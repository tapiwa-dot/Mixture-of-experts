# A decoder-only Transformer whose feed-forward blocks are Mixture-of-Experts layers

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MoEConfig


# Building blocks

class RMSNorm(nn.Module):
    """Root-mean-square layer norm, applied to each token's feature vector."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FIX: normalize over the feature dimension (-1), not the time dimension (1).
        # Averaging over time mixed information across positions and leaked future tokens.
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.scale


class DenseNoBias(nn.Module):
    """y = x @ W, with W of shape [in, out]."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.kernel = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.normal_(self.kernel, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.kernel


def causal_mask(t: int, *, device: torch.device) -> torch.Tensor:
    """Lower-triangular boolean mask: position i may attend to positions <= i."""
    return torch.tril(torch.ones((t, t), dtype=torch.bool, device=device))


casual_mask = causal_mask  # old (misspelled) name kept so existing imports still work


class MultiHeadAttention(nn.Module):
    """Standard multi-head causal self-attention."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = DenseNoBias(d_model, d_model)
        self.k_proj = DenseNoBias(d_model, d_model)
        self.v_proj = DenseNoBias(d_model, d_model)
        self.out_proj = DenseNoBias(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, *, attn_mask: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(b, t, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(b, t, self.num_heads, self.head_dim)

        scale = 1.0 / math.sqrt(self.head_dim)
        att = torch.einsum("bthd,bshd->bhts", q, k) * scale
        att = att.masked_fill(~attn_mask.view(1, 1, t, t), float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.attn_drop(att)
        out = torch.einsum("bhts,bshd->bthd", att, v).reshape(b, t, d)
        return self.out_proj(out)


# Mixture of experts

class Router(nn.Module):
    """Scores every token against every expert and keeps the top-k."""

    def __init__(self, d_model: int, num_experts: int, top_k: int, jitter_noise: float = 0.0):
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError(f"top_k must be between 1 and num_experts ({num_experts}), got {top_k}")
        self.num_experts = num_experts
        self.top_k = top_k
        self.jitter_noise = jitter_noise
        self.gate = DenseNoBias(d_model, num_experts)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.gate(x)

        # Noise encourages exploration early in training
        if self.training and self.jitter_noise > 0:
            logits = logits + torch.randn_like(logits) * self.jitter_noise

        probs = torch.softmax(logits, dim=-1)
        topk_vals, topk_idx = torch.topk(probs, k=self.top_k, dim=-1)
        if self.top_k > 1:
            # Mixtral-style: renormalize over the chosen experts
            gates = topk_vals / topk_vals.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        else:
            # FIX: with top_k = 1, renormalizing gives a constant 1.0 and the router
            # gets no gradient from the task loss. Switch Transformer uses the raw probability.
            gates = topk_vals
        return topk_idx, gates, probs


def load_balancing_loss(probs: torch.Tensor, topk_idx: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Switch Transformer auxiliary loss.

    Minimized (value 1.0) when tokens are spread evenly across experts.
    f_i = fraction of routing slots sent to expert i
    P_i = mean router probability for expert i
    loss = E * sum_i f_i * P_i
    """
    probs = probs.reshape(-1, num_experts)
    one_hot = F.one_hot(topk_idx.reshape(-1), num_experts).float()
    f = one_hot.mean(dim=0)
    p = probs.mean(dim=0)
    return num_experts * torch.sum(f * p)


class ExpertMLPBank(nn.Module):
    """N independent two-layer MLPs stored as stacked weight tensors.

    Tokens are grouped by expert, so each expert runs one matmul over only
    the tokens routed to it.
    """

    def __init__(self, d_model: int, hidden_dim: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        self.w1 = nn.Parameter(torch.empty(num_experts, d_model, hidden_dim))
        self.b1 = nn.Parameter(torch.zeros(num_experts, hidden_dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, hidden_dim, d_model))
        self.b2 = nn.Parameter(torch.zeros(num_experts, d_model))
        nn.init.normal_(self.w1, std=0.02)
        nn.init.normal_(self.w2, std=0.02)

    def forward(self, x: torch.Tensor, expert_idx: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
        """
        x:          [n, d] flattened tokens
        expert_idx: [n, k] chosen experts per token
        gates:      [n, k] weight for each chosen expert
        returns     [n, d] sum_k gate_k * expert_k(x)
        """
        out = torch.zeros_like(x)
        for e in range(self.num_experts):
            token_pos, slot = torch.where(expert_idx == e)
            if token_pos.numel() == 0:
                continue
            h = F.silu(x[token_pos] @ self.w1[e] + self.b1[e])
            y = h @ self.w2[e] + self.b2[e]
            out.index_add_(0, token_pos, y * gates[token_pos, slot].unsqueeze(-1))
        return out


class MoELayer(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, num_experts: int, top_k: int, jitter_noise: float = 0.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = Router(d_model, num_experts, top_k, jitter_noise)
        self.experts = ExpertMLPBank(d_model, hidden_dim, num_experts)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, t, d = x.shape
        topk_idx, gates, probs = self.router(x)
        y = self.experts(
            x.reshape(b * t, d),
            topk_idx.reshape(b * t, self.top_k),
            gates.reshape(b * t, self.top_k),
        )
        aux = load_balancing_loss(probs, topk_idx, self.num_experts)
        return y.reshape(b, t, d), aux, topk_idx


class Block(nn.Module):
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.rmsnorm_0 = RMSNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg.d_model, cfg.num_heads, cfg.dropout)
        self.rmsnorm_1 = RMSNorm(cfg.d_model)
        self.moe = MoELayer(cfg.d_model, cfg.ffn_dim, cfg.num_experts, cfg.top_k, cfg.router_jitter)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, *, attn_mask: torch.Tensor):
        # FIX: attn_mask is keyword-only in MultiHeadAttention.forward
        x = x + self.drop(self.attn(self.rmsnorm_0(x), attn_mask=attn_mask))
        moe_out, aux, topk_idx = self.moe(self.rmsnorm_1(x))
        x = x + self.drop(moe_out)
        return x, aux, topk_idx


# Full model

@dataclass
class MOEOutput:
    logits: torch.Tensor                    # [b, t, vocab]
    aux_loss: torch.Tensor                  # scalar, mean over layers
    routing: Optional[List[torch.Tensor]]   # per layer [b, t, k]


class MoE(nn.Module):
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.num_layers)])
        self.rmsnorm_f = RMSNorm(cfg.d_model)
        self.lm_head = DenseNoBias(cfg.d_model, cfg.vocab_size)
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def num_params(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        expert = sum(p.numel() for n, p in self.named_parameters() if ".experts." in n)
        per_expert = expert // self.cfg.num_experts
        active = total - expert + per_expert * self.cfg.top_k
        # FIX: key renamed from "activate_per_token" to match train.py
        return {"total": total, "active_per_token": active}

    def forward(self, input_ids: torch.Tensor, *, return_routing: bool = False) -> MOEOutput:
        b, t = input_ids.shape
        if t > self.cfg.max_seq_len:
            # FIX: max_seq_len lives on self.cfg
            raise ValueError(f"input length {t} exceeds max_seq_len {self.cfg.max_seq_len}")
        device = input_ids.device

        pos = torch.arange(t, device=device).unsqueeze(0)
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))
        mask = causal_mask(t, device=device)

        aux_total = x.new_zeros(())
        routing: List[torch.Tensor] = []
        for blk in self.blocks:
            x, aux, topk_idx = blk(x, attn_mask=mask)
            aux_total = aux_total + aux
            if return_routing:
                routing.append(topk_idx)

        logits = self.lm_head(self.rmsnorm_f(x))
        return MOEOutput(logits, aux_total / len(self.blocks), routing if return_routing else None)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: Optional[int] = None) -> torch.Tensor:
        """Autoregressive sampling. temperature <= 0 means greedy."""
        for _ in range(max_new_tokens):
            ctx = input_ids[:, -self.cfg.max_seq_len:]
            logits = self(ctx).logits[:, -1, :]
            if temperature <= 0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
                nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1)
            input_ids = torch.cat([input_ids, nxt], dim=1)
        return input_ids


# Checkpoints

def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(path: str, model: MoE, extra: Optional[Dict[str, Any]] = None):
    ckpt = {"config": model.cfg.to_dict(), "state_dict": model.state_dict()}
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_model_from_checkpoint(
        checkpoint_path: str,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
) -> Tuple[MoE, Dict[str, Any]]:
    """Return (model in eval mode, full checkpoint dict)."""
    device = torch.device(device) if device is not None else default_device()
    # FIX: the argument is weights_only (plural)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = MoEConfig.from_dict(ckpt.get("config", {}))
    model = MoE(cfg)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device=device, dtype=dtype).eval()
    return model, ckpt