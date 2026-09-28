"""Model configuration and size presets."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict


@dataclass(frozen=True)
class MoEConfig:
    vocab_size: int = 50257
    d_model: int = 768
    num_layers: int = 12
    num_heads: int = 16
    max_seq_len: int = 512
    num_experts: int = 8
    top_k: int = 2
    ffn_dim: int = 2048
    dropout: float = 0.0
    router_jitter: float = 0.0
    # FIX: renamed from aux_loss_coeff to match cfg.aux_loss_coef in train.py
    aux_loss_coef: float = 0.01

    # FIX: was __post_ini__, so these checks never ran
    def __post_init__(self) -> None:
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if not 1 <= self.top_k <= self.num_experts:
            raise ValueError("top_k must be between 1 and num_experts")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MoEConfig":
        d = dict(d)
        if "aux_loss_coeff" in d and "aux_loss_coef" not in d:   # old checkpoints
            d["aux_loss_coef"] = d.pop("aux_loss_coeff")
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


PRESETS: Dict[str, MoEConfig] = {
    "tiny": MoEConfig(
        vocab_size=65, d_model=128, num_layers=4, num_heads=4, max_seq_len=128,
        num_experts=8, top_k=2, ffn_dim=256, dropout=0.1, router_jitter=0.01,
    ),
    "small": MoEConfig(
        vocab_size=65, d_model=384, num_layers=6, num_heads=6, max_seq_len=256,
        num_experts=8, top_k=2, ffn_dim=1024, dropout=0.1, router_jitter=0.01,
    ),
    "base": MoEConfig(),
}