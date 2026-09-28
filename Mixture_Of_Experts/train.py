"""Train the MoE Transformer on a text file (Tiny Shakespeare by default).

    python train.py --preset tiny --steps 2000

Writes:
    checkpoints/model.pt      best checkpoint by validation loss
    checkpoints/metrics.jsonl one JSON line per evaluation
    assets/training_curves.png
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from moe import PRESETS, MoE, save_checkpoint
from moe.data import CharTokenizer, get_batch, load_text, train_val_split
from moe.model import default_device


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default="tiny", choices=list(PRESETS))
    p.add_argument("--data", default="data/tinyshakespeare.txt")
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=None, help="defaults to the preset's max_seq_len")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--min-lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-batches", type=int, default=10)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default=None)
    return p.parse_args()


def lr_at(step: int, args) -> float:
    """Linear warmup then cosine decay to min_lr."""
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    progress = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model: MoE, data, args, seq_len, device):
    """Mean loss, next-token accuracy, and per-layer expert load on `data`."""
    model.eval()
    losses, accs = [], []
    counts = torch.zeros(model.cfg.num_layers, model.cfg.num_experts)
    for _ in range(args.eval_batches):
        x, y = get_batch(data, args.batch_size, seq_len, device)
        out = model(x, return_routing=True)
        losses.append(F.cross_entropy(out.logits.flatten(0, 1), y.flatten()).item())
        accs.append((out.logits.argmax(-1) == y).float().mean().item())
        for layer, idx in enumerate(out.routing):
            counts[layer] += torch.bincount(idx.flatten().cpu(), minlength=model.cfg.num_experts)
    model.train()
    load = (counts / counts.sum(dim=1, keepdim=True)).tolist()
    return sum(losses) / len(losses), sum(accs) / len(accs), load


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device) if args.device else default_device()

    text = load_text(args.data)
    tok = CharTokenizer.from_text(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    train_data, val_data = train_val_split(ids)

    cfg = dataclasses.replace(PRESETS[args.preset], vocab_size=tok.vocab_size)
    seq_len = args.seq_len or cfg.max_seq_len
    model = MoE(cfg).to(device)
    n = model.num_params()
    print(f"device={device}  params={n['total']/1e6:.2f}M  active/token={n['active_per_token']/1e6:.2f}M")

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95),
    )

    os.makedirs(args.out, exist_ok=True)
    metrics_path = os.path.join(args.out, "metrics.jsonl")
    ckpt_path = os.path.join(args.out, "model.pt")
    open(metrics_path, "w").close()

    best_val = float("inf")
    run_loss = run_acc = run_aux = 0.0
    run_n = 0
    t0 = time.time()
    model.train()

    for step in range(args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args)

        if step % args.eval_every == 0 or step == args.steps:
            val_loss, val_acc, load = evaluate(model, val_data, args, seq_len, device)
            record = {
                "step": step,
                "train_loss": run_loss / run_n if run_n else None,
                "train_acc": run_acc / run_n if run_n else None,
                "aux_loss": run_aux / run_n if run_n else None,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": opt.param_groups[0]["lr"],
                "expert_load": load,
                "elapsed_s": round(time.time() - t0, 1),
            }
            with open(metrics_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            tl = f"{record['train_loss']:.3f}" if run_n else "  -  "
            print(f"step {step:5d} | train {tl} | val {val_loss:.3f} | val acc {val_acc:.1%} "
                  f"| {record['elapsed_s']:.0f}s", flush=True)
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(ckpt_path, model, {
                    "tokenizer": tok.to_dict(), "step": step, "val_loss": val_loss, "val_acc": val_acc,
                })
            run_loss = run_acc = run_aux = 0.0
            run_n = 0
            if step == args.steps:
                break

        x, y = get_batch(train_data, args.batch_size, seq_len, device)
        out = model(x)
        ce = F.cross_entropy(out.logits.flatten(0, 1), y.flatten())
        loss = ce + cfg.aux_loss_coef * out.aux_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        run_loss += ce.item()
        run_acc += (out.logits.argmax(-1) == y).float().mean().item()
        run_aux += out.aux_loss.item()
        run_n += 1

    print(f"best val loss {best_val:.3f}  ->  {ckpt_path}")
    try:
        from plot_metrics import plot
        plot(metrics_path, "assets")
    except Exception as e:  # plotting should never fail a training run
        print(f"(skipped plotting: {e})")


if __name__ == "__main__":
    main()
