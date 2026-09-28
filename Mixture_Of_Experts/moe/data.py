"""Character-level tokenizer and a simple random-window batch loader."""

from __future__ import annotations

import os
import urllib.request
from typing import Dict, List, Tuple

import torch

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


class CharTokenizer:
    def __init__(self, chars: List[str]):
        self.chars = chars
        self.stoi: Dict[str, int] = {c: i for i, c in enumerate(chars)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> List[int]:
        # Unknown characters are dropped rather than raising an error
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.chars[i] for i in ids)

    def to_dict(self) -> Dict:
        return {"type": "char", "chars": self.chars}

    @classmethod
    def from_dict(cls, d: Dict) -> "CharTokenizer":
        return cls(d["chars"])


def load_text(path: str) -> str:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        print(f"Downloading Tiny Shakespeare to {path} ...")
        tmp = path + ".part"
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, tmp)
        os.replace(tmp, path)   # only appears under the real name once complete
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def train_val_split(ids: torch.Tensor, val_frac: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    n = int(len(ids) * (1 - val_frac))
    return ids[:n], ids[n:]


def get_batch(data: torch.Tensor, batch_size: int, seq_len: int, device: torch.device):
    """Random contiguous windows; targets are the inputs shifted by one."""
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    # FIX: x was data[i : i + seq_len - 1], one token shorter than y
    x = torch.stack([data[i : i + seq_len] for i in ix])
    y = torch.stack([data[i + 1 : i + seq_len + 1] for i in ix])
    return x.to(device), y.to(device)