import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moe import MoE, MoEConfig, load_model_from_checkpoint, save_checkpoint
from moe.model import ExpertMLPBank, Router, casual_mask, load_balancing_loss

CFG = MoEConfig(vocab_size=50, d_model=32, num_layers=2, num_heads=4, max_seq_len=16,
                num_experts=4, top_k=2, ffn_dim=64)

def test_forward_shapes():
    model = MoE(CFG)
    x = torch.randint(0, 50, (3, 10))
    out = model(x, return_routing=True)
    assert out.logits.shape == (3, 10, 50)
    assert out.aux_loss.ndim == 0
    assert len(out.routing) == CFG.num_layers
    assert out.routing[0].shape == (3, 10, CFG.top_k)

def test_casual_mask_blocks_future():
    torch.manual_seed(0)
    model = MoE(CFG).eval()
    x = torch.randint(0, 50, (1, 8))
    y = x.clone()
    y[0, -1] = (y[0, -1] + 1) % 50
    a, b = model(x).logits, model(y).logits
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-5)

def test_router_gates_sum_to_one():
    r = Router(32, 4, 2)
    idx, gates, probs = r(torch.randn(2, 5, 32))
    assert torch.allclose(gates.sum(-1), torch.ones(2, 5), atol=1e-5)
    assert torch.allclose(probs.sum(-1), torch.ones(2, 5), atol=1e-5)
    assert idx.max() < 4

def test_expert_bank_matches_naive_loop():
    torch.manual_seed(0)
    bank = ExpertMLPBank(8, 16, 3)
    x = torch.randn(5, 8)
    idx = torch.tensor([[0, 1], [2, 0], [1, 2], [0, 2], [1, 0]])
    gates = torch.softmax(torch.randn(5, 2), -1)
    expected = torch.zeros(5, 8)
    for n in range(5):
        for s in range(2):
            e = idx[n, s]
            h = torch.nn.functional.silu(x[n] @ bank.w1[e] + bank.b1[e])
            expected[n] += gates[n, s] * (h @ bank.w2[e] + bank.b2[e])
    assert torch.allclose(bank(x, idx, gates), expected, atol=1e-5)

def test_balance_loss():
    probs = torch.full((1, 4, 4), 0.25)
    idx = torch.tensor([[[0], [1], [2], [3]]])
    assert torch.isclose(load_balancing_loss(probs, idx, 4), torch.tensor(1.0))


def test_casual_mask_values():
    m = casual_mask(3, device=torch.device("cpu"))
    assert m.tolist() == [[True, False, False], [True, True, False], [True, True, True]]

def test_checkpoint_roundtrip(tmp_path):
    model = MoE(CFG).eval()
    path = tmp_path / "m.pt"
    save_checkpoint(str(path), model)
    loaded, _ = load_model_from_checkpoint(str(path), device="cpu")
    x = torch.randint(0, 50, (1, 6))
    assert torch.allclose(model(x).logits, loaded(x).logits)

def test_generate_length():
    model = MoE(CFG).eval()
    out = model.generate(torch.zeros(1, 3, dtype=torch.long), max_new_tokens=20, temperature=0.8, top_k=5)
    assert out.shape == (1, 23) 