"""Is the fla kernel (fla-core) a drop-in for the reference PyTorch gated delta rule that transformers falls back to when fla is missing?
Qwen3.5-9B: 24 of its 32 layers are Gated DeltaNet, inputs [B, T, 32 heads, 128] after the k->v head repeat, g float32, beta bf16.
Forward and backward on random inputs of that shape, both implementations, relative L2 error of the output and of every gradient.
    .venv-train/bin/python training/check_fla.py [--T 4096]      -> last line "fla check PASS ..." or "fla check FAIL ..."
"""
import argparse
import inspect
import time

import torch
import torch.nn.functional as F
from transformers.models.qwen3_5 import modeling_qwen3_5 as M
from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_chunk

ap = argparse.ArgumentParser()
ap.add_argument("--T", type=int, default=4096)
ap.add_argument("--tol-out", type=float, default=2e-2)
ap.add_argument("--tol-grad", type=float, default=5e-2)
a = ap.parse_args()

ref = inspect.unwrap(M.torch_chunk_gated_delta_rule)  # the hub/fla dispatch wraps the torch function: take the torch one
assert ref.__module__ == M.__name__ and ref is not fla_chunk, ref

torch.manual_seed(0)
dev, B, T, H, D = "cuda", 1, a.T, 32, 128
q0 = torch.randn(B, T, H, D, device=dev, dtype=torch.bfloat16)
k0 = torch.randn(B, T, H, D, device=dev, dtype=torch.bfloat16)
v0 = torch.randn(B, T, H, D, device=dev, dtype=torch.bfloat16)
g0 = -torch.rand(B, T, H, device=dev, dtype=torch.float32).mul(2).exp().mul(0.05) * F.softplus(torch.randn(B, T, H, device=dev))
b0 = torch.randn(B, T, H, device=dev, dtype=torch.bfloat16).sigmoid()
do = torch.randn(B, T, H, D, device=dev, dtype=torch.bfloat16)


def run(fn):
    xs = [x.clone().requires_grad_(True) for x in (q0, k0, v0, g0, b0)]
    torch.cuda.synchronize(); t = time.time()
    o, _ = fn(xs[0], xs[1], xs[2], g=xs[3], beta=xs[4], initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True)
    (o.float() * do.float()).sum().backward()
    torch.cuda.synchronize()
    return o.float(), [x.grad.float() for x in xs], time.time() - t


run(fla_chunk)  # Triton autotune / compile once
o_r, g_r, t_r = run(ref)
o_f, g_f, t_f = run(fla_chunk)
rel = lambda x, y: ((x - y).norm() / y.norm().clamp_min(1e-12)).item()
eo = rel(o_f, o_r)
eg = {n: rel(x, y) for n, x, y in zip(("dq", "dk", "dv", "dg", "dbeta"), g_f, g_r)}
finite = bool(torch.isfinite(o_f).all()) and all(bool(torch.isfinite(x).all()) for x in g_f)
ok = finite and eo < a.tol_out and max(eg.values()) < a.tol_grad
print(f"T={T}: reference {t_r:.2f}s, fla {t_f:.3f}s ({t_r / max(t_f, 1e-9):.0f}x) for one layer forward+backward")
print(f"rel L2 error: out {eo:.2e}, " + ", ".join(f"{n} {e:.2e}" for n, e in eg.items()))
print(f"fla check {'PASS' if ok else 'FAIL'}: out {eo:.1e} grad max {max(eg.values()):.1e} finite {finite} speedup {t_r / max(t_f, 1e-9):.0f}x")
