
import os
os.environ.setdefault("TORCH_COMPILE_DEBUG", "1")  # must precede import torch

from pathlib import Path

import torch
import torch.nn.functional as F


def norm_transformer(H, X, alpha):
    H_norm = F.rms_norm(H, normalized_shape=(H.shape[-1],))
    A = H_norm + X
    B = alpha * A
    C = X + B
    return F.rms_norm(C, normalized_shape=(C.shape[-1],))


bs = 1
H = torch.randn(8 * bs, 4096, device="cuda", dtype=torch.float16)
X = torch.randn(8 * bs, 4096, device="cuda", dtype=torch.float16)
alpha = torch.randn(8 * bs, 4096, device="cuda", dtype=torch.float16)

compiled = torch.compile(norm_transformer)
with torch.no_grad():
    out = compiled(H, X, alpha)
torch.cuda.synchronize()
print("output:", tuple(out.shape), out.dtype)
for p in sorted(Path("torch_compile_debug").rglob("output_code.py")):
    print("dumped kernels ->", p)
