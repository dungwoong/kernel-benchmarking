"""
python compile_ir.py /path/to/_helion_rmsnorm_lin_kernel

Uses <base>.json for options and compiles <base>.<IR_EXT>.
  IR_EXT = "ttgir" -> only LLVM / PTX / cubin run on your file
  IR_EXT = "ttir"  -> the full TTGIR pipeline runs again on your file

Accuracy: RMSE of (torch bf16) and (compiled IR kernel) against torch on fp64 inputs.
"""
import json
import sys

import torch
import triton

IR_EXT = "ttir"          # "ttgir" or "ttir"
M = N = K = 4096
EPS = 1e-6

# NOTE had to fetch this from kernel_bundles/.../source.py
GRID = (132, 1, 1)        # persistent grid; check the launch call in the Helion-generated .py

base = sys.argv[1]
ir_path = f"{base}.{IR_EXT}"


# ---- reference + metric -----------------------------------------------------------------
@torch.compile
def torch_kernel(a: torch.Tensor, b: torch.Tensor, eps: float = EPS):
    a_rms = torch.nn.functional.rms_norm(a, normalized_shape=(a.shape[1],), eps=eps)
    return a_rms @ b.t()


def get_rmse(ref: torch.Tensor, o: torch.Tensor):
    assert o.dtype == ref.dtype
    assert o.numel() == ref.numel(), f'{o.shape=} {ref.shape=}'
    mse = torch.nn.functional.mse_loss(o.flatten(), ref.flatten(), reduction='mean')
    rmse = mse.sqrt().item()
    return rmse


# ---- options from the original compile ------------------------------------------------
with open(f"{base}.json") as f:
    meta = json.load(f)
options = {k: meta[k] for k in ("num_warps", "num_ctas", "num_stages", "maxnreg", "enable_fp_fusion")
           if meta.get(k) is not None}
print(f"compiling {ir_path}")
print(f"options: {options}")
print(f"triton: dumped with {meta.get('triton_version')}, running {triton.__version__}")

# ---- device-side TMA descriptors need a global scratch allocator -----------------------
triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8, device="cuda"))

# ---- compile ------------------------------------------------------------------------------
k = triton.compile(ir_path, options=options)

# ---- inputs ---------------------------------------------------------------------------------
torch.manual_seed(0)
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
y = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)


def launch():
    k[GRID](x, y, out, EPS)   # (x, y, out, eps) = order of the tt.func arguments


launch()
torch.cuda.synchronize()
print(f"n_regs: {k.n_regs}  n_spills: {k.n_spills}  shared: {k.metadata.shared} B")

# ---- accuracy: everything against torch on fp64 copies of the SAME bf16 inputs ------------
ref64 = torch_kernel(x.double(), y.double())       # fp64 reference
torch_bf16 = torch_kernel(x, y)                    # torch in bf16 (baseline error)
torch.cuda.synchronize()

rmse_torch = get_rmse(ref64, torch_bf16.double())
rmse_ir = get_rmse(ref64, out.double())
print(f"RMSE vs fp64 | torch bf16: {rmse_torch:.6g}   {IR_EXT} kernel: {rmse_ir:.6g}   "
      f"ratio: {rmse_ir / rmse_torch:.3f}x")

# ---- time ---------------------------------------------------------------------------------
ms = triton.testing.do_bench(launch)
ms_torch = triton.testing.do_bench(lambda: torch_kernel(x, y))
tflops = lambda t: 2 * M * N * K / (t * 1e-3) / 1e12
print(f"{IR_EXT} kernel: {ms:.3f} ms  ~{tflops(ms):.0f} TFLOP/s")
print(f"torch bf16:   {ms_torch:.3f} ms  ~{tflops(ms_torch):.0f} TFLOP/s")

# ---- save what came out, next to the input ------------------------------------------------
for ext in ("ttgir", "llir", "ptx"):
    with open(f"{base}.recompiled_from_{IR_EXT}.{ext}", "w") as f:
        f.write(k.asm[ext])
with open(f"{base}.recompiled_from_{IR_EXT}.sass", "w") as f:
    f.write(k.asm["sass"])   # needs nvdisasm on PATH
print(f"wrote {base}.recompiled_from_{IR_EXT}.{{ttgir,llir,ptx,sass}}")