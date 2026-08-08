"""
Autotunes the Helion GEMM and saves Configs to .cache/helion/

How to run:
python3 demo/helion_gemm_autotune.py                      # all shapes, quick
python3 demo/helion_gemm_autotune.py --effort full        # all shapes, full
python3 demo/helion_gemm_autotune.py --shape 3 --effort full   # one shape, full
"""
import argparse
import os

# Run each candidate in a subprocess so a crashing config doesn't kill the whole tuner
os.environ.setdefault("HELION_AUTOTUNE_PRECOMPILE", "spawn")

import torch
from workload_shapes import GEMM_SHAPES
from helion_gemm import autotune, cache_path

torch.manual_seed(18)


def tune_one(cfg, effort):
    m, n, k = cfg["m"], cfg["n"], cfg["k"]
    path = cache_path(m, n, k, torch.bfloat16, effort)
    if os.path.exists(path):
        print(f"Skipping {m}x{n}x{k} [{effort}] (exists)")
        return
    print(f"Autotuning {m}x{n}x{k} [{effort}] ...")
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    saved = autotune(a, b, effort)
    print(f"  Saved -> {saved}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--shape", type=int, default=None,
                   help="GEMM_SHAPES index to tune; omit to tune all shapes.")
    p.add_argument("--effort", choices=["quick", "full"], default="quick")
    args = p.parse_args()
    shapes = [GEMM_SHAPES[args.shape]] if args.shape is not None else GEMM_SHAPES
    for cfg in shapes:
        tune_one(cfg, args.effort)
