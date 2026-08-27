"""
Times a Helion autotune of the gemm kernel for one shape

Output file:
bench_kernels/helion/autotune_cache/matmul/<shape key>.json

Skips a shape whose config is already cached
"""
import torch
from autotune_common import autotune_runner
from profile_utils import get_profiling_job_args
from workload_shapes import GEMM_ARGS_NT
from helion_utils.kernel_runner import Matmul

torch.manual_seed(18)

MASK = (0, 1)  # a, b from GEMM_ARGS_NT

if __name__ == "__main__":
    args = get_profiling_job_args()
    autotune_runner(Matmul, GEMM_ARGS_NT.with_config(args.config), MASK, "matmul", args.csv)
