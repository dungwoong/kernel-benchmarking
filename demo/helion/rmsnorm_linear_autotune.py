"""
Times a Helion autotune of the RMSNorm + Linear kernel for one shape

Output file:
bench_kernels/helion/autotune_cache/rmsnorm_lin/<shape key>.json

Skips a shape whose config is already cached
"""
import torch
from autotune_common import autotune_runner
from profile_utils import get_profiling_job_args
from workload_shapes import RMSNORM_LINEAR_ARGS_NT
from helion_utils.kernel_runner import RMSNormLinear

torch.manual_seed(18)

MASK = (0, 1, 3)  # a, b, eps from RMSNORM_LINEAR_ARGS_NT

if __name__ == "__main__":
    args = get_profiling_job_args()
    autotune_runner(RMSNormLinear, RMSNORM_LINEAR_ARGS_NT.with_config(args.config),
                    MASK, "rmsnorm_lin", args.csv)
