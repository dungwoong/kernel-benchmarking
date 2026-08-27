"""
Times a Helion autotune of the SwiGLU kernel for one shape

Output file:
bench_kernels/helion/autotune_cache/swiglu/<shape key>.json

Skips a shape whose config is already cached
"""
import torch
from autotune_common import autotune_runner
from profile_utils import get_profiling_job_args
from workload_shapes import SWIGLU_NT
from helion_utils.kernel_runner import SwiGLU

torch.manual_seed(18)

MASK = (0, 1, 2)  # x, w1, w2 from SWIGLU_NT

if __name__ == "__main__":
    args = get_profiling_job_args()
    autotune_runner(SwiGLU, SWIGLU_NT.with_config(args.config), MASK, "swiglu", args.csv)
