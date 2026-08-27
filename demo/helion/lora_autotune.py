"""
Times a Helion autotune of the LoRA kernel for one shape

Output file:
bench_kernels/helion/autotune_cache/LoRA/<shape key>.json

Skips a shape whose config is already cached
"""
import torch
from autotune_common import autotune_runner
from profile_utils import get_profiling_job_args
from workload_shapes import LORA_NT_16
from helion_utils.kernel_runner import LoRA

torch.manual_seed(18)

MASK = (0, 1, 2, 3)  # x, W, A, B from LORA_NT_16

if __name__ == "__main__":
    args = get_profiling_job_args()
    autotune_runner(LoRA, LORA_NT_16.with_config(args.config), MASK, "lora", args.csv)
