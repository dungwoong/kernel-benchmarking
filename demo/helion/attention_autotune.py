"""
Times a Helion autotune of the attention kernel for one shape

Output file:
helion_utils/autotune_cache/attention/<shape key>.json

Skips a shape whose config is already cached
"""
import torch
from autotune_common import autotune_runner
from profile_utils import get_profiling_job_args, AttentionOutput
from workload_shapes import ATTENTION_ARGS_BHSD
from helion_utils.kernel_runner import Attention

torch.manual_seed(18)

MASK = (0, 1, 2)  # q, k, v from ATTENTION_ARGS_BHSD, already (B, H, S, D)

if __name__ == "__main__":
    args = get_profiling_job_args()
    autotune_runner(Attention, ATTENTION_ARGS_BHSD.with_config(args.config), MASK,
                    "attn", args.csv, output_cls=AttentionOutput)

