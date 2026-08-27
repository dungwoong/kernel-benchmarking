"""
Times a Helion autotune of the attention kernel for one shape

Output file:
helion_utils/autotune_cache/attention/<shape key>.json

Skips a shape whose config is already cached
"""
import torch
from autotune_common import autotune_runner
from profile_utils import get_profiling_job_args, AttentionOutput
from workload_shapes import ATTENTION_ARGS
from helion_utils.kernel_runner import Attention

torch.manual_seed(18)

MASK = (0, 1, 2)


if __name__ == "__main__":
    args = get_profiling_job_args()
    prob_args = ATTENTION_ARGS.with_config(args.config)
    tensors = tuple(t.transpose(1, 2) for t in prob_args.tensors(MASK))
    autotune_runner(Attention, prob_args, MASK, "attn", args.csv, output_cls=AttentionOutput, tensors=tensors)

