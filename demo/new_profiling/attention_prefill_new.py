import torch

import flash_attn_interface
from torch.nn.attention import SDPBackend, sdpa_kernel

from profile_utils import ProfilingJob, get_profiling_job_args, AttentionOutput
from workload_shapes import ATTENTION_ARGS

"""
Compute-bound batch-1 attention comparing cuDNN vs FA3 vs Torch

Config indexes into ATTENTION_SHAPES
q/k/v are (B, S, H, D) with B=1 and each kernel reshapes to the expected layout
"""
torch.manual_seed(18)

CAUSAL = False


def cudnn_attention(q, k, v):
    # SDPA wants (B, H, S, D)
    qh, kh, vh = (t.transpose(1, 2) for t in (q, k, v))
    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
        o = torch.nn.functional.scaled_dot_product_attention(qh, kh, vh, is_causal=CAUSAL)
    return o.transpose(1, 2)


def fa3_attention(q, k, v):
    # FA3 wants (B, S, H, D)
    return flash_attn_interface.flash_attn_func(q, k, v, causal=CAUSAL)


@torch.compile
def torch_attention(q, k, v):
    # pass (H,S,D)
    qh, kh, vh = (t.squeeze(0).transpose(0, 1) for t in (q, k, v))
    scale = q.shape[-1] ** -0.5
    attn_weights = torch.matmul(qh, kh.transpose(-2, -1)) * scale
    attn_probs = torch.nn.functional.softmax(attn_weights, dim=-1)
    o = torch.matmul(attn_probs, vh)
    return o.transpose(0, 1).unsqueeze(0)


if __name__ == "__main__":
    args = get_profiling_job_args()
    prob_args = ATTENTION_ARGS.with_config(args.config)

    p = ProfilingJob(
        "attn",
        kernels={"cudnn": cudnn_attention, "fa3": fa3_attention, "torch": torch_attention},
        args=prob_args,
        baseline="cudnn",
        ref="torch",
        output_cls=AttentionOutput,
    )
    p.run(ncu=args.ncu, csv=args.csv)
