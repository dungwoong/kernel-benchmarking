import torch
from profile_utils import ProfilingJob, get_profiling_job_args, AttentionOutput
from workload_shapes import ATTENTION_ARGS
from helion_utils.kernel_runner import Attention

torch.manual_seed(18)

CAUSAL = False
MASK = (0, 1, 2)  

@torch.compile
def torch_sdpa(q, k, v):
    # SDPA wants (B, H, S, D)
    o = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=CAUSAL,
    )
    return o.transpose(1, 2)


if __name__ == '__main__':
    args = get_profiling_job_args()
    prob_args = ATTENTION_ARGS.with_config(args.config)
    tensors = tuple(t.transpose(1, 2) for t in prob_args.tensors(MASK))
    compiled_attention = Attention.compile(*tensors)
    Attention.dump_ir(compiled_attention, tensors)

    def helion_attention(q, k, v):
        # Helion kernel wants (B, H, S, D) 
        o = compiled_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return o.transpose(1, 2)

    p = ProfilingJob(
        "attn",
        kernels={"helion": helion_attention, "torch": torch_sdpa},
        args=prob_args,
        baseline="torch",
        ref="torch",
        output_cls=AttentionOutput,
    )
    p.run(ncu=args.ncu, csv=args.csv)
