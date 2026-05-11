import torch

from cutedsl_kernels import Gemm4SM90
from workload_shapes import GEMM_ARGS_NT
from profile_utils import ProfilingJob, get_profiling_job_args
from cdsl_fn_utils import compile_cutedsl

"""
Profiles torch gemm using two profiling methods

This script is likely unneeded anymore since we use do_bench for everything now.
"""
torch.manual_seed(18)

@torch.compile
def torch_kernel(a: torch.Tensor, b: torch.Tensor):
    return a @ b.t()

gemm = Gemm4SM90(
    tile_shape_mnk=(128, 256, 32),
    epi_tile_mn=(128, 128),
    cluster_shape_mnk=(2, 1, 1),
    atom_layout_mn=(2, 1),
    ab_stage=6,
    epi_stage=2,
    is_persistent=True,
    gemm_n_prologue=1,
)


if __name__ == "__main__":
    args = get_profiling_job_args()
    prob_args = GEMM_ARGS_NT.with_config(args.config)
    prob_args.cuda()
    compiled_gemm = compile_cutedsl(prob_args.tensors(), gemm, include_stream=False)

    def cdsl_kernel(a_: torch.Tensor, b_: torch.Tensor):
        o = torch.empty(a_.shape[0], b_.shape[0], dtype=torch.bfloat16, device='cuda')
        compiled_gemm(a_, b_, o)
        return o

    p = ProfilingJob(
        "gemm",
        kernels={"cutedsl": cdsl_kernel, "torch": torch_kernel},
        args=prob_args,
        arg_mask={"torch": (0, 1), "cutedsl": (0, 1)},
        baseline="torch",
        ref="torch")
    
    p.run(ncu=args.ncu)