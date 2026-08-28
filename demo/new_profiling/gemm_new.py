import torch

from cutedsl_kernels import Gemm4SM90
from workload_shapes import GEMM_ARGS_NT
from profile_utils import ProfilingJob, get_profiling_job_args
from cdsl_fn_utils import compile_cutedsl

from compiler_2.kernels.hel.gemm import get_kernel as get_c2_kernel
from c2_utils import compile_hel
from helion_utils.kernel_runner import Matmul

"""
Profiles torch gemm vs cutedsl gemm vs compiler_2's autotuned gemm

Uses new profiling setup
"""
torch.manual_seed(18)

C2_TILE_M, C2_TILE_N, C2_TILE_K = 128, 256, 64

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
    prob_args = GEMM_ARGS_NT.with_config(args.config) # Get set of tensors defining the problem

    # Compile CuteDSL manually
    compiled_gemm = compile_cutedsl(prob_args.tensors(), gemm, include_stream=False)

    m, n, k = prob_args.arg('m', 'n', 'k')
    c2_kernel = compile_hel(get_c2_kernel(m, n, k, C2_TILE_M, C2_TILE_N, C2_TILE_K), name=f'gemm_{m}x{n}x{k}')

    def cdsl_kernel(a_: torch.Tensor, b_: torch.Tensor):
        o = torch.empty(a_.shape[0], b_.shape[0], dtype=torch.bfloat16, device='cuda')
        compiled_gemm(a_, b_, o)
        return o

    def c2_kernel_fn(a_: torch.Tensor, b_: torch.Tensor):
        o = torch.empty(a_.shape[0], b_.shape[0], dtype=torch.bfloat16, device='cuda')
        c2_kernel(a_, b_, o)
        return o

    # Helion will load its tuned config from helion_utils/autotune_cache
    helion_kernel = Matmul.compile(*prob_args.tensors((0, 1)))

    p = ProfilingJob(
        "gemm",
        kernels={"cutedsl": cdsl_kernel, "torch": torch_kernel, "c2": c2_kernel_fn, "helion": helion_kernel},
        args=prob_args,
        arg_mask={"torch": (0, 1), "cutedsl": (0, 1), "c2": (0, 1), "helion": (0, 1)},
        baseline="torch",
        ref="torch")

    p.run(ncu=args.ncu, csv=args.csv)
