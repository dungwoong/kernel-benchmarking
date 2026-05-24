import torch

from cutedsl_kernels import Swiglu3SM90 # RS
from cutedsl_kernels import Swiglu3SM90, Swiglu2SM90
from workload_shapes import SWIGLU_NT
from profile_utils import ProfilingJob, get_profiling_job_args
from cdsl_fn_utils import compile_cutedsl
from trt_utils import build_trt_runner
from baselines.rmsnorm_swiglu_trt import SwigluSeparateModule

"""
Profiles torch gemm vs cutedsl gemm

Uses new profiling setup
"""
torch.manual_seed(18)

# can't do torch.compile
def torch_gemm(a: torch.Tensor, b: torch.Tensor, b1: torch.Tensor):
    a @ b1.t()
    return a @ b.t()

@torch.compile
def torch_swiglu(a, b, b1): # no concat
        o1 = a @ b.t()
        o2 = a @ b1.t()
        return torch.nn.functional.silu(o1) * o2

cdsl_swiglu = Swiglu3SM90(
    tile_shape_mnk=(128, 128, 64),
    epi_tile_mn=(128, 32),
    cluster_shape_mnk=(2, 1, 1),
    atom_layout_mn=(2, 1),
    ab_stage=3,
    epi_stage=2,
    reuse_ab=False,
    is_persistent=True,
    gemm_n_prologue=0,
)


if __name__ == "__main__":
    args = get_profiling_job_args()
    prob_args = SWIGLU_NT.with_config(args.config) # Get set of tensors defining the problem

    # Compile CuteDSL manually
    compiled_cdsl = compile_cutedsl(prob_args.tensors(), cdsl_swiglu, include_stream=False)

    def cdsl_kernel(a_: torch.Tensor, b_: torch.Tensor, b1_: torch.Tensor):
        o = torch.empty(a_.shape[0], b_.shape[0], dtype=torch.bfloat16, device='cuda')
        compiled_cdsl(a_, b_, b1_, o)
        return o
    
    m, n, k = prob_args.arg('m', 'n', 'k')
    trt_tensors = prob_args.tensors((0, 1, 2))
    trt_runner = build_trt_runner(
        module=SwigluSeparateModule(),
        example_inputs=trt_tensors,
        output_shape=(m, n),
        cache_key=f"swiglu_separate_m{m}_n{n}_k{k}_bf16",
        input_names=["a", "b", "b1"],
    )
    print('built TRT')

    p = ProfilingJob(
        "swiglu",
        kernels={"cutedsl": cdsl_kernel, "torch": torch_swiglu, "tensorrt": trt_runner, "max": torch_gemm},
        args=prob_args,
        arg_mask={"torch": (0, 1, 2), "cutedsl": (0, 1, 2), "tensorrt": (0, 1, 2), "max": (0, 1, 2)},
        baseline="torch",
        ref="torch")
    
    p.run(ncu=args.ncu)