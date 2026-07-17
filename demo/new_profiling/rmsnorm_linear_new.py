import torch
from cutedsl_kernels import RMSNormLinear2SM90
from profile_utils import ProfilingJob, get_profiling_job_args
from workload_shapes import RMSNORM_LINEAR_ARGS_NT
from cdsl_fn_utils import make_fake_tensor, compile_cutedsl, STREAM
from trt_utils import build_trt_runner
from baselines.rmsnorm_swiglu_trt import RMSNormLinearModule
from triton.testing import do_bench
from triton_rmsnorm_linear import rmsnorm_linear_persistent, rmsnorm_linear_persistent_nn

"""
RMSNorm + Linear
"""
torch.manual_seed(18)

EPS = 1e-5

@torch.compile
def torch_kernel(a: torch.Tensor, b: torch.Tensor, eps: float=EPS):
    a_rms = torch.nn.functional.rms_norm(a, normalized_shape=(a.shape[1],), eps=eps)
    return a_rms @ b.t()

@torch.compile
def torch_gemm(a: torch.Tensor, b: torch.Tensor):
    return a @ b.t()

# Caching to minimize timing of allocating mem for contiguous() and for creating .t() multiple times
b_nn_transp_cache = {}
b_nn_transp_contig_cache = {}

def triton_nn_transp_kernel(a: torch.Tensor, b: torch.Tensor, eps: float=EPS):
    # b.data_ptr is just a key for the cache
    b_nn = b_nn_transp_cache.get(b.data_ptr())
    if b_nn is None:
        b_nn = b.t()
        b_nn_transp_cache[b.data_ptr()] = b_nn
    return rmsnorm_linear_persistent_nn(a, b_nn, eps)

def triton_nn_transp_contig_kernel(a: torch.Tensor, b: torch.Tensor, eps: float=EPS):
    b_nn = b_nn_transp_contig_cache.get(b.data_ptr())
    if b_nn is None:
        b_nn = b.t().contiguous()
        b_nn_transp_contig_cache[b.data_ptr()] = b_nn
    return rmsnorm_linear_persistent_nn(a, b_nn, eps)

def get_cdsl_kernel(a, b, *args, **kwargs):
    m, k = a.shape
    n = b.shape[0]

    if (m, n, k) == (4096, 1536, 7168):
        return RMSNormLinear2SM90(
            tile_shape_mnk=(192, 128, 64),
            epi_tile_mn=(192, 128),
            cluster_shape_mnk=(1, 2, 1),
            atom_layout_mn=(3, 1),
            ab_stage=4,
            epi_stage=1,
            is_persistent=True,
            gemm_n_prologue=0,
            pingpong=False,
        )
    
    return RMSNormLinear2SM90(
        tile_shape_mnk=(128, 256, 64),
        epi_tile_mn=(128, 64),
        cluster_shape_mnk=(1, 2, 1),
        atom_layout_mn=(2, 1),
        ab_stage=4,
        epi_stage=2,
        is_persistent=True,
        gemm_n_prologue=0,
        pingpong=False,
    )


if __name__ == "__main__":
    args = get_profiling_job_args()
    prob_args = RMSNORM_LINEAR_ARGS_NT.with_config(args.config) # Get set of tensors defining the problem

    # Compile CuteDSL manually
    kernel = get_cdsl_kernel(*prob_args.tensors())
    compiled_cdsl = compile_cutedsl(prob_args.tensors(), kernel, include_stream=False)

    def cdsl_kernel(a_: torch.Tensor, b_: torch.Tensor, eps: float=EPS):
        o = torch.empty(a_.shape[0], b_.shape[0], dtype=torch.bfloat16, device='cuda')
        compiled_cdsl(a_, b_, o, eps)
        return o
    
    # NOTE TRT initializes eps in the object
    m, n, k = prob_args.arg('m', 'n', 'k')
    trt_runner = build_trt_runner(
        module=RMSNormLinearModule(eps=EPS),
        example_inputs=prob_args.tensors((0, 1)),
        output_shape=(m, n),
        cache_key=f"rmsnorm_linear_m{m}_n{n}_k{k}_bf16",
        input_names=["a", "b"],
    )

    p = ProfilingJob(
        "rmsnorm_lin",
        kernels={
            "cutedsl": cdsl_kernel,
            "torch": torch_kernel,
            "trt": trt_runner,
            "triton_persistent_nt": rmsnorm_linear_persistent,
            "triton_persistent_nn_transp": triton_nn_transp_kernel,
            "triton_persistent_nn_transp_contig": triton_nn_transp_contig_kernel,
            "max": torch_gemm,
        },
        args=prob_args,
        arg_mask={
            "torch": (0, 1, 3),
            "cutedsl": (0, 1, 3),
            "trt": (0, 1),
            "triton_persistent_nt": (0, 1, 3),
            "triton_persistent_nn_transp": (0, 1, 3),
            "triton_persistent_nn_transp_contig": (0, 1, 3),
            "max": (0, 1),
        },
        baseline="torch",
        ref="torch")
    
    p.run(ncu=args.ncu)
