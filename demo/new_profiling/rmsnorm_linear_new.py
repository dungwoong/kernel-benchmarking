import torch
from cutedsl_kernels import RMSNormLinear2SM90
from profile_utils import ProfilingJob, get_profiling_job_args
from workload_shapes import RMSNORM_LINEAR_ARGS_NT
from cdsl_fn_utils import make_fake_tensor, compile_cutedsl, STREAM
from trt_utils import build_trt_runner
from baselines.rmsnorm_swiglu_trt import RMSNormLinearModule
from triton.testing import do_bench

from c2.hl.nodes import Builder, ForLoop, PersistentForLoopSM90
import c2.hl.user_nodes
from c2.hl.util import DType
import hlgraph.nodes as hlN
from hel import nodes
from hlgraph.rasterization_helper import PersistentRasterizationHelper
from compiler import compile_hel

"""
RMSNorm + Linear
"""
torch.manual_seed(18)

EPS = 1e-5

C2_TILE_M, C2_TILE_N, C2_TILE_K = 128, 256, 64

@torch.compile
def torch_kernel(a: torch.Tensor, b: torch.Tensor, eps: float=EPS):
    a_rms = torch.nn.functional.rms_norm(a, normalized_shape=(a.shape[1],), eps=eps)
    return a_rms @ b.t()

@torch.compile
def torch_gemm(a: torch.Tensor, b: torch.Tensor):
    return a @ b.t()

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


def build_c2_rmsnorm_linear(m, n, k, eps=EPS):
    """See compiler_2's python/kernels/hel/rmsnorm_linear.py
    """
    raster = PersistentRasterizationHelper(m, n, C2_TILE_M, C2_TILE_N).with_block(8, 8)
    builder = Builder(_p=hlN.HLProgram(raster.grid_size(), 256, raster.cluster_shape, tma_stages=3))
    gA = builder.GlobalTensor['a', DType.bfloat16, (m, k)]()
    gB = builder.GlobalTensor['b', DType.bfloat16, (n, k)]()
    gC = builder.GlobalTensor['c', DType.bfloat16, (m, n)]()

    with builder.enter_scope(PersistentForLoopSM90('sched_idx', raster.total_tiles(), raster.total_blocks(), block_dim=raster.block_dim())) as sched_idx:
        coords = builder.ScheduleFunction['sched_coord', *raster.layout()](sched_idx)
        tm = builder.SliceCoord[0](coords)
        tn = builder.SliceCoord[1](coords)
        acc = builder.hel_Zeros['acc', C2_TILE_M, C2_TILE_N, DType.float32]()
        sum_acc = builder.hel_Zeros['sum_acc', C2_TILE_M, 1, DType.float32]()

        with builder.enter_scope(ForLoop('k', 0, k // C2_TILE_K, 1)) as tk:
            dA = builder.hel_Load['dA', C2_TILE_M, C2_TILE_K](gA, tm, tk)
            dB = builder.hel_Load['dB', C2_TILE_N, C2_TILE_K](gB, tn, tk)
            builder.hel_AccumulatingGemm(dA, dB, acc)

            dAsq = builder.hel_Elementwise['square'](dA)
            builder.hel_RowsumAccumulating(dAsq, sum_acc)
        s_div = builder.hel_Elementwise['div', k](sum_acc)
        s_add = builder.hel_Elementwise['add', eps](s_div)
        s_rsqrt = builder.hel_Elementwise['rsqrt'](s_add)
        acc_scaled = builder.hel_RowBcastOp['mul'](acc, s_rsqrt)
        builder.hel_Store(acc_scaled, gC, tm, tn)

    return builder


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
    # NOTE c2 sets eps in the program
    c2_kernel = compile_hel(build_c2_rmsnorm_linear(m, n, k), name=f'rmsnorm_linear_{m}x{n}x{k}')

    def c2_kernel_fn(a_: torch.Tensor, b_: torch.Tensor):
        o = torch.empty(a_.shape[0], b_.shape[0], dtype=torch.bfloat16, device='cuda')
        c2_kernel(a_, b_, o)
        return o

    trt_runner = build_trt_runner(
        module=RMSNormLinearModule(eps=EPS),
        example_inputs=prob_args.tensors((0, 1)),
        output_shape=(m, n),
        cache_key=f"rmsnorm_linear_m{m}_n{n}_k{k}_bf16",
        input_names=["a", "b"],
    )

    p = ProfilingJob(
        "rmsnorm_lin",
        kernels={"cutedsl": cdsl_kernel, "torch": torch_kernel, 'trt': trt_runner, 'max': torch_gemm, 'c2': c2_kernel_fn},
        args=prob_args,
        arg_mask={"torch": (0, 1, 3), "cutedsl": (0, 1, 3), "trt": (0, 1), 'max': (0, 1), 'c2': (0, 1)},
        baseline="torch",
        ref="torch")
    
    p.run(ncu=args.ncu, csv=args.csv)