import torch

from cutedsl_kernels import Gemm4SM90
from workload_shapes import GEMM_ARGS_NT
from profile_utils import ProfilingJob, get_profiling_job_args
from cdsl_fn_utils import compile_cutedsl

from c2.hl.nodes import Builder, ForLoop, PersistentForLoopSM90
import c2.hl.user_nodes
from c2.hl.util import DType
import hlgraph.nodes as hlN
from hel import nodes
from hlgraph.rasterization_helper import PersistentRasterizationHelper
from compiler import compile_hel

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


def build_c2_gemm(m, n, k):
    """See compiler_2's python/kernels/hel/gemm.py."""
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
        with builder.enter_scope(ForLoop('k', 0, k // C2_TILE_K, 1)) as tk:
            dA = builder.hel_Load['dA', C2_TILE_M, C2_TILE_K](gA, tm, tk)
            dB = builder.hel_Load['dB', C2_TILE_N, C2_TILE_K](gB, tn, tk)
            builder.hel_AccumulatingGemm(dA, dB, acc)
        builder.hel_Store(acc, gC, tm, tn)

    return builder


if __name__ == "__main__":
    args = get_profiling_job_args()
    prob_args = GEMM_ARGS_NT.with_config(args.config) # Get set of tensors defining the problem

    # Compile CuteDSL manually
    compiled_gemm = compile_cutedsl(prob_args.tensors(), gemm, include_stream=False)

    m, n, k = prob_args.arg('m', 'n', 'k')
    c2_kernel = compile_hel(build_c2_gemm(m, n, k), name=f'gemm_{m}x{n}x{k}')

    def cdsl_kernel(a_: torch.Tensor, b_: torch.Tensor):
        o = torch.empty(a_.shape[0], b_.shape[0], dtype=torch.bfloat16, device='cuda')
        compiled_gemm(a_, b_, o)
        return o

    def c2_kernel_fn(a_: torch.Tensor, b_: torch.Tensor):
        o = torch.empty(a_.shape[0], b_.shape[0], dtype=torch.bfloat16, device='cuda')
        c2_kernel(a_, b_, o)
        return o

    p = ProfilingJob(
        "gemm",
        kernels={"cutedsl": cdsl_kernel, "torch": torch_kernel, "c2": c2_kernel_fn},
        args=prob_args,
        arg_mask={"torch": (0, 1), "cutedsl": (0, 1), "c2": (0, 1)},
        baseline="torch",
        ref="torch")

    p.run(ncu=args.ncu, csv=args.csv)
