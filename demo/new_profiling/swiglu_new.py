import torch

from cutedsl_kernels import Swiglu3SM90 # RS
from cutedsl_kernels import Swiglu3SM90, Swiglu2SM90
from workload_shapes import SWIGLU_NT
from profile_utils import ProfilingJob, get_profiling_job_args
from cdsl_fn_utils import compile_cutedsl
from trt_utils import build_trt_runner
from baselines.rmsnorm_swiglu_trt import SwigluSeparateModule

from c2.hl.nodes import Builder, ForLoop, PersistentForLoopSM90
import c2.hl.user_nodes
from c2.hl.util import DType
import hlgraph.nodes as hlN
from hel import nodes
from hlgraph.rasterization_helper import PersistentRasterizationHelper
from compiler import compile_hel

"""
Profiles torch gemm vs cutedsl gemm

Uses new profiling setup
"""
torch.manual_seed(18)

C2_TILE_M, C2_TILE_N, C2_TILE_K = 128, 128, 64

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


def build_c2_swiglu(m, n, k):
    """See compiler_2's python/kernels/hel/swiglu.py."""
    raster = PersistentRasterizationHelper(m, n, C2_TILE_M, C2_TILE_N).with_block(8, 8)
    builder = Builder(_p=hlN.HLProgram(raster.grid_size(), 256, raster.cluster_shape, tma_stages=3))
    gA = builder.GlobalTensor['a', DType.bfloat16, (m, k)]()
    gB = builder.GlobalTensor['b', DType.bfloat16, (n, k)]()
    gB1 = builder.GlobalTensor['b1', DType.bfloat16, (n, k)]()
    gC = builder.GlobalTensor['c', DType.bfloat16, (m, n)]()

    with builder.enter_scope(PersistentForLoopSM90('sched_idx', raster.total_tiles(), raster.total_blocks(), block_dim=raster.block_dim())) as sched_idx:
        coords = builder.ScheduleFunction['sched_coord', *raster.layout()](sched_idx)
        tm = builder.SliceCoord[0](coords)
        tn = builder.SliceCoord[1](coords)
        acc = builder.hel_Zeros['acc', C2_TILE_M, C2_TILE_N, DType.float32]()
        acc1 = builder.hel_Zeros['acc1', C2_TILE_M, C2_TILE_N, DType.float32]()
        
        with builder.enter_scope(ForLoop('k', 0, k // C2_TILE_K, 1)) as tk:
            dA = builder.hel_Load['dA', C2_TILE_M, C2_TILE_K](gA, tm, tk)
            dB = builder.hel_Load['dB', C2_TILE_N, C2_TILE_K](gB, tn, tk)
            dB1 = builder.hel_Load['dB1', C2_TILE_N, C2_TILE_K](gB1, tn, tk)
            builder.hel_AccumulatingGemm(dA, dB, acc)
            builder.hel_AccumulatingGemm(dA, dB1, acc1)
        acc_silu = builder.hel_Elementwise['silu'](acc)
        acc_final = builder.hel_Tilewise['mul'](acc_silu, acc1)
        builder.hel_Store(acc_final, gC, tm, tn)

    return builder


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
    c2_kernel = compile_hel(build_c2_swiglu(m, n, k), name=f'swiglu_{m}x{n}x{k}')

    def c2_kernel_fn(a_: torch.Tensor, b_: torch.Tensor, b1_: torch.Tensor):
        o = torch.empty(a_.shape[0], b_.shape[0], dtype=torch.bfloat16, device='cuda')
        c2_kernel(a_, b_, b1_, o)
        return o

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
        kernels={"cutedsl": cdsl_kernel, "torch": torch_swiglu, "tensorrt": trt_runner, "max": torch_gemm, "c2": c2_kernel_fn},
        args=prob_args,
        arg_mask={"torch": (0, 1, 2), "cutedsl": (0, 1, 2), "tensorrt": (0, 1, 2), "max": (0, 1, 2), "c2": (0, 1, 2)},
        baseline="torch",
        ref="torch")
    
    p.run(ncu=args.ncu, csv=args.csv)