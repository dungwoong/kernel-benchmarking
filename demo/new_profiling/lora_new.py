import torch

from cutedsl_kernels import LoRASM90
from workload_shapes import LORA_NT_16
from profile_utils import ProfilingJob, get_profiling_job_args, ExperimentOutput
from cdsl_fn_utils import compile_cutedsl
from trt_utils import build_trt_runner
from baselines.lora import LoraModule

from c2.hl.nodes import Builder, ForLoop, PersistentForLoopSM90, ConstantExpr
import c2.hl.user_nodes
from c2.hl.util import DType
import hlgraph.nodes as hlN
from hel import nodes
from hlgraph.rasterization_helper import PersistentRasterizationHelper
from compiler import compile_hel

"""
LoRA, uses lora_dim=16
"""
torch.manual_seed(18)

LORA_DIM = 16
C2_TILE_M, C2_TILE_N, C2_TILE_K = 128, 256, 64

@torch.compile
def torch_kernel(a: torch.Tensor, b: torch.Tensor, lA: torch.Tensor, lB: torch.Tensor):
    return (a @ b.t()) + ((a @ lA.t()) @ lB.t())

gemm = LoRASM90(
    tile_shape_mn=(128, 256),
    lora_dim=16,
    epi_tile_mn=(128, 32),
    cluster_shape_mnk=(2, 1, 1), 
    atom_layout_mn=(2, 1),
    ab_stage=3,
    reuse_ab=True,
    is_persistent=False)


def build_c2_lora(m, n, k):
    """See compiler_2's python/kernels/hel/lora.py.
    """
    raster = PersistentRasterizationHelper(m, n, C2_TILE_M, C2_TILE_N).with_block(8, 8)
    builder = Builder(_p=hlN.HLProgram(raster.grid_size(), 256, raster.cluster_shape, tma_stages=3))
    
    gA = builder.GlobalTensor['a', DType.bfloat16, (m, k)]()
    gB = builder.GlobalTensor['b', DType.bfloat16, (n, k)]()
    gXA = builder.GlobalTensor['xa', DType.bfloat16, (m, LORA_DIM)]()
    gLB = builder.GlobalTensor['lora_b', DType.bfloat16, (n, LORA_DIM)]()
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
        # the lora update is a single k step, so it sits outside the k loop
        dxA = builder.hel_Load['dxA', C2_TILE_M, LORA_DIM](gXA, tm, ConstantExpr(0))
        dlB = builder.hel_Load['dLB', C2_TILE_N, LORA_DIM](gLB, tn, ConstantExpr(0))
        builder.hel_AccumulatingGemm(dxA, dlB, acc)
        builder.hel_Store(acc, gC, tm, tn)

    return builder


if __name__ == "__main__":
    args = get_profiling_job_args()
    prob_args = LORA_NT_16.with_config(args.config)

    # Compile CuteDSL manually
    xTensor, aTensor = prob_args.tensors((0, 2))
    xA = xTensor @ aTensor.t()
    cdsl_tensors = prob_args.tensors((0, 1)) + (xA,) + prob_args.tensors((3, 4))
    compiled_gemm = compile_cutedsl(cdsl_tensors, gemm, include_stream=False)

    def cdsl_kernel(a_: torch.Tensor, b_: torch.Tensor, lA_: torch.Tensor, lB_: torch.Tensor):
        o = torch.empty(a_.shape[0], b_.shape[0], dtype=torch.bfloat16, device='cuda')
        lxa = a_ @ lA_.t()
        compiled_gemm(a_, b_, lxa, lB_, o)
        return o
    
    m, n, k = prob_args.arg('m', 'n', 'k')
    c2_kernel = compile_hel(build_c2_lora(m, n, k), name=f'lora_{m}x{n}x{k}')

    def c2_kernel_fn(a_: torch.Tensor, b_: torch.Tensor, lA_: torch.Tensor, lB_: torch.Tensor):
        o = torch.empty(a_.shape[0], b_.shape[0], dtype=torch.bfloat16, device='cuda')
        lxa = a_ @ lA_.t()
        c2_kernel(a_, b_, lxa, lB_, o)
        return o

    trt_runner = build_trt_runner(
        module=LoraModule(),
        example_inputs=prob_args.tensors((0, 1, 2, 3)),
        output_shape=(m, n),
        cache_key=f"lora_m{m}_n{n}_k{k}_bf16",
        input_names=["a", "b", "lA", "lB"],
    )

    p = ProfilingJob(
        "lora",
        kernels={"cutedsl": cdsl_kernel, "torch": torch_kernel, 'trt': trt_runner, 'c2': c2_kernel_fn},
        args=prob_args,
        arg_mask={"torch": (0, 1, 2, 3), "cutedsl": (0, 1, 2, 3), 'c2': (0, 1, 2, 3)},
        baseline="torch",
        ref="torch")
    
    p.run(ncu=args.ncu, csv=args.csv)