import torch

from cutedsl_kernels import LoRASM90
from workload_shapes import LORA_NT_16
from profile_utils import ProfilingJob, get_profiling_job_args, ExperimentOutput
from cdsl_fn_utils import compile_cutedsl
from trt_utils import build_trt_runner
from baselines.lora import LoraModule

from kernels.hel.lora import get_kernel as get_c2_kernel
from compiler import compile_hel
from helion_utils.kernel_runner import LoRA

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
    c2_kernel = compile_hel(get_c2_kernel(m, n, k, C2_TILE_M, C2_TILE_N, C2_TILE_K, LORA_DIM), name=f'lora_{m}x{n}x{k}')

    def c2_kernel_fn(a_: torch.Tensor, b_: torch.Tensor, lA_: torch.Tensor, lB_: torch.Tensor):
        o = torch.empty(a_.shape[0], b_.shape[0], dtype=torch.bfloat16, device='cuda')
        lxa = a_ @ lA_.t()
        c2_kernel(a_, b_, lxa, lB_, o)
        return o

    # Helion loads tuned config from helion_utils/autotune_cache
    helion_kernel = LoRA.compile(*prob_args.tensors((0, 1, 2, 3)))

    trt_runner = build_trt_runner(
        module=LoraModule(),
        example_inputs=prob_args.tensors((0, 1, 2, 3)),
        output_shape=(m, n),
        cache_key=f"lora_m{m}_n{n}_k{k}_bf16",
        input_names=["a", "b", "lA", "lB"],
    )

    p = ProfilingJob(
        "lora",
        kernels={"cutedsl": cdsl_kernel, "torch": torch_kernel, 'trt': trt_runner, 'c2': c2_kernel_fn, 'helion': helion_kernel},
        args=prob_args,
        arg_mask={"torch": (0, 1, 2, 3), "cutedsl": (0, 1, 2, 3), 'c2': (0, 1, 2, 3), 'helion': (0, 1, 2, 3)},
        baseline="torch",
        ref="torch")
    
    p.run(ncu=args.ncu, csv=args.csv)