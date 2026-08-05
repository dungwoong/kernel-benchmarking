import torch

from cutedsl_kernels import LoRASM90
from workload_shapes import LORA_NT_16
from profile_utils import ProfilingJob, get_profiling_job_args, ExperimentOutput
from cdsl_fn_utils import compile_cutedsl
from trt_utils import build_trt_runner
from baselines.lora import LoraModule

"""
LoRA, uses lora_dim=16
"""
torch.manual_seed(18)

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
    trt_runner = build_trt_runner(
        module=LoraModule(),
        example_inputs=prob_args.tensors((0, 1, 2, 3)),
        output_shape=(m, n),
        cache_key=f"lora_m{m}_n{n}_k{k}_bf16",
        input_names=["a", "b", "lA", "lB"],
    )

    p = ProfilingJob(
        "lora",
        kernels={"cutedsl": cdsl_kernel, "torch": torch_kernel, 'trt': trt_runner},
        args=prob_args,
        arg_mask={"torch": (0, 1, 2, 3), "cutedsl": (0, 1, 2, 3)},
        baseline="torch",
        ref="torch")
    
    p.run(ncu=args.ncu)