import time
import torch
import math
import cutlass
from cutlass import cute
import cuda.bindings.driver as cuda

from profile_utils import ExperimentOutput, get_normal_bernoulli, get_args
from cutedsl_kernels import LoRASM90
from cdsl_fn_utils import make_fake_tensor, compile_cutedsl, STREAM

"""
Profiles torch gemm(cuBLAS) + cutedsl kernel
"""
torch.manual_seed(18)

def torch_kernel(a: torch.Tensor, b: torch.Tensor, lA: torch.Tensor, lB: torch.Tensor):
    return (a @ b.t()) + ((a @ lA.t()) @ lB.t())


if __name__ == "__main__":
    parser = get_args(parse=False)
    parser.add_argument("lora_dim", type=int)
    args = parser.parse_args()
    assert args.lora_dim in (16, 32, 64), "unsupported lora dim"

    metadata_str = f'lora_dim={args.lora_dim}'
    torch_output = ExperimentOutput('lora_torch', args.m, args.n, args.k, metadata=metadata_str)
    cdsl_output = ExperimentOutput('lora_cdsl', args.m, args.n, args.k, metadata=metadata_str)
    
    m, n, k = args.m, args.n, args.k
    lora_dim = args.lora_dim

    gemm = LoRASM90(
        tile_shape_mn=(128, 256),
        lora_dim=lora_dim,
        epi_tile_mn=(128, 32),
        cluster_shape_mnk=(2, 1, 1), 
        atom_layout_mn=(2, 1),
        ab_stage=3,
        reuse_ab=True,
        is_persistent=False)
    
    a64 = get_normal_bernoulli((m, k), dtype=torch.float64)
    b64 = get_normal_bernoulli((n, k), dtype=torch.float64)
    c = torch.empty((m, n), dtype=torch.bfloat16).to('cuda')
    lA64 = get_normal_bernoulli((lora_dim, k), dtype=torch.float64)
    lB64 = get_normal_bernoulli((n, lora_dim), dtype=torch.float64)

    ref = torch_kernel(a64, b64, lA64, lB64)
    a, b, lA, lB = (t.to(torch.bfloat16) for t in (a64, b64, lA64, lB64))
    tensors = (a, b, lA, lB)

    compiled_gemm = compile_cutedsl((a, b, a @ lA.t(), lB, c), gemm)
    torch_compiled = torch.compile(torch_kernel)

    def cdsl_kernel(a_: torch.Tensor, b_: torch.Tensor, lA_: torch.Tensor, lB_: torch.Tensor):
        o = torch.empty(a_.shape[0], b_.shape[0], dtype=torch.bfloat16, device='cuda')
        lxa = a_ @ lA_.t()
        compiled_gemm(a_, b_, lxa, lB_, o, STREAM)
        return o
    
    cdsl_output.run(cdsl_kernel, tensors, ref)
    time.sleep(2)
    torch_output.run(torch_compiled, tensors, ref)

    if args.to_csv:
        print(ExperimentOutput.list_to_csv(torch_output.values()))
        print(ExperimentOutput.list_to_csv(cdsl_output.values()))
    else:
        print(ExperimentOutput.header())
        print(torch_output.values())
        print(cdsl_output.values())
        print(torch_output.q50_ms / cdsl_output.q50_ms)