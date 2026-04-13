import time
import torch
import math
import cutlass
from cutlass import cute
import cuda.bindings.driver as cuda

from profile_utils import ExperimentOutput, get_normal_bernoulli, get_kaiming, get_args
from cutedsl_kernels import SwigluSM90
from cdsl_fn_utils import make_fake_tensor, compile_cutedsl, STREAM

"""
Profiles torch swiglu(torch) + cutedsl kernel
"""
torch.manual_seed(18)

swiglu = SwigluSM90(
    tile_shape_mn=(128, 128), 
    epi_tile_mn=(128, 32),
    cluster_shape_mnk=(2, 1, 1), 
    atom_layout_mn=(2, 1),
    ab_stage=3,
    reuse_ab=False,
    is_persistent=True,
    gemm_n_prologue=0)

def torch_kernel(a: torch.Tensor, bb1: torch.Tensor):
    o = a @ bb1.t()
    o1, o2 = o.chunk(2, dim=1)
    return torch.nn.functional.silu(o1) * o2


if __name__ == "__main__":
    args = get_args()
    torch_output = ExperimentOutput('swiglu_torch', args.m, args.n, args.k)
    cdsl_output = ExperimentOutput('swiglu_cdsl', args.m, args.n, args.k)
    
    m, n, k = args.m, args.n, args.k
    
    a64 = get_normal_bernoulli((m, k), dtype=torch.float64)
    b64 = get_kaiming((n, k), dtype=torch.float64)
    b164 = get_kaiming((n, k), dtype=torch.float64)
    bb164 = torch.cat((b64, b164), dim=0) # 2n, k

    a = a64.to(torch.bfloat16)
    b = b64.to(torch.bfloat16)
    b1 = b164.to(torch.bfloat16)
    bb1 = bb164.to(torch.bfloat16)
    c = torch.empty((m, n), dtype=torch.bfloat16).to('cuda')
    tensors = (a, b, b1)
    torch_tensors = (a, bb1)
    compiled_gemm = compile_cutedsl((a, b, b1, c), swiglu)
    ref = torch_kernel(a64, bb164)

    def cdsl_kernel(a_: torch.Tensor, b_: torch.Tensor, b1_: torch.Tensor):
        o = torch.empty(a_.shape[0], b_.shape[0], dtype=torch.bfloat16, device='cuda')
        compiled_gemm(a_, b_, b1_, o, STREAM)
        return o
    
    torch_compiled = torch.compile(torch_kernel)
    
    cdsl_output.run(cdsl_kernel, tensors, ref)
    time.sleep(2)
    torch_output.run(torch_compiled, torch_tensors, ref)
    print(ExperimentOutput.header())
    print(torch_output.values())
    print(cdsl_output.values())
    print(torch_output.time_ms / cdsl_output.time_ms)