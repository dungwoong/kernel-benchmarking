import time
import torch
import math
import cutlass
from cutlass import cute
import cuda.bindings.driver as cuda

from profile_utils import ExperimentOutput, get_normal_bernoulli, get_args
from cutedsl_kernels import Gemm1SM90
from cdsl_fn_utils import make_fake_tensor, compile_cutedsl, STREAM

"""
Profiles torch gemm(cuBLAS) + cutedsl kernel
"""
torch.manual_seed(18)

gemm = Gemm1SM90(
    tile_shape_mn=(128, 256), 
    epi_tile_mn=(128, 32),
    cluster_shape_mnk=(2, 1, 1), 
    atom_layout_mn=(2, 1),
    ab_stage=3,
    reuse_ab=False,
    is_persistent=True)

def torch_kernel(a: torch.Tensor, b: torch.Tensor):
    return a @ b.t()


if __name__ == "__main__":
    args = get_args()
    torch_output = ExperimentOutput('torch', args.m, args.n, args.k)
    cdsl_output = ExperimentOutput('cdsl', args.m, args.n, args.k)
    
    m, n, k = args.m, args.n, args.k
    
    a = get_normal_bernoulli((m, k))
    b = get_normal_bernoulli((n, k))
    c = torch.empty((m, n), dtype=torch.bfloat16).to('cuda')
    tensors = (a, b)
    compiled_gemm = compile_cutedsl((a, b, c), gemm)
    ref = a.to(torch.float64) @ b.to(torch.float64).t()

    def cdsl_kernel(a_: torch.Tensor, b_: torch.Tensor):
        o = torch.empty(a_.shape[0], b_.shape[0], dtype=torch.bfloat16, device='cuda')
        compiled_gemm(a_, b_, o, STREAM)
        return o
    
    cdsl_output.run(cdsl_kernel, tensors, ref)
    time.sleep(2)
    torch_output.run(torch_kernel, tensors, ref)
    print(ExperimentOutput.header())
    print(torch_output.values())
    print(cdsl_output.values())
    print(torch_output.time_ms / cdsl_output.time_ms)