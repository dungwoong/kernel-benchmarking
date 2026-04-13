import torch
import math
import cutlass
from cutlass import cute
import cuda.bindings.driver as cuda

from profile_utils import ExperimentOutput, get_normal_bernoulli, get_args
from cutedsl_kernels import Gemm1SM90
from cdsl_fn_utils import make_fake_tensor

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

    # Compile CuteDSL kernel
    # TODO add helper functions for this later
    dtype = cutlass.BFloat16
    div = math.gcd(128//dtype.width, k)
    divn = math.gcd(128//dtype.width, n)
    fA = make_fake_tensor(dtype, (m, k), div)
    fB = make_fake_tensor(dtype, (n, k), div)
    fC = make_fake_tensor(dtype, (m, n), divn)
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled_gemm = cute.compile(gemm, fA, fB, fC, current_stream, options='--enable-tvm-ffi')
    
    def cdsl_kernel(a_: torch.Tensor, b_: torch.Tensor):
        o = torch.empty(a_.shape[0], b_.shape[0], dtype=torch.bfloat16, device='cuda')
        compiled_gemm(a_, b_, o, current_stream)
        return o
    
    a = get_normal_bernoulli((m, k))
    b = get_normal_bernoulli((n, k))
    print(a.dtype)
    tensors = (a, b)
    ref = a.to(torch.float64) @ b.to(torch.float64).t()
    
    torch_output.run(torch_kernel, tensors, ref)
    cdsl_output.run(cdsl_kernel, tensors, ref)
    print(ExperimentOutput.header())
    print(torch_output.values())
    print(cdsl_output.values())