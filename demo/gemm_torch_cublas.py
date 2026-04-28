import time
import torch
import math
import cutlass
from cutlass import cute
import cuda.bindings.driver as cuda
from cublas_mm import matmul as cublas_matmul

from profile_utils import ExperimentOutput, get_normal_bernoulli, get_args
from cutedsl_kernels import Gemm4SM90
from cdsl_fn_utils import make_fake_tensor, compile_cutedsl, STREAM

"""
Profiles torch gemm(cuBLAS) + torch kernel
"""
torch.manual_seed(18)

def torch_kernel(a: torch.Tensor, b: torch.Tensor):
    return a @ b.t()


if __name__ == "__main__":
    args = get_args()
    torch_output = ExperimentOutput('gemm_torch', args.m, args.n, args.k)
    cublas_output = ExperimentOutput('gemm_cublas', args.m, args.n, args.k)
    
    m, n, k = args.m, args.n, args.k
    
    a = get_normal_bernoulli((m, k))
    b = get_normal_bernoulli((n, k))
    c = torch.empty((m, n), dtype=torch.bfloat16).to('cuda')
    tensors = (a, b)
    ref = a.to(torch.float64) @ b.to(torch.float64).t()
    
    torch_output.run(torch_kernel, tensors, ref)
    time.sleep(2)
    cublas_output.run(cublas_matmul, tensors, ref)
    
    if args.to_csv:
        print(ExperimentOutput.list_to_csv(torch_output.values()))
        print(ExperimentOutput.list_to_csv(cublas_output.values()))
    else:
        print(ExperimentOutput.header())
        print(torch_output.values())
        print(cublas_output.values())
        print(torch_output.ms_median / cublas_output.ms_median)