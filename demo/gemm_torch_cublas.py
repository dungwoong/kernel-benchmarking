import time
import torch
from cublas_mm import matmul as cublas_matmul

from profile_utils import ExperimentOutput, get_normal_bernoulli, get_args

"""
Profiles torch gemm(cuBLAS) + torch kernel, looking for discrepancies in timings
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