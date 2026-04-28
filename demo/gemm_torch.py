import time
import torch
import math

from profile_utils import TimingComparisonOutput, get_normal_bernoulli, get_args

"""
Profiles torch gemm(cuBLAS) + cutedsl kernel
"""
torch.manual_seed(18)

def torch_kernel(a: torch.Tensor, b: torch.Tensor):
    return a @ b.t()


if __name__ == "__main__":
    args = get_args()
    torch_output = TimingComparisonOutput('gemm_torch_cuda_timing', args.m, args.n, args.k)
    
    m, n, k = args.m, args.n, args.k
    
    a = get_normal_bernoulli((m, k))
    b = get_normal_bernoulli((n, k))
    tensors = (a, b)
    ref = a.to(torch.float64) @ b.to(torch.float64).t()
    torch_output.run(torch_kernel, tensors, ref)
    
    if args.to_csv:
        print(TimingComparisonOutput.list_to_csv(torch_output.values()))
    else:
        # print(TimingComparisonOutput.header())
        # print(torch_output.values())
        print(torch_output.ms_median, torch_output.ms_median_2)
        print(torch_output.ms_mean, torch_output.ms_mean_2)