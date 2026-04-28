import time
import torch
import math
import cutlass
from cutlass import cute
import cuda.bindings.driver as cuda

from profile_utils import ExperimentOutput, get_normal_bernoulli, get_args
from cutedsl_kernels import Gemm4SM90
from cdsl_fn_utils import make_fake_tensor, compile_cutedsl, STREAM

"""
Profiles torch gemm(cuBLAS) + cutedsl kernel
"""
torch.manual_seed(18)

gemm = Gemm4SM90(
    tile_shape_mnk=(128, 256, 32),
    epi_tile_mn=(128, 128),
    cluster_shape_mnk=(2, 1, 1),
    atom_layout_mn=(2, 1),
    ab_stage=6,
    epi_stage=2,
    is_persistent=True,
    gemm_n_prologue=1,
)

def torch_kernel(a: torch.Tensor, b: torch.Tensor):
    return a @ b.t()


if __name__ == "__main__":
    args = get_args()
    torch_output = ExperimentOutput('gemm_torch', args.m, args.n, args.k)
    cdsl_output = ExperimentOutput('gemm_cdsl', args.m, args.n, args.k)
    
    m, n, k = args.m, args.n, args.k
    
    a = get_normal_bernoulli((m, k))
    b = get_normal_bernoulli((n, k))
    c = torch.empty((m, n), dtype=torch.bfloat16).to('cuda')
    tensors = (a, b)
    compiled_gemm = compile_cutedsl((a, b, c), gemm, include_stream=False)
    ref = a.to(torch.float64) @ b.to(torch.float64).t()

    def cdsl_kernel(a_: torch.Tensor, b_: torch.Tensor):
        o = torch.empty(a_.shape[0], b_.shape[0], dtype=torch.bfloat16, device='cuda')
        compiled_gemm(a_, b_, o)
        return o
    
    cdsl_output.run(cdsl_kernel, tensors, ref)
    time.sleep(2)
    torch_output.run(torch_kernel, tensors, ref)
    
    if args.to_csv:
        print(ExperimentOutput.list_to_csv(torch_output.values()))
        print(ExperimentOutput.list_to_csv(cdsl_output.values()))
    else:
        print(ExperimentOutput.header())
        print(torch_output.values())
        print(cdsl_output.values())
        print(torch_output.ms_median / cdsl_output.ms_median)