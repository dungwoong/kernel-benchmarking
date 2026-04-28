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
    
    m, n, k = args.m, args.n, args.k
    
    a = get_normal_bernoulli((m, k), ncu=True)
    b = get_normal_bernoulli((n, k), ncu=True)
    c = torch.empty((m, n), dtype=torch.bfloat16).to('cuda')
    tensors = (a, b)
    compiled_gemm = compile_cutedsl((a, b, c), gemm, include_stream=False)

    def cdsl_kernel(a_: torch.Tensor, b_: torch.Tensor):
        o = torch.empty(a_.shape[0], b_.shape[0], dtype=torch.bfloat16).to('cuda')
        compiled_gemm(a_, b_, o)
        return o
    
    torch_kernel(*tensors)
    cdsl_kernel(*tensors)
    cublas_matmul(*tensors)