import time
import torch
import math
import cutlass
from cutlass import cute
import cuda.bindings.driver as cuda

from profile_utils import ExperimentOutput, get_normal_bernoulli, get_args
from cutedsl_kernels import RMSNormLinear2SM90
from cdsl_fn_utils import make_fake_tensor, compile_cutedsl, STREAM
from trt_utils import build_trt_runner
from baselines.rmsnorm_swiglu_trt import RMSNormLinearModule
from triton.testing import do_bench

"""
Profiles RMSNorm + Linear kernels
- you get around 1.15x speedup on mnk2048
- however, you do get an increase in RMSE, so we need to try lowering that
"""
torch.manual_seed(18)

EPS = 1e-5

gemm = RMSNormLinear2SM90(
    tile_shape_mnk=(128, 256, 64),
    epi_tile_mn=(128, 64),
    cluster_shape_mnk=(1, 2, 1),
    atom_layout_mn=(2, 1),
    ab_stage=4,
    epi_stage=2,
    is_persistent=True,
    gemm_n_prologue=0,
    pingpong=False,
)

def torch_kernel(a: torch.Tensor, b: torch.Tensor, eps: float=EPS):
    a_rms = torch.nn.functional.rms_norm(a, normalized_shape=(a.shape[1],), eps=eps)
    return a_rms @ b.t()

@torch.compile
def rmsnorm_kernel(a: torch.Tensor, eps: float=EPS):
    return torch.nn.functional.rms_norm(a, normalized_shape=(a.shape[1],), eps=eps)


if __name__ == "__main__":
    args = get_args()
    torch_output = ExperimentOutput('rmsnorm_lin_torch', args.m, args.n, args.k)
    cdsl_output = ExperimentOutput('rmsnorm_lin_cdsl', args.m, args.n, args.k)
    trt_output = ExperimentOutput('rmsnorm_lin_trt', args.m, args.n, args.k)
    max_output = ExperimentOutput('rmsnorm_lin_max', args.m, args.n, args.k)

    m, n, k = args.m, args.n, args.k
    
    a64 = get_normal_bernoulli((m, k), dtype=torch.float64)
    b64 = get_normal_bernoulli((n, k), dtype=torch.float64)
    a, b = (t.to(torch.bfloat16) for t in (a64, b64))
    c = torch.empty((m, n), dtype=torch.bfloat16).to('cuda')
    tensors = (a, b)
    compiled_gemm = compile_cutedsl((a, b, c, EPS), gemm, False)
    ref = torch_kernel(a64, b64)
    ref_gemm = a64 @ b64.t()
    
    compiled_torch = torch.compile(torch_kernel)

    def cdsl_kernel(a_: torch.Tensor, b_: torch.Tensor, eps=EPS):
        o = torch.empty(a_.shape[0], b_.shape[0], dtype=torch.bfloat16, device='cuda')
        compiled_gemm(a_, b_, o, eps)
        return o
    
    def gemm(a_, b_):
        return a_ @ b_.t()
    
    trt_runner = build_trt_runner(
        module=RMSNormLinearModule(eps=EPS),
        example_inputs=(a, b),
        output_shape=(m, n),
        cache_key=f"rmsnorm_linear_m{m}_n{n}_k{k}_bf16",
        input_names=["a", "b"],
    )

    cdsl_output.run(cdsl_kernel, tensors, ref)
    time.sleep(2)
    trt_output.run(trt_runner, tensors, ref)
    time.sleep(2)
    torch_output.run(compiled_torch, tensors, ref)
    time.sleep(2)
    max_output.run(gemm, tensors, ref_gemm)

    if args.to_csv:
        print(ExperimentOutput.list_to_csv(torch_output.values()))
        print(ExperimentOutput.list_to_csv(cdsl_output.values()))
        print(ExperimentOutput.list_to_csv(trt_output.values()))
        print(ExperimentOutput.list_to_csv(max_output.values()))
    else:
        print(ExperimentOutput.header())
        print(torch_output.values())
        print(cdsl_output.values())
        print(trt_output.values())
        print(torch_output.ms_median / cdsl_output.ms_median)

        # Check max attainable speedup (rmsnorm + gemm / gemm)
        rmsnorm_ms = do_bench(lambda: rmsnorm_kernel(a))
        gemm_ms = do_bench(lambda: a @ b.t())
        print(f'{rmsnorm_ms=}, {gemm_ms=}. Max speedup = {torch_output.ms_median / gemm_ms}')