import torch
from profile_utils import ProfilingJob, get_profiling_job_args
from workload_shapes import RMSNORM_LINEAR_ARGS_NT
from helion_utils.kernel_runner import RMSNormLinear

torch.manual_seed(18)

EPS = 1e-5

@torch.compile
def torch_kernel(a: torch.Tensor, b: torch.Tensor, eps: float=EPS):
    a_rms = torch.nn.functional.rms_norm(a, normalized_shape=(a.shape[1],), eps=eps)
    return a_rms @ b.t()

@torch.compile
def torch_gemm(a: torch.Tensor, b: torch.Tensor):
    return a @ b.t()

if __name__ == '__main__':
    args = get_profiling_job_args()
    prob_args = RMSNORM_LINEAR_ARGS_NT.with_config(args.config)
    compiled_rmsnorm_lin = RMSNormLinear.compile(*prob_args.tensors((0, 1, 3)))
    RMSNormLinear.dump_ir(compiled_rmsnorm_lin, prob_args.tensors((0, 1, 3)))

    p = ProfilingJob(
        "rmsnorm_lin",
        kernels={"helion": compiled_rmsnorm_lin, "torch": torch_kernel, "gemm": torch_gemm},
        args=prob_args,
        arg_mask={"torch": (0, 1, 3), "helion": (0, 1, 3), "gemm": (0, 1)},
        baseline="torch",
        ref="torch",
    )
    p.run(ncu=args.ncu)