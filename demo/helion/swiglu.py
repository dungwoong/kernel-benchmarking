import torch
from profile_utils import ProfilingJob, get_profiling_job_args
from workload_shapes import SWIGLU_NT
from helion_kernels.kernel_runner import SwiGLU

torch.manual_seed(18)

EPS = 1e-5

@torch.compile
def torch_unfused(a: torch.Tensor, b: torch.Tensor, b1: torch.Tensor):
    in1 = a @ b.t()
    in2 = a @ b1.t()
    return torch.nn.functional.silu(in1) * in2

@torch.compile
def torch_gemm(a: torch.Tensor, b: torch.Tensor, b1: torch.Tensor):
    return a @ b.t() + a @ b1.t()

if __name__ == '__main__':
    args = get_profiling_job_args()
    prob_args = SWIGLU_NT.with_config(args.config)
    compiled_swiglu = SwiGLU.compile(*prob_args.tensors((0, 1, 2)))
    SwiGLU.dump_ir(compiled_swiglu, prob_args.tensors((0, 1, 2)))

    p = ProfilingJob(
        "rmsnorm_lin",
        kernels={"helion": compiled_swiglu, "torch": torch_unfused, "gemm": torch_gemm},
        args=prob_args,
        arg_mask={"torch": (0, 1, 2), "helion": (0, 1, 2), "gemm": (0, 1, 2)},
        baseline="torch",
        ref="torch",
    )
    p.run(ncu=args.ncu)