import torch
from profile_utils import ProfilingJob, get_profiling_job_args
from workload_shapes import GEMM_ARGS_NT
from helion_utils.kernel_runner import Matmul

torch.manual_seed(18)

EPS = 1e-5

@torch.compile
def torch_gemm(a: torch.Tensor, b: torch.Tensor):
    return a @ b.t()

if __name__ == '__main__':
    args = get_profiling_job_args()
    prob_args = GEMM_ARGS_NT.with_config(args.config)
    compiled_matmul = Matmul.compile(*prob_args.tensors((0, 1)))
    Matmul.dump_ir(compiled_matmul, prob_args.tensors((0, 1)))

    p = ProfilingJob(
        "matmul",
        kernels={"helion": compiled_matmul, "torch": torch_gemm},
        args=prob_args,
        arg_mask={"torch": (0, 1), "helion": (0, 1)},
        baseline="torch",
        ref="torch",
    )
    p.run(ncu=args.ncu)