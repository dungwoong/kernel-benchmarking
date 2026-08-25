import torch
from profile_utils import ProfilingJob, get_profiling_job_args
from workload_shapes import LORA_NT_16
from helion_utils.kernel_runner import LoRA

torch.manual_seed(18)

LORA_DIM = 16

@torch.compile
def torch_kernel(a: torch.Tensor, b: torch.Tensor, lA: torch.Tensor, lB: torch.Tensor):
    return (a @ b.t()) + ((a @ lA.t()) @ lB.t())

if __name__ == '__main__':
    args = get_profiling_job_args()
    prob_args = LORA_NT_16.with_config(args.config)
    helion_kernel = LoRA.compile(*prob_args.tensors((0, 1, 2, 3)))
    LoRA.dump_ir(helion_kernel, prob_args.tensors((0, 1, 2, 3)))

    p = ProfilingJob(
        "lora",
        kernels={"helion": helion_kernel, "torch": torch_kernel},
        args=prob_args,
        arg_mask={"torch": (0, 1, 2, 3), "helion": (0, 1, 2, 3)},
        baseline="torch",
        ref="torch",
    )
    p.run(ncu=args.ncu)
