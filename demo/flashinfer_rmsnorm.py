import torch
import flashinfer

from profile_utils import ExperimentOutput, get_normal_bernoulli, get_args

torch.manual_seed(18)

EPS = 1e-5


def torch_kernel(x: torch.Tensor, w: torch.Tensor):
    return torch.nn.functional.rms_norm(
        x, normalized_shape=(x.shape[-1],), weight=w, eps=EPS
    )


def flashinfer_kernel(x: torch.Tensor, w: torch.Tensor):
    return flashinfer.norm.rmsnorm(x, w, eps=EPS)


if __name__ == "__main__":
    args = get_args()
    m, n, k = args.m, args.n, args.k

    torch_output = ExperimentOutput("rmsnorm_torch", m, n, k)
    fi_output = ExperimentOutput("rmsnorm_flashinfer", m, n, k)

    x64 = get_normal_bernoulli((m, k), dtype=torch.float64)
    w64 = get_normal_bernoulli((k,), dtype=torch.float64)
    x = x64.to(torch.bfloat16)
    w = w64.to(torch.bfloat16)
    tensors = (x, w)

    ref = torch_kernel(x64, w64)
    compiled_torch = torch.compile(torch_kernel)

    fi_output.run(flashinfer_kernel, tensors, ref)
    torch_output.run(compiled_torch, tensors, ref)

    if args.to_csv:
        print(ExperimentOutput.list_to_csv(torch_output.values()))
        print(ExperimentOutput.list_to_csv(fi_output.values()))
    else:
        print(ExperimentOutput.header())
        print(torch_output.values())
        print(fi_output.values())
        print(torch_output.ms_median / fi_output.ms_median)
