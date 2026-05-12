import torch
from torch import nn

from baselines.rmsnorm_swiglu import swiglu_concat


def _rmsnorm(x: torch.Tensor, eps: float) -> torch.Tensor:
    # aten::rms_norm has no ONNX exporter symbolic, so decompose
    # TODO confirm TRT re-fuses this (TRT_DEBUG=1, check Normalization layer).
    x_fp32 = x.to(torch.float32)
    variance = (x_fp32 * x_fp32).mean(dim=-1, keepdim=True)
    return (x_fp32 * torch.rsqrt(variance + eps)).to(x.dtype)


class RMSNormLinearModule(nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_norm = _rmsnorm(a, self.eps)
        return a_norm @ b.t()


class SwigluConcatModule(nn.Module):
    def forward(self, a: torch.Tensor, bb1: torch.Tensor) -> torch.Tensor:
        return swiglu_concat(a, bb1)
