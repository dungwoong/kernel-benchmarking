import torch

def swiglu_concat(a: torch.Tensor, bb1: torch.Tensor):
    """
    a   (m, k)
    bb1 (2n, k)
    """
    o = a @ bb1.t()
    o1, o2 = o.chunk(2, dim=1)
    return torch.nn.functional.silu(o1) * o2

def swiglu_separate(a: torch.Tensor, b: torch.Tensor, b1: torch.Tensor):
    """
    This version is less efficient,
    the first version is what you would see in torch
    a  (m, k)
    b  (n, k)
    b1 (n, k)
    """
    in1 = a @ b.t()
    in2 = a @ b1.t()
    return torch.nn.functional.silu(in1) * in2

# norms across the last dim of the tensor
def rmsnorm(a: torch.Tensor, eps=1e-5):
    return torch.nn.functional.rms_norm(a, normalized_shape=(a.shape[-1],), eps=eps)

def rmsnorm_linear(a: torch.Tensor, b: torch.Tensor, eps=1e-5):
    """
    a (m, k)
    b (n, k)
    """
    a_rms = rmsnorm(a, eps)
    return a_rms @ b.t()

def rmsnorm_swiglu(a: torch.Tensor, bb1: torch.Tensor, eps=1e-5):
    """
    a   (m, k)
    bb1 (2n, k)
    """
    a_rms = rmsnorm(a, eps)
    return swiglu_concat(a_rms, bb1)