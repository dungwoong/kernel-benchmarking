import torch
from .rmsnorm_swiglu import rmsnorm

def ntrans(H: torch.Tensor, X: torch.Tensor, alpha: torch.Tensor):
    H_norm = rmsnorm(H)
    A = H_norm - X
    B = alpha * A # mirage puts X here, that is incorrect
    C = X + B
    O = rmsnorm(C)
    return O

def example():
    batch = 128
    H = torch.randn((batch, 4096), dtype=torch.bfloat16, device='cuda')
    X = torch.randn_like(H)
    alpha = torch.rand_like(H)
    ntrans(H, X, alpha)