import torch

def torch_lora(a: torch.Tensor, b: torch.Tensor, lA: torch.Tensor, lB: torch.Tensor):
    """
    Assumes everything is k-major.
    a  (m, k)
    b  (n, k)
    lA (l, k)
    lB (n, l)
    l = lora_dim
    """
    return (a @ b.t()) + ((a @ lA.t()) @ lB.t())