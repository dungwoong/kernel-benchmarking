import torch

def simple_gla(q, k, v, scale=1):
    """
    q, k are (B, T, H, K) where T is seqlen, K is dim
    v is (B, T, H, V)
    """
    
    # Align for batch matrix multiplication: (B, H, T, K)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    
    # 1. (QK^T)
    # Result: (B, H, T, T)
    A = (q @ k.transpose(-1, -2)) * scale
    
    # 2. Mask M where M_ij = 1 if i >= j else 0
    # torch.tril includes the diagonal (i == j)
    M = torch.tril(torch.ones((q.shape[-2], q.shape[-2]), dtype=q.dtype, device=q.device))
    
    # 3. Element-wise product (Hadamard) and V multiply
    # O = (A * M) @ V
    o = (A * M) @ v
    
    # Restore to (B, T, H, V)
    return o.transpose(1, 2)