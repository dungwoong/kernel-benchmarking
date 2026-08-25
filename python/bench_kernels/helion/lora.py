import torch
import helion.language as hl


# Validated against CuTE DSL rmse
# From previous library helion_kernels.py
def lora_kernel(
    x: torch.Tensor,
    W: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,  # pre-computed x @ B.T, passed in to keep @ outside helion
) -> torch.Tensor:
    """
    Computes xW + (xA)B

    x: m, k
    w: n, k --> k, n
    A: lora_dim, k --> k, lora_dim
    B: n, lora_dim --> lora_dim, n

    Helion ships no lora example, so this has no official counterpart.
    """
    W = W.t()
    B = B.t()
    m, k = x.shape
    n = W.shape[1]
    xA = x @ A.t() # m, lora_dim
    out = torch.empty(m, n, dtype=x.dtype, device=x.device)

    for tile_i, tile_j in hl.tile([m, n]):
        acc = hl.zeros([tile_i, tile_j], dtype=torch.float32)
        for tile_k in hl.tile(k):
            # accumulation can also be written with hl.dot (previous lib also autotuned that)
            acc = torch.addmm(acc, x[tile_i, tile_k], W[tile_k, tile_j])
        acc = torch.addmm(acc, xA[tile_i, :], B[:, tile_j])
        out[tile_i, tile_j] = acc.to(x.dtype)
    return out
