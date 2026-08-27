import torch
import helion.language as hl


def rmsnorm_lin_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float=1e-5,
) -> torch.Tensor:
    """
    performs rmsnorm(x) y
    x is (m, k), y is (k, n)
    """
    y = y.t()
    m, k = x.size()
    k2, n = y.size()
    out = torch.empty(
        [m, n], dtype=x.dtype, device=x.device
    )
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        sum_acc = hl.zeros([tile_m], dtype=torch.float32)
        for tile_k in hl.tile(k):
            xTile = x[tile_m, tile_k]
            # accumulation can also be written with hl.dot
            acc = torch.addmm(acc, xTile, y[tile_k, tile_n])

            # This is taken from rmsnorm tutorial
            x_squared = xTile * xTile
            sum_acc += torch.sum(x_squared, dim=-1)

        inv_rms = torch.rsqrt((sum_acc / k) + eps)
        normalized = acc * inv_rms[:, None]
        out[tile_m, tile_n] = normalized
    return out
