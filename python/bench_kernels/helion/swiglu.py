import torch
import helion.language as hl


def swiglu_kernel(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """
    performs swiglu
    expects (m, k), (n, k), (n, k) inputs
    x is (m, k)
    w1, w2 are (k, n)
    """
    w1, w2 = w1.t(), w2.t()
    m = x.shape[0]
    n = w1.shape[1]
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    for tile_i, tile_j in hl.tile([m, n]):
        gate_acc = hl.zeros([tile_i, tile_j], dtype=torch.float32)
        up_acc = hl.zeros([tile_i, tile_j], dtype=torch.float32)
        for tile_k in hl.tile(x.shape[1]):
            xTile = x[tile_i, tile_k]
            gate_acc = torch.addmm(gate_acc, xTile, w1[tile_k, tile_j])
            up_acc = torch.addmm(up_acc, xTile, w2[tile_k, tile_j])
        silu_gate = gate_acc * torch.sigmoid(gate_acc)
        # Convert to BF16 before writing to out
        out[tile_i, tile_j] = (silu_gate * up_acc).to(x.dtype)
    return out
