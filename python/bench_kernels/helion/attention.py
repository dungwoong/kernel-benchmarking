import math
import torch
import helion
import helion.language as hl


# Implementation similar to Helion's example
# https://helionlang.com/examples/attention.html
def attention_kernel_fn(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
) -> torch.Tensor:
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))

    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim]).transpose(1, 2)
    out = torch.empty_like(q_view)

    # similar to reference
    sm_scale = 1.0 / math.sqrt(head_dim)
    # multiply by log2(e) so we can use exp2 instead of exp (faster on hardware)
    qk_scale = sm_scale * 1.44269504

    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        q = q_view[tile_b, tile_m, :]

        for tile_n in hl.tile(v_view.size(1)):
            k = k_view[tile_b, :, tile_n]
            qk = torch.bmm(q, k)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, :, None]
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            v = v_view[tile_b, tile_n, :]
            p = p.to(v.dtype)
            acc = torch.baddbmm(acc, p, v)
            m_i = m_ij

        m_i += torch.log2(l_i)
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)

    return out.view(q_in.size())


# Constrained search space, unused by default. Matters for larger batch sizes.
SAFE_CONFIGS = [
    helion.Config(block_sizes=[1, 64, 64]),
    helion.Config(block_sizes=[1, 64, 128]),
    helion.Config(block_sizes=[1, 32, 64]),
    helion.Config(block_sizes=[1, 32, 128]),
    helion.Config(block_sizes=[2, 64, 64]),
    helion.Config(block_sizes=[2, 32, 64]),
]
