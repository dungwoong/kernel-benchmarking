"""
Decode attention (q_len = 1) forward, non-causal: one query token per head attends to a long K/V cache.
Same FlashAttention-2 online softmax (exp2 domain) as demo/kernels/attention.py, but the query tile is padded.

The padding trick: a wgmma needs M >= 64 rows, but there is one query row. The Q descriptor's SHAPE is the real
[1, HEAD_DIM] while its BLOCK is [BLOCK_M, HEAD_DIM]; TMA zero-fills the rows outside the shape, so the shared-memory Q
tile is row 0 plus zeros, with no masking in the kernel. The output descriptor has the same shape/block split, so its
store is clipped to row 0. The 63 padding rows cost MMA work and registers, not memory traffic (decode is bound by
reading K and V, not by the MMA).

Shapes: q is (Z, H, 1, HEAD_DIM); k, v are (Z, H, N_CTX, HEAD_DIM); all with a unit-stride last dim, any head and
sequence strides (Z and H must be mergeable into one axis, always true for Z == 1). The output is a contiguous
(Z, H, 1, HEAD_DIM). N_CTX can be any length: the last K/V tile is zero-filled by TMA and its scores are masked to -inf.
One program per (batch, head): no split over the K/V length yet, so only Z * H programs run (32 for H = 32).

Expect Triton to lower the two tl.dot calls to wgmma when BLOCK_M = 64 and num_warps = 4 (one warpgroup, M = 64);
with BLOCK_M = 16 it would use mma.sync (MMAv2). Check the TTGIR for ttng.warp_group_dot.

Copied from the tpot repo (demo/kernels/decode_attention.py), where it is also used as an input for the tpot rewrites.
"""

import torch
import triton
import triton.language as tl

BLOCK_M = 64   # query rows in the MMA tile; only row 0 is real
BLOCK_N = 128  # keys per K/V tile


def _alloc_fn(size: int, align: int, stream):
    # scratch for device-side tensor descriptors
    return torch.empty(size, dtype=torch.int8, device="cuda")


triton.set_allocator(_alloc_fn)


@triton.jit
def _decode_fwd(q, k, v, out, sm_scale,
                q_stride_h, q_stride_s, k_stride_h, k_stride_s, v_stride_h, v_stride_s, N_CTX,
                HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    off_h = tl.program_id(0).to(tl.int64)  # flattened (batch, head)
    qk_scale = sm_scale * 1.44269504       # 1 / ln(2): softmax in the exp2 domain

    # Q and the output: real shape [1, HEAD_DIM], block [BLOCK_M, HEAD_DIM] (TMA zero-fills / clips the other rows).
    q_desc = tl.make_tensor_descriptor(q + off_h * q_stride_h, [1, HEAD_DIM], [q_stride_s, 1], [BLOCK_M, HEAD_DIM])
    o_desc = tl.make_tensor_descriptor(out + off_h * HEAD_DIM, [1, HEAD_DIM], [HEAD_DIM, 1], [BLOCK_M, HEAD_DIM])
    k_desc = tl.make_tensor_descriptor(k + off_h * k_stride_h, [N_CTX, HEAD_DIM], [k_stride_s, 1], [BLOCK_N, HEAD_DIM])
    v_desc = tl.make_tensor_descriptor(v + off_h * v_stride_h, [N_CTX, HEAD_DIM], [v_stride_s, 1], [BLOCK_N, HEAD_DIM])

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

    q_tile = q_desc.load([0, 0])  # row 0 = the query, rows 1.. = 0

    for start_n in tl.range(0, N_CTX, BLOCK_N):
        k_tile = k_desc.load([start_n, 0])
        qk = tl.dot(q_tile, k_tile.T) * qk_scale
        # keys past N_CTX were zero-filled by TMA (score 0, not -inf): mask them out of the softmax
        cols = start_n + tl.arange(0, BLOCK_N)
        qk = tl.where(cols[None, :] < N_CTX, qk, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v_tile = v_desc.load([start_n, 0])
        acc = tl.dot(p.to(v_tile.dtype), v_tile, acc)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    acc = acc / l_i[:, None]
    o_desc.store([0, 0], acc.to(q_tile.dtype))  # clipped to row 0 by the descriptor's shape


def decode_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float | None = None,
                     *, num_warps: int = 4, num_stages: int = 3):
    """
    Non-causal softmax(q k^T * sm_scale) v for a single query token. q is (Z, H, 1, HEAD_DIM); k, v are
    (Z, H, N_CTX, HEAD_DIM); fp16 or bf16, last dim unit-stride. Returns a contiguous (Z, H, 1, HEAD_DIM) tensor.
    sm_scale defaults to HEAD_DIM ** -0.5. num_warps = 4 is one warpgroup (the wgmma M = 64 tile).
    """
    Z, H, q_len, head_dim = q.shape
    n_ctx = k.shape[2]
    # assert q_len == 1 and k.shape == v.shape and q.dtype == k.dtype == v.dtype
    # assert head_dim in (64, 128, 256) and q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
    if sm_scale is None:
        sm_scale = head_dim ** -0.5
    # merge (Z, H) into one head axis; .view raises if the strides do not allow it (never for Z == 1)
    q3, k3, v3 = q.view(Z * H, 1, head_dim), k.view(Z * H, n_ctx, head_dim), v.view(Z * H, n_ctx, head_dim)
    out = torch.empty((Z * H, 1, head_dim), dtype=q.dtype, device=q.device)
    _decode_fwd[(Z * H,)](q3, k3, v3, out, sm_scale,
                          q3.stride(0), q3.stride(1), k3.stride(0), k3.stride(1), v3.stride(0), v3.stride(1), n_ctx,
                          HEAD_DIM=head_dim, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                          num_warps=num_warps, num_stages=num_stages)
    return out.view(Z, H, 1, head_dim)
