"""
Fused attention forward (non-causal), simplified from the Triton fused-attention tutorial
(FlashAttention-2 online softmax in the exp2 domain). Hand-written, 2D, TMA descriptors, one function plus a runner.
Copied from the tpot repo (demo/kernels/attention.py), where it is also used as an input for the tpot rewrites.

Compared with the tutorial: forward only, non-causal, fp16/bf16 only (no FP8), no warp-specialize / Hopper / Blackwell
branches, no autotuner, no logsumexp output, and the K/V offsets come from the loop variable instead of being carried.
The loop-carried values are exactly m_i, l_i and acc.

Shapes: q, k, v are (Z, H, N_CTX, HEAD_DIM) with a unit-stride last dim; the head and sequence strides can be
anything (e.g. a head-major view of a (batch, seq, head, dim) tensor), as long as Z and H can be merged into one axis
(always true for Z == 1). q, k, v share one N_CTX. N_CTX is a runtime argument (the K/V loop bound is not hard-coded);
HEAD_DIM and the tile sizes are constexpr. N_CTX must be a multiple of BLOCK_M and BLOCK_N (there is no masking).
Each program builds one 2D descriptor per tensor over ITS head (base pointer + head offset, shape [N_CTX, HEAD_DIM],
the real sequence stride), so the strides only enter through the descriptors. The output is written contiguous.
"""
import torch
import triton
import triton.language as tl

BLOCK_M = 128
BLOCK_N = 128


def _alloc_fn(size: int, align: int, stream):
    # scratch for device-side tensor descriptors (the Helion modules do this via helion.runtime.set_triton_allocator())
    return torch.empty(size, dtype=torch.int8, device="cuda")


triton.set_allocator(_alloc_fn)


@triton.jit
def _attn_fwd(q, k, v, out, sm_scale,
              q_stride_h, q_stride_s, k_stride_h, k_stride_s, v_stride_h, v_stride_s, N_CTX,
              HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    start_m = tl.program_id(0)
    off_h = tl.program_id(1).to(tl.int64)  # flattened (batch, head)

    # Device-side TMA descriptors over this head's (N_CTX, HEAD_DIM) slice of each tensor.
    q_desc = tl.make_tensor_descriptor(q + off_h * q_stride_h, [N_CTX, HEAD_DIM], [q_stride_s, 1], [BLOCK_M, HEAD_DIM])
    k_desc = tl.make_tensor_descriptor(k + off_h * k_stride_h, [N_CTX, HEAD_DIM], [k_stride_s, 1], [BLOCK_N, HEAD_DIM])
    v_desc = tl.make_tensor_descriptor(v + off_h * v_stride_h, [N_CTX, HEAD_DIM], [v_stride_s, 1], [BLOCK_N, HEAD_DIM])
    o_desc = tl.make_tensor_descriptor(out + off_h * N_CTX * HEAD_DIM, [N_CTX, HEAD_DIM], [HEAD_DIM, 1],
                                       [BLOCK_M, HEAD_DIM])

    q_row = start_m * BLOCK_M         # first row of this program's Q tile
    qk_scale = sm_scale * 1.44269504  # 1 / ln(2): softmax in the exp2 domain

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

    # Q stays on chip for the whole K/V loop.
    q_tile = q_desc.load([q_row, 0])

    for start_n in tl.range(0, N_CTX, BLOCK_N):
        k_tile = k_desc.load([start_n, 0])
        qk = tl.dot(q_tile, k_tile.T)
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        p = tl.math.exp2(qk * qk_scale - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v_tile = v_desc.load([start_n, 0])
        acc = tl.dot(p.to(v_tile.dtype), v_tile, acc)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    acc = acc / l_i[:, None]
    o_desc.store([q_row, 0], acc.to(q_tile.dtype))


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float | None = None,
              *, num_warps: int = 8, num_stages: int = 3):
    """
    Non-causal softmax(q k^T * sm_scale) v.  q, k, v are (Z, H, N_CTX, HEAD_DIM), fp16 or bf16, last dim unit-stride,
    any head / sequence strides.  Returns a contiguous (Z, H, N_CTX, HEAD_DIM) tensor.
    sm_scale defaults to HEAD_DIM ** -0.5.  num_warps / num_stages are untuned starting points.
    """
    Z, H, n_ctx, head_dim = q.shape
    # assert q.shape == k.shape == v.shape, "q, k, v must have the same shape"
    # assert q.dtype == k.dtype == v.dtype and q.dtype in (torch.float16, torch.bfloat16)
    # assert q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
    # assert head_dim in (16, 32, 64, 128, 256)
    # assert n_ctx % BLOCK_M == 0 and n_ctx % BLOCK_N == 0, f"N_CTX must be a multiple of {BLOCK_M} and {BLOCK_N}"
    if sm_scale is None:
        sm_scale = head_dim ** -0.5
    # merge (Z, H) into one head axis; .view raises if the strides do not allow it (never for Z == 1)
    q3, k3, v3 = q.view(Z * H, n_ctx, head_dim), k.view(Z * H, n_ctx, head_dim), v.view(Z * H, n_ctx, head_dim)
    out = torch.empty((Z * H, n_ctx, head_dim), dtype=q.dtype, device=q.device)
    grid = (n_ctx // BLOCK_M, Z * H)
    _attn_fwd[grid](q3, k3, v3, out, sm_scale,
                    q3.stride(0), q3.stride(1), k3.stride(0), k3.stride(1), v3.stride(0), v3.stride(1), n_ctx,
                    HEAD_DIM=head_dim, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                    num_warps=num_warps, num_stages=num_stages)
    return out.view(Z, H, n_ctx, head_dim)
