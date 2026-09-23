import torch
import tilelang
import tilelang.language as T
from tilelang.carver.arch import driver

@tilelang.jit
def lora_matmul_persistent(
    X, W, XA, B, block_M=128, block_N=256, block_K=64, threads=256, num_stages=3,
    dtype=T.bfloat16, accum_dtype=T.float32, use_persistent_primitive=True
):
    M, K = T.const("M, K")
    N = T.const("N")
    R = T.const("R")   # lora_dim

    X: T.Tensor((M, K), dtype)
    W: T.Tensor((N, K), dtype)    # nn.Linear-style weight: (out_features, in_features)
    XA: T.Tensor((M, R), dtype)   # precomputed x @ A^T
    B: T.Tensor((N, R), dtype)    # lora "up" weight, (out_features, lora_dim)
    Out = T.empty((M, N), dtype)

    sm_num = driver.get_num_sms()
    m_blocks = T.ceildiv(M, block_M)
    n_blocks = T.ceildiv(N, block_N)
    waves = T.ceildiv(m_blocks * n_blocks, sm_num)
    group_size = 8

    with T.Kernel(sm_num, threads=threads) as (block_id):
        X_shared = T.alloc_shared((block_M, block_K), dtype)
        X_local = T.alloc_fragment((block_M, block_K), dtype)
        W_shared = T.alloc_shared((block_N, block_K), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
        C_shared = T.alloc_shared((block_M, block_N), dtype)

        XA_shared = T.alloc_shared((block_M, R), dtype)   # NEW: lora activation tile
        # XA_local = T.alloc_fragment((block_M, R), dtype)  # NEW: register copy for gemm operand
        B_shared = T.alloc_shared((block_N, R), dtype)    # NEW: lora "up" weight tile

        if use_persistent_primitive:
            for bx, by in T.Persistent([T.ceildiv(M, block_M), T.ceildiv(N, block_N)], sm_num, block_id):
                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    T.copy(X[bx * block_M, k * block_K], X_shared)
                    T.copy(X_shared, X_local)
                    T.copy(W[by * block_N, k * block_K], W_shared)
                    T.gemm(X_local, W_shared, C_local, transpose_B=True,
                           policy=T.GemmWarpPolicy.FullRow)

                # second accumulation: (xA) @ B^T, added into the same C_local  # NEW
                T.copy(XA[bx * block_M, 0], XA_shared)
                # T.copy(XA_shared, XA_local)
                T.copy(B[by * block_N, 0], B_shared)
                T.gemm(XA_shared, B_shared, C_local, transpose_B=True,
                       policy=T.GemmWarpPolicy.FullRow)

                T.copy(C_local, C_shared)
                T.copy(C_shared, Out[bx * block_M, by * block_N])
        else:
            for w in T.serial(waves):
                tile_id = sm_num * w + block_id
                bx = (tile_id // group_size) % m_blocks
                by = (tile_id % group_size) + (tile_id // group_size) // m_blocks * group_size

                if bx * block_M < M and by * block_N < N:
                    T.clear(C_local)
                    for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                        T.copy(X[bx * block_M, k * block_K], X_shared)
                        T.copy(X_shared, X_local)
                        T.copy(W[by * block_N, k * block_K], W_shared)
                        T.gemm(X_local, W_shared, C_local, transpose_B=True,
                               policy=T.GemmWarpPolicy.FullRow)

                    T.copy(XA[bx * block_M, 0], XA_shared)
                    # T.copy(XA_shared, XA_local)
                    T.copy(B[by * block_N, 0], B_shared)
                    T.gemm(XA_shared, B_shared, C_local, transpose_B=True,
                           policy=T.GemmWarpPolicy.FullRow)

                    T.copy(C_local, C_shared)
                    T.copy(C_shared, Out[bx * block_M, by * block_N])

    return Out

def lora_launcher(
    x: torch.Tensor,   # (m, k)
    W: torch.Tensor,   # (n, k)  nn.Linear-style
    A: torch.Tensor,   # (lora_dim, k)
    B: torch.Tensor,   # (n, lora_dim)
) -> torch.Tensor:
    xA = x @ A.t()      # (m, lora_dim) — small gemm, just use PyTorch
    return lora_matmul_persistent(x, W, xA, B)