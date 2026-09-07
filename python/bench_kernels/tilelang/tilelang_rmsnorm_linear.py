import torch
import tilelang
import tilelang.language as T
from tilelang.carver.arch import driver

# Modify matmul tutorial to get this kernel
@tilelang.jit
def rmsnorm_matmul_persistent(
    X, W, eps=1e-5, block_M=128, block_N=256, block_K=64, threads=256, num_stages=3,
    dtype=T.bfloat16, accum_dtype=T.float32, use_persistent_primitive=True
):
    M, K = T.const("M, K")
    N = T.const("N")

    X: T.Tensor((M, K), dtype)
    W: T.Tensor((N, K), dtype)   # nn.Linear-style weight: (out_features, in_features)
    Out = T.empty((M, N), dtype)

    sm_num = driver.get_num_sms()
    m_blocks = T.ceildiv(M, block_M)
    n_blocks = T.ceildiv(N, block_N)
    waves = T.ceildiv(m_blocks * n_blocks, sm_num)
    group_size = 8

    with T.Kernel(sm_num, threads=threads) as (block_id):
        X_shared = T.alloc_shared((block_M, block_K), dtype)
        X_local = T.alloc_fragment((block_M, block_K), dtype)   # NEW: register copy of X tile
        W_shared = T.alloc_shared((block_N, block_K), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
        C_shared = T.alloc_shared((block_M, block_N), dtype)

        X_sq = T.alloc_fragment((block_M, block_K), accum_dtype)
        sumsq = T.alloc_fragment((block_M,), accum_dtype)
        inv_rms = T.alloc_fragment((block_M,), accum_dtype)

        if use_persistent_primitive:
            for bx, by in T.Persistent([T.ceildiv(M, block_M), T.ceildiv(N, block_N)], sm_num, block_id):
                T.clear(C_local)
                T.clear(sumsq)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    T.copy(X[bx * block_M, k * block_K], X_shared)
                    T.copy(X_shared, X_local)                     # NEW: shared -> registers, once
                    T.copy(W[by * block_N, k * block_K], W_shared)

                    T.gemm(X_local, W_shared, C_local, transpose_B=True,
                           policy=T.GemmWarpPolicy.FullRow)       # NEW: X_local is the operand now

                    for i, kk in T.Parallel(block_M, block_K):
                        X_sq[i, kk] = X_local[i, kk] * X_local[i, kk]   # reuse X_local, no re-load
                    T.reduce_sum(X_sq, sumsq, dim=1, clear=False)

                for i in T.Parallel(block_M):
                    inv_rms[i] = T.rsqrt(sumsq[i] / K + eps)
                for i, j in T.Parallel(block_M, block_N):
                    C_local[i, j] *= inv_rms[i]

                T.copy(C_local, C_shared)
                T.copy(C_shared, Out[bx * block_M, by * block_N])
        else:
            for w in T.serial(waves):
                tile_id = sm_num * w + block_id
                bx = (tile_id // group_size) % m_blocks
                by = (tile_id % group_size) + (tile_id // group_size) // m_blocks * group_size

                if bx * block_M < M and by * block_N < N:
                    T.clear(C_local)
                    T.clear(sumsq)
                    for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                        T.copy(X[bx * block_M, k * block_K], X_shared)
                        T.copy(X_shared, X_local)
                        T.copy(W[by * block_N, k * block_K], W_shared)

                        T.gemm(X_local, W_shared, C_local, transpose_B=True,
                               policy=T.GemmWarpPolicy.FullRow)

                        for i, kk in T.Parallel(block_M, block_K):
                            X_sq[i, kk] = X_local[i, kk] * X_local[i, kk]
                        T.reduce_sum(X_sq, sumsq, dim=1, clear=False)

                    for i in T.Parallel(block_M):
                        inv_rms[i] = T.rsqrt(sumsq[i] / K + eps)
                    for i, j in T.Parallel(block_M, block_N):
                        C_local[i, j] *= inv_rms[i]

                    T.copy(C_local, C_shared)
                    T.copy(C_shared, Out[bx * block_M, by * block_N])

    return Out