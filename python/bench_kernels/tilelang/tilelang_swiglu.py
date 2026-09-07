import torch
import tilelang
import tilelang.language as T
from tilelang.carver.arch import driver

# NOTE: tested loading X into registers vs SMEM, SMEM is better
@tilelang.jit
def swiglu_persistent(
    X, W1, W2, block_M=128, block_N=128, block_K=64, threads=256, num_stages=3,
    dtype=T.bfloat16, accum_dtype=T.float32, use_persistent_primitive=True
):
    M, K = T.const("M, K")
    N = T.const("N")

    X: T.Tensor((M, K), dtype)
    W1: T.Tensor((N, K), dtype)   # gate weight, nn.Linear-style (out, in)
    W2: T.Tensor((N, K), dtype)   # up weight
    Out = T.empty((M, N), dtype)

    sm_num = driver.get_num_sms()
    m_blocks = T.ceildiv(M, block_M)
    n_blocks = T.ceildiv(N, block_N)
    waves = T.ceildiv(m_blocks * n_blocks, sm_num)
    group_size = 8

    with T.Kernel(sm_num, threads=threads) as (block_id):
        X_shared = T.alloc_shared((block_M, block_K), dtype)
        W1_shared = T.alloc_shared((block_N, block_K), dtype)   # NEW: gate weight tile
        W2_shared = T.alloc_shared((block_N, block_K), dtype)   # NEW: up weight tile

        gate_local = T.alloc_fragment((block_M, block_N), accum_dtype)  # NEW
        up_local = T.alloc_fragment((block_M, block_N), accum_dtype)    # NEW
        C_shared = T.alloc_shared((block_M, block_N), dtype)

        if use_persistent_primitive:
            for bx, by in T.Persistent([T.ceildiv(M, block_M), T.ceildiv(N, block_N)], sm_num, block_id):
                T.clear(gate_local)
                T.clear(up_local)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    T.copy(X[bx * block_M, k * block_K], X_shared)
                    T.copy(W1[by * block_N, k * block_K], W1_shared)
                    T.copy(W2[by * block_N, k * block_K], W2_shared)
                    T.gemm(X_shared, W1_shared, gate_local, transpose_B=True)  # X @ W1^T
                    T.gemm(X_shared, W2_shared, up_local, transpose_B=True)    # X @ W2^T

                # silu(gate) * up, then cast to output dtype               # NEW
                for i, j in T.Parallel(block_M, block_N):
                    g = gate_local[i, j]
                    silu_g = g * (1.0 / (1.0 + T.exp(-g)))
                    gate_local[i, j] = silu_g * up_local[i, j]

                T.copy(gate_local, C_shared)
                T.copy(C_shared, Out[bx * block_M, by * block_N])
        else:
            for w in T.serial(waves):
                tile_id = sm_num * w + block_id
                bx = (tile_id // group_size) % m_blocks
                by = (tile_id % group_size) + (tile_id // group_size) // m_blocks * group_size

                if bx * block_M < M and by * block_N < N:
                    T.clear(gate_local)
                    T.clear(up_local)
                    for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                        T.copy(X[bx * block_M, k * block_K], X_shared)
                        T.copy(W1[by * block_N, k * block_K], W1_shared)
                        T.copy(W2[by * block_N, k * block_K], W2_shared)
                        T.gemm(X_shared, W1_shared, gate_local, transpose_B=True)
                        T.gemm(X_shared, W2_shared, up_local, transpose_B=True)

                    for i, j in T.Parallel(block_M, block_N):
                        g = gate_local[i, j]
                        silu_g = g * (1.0 / (1.0 + T.exp(-g)))
                        gate_local[i, j] = silu_g * up_local[i, j]

                    T.copy(gate_local, C_shared)
                    T.copy(C_shared, Out[bx * block_M, by * block_N])

    return Out