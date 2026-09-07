import torch
import tilelang
import tilelang.language as T
from tilelang.carver.arch import driver


@tilelang.jit
def matmul_persistent(
    A, B, block_M=128, block_N=256, block_K=64, threads=256, num_stages=3, dtype=T.bfloat16, accum_dtype=T.float32, use_persistent_primitive=True
):
    M, N, K = T.const("M, N, K")

    A: T.Tensor((M, K), dtype)
    B: T.Tensor((N, K), dtype)   # <-- B is now (N, K), not (K, N)
    C = T.empty((M, N), dtype)

    sm_num = driver.get_num_sms()
    m_blocks = T.ceildiv(M, block_M)
    n_blocks = T.ceildiv(N, block_N)
    waves = T.ceildiv(m_blocks * n_blocks, sm_num)
    group_size = 8

    with T.Kernel(sm_num, threads=threads) as (block_id):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_N, block_K), dtype)  # <-- (N, K) tile shape
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
        C_shared = T.alloc_shared((block_M, block_N), dtype)

        if use_persistent_primitive:
            for bx, by in T.Persistent([T.ceildiv(M, block_M), T.ceildiv(N, block_N)], sm_num, block_id):
                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    T.copy(A[bx * block_M, k * block_K], A_shared)
                    T.copy(B[by * block_N, k * block_K], B_shared)   # <-- indexing swapped
                    T.gemm(A_shared, B_shared, C_local, transpose_B=True)  # <-- key change

                T.copy(C_local, C_shared)
                T.copy(C_shared, C[bx * block_M, by * block_N])
        else:
            for w in T.serial(waves):
                tile_id = sm_num * w + block_id
                bx = (tile_id // group_size) % m_blocks
                by = (tile_id % group_size) + (tile_id // group_size) // m_blocks * group_size

                if bx * block_M < M and by * block_N < N:
                    T.clear(C_local)
                    for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                        T.copy(A[bx * block_M, k * block_K], A_shared)
                        T.copy(B[by * block_N, k * block_K], B_shared)
                        T.gemm(A_shared, B_shared, C_local, transpose_B=True)

                    T.copy(C_local, C_shared)
                    T.copy(C_shared, C[bx * block_M, by * block_N])

    return C