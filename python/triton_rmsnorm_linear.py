"""
Fused RMSNorm-linear in NT layout: C = RMSNorm(A) @ B.T, with A=(M,K), B=(N,K)
"""
import torch
import triton
import triton.language as tl

from triton_persistent_gemm import (
    matmul_get_configs,
    _compute_pid,
    _validate_inputs,
    _output,
)


@triton.autotune(configs=matmul_get_configs(), key=["M", "N", "K"])
@triton.jit
def rmsnorm_linear_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    eps,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    offs_k_mask = tl.arange(0, BLOCK_SIZE_K)

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        safe_offs_m = tl.max_contiguous(tl.multiple_of(offs_m % M, BLOCK_SIZE_M), BLOCK_SIZE_M)
        safe_offs_n = tl.max_contiguous(tl.multiple_of(offs_n % N, BLOCK_SIZE_N), BLOCK_SIZE_N)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        # row_sumsq only depends on pid_m, so it is recomputed in every N-tile of this row
        row_sumsq = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        for ki in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + safe_offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            b_ptrs = b_ptr + safe_offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk
            k_remaining = K - ki * BLOCK_SIZE_K
            a = tl.load(a_ptrs, mask=offs_k_mask[None, :] < k_remaining, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k_mask[:, None] < k_remaining, other=0.0)
            accumulator = tl.dot(a, b, accumulator)
            a_f32 = a.to(tl.float32)
            row_sumsq += tl.sum(a_f32 * a_f32, axis=1)

        inv_rms = tl.rsqrt(row_sumsq / K + eps)
        accumulator *= inv_rms[:, None]

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=c_mask)


def rmsnorm_linear_persistent(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-5):
    _validate_inputs(a, b)
    M, K = a.shape
    N = b.shape[0]
    c = _output(a, b)
    num_sms = torch.cuda.get_device_properties(a.device).multi_processor_count
    grid = lambda META: (
        min(num_sms, triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])),
    )
    rmsnorm_linear_kernel[grid](
        a, b, c,
        M, N, K,
        eps,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        NUM_SMS=num_sms,
    )
    return c
