import math
from typing import Type

import cutlass
import cutlass.cute as cute
from cutlass import Float32


def _tiled_copy_2d(dtype, threads_per_row, num_threads, vecsize):
    # threads stride across the row, each thread owns vecsize contiguous elems
    copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), dtype, num_bits_per_copy=vecsize * dtype.width
    )
    thr_layout = cute.make_ordered_layout(
        (num_threads // threads_per_row, threads_per_row), order=(1, 0)
    )
    val_layout = cute.make_layout((1, vecsize))
    return cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)


@cute.jit
def _copy(src: cute.Tensor, dst: cute.Tensor):
    elems = src.shape[0][0]
    atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        src.element_type,
        num_bits_per_copy=min(128, elems * src.element_type.width),
    )
    cute.copy(atom, src, dst)


@cute.jit
def _warp_reduce_add(val: Float32) -> Float32:
    # butterfly sum over 32 lanes, result broadcast to every lane
    for i in cutlass.range_constexpr(5):
        val = val + cute.arch.shuffle_sync_bfly(val, offset=1 << i)
    return val


@cute.jit
def _block_reduce_add(val: Float32, buf: cute.Tensor, stage: int, warps_per_row: int) -> Float32:
    # buf is (warps_per_row, 2), one column per reduction stage
    lane, warp = cute.arch.lane_idx(), cute.arch.warp_idx()
    if lane == 0:
        buf[warp, stage] = val
    cute.arch.barrier()
    acc = Float32(0.0)
    if lane < warps_per_row:
        acc = buf[lane, stage]
    return _warp_reduce_add(acc)


@cute.jit
def _row_sumsq(x: cute.TensorSSA, buf: cute.Tensor, stage: int, warps_per_row: int) -> Float32:
    val = (x * x).reduce(cute.ReductionOp.ADD, init_val=Float32(0.0), reduction_profile=0)
    val = _warp_reduce_add(val)
    return _block_reduce_add(val, buf, stage, warps_per_row)


class NtransMirage:

    def __init__(self, dtype: Type[cutlass.Numeric], N: int):
        self.dtype = dtype
        self.N = N
        self.num_threads = 128
        self.threads_per_row = 128
        self.warps_per_row = self.threads_per_row // cute.arch.WARP_SIZE
        self.vecsize = math.gcd(N, 128 // dtype.width)
        assert N % (self.vecsize * self.threads_per_row) == 0, "N must tile evenly"

    @cute.jit
    def __call__(self, mH, mX, mAlpha, mO, eps: Float32):
        tiled_copy = _tiled_copy_2d(self.dtype, self.threads_per_row, self.num_threads, self.vecsize)
        rows_per_block = self.num_threads // self.threads_per_row
        self.kernel(mH, mX, mAlpha, mO, eps, tiled_copy).launch(
            grid=[cute.ceil_div(mH.shape[0], rows_per_block), 1, 1],
            block=[self.num_threads, 1, 1],
        )

    @cute.kernel
    def kernel(self, mH, mX, mAlpha, mO, eps: Float32, tiled_copy):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        # static tile built from self.N (passing it across the kernel boundary makes it dynamic)
        tile = (self.num_threads // self.threads_per_row, self.N)
        smem = cutlass.utils.SmemAllocator()
        smem_layout = cute.make_ordered_layout(tile, order=(1, 0))
        sH = smem.allocate_tensor(self.dtype, smem_layout, byte_alignment=16)
        sX = smem.allocate_tensor(self.dtype, smem_layout, byte_alignment=16)
        sA = smem.allocate_tensor(self.dtype, smem_layout, byte_alignment=16)
        buf = smem.allocate_tensor(
            Float32, cute.make_ordered_layout((self.warps_per_row, 2), order=(0, 1)), byte_alignment=8
        )

        gH, gX, gA, gO = [
            cute.local_tile(mT, tile, (bidx, 0)) for mT in (mH, mX, mAlpha, mO)
        ]
        thr = tiled_copy.get_slice(tidx)
        # gmem -> smem (cute.copy handles dynamic gmem extents)
        _copy(thr.partition_S(gH), thr.partition_D(sH))
        _copy(thr.partition_S(gX), thr.partition_D(sX))
        _copy(thr.partition_S(gA), thr.partition_D(sA))

        # smem (static layout) -> rmem, so .load() sees a static layout
        tHs, tXs, tAs = thr.partition_D(sH), thr.partition_D(sX), thr.partition_D(sA)
        rH, rX, rA = [cute.make_rmem_tensor_like(t) for t in (tHs, tXs, tAs)]
        cute.autovec_copy(tHs, rH)
        cute.autovec_copy(tXs, rX)
        cute.autovec_copy(tAs, rA)
        h = rH.load().to(Float32)
        x = rX.load().to(Float32)
        a = rA.load().to(Float32)

        rstd_h = cute.math.rsqrt(_row_sumsq(h, buf, 0, self.warps_per_row) / self.N + eps, fastmath=True)
        c = x + a * (h * rstd_h + x)
        rstd_c = cute.math.rsqrt(_row_sumsq(c, buf, 1, self.warps_per_row) / self.N + eps, fastmath=True)

        rO = cute.make_rmem_tensor_like(rH)
        rO.store((c * rstd_c).to(self.dtype))
        _copy(rO, thr.partition_D(gO))
