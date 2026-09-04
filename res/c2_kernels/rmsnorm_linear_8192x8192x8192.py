import cutlass
from cutlass import cute
from cdsl_helpers import shared
from cdsl_helpers import pipeline
from cdsl_helpers import reduction
from cdsl_helpers import scheduler
from cdsl_helpers import mma
from cdsl_helpers import elementwise
from cdsl_helpers import store

# kwargs={'tma_stages': 3}


class Kernel:
  @cute.jit
  def __call__(self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    st_1577_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 64, 3)
    st_1595_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 256, 64, 3)
    rt_1642_epi_smem_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 64, 2)
    tiled_mma_1616 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 16, False)
    wgmma_acc_1606_tiled_mma = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 256, False)
    tiled_mma_5457 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 16, True)
    tiled_mma_5506 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 256, True)
    tiled_mma_5525 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 256, False)
    c_tma_atom_1, c_tma_tensor_1 = shared.get_tma_epi_tensor_and_atom(c, rt_1642_epi_smem_layout, 128, 64)
    a_tma_atom_2, a_tma_tensor_2 = shared.get_tma_tensor_and_atom(a, st_1577_layout, 128, 64, 1)
    b_tma_atom_3, b_tma_tensor_3 = shared.get_tma_tensor_and_atom(b, st_1595_layout, 256, 64, 1)
    self.kernel(a, b, st_1577_layout, st_1595_layout, c, rt_1642_epi_smem_layout, tiled_mma_1616, wgmma_acc_1606_tiled_mma, tiled_mma_5457, tiled_mma_5506, tiled_mma_5525, c_tma_atom_1, c_tma_tensor_1, a_tma_atom_2, a_tma_tensor_2, b_tma_atom_3, b_tma_tensor_3).launch(grid=[132, 1, 1], block=384)

  @cute.kernel
  def kernel(self, a: cute.Tensor, b: cute.Tensor, st_1577_layout, st_1595_layout, c: cute.Tensor, rt_1642_epi_smem_layout, tiled_mma_1616, wgmma_acc_1606_tiled_mma, tiled_mma_5457, tiled_mma_5506, tiled_mma_5525, c_tma_atom_1, c_tma_tensor_1, a_tma_atom_2, a_tma_tensor_2, b_tma_atom_3, b_tma_tensor_3):
    SharedStorage_t = shared.get_smem_struct()
    shared.smem_add_shared_tensor(SharedStorage_t, 'st_1577_ptr', cutlass.BFloat16, st_1577_layout, 1024)
    shared.smem_add_shared_tensor(SharedStorage_t, 'st_1595_ptr', cutlass.BFloat16, st_1595_layout, 1024)
    shared.smem_add_barrier_array(SharedStorage_t, 'st_1595_pipe_ptr', 3)
    shared.smem_add_shared_tensor(SharedStorage_t, 'rt_1642_epi_smem_ptr', cutlass.BFloat16, rt_1642_epi_smem_layout, 1024)
    smem_alloc = cutlass.utils.SmemAllocator()
    smem_ = smem_alloc.allocate(cute.struct(SharedStorage_t))
    st_1577 = shared.smem_get_tensor(smem_, 'st_1577_ptr', st_1577_layout)
    st_1595 = shared.smem_get_tensor(smem_, 'st_1595_ptr', st_1595_layout)
    st_1595_pipe = pipeline.make_tma_pipeline_alt(smem_, 'st_1595_pipe_ptr', 3, shared.staged_tensor_sizes(cutlass.BFloat16, st_1595_layout, st_1577_layout), 8, None, 1)
    rt_1642_epi_smem = shared.smem_get_tensor(smem_, 'rt_1642_epi_smem_ptr', rt_1642_epi_smem_layout)
    warpidx_ = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx_, _, _ = cute.arch.thread_idx()
    st_1577_pstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 3)
    st_1577_cstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 3)
    st_1595_pstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 3)
    st_1595_cstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 3)
    if warpidx_ >= 0 and warpidx_ < 8:
      cute.arch.setmaxregister_increase(232)
      # No change to min warp
      for sched_idx in cutlass.range(cute.arch.block_idx()[0], 2048, 132):
        rt_1615 = reduction.make_mma_A_reduction_tensor(tiled_mma_1616, 128, 16, cutlass.Float32)
        sched_coord = scheduler.remap_1d_idx(sched_idx, ((8, 8), (4, 8)), ((64, 1), (512, 8)), (64, 32), 8)
        wgmma_acc_1606 = mma.get_acc(wgmma_acc_1606_tiled_mma, 128, 256, cutlass.Float32)
        wgmma_acc_1606_accumulate = False
        for k in cutlass.range(0, 128, 1):
          st_1595_pipe.consumer_wait(st_1595_cstate, st_1595_pipe.consumer_try_wait(st_1595_cstate))
          rt_1587 = mma.copy_a_wgmma(tidx_, tiled_mma_5457, st_1577[None, None, st_1595_cstate.index], 128, 64, cutlass.BFloat16)
          mma.accumulating_gemm_rs(tidx_, tiled_mma_5506, rt_1587, st_1595, wgmma_acc_1606, st_1595_cstate, wgmma_acc_1606_accumulate, -1)
          wgmma_acc_1606_accumulate = True
          rt_1618 = cute.make_rmem_tensor_like(rt_1587, cutlass.BFloat16)
          rt_1618.store(elementwise.square_elementwise(rt_1587.load()))
          reduction.row_sum_mixed_types(rt_1618, rt_1615, cutlass.BFloat16)
          cute.nvgpu.warpgroup.wait_group(0)
          st_1595_pipe.consumer_release(st_1595_cstate)
          st_1577_pstate.advance()
          st_1595_cstate.advance()
        rt_1615.store(reduction.warp_sum_row_mma_layout(rt_1615.load()))
        rt_1630 = cute.make_rmem_tensor_like(rt_1615, cutlass.Float32)
        rt_1630.store(elementwise.const_div(rt_1615.load(), 8192.0))
        rt_1634 = cute.make_rmem_tensor_like(rt_1630, cutlass.Float32)
        rt_1634.store(elementwise.const_add(rt_1630.load(), 1e-05))
        rt_1638 = cute.make_rmem_tensor_like(rt_1634, cutlass.Float32)
        rt_1638.store(elementwise.const_rsqrt(rt_1634.load()))
        rt_1642 = elementwise.row_mul(wgmma_acc_1606, rt_1638)
        store.mma_epilogue_tma(tiled_mma_5525, c_tma_tensor_1, c_tma_atom_1, rt_1642_epi_smem, rt_1642, 128, 256, sched_coord[0], sched_coord[1], tidx_, warpidx_, cutlass.Float32)
    if warpidx_ >= 8 and warpidx_ < 12:
      cute.arch.setmaxregister_decrease(40)
      if warpidx_ == 8:
        warpidx_ = warpidx_ + 8
        tidx_ = tidx_ + 256
        for sched_idx in cutlass.range(cute.arch.block_idx()[0], 2048, 132):
          sched_coord = scheduler.remap_1d_idx(sched_idx, ((8, 8), (4, 8)), ((64, 1), (512, 8)), (64, 32), 8)
          for k in cutlass.range(0, 128, 1):
            if cutlass.const_expr(True):
              st_1595_pipe.producer_acquire(st_1577_pstate, st_1595_pipe.producer_try_acquire(st_1577_pstate))
              mcast_mask_2, cta_coord_2, cta_layout_2 = shared.get_multicast_info(None, -1)
              shared.tma_copy(a_tma_atom_2, a_tma_tensor_2, st_1577, 128, 64, sched_coord[0], k, st_1595_pipe, st_1577_pstate, cta_coord_2, cta_layout_2, mcast_mask_2)
              mcast_mask_3, cta_coord_3, cta_layout_3 = shared.get_multicast_info(None, -1)
              shared.tma_copy(b_tma_atom_3, b_tma_tensor_3, st_1595, 256, 64, sched_coord[1], k, st_1595_pipe, st_1577_pstate, cta_coord_3, cta_layout_3, mcast_mask_3)
            st_1577_pstate.advance()
