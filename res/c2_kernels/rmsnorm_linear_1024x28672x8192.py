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
    st_1785_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 64, 3)
    st_1802_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 256, 64, 3)
    rt_1844_epi_smem_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 64, 2)
    tiled_mma_1822 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 16, False)
    wgmma_acc_1812_tiled_mma = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 256, False)
    tiled_mma_5099 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 16, True)
    tiled_mma_5148 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 256, True)
    tiled_mma_5167 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 256, False)
    c_tma_atom_1, c_tma_tensor_1 = shared.get_tma_epi_tensor_and_atom(c, rt_1844_epi_smem_layout, 128, 64)
    a_tma_atom_2, a_tma_tensor_2 = shared.get_tma_tensor_and_atom(a, st_1785_layout, 128, 64, 1)
    b_tma_atom_3, b_tma_tensor_3 = shared.get_tma_tensor_and_atom(b, st_1802_layout, 256, 64, 1)
    self.kernel(a, b, st_1785_layout, st_1802_layout, c, rt_1844_epi_smem_layout, tiled_mma_1822, wgmma_acc_1812_tiled_mma, tiled_mma_5099, tiled_mma_5148, tiled_mma_5167, c_tma_atom_1, c_tma_tensor_1, a_tma_atom_2, a_tma_tensor_2, b_tma_atom_3, b_tma_tensor_3).launch(grid=[132, 1, 1], block=384)

  @cute.kernel
  def kernel(self, a: cute.Tensor, b: cute.Tensor, st_1785_layout, st_1802_layout, c: cute.Tensor, rt_1844_epi_smem_layout, tiled_mma_1822, wgmma_acc_1812_tiled_mma, tiled_mma_5099, tiled_mma_5148, tiled_mma_5167, c_tma_atom_1, c_tma_tensor_1, a_tma_atom_2, a_tma_tensor_2, b_tma_atom_3, b_tma_tensor_3):
    SharedStorage_t = shared.get_smem_struct()
    shared.smem_add_shared_tensor(SharedStorage_t, 'st_1785_ptr', cutlass.BFloat16, st_1785_layout, 1024)
    shared.smem_add_shared_tensor(SharedStorage_t, 'st_1802_ptr', cutlass.BFloat16, st_1802_layout, 1024)
    shared.smem_add_barrier_array(SharedStorage_t, 'st_1802_pipe_ptr', 3)
    shared.smem_add_shared_tensor(SharedStorage_t, 'rt_1844_epi_smem_ptr', cutlass.BFloat16, rt_1844_epi_smem_layout, 1024)
    smem_alloc = cutlass.utils.SmemAllocator()
    smem_ = smem_alloc.allocate(cute.struct(SharedStorage_t))
    st_1785 = shared.smem_get_tensor(smem_, 'st_1785_ptr', st_1785_layout)
    st_1802 = shared.smem_get_tensor(smem_, 'st_1802_ptr', st_1802_layout)
    st_1802_pipe = pipeline.make_tma_pipeline_alt(smem_, 'st_1802_pipe_ptr', 3, shared.staged_tensor_sizes(cutlass.BFloat16, st_1802_layout, st_1785_layout), 8, None, 1)
    rt_1844_epi_smem = shared.smem_get_tensor(smem_, 'rt_1844_epi_smem_ptr', rt_1844_epi_smem_layout)
    warpidx_ = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx_, _, _ = cute.arch.thread_idx()
    st_1785_pstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 3)
    st_1785_cstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 3)
    st_1802_pstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 3)
    st_1802_cstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 3)
    if warpidx_ >= 0 and warpidx_ < 8:
      cute.arch.setmaxregister_increase(232)
      # No change to min warp
      for sched_idx in cutlass.range(cute.arch.block_idx()[0], 896, 132):
        rt_1821 = reduction.make_mma_A_reduction_tensor(tiled_mma_1822, 128, 16, cutlass.Float32)
        sched_coord = scheduler.remap_1d_idx(sched_idx, ((8, 8), (14, 1)), ((8, 1), (64, 8)), (8, 112), 8)
        wgmma_acc_1812 = mma.get_acc(wgmma_acc_1812_tiled_mma, 128, 256, cutlass.Float32)
        wgmma_acc_1812_accumulate = False
        for k in cutlass.range(0, 128, 1):
          st_1802_pipe.consumer_wait(st_1802_cstate, st_1802_pipe.consumer_try_wait(st_1802_cstate))
          rt_1795 = mma.copy_a_wgmma(tidx_, tiled_mma_5099, st_1785[None, None, st_1802_cstate.index], 128, 64, cutlass.BFloat16)
          mma.accumulating_gemm_rs(tidx_, tiled_mma_5148, rt_1795, st_1802, wgmma_acc_1812, st_1802_cstate, wgmma_acc_1812_accumulate, -1)
          wgmma_acc_1812_accumulate = True
          reduction.row_sum_square_mixed_types(rt_1795, rt_1821, cutlass.BFloat16)
          cute.nvgpu.warpgroup.wait_group(0)
          st_1802_pipe.consumer_release(st_1802_cstate)
          st_1785_pstate.advance()
          st_1802_cstate.advance()
        rt_1821.store(reduction.warp_sum_row_mma_layout(rt_1821.load()))
        rt_1832 = cute.make_rmem_tensor_like(rt_1821, cutlass.Float32)
        rt_1832.store(elementwise.const_div(rt_1821.load(), 8192.0))
        rt_1836 = cute.make_rmem_tensor_like(rt_1832, cutlass.Float32)
        rt_1836.store(elementwise.const_add(rt_1832.load(), 1e-05))
        rt_1840 = cute.make_rmem_tensor_like(rt_1836, cutlass.Float32)
        rt_1840.store(elementwise.const_rsqrt(rt_1836.load()))
        rt_1844 = elementwise.row_mul(wgmma_acc_1812, rt_1840)
        store.mma_epilogue_tma(tiled_mma_5167, c_tma_tensor_1, c_tma_atom_1, rt_1844_epi_smem, rt_1844, 128, 256, sched_coord[0], sched_coord[1], tidx_, warpidx_, cutlass.Float32)
    if warpidx_ >= 8 and warpidx_ < 12:
      cute.arch.setmaxregister_decrease(40)
      if warpidx_ == 8:
        warpidx_ = warpidx_ + 8
        tidx_ = tidx_ + 256
        for sched_idx in cutlass.range(cute.arch.block_idx()[0], 896, 132):
          sched_coord = scheduler.remap_1d_idx(sched_idx, ((8, 8), (14, 1)), ((8, 1), (64, 8)), (8, 112), 8)
          for k in cutlass.range(0, 128, 1):
            if cutlass.const_expr(True):
              st_1802_pipe.producer_acquire(st_1785_pstate, st_1802_pipe.producer_try_acquire(st_1785_pstate))
              mcast_mask_2, cta_coord_2, cta_layout_2 = shared.get_multicast_info(None, -1)
              shared.tma_copy(a_tma_atom_2, a_tma_tensor_2, st_1785, 128, 64, sched_coord[0], k, st_1802_pipe, st_1785_pstate, cta_coord_2, cta_layout_2, mcast_mask_2)
              mcast_mask_3, cta_coord_3, cta_layout_3 = shared.get_multicast_info(None, -1)
              shared.tma_copy(b_tma_atom_3, b_tma_tensor_3, st_1802, 256, 64, sched_coord[1], k, st_1802_pipe, st_1785_pstate, cta_coord_3, cta_layout_3, mcast_mask_3)
            st_1785_pstate.advance()
