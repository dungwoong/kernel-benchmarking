import cutlass
from cutlass import cute
from cdsl_helpers import shared
from cdsl_helpers import pipeline
from cdsl_helpers import scheduler
from cdsl_helpers import mma
from cdsl_helpers import elementwise
from cdsl_helpers import store

# kwargs={'tma_stages': 3}


class Kernel:
  @cute.jit
  def __call__(self, a: cute.Tensor, b: cute.Tensor, b1: cute.Tensor, c: cute.Tensor):
    st_4544_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 64, 3)
    st_4558_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 64, 3)
    st_4577_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 64, 3)
    rt_4600_epi_smem_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 64, 2)
    wgmma_acc_4568_tiled_mma = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 128, False)
    wgmma_acc_4587_tiled_mma = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 128, False)
    tiled_mma_7689 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 128, False)
    tiled_mma_7586 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 16, True)
    tiled_mma_7635 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 128, True)
    tiled_mma_7710 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 128, False)
    c_tma_atom_1, c_tma_tensor_1 = shared.get_tma_epi_tensor_and_atom(c, rt_4600_epi_smem_layout, 128, 64)
    a_tma_atom_2, a_tma_tensor_2 = shared.get_tma_tensor_and_atom(a, st_4544_layout, 128, 64, 1)
    b_tma_atom_3, b_tma_tensor_3 = shared.get_tma_tensor_and_atom(b, st_4558_layout, 128, 64, 1)
    b1_tma_atom_4, b1_tma_tensor_4 = shared.get_tma_tensor_and_atom(b1, st_4577_layout, 128, 64, 1)
    self.kernel(a, b, b1, st_4544_layout, st_4558_layout, st_4577_layout, c, rt_4600_epi_smem_layout, wgmma_acc_4568_tiled_mma, wgmma_acc_4587_tiled_mma, tiled_mma_7689, tiled_mma_7586, tiled_mma_7635, tiled_mma_7710, c_tma_atom_1, c_tma_tensor_1, a_tma_atom_2, a_tma_tensor_2, b_tma_atom_3, b_tma_tensor_3, b1_tma_atom_4, b1_tma_tensor_4).launch(grid=[132, 1, 1], block=384)

  @cute.kernel
  def kernel(self, a: cute.Tensor, b: cute.Tensor, b1: cute.Tensor, st_4544_layout, st_4558_layout, st_4577_layout, c: cute.Tensor, rt_4600_epi_smem_layout, wgmma_acc_4568_tiled_mma, wgmma_acc_4587_tiled_mma, tiled_mma_7689, tiled_mma_7586, tiled_mma_7635, tiled_mma_7710, c_tma_atom_1, c_tma_tensor_1, a_tma_atom_2, a_tma_tensor_2, b_tma_atom_3, b_tma_tensor_3, b1_tma_atom_4, b1_tma_tensor_4):
    SharedStorage_t = shared.get_smem_struct()
    shared.smem_add_shared_tensor(SharedStorage_t, 'st_4544_ptr', cutlass.BFloat16, st_4544_layout, 1024)
    shared.smem_add_shared_tensor(SharedStorage_t, 'st_4558_ptr', cutlass.BFloat16, st_4558_layout, 1024)
    shared.smem_add_shared_tensor(SharedStorage_t, 'st_4577_ptr', cutlass.BFloat16, st_4577_layout, 1024)
    shared.smem_add_barrier_array(SharedStorage_t, 'st_4577_pipe_ptr', 3)
    shared.smem_add_shared_tensor(SharedStorage_t, 'rt_4600_epi_smem_ptr', cutlass.BFloat16, rt_4600_epi_smem_layout, 1024)
    smem_alloc = cutlass.utils.SmemAllocator()
    smem_ = smem_alloc.allocate(cute.struct(SharedStorage_t))
    st_4544 = shared.smem_get_tensor(smem_, 'st_4544_ptr', st_4544_layout)
    st_4558 = shared.smem_get_tensor(smem_, 'st_4558_ptr', st_4558_layout)
    st_4577 = shared.smem_get_tensor(smem_, 'st_4577_ptr', st_4577_layout)
    st_4577_pipe = pipeline.make_tma_pipeline_alt(smem_, 'st_4577_pipe_ptr', 3, shared.staged_tensor_sizes(cutlass.BFloat16, st_4577_layout, st_4558_layout, st_4544_layout), 8, None, 1)
    rt_4600_epi_smem = shared.smem_get_tensor(smem_, 'rt_4600_epi_smem_ptr', rt_4600_epi_smem_layout)
    warpidx_ = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx_, _, _ = cute.arch.thread_idx()
    st_4544_pstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 3)
    st_4544_cstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 3)
    st_4558_pstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 3)
    st_4558_cstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 3)
    st_4577_pstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 3)
    st_4577_cstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 3)
    if warpidx_ >= 0 and warpidx_ < 8:
      cute.arch.setmaxregister_increase(232)
      # No change to min warp
      for sched_idx in cutlass.range(cute.arch.block_idx()[0], 1024, 132):
        sched_coord = scheduler.remap_1d_idx(sched_idx, ((8, 8), (4, 4)), ((32, 1), (256, 8)), (32, 32), 8)
        wgmma_acc_4568 = mma.get_acc(wgmma_acc_4568_tiled_mma, 128, 128, cutlass.Float32)
        wgmma_acc_4568_accumulate = False
        wgmma_acc_4587 = mma.get_acc(wgmma_acc_4587_tiled_mma, 128, 128, cutlass.Float32)
        wgmma_acc_4587_accumulate = False
        for k in cutlass.range(0, 64, 1):
          st_4577_pipe.consumer_wait(st_4558_cstate, st_4577_pipe.consumer_try_wait(st_4558_cstate))
          mma.accumulating_gemm_ss(tidx_, tiled_mma_7689, st_4544, st_4577, wgmma_acc_4587, st_4558_cstate, st_4558_cstate, wgmma_acc_4587_accumulate, -1)
          wgmma_acc_4587_accumulate = True
          rt_4554 = mma.copy_a_wgmma(tidx_, tiled_mma_7586, st_4544[None, None, st_4558_cstate.index], 128, 64, cutlass.BFloat16)
          mma.accumulating_gemm_rs(tidx_, tiled_mma_7635, rt_4554, st_4558, wgmma_acc_4568, st_4558_cstate, wgmma_acc_4568_accumulate, -1)
          wgmma_acc_4568_accumulate = True
          cute.nvgpu.warpgroup.wait_group(0)
          st_4577_pipe.consumer_release(st_4558_cstate)
          st_4544_pstate.advance()
          st_4558_cstate.advance()
        rt_4596 = cute.make_rmem_tensor_like(wgmma_acc_4568, cutlass.Float32)
        rt_4596.store(elementwise.silu(wgmma_acc_4568.load()))
        rt_4600 = elementwise.tilewise_mul(rt_4596, wgmma_acc_4587)
        store.mma_epilogue_tma(tiled_mma_7710, c_tma_tensor_1, c_tma_atom_1, rt_4600_epi_smem, rt_4600, 128, 128, sched_coord[0], sched_coord[1], tidx_, warpidx_, cutlass.Float32)
    if warpidx_ >= 8 and warpidx_ < 12:
      cute.arch.setmaxregister_decrease(40)
      if warpidx_ == 8:
        warpidx_ = warpidx_ + 8
        tidx_ = tidx_ + 256
        for sched_idx in cutlass.range(cute.arch.block_idx()[0], 1024, 132):
          sched_coord = scheduler.remap_1d_idx(sched_idx, ((8, 8), (4, 4)), ((32, 1), (256, 8)), (32, 32), 8)
          for k in cutlass.range(0, 64, 1):
            if cutlass.const_expr(True):
              st_4577_pipe.producer_acquire(st_4544_pstate, st_4577_pipe.producer_try_acquire(st_4544_pstate))
              mcast_mask_2, cta_coord_2, cta_layout_2 = shared.get_multicast_info(None, -1)
              shared.tma_copy(a_tma_atom_2, a_tma_tensor_2, st_4544, 128, 64, sched_coord[0], k, st_4577_pipe, st_4544_pstate, cta_coord_2, cta_layout_2, mcast_mask_2)
              mcast_mask_3, cta_coord_3, cta_layout_3 = shared.get_multicast_info(None, -1)
              shared.tma_copy(b_tma_atom_3, b_tma_tensor_3, st_4558, 128, 64, sched_coord[1], k, st_4577_pipe, st_4544_pstate, cta_coord_3, cta_layout_3, mcast_mask_3)
              mcast_mask_4, cta_coord_4, cta_layout_4 = shared.get_multicast_info(None, -1)
              shared.tma_copy(b1_tma_atom_4, b1_tma_tensor_4, st_4577, 128, 64, sched_coord[1], k, st_4577_pipe, st_4544_pstate, cta_coord_4, cta_layout_4, mcast_mask_4)
            st_4544_pstate.advance()
