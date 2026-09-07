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
    st_6596_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 64, 3)
    st_6606_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 64, 3)
    st_6616_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 64, 3)
    rt_6648_epi_smem_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 64, 2)
    wgmma_acc_6626_tiled_mma = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 128, False)
    wgmma_acc_6628_tiled_mma = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 128, False)
    tiled_mma_8069 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 128, False)
    tiled_mma_8125 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 128, False)
    tiled_mma_8146 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 128, False)
    c_tma_atom_1, c_tma_tensor_1 = shared.get_tma_epi_tensor_and_atom(c, rt_6648_epi_smem_layout, 128, 64)
    a_tma_atom_2, a_tma_tensor_2 = shared.get_tma_tensor_and_atom(a, st_6596_layout, 128, 64, 1)
    b_tma_atom_3, b_tma_tensor_3 = shared.get_tma_tensor_and_atom(b, st_6606_layout, 128, 64, 1)
    b1_tma_atom_4, b1_tma_tensor_4 = shared.get_tma_tensor_and_atom(b1, st_6616_layout, 128, 64, 1)
    self.kernel(a, b, b1, st_6596_layout, st_6606_layout, st_6616_layout, c, rt_6648_epi_smem_layout, wgmma_acc_6626_tiled_mma, wgmma_acc_6628_tiled_mma, tiled_mma_8069, tiled_mma_8125, tiled_mma_8146, c_tma_atom_1, c_tma_tensor_1, a_tma_atom_2, a_tma_tensor_2, b_tma_atom_3, b_tma_tensor_3, b1_tma_atom_4, b1_tma_tensor_4).launch(grid=[132, 1, 1], block=384)

  @cute.kernel
  def kernel(self, a: cute.Tensor, b: cute.Tensor, b1: cute.Tensor, st_6596_layout, st_6606_layout, st_6616_layout, c: cute.Tensor, rt_6648_epi_smem_layout, wgmma_acc_6626_tiled_mma, wgmma_acc_6628_tiled_mma, tiled_mma_8069, tiled_mma_8125, tiled_mma_8146, c_tma_atom_1, c_tma_tensor_1, a_tma_atom_2, a_tma_tensor_2, b_tma_atom_3, b_tma_tensor_3, b1_tma_atom_4, b1_tma_tensor_4):
    SharedStorage_t = shared.get_smem_struct()
    shared.smem_add_shared_tensor(SharedStorage_t, 'st_6596_ptr', cutlass.BFloat16, st_6596_layout, 1024)
    shared.smem_add_shared_tensor(SharedStorage_t, 'st_6606_ptr', cutlass.BFloat16, st_6606_layout, 1024)
    shared.smem_add_shared_tensor(SharedStorage_t, 'st_6616_ptr', cutlass.BFloat16, st_6616_layout, 1024)
    shared.smem_add_barrier_array(SharedStorage_t, 'st_6616_pipe_ptr', 3)
    shared.smem_add_shared_tensor(SharedStorage_t, 'rt_6648_epi_smem_ptr', cutlass.BFloat16, rt_6648_epi_smem_layout, 1024)
    smem_alloc = cutlass.utils.SmemAllocator()
    smem_ = smem_alloc.allocate(cute.struct(SharedStorage_t))
    st_6596 = shared.smem_get_tensor(smem_, 'st_6596_ptr', st_6596_layout)
    st_6606 = shared.smem_get_tensor(smem_, 'st_6606_ptr', st_6606_layout)
    st_6616 = shared.smem_get_tensor(smem_, 'st_6616_ptr', st_6616_layout)
    st_6616_pipe = pipeline.make_tma_pipeline_alt(smem_, 'st_6616_pipe_ptr', 3, shared.staged_tensor_sizes(cutlass.BFloat16, st_6616_layout, st_6606_layout, st_6596_layout), 8, None, 1)
    rt_6648_epi_smem = shared.smem_get_tensor(smem_, 'rt_6648_epi_smem_ptr', rt_6648_epi_smem_layout)
    warpidx_ = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx_, _, _ = cute.arch.thread_idx()
    st_6596_pstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 3)
    st_6596_cstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 3)
    st_6606_pstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 3)
    st_6606_cstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 3)
    st_6616_pstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 3)
    st_6616_cstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 3)
    if warpidx_ >= 0 and warpidx_ < 8:
      cute.arch.setmaxregister_increase(232)
      # No change to min warp
      for sched_idx in cutlass.range(cute.arch.block_idx()[0], 256, 132):
        sched_coord = scheduler.remap_1d_idx(sched_idx, ((8, 8), (2, 2)), ((16, 1), (128, 8)), (16, 16), 8)
        wgmma_acc_6626 = mma.get_acc(wgmma_acc_6626_tiled_mma, 128, 128, cutlass.Float32)
        wgmma_acc_6626_accumulate = False
        wgmma_acc_6628 = mma.get_acc(wgmma_acc_6628_tiled_mma, 128, 128, cutlass.Float32)
        wgmma_acc_6628_accumulate = False
        for k in cutlass.range(0, 32, 1):
          st_6616_pipe.consumer_wait(st_6616_cstate, st_6616_pipe.consumer_try_wait(st_6616_cstate))
          mma.accumulating_gemm_ss(tidx_, tiled_mma_8069, st_6596, st_6606, wgmma_acc_6628, st_6616_cstate, st_6616_cstate, wgmma_acc_6628_accumulate, -1)
          wgmma_acc_6628_accumulate = True
          mma.accumulating_gemm_ss(tidx_, tiled_mma_8125, st_6596, st_6616, wgmma_acc_6626, st_6616_cstate, st_6616_cstate, wgmma_acc_6626_accumulate, -1)
          wgmma_acc_6626_accumulate = True
          cute.nvgpu.warpgroup.wait_group(0)
          st_6616_pipe.consumer_release(st_6616_cstate)
          st_6596_pstate.advance()
          st_6616_cstate.advance()
        rt_6637 = cute.make_rmem_tensor_like(wgmma_acc_6628, cutlass.Float32)
        rt_6637.store(elementwise.silu(wgmma_acc_6628.load()))
        rt_6648 = elementwise.tilewise_mul(rt_6637, wgmma_acc_6626)
        store.mma_epilogue_tma(tiled_mma_8146, c_tma_tensor_1, c_tma_atom_1, rt_6648_epi_smem, rt_6648, 128, 128, sched_coord[0], sched_coord[1], tidx_, warpidx_, cutlass.Float32)
    if warpidx_ >= 8 and warpidx_ < 12:
      cute.arch.setmaxregister_decrease(40)
      if warpidx_ == 8:
        warpidx_ = warpidx_ + 8
        tidx_ = tidx_ + 256
        for sched_idx in cutlass.range(cute.arch.block_idx()[0], 256, 132):
          sched_coord = scheduler.remap_1d_idx(sched_idx, ((8, 8), (2, 2)), ((16, 1), (128, 8)), (16, 16), 8)
          for k in cutlass.range(0, 32, 1):
            if cutlass.const_expr(True):
              st_6616_pipe.producer_acquire(st_6596_pstate, st_6616_pipe.producer_try_acquire(st_6596_pstate))
              mcast_mask_2, cta_coord_2, cta_layout_2 = shared.get_multicast_info(None, -1)
              shared.tma_copy(a_tma_atom_2, a_tma_tensor_2, st_6596, 128, 64, sched_coord[0], k, st_6616_pipe, st_6596_pstate, cta_coord_2, cta_layout_2, mcast_mask_2)
              mcast_mask_3, cta_coord_3, cta_layout_3 = shared.get_multicast_info(None, -1)
              shared.tma_copy(b_tma_atom_3, b_tma_tensor_3, st_6606, 128, 64, sched_coord[1], k, st_6616_pipe, st_6596_pstate, cta_coord_3, cta_layout_3, mcast_mask_3)
              mcast_mask_4, cta_coord_4, cta_layout_4 = shared.get_multicast_info(None, -1)
              shared.tma_copy(b1_tma_atom_4, b1_tma_tensor_4, st_6616, 128, 64, sched_coord[1], k, st_6616_pipe, st_6596_pstate, cta_coord_4, cta_layout_4, mcast_mask_4)
            st_6596_pstate.advance()
