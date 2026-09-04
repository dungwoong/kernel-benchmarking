import cutlass
from cutlass import cute
from cdsl_helpers import shared
from cdsl_helpers import pipeline
from cdsl_helpers import scheduler
from cdsl_helpers import mma
from cdsl_helpers import store

# kwargs={'tma_stages': 1}


class Kernel:
  @cute.jit
  def __call__(self, a: cute.Tensor, b: cute.Tensor, xa: cute.Tensor, lora_b: cute.Tensor, c: cute.Tensor):
    st_3259_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 64, 3)
    st_3269_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 256, 64, 3)
    st_3235_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 16, 1)
    st_3249_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 256, 16, 1)
    wgmma_acc_3233_epi_smem_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 64, 2)
    wgmma_acc_3233_tiled_mma = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 256, False)
    tiled_mma_7368 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 256, False)
    tiled_mma_7424 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 16, True)
    tiled_mma_7473 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 256, True)
    tiled_mma_7492 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 256, False)
    c_tma_atom_1, c_tma_tensor_1 = shared.get_tma_epi_tensor_and_atom(c, wgmma_acc_3233_epi_smem_layout, 128, 64)
    a_tma_atom_2, a_tma_tensor_2 = shared.get_tma_tensor_and_atom(a, st_3259_layout, 128, 64, 1)
    b_tma_atom_3, b_tma_tensor_3 = shared.get_tma_tensor_and_atom(b, st_3269_layout, 256, 64, 1)
    xa_tma_atom_4, xa_tma_tensor_4 = shared.get_tma_tensor_and_atom(xa, st_3235_layout, 128, 16, 1)
    lora_b_tma_atom_5, lora_b_tma_tensor_5 = shared.get_tma_tensor_and_atom(lora_b, st_3249_layout, 256, 16, 1)
    self.kernel(a, b, xa, lora_b, st_3259_layout, st_3269_layout, st_3235_layout, st_3249_layout, c, wgmma_acc_3233_epi_smem_layout, wgmma_acc_3233_tiled_mma, tiled_mma_7368, tiled_mma_7424, tiled_mma_7473, tiled_mma_7492, c_tma_atom_1, c_tma_tensor_1, a_tma_atom_2, a_tma_tensor_2, b_tma_atom_3, b_tma_tensor_3, xa_tma_atom_4, xa_tma_tensor_4, lora_b_tma_atom_5, lora_b_tma_tensor_5).launch(grid=[132, 1, 1], block=384)

  @cute.kernel
  def kernel(self, a: cute.Tensor, b: cute.Tensor, xa: cute.Tensor, lora_b: cute.Tensor, st_3259_layout, st_3269_layout, st_3235_layout, st_3249_layout, c: cute.Tensor, wgmma_acc_3233_epi_smem_layout, wgmma_acc_3233_tiled_mma, tiled_mma_7368, tiled_mma_7424, tiled_mma_7473, tiled_mma_7492, c_tma_atom_1, c_tma_tensor_1, a_tma_atom_2, a_tma_tensor_2, b_tma_atom_3, b_tma_tensor_3, xa_tma_atom_4, xa_tma_tensor_4, lora_b_tma_atom_5, lora_b_tma_tensor_5):
    SharedStorage_t = shared.get_smem_struct()
    shared.smem_add_shared_tensor(SharedStorage_t, 'st_3259_ptr', cutlass.BFloat16, st_3259_layout, 1024)
    shared.smem_add_shared_tensor(SharedStorage_t, 'st_3269_ptr', cutlass.BFloat16, st_3269_layout, 1024)
    shared.smem_add_barrier_array(SharedStorage_t, 'st_3269_pipe_ptr', 3)
    shared.smem_add_shared_tensor(SharedStorage_t, 'st_3235_ptr', cutlass.BFloat16, st_3235_layout, 1024)
    shared.smem_add_shared_tensor(SharedStorage_t, 'st_3249_ptr', cutlass.BFloat16, st_3249_layout, 1024)
    shared.smem_add_barrier_array(SharedStorage_t, 'st_3249_pipe_ptr', 1)
    shared.smem_add_shared_tensor(SharedStorage_t, 'wgmma_acc_3233_epi_smem_ptr', cutlass.BFloat16, wgmma_acc_3233_epi_smem_layout, 1024)
    smem_alloc = cutlass.utils.SmemAllocator()
    smem_ = smem_alloc.allocate(cute.struct(SharedStorage_t))
    st_3259 = shared.smem_get_tensor(smem_, 'st_3259_ptr', st_3259_layout)
    st_3269 = shared.smem_get_tensor(smem_, 'st_3269_ptr', st_3269_layout)
    st_3269_pipe = pipeline.make_tma_pipeline_alt(smem_, 'st_3269_pipe_ptr', 3, shared.staged_tensor_sizes(cutlass.BFloat16, st_3269_layout, st_3259_layout), 8, None, 1)
    st_3235 = shared.smem_get_tensor(smem_, 'st_3235_ptr', st_3235_layout)
    st_3249 = shared.smem_get_tensor(smem_, 'st_3249_ptr', st_3249_layout)
    st_3249_pipe = pipeline.make_tma_pipeline_alt(smem_, 'st_3249_pipe_ptr', 1, shared.staged_tensor_sizes(cutlass.BFloat16, st_3249_layout, st_3235_layout), 8, None, 1)
    wgmma_acc_3233_epi_smem = shared.smem_get_tensor(smem_, 'wgmma_acc_3233_epi_smem_ptr', wgmma_acc_3233_epi_smem_layout)
    warpidx_ = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx_, _, _ = cute.arch.thread_idx()
    st_3259_pstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 3)
    st_3259_cstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 3)
    st_3269_pstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 3)
    st_3269_cstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 3)
    st_3235_pstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 1)
    st_3235_cstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 1)
    st_3249_pstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 1)
    st_3249_cstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 1)
    if warpidx_ >= 0 and warpidx_ < 8:
      cute.arch.setmaxregister_increase(232)
      # No change to min warp
      for sched_idx in cutlass.range(cute.arch.block_idx()[0], 128, 132):
        sched_coord = scheduler.remap_1d_idx(sched_idx, ((4, 32), (1, 1)), ((32, 1), (128, 32)), (32, 4), 4)
        wgmma_acc_3233 = mma.get_acc(wgmma_acc_3233_tiled_mma, 128, 256, cutlass.Float32)
        wgmma_acc_3233_accumulate = False
        for k in cutlass.range(0, 256, 1):
          st_3269_pipe.consumer_wait(st_3269_cstate, st_3269_pipe.consumer_try_wait(st_3269_cstate))
          mma.accumulating_gemm_ss(tidx_, tiled_mma_7368, st_3259, st_3269, wgmma_acc_3233, st_3269_cstate, st_3269_cstate, wgmma_acc_3233_accumulate, -1)
          wgmma_acc_3233_accumulate = True
          cute.nvgpu.warpgroup.wait_group(0)
          st_3269_pipe.consumer_release(st_3269_cstate)
          st_3259_pstate.advance()
          st_3269_cstate.advance()
        st_3249_pipe.consumer_wait(st_3235_cstate, st_3249_pipe.consumer_try_wait(st_3235_cstate))
        rt_3245 = mma.copy_a_wgmma(tidx_, tiled_mma_7424, st_3235[None, None, st_3235_cstate.index], 128, 16, cutlass.BFloat16)
        mma.accumulating_gemm_rs(tidx_, tiled_mma_7473, rt_3245, st_3249, wgmma_acc_3233, st_3235_cstate, wgmma_acc_3233_accumulate, -1)
        wgmma_acc_3233_accumulate = True
        cute.nvgpu.warpgroup.wait_group(0)
        st_3249_pipe.consumer_release(st_3235_cstate)
        st_3235_pstate.advance()
        st_3235_cstate.advance()
        store.mma_epilogue_tma(tiled_mma_7492, c_tma_tensor_1, c_tma_atom_1, wgmma_acc_3233_epi_smem, wgmma_acc_3233, 128, 256, sched_coord[0], sched_coord[1], tidx_, warpidx_, cutlass.Float32)
    if warpidx_ >= 8 and warpidx_ < 12:
      cute.arch.setmaxregister_decrease(40)
      if warpidx_ == 8:
        warpidx_ = warpidx_ + 8
        tidx_ = tidx_ + 256
        for sched_idx in cutlass.range(cute.arch.block_idx()[0], 128, 132):
          sched_coord = scheduler.remap_1d_idx(sched_idx, ((4, 32), (1, 1)), ((32, 1), (128, 32)), (32, 4), 4)
          for k in cutlass.range(0, 256, 1):
            if cutlass.const_expr(True):
              st_3269_pipe.producer_acquire(st_3259_pstate, st_3269_pipe.producer_try_acquire(st_3259_pstate))
              mcast_mask_2, cta_coord_2, cta_layout_2 = shared.get_multicast_info(None, -1)
              shared.tma_copy(a_tma_atom_2, a_tma_tensor_2, st_3259, 128, 64, sched_coord[0], k, st_3269_pipe, st_3259_pstate, cta_coord_2, cta_layout_2, mcast_mask_2)
              mcast_mask_3, cta_coord_3, cta_layout_3 = shared.get_multicast_info(None, -1)
              shared.tma_copy(b_tma_atom_3, b_tma_tensor_3, st_3269, 256, 64, sched_coord[1], k, st_3269_pipe, st_3259_pstate, cta_coord_3, cta_layout_3, mcast_mask_3)
            st_3259_pstate.advance()
          if cutlass.const_expr(True):
            st_3249_pipe.producer_acquire(st_3235_pstate, st_3249_pipe.producer_try_acquire(st_3235_pstate))
            mcast_mask_4, cta_coord_4, cta_layout_4 = shared.get_multicast_info(None, -1)
            shared.tma_copy(xa_tma_atom_4, xa_tma_tensor_4, st_3235, 128, 16, sched_coord[0], 0, st_3249_pipe, st_3235_pstate, cta_coord_4, cta_layout_4, mcast_mask_4)
            mcast_mask_5, cta_coord_5, cta_layout_5 = shared.get_multicast_info(None, -1)
            shared.tma_copy(lora_b_tma_atom_5, lora_b_tma_tensor_5, st_3249, 256, 16, sched_coord[1], 0, st_3249_pipe, st_3235_pstate, cta_coord_5, cta_layout_5, mcast_mask_5)
          st_3235_pstate.advance()
