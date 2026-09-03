import cutlass
from cutlass import cute
from cdsl_helpers import layout
from cdsl_helpers import shared
from cdsl_helpers import pipeline
from cdsl_helpers import scheduler
from cdsl_helpers import reduction
from cdsl_helpers import mma
from cdsl_helpers import elementwise
from cdsl_helpers import conversion
from cdsl_helpers import store

# kwargs={'tma_stages': 1}


class Kernel:
  @cute.jit
  def __call__(self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, mO: cute.Tensor):
    mQ = layout.select(mQ, (1, 2, 0))
    mK = layout.select(mK, (1, 2, 0))
    mV = layout.select(mV, (1, 2, 0))
    st_5538_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 64, 128, 1)
    st_5556_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 128, 2)
    st_5637_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 128, 2)
    mO = layout.select(mO, (1, 2, 0))
    rt_5657_epi_smem_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 64, 64, 2)
    tiled_mma_5523 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 64, 16, False)
    tiled_mma_5605 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 64, 16, False)
    wgmma_acc_5627_tiled_mma = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 64, 128, False)
    tiled_mma_6327 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 64, 128, False)
    tiled_mma_6374 = mma.get_tiled_mma(cutlass.BFloat16, True, False, cutlass.Float32, 64, 128, True)
    tiled_mma_6392 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 64, 128, False)
    mO_tma_atom_1, mO_tma_tensor_1 = shared.get_tma_epi_tensor_and_atom(mO, rt_5657_epi_smem_layout, 64, 64)
    mQ_tma_atom_2, mQ_tma_tensor_2 = shared.get_tma_tensor_and_atom(mQ, st_5538_layout, 64, 128, 1)
    mK_tma_atom_3, mK_tma_tensor_3 = shared.get_tma_tensor_and_atom(mK, st_5556_layout, 128, 128, 1)
    mV_tma_atom_4, mV_tma_tensor_4 = shared.get_tma_tensor_and_atom(mV, st_5637_layout, 128, 128, 1)
    self.kernel(mQ, mK, mV, st_5538_layout, st_5556_layout, st_5637_layout, mO, rt_5657_epi_smem_layout, tiled_mma_5523, tiled_mma_5605, wgmma_acc_5627_tiled_mma, tiled_mma_6327, tiled_mma_6374, tiled_mma_6392, mO_tma_atom_1, mO_tma_tensor_1, mQ_tma_atom_2, mQ_tma_tensor_2, mK_tma_atom_3, mK_tma_tensor_3, mV_tma_atom_4, mV_tma_tensor_4).launch(grid=[132, 1, 1], block=256, cluster=[1, 1, 1])

  @cute.kernel
  def kernel(self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, st_5538_layout, st_5556_layout, st_5637_layout, mO: cute.Tensor, rt_5657_epi_smem_layout, tiled_mma_5523, tiled_mma_5605, wgmma_acc_5627_tiled_mma, tiled_mma_6327, tiled_mma_6374, tiled_mma_6392, mO_tma_atom_1, mO_tma_tensor_1, mQ_tma_atom_2, mQ_tma_tensor_2, mK_tma_atom_3, mK_tma_tensor_3, mV_tma_atom_4, mV_tma_tensor_4):
    SharedStorage_t = shared.get_smem_struct()
    shared.smem_add_shared_tensor(SharedStorage_t, 'st_5538_ptr', cutlass.BFloat16, st_5538_layout, 1024)
    shared.smem_add_barrier_array(SharedStorage_t, 'st_5538_pipe_ptr', 1)
    shared.smem_add_shared_tensor(SharedStorage_t, 'st_5556_ptr', cutlass.BFloat16, st_5556_layout, 1024)
    shared.smem_add_barrier_array(SharedStorage_t, 'st_5556_pipe_ptr', 2)
    shared.smem_add_shared_tensor(SharedStorage_t, 'st_5637_ptr', cutlass.BFloat16, st_5637_layout, 1024)
    shared.smem_add_barrier_array(SharedStorage_t, 'st_5637_pipe_ptr', 2)
    shared.smem_add_shared_tensor(SharedStorage_t, 'rt_5657_epi_smem_ptr', cutlass.BFloat16, rt_5657_epi_smem_layout, 1024)
    smem_alloc = cutlass.utils.SmemAllocator()
    smem_ = smem_alloc.allocate(cute.struct(SharedStorage_t))
    st_5538 = shared.smem_get_tensor(smem_, 'st_5538_ptr', st_5538_layout)
    st_5538_pipe = pipeline.make_tma_pipeline_alt(smem_, 'st_5538_pipe_ptr', 1, shared.staged_tensor_sizes(cutlass.BFloat16, st_5538_layout), 4, cute.make_layout((1, 1, 1, 1)), 1)
    st_5556 = shared.smem_get_tensor(smem_, 'st_5556_ptr', st_5556_layout)
    st_5556_pipe = pipeline.make_tma_pipeline_alt(smem_, 'st_5556_pipe_ptr', 2, shared.staged_tensor_sizes(cutlass.BFloat16, st_5556_layout), 4, cute.make_layout((1, 1, 1, 1)), 1)
    st_5637 = shared.smem_get_tensor(smem_, 'st_5637_ptr', st_5637_layout)
    st_5637_pipe = pipeline.make_tma_pipeline_alt(smem_, 'st_5637_pipe_ptr', 2, shared.staged_tensor_sizes(cutlass.BFloat16, st_5637_layout), 4, cute.make_layout((1, 1, 1, 1)), 1)
    rt_5657_epi_smem = shared.smem_get_tensor(smem_, 'rt_5657_epi_smem_ptr', rt_5657_epi_smem_layout)
    warpidx_ = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx_, _, _ = cute.arch.thread_idx()
    st_5538_pstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 1)
    st_5538_cstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 1)
    st_5556_pstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 2)
    st_5556_cstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 2)
    st_5637_pstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 2)
    st_5637_cstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 2)
    if warpidx_ >= 0 and warpidx_ < 4:
      cute.arch.setmaxregister_increase(232)
      # No change to min warp
      for sched_idx in cutlass.range(cute.arch.block_idx()[0], 32, 132):
        sched_coord = scheduler.remap_1d_idx(sched_idx, ((1, 32), (1, 1)), ((32, 1), (32, 32)), (32, 1), 1)
        rt_5522 = reduction.make_mma_A_ninf_tensor(tiled_mma_5523, 64, 16, cutlass.Float32)
        rt_5604 = reduction.make_mma_A_reduction_tensor(tiled_mma_5605, 64, 16, cutlass.Float32)
        wgmma_acc_5627 = mma.get_acc(wgmma_acc_5627_tiled_mma, 64, 128, cutlass.Float32)
        wgmma_acc_5627_accumulate = False
        gt_slice_5548 = mK[None, None, sched_coord[0]]
        gt_slice_5629 = mV[None, None, sched_coord[0]]
        gt_slice_5529 = mQ[None, None, sched_coord[0]]
        gt_slice_5663 = mO[None, None, sched_coord[0]]
        st_5538_pipe.consumer_wait(st_5538_cstate, st_5538_pipe.consumer_try_wait(st_5538_cstate))
        for k in cutlass.range(0, 8, 1):
          rt_5525 = cute.make_rmem_tensor_like(rt_5522, cutlass.Float32)
          rt_5525.store(elementwise.copy_elementwise(rt_5522.load()))
          st_5556_pipe.consumer_wait(st_5556_cstate, st_5556_pipe.consumer_try_wait(st_5556_cstate))
          acc_6327 = mma.single_gemm_ss(tidx_, 64, 128, tiled_mma_6327, st_5538, st_5556, st_5538_cstate, st_5556_cstate, 0)
          st_5556_pipe.consumer_release(st_5556_cstate)
          rt_5569 = cute.make_rmem_tensor_like(acc_6327, cutlass.Float32)
          rt_5569.store(elementwise.const_mul(acc_6327.load(), 0.12751743082459868))
          reduction.row_max_f32(rt_5569, rt_5522)
          rt_5522.store(reduction.warp_max_row_mma_layout(rt_5522.load()))
          rt_5581 = elementwise.row_bcast_sub(rt_5569, rt_5522)
          rt_5597 = cute.make_rmem_tensor_like(rt_5581, cutlass.Float32)
          rt_5597.store(elementwise.exp2f(rt_5581.load()))
          rt_5597_frgA = cute.make_tensor(rt_5597.iterator, layout.convert_layout_acc_frgA(rt_5597.layout))
          rt_5587 = elementwise.tilewise_sub(rt_5525, rt_5522)
          rt_5593 = cute.make_rmem_tensor_like(rt_5587, cutlass.Float32)
          rt_5593.store(elementwise.exp2f(rt_5587.load()))
          rt_5604.store(elementwise.tilewise_mul(rt_5604, rt_5593).load())
          wgmma_acc_5627.store(elementwise.row_mul(wgmma_acc_5627, rt_5593).load())
          reduction.row_sum_mixed_types(rt_5597_frgA, rt_5604, cutlass.Float32)
          rt_6888 = conversion.cvt_f16(rt_5597_frgA, cutlass.BFloat16)
          st_5556_pstate.advance()
          st_5556_cstate.advance()
          st_5637_pipe.consumer_wait(st_5637_cstate, st_5637_pipe.consumer_try_wait(st_5637_cstate))
          mma.accumulating_gemm_rs(tidx_, tiled_mma_6374, rt_6888, layout.transpose_view(st_5637), wgmma_acc_5627, st_5637_cstate, wgmma_acc_5627_accumulate, -1)
          wgmma_acc_5627_accumulate = True
          cute.nvgpu.warpgroup.wait_group(0)
          st_5637_pipe.consumer_release(st_5637_cstate)
          st_5637_pstate.advance()
          st_5637_cstate.advance()
        st_5538_pipe.consumer_release(st_5538_cstate)
        rt_5604.store(reduction.warp_sum_row_mma_layout(rt_5604.load()))
        rt_5623 = cute.make_rmem_tensor_like(rt_5604, cutlass.Float32)
        rt_5623.store(elementwise.rcp(rt_5604.load()))
        rt_5657 = elementwise.row_mul(wgmma_acc_5627, rt_5623)
        st_5538_pstate.advance()
        st_5538_cstate.advance()
        mO_tma_tensor_1_slice = mO_tma_tensor_1[None, None, sched_coord[0]]
        store.mma_epilogue_tma(tiled_mma_6392, mO_tma_tensor_1_slice, mO_tma_atom_1, rt_5657_epi_smem, rt_5657, 64, 128, sched_coord[1], 0, tidx_, warpidx_, cutlass.Float32)
    if warpidx_ >= 4 and warpidx_ < 8:
      cute.arch.setmaxregister_decrease(40)
      if warpidx_ == 4:
        warpidx_ = warpidx_ + 4
        tidx_ = tidx_ + 128
        for sched_idx in cutlass.range(cute.arch.block_idx()[0], 32, 132):
          sched_coord = scheduler.remap_1d_idx(sched_idx, ((1, 32), (1, 1)), ((32, 1), (32, 32)), (32, 1), 1)
          gt_slice_5548 = mK[None, None, sched_coord[0]]
          gt_slice_5629 = mV[None, None, sched_coord[0]]
          gt_slice_5529 = mQ[None, None, sched_coord[0]]
          if cutlass.const_expr(True):
            st_5538_pipe.producer_acquire(st_5538_pstate, st_5538_pipe.producer_try_acquire(st_5538_pstate))
            mcast_mask_2, cta_coord_2, cta_layout_2 = shared.get_multicast_info([1, 1, 1], -1)
            mQ_tma_tensor_2_slice = mQ_tma_tensor_2[None, None, sched_coord[0]]
            shared.tma_copy(mQ_tma_atom_2, mQ_tma_tensor_2_slice, st_5538, 64, 128, sched_coord[1], 0, st_5538_pipe, st_5538_pstate, cta_coord_2, cta_layout_2, mcast_mask_2)
          for k in cutlass.range(0, 8, 1):
            if cutlass.const_expr(True):
              st_5556_pipe.producer_acquire(st_5556_pstate, st_5556_pipe.producer_try_acquire(st_5556_pstate))
              mcast_mask_3, cta_coord_3, cta_layout_3 = shared.get_multicast_info([1, 1, 1], -1)
              mK_tma_tensor_3_slice = mK_tma_tensor_3[None, None, sched_coord[0]]
              shared.tma_copy(mK_tma_atom_3, mK_tma_tensor_3_slice, st_5556, 128, 128, k, 0, st_5556_pipe, st_5556_pstate, cta_coord_3, cta_layout_3, mcast_mask_3)
            st_5556_pstate.advance()
            if cutlass.const_expr(True):
              st_5637_pipe.producer_acquire(st_5637_pstate, st_5637_pipe.producer_try_acquire(st_5637_pstate))
              mcast_mask_4, cta_coord_4, cta_layout_4 = shared.get_multicast_info([1, 1, 1], -1)
              mV_tma_tensor_4_slice = mV_tma_tensor_4[None, None, sched_coord[0]]
              shared.tma_copy(mV_tma_atom_4, mV_tma_tensor_4_slice, st_5637, 128, 128, k, 0, st_5637_pipe, st_5637_pstate, cta_coord_4, cta_layout_4, mcast_mask_4)
            st_5637_pstate.advance()
          st_5538_pstate.advance()
