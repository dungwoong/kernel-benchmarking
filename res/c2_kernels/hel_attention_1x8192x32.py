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
    st_5531_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 64, 128, 1)
    st_5549_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 128, 2)
    st_5574_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 128, 2)
    mO = layout.select(mO, (1, 2, 0))
    rt_5665_epi_smem_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 64, 64, 2)
    tiled_mma_5591 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 64, 16, False)
    tiled_mma_5629 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 64, 16, False)
    wgmma_acc_5651_tiled_mma = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 64, 128, False)
    tiled_mma_6327 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 64, 128, False)
    tiled_mma_6374 = mma.get_tiled_mma(cutlass.BFloat16, True, False, cutlass.Float32, 64, 128, True)
    tiled_mma_6392 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 64, 128, False)
    mO_tma_atom_1, mO_tma_tensor_1 = shared.get_tma_epi_tensor_and_atom(mO, rt_5665_epi_smem_layout, 64, 64)
    mQ_tma_atom_2, mQ_tma_tensor_2 = shared.get_tma_tensor_and_atom(mQ, st_5531_layout, 64, 128, 1)
    mK_tma_atom_3, mK_tma_tensor_3 = shared.get_tma_tensor_and_atom(mK, st_5549_layout, 128, 128, 1)
    mV_tma_atom_4, mV_tma_tensor_4 = shared.get_tma_tensor_and_atom(mV, st_5574_layout, 128, 128, 1)
    self.kernel(mQ, mK, mV, st_5531_layout, st_5549_layout, st_5574_layout, mO, rt_5665_epi_smem_layout, tiled_mma_5591, tiled_mma_5629, wgmma_acc_5651_tiled_mma, tiled_mma_6327, tiled_mma_6374, tiled_mma_6392, mO_tma_atom_1, mO_tma_tensor_1, mQ_tma_atom_2, mQ_tma_tensor_2, mK_tma_atom_3, mK_tma_tensor_3, mV_tma_atom_4, mV_tma_tensor_4).launch(grid=[132, 1, 1], block=256, cluster=[1, 1, 1])

  @cute.kernel
  def kernel(self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, st_5531_layout, st_5549_layout, st_5574_layout, mO: cute.Tensor, rt_5665_epi_smem_layout, tiled_mma_5591, tiled_mma_5629, wgmma_acc_5651_tiled_mma, tiled_mma_6327, tiled_mma_6374, tiled_mma_6392, mO_tma_atom_1, mO_tma_tensor_1, mQ_tma_atom_2, mQ_tma_tensor_2, mK_tma_atom_3, mK_tma_tensor_3, mV_tma_atom_4, mV_tma_tensor_4):
    SharedStorage_t = shared.get_smem_struct()
    shared.smem_add_shared_tensor(SharedStorage_t, 'st_5531_ptr', cutlass.BFloat16, st_5531_layout, 1024)
    shared.smem_add_barrier_array(SharedStorage_t, 'st_5531_pipe_ptr', 1)
    shared.smem_add_shared_tensor(SharedStorage_t, 'st_5549_ptr', cutlass.BFloat16, st_5549_layout, 1024)
    shared.smem_add_barrier_array(SharedStorage_t, 'st_5549_pipe_ptr', 2)
    shared.smem_add_shared_tensor(SharedStorage_t, 'st_5574_ptr', cutlass.BFloat16, st_5574_layout, 1024)
    shared.smem_add_barrier_array(SharedStorage_t, 'st_5574_pipe_ptr', 2)
    shared.smem_add_shared_tensor(SharedStorage_t, 'rt_5665_epi_smem_ptr', cutlass.BFloat16, rt_5665_epi_smem_layout, 1024)
    smem_alloc = cutlass.utils.SmemAllocator()
    smem_ = smem_alloc.allocate(cute.struct(SharedStorage_t))
    st_5531 = shared.smem_get_tensor(smem_, 'st_5531_ptr', st_5531_layout)
    st_5531_pipe = pipeline.make_tma_pipeline_alt(smem_, 'st_5531_pipe_ptr', 1, shared.staged_tensor_sizes(cutlass.BFloat16, st_5531_layout), 4, cute.make_layout((1, 1, 1, 1)), 1)
    st_5549 = shared.smem_get_tensor(smem_, 'st_5549_ptr', st_5549_layout)
    st_5549_pipe = pipeline.make_tma_pipeline_alt(smem_, 'st_5549_pipe_ptr', 2, shared.staged_tensor_sizes(cutlass.BFloat16, st_5549_layout), 4, cute.make_layout((1, 1, 1, 1)), 1)
    st_5574 = shared.smem_get_tensor(smem_, 'st_5574_ptr', st_5574_layout)
    st_5574_pipe = pipeline.make_tma_pipeline_alt(smem_, 'st_5574_pipe_ptr', 2, shared.staged_tensor_sizes(cutlass.BFloat16, st_5574_layout), 4, cute.make_layout((1, 1, 1, 1)), 1)
    rt_5665_epi_smem = shared.smem_get_tensor(smem_, 'rt_5665_epi_smem_ptr', rt_5665_epi_smem_layout)
    warpidx_ = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx_, _, _ = cute.arch.thread_idx()
    st_5531_pstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 1)
    st_5531_cstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 1)
    st_5549_pstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 2)
    st_5549_cstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 2)
    st_5574_pstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 2)
    st_5574_cstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 2)
    if warpidx_ >= 0 and warpidx_ < 4:
      cute.arch.setmaxregister_increase(232)
      # No change to min warp
      for sched_idx in cutlass.range(cute.arch.block_idx()[0], 32, 132):
        sched_coord = scheduler.remap_1d_idx(sched_idx, ((1, 32), (1, 1)), ((32, 1), (32, 32)), (32, 1), 1)
        rt_5590 = reduction.make_mma_A_ninf_tensor(tiled_mma_5591, 64, 16, cutlass.Float32)
        rt_5628 = reduction.make_mma_A_reduction_tensor(tiled_mma_5629, 64, 16, cutlass.Float32)
        wgmma_acc_5651 = mma.get_acc(wgmma_acc_5651_tiled_mma, 64, 128, cutlass.Float32)
        wgmma_acc_5651_accumulate = False
        gt_slice_5541 = mK[None, None, sched_coord[0]]
        gt_slice_5566 = mV[None, None, sched_coord[0]]
        gt_slice_5523 = mQ[None, None, sched_coord[0]]
        gt_slice_5582 = mO[None, None, sched_coord[0]]
        st_5531_pipe.consumer_wait(st_5531_cstate, st_5531_pipe.consumer_try_wait(st_5531_cstate))
        for k in cutlass.range(0, 64, 1):
          rt_5593 = cute.make_rmem_tensor_like(rt_5590, cutlass.Float32)
          rt_5593.store(elementwise.copy_elementwise(rt_5590.load()))
          st_5549_pipe.consumer_wait(st_5549_cstate, st_5549_pipe.consumer_try_wait(st_5549_cstate))
          acc_6327 = mma.single_gemm_ss(tidx_, 64, 128, tiled_mma_6327, st_5531, st_5549, st_5531_cstate, st_5549_cstate, 0)
          st_5549_pipe.consumer_release(st_5549_cstate)
          rt_5562 = cute.make_rmem_tensor_like(acc_6327, cutlass.Float32)
          rt_5562.store(elementwise.const_mul(acc_6327.load(), 0.12751743082459868))
          reduction.row_max_f32(rt_5562, rt_5590)
          rt_5590.store(reduction.warp_max_row_mma_layout(rt_5590.load()))
          rt_5605 = elementwise.row_bcast_sub(rt_5562, rt_5590)
          rt_5611 = cute.make_rmem_tensor_like(rt_5605, cutlass.Float32)
          rt_5611.store(elementwise.exp2f(rt_5605.load()))
          rt_5611_frgA = cute.make_tensor(rt_5611.iterator, layout.convert_layout_acc_frgA(rt_5611.layout))
          rt_5618 = elementwise.tilewise_sub(rt_5593, rt_5590)
          rt_5624 = cute.make_rmem_tensor_like(rt_5618, cutlass.Float32)
          rt_5624.store(elementwise.exp2f(rt_5618.load()))
          rt_5628.store(elementwise.tilewise_mul(rt_5628, rt_5624).load())
          wgmma_acc_5651.store(elementwise.row_mul(wgmma_acc_5651, rt_5624).load())
          reduction.row_sum_mixed_types(rt_5611_frgA, rt_5628, cutlass.Float32)
          rt_6888 = conversion.cvt_f16(rt_5611_frgA, cutlass.BFloat16)
          st_5549_pstate.advance()
          st_5549_cstate.advance()
          st_5574_pipe.consumer_wait(st_5574_cstate, st_5574_pipe.consumer_try_wait(st_5574_cstate))
          mma.accumulating_gemm_rs(tidx_, tiled_mma_6374, rt_6888, layout.transpose_view(st_5574), wgmma_acc_5651, st_5574_cstate, wgmma_acc_5651_accumulate, -1)
          wgmma_acc_5651_accumulate = True
          cute.nvgpu.warpgroup.wait_group(0)
          st_5574_pipe.consumer_release(st_5574_cstate)
          st_5574_pstate.advance()
          st_5574_cstate.advance()
        st_5531_pipe.consumer_release(st_5531_cstate)
        rt_5628.store(reduction.warp_sum_row_mma_layout(rt_5628.load()))
        rt_5647 = cute.make_rmem_tensor_like(rt_5628, cutlass.Float32)
        rt_5647.store(elementwise.rcp(rt_5628.load()))
        rt_5665 = elementwise.row_mul(wgmma_acc_5651, rt_5647)
        st_5531_pstate.advance()
        st_5531_cstate.advance()
        mO_tma_tensor_1_slice = mO_tma_tensor_1[None, None, sched_coord[0]]
        store.mma_epilogue_tma(tiled_mma_6392, mO_tma_tensor_1_slice, mO_tma_atom_1, rt_5665_epi_smem, rt_5665, 64, 128, sched_coord[1], 0, tidx_, warpidx_, cutlass.Float32)
    if warpidx_ >= 4 and warpidx_ < 8:
      cute.arch.setmaxregister_decrease(40)
      if warpidx_ == 4:
        warpidx_ = warpidx_ + 4
        tidx_ = tidx_ + 128
        for sched_idx in cutlass.range(cute.arch.block_idx()[0], 32, 132):
          sched_coord = scheduler.remap_1d_idx(sched_idx, ((1, 32), (1, 1)), ((32, 1), (32, 32)), (32, 1), 1)
          gt_slice_5541 = mK[None, None, sched_coord[0]]
          gt_slice_5566 = mV[None, None, sched_coord[0]]
          gt_slice_5523 = mQ[None, None, sched_coord[0]]
          if cutlass.const_expr(True):
            st_5531_pipe.producer_acquire(st_5531_pstate, st_5531_pipe.producer_try_acquire(st_5531_pstate))
            mcast_mask_2, cta_coord_2, cta_layout_2 = shared.get_multicast_info([1, 1, 1], -1)
            mQ_tma_tensor_2_slice = mQ_tma_tensor_2[None, None, sched_coord[0]]
            shared.tma_copy(mQ_tma_atom_2, mQ_tma_tensor_2_slice, st_5531, 64, 128, sched_coord[1], 0, st_5531_pipe, st_5531_pstate, cta_coord_2, cta_layout_2, mcast_mask_2)
          for k in cutlass.range(0, 64, 1):
            if cutlass.const_expr(True):
              st_5549_pipe.producer_acquire(st_5549_pstate, st_5549_pipe.producer_try_acquire(st_5549_pstate))
              mcast_mask_3, cta_coord_3, cta_layout_3 = shared.get_multicast_info([1, 1, 1], -1)
              mK_tma_tensor_3_slice = mK_tma_tensor_3[None, None, sched_coord[0]]
              shared.tma_copy(mK_tma_atom_3, mK_tma_tensor_3_slice, st_5549, 128, 128, k, 0, st_5549_pipe, st_5549_pstate, cta_coord_3, cta_layout_3, mcast_mask_3)
            st_5549_pstate.advance()
            if cutlass.const_expr(True):
              st_5574_pipe.producer_acquire(st_5574_pstate, st_5574_pipe.producer_try_acquire(st_5574_pstate))
              mcast_mask_4, cta_coord_4, cta_layout_4 = shared.get_multicast_info([1, 1, 1], -1)
              mV_tma_tensor_4_slice = mV_tma_tensor_4[None, None, sched_coord[0]]
              shared.tma_copy(mV_tma_atom_4, mV_tma_tensor_4_slice, st_5574, 128, 128, k, 0, st_5574_pipe, st_5574_pstate, cta_coord_4, cta_layout_4, mcast_mask_4)
            st_5574_pstate.advance()
          st_5531_pstate.advance()
