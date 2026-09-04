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
    st_10352_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 64, 128, 1)
    st_10370_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 128, 2)
    st_10420_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 128, 2)
    mO = layout.select(mO, (1, 2, 0))
    rt_10445_epi_smem_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 64, 64, 2)
    tiled_mma_10395 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 64, 16, False)
    wgmma_acc_10436_tiled_mma = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 64, 128, False)
    tiled_mma_11871 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 64, 128, False)
    tiled_mma_11918 = mma.get_tiled_mma(cutlass.BFloat16, True, False, cutlass.Float32, 64, 128, True)
    tiled_mma_11936 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 64, 128, False)
    mO_tma_atom_1, mO_tma_tensor_1 = shared.get_tma_epi_tensor_and_atom(mO, rt_10445_epi_smem_layout, 64, 64)
    mQ_tma_atom_2, mQ_tma_tensor_2 = shared.get_tma_tensor_and_atom(mQ, st_10352_layout, 64, 128, 1)
    mK_tma_atom_3, mK_tma_tensor_3 = shared.get_tma_tensor_and_atom(mK, st_10370_layout, 128, 128, 1)
    mV_tma_atom_4, mV_tma_tensor_4 = shared.get_tma_tensor_and_atom(mV, st_10420_layout, 128, 128, 1)
    self.kernel(mQ, mK, mV, st_10352_layout, st_10370_layout, st_10420_layout, mO, rt_10445_epi_smem_layout, tiled_mma_10395, wgmma_acc_10436_tiled_mma, tiled_mma_11871, tiled_mma_11918, tiled_mma_11936, mO_tma_atom_1, mO_tma_tensor_1, mQ_tma_atom_2, mQ_tma_tensor_2, mK_tma_atom_3, mK_tma_tensor_3, mV_tma_atom_4, mV_tma_tensor_4).launch(grid=[132, 1, 1], block=256, cluster=[1, 1, 1])

  @cute.kernel
  def kernel(self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, st_10352_layout, st_10370_layout, st_10420_layout, mO: cute.Tensor, rt_10445_epi_smem_layout, tiled_mma_10395, wgmma_acc_10436_tiled_mma, tiled_mma_11871, tiled_mma_11918, tiled_mma_11936, mO_tma_atom_1, mO_tma_tensor_1, mQ_tma_atom_2, mQ_tma_tensor_2, mK_tma_atom_3, mK_tma_tensor_3, mV_tma_atom_4, mV_tma_tensor_4):
    SharedStorage_t = shared.get_smem_struct()
    shared.smem_add_shared_tensor(SharedStorage_t, 'st_10352_ptr', cutlass.BFloat16, st_10352_layout, 1024)
    shared.smem_add_barrier_array(SharedStorage_t, 'st_10352_pipe_ptr', 1)
    shared.smem_add_shared_tensor(SharedStorage_t, 'st_10370_ptr', cutlass.BFloat16, st_10370_layout, 1024)
    shared.smem_add_barrier_array(SharedStorage_t, 'st_10370_pipe_ptr', 2)
    shared.smem_add_shared_tensor(SharedStorage_t, 'st_10420_ptr', cutlass.BFloat16, st_10420_layout, 1024)
    shared.smem_add_barrier_array(SharedStorage_t, 'st_10420_pipe_ptr', 2)
    shared.smem_add_shared_tensor(SharedStorage_t, 'rt_10445_epi_smem_ptr', cutlass.BFloat16, rt_10445_epi_smem_layout, 1024)
    smem_alloc = cutlass.utils.SmemAllocator()
    smem_ = smem_alloc.allocate(cute.struct(SharedStorage_t))
    st_10352 = shared.smem_get_tensor(smem_, 'st_10352_ptr', st_10352_layout)
    st_10352_pipe = pipeline.make_tma_pipeline_alt(smem_, 'st_10352_pipe_ptr', 1, shared.staged_tensor_sizes(cutlass.BFloat16, st_10352_layout), 4, cute.make_layout((1, 1, 1, 1)), 1)
    st_10370 = shared.smem_get_tensor(smem_, 'st_10370_ptr', st_10370_layout)
    st_10370_pipe = pipeline.make_tma_pipeline_alt(smem_, 'st_10370_pipe_ptr', 2, shared.staged_tensor_sizes(cutlass.BFloat16, st_10370_layout), 4, cute.make_layout((1, 1, 1, 1)), 1)
    st_10420 = shared.smem_get_tensor(smem_, 'st_10420_ptr', st_10420_layout)
    st_10420_pipe = pipeline.make_tma_pipeline_alt(smem_, 'st_10420_pipe_ptr', 2, shared.staged_tensor_sizes(cutlass.BFloat16, st_10420_layout), 4, cute.make_layout((1, 1, 1, 1)), 1)
    rt_10445_epi_smem = shared.smem_get_tensor(smem_, 'rt_10445_epi_smem_ptr', rt_10445_epi_smem_layout)
    warpidx_ = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx_, _, _ = cute.arch.thread_idx()
    st_10352_pstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 1)
    st_10352_cstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 1)
    st_10370_pstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 2)
    st_10370_cstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 2)
    st_10420_pstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 2)
    st_10420_cstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 2)
    if warpidx_ >= 0 and warpidx_ < 4:
      cute.arch.setmaxregister_increase(232)
      # No change to min warp
      for sched_idx in cutlass.range(cute.arch.block_idx()[0], 32, 132):
        sched_coord = scheduler.remap_1d_idx(sched_idx, ((1, 32), (1, 1)), ((32, 1), (32, 32)), (32, 1), 1)
        rt_10394 = reduction.make_mma_A_reduction_tensor(tiled_mma_10395, 64, 16, cutlass.Float32)
        wgmma_acc_10436 = mma.get_acc(wgmma_acc_10436_tiled_mma, 64, 128, cutlass.Float32)
        wgmma_acc_10436_accumulate = False
        gt_slice_10362 = mK[None, None, sched_coord[0]]
        gt_slice_10412 = mV[None, None, sched_coord[0]]
        gt_slice_10344 = mQ[None, None, sched_coord[0]]
        gt_slice_10428 = mO[None, None, sched_coord[0]]
        st_10352_pipe.consumer_wait(st_10352_cstate, st_10352_pipe.consumer_try_wait(st_10352_cstate))
        for k in cutlass.range(0, 16, 1):
          st_10370_pipe.consumer_wait(st_10370_cstate, st_10370_pipe.consumer_try_wait(st_10370_cstate))
          acc_11871 = mma.single_gemm_ss(tidx_, 64, 128, tiled_mma_11871, st_10352, st_10370, st_10352_cstate, st_10370_cstate, 0)
          st_10370_pipe.consumer_release(st_10370_cstate)
          rt_10383 = cute.make_rmem_tensor_like(acc_11871, cutlass.Float32)
          rt_10383.store(elementwise.const_mul(acc_11871.load(), 0.12751743082459868))
          rt_10387 = cute.make_rmem_tensor_like(rt_10383, cutlass.Float32)
          rt_10387.store(elementwise.exp2f(rt_10383.load()))
          rt_10387_frgA = cute.make_tensor(rt_10387.iterator, layout.convert_layout_acc_frgA(rt_10387.layout))
          reduction.row_sum_mixed_types(rt_10387_frgA, rt_10394, cutlass.Float32)
          rt_12362 = conversion.cvt_f16(rt_10387_frgA, cutlass.BFloat16)
          st_10370_pstate.advance()
          st_10370_cstate.advance()
          st_10420_pipe.consumer_wait(st_10420_cstate, st_10420_pipe.consumer_try_wait(st_10420_cstate))
          mma.accumulating_gemm_rs(tidx_, tiled_mma_11918, rt_12362, layout.transpose_view(st_10420), wgmma_acc_10436, st_10420_cstate, wgmma_acc_10436_accumulate, -1)
          wgmma_acc_10436_accumulate = True
          cute.nvgpu.warpgroup.wait_group(0)
          st_10420_pipe.consumer_release(st_10420_cstate)
          st_10420_pstate.advance()
          st_10420_cstate.advance()
        st_10352_pipe.consumer_release(st_10352_cstate)
        rt_10394.store(reduction.warp_sum_row_mma_layout(rt_10394.load()))
        rt_10408 = cute.make_rmem_tensor_like(rt_10394, cutlass.Float32)
        rt_10408.store(elementwise.rcp(rt_10394.load()))
        rt_10445 = elementwise.row_mul(wgmma_acc_10436, rt_10408)
        st_10352_pstate.advance()
        st_10352_cstate.advance()
        mO_tma_tensor_1_slice = mO_tma_tensor_1[None, None, sched_coord[0]]
        store.mma_epilogue_tma(tiled_mma_11936, mO_tma_tensor_1_slice, mO_tma_atom_1, rt_10445_epi_smem, rt_10445, 64, 128, sched_coord[1], 0, tidx_, warpidx_, cutlass.Float32)
    if warpidx_ >= 4 and warpidx_ < 8:
      cute.arch.setmaxregister_decrease(40)
      if warpidx_ == 4:
        warpidx_ = warpidx_ + 4
        tidx_ = tidx_ + 128
        for sched_idx in cutlass.range(cute.arch.block_idx()[0], 32, 132):
          sched_coord = scheduler.remap_1d_idx(sched_idx, ((1, 32), (1, 1)), ((32, 1), (32, 32)), (32, 1), 1)
          gt_slice_10362 = mK[None, None, sched_coord[0]]
          gt_slice_10412 = mV[None, None, sched_coord[0]]
          gt_slice_10344 = mQ[None, None, sched_coord[0]]
          if cutlass.const_expr(True):
            st_10352_pipe.producer_acquire(st_10352_pstate, st_10352_pipe.producer_try_acquire(st_10352_pstate))
            mcast_mask_2, cta_coord_2, cta_layout_2 = shared.get_multicast_info([1, 1, 1], -1)
            mQ_tma_tensor_2_slice = mQ_tma_tensor_2[None, None, sched_coord[0]]
            shared.tma_copy(mQ_tma_atom_2, mQ_tma_tensor_2_slice, st_10352, 64, 128, sched_coord[1], 0, st_10352_pipe, st_10352_pstate, cta_coord_2, cta_layout_2, mcast_mask_2)
          for k in cutlass.range(0, 16, 1):
            if cutlass.const_expr(True):
              st_10370_pipe.producer_acquire(st_10370_pstate, st_10370_pipe.producer_try_acquire(st_10370_pstate))
              mcast_mask_3, cta_coord_3, cta_layout_3 = shared.get_multicast_info([1, 1, 1], -1)
              mK_tma_tensor_3_slice = mK_tma_tensor_3[None, None, sched_coord[0]]
              shared.tma_copy(mK_tma_atom_3, mK_tma_tensor_3_slice, st_10370, 128, 128, k, 0, st_10370_pipe, st_10370_pstate, cta_coord_3, cta_layout_3, mcast_mask_3)
            st_10370_pstate.advance()
            if cutlass.const_expr(True):
              st_10420_pipe.producer_acquire(st_10420_pstate, st_10420_pipe.producer_try_acquire(st_10420_pstate))
              mcast_mask_4, cta_coord_4, cta_layout_4 = shared.get_multicast_info([1, 1, 1], -1)
              mV_tma_tensor_4_slice = mV_tma_tensor_4[None, None, sched_coord[0]]
              shared.tma_copy(mV_tma_atom_4, mV_tma_tensor_4_slice, st_10420, 128, 128, k, 0, st_10420_pipe, st_10420_pstate, cta_coord_4, cta_layout_4, mcast_mask_4)
            st_10420_pstate.advance()
          st_10352_pstate.advance()
