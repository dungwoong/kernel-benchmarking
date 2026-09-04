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
    st_4677_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 64, 3)
    st_4695_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 256, 64, 3)
    rt_4742_epi_smem_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 64, 2)
    tiled_mma_4716 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 16, False)
    wgmma_acc_4706_tiled_mma = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 256, False)
    tiled_mma_6234 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 256, False)
    tiled_mma_6185 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 16, True)
    tiled_mma_6255 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 256, False)
    c_tma_atom_1, c_tma_tensor_1 = shared.get_tma_epi_tensor_and_atom(c, rt_4742_epi_smem_layout, 128, 64)
    a_tma_atom_2, a_tma_tensor_2 = shared.get_tma_tensor_and_atom(a, st_4677_layout, 128, 64, 1)
    b_tma_atom_3, b_tma_tensor_3 = shared.get_tma_tensor_and_atom(b, st_4695_layout, 256, 64, 1)
    self.kernel(a, b, st_4677_layout, st_4695_layout, c, rt_4742_epi_smem_layout, tiled_mma_4716, wgmma_acc_4706_tiled_mma, tiled_mma_6234, tiled_mma_6185, tiled_mma_6255, c_tma_atom_1, c_tma_tensor_1, a_tma_atom_2, a_tma_tensor_2, b_tma_atom_3, b_tma_tensor_3).launch(grid=[132, 1, 1], block=384)

  @cute.kernel
  def kernel(self, a: cute.Tensor, b: cute.Tensor, st_4677_layout, st_4695_layout, c: cute.Tensor, rt_4742_epi_smem_layout, tiled_mma_4716, wgmma_acc_4706_tiled_mma, tiled_mma_6234, tiled_mma_6185, tiled_mma_6255, c_tma_atom_1, c_tma_tensor_1, a_tma_atom_2, a_tma_tensor_2, b_tma_atom_3, b_tma_tensor_3):
    SharedStorage_t = shared.get_smem_struct()
    shared.smem_add_shared_tensor(SharedStorage_t, 'st_4677_ptr', cutlass.BFloat16, st_4677_layout, 1024)
    shared.smem_add_shared_tensor(SharedStorage_t, 'st_4695_ptr', cutlass.BFloat16, st_4695_layout, 1024)
    shared.smem_add_barrier_array(SharedStorage_t, 'st_4695_pipe_ptr', 3)
    shared.smem_add_shared_tensor(SharedStorage_t, 'rt_4742_epi_smem_ptr', cutlass.BFloat16, rt_4742_epi_smem_layout, 1024)
    smem_alloc = cutlass.utils.SmemAllocator()
    smem_ = smem_alloc.allocate(cute.struct(SharedStorage_t))
    st_4677 = shared.smem_get_tensor(smem_, 'st_4677_ptr', st_4677_layout)
    st_4695 = shared.smem_get_tensor(smem_, 'st_4695_ptr', st_4695_layout)
    st_4695_pipe = pipeline.make_tma_pipeline_alt(smem_, 'st_4695_pipe_ptr', 3, shared.staged_tensor_sizes(cutlass.BFloat16, st_4695_layout, st_4677_layout), 8, None, 1)
    rt_4742_epi_smem = shared.smem_get_tensor(smem_, 'rt_4742_epi_smem_ptr', rt_4742_epi_smem_layout)
    warpidx_ = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx_, _, _ = cute.arch.thread_idx()
    st_4677_pstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 3)
    st_4677_cstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 3)
    st_4695_pstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 3)
    st_4695_cstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 3)
    if warpidx_ >= 0 and warpidx_ < 8:
      cute.arch.setmaxregister_increase(232)
      # No change to min warp
      for sched_idx in cutlass.range(cute.arch.block_idx()[0], 512, 132):
        rt_4715 = reduction.make_mma_A_reduction_tensor(tiled_mma_4716, 128, 16, cutlass.Float32)
        sched_coord = scheduler.remap_1d_idx(sched_idx, ((8, 8), (8, 1)), ((8, 1), (64, 8)), (8, 64), 8)
        wgmma_acc_4706 = mma.get_acc(wgmma_acc_4706_tiled_mma, 128, 256, cutlass.Float32)
        wgmma_acc_4706_accumulate = False
        for k in cutlass.range(0, 32, 1):
          st_4695_pipe.consumer_wait(st_4695_cstate, st_4695_pipe.consumer_try_wait(st_4695_cstate))
          mma.accumulating_gemm_ss(tidx_, tiled_mma_6234, st_4677, st_4695, wgmma_acc_4706, st_4695_cstate, st_4695_cstate, wgmma_acc_4706_accumulate, -1)
          wgmma_acc_4706_accumulate = True
          rt_4687 = mma.copy_a_wgmma(tidx_, tiled_mma_6185, st_4677[None, None, st_4695_cstate.index], 128, 64, cutlass.BFloat16)
          rt_4718 = cute.make_rmem_tensor_like(rt_4687, cutlass.BFloat16)
          rt_4718.store(elementwise.square_elementwise(rt_4687.load()))
          reduction.row_sum_mixed_types(rt_4718, rt_4715, cutlass.BFloat16)
          cute.nvgpu.warpgroup.wait_group(0)
          st_4695_pipe.consumer_release(st_4695_cstate)
          st_4677_pstate.advance()
          st_4695_cstate.advance()
        rt_4715.store(reduction.warp_sum_row_mma_layout(rt_4715.load()))
        rt_4730 = cute.make_rmem_tensor_like(rt_4715, cutlass.Float32)
        rt_4730.store(elementwise.const_div(rt_4715.load(), 2048.0))
        rt_4734 = cute.make_rmem_tensor_like(rt_4730, cutlass.Float32)
        rt_4734.store(elementwise.const_add(rt_4730.load(), 1e-05))
        rt_4738 = cute.make_rmem_tensor_like(rt_4734, cutlass.Float32)
        rt_4738.store(elementwise.const_rsqrt(rt_4734.load()))
        rt_4742 = elementwise.row_mul(wgmma_acc_4706, rt_4738)
        store.mma_epilogue_tma(tiled_mma_6255, c_tma_tensor_1, c_tma_atom_1, rt_4742_epi_smem, rt_4742, 128, 256, sched_coord[0], sched_coord[1], tidx_, warpidx_, cutlass.Float32)
    if warpidx_ >= 8 and warpidx_ < 12:
      cute.arch.setmaxregister_decrease(40)
      if warpidx_ == 8:
        warpidx_ = warpidx_ + 8
        tidx_ = tidx_ + 256
        for sched_idx in cutlass.range(cute.arch.block_idx()[0], 512, 132):
          sched_coord = scheduler.remap_1d_idx(sched_idx, ((8, 8), (8, 1)), ((8, 1), (64, 8)), (8, 64), 8)
          for k in cutlass.range(0, 32, 1):
            if cutlass.const_expr(True):
              st_4695_pipe.producer_acquire(st_4677_pstate, st_4695_pipe.producer_try_acquire(st_4677_pstate))
              mcast_mask_2, cta_coord_2, cta_layout_2 = shared.get_multicast_info(None, -1)
              shared.tma_copy(a_tma_atom_2, a_tma_tensor_2, st_4677, 128, 64, sched_coord[0], k, st_4695_pipe, st_4677_pstate, cta_coord_2, cta_layout_2, mcast_mask_2)
              mcast_mask_3, cta_coord_3, cta_layout_3 = shared.get_multicast_info(None, -1)
              shared.tma_copy(b_tma_atom_3, b_tma_tensor_3, st_4695, 256, 64, sched_coord[1], k, st_4695_pipe, st_4677_pstate, cta_coord_3, cta_layout_3, mcast_mask_3)
            st_4677_pstate.advance()
