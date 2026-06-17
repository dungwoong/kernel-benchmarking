import mirage as mi
import numpy as np
import torch
import argparse

"""
Modification of mirage/benchmark/rmsnorm.py, to output cuda code

python3 demo/rmsnorm.py --warmup 10 --profile 100 --file third_party/mirage/benchmark/saved_mugraphs_hopper/rmsnorm_bs1.json
"""

def get_cuda(self, async_=False, **kwargs):
    input_tensors = kwargs.get("inputs", [])
    input_strides = []
    # Check that the input_strides match uGraph's specification
    dtensors = self.cygraph.get_input_dtensors()
    assert len(dtensors) == len(
        input_tensors
    ), "Given number of inputs do not match the uGraph's inputs"
    for i in range(len(dtensors)):
        dims, strides = self.cygraph.get_input_dtensor_shape_and_stride(dtensors[i])
        assert (
            dims == input_tensors[i].shape
        ), "Expected input dims {}, got input dims {}".format(
            dims, input_tensors[i].shape
        )
        assert (
            strides == input_tensors[i].stride()
        ), "Expected input strides {}, got input strides {}".format(
            strides, input_tensors[i].stride()
        )
        input_strides.append(strides)
    target_cc = kwargs.get(
        "target_cc",
        torch.cuda.get_device_properties(0).major * 10
        + torch.cuda.get_device_properties(0).minor,
    )
    num_warp_groups = kwargs.get("num_warp_groups", 2)
    pipeline_stages = kwargs.get("pipeline_stages", 2)
    # TODO, add profling for Ampere later to show gpu wave
    profiling = kwargs.get("profiling", False)
    enable_online_softmax = kwargs.get("enable_online_softmax", False)

    result = mi.generate_cuda_program(
        self.cygraph,
        target_cc=target_cc,
        input_strides=input_strides,
        num_warp_groups=num_warp_groups,
        pipeline_stages=pipeline_stages,
        profiling=profiling,
        enable_online_softmax=enable_online_softmax,
    )
    return result["code"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('--file', type=str, default='rmsnorm.json')
    parser.add_argument('--backend', type=str, default='cuda')
    parser.add_argument('--warmup', type=int, default=16)
    parser.add_argument('--profile', type=int, default=1000)
    parser.add_argument('--save_codes', type=bool, default=False)
    parser.add_argument('-t', '--thread', type=int, default=-1)
    parser.add_argument('--max_num_threadblock_graph_op', type=int, default=-1)

    args = parser.parse_args()
    batch_size = args.batch_size
    filename = args.file
    backend = args.backend
    warmup_iters = args.warmup
    profile_iters = args.profile
    save_codes = args.save_codes

    graph = mi.new_kernel_graph()
    X = graph.new_input(dims=(2 * batch_size, 4096), dtype=mi.float16)
    W = graph.new_input(dims=(4096, 6144), dtype=mi.float16)
    D = graph.rms_norm(X, normalized_shape=(4096,))
    O = graph.matmul(D, W)
    graph.mark_output(O)
    optimized_graph = graph.superoptimize(
        config="mlp",
        previous_checkpoint=filename,
        backend=backend,
        save_codes=save_codes,
        warmup_iters=warmup_iters,
        profile_iters=profile_iters,
        num_search_thread=args.thread,
        max_num_threadblock_graph_op=args.max_num_threadblock_graph_op)

    input_tensors = [
        torch.randn(2 * batch_size, 4096, dtype=torch.float16, device='cuda:0'),
        # MAKE SURE THE SHAPES MATCH!
        torch.randn(4096, 6144, dtype=torch.float16, device='cuda:0'),
    ]

    print(get_cuda(optimized_graph, inputs=input_tensors))

    outputs = optimized_graph(inputs=input_tensors)
    output = outputs[0]
    print(output.shape)
    print(output.stride(0), output.stride(1))

    for _ in range(16):
        optimized_graph(inputs=input_tensors)

    torch.cuda.synchronize()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    starter.record()
    for _ in range(1000):
        optimized_graph(inputs=input_tensors)
    ender.record()
    torch.cuda.synchronize()
    curr_time = starter.elapsed_time(ender)
    mean_syn = curr_time / 1000

    print("Best muGraph run time (ms): ", mean_syn)
