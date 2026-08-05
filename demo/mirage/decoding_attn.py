import mirage as mi
import numpy as np
import torch
import argparse
from triton.testing import do_bench

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch', type=int, default=32)
    parser.add_argument('--file', type=str, default='group_query_attention.json')
    args = parser.parse_args()
    batch_size = args.batch
    filename = args.file

    graph = mi.new_kernel_graph()
    Q = graph.new_input(dims=(batch_size, 16, 128), dtype=mi.float16)
    K = graph.new_input(dims=(batch_size, 128, 4096), dtype=mi.float16)
    V = graph.new_input(dims=(batch_size, 4096, 128), dtype=mi.float16)
    A = graph.matmul(Q, K)
    E = graph.exp(A)
    S = graph.reduction(E, 2)
    D = graph.div(E, S)
    O = graph.matmul(D, V)
    graph.mark_output(O)
    optimized_graph = graph.superoptimize(config="attention", previous_checkpoint=filename)

    input_tensors = [
        torch.randn(batch_size, 16, 128, dtype=torch.float16, device='cuda:0'),
        torch.randn(batch_size, 128, 4096, dtype=torch.float16, device='cuda:0'),
        torch.randn(batch_size, 4096, 128, dtype=torch.float16, device='cuda:0')
    ]

    mean_syn = do_bench(lambda: optimized_graph(inputs=input_tensors))

    print("Best muGraph run time (ms): ", mean_syn)

