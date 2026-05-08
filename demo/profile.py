from workload_shapes import GEMM_ARGS_NT

args = GEMM_ARGS_NT.with_config(-2)
a, = args.tensors(mask=[0])
print(a.shape)