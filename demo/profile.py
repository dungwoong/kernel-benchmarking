from workload_shapes import RMSNORM_LINEAR_ARGS_NT

args = RMSNORM_LINEAR_ARGS_NT.with_config(-2)
print(args)
a, b = args.tensors(mask=[0, 1])
print(a.shape)