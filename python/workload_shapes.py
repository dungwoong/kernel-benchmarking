from profile_utils import ProfilingTensor, KernelArgs

# Combine these args with a gemm shape
GEMM_SHAPES = [
    {'m': 2048, 'n': 2048, 'k': 2048},
    {'m': 4096, 'n': 4096, 'k': 4096},
    {'m': 1024, 'n': 16384, 'k': 2048},
    {'m': 1024, 'n': 32768, 'k': 4096},
    {'m': 1024, 'n': 4096, 'k': 16384},
    {'m': 8192, 'n': 1024, 'k': 2048},
    {'m': 8192, 'n': 2048, 'k': 4096}
]

# https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html
GEMM_ARGS_NT = KernelArgs(
    ProfilingTensor(('m', 'k')), 
    ProfilingTensor(('n', 'k')),
    configs=GEMM_SHAPES)

RMSNORM_LINEAR_ARGS_NT = KernelArgs(
    ProfilingTensor(('m', 'k')), 
    ProfilingTensor(('n', 'k')),
    1e-5,
    configs=GEMM_SHAPES)

SWIGLU_NT = KernelArgs(
    ProfilingTensor(('m', 'k')),
    ProfilingTensor(('n', 'k')),
    ProfilingTensor(('n', 'k')),
    configs=GEMM_SHAPES)
