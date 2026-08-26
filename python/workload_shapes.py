from profile_utils import ProfilingTensor, EmptyTensor, KernelArgs

# Combine these args with a gemm shape
GEMM_SHAPES = [
    {'m': 2048, 'n': 2048, 'k': 2048},
    {'m': 4096, 'n': 4096, 'k': 4096},
    {'m': 1024, 'n': 16384, 'k': 2048},
    {'m': 1024, 'n': 32768, 'k': 4096},
    {'m': 1024, 'n': 4096, 'k': 16384},
    {'m': 8192, 'n': 1024, 'k': 2048},
    {'m': 8192, 'n': 2048, 'k': 4096},
    {'m': 4096, 'n': 1024, 'k': 4096}, # meta-llama/Meta-Llama-3.1-8B k_proj, you can also do m=8192
    {'m': 4096, 'n': 1024, 'k': 8192}, # meta-llama/Meta-Llama-3.1-70B k_proj
    {'m': 4096, 'n': 1536, 'k': 7168}, # deepseek-ai/DeepSeek-V3 q a proj
    {'m': 4096, 'n': 1024, 'k': 16384}, # meta-llama/Meta-Llama-3.1-405B kv proj
]

# https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html
GEMM_ARGS_NT = KernelArgs(
    ProfilingTensor(('m', 'k')), 
    ProfilingTensor(('n', 'k')),
    EmptyTensor(('m', 'n')),
    configs=GEMM_SHAPES)

RMSNORM_LINEAR_ARGS_NT = KernelArgs(
    ProfilingTensor(('m', 'k')), 
    ProfilingTensor(('n', 'k')),
    EmptyTensor(('m', 'n')),
    1e-5,
    configs=GEMM_SHAPES)

SWIGLU_NT = KernelArgs(
    ProfilingTensor(('m', 'k')),
    ProfilingTensor(('n', 'k')),
    ProfilingTensor(('n', 'k')),
    EmptyTensor(('m', 'n')),
    configs=GEMM_SHAPES)

LORA_NT_16 = KernelArgs(
    ProfilingTensor(('m', 'k')), # X
    ProfilingTensor(('n', 'k')), # W
    ProfilingTensor((16, 'k')), # A
    ProfilingTensor(('n', 16)), # B
    EmptyTensor(('m', 'n')),
    configs=GEMM_SHAPES
)

# seq_len stops at 8192 because ProfilingJob runs the ref kernel in fp64 
# and with larger seq_len the job errors with OOM
ATTENTION_SHAPES = [
    {'batch': 1, 'heads': 32, 'seq_len': s, 'head_dim': d}
    for d in (64, 128)
    for s in (1024, 2048, 4096, 8192)
]

ATTENTION_ARGS = KernelArgs(
    ProfilingTensor(('batch', 'seq_len', 'heads', 'head_dim')), # q (B, S, H, D)
    ProfilingTensor(('batch', 'seq_len', 'heads', 'head_dim')), # k
    ProfilingTensor(('batch', 'seq_len', 'heads', 'head_dim')), # v
    configs=ATTENTION_SHAPES,
)