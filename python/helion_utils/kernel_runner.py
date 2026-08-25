import os
import torch
import helion
from helion import Config
from bench_kernels.helion.matmul import matmul_kernel
from bench_kernels.helion.rmsnorm_linear import rmsnorm_lin_kernel
from bench_kernels.helion.swiglu import swiglu_kernel
from bench_kernels.helion.lora import lora_kernel
from bench_kernels.helion.attention import attention_kernel_fn
from helion_utils.dump_ir import dump_ir

CACHE_DIR = os.path.join(os.path.dirname(__file__), "autotune_cache")

def _get_cache_path(label, key):
    return os.path.join(CACHE_DIR, label, f'{key}.json')

def autotune_helion_kernel_single(kernel_fn, label, key, tensors):
    cache_path = _get_cache_path(label, key)
    if os.path.exists(cache_path):
        print(f'Skipping {key} (cache exists)')
        return
    os.makedirs(os.path.join(CACHE_DIR, label), exist_ok=True)

    print(f"Autotuning {key}")
    # this was used in the matmul example https://github.com/pytorch/helion/blob/main/examples/matmul.py
    # static shapes gives perf boost
    # tl.dot is pipelined with num_stages
    hk = helion.kernel(
        static_shapes=True,
        )
    best_config = hk(kernel_fn).autotune(tensors)
    best_config.save(cache_path)
    print(f"Saved -> {cache_path}")

def autotune_helion_kernel(kernel_fn, label, shapes_iter):
    """
    shapes is a callable that yields (key, [tensors]) tuples
    """
    for key, tensors in shapes_iter():
        autotune_helion_kernel_single(kernel_fn, label, key, tensors)



class HelionKernel:
    """
    Helps automatically designate kernel label, etc.
    """
    def __init__(self, fn, label):
        self.fn = fn
        self.label = label

    def _tensors_to_key(self, *tensors):
        tensors = [t for t in tensors if isinstance(t, torch.Tensor)] # filter out constant args
        dim_list = []
        for t in tensors:
            dim_list.extend([str(dim) for dim in t.shape])
        return '_'.join(dim_list)

    def compile(self, *tensors, allow_compile=True):
        key = self._tensors_to_key(*tensors)
        cache_path = _get_cache_path(self.label, key)
        if not os.path.exists(cache_path):
            assert allow_compile, f"Kernel {self.label} {key} not found in cache. Set allow_compile to True or pre-compile"
            autotune_helion_kernel_single(self.fn, self.label, key, tensors)

        # compiled_fn should be a complete function that includes host + device code
        compiled_fn = helion.kernel(config=Config.load(cache_path))(self.fn)
        return compiled_fn

    def dump_ir(self, compiled_kernel, sample_args):
        """
        Dumps IR to ir_dumps/{self.label}/(triton, ttir, etc.)
        """
        dump_ir(self.label, compiled_kernel, sample_args, self._tensors_to_key(*sample_args))

Matmul = HelionKernel(matmul_kernel, 'matmul')
RMSNormLinear = HelionKernel(rmsnorm_lin_kernel, 'rmsnorm_lin')
SwiGLU = HelionKernel(swiglu_kernel, 'swiglu')
LoRA = HelionKernel(lora_kernel, 'LoRA')
Attention = HelionKernel(attention_kernel_fn, 'attention')
