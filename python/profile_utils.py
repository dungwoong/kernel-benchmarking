import time
import torch
import argparse
from triton import runtime
from triton.testing import do_bench
import math
import numpy as np
from typing import List, Dict
from dataclasses import dataclass, fields
from cdsl_fn_utils import compile_cutedsl

# ---------------------------------------
# Output format for any experiments
# Add more if necessary
# ---------------------------------------

@dataclass
class ExperimentOutput:
    """
    Enables most gemm-based experiments,
    metadata helps capture any additional params(e.g. lora dim)
    """
    label: str
    m: int=None
    n: int=None
    k: int=None
    ms_median: float=None
    ms_mean: float=None
    ms_std: float=None
    max_abs: float=None
    max_rel: float=None
    rmse: float=None
    metadata: str=None

    @classmethod
    def header(cls) -> tuple:
        return tuple(f.name for f in fields(cls))
    
    def values(self):
        return tuple(getattr(self, f.name) for f in fields(self))
    
    def items(self):
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def run(self, kernel, tensors, ref_output: torch.Tensor):
        """
        Make sure tensors are detached if necessary.
        Max abs/rel aren't that useful since the scale of the tensors influences it heavily
        """
        o = kernel(*tensors)
        o_casted = o.to(ref_output.dtype)

        self.rmse = get_rmse(ref_output, o_casted)
        self.max_abs, self.max_rel = get_max_errors(o_casted, ref_output)
        timings = do_bench(lambda: kernel(*tensors), return_mode='all')
        self.ms_median = np.median(timings).item()
        self.ms_mean = np.mean(timings).item()
        self.ms_std = np.std(timings).item()
    
    def run_ncu(self, kernel, tensors):
        kernel(*tensors)

    @staticmethod
    def list_to_csv(lst):
        return ','.join(str(x) for x in lst)


@dataclass
class TimingComparisonOutput:
    """
    Compare multiple methods for timing on the same workload
    """
    label: str
    m: int
    n: int
    k: int
    ms_median: float=None
    ms_mean: float=None
    ms_std: float=None
    ms_median_2: float=None
    ms_mean_2: float=None
    ms_std_2: float=None
    max_abs: float=None
    max_rel: float=None
    rmse: float=None
    metadata: str=None

    @classmethod
    def header(cls) -> tuple:
        return tuple(f.name for f in fields(cls))
    
    def values(self):
        return tuple(getattr(self, f.name) for f in fields(self))
    
    def items(self):
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def run(self, kernel, tensors, ref_output: torch.Tensor):
        """
        Make sure tensors are detached if necessary.
        Max abs/rel aren't that useful since the scale of the tensors influences it heavily
        """
        o = kernel(*tensors)
        o_casted = o.to(ref_output.dtype)

        self.rmse = get_rmse(ref_output, o_casted)
        self.max_abs, self.max_rel = get_max_errors(o_casted, ref_output)

        timings = do_bench(lambda: kernel(*tensors), return_mode='all')
        time.sleep(2)
        timings2 = cuda_timings_tritonbench(lambda: kernel(*tensors))
        self.ms_median = np.median(timings).item()
        self.ms_mean = np.mean(timings).item()
        self.ms_std = np.std(timings).item()
        self.ms_median_2 = np.median(timings2).item()
        self.ms_mean_2 = np.mean(timings2).item()
        self.ms_std_2 = np.std(timings2).item()
    
    def run_ncu(self, kernel, tensors):
        kernel(*tensors)

    @staticmethod
    def list_to_csv(lst):
        return ','.join(str(x) for x in lst)

def cuda_timings(func, warmup=10, bench=50):
    with torch.no_grad():
        for _ in range(warmup):
            func()
        
        torch.cuda.synchronize()
        
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(bench)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(bench)]
        
        for i in range(bench):
            start_events[i].record()
            func()
            end_events[i].record()
        torch.cuda.synchronize()
        
        timings = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        
    return timings

def cuda_timings_tritonbench(func, warmup=10, bench=50):
    di = runtime.driver.active.get_device_interface()
    cache = runtime.driver.active.get_empty_cache_for_benchmark()
    with torch.no_grad():
        for _ in range(warmup):
            func()
        
        start_events = [di.Event(enable_timing=True) for _ in range(bench)]
        end_events = [di.Event(enable_timing=True) for _ in range(bench)]
        
        for i in range(bench):
            runtime.driver.active.clear_cache(cache)
            start_events[i].record()
            func()
            end_events[i].record()

        di.synchronize()
        
        timings = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        
    return timings

def get_rmse(ref: torch.Tensor, o: torch.Tensor):
    assert o.dtype == ref.dtype
    mse = torch.nn.functional.mse_loss(o, ref, reduction='mean')
    rmse = mse.sqrt().item()
    return rmse

def get_max_errors(ref: torch.Tensor, o: torch.Tensor):
    ref = ref.float()
    o = o.float()
    diff = (o - ref).abs()
    max_abs = diff.max().item()
    max_rel = (diff / (ref.abs().clamp(min=1.0))).max().item()
    return max_abs, max_rel

def get_args(parse=True):
    """
    Gets typical arguments for a gemm-based profiling program
    User can add optional args on top if needed
    """
    parser = argparse.ArgumentParser(description="Profiling Program For Gemm-Based Kernel")
    parser.add_argument("m", type=int)
    parser.add_argument("n", type=int)
    parser.add_argument("k", type=int)

    # you can use this as a print flag
    parser.add_argument("--to_csv", action='store_true')
    if parse:
        args = parser.parse_args()
        return args
    return parser

def get_normal_bernoulli(shape, p=0.001, dtype=torch.bfloat16, device="cuda", ncu=False) -> torch.Tensor:
    """References FlashAttention3"""
    gen_device = "cpu" if ncu else device
    base_noise = torch.randn(shape, dtype=dtype, device=gen_device)
    large_noise = torch.randn(shape, dtype=dtype, device=gen_device) * 10
    mask = torch.bernoulli(torch.full(shape, p, dtype=torch.bfloat16, device=gen_device))

    return (base_noise + (large_noise * mask)).to(device)

def get_kaiming(shape, gain=2, dtype=torch.bfloat16, device="cuda", ncu=False):
    # weights, k dim(shape[1]) is fan in
    gen_device = "cpu" if ncu else device
    multiplier = gain / math.sqrt(shape[1])
    return torch.randn(shape, dtype=dtype, device=gen_device).mul(multiplier).to(device)


class ProfilingTensor:
    """
    Allows quick creation of tensors without launching
    CUDA kernels, and quick casting.

    The two tensors in this object should always hold the same data.
    """
    def __init__(self, shape=None):
        self.tensor_64 = None
        self.tensor_16 = None
        self._shape = shape
        self.is_cuda = False
    
    def with_config(self, config: Dict[str, int]):
        assert self._shape is not None
        ret = ProfilingTensor(self._shape)
        ret.populate(config)
        return ret
    
    def populate(self, config: Dict[str, int]):
        assert self._shape is not None or self.tensor_16 is not None
        if self._shape is not None:
            shape = [config[s] if isinstance(s, str) else s for s in self._shape]
            self._set_tensors(shape)
    
    def _set_tensors(self, shape):
        # Use CPU to avoid launching a cuda kernel
        self.tensor_64 = get_normal_bernoulli(shape, dtype=torch.float64, device="cpu")
        self.tensor_16 = self.tensor_64.to(torch.bfloat16)
    
    def cuda(self):
        """
        Call this right before profiling
        """
        self.tensor_64 = self.tensor_64.to('cuda')
        self.tensor_16 = self.tensor_16.to('cuda')
        self.is_cuda = True
        return self

    def concat(self, other, dim=-1):
        assert not self.is_cuda, 'call is_cuda after doing ops'
        obj = ProfilingTensor()
        obj.tensor_64 = torch.concat((self.tensor_64, other.tensor_64), dim=dim)
        obj.tensor_16 = torch.concat((self.tensor_16, other.tensor_16), dim=dim)
        return obj

    def __repr__(self):
        cuda_tag = '<cuda>' if self.is_cuda else ''
        return f"{self.__class__.__name__}{cuda_tag}{tuple(self.tensor_16.shape)}"

class EmptyTensor(ProfilingTensor):
    def with_config(self, config: Dict[str, int]):
        assert self._shape is not None
        ret = EmptyTensor(self._shape)
        ret.populate(config)
        return ret
    
    def _set_tensors(self, shape):
        # Only populate the bf16 tensor
        self.tensor_16 = torch.empty(shape, dtype=torch.bfloat16, device="cpu")
    
    def cuda(self):
        self.tensor_16 = self.tensor_16.to('cuda')
        self.is_cuda = True
        return self
    
    def concat(self, other, dim=-1):
        raise NotImplementedError()

class KernelArgs:
    """
    User chooses from a set of predefined configs, 
    rather than passing arbitrary values

    Example:
    t1 = ProfilingTensor((64, 'k'))
    t2 = ProfilingTensor(('k', 128))
    args = KernelArgs(t1, t2, configs=[{'k': 5}, {'k': 6}])
    a1 = args.with_config(1)
    """
    def __init__(self, *args, configs=None):
        self.args = args
        self._configs = configs
        self._populated_idx = None

    def with_config(self, idx: int):
        """
        Returned object will have populated torch tensors
        """
        assert idx < len(self._configs), "Invalid config"
        new_args = [a.with_config(self._configs[idx]) if isinstance(a, ProfilingTensor) else a for a in self.args]
        k = KernelArgs(*new_args, configs=self._configs)
        k._populated_idx = idx
        k.cuda()
        return k

    def cuda(self):
        assert self._populated_idx is not None
        [a.cuda() if isinstance(a, ProfilingTensor) else None for a in self.args]
    
    def _get(self, getter, mask=None):
        # always return list so you can spread into args
        return [
            getter(a) if isinstance(a, ProfilingTensor) else a 
            for idx, a in enumerate(self.args) 
            if (mask is None or idx in mask)]
    
    def tensors_64(self, mask=None):
        return self._get(lambda x: x.tensor_64, mask=mask)
    
    def tensors(self, mask=None):
        return self._get(lambda x: x.tensor_16, mask=mask)
    
    def __repr__(self):
        return f'KernelArgs({self.args})'


class ProfilingJob:
    def __init__(self, label: str, kernels: dict, args: KernelArgs, baseline: str, ref: str, arg_mask: dict=None):
        """
        label: workload
        kernels: dict of (name: str, kernel: callable)
        args: KernelArgs containing a set of tensors
        baseline: str index of the baseline
        ref: str index of the ref. NOTE: the callable kernels[ref] must be able to execute on FP64 inputs
        arg_mask: Each kernel may use different tensors from <args>. Index them with <arg_mask>
            e.g. {'k1': (0, 1), 'k2': (0, 1, 2)}
            Passing None as the value will default to all args. Passing no arg mask will use all args for all kernels.
        """
        arg_mask = dict() if arg_mask is None else arg_mask
        self.kernels = kernels
        self.baseline = baseline
        self.args = {k: args.tensors(arg_mask.get(k, None)) for k in kernels}
        self.ref = self.kernels[ref](*args.tensors_64(arg_mask.get(ref, None)))
        self.outputs = {k: ExperimentOutput(f'{label}:{args._populated_idx}:{k}') for k in kernels}
    
    def _run(self):
        for k in self.kernels:
            print(f'running {k}')
            self.outputs[k].run(self.kernels[k], self.args[k], self.ref)
            time.sleep(2)
    
    def _run_ncu(self):
        for k in self.kernels:
            self.outputs[k].run_ncu(self.kernels[k], self.args[k])
    
    def run(self, ncu=False):
        if ncu:
            self._run_ncu()
        else:
            self._run()
        self.output_results()
    
    def output_results(self):
        for k in self.outputs:
            print(ExperimentOutput.list_to_csv(self.outputs[k].values()))

def get_profiling_job_args(parse=True):
    """
    Arguments specifically for ProfilingJob setup
    """
    parser = argparse.ArgumentParser(description="Profiling Program For Gemm-Based Kernel")
    parser.add_argument("config", type=int, help="the config number")

    parser.add_argument("--ncu", action='store_true')
    if parse:
        args = parser.parse_args()
        return args
    return parser