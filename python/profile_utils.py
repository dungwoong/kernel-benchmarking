import torch
import argparse
from triton.testing import do_bench
import math
import numpy as np
from dataclasses import dataclass, fields

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
    m: int
    n: int
    k: int
    use_do_bench: bool=True
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
        if self.use_do_bench:
            timings = do_bench(lambda: kernel(*tensors), return_mode='all')
        else:
            timings = self.cuda_timings(lambda: kernel(*tensors))
        self.ms_median = np.median(timings).item()
        self.ms_mean = np.mean(timings).item()
        self.ms_std = np.std(timings).item()
    
    def cuda_timings(self, func, warmup=10, bench=50):
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
    
    def run_ncu(self, kernel, tensors):
        kernel(*tensors)

    @staticmethod
    def list_to_csv(lst):
        return ','.join(str(x) for x in lst)

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
