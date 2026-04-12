import torch
import argparse
import cuda.bindings.driver as cuda
from triton.testing import do_bench
import time



def validate(expected, out):
    expected = expected.float()
    out = out.float()
    diff = (out - expected).abs()
    max_abs = diff.max().item()
    max_rel = (diff / (expected.abs().clamp(min=1.0))).max().item()
    return max_abs, max_rel

def get_args():
    parser = argparse.ArgumentParser(description="Profiling Program For Gemm-Based Kernel")
    parser.add_argument("m", type=int)
    parser.add_argument("n", type=int)
    parser.add_argument("k", type=int)
    parser.add_argument("--atol", type=float, default=1e-1)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--label", type=str, default="unknown")
    parser.add_argument("--output_keys_only", type=bool, default=False)
    args = parser.parse_args()
    return args

# tensors[-1] must be the output tensor
def run_experiment(args, torch_tensors, torch_ref, other_tensors, other_kernel):
    output = {
        "label": args.label,
        "atol": args.atol,
        "rtol": args.rtol,
        "time_ms": None, 
        "allclose": None,
        "max_abs": None,
        "max_rel": None,
    }

    def print_keys():
        output_str = ','.join(str(k) for k in output.keys())
        print(output_str)
    def print_output():
        output_str = ','.join(str(v) for v in output.values())
        print(output_str)
    
    if args.output_keys_only:
        print_keys()
        return
    
    # try:
    ref = torch_ref(*torch_tensors)
    o = other_tensors(*other_tensors)

    output['allclose'] = torch.allclose(ref, o, atol=args.atol, rtol = args.rtol)
    output['max_abs'], output['max_rel'] = validate(ref, o)

    # timing
    time.sleep(1) # ??
    output['time_ms'] = do_bench(lambda: other_kernel(*other_tensors))
    # except Exception as e:
    #     print(e)
    # finally:
    print_output()

