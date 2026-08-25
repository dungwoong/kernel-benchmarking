"""
Dumps all IR representations of swiglu helion kernels for analysis.

Output structure:
  ir_dumps/label/triton/<key>.py    — helion-generated Triton code
  ir_dumps/label/ttir/<key>.ttir    — Triton IR (MLIR)
  ir_dumps/label/ttgir/<key>.ttgir  — Triton GPU IR
  ir_dumps/label/llir/<key>.ll      — LLVM IR / NVVM IR
  ir_dumps/label/ptx/<key>.ptx      — PTX assembly

"""

import importlib.util
import inspect
import os
import traceback

import torch
import triton
from triton.compiler import ASTSource
import helion
from helion import Config

CACHE_DIR = os.path.join(os.path.dirname(__file__), "autotune_cache")
OUT_DIR   = os.path.join(os.path.dirname(__file__), "ir_dumps")

DTYPE_TO_TRITON = {
    torch.bfloat16: "*bf16",
    torch.float16:  "*fp16",
    torch.float32:  "*fp32",
}

def get_base_folder(label):
    return os.path.join(OUT_DIR, label)


def write_file(label, subdir, filename, content):
    path = os.path.join(get_base_folder(label), subdir)
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, filename)
    mode = "wb" if isinstance(content, bytes) else "w"
    with open(full_path, mode) as f:
        f.write(content)
    print(f"  [{subdir:<5}]  {full_path}")


def _is_wrapper(k, v):
    if k.startswith("_") or not callable(v):
        return False
    try:
        return "_launcher" in inspect.signature(v).parameters
    except (ValueError, TypeError):
        return False


def dump_ir(label, kernel_fn, sample_args, key):
    # Triton code
    bound = kernel_fn.bind(sample_args)
    triton_code = bound.to_triton_code()
    write_file(label, "triton", f"{key}.py", triton_code)

    # Load from file so @triton.jit can call inspect.getsourcelines
    triton_path = os.path.join(get_base_folder(label), "triton", f"{key}.py")
    spec = importlib.util.spec_from_file_location(key, triton_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    jit_fn = next(v for k, v in vars(mod).items() if k.startswith("_helion_"))

    # Intercept the Python wrapper's _launcher to capture actual constexpr
    # values (e.g. _NUM_SM, _RDIM_SIZE_*) that helion passes at runtime.
    captured = {}
    def capturing_launcher(fn, grid, *args, **kwargs):
        captured["args"] = args

    wrapper_fn = next(v for k, v in vars(mod).items() if _is_wrapper(k, v))
    wrapper_fn(*sample_args, _launcher=capturing_launcher)

    # dtype = sample_args[0].dtype
    # ptr_type = DTYPE_TO_TRITON[dtype]
    signature = {}
    constexprs = {}
    for i, (name, _) in enumerate(inspect.signature(jit_fn.fn).parameters.items()):
        val = captured["args"][i]
        if isinstance(val, int):
            constexprs[name] = val
        elif isinstance(val, float):
            signature[name] = "fp32"
        else:
            assert hasattr(val, 'dtype') and val.dtype in DTYPE_TO_TRITON, f'Could not get dtype of {type(val)}'
            t = DTYPE_TO_TRITON[val.dtype]
            signature[name] = t

    try:
        target = triton.runtime.driver.active.get_current_target()
        src = ASTSource(fn=jit_fn, signature=signature, constexprs=constexprs)
        compiled = triton.compile(src, target=target)

        if "ttir"  in compiled.asm: write_file(label, "ttir",  f"{key}.ttir",  compiled.asm["ttir"])
        if "ttgir" in compiled.asm: write_file(label, "ttgir", f"{key}.ttgir", compiled.asm["ttgir"])
        if "llir"  in compiled.asm: write_file(label, "llir",  f"{key}.ll",    compiled.asm["llir"])
        if "ptx"   in compiled.asm: write_file(label, "ptx",   f"{key}.ptx",   compiled.asm["ptx"])
    except Exception:
        traceback.print_exc()