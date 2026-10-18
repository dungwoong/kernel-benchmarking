"""Dump TorchInductor kernel artifacts from output_code.py"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
from pathlib import Path
from types import ModuleType
from typing import Any


def _set_inductor_debug_flags() -> dict[str, str]:
    changed = {}
    defaults = {
        "TORCHINDUCTOR_CACHE_DIR": str(Path.cwd() / "torchinductor_cache_dump"),
        "TRITON_CACHE_DIR": str(Path.cwd() / "triton_cache_dump"),
    }
    for key, value in defaults.items():
        if key not in os.environ:
            os.environ[key] = value
            changed[key] = value
    return changed


def _enable_artifacts() -> dict[str, Any]:
    import torch 

    triton_cfg = torch._inductor.config.triton 
    triton_cfg.store_cubin = True
    return {
        "triton.store_cubin": True,
    }


def _import_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_module_once(module: ModuleType) -> str:
    if hasattr(module, "benchmark_compiled_module"):
        fn = getattr(module, "benchmark_compiled_module")
        fn(times=1, repeat=1)
        return "benchmark_compiled_module(times=1, repeat=1)"
    if hasattr(module, "call"):
        return "call() available but not invoked automatically"
    return "no runnable entrypoint found"


def _is_kernel_object(name: str, value: Any) -> bool:
    if not name.startswith("triton_"):
        return False
    return hasattr(value, "run") or hasattr(value, "asm") or hasattr(value, "kernel")


def _safe_write_text(path: Path, value: Any) -> None:
    path.write_text(value if isinstance(value, str) else repr(value))


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return repr(value)


def _dump_kernel(name: str, kernel: Any, outdir: Path) -> dict[str, Any]:
    kdir = outdir / name
    kdir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"name": name, "written": []}

    attrs = sorted(set(dir(kernel)))
    (kdir / "dir.txt").write_text("\n".join(attrs))
    summary["written"].append("dir.txt")

    public_attrs = {}
    for attr in attrs:
        if attr.startswith("_"):
            continue
        try:
            value = getattr(kernel, attr)
        except Exception as exc:
            public_attrs[attr] = f"<error: {exc}>"
            continue
        if inspect.ismethod(value) or inspect.isfunction(value):
            public_attrs[attr] = f"<callable {type(value).__name__}>"
        else:
            public_attrs[attr] = _jsonable(value)
    (kdir / "public_attrs.json").write_text(json.dumps(public_attrs, indent=2, sort_keys=True))
    summary["written"].append("public_attrs.json")

    for attr in ("asm", "src", "metadata", "signature", "kernel", "fn", "constexprs"):
        if not hasattr(kernel, attr):
            continue
        try:
            value = getattr(kernel, attr)
        except Exception as exc:
            _safe_write_text(kdir / f"{attr}.txt", f"<error: {exc}>")
            summary["written"].append(f"{attr}.txt")
            continue

        if attr == "asm" and isinstance(value, dict):
            adir = kdir / "asm"
            adir.mkdir(exist_ok=True)
            for key, text in value.items():
                suffix = ".bin" if isinstance(text, (bytes, bytearray)) else ".txt"
                out = adir / f"{key}{suffix}"
                if isinstance(text, (bytes, bytearray)):
                    out.write_bytes(bytes(text))
                else:
                    _safe_write_text(out, text)
                summary["written"].append(str(out.relative_to(kdir)))
        elif isinstance(value, (bytes, bytearray)):
            out = kdir / f"{attr}.bin"
            out.write_bytes(bytes(value))
            summary["written"].append(out.name)
        elif isinstance(value, (dict, list, tuple)):
            out = kdir / f"{attr}.json"
            out.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True))
            summary["written"].append(out.name)
        else:
            out = kdir / f"{attr}.txt"
            _safe_write_text(out, value)
            summary["written"].append(out.name)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=Path, required=True, help="Path to TorchInductor output_code.py")
    parser.add_argument("--outdir", type=Path, required=True, help="Directory to write dumped artifacts into")
    args = parser.parse_args()

    env_changes = _set_inductor_debug_flags()
    debug_status = _enable_artifacts()

    args.outdir.mkdir(parents=True, exist_ok=True)
    module = _import_module(args.module)
    ran = _run_module_once(module)

    kernels = {
        name: value
        for name, value in vars(module).items()
        if _is_kernel_object(name, value)
    }

    summary = {
        "module": str(args.module),
        "outdir": str(args.outdir),
        "env_changes": env_changes,
        "debug_status": debug_status,
        "run_step": ran,
        "kernels": [],
    }

    for name, kernel in kernels.items():
        summary["kernels"].append(_dump_kernel(name, kernel, args.outdir))

    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
