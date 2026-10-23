"""
Bundle everything Triton produces for a kernel into one folder per compile:

    <out>/<kernel_name>/<cache_hash[:12]>/
        <name>.ttir  .ttgir  .llir  .ptx  .cubin  .json   (copied from the Triton cache)
        <name>.sass                                         (disassembled from the cubin)
        source.py                                           (the @triton.jit source file, e.g. Helion's generated code
                                                             including its host/launch function)
        metadata.json                                       (options + launch metadata, pretty-printed)

Usage: import this module (and call install()) before the kernel runs.

    import triton_bundle
    triton_bundle.install("kernel_bundles", only="_helion_rmsnorm_lin_kernel")

Works on cache hits too, so TRITON_ALWAYS_COMPILE is not needed. Intermediate IR files are
only in the cache if TRITON_STORE_BINARY_ONLY is unset.
"""
import inspect
import json
import os
import shutil

import triton
from triton import knobs


def install(out_dir="kernel_bundles", only=None):
    os.makedirs(out_dir, exist_ok=True)
    previous = knobs.compilation.listener

    def listener(*, src, metadata, metadata_group, times, cache_hit):
        if previous is not None:
            previous(src=src, metadata=metadata, metadata_group=metadata_group, times=times, cache_hit=cache_hit)

        name = metadata.get("name", "kernel")
        if only is not None and name != only:
            return
        dest = os.path.join(out_dir, name, str(metadata.get("hash", "nohash"))[:12])
        os.makedirs(dest, exist_ok=True)

        # 1. every stage file Triton put in the cache (ttir, ttgir, llir, ptx, cubin, json)
        cubin_path = None
        for filename, path in metadata_group.items():
            shutil.copy(path, os.path.join(dest, filename))
            if filename.endswith(".cubin"):
                cubin_path = path

        # 2. SASS (needs nvdisasm / cuobjdump on PATH)
        if cubin_path is not None:
            try:
                from triton.tools.disasm import get_sass
                with open(cubin_path, "rb") as f:
                    sass = get_sass(f.read())
                with open(os.path.join(dest, f"{name}.sass"), "w") as f:
                    f.write(sass)
            except Exception as e:  # disassembler missing etc.
                print(f"[triton_bundle] sass skipped: {e}")

        # 3. the source file the kernel came from (for Helion: the generated Triton module)
        fn = getattr(src, "fn", None)  # ASTSource -> JITFunction
        py_fn = getattr(fn, "fn", None)
        if py_fn is not None:
            try:
                shutil.copy(inspect.getsourcefile(py_fn), os.path.join(dest, "source.py"))
            except (TypeError, OSError):
                with open(os.path.join(dest, "source.py"), "w") as f:
                    f.write(getattr(fn, "src", ""))

        # 4. readable metadata (num_warps, num_stages, maxnreg, enable_fp_fusion, shared, ...)
        with open(os.path.join(dest, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"[triton_bundle] {name} -> {dest} (cache_hit={cache_hit})")

    knobs.compilation.listener = listener