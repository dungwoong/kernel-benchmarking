"""
compile_hel with a cache

C2 writes candidates to cache that is not committed. 
The wrapper copies the best kernel for each shape to a commitable folder 
with extra information on the best kernel in results.json

The files created for the cache:
res/c2_kernels/<tag>.py:  the winning CuteDSL file
res/c2_kernels/results.json: one record per tag

Delete res/c2_kernels after updating the compiler_2 submodule

Set C2_CACHE_DISABLE=1 to always recompile
"""
import json
import os
import shutil
from pathlib import Path

from compiler_2 import compile_hel as _compile_hel
from compiler_2.compiler import LoweredKernel

# TODO: consider using REPO_ROOT env variable throughout repository scripts
REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "res" / "c2_kernels"
RESULTS_FILE = "results.json"


def _record(best, cache_dir, tag):

    path = Path(cache_dir) / RESULTS_FILE
    records = json.loads(path.read_text()) if path.exists() else []

    run = json.loads((best.file_path.parent / RESULTS_FILE).read_text())
    won = next((r for r in run if r["file"] == best.file_path.name), {})

    records = [r for r in records if r["tag"] != tag]  # update record if tag matches (no duplicate)
    records.append({
        "tag": tag,
        "kernel_file": f"{tag}.py",
        "candidate": best.file_path.name,
        "num_candidates": len(run),
        "ms_median": won.get("ms_median"),
        "predicted_score": won.get("score"),
    })
    path.write_text(json.dumps(sorted(records, key=lambda r: r["tag"]), indent=2) + "\n")


def compile_hel(builder, *sample_inputs, name, cache_dir=CACHE_DIR, **kwargs):
    """
    Reuse the winning kernel if it has been compiled before
    If the winning kernel does not exist compile, save it and record what it measured
    """
    cache_dir = Path(cache_dir)
    path = cache_dir / f"{name}.py"

    if path.exists() and not os.environ.get("C2_CACHE_DISABLE"):
        print(f"c2 cache hit {path}")
        return LoweredKernel(path)

    best = _compile_hel(builder, *sample_inputs, name=name, **kwargs)
    cache_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best.file_path, path)  # Copy to commitable cache
    _record(best, cache_dir, name)
    print(f"c2 cache write {path}")
    return best
