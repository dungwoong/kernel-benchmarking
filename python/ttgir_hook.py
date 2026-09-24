"""
Pause Triton's TTGIR pipeline at a chosen pass, dump the module, optionally swap in your
hand-edited version, then run the rest of the pipeline. Triton 3.8 (NVIDIA), uses
knobs.runtime.add_stages_inspection_hook - no Triton rebuild.

Usage (before the kernel compiles, e.g. top of your Helion script):

    import ttgir_hook
    ttgir_hook.install(cut="tritongpu-coalesce", kernel="_helion_rmsnorm_lin_kernel", workdir="ttgir_hook")

Each compile of that kernel then writes, in `workdir`:

    <kernel>.<cut>.orig.mlir    module right BEFORE the cut pass (read-only reference, overwritten every run)
    <kernel>.<cut>.final.ttgir  final TTGIR after the whole pipeline (to diff / check your edit survived)

and if this file exists it is parsed and used instead of the .orig module:

    <kernel>.<cut>.edit.mlir    your edited copy -> `cp X.orig.mlir X.edit.mlir`, then edit

Cut names are the pass names from the MLIR_ENABLE_DUMP headers. Passes that appear more than once
take an occurrence suffix: "tritongpu-remove-layout-conversions#2". cut="end" pauses after the last
TTGIR pass (i.e. edit the final TTGIR before LLVM lowering). `python ttgir_hook.py` lists the names.

Edits change the compile-cache key automatically; TRITON_ALWAYS_COMPILE=1 is still a good idea.
"""
import hashlib
import os
import pathlib

from triton import knobs
from triton._C.libtriton import ir, passes, nvidia
from triton.backends.compiler import Language

_THIS_FILE = pathlib.Path(__file__)
_CFG = {"cut": "tritongpu-coalesce", "kernel": None, "workdir": "ttgir_hook"}


# --------------------------------------------------------------------------------------------
# The pass list of CUDABackend.make_ttgir (triton release/3.8.x), one entry per pass, so it can
# be split anywhere. Keep in sync with third_party/nvidia/backend/compiler.py if you upgrade.
# --------------------------------------------------------------------------------------------
def _ttgir_passes(opt, capability, dump_enabled):
    emuTF32 = capability // 10 >= 8
    cc = capability // 10
    P = []
    add = lambda name, fn: P.append((name, fn))

    add("convert-triton-to-tritongpu",
        lambda pm: passes.ttir.add_convert_to_ttgpuir(pm, f"cuda:{capability}", opt.num_warps, 32, opt.num_ctas))
    add("tritongpu-coalesce", lambda pm: passes.ttgpuir.add_coalesce(pm))
    add("tritongpu-F32DotTC", lambda pm: passes.ttgpuir.add_f32_dot_tc(pm, emuTF32))
    add("triton-nvidia-gpu-plan-cta", lambda pm: nvidia.passes.ttnvgpuir.add_plan_cta(pm))
    add("tritongpu-remove-layout-conversions", lambda pm: passes.ttgpuir.add_remove_layout_conversions(pm))
    add("tritongpu-optimize-thread-locality", lambda pm: passes.ttgpuir.add_optimize_thread_locality(pm))
    add("tritongpu-accelerate-matmul", lambda pm: passes.ttgpuir.add_accelerate_matmul(pm))
    add("tritongpu-remove-layout-conversions", lambda pm: passes.ttgpuir.add_remove_layout_conversions(pm))
    add("tritongpu-optimize-dot-operands", lambda pm: passes.ttgpuir.add_optimize_dot_operands(pm, capability >= 80))
    add("triton-nvidia-optimize-descriptor-encoding",
        lambda pm: nvidia.passes.ttnvgpuir.add_optimize_descriptor_encoding(pm))
    add("triton-loop-aware-cse", lambda pm: passes.ttir.add_loop_aware_cse(pm))
    if cc in (8, 9):
        add("tritongpu-fuse-nested-loops", lambda pm: passes.ttgpuir.add_fuse_nested_loops(pm))
        add("canonicalize", lambda pm: passes.common.add_canonicalizer(pm))
        add("triton-licm", lambda pm: passes.ttir.add_triton_licm(pm))
        add("canonicalize", lambda pm: passes.common.add_canonicalizer(pm))
        add("tritongpu-combine-tensor-select-and-if", lambda pm: passes.ttgpuir.add_combine_tensor_select_and_if(pm))
        add("nvgpu-warp-specialization",
            lambda pm: nvidia.passes.hopper.add_hopper_warpspec(pm, opt.num_stages, dump_enabled))
        add("tritongpu-assign-latencies", lambda pm: passes.ttgpuir.add_assign_latencies(pm, opt.num_stages))
        add("tritongpu-schedule-loops", lambda pm: passes.ttgpuir.add_schedule_loops(pm))
        add("tritongpu-pipeline", lambda pm: passes.ttgpuir.add_pipeline(pm, opt.num_stages, dump_enabled))
    elif cc >= 10:
        add("tritongpu-fuse-nested-loops", lambda pm: passes.ttgpuir.add_fuse_nested_loops(pm))
        add("canonicalize", lambda pm: passes.common.add_canonicalizer(pm))
        add("triton-licm", lambda pm: passes.ttir.add_triton_licm(pm))
        add("tritongpu-optimize-accumulator-init", lambda pm: passes.ttgpuir.add_optimize_accumulator_init(pm))
        add("tritongpu-hoist-tmem-alloc", lambda pm: passes.ttgpuir.add_hoist_tmem_alloc(pm, False))
        add("triton-nvidia-gpu-promote-lhs-to-tmem", lambda pm: nvidia.passes.ttnvgpuir.add_promote_lhs_to_tmem(pm))
        add("tritongpu-assign-latencies", lambda pm: passes.ttgpuir.add_assign_latencies(pm, opt.num_stages))
        add("tritongpu-schedule-loops", lambda pm: passes.ttgpuir.add_schedule_loops(pm))
        add("tritongpu-warp-specialize", lambda pm: passes.ttgpuir.add_warp_specialize(pm, opt.num_stages))
        add("tritongpu-pipeline", lambda pm: passes.ttgpuir.add_pipeline(pm, opt.num_stages, dump_enabled))
        add("tritongpu-optimize-partition-warps", lambda pm: passes.ttgpuir.add_optimize_partition_warps(pm))
        add("tritongpu-combine-tensor-select-and-if", lambda pm: passes.ttgpuir.add_combine_tensor_select_and_if(pm))
        add("tritongpu-hoist-tmem-alloc", lambda pm: passes.ttgpuir.add_hoist_tmem_alloc(pm, True))
        add("triton-nvidia-gpu-remove-tmem-tokens", lambda pm: nvidia.passes.ttnvgpuir.add_remove_tmem_tokens(pm))
    else:
        add("triton-licm", lambda pm: passes.ttir.add_triton_licm(pm))
    add("canonicalize", lambda pm: passes.common.add_canonicalizer(pm))
    add("triton-loop-aware-cse", lambda pm: passes.ttir.add_loop_aware_cse(pm))
    if cc == 8:
        add("tritongpu-prefetch", lambda pm: passes.ttgpuir.add_prefetch(pm))
    add("tritongpu-optimize-dot-operands", lambda pm: passes.ttgpuir.add_optimize_dot_operands(pm, capability >= 80))
    add("tritongpu-coalesce-async-copy", lambda pm: passes.ttgpuir.add_coalesce_async_copy(pm))
    add("triton-nvidia-optimize-tmem-layouts", lambda pm: nvidia.passes.ttnvgpuir.add_optimize_tmem_layouts(pm))
    add("triton-nvidia-tmem-load-reduce", lambda pm: nvidia.passes.ttnvgpuir.add_tmem_load_reduce(pm))
    if cc >= 9:
        add("triton-nvidia-tma-lowering", lambda pm: nvidia.passes.ttnvgpuir.add_tma_lowering(pm))
    add("tritongpu-remove-layout-conversions", lambda pm: passes.ttgpuir.add_remove_layout_conversions(pm))
    add("triton-nvidia-interleave-tmem", lambda pm: nvidia.passes.ttnvgpuir.add_interleave_tmem(pm))
    add("tritongpu-reduce-data-duplication", lambda pm: passes.ttgpuir.add_reduce_data_duplication(pm))
    add("tritongpu-reorder-instructions", lambda pm: passes.ttgpuir.add_reorder_instructions(pm))
    add("triton-loop-aware-cse", lambda pm: passes.ttir.add_loop_aware_cse(pm))
    add("symbol-dce", lambda pm: passes.common.add_symbol_dce(pm))
    add("triton-nvidia-gpu-fence-insertion", lambda pm: nvidia.passes.ttnvgpuir.add_fence_insertion(pm, capability))
    add("triton-nvidia-mma-lowering", lambda pm: nvidia.passes.ttnvgpuir.add_lower_mma(pm))
    add("sccp", lambda pm: passes.common.add_sccp(pm))
    add("cse", lambda pm: passes.common.add_cse(pm))
    add("canonicalize", lambda pm: passes.common.add_canonicalizer(pm))
    if "fpsan" in getattr(opt, "instrumentation_mode", ""):
        add("tritongpu-fp-sanitizer", lambda pm: passes.ttgpuir.add_fp_sanitizer(pm))
        add("tritongpu-remove-layout-conversions", lambda pm: passes.ttgpuir.add_remove_layout_conversions(pm))
        add("canonicalize", lambda pm: passes.common.add_canonicalizer(pm))
        add("cse", lambda pm: passes.common.add_cse(pm))

    # label repeated passes: name, name#2, name#3, ...
    seen, labeled = {}, []
    for name, fn in P:
        seen[name] = seen.get(name, 0) + 1
        labeled.append((name if seen[name] == 1 else f"{name}#{seen[name]}", fn))
    return labeled


def _split(labeled, cut):
    """
    Splits the list of passes <labeled> at <cut>
    """
    if cut == "end":
        return labeled, []
    names = [n for n, _ in labeled]
    if cut not in names:
        raise ValueError(f"cut '{cut}' not in pass list:\n  " + "\n  ".join(names))
    i = names.index(cut)
    return labeled[:i], labeled[i:]


def _run(mod, labeled, tag):
    """
    Run passes in <labeled> on mod
    """
    if not labeled:
        return
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    for _, fn in labeled:
        fn(pm)
    pm.run(mod, tag)


def _paths(kernel):
    d = pathlib.Path(_CFG["workdir"])
    stem = f"{kernel}.{_CFG['cut']}"
    return d / f"{stem}.orig.mlir", d / f"{stem}.edit.mlir", d / f"{stem}.final.ttgir"


def _make_ttgir_hooked(backend, mod, metadata, opt, capability):
    kernel = mod.get_entry_func_name()

    # Proceed normally if kernel name doesn't match
    if _CFG["kernel"] is not None and kernel != _CFG["kernel"]:
        return backend.make_ttgir(mod, metadata, opt, capability)

    if opt.maxnreg is not None:
        mod.set_attr("ttg.maxnreg", ir.builder(mod.context).get_int32_attr(opt.maxnreg))

    # dump_enabled must match what enable_debug() reports (the pipeliner / warpspec use it)
    dump_enabled = ir.pass_manager(mod.context).enable_debug()
    pre, post = _split(_ttgir_passes(opt, capability, dump_enabled), _CFG["cut"])

    orig, edit, final = _paths(kernel)
    orig.parent.mkdir(parents=True, exist_ok=True)

    # Run the pre-split passes
    _run(mod, pre, "make_ttgir_pre")
    orig.write_text(mod.str())
    print(f"[ttgir_hook] {kernel}: wrote module before '{_CFG['cut']}' -> {orig}")

    if edit.exists():
        print(f"[ttgir_hook] {kernel}: using EDITED module {edit}")
        # Pass modules context to edited version, else next passes won't work.
        ctx = mod.context
        mod = ir.parse_mlir_module(str(edit), ctx)
        # `.context` is a plain Python attribute that Triton's compiler attaches to modules
        # (see IRSource.make_ir); a freshly parsed module doesn't have it, so set it.
        mod.context = ctx

    _run(mod, post, "make_ttgir_post")
    final.write_text(mod.str())
    print(f"[ttgir_hook] {kernel}: final TTGIR -> {final}")

    metadata["tensordesc_meta"] = mod.get_tensordesc_metadata()
    return mod


def _key():
    # recompile whenever this file, the cut, or any edit file changes
    h = hashlib.sha256()
    h.update(_THIS_FILE.read_bytes())
    h.update(repr(sorted(_CFG.items())).encode())
    for p in sorted(pathlib.Path(_CFG["workdir"]).glob("*.edit.mlir")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    digest = h.hexdigest()
    return f"ttgir_hook:{digest}", digest


def _hook(self=None, stages=None, options=None, language=None, capability=None):
    if all(a is None for a in (stages, options, language, capability)):
        return _key()
    if language == Language.TRITON:
        stages["ttgir"] = lambda src, metadata: _make_ttgir_hooked(self, src, metadata, options, capability)
    return _key()


def install(cut="tritongpu-coalesce", kernel=None, workdir="ttgir_hook"):
    _CFG.update(cut=cut, kernel=kernel, workdir=workdir)
    knobs.runtime.add_stages_inspection_hook = _hook
    print(f"[ttgir_hook] installed: cut={cut} kernel={kernel} workdir={workdir}")


def uninstall():
    knobs.runtime.add_stages_inspection_hook = None


if __name__ == "__main__":
    class _Opt:  # dummy options just to print names
        num_warps, num_ctas, num_stages, maxnreg, instrumentation_mode = 4, 1, 3, None, ""
    for i, (n, _) in enumerate(_ttgir_passes(_Opt, 90, False)):
        print(f"{i:2d}  {n}")