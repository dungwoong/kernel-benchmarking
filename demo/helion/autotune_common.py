"""
Shared components for the *_autotune.py demos
"""
import os
import time
from dataclasses import fields

import torch
from helion_utils.kernel_runner import _get_cache_path, autotune_helion_kernel_single
from profile_utils import ExperimentOutput, write_csv_rows


def autotune_runner(runner, prob_args, mask, kernel, csv, output_cls=ExperimentOutput):
    """
    Autotunes for one shape using Helion 

    output_cls decides the shape columns
    """
    tensors = prob_args.tensors(mask)

    key = runner._tensors_to_key(*tensors)
    label = f"{kernel}_{key}"
    cache_path = _get_cache_path(runner.label, key)
    cached = os.path.exists(cache_path)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    autotune_helion_kernel_single(runner.fn, runner.label, key, tensors)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    shape = {f.name: prob_args._arg(f.name) for f in fields(output_cls)}
    shape = {dim: v for dim, v in shape.items() if v is not None}
    out = output_cls(
        label=label,
        autotune_seconds=None if cached else round(elapsed, 1),
        **shape,
    )
    write_csv_rows([output_cls.list_to_csv(out.values())], csv)
