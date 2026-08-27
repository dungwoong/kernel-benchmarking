"""
Shared components for the *_autotune.py demos
"""
import time
from dataclasses import fields

import torch
from profile_utils import ExperimentOutput, write_csv_rows


def autotune_runner(runner, prob_args, mask, kernel, csv, output_cls=ExperimentOutput, tensors=None):
    """
    Autotunes for one shape using Helion 

    output_cls decides the shape columns
    """
    if tensors is None:
        tensors = prob_args.tensors(mask)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    key, ran = runner.autotune(*tensors)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    shape = {f.name: prob_args._arg(f.name) for f in fields(output_cls)}
    shape = {dim: v for dim, v in shape.items() if v is not None}
    out = output_cls(
        label=f"{kernel}_{key}",
        autotune_seconds=round(elapsed, 1) if ran else None,
        **shape,
    )
    write_csv_rows([output_cls.list_to_csv(out.values())], csv)
