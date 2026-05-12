import os
import io
from pathlib import Path
from typing import Callable

import torch
import tensorrt as trt

REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINE_CACHE_DIR = REPO_ROOT / ".cache" / "tensorrt"
ENGINE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_VERBOSE = os.environ.get("TRT_DEBUG") == "1"
_LOGGER = trt.Logger(trt.Logger.VERBOSE if _VERBOSE else trt.Logger.WARNING)
_RUNTIME = trt.Runtime(_LOGGER)


def _build_engine(onnx_bytes: bytes) -> bytes:
    builder = trt.Builder(_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, _LOGGER)
    if not parser.parse(onnx_bytes):
        errs = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed:\n{errs}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    config.set_flag(trt.BuilderFlag.BF16)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build returned None")
    return bytes(serialized)


def _export_onnx(module: torch.nn.Module,
                 example_inputs: tuple,
                 input_names: list,
                 output_name: str) -> bytes:
    buf = io.BytesIO()
    module = module.eval()
    with torch.no_grad():
        torch.onnx.export(
            module,
            example_inputs,
            buf,
            input_names=input_names,
            output_names=[output_name],
            opset_version=17,  # 17 adds LayerNormalization so TRT can fuse RMSNorm (Trinity uses 13)
            do_constant_folding=True,
            dynamo=False,
        )
    return buf.getvalue()


def build_trt_runner(
    module: torch.nn.Module,
    example_inputs: tuple,
    output_shape: tuple,
    cache_key: str,
    input_names: list,
    output_name: str = "output",
    output_dtype: torch.dtype = torch.bfloat16,
) -> Callable:
    # Cache by (workload, shape, dtype): Trinity rebuilds per process
    # Cache key does not hash module source: rm .cache/tensorrt/ after editing a module
    engine_path = ENGINE_CACHE_DIR / f"{cache_key}.engine"

    if engine_path.exists():
        engine_bytes = engine_path.read_bytes()
    else:
        onnx_bytes = _export_onnx(module, example_inputs, input_names, output_name)
        (ENGINE_CACHE_DIR / f"{cache_key}.onnx").write_bytes(onnx_bytes)
        engine_bytes = _build_engine(onnx_bytes)
        engine_path.write_bytes(engine_bytes)

    engine = _RUNTIME.deserialize_cuda_engine(engine_bytes)
    context = engine.create_execution_context()
    output = torch.empty(output_shape, dtype=output_dtype, device="cuda")
    # Dedicated non-default stream: TRT warns on default stream and adds extra syncs
    stream = torch.cuda.Stream()

    _refs = (engine, context, output, stream)  # keep alive across closure use

    def run(*tensors: torch.Tensor) -> torch.Tensor:
        assert len(tensors) == len(input_names), (
            f"expected {len(input_names)} inputs, got {len(tensors)}"
        )
        for name, t in zip(input_names, tensors):
            context.set_tensor_address(name, t.data_ptr())
        context.set_tensor_address(output_name, output.data_ptr())
        # execute_async_v3 (vs Trinity's execute_v2): name-based bindings, explicit stream
        with torch.cuda.stream(stream):
            context.execute_async_v3(stream_handle=stream.cuda_stream)
        stream.synchronize()
        return output

    run._trt_refs = _refs
    return run
