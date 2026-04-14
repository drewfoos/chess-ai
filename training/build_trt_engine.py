"""Build a TensorRT engine (.trt) from an ONNX file.

The engine is built for FP16 with a batch-dim optimization profile so a single
engine covers batch=1 (UCI root expand) through the max self-play batch size.

Usage:
    python -m training.build_trt_engine --onnx model.onnx --output model.trt
"""

import argparse
import os
import sys


def _ensure_trt_dlls_on_path():
    """Windows: prepend $TENSORRT_PATH/bin to PATH before importing tensorrt.

    TensorRT's Python loader walks os.environ["PATH"] directly (it doesn't
    respect the Windows DLL search path), so os.add_dll_directory alone does
    not make nvinfer_10.dll discoverable — we have to mutate PATH.
    """
    if sys.platform != 'win32':
        return
    trt_root = os.environ.get('TENSORRT_PATH')
    if not trt_root:
        return
    bin_dir = os.path.join(trt_root, 'bin')
    if not os.path.isdir(bin_dir):
        return
    current = os.environ.get('PATH', '')
    if bin_dir not in current.split(os.pathsep):
        os.environ['PATH'] = bin_dir + os.pathsep + current
    try:
        os.add_dll_directory(bin_dir)
    except (AttributeError, OSError):
        pass


_ensure_trt_dlls_on_path()

import tensorrt as trt  # noqa: E402


def build_engine(
    onnx_path: str,
    output_path: str,
    fp16: bool = True,
    min_batch: int = 1,
    opt_batch: int = 128,
    max_batch: int = 256,
    workspace_mb: int = 2048,
    input_name: str = 'input',
    verbose: bool = False,
    refittable: bool = True,
) -> str:
    """Build a serialized TRT engine and write it to `output_path`.

    When `refittable=True` the engine includes refit data so subsequent
    generations can swap in new weights via `refit_engine()` without
    re-running the 30-90s plan-compilation step.

    Returns the output path. Raises RuntimeError on build failure.
    """
    severity = trt.Logger.INFO if verbose else trt.Logger.WARNING
    logger = trt.Logger(severity)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            errors = '\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"ONNX parse failed:\n{errors}")

    # Optimization profile: batch axis only. Spatial dims + channels are fixed.
    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, workspace_mb * (1 << 20)
    )
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if refittable:
        # REFIT lets subsequent generations reuse this engine with new weights
        # via OnnxParserRefitter, skipping plan compilation (~5-10s vs 60-90s).
        config.set_flag(trt.BuilderFlag.REFIT)

    # Determine the (C, H, W) part of the input shape from the network input.
    in_tensor = None
    for i in range(network.num_inputs):
        t = network.get_input(i)
        if t.name == input_name:
            in_tensor = t
            break
    if in_tensor is None:
        raise RuntimeError(f"Input '{input_name}' not found in ONNX graph")
    shape = in_tensor.shape  # e.g. (-1, 112, 8, 8) — dim 0 is dynamic
    chw = tuple(shape[1:])

    profile = builder.create_optimization_profile()
    profile.set_shape(
        input_name,
        (min_batch,) + chw,
        (opt_batch,) + chw,
        (max_batch,) + chw,
    )
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build returned None")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(bytes(serialized))
    return output_path


def refit_engine(
    onnx_path: str,
    engine_path: str,
    verbose: bool = False,
) -> str:
    """Refit an existing refittable engine with weights from a new ONNX file.

    The ONNX must have the same graph topology as the one the engine was
    built from (identical layer shapes and initializer names). Typically
    5-10s vs 30-90s for a full `build_engine()` on RTX-class hardware.

    Raises RuntimeError if the existing engine wasn't built with REFIT, if
    topology differs, or if the refit itself fails. Callers should catch
    and fall back to `build_engine()` on any failure.
    """
    if not os.path.exists(engine_path):
        raise RuntimeError(f"Engine not found for refit: {engine_path}")

    severity = trt.Logger.INFO if verbose else trt.Logger.WARNING
    logger = trt.Logger(severity)
    runtime = trt.Runtime(logger)
    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize engine at {engine_path}")

    try:
        refitter = trt.Refitter(engine, logger)
    except TypeError as e:
        # TensorRT's Refitter constructor returns nullptr for engines built
        # without REFIT, which pybind11 surfaces as TypeError. Normalize to
        # RuntimeError so callers only need to catch one type.
        raise RuntimeError(
            f"Engine at {engine_path} is not refittable "
            f"(built without REFIT flag?): {e}"
        )
    onnx_refitter = trt.OnnxParserRefitter(refitter, logger)
    if not onnx_refitter.refit_from_file(onnx_path):
        errs = '\n'.join(
            str(onnx_refitter.get_error(i)) for i in range(onnx_refitter.num_errors)
        )
        raise RuntimeError(f"OnnxParserRefitter failed:\n{errs}")

    missing = refitter.get_missing_weights()
    if missing:
        raise RuntimeError(
            f"Refit incomplete — {len(missing)} weights unset "
            f"(first: {missing[0] if missing else 'n/a'})"
        )

    if not refitter.refit_cuda_engine():
        raise RuntimeError("refit_cuda_engine() returned False")

    # Re-serialize so the on-disk engine reflects the refit weights. C++
    # TRTEvaluator reloads from disk each generation, so the updated plan
    # is picked up next time the engine path is consumed.
    serialized = engine.serialize()
    if serialized is None:
        raise RuntimeError("engine.serialize() returned None after refit")
    with open(engine_path, 'wb') as f:
        f.write(bytes(serialized))
    return engine_path


def build_or_refit_engine(
    onnx_path: str,
    output_path: str,
    can_refit: bool,
    **build_kwargs,
) -> tuple[str, bool]:
    """Refit an existing engine if possible, otherwise build from scratch.

    `can_refit` is the caller's assertion that the existing engine at
    `output_path` has the same architecture as the new ONNX. Even with
    `can_refit=True`, refit failures (stale engine, missing REFIT flag,
    topology drift) fall back to a full build.

    Returns (engine_path, refitted_flag) so callers can report timing.
    """
    if can_refit and os.path.exists(output_path):
        try:
            refit_engine(onnx_path, output_path)
            return output_path, True
        except Exception as e:
            print(f"  [trt] refit failed ({e}) — falling back to full build")
    return build_engine(onnx_path, output_path, **build_kwargs), False


def main():
    parser = argparse.ArgumentParser(description="Build TensorRT engine from ONNX")
    parser.add_argument('--onnx', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--no-fp16', action='store_true')
    parser.add_argument('--min-batch', type=int, default=1)
    parser.add_argument('--opt-batch', type=int, default=128)
    parser.add_argument('--max-batch', type=int, default=256)
    parser.add_argument('--workspace-mb', type=int, default=2048)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    out = build_engine(
        args.onnx, args.output,
        fp16=not args.no_fp16,
        min_batch=args.min_batch, opt_batch=args.opt_batch, max_batch=args.max_batch,
        workspace_mb=args.workspace_mb,
        verbose=args.verbose,
    )
    print(f"Wrote engine: {out} ({os.path.getsize(out):,} bytes)")


if __name__ == '__main__':
    main()
