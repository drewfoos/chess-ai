"""Build a TensorRT engine (.trt) from an ONNX file.

The engine is built for FP16 with a batch-dim optimization profile so a single
engine covers batch=1 (UCI root expand) through the max self-play batch size.

We follow Lc0's pattern (src/neural/backends/onnx/network_onnx.cc:675-742):
build fresh per weight-set, use a persistent **timing cache** to amortize
kernel-selection across rebuilds. A warm timing cache turns a 60s rebuild into
~10s for our 20b×256f network on RTX 3080.

We deliberately do NOT use TRT's native refit (`BuilderFlag.REFIT` +
`OnnxParserRefitter`). `torch.onnx.export` with `do_constant_folding=True`
folds BN-fused conv weights into anonymous constants; those are no longer
refittable, but `refitter.get_missing_weights()` doesn't flag them because
TRT never knew they were initializers. Refit silently preserved build-time
weights for folded paths while updating the rest — producing an engine that
ran, but played catastrophically (50x slower, instant blunders). Lc0
sidesteps this by rebuilding on every weight-set change.

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
    opt_batch: int = 256,
    max_batch: int = 512,
    workspace_mb: int = 2048,
    input_name: str = 'input',
    verbose: bool = False,
    timing_cache_path: str | None = None,
) -> str:
    """Build a serialized TRT engine and write it to `output_path`.

    When `timing_cache_path` is set, the builder loads that cache (if present)
    before building and writes the updated cache back after. The cache holds
    per-kernel timing decisions and survives across weight-set changes as long
    as the network topology is the same, so subsequent builds with different
    weights but identical architecture are ~5x faster.

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

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, workspace_mb * (1 << 20)
    )
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Timing cache: replay kernel-selection decisions from prior builds so a
    # cold build (~60s) becomes a warm rebuild (~10s). Lc0 achieves the same
    # via ORT's trt_timing_cache_enable setting.
    timing_cache = None
    if timing_cache_path is not None:
        if os.path.exists(timing_cache_path):
            with open(timing_cache_path, 'rb') as f:
                cache_bytes = f.read()
            timing_cache = config.create_timing_cache(cache_bytes)
        else:
            timing_cache = config.create_timing_cache(b'')
        # ignore_mismatch=False: if the cache came from a different GPU/driver,
        # TRT discards incompatible entries rather than failing the build.
        config.set_timing_cache(timing_cache, ignore_mismatch=False)

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

    # Persist the (possibly enriched) timing cache for the next build.
    if timing_cache_path is not None and timing_cache is not None:
        cache_out = timing_cache.serialize()
        os.makedirs(
            os.path.dirname(os.path.abspath(timing_cache_path)) or '.',
            exist_ok=True,
        )
        with open(timing_cache_path, 'wb') as f:
            f.write(bytes(cache_out))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(bytes(serialized))
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Build TensorRT engine from ONNX")
    parser.add_argument('--onnx', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--no-fp16', action='store_true')
    parser.add_argument('--min-batch', type=int, default=1)
    parser.add_argument('--opt-batch', type=int, default=128)
    parser.add_argument('--max-batch', type=int, default=256)
    parser.add_argument('--workspace-mb', type=int, default=2048)
    parser.add_argument('--timing-cache', type=str, default=None,
                        help='Path to persistent timing cache file. Warm cache '
                             'cuts rebuild time ~6x (60s → 10s on RTX 3080).')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    out = build_engine(
        args.onnx, args.output,
        fp16=not args.no_fp16,
        min_batch=args.min_batch, opt_batch=args.opt_batch, max_batch=args.max_batch,
        workspace_mb=args.workspace_mb,
        timing_cache_path=args.timing_cache,
        verbose=args.verbose,
    )
    print(f"Wrote engine: {out} ({os.path.getsize(out):,} bytes)")


if __name__ == '__main__':
    main()
