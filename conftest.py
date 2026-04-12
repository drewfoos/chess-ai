"""Global pytest setup.

On Windows, register the native DLL directories that `chess_mcts.pyd`
(LibTorch, optional TensorRT) and `import tensorrt` depend on, so tests run
cleanly in subprocesses without relying on inherited PATH.
"""
import os
import sys


def _ensure_trt_dlls_on_path():
    if sys.platform != 'win32':
        return
    trt_root = os.environ.get('TENSORRT_PATH')
    if not trt_root:
        return
    bin_dir = os.path.join(trt_root, 'bin')
    if not os.path.isdir(bin_dir):
        return
    # TensorRT's Python __init__ uses its own loader that walks
    # os.environ["PATH"] (not the Windows DLL search path), so
    # os.add_dll_directory alone isn't enough — we must mutate PATH too.
    current = os.environ.get('PATH', '')
    if bin_dir not in current.split(os.pathsep):
        os.environ['PATH'] = bin_dir + os.pathsep + current
    try:
        os.add_dll_directory(bin_dir)
    except (AttributeError, OSError):
        pass


_ensure_trt_dlls_on_path()

# Import the training package so its DLL-directory hook (libtorch/lib + any
# TensorRT lib dirs) runs before any test touches chess_mcts.
try:
    import training  # noqa: F401
except ImportError:
    pass
