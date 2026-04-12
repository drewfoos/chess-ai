"""Training package.

`chess_mcts` self-registers its native DLL directories in its own
__init__.py (see src/python/chess_mcts/__init__.py). We import it here as
a belt-and-suspenders safeguard so code paths that only touch `training`
(e.g. `python -m training.selfplay ...`) still get LibTorch/TensorRT DLLs
resolved before any downstream `import torch` or `import tensorrt`.
"""

try:
    import chess_mcts  # noqa: F401
except ImportError:
    # Dev workflows that don't need the C++ engine (pure-Python MCTS
    # fallback, docs generation, etc.) can run without it.
    pass
