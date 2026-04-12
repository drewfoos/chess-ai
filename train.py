"""Interactive training launcher with sensible defaults."""
import sys
import os

# Import torch BEFORE adding build/Release to path, to avoid DLL conflicts
# between PyTorch's bundled CUDA and LibTorch's CUDA
import torch

# Add C++ MCTS module to path if available
build_release = os.path.join(os.path.dirname(__file__), 'build', 'Release')
if os.path.isdir(build_release):
    sys.path.insert(0, build_release)


def prompt(label, default):
    """Prompt user for a value, using default if they press Enter."""
    val = input(f"  {label} [{default}]: ").strip()
    return val if val else str(default)


def main():
    print("=" * 50)
    print("  Chess AI Training Launcher")
    print("=" * 50)
    print()

    # Check for C++ MCTS
    try:
        import chess_mcts  # noqa: F401
        print("  C++ MCTS: Available (CUDA)")
    except ImportError:
        print("  C++ MCTS: Not available (using Python MCTS)")

    # Probe for TensorRT (needs tensorrt wheel + TENSORRT_PATH on Windows)
    try:
        from training import build_trt_engine  # noqa: F401
        import tensorrt  # noqa: F401
        has_trt = True
        print(f"  TensorRT: Available ({tensorrt.__version__})")
    except Exception:
        has_trt = False
        print("  TensorRT: Not available (falling back to LibTorch FP16)")

    has_cuda = torch.cuda.is_available()
    if has_cuda:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  GPU: None (CPU only)")
    print()

    print("Configure training (press Enter for defaults):\n")

    generations = int(prompt("Generations", 10))
    games = int(prompt("Games per generation", 400))
    simulations = int(prompt("Simulations per move", 400))
    max_moves = int(prompt("Max moves per game", 200))
    device = prompt("Device", "cuda" if has_cuda else "cpu")
    batch_size = int(prompt("Training batch size", 2048))
    lr = float(prompt("Learning rate", 0.001))
    blocks = int(prompt("Network blocks", 10))
    filters = int(prompt("Network filters", 128))

    # Tiered-net schedule: smaller net early (faster self-play → more data per hour),
    # scale up to the full size once data has accumulated.
    network_schedule = None
    tier_reply = input("  Enable tiered-net schedule? [y/N]: ").strip().lower()
    if tier_reply == 'y':
        init_blocks = int(prompt("  Initial blocks", 6))
        init_filters = int(prompt("  Initial filters", 64))
        scale_gen = int(prompt("  Scale-up at generation", 20))
        final_blocks = int(prompt("  Final blocks", blocks))
        final_filters = int(prompt("  Final filters", filters))
        network_schedule = [(1, init_blocks, init_filters), (scale_gen, final_blocks, final_filters)]
        # When tiered, the initial tier drives the model size.
        blocks, filters = init_blocks, init_filters

    window_size = int(prompt("Sliding window size", 5))
    parallel_games = int(prompt("Parallel games (C++ cross-game batching)", 128))
    output_dir = prompt("Output directory", "models/current_run")

    # TensorRT is the fast path on consumer Nvidia GPUs (RTX 3080 etc.).
    # Default to on when available; otherwise don't offer the prompt.
    use_trt = False
    if has_trt and has_cuda:
        trt_reply = input("  Use TensorRT backend for self-play? [Y/n]: ").strip().lower()
        use_trt = trt_reply != 'n'

    # Optional enhancements — Y/N prompts when a default resource is present.
    default_syzygy = os.path.join(os.path.dirname(__file__), 'syzygy')
    syzygy_path = None
    if os.path.isdir(default_syzygy):
        reply = input(f"  Use Syzygy tablebases at {default_syzygy}? [Y/n]: ").strip().lower()
        if reply != 'n':
            syzygy_path = default_syzygy
    else:
        custom = prompt("Syzygy tablebase path (blank to disable)", "")
        if custom and os.path.isdir(custom):
            syzygy_path = custom
        elif custom:
            print(f"  Warning: Syzygy path '{custom}' not found, disabling")

    default_book = os.path.join(os.path.dirname(__file__), 'openings.fen')
    opening_book_path = None
    opening_book_fraction = 0.5
    if os.path.isfile(default_book):
        reply = input(f"  Use opening book at {default_book}? [Y/n]: ").strip().lower()
        if reply != 'n':
            opening_book_path = default_book
            opening_book_fraction = float(prompt("Book-seeded game fraction", 0.5))
    else:
        custom = prompt("Opening book FEN file (blank to disable)", "")
        if custom and os.path.isfile(custom):
            opening_book_path = custom
            opening_book_fraction = float(prompt("Book-seeded game fraction", 0.5))
        elif custom:
            print(f"  Warning: Opening book '{custom}' not found, disabling")

    # Resume option
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    resume_from = None
    if os.path.isdir(checkpoint_dir):
        checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith('model_gen_')])
        if checkpoints:
            latest = checkpoints[-1]
            resume = input(f"\n  Found {len(checkpoints)} checkpoint(s), latest: {latest}\n  Resume from it? [Y/n]: ").strip().lower()
            if resume != 'n':
                resume_from = os.path.join(checkpoint_dir, latest)
                print(f"  Resuming from {resume_from}")

    print()
    print("-" * 50)
    print(f"  Generations:  {generations}")
    print(f"  Games/gen:    {games}")
    print(f"  Simulations:  {simulations}")
    print(f"  Max moves:    {max_moves}")
    print(f"  Device:       {device}")
    print(f"  Batch size:   {batch_size}")
    print(f"  Network:      {blocks} blocks, {filters} filters")
    if network_schedule:
        print(f"  Tier schedule: {network_schedule}")
    print(f"  Parallel:     {parallel_games} games")
    print(f"  Backend:      {'TensorRT (FP16)' if use_trt else 'LibTorch (FP16 on CUDA)'}")
    print(f"  Output:       {output_dir}")
    if resume_from:
        print(f"  Resume from:  {resume_from}")
    print("-" * 50)
    print()

    confirm = input("  Start training? [Y/n]: ").strip().lower()
    if confirm == 'n':
        print("Cancelled.")
        return

    print()
    from training.selfplay import training_loop
    training_loop(
        generations=generations,
        games_per_gen=games,
        num_simulations=simulations,
        max_moves=max_moves,
        device=device,
        batch_size=batch_size,
        lr=lr,
        blocks=blocks,
        filters=filters,
        window_size=window_size,
        output_dir=output_dir,
        resume_from=resume_from,
        parallel_games=parallel_games,
        syzygy_path=syzygy_path,
        opening_book_path=opening_book_path,
        opening_book_fraction=opening_book_fraction,
        network_schedule=network_schedule,
        use_trt=use_trt,
    )


if __name__ == '__main__':
    main()
