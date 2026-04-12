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
        import chess_mcts
        print("  C++ MCTS: Available (CUDA)")
    except ImportError:
        print("  C++ MCTS: Not available (using Python MCTS)")
    print()
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  GPU: None (CPU only)")
    print()

    print("Configure training (press Enter for defaults):\n")

    generations = int(prompt("Generations", 10))
    games = int(prompt("Games per generation", 100))
    simulations = int(prompt("Simulations per move", 200))
    max_moves = int(prompt("Max moves per game", 200))
    device = prompt("Device", "cuda" if has_cuda else "cpu")
    batch_size = int(prompt("Training batch size", 2048))
    lr = float(prompt("Learning rate", 0.001))
    blocks = int(prompt("Network blocks", 10))
    filters = int(prompt("Network filters", 128))
    window_size = int(prompt("Sliding window size", 5))
    output_dir = prompt("Output directory", "selfplay_output")

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
    )


if __name__ == '__main__':
    main()
