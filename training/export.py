"""Export trained chess network to TorchScript for C++ inference.

Usage:
    python -m training.export --checkpoint checkpoints/model_epoch_10.pt --output model.pt
"""

import argparse

import torch

from training.config import NetworkConfig
from training.model import ChessNetwork


def export_torchscript(
    model: ChessNetwork,
    output_path: str,
    device: str = 'cpu',
):
    """Export a ChessNetwork to TorchScript via tracing.

    Args:
        model: Trained ChessNetwork instance.
        output_path: Path to save the .pt TorchScript file.
        device: Device to export on ('cpu' recommended for portability).
    """
    model = model.to(device)
    model.eval()

    # Trace with example input
    example = torch.randn(1, model.config.input_planes, 8, 8, device=device)
    with torch.no_grad():
        traced = torch.jit.trace(model, example)

    traced.save(output_path)
    print(f"Exported TorchScript model to {output_path}")


def export_from_checkpoint(checkpoint_path: str, output_path: str):
    """Load a training checkpoint and export to TorchScript."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    model = ChessNetwork(config)
    model.load_state_dict(checkpoint['model_state_dict'])

    export_torchscript(model, output_path)
    print(f"  Config: {config.num_blocks} blocks, {config.num_filters} filters")
    print(f"  Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export chess network to TorchScript')
    parser.add_argument('--checkpoint', required=True, help='Training checkpoint .pt file')
    parser.add_argument('--output', required=True, help='Output TorchScript .pt file')
    args = parser.parse_args()

    export_from_checkpoint(args.checkpoint, args.output)
