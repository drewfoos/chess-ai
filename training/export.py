"""Export trained chess network to TorchScript for C++ inference.

Usage:
    python -m training.export --checkpoint checkpoints/model_epoch_10.pt --output model.pt
"""

import argparse

import torch

from training.config import NetworkConfig
from training.model import ChessNetwork


class _ExportWrapper(torch.nn.Module):
    """Wraps ChessNetwork to apply softmax to value head for inference.

    Returns (policy_logits, value_probs, mlh). MLH is exposed to the C++ side
    so the MCTS PUCT can apply a moves-left bonus (prefer shorter wins / longer
    losses). The Python MCTS uses its own path for MLH.
    """

    def __init__(self, model: ChessNetwork):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        policy_logits, value_logits, mlh = self.model(x)
        value_probs = torch.softmax(value_logits, dim=1)
        return policy_logits, value_probs, mlh


def export_torchscript(
    model: ChessNetwork,
    output_path: str,
    device: str = 'cpu',
):
    """Export a ChessNetwork to TorchScript via tracing.

    The exported model applies softmax to the value head output,
    so it returns (policy_logits, value_probabilities).
    """
    # Preserve the model's training device — callers pass a live model mid-run.
    original_device = next(model.parameters()).device
    was_training = model.training
    try:
        model.to(device)
        model.eval()

        wrapper = _ExportWrapper(model)
        wrapper.eval()

        example = torch.randn(1, model.config.input_planes, 8, 8, device=device)
        with torch.no_grad():
            traced = torch.jit.trace(wrapper, example)

        traced.save(output_path)
        print(f"Exported TorchScript model to {output_path}")
    finally:
        model.to(original_device)
        if was_training:
            model.train()


def export_onnx(
    model: ChessNetwork,
    output_path: str,
    opset_version: int = 18,
    device: str = 'cpu',
):
    """Export a ChessNetwork to ONNX with a dynamic batch axis.

    Produces (policy_logits, value_probs, mlh) — three outputs matching the
    TorchScript export. Policy temperature is applied at MCTS time, not here.

    Uses opset 18 for native Mish support (no custom operators required).
    Dynamic batch dim lets the TRT engine cover batch=1 (uci/root-expand)
    through batch=256 (self-play) from a single ONNX file.
    """
    original_device = next(model.parameters()).device
    was_training = model.training
    try:
        model.to(device)
        model.eval()
        wrapper = _ExportWrapper(model)
        wrapper.eval()

        example = torch.zeros(1, model.config.input_planes, 8, 8, device=device)
        with torch.no_grad():
            # dynamo=False uses the classic tracer-based exporter which honors
            # dynamic_axes cleanly. Dynamo bakes batch=1 into downstream Gemm shapes
            # even with dynamic_axes set, producing an engine that only runs at bs=1.
            torch.onnx.export(
                wrapper,
                example,
                output_path,
                input_names=['input'],
                output_names=['policy', 'value', 'mlh'],
                opset_version=opset_version,
                dynamic_axes={
                    'input':  {0: 'batch'},
                    'policy': {0: 'batch'},
                    'value':  {0: 'batch'},
                    'mlh':    {0: 'batch'},
                },
                do_constant_folding=True,
                dynamo=False,
            )
    finally:
        model.to(original_device)
        if was_training:
            model.train()


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
