"""Chess training dataset: loads .npz files into a PyTorch Dataset."""

import numpy as np
import torch
from torch.utils.data import Dataset


class ChessDataset(Dataset):
    """Dataset that loads chess training positions from .npz files.

    Each .npz file contains:
        planes:   float32[N, 112, 8, 8]  — encoded board position
        policies: float32[N, 1858]        — MCTS visit distribution target
        values:   float32[N, 3]           — WDL target (win, draw, loss)
    """

    def __init__(self, npz_paths: list[str]):
        planes_list = []
        policies_list = []
        values_list = []

        for path in npz_paths:
            data = np.load(path)
            planes_list.append(data['planes'])
            policies_list.append(data['policies'])
            values_list.append(data['values'])

        self.planes = torch.from_numpy(np.concatenate(planes_list, axis=0))
        self.policies = torch.from_numpy(np.concatenate(policies_list, axis=0))
        self.values = torch.from_numpy(np.concatenate(values_list, axis=0))

    def __len__(self) -> int:
        return len(self.planes)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.planes[idx], self.policies[idx], self.values[idx]
