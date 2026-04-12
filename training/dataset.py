"""Chess training dataset: loads .npz files into a PyTorch Dataset.

Supports horizontal mirror augmentation (flip a↔h files) for 2× training data.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from training.encoder import POLICY_SIZE, index_to_move, move_to_index, file_of, rank_of, make_square


def _build_policy_mirror_table() -> np.ndarray:
    """Build lookup table mapping each policy index to its mirror (file-flipped) index.

    Returns:
        int32 array of shape (1858,) where table[i] = mirrored index of policy i.
        Entries are -1 if the mirrored move doesn't exist in the encoding.
    """
    table = np.full(POLICY_SIZE, -1, dtype=np.int32)
    for idx in range(POLICY_SIZE):
        from_sq, to_sq, promo = index_to_move(idx)
        # Flip files: file -> 7 - file, rank stays the same
        from_f, from_r = file_of(from_sq), rank_of(from_sq)
        to_f, to_r = file_of(to_sq), rank_of(to_sq)
        mirror_from = make_square(7 - from_f, from_r)
        mirror_to = make_square(7 - to_f, to_r)
        mirror_idx = move_to_index(mirror_from, mirror_to, promo)
        if mirror_idx is not None:
            table[idx] = mirror_idx
    return table


_POLICY_MIRROR = _build_policy_mirror_table()


def mirror_planes(planes: np.ndarray) -> np.ndarray:
    """Flip board planes horizontally (a↔h files).

    Args:
        planes: float32 array of shape (112, 8, 8).

    Returns:
        New array with files flipped (axis 2 reversed).
    """
    return np.flip(planes, axis=2).copy()


def mirror_policy(policy: np.ndarray) -> np.ndarray:
    """Mirror a policy vector to match a file-flipped board.

    Args:
        policy: float32 array of shape (1858,).

    Returns:
        New policy with move indices remapped for the mirrored board.
    """
    mirrored = np.zeros_like(policy)
    for idx in range(POLICY_SIZE):
        mirror_idx = _POLICY_MIRROR[idx]
        if mirror_idx >= 0:
            mirrored[mirror_idx] = policy[idx]
    return mirrored


class ChessDataset(Dataset):
    """Dataset that loads chess training positions from .npz files.

    Each .npz file contains:
        planes:      float32[N, 112, 8, 8]  — encoded board position
        policies:    float32[N, 1858]        — MCTS visit distribution target
        values:      float32[N, 3]           — WDL target (win, draw, loss)
        moves_left:  float32[N]              — remaining moves target (optional)

    With mirror=True, each position also produces a horizontally flipped
    copy (a↔h files), doubling the effective dataset size.
    """

    def __init__(self, npz_paths: list[str], mirror: bool = False):
        planes_list = []
        policies_list = []
        values_list = []
        mlh_list = []
        surprise_list = []
        use_policy_list = []

        for path in npz_paths:
            data = np.load(path)
            planes_list.append(data['planes'])
            policies_list.append(data['policies'])
            values_list.append(data['values'])
            if 'moves_left' in data:
                mlh_list.append(data['moves_left'])
            if 'surprise' in data:
                surprise_list.append(data['surprise'])
            if 'use_policy' in data:
                use_policy_list.append(data['use_policy'])

        planes = np.concatenate(planes_list, axis=0)
        policies = np.concatenate(policies_list, axis=0)
        values = np.concatenate(values_list, axis=0)
        has_mlh = len(mlh_list) == len(planes_list)
        moves_left = np.concatenate(mlh_list, axis=0) if has_mlh else None
        has_surprise = len(surprise_list) == len(planes_list)
        surprise = np.concatenate(surprise_list, axis=0) if has_surprise else None
        has_use_policy = len(use_policy_list) == len(planes_list)
        use_policy = np.concatenate(use_policy_list, axis=0) if has_use_policy else None

        if mirror:
            mirror_planes_arr = np.stack([mirror_planes(p) for p in planes])
            mirror_policies_arr = np.stack([mirror_policy(p) for p in policies])
            planes = np.concatenate([planes, mirror_planes_arr], axis=0)
            policies = np.concatenate([policies, mirror_policies_arr], axis=0)
            values = np.concatenate([values, values], axis=0)  # WDL unchanged by mirror
            if moves_left is not None:
                moves_left = np.concatenate([moves_left, moves_left], axis=0)
            if surprise is not None:
                surprise = np.concatenate([surprise, surprise], axis=0)
            if use_policy is not None:
                use_policy = np.concatenate([use_policy, use_policy], axis=0)

        self.planes = torch.from_numpy(planes)
        self.policies = torch.from_numpy(policies)
        self.values = torch.from_numpy(values)
        self.moves_left = torch.from_numpy(moves_left) if moves_left is not None else None
        self.surprise_weights = torch.from_numpy(surprise).float() if surprise is not None else None
        self.use_policy = torch.from_numpy(use_policy) if use_policy is not None else None

    def __len__(self) -> int:
        return len(self.planes)

    def __getitem__(self, idx: int) -> tuple:
        items = (self.planes[idx], self.policies[idx], self.values[idx])
        if self.moves_left is not None:
            items = items + (self.moves_left[idx],)
        if self.use_policy is not None:
            items = items + (self.use_policy[idx],)
        return items
