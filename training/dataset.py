"""Chess training dataset: loads .npz files into a PyTorch Dataset.

Supports two on-disk formats:
  - Legacy dense: key `planes` with shape (N, 112, 8, 8) float32.
  - Packed v2:   keys `bitboards` (N, 104) uint64 + scalar metadata
                 (`stm`, `castling`, `rule50`, `fullmove`) + `format_version`.
The packed form stores only the 104 binary piece/history planes as bitboards
(32× smaller than dense), regenerating the 8 scalar feature planes at batch
time. Horizontal mirror augmentation (a↔h files) is applied lazily in either
format.
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

# Cached gather indices for vectorized policy mirroring. `_POLICY_MIRROR_SRC`
# holds the source positions whose mirror move exists in the encoding;
# `_POLICY_MIRROR_DST` holds where each one writes to in the mirrored array.
# A batched mirror is then `out[:, dst] = policies[:, src]` — eliminates the
# 1858-iter Python loop that dominated dataset construction time.
_POLICY_MIRROR_SRC = np.where(_POLICY_MIRROR >= 0)[0].astype(np.int64)
_POLICY_MIRROR_DST = _POLICY_MIRROR[_POLICY_MIRROR_SRC].astype(np.int64)


def mirror_policies_batched(policies: np.ndarray) -> np.ndarray:
    """Vectorized mirror of a (N, 1858) policy batch. Equivalent to
    `np.stack([mirror_policy(p) for p in policies])` but ~100× faster."""
    out = np.zeros_like(policies)
    out[:, _POLICY_MIRROR_DST] = policies[:, _POLICY_MIRROR_SRC]
    return out


def _build_byte_bitreverse_table() -> np.ndarray:
    """Per-byte bit-reverse lookup (reverses bit order within each byte)."""
    table = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        v = i
        v = ((v >> 1) & 0x55) | ((v & 0x55) << 1)
        v = ((v >> 2) & 0x33) | ((v & 0x33) << 2)
        v = ((v >> 4) & 0x0F) | ((v & 0x0F) << 4)
        table[i] = v & 0xFF
    return table


_BYTE_REVERSE = _build_byte_bitreverse_table()


def unpack_bitboards(bb: np.ndarray) -> np.ndarray:
    """Expand 104 uint64 bitboards into a dense (104, 8, 8) float32 tensor.

    LERF layout: bit `i` in a uint64 maps to square `i` (a1 = LSB, h8 = MSB),
    so byte `r` of each uint64 holds rank `r`, and within that byte the LSB
    is file 0 (a-file). np.unpackbits(..., bitorder='little') yields the
    bits in (file-0 .. file-7) order per byte — exactly [plane, rank, file].
    """
    assert bb.shape[-1] == 104, f"expected last dim 104, got {bb.shape}"
    bb = np.ascontiguousarray(bb, dtype=np.uint64)
    bytes_view = bb.view(np.uint8).reshape(*bb.shape[:-1], 104, 8)
    bits = np.unpackbits(bytes_view, axis=-1, bitorder='little')
    return bits.reshape(*bb.shape[:-1], 104, 8, 8).astype(np.float32)


def mirror_bitboards(bb: np.ndarray) -> np.ndarray:
    """Horizontally mirror (a↔h files) every plane in a packed bitboard block.

    In LERF each rank is stored as a single byte of a uint64. Mirroring files
    is a per-byte bit reverse, applied independently to all 8 bytes.
    """
    bb = np.ascontiguousarray(bb, dtype=np.uint64)
    bytes_view = bb.view(np.uint8).reshape(*bb.shape, 8).copy()
    mirrored_bytes = _BYTE_REVERSE[bytes_view]
    return mirrored_bytes.view(np.uint64).reshape(bb.shape).copy()


def mirror_castling(castling: np.ndarray) -> np.ndarray:
    """Swap kingside↔queenside bits within each color's nibble.

    Bits: 0=STM-K, 1=STM-Q, 2=OPP-K, 3=OPP-Q. A horizontal mirror converts
    kingside castling to queenside and vice-versa for both colors.
    """
    c = np.asarray(castling, dtype=np.uint8)
    # Swap pairs (bit0,bit1) and (bit2,bit3) in each byte.
    b0 = (c >> 0) & 1
    b1 = (c >> 1) & 1
    b2 = (c >> 2) & 1
    b3 = (c >> 3) & 1
    return ((b1 << 0) | (b0 << 1) | (b3 << 2) | (b2 << 3)).astype(np.uint8)


def mirror_planes(planes: np.ndarray) -> np.ndarray:
    """Flip dense board planes horizontally (a↔h files).

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


def _extract_metadata_from_dense(planes: np.ndarray) -> dict:
    """Recover packed scalar metadata (stm/castling/rule50/fullmove) from a
    dense (N, 112, 8, 8) plane tensor. Used when loading legacy .npz files
    alongside a packed window — lets the Dataset present one unified format.
    """
    stm = planes[:, 104, 0, 0].astype(bool)
    fullmove = np.rint(planes[:, 105, 0, 0] * 200.0).astype(np.uint16)
    c0 = planes[:, 106, 0, 0] > 0.5
    c1 = planes[:, 107, 0, 0] > 0.5
    c2 = planes[:, 108, 0, 0] > 0.5
    c3 = planes[:, 109, 0, 0] > 0.5
    castling = (c0.astype(np.uint8) << 0) \
             | (c1.astype(np.uint8) << 1) \
             | (c2.astype(np.uint8) << 2) \
             | (c3.astype(np.uint8) << 3)
    rule50 = np.rint(planes[:, 110, 0, 0] * 100.0).astype(np.uint8)
    return dict(stm=stm, castling=castling, rule50=rule50, fullmove=fullmove)


def _pack_dense_planes(planes: np.ndarray) -> np.ndarray:
    """Pack the 104 binary piece planes of a dense tensor into uint64 bitboards.

    Input: (N, 112, 8, 8) float32. Output: (N, 104) uint64 in LERF layout.
    """
    bits = (planes[:, :104] > 0.5).astype(np.uint8)  # (N, 104, 8, 8)
    bits = bits.reshape(bits.shape[0], 104, 64)
    # Pack along the 64-bit axis in little-endian order so bit i == square i.
    packed_bytes = np.packbits(bits, axis=-1, bitorder='little')  # (N, 104, 8)
    return np.ascontiguousarray(packed_bytes).view(np.uint64).reshape(planes.shape[0], 104)


class ChessDataset(Dataset):
    """Dataset that loads chess training positions from .npz files.

    Two on-disk layouts are accepted:
      - Packed v2: `bitboards` (N,104) uint64 + scalar metadata arrays + a
        marker key `format_version == 2`. Preferred.
      - Legacy dense: `planes` (N,112,8,8) float32. Auto-converted to packed
        form at load so the in-memory representation is always packed.

    Each sample is expanded to a (112, 8, 8) float32 tensor on __getitem__,
    combining the 104 unpacked piece planes with 8 scalar feature planes.

    With mirror=True, each position also produces a horizontally flipped
    copy (a↔h files), doubling the effective dataset size.
    """

    def __init__(
        self,
        npz_paths: list[str],
        mirror: bool = False,
        value_blend: dict | None = None,
        adjudicated_weight: float = 1.0,
    ):
        """
        Args:
            value_blend: when provided and v2 eval signals are present, blend
                `values` (game_result) with best_eval/played_eval/raw_nn_eval
                per row using these weights. Must sum to 1.0. When None, the
                on-disk `values` array is used as-is.
            adjudicated_weight: multiplier applied to sample weights for rows
                flagged `adjudicated=True`. Default 1.0 (no effect). Lc0's
                guidance is 0.5 — downweight ply-cap draws relative to real
                terminals.
        """
        bb_list: list[np.ndarray] = []
        stm_list: list[np.ndarray] = []
        castling_list: list[np.ndarray] = []
        rule50_list: list[np.ndarray] = []
        fullmove_list: list[np.ndarray] = []
        policies_list: list[np.ndarray] = []
        values_list: list[np.ndarray] = []
        mlh_list: list[np.ndarray] = []
        surprise_list: list[np.ndarray] = []
        use_policy_list: list[np.ndarray] = []
        best_eval_list: list[np.ndarray] = []
        played_eval_list: list[np.ndarray] = []
        raw_nn_eval_list: list[np.ndarray] = []
        adjudicated_list: list[np.ndarray] = []

        for path in npz_paths:
            data = np.load(path)
            files = set(data.files)
            if 'bitboards' in files:
                bb = np.asarray(data['bitboards'], dtype=np.uint64)
                stm = np.asarray(data['stm'], dtype=bool)
                castling = np.asarray(data['castling'], dtype=np.uint8)
                rule50 = np.asarray(data['rule50'], dtype=np.uint8)
                fullmove = np.asarray(data['fullmove'], dtype=np.uint16)
            else:
                # Legacy dense format: pack on load so the rest of the pipeline
                # only has to deal with one representation.
                planes = data['planes']
                bb = _pack_dense_planes(planes)
                meta = _extract_metadata_from_dense(planes)
                stm = meta['stm']
                castling = meta['castling']
                rule50 = meta['rule50']
                fullmove = meta['fullmove']

            bb_list.append(bb)
            stm_list.append(stm)
            castling_list.append(castling)
            rule50_list.append(rule50)
            fullmove_list.append(fullmove)
            policies_list.append(data['policies'])
            values_list.append(data['values'])
            if 'moves_left' in files:
                mlh_list.append(data['moves_left'])
            if 'surprise' in files:
                surprise_list.append(data['surprise'])
            if 'use_policy' in files:
                use_policy_list.append(data['use_policy'])
            if 'best_eval' in files:
                best_eval_list.append(np.asarray(data['best_eval'], dtype=np.float32))
            if 'played_eval' in files:
                played_eval_list.append(np.asarray(data['played_eval'], dtype=np.float32))
            if 'raw_nn_eval' in files:
                raw_nn_eval_list.append(np.asarray(data['raw_nn_eval'], dtype=np.float32))
            if 'adjudicated' in files:
                adjudicated_list.append(np.asarray(data['adjudicated'], dtype=np.bool_))

        bitboards = np.concatenate(bb_list, axis=0)
        stm = np.concatenate(stm_list, axis=0)
        castling = np.concatenate(castling_list, axis=0)
        rule50 = np.concatenate(rule50_list, axis=0)
        fullmove = np.concatenate(fullmove_list, axis=0)
        policies = np.concatenate(policies_list, axis=0)
        values = np.concatenate(values_list, axis=0)
        n_files = len(bb_list)
        has_mlh = len(mlh_list) == n_files
        moves_left = np.concatenate(mlh_list, axis=0) if has_mlh else None
        has_surprise = len(surprise_list) == n_files
        surprise = np.concatenate(surprise_list, axis=0) if has_surprise else None
        has_use_policy = len(use_policy_list) == n_files
        use_policy = np.concatenate(use_policy_list, axis=0) if has_use_policy else None
        has_v2_eval = (
            len(best_eval_list) == n_files
            and len(played_eval_list) == n_files
            and len(raw_nn_eval_list) == n_files
        )
        best_eval = np.concatenate(best_eval_list, axis=0) if has_v2_eval else None
        played_eval = np.concatenate(played_eval_list, axis=0) if has_v2_eval else None
        raw_nn_eval = np.concatenate(raw_nn_eval_list, axis=0) if has_v2_eval else None
        has_adjudicated = len(adjudicated_list) == n_files
        adjudicated = np.concatenate(adjudicated_list, axis=0) if has_adjudicated else None

        # Apply the value_blend at load time if the shard carries the v2
        # decomposed signals. Otherwise `values` stays as whatever the shard
        # wrote (legacy v1 q-blend or raw z_wdl).
        if value_blend is not None and has_v2_eval:
            from training.train import blend_value_target
            values = blend_value_target(
                value_blend, values, best_eval, played_eval, raw_nn_eval,
            ).astype(np.float32)

        if mirror:
            m_bb = mirror_bitboards(bitboards)
            m_castling = mirror_castling(castling)
            m_policies = mirror_policies_batched(policies)

            bitboards = np.concatenate([bitboards, m_bb], axis=0)
            stm = np.concatenate([stm, stm], axis=0)
            castling = np.concatenate([castling, m_castling], axis=0)
            rule50 = np.concatenate([rule50, rule50], axis=0)
            fullmove = np.concatenate([fullmove, fullmove], axis=0)
            policies = np.concatenate([policies, m_policies], axis=0)
            values = np.concatenate([values, values], axis=0)  # WDL unchanged by mirror
            if moves_left is not None:
                moves_left = np.concatenate([moves_left, moves_left], axis=0)
            if surprise is not None:
                surprise = np.concatenate([surprise, surprise], axis=0)
            if use_policy is not None:
                use_policy = np.concatenate([use_policy, use_policy], axis=0)
            if best_eval is not None:
                best_eval = np.concatenate([best_eval, best_eval], axis=0)
            if played_eval is not None:
                played_eval = np.concatenate([played_eval, played_eval], axis=0)
            if raw_nn_eval is not None:
                raw_nn_eval = np.concatenate([raw_nn_eval, raw_nn_eval], axis=0)
            if adjudicated is not None:
                adjudicated = np.concatenate([adjudicated, adjudicated], axis=0)

        # Store packed, per-position metadata as numpy arrays (cheap); expand to
        # full (112,8,8) tensors lazily in __getitem__ to keep RAM low.
        self.bitboards = np.ascontiguousarray(bitboards, dtype=np.uint64)
        self.stm = np.ascontiguousarray(stm, dtype=bool)
        self.castling = np.ascontiguousarray(castling, dtype=np.uint8)
        self.rule50 = np.ascontiguousarray(rule50, dtype=np.uint8)
        self.fullmove = np.ascontiguousarray(fullmove, dtype=np.uint16)
        self.policies = torch.from_numpy(np.ascontiguousarray(policies))
        self.values = torch.from_numpy(np.ascontiguousarray(values))
        self.moves_left = torch.from_numpy(np.ascontiguousarray(moves_left)) if moves_left is not None else None
        self.surprise_weights = torch.from_numpy(np.ascontiguousarray(surprise)).float() if surprise is not None else None
        self.use_policy = torch.from_numpy(np.ascontiguousarray(use_policy)) if use_policy is not None else None
        self.adjudicated = adjudicated

        # Adjudicated-row downweighting: fold it into surprise_weights so the
        # existing WeightedRandomSampler picks up the effect for free. If the
        # shard has no surprise array, synthesize a uniform one first so the
        # multiplier has something to scale.
        if adjudicated is not None and adjudicated_weight != 1.0:
            if self.surprise_weights is None:
                self.surprise_weights = torch.ones(len(adjudicated), dtype=torch.float32)
            factor = torch.tensor(
                np.where(adjudicated, adjudicated_weight, 1.0),
                dtype=torch.float32,
            )
            self.surprise_weights = self.surprise_weights * factor

    def __len__(self) -> int:
        return len(self.bitboards)

    def _build_planes(self, idx: int) -> torch.Tensor:
        """Expand a single sample's packed bitboards + scalars into (112, 8, 8)."""
        piece_planes = unpack_bitboards(self.bitboards[idx])  # (104, 8, 8)
        planes = np.zeros((112, 8, 8), dtype=np.float32)
        planes[:104] = piece_planes
        if self.stm[idx]:
            planes[104] = 1.0
        planes[105] = float(self.fullmove[idx]) / 200.0
        c = int(self.castling[idx])
        if c & 0x1: planes[106] = 1.0
        if c & 0x2: planes[107] = 1.0
        if c & 0x4: planes[108] = 1.0
        if c & 0x8: planes[109] = 1.0
        planes[110] = float(self.rule50[idx]) / 100.0
        planes[111] = 1.0
        return torch.from_numpy(planes)

    def __getitem__(self, idx: int) -> tuple:
        planes = self._build_planes(idx)
        items = (planes, self.policies[idx], self.values[idx])
        if self.moves_left is not None:
            items = items + (self.moves_left[idx],)
        if self.use_policy is not None:
            items = items + (self.use_policy[idx],)
        return items
