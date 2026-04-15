from dataclasses import dataclass


@dataclass
class NetworkConfig:
    num_blocks: int = 10
    num_filters: int = 128
    se_ratio: int = 4
    input_planes: int = 112
    policy_size: int = 1858
    value_size: int = 3  # WDL: win, draw, loss
    policy_conv_filters: int = 80
    value_conv_filters: int = 32
    value_fc_size: int = 128
    # Moves-left head
    mlh_conv_filters: int = 8
    mlh_fc_size: int = 128
    # Attention policy head
    use_attention_policy: bool = True
    policy_embedding_size: int = 64
    policy_d_model: int = 64


@dataclass
class SelfPlayConfig:
    """Self-play generation config.

    Historical fields (pre Lc0-parity refactor): temperature_moves, max_moves,
    resign_threshold, consecutive_resign, q_ratio, playout_cap_randomization,
    playout_cap_fraction, playout_cap_quick_sims, kld_adaptive, kld_min_sims,
    kld_max_sims, kld_threshold, syzygy_path, random_opening_fraction,
    random_opening_moves, opening_book_path, opening_book_fraction.

    Stage 2 Lc0-parity fields (consumed by GameLoopManager):
    num_games, full_sims, quick_sims, min_sims, playout_cap_p,
    opening_temp, opening_temp_plies, temp_floor, temp_decay_plies,
    use_kld_adaptive, max_ply.
    """
    # --- Legacy fields (still used by training/selfplay.py and tests) ---
    temperature_moves: int = 30
    max_moves: int = 512
    resign_threshold: float = -0.95
    consecutive_resign: int = 5
    # Lc0-style Q-value blending into the value target: 0 = pure game result,
    # 1 = pure search-Q. A moderate blend (~0.25) de-noises terminal-only
    # training signal, especially in long games where the final result is a
    # weak label for most positions.
    q_ratio: float = 0.25
    playout_cap_randomization: bool = True   # Alternate full/quick search
    playout_cap_fraction: float = 0.25       # Fraction of moves that get full search
    playout_cap_quick_sims: int = 100        # Simulations for quick search moves
    kld_adaptive: bool = True               # Adapt visit count based on KL divergence
    kld_min_sims: int = 100                 # Minimum simulations (used when KLD is low)
    kld_max_sims: int = 800                 # Maximum simulations (used when KLD is high)
    kld_threshold: float = 0.5              # KLD above this gets max sims
    syzygy_path: str | None = None          # Path to Syzygy tablebase files (None = disabled)
    random_opening_fraction: float = 0.05   # Fraction of games with random openings
    random_opening_moves: int = 8           # Max random moves for opening randomization
    opening_book_path: str | None = None    # File of starting FENs, one per line (None = disabled)
    opening_book_fraction: float = 0.5      # Fraction of games seeded from book FENs

    # --- Stage 2 (Lc0-parity) fields, consumed by GameLoopManager ---
    num_games: int = 1
    full_sims: int = 400
    quick_sims: int = 100
    min_sims: int = 80
    playout_cap_p: float = 0.25              # P(quick search) per move in GameLoopManager
    opening_temp: float = 1.0
    opening_temp_plies: int = 30
    temp_floor: float = 0.4
    temp_decay_plies: int = 30
    use_kld_adaptive: bool = True
    max_ply: int = 450

    # --- Stage 4 (Lc0-parity): WDL-aware resign + ply-cap adjudication ---
    # All three thresholds expressed in side-to-move POV. Lc0 defaults.
    resign_w: float = 0.02            # resign if own W drops below
    resign_d: float = 0.98            # resign if draw-prob exceeds (both sides agree on draw)
    resign_l: float = 0.98            # resign if own L exceeds
    resign_earliest_ply: int = 30     # no resign before this ply (avoids openings)
    adjudicated_weight: float = 0.5   # trainer down-weights adjudicated-draw rows

    # --- Stage 5 (Lc0-parity): min-visit floor on temperature-sampled moves ---
    # None = auto-scale as max(5, 1% of target_sims). Otherwise an absolute
    # visit count. Moves with fewer visits get rejected and resampled (up to 3
    # retries) before falling back to argmax.
    min_visits_floor: int | None = None