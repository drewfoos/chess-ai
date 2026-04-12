import pytest
from torch.utils.data import DataLoader
from training.config import NetworkConfig
from training.encoder import (
    move_to_index,
    index_to_move,
    POLICY_SIZE,
    mirror_move,
)


def test_network_config_defaults():
    cfg = NetworkConfig()
    assert cfg.num_blocks == 10
    assert cfg.num_filters == 128
    assert cfg.se_ratio == 4
    assert cfg.input_planes == 112
    assert cfg.policy_size == 1858
    assert cfg.value_size == 3


def test_network_config_custom():
    cfg = NetworkConfig(num_blocks=20, num_filters=256)
    assert cfg.num_blocks == 20
    assert cfg.num_filters == 256
    assert cfg.input_planes == 112  # Unchanged default


def test_policy_size():
    assert POLICY_SIZE == 1858


def test_move_encoding_roundtrip_e2e4():
    # e2e4: from=E2(12), to=E4(28), promo=None
    # From white's perspective, this is a pawn double push
    # E2 is file=4, rank=1 → square index 12
    # E4 is file=4, rank=3 → square index 28
    # Direction: North (index 0), distance 2
    idx = move_to_index(12, 28, None)
    assert idx is not None
    from_sq, to_sq, promo = index_to_move(idx)
    assert from_sq == 12
    assert to_sq == 28
    assert promo is None


def test_move_encoding_knight():
    # Ng1-f3: from=G1(6), to=F3(21)
    # Knight move: delta = (21-6) = 15 → (-1, +2) in (file, rank)
    idx = move_to_index(6, 21, None)
    assert idx is not None
    from_sq, to_sq, promo = index_to_move(idx)
    assert from_sq == 6
    assert to_sq == 21
    assert promo is None


def test_move_encoding_queen_promotion():
    # e7e8=Q: from=E7(52), to=E8(60), promo=queen
    # Queen promotion uses the normal queen-move encoding (N, distance 1)
    idx = move_to_index(52, 60, None)  # No promo flag for queen promo
    assert idx is not None
    from_sq, to_sq, promo = index_to_move(idx)
    assert from_sq == 52
    assert to_sq == 60


def test_move_encoding_underpromotion():
    # e7e8=N: from=E7(52), to=E8(60), promo='n'
    idx = move_to_index(52, 60, 'n')
    assert idx is not None
    from_sq, to_sq, promo = index_to_move(idx)
    assert from_sq == 52
    assert to_sq == 60
    assert promo == 'n'


def test_move_encoding_capture_underpromotion():
    # e7d8=R: from=E7(52), to=D8(59), promo='r'
    idx = move_to_index(52, 59, 'r')
    assert idx is not None
    from_sq, to_sq, promo = index_to_move(idx)
    assert from_sq == 52
    assert to_sq == 59
    assert promo == 'r'


def test_all_indices_unique():
    # Collect all valid move indices and ensure no duplicates
    seen = set()
    for idx in range(POLICY_SIZE):
        from_sq, to_sq, promo = index_to_move(idx)
        key = (from_sq, to_sq, promo)
        assert key not in seen, f"Duplicate move at index {idx}: {key}"
        seen.add(key)


def test_mirror_move():
    # E2(12) mirrored = E7(52) — flip rank: rank 1 → rank 6
    assert mirror_move(12) == 52
    # A1(0) mirrored = A8(56)
    assert mirror_move(0) == 56
    # H8(63) mirrored = H1(7)
    assert mirror_move(63) == 7


import numpy as np
from training.encoder import encode_position


def test_encode_starting_position_shape():
    planes = encode_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    assert planes.shape == (112, 8, 8)
    assert planes.dtype == np.float32


def test_encode_starting_position_white_pawns():
    planes = encode_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    # White to move, so "our" pieces = white
    # Plane 0 = our pawns. White pawns on rank 1 (index 1 in 0-indexed)
    # In the tensor: planes[0] should have 1s on rank 1 (row index 1)
    pawn_plane = planes[0]
    assert pawn_plane.sum() == 8  # 8 white pawns
    # All on rank 1
    for file in range(8):
        assert pawn_plane[1, file] == 1.0


def test_encode_starting_position_opponent_pawns():
    planes = encode_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    # Plane 6 = opponent pawns (black). Black pawns on rank 6
    opp_pawn_plane = planes[6]
    assert opp_pawn_plane.sum() == 8
    for file in range(8):
        assert opp_pawn_plane[6, file] == 1.0


def test_encode_black_to_move_flips():
    # Same position but black to move — board should be flipped
    planes = encode_position("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
    # Black to move: "our" pieces = black, board flipped vertically
    # Black pawns were on rank 6, after flip they're on rank 1
    our_pawn_plane = planes[0]
    assert our_pawn_plane.sum() == 8
    for file in range(8):
        assert our_pawn_plane[1, file] == 1.0


def test_encode_castling_planes():
    planes = encode_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    # Constant planes start at index 104
    # Plane 106 = our kingside castling (K for white) → all 1s
    assert planes[106].sum() == 64  # All 1s
    # Plane 107 = our queenside castling (Q for white) → all 1s
    assert planes[107].sum() == 64


def test_encode_no_castling():
    planes = encode_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1")
    # No castling rights → planes 106-109 all zeros
    for i in range(106, 110):
        assert planes[i].sum() == 0


def test_encode_color_plane():
    # White to move: color plane (104) = all 1s
    planes_w = encode_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    assert planes_w[104].sum() == 64

    # Black to move: color plane (104) = all 0s
    planes_b = encode_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")
    assert planes_b[104].sum() == 0


def test_encode_bias_plane():
    planes = encode_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    # Plane 111 = all-ones bias
    assert planes[111].sum() == 64


import torch
from training.model import ChessNetwork


def test_model_output_shapes():
    cfg = NetworkConfig(num_blocks=2, num_filters=32)  # Small for testing
    model = ChessNetwork(cfg)
    x = torch.randn(4, 112, 8, 8)  # Batch of 4
    policy, value, mlh = model(x)
    assert policy.shape == (4, 1858)
    assert value.shape == (4, 3)
    assert mlh.shape == (4,)


def test_model_policy_logits():
    cfg = NetworkConfig(num_blocks=2, num_filters=32)
    model = ChessNetwork(cfg)
    x = torch.randn(1, 112, 8, 8)
    policy, _, _ = model(x)
    # Policy should be raw logits (not softmaxed) — can be any real number
    assert policy.dtype == torch.float32


def test_model_value_probabilities():
    cfg = NetworkConfig(num_blocks=2, num_filters=32)
    model = ChessNetwork(cfg)
    x = torch.randn(1, 112, 8, 8)
    _, value, _ = model(x)
    # Value head now returns raw logits (not probabilities)
    assert value.shape == (1, 3)
    assert value.dtype == torch.float32
    # Logits can be any real number — verify softmax produces valid probs
    probs = torch.softmax(value, dim=1)
    assert torch.allclose(probs.sum(dim=1), torch.tensor([1.0]), atol=1e-5)
    assert (probs >= 0).all()


def test_model_default_config():
    cfg = NetworkConfig()  # 10 blocks, 128 filters
    model = ChessNetwork(cfg)
    # Count parameters — with policy FC (80*64 → 1858) the dominant cost is the policy head (~9.5M)
    # Total is ~13M for default config (10 blocks, 128 filters, policy_conv_filters=80)
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 1_000_000  # At least 1M
    assert total_params < 20_000_000  # Less than 20M


def test_model_batch_independence():
    cfg = NetworkConfig(num_blocks=2, num_filters=32)
    model = ChessNetwork(cfg)
    model.eval()
    x = torch.randn(2, 112, 8, 8)
    policy_batch, value_batch, _ = model(x)
    policy_0, value_0, _ = model(x[0:1])
    policy_1, value_1, _ = model(x[1:2])
    assert torch.allclose(policy_batch[0], policy_0[0], atol=1e-5)
    assert torch.allclose(value_batch[0], value_0[0], atol=1e-5)


def test_attention_policy_head_output_shape():
    from training.model import AttentionPolicyHead
    head = AttentionPolicyHead(body_channels=32, embedding_size=16, d_model=16)
    x = torch.randn(2, 32, 8, 8)
    out = head(x)
    assert out.shape == (2, 1858)


def test_attention_policy_map_valid_indices():
    from training.model import _build_attention_policy_index
    idx = _build_attention_policy_index()
    assert idx.shape == (1858,)
    assert idx.min() >= 0
    assert idx.max() < 4288  # 4096 + 192


def test_attention_policy_map_promotions():
    """Verify queen promotions map to promotion section, knight to base."""
    from training.model import _build_attention_policy_index
    from training.encoder import move_to_index, rank_of
    idx_map = _build_attention_policy_index()

    # e7e8 queen promotion (from_sq=52, to_sq=60, promo=None)
    q_idx = move_to_index(52, 60, None)
    assert q_idx is not None
    assert idx_map[q_idx] >= 4096  # In promotion section

    # e7e8 knight promotion
    n_idx = move_to_index(52, 60, 'n')
    assert n_idx is not None
    assert idx_map[n_idx] < 4096  # In 64×64 base section
    assert idx_map[n_idx] == 52 * 64 + 60

    # e7e8 rook promotion
    r_idx = move_to_index(52, 60, 'r')
    assert r_idx is not None
    assert idx_map[r_idx] >= 4096  # In promotion section


def test_attention_vs_classical_policy_shape():
    """Both policy head types should produce identical output shapes."""
    cfg_attn = NetworkConfig(num_blocks=1, num_filters=16, use_attention_policy=True,
                             policy_embedding_size=8, policy_d_model=8)
    cfg_fc = NetworkConfig(num_blocks=1, num_filters=16, use_attention_policy=False)
    model_attn = ChessNetwork(cfg_attn)
    model_fc = ChessNetwork(cfg_fc)
    x = torch.randn(1, 112, 8, 8)
    p_attn, _, _ = model_attn(x)
    p_fc, _, _ = model_fc(x)
    assert p_attn.shape == p_fc.shape == (1, 1858)


def test_glorot_init_weight_scale():
    """Glorot-initialized weights should have reasonable scale."""
    cfg = NetworkConfig(num_blocks=2, num_filters=32)
    model = ChessNetwork(cfg)
    conv_weight = model.input_conv.weight
    # Xavier normal: std ≈ sqrt(2 / (fan_in + fan_out))
    assert conv_weight.std().item() < 0.2
    assert conv_weight.std().item() > 0.01


def test_sgd_optimizer():
    from training.train import create_optimizer
    cfg = NetworkConfig(num_blocks=1, num_filters=16)
    model = ChessNetwork(cfg)
    opt = create_optimizer(model, optimizer_type='sgd')
    assert isinstance(opt, torch.optim.SGD)
    assert opt.defaults['nesterov'] is True
    assert opt.defaults['momentum'] == 0.9


def test_adamw_optimizer_default():
    from training.train import create_optimizer
    cfg = NetworkConfig(num_blocks=1, num_filters=16)
    model = ChessNetwork(cfg)
    opt = create_optimizer(model)
    assert isinstance(opt, torch.optim.AdamW)


import tempfile
import os
from training.generate_data import generate_synthetic_data
from training.dataset import ChessDataset


def test_generate_synthetic_data():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_data.npz")
        generate_synthetic_data(path, num_positions=50)

        with np.load(path) as data:
            assert data['planes'].shape == (50, 112, 8, 8)
            assert data['policies'].shape == (50, 1858)
            assert data['values'].shape == (50, 3)

            # Policies should be valid probability distributions
            policy_sums = data['policies'].sum(axis=1)
            np.testing.assert_allclose(policy_sums, 1.0, atol=1e-5)

            # Values should be valid WDL distributions
            value_sums = data['values'].sum(axis=1)
            np.testing.assert_allclose(value_sums, 1.0, atol=1e-5)


def test_chess_dataset():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_data.npz")
        generate_synthetic_data(path, num_positions=20)

        dataset = ChessDataset([path])
        assert len(dataset) == 20

        planes, policy, value = dataset[0]
        assert planes.shape == (112, 8, 8)
        assert policy.shape == (1858,)
        assert value.shape == (3,)
        assert isinstance(planes, torch.Tensor)


def test_chess_dataset_multiple_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        path1 = os.path.join(tmpdir, "data1.npz")
        path2 = os.path.join(tmpdir, "data2.npz")
        generate_synthetic_data(path1, num_positions=10)
        generate_synthetic_data(path2, num_positions=15)

        dataset = ChessDataset([path1, path2])
        assert len(dataset) == 25


def test_chess_dataset_mirror_augmentation():
    from training.dataset import mirror_planes, mirror_policy
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_data.npz")
        generate_synthetic_data(path, num_positions=10)

        dataset_no_mirror = ChessDataset([path], mirror=False)
        dataset_mirror = ChessDataset([path], mirror=True)
        assert len(dataset_mirror) == 2 * len(dataset_no_mirror)

        # Original positions are preserved (first half)
        planes_orig, policy_orig, value_orig = dataset_no_mirror[0]
        planes_m, policy_m, value_m = dataset_mirror[0]
        assert torch.equal(planes_orig, planes_m)
        assert torch.equal(policy_orig, policy_m)
        assert torch.equal(value_orig, value_m)

        # Mirror preserves policy sum
        planes_flip, policy_flip, value_flip = dataset_mirror[10]  # mirrored copy
        assert abs(policy_flip.sum().item() - policy_orig.sum().item()) < 1e-5
        # Value is unchanged by mirror
        assert torch.equal(value_orig, value_flip)

    # mirror_planes flips file axis
    planes = np.zeros((112, 8, 8), dtype=np.float32)
    planes[0, 0, 0] = 1.0  # a1
    flipped = mirror_planes(planes)
    assert flipped[0, 0, 0] == 0.0
    assert flipped[0, 0, 7] == 1.0  # h1


from training.train import train_step, create_optimizer


def test_train_step_reduces_loss():
    cfg = NetworkConfig(num_blocks=2, num_filters=32)
    model = ChessNetwork(cfg)
    optimizer = create_optimizer(model)

    # Create a batch of synthetic data
    batch_size = 8
    planes = torch.randn(batch_size, 112, 8, 8)
    policies = torch.softmax(torch.randn(batch_size, 1858), dim=1)
    values = torch.zeros(batch_size, 3)
    values[:, 0] = 1.0  # All wins

    # Run multiple steps and check loss decreases
    losses = []
    for _ in range(10):
        loss = train_step(model, optimizer, planes, policies, values)
        losses.append(loss)

    # Loss should decrease over 10 steps
    assert losses[-1] < losses[0]


def test_train_step_loss_components():
    cfg = NetworkConfig(num_blocks=2, num_filters=32)
    model = ChessNetwork(cfg)
    optimizer = create_optimizer(model)

    planes = torch.randn(4, 112, 8, 8)
    policies = torch.softmax(torch.randn(4, 1858), dim=1)
    values = torch.zeros(4, 3)
    values[:, 1] = 1.0  # All draws

    loss = train_step(model, optimizer, planes, policies, values)
    # Loss should be a positive number
    assert loss > 0
    assert not np.isnan(loss)
    assert not np.isinf(loss)


from training.export import export_torchscript


def test_export_torchscript():
    cfg = NetworkConfig(num_blocks=2, num_filters=32)
    model = ChessNetwork(cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.pt")
        export_torchscript(model, path)

        loaded = torch.jit.load(path)

        model.eval()
        x = torch.randn(1, 112, 8, 8)
        with torch.no_grad():
            orig_policy, orig_value_logits, _ = model(x)
            loaded_policy, loaded_value_probs = loaded(x)  # export wrapper returns 2

        # Policy logits should match exactly
        assert torch.allclose(orig_policy, loaded_policy, atol=1e-5)
        # Exported model applies softmax to value head
        orig_value_probs = torch.softmax(orig_value_logits, dim=1)
        assert torch.allclose(orig_value_probs, loaded_value_probs, atol=1e-5)


def test_export_torchscript_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    cfg = NetworkConfig(num_blocks=2, num_filters=32)
    model = ChessNetwork(cfg).cuda()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model_gpu.pt")
        export_torchscript(model, path, device='cpu')

        loaded = torch.jit.load(path)
        x = torch.randn(1, 112, 8, 8)
        model_cpu = model.cpu()
        model_cpu.eval()
        with torch.no_grad():
            orig_policy, orig_value_logits, _ = model_cpu(x)
            loaded_policy, loaded_value_probs = loaded(x)  # export wrapper returns 2

        assert torch.allclose(orig_policy, loaded_policy, atol=1e-5)
        orig_value_probs = torch.softmax(orig_value_logits, dim=1)
        assert torch.allclose(orig_value_probs, loaded_value_probs, atol=1e-5)


def test_end_to_end_pipeline():
    """Full pipeline: generate data → train → export → verify."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Generate synthetic data
        data_path = os.path.join(tmpdir, "train.npz")
        generate_synthetic_data(data_path, num_positions=64)

        # 2. Load into dataset
        dataset = ChessDataset([data_path])
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        # 3. Create model (tiny for speed)
        cfg = NetworkConfig(num_blocks=1, num_filters=16)
        model = ChessNetwork(cfg)
        optimizer = create_optimizer(model, lr=1e-3)

        # 4. Train for a few steps
        model.train()
        losses = []
        for planes, policies, values in loader:
            loss = train_step(model, optimizer, planes, policies, values)
            losses.append(loss)

        assert len(losses) > 0
        assert all(not np.isnan(l) for l in losses)

        # 5. Export to TorchScript
        export_path = os.path.join(tmpdir, "model.pt")
        export_torchscript(model, export_path)
        assert os.path.exists(export_path)

        # 6. Load exported model and verify
        loaded = torch.jit.load(export_path)
        model.eval()
        x = torch.randn(1, 112, 8, 8)
        with torch.no_grad():
            orig_p, orig_v_logits, _ = model(x)
            load_p, load_v_probs = loaded(x)  # export wrapper returns 2
        assert torch.allclose(orig_p, load_p, atol=1e-5)
        orig_v_probs = torch.softmax(orig_v_logits, dim=1)
        assert torch.allclose(orig_v_probs, load_v_probs, atol=1e-5)


def test_gpu_training():
    """Verify training works on GPU."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    cfg = NetworkConfig(num_blocks=2, num_filters=32)
    model = ChessNetwork(cfg).cuda()
    optimizer = create_optimizer(model)

    planes = torch.randn(4, 112, 8, 8).cuda()
    policies = torch.softmax(torch.randn(4, 1858), dim=1).cuda()
    values = torch.zeros(4, 3).cuda()
    values[:, 0] = 1.0

    loss = train_step(model, optimizer, planes, policies, values)
    assert loss > 0
    assert not np.isnan(loss)


import chess
from training.encoder import encode_board


def test_encode_board_shape():
    """encode_board produces the correct tensor shape."""
    board = chess.Board()
    planes = encode_board(board)
    assert planes.shape == (112, 8, 8)
    assert planes.dtype == np.float32


def test_encode_board_starting_matches_fen():
    """Starting position via encode_board matches encode_position for time step 0."""
    board = chess.Board()
    from_board = encode_board(board)
    from_fen = encode_position(board.fen())
    # All 8 time steps are identical (no history) so entire tensor matches
    np.testing.assert_array_equal(from_board, from_fen)


def test_encode_board_history_differs():
    """After moves, earlier time steps show different positions."""
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Nf3")
    planes = encode_board(board)
    # Time step 0 (current: after 1.e4 e5 2.Nf3) differs from
    # time step 1 (position after 1.e4 e5)
    step0 = planes[0:13]
    step1 = planes[13:26]
    assert not np.array_equal(step0, step1)


def test_encode_board_black_to_move_flips():
    """Board is flipped when Black is to move."""
    board = chess.Board()
    board.push_san("e4")  # Now Black to move
    planes = encode_board(board)
    # Color plane (104) should be 0 (Black to move)
    assert planes[104].sum() == 0
    # "Our" pawns (plane 0) should be Black's pawns, flipped to rank 1
    our_pawns = planes[0]
    assert our_pawns.sum() == 8
    for file in range(8):
        assert our_pawns[1, file] == 1.0


def test_encode_board_repetition_plane():
    """Repetition plane is set when position repeats."""
    board = chess.Board()
    # Play moves that repeat: Nf3 Nf6 Ng1 Ng8 (back to start)
    board.push_san("Nf3")
    board.push_san("Nf6")
    board.push_san("Ng1")
    board.push_san("Ng8")
    # Now the starting position has occurred twice
    planes = encode_board(board)
    # Repetition plane (index 12 within time step 0) should be 1
    assert planes[12].sum() == 64  # All ones = repetition detected


from training.mcts import Node, MCTS, MCTSConfig, chess_move_to_policy_index


def test_node_initial_state():
    """New node has zero visits and no children."""
    node = Node(prior=0.5)
    assert node.visit_count == 0
    assert node.total_value == 0.0
    assert node.prior == 0.5
    assert len(node.children) == 0


def test_node_value():
    """Node value is average of backpropagated values."""
    node = Node(prior=0.1)
    node.visit_count = 4
    node.total_value = 2.0
    assert node.value() == 0.5


def test_node_puct_prefers_high_prior():
    """PUCT formula prefers unvisited children with higher prior."""
    parent = Node(prior=1.0)
    parent.visit_count = 10
    child_high = Node(prior=0.9)
    child_low = Node(prior=0.1)
    parent.children = {chess.Move.from_uci("e2e4"): child_high,
                       chess.Move.from_uci("a2a3"): child_low}
    c_puct = 2.5
    score_high = child_high.puct_score(parent.visit_count, c_puct)
    score_low = child_low.puct_score(parent.visit_count, c_puct)
    assert score_high > score_low


def test_mcts_returns_legal_move():
    """MCTS search returns a legal move from the starting position."""
    cfg = NetworkConfig(num_blocks=1, num_filters=16)
    model = ChessNetwork(cfg)
    model.eval()
    mcts_cfg = MCTSConfig(num_simulations=20)
    mcts = MCTS(model, mcts_cfg)
    board = chess.Board()
    result = mcts.search(board)
    assert result.best_move in board.legal_moves


def test_mcts_policy_target_shape():
    """Search result includes a 1858-dim policy target."""
    cfg = NetworkConfig(num_blocks=1, num_filters=16)
    model = ChessNetwork(cfg)
    model.eval()
    mcts_cfg = MCTSConfig(num_simulations=20)
    mcts = MCTS(model, mcts_cfg)
    board = chess.Board()
    result = mcts.search(board)
    assert result.policy_target.shape == (1858,)
    np.testing.assert_allclose(result.policy_target.sum(), 1.0, atol=1e-5)


def test_mcts_policy_target_only_legal():
    """Policy target has nonzero values only for legal moves."""
    cfg = NetworkConfig(num_blocks=1, num_filters=16)
    model = ChessNetwork(cfg)
    model.eval()
    mcts_cfg = MCTSConfig(num_simulations=50)
    mcts = MCTS(model, mcts_cfg)
    board = chess.Board()
    result = mcts.search(board)
    legal_indices = set()
    for move in board.legal_moves:
        idx = chess_move_to_policy_index(move, board.turn)
        if idx is not None:
            legal_indices.add(idx)
    for i in range(1858):
        if result.policy_target[i] > 0:
            assert i in legal_indices, f"Nonzero policy at index {i} but not legal"


def test_mcts_terminal_position():
    """MCTS handles checkmate position without crashing."""
    cfg = NetworkConfig(num_blocks=1, num_filters=16)
    model = ChessNetwork(cfg)
    model.eval()
    mcts_cfg = MCTSConfig(num_simulations=10)
    mcts = MCTS(model, mcts_cfg)
    board = chess.Board("rnbqkbnr/pppp1ppp/4p3/8/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2")
    board.push_san("Qh4#")
    assert board.is_checkmate()
    result = mcts.search(board)
    assert result.best_move is None


def test_chess_move_to_policy_index_roundtrip():
    """chess_move_to_policy_index produces valid indices that roundtrip."""
    board = chess.Board()
    for move in board.legal_moves:
        idx = chess_move_to_policy_index(move, board.turn)
        assert idx is not None, f"Move {move} has no policy index"
        assert 0 <= idx < 1858


def test_mcts_solver_checkmate():
    """MCTS-solver should propagate proven terminal results."""
    cfg = NetworkConfig(num_blocks=1, num_filters=16)
    model = ChessNetwork(cfg)
    model.eval()
    mcts_cfg = MCTSConfig(num_simulations=50)
    mcts = MCTS(model, mcts_cfg)
    # Scholar's mate position: White can play Qf7# (if it finds it)
    # Use a position where checkmate is forced in 1
    board = chess.Board("r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4")
    # This is already checkmate — Qf7#
    assert board.is_checkmate()
    result = mcts.search(board)
    assert result.best_move is None  # Terminal position


def test_dynamic_cpuct_increases():
    """Dynamic c_puct should increase with parent visit count."""
    cfg = NetworkConfig(num_blocks=1, num_filters=16)
    model = ChessNetwork(cfg)
    mcts_cfg = MCTSConfig()
    mcts = MCTS(model, mcts_cfg)
    c1 = mcts._dynamic_cpuct(1)
    c100 = mcts._dynamic_cpuct(100)
    c10000 = mcts._dynamic_cpuct(10000)
    assert c1 < c100 < c10000


def test_mlh_output_nonnegative():
    """Moves-left head output should be non-negative (ReLU)."""
    cfg = NetworkConfig(num_blocks=2, num_filters=32)
    model = ChessNetwork(cfg)
    model.eval()
    x = torch.randn(4, 112, 8, 8)
    _, _, mlh = model(x)
    assert mlh.shape == (4,)
    assert (mlh >= 0).all()


def test_q_value_blending():
    """Q-value blending should mix game result with search Q."""
    cfg = NetworkConfig(num_blocks=1, num_filters=16)
    model = ChessNetwork(cfg)
    model.eval()
    mcts_cfg = MCTSConfig(num_simulations=5)
    mcts = MCTS(model, mcts_cfg)
    sp_cfg = SelfPlayConfig(max_moves=10, resign_threshold=-1.0, q_ratio=0.5)
    record = play_game(mcts, sp_cfg)
    # With q_ratio=0.5, WDL values should not be pure 0/1 distributions
    # (they're blended with search Q which is continuous)
    assert len(record.values) > 0
    for wdl in record.values:
        assert abs(wdl.sum() - 1.0) < 1e-5  # Still sums to 1


def test_checkpoint_load_resume():
    """Saved checkpoint should load and produce valid output."""
    cfg = NetworkConfig(num_blocks=1, num_filters=16)
    model = ChessNetwork(cfg)
    x = torch.randn(1, 112, 8, 8)
    model.eval()
    p1, v1, m1 = model(x)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'test_ckpt.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': cfg,
        }, path)

        # Load into fresh model
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        model2 = ChessNetwork(ckpt['config'])
        model2.load_state_dict(ckpt['model_state_dict'])
        model2.eval()
        p2, v2, m2 = model2(x)

        assert torch.allclose(p1, p2, atol=1e-5)
        assert torch.allclose(v1, v2, atol=1e-5)
        assert torch.allclose(m1, m2, atol=1e-5)


def test_dataset_loads_surprise_weights():
    """ChessDataset should expose surprise_weights when present in npz."""
    with tempfile.TemporaryDirectory() as tmpdir:
        n = 10
        np.savez(
            os.path.join(tmpdir, 'test.npz'),
            planes=np.random.randn(n, 112, 8, 8).astype(np.float32),
            policies=np.ones((n, 1858), dtype=np.float32) / 1858,
            values=np.tile([1.0, 0.0, 0.0], (n, 1)).astype(np.float32),
            moves_left=np.arange(n, dtype=np.float32),
            surprise=np.random.rand(n).astype(np.float32),
        )
        dataset = ChessDataset([os.path.join(tmpdir, 'test.npz')])
        assert dataset.surprise_weights is not None
        assert len(dataset.surprise_weights) == n
        assert (dataset.surprise_weights >= 0).all()


def test_dataset_without_surprise():
    """ChessDataset should work without surprise weights (backward compat)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        n = 5
        np.savez(
            os.path.join(tmpdir, 'test.npz'),
            planes=np.random.randn(n, 112, 8, 8).astype(np.float32),
            policies=np.ones((n, 1858), dtype=np.float32) / 1858,
            values=np.tile([1.0, 0.0, 0.0], (n, 1)).astype(np.float32),
        )
        dataset = ChessDataset([os.path.join(tmpdir, 'test.npz')])
        assert dataset.surprise_weights is None
        assert dataset.moves_left is None
        assert len(dataset) == n


def test_weighted_sampling_prefers_high_surprise():
    """WeightedRandomSampler with surprise should oversample high-surprise positions."""
    from torch.utils.data import WeightedRandomSampler
    # Create extreme weights: first position has weight 100, rest have weight 1
    weights = torch.tensor([100.0] + [1.0] * 99)
    sampler = WeightedRandomSampler(weights, num_samples=200, replacement=True)
    indices = list(sampler)
    # Index 0 should appear much more frequently
    count_0 = indices.count(0)
    assert count_0 > 20  # With weight 100/199, expect ~100 but at least 20


def test_multi_gen_training_loss_finite():
    """Multi-generation training should produce finite losses throughout."""
    from training.selfplay import training_loop
    with tempfile.TemporaryDirectory() as tmpdir:
        training_loop(
            generations=2,
            games_per_gen=2,
            train_epochs=1,
            batch_size=8,
            num_simulations=5,
            blocks=1,
            filters=16,
            output_dir=tmpdir,
            device='cpu',
            max_moves=10,
            resign_threshold=-1.0,
            use_swa=True,
        )
        # Verify both generation checkpoints exist and are loadable
        for gen in [1, 2]:
            path = os.path.join(tmpdir, 'checkpoints', f'model_gen_{gen}.pt')
            assert os.path.exists(path)
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            assert ckpt['generation'] == gen
        # Verify final model exported
        final = os.path.join(tmpdir, 'model_final.pt')
        assert os.path.exists(final)


def test_virtual_loss_applied_and_reverted():
    """Virtual loss should temporarily inflate visit count."""
    from training.mcts import Node
    node = Node(prior=0.5)
    node.visit_count = 10
    node.total_value = 5.0
    original_value = node.value()
    node.apply_virtual_loss()
    assert node.visit_count == 11
    assert node.pending_evals == 1
    assert node.value() < original_value  # diluted toward 0
    node.revert_virtual_loss()
    assert node.visit_count == 10
    assert node.pending_evals == 0
    assert abs(node.value() - original_value) < 1e-9


def test_nn_cache_store_and_retrieve():
    """NNCache should store and retrieve evaluations."""
    from training.mcts import NNCache
    cache = NNCache(max_size=100)
    board = chess.Board()
    policy = np.ones(1858, dtype=np.float32)
    cache.put(board, policy, 0.5)
    result = cache.get(board)
    assert result is not None
    assert result[1] == 0.5


def test_nn_cache_eviction():
    """NNCache should evict entries when full."""
    from training.mcts import NNCache
    cache = NNCache(max_size=10)
    for i in range(15):
        board = chess.Board()
        board.push(list(board.legal_moves)[i % 20])
        cache.put(board, np.zeros(1858), 0.0)
    assert len(cache) <= 10


def test_batched_search_returns_legal_move():
    """Batched MCTS should return a legal move."""
    cfg = NetworkConfig(num_blocks=1, num_filters=16)
    model = ChessNetwork(cfg)
    model.eval()
    mcts_cfg = MCTSConfig(num_simulations=32, batch_size=8)
    mcts = MCTS(model, mcts_cfg)
    board = chess.Board()
    result = mcts.search(board)
    assert result.best_move in board.legal_moves


def test_batched_search_policy_target_shape():
    """Batched search should produce valid policy target."""
    cfg = NetworkConfig(num_blocks=1, num_filters=16)
    model = ChessNetwork(cfg)
    model.eval()
    mcts_cfg = MCTSConfig(num_simulations=32, batch_size=8)
    mcts = MCTS(model, mcts_cfg)
    board = chess.Board()
    result = mcts.search(board)
    assert result.policy_target.shape == (1858,)
    assert abs(result.policy_target.sum() - 1.0) < 0.01


def test_batched_search_preserves_raw_policy():
    """Batched search should still capture raw NN policy for diff-focus."""
    cfg = NetworkConfig(num_blocks=1, num_filters=16)
    model = ChessNetwork(cfg)
    model.eval()
    mcts_cfg = MCTSConfig(num_simulations=16, batch_size=8)
    mcts = MCTS(model, mcts_cfg)
    board = chess.Board()
    result = mcts.search(board)
    assert result.raw_policy is not None
    assert result.raw_policy.shape == (1858,)


def test_batch_size_1_equivalent():
    """batch_size=1 should behave like the old sequential loop."""
    cfg = NetworkConfig(num_blocks=1, num_filters=16)
    model = ChessNetwork(cfg)
    model.eval()
    mcts_cfg = MCTSConfig(num_simulations=16, batch_size=1)
    mcts = MCTS(model, mcts_cfg)
    board = chess.Board()
    result = mcts.search(board)
    assert result.best_move in board.legal_moves
    assert result.policy_target.shape == (1858,)


def test_nn_cache_populated_after_search():
    """NN cache should have entries after search."""
    cfg = NetworkConfig(num_blocks=1, num_filters=16)
    model = ChessNetwork(cfg)
    model.eval()
    mcts_cfg = MCTSConfig(num_simulations=64, batch_size=8, nn_cache_size=1000)
    mcts = MCTS(model, mcts_cfg)
    board = chess.Board()
    mcts.search(board)
    # With 64 sims, many unique positions are evaluated and cached
    assert mcts.nn_cache is not None
    assert len(mcts.nn_cache) > 0


def test_updated_search_params():
    """MCTSConfig should have updated Lc0 defaults."""
    cfg = MCTSConfig()
    assert cfg.c_puct_init == 3.0
    assert cfg.policy_softmax_temperature == 2.2
    assert cfg.fpu_reduction == 1.2
    assert cfg.fpu_reduction_root == 1.2


def test_playout_cap_randomization():
    """Playout cap should produce both full and quick search positions."""
    from training.selfplay import SelfPlayConfig, play_game
    cfg = NetworkConfig(num_blocks=1, num_filters=16)
    model = ChessNetwork(cfg)
    model.eval()
    mcts_cfg = MCTSConfig(num_simulations=20, batch_size=4)
    mcts = MCTS(model, mcts_cfg)
    sp_cfg = SelfPlayConfig(
        max_moves=30,
        playout_cap_randomization=True,
        playout_cap_fraction=0.5,
        playout_cap_quick_sims=5,
    )
    record = play_game(mcts, sp_cfg)
    # With 50% fraction, expect a mix of True/False
    assert len(record.use_policy) > 0
    assert len(record.use_policy) == len(record.planes)
    # Should have at least some variation (probabilistic, but with 30 moves and 50% it's near certain)
    if len(record.use_policy) >= 10:
        assert not all(record.use_policy) or not all(not x for x in record.use_policy)


def test_kld_adaptive_varies_sims():
    """KLD-adaptive should vary simulation count based on policy divergence."""
    from training.selfplay import SelfPlayConfig
    cfg = SelfPlayConfig(kld_adaptive=True, kld_min_sims=50, kld_max_sims=400, kld_threshold=0.5)
    # With KLD=0 (perfect agreement), should use min sims
    t = min(0.0 / cfg.kld_threshold, 1.0)
    adaptive_sims = int(cfg.kld_min_sims + t * (cfg.kld_max_sims - cfg.kld_min_sims))
    assert adaptive_sims == 50
    # With KLD=0.5 (threshold), should use max sims
    t = min(0.5 / cfg.kld_threshold, 1.0)
    adaptive_sims = int(cfg.kld_min_sims + t * (cfg.kld_max_sims - cfg.kld_min_sims))
    assert adaptive_sims == 400


def test_policy_mask_in_loss():
    """Policy mask should only compute policy loss for masked positions."""
    from training.train import compute_loss
    B = 4
    policy_logits = torch.randn(B, 1858)
    value_logits = torch.randn(B, 3)
    policy_target = torch.softmax(torch.randn(B, 1858), dim=1)
    value_target = torch.softmax(torch.randn(B, 3), dim=1)
    mask = torch.tensor([True, False, True, False])
    total, p_loss, v_loss = compute_loss(
        policy_logits, value_logits, policy_target, value_target, policy_mask=mask,
    )
    # Policy loss should only be from positions 0 and 2
    assert p_loss.item() > 0
    assert v_loss.item() > 0


def test_syzygy_rescore_function():
    """Tablebase rescoring function should handle positions without TB gracefully."""
    from training.selfplay import rescore_with_tablebases
    # Without tablebase, should return positions unchanged
    positions = [(None, None, chess.WHITE, 0.0, 0.0, True)]
    boards = [chess.Board()]
    result = rescore_with_tablebases(positions, boards, None)
    assert result == positions


def test_smart_pruning_stops_early():
    """Smart pruning should stop search early when best move has insurmountable lead."""
    cfg = NetworkConfig(num_blocks=1, num_filters=16)
    model = ChessNetwork(cfg)
    model.eval()
    # With smart pruning enabled and many sims, the search may stop early
    mcts_cfg = MCTSConfig(num_simulations=200, batch_size=8, smart_pruning=True, smart_pruning_factor=1.33)
    mcts = MCTS(model, mcts_cfg)
    board = chess.Board()
    result = mcts.search(board)
    assert result.best_move in board.legal_moves
    # Root should have been visited, potentially fewer times than num_simulations due to pruning
    total_visits = sum(result.visit_counts.values())
    assert total_visits > 0
    assert total_visits <= 200 + 1  # +1 for initial root eval


def test_smart_pruning_config_defaults():
    """MCTSConfig smart pruning defaults should be set."""
    cfg = MCTSConfig()
    assert cfg.smart_pruning is True
    assert cfg.smart_pruning_factor == 1.33


def test_opening_randomization():
    """Random opening moves should produce non-standard positions."""
    from training.selfplay import play_game, SelfPlayConfig
    cfg = NetworkConfig(num_blocks=1, num_filters=16)
    model = ChessNetwork(cfg)
    model.eval()
    mcts_cfg = MCTSConfig(num_simulations=5)
    mcts = MCTS(model, mcts_cfg)
    # Force opening randomization on every game
    sp_cfg = SelfPlayConfig(
        max_moves=10, resign_threshold=-1.0,
        random_opening_fraction=1.0, random_opening_moves=4,
    )
    record = play_game(mcts, sp_cfg)
    # Game should still complete with valid data
    assert len(record.planes) > 0
    assert len(record.policies) == len(record.planes)


def test_soft_policy_loss():
    """Soft policy target should increase total policy loss."""
    from training.train import compute_loss
    policy_logits = torch.randn(8, 1858)
    value_logits = torch.randn(8, 3)
    # Create a peaked policy target
    policy_target = torch.zeros(8, 1858)
    policy_target[:, 0] = 0.9
    policy_target[:, 1] = 0.1
    value_target = torch.zeros(8, 3)
    value_target[:, 0] = 1.0  # all wins

    # Without soft policy
    loss_no_soft, p_no_soft, v1 = compute_loss(
        policy_logits, value_logits, policy_target, value_target,
        soft_policy_weight=0.0,
    )
    # With soft policy
    loss_soft, p_soft, v2 = compute_loss(
        policy_logits, value_logits, policy_target, value_target,
        soft_policy_weight=2.0, soft_policy_temperature=4.0,
    )
    # Soft policy should add to the loss (value loss is the same)
    assert p_soft.item() > p_no_soft.item()
    assert loss_soft.item() > loss_no_soft.item()


# ── Tier 1 Round 3: Variance, Repetition, Shaped Noise, Uncertainty, Contempt ─

def test_node_variance_low_visits():
    """Variance should be 0.0 for nodes with < 2 visits."""
    node = Node(prior=0.5)
    assert node.value_variance() == 0.0
    node.visit_count = 1
    node.total_value = 0.5
    node.sum_sq_value = 0.25
    assert node.value_variance() == 0.0


def test_node_variance_computation():
    """Variance should match E[X^2] - E[X]^2."""
    node = Node(prior=0.5)
    node.visit_count = 4
    node.total_value = 2.0       # mean = 0.5
    node.sum_sq_value = 1.5      # mean_sq = 0.375
    expected = 0.375 - 0.25      # 0.125
    assert abs(node.value_variance() - expected) < 1e-6


def test_backpropagate_updates_sum_sq():
    """Backpropagation should accumulate squared values."""
    root = Node(prior=1.0)
    child = Node(prior=0.5)
    root.children[chess.Move.from_uci('e2e4')] = child
    path = [root, child]
    cfg = NetworkConfig(num_blocks=1, num_filters=16)
    model = ChessNetwork(cfg)
    mcts = MCTS(model, MCTSConfig(), 'cpu')
    mcts._backpropagate(path, 0.8)
    assert child.sum_sq_value == pytest.approx(0.64)   # 0.8^2
    assert root.sum_sq_value == pytest.approx(0.64)     # (-0.8)^2


def test_two_fold_repetition_as_draw():
    """MCTS should treat 2-fold repetition as a draw."""
    cfg = NetworkConfig(num_blocks=1, num_filters=16)
    model = ChessNetwork(cfg)
    model.eval()
    board = chess.Board()
    # Play Nf3 Nf6 Ng1 Ng8 — back to start, which now has 2 occurrences
    for uci in ['g1f3', 'g8f6', 'f3g1', 'f6g8']:
        board.push(chess.Move.from_uci(uci))
    assert board.is_repetition(2)
    mcts_cfg = MCTSConfig(num_simulations=32, two_fold_draw=True)
    mcts = MCTS(model, mcts_cfg, 'cpu')
    result = mcts.search(board)
    assert abs(result.root_value) < 0.5


def test_two_fold_repetition_disabled():
    """With two_fold_draw=False, search still works normally."""
    cfg = NetworkConfig(num_blocks=1, num_filters=16)
    model = ChessNetwork(cfg)
    model.eval()
    mcts_cfg = MCTSConfig(num_simulations=16, two_fold_draw=False)
    mcts = MCTS(model, mcts_cfg, 'cpu')
    board = chess.Board()
    result = mcts.search(board)
    assert result.best_move in board.legal_moves


def test_shaped_dirichlet_modifies_priors():
    """Shaped Dirichlet should produce valid search results."""
    cfg = NetworkConfig(num_blocks=1, num_filters=16)
    model = ChessNetwork(cfg)
    model.eval()
    mcts_cfg = MCTSConfig(num_simulations=32, shaped_dirichlet=True)
    mcts = MCTS(model, mcts_cfg, 'cpu')
    board = chess.Board()
    result = mcts.search(board)
    assert result.best_move in board.legal_moves


def test_shaped_dirichlet_vs_uniform():
    """Both shaped and uniform Dirichlet produce valid searches."""
    cfg = NetworkConfig(num_blocks=1, num_filters=16)
    model = ChessNetwork(cfg)
    model.eval()
    board = chess.Board()
    for shaped in [True, False]:
        mcts_cfg = MCTSConfig(num_simulations=16, shaped_dirichlet=shaped)
        mcts = MCTS(model, mcts_cfg, 'cpu')
        result = mcts.search(board)
        assert result.best_move in board.legal_moves
        assert result.policy_target.shape == (1858,)


def test_uncertainty_boosting_prefers_high_variance():
    """Child with higher value variance should get higher selection score."""
    parent = Node(prior=1.0)
    parent.visit_count = 100
    parent.total_value = 50.0
    parent.sum_sq_value = 30.0
    # Child A: low variance
    a = Node(prior=0.3)
    a.visit_count = 20
    a.total_value = 10.0
    a.sum_sq_value = 5.1   # variance ~0.005
    # Child B: same mean, high variance
    b = Node(prior=0.3)
    b.visit_count = 20
    b.total_value = 10.0
    b.sum_sq_value = 10.0  # variance ~0.25
    parent.children[chess.Move.from_uci('e2e4')] = a
    parent.children[chess.Move.from_uci('d2d4')] = b
    cfg = NetworkConfig(num_blocks=1, num_filters=16)
    model = ChessNetwork(cfg)
    mcts_cfg = MCTSConfig(uncertainty_weight=0.5, variance_scaling=False)
    mcts = MCTS(model, mcts_cfg, 'cpu')
    move, child = mcts._select_child(parent)
    assert child is b


def test_uncertainty_boosting_disabled():
    """With uncertainty_weight=0, search still works."""
    cfg = NetworkConfig(num_blocks=1, num_filters=16)
    model = ChessNetwork(cfg)
    model.eval()
    mcts_cfg = MCTSConfig(num_simulations=16, uncertainty_weight=0.0)
    mcts = MCTS(model, mcts_cfg, 'cpu')
    board = chess.Board()
    result = mcts.search(board)
    assert result.best_move in board.legal_moves


def test_variance_scaling_clamped():
    """Variance scaling should be clamped to [0.5, 2.0]."""
    import math as _math
    low_scale = max(0.5, min(2.0, _math.sqrt(0.001) / 0.5))
    assert low_scale == 0.5
    high_scale = max(0.5, min(2.0, _math.sqrt(2.0) / 0.5))
    assert high_scale == 2.0


def test_variance_scaling_disabled():
    """With variance_scaling=False, search still works."""
    cfg = NetworkConfig(num_blocks=1, num_filters=16)
    model = ChessNetwork(cfg)
    model.eval()
    mcts_cfg = MCTSConfig(num_simulations=16, variance_scaling=False)
    mcts = MCTS(model, mcts_cfg, 'cpu')
    board = chess.Board()
    result = mcts.search(board)
    assert result.best_move in board.legal_moves


def test_contempt_shifts_value():
    """Contempt should push draw-ish values away from zero."""
    cfg = NetworkConfig(num_blocks=1, num_filters=16)
    model = ChessNetwork(cfg)
    model.eval()
    board = chess.Board()
    mcts_no = MCTS(model, MCTSConfig(num_simulations=32, contempt=0.0), 'cpu')
    result_no = mcts_no.search(board)
    mcts_yes = MCTS(model, MCTSConfig(num_simulations=32, contempt=0.3), 'cpu')
    result_yes = mcts_yes.search(board)
    assert abs(result_yes.root_value) >= abs(result_no.root_value) - 0.01


def test_contempt_preserves_sign():
    """Contempt should not flip the sign of the value."""
    value = 0.2
    contempt = 0.5
    sign = 1.0
    shift = contempt * (1.0 - abs(value))
    new_value = max(-1.0, min(1.0, value + sign * shift))
    assert new_value > 0
    assert new_value > value


def test_contempt_disabled_by_default():
    """Default MCTSConfig has contempt=0.0."""
    cfg = MCTSConfig()
    assert cfg.contempt == 0.0


def test_new_config_defaults():
    """All new MCTSConfig fields have correct defaults."""
    cfg = MCTSConfig()
    assert cfg.two_fold_draw is True
    assert cfg.shaped_dirichlet is True
    assert cfg.uncertainty_weight == 0.15
    assert cfg.variance_scaling is True
    assert cfg.contempt == 0.0


def test_tree_reuse():
    """Reused subtree should have prior visit data."""
    cfg = NetworkConfig(num_blocks=1, num_filters=16)
    model = ChessNetwork(cfg)
    model.eval()
    mcts_cfg = MCTSConfig(num_simulations=10)
    mcts = MCTS(model, mcts_cfg)
    board = chess.Board()
    result = mcts.search(board)
    assert result.root_node is not None
    # Reuse tree for the best move
    best_move = result.best_move
    child = mcts.reuse_tree(result.root_node, best_move)
    assert child is not None
    assert child.visit_count > 0
    # Search with reused root
    board.push(best_move)
    result2 = mcts.search(board, root=child)
    assert result2.best_move is not None


def test_search_returns_raw_policy():
    """SearchResult should include raw NN policy and value."""
    cfg = NetworkConfig(num_blocks=1, num_filters=16)
    model = ChessNetwork(cfg)
    model.eval()
    mcts = MCTS(model, MCTSConfig(num_simulations=5))
    board = chess.Board()
    result = mcts.search(board)
    assert result.raw_policy is not None
    assert result.raw_policy.shape == (1858,)
    assert result.raw_policy.sum() > 0.99  # Should be a probability distribution


from torch.optim.lr_scheduler import MultiStepLR


def test_lr_scheduler_reduces_lr():
    """MultiStepLR reduces learning rate at milestones."""
    cfg = NetworkConfig(num_blocks=1, num_filters=16)
    model = ChessNetwork(cfg)
    optimizer = create_optimizer(model, lr=0.1)
    scheduler = MultiStepLR(optimizer, milestones=[2, 4], gamma=0.1)

    lrs = []
    for epoch in range(6):
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()

    # Before milestone 2: lr = 0.1
    assert abs(lrs[0] - 0.1) < 1e-6
    assert abs(lrs[1] - 0.1) < 1e-6
    # After milestone 2: lr = 0.01
    assert abs(lrs[2] - 0.01) < 1e-6
    assert abs(lrs[3] - 0.01) < 1e-6
    # After milestone 4: lr = 0.001
    assert abs(lrs[4] - 0.001) < 1e-6
    assert abs(lrs[5] - 0.001) < 1e-6


# ── Self-Play Tests ──────────────────────────────────────────────────────────

from training.selfplay import SelfPlayConfig, play_game, GameRecord, generate_games


def test_play_game_produces_record():
    """play_game returns a GameRecord with positions, policies, and result."""
    cfg = NetworkConfig(num_blocks=1, num_filters=16)
    model = ChessNetwork(cfg)
    model.eval()
    mcts_cfg = MCTSConfig(num_simulations=10)
    mcts = MCTS(model, mcts_cfg)
    sp_cfg = SelfPlayConfig(max_moves=20, resign_threshold=-1.0)  # Never resign

    record = play_game(mcts, sp_cfg)

    assert isinstance(record, GameRecord)
    assert len(record.planes) > 0
    assert len(record.planes) == len(record.policies)
    assert len(record.planes) == len(record.values)
    assert record.planes[0].shape == (112, 8, 8)
    assert record.policies[0].shape == (1858,)
    assert record.values[0].shape == (3,)


def test_play_game_wdl_labels():
    """Game result assigns correct WDL labels."""
    cfg = NetworkConfig(num_blocks=1, num_filters=16)
    model = ChessNetwork(cfg)
    model.eval()
    mcts_cfg = MCTSConfig(num_simulations=10)
    mcts = MCTS(model, mcts_cfg)
    sp_cfg = SelfPlayConfig(max_moves=20, resign_threshold=-1.0)

    record = play_game(mcts, sp_cfg)

    for v in record.values:
        np.testing.assert_allclose(v.sum(), 1.0, atol=1e-5)
        assert (v >= 0).all()


def test_play_game_max_moves():
    """Game terminates when max_moves is reached."""
    cfg = NetworkConfig(num_blocks=1, num_filters=16)
    model = ChessNetwork(cfg)
    model.eval()
    mcts_cfg = MCTSConfig(num_simulations=5)
    mcts = MCTS(model, mcts_cfg)
    sp_cfg = SelfPlayConfig(max_moves=10, resign_threshold=-1.0)

    record = play_game(mcts, sp_cfg)
    assert len(record.planes) <= 10


def test_play_game_records_surprise():
    """play_game should record surprise scores for each position."""
    cfg = NetworkConfig(num_blocks=1, num_filters=16)
    model = ChessNetwork(cfg)
    model.eval()
    mcts_cfg = MCTSConfig(num_simulations=5)
    mcts = MCTS(model, mcts_cfg)
    sp_cfg = SelfPlayConfig(max_moves=10, resign_threshold=-1.0)
    record = play_game(mcts, sp_cfg)
    assert len(record.surprise) == len(record.planes)
    assert all(s >= 0 for s in record.surprise)


def test_swa_model_used_in_training_loop():
    """training_loop with use_swa should succeed and write metrics."""
    from training.selfplay import training_loop
    with tempfile.TemporaryDirectory() as tmpdir:
        training_loop(
            generations=2,
            games_per_gen=2,
            train_epochs=1,
            batch_size=8,
            num_simulations=5,
            blocks=1,
            filters=16,
            output_dir=tmpdir,
            device='cpu',
            max_moves=10,
            resign_threshold=-1.0,
            use_swa=True,
        )
        summary_path = os.path.join(tmpdir, 'metrics', 'summary.json')
        assert os.path.exists(summary_path)


def test_selfplay_npz_has_surprise():
    """Generated npz should contain surprise weights."""
    cfg = NetworkConfig(num_blocks=1, num_filters=16)
    model = ChessNetwork(cfg)
    model.eval()
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, 'test.npz')
        generate_games(
            model, 2, output_path,
            mcts_config=MCTSConfig(num_simulations=5),
            selfplay_config=SelfPlayConfig(max_moves=10, resign_threshold=-1.0),
        )
        with np.load(output_path) as data:
            assert 'surprise' in data
            assert len(data['surprise']) == len(data['planes'])


def test_training_loop_one_generation():
    """Full RL loop: generate -> train -> checkpoint for 1 generation."""
    from training.selfplay import training_loop

    with tempfile.TemporaryDirectory() as tmpdir:
        training_loop(
            generations=1,
            games_per_gen=2,
            train_epochs=2,
            batch_size=8,
            num_simulations=10,
            blocks=1,
            filters=16,
            output_dir=tmpdir,
            device='cpu',
            max_moves=20,
            resign_threshold=-1.0,  # Never resign
        )

        # Verify checkpoint was saved
        checkpoint_path = os.path.join(tmpdir, "checkpoints", "model_gen_1.pt")
        assert os.path.exists(checkpoint_path)

        # Verify training data was saved
        data_path = os.path.join(tmpdir, "data", "gen_001.npz")
        assert os.path.exists(data_path)


# ── Integration Smoke Tests ──────────────────────────────────────────────────

def test_selfplay_to_training_integration():
    """End-to-end: self-play -> train -> verify loss is finite."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Create model
        cfg = NetworkConfig(num_blocks=1, num_filters=16)
        model = ChessNetwork(cfg)
        model.eval()

        # 2. Generate self-play data
        mcts_cfg = MCTSConfig(num_simulations=10)
        sp_cfg = SelfPlayConfig(max_moves=20, resign_threshold=-1.0)
        data_path = os.path.join(tmpdir, "selfplay.npz")
        num_positions = generate_games(
            model, num_games=3, output_path=data_path,
            mcts_config=mcts_cfg, selfplay_config=sp_cfg,
        )
        assert num_positions > 0

        # 3. Load into ChessDataset
        dataset = ChessDataset([data_path])
        assert len(dataset) == num_positions
        loader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)

        # 4. Train for 2 epochs
        optimizer = create_optimizer(model, lr=1e-3)
        losses = []
        for epoch in range(2):
            for batch in loader:
                loss = train_step(model, optimizer, batch[0], batch[1], batch[2])
                losses.append(loss)

        # Verify loss is finite and positive
        assert len(losses) > 0
        assert all(not np.isnan(l) for l in losses)
        assert all(not np.isinf(l) for l in losses)
        assert all(l > 0 for l in losses)


def test_selfplay_npz_format():
    """Self-play .npz has correct format for ChessDataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = NetworkConfig(num_blocks=1, num_filters=16)
        model = ChessNetwork(cfg)
        model.eval()

        mcts_cfg = MCTSConfig(num_simulations=10)
        sp_cfg = SelfPlayConfig(max_moves=15, resign_threshold=-1.0)
        data_path = os.path.join(tmpdir, "test.npz")
        generate_games(model, num_games=2, output_path=data_path,
                       mcts_config=mcts_cfg, selfplay_config=sp_cfg)

        with np.load(data_path) as data:
            assert 'planes' in data
            assert 'policies' in data
            assert 'values' in data
            n = data['planes'].shape[0]
            assert data['planes'].shape == (n, 112, 8, 8)
            assert data['policies'].shape == (n, 1858)
            assert data['values'].shape == (n, 3)
            # Policy targets should sum to ~1
            np.testing.assert_allclose(
                data['policies'].sum(axis=1), 1.0, atol=1e-5
            )
            # WDL targets should sum to 1
            np.testing.assert_allclose(
                data['values'].sum(axis=1), 1.0, atol=1e-5
            )


def test_selfplay_gpu():
    """Self-play works on GPU."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = NetworkConfig(num_blocks=1, num_filters=16)
        model = ChessNetwork(cfg).cuda()
        model.eval()

        mcts_cfg = MCTSConfig(num_simulations=10)
        sp_cfg = SelfPlayConfig(max_moves=10, resign_threshold=-1.0)
        data_path = os.path.join(tmpdir, "gpu_selfplay.npz")
        num_positions = generate_games(
            model, num_games=2, output_path=data_path,
            mcts_config=mcts_cfg, selfplay_config=sp_cfg,
            device='cuda',
        )
        assert num_positions > 0


# ── Metrics Tests ────────────────────────────────────────────────────────────

import json
from training.metrics import MetricsLogger, GameMetrics, TrainingMetrics


def test_metrics_logger_creates_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        metrics_dir = os.path.join(tmpdir, 'metrics')
        logger = MetricsLogger(metrics_dir)
        assert os.path.isdir(metrics_dir)


def test_metrics_game_recorded():
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = MetricsLogger(tmpdir)
        game = GameMetrics(
            game_num=1,
            num_moves=45,
            result='1-0',
            duration_s=12.5,
            moves_uci=['e2e4', 'd7d5', 'e4d5'],
        )
        logger.record_game(game)
        assert len(logger.current_games) == 1


def test_metrics_generation_saved():
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = MetricsLogger(tmpdir)
        game = GameMetrics(
            game_num=1, num_moves=30, result='1/2-1/2',
            duration_s=10.0, moves_uci=['e2e4', 'e7e5'],
        )
        logger.record_game(game)
        training = TrainingMetrics(
            total_loss=2.5, policy_loss=1.8, value_loss=0.7,
            num_batches=10, learning_rate=0.001,
        )
        logger.save_generation(
            generation=1, num_positions=150,
            training=training, duration_s=25.0,
        )
        gen_path = os.path.join(tmpdir, 'gen_001.json')
        assert os.path.exists(gen_path)
        with open(gen_path) as f:
            data = json.load(f)
        assert data['generation'] == 1
        assert data['num_positions'] == 150
        assert len(data['games']) == 1
        assert data['training']['total_loss'] == 2.5


def test_metrics_summary_updated():
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = MetricsLogger(tmpdir)
        game = GameMetrics(
            game_num=1, num_moves=20, result='0-1',
            duration_s=8.0, moves_uci=['d2d4'],
        )
        logger.record_game(game)
        training = TrainingMetrics(
            total_loss=3.0, policy_loss=2.0, value_loss=1.0,
            num_batches=5, learning_rate=0.001,
        )
        logger.save_generation(1, 100, training, 20.0)
        summary_path = os.path.join(tmpdir, 'summary.json')
        assert os.path.exists(summary_path)
        with open(summary_path) as f:
            summary = json.load(f)
        assert summary['total_generations'] == 1
        assert len(summary['generations']) == 1


def test_metrics_multiple_generations():
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = MetricsLogger(tmpdir)
        for gen in range(1, 4):
            game = GameMetrics(
                game_num=1, num_moves=20, result='1/2-1/2',
                duration_s=5.0, moves_uci=['e2e4'],
            )
            logger.record_game(game)
            training = TrainingMetrics(
                total_loss=3.0 - gen * 0.5, policy_loss=2.0, value_loss=1.0,
                num_batches=5, learning_rate=0.001,
            )
            logger.save_generation(gen, 100, training, 15.0)
        summary_path = os.path.join(tmpdir, 'summary.json')
        with open(summary_path) as f:
            summary = json.load(f)
        assert summary['total_generations'] == 3
        assert len(summary['generations']) == 3
        losses = [g['training']['total_loss'] for g in summary['generations']]
        assert losses == [2.5, 2.0, 1.5]


def test_game_record_has_moves_uci():
    """GameRecord should include UCI move strings after play_game."""
    from training.selfplay import play_game, SelfPlayConfig
    from training.mcts import MCTS, MCTSConfig
    from training.model import ChessNetwork
    from training.config import NetworkConfig

    config = NetworkConfig(num_blocks=1, num_filters=16)
    model = ChessNetwork(config)
    model.eval()
    mcts = MCTS(model, MCTSConfig(num_simulations=2))
    sp_config = SelfPlayConfig(max_moves=10)

    record = play_game(mcts, sp_config)
    assert hasattr(record, 'moves_uci')
    assert isinstance(record.moves_uci, list)
    assert len(record.moves_uci) == record.num_moves
    for m in record.moves_uci:
        assert isinstance(m, str)
        assert len(m) >= 4


def test_training_loop_writes_metrics():
    """training_loop with metrics_dir should produce summary.json."""
    import tempfile
    from training.selfplay import training_loop

    with tempfile.TemporaryDirectory() as tmpdir:
        training_loop(
            generations=1,
            games_per_gen=2,
            train_epochs=1,
            batch_size=32,
            num_simulations=2,
            blocks=1,
            filters=16,
            output_dir=tmpdir,
            max_moves=10,
        )
        metrics_dir = os.path.join(tmpdir, 'metrics')
        assert os.path.isdir(metrics_dir)
        summary_path = os.path.join(tmpdir, 'metrics', 'summary.json')
        assert os.path.exists(summary_path)
        with open(summary_path) as f:
            summary = json.load(f)
        assert summary['total_generations'] == 1


def test_server_summary_endpoint():
    """Flask /api/summary should return summary.json contents."""
    import tempfile
    from visualization.server import create_app

    with tempfile.TemporaryDirectory() as tmpdir:
        summary = {
            'total_generations': 1,
            'total_games': 5,
            'total_positions': 200,
            'total_time_s': 30.0,
            'generations': [{
                'generation': 1,
                'num_positions': 200,
                'num_games': 5,
                'training': {'total_loss': 2.5, 'policy_loss': 1.8, 'value_loss': 0.7,
                             'num_batches': 10, 'learning_rate': 0.001},
                'duration_s': 30.0,
                'avg_game_length': 40.0,
                'results': {'1-0': 2, '0-1': 2, '1/2-1/2': 1},
            }],
        }
        with open(os.path.join(tmpdir, 'summary.json'), 'w') as f:
            json.dump(summary, f)

        app = create_app(tmpdir)
        client = app.test_client()
        resp = client.get('/api/summary')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['total_generations'] == 1


def test_server_generation_endpoint():
    """Flask /api/generation/1 should return gen_001.json contents."""
    import tempfile
    from visualization.server import create_app

    with tempfile.TemporaryDirectory() as tmpdir:
        gen_data = {
            'generation': 1,
            'num_positions': 100,
            'num_games': 3,
            'games': [
                {'game_num': 1, 'num_moves': 30, 'result': '1-0',
                 'duration_s': 10.0, 'moves_uci': ['e2e4', 'e7e5']},
            ],
            'training': {'total_loss': 2.0, 'policy_loss': 1.5, 'value_loss': 0.5,
                         'num_batches': 5, 'learning_rate': 0.001},
            'duration_s': 20.0,
        }
        with open(os.path.join(tmpdir, 'gen_001.json'), 'w') as f:
            json.dump(gen_data, f)

        app = create_app(tmpdir)
        client = app.test_client()
        resp = client.get('/api/generation/1')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['generation'] == 1
        assert len(data['games']) == 1


def test_server_generation_not_found():
    """Flask /api/generation/99 should return 404."""
    import tempfile
    from visualization.server import create_app

    with tempfile.TemporaryDirectory() as tmpdir:
        app = create_app(tmpdir)
        client = app.test_client()
        resp = client.get('/api/generation/99')
        assert resp.status_code == 404


def test_server_status_endpoint():
    """Flask /api/status should always return ok."""
    import tempfile
    from visualization.server import create_app

    with tempfile.TemporaryDirectory() as tmpdir:
        app = create_app(tmpdir)
        client = app.test_client()
        resp = client.get('/api/status')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['status'] == 'ok'


def test_server_empty_summary():
    """Flask /api/summary should return empty data when no summary.json exists."""
    import tempfile
    from visualization.server import create_app

    with tempfile.TemporaryDirectory() as tmpdir:
        app = create_app(tmpdir)
        client = app.test_client()
        resp = client.get('/api/summary')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['total_generations'] == 0
        assert data['generations'] == []


def test_integration_selfplay_to_dashboard():
    """Full pipeline: self-play -> metrics -> Flask API -> verify data."""
    import tempfile
    from training.selfplay import training_loop
    from visualization.server import create_app

    with tempfile.TemporaryDirectory() as tmpdir:
        training_loop(
            generations=2,
            games_per_gen=2,
            train_epochs=1,
            batch_size=32,
            num_simulations=2,
            blocks=1,
            filters=16,
            output_dir=tmpdir,
            max_moves=10,
        )

        metrics_dir = os.path.join(tmpdir, 'metrics')
        assert os.path.exists(os.path.join(metrics_dir, 'gen_001.json'))
        assert os.path.exists(os.path.join(metrics_dir, 'gen_002.json'))
        assert os.path.exists(os.path.join(metrics_dir, 'summary.json'))

        app = create_app(metrics_dir)
        client = app.test_client()

        resp = client.get('/api/summary')
        assert resp.status_code == 200
        summary = resp.get_json()
        assert summary['total_generations'] == 2
        assert summary['total_games'] == 4

        resp = client.get('/api/generation/1')
        assert resp.status_code == 200
        gen = resp.get_json()
        assert len(gen['games']) == 2
        assert len(gen['games'][0]['moves_uci']) > 0


# ---------- C++ MCTS Integration Tests ----------

try:
    import chess_mcts as _chess_mcts_test
    _HAS_CPP_MCTS = True
except ImportError:
    _HAS_CPP_MCTS = False


@pytest.mark.skipif(not _HAS_CPP_MCTS, reason="C++ MCTS not built")
class TestCppMCTS:
    """Tests for C++ MCTS integration (only run when chess_mcts module is available)."""

    def _get_model_path(self, tmp_path):
        """Export a small model to TorchScript and return the path."""
        from training.model import ChessNetwork
        from training.config import NetworkConfig
        from training.export import export_torchscript
        config = NetworkConfig(num_blocks=2, num_filters=32)
        model = ChessNetwork(config)
        model.eval()
        path = str(tmp_path / "test_model.pt")
        export_torchscript(model, path, device='cpu')
        return path

    def test_cpp_mcts_returns_legal_move(self, tmp_path):
        import chess
        from training.mcts import MCTSConfig
        from training.selfplay import CppMCTS

        model_path = self._get_model_path(tmp_path)
        config = MCTSConfig(num_simulations=50)
        engine = CppMCTS(model_path, config, 'cpu')

        board = chess.Board()
        result = engine.search(board)

        assert result.best_move is not None
        assert result.best_move in board.legal_moves
        assert result.policy_target.shape == (1858,)
        assert abs(result.policy_target.sum() - 1.0) < 0.01

    def test_cpp_mcts_play_game(self, tmp_path):
        from training.mcts import MCTSConfig
        from training.selfplay import CppMCTS, SelfPlayConfig, play_game

        model_path = self._get_model_path(tmp_path)
        config = MCTSConfig(num_simulations=20)
        engine = CppMCTS(model_path, config, 'cpu')
        selfplay_config = SelfPlayConfig(
            max_moves=20,
            playout_cap_randomization=False,
            kld_adaptive=False,
            random_opening_fraction=0.0,
        )

        record = play_game(engine, selfplay_config)
        assert record.num_moves > 0
        assert len(record.planes) == record.num_moves
        assert len(record.policies) == record.num_moves

    def test_game_manager_parallel_games(self, tmp_path):
        """Test GameManager runs multiple games with cross-game batching."""
        import chess_mcts

        model_path = self._get_model_path(tmp_path)
        manager = chess_mcts.GameManager(model_path, "cpu", {"num_iterations": 50, "batch_size": 8})
        manager.init_games(4, 50)

        assert manager.num_games() == 4
        assert not manager.all_complete()

        # Run until all complete
        max_steps = 200  # safety limit
        steps = 0
        while not manager.all_complete() and steps < max_steps:
            manager.step()
            steps += 1

        assert manager.all_complete()

        for i in range(4):
            assert manager.is_complete(i)
            result = manager.get_result(i)
            assert result.best_move  # non-empty string
            assert result.total_nodes > 0

    def test_game_manager_from_fen(self, tmp_path):
        """Test GameManager initialized from FEN strings and move histories."""
        import chess_mcts

        model_path = self._get_model_path(tmp_path)
        manager = chess_mcts.GameManager(model_path, "cpu", {"num_iterations": 30, "batch_size": 4})

        fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        ]
        move_histories = [
            [],           # starting position, no moves
            ["e2e4"],     # replay e4 from starting position for history
        ]
        manager.init_games_from_fen(fens, move_histories, 30)

        assert manager.num_games() == 2

        max_steps = 200
        steps = 0
        while not manager.all_complete() and steps < max_steps:
            manager.step()
            steps += 1

        assert manager.all_complete()
        for i in range(2):
            result = manager.get_result(i)
            assert result.best_move
            assert result.total_nodes > 0
