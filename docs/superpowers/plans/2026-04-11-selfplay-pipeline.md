# Self-Play & Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the reinforcement learning loop — generate self-play training games using Python MCTS + PyTorch model, then train on that data, entirely in Python.

**Architecture:** Python-only pipeline using `python-chess` for move generation and the PyTorch model directly for evaluation. Self-play generates `.npz` files consumed by the existing `ChessDataset`. The full RL loop runs: generate games → train network → export weights → repeat.

**Tech Stack:** Python 3.11, PyTorch 2.11, python-chess, numpy

---

## Pre-Execution Setup

1. Install python-chess: `pip install python-chess`
2. Create feature branch `feature/selfplay-pipeline` from `main`
3. Verify existing tests pass: `python -m pytest training/test_training.py -v`

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `training/model.py` | Modify | Remove softmax from value head (return raw logits) |
| `training/train.py` | Modify | Use `F.cross_entropy` for value loss, add LR scheduler, split loss logging |
| `training/export.py` | Modify | Apply softmax to value head at export time |
| `training/encoder.py` | Modify | Add `encode_board()` for python-chess Board with 8-step history |
| `training/mcts.py` | Create | Python MCTS: Node, PUCT, search, move-to-policy bridge |
| `training/selfplay.py` | Create | Self-play game generation + full RL training loop |
| `training/test_training.py` | Modify | Add tests for all new functionality (~18 new tests) |
| `docs/changelog.md` | Modify | Stamp v0.4.0 |
| `docs/architecture.md` | Modify | Update Plan 4 section |

---

### Task 1: Fix value head numerical stability (model.py + train.py)

**Files:**
- Modify: `training/model.py:106` (remove softmax)
- Modify: `training/train.py:27-53` (replace value loss)
- Modify: `training/export.py:15-36` (apply softmax at export)
- Modify: `training/test_training.py:203-209` (update value head test)

- [ ] **Step 1: Update the value head test to expect raw logits**

In `training/test_training.py`, replace `test_model_value_probabilities`:

```python
def test_model_value_probabilities():
    cfg = NetworkConfig(num_blocks=2, num_filters=32)
    model = ChessNetwork(cfg)
    x = torch.randn(1, 112, 8, 8)
    _, value = model(x)
    # Value head now returns raw logits (not probabilities)
    assert value.shape == (1, 3)
    assert value.dtype == torch.float32
    # Logits can be any real number — verify softmax produces valid probs
    probs = torch.softmax(value, dim=1)
    assert torch.allclose(probs.sum(dim=1), torch.tensor([1.0]), atol=1e-5)
    assert (probs >= 0).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest training/test_training.py::test_model_value_probabilities -v`
Expected: FAIL — model still returns softmaxed probabilities, so logits can be any real number doesn't fail, but the key test is that after this change and step 3, all tests still pass.

Actually, this test will pass even with the current model since softmax output is also float32 with shape (1,3) and softmax of softmax still sums to ~1. The real validation comes from the full suite passing after the model change.

- [ ] **Step 3: Remove softmax from value head in model.py**

In `training/model.py`, line 106, change:

```python
        v = F.softmax(self.value_fc2(v), dim=1)  # WDL probabilities
```

to:

```python
        v = self.value_fc2(v)  # Raw WDL logits (softmax applied in loss / at inference)
```

- [ ] **Step 4: Replace value loss with F.cross_entropy in train.py**

In `training/train.py`, replace the `compute_loss` function:

```python
def compute_loss(
    policy_logits: torch.Tensor,
    value_logits: torch.Tensor,
    policy_target: torch.Tensor,
    value_target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute combined policy + value loss.

    Args:
        policy_logits: Raw logits from policy head (B, 1858)
        value_logits: Raw logits from value head (B, 3)
        policy_target: MCTS visit distribution target (B, 1858)
        value_target: WDL target (B, 3)

    Returns:
        (total_loss, policy_loss, value_loss) tuple.
    """
    # Policy loss: cross-entropy with soft targets
    # = -sum(target * log_softmax(logits))
    log_probs = F.log_softmax(policy_logits, dim=1)
    policy_loss = -(policy_target * log_probs).sum(dim=1).mean()

    # Value loss: cross-entropy with soft WDL targets
    # = -sum(target * log_softmax(logits))
    value_log_probs = F.log_softmax(value_logits, dim=1)
    value_loss = -(value_target * value_log_probs).sum(dim=1).mean()

    return policy_loss + value_loss, policy_loss, value_loss
```

- [ ] **Step 5: Update train_step to use new compute_loss signature**

In `training/train.py`, update `train_step`:

```python
def train_step(
    model: ChessNetwork,
    optimizer: torch.optim.Optimizer,
    planes: torch.Tensor,
    policy_target: torch.Tensor,
    value_target: torch.Tensor,
) -> float:
    """Execute one training step. Returns the loss value."""
    model.train()
    optimizer.zero_grad()

    policy_logits, value_logits = model(planes)
    loss, _, _ = compute_loss(policy_logits, value_logits, policy_target, value_target)

    loss.backward()
    optimizer.step()

    return loss.item()
```

- [ ] **Step 6: Update export.py to apply softmax at export time**

In `training/export.py`, wrap the model in a module that applies softmax before tracing:

```python
class _ExportWrapper(torch.nn.Module):
    """Wraps ChessNetwork to apply softmax to value head for inference."""

    def __init__(self, model: ChessNetwork):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        policy_logits, value_logits = self.model(x)
        value_probs = torch.softmax(value_logits, dim=1)
        return policy_logits, value_probs
```

Add `import torch.nn` to the imports if not present. Then in `export_torchscript`, change the tracing to use the wrapper:

```python
def export_torchscript(
    model: ChessNetwork,
    output_path: str,
    device: str = 'cpu',
):
    """Export a ChessNetwork to TorchScript via tracing.

    The exported model applies softmax to the value head output,
    so it returns (policy_logits, value_probabilities).
    """
    model = model.to(device)
    model.eval()

    wrapper = _ExportWrapper(model)
    wrapper.eval()

    example = torch.randn(1, model.config.input_planes, 8, 8, device=device)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example)

    traced.save(output_path)
    print(f"Exported TorchScript model to {output_path}")
```

- [ ] **Step 7: Update export tests to account for softmax in export vs raw logits in model**

In `training/test_training.py`, update `test_export_torchscript`:

```python
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
            orig_policy, orig_value_logits = model(x)
            loaded_policy, loaded_value_probs = loaded(x)

        # Policy logits should match exactly
        assert torch.allclose(orig_policy, loaded_policy, atol=1e-5)
        # Exported model applies softmax to value head
        orig_value_probs = torch.softmax(orig_value_logits, dim=1)
        assert torch.allclose(orig_value_probs, loaded_value_probs, atol=1e-5)
```

Update `test_export_torchscript_gpu` similarly:

```python
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
            orig_policy, orig_value_logits = model_cpu(x)
            loaded_policy, loaded_value_probs = loaded(x)

        assert torch.allclose(orig_policy, loaded_policy, atol=1e-5)
        orig_value_probs = torch.softmax(orig_value_logits, dim=1)
        assert torch.allclose(orig_value_probs, loaded_value_probs, atol=1e-5)
```

Update `test_end_to_end_pipeline` to also handle the export comparison:

```python
def test_end_to_end_pipeline():
    """Full pipeline: generate data -> train -> export -> verify."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, "train.npz")
        generate_synthetic_data(data_path, num_positions=64)

        dataset = ChessDataset([data_path])
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        cfg = NetworkConfig(num_blocks=1, num_filters=16)
        model = ChessNetwork(cfg)
        optimizer = create_optimizer(model, lr=1e-3)

        model.train()
        losses = []
        for planes, policies, values in loader:
            loss = train_step(model, optimizer, planes, policies, values)
            losses.append(loss)

        assert len(losses) > 0
        assert all(not np.isnan(l) for l in losses)

        export_path = os.path.join(tmpdir, "model.pt")
        export_torchscript(model, export_path)
        assert os.path.exists(export_path)

        loaded = torch.jit.load(export_path)
        model.eval()
        x = torch.randn(1, 112, 8, 8)
        with torch.no_grad():
            orig_p, orig_v_logits = model(x)
            load_p, load_v_probs = loaded(x)
        assert torch.allclose(orig_p, load_p, atol=1e-5)
        orig_v_probs = torch.softmax(orig_v_logits, dim=1)
        assert torch.allclose(orig_v_probs, load_v_probs, atol=1e-5)
```

- [ ] **Step 8: Run all tests**

Run: `python -m pytest training/test_training.py -v`
Expected: All 32 tests PASS

- [ ] **Step 9: Commit**

```bash
git add training/model.py training/train.py training/export.py training/test_training.py
git commit -m "refactor: value head returns raw logits, use log_softmax in loss

Value head no longer applies softmax — returns raw logits for numerical
stability. Loss uses log_softmax instead of log(softmax+eps). Export
wrapper applies softmax for C++ inference. compute_loss now returns
(total, policy, value) tuple for split logging."
```

---

### Task 2: Add position history encoding (encoder.py)

**Files:**
- Modify: `training/encoder.py` (add `encode_board` function)
- Modify: `training/test_training.py` (add history encoding tests)

- [ ] **Step 1: Write the failing tests**

Add to `training/test_training.py`:

```python
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
    # Time step 0 (current position after 1.e4 e5 2.Nf3) differs from
    # time step 1 (position after 1.e4 e5) — at minimum the knight plane changes
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest training/test_training.py::test_encode_board_shape -v`
Expected: FAIL — `encode_board` does not exist yet

- [ ] **Step 3: Implement encode_board in encoder.py**

Add to `training/encoder.py`, after the existing `encode_position` function:

```python
def encode_board(board) -> np.ndarray:
    """Encode a python-chess Board as a 112x8x8 float32 tensor with position history.

    Unlike encode_position(fen), this uses the board's move stack to fill
    8 time steps with actual history positions instead of repeating the current one.

    Args:
        board: A python-chess Board object with move history.

    Returns:
        np.ndarray of shape (112, 8, 8), dtype float32.
    """
    import chess

    is_white = (board.turn == chess.WHITE)
    planes = np.zeros((112, 8, 8), dtype=np.float32)

    # We need to encode 8 time steps: current position + 7 history positions.
    # Walk backwards through the move stack to reconstruct earlier positions.
    # We work on a copy so we don't mutate the original board.
    history_board = board.copy()
    for t in range(8):
        step_planes = np.zeros((13, 8, 8), dtype=np.float32)

        # Encode pieces on the board
        for sq in range(64):
            piece = history_board.piece_at(sq)
            if piece is None:
                continue
            piece_is_white = (piece.color == chess.WHITE)
            piece_type = piece.piece_type - 1  # chess.PAWN=1..KING=6 → 0..5

            # Flip square if side-to-move is Black (we always orient from STM perspective)
            actual_sq = sq if is_white else mirror_move(sq)
            r, f = rank_of(actual_sq), file_of(actual_sq)

            if piece_is_white == is_white:
                step_planes[piece_type, r, f] = 1.0  # Our piece
            else:
                step_planes[6 + piece_type, r, f] = 1.0  # Opponent piece

        # Repetition plane (plane 12): 1 if position has occurred before
        if history_board.is_repetition(1):
            step_planes[12] = 1.0

        planes[t * 13:(t + 1) * 13] = step_planes

        # Step back one move for the next time step
        if history_board.move_stack:
            history_board.pop()
        # If no more history, remaining time steps will repeat this earliest position
        # (the loop continues but the board doesn't change)

    # Constant planes (104-111)
    if is_white:
        planes[104] = 1.0  # Color to move

    # Total move count (normalized)
    planes[105] = board.fullmove_number / 200.0

    # Castling rights (from side-to-move perspective)
    if is_white:
        if board.has_kingside_castling_rights(chess.WHITE):
            planes[106] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE):
            planes[107] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK):
            planes[108] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK):
            planes[109] = 1.0
    else:
        if board.has_kingside_castling_rights(chess.BLACK):
            planes[106] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK):
            planes[107] = 1.0
        if board.has_kingside_castling_rights(chess.WHITE):
            planes[108] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE):
            planes[109] = 1.0

    # Halfmove clock (normalized)
    planes[110] = board.halfmove_clock / 100.0

    # All-ones bias
    planes[111] = 1.0

    return planes
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest training/test_training.py -k "encode_board" -v`
Expected: All 5 new tests PASS

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest training/test_training.py -v`
Expected: All tests PASS (32 existing + 5 new = 37)

- [ ] **Step 6: Commit**

```bash
git add training/encoder.py training/test_training.py
git commit -m "feat: add encode_board() with 8-step position history

Uses python-chess Board.move_stack to walk back through game history,
filling 8 time steps with actual previous positions instead of repeating
the current position. Includes repetition detection via board.is_repetition()."
```

---

### Task 3: Add learning rate scheduling (train.py)

**Files:**
- Modify: `training/train.py` (add scheduler, split loss logging)
- Modify: `training/test_training.py` (add scheduler test)

- [ ] **Step 1: Write the failing test**

Add to `training/test_training.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it passes**

Run: `python -m pytest training/test_training.py::test_lr_scheduler_reduces_lr -v`
Expected: PASS — this test uses PyTorch's built-in `MultiStepLR`, no custom code needed yet.

- [ ] **Step 3: Add scheduler to the train() function**

In `training/train.py`, add the import:

```python
from torch.optim.lr_scheduler import MultiStepLR
```

Update the `train` function signature and body to include scheduler and split loss logging:

```python
def train(
    data_paths: list[str],
    config: NetworkConfig = NetworkConfig(),
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    lr_milestones: list[int] | None = None,
    lr_gamma: float = 0.1,
    checkpoint_dir: str = 'checkpoints',
    device: str = 'auto',
):
    """Full training loop."""
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Device: {device}")
    print(f"Config: {config.num_blocks} blocks, {config.num_filters} filters")

    dataset = ChessDataset(data_paths)
    print(f"Training data: {len(dataset)} positions from {len(data_paths)} files")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = ChessNetwork(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    optimizer = create_optimizer(model, lr=lr, weight_decay=weight_decay)

    scheduler = None
    if lr_milestones:
        scheduler = MultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_gamma)
        print(f"LR schedule: milestones={lr_milestones}, gamma={lr_gamma}")

    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        num_batches = 0
        start = time.time()

        for planes, policies, values in loader:
            planes = planes.to(device)
            policies = policies.to(device)
            values = values.to(device)

            model.train()
            optimizer.zero_grad()

            policy_logits, value_logits = model(planes)
            loss, p_loss, v_loss = compute_loss(
                policy_logits, value_logits, policies, values
            )

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_policy_loss += p_loss.item()
            epoch_value_loss += v_loss.item()
            num_batches += 1

        if scheduler:
            scheduler.step()

        elapsed = time.time() - start
        avg_loss = epoch_loss / max(num_batches, 1)
        avg_p = epoch_policy_loss / max(num_batches, 1)
        avg_v = epoch_value_loss / max(num_batches, 1)
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {avg_loss:.4f} (policy: {avg_p:.4f}, value: {avg_v:.4f}) | "
            f"LR: {current_lr:.6f} | Time: {elapsed:.1f}s"
        )

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'loss': avg_loss,
            }, path)
            print(f"  Saved checkpoint: {path}")
```

Note: `train_step` is kept as-is for use in tests and the selfplay loop. The `train()` function now inlines the training logic so it can log split losses.

- [ ] **Step 4: Run all tests**

Run: `python -m pytest training/test_training.py -v`
Expected: All tests PASS (37 existing + 1 new = 38)

- [ ] **Step 5: Commit**

```bash
git add training/train.py training/test_training.py
git commit -m "feat: add MultiStepLR scheduler and split loss logging

train() now accepts lr_milestones and lr_gamma for step LR decay.
Loss is logged as total, policy, and value components per epoch.
compute_loss returns (total, policy, value) tuple."
```

---

### Task 4: Python MCTS implementation (mcts.py)

**Files:**
- Create: `training/mcts.py`
- Modify: `training/test_training.py` (add MCTS tests)

- [ ] **Step 1: Write the failing tests**

Add to `training/test_training.py`:

```python
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
    assert node.value() == 0.5  # 2.0 / 4


def test_node_puct_prefers_high_prior():
    """PUCT formula prefers unvisited children with higher prior."""
    parent = Node(prior=1.0)
    parent.visit_count = 10
    child_high = Node(prior=0.9)
    child_low = Node(prior=0.1)
    parent.children = {chess.Move.from_uci("e2e4"): child_high,
                       chess.Move.from_uci("a2a3"): child_low}

    # With zero visits, PUCT score is dominated by prior * exploration term
    # High prior child should have higher PUCT score
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
    # Policy target should sum to ~1 (normalized visit counts)
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

    # Get legal move indices
    legal_indices = set()
    for move in board.legal_moves:
        idx = chess_move_to_policy_index(move, board.turn)
        if idx is not None:
            legal_indices.add(idx)

    # All nonzero entries in policy_target should be at legal indices
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

    # Scholar's mate position (Black is checkmated)
    board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    # White is checkmated (Qh4# is already played) — actually let me use a real checkmate
    board = chess.Board("rnbqkbnr/pppp1ppp/4p3/8/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2")
    board.push_san("Qh4#")  # Now White is checkmated
    assert board.is_checkmate()
    # Search on a checkmated position should handle gracefully
    result = mcts.search(board)
    assert result.best_move is None  # No legal moves


def test_chess_move_to_policy_index_roundtrip():
    """chess_move_to_policy_index produces valid indices that roundtrip."""
    board = chess.Board()
    for move in board.legal_moves:
        idx = chess_move_to_policy_index(move, board.turn)
        assert idx is not None, f"Move {move} has no policy index"
        assert 0 <= idx < POLICY_SIZE
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest training/test_training.py::test_node_initial_state -v`
Expected: FAIL — `training/mcts.py` does not exist yet

- [ ] **Step 3: Implement mcts.py**

Create `training/mcts.py`:

```python
"""Python MCTS implementation for self-play.

Uses python-chess for move generation and PyTorch model for position evaluation.
Single-threaded, single-position inference — correct and simple, not fast.
"""

import math
from dataclasses import dataclass, field

import chess
import numpy as np
import torch

from training.config import NetworkConfig
from training.encoder import encode_board, move_to_index, mirror_move, POLICY_SIZE


@dataclass
class MCTSConfig:
    num_simulations: int = 400
    c_puct: float = 2.5
    fpu_reduction: float = 0.25
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0


@dataclass
class SearchResult:
    best_move: chess.Move | None
    visit_counts: dict[chess.Move, int]
    root_value: float
    policy_target: np.ndarray  # shape (1858,)


class Node:
    """MCTS tree node."""

    __slots__ = ['prior', 'visit_count', 'total_value', 'children']

    def __init__(self, prior: float):
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0
        self.children: dict[chess.Move, 'Node'] = {}

    def value(self) -> float:
        """Average value from this node's perspective."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def puct_score(self, parent_visits: int, c_puct: float) -> float:
        """PUCT = Q + c * P * sqrt(N_parent) / (1 + N_child)."""
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.value() + exploration

    def is_expanded(self) -> bool:
        return len(self.children) > 0


def chess_move_to_policy_index(move: chess.Move, turn: chess.Color) -> int | None:
    """Convert a python-chess Move to a policy index in [0, 1857].

    The policy encoding assumes the board is oriented from the side-to-move's
    perspective. For Black, squares are mirrored vertically.

    Args:
        move: A python-chess Move.
        turn: chess.WHITE or chess.BLACK — the side that made the move.

    Returns:
        Policy index, or None if the move cannot be encoded.
    """
    from_sq = move.from_square
    to_sq = move.to_square

    # Mirror for Black (our encoding assumes white's perspective)
    if turn == chess.BLACK:
        from_sq = mirror_move(from_sq)
        to_sq = mirror_move(to_sq)

    # Determine promotion piece
    promo = None
    if move.promotion is not None and move.promotion != chess.QUEEN:
        promo_map = {chess.KNIGHT: 'n', chess.BISHOP: 'b', chess.ROOK: 'r'}
        promo = promo_map.get(move.promotion)

    return move_to_index(from_sq, to_sq, promo)


class MCTS:
    """Monte Carlo Tree Search using a neural network for evaluation."""

    def __init__(self, model: torch.nn.Module, config: MCTSConfig, device: str = 'cpu'):
        self.model = model
        self.config = config
        self.device = device

    def search(self, board: chess.Board) -> SearchResult:
        """Run MCTS search on the given position.

        Args:
            board: Current chess position.

        Returns:
            SearchResult with best move, visit counts, and policy target.
        """
        if board.is_game_over():
            return SearchResult(
                best_move=None,
                visit_counts={},
                root_value=self._terminal_value(board),
                policy_target=np.zeros(POLICY_SIZE, dtype=np.float32),
            )

        root = Node(prior=1.0)
        # Expand root
        policy, value = self._evaluate(board)
        self._expand(root, board, policy)

        # Add Dirichlet noise at root for exploration
        self._add_dirichlet_noise(root)

        root.visit_count = 1
        root.total_value = value

        for _ in range(self.config.num_simulations):
            node = root
            scratch_board = board.copy()
            path = [node]

            # Select
            while node.is_expanded() and not scratch_board.is_game_over():
                move, node = self._select_child(node)
                scratch_board.push(move)
                path.append(node)

            # Evaluate
            if scratch_board.is_game_over():
                leaf_value = self._terminal_value(scratch_board)
            elif not node.is_expanded():
                policy, leaf_value = self._evaluate(scratch_board)
                self._expand(node, scratch_board, policy)
            else:
                leaf_value = self._terminal_value(scratch_board)

            # Backpropagate — negate value as we go up (alternating perspective)
            self._backpropagate(path, leaf_value)

        return self._build_result(root, board)

    def _evaluate(self, board: chess.Board) -> tuple[np.ndarray, float]:
        """Evaluate position with neural network.

        Returns:
            (policy, value): policy is 1858-dim numpy array, value is float in [-1, 1].
        """
        planes = encode_board(board)
        tensor = torch.from_numpy(planes).unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_logits, value_logits = self.model(tensor)

        # Policy: softmax over logits
        policy_probs = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()

        # Value: WDL logits → softmax → scalar value (W - L)
        wdl = torch.softmax(value_logits, dim=1).squeeze(0).cpu().numpy()
        value = float(wdl[0] - wdl[2])  # Win - Loss

        return policy_probs, value

    def _expand(self, node: Node, board: chess.Board, policy: np.ndarray):
        """Create child nodes for all legal moves."""
        for move in board.legal_moves:
            idx = chess_move_to_policy_index(move, board.turn)
            prior = policy[idx] if idx is not None else 1e-6
            node.children[move] = Node(prior=prior)

        # Normalize priors
        total_prior = sum(child.prior for child in node.children.values())
        if total_prior > 0:
            for child in node.children.values():
                child.prior /= total_prior

    def _select_child(self, node: Node) -> tuple[chess.Move, 'Node']:
        """Select child with highest PUCT score."""
        best_score = -float('inf')
        best_move = None
        best_child = None

        fpu_value = node.value() - self.config.fpu_reduction

        for move, child in node.children.items():
            if child.visit_count == 0:
                score = fpu_value + self.config.c_puct * child.prior * math.sqrt(node.visit_count)
            else:
                score = child.puct_score(node.visit_count, self.config.c_puct)
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child

        return best_move, best_child

    def _add_dirichlet_noise(self, root: Node):
        """Add Dirichlet noise to root node priors for exploration."""
        if not root.children:
            return
        noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(root.children))
        eps = self.config.dirichlet_epsilon
        for i, child in enumerate(root.children.values()):
            child.prior = (1 - eps) * child.prior + eps * noise[i]

    def _terminal_value(self, board: chess.Board) -> float:
        """Return terminal position value from the current side-to-move's perspective."""
        if board.is_checkmate():
            return -1.0  # Current player is checkmated
        return 0.0  # Draw (stalemate, insufficient material, etc.)

    def _backpropagate(self, path: list[Node], leaf_value: float):
        """Backpropagate value through the path, negating at each level."""
        # The leaf_value is from the perspective of the player at the leaf.
        # As we walk up, we negate since each level is the opposite player.
        value = leaf_value
        for node in reversed(path):
            # Negate because parent is the opponent of child
            node.total_value += value
            node.visit_count += 1
            value = -value

    def _build_result(self, root: Node, board: chess.Board) -> SearchResult:
        """Build SearchResult from completed search."""
        visit_counts = {move: child.visit_count for move, child in root.children.items()}

        # Select best move by visit count (with temperature)
        best_move = self._select_move(root, board)

        # Build 1858-dim policy target from visit counts
        policy_target = np.zeros(POLICY_SIZE, dtype=np.float32)
        total_visits = sum(visit_counts.values())
        if total_visits > 0:
            for move, visits in visit_counts.items():
                idx = chess_move_to_policy_index(move, board.turn)
                if idx is not None:
                    policy_target[idx] = visits / total_visits

        root_value = root.value()
        return SearchResult(
            best_move=best_move,
            visit_counts=visit_counts,
            root_value=root_value,
            policy_target=policy_target,
        )

    def _select_move(self, root: Node, board: chess.Board) -> chess.Move:
        """Select a move from the root based on visit counts and temperature."""
        moves = list(root.children.keys())
        visits = np.array([root.children[m].visit_count for m in moves], dtype=np.float64)

        if self.config.temperature <= 0.01:
            # Greedy: pick highest visit count
            best_idx = np.argmax(visits)
        else:
            # Temperature-scaled sampling
            visits = visits ** (1.0 / self.config.temperature)
            probs = visits / visits.sum()
            best_idx = np.random.choice(len(moves), p=probs)

        return moves[best_idx]
```

- [ ] **Step 4: Run MCTS tests**

Run: `python -m pytest training/test_training.py -k "mcts or node or puct or policy_index" -v`
Expected: All 8 new MCTS tests PASS

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest training/test_training.py -v`
Expected: All tests PASS (38 existing + 8 new = 46)

- [ ] **Step 6: Commit**

```bash
git add training/mcts.py training/test_training.py
git commit -m "feat: add Python MCTS with PUCT, Dirichlet noise, temperature selection

Node with visit count, value, prior, PUCT scoring. MCTS search loop
with neural network evaluation. chess_move_to_policy_index bridges
python-chess moves to our 1858-dim encoding with Black mirroring.
Configurable: simulations, c_puct, FPU reduction, Dirichlet noise."
```

---

### Task 5: Self-play game generation (selfplay.py)

**Files:**
- Create: `training/selfplay.py`
- Modify: `training/test_training.py` (add self-play tests)

- [ ] **Step 1: Write the failing tests**

Add to `training/test_training.py`:

```python
from training.selfplay import SelfPlayConfig, play_game, GameRecord


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

    # All value labels should be valid WDL distributions
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest training/test_training.py::test_play_game_produces_record -v`
Expected: FAIL — `training/selfplay.py` does not exist

- [ ] **Step 3: Implement selfplay.py (game generation)**

Create `training/selfplay.py`:

```python
"""Self-play game generation and reinforcement learning loop.

Usage:
    python -m training.selfplay generate --games 50 --simulations 100
    python -m training.selfplay loop --generations 10 --games-per-gen 50
"""

import argparse
import os
import time
from dataclasses import dataclass, field

import chess
import numpy as np
import torch

from training.config import NetworkConfig
from training.encoder import encode_board, POLICY_SIZE
from training.model import ChessNetwork
from training.mcts import MCTS, MCTSConfig


@dataclass
class SelfPlayConfig:
    temperature_moves: int = 30  # Use temperature=1.0 for first N moves
    max_moves: int = 512
    resign_threshold: float = -0.95  # Resign if value < this for consecutive_resign moves
    consecutive_resign: int = 5


@dataclass
class GameRecord:
    planes: list[np.ndarray] = field(default_factory=list)
    policies: list[np.ndarray] = field(default_factory=list)
    values: list[np.ndarray] = field(default_factory=list)
    result: str = '*'  # '1-0', '0-1', '1/2-1/2', '*'
    num_moves: int = 0


def play_game(mcts: MCTS, config: SelfPlayConfig) -> GameRecord:
    """Play a single self-play game.

    Args:
        mcts: MCTS search instance with model.
        config: Self-play configuration.

    Returns:
        GameRecord with encoded positions, policy targets, and WDL labels.
    """
    board = chess.Board()
    record = GameRecord()

    positions = []  # (planes, policy_target, side_to_move)
    resign_count = 0

    for move_num in range(config.max_moves):
        if board.is_game_over(claim_draw=True):
            break

        # Set temperature based on move number
        if move_num < config.temperature_moves:
            mcts.config.temperature = 1.0
        else:
            mcts.config.temperature = 0.01  # Near-greedy

        # Search
        result = mcts.search(board)
        if result.best_move is None:
            break

        # Record position
        planes = encode_board(board)
        positions.append((planes, result.policy_target, board.turn))

        # Check resign condition
        if result.root_value < config.resign_threshold:
            resign_count += 1
            if resign_count >= config.consecutive_resign:
                # Current side resigns
                if board.turn == chess.WHITE:
                    record.result = '0-1'
                else:
                    record.result = '1-0'
                break
        else:
            resign_count = 0

        board.push(result.best_move)

    # Determine game result if not resigned
    if record.result == '*':
        outcome = board.outcome(claim_draw=True)
        if outcome is None:
            record.result = '1/2-1/2'  # Max moves reached → draw
        elif outcome.winner is None:
            record.result = '1/2-1/2'
        elif outcome.winner == chess.WHITE:
            record.result = '1-0'
        else:
            record.result = '0-1'

    # Label all positions with WDL based on game result
    for planes, policy_target, side_to_move in positions:
        if record.result == '1/2-1/2':
            wdl = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        elif record.result == '1-0':
            if side_to_move == chess.WHITE:
                wdl = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # Win
            else:
                wdl = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # Loss
        else:  # 0-1
            if side_to_move == chess.BLACK:
                wdl = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # Win
            else:
                wdl = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # Loss

        record.planes.append(planes)
        record.policies.append(policy_target)
        record.values.append(wdl)

    record.num_moves = len(positions)
    return record


def generate_games(
    model: ChessNetwork,
    num_games: int,
    output_path: str,
    mcts_config: MCTSConfig = MCTSConfig(),
    selfplay_config: SelfPlayConfig = SelfPlayConfig(),
    device: str = 'cpu',
) -> int:
    """Generate self-play games and save as .npz file.

    Args:
        model: Neural network for evaluation.
        num_games: Number of games to generate.
        output_path: Path to save .npz file.
        mcts_config: MCTS search configuration.
        selfplay_config: Self-play configuration.
        device: Device for model inference.

    Returns:
        Total number of training positions generated.
    """
    model = model.to(device)
    model.eval()
    mcts = MCTS(model, mcts_config, device=device)

    all_planes = []
    all_policies = []
    all_values = []

    start = time.time()

    for game_num in range(num_games):
        game_start = time.time()
        record = play_game(mcts, selfplay_config)
        game_time = time.time() - game_start

        all_planes.extend(record.planes)
        all_policies.extend(record.policies)
        all_values.extend(record.values)

        total_positions = len(all_planes)
        print(
            f"  Game {game_num + 1}/{num_games}: "
            f"{record.num_moves} moves, {record.result}, "
            f"{game_time:.1f}s ({total_positions} positions total)"
        )

    elapsed = time.time() - start
    total_positions = len(all_planes)

    # Save as .npz
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    np.savez(
        output_path,
        planes=np.array(all_planes, dtype=np.float32),
        policies=np.array(all_policies, dtype=np.float32),
        values=np.array(all_values, dtype=np.float32),
    )

    print(f"Generated {num_games} games, {total_positions} positions in {elapsed:.1f}s")
    print(f"Saved to {output_path}")
    return total_positions
```

- [ ] **Step 4: Run self-play tests**

Run: `python -m pytest training/test_training.py -k "play_game" -v`
Expected: All 3 self-play tests PASS

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest training/test_training.py -v`
Expected: All tests PASS (46 existing + 3 new = 49)

- [ ] **Step 6: Commit**

```bash
git add training/selfplay.py training/test_training.py
git commit -m "feat: add self-play game generation with temperature and resign

SelfPlayConfig with temperature schedule (tau=1 for first 30 moves,
near-greedy after), resign threshold, and max move limit. play_game
generates a full game with MCTS search and WDL labeling. generate_games
saves training data as .npz compatible with ChessDataset."
```

---

### Task 6: Full training loop (selfplay.py)

**Files:**
- Modify: `training/selfplay.py` (add `training_loop` and CLI)
- Modify: `training/test_training.py` (add training loop test)

- [ ] **Step 1: Write the failing test**

Add to `training/test_training.py`:

```python
from training.selfplay import training_loop


def test_training_loop_one_generation():
    """Full RL loop: generate -> train -> checkpoint for 1 generation."""
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest training/test_training.py::test_training_loop_one_generation -v`
Expected: FAIL — `training_loop` does not exist yet

- [ ] **Step 3: Implement training_loop and CLI in selfplay.py**

Add to `training/selfplay.py`, after the `generate_games` function:

```python
def training_loop(
    generations: int = 10,
    games_per_gen: int = 50,
    train_epochs: int = 5,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_simulations: int = 400,
    blocks: int = 10,
    filters: int = 128,
    window_size: int = 5,
    output_dir: str = 'selfplay_output',
    device: str = 'auto',
    max_moves: int = 512,
    resign_threshold: float = -0.95,
):
    """Full self-play reinforcement learning loop.

    Each generation: generate games -> train on recent data -> save checkpoint.

    Args:
        generations: Number of RL generations.
        games_per_gen: Games to generate per generation.
        train_epochs: Training epochs per generation.
        batch_size: Training batch size.
        lr: Learning rate.
        weight_decay: Weight decay.
        num_simulations: MCTS simulations per move.
        blocks: Number of residual blocks.
        filters: Number of filters.
        window_size: Number of recent generations to train on.
        output_dir: Base directory for output files.
        device: Device for training and inference.
        max_moves: Maximum moves per game.
        resign_threshold: Resign if value below this.
    """
    from torch.utils.data import DataLoader
    from training.dataset import ChessDataset
    from training.train import compute_loss, create_optimizer
    from training.export import export_torchscript

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_dir = os.path.join(output_dir, 'data')
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    config = NetworkConfig(num_blocks=blocks, num_filters=filters)
    model = ChessNetwork(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Self-Play Training Loop")
    print(f"  Device: {device}")
    print(f"  Network: {blocks} blocks, {filters} filters ({total_params:,} params)")
    print(f"  Generations: {generations}, Games/gen: {games_per_gen}")
    print(f"  Simulations/move: {num_simulations}")
    print()

    mcts_config = MCTSConfig(num_simulations=num_simulations)
    selfplay_config = SelfPlayConfig(
        max_moves=max_moves,
        resign_threshold=resign_threshold,
    )

    optimizer = create_optimizer(model, lr=lr, weight_decay=weight_decay)

    for gen in range(1, generations + 1):
        gen_start = time.time()
        print(f"=== Generation {gen}/{generations} ===")

        # 1. Generate self-play games
        print(f"Generating {games_per_gen} games...")
        data_path = os.path.join(data_dir, f"gen_{gen:03d}.npz")
        num_positions = generate_games(
            model, games_per_gen, data_path,
            mcts_config=mcts_config,
            selfplay_config=selfplay_config,
            device=device,
        )

        # 2. Train on recent data (sliding window)
        recent_paths = []
        for g in range(max(1, gen - window_size + 1), gen + 1):
            p = os.path.join(data_dir, f"gen_{g:03d}.npz")
            if os.path.exists(p):
                recent_paths.append(p)

        print(f"Training on {len(recent_paths)} generation(s) of data...")
        dataset = ChessDataset(recent_paths)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        for epoch in range(train_epochs):
            epoch_loss = 0.0
            epoch_p_loss = 0.0
            epoch_v_loss = 0.0
            num_batches = 0

            for planes, policies, values in loader:
                planes = planes.to(device)
                policies = policies.to(device)
                values = values.to(device)

                model.train()
                optimizer.zero_grad()
                policy_logits, value_logits = model(planes)
                loss, p_loss, v_loss = compute_loss(
                    policy_logits, value_logits, policies, values
                )
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_p_loss += p_loss.item()
                epoch_v_loss += v_loss.item()
                num_batches += 1

            if num_batches > 0:
                avg = epoch_loss / num_batches
                avg_p = epoch_p_loss / num_batches
                avg_v = epoch_v_loss / num_batches
                print(
                    f"  Epoch {epoch + 1}/{train_epochs}: "
                    f"loss={avg:.4f} (policy={avg_p:.4f}, value={avg_v:.4f})"
                )

        # 3. Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"model_gen_{gen}.pt")
        torch.save({
            'generation': gen,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
        }, checkpoint_path)

        gen_time = time.time() - gen_start
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"  Generation time: {gen_time:.1f}s")
        print()

    # Export final model
    export_path = os.path.join(checkpoint_dir, "model_final.pt")
    export_torchscript(model, export_path, device='cpu')
    print(f"Training complete. Final model: {export_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Self-play training for chess AI')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Generate subcommand
    gen_parser = subparsers.add_parser('generate', help='Generate self-play games')
    gen_parser.add_argument('--games', type=int, default=50)
    gen_parser.add_argument('--simulations', type=int, default=400)
    gen_parser.add_argument('--output', type=str, default='data/selfplay.npz')
    gen_parser.add_argument('--blocks', type=int, default=10)
    gen_parser.add_argument('--filters', type=int, default=128)
    gen_parser.add_argument('--device', type=str, default='auto')
    gen_parser.add_argument('--max-moves', type=int, default=512)

    # Loop subcommand
    loop_parser = subparsers.add_parser('loop', help='Run full RL training loop')
    loop_parser.add_argument('--generations', type=int, default=10)
    loop_parser.add_argument('--games-per-gen', type=int, default=50)
    loop_parser.add_argument('--train-epochs', type=int, default=5)
    loop_parser.add_argument('--batch-size', type=int, default=256)
    loop_parser.add_argument('--lr', type=float, default=1e-3)
    loop_parser.add_argument('--simulations', type=int, default=400)
    loop_parser.add_argument('--blocks', type=int, default=10)
    loop_parser.add_argument('--filters', type=int, default=128)
    loop_parser.add_argument('--window-size', type=int, default=5)
    loop_parser.add_argument('--output-dir', type=str, default='selfplay_output')
    loop_parser.add_argument('--device', type=str, default='auto')
    loop_parser.add_argument('--max-moves', type=int, default=512)

    args = parser.parse_args()

    if args.command == 'generate':
        device = args.device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        config = NetworkConfig(num_blocks=args.blocks, num_filters=args.filters)
        model = ChessNetwork(config).to(device)
        model.eval()

        mcts_config = MCTSConfig(num_simulations=args.simulations)
        selfplay_config = SelfPlayConfig(max_moves=args.max_moves)

        generate_games(
            model, args.games, args.output,
            mcts_config=mcts_config,
            selfplay_config=selfplay_config,
            device=device,
        )

    elif args.command == 'loop':
        training_loop(
            generations=args.generations,
            games_per_gen=args.games_per_gen,
            train_epochs=args.train_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            num_simulations=args.simulations,
            blocks=args.blocks,
            filters=args.filters,
            window_size=args.window_size,
            output_dir=args.output_dir,
            device=args.device,
            max_moves=args.max_moves,
        )
```

- [ ] **Step 4: Run the training loop test**

Run: `python -m pytest training/test_training.py::test_training_loop_one_generation -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest training/test_training.py -v`
Expected: All tests PASS (49 existing + 1 new = 50)

- [ ] **Step 6: Commit**

```bash
git add training/selfplay.py training/test_training.py
git commit -m "feat: add full RL training loop with sliding window

training_loop generates self-play games, trains on sliding window of
recent generations, saves checkpoints and exports final TorchScript model.
CLI with generate and loop subcommands."
```

---

### Task 7: Integration smoke test

**Files:**
- Modify: `training/test_training.py` (add integration tests)

- [ ] **Step 1: Write integration tests**

Add to `training/test_training.py`:

```python
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
            for planes, policies, values in loader:
                loss = train_step(model, optimizer, planes, policies, values)
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
```

- [ ] **Step 2: Run integration tests**

Run: `python -m pytest training/test_training.py -k "integration or npz_format or selfplay_gpu" -v`
Expected: All 3 tests PASS (GPU test skipped if no CUDA)

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest training/test_training.py -v`
Expected: All tests PASS (~53 total)

- [ ] **Step 4: Commit**

```bash
git add training/test_training.py
git commit -m "test: add integration smoke tests for self-play pipeline

End-to-end: random model -> self-play -> train -> verify finite loss.
Verify .npz format matches ChessDataset expectations. GPU self-play test."
```

---

### Task 8: Documentation

**Files:**
- Modify: `docs/changelog.md`
- Modify: `docs/architecture.md`

- [ ] **Step 1: Update changelog**

Add under `## [Unreleased]` in `docs/changelog.md`, replacing "Nothing yet.":

```markdown
## [0.4.0] - 2026-04-11

**Self-Play & Data Pipeline — Plan 4 complete.**

### Added
- Python MCTS implementation: PUCT selection, Dirichlet noise, temperature selection (`training/mcts.py`)
- Self-play game generation: temperature schedule, resign logic, WDL labeling (`training/selfplay.py`)
- Full RL training loop: generate → train → export → repeat with sliding window (`training/selfplay.py`)
- Position history encoding: `encode_board()` fills 8 time steps from move stack (`training/encoder.py`)
- Move bridge: `chess_move_to_policy_index()` maps python-chess moves to 1858-dim encoding
- LR scheduling: `MultiStepLR` support in training loop
- CLI: `python -m training.selfplay generate` and `python -m training.selfplay loop`
- Integration tests: self-play → train end-to-end, GPU self-play

### Changed
- Value head returns raw logits (previously applied softmax in model)
- Value loss uses `log_softmax` for numerical stability (previously `log(softmax + eps)`)
- `compute_loss` returns `(total, policy, value)` tuple for split loss logging
- TorchScript export applies softmax wrapper for value head
- Training loop logs policy and value loss separately

### Dependencies
- Added: `python-chess` for self-play move generation
```

- [ ] **Step 2: Update architecture.md Plan 4 section**

Replace the Plan 4 section in `docs/architecture.md` (the part starting with `### 4. Self-Play Pipeline (C++)`) with:

```markdown
### 4. Self-Play & Data Pipeline (Python) — Plan 4

Generates training games using Python MCTS + PyTorch model directly. Written entirely in Python to prove the RL loop before optimizing in C++ (Plan 5).

```
training/
├── mcts.py         Python MCTS: Node, PUCT, search, move-to-policy bridge
├── selfplay.py     Self-play game generation + full RL training loop
├── encoder.py      (modified) Added encode_board() with 8-step history
├── model.py        (modified) Value head returns raw logits
├── train.py        (modified) log_softmax loss, LR scheduler, split logging
└── export.py       (modified) Softmax wrapper for TorchScript export
```

**Data flow:**
```
Current PyTorch model
  ↓
Self-play game loop (python-chess + MCTS):
  Position → encode_board() → model → (policy, value)
  MCTS search (N simulations) → move selection (with temperature)
  Record: (encoded position, visit distribution, side to move)
  ↓
Game result: W/D/L
  ↓
Label all positions with WDL
  ↓
Save training data to .npz
  ↓
Train on sliding window of recent generations
  ↓
Export TorchScript → loop back
```

**Key design decisions:**
- Python-only (no C++ needed) — uses python-chess for move generation
- Single-position inference (no batching) — simple, correct, fast enough for validation
- Small network (2 blocks, 32 filters) for development speed (~10-30s per game)
- Value head returns raw logits for numerical stability (softmax applied at inference/export)
- En passant not encoded as separate input plane (matches Lc0; inferable from position history)

**Implementation status:** Complete. Full RL loop running: self-play → train → export → repeat.
```

- [ ] **Step 3: Commit**

```bash
git add docs/changelog.md docs/architecture.md
git commit -m "docs: stamp v0.4.0, update architecture for Python self-play pipeline"
```

---

## Verification

After all tasks:

1. `python -m pytest training/test_training.py -v` — all ~53 tests pass
2. `python -m training.selfplay generate --games 5 --simulations 100 --blocks 2 --filters 32` — generates games
3. `python -m training.selfplay loop --generations 3 --games-per-gen 10 --train-epochs 5 --blocks 2 --filters 32` — full RL loop runs
4. Verify loss is finite across all generations
5. `ctest --test-dir build --build-config Release` — C++ tests still pass (96 tests, no regression)
6. Final code review subagent across entire implementation
7. Use `superpowers:finishing-a-development-branch` to wrap up
