# Lc0 Optimizations Research

Research findings from analyzing the Leela Chess Zero source code ([LeelaChessZero/lc0](https://github.com/LeelaChessZero/lc0), [lczero-training](https://github.com/LeelaChessZero/lczero-training)) and documentation, compared against our current implementation. Prioritized for an RTX 3080 + Ryzen 7 5800X setup.

---

## Already Implemented

These Lc0 features are already in our codebase:

| Feature | Status | Notes |
|---|---|---|
| SE layers (Lc0 variant: weights + biases) | Done | `training/model.py` |
| WDL value head | Done | 3-output softmax |
| FPU reduction (root=0.44, non-root=0.25) | Done | `training/mcts.py`, `src/mcts/search.cpp` |
| Dirichlet noise (α=0.3, ε=0.25) | Done | Root only |
| Mirror data augmentation (a↔h flip) | Done | `training/dataset.py` |
| Mixed precision training (AMP) | Done | `training/selfplay.py`, `training/train.py` |
| 112-plane input encoding | Done | Matches Lc0 classic format |
| 1858-dim policy encoding | Done | Lc0/AlphaZero move encoding |
| Dynamic c_puct (Tier 1) | Done | `training/mcts.py`, `src/mcts/search.cpp` — AlphaZero defaults |
| Policy softmax temperature (Tier 1) | Done | `training/mcts.py`, `src/neural/neural_evaluator.cpp` — PST=1.61 |
| Gradient clipping (Tier 1) | Done | `training/train.py`, `training/selfplay.py` — max_norm=10.0 |
| LR warmup (Tier 1) | Done | `training/train.py` — 250 steps LinearLR + SequentialLR |
| Mish activation (Tier 1) | Done | `training/model.py` — replaced all ReLU |
| Moves-left head (Tier 2) | Done | `training/model.py`, `training/train.py` — Huber loss, δ=10, scaled 1/20 |
| Q-value blending (Tier 2) | Done | `training/selfplay.py` — q_ratio config, soft WDL blending |
| MCTS-solver (Tier 2) | Done | `training/mcts.py` — terminal_status propagation in tree |
| Attention policy head (Tier 2) | Done | `training/model.py` — Q@K^T attention with promotion offsets |
| SGD + Nesterov (Tier 3) | Done | `training/train.py` — config option alongside AdamW |
| Tree reuse (Tier 3) | Done | `training/mcts.py` — `reuse_tree()` + `search(root=)` |
| Diff-focus sampling (Tier 3) | Done | `training/selfplay.py`, `training/dataset.py` — surprise-weighted sampling |
| SWA (Tier 3) | Done | `training/selfplay.py` — `AveragedModel` for self-play |
| Glorot normal init (Tier 3) | Done | `training/model.py` — `_init_weights()` with `xavier_normal_` |
| Two-fold repetition as draw (Round 3) | Done | `training/mcts.py` — `is_repetition(2)` check in search loop |
| WDL rescale/contempt (Round 3) | Done | `training/mcts.py` — `contempt` config, applied in `_build_result` |
| Shaped Dirichlet noise (Round 3) | Done | `training/mcts.py` — KataGo-style `_add_dirichlet_noise` |
| Uncertainty boosting (Round 3) | Done | `training/mcts.py` — Ceres-style variance bonus in `_select_child` |
| Variance-scaled cPUCT (Round 3) | Done | `training/mcts.py` — KataGo-style parent variance scaling |
| Node value variance tracking (Round 3) | Done | `training/mcts.py` — `sum_sq_value` + `value_variance()` |
| Temperature decay schedule (Round 3 T2) | Done | `training/selfplay.py` — smooth decay with 0.4 floor |
| Badgame split (Round 3 T2) | Done | `training/selfplay.py` — fork on Q-gap, replay greedy |
| Sibling blending / Ceres FPU (Round 3 T2) | Done | `training/mcts.py` — `_sibling_fpu()` in `_select_child` |
| Policy edge sorting (Round 3 T2) | Done | `training/mcts.py` — sorted by prior in `_expand()` |
| Prior compression float16 (Round 3 T2) | Done | `training/mcts.py` — `np.float16` in `Node.__init__` |

---

## Tier 1: High Impact, Easy to Implement ✓ (All Complete)

### 1. Dynamic c_puct ✓

**What:** c_puct grows logarithmically with parent visit count instead of staying fixed.

**Lc0 formula:**
```
c_puct = c_init + c_factor * log((N_parent + c_base) / c_base)
```

**Lc0 defaults:** `c_init=2.15`, `c_base=18368`, `c_factor=2.82`
**AlphaZero paper:** `c_init=2.5`, `c_base=19652`, `c_factor=2.0`

**Why:** Early search trusts the policy network more (lower effective c_puct). As nodes accumulate visits, exploration grows, forcing the engine to consider less-favored moves. Prevents both under-exploration early and over-exploration late.

**Where to change:** `training/mcts.py` (`MCTSConfig` + `puct_score`), `src/mcts/search.cpp` (`SearchParams` + `select_child`)

### 2. Policy Softmax Temperature ✓

**What:** Apply a temperature > 1.0 to NN policy logits before softmax, widening the prior distribution.

**Lc0 default:** `PolicyTemperature=1.61`

**Why:** Compensates for overconfident policy heads that would starve good-but-low-prior moves of visits. Especially important for smaller networks.

**Implementation:** In `_evaluate()`: `policy_probs = softmax(logits / policy_temperature)`. Add `policy_temperature: float = 1.61` to `MCTSConfig`.

**Where to change:** `training/mcts.py` (`_evaluate`), `src/neural/neural_evaluator.cpp`

### 3. Gradient Clipping ✓

**What:** Clip gradient global norm to prevent catastrophic spikes during training.

**Lc0:** Uses gradient clipping with a high threshold.

**Implementation:** Add `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)` before `optimizer.step()`.

**Where to change:** `training/selfplay.py` (training loop), `training/train.py`

### 4. LR Warmup ✓

**What:** Linearly ramp learning rate from 0 to target over first N steps.

**Lc0 default:** 250 warmup steps.

**Why:** Prevents early divergence when model weights are random and gradients are noisy.

**Implementation:** Use `torch.optim.lr_scheduler.LinearLR` for warmup, chained with the existing MultiStepLR via `SequentialLR`.

**Where to change:** `training/selfplay.py` (training loop), `training/train.py`

### 5. Mish Activation ✓

**What:** Drop-in replacement for ReLU: `mish(x) = x * tanh(softplus(x))`

**Lc0:** Default activation in recent networks (T78+).

**Why:** Smoother gradients than ReLU, avoids dying neurons. Consistent small Elo gain across network sizes.

**Implementation:** `F.mish(x)` in PyTorch (built-in). Replace `F.relu` calls in `ResidualBlock`, policy head, value head.

**Where to change:** `training/model.py`

---

## Tier 2: High Impact, Moderate Effort ✓ (All Complete)

### 6. Moves-Left Head (MLH) ✓

**What:** Auxiliary network output predicting remaining game plies.

**Architecture (from Lc0 source):**
```
body output → 1×1 conv(filters → 8) → BN → ReLU → flatten(512) → FC(128) → ReLU → FC(1) → ReLU
```

**Loss:** Huber loss (delta=10.0), scaled by 1/20. Target is actual remaining moves.

**Why:** Helps search prefer shorter winning lines. Improves time management. Auxiliary training signal.

**Where to change:** `training/model.py` (add head), `training/train.py` (add loss), `training/selfplay.py` (record remaining moves)

### 7. Q-Value Blending ✓

**What:** Blend game outcome with MCTS root Q-value for value targets.

**Formula:** `value_target = q_ratio * root_q + (1 - q_ratio) * game_result`

**Lc0:** Default `q_ratio=0` but production runs use 0.2–0.5.

**Why:** Reduces variance in value targets. Search Q-values are more informative than binary game outcomes, especially in long games with noise.

**Where to change:** `training/selfplay.py` (record root Q in `play_game`, blend in WDL labeling)

### 8. MCTS-Solver (Certainty Propagation) ✓

**What:** When a terminal node is reached (checkmate/stalemate), propagate the proven result up the tree.

**Rules:**
- If any child is a proven win for us → parent is a proven loss (for opponent)
- If all children are proven → parent is proven with best result
- Proven nodes get infinite visits (always selected/avoided appropriately)

**Why:** Solves shallow forced mates. Avoids exploring dead branches in endgames.

**Where to change:** `training/mcts.py` (add `terminal_status` to Node), `src/mcts/node.h` (same for C++)

### 9. Attention Policy Head ✓

**What:** Replace classical FC policy head with attention-based policy.

**Architecture (from Lc0 source):**
```
body output → reshape to 64 tokens (one per square)
→ dense embedding + SELU activation
→ optional encoder layer(s) (self-attention + FFN)
→ Q projection and K projection (dense layers)
→ policy logits = Q @ K^T (64×64 from-to matrix)
→ promotion: separate dense on last-rank keys → 4 offsets (Q/R/B/N)
→ map to 1858 via sparse constant matrix
```

**Why:** Biggest single architectural Elo gain. Understands piece interactions rather than treating policy as flat classification.

**Where to change:** `training/model.py` (new policy head class)

---

## Tier 3: Worth Doing Later ✓ (All Complete)

### 10. SGD + Nesterov Momentum ✓

**What:** Lc0 uses SGD (momentum=0.9, Nesterov=True) instead of Adam/AdamW.

**Why:** Better generalization for RL where data distribution shifts constantly. Adam's adaptive rates can overfit to stale patterns.

**Lc0 LR:** Starts at 0.02 (20× higher than our 1e-3), with step decay.

### 11. Tree Reuse ✓

**What:** After a move is played, make the chosen child the new root and discard the rest.

**Note:** Lc0 disables tree reuse during self-play training (independent root distributions) but enables it during play.

### 12. Diff Focus Sampling ✓

**What:** Oversample positions where MCTS search disagreed with the raw NN evaluation (policy or value changed significantly after search).

**Why:** Focuses training on informative positions rather than easy ones.

### 13. Stochastic Weight Averaging (SWA) ✓

**What:** Maintain a running average of weights, use the averaged model for self-play.

**Why:** Reduces noise, smoother self-play policy. Low cost with `torch.optim.swa_utils.AveragedModel`.

### 14. Glorot Normal Initialization ✓

**What:** Lc0 uses Xavier/Glorot normal initialization for all layers.

**Implementation:** `nn.init.xavier_normal_` on conv and FC weights.

---

## References

- [Lc0 Engine Parameters](https://lczero.org/play/flags/)
- [Technical Explanation of Leela Chess Zero](https://lczero.org/dev/wiki/technical-explanation-of-leela-chess-zero/)
- [AlphaZero Paper and Lc0](https://lczero.org/blog/2018/12/alphazero-paper-and-lc0-v0191/)
- [Lc0 GitHub — C++ engine](https://github.com/LeelaChessZero/lc0)
- [Lc0 GitHub — Training](https://github.com/LeelaChessZero/lczero-training)
- [MCTS Solver / Certainty Propagation](https://github.com/LeelaChessZero/lc0/pull/700)
- [Moves Left Head PR #961](https://github.com/LeelaChessZero/lc0/pull/961)
- [WDL Rescale/Contempt](https://lczero.org/blog/2023/07/the-lc0-v0.30.0-wdl-rescale/contempt-implementation/)
