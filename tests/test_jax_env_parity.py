"""Parity tests: numpy BatchEnv ↔ JAX env.

Test 4 — KS distribution test (random policy, 500 episodes, ep_length CDF
eşitliği pencerelerde).
Test 5 — Action mask byte-equivalence (10K random state injection, CPU mode'da
jax.numpy ≡ numpy bit-identical placement_mask).

Test 5'te JAX CPU mode kullanılır (parity için float64 / float32 farkı yok,
boolean/int tamamen deterministik). Refill RNG path farkı kabul edildi →
state injection'da tray sabit tutulur, refill tetiklenmez.
"""

from __future__ import annotations

import os
# CPU mode for parity (deterministic; bool/int byte-identical with numpy)
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from src.env.batch_env import BlockBlastBatchEnv
from src.env.encoding import (
    BOARD_SIZE,
    HISTORY_LEN,
    HISTORY_SENTINEL,
    NUM_ACTIONS,
    NUM_PIECE_TYPES,
    TRAY_SIZE,
)
from src.env.jax_env import JaxState, build_action_mask, _build_piece_tensor
from src.env.jax_vec_wrapper import JaxBlockBlastVecEnv


CFG = "configs/env.yaml"
PIECES = "configs/pieces.yaml"


# ---------------------------------------------------------------------------
# Test 5 — action_mask byte-equivalence (10K random state injection)
# ---------------------------------------------------------------------------

def test_action_mask_byte_equivalent_to_numpy_env():
    """10K random (board, tray) state için JAX build_action_mask vs numpy
    BatchEnv._build_action_mask byte-identical."""
    rng = np.random.default_rng(20260506)
    N = 10_000

    # Random boards: ~30% fill (mix of empty / sparse / dense)
    fill = rng.uniform(0.0, 0.6, size=N)
    boards_np = (rng.random((N, BOARD_SIZE, BOARD_SIZE)) < fill[:, None, None]).astype(bool)

    # Random tray: 3 unique pieces per env (default same_tray_unique=True logic)
    tray_np = np.zeros((N, TRAY_SIZE), dtype=np.int8)
    for i in range(N):
        tray_np[i] = rng.choice(NUM_PIECE_TYPES, size=TRAY_SIZE, replace=False).astype(np.int8)

    # Some envs: empty slot to test the (-1) handling
    # First 200 envs: middle slot empty
    tray_np[:200, 1] = -1
    # Next 100 envs: all slots empty (game over scenario)
    tray_np[200:300] = -1

    game_over_np = np.zeros(N, dtype=bool)
    # Last 50: simulate game_over → mask should be all False
    game_over_np[-50:] = True

    # Numpy reference: BatchEnv._build_action_mask (state injection)
    np_env = BlockBlastBatchEnv(num_envs=N, config_path=CFG, seed=0)
    np_env.boards = boards_np
    np_env.tray_idx = tray_np
    np_env.game_over = game_over_np
    mask_np = np_env._build_action_mask()  # (N, 192) bool

    # JAX implementation
    pcp = _build_piece_tensor(PIECES)
    boards_j = jnp.asarray(boards_np, dtype=bool)
    tray_j = jnp.asarray(tray_np, dtype=jnp.int8)
    go_j = jnp.asarray(game_over_np, dtype=bool)
    mask_j = build_action_mask(boards_j, tray_j, go_j, pcp)
    mask_j_np = np.asarray(mask_j, dtype=bool)

    # Byte-identical
    assert mask_np.shape == mask_j_np.shape == (N, NUM_ACTIONS)
    diff = mask_np != mask_j_np
    n_diff = int(diff.sum())
    if n_diff > 0:
        # Diagnostic
        idx = np.argwhere(diff)[:5]
        print(f"\n[FAIL] {n_diff} cell mismatches; first 5 (env, action): {idx.tolist()}")
    assert n_diff == 0, f"action_mask byte mismatch: {n_diff} cells differ"


# ---------------------------------------------------------------------------
# Test 4 — KS distribution test (random policy, 500 episodes)
# ---------------------------------------------------------------------------

def _random_masked_action(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Per-env: uniform sample over valid actions; if no valid, return 0."""
    N, A = mask.shape
    # Gumbel-max trick over masked logits
    g = rng.standard_normal((N, A))
    g = np.where(mask, g, -np.inf)
    # If a row is fully masked (game-over carried), pick 0 (won't matter, env is_done)
    no_legal = ~mask.any(axis=1)
    actions = np.where(no_legal, 0, np.argmax(g, axis=1))
    return actions.astype(np.int64)


def _collect_episode_lengths(venv, n_episodes: int, rng: np.random.Generator) -> list[int]:
    """Random masked policy with given env (numpy or JAX VecEnv). Collect first
    n_episodes completed episode lengths."""
    obs = venv.reset()
    lengths: list[int] = []
    while len(lengths) < n_episodes:
        mask = venv.action_masks()
        actions = _random_masked_action(mask, rng)
        venv.step_async(actions)
        obs, reward, done, infos = venv.step_wait()
        for info in infos:
            ep = info.get("episode")
            if ep is not None:
                lengths.append(int(ep["l"]))
                if len(lengths) >= n_episodes:
                    break
    return lengths


def _ks_2samp(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Manual two-sample KS (no scipy)."""
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    n_a, n_b = a.size, b.size
    if n_a == 0 or n_b == 0:
        return 1.0, 0.0
    all_v = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, all_v, side="right") / n_a
    cdf_b = np.searchsorted(b, all_v, side="right") / n_b
    D = float(np.abs(cdf_a - cdf_b).max())
    en = np.sqrt(n_a * n_b / (n_a + n_b))
    lam = (en + 0.12 + 0.11 / en) * D
    j = np.arange(1, 101)
    p = 2.0 * float(np.sum(((-1.0) ** (j - 1)) * np.exp(-2.0 * (lam ** 2) * (j ** 2))))
    return D, max(0.0, min(1.0, p))


def test_ks_distribution_matches_numpy_env():
    """500 episode random policy. ep_length CDF KS-test 100-ep window'larda
    ≥ %80 PASS (p > 0.05) ve |Δmean(final 100)| < 1.5."""
    n_ep = 500
    np_env = BlockBlastBatchEnv(num_envs=64, config_path=CFG, seed=42)
    rng = np.random.default_rng(123)
    lens_np = _collect_episode_lengths(np_env, n_ep, rng)

    jx_env = JaxBlockBlastVecEnv(num_envs=64, config_path=CFG, seed=42)
    rng2 = np.random.default_rng(123)
    lens_jx = _collect_episode_lengths(jx_env, n_ep, rng2)

    a = np.asarray(lens_np[:n_ep], dtype=np.int32)
    b = np.asarray(lens_jx[:n_ep], dtype=np.int32)

    # Window-based KS over chunks of 50 episodes
    win = 50
    pass_n = 0
    total = 0
    for i in range(0, min(a.size, b.size), win):
        chunk_a = a[i : i + win]
        chunk_b = b[i : i + win]
        if chunk_a.size < 30 or chunk_b.size < 30:
            continue
        D, p = _ks_2samp(chunk_a, chunk_b)
        total += 1
        if p > 0.05:
            pass_n += 1
    pct = pass_n / max(1, total)
    print(f"\n[KS] window pass rate: {pass_n}/{total} ({pct:.0%})")

    # Final 100 ep mean diff
    diff = float(abs(a[-100:].mean() - b[-100:].mean()))
    print(f"[KS] |Δmean final 100|: {diff:.2f}")
    print(f"[KS] mean(np)={a.mean():.2f}  mean(jx)={b.mean():.2f}")

    assert pct >= 0.80, f"KS window pass rate {pct:.0%} < 80%"
    assert diff < 1.5, f"|Δmean| {diff:.2f} >= 1.5"
