"""BlockBlastBatchEnv ↔ SubprocVecEnv(BlockBlastEnv) parity test suite.

İki implementasyonun davranışsal eşitlikten ne kadar emin olabileceğimizi
ölçer. SubprocVec referans (single-env mantığı). BatchEnv test edilen.

Test 1: tek-env determinism (1000 step, 8 obs alanı + reward + done + mask)
Test 2: reward bileşen parity (step_bonus, line_clear, hole_delta, comp_delta, terminal)
Test 3: multi-env determinism (N=16, aynı action sequence)
Test 4: ep_length distribution (KS-test, p > 0.05)
Test 5: action_mask coverage (10000 random state, byte-eşitlik)

Mismatch'lerde ilk fark eden step + alan + beklenen vs bulunan + hipotez raporlanır.

NOT: SubprocVec sadece infos[i]["action_mask"] (BlockBlastEnv tarafından doldurulur)
ile mask alır; env_method('action_masks') de geçerli. BatchEnv `action_masks()`
property + env_method('action_masks') ikisini de destekler.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from stable_baselines3.common.vec_env import SubprocVecEnv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.env.batch_env import BlockBlastBatchEnv  # noqa: E402
from src.env.env import BlockBlastEnv  # noqa: E402

CFG_SURVIVAL = "configs/env.yaml"          # v4a sade
CFG_REWARD_SHAPING = "configs/env_v4b.yaml"  # hole + bumpiness + line_clear table


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sub_init(rank: int, cfg: str, seed: int):
    def _init():
        return BlockBlastEnv(config_path=cfg, seed=seed + rank)
    return _init


def make_subproc(n: int, cfg: str, seed: int) -> SubprocVecEnv:
    return SubprocVecEnv(
        [_sub_init(i, cfg, seed) for i in range(n)],
        start_method="forkserver",
    )


def get_masks(venv) -> np.ndarray:
    """Return (N, 192) bool. SubprocVec uses env_method, BatchEnv has direct property."""
    if hasattr(venv, "action_masks") and callable(venv.action_masks):
        return venv.action_masks()
    return np.stack(venv.env_method("action_masks"))


def first_field_mismatch(d_a: dict, d_b: dict, atol: float = 0.0) -> tuple[str, np.ndarray, np.ndarray] | None:
    """Return (field_name, val_a, val_b) of first mismatching obs field, else None."""
    for k in d_a.keys():
        a = np.asarray(d_a[k])
        b = np.asarray(d_b[k])
        if a.shape != b.shape:
            return (k, a, b)
        if a.dtype.kind == "f":
            if not np.allclose(a, b, atol=atol):
                return (k, a, b)
        else:
            if not np.array_equal(a, b):
                return (k, a, b)
    return None


def ks_2samp_manual(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sample Kolmogorov–Smirnov test (no scipy)."""
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    n_a, n_b = a.size, b.size
    all_v = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, all_v, side="right") / n_a
    cdf_b = np.searchsorted(b, all_v, side="right") / n_b
    D = float(np.abs(cdf_a - cdf_b).max())
    en = np.sqrt(n_a * n_b / (n_a + n_b))
    # Asymptotic Kolmogorov approximation
    lam = (en + 0.12 + 0.11 / en) * D
    j = np.arange(1, 101)
    p = 2.0 * float(np.sum(((-1.0) ** (j - 1)) * np.exp(-2.0 * (lam ** 2) * (j ** 2))))
    p = max(0.0, min(1.0, p))
    return D, p


def random_legal_actions(masks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Vectorized: argmax(uniform * mask). Same RNG draws on both sides → same actions."""
    n, a = masks.shape
    priority = rng.random((n, a)) * masks.astype(np.float32)
    return priority.argmax(axis=1).astype(np.int64)


def split_reward_components(env_cfg_rewards: dict, info: dict, reward: float, done: bool) -> dict:
    """Reconstruct reward components from a single-env info dict.

    Single-env reward formula (env.py):
        if invalid: r = r_invalid (terminal)
        else:
            r = r_step
            if game_over: r += r_game_over
            r += r_hole * holes_delta
            r += r_bumpiness * comps_delta
            r += line_clear_reward (from info)
    """
    r_step = float(env_cfg_rewards.get("step", 1.0))
    r_invalid = float(env_cfg_rewards.get("invalid_action", -1.0))
    r_game_over = float(env_cfg_rewards.get("game_over", -10.0))
    r_hole = float(env_cfg_rewards.get("hole_penalty", 0.0))
    r_bumpiness = float(env_cfg_rewards.get("bumpiness_penalty", 0.0))

    invalid = bool(info.get("invalid_action", False))
    if invalid:
        return {
            "step_bonus": 0.0,
            "line_clear": 0.0,
            "hole_delta": 0.0,
            "comp_delta": 0.0,
            "terminal_game_over": 0.0,
            "invalid_term": r_invalid,
        }

    holes_delta = int(info.get("holes_delta", 0))
    comps_delta = int(info.get("comps_delta", 0))
    line_clear = float(info.get("line_clear_reward", 0.0))
    game_over_now = bool(info.get("game_over", False))
    return {
        "step_bonus": r_step,
        "line_clear": line_clear,
        "hole_delta": r_hole * holes_delta,
        "comp_delta": r_bumpiness * comps_delta,
        "terminal_game_over": r_game_over if game_over_now else 0.0,
        "invalid_term": 0.0,
    }


# ---------------------------------------------------------------------------
# Test 1 — Tek env determinism (n=1, 1000 step, all obs fields + reward + done + mask)
# ---------------------------------------------------------------------------

def test_1_single_env_determinism():
    seed = 123
    n_steps = 1000
    cfg = CFG_SURVIVAL

    sub = make_subproc(1, cfg, seed)
    bat = BlockBlastBatchEnv(num_envs=1, config_path=cfg, seed=seed)

    try:
        obs_s = sub.reset()
        obs_b = bat.reset()

        # Initial-state field comparison (pre-step)
        m0 = first_field_mismatch(obs_s, obs_b, atol=1e-6)
        if m0 is not None:
            field, va, vb = m0
            pytest.fail(
                f"[Test 1] step=0 (post-reset, pre-step) mismatch in obs['{field}'].\n"
                f"  shapes: subproc={va.shape}, batch={vb.shape}\n"
                f"  HYPOTHESIS: tray refill RNG path farklı. Single-env: "
                f"Tray._sample_one() rng.choice(31, p=weights) per slot. "
                f"BatchEnv: rng.random(31)+argsort. Aynı seed altında farklı "
                f"piece type'ları seçiyor → tray, placements (action_mask reshape), "
                f"meta(n_legal_norm) hepsi farklı."
            )

        m_s = get_masks(sub)
        m_b = get_masks(bat)
        if not np.array_equal(m_s, m_b):
            pytest.fail(
                f"[Test 1] step=0 action_masks() farklı. legal: subproc={int(m_s.sum())}, "
                f"batch={int(m_b.sum())}, ilk diff index={int(np.where(m_s != m_b)[0][0])}\n"
                f"  HYPOTHESIS: tray refill RNG path farklı."
            )

        rng = np.random.default_rng(2024)
        for step in range(n_steps):
            m_s = get_masks(sub)
            m_b = get_masks(bat)
            if not np.array_equal(m_s, m_b):
                ix = int(np.where(m_s != m_b)[0][0])
                pytest.fail(
                    f"[Test 1] step={step} action_masks differ. ilk diff idx={ix}, "
                    f"sub={bool(m_s.flat[ix])} batch={bool(m_b.flat[ix])}"
                )
            actions = random_legal_actions(m_s, rng)

            obs_s, r_s, d_s, infos_s = sub.step(actions)
            obs_b, r_b, d_b, infos_b = bat.step(actions)

            mm = first_field_mismatch(obs_s, obs_b, atol=1e-6)
            if mm is not None:
                field, va, vb = mm
                pytest.fail(
                    f"[Test 1] step={step+1} obs['{field}'] differ.\n"
                    f"  subproc.shape={va.shape} batch.shape={vb.shape}\n"
                    f"  first diff idx={np.argwhere((va != vb))[0].tolist() if va.shape == vb.shape else 'shape'}"
                )
            if not np.allclose(r_s, r_b, atol=1e-6):
                pytest.fail(
                    f"[Test 1] step={step+1} reward differ: sub={r_s} batch={r_b}"
                )
            if not np.array_equal(d_s, d_b):
                pytest.fail(
                    f"[Test 1] step={step+1} done differ: sub={d_s} batch={d_b}"
                )
    finally:
        sub.close()
        bat.close()


# ---------------------------------------------------------------------------
# Test 2 — Reward bileşen parity (env_v4b ile hole/bumpiness/line_clear tetikli)
# ---------------------------------------------------------------------------

def test_2_reward_component_parity():
    seed = 7
    n_steps = 1000
    cfg = CFG_REWARD_SHAPING

    sub = make_subproc(1, cfg, seed)
    bat = BlockBlastBatchEnv(num_envs=1, config_path=cfg, seed=seed)

    try:
        sub.reset()
        bat.reset()
        # Read env config rewards (single-env yaml)
        import yaml
        with open(cfg) as f:
            env_cfg = yaml.safe_load(f)
        env_rewards = env_cfg.get("reward", {})

        rng = np.random.default_rng(99)
        # We can only compare reward COMPONENTS if we can derive them on both
        # sides. Single-env emits per-component info fields. BatchEnv currently
        # does NOT (its info is sparse). For Test 2, use SubprocVec mask to
        # drive both, and compare TOTAL reward + best-effort component split
        # using info on the subproc side and recomputing on the batch side.
        for step in range(n_steps):
            m_s = get_masks(sub)
            m_b = get_masks(bat)
            if not np.array_equal(m_s, m_b):
                # If masks already diverge, all subsequent reward analysis is
                # meaningless. Surface the divergence as the test 2 failure.
                pytest.fail(
                    f"[Test 2] step={step}: masks already diverge → "
                    f"reward bileşen karşılaştırması imkansız. "
                    f"HYPOTHESIS: tray refill RNG path farkı (Test 1 ile aynı kök sebep)."
                )
            actions = random_legal_actions(m_s, rng)

            _, r_s, d_s, infos_s = sub.step(actions)
            _, r_b, d_b, infos_b = bat.step(actions)

            if not np.allclose(r_s, r_b, atol=1e-6):
                # Inspect components from subproc side
                comp = split_reward_components(env_rewards, infos_s[0], float(r_s[0]), bool(d_s[0]))
                pytest.fail(
                    f"[Test 2] step={step+1}: total reward differs "
                    f"sub={float(r_s[0]):.4f} bat={float(r_b[0]):.4f}.\n"
                    f"  subproc components: {comp}\n"
                    f"  HYPOTHESIS: ya bileşen hesabı farklı (tablo lookup, "
                    f"hole_max_size, asymmetry kuralı), ya da prior-step tray "
                    f"divergence cumulative."
                )
    finally:
        sub.close()
        bat.close()


# ---------------------------------------------------------------------------
# Test 3 — Multi-env determinism (N=16)
# ---------------------------------------------------------------------------

def test_3_multi_env_determinism():
    seed = 555
    N = 16
    n_steps = 200  # 16 × 200 = 3200 env-step
    cfg = CFG_SURVIVAL

    sub = make_subproc(N, cfg, seed)
    bat = BlockBlastBatchEnv(num_envs=N, config_path=cfg, seed=seed)

    try:
        sub.reset()
        bat.reset()
        rng = np.random.default_rng(31337)

        for step in range(n_steps):
            m_s = get_masks(sub)
            m_b = get_masks(bat)
            if not np.array_equal(m_s, m_b):
                env_idx = int(np.argmax((m_s != m_b).any(axis=1)))
                idx = int(np.where(m_s[env_idx] != m_b[env_idx])[0][0])
                pytest.fail(
                    f"[Test 3] step={step} env={env_idx} mask diff. legal: "
                    f"sub={int(m_s[env_idx].sum())} bat={int(m_b[env_idx].sum())}, "
                    f"ilk diff idx={idx}\n"
                    f"  HYPOTHESIS: tray refill RNG farkı (her env için unique seed "
                    f"BatchEnv'de tek shared RNG ile gerçeklenmiş; subproc her worker "
                    f"farklı per-env RNG'ye sahip)."
                )
            actions = random_legal_actions(m_s, rng)
            obs_s, r_s, d_s, _ = sub.step(actions)
            obs_b, r_b, d_b, _ = bat.step(actions)

            if not np.allclose(r_s, r_b, atol=1e-6):
                env_idx = int(np.argmax(np.abs(r_s - r_b) > 1e-6))
                pytest.fail(
                    f"[Test 3] step={step+1} env={env_idx} reward diff: "
                    f"sub={r_s[env_idx]:.4f} bat={r_b[env_idx]:.4f}"
                )
            if not np.array_equal(d_s, d_b):
                env_idx = int(np.argmax(d_s != d_b))
                pytest.fail(
                    f"[Test 3] step={step+1} env={env_idx} done diff"
                )
    finally:
        sub.close()
        bat.close()


# ---------------------------------------------------------------------------
# Test 4 — Episode length distribution (KS-test, random policy, 500 episode)
# ---------------------------------------------------------------------------

def _collect_episode_lengths(venv, n_episodes: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    venv.reset()
    lengths: list[int] = []
    cur_len = np.zeros(venv.num_envs, dtype=np.int32)
    while len(lengths) < n_episodes:
        masks = get_masks(venv)
        actions = random_legal_actions(masks, rng)
        _, _, dones, _ = venv.step(actions)
        cur_len += 1
        for i in range(venv.num_envs):
            if dones[i]:
                lengths.append(int(cur_len[i]))
                cur_len[i] = 0
    return lengths[:n_episodes]


def test_4_episode_length_distribution():
    seed = 4242
    N = 8
    n_eps = 500
    cfg = CFG_SURVIVAL

    sub = make_subproc(N, cfg, seed)
    bat = BlockBlastBatchEnv(num_envs=N, config_path=cfg, seed=seed)
    try:
        lens_s = _collect_episode_lengths(sub, n_eps, seed=1)
        lens_b = _collect_episode_lengths(bat, n_eps, seed=1)

        D, p = ks_2samp_manual(np.asarray(lens_s), np.asarray(lens_b))
        mean_s = float(np.mean(lens_s))
        mean_b = float(np.mean(lens_b))
        std_s = float(np.std(lens_s))
        std_b = float(np.std(lens_b))
        print(
            f"[Test 4] ep_len  sub: mean={mean_s:.2f} std={std_s:.2f}  "
            f"bat: mean={mean_b:.2f} std={std_b:.2f}  KS D={D:.4f} p={p:.4f}"
        )
        assert p > 0.05, (
            f"[Test 4] ep_length dağılımları farklı: KS D={D:.4f} p={p:.4f} "
            f"(sub mean={mean_s:.2f} bat mean={mean_b:.2f}). "
            f"HYPOTHESIS: tray refill stratejisi farkı dağılımsal olarak da "
            f"farklı; weights normalize / argsort_unique birinde same_tray_unique "
            f"semantic'i uygulanmıyor olabilir."
        )
    finally:
        sub.close()
        bat.close()


# ---------------------------------------------------------------------------
# Test 5 — Action mask coverage (10000 random state)
# ---------------------------------------------------------------------------

def test_5_action_mask_coverage():
    """10000 (board, tray) state'inde action_mask byte-by-byte eşit mi?

    State'i her iki env'e doğrudan inject ederek RNG-bağımsız bir karşılaştırma
    yapar. Bu test tray refill stratejisi farkından etkilenmez — sadece
    placement_mask + masking semantic'i test eder.
    """
    cfg = CFG_SURVIVAL
    n_states = 10000

    rng = np.random.default_rng(2025)

    # Single-env tarafı (referans)
    single = BlockBlastEnv(config_path=cfg, seed=0)
    # Batch tarafı (test edilen)
    batch = BlockBlastBatchEnv(num_envs=1, config_path=cfg, seed=0)

    # Reuse single-env piece set & build action mask
    from src.env.masking import build_action_mask  # noqa: PLC0415

    mismatches = 0
    first_mismatch = None
    for k in range(n_states):
        # Random board fill density: 5% to 70%
        density = float(rng.uniform(0.05, 0.70))
        board = (rng.random((8, 8)) < density)
        # Random tray: 3 unique piece types from 0..30
        tray_idx = rng.choice(31, size=3, replace=False).astype(np.int8)

        # Inject into single
        single.game.board.grid = board.copy()
        single.game.tray.slots = [single.game.piece_set[i] for i in tray_idx]
        single.game.game_over = False
        m_s = build_action_mask(single.game)

        # Inject into batch
        batch.boards[0] = board.copy()
        batch.tray_idx[0] = tray_idx
        batch.game_over[0] = False
        m_b = batch._build_action_mask()[0]

        if not np.array_equal(m_s, m_b):
            mismatches += 1
            if first_mismatch is None:
                first_mismatch = (k, board.copy(), tray_idx.copy(),
                                  m_s.copy(), m_b.copy())

    if mismatches > 0:
        k, board, tray_idx, m_s, m_b = first_mismatch
        diff_idx = int(np.where(m_s != m_b)[0][0])
        piece_slot, rem = divmod(diff_idx, 64)
        y, x = divmod(rem, 8)
        pytest.fail(
            f"[Test 5] {mismatches}/{n_states} state'te action_mask farkı. "
            f"İlk mismatch state #{k}: tray={tray_idx.tolist()} fill={int(board.sum())}/64, "
            f"diff at action_id={diff_idx} (slot={piece_slot}, y={y}, x={x}), "
            f"sub={bool(m_s[diff_idx])} bat={bool(m_b[diff_idx])}\n"
            f"  HYPOTHESIS: placement_mask boundary handling — pad-true "
            f"sliding window y/x sınır kontrolü single-env'in (h,w) "
            f"check'inden farklı bir yerde sapıyor olabilir."
        )
    print(f"[Test 5] action_mask coverage: 0/{n_states} mismatches")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
