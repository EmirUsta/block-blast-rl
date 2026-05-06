"""GPU-resident, JAX-backed Block Blast vectorized env.

Davranış `src/env/batch_env.py` ile birebir aynıdır; tek fark tray refill
RNG path'i (numpy `np.random.default_rng().random` vs `jax.random.uniform`).
Boolean / integer state alanları CPU-mode'da byte-identical, float alanlar
1e-5 toleransta eşit. Parity testleri `tests/test_jax_env_parity.py`'de.

Kontrat:
  - Pure functional step kernel (jit + pytree state).
  - Auto-reset + terminal_observation: pre-reset obs ve post-reset obs ayrı
    döndürülür; SB3 wrapper terminal_obs'u info'ya host'ta enjekte eder.
  - PRNG: tek master key state'te, her step'te split.
  - Reward shaping (hole/component): build-time flag (`use_hole_penalty`,
    `use_bumpiness`). v4a default False → BFS kodu hiç compile edilmez.

Sliding window mask hesabı: 12×12 pad-true scratch + fancy indexing
(N,8,8,5,5) gather. Piece OR-scatter: 12×12 pad-false scratch +
`lax.dynamic_update_slice` per-env vmap.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import NamedTuple

import flax.struct
import jax
import jax.numpy as jnp
import yaml
from jax import lax

from src.env.encoding import (
    BOARD_SIZE,
    FRAME_STACK,
    HISTORY_LEN,
    HISTORY_SENTINEL,
    MAX_PIECE_DIM,
    META_DIM,
    NUM_ACTIONS,
    NUM_PIECE_TYPES,
    TRAY_SIZE,
)
from src.game.pieces import load_piece_set


# ---------------------------------------------------------------------------
# Static config — implementer notu: hashable NamedTuple, jit static arg olarak
# geçirilir. Build-time bilinen flag'ler (use_hole_penalty, use_bumpiness)
# Python `if` ile compile-edilmemiş code path bypass için kullanılır.
# ---------------------------------------------------------------------------

class JaxEnvCfg(NamedTuple):
    r_step: float
    r_game_over: float
    r_invalid: float
    r_hole: float
    r_bumpiness: float
    r_line_base: float
    r_line_exp: float
    hole_max_size: int
    max_steps: int
    same_tray_unique: bool
    use_hole_penalty: bool
    use_bumpiness: bool
    use_line_clear: bool


@flax.struct.dataclass
class JaxState:
    """Tüm batched state. flax struct → pytree-native, jit-compatible."""
    boards: jnp.ndarray           # (N, 8, 8) bool
    frame_buf: jnp.ndarray        # (N, 4, 8, 8) float32
    tray_idx: jnp.ndarray         # (N, 3) int8 — empty slot = -1
    history: jnp.ndarray          # (N, 8) int32 (obs'a int64 cast)
    step_count: jnp.ndarray       # (N,) int32
    score: jnp.ndarray            # (N,) int32 — info-only
    game_over: jnp.ndarray        # (N,) bool
    episode_returns: jnp.ndarray  # (N,) float32
    episode_lengths: jnp.ndarray  # (N,) int32
    cached_mask: jnp.ndarray      # (N, 192) bool
    master_key: jnp.ndarray       # (2,) uint32 — TEK master key


# ---------------------------------------------------------------------------
# Piece tensor build (host-side, runtime sabit)
# ---------------------------------------------------------------------------

def _build_piece_tensor(pieces_path: str | Path) -> jnp.ndarray:
    """Pre-baked (31, 5, 5) bool padded piece cells. CPU'da inşa edilip device'a aktarılır."""
    piece_set = load_piece_set(Path(pieces_path))
    if len(piece_set) != NUM_PIECE_TYPES:
        raise ValueError(f"piece_set size {len(piece_set)} != {NUM_PIECE_TYPES}")
    import numpy as np
    pcp = np.zeros((NUM_PIECE_TYPES, MAX_PIECE_DIM, MAX_PIECE_DIM), dtype=bool)
    for p in piece_set:
        pcp[p.type_idx, : p.height, : p.width] = p.cells
    return jnp.asarray(pcp, dtype=bool)


def cfg_from_yaml(config_path: str | Path) -> JaxEnvCfg:
    """env.yaml → JaxEnvCfg. v4a default'unda use_* flag'leri tamamen False."""
    with open(config_path, "r", encoding="utf-8") as f:
        c = yaml.safe_load(f)
    rw = c.get("reward", {})
    r_hole = float(rw.get("hole_penalty", 0.0))
    r_bump = float(rw.get("bumpiness_penalty", 0.0))
    r_lb = float(rw.get("line_clear_base", 0.0))
    return JaxEnvCfg(
        r_step=float(rw.get("step", 1.0)),
        r_game_over=float(rw.get("game_over", -10.0)),
        r_invalid=float(rw.get("invalid_action", -1.0)),
        r_hole=r_hole,
        r_bumpiness=r_bump,
        r_line_base=r_lb,
        r_line_exp=float(rw.get("line_clear_exp", 3.0)),
        hole_max_size=int(rw.get("hole_max_size", 2)),
        max_steps=int(c["episode"]["max_steps"]),
        same_tray_unique=bool(c["tray"].get("same_tray_unique", True)),
        use_hole_penalty=(r_hole != 0.0),
        use_bumpiness=(r_bump != 0.0),
        use_line_clear=(r_lb != 0.0 or rw.get("line_clear_table") is not None),
    )


# ---------------------------------------------------------------------------
# Sliding window action mask
# ---------------------------------------------------------------------------

# Static gather grids for sliding-(5,5) over 8×8 valid origins on a 12×12 pad.
# yy[y, dy] = y + dy  (y in [0,7], dy in [0,4])
# xx[x, dx] = x + dx  (x in [0,7], dx in [0,4])
_YY = (jnp.arange(BOARD_SIZE)[:, None] + jnp.arange(MAX_PIECE_DIM)[None, :])  # (8,5)
_XX = (jnp.arange(BOARD_SIZE)[:, None] + jnp.arange(MAX_PIECE_DIM)[None, :])  # (8,5)


def build_action_mask(
    boards: jnp.ndarray,
    tray_idx: jnp.ndarray,
    game_over: jnp.ndarray,
    piece_cells_pad: jnp.ndarray,
) -> jnp.ndarray:
    """(N,192) bool. Pad-true 12x12 + fancy gather (N,8,8,5,5) windows."""
    N = boards.shape[0]
    pad_size = BOARD_SIZE + (MAX_PIECE_DIM - 1)  # 12
    pad = jnp.ones((N, pad_size, pad_size), dtype=bool)
    pad = pad.at[:, :BOARD_SIZE, :BOARD_SIZE].set(boards)
    # Fancy index: pad[:, yy[:, None, :, None], xx[None, :, None, :]]
    # → (N, 8, 8, 5, 5)
    windows = pad[:, _YY[:, None, :, None], _XX[None, :, None, :]]

    safe_tray = jnp.where(tray_idx >= 0, tray_idx, 0).astype(jnp.int32)
    tray_pieces = piece_cells_pad[safe_tray]  # (N, 3, 5, 5)
    # conflict: (N, 1, 8, 8, 5, 5) & (N, 3, 1, 1, 5, 5) → reduce → (N, 3, 8, 8)
    conflict = (windows[:, None] & tray_pieces[:, :, None, None]).any(axis=(-1, -2))
    per_slot = ~conflict
    per_slot = per_slot & (tray_idx >= 0)[:, :, None, None]
    per_slot = per_slot & (~game_over)[:, None, None, None]
    return per_slot.reshape(N, NUM_ACTIONS)


# ---------------------------------------------------------------------------
# Place: OR piece cells onto each env's board at (y, x). 12×12 pad scratch
# (pad=False) so y+5 / x+5 spill into pad area is harmless.
# ---------------------------------------------------------------------------

def _place_one(board: jnp.ndarray, cells: jnp.ndarray, py: jnp.ndarray, px: jnp.ndarray) -> jnp.ndarray:
    """Single env: OR `cells` (5,5) into `board` (8,8) at (py, px). Returns (8,8) bool."""
    pad_size = BOARD_SIZE + (MAX_PIECE_DIM - 1)
    pad = jnp.zeros((pad_size, pad_size), dtype=bool)
    pad = pad.at[:BOARD_SIZE, :BOARD_SIZE].set(board)
    region = lax.dynamic_slice(pad, (py, px), (MAX_PIECE_DIM, MAX_PIECE_DIM))
    pad = lax.dynamic_update_slice(pad, region | cells, (py, px))
    return pad[:BOARD_SIZE, :BOARD_SIZE]


_place_batch = jax.vmap(_place_one, in_axes=(0, 0, 0, 0))


# ---------------------------------------------------------------------------
# Tray refill (RNG path — numpy ile farklı, dağılım eşit)
# ---------------------------------------------------------------------------

def _refill_one_unique(key: jnp.ndarray) -> jnp.ndarray:
    """argsort(uniform(31))[:3] — 3 unique piece type, int8."""
    u = jax.random.uniform(key, (NUM_PIECE_TYPES,))
    # argsort ascending; ilk 3 = en küçük 3 → permütasyondan ilk 3 unique pick.
    return jnp.argsort(u)[:TRAY_SIZE].astype(jnp.int8)


def _refill_one_dup(key: jnp.ndarray) -> jnp.ndarray:
    """3 independent piece type (duplicates allowed), int8."""
    return jax.random.randint(key, (TRAY_SIZE,), 0, NUM_PIECE_TYPES).astype(jnp.int8)


def refill_empty_trays(
    tray_idx: jnp.ndarray,
    master_key: jnp.ndarray,
    same_tray_unique: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """tray_idx all -1 olan env'lere 3 piece doldur. Diğerleri değişmez.

    Returns (new_tray_idx, new_master_key).
    """
    N = tray_idx.shape[0]
    master_key, sub = jax.random.split(master_key)
    per_env = jax.random.split(sub, N)
    if same_tray_unique:
        picks = jax.vmap(_refill_one_unique)(per_env)  # (N, 3) int8
    else:
        picks = jax.vmap(_refill_one_dup)(per_env)  # (N, 3) int8
    all_empty = (tray_idx == -1).all(axis=1)  # (N,)
    new_tray = jnp.where(all_empty[:, None], picks, tray_idx)
    return new_tray, master_key


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

def build_obs(
    boards: jnp.ndarray,
    frame_buf: jnp.ndarray,
    cached_mask: jnp.ndarray,
    tray_idx: jnp.ndarray,
    history: jnp.ndarray,
    step_count: jnp.ndarray,
    piece_cells_pad: jnp.ndarray,
    max_steps: int,
) -> dict:
    """SB3 Dict observation. `history` int32 — adapter int64'e cast."""
    N = boards.shape[0]
    placements = cached_mask.reshape(N, TRAY_SIZE, BOARD_SIZE, BOARD_SIZE).astype(jnp.float32)
    safe_tray = jnp.where(tray_idx >= 0, tray_idx, 0).astype(jnp.int32)
    tray_one_hot = jnp.eye(NUM_PIECE_TYPES, dtype=jnp.float32)[safe_tray]
    empty = (tray_idx < 0)[:, :, None]
    tray_one_hot = jnp.where(empty, 0.0, tray_one_hot)
    tray_shapes = piece_cells_pad[safe_tray].astype(jnp.float32)
    tray_shapes = jnp.where(empty[:, :, :, None], 0.0, tray_shapes)
    steps_norm = jnp.minimum(1.0, step_count.astype(jnp.float32) / float(max(1, max_steps)))
    fill_ratio = boards.sum(axis=(1, 2)).astype(jnp.float32) / float(BOARD_SIZE * BOARD_SIZE)
    n_legal = cached_mask.sum(axis=1).astype(jnp.float32)
    n_legal_norm = n_legal / float(NUM_ACTIONS)
    slots_norm = (tray_idx >= 0).sum(axis=1).astype(jnp.float32) / float(TRAY_SIZE)
    meta = jnp.stack([steps_norm, fill_ratio, n_legal_norm, slots_norm], axis=1)
    return {
        "grid": frame_buf,
        "placements": placements,
        "tray": tray_one_hot,
        "tray_shapes": tray_shapes,
        "history": history,  # int32, adapter int64'e cast eder
        "meta": meta,
    }


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

def initial_state(num_envs: int, master_key: jnp.ndarray, cfg: JaxEnvCfg, piece_cells_pad: jnp.ndarray) -> JaxState:
    """Tüm env'ler temiz: board zeros, frame zeros, history=192, tray refill."""
    N = num_envs
    boards = jnp.zeros((N, BOARD_SIZE, BOARD_SIZE), dtype=bool)
    frame_buf = jnp.zeros((N, FRAME_STACK, BOARD_SIZE, BOARD_SIZE), dtype=jnp.float32)
    tray_idx = jnp.full((N, TRAY_SIZE), -1, dtype=jnp.int8)
    history = jnp.full((N, HISTORY_LEN), HISTORY_SENTINEL, dtype=jnp.int32)
    step_count = jnp.zeros(N, dtype=jnp.int32)
    score = jnp.zeros(N, dtype=jnp.int32)
    game_over = jnp.zeros(N, dtype=bool)
    ep_returns = jnp.zeros(N, dtype=jnp.float32)
    ep_lengths = jnp.zeros(N, dtype=jnp.int32)
    tray_idx, master_key = refill_empty_trays(tray_idx, master_key, cfg.same_tray_unique)
    cached_mask = build_action_mask(boards, tray_idx, game_over, piece_cells_pad)
    return JaxState(
        boards=boards,
        frame_buf=frame_buf,
        tray_idx=tray_idx,
        history=history,
        step_count=step_count,
        score=score,
        game_over=game_over,
        episode_returns=ep_returns,
        episode_lengths=ep_lengths,
        cached_mask=cached_mask,
        master_key=master_key,
    )


# ---------------------------------------------------------------------------
# Step kernel — pure functional, jit'lenecek
# ---------------------------------------------------------------------------

def _step_kernel(
    state: JaxState,
    actions: jnp.ndarray,
    cfg: JaxEnvCfg,
    piece_cells_pad: jnp.ndarray,
):
    """Numpy env step_wait'in pure functional ve vectorized JAX karşılığı.

    Sıra: validity check → place (valid only) → line clear (valid only) →
    tray take → tray refill → mask refresh → invalid→game_over → no-legal
    game_over → step_count/history/frame push (valid only) → reward → ep
    stats → done (terminate ∪ truncate) → reset → final mask & obs.

    Returns:
        new_state, post_reset_obs, pre_reset_obs, reward, done, terminated,
        truncated, ep_returns_snapshot, ep_lengths_snapshot
    """
    N = state.boards.shape[0]
    valid = state.cached_mask[jnp.arange(N), actions]  # (N,) bool

    # Decode action
    piece_slot = actions // (BOARD_SIZE * BOARD_SIZE)
    rem = actions % (BOARD_SIZE * BOARD_SIZE)
    y = (rem // BOARD_SIZE).astype(jnp.int32)
    x = (rem % BOARD_SIZE).astype(jnp.int32)

    # Piece type for chosen slot (may be -1 for empty slot — invalid action zaten)
    piece_type = state.tray_idx[jnp.arange(N), piece_slot]
    safe_pt = jnp.where(piece_type >= 0, piece_type, 0).astype(jnp.int32)
    cells = piece_cells_pad[safe_pt]  # (N, 5, 5) bool

    # Place (compute for all, then where(valid) to keep)
    placed = _place_batch(state.boards, cells, y, x)  # (N, 8, 8)
    boards1 = jnp.where(valid[:, None, None], placed, state.boards)

    # Line clear (compute for all, then where(valid))
    full_rows = boards1.all(axis=2)  # (N, 8)
    full_cols = boards1.all(axis=1)  # (N, 8)
    clear_mask = full_rows[:, :, None] | full_cols[:, None, :]
    cleared = boards1 & ~clear_mask
    boards2 = jnp.where(valid[:, None, None], cleared, state.boards)

    # n_cleared per env (used for line_clear reward only if enabled)
    n_cleared = (full_rows.sum(axis=1) + full_cols.sum(axis=1)).astype(jnp.int32)
    n_cleared = jnp.where(valid, n_cleared, jnp.int32(0))

    # Take piece from tray (slot → -1) for valid envs
    new_slot = jnp.where(valid, jnp.int8(-1), piece_type)
    tray1 = state.tray_idx.at[jnp.arange(N), piece_slot].set(new_slot)

    # Refill empty trays
    tray2, master_key1 = refill_empty_trays(tray1, state.master_key, cfg.same_tray_unique)

    # Invalid → immediate game_over
    game_over1 = state.game_over | ~valid

    # Recompute mask (post-place + post-clear + post-refill)
    mask_post_step = build_action_mask(boards2, tray2, game_over1, piece_cells_pad)

    # Reward: r_step (valid) ya da r_invalid (~valid)
    reward = jnp.where(valid, jnp.float32(cfg.r_step), jnp.float32(cfg.r_invalid))

    # Optional line clear reward (build-time flag → compile dışında bypass)
    if cfg.use_line_clear and cfg.r_line_base != 0.0:
        non_zero = n_cleared > 0
        lc = jnp.where(
            non_zero,
            jnp.float32(cfg.r_line_base) * (jnp.float32(cfg.r_line_exp) ** jnp.maximum(0, n_cleared - 1).astype(jnp.float32)),
            jnp.float32(0.0),
        )
        reward = reward + lc

    # No-legal → terminate (yalnızca daha önce game_over olmayanlar için)
    no_legal = ~mask_post_step.any(axis=1)
    new_go = no_legal & ~game_over1
    reward = reward + jnp.where(new_go, jnp.float32(cfg.r_game_over), jnp.float32(0.0))
    game_over2 = game_over1 | new_go

    # step_count / history / frame_buf push (valid only)
    step_count1 = jnp.where(valid, state.step_count + 1, state.step_count)
    history_pushed = jnp.concatenate(
        [state.history[:, 1:], actions.astype(jnp.int32)[:, None]], axis=1
    )
    history1 = jnp.where(valid[:, None], history_pushed, state.history)
    frame_pushed = jnp.concatenate(
        [state.frame_buf[:, 1:], boards2.astype(jnp.float32)[:, None]], axis=1
    )
    frame_buf1 = jnp.where(valid[:, None, None, None], frame_pushed, state.frame_buf)

    # Truncation
    truncated = step_count1 >= cfg.max_steps
    terminated = game_over2
    done = terminated | truncated

    # Episode stats: pre-reset değer (info["episode"] için)
    ep_returns_pre = state.episode_returns + reward
    ep_lengths_pre = state.episode_lengths + jnp.int32(1)  # numpy env: tüm env'lerde +1

    # Pre-reset obs (terminal_observation için): mask_post_step kullan
    # Numpy env terminal_obs'ı reset'ten ÖNCE inşa ediyor (line 553-554).
    pre_reset_obs = build_obs(
        boards=boards2,
        frame_buf=frame_buf1,
        cached_mask=mask_post_step,
        tray_idx=tray2,
        history=history1,
        step_count=step_count1,
        piece_cells_pad=piece_cells_pad,
        max_steps=cfg.max_steps,
    )

    # Reset done envs
    boards_r = jnp.where(done[:, None, None], jnp.zeros_like(boards2), boards2)
    frame_r = jnp.where(done[:, None, None, None], jnp.zeros_like(frame_buf1), frame_buf1)
    tray_r = jnp.where(done[:, None], jnp.full_like(tray2, -1), tray2)
    history_r = jnp.where(done[:, None], jnp.full_like(history1, HISTORY_SENTINEL), history1)
    step_r = jnp.where(done, jnp.zeros_like(step_count1), step_count1)
    score_r = jnp.where(done, jnp.zeros_like(state.score), state.score)
    go_r = jnp.where(done, jnp.zeros_like(game_over2), game_over2)
    ep_returns_post = jnp.where(done, jnp.zeros_like(ep_returns_pre), ep_returns_pre)
    ep_lengths_post = jnp.where(done, jnp.zeros_like(ep_lengths_pre), ep_lengths_pre)

    # Refill done env'lerin tray'ini (artık tümü -1)
    tray_final, master_key2 = refill_empty_trays(tray_r, master_key1, cfg.same_tray_unique)

    # Final mask + post-reset obs
    final_mask = build_action_mask(boards_r, tray_final, go_r, piece_cells_pad)
    post_reset_obs = build_obs(
        boards=boards_r,
        frame_buf=frame_r,
        cached_mask=final_mask,
        tray_idx=tray_final,
        history=history_r,
        step_count=step_r,
        piece_cells_pad=piece_cells_pad,
        max_steps=cfg.max_steps,
    )

    new_state = state.replace(
        boards=boards_r,
        frame_buf=frame_r,
        tray_idx=tray_final,
        history=history_r,
        step_count=step_r,
        score=score_r,
        game_over=go_r,
        episode_returns=ep_returns_post,
        episode_lengths=ep_lengths_post,
        cached_mask=final_mask,
        master_key=master_key2,
    )

    return (
        new_state,
        post_reset_obs,
        pre_reset_obs,
        reward.astype(jnp.float32),
        done,
        terminated,
        truncated,
        ep_returns_pre,
        ep_lengths_pre,
    )


# ---------------------------------------------------------------------------
# JIT'lenmiş public step. cfg & piece_cells_pad static (cfg hashable, piece
# cells_pad fonksiyon kapanışında sabit). num_envs trace anında bilinir.
# ---------------------------------------------------------------------------

def make_step_fn(cfg: JaxEnvCfg, piece_cells_pad: jnp.ndarray):
    """jit'lenmiş step fonksiyonu döndürür. piece_cells_pad closure'a yakalanır."""
    @partial(jax.jit, static_argnames=())
    def step_fn(state: JaxState, actions: jnp.ndarray):
        return _step_kernel(state, actions, cfg, piece_cells_pad)
    return step_fn


def make_reset_fn(cfg: JaxEnvCfg, piece_cells_pad: jnp.ndarray):
    """Tam reset (tüm env'ler). N runtime'da bilinir → static_argnums=(0,)."""
    @partial(jax.jit, static_argnums=(0,))
    def reset_fn(num_envs: int, master_key: jnp.ndarray) -> JaxState:
        return initial_state(num_envs, master_key, cfg, piece_cells_pad)
    return reset_fn
