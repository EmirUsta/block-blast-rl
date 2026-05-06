"""BlockBlastBatchEnv — tek process içinde N oyunu batched numpy ile paralel çalıştıran SB3 VecEnv.

SubprocVecEnv ile drop-in geçiş için:
    from src.env.batch_env import BlockBlastBatchEnv
    venv = BlockBlastBatchEnv(num_envs=16, config_path="configs/env_v4a.yaml", seed=42)
    model = MaskablePPO(..., env=venv, ...)

Davranış BlockBlastEnv (src/env/env.py) ile birebir eşittir; ikisi de aynı YAML
config'i okur ve aynı reward fonksiyonunu uygular. Single-env Python kodu hiç
değiştirilmedi; bu dosya tamamen ayrı bir vectorized implementasyon.

Hot-path stateları batched numpy:
  - boards:        (N, 8, 8) bool        — mevcut tahta
  - frame_buf:     (N, 4, 8, 8) float32  — frame stack (rolling)
  - tray_idx:      (N, 3) int8           — slot piece type idx, -1 = boş
  - history:       (N, 8) int64          — son action_id, sentinel=192
  - step_count:    (N,) int32
  - score:         (N,) int32
  - game_over:     (N,) bool
  - episode_returns/lengths: VecEnv standart episode istatistikleri

Önemli mimari kararlar:
  - 31 piece (rotation dahil) tek 5×5 padded bool tensor (`piece_cells_pad`).
  - Placement check: tahtayı 12×12 padded'e koy (kenar = "dolu") → tek
    sliding_window_view + bool reduction → (N, 31, 8, 8) full mask. y/x sınır
    kontrolü pad mantığıyla otomatik çözülür.
  - Place: 5×5 scratch scatter ile vectorized OR (Python loop yok).
  - Line clear, refill, game_over: tamamen vectorized.
  - Hole / component metrikleri: per-env BFS loop (sadece reward ağırlığı
    nonzero olan env'lerde tetiklenir → v4a config'inde hiç çalışmaz).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces
from numpy.lib.stride_tricks import sliding_window_view
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvIndices, VecEnvObs, VecEnvStepReturn

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
# Hole / component BFS (vectorize edilmedi — pahalı reward shaping path'i)
# ---------------------------------------------------------------------------

def _count_small_islands_single(grid: np.ndarray, max_size: int) -> int:
    h, w = grid.shape
    visited = grid.copy()
    total = 0
    stack: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if visited[r, c]:
                continue
            stack.clear()
            stack.append((r, c))
            visited[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            if len(cells) <= max_size:
                total += len(cells)
    return total


def _count_empty_components_single(grid: np.ndarray) -> int:
    h, w = grid.shape
    visited = grid.copy()
    n = 0
    stack: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if visited[r, c]:
                continue
            n += 1
            stack.clear()
            stack.append((r, c))
            visited[r, c] = True
            while stack:
                y, x = stack.pop()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
    return n


def _holes_batch(boards: np.ndarray, max_size: int) -> np.ndarray:
    out = np.zeros(boards.shape[0], dtype=np.int32)
    for i in range(boards.shape[0]):
        out[i] = _count_small_islands_single(boards[i], max_size)
    return out


def _components_batch(boards: np.ndarray) -> np.ndarray:
    out = np.zeros(boards.shape[0], dtype=np.int32)
    for i in range(boards.shape[0]):
        out[i] = _count_empty_components_single(boards[i])
    return out


# ---------------------------------------------------------------------------
# BlockBlastBatchEnv
# ---------------------------------------------------------------------------

class BlockBlastBatchEnv(VecEnv):
    """SB3 VecEnv: N oyunu tek process'te batched numpy ile paralel çalıştırır."""

    def __init__(
        self,
        num_envs: int,
        config_path: str | Path,
        seed: int | None = None,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be > 0")

        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        rw = self.cfg.get("reward", {})
        self.r_step: float = float(rw.get("step", 1.0))
        self.r_game_over: float = float(rw.get("game_over", -10.0))
        self.r_invalid: float = float(rw.get("invalid_action", -1.0))
        self.r_hole: float = float(rw.get("hole_penalty", 0.0))
        self.hole_max_size: int = int(rw.get("hole_max_size", 2))
        self.r_bumpiness: float = float(rw.get("bumpiness_penalty", 0.0))
        table_cfg = rw.get("line_clear_table")
        self.r_line_table: np.ndarray | None = (
            np.asarray(table_cfg, dtype=np.float32) if table_cfg is not None else None
        )
        self.r_line_base: float = float(rw.get("line_clear_base", 0.0))
        self.r_line_exp: float = float(rw.get("line_clear_exp", 3.0))

        self.board_size = int(self.cfg["board"]["size"])
        if self.board_size != BOARD_SIZE:
            raise ValueError(f"board.size must be {BOARD_SIZE}, got {self.board_size}")
        self.tray_size = int(self.cfg["tray"]["size"])
        if self.tray_size != TRAY_SIZE:
            raise ValueError(f"tray.size must be {TRAY_SIZE}, got {self.tray_size}")
        self.same_tray_unique = bool(self.cfg["tray"].get("same_tray_unique", True))
        self.max_steps = int(self.cfg["episode"]["max_steps"])

        pieces_path = Path(self.cfg["pieces"]["config_file"])
        piece_set = load_piece_set(pieces_path)
        if len(piece_set) != NUM_PIECE_TYPES:
            raise ValueError(
                f"piece_set size {len(piece_set)} != NUM_PIECE_TYPES {NUM_PIECE_TYPES}"
            )

        # Pre-bake piece tensors (5x5 padded) and metadata
        P = NUM_PIECE_TYPES
        D = MAX_PIECE_DIM
        self.piece_cells_pad = np.zeros((P, D, D), dtype=bool)
        self.piece_h = np.zeros(P, dtype=np.int8)
        self.piece_w = np.zeros(P, dtype=np.int8)
        self.piece_n_cells = np.zeros(P, dtype=np.int8)
        for p in piece_set:
            h, w = p.height, p.width
            self.piece_cells_pad[p.type_idx, :h, :w] = p.cells
            self.piece_h[p.type_idx] = h
            self.piece_w[p.type_idx] = w
            self.piece_n_cells[p.type_idx] = p.n_cells

        # Pad board-side so sliding_window_view (5,5) covers all valid origins.
        # Real board: rows [0..7], cols [0..7]; pad rows [8..11], cols [8..11].
        # For placement check, pad area is always "filled" (True) → conflict on
        # any piece cell falling into pad → automatic boundary rejection.
        self._pad_size = BOARD_SIZE + (D - 1)  # 12
        self._padded_buf = np.zeros((num_envs, self._pad_size, self._pad_size), dtype=bool)
        # Persistent pad mask: True at indices >= 8
        self._pad_mask_static = np.zeros((self._pad_size, self._pad_size), dtype=bool)
        self._pad_mask_static[BOARD_SIZE:, :] = True
        self._pad_mask_static[:, BOARD_SIZE:] = True

        # ------------------------------------------------------------------
        # VecEnv interface metadata
        # ------------------------------------------------------------------
        observation_space = spaces.Dict(
            {
                "grid": spaces.Box(0.0, 1.0, (FRAME_STACK, BOARD_SIZE, BOARD_SIZE), dtype=np.float32),
                "placements": spaces.Box(0.0, 1.0, (TRAY_SIZE, BOARD_SIZE, BOARD_SIZE), dtype=np.float32),
                "tray": spaces.Box(0.0, 1.0, (TRAY_SIZE, NUM_PIECE_TYPES), dtype=np.float32),
                "tray_shapes": spaces.Box(0.0, 1.0, (TRAY_SIZE, MAX_PIECE_DIM, MAX_PIECE_DIM), dtype=np.float32),
                "history": spaces.Box(0, HISTORY_SENTINEL, (HISTORY_LEN,), dtype=np.int64),
                "meta": spaces.Box(0.0, 1.0, (META_DIM,), dtype=np.float32),
            }
        )
        action_space = spaces.Discrete(NUM_ACTIONS)
        super().__init__(num_envs, observation_space, action_space)

        # ------------------------------------------------------------------
        # Batched state
        # ------------------------------------------------------------------
        self.N = num_envs
        self._rng = np.random.default_rng(seed if seed is not None else 0)

        self.boards = np.zeros((self.N, BOARD_SIZE, BOARD_SIZE), dtype=bool)
        self.frame_buf = np.zeros((self.N, FRAME_STACK, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        self.tray_idx = np.full((self.N, TRAY_SIZE), -1, dtype=np.int8)
        self.history = np.full((self.N, HISTORY_LEN), HISTORY_SENTINEL, dtype=np.int64)
        self.step_count = np.zeros(self.N, dtype=np.int32)
        self.score = np.zeros(self.N, dtype=np.int32)
        self.game_over = np.zeros(self.N, dtype=bool)
        self.episode_returns = np.zeros(self.N, dtype=np.float64)
        self.episode_lengths = np.zeros(self.N, dtype=np.int32)

        self._cached_action_mask = np.zeros((self.N, NUM_ACTIONS), dtype=bool)
        self._actions: np.ndarray | None = None  # set by step_async

        # Initial reset: full population
        self._reset_all()

    # ----------------------------------------------------------------------
    # Internal: tray refill
    # ----------------------------------------------------------------------

    def _refill_empty_trays(self) -> None:
        """Refill any tray whose all 3 slots are -1. Uses argsort trick for unique sampling."""
        all_empty = (self.tray_idx == -1).all(axis=1)  # (N,)
        n = int(all_empty.sum())
        if n == 0:
            return
        if self.same_tray_unique:
            # 3 unique piece types per env → argsort of random uniform (n, 31) gives random permutation
            rand = self._rng.random((n, NUM_PIECE_TYPES))
            picks = np.argsort(rand, axis=1)[:, :TRAY_SIZE].astype(np.int8)
        else:
            picks = self._rng.integers(0, NUM_PIECE_TYPES, size=(n, TRAY_SIZE), dtype=np.int8)
        self.tray_idx[all_empty] = picks

    # ----------------------------------------------------------------------
    # Internal: placement_mask_all → (N, 31, 8, 8) and per-tray (N, 3, 8, 8)
    # ----------------------------------------------------------------------

    def _build_action_mask(self) -> np.ndarray:
        """Returns (N, 192) bool. Pad-true sliding window over tray slot pieces only.

        Per-slot vectorized:  (N, 3, 8, 8) ← (N, 1, 8, 8, 5, 5) AND (N, 3, 1, 1, 5, 5).
        Pad area is True → out-of-bounds origins auto-rejected.
        """
        # Build padded board: real area = boards, pad area = True (always filled)
        self._padded_buf[:, :BOARD_SIZE, :BOARD_SIZE] = self.boards
        self._padded_buf[:, BOARD_SIZE:, :] = True
        self._padded_buf[:, :, BOARD_SIZE:] = True

        # Sliding (5,5) windows: (N, 8, 8, 5, 5)
        windows = sliding_window_view(self._padded_buf, (MAX_PIECE_DIM, MAX_PIECE_DIM), axis=(1, 2))

        # Gather tray pieces: (N, 3, 5, 5)
        safe_tray = np.where(self.tray_idx >= 0, self.tray_idx, 0).astype(np.int64)
        tray_pieces = self.piece_cells_pad[safe_tray]

        # conflict: (N, 1, 8, 8, 5, 5) & (N, 3, 1, 1, 5, 5) → reduce → (N, 3, 8, 8)
        conflict = (
            windows[:, None, :, :, :, :] & tray_pieces[:, :, None, None, :, :]
        ).any(axis=(-1, -2))
        per_slot = ~conflict  # (N, 3, 8, 8)

        # Empty slots / game_over masking
        slot_alive = (self.tray_idx >= 0)[:, :, None, None]
        per_slot &= slot_alive
        per_slot &= (~self.game_over)[:, None, None, None]

        return per_slot.reshape(self.N, NUM_ACTIONS)

    # ----------------------------------------------------------------------
    # Internal: reset for given env indices (in-place)
    # ----------------------------------------------------------------------

    def _reset_indices(self, indices: np.ndarray) -> None:
        if indices.size == 0:
            return
        self.boards[indices] = False
        self.tray_idx[indices] = -1
        self.history[indices] = HISTORY_SENTINEL
        self.step_count[indices] = 0
        self.score[indices] = 0
        self.game_over[indices] = False
        self.episode_returns[indices] = 0.0
        self.episode_lengths[indices] = 0
        # Reset frame buffer (float32 zeros — board initially empty)
        self.frame_buf[indices] = 0.0
        # Refill trays (in-place; only those with all slots -1 — all reset env'leri)
        self._refill_empty_trays()

    def _reset_all(self) -> None:
        self._reset_indices(np.arange(self.N))
        self._cached_action_mask = self._build_action_mask()

    # ----------------------------------------------------------------------
    # Observation construction (vectorized)
    # ----------------------------------------------------------------------

    def _build_obs(self) -> dict:
        # placements: action_mask reshape (N, 3, 8, 8) float32 — bedava
        placements = self._cached_action_mask.reshape(self.N, TRAY_SIZE, BOARD_SIZE, BOARD_SIZE).astype(np.float32)

        # tray one-hot (N, 3, 31)
        safe_tray = np.where(self.tray_idx >= 0, self.tray_idx, 0).astype(np.int64)
        tray_one_hot = np.eye(NUM_PIECE_TYPES, dtype=np.float32)[safe_tray]
        # Zero out empty slots
        empty_slot = (self.tray_idx < 0)[:, :, None]
        tray_one_hot = np.where(empty_slot, 0.0, tray_one_hot)

        # tray_shapes (N, 3, 5, 5)
        tray_shapes = self.piece_cells_pad[safe_tray].astype(np.float32)
        tray_shapes = np.where(empty_slot[:, :, :, None], 0.0, tray_shapes)

        # meta (N, 4)
        steps_norm = np.minimum(1.0, self.step_count.astype(np.float32) / max(1, self.max_steps))
        fill_ratio = self.boards.sum(axis=(1, 2)).astype(np.float32) / (BOARD_SIZE * BOARD_SIZE)
        n_legal = self._cached_action_mask.sum(axis=1).astype(np.float32)
        n_legal_norm = n_legal / NUM_ACTIONS
        slots_remaining = (self.tray_idx >= 0).sum(axis=1).astype(np.float32)
        slots_norm = slots_remaining / TRAY_SIZE
        meta = np.stack([steps_norm, fill_ratio, n_legal_norm, slots_norm], axis=1)

        return {
            "grid": self.frame_buf.copy(),
            "placements": placements,
            "tray": tray_one_hot,
            "tray_shapes": tray_shapes,
            "history": self.history.copy(),
            "meta": meta,
        }

    # ----------------------------------------------------------------------
    # Vectorized place: scatter piece_cells onto boards using 5×5 OR
    # ----------------------------------------------------------------------

    def _place_pieces(
        self, env_idx: np.ndarray, piece_type: np.ndarray, y: np.ndarray, x: np.ndarray
    ) -> None:
        """For each entry, write piece_cells_pad[piece_type[k]] OR'd into
        boards[env_idx[k], y[k]:y[k]+5, x[k]:x[k]+5] — using a padded scratch
        so y+5 / x+5 can spill into pad area (which is discarded)."""
        if env_idx.size == 0:
            return
        # padded boards (env_idx subset) for safe scatter
        n = env_idx.size
        # Use the per-step _padded_buf zone (NOT the placement-check version
        # which has pad=True). Allocate a fresh scratch with pad=False so that
        # the OR onto pad area is benign and harvested back into boards.
        pad = np.zeros((n, self._pad_size, self._pad_size), dtype=bool)
        pad[:, :BOARD_SIZE, :BOARD_SIZE] = self.boards[env_idx]
        # Build coordinate grids: rows = y[:, None] + arange(5)[None, :, None]
        ar = np.arange(MAX_PIECE_DIM)
        rows = (y[:, None, None] + ar[None, :, None])  # (n, 5, 1)
        cols = (x[:, None, None] + ar[None, None, :])  # (n, 1, 5)
        rows_b = np.broadcast_to(rows, (n, MAX_PIECE_DIM, MAX_PIECE_DIM))
        cols_b = np.broadcast_to(cols, (n, MAX_PIECE_DIM, MAX_PIECE_DIM))
        envs_b = np.broadcast_to(np.arange(n)[:, None, None], (n, MAX_PIECE_DIM, MAX_PIECE_DIM))
        cells = self.piece_cells_pad[piece_type]  # (n, 5, 5)
        # Vectorized OR: pad |= cells (only where cells is True)
        pad[envs_b, rows_b, cols_b] |= cells
        # Harvest back the real 8×8 area
        self.boards[env_idx] = pad[:, :BOARD_SIZE, :BOARD_SIZE]

    # ----------------------------------------------------------------------
    # Step
    # ----------------------------------------------------------------------

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = np.asarray(actions, dtype=np.int64)

    def step_wait(self) -> VecEnvStepReturn:
        actions = self._actions
        if actions is None:
            raise RuntimeError("step_async() must be called before step_wait()")
        N = self.N
        if actions.shape != (N,):
            raise ValueError(f"actions shape {actions.shape} != ({N},)")

        rewards = np.zeros(N, dtype=np.float32)
        dones = np.zeros(N, dtype=bool)

        # Validity using cached pre-step mask
        valid = self._cached_action_mask[np.arange(N), actions]
        # game_over envs: skip step entirely (caller shouldn't have called, but be safe)
        # (Won't actually happen because terminal envs were reset at end of previous step.)

        # Decode actions: piece_slot, y, x
        piece_slot = actions // (BOARD_SIZE * BOARD_SIZE)        # (N,)
        rem = actions % (BOARD_SIZE * BOARD_SIZE)
        y = rem // BOARD_SIZE
        x = rem % BOARD_SIZE

        invalid = ~valid
        # Pre-step metrics (only when reward weights nonzero)
        need_holes = self.r_hole != 0.0
        need_comps = self.r_bumpiness != 0.0
        if need_holes:
            holes_before = _holes_batch(self.boards, self.hole_max_size)
        else:
            holes_before = None
        if need_comps:
            comps_before = _components_batch(self.boards)
        else:
            comps_before = None

        # ---- Apply place for valid actions ----
        valid_idx_arr = np.nonzero(valid)[0]
        if valid_idx_arr.size:
            # Lookup piece type from tray
            v_piece_type = self.tray_idx[valid_idx_arr, piece_slot[valid_idx_arr]].astype(np.int64)
            # piece_n_cells for score (info-only)
            v_n_cells = self.piece_n_cells[v_piece_type].astype(np.int32)
            self._place_pieces(valid_idx_arr, v_piece_type, y[valid_idx_arr], x[valid_idx_arr])
            # Take piece from tray (slot → -1)
            self.tray_idx[valid_idx_arr, piece_slot[valid_idx_arr]] = -1

            # Line clear (vectorized) on valid envs only
            v_boards = self.boards[valid_idx_arr]
            full_rows = v_boards.all(axis=2)        # (V, 8)
            full_cols = v_boards.all(axis=1)        # (V, 8)
            n_cleared = full_rows.sum(axis=1) + full_cols.sum(axis=1)  # (V,)
            clear_mask = full_rows[:, :, None] | full_cols[:, None, :]
            v_boards = v_boards & ~clear_mask
            # board_was_cleared: post-clear all empty AND cleared > 0
            board_was_cleared = (n_cleared > 0) & (~v_boards.any(axis=(1, 2)))
            self.boards[valid_idx_arr] = v_boards

            # Refill empty trays (only for valid envs; vectorized check handles
            # all N but only those with all -1 slots get refilled)
            self._refill_empty_trays()

            # Score update (info-only; not used by reward)
            _LINES_TABLE = (0, 10, 30, 60, 100, 150, 210, 280, 360, 450)
            lines_table_arr = np.asarray(_LINES_TABLE, dtype=np.int32)
            n_cleared_idx = np.minimum(n_cleared, len(lines_table_arr) - 1)
            line_score = lines_table_arr[n_cleared_idx]
            cell_score = v_n_cells
            score_delta = line_score + cell_score + np.where(board_was_cleared, 200, 0)
            self.score[valid_idx_arr] += score_delta

            # Reward: r_step
            rewards[valid_idx_arr] += self.r_step

            # Line clear reward
            if self.r_line_table is not None:
                lc_idx = np.minimum(n_cleared, len(self.r_line_table) - 1)
                lc_reward = self.r_line_table[lc_idx]
                rewards[valid_idx_arr] += lc_reward
            elif self.r_line_base != 0.0:
                non_zero = n_cleared > 0
                lc_reward = np.where(
                    non_zero,
                    self.r_line_base * (self.r_line_exp ** (n_cleared - 1).clip(min=0)),
                    0.0,
                ).astype(np.float32)
                rewards[valid_idx_arr] += lc_reward
        else:
            n_cleared = np.zeros(0, dtype=np.int32)
            board_was_cleared = np.zeros(0, dtype=bool)

        # ---- Invalid action handling ----
        if invalid.any():
            inv_idx = np.nonzero(invalid)[0]
            rewards[inv_idx] += self.r_invalid
            self.game_over[inv_idx] = True

        # ---- step_count + history + frame_buf push (for valid envs) ----
        if valid_idx_arr.size:
            self.step_count[valid_idx_arr] += 1
            # Frame buf rolling: shift left by 1, append current board (float32)
            self.frame_buf[valid_idx_arr, :-1] = self.frame_buf[valid_idx_arr, 1:]
            self.frame_buf[valid_idx_arr, -1] = self.boards[valid_idx_arr].astype(np.float32)
            # History rolling
            self.history[valid_idx_arr, :-1] = self.history[valid_idx_arr, 1:]
            self.history[valid_idx_arr, -1] = actions[valid_idx_arr]
        # Invalid envs: also push action to history? In single-env it does
        # NOT push (early return). Match that behavior: skip history/frame_buf
        # update for invalid envs (they're terminating anyway).

        # ---- Post-step action mask (for game_over check + next obs) ----
        # After place + clear + refill, recompute placement mask
        self._cached_action_mask = self._build_action_mask()

        # Post-step hole/component delta penalties
        if need_holes and valid_idx_arr.size:
            holes_after = _holes_batch(self.boards, self.hole_max_size)
            holes_delta = np.maximum(0, holes_after - holes_before)
            rewards += (self.r_hole * holes_delta).astype(np.float32)  # only valid envs have nonzero delta meaningfully
            # Restrict to valid envs (invalid envs have unchanged board, delta=0 anyway)
        if need_comps and valid_idx_arr.size:
            comps_after = _components_batch(self.boards)
            comps_delta = np.maximum(0, comps_after - comps_before)
            rewards += (self.r_bumpiness * comps_delta).astype(np.float32)

        # ---- game_over detection (post-step): valid env ile no legal move kaldıysa ----
        if valid_idx_arr.size:
            slot_alive = (self.tray_idx >= 0)
            no_pieces = ~slot_alive.any(axis=1)
            # If trays empty but were just refilled this step, no_pieces is False.
            # Real check: is there ANY legal action in cached_action_mask?
            no_legal = ~self._cached_action_mask.any(axis=1)
            new_game_over = no_legal & (~self.game_over)
            if new_game_over.any():
                self.game_over[new_game_over] = True
                rewards[new_game_over] += self.r_game_over

        # ---- Truncation (max_steps) ----
        truncated = self.step_count >= self.max_steps
        # SB3 VecEnv historically squashes terminated/truncated into done=True;
        # we follow the standard convention (auto-reset on either).
        terminated = self.game_over.copy()
        dones = terminated | truncated

        # Episode statistics + auto-reset
        self.episode_returns += rewards.astype(np.float64)
        self.episode_lengths += 1  # increment for ALL env (including invalid)

        infos: list[dict[str, Any]] = [{} for _ in range(N)]
        # Mark TimeLimit.truncated (Gymnasium convention) on truncated-but-not-terminated
        for i in range(N):
            if truncated[i] and not terminated[i]:
                infos[i]["TimeLimit.truncated"] = True
            if dones[i]:
                infos[i]["episode"] = {
                    "r": float(self.episode_returns[i]),
                    "l": int(self.episode_lengths[i]),
                }

        # Auto-reset terminal envs and inject terminal_observation
        done_idx = np.nonzero(dones)[0]
        if done_idx.size:
            # Build terminal_observation BEFORE reset (per SB3 convention)
            term_obs = self._build_obs()
            for i in done_idx:
                infos[int(i)]["terminal_observation"] = {
                    k: term_obs[k][int(i)].copy() for k in term_obs
                }
            # Reset terminal envs
            self._reset_indices(done_idx)
            self._cached_action_mask = self._build_action_mask()

        obs = self._build_obs()
        return obs, rewards.astype(np.float32), dones, infos

    # ----------------------------------------------------------------------
    # Reset
    # ----------------------------------------------------------------------

    def reset(self) -> VecEnvObs:
        self._reset_all()
        return self._build_obs()

    # ----------------------------------------------------------------------
    # MaskablePPO hook
    # ----------------------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        """Return cached (N, 192) bool mask. Recomputed on every step/reset."""
        return self._cached_action_mask.copy()

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs):
        """Only env_method('action_masks') is supported (sb3-contrib calls it
        as a fallback). Returns list of per-env (192,) bool arrays."""
        idx = self._get_indices(indices)
        if method_name == "action_masks":
            return [self._cached_action_mask[i].copy() for i in idx]
        raise NotImplementedError(
            f"BlockBlastBatchEnv.env_method('{method_name}') is not supported"
        )

    # ----------------------------------------------------------------------
    # VecEnv boilerplate
    # ----------------------------------------------------------------------

    def seed(self, seed: int | None = None) -> Sequence[None]:
        self._rng = np.random.default_rng(seed if seed is not None else 0)
        return [None] * self.N

    def close(self) -> None:
        return None

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        idx = self._get_indices(indices)
        if hasattr(self, attr_name):
            val = getattr(self, attr_name)
            return [val for _ in idx]
        raise AttributeError(f"BlockBlastBatchEnv has no attribute '{attr_name}'")

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        # Per-env attribute writes are not supported by this batched impl.
        raise NotImplementedError(
            "BlockBlastBatchEnv.set_attr is not supported (state is shared/batched)."
        )

    def env_is_wrapped(self, wrapper_class, indices: VecEnvIndices = None) -> list[bool]:
        idx = self._get_indices(indices)
        return [False for _ in idx]

    def render(self, mode: str = "rgb_array") -> Any:  # noqa: ARG002
        return self.boards.copy()

    # ----------------------------------------------------------------------
    # Index helper (compat with SB3 VecEnv signature)
    # ----------------------------------------------------------------------

    def _get_indices(self, indices: VecEnvIndices) -> Sequence[int]:
        if indices is None:
            return list(range(self.N))
        if isinstance(indices, int):
            return [indices]
        return list(indices)
