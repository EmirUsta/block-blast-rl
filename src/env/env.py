"""Block Blast Gymnasium environment — survival-only reward.

Reward fonksiyonu:
    +1.0  her başarılı yerleştirme
   -10.0  game over olduğunda (terminal step)
   -1.0   invalid action (mask zaten engellemeli; emniyet için)
    0.0   diğer her şey (line clear, combo, score, board clear, holes, …)

Observation: src/env/encoding.py dokümante eder.
Action mask: `info["action_mask"]` üzerinden MaskablePPO'ya verilir.
`action_masks()` metodu da sağlanır (sb3-contrib bunu otomatik bulur).
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

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
    count_empty_components,
    count_small_islands,
    decode_action,
    encode_observation,
)
from src.env.masking import build_action_mask
from src.game.game import Game


def _zero_grid() -> np.ndarray:
    return np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)


class BlockBlastEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config_path: str | Path, seed: int | None = None) -> None:
        super().__init__()
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        rw = self.cfg.get("reward", {})
        self.r_step: float = float(rw.get("step", 1.0))
        self.r_game_over: float = float(rw.get("game_over", -10.0))
        self.r_invalid: float = float(rw.get("invalid_action", -1.0))
        # Hole penalty: küçük (size <= hole_max_size) izole boş ada başına
        # Sadece pozitif delta cezalandırılır; line clear sonrası azalma ödüllendirilmez.
        # Default 0.0 → backwards compatible (env.yaml dokunulmadan v1/v2 davranışı).
        self.r_hole: float = float(rw.get("hole_penalty", 0.0))
        self.hole_max_size: int = int(rw.get("hole_max_size", 2))
        # Bumpiness/fragmentation: boş empty-component sayısının artışı cezalandırılır.
        # Asimetrik (line clear sonrası azalma ödüllendirilmez; survival zaten bunu içeriyor).
        self.r_bumpiness: float = float(rw.get("bumpiness_penalty", 0.0))
        # Line clear ödülü — iki mod:
        #   (a) line_clear_table: [r0, r1, r2, ...] — n_lines'a doğrudan lookup. Öncelikli.
        #   (b) line_clear_base * exp^(n-1) for n >= 1 — fallback.
        # Defaults: tablo None, base 0.0 → kapalı (geri uyumlu).
        table_cfg = rw.get("line_clear_table")
        self.r_line_table: list[float] | None = (
            [float(x) for x in table_cfg] if table_cfg is not None else None
        )
        self.r_line_base: float = float(rw.get("line_clear_base", 0.0))
        self.r_line_exp: float = float(rw.get("line_clear_exp", 3.0))

        self._seed = int(seed) if seed is not None else 0
        rng = np.random.default_rng(self._seed)
        self.game = Game(self.cfg, rng=rng)

        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(0.0, 1.0, (FRAME_STACK, BOARD_SIZE, BOARD_SIZE), dtype=np.float32),
                "placements": spaces.Box(0.0, 1.0, (TRAY_SIZE, BOARD_SIZE, BOARD_SIZE), dtype=np.float32),
                "tray": spaces.Box(0.0, 1.0, (TRAY_SIZE, NUM_PIECE_TYPES), dtype=np.float32),
                "tray_shapes": spaces.Box(0.0, 1.0, (TRAY_SIZE, MAX_PIECE_DIM, MAX_PIECE_DIM), dtype=np.float32),
                "history": spaces.Box(0, HISTORY_SENTINEL, (HISTORY_LEN,), dtype=np.int64),
                "meta": spaces.Box(0.0, 1.0, (META_DIM,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        self._frame_buf: deque[np.ndarray] = deque(maxlen=FRAME_STACK)
        self._history_buf: deque[int] = deque(maxlen=HISTORY_LEN)
        self._cached_mask: np.ndarray = np.zeros(NUM_ACTIONS, dtype=bool)
        self._ep_steps: int = 0

    def _reset_buffers(self) -> None:
        self._frame_buf.clear()
        for _ in range(FRAME_STACK):
            self._frame_buf.append(self.game.board.grid.astype(np.float32))
        self._history_buf.clear()
        for _ in range(HISTORY_LEN):
            self._history_buf.append(HISTORY_SENTINEL)
        self._ep_steps = 0

    def _push_frame(self) -> None:
        self._frame_buf.append(self.game.board.grid.astype(np.float32))

    def _push_action(self, action_id: int) -> None:
        self._history_buf.append(int(action_id))

    def _build_obs(self) -> dict:
        return encode_observation(
            self.game, self._frame_buf, self._history_buf, action_mask=self._cached_mask
        )

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict, dict]:
        if seed is not None:
            self._seed = int(seed)
        self.game.reset(seed=self._seed)
        # Her reset'te seed'i kademeli ilerlet ki tekrarlama olmasın
        self._seed += 1
        self._reset_buffers()
        self._cached_mask = build_action_mask(self.game)
        obs = self._build_obs()
        info = {
            "action_mask": self._cached_mask.copy(),
            "score": self.game.score,
            "episode_length": self._ep_steps,
        }
        return obs, info

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        action = int(action)
        terminated = False
        truncated = False

        # Mask kontrolü — emniyet (MaskablePPO mask'i kullanır, ama doğrulayalım)
        if not self._cached_mask[action]:
            reward = self.r_invalid
            terminated = True
            self.game.game_over = True
            self._cached_mask = np.zeros(NUM_ACTIONS, dtype=bool)
            obs = self._build_obs()
            info = {
                "action_mask": self._cached_mask.copy(),
                "score": self.game.score,
                "episode_length": self._ep_steps,
                "invalid_action": True,
            }
            return obs, reward, terminated, truncated, info

        piece_idx, x, y = decode_action(action)

        # Pre-step metrics (yalnız penalty etkinse hesapla → CPU tasarrufu)
        holes_before = (
            count_small_islands(self.game.board.grid, self.hole_max_size)
            if self.r_hole != 0.0 else 0
        )
        comps_before = (
            count_empty_components(self.game.board.grid)
            if self.r_bumpiness != 0.0 else 0
        )

        result = self.game.step(piece_idx, x, y)

        if result.invalid:
            reward = self.r_invalid
            terminated = True
            holes_after = holes_before
            holes_delta = 0
            comps_after = comps_before
            comps_delta = 0
            line_clear_reward = 0.0
        else:
            reward = self.r_step
            if result.game_over:
                reward += self.r_game_over
                terminated = True
            elif self.game.is_truncated():
                truncated = True
            # Hole metric: post-step
            if self.r_hole != 0.0:
                holes_after = count_small_islands(
                    self.game.board.grid, self.hole_max_size
                )
                holes_delta = max(0, holes_after - holes_before)
                reward += self.r_hole * float(holes_delta)
            else:
                holes_after = 0
                holes_delta = 0
            # Bumpiness metric: post-step
            if self.r_bumpiness != 0.0:
                comps_after = count_empty_components(self.game.board.grid)
                comps_delta = max(0, comps_after - comps_before)
                reward += self.r_bumpiness * float(comps_delta)
            else:
                comps_after = 0
                comps_delta = 0
            # Line clear ödülü — tablo öncelikli, yoksa base*exp formülü
            n_lines = int(result.cleared_lines)
            if n_lines > 0 and self.r_line_table is not None:
                # Tablo lookup; index aşımında son değeri kullan (cap)
                idx = min(n_lines, len(self.r_line_table) - 1)
                line_clear_reward = self.r_line_table[idx]
                reward += line_clear_reward
            elif n_lines > 0 and self.r_line_base != 0.0:
                line_clear_reward = self.r_line_base * (
                    self.r_line_exp ** (n_lines - 1)
                )
                reward += line_clear_reward
            else:
                line_clear_reward = 0.0

        self._ep_steps += 1
        self._push_frame()
        self._push_action(action)
        self._cached_mask = build_action_mask(self.game)
        obs = self._build_obs()
        info = {
            "action_mask": self._cached_mask.copy(),
            "score": self.game.score,
            "score_delta": result.score_delta,
            "cleared_lines": result.cleared_lines,
            "board_was_cleared": result.board_was_cleared,
            "cells_placed": result.cells_placed,
            "holes_before": holes_before,
            "holes_after": holes_after,
            "holes_delta": holes_delta,
            "comps_before": comps_before,
            "comps_after": comps_after,
            "comps_delta": comps_delta,
            "line_clear_reward": line_clear_reward,
            "episode_length": self._ep_steps,
            "game_over": result.game_over,
        }
        return obs, float(reward), terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """sb3-contrib MaskablePPO bu metodu otomatik tarar."""
        return self._cached_mask.copy()

    def render(self) -> Any:  # noqa: ANN401
        return self.game.board.grid.copy()
