"""Game state machine — sade, scoring info-only.

1. denemeden farklı:
- Scoring modülü ayrı bir karmaşıklık kaynağıydı; burada hesap satır içi.
- combo / line / cell skoru `info["score"]` olarak monitoring için tutulur, **reward'a
  girmez** (env reward fonksiyonu sadece survival).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from src.game.board import Board
from src.game.pieces import Piece, load_piece_set
from src.game.tray import Tray


@dataclass
class StepResult:
    invalid: bool
    cells_placed: int
    cleared_lines: int
    board_was_cleared: bool
    score_delta: int       # info-only
    game_over: bool


def load_env_config(yaml_path: str | Path) -> dict:
    yaml_path = Path(yaml_path)
    with yaml_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Sade skor formülü (info için):
#   line_score = lines_table[clears]   (clears=0..N)
#   cell_score = n_cells * 1
#   board_clear → +200
# Bunlar reward'a yansımaz.
_LINES_TABLE = (0, 10, 30, 60, 100, 150, 210, 280, 360, 450)
_BOARD_CLEAR_BONUS = 200


class Game:
    def __init__(self, config: dict, rng: np.random.Generator | None = None) -> None:
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng()

        self.board_size = int(config["board"]["size"])
        self.tray_size = int(config["tray"]["size"])
        self.same_tray_unique = bool(config["tray"].get("same_tray_unique", True))
        self.max_steps = int(config["episode"]["max_steps"])

        pieces_path = Path(config["pieces"]["config_file"])
        self.piece_set: list[Piece] = load_piece_set(pieces_path)

        self.board = Board(self.board_size)
        self.tray = Tray(
            self.piece_set,
            self.tray_size,
            self.rng,
            same_tray_unique=self.same_tray_unique,
        )

        self.score: int = 0
        self.steps: int = 0
        self.game_over: bool = False

        self.tray.refill_if_all_empty()

    @classmethod
    def from_config_file(
        cls, yaml_path: str | Path, rng: np.random.Generator | None = None
    ) -> "Game":
        return cls(load_env_config(yaml_path), rng)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.tray.rng = self.rng
        self.board.reset()
        self.tray.reset()
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.tray.refill_if_all_empty()

    def step(self, piece_idx: int, x: int, y: int) -> StepResult:
        if self.game_over:
            return StepResult(True, 0, 0, False, 0, True)

        piece = self.tray.get(piece_idx)
        if piece is None or not self.board.can_place(piece, x, y):
            return StepResult(True, 0, 0, False, 0, self.game_over)

        cells_placed = piece.n_cells
        self.board.place(piece, x, y)
        self.tray.take(piece_idx)

        cleared = self.board.clear_full_lines()
        board_was_cleared = self.board.is_empty() if cleared > 0 else False

        idx = min(cleared, len(_LINES_TABLE) - 1)
        line_score = _LINES_TABLE[idx]
        cell_score = cells_placed
        score_delta = line_score + cell_score + (_BOARD_CLEAR_BONUS if board_was_cleared else 0)
        self.score += score_delta

        self.tray.refill_if_all_empty()

        if not self.tray.any_placeable(self.board):
            self.game_over = True

        self.steps += 1

        return StepResult(
            invalid=False,
            cells_placed=cells_placed,
            cleared_lines=cleared,
            board_was_cleared=board_was_cleared,
            score_delta=score_delta,
            game_over=self.game_over,
        )

    def is_truncated(self) -> bool:
        return self.steps >= self.max_steps
