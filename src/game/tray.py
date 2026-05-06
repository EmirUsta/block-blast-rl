"""3-slot tray — ham Block Blast davranışı.

1. denemeden farklı: anti-frustration ve ensure-solvable KAPALI default.
Ajanın gerçek dağılımı ve game-over riskini görmesi gerek; suni güvenlik öğrenmeyi
zorlaştırır.

Sadece koruyucu davranışlar:
- `same_tray_unique`: aynı tray'de 3 farklı parça type'ı (oyunun tipik davranışı).
"""

from __future__ import annotations

import numpy as np

from src.game.board import Board
from src.game.pieces import Piece


class Tray:
    def __init__(
        self,
        piece_set: list[Piece],
        size: int = 3,
        rng: np.random.Generator | None = None,
        same_tray_unique: bool = True,
    ) -> None:
        if not piece_set:
            raise ValueError("piece_set must not be empty")
        self.piece_set = piece_set
        self.size = size
        self.rng = rng if rng is not None else np.random.default_rng()
        self.same_tray_unique = same_tray_unique
        self.slots: list[Piece | None] = [None] * size

    def _sample_one(self, used_types: set[int]) -> Piece:
        n = len(self.piece_set)
        weights = np.ones(n, dtype=np.float64)
        if self.same_tray_unique:
            for t in used_types:
                if 0 <= t < n:
                    weights[t] = 0.0
            if weights.sum() <= 0:
                weights = np.ones(n, dtype=np.float64)
        weights /= weights.sum()
        idx = int(self.rng.choice(n, p=weights))
        return self.piece_set[idx]

    def refill_if_all_empty(self) -> bool:
        if not all(s is None for s in self.slots):
            return False
        used: set[int] = set()
        for i in range(self.size):
            p = self._sample_one(used)
            self.slots[i] = p
            used.add(p.type_idx)
        return True

    def take(self, piece_idx: int) -> Piece:
        if piece_idx < 0 or piece_idx >= self.size:
            raise IndexError(f"piece_idx {piece_idx} out of range")
        piece = self.slots[piece_idx]
        if piece is None:
            raise ValueError(f"Slot {piece_idx} is empty")
        self.slots[piece_idx] = None
        return piece

    def has_piece(self, piece_idx: int) -> bool:
        return 0 <= piece_idx < self.size and self.slots[piece_idx] is not None

    def get(self, piece_idx: int) -> Piece | None:
        if piece_idx < 0 or piece_idx >= self.size:
            return None
        return self.slots[piece_idx]

    def any_placeable(self, board: Board) -> bool:
        for piece in self.slots:
            if piece is None:
                continue
            if board.placement_mask(piece).any():
                return True
        return False

    def all_empty(self) -> bool:
        return all(s is None for s in self.slots)

    def slots_remaining(self) -> int:
        return sum(1 for s in self.slots if s is not None)

    def reset(self) -> None:
        self.slots = [None] * self.size
