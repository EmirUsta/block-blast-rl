"""8×8 Block Blast tahtası — saf NumPy.

Numba bağımlılığı yok (1. denemede vardı); bu projede performans yeterli, sadelik öncelik.
"""

from __future__ import annotations

import numpy as np

from src.game.pieces import Piece


class Board:
    def __init__(self, size: int = 8) -> None:
        self.size = size
        self.grid: np.ndarray = np.zeros((size, size), dtype=bool)

    def reset(self) -> None:
        self.grid.fill(False)

    def can_place(self, piece: Piece, x: int, y: int) -> bool:
        h, w = piece.height, piece.width
        if x < 0 or y < 0 or x + w > self.size or y + h > self.size:
            return False
        sub = self.grid[y : y + h, x : x + w]
        return not bool(np.any(sub & piece.cells))

    def placement_mask(self, piece: Piece) -> np.ndarray:
        """Tüm (x, y) konumları için can_place sonucunu vectorized hesaplar.

        Returns: (size-h+1, size-w+1) shape bool. mask[y, x] = piece (x, y)'e konabilir mi?
        """
        h, w = piece.height, piece.width
        if h > self.size or w > self.size:
            return np.zeros(
                (max(0, self.size - h + 1), max(0, self.size - w + 1)), dtype=bool
            )
        # Sliding window with NumPy
        from numpy.lib.stride_tricks import sliding_window_view

        windows = sliding_window_view(self.grid, (h, w))  # (S-h+1, S-w+1, h, w)
        # Conflict: any(window & piece.cells) → invalid; we want NOT
        conflict = (windows & piece.cells).any(axis=(2, 3))
        return ~conflict

    def place(self, piece: Piece, x: int, y: int) -> None:
        if not self.can_place(piece, x, y):
            raise ValueError(f"Cannot place {piece.id} at ({x},{y})")
        h, w = piece.height, piece.width
        self.grid[y : y + h, x : x + w] |= piece.cells

    def clear_full_lines(self) -> int:
        """Aynı step'te dolu satır VE sütunları toplu temizler.

        Returns: temizlenen satır + sütun sayısı.
        """
        full_rows = np.all(self.grid, axis=1)
        full_cols = np.all(self.grid, axis=0)
        n = int(full_rows.sum() + full_cols.sum())
        if n == 0:
            return 0
        mask = np.zeros_like(self.grid)
        mask[full_rows, :] = True
        mask[:, full_cols] = True
        self.grid &= ~mask
        return n

    def is_empty(self) -> bool:
        return not bool(np.any(self.grid))

    def is_full(self) -> bool:
        return bool(np.all(self.grid))

    def filled_count(self) -> int:
        return int(self.grid.sum())

    def copy(self) -> "Board":
        new = Board(self.size)
        new.grid = self.grid.copy()
        return new
