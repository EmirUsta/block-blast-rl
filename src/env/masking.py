"""Action masking: 192-uzunluk bool array.

mask[a] = True ⇔ (piece_idx, x, y) = decode_action(a) için:
  1. tray.has_piece(piece_idx)
  2. board.can_place(piece, x, y)
"""

from __future__ import annotations

import numpy as np

from src.env.encoding import BOARD_SIZE, NUM_ACTIONS, TRAY_SIZE
from src.game.game import Game


def build_action_mask(game: Game) -> np.ndarray:
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    if game.game_over:
        return mask
    for pi in range(TRAY_SIZE):
        piece = game.tray.get(pi)
        if piece is None:
            continue
        pm = game.board.placement_mask(piece)  # (S-h+1, S-w+1)
        if pm.size == 0 or not pm.any():
            continue
        ys, xs = np.where(pm)
        offsets = pi * (BOARD_SIZE * BOARD_SIZE) + ys * BOARD_SIZE + xs
        mask[offsets] = True
    return mask
