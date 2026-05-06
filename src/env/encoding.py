"""Observation encoding ve action ↔ (piece_idx, x, y) bijection.

Sabitler:
- BOARD_SIZE = 8
- TRAY_SIZE = 3
- NUM_PIECE_TYPES = 31  (rotation'lar dahil tam set)
- MAX_PIECE_DIM = 5     (tray_shapes padding boyutu)
- FRAME_STACK = 4
- HISTORY_LEN = 8
- META_DIM = 4
- NUM_ACTIONS = TRAY_SIZE * BOARD_SIZE * BOARD_SIZE = 192
- HISTORY_SENTINEL = NUM_ACTIONS  (henüz alınmamış adım için "boş" değer)
"""

from __future__ import annotations

from collections import deque

import numpy as np

from src.game.game import Game
from src.game.pieces import Piece

BOARD_SIZE = 8
TRAY_SIZE = 3
NUM_PIECE_TYPES = 31
MAX_PIECE_DIM = 5
FRAME_STACK = 4
HISTORY_LEN = 8
META_DIM = 4
NUM_ACTIONS = TRAY_SIZE * BOARD_SIZE * BOARD_SIZE  # 192
HISTORY_SENTINEL = NUM_ACTIONS  # 192 — embedding için "boş" indeks


def encode_action(piece_idx: int, x: int, y: int) -> int:
    return piece_idx * (BOARD_SIZE * BOARD_SIZE) + y * BOARD_SIZE + x


def decode_action(action_id: int) -> tuple[int, int, int]:
    piece_idx, rem = divmod(action_id, BOARD_SIZE * BOARD_SIZE)
    y, x = divmod(rem, BOARD_SIZE)
    return piece_idx, x, y


def encode_piece_shape(piece: Piece | None) -> np.ndarray:
    """5×5 padded shape (sol-üst origin). None → tüm sıfır."""
    out = np.zeros((MAX_PIECE_DIM, MAX_PIECE_DIM), dtype=np.float32)
    if piece is None:
        return out
    h, w = piece.height, piece.width
    out[:h, :w] = piece.cells.astype(np.float32)
    return out


def encode_tray_one_hot(piece: Piece | None) -> np.ndarray:
    out = np.zeros(NUM_PIECE_TYPES, dtype=np.float32)
    if piece is None:
        return out
    if piece.type_idx < NUM_PIECE_TYPES:
        out[piece.type_idx] = 1.0
    return out


def count_empty_components(grid: np.ndarray) -> int:
    """Boş hücrelerin 4-connected component sayısı.

    İdeal = 1 (tek büyük boş bölge → kümelenmiş dolu parçalar).
    Kötü = çok sayıda küçük ada (saçılmış yerleştirme).

    Block Blast'ta klasik Tetris bumpiness'ı (column heights diff) anlamlı
    değil çünkü yerçekimi yok. Onun yerine "boş bölge parçalanması" daha doğru
    bir saçılma metriği. Ajan kümelenirse parçalanma azalır.
    """
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


def count_small_islands(grid: np.ndarray, max_size: int = 2) -> int:
    """Boş hücrelerin 4-connected component'larından size <= max_size olanların
    toplam hücre sayısı.

    Block Blast'ta yerçekimi yok, klasik Tetris "hole" tanımı (üstü dolu boş
    hücre) anlamlı değil. Onun yerine: izole edilmiş, **küçük** boş ada'lar
    tehlikelidir çünkü tek-hücre veya iki-hücre boşluğa sadece çok özel
    parçalar (dot, sq2, h2/v2) yerleşir, ajan oraya o parça gelmezse tıkanır.

    Args:
        grid: (8, 8) bool — True = dolu.
        max_size: bu boyut ve altındaki adalar "hole" sayılır.

    Returns:
        Küçük adalardaki toplam hücre sayısı (1-ada = 1, 2-ada = 2, ...).

    BFS ile manuel connected components — 8x8 küçük grid için yeterince hızlı.
    """
    h, w = grid.shape
    visited = grid.copy()  # dolu hücreleri "ziyaret edilmiş" say
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


def encode_meta(game: Game, n_legal: int) -> np.ndarray:
    """4 skaler: [steps_norm, fill_ratio, n_legal_norm, slots_remaining_norm]."""
    steps_norm = min(1.0, game.steps / max(1, game.max_steps))
    fill_ratio = game.board.filled_count() / (BOARD_SIZE * BOARD_SIZE)
    n_legal_norm = n_legal / NUM_ACTIONS
    slots_norm = game.tray.slots_remaining() / TRAY_SIZE
    return np.array(
        [steps_norm, fill_ratio, n_legal_norm, slots_norm], dtype=np.float32
    )


def encode_observation(
    game: Game,
    frame_buf: deque,
    history_buf: deque,
    action_mask: np.ndarray,
) -> dict:
    """Tüm Dict observation'ı oluşturur.

    `frame_buf` deque(maxlen=FRAME_STACK), her elemanı (8,8) float32 grid.
    `history_buf` deque(maxlen=HISTORY_LEN), her elemanı action_id (int).
    `action_mask` 192-uzunluk bool — placement planes (3,8,8) buradan reshape edilir.

    Çıktı:
      - `grid`: (4, 8, 8) frame stack
      - `placements`: (3, 8, 8) — her piece için "yerleşebileceği origin" mask'i.
        Bu, action_mask'in (3, 8, 8) reshape'i (sıfır ek hesap). CNN early-fusion'da
        bu kanallar grid ile birlikte (toplam 7) verilir.
      - `tray`, `tray_shapes`: piece kimliği için late-fusion encoder
      - `history`: son 8 action embedding
      - `meta`: 4 skaler
    """
    grid_stack = np.stack(list(frame_buf), axis=0).astype(np.float32)  # (4,8,8)

    # Placement availability planes — action_mask'tan bedava türetilir.
    # action_id = piece_idx*64 + y*8 + x → reshape(3, 8, 8): axis0=piece, axis1=y, axis2=x.
    placements = action_mask.reshape(TRAY_SIZE, BOARD_SIZE, BOARD_SIZE).astype(np.float32)

    tray = np.zeros((TRAY_SIZE, NUM_PIECE_TYPES), dtype=np.float32)
    tray_shapes = np.zeros((TRAY_SIZE, MAX_PIECE_DIM, MAX_PIECE_DIM), dtype=np.float32)
    for i in range(TRAY_SIZE):
        p = game.tray.get(i)
        tray[i] = encode_tray_one_hot(p)
        tray_shapes[i] = encode_piece_shape(p)

    history = np.array(list(history_buf), dtype=np.int64)
    n_legal = int(action_mask.sum())
    meta = encode_meta(game, n_legal)

    return {
        "grid": grid_stack,
        "placements": placements,
        "tray": tray,
        "tray_shapes": tray_shapes,
        "history": history,
        "meta": meta,
    }
