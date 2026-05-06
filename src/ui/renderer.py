"""Pygame renderer — sade görselleştirme.

Sol: 8×8 board. Alt: 3 piece tray. Sağ: HUD (steps, score, clears, ep_len ortalama).
Animasyon yok, her step ekran direkt güncellenir.
"""

from __future__ import annotations

import pygame
import numpy as np

from src.game.board import Board
from src.game.tray import Tray

# Boyutlar
CELL = 56
PAD = 24
GRID_PX = 8 * CELL                          # 448
TRAY_CELL = 28
TRAY_SLOT_W = 5 * TRAY_CELL                 # 140
TRAY_GAP = 24
TRAY_PX_W = 3 * TRAY_SLOT_W + 2 * TRAY_GAP  # 468
TRAY_PX_H = 5 * TRAY_CELL                   # 140
PANEL_W = 260

WINDOW_W = PAD + GRID_PX + PAD + PANEL_W + PAD     # 24+448+24+260+24 = 780
WINDOW_H = PAD + GRID_PX + PAD + TRAY_PX_H + PAD    # 24+448+24+140+24 = 660

# Renkler
BG = (24, 26, 35)
GRID_BG = (40, 44, 56)
GRID_LINE = (60, 65, 80)
CELL_FILLED = (90, 200, 255)
CELL_HIGHLIGHT = (255, 220, 100)
PIECE_COLOR = (255, 200, 120)
PIECE_DIM = (140, 110, 60)
TEXT = (220, 220, 230)
TEXT_DIM = (150, 150, 160)
GAME_OVER = (255, 90, 90)


def _grid_origin() -> tuple[int, int]:
    return PAD, PAD


def _tray_origin() -> tuple[int, int]:
    gx, gy = _grid_origin()
    return gx, gy + GRID_PX + PAD


def _panel_origin() -> tuple[int, int]:
    gx, gy = _grid_origin()
    return gx + GRID_PX + PAD, gy


def draw_board(surface: pygame.Surface, board: Board, last_placed: np.ndarray | None = None) -> None:
    """Grid arka planı + dolu hücreler + son yerleştirme highlight."""
    gx, gy = _grid_origin()
    pygame.draw.rect(surface, GRID_BG, (gx - 4, gy - 4, GRID_PX + 8, GRID_PX + 8), border_radius=6)
    for r in range(8):
        for c in range(8):
            x = gx + c * CELL
            y = gy + r * CELL
            color = None
            if last_placed is not None and last_placed[r, c]:
                color = CELL_HIGHLIGHT
            elif board.grid[r, c]:
                color = CELL_FILLED
            if color is not None:
                pygame.draw.rect(surface, color, (x + 2, y + 2, CELL - 4, CELL - 4), border_radius=4)
            else:
                pygame.draw.rect(surface, GRID_LINE, (x + 1, y + 1, CELL - 2, CELL - 2), 1, border_radius=3)


def draw_tray(surface: pygame.Surface, tray: Tray, last_used_idx: int | None = None) -> None:
    """3 slot, her birinde piece şekli. Az önce kullanılan slot soluk renkte."""
    tx, ty = _tray_origin()
    for i in range(3):
        slot_x = tx + i * (TRAY_SLOT_W + TRAY_GAP)
        # Slot çerçevesi
        pygame.draw.rect(surface, GRID_BG, (slot_x, ty, TRAY_SLOT_W, TRAY_PX_H), border_radius=6)
        piece = tray.get(i)
        if piece is None:
            continue
        color = PIECE_DIM if i == last_used_idx else PIECE_COLOR
        # Parçayı slot içinde ortala
        pw = piece.width * TRAY_CELL
        ph = piece.height * TRAY_CELL
        ox = slot_x + (TRAY_SLOT_W - pw) // 2
        oy = ty + (TRAY_PX_H - ph) // 2
        for r in range(piece.height):
            for c in range(piece.width):
                if piece.cells[r, c]:
                    pygame.draw.rect(
                        surface, color,
                        (ox + c * TRAY_CELL + 2, oy + r * TRAY_CELL + 2, TRAY_CELL - 4, TRAY_CELL - 4),
                        border_radius=3,
                    )


def draw_panel(
    surface: pygame.Surface,
    fonts: dict,
    *,
    steps: int,
    score: int,
    cleared_total: int,
    last_clears: int,
    last_action: tuple[int, int, int] | None,
    game_over: bool,
    game_count: int,
    ep_len_history: list[int],
) -> None:
    """Sağ panel: tüm istatistikler."""
    px, py = _panel_origin()
    big = fonts["big"]
    med = fonts["med"]
    sm = fonts["sm"]

    title = big.render("Block Blast AI", True, TEXT)
    surface.blit(title, (px, py))
    sub = sm.render("v2_parallel · 10M step", True, TEXT_DIM)
    surface.blit(sub, (px, py + title.get_height() + 2))

    y = py + title.get_height() + 30

    def kv(label: str, value: str, color=TEXT) -> None:
        nonlocal y
        l = sm.render(label, True, TEXT_DIM)
        v = med.render(value, True, color)
        surface.blit(l, (px, y))
        surface.blit(v, (px, y + l.get_height() + 2))
        y += l.get_height() + v.get_height() + 12

    kv("Adım (steps)", str(steps))
    kv("Skor (info-only)", str(score))
    kv("Toplam clear", str(cleared_total))
    kv("Bu hamle clear", f"+{last_clears}" if last_clears > 0 else "0")

    if last_action is not None:
        pi, x, ya = last_action
        kv("Son hamle", f"piece#{pi} → ({x},{ya})")

    if ep_len_history:
        last5 = ep_len_history[-5:]
        avg = sum(last5) / len(last5)
        kv(f"Son {len(last5)} ep ort", f"{avg:.1f}")
        kv(f"Toplam oyun", str(game_count))

    if game_over:
        go = big.render("GAME OVER", True, GAME_OVER)
        surface.blit(go, (px, WINDOW_H - PAD - go.get_height() - 30))
        hint = sm.render("Yeni oyun başlıyor…", True, TEXT_DIM)
        surface.blit(hint, (px, WINDOW_H - PAD - hint.get_height() - 6))


def init_window(caption: str = "Block Blast AI") -> tuple[pygame.Surface, dict, pygame.time.Clock]:
    pygame.init()
    pygame.display.set_caption(caption)
    surface = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    fonts = {
        "big": pygame.font.SysFont("DejaVu Sans", 26, bold=True),
        "med": pygame.font.SysFont("DejaVu Sans", 22, bold=True),
        "sm": pygame.font.SysFont("DejaVu Sans", 14),
    }
    clock = pygame.time.Clock()
    return surface, fonts, clock
