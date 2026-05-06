"""Pygame'de AI canlı oynar.

Kullanım:
    python -m src.scripts.play_ai --model models/best/v2_parallel/best_model.zip
    python -m src.scripts.play_ai --fps 3 --deterministic

Tuşlar:
    R    — yeni oyun
    +/-  — hızı değiştir
    SPACE — duraklat / devam
    ESC  — çık
"""

from __future__ import annotations

import argparse
from time import time

import numpy as np
import pygame
from sb3_contrib import MaskablePPO

from src.env.encoding import decode_action
from src.env.env import BlockBlastEnv
from src.game.pieces import Piece
from src.ui import renderer as R


def _last_placed_mask(piece: Piece | None, x: int, y: int) -> np.ndarray | None:
    if piece is None:
        return None
    out = np.zeros((8, 8), dtype=bool)
    h, w = piece.height, piece.width
    out[y : y + h, x : x + w] = piece.cells
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/best/v2_parallel/best_model.zip")
    parser.add_argument("--config", default="configs/env.yaml")
    parser.add_argument("--fps", type=float, default=3.0,
                        help="kaç hamle / saniye")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--stochastic", action="store_true",
                        help="deterministic yerine sample")
    args = parser.parse_args()
    deterministic = not args.stochastic

    print(f"Loading model: {args.model}")
    model = MaskablePPO.load(args.model, device="auto")
    env = BlockBlastEnv(config_path=args.config, seed=args.seed)

    surface, fonts, clock = R.init_window("Block Blast AI · v2_parallel")

    obs, info = env.reset(seed=args.seed)
    last_placed_mask: np.ndarray | None = None
    last_action: tuple[int, int, int] | None = None
    last_clears: int = 0
    cleared_total: int = 0
    game_count: int = 1
    ep_len_history: list[int] = []
    paused = False
    fps = float(args.fps)

    t_last_step = time()
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_r:
                    obs, info = env.reset()
                    last_placed_mask = None
                    last_action = None
                    last_clears = 0
                    cleared_total = 0
                    game_count += 1
                elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    fps = min(60.0, fps * 1.5)
                    print(f"fps → {fps:.1f}")
                elif ev.key == pygame.K_MINUS:
                    fps = max(0.5, fps / 1.5)
                    print(f"fps → {fps:.1f}")
                elif ev.key == pygame.K_SPACE:
                    paused = not paused
                    print("paused" if paused else "resumed")

        # Hamle zamanı geldi mi?
        now = time()
        step_due = (now - t_last_step) >= (1.0 / fps)

        if not paused and step_due and not env.game.game_over:
            mask = info["action_mask"]
            action, _ = model.predict(obs, action_masks=mask, deterministic=deterministic)
            action = int(action)
            piece_idx, x, y = decode_action(action)
            piece = env.game.tray.get(piece_idx)
            last_placed_mask = _last_placed_mask(piece, x, y)
            last_action = (piece_idx, x, y)
            obs, r, term, trunc, info = env.step(action)
            last_clears = int(info.get("cleared_lines", 0))
            cleared_total += last_clears
            t_last_step = now
            if term or trunc:
                ep_len_history.append(env.game.steps)

        # Çiz
        surface.fill(R.BG)
        R.draw_board(surface, env.game.board, last_placed=last_placed_mask)
        R.draw_tray(
            surface, env.game.tray,
            last_used_idx=(last_action[0] if last_action is not None else None),
        )
        R.draw_panel(
            surface, fonts,
            steps=env.game.steps,
            score=env.game.score,
            cleared_total=cleared_total,
            last_clears=last_clears,
            last_action=last_action,
            game_over=env.game.game_over,
            game_count=game_count,
            ep_len_history=ep_len_history,
        )
        pygame.display.flip()

        # Game over → 2 sn bekle, yeni oyun
        if env.game.game_over and not paused:
            pygame.time.wait(1500)
            obs, info = env.reset()
            last_placed_mask = None
            last_action = None
            last_clears = 0
            cleared_total = 0
            game_count += 1
            t_last_step = time()

        clock.tick(60)  # 60 fps render, hamle hızı ayrı

    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
