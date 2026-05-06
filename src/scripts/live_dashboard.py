"""Live training dashboard — pasif log okuyucu.

Pygame penceresinde 1 saniyede bir log dosyasını yeniden parse eder ve eğitim
metriklerini gösterir. Eğitime müdahale etmez, sadece izler.

Gösterilenler:
- Run name, timesteps progress (bar + %)
- fps, ETA, elapsed time
- Eval ep_len son 100 ölçüm — line graph
- Eval ep_reward son 100 ölçüm — line graph
- Son 5 eval satırı tablo
- v2/v4a baseline çizgileri (opsiyonel)

Kullanım:
    python -m src.scripts.live_dashboard --log logs/train_v4b.log
    python -m src.scripts.live_dashboard --log logs/train_v4b.log --total 10000000
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from time import time
from typing import Optional

import pygame

# Renkler
BG = (18, 20, 28)
PANEL = (28, 32, 42)
ACCENT = (90, 200, 255)
ACCENT_2 = (255, 200, 120)
TEXT = (220, 220, 230)
TEXT_DIM = (140, 140, 150)
GRID_LINE = (50, 55, 68)
PLOT_LINE = (90, 200, 255)
PLOT_LINE_2 = (255, 180, 100)
BAR_BG = (48, 54, 70)
BAR_FILL = (90, 200, 255)
GOOD = (120, 220, 140)
BAD = (255, 100, 110)
BASELINE_V2 = (180, 130, 220)
BASELINE_V4A = (130, 220, 180)

WINDOW_W = 1100
WINDOW_H = 680


@dataclass
class TrainStats:
    run_name: str = "?"
    total_timesteps: int = 0
    target_timesteps: int = 10_000_000
    fps: int = 0
    elapsed_sec: int = 0
    last_eval_ts: int = 0
    eval_ts: list[int] = None
    eval_ep_len: list[float] = None
    eval_ep_rew: list[float] = None
    last_lines: list[str] = None

    def __post_init__(self):
        if self.eval_ts is None: self.eval_ts = []
        if self.eval_ep_len is None: self.eval_ep_len = []
        if self.eval_ep_rew is None: self.eval_ep_rew = []
        if self.last_lines is None: self.last_lines = []


_RE_TOTAL = re.compile(r"total_timesteps\s*\|\s*(\d+)")
_RE_FPS = re.compile(r"\|\s*fps\s*\|\s*(\d+)")
_RE_ELAPSED = re.compile(r"\|\s*time_elapsed\s*\|\s*(\d+)")
_RE_RUN = re.compile(r"^Run:\s*(\S+)")
_RE_TARGET = re.compile(r"^Total timesteps:\s*([\d,_]+)")
_RE_EVAL_TS = re.compile(r"Eval num_timesteps=(\d+),\s*episode_reward=([+-]?[\d.]+)")
_RE_EVAL_LEN = re.compile(r"Episode length:\s*([\d.]+)")


def parse_log(path: str) -> TrainStats:
    s = TrainStats()
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except FileNotFoundError:
        return s

    if m := _RE_RUN.search(text):
        s.run_name = m.group(1)
    if m := _RE_TARGET.search(text):
        s.target_timesteps = int(m.group(1).replace(",", "").replace("_", ""))

    # Tüm timestep kayıtlarını bul, son fps + elapsed
    totals = _RE_TOTAL.findall(text)
    if totals:
        s.total_timesteps = int(totals[-1])
    fpss = _RE_FPS.findall(text)
    if fpss:
        s.fps = int(fpss[-1])
    elapseds = _RE_ELAPSED.findall(text)
    if elapseds:
        s.elapsed_sec = int(elapseds[-1])

    # Eval kayıtları (Eval num_timesteps + Episode length pair'leri)
    eval_ts_rew_pairs = _RE_EVAL_TS.findall(text)
    ep_lens = _RE_EVAL_LEN.findall(text)
    n = min(len(eval_ts_rew_pairs), len(ep_lens))
    if n > 0:
        s.eval_ts = [int(t) for t, _ in eval_ts_rew_pairs[:n]]
        s.eval_ep_rew = [float(r) for _, r in eval_ts_rew_pairs[:n]]
        s.eval_ep_len = [float(x) for x in ep_lens[:n]]
        s.last_eval_ts = s.eval_ts[-1]

    # Son 5 eval satırı (kullanıcıya tablo için)
    lines = text.splitlines()
    eval_blocks = []
    for i, ln in enumerate(lines):
        if "Eval num_timesteps=" in ln:
            block = [ln]
            if i + 1 < len(lines):
                block.append(lines[i + 1])
            eval_blocks.append(block)
    s.last_lines = eval_blocks[-5:]
    return s


def draw_progress_bar(surf, x, y, w, h, frac):
    pygame.draw.rect(surf, BAR_BG, (x, y, w, h), border_radius=4)
    fw = max(0, min(int(w * frac), w))
    if fw > 0:
        pygame.draw.rect(surf, BAR_FILL, (x, y, fw, h), border_radius=4)


def draw_line_chart(
    surf, x, y, w, h, values: list[float], xs: list[int],
    color, label: str, fonts, target_x: int | None = None,
    baselines: list[tuple[float, tuple[int, int, int], str]] | None = None,
):
    """Time-series line plot.

    values: y-axis. xs: x-axis (timesteps).
    target_x: opsiyonel, x-ekseni right-bound (örn 10M). None ise xs'in maxı.
    baselines: yatay referans çizgileri [(y_value, color, label), ...]
    """
    pygame.draw.rect(surf, PANEL, (x, y, w, h), border_radius=6)
    inner_pad = 32
    px, py = x + inner_pad, y + inner_pad - 6
    pw, ph = w - 2 * inner_pad, h - 2 * inner_pad

    surf.blit(fonts["sm"].render(label, True, TEXT), (x + 12, y + 8))

    if not values or not xs:
        msg = fonts["sm"].render("(no data yet)", True, TEXT_DIM)
        surf.blit(msg, (x + w // 2 - msg.get_width() // 2, y + h // 2))
        return

    vmin = min(values)
    vmax = max(values)
    if vmax == vmin:
        vmax = vmin + 1
    # Padding: %10 üst-alt
    span = vmax - vmin
    vmin_p = vmin - 0.1 * span
    vmax_p = vmax + 0.1 * span

    # X bound
    x_max = target_x if target_x else xs[-1]
    x_min = 0

    # Grid lines (yatay 4 adım)
    for i in range(5):
        gy = py + ph - int(ph * i / 4)
        pygame.draw.line(surf, GRID_LINE, (px, gy), (px + pw, gy), 1)
        gv = vmin_p + (vmax_p - vmin_p) * i / 4
        gtxt = fonts["xs"].render(f"{gv:.1f}", True, TEXT_DIM)
        surf.blit(gtxt, (x + 4, gy - 6))

    # Baselines
    if baselines:
        for yv, color_b, lbl in baselines:
            if vmin_p <= yv <= vmax_p:
                gy = py + ph - int(ph * (yv - vmin_p) / (vmax_p - vmin_p))
                pygame.draw.line(surf, color_b, (px, gy), (px + pw, gy), 1)
                bt = fonts["xs"].render(lbl, True, color_b)
                surf.blit(bt, (px + pw - bt.get_width() - 2, gy - 14))

    # Path
    pts = []
    for tv, vv in zip(xs, values):
        gx = px + int(pw * (tv - x_min) / max(1, x_max - x_min))
        gy = py + ph - int(ph * (vv - vmin_p) / (vmax_p - vmin_p))
        pts.append((gx, gy))
    if len(pts) >= 2:
        pygame.draw.lines(surf, color, False, pts, 2)
    for p in pts[-1:]:
        pygame.draw.circle(surf, color, p, 4)

    # Last value label
    last = fonts["md"].render(f"{values[-1]:.2f}", True, color)
    surf.blit(last, (x + w - last.get_width() - 12, y + 8))


def fmt_time(sec: int) -> str:
    if sec < 60: return f"{sec}s"
    if sec < 3600: return f"{sec//60}m {sec%60}s"
    return f"{sec//3600}h {(sec%3600)//60}m"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="logs/train_v4b.log")
    parser.add_argument("--total", type=int, default=10_000_000)
    parser.add_argument("--update-ms", type=int, default=1000)
    parser.add_argument(
        "--baseline-v2", type=float, default=21.3, help="v2 ep_len baseline"
    )
    parser.add_argument(
        "--baseline-v4a", type=float, default=21.9, help="v4a ep_len baseline"
    )
    args = parser.parse_args()

    pygame.init()
    pygame.display.set_caption(f"Live Dashboard — {args.log}")
    surf = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    fonts = {
        "xl": pygame.font.SysFont("DejaVu Sans", 28, bold=True),
        "lg": pygame.font.SysFont("DejaVu Sans", 22, bold=True),
        "md": pygame.font.SysFont("DejaVu Sans", 18, bold=True),
        "sm": pygame.font.SysFont("DejaVu Sans", 14),
        "xs": pygame.font.SysFont("DejaVu Sans", 11),
        "mono": pygame.font.SysFont("DejaVu Sans Mono", 13),
    }
    clock = pygame.time.Clock()

    last_parse = 0.0
    stats = TrainStats(target_timesteps=args.total)

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                running = False

        now = time()
        if now - last_parse >= args.update_ms / 1000:
            stats = parse_log(args.log)
            stats.target_timesteps = args.total
            last_parse = now

        surf.fill(BG)

        # Header
        title = fonts["xl"].render(stats.run_name, True, TEXT)
        surf.blit(title, (24, 16))
        sub = fonts["sm"].render(f"log: {args.log}", True, TEXT_DIM)
        surf.blit(sub, (24, 16 + title.get_height() + 2))

        # Progress
        frac = (stats.total_timesteps / max(1, stats.target_timesteps))
        pb_x, pb_y = 24, 80
        pb_w, pb_h = WINDOW_W - 48, 18
        draw_progress_bar(surf, pb_x, pb_y, pb_w, pb_h, frac)
        pct_txt = fonts["md"].render(
            f"{stats.total_timesteps:,} / {stats.target_timesteps:,}  ({frac*100:.1f}%)",
            True, TEXT,
        )
        surf.blit(pct_txt, (pb_x, pb_y - 22))

        # KPI row
        kpi_y = 110
        eta = (stats.target_timesteps - stats.total_timesteps) / max(1, stats.fps)
        kpis = [
            ("fps", f"{stats.fps:,}"),
            ("elapsed", fmt_time(stats.elapsed_sec)),
            ("ETA", fmt_time(int(eta))),
            ("eval count", str(len(stats.eval_ts))),
        ]
        kw = (WINDOW_W - 48) // len(kpis)
        for i, (k, v) in enumerate(kpis):
            x0 = 24 + i * kw
            pygame.draw.rect(surf, PANEL, (x0, kpi_y, kw - 8, 56), border_radius=6)
            kt = fonts["sm"].render(k, True, TEXT_DIM)
            vt = fonts["lg"].render(v, True, TEXT)
            surf.blit(kt, (x0 + 12, kpi_y + 6))
            surf.blit(vt, (x0 + 12, kpi_y + 22))

        # Charts
        ch_y = 180
        ch_w = (WINDOW_W - 48 - 16) // 2
        ch_h = 250

        # ep_len chart with baselines
        baselines_len = []
        if args.baseline_v2 > 0:
            baselines_len.append((args.baseline_v2, BASELINE_V2, f"v2={args.baseline_v2}"))
        if args.baseline_v4a > 0:
            baselines_len.append((args.baseline_v4a, BASELINE_V4A, f"v4a={args.baseline_v4a}"))

        draw_line_chart(
            surf, 24, ch_y, ch_w, ch_h,
            stats.eval_ep_len, stats.eval_ts,
            PLOT_LINE, "Eval Episode Length",
            fonts, target_x=stats.target_timesteps,
            baselines=baselines_len,
        )
        draw_line_chart(
            surf, 24 + ch_w + 16, ch_y, ch_w, ch_h,
            stats.eval_ep_rew, stats.eval_ts,
            PLOT_LINE_2, "Eval Episode Reward",
            fonts, target_x=stats.target_timesteps,
        )

        # Last 5 eval table
        tbl_y = ch_y + ch_h + 16
        tbl_h = WINDOW_H - tbl_y - 16
        pygame.draw.rect(surf, PANEL, (24, tbl_y, WINDOW_W - 48, tbl_h), border_radius=6)
        hdr = fonts["sm"].render("Last 5 evals", True, TEXT)
        surf.blit(hdr, (36, tbl_y + 8))
        n_show = min(5, len(stats.eval_ts))
        if n_show > 0:
            for i in range(n_show):
                idx = len(stats.eval_ts) - n_show + i
                ts = stats.eval_ts[idx]
                el = stats.eval_ep_len[idx]
                er = stats.eval_ep_rew[idx]
                # Color: v4a baseline'a göre
                color = GOOD if el >= args.baseline_v4a else (TEXT if el >= args.baseline_v2 else TEXT_DIM)
                row = f"{ts:>10,}  ep_len={el:>6.2f}   ep_rew={er:>+7.2f}"
                txt = fonts["mono"].render(row, True, color)
                surf.blit(txt, (40, tbl_y + 30 + i * 18))
        else:
            msg = fonts["sm"].render("(awaiting first eval...)", True, TEXT_DIM)
            surf.blit(msg, (40, tbl_y + 30))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
