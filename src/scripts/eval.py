"""Eval scripti — N episode oynat, istatistik döndür.

Kullanım:
    python -m src.scripts.eval --model models/best/ppo/best_model.zip --episodes 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from src.env.env import BlockBlastEnv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default="configs/env.yaml")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--deterministic", action="store_true", default=True)
    args = parser.parse_args()

    env = BlockBlastEnv(config_path=args.config, seed=args.seed)
    model = MaskablePPO.load(args.model, device="auto")

    lengths: list[int] = []
    scores: list[int] = []
    cleared: list[int] = []
    rewards: list[float] = []

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        done = False
        ep_reward = 0.0
        ep_length = 0
        ep_lines = 0
        while not done:
            mask = info["action_mask"]
            action, _ = model.predict(obs, action_masks=mask, deterministic=args.deterministic)
            obs, r, term, trunc, info = env.step(int(action))
            ep_reward += float(r)
            ep_length += 1
            ep_lines += int(info.get("cleared_lines", 0))
            done = term or trunc
        lengths.append(ep_length)
        scores.append(int(info.get("score", 0)))
        cleared.append(ep_lines)
        rewards.append(ep_reward)

    L = np.array(lengths); S = np.array(scores); C = np.array(cleared); R = np.array(rewards)
    # Hamle başı clear: global (toplam clear / toplam hamle) ve episode-bazlı ortalama
    total_steps = int(L.sum())
    total_clears = int(C.sum())
    clears_per_move_global = total_clears / max(1, total_steps)
    # Episode-bazlı ortalama (kısa episode'da yüksek oran daha çok ağırlık almasın diye)
    per_ep_rate = np.where(L > 0, C / np.maximum(L, 1), 0.0)
    clears_per_move_ep_mean = float(per_ep_rate.mean())

    print(f"\n=== Eval ({args.episodes} episodes) ===")
    print(f"length         | mean={L.mean():.1f}  median={np.median(L):.0f}  max={L.max()}  min={L.min()}")
    print(f"score          | mean={S.mean():.1f}  median={np.median(S):.0f}  max={S.max()}")
    print(f"clears/ep      | mean={C.mean():.2f}  median={np.median(C):.0f}  max={C.max()}")
    print(f"clears/move    | global={clears_per_move_global:.4f}  ep_mean={clears_per_move_ep_mean:.4f}")
    print(f"reward/ep      | mean={R.mean():.2f}  median={np.median(R):.2f}")
    print(f"totals         | episodes={args.episodes}  steps={total_steps:,}  clears={total_clears:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
