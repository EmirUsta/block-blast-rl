"""500K BatchEnv vs SubprocVec eğitimleri sırasında episode-bazlı ep_length
örneklemleri toplayıp 50K-step pencerelerde KS-test uygular.

İki training paralel koşmaz; ardışık çalışır. Her training kendi `runs/`
dizinine episodelar.jsonl yazar (her episode için step_when_completed + length).
Sonra ks_window_compare.py --analyze ile KS pencere analizi yapılır.

Kullanım:
    # 1) BatchEnv eğitimi (500K)
    python -m examples.ks_window_compare --mode train --backend batch --total 500000 --out logs/ks_batch.jsonl
    # 2) SubprocVec eğitimi (500K)
    python -m examples.ks_window_compare --mode train --backend subproc --total 500000 --out logs/ks_subproc.jsonl
    # 3) Karşılaştırma
    python -m examples.ks_window_compare --mode analyze \
        --a logs/ks_batch.jsonl --b logs/ks_subproc.jsonl \
        --window 50000 --total 500000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.ai.policy import BlockBlastFeatureExtractor
from src.env.batch_env import BlockBlastBatchEnv
from src.env.env import BlockBlastEnv


CFG = "configs/env.yaml"

PPO_KW = dict(
    learning_rate=3e-4,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    target_kl=0.03,
)


# ---------------------------------------------------------------------------
# Custom callback — log every episode completion
# ---------------------------------------------------------------------------

class EpisodeLogger(BaseCallback):
    """Write JSONL of {step, length, reward} per finished episode."""

    def __init__(self, out_path: Path) -> None:
        super().__init__(verbose=0)
        self.out_path = out_path
        self.fp = None

    def _on_training_start(self) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.fp = open(self.out_path, "w")

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if ep is not None:
                self.fp.write(json.dumps({
                    "step": int(self.num_timesteps),
                    "length": int(ep["l"]),
                    "reward": float(ep["r"]),
                }) + "\n")
        return True

    def _on_training_end(self) -> None:
        if self.fp is not None:
            self.fp.flush()
            self.fp.close()


# ---------------------------------------------------------------------------
# train()
# ---------------------------------------------------------------------------

def make_subproc_vec(n_envs: int, seed: int) -> SubprocVecEnv:
    def _init(rank: int):
        def _f():
            # Wrap with Monitor so info["episode"]={r,l} is emitted on done.
            # BlockBlastEnv tarafından üretilmiyor; EpisodeLogger callback'i
            # bu alanı dinliyor.
            return Monitor(BlockBlastEnv(config_path=CFG, seed=seed + rank))
        return _f
    return SubprocVecEnv([_init(i) for i in range(n_envs)], start_method="forkserver")


def cmd_train(backend: str, total: int, out: str, n_envs: int, n_steps: int, batch_size: int, seed: int) -> None:
    if backend == "batch":
        venv = BlockBlastBatchEnv(num_envs=n_envs, config_path=CFG, seed=seed)
    elif backend == "subproc":
        venv = make_subproc_vec(n_envs, seed)
    else:
        raise ValueError(backend)

    policy_kwargs = dict(
        features_extractor_class=BlockBlastFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256, cnn_channels=(32, 64), history_embed_dim=16),
    )
    model = MaskablePPO(
        "MultiInputPolicy",
        venv,
        n_steps=n_steps,
        batch_size=batch_size,
        policy_kwargs=policy_kwargs,
        tensorboard_log=None,
        verbose=0,
        seed=seed,
        device="auto",
        **PPO_KW,
    )
    cb = EpisodeLogger(Path(out))
    print(f"[train.{backend}] N={n_envs} n_steps={n_steps} batch={batch_size} total={total:,} → {out}")
    t0 = time.perf_counter()
    model.learn(total_timesteps=total, callback=cb, progress_bar=False)
    el = time.perf_counter() - t0
    print(f"[train.{backend}] done in {el:.1f}s ({total/el:,.0f} ts/s)")
    venv.close()


# ---------------------------------------------------------------------------
# analyze()
# ---------------------------------------------------------------------------

def _ks_2samp(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    n_a, n_b = a.size, b.size
    if n_a == 0 or n_b == 0:
        return (1.0, 0.0)
    all_v = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, all_v, side="right") / n_a
    cdf_b = np.searchsorted(b, all_v, side="right") / n_b
    D = float(np.abs(cdf_a - cdf_b).max())
    en = np.sqrt(n_a * n_b / (n_a + n_b))
    lam = (en + 0.12 + 0.11 / en) * D
    j = np.arange(1, 101)
    p = 2.0 * float(np.sum(((-1.0) ** (j - 1)) * np.exp(-2.0 * (lam ** 2) * (j ** 2))))
    p = max(0.0, min(1.0, p))
    return D, p


def _load_jsonl(path: str) -> tuple[np.ndarray, np.ndarray]:
    steps: list[int] = []
    lengths: list[int] = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            steps.append(obj["step"])
            lengths.append(obj["length"])
    return np.asarray(steps, dtype=np.int64), np.asarray(lengths, dtype=np.int32)


def cmd_analyze(a_path: str, b_path: str, window: int, total: int) -> int:
    sa, la = _load_jsonl(a_path)
    sb, lb = _load_jsonl(b_path)
    print(f"[analyze] A: {a_path}  episodes={la.size}  steps_max={sa.max() if sa.size else 0}")
    print(f"[analyze] B: {b_path}  episodes={lb.size}  steps_max={sb.max() if sb.size else 0}")

    print()
    print(f"{'window':<22} {'n_a':>6} {'n_b':>6} {'mean_a':>8} {'mean_b':>8} "
          f"{'D':>7} {'p':>8} {'pass':>5}")
    print("-" * 80)
    bound_lo = 0
    pass_count = 0
    total_windows = 0
    while bound_lo < total:
        bound_hi = bound_lo + window
        ma_in = (sa >= bound_lo) & (sa < bound_hi)
        mb_in = (sb >= bound_lo) & (sb < bound_hi)
        a_lens = la[ma_in]
        b_lens = lb[mb_in]
        D, p = _ks_2samp(a_lens, b_lens)
        ok = (p > 0.05) and (a_lens.size >= 30) and (b_lens.size >= 30)
        flag = "OK" if ok else "X"
        print(
            f"[{bound_lo:>7,}, {bound_hi:>7,}) {a_lens.size:>6} {b_lens.size:>6} "
            f"{(a_lens.mean() if a_lens.size else 0.0):>8.2f} "
            f"{(b_lens.mean() if b_lens.size else 0.0):>8.2f} "
            f"{D:>7.4f} {p:>8.4f} {flag:>5}"
        )
        total_windows += 1
        if ok:
            pass_count += 1
        bound_lo = bound_hi
    print()
    pct = (pass_count / total_windows * 100) if total_windows else 0.0
    print(f"[analyze] PASS rate: {pass_count}/{total_windows} ({pct:.0f}%)")

    # Final 100K window mean diff
    final_lo = max(0, total - 100_000)
    a_final = la[(sa >= final_lo) & (sa < total)]
    b_final = lb[(sb >= final_lo) & (sb < total)]
    diff = float("inf")
    if a_final.size and b_final.size:
        diff = float(abs(a_final.mean() - b_final.mean()))
        print(f"[analyze] Final 100K mean: A={a_final.mean():.2f} B={b_final.mean():.2f}  |Δ|={diff:.2f}")
    else:
        print(f"[analyze] Final 100K window: no episodes one of: a={a_final.size} b={b_final.size}")

    target_pct = 80.0
    target_diff = 1.5
    target_diff_ok = (diff < target_diff)
    target_pct_ok = pct >= target_pct
    if target_pct_ok and target_diff_ok:
        print(f"[analyze] CRITERIA MET: {pct:.0f}% ≥ {target_pct:.0f}% AND |Δ| < {target_diff}")
        return 0
    diff_str = f"{diff:.2f}" if diff != float("inf") else "n/a"
    print(f"[analyze] CRITERIA FAILED: PASS%={pct:.0f}/{target_pct:.0f}  |Δ|={diff_str}/{target_diff}")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "analyze"], required=True)
    parser.add_argument("--backend", choices=["batch", "subproc"])
    parser.add_argument("--total", type=int, default=500_000)
    parser.add_argument("--out", default="logs/ks_episodes.jsonl")
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--a")
    parser.add_argument("--b")
    parser.add_argument("--window", type=int, default=50_000)
    args = parser.parse_args()

    if args.mode == "train":
        cmd_train(
            backend=args.backend,
            total=args.total,
            out=args.out,
            n_envs=args.n_envs,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        return 0
    return cmd_analyze(args.a, args.b, args.window, args.total)


if __name__ == "__main__":
    raise SystemExit(main())
