"""MaskablePPO entegrasyon throughput ölçümü (gerçek rollout + gradient).

Kullanım:
    python -m examples.ppo_throughput \
        --n-envs 64 --n-steps 512 --batch-size 1024 \
        --total-timesteps 200000

Çıktı: wallclock, timesteps/sec, GPU util avg/p95, RAM peak.
TensorBoard log'u kapalı (saf throughput).
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import pynvml
import yaml
from sb3_contrib import MaskablePPO

from src.ai.policy import BlockBlastFeatureExtractor
from src.env.batch_env import BlockBlastBatchEnv

try:
    pynvml.nvmlInit()
    _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    _NVML_OK = True
except Exception:
    _NVML_OK = False


class GPUSampler(threading.Thread):
    """Background sampler: GPU util, GPU mem, process RSS — every interval."""

    def __init__(self, interval: float = 0.5) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self.gpu_util: list[int] = []
        self.gpu_mem_mb: list[int] = []
        self.rss_mb: list[int] = []
        self._stop_evt = threading.Event()

    def run(self) -> None:
        proc = psutil.Process(os.getpid())
        while not self._stop_evt.is_set():
            if _NVML_OK:
                try:
                    u = pynvml.nvmlDeviceGetUtilizationRates(_NVML_HANDLE)
                    m = pynvml.nvmlDeviceGetMemoryInfo(_NVML_HANDLE)
                    self.gpu_util.append(int(u.gpu))
                    self.gpu_mem_mb.append(int(m.used // (1024 * 1024)))
                except Exception:
                    pass
            try:
                rss = proc.memory_info().rss
                for c in proc.children(recursive=True):
                    try:
                        rss += c.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                self.rss_mb.append(int(rss // (1024 * 1024)))
            except Exception:
                pass
            self._stop_evt.wait(self.interval)

    def stop(self) -> None:
        self._stop_evt.set()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/env.yaml")
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    label = args.label or f"N{args.n_envs}_steps{args.n_steps}_bs{args.batch_size}"
    rollout_size = args.n_envs * args.n_steps
    minibatch_count = rollout_size // args.batch_size

    print(f"=== {label} ===")
    print(
        f"  N={args.n_envs}  n_steps={args.n_steps}  batch_size={args.batch_size}  "
        f"rollout={rollout_size:,}  minibatches/update={minibatch_count}  "
        f"total={args.total_timesteps:,}"
    )

    venv = BlockBlastBatchEnv(
        num_envs=args.n_envs, config_path=args.config, seed=args.seed
    )

    policy_kwargs = dict(
        features_extractor_class=BlockBlastFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256, cnn_channels=(32, 64), history_embed_dim=16),
    )

    model = MaskablePPO(
        "MultiInputPolicy",
        venv,
        learning_rate=3e-4,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.03,
        policy_kwargs=policy_kwargs,
        tensorboard_log=None,
        verbose=0,
        seed=args.seed,
        device="auto",
    )

    sampler = GPUSampler(interval=0.5)
    sampler.start()
    t0 = time.perf_counter()
    model.learn(total_timesteps=args.total_timesteps, progress_bar=False)
    elapsed = time.perf_counter() - t0
    sampler.stop()
    sampler.join(timeout=2.0)

    fps = args.total_timesteps / elapsed
    gpu_util = np.asarray(sampler.gpu_util, dtype=np.float64) if sampler.gpu_util else np.zeros(0)
    rss = np.asarray(sampler.rss_mb, dtype=np.int32) if sampler.rss_mb else np.zeros(0, dtype=np.int32)
    gpu_mem = np.asarray(sampler.gpu_mem_mb, dtype=np.int32) if sampler.gpu_mem_mb else np.zeros(0, dtype=np.int32)

    print()
    print(f"  wallclock:   {elapsed:.2f} s")
    print(f"  timesteps/s: {fps:,.0f}")
    if gpu_util.size:
        print(
            f"  GPU util:    avg={gpu_util.mean():.1f}%  p50={np.percentile(gpu_util, 50):.0f}%  "
            f"p95={np.percentile(gpu_util, 95):.0f}%  max={gpu_util.max():.0f}%"
        )
    if gpu_mem.size:
        print(f"  GPU mem:     peak={gpu_mem.max()} MB")
    if rss.size:
        print(f"  RSS:         peak={rss.max()} MB")
    print(f"  rollout_size={rollout_size:,}  minibatches/update={minibatch_count}")

    venv.close()
    # Output a one-line CSV for easy aggregation
    print()
    print("CSV: " + ",".join([
        label, str(args.n_envs), str(args.n_steps), str(args.batch_size),
        str(rollout_size), str(minibatch_count),
        f"{elapsed:.2f}", f"{fps:.0f}",
        f"{gpu_util.mean():.1f}" if gpu_util.size else "n/a",
        f"{np.percentile(gpu_util, 95):.0f}" if gpu_util.size else "n/a",
        str(int(rss.max())) if rss.size else "n/a",
        str(int(gpu_mem.max())) if gpu_mem.size else "n/a",
    ]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
