"""PPO 200K real-loop throughput karşılaştırması: numpy BatchEnv vs JAX env.

3 config'i sıralı koşar (her biri rollout=32K SABİT):
  a) v5_batch64       : numpy BatchEnv, N=64,  n_steps=512, batch=1024
  b) v6_jax_512       : JAX env,        N=512, n_steps=64,  batch=1024
  c) v6_jax_2048      : JAX env,        N=2048,n_steps=16,  batch=1024

Çıktı: tablo (wallclock, ts/s, speedup, GPU avg/p95, GPU mem peak, RSS peak).

Kullanım:
    python -m examples.ppo_jax_compare --total-timesteps 200000
"""

from __future__ import annotations

import argparse
import gc
import os
import threading
import time
from pathlib import Path

# JAX env vars — JAX import'undan ÖNCE set edilmeli
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.4")

import numpy as np
import psutil
import pynvml
from sb3_contrib import MaskablePPO

from src.ai.policy import BlockBlastFeatureExtractor
from src.env.batch_env import BlockBlastBatchEnv

try:
    pynvml.nvmlInit()
    _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    _NVML_OK = True
except Exception:
    _NVML_OK = False


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


CFG_ENV = "configs/env.yaml"


class GPUSampler(threading.Thread):
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


def _make_venv(backend: str, n_envs: int, seed: int):
    if backend == "numpy":
        return BlockBlastBatchEnv(num_envs=n_envs, config_path=CFG_ENV, seed=seed)
    elif backend == "jax":
        # Lazy import to keep numpy-only paths import-clean
        from src.env.jax_vec_wrapper import JaxBlockBlastVecEnv
        return JaxBlockBlastVecEnv(num_envs=n_envs, config_path=CFG_ENV, seed=seed)
    raise ValueError(backend)


def run_once(label: str, backend: str, n_envs: int, n_steps: int, batch_size: int, total: int, seed: int) -> dict:
    rollout = n_envs * n_steps
    minibatches = rollout // batch_size
    print(f"\n=== {label} ({backend}) ===")
    print(f"  N={n_envs}  n_steps={n_steps}  batch={batch_size}  rollout={rollout:,}  minibatches/update={minibatches}  total={total:,}")

    venv = _make_venv(backend, n_envs, seed)

    pkw = dict(
        features_extractor_class=BlockBlastFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256, cnn_channels=(32, 64), history_embed_dim=16),
    )
    model = MaskablePPO(
        "MultiInputPolicy",
        venv,
        n_steps=n_steps,
        batch_size=batch_size,
        policy_kwargs=pkw,
        tensorboard_log=None,
        verbose=0,
        seed=seed,
        device="auto",
        **PPO_KW,
    )

    sampler = GPUSampler(0.5)
    sampler.start()
    t0 = time.perf_counter()
    model.learn(total_timesteps=total, progress_bar=False)
    elapsed = time.perf_counter() - t0
    sampler.stop()
    sampler.join(timeout=2.0)

    fps = total / elapsed
    gpu_util = np.asarray(sampler.gpu_util, dtype=np.float64) if sampler.gpu_util else np.zeros(0)
    gpu_mem = np.asarray(sampler.gpu_mem_mb, dtype=np.int32) if sampler.gpu_mem_mb else np.zeros(0, dtype=np.int32)
    rss = np.asarray(sampler.rss_mb, dtype=np.int32) if sampler.rss_mb else np.zeros(0, dtype=np.int32)

    venv.close()
    del model
    gc.collect()

    print(f"  wallclock: {elapsed:.1f}s   ts/s: {fps:,.0f}")
    if gpu_util.size:
        print(f"  GPU util: avg={gpu_util.mean():.1f}%  p95={np.percentile(gpu_util, 95):.0f}%  max={gpu_util.max():.0f}%")
    if gpu_mem.size:
        print(f"  GPU mem peak: {gpu_mem.max()} MB")
    if rss.size:
        print(f"  RSS peak: {rss.max()} MB")

    return {
        "label": label,
        "backend": backend,
        "n_envs": n_envs,
        "n_steps": n_steps,
        "batch": batch_size,
        "rollout": rollout,
        "wallclock_s": elapsed,
        "ts_s": fps,
        "gpu_avg": float(gpu_util.mean()) if gpu_util.size else 0.0,
        "gpu_p95": float(np.percentile(gpu_util, 95)) if gpu_util.size else 0.0,
        "gpu_mem_peak": int(gpu_mem.max()) if gpu_mem.size else 0,
        "rss_peak": int(rss.max()) if rss.size else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-numpy", action="store_true")
    parser.add_argument("--only", default=None, help="Sadece bu label'ı koş (örn. v6_jax_2048)")
    args = parser.parse_args()

    configs = []
    if not args.skip_numpy:
        configs.append(("v5_batch64", "numpy", 64, 512, 1024))
    configs.extend([
        ("v6_jax_512",  "jax",   512,  64, 1024),
        ("v6_jax_2048", "jax",  2048,  16, 1024),
    ])
    if args.only:
        configs = [c for c in configs if c[0] == args.only]

    results = []
    for label, backend, n_envs, n_steps, bs in configs:
        try:
            r = run_once(label, backend, n_envs, n_steps, bs, args.total_timesteps, args.seed)
            results.append(r)
        except Exception as e:
            print(f"  FAILED: {e!r}")
            results.append({"label": label, "backend": backend, "n_envs": n_envs, "error": repr(e), "ts_s": 0.0})

    # Tablo
    print()
    print("=" * 110)
    print(f"{'Config':<14} {'Backend':<7} {'N':>5} {'n_steps':>8} {'batch':>6} "
          f"{'wallclock':>10} {'ts/s':>10} {'speedup':>8} "
          f"{'gpu_avg%':>9} {'gpu_p95%':>9} {'gpu_mem_MB':>11} {'rss_GB':>7}")
    print("-" * 110)
    base_ts = next((r["ts_s"] for r in results if r.get("label") == "v5_batch64"), None)
    for r in results:
        if "error" in r:
            print(f"{r['label']:<14} {r['backend']:<7} {r['n_envs']:>5} {'-':>8} {'-':>6} {'FAIL':>10} {'-':>10} {'-':>8} {'-':>9} {'-':>9} {'-':>11} {'-':>7}")
            continue
        sp = (r["ts_s"] / base_ts) if base_ts else 1.0
        rss_gb = r["rss_peak"] / 1024.0
        print(
            f"{r['label']:<14} {r['backend']:<7} {r['n_envs']:>5} {r['n_steps']:>8} {r['batch']:>6} "
            f"{r['wallclock_s']:>9.1f}s {r['ts_s']:>10,.0f} {sp:>7.2f}x "
            f"{r['gpu_avg']:>8.1f}% {r['gpu_p95']:>8.0f}% {r['gpu_mem_peak']:>9} MB {rss_gb:>5.2f} GB"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
