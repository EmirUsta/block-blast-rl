"""JaxBlockBlastVecEnv random-policy throughput sweep.

Kullanım:
    python -m examples.jax_throughput --duration 60 --sizes 64,256,512,2048,8192

Çıktı: her N için ts/s, GPU util avg/p95, GPU mem peak, RSS peak. JIT warmup
benchmark'a dahil değil.
"""

from __future__ import annotations

import argparse
import os
import threading
import time

# JAX env vars must be set BEFORE importing jax (XLA backend init time)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.4")

import numpy as np
import psutil
import pynvml

from src.env.jax_vec_wrapper import JaxBlockBlastVecEnv

try:
    pynvml.nvmlInit()
    _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    _NVML_OK = True
except Exception:
    _NVML_OK = False


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
                self.rss_mb.append(int(rss // (1024 * 1024)))
            except Exception:
                pass
            self._stop_evt.wait(self.interval)

    def stop(self) -> None:
        self._stop_evt.set()


def _random_masked_actions(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Vectorized: argmax(uniform * mask + (-inf) * ~mask). Game-over rows → 0."""
    g = rng.standard_normal(mask.shape).astype(np.float32)
    g = np.where(mask, g, -np.inf)
    no_legal = ~mask.any(axis=1)
    a = np.argmax(g, axis=1)
    a = np.where(no_legal, 0, a)
    return a.astype(np.int64)


def bench_n(n: int, duration: float, config: str, seed: int) -> dict:
    venv = JaxBlockBlastVecEnv(num_envs=n, config_path=config, seed=seed)
    rng = np.random.default_rng(seed)

    # Warmup: 5 steps to compile the jit graph
    print(f"  warmup (jit compile)...", flush=True, end=" ")
    t_warm = time.perf_counter()
    obs = venv.reset()
    for _ in range(5):
        mask = venv.action_masks()
        a = _random_masked_actions(mask, rng)
        venv.step_async(a)
        venv.step_wait()
    print(f"done in {time.perf_counter() - t_warm:.1f}s", flush=True)

    # Benchmark window
    sampler = GPUSampler(0.5)
    sampler.start()
    n_steps = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < duration:
        mask = venv.action_masks()
        a = _random_masked_actions(mask, rng)
        venv.step_async(a)
        venv.step_wait()
        n_steps += 1
    elapsed = time.perf_counter() - t0
    sampler.stop()
    sampler.join(timeout=2.0)

    fps = n_steps * n / elapsed
    gpu_util = np.asarray(sampler.gpu_util, dtype=np.float64) if sampler.gpu_util else np.zeros(0)
    gpu_mem = np.asarray(sampler.gpu_mem_mb, dtype=np.int32) if sampler.gpu_mem_mb else np.zeros(0, dtype=np.int32)
    rss = np.asarray(sampler.rss_mb, dtype=np.int32) if sampler.rss_mb else np.zeros(0, dtype=np.int32)

    venv.close()
    return {
        "n": n,
        "duration": elapsed,
        "n_steps": n_steps,
        "fps": fps,
        "gpu_avg": float(gpu_util.mean()) if gpu_util.size else 0.0,
        "gpu_p95": float(np.percentile(gpu_util, 95)) if gpu_util.size else 0.0,
        "gpu_max": float(gpu_util.max()) if gpu_util.size else 0.0,
        "gpu_mem_peak": int(gpu_mem.max()) if gpu_mem.size else 0,
        "rss_peak": int(rss.max()) if rss.size else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/env.yaml")
    parser.add_argument("--sizes", default="64,256,512,2048,8192")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    print(
        f"=== JAX throughput sweep ===\n"
        f"  config:   {args.config}\n"
        f"  duration: {args.duration:.0f}s per N\n"
        f"  N values: {sizes}\n"
        f"  seed:     {args.seed}\n"
    )

    results = []
    for n in sizes:
        print(f"--- N = {n} ---", flush=True)
        try:
            r = bench_n(n, args.duration, args.config, args.seed)
            results.append(r)
            print(
                f"  ts/s:        {r['fps']:>10,.0f}\n"
                f"  steps:       {r['n_steps']:>10,}\n"
                f"  GPU util:    avg={r['gpu_avg']:.1f}%  p95={r['gpu_p95']:.0f}%  max={r['gpu_max']:.0f}%\n"
                f"  GPU mem peak: {r['gpu_mem_peak']} MB\n"
                f"  RSS peak:     {r['rss_peak']} MB\n",
                flush=True,
            )
        except Exception as e:  # OOM / device error → skip
            print(f"  FAILED: {e!r}\n", flush=True)
            results.append({"n": n, "fps": 0.0, "error": repr(e)})

    print("\n=== SUMMARY ===")
    print(f"{'N':>6} {'ts/s':>12} {'GPU avg':>9} {'GPU p95':>9} {'GPU mem':>9} {'RSS':>9}")
    for r in results:
        if "error" in r:
            print(f"{r['n']:>6} {'FAIL':>12} {'-':>9} {'-':>9} {'-':>9} {'-':>9}")
            continue
        print(
            f"{r['n']:>6} {r['fps']:>12,.0f} "
            f"{r['gpu_avg']:>8.1f}% {r['gpu_p95']:>8.0f}% "
            f"{r['gpu_mem_peak']:>7} MB {r['rss_peak']:>7} MB"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
