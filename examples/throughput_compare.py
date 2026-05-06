"""SubprocVec16 vs BatchEnv16 vs BatchEnv64 vs BatchEnv128 throughput karşılaştırması.

Kullanım:
    python -m examples.throughput_compare
    python -m examples.throughput_compare --duration 30 --config configs/env.yaml

Her konfigürasyon için 60 sn random-policy step koşar (legal action mask
respect edilir), fps tablosu basar. SubprocVec ile drop-in equivalence
(BatchEnv VecEnv arayüzünü tam destekliyor mu) gözlenir.
"""

from __future__ import annotations

import argparse
import os
import time
from contextlib import contextmanager

import numpy as np
import psutil
import pynvml
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.env.batch_env import BlockBlastBatchEnv
from src.env.env import BlockBlastEnv

# NVML init (lazy; if no GPU, gpu_util/mem reported as None)
try:
    pynvml.nvmlInit()
    _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    _NVML_OK = True
except Exception:
    _NVML_HANDLE = None
    _NVML_OK = False


def _proc_tree_rss_mb() -> int:
    """Sum RSS (MB) of current process and all children."""
    p = psutil.Process(os.getpid())
    rss = p.memory_info().rss
    for c in p.children(recursive=True):
        try:
            rss += c.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return int(rss // (1024 * 1024))


def _gpu_snapshot() -> tuple[int | None, int | None]:
    if not _NVML_OK:
        return (None, None)
    try:
        u = pynvml.nvmlDeviceGetUtilizationRates(_NVML_HANDLE)
        m = pynvml.nvmlDeviceGetMemoryInfo(_NVML_HANDLE)
        return (int(u.gpu), int(m.used // (1024 * 1024)))
    except Exception:
        return (None, None)


def make_subproc(num_envs: int, config_path: str, seed: int) -> SubprocVecEnv:
    def _maker(rank: int):
        def _init():
            return BlockBlastEnv(config_path=config_path, seed=seed + rank)
        return _init

    return SubprocVecEnv(
        [_maker(i) for i in range(num_envs)], start_method="forkserver"
    )


def random_step_loop(
    venv,
    duration_sec: float,
    rng: np.random.Generator,
) -> dict:
    """Run random-policy step loop for `duration_sec`. Returns metrics dict
    including peak RAM and GPU util/mem snapshots."""
    n = venv.num_envs
    venv.reset()
    steps = 0
    eps = 0
    rss_peak_mb = 0
    gpu_util_samples: list[int] = []
    gpu_mem_samples: list[int] = []
    last_meas = 0.0
    t0 = time.perf_counter()
    while True:
        if hasattr(venv, "action_masks"):
            masks = venv.action_masks()
        else:
            masks = np.stack(venv.env_method("action_masks"))
        priority = rng.random((n, masks.shape[1])) * masks.astype(np.float32)
        actions = priority.argmax(axis=1).astype(np.int64)
        _, _, dones, _ = venv.step(actions)
        steps += n
        eps += int(dones.sum())
        now = time.perf_counter()
        # Sample resource usage every ~1s
        if now - last_meas >= 1.0:
            rss = _proc_tree_rss_mb()
            if rss > rss_peak_mb:
                rss_peak_mb = rss
            gu, gm = _gpu_snapshot()
            if gu is not None:
                gpu_util_samples.append(gu)
                gpu_mem_samples.append(gm)
            last_meas = now
        if now - t0 >= duration_sec:
            break
    elapsed = time.perf_counter() - t0
    return {
        "steps": steps,
        "eps": eps,
        "elapsed": elapsed,
        "fps_total": steps / elapsed,
        "fps_per_env": (steps / elapsed) / n,
        "rss_peak_mb": rss_peak_mb,
        "gpu_util_avg": float(np.mean(gpu_util_samples)) if gpu_util_samples else None,
        "gpu_mem_max_mb": int(max(gpu_mem_samples)) if gpu_mem_samples else None,
    }


@contextmanager
def banner(title: str):
    print(f"\n=== {title} ===")
    yield
    print(f"=== /{title} ===")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/env.yaml")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="seconds per benchmark scenario")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenarios", default="subproc16,batch16,batch64,batch128",
                        help="comma-separated subset of "
                             "[subproc16,batch16,batch64,batch128]")
    args = parser.parse_args()

    scenarios = args.scenarios.split(",")
    rng = np.random.default_rng(args.seed)

    print(f"Block Blast VecEnv throughput compare")
    print(f"  config:        {args.config}")
    print(f"  duration/run:  {args.duration:.0f} s")
    print(f"  seed:          {args.seed}")
    print(f"  scenarios:     {scenarios}")

    results = []  # (label, n, steps, eps, elapsed, fps_total, fps_per_env)

    def _run(label: str, venv, n: int) -> None:
        try:
            m = random_step_loop(venv, args.duration, rng)
            print(f"  steps={m['steps']:,}  eps={m['eps']:,}  elapsed={m['elapsed']:.2f}s")
            print(
                f"  TOTAL fps={m['fps_total']:,.0f}  per_env={m['fps_per_env']:,.0f}  "
                f"RSS_peak={m['rss_peak_mb']} MB  "
                f"GPU_util_avg={m['gpu_util_avg']}  GPU_mem_max={m['gpu_mem_max_mb']} MB"
            )
            results.append((label, n, m))
        finally:
            venv.close()

    if "subproc16" in scenarios:
        with banner("SubprocVecEnv (n_envs=16)"):
            _run("SubprocVecEnv-16", make_subproc(16, args.config, args.seed), 16)

    # Generic batch{N} parsing
    for scen in scenarios:
        if not scen.startswith("batch"):
            continue
        try:
            n = int(scen.replace("batch", ""))
        except ValueError:
            continue
        with banner(f"BatchEnv-{n}"):
            _run(f"BatchEnv-{n}", BlockBlastBatchEnv(num_envs=n, config_path=args.config, seed=args.seed), n)

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------
    print()
    print("=" * 100)
    print(
        f"{'scenario':<22} {'N':>4} {'fps_tot':>10} {'fps/env':>9} "
        f"{'eps':>8} {'sec':>7} {'RSS_MB':>8} {'GPU%':>6} {'GPU_MB':>8}"
    )
    print("-" * 100)
    for label, n, m in results:
        gu = f"{m['gpu_util_avg']:.1f}" if m['gpu_util_avg'] is not None else "n/a"
        gm = f"{m['gpu_mem_max_mb']}" if m['gpu_mem_max_mb'] is not None else "n/a"
        print(
            f"{label:<22} {n:>4} {m['fps_total']:>10,.0f} {m['fps_per_env']:>9,.0f} "
            f"{m['eps']:>8,} {m['elapsed']:>7.2f} {m['rss_peak_mb']:>8} {gu:>6} {gm:>8}"
        )
    print("=" * 100)
    sub = next((r for r in results if r[0] == "SubprocVecEnv-16"), None)
    if sub is not None:
        for label, n, m in results:
            if label == "SubprocVecEnv-16":
                continue
            speedup = m["fps_total"] / sub[2]["fps_total"]
            print(f"  {label:<22} → speedup vs SubprocVec16: {speedup:.2f}×")


if __name__ == "__main__":
    main()
