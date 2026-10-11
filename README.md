# Block Blast RL: Survival-First PPO

A MaskablePPO agent for an 8x8 Block Blast clone, trained with a deliberately minimal reward: **+1 for every placed piece, -10 on game over**. This is the first version of the project; the faster JAX rewrite with AlphaZero-style agents lives in [block-blast-jax-rl](https://github.com/EmirUsta/block-blast-jax-rl).

## Key results

1000-episode evaluation, episode length = pieces placed before game over.

| Model | Change | Ep. length | Clears / ep | Score |
|---|---|---|---|---|
| v1 (2M steps) | baseline | 18.4 | 3.27 | 103.5 |
| v2 (10M) | longer training | 21.3 | 4.67 | 129.1 |
| v3 (10M) | architecture + 3 reward-shaping terms + batch | 19.3 | 4.11 | 115.2 |
| v4a (10M) | architecture only (early fusion CNN, 53% fewer params) | 21.9 | 4.96 | 134.3 |
| v4b (10M) | v4a + "balanced" reward shaping | 17.1 | 2.83 | 93.9 |
| v5 (10M) | vectorized NumPy batch env | 22.5 | 5.20 | 139.2 |
| **v6b (10M)** | **JAX GPU env, rollout horizon recalibrated** | **22.6** | **5.28** | **140.7** |
| v7 (10M) | v6b + soft line-clear bonus | 21.1 | 4.64 | 128.5 |

What the ablations showed:

- **Reward shaping hurt every time.** Hole, bumpiness and line-clear bonuses made the agent play greedy short games (v3, v4b, v7). The plain survival reward won.
- **Architecture helped.** Early fusion of the grid with placement masks and a stride-2 bottleneck cut parameters by 53% and improved all metrics (v4a).
- **Simulation speed:** profiling showed subprocess IPC dominated, so I wrote a vectorized batch env (training 46 min -> 23.5 min) and then a GPU-resident JAX env (~29x raw throughput, 208K steps/s at 2048 envs on an RTX 4050).
- **Horizon matters:** the first JAX run regressed because `n_steps` (16) was shorter than an average episode (~22). Recalibrating to 64 fixed it and gave the best model (v6b).

Every backend change was checked with parity tests (10K random states, identical action masks) and KS tests on episode-length distributions.

## Setup

```bash
git clone https://github.com/EmirUsta/block-blast-rl.git
cd block-blast-rl
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -q
python -m src.scripts.train --config configs/ppo_v6b_jax.yaml
python -m src.scripts.eval --model <path/to/model.zip> --episodes 1000
```

The JAX env needs a JAX install (`pip install -U "jax[cuda12]"` for GPU). Trained models are not included.

## Repository

| Path | Content |
|---|---|
| `src/game/` | Game logic: board, 31-piece set, tray refill |
| `src/env/` | Gymnasium env, action masking, NumPy batch env, JAX env and SB3 wrapper |
| `src/ai/` | Custom policy network and training callbacks |
| `src/scripts/` | Train, eval, play and live dashboard |
| `configs/` | Env and PPO configs for every version above |
| `tests/` | Smoke, masking, holes and backend parity tests |
| `examples/` | Throughput benchmarks |

Design notes and the full experiment log (in Turkish): [DESIGN.md](DESIGN.md), [DOCS.md](DOCS.md), [docs/log.md](docs/log.md), [docs/architecture.md](docs/architecture.md).
