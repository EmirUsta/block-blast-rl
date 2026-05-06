"""MaskablePPO eğitim scripti — sade.

Kullanım:
    python -m src.scripts.train --config configs/ppo.yaml
    python -m src.scripts.train --config configs/ppo.yaml --total-timesteps 500000 --run-name run1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from src.ai.policy import BlockBlastFeatureExtractor
from src.env.batch_env import BlockBlastBatchEnv
from src.env.env import BlockBlastEnv


def _make_env(config_path: str, seed: int):
    def _init() -> BlockBlastEnv:
        return BlockBlastEnv(config_path=config_path, seed=seed)
    return _init


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ppo.yaml")
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help="config'i override eder")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--continue-from", default=None,
                        help="Önceden kaydedilmiş .zip checkpoint")
    parser.add_argument("--use-subproc", action="store_true",
                        help="SubprocVecEnv kullan (default: DummyVecEnv)")
    parser.add_argument("--use-batch", action="store_true",
                        help="BlockBlastBatchEnv (tek-process vectorized) kullan; --use-subproc'i ezer")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.total_timesteps is not None:
        cfg["total_timesteps"] = int(args.total_timesteps)

    run_name = args.run_name or Path(args.config).stem
    n_envs = int(cfg["n_envs"])
    seed = int(cfg.get("seed", 42))

    eval_fns = [_make_env(cfg["env_config"], seed + 1000 + i) for i in range(2)]

    if args.use_batch:
        train_env = BlockBlastBatchEnv(num_envs=n_envs, config_path=cfg["env_config"], seed=seed)
        eval_env = DummyVecEnv(eval_fns)
    elif args.use_subproc:
        env_fns = [_make_env(cfg["env_config"], seed + i) for i in range(n_envs)]
        train_env = SubprocVecEnv(env_fns)
        eval_env = SubprocVecEnv(eval_fns)
    else:
        env_fns = [_make_env(cfg["env_config"], seed + i) for i in range(n_envs)]
        train_env = DummyVecEnv(env_fns)
        eval_env = DummyVecEnv(eval_fns)

    pkw = cfg.get("policy_kwargs", {})
    policy_kwargs = dict(
        features_extractor_class=BlockBlastFeatureExtractor,
        features_extractor_kwargs=dict(
            features_dim=int(pkw.get("features_dim", 256)),
            cnn_channels=tuple(pkw.get("cnn_channels", [32, 64])),
            history_embed_dim=int(pkw.get("history_embed_dim", 16)),
        ),
    )

    tb_log = cfg.get("tensorboard_log", "logs/tensorboard")
    Path(tb_log).mkdir(parents=True, exist_ok=True)

    if args.continue_from and Path(args.continue_from).exists():
        print(f"Continuing from: {args.continue_from}")
        model = MaskablePPO.load(
            args.continue_from,
            env=train_env,
            tensorboard_log=tb_log,
            device=cfg.get("device", "auto"),
        )
    else:
        model = MaskablePPO(
            cfg["policy"],
            train_env,
            learning_rate=float(cfg["learning_rate"]),
            n_steps=int(cfg["n_steps"]),
            batch_size=int(cfg["batch_size"]),
            n_epochs=int(cfg["n_epochs"]),
            gamma=float(cfg["gamma"]),
            gae_lambda=float(cfg["gae_lambda"]),
            clip_range=float(cfg["clip_range"]),
            ent_coef=float(cfg["ent_coef"]),
            vf_coef=float(cfg["vf_coef"]),
            max_grad_norm=float(cfg["max_grad_norm"]),
            target_kl=float(cfg["target_kl"]),
            policy_kwargs=policy_kwargs,
            tensorboard_log=tb_log,
            verbose=int(cfg.get("verbose", 1)),
            seed=seed,
            device=cfg.get("device", "auto"),
        )

    ckpt_dir = Path(f"models/checkpoints/{run_name}")
    best_dir = Path(f"models/best/{run_name}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        CheckpointCallback(
            save_freq=max(1, int(cfg.get("checkpoint_freq", 100_000)) // n_envs),
            save_path=str(ckpt_dir),
            name_prefix="ppo",
            save_replay_buffer=False,
        ),
        MaskableEvalCallback(
            eval_env,
            best_model_save_path=str(best_dir),
            log_path=str(best_dir),
            eval_freq=max(1, int(cfg.get("eval_freq", 25_000)) // n_envs),
            n_eval_episodes=int(cfg.get("n_eval_episodes", 20)),
            deterministic=True,
            render=False,
        ),
    ]

    print(f"Run: {run_name}")
    print(f"Total timesteps: {int(cfg['total_timesteps']):,}")
    print(f"n_envs: {n_envs} | n_steps: {cfg['n_steps']} | batch: {cfg['batch_size']}")
    print(f"gamma={cfg['gamma']} | ent={cfg['ent_coef']} | vf={cfg['vf_coef']}")

    model.learn(
        total_timesteps=int(cfg["total_timesteps"]),
        callback=callbacks,
        tb_log_name=run_name,
        progress_bar=True,
    )

    final_path = ckpt_dir / "final.zip"
    model.save(str(final_path))
    print(f"Final model: {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
