"""SB3 entegrasyon smoke test.

MaskablePPO modelini env üzerinde inşa edebiliyor muyuz, küçük bir learn() çalışıyor mu,
predict + step döngüsü hatasız mı?

Bu test biraz ağır (torch + SB3 import), ama eğitim öncesi tek seferlik garanti.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sb3_contrib = pytest.importorskip("sb3_contrib")
from sb3_contrib import MaskablePPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from src.ai.policy import BlockBlastFeatureExtractor  # noqa: E402
from src.env.env import BlockBlastEnv  # noqa: E402


def _make_env(seed: int):
    def _init():
        return BlockBlastEnv(config_path="configs/env.yaml", seed=seed)
    return _init


def test_model_build_and_short_learn():
    env = DummyVecEnv([_make_env(0)])
    policy_kwargs = dict(
        features_extractor_class=BlockBlastFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256, cnn_channels=(32, 64), history_embed_dim=16),
    )
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        learning_rate=3e-4,
        n_steps=64,
        batch_size=16,
        n_epochs=1,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=0,
        device="cpu",
        seed=0,
    )
    # Tek bir kısa rollout ve update — eğer build/forward hatası varsa burası yakalar.
    model.learn(total_timesteps=128, progress_bar=False)


def test_predict_step_loop():
    env = BlockBlastEnv(config_path="configs/env.yaml", seed=42)
    obs, info = env.reset(seed=42)
    policy_kwargs = dict(
        features_extractor_class=BlockBlastFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256, cnn_channels=(32, 64), history_embed_dim=16),
    )
    venv = DummyVecEnv([_make_env(43)])
    model = MaskablePPO(
        "MultiInputPolicy", venv,
        n_steps=64, batch_size=16, n_epochs=1,
        policy_kwargs=policy_kwargs, verbose=0, device="cpu", seed=0,
    )
    for _ in range(5):
        a, _ = model.predict(obs, action_masks=info["action_mask"], deterministic=True)
        obs, r, term, trunc, info = env.step(int(a))
        assert isinstance(r, float)
        if term or trunc:
            obs, info = env.reset()
