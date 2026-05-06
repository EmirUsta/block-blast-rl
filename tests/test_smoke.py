"""Smoke testler — temel akışın çalıştığını doğrular.

- Pieces yüklenir, board place/clear çalışır
- Env reset/step döngüsü hatasız ilerler ve reward survival-only davranır
- Action mask shape ve content doğrulanır
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.env.encoding import (
    BOARD_SIZE,
    FRAME_STACK,
    HISTORY_LEN,
    MAX_PIECE_DIM,
    META_DIM,
    NUM_ACTIONS,
    NUM_PIECE_TYPES,
    TRAY_SIZE,
    decode_action,
    encode_action,
)
from src.env.env import BlockBlastEnv
from src.env.masking import build_action_mask
from src.game.game import Game, load_env_config

CONFIG = "configs/env.yaml"


def test_action_id_bijection():
    for pi in range(TRAY_SIZE):
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                a = encode_action(pi, x, y)
                pi2, x2, y2 = decode_action(a)
                assert (pi, x, y) == (pi2, x2, y2), f"mismatch at {(pi, x, y)}"
    assert encode_action(2, 7, 7) == NUM_ACTIONS - 1


def test_game_reset_and_step():
    cfg = load_env_config(CONFIG)
    g = Game(cfg, rng=np.random.default_rng(0))
    assert g.score == 0 and g.steps == 0 and not g.game_over
    # tray full?
    assert g.tray.slots_remaining() == TRAY_SIZE
    mask = build_action_mask(g)
    assert mask.shape == (NUM_ACTIONS,)
    assert mask.any(), "Initial board boş, en az 1 valid action olmalı"
    # bir tane valid action al
    a = int(np.where(mask)[0][0])
    pi, x, y = decode_action(a)
    res = g.step(pi, x, y)
    assert not res.invalid
    assert g.steps == 1


def test_env_observation_shapes():
    env = BlockBlastEnv(config_path=CONFIG, seed=0)
    obs, info = env.reset(seed=0)
    assert obs["grid"].shape == (FRAME_STACK, BOARD_SIZE, BOARD_SIZE)
    assert obs["placements"].shape == (TRAY_SIZE, BOARD_SIZE, BOARD_SIZE)
    assert obs["tray"].shape == (TRAY_SIZE, NUM_PIECE_TYPES)
    assert obs["tray_shapes"].shape == (TRAY_SIZE, MAX_PIECE_DIM, MAX_PIECE_DIM)
    assert obs["history"].shape == (HISTORY_LEN,)
    assert obs["meta"].shape == (META_DIM,)
    assert info["action_mask"].shape == (NUM_ACTIONS,)
    # placements her piece için 64 hücre — action_mask'in (3,8,8) reshape'i olmalı
    flat = obs["placements"].astype(bool).reshape(-1)
    assert (flat == info["action_mask"]).all()


def test_env_step_reward_survival():
    """Geçerli bir adım reward=+1 vermeli (default config)."""
    env = BlockBlastEnv(config_path=CONFIG, seed=0)
    obs, info = env.reset(seed=0)
    mask = info["action_mask"]
    a = int(np.where(mask)[0][0])
    obs, r, term, trunc, info = env.step(a)
    assert r == pytest.approx(1.0), f"survival reward beklenir, alındı: {r}"
    assert not term and not trunc


def test_env_action_history_updates():
    env = BlockBlastEnv(config_path=CONFIG, seed=1)
    obs, info = env.reset(seed=1)
    a = int(np.where(info["action_mask"])[0][0])
    obs, _, _, _, _ = env.step(a)
    # history'nin son elemanı bizim aldığımız action olmalı
    assert int(obs["history"][-1]) == a


def test_env_frame_stack_advances():
    env = BlockBlastEnv(config_path=CONFIG, seed=2)
    obs0, info = env.reset(seed=2)
    g0 = obs0["grid"][-1].copy()
    a = int(np.where(info["action_mask"])[0][0])
    obs1, _, _, _, _ = env.step(a)
    g_now = obs1["grid"][-1]
    # En azından bir hücre değişmiş olmalı (yerleştirme ya da clear sonrası)
    assert not np.array_equal(g_now, g0)


def test_invalid_action_terminates():
    env = BlockBlastEnv(config_path=CONFIG, seed=3)
    obs, info = env.reset(seed=3)
    invalid = int(np.where(~info["action_mask"])[0][0])
    obs, r, term, trunc, info = env.step(invalid)
    assert term, "Invalid action terminate etmeli"
    assert r == pytest.approx(-1.0)


def test_pieces_yaml_exists():
    assert Path("configs/pieces.yaml").exists()
    assert Path("configs/env.yaml").exists()
    assert Path("configs/ppo.yaml").exists()
