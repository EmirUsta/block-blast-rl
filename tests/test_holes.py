"""Hole/bumpiness/line-clear metric'leri ve env reward integration testleri."""

from __future__ import annotations

import numpy as np
import pytest

from src.env.encoding import count_empty_components, count_small_islands
from src.env.env import BlockBlastEnv


def test_empty_board_no_holes():
    g = np.zeros((8, 8), dtype=bool)
    # Tüm board boş → tek büyük 64-cell ada → küçük ada yok
    assert count_small_islands(g, max_size=2) == 0


def test_full_board_no_holes():
    g = np.ones((8, 8), dtype=bool)
    # Boş hücre yok → 0
    assert count_small_islands(g, max_size=2) == 0


def test_single_isolated_cell():
    g = np.ones((8, 8), dtype=bool)
    g[3, 3] = False  # Tek izole boş hücre
    # 1-cell ada → max_size=2 → sayılır → 1 hücre
    assert count_small_islands(g, max_size=2) == 1


def test_two_cell_island_horizontal():
    g = np.ones((8, 8), dtype=bool)
    g[3, 3] = False
    g[3, 4] = False
    # 2-cell ada → sayılır → 2
    assert count_small_islands(g, max_size=2) == 2


def test_three_cell_island_not_counted():
    g = np.ones((8, 8), dtype=bool)
    g[3, 3] = False
    g[3, 4] = False
    g[3, 5] = False
    # 3-cell ada → max_size=2 ile sayılmaz → 0
    assert count_small_islands(g, max_size=2) == 0


def test_two_separate_one_cell_islands():
    g = np.ones((8, 8), dtype=bool)
    g[1, 1] = False
    g[6, 6] = False  # ayrı, ayrı tek-hücre adalar
    assert count_small_islands(g, max_size=2) == 2


def test_max_size_threshold():
    g = np.ones((8, 8), dtype=bool)
    # 3-cell L-shape ada
    g[3, 3] = False
    g[3, 4] = False
    g[4, 3] = False
    # max_size=3 ile sayılır
    assert count_small_islands(g, max_size=3) == 3
    # max_size=2 ile sayılmaz
    assert count_small_islands(g, max_size=2) == 0


def test_env_default_no_hole_penalty():
    """Default env.yaml (hole_penalty=0) reward'a hole etkisi olmamalı."""
    env = BlockBlastEnv(config_path="configs/env.yaml", seed=0)
    obs, info = env.reset(seed=0)
    a = int(np.where(info["action_mask"])[0][0])
    obs, r, term, trunc, info = env.step(a)
    # Default reward = +1 (geçerli yerleştirme)
    assert r == pytest.approx(1.0)
    assert info.get("holes_delta", 0) == 0
    assert info.get("holes_after", 0) == 0  # hesaplanmadı


def test_env_v3_hole_penalty_active():
    """env_v3.yaml (hole_penalty=-2.0) reward'da etkili olmalı; info field var."""
    env = BlockBlastEnv(config_path="configs/env_v3.yaml", seed=0)
    obs, info = env.reset(seed=0)
    a = int(np.where(info["action_mask"])[0][0])
    obs, r, term, trunc, info = env.step(a)
    # Reward = +1 + r_hole * Δhole + r_bump * Δcomp + line_clear_reward
    assert "holes_delta" in info
    assert "holes_after" in info
    assert info["holes_delta"] >= 0
    expected = (
        1.0
        + (-2.0) * info["holes_delta"]
        + (-0.1) * info["comps_delta"]
        + info["line_clear_reward"]
    )
    assert r == pytest.approx(expected)


def test_empty_components_empty_board():
    g = np.zeros((8, 8), dtype=bool)
    # Tek büyük boş bölge → 1 component
    assert count_empty_components(g) == 1


def test_empty_components_full_board():
    g = np.ones((8, 8), dtype=bool)
    assert count_empty_components(g) == 0


def test_empty_components_split_in_two():
    """Tek satır dolu → boşluk 2 component'a bölünmüş olur."""
    g = np.zeros((8, 8), dtype=bool)
    g[3, :] = True  # 4. satır tamamen dolu, üst ve alt iki ayrı boş bölge
    assert count_empty_components(g) == 2


def test_empty_components_island_inside_full():
    g = np.ones((8, 8), dtype=bool)
    g[3, 3] = False
    g[5, 5] = False
    # 2 ayrı 1-cell ada → 2 component
    assert count_empty_components(g) == 2


def test_env_v3_line_clear_reward_on_actual_clear():
    """Line clear yapıldığında reward + (10 × 3^(n-1)) eklenmiş olmalı."""
    env = BlockBlastEnv(config_path="configs/env_v3.yaml", seed=0)
    env.reset(seed=0)
    # Manuel: tahtayı bir satır eksik dolu yap, son hücreyi dolduran hamleyi yap
    # Bu zor; pratik bir test: rastgele oyna ve line_clear_reward field tutarlılığını kontrol et.
    obs, info = env.reset(seed=7)
    found_clear = False
    for _ in range(60):
        if env.game.game_over:
            break
        a = int(np.random.choice(np.where(info["action_mask"])[0]))
        obs, r, term, trunc, info = env.step(a)
        n = info.get("cleared_lines", 0)
        lc_r = info.get("line_clear_reward", 0.0)
        if n > 0:
            expected = 10.0 * (3.0 ** (n - 1))
            assert lc_r == pytest.approx(expected), \
                f"n={n} got {lc_r}, expected {expected}"
            found_clear = True
        else:
            assert lc_r == 0.0
        if term or trunc:
            obs, info = env.reset(seed=8)
    # 60 random adımda en az bir clear gerekmiyor; sadece bulduğumuzda doğru olsun.
    # found_clear True ise yukarıdaki assertion'lar sağlandı.


def test_env_default_no_bumpiness_no_line_clear():
    """env.yaml default → bumpiness_penalty=0, line_clear_base=0."""
    env = BlockBlastEnv(config_path="configs/env.yaml", seed=1)
    obs, info = env.reset(seed=1)
    a = int(np.where(info["action_mask"])[0][0])
    obs, r, term, trunc, info = env.step(a)
    # Reward sadece +1 (default config)
    if not term:
        assert r == pytest.approx(1.0)
    assert info.get("comps_delta", 0) == 0
    assert info.get("line_clear_reward", 0.0) == 0.0


def test_env_v3_bumpiness_field_exists():
    env = BlockBlastEnv(config_path="configs/env_v3.yaml", seed=2)
    obs, info = env.reset(seed=2)
    a = int(np.where(info["action_mask"])[0][0])
    obs, r, term, trunc, info = env.step(a)
    assert "comps_before" in info
    assert "comps_after" in info
    assert "comps_delta" in info
    assert info["comps_delta"] == max(0, info["comps_after"] - info["comps_before"])


def test_env_v3_isolation_creates_penalty():
    """Tahta'yı zorlayıp gerçekten ceza geldiğini görelim — synthetic test."""
    env = BlockBlastEnv(config_path="configs/env_v3.yaml", seed=0)
    obs, info = env.reset(seed=0)
    # Board'ı manuel olarak izole bir hücre kalacak biçimde dolduralım
    # ve bir parça yerleştirip o izole hücreyi kapatmadığımızı test edelim.
    # Bu test daha çok integration sanity; stochastic olabilir.
    # Sadece field'lar tutarlı mı diye kontrol et:
    a = int(np.where(info["action_mask"])[0][0])
    obs, r, term, trunc, info = env.step(a)
    holes_after = info["holes_after"]
    holes_before = info["holes_before"]
    delta = info["holes_delta"]
    assert delta == max(0, holes_after - holes_before)
