"""SB3 callback'leri.

`RewardComponentLogger`: JaxBlockBlastVecEnv (ya da uyumlu başka VecEnv)
infos[0]["rew_components"] alanından reward bileşenlerini okuyup
TensorBoard'a yazar. Her rollout sonu (n_steps × n_envs step) bir kez
ortalama yazar.

JaxBlockBlastVecEnv her step'te infos[0]["rew_components"] = {
    "step": float,           # batch ortalama r_step (game_over hariç)
    "line_clear": float,     # batch ortalama r_line_clear
    "terminal": float,       # batch ortalama r_terminal (sadece game_over'larda nonzero)
    "clears_step_mean": float,
    "clears_step_max": int,
} koyar; bu callback bu sözlüklerin running mean'ini tutar ve rollout
sonu TB'ye yazar.
"""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class RewardComponentLogger(BaseCallback):
    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self._reset_acc()

    def _reset_acc(self) -> None:
        self._sum_step = 0.0
        self._sum_lc = 0.0
        self._sum_term = 0.0
        self._sum_clears = 0.0
        self._max_clears = 0
        self._n_steps = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not infos:
            return True
        comp = infos[0].get("rew_components")
        if comp is None:
            return True
        self._sum_step += float(comp["step"])
        self._sum_lc += float(comp["line_clear"])
        self._sum_term += float(comp["terminal"])
        self._sum_clears += float(comp["clears_step_mean"])
        cm = int(comp["clears_step_max"])
        if cm > self._max_clears:
            self._max_clears = cm
        self._n_steps += 1
        return True

    def _on_rollout_end(self) -> None:
        if self._n_steps == 0:
            return
        n = self._n_steps
        self.logger.record("rew/step_bonus", self._sum_step / n)
        self.logger.record("rew/line_clear", self._sum_lc / n)
        self.logger.record("rew/terminal", self._sum_term / n)
        self.logger.record("rew/clears_step_mean", self._sum_clears / n)
        self.logger.record("rew/clears_step_max", self._max_clears)
        self._reset_acc()
