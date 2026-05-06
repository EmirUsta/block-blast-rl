"""SB3 VecEnv adapter for the JAX-backed Block Blast env.

Drop-in replacement for `BlockBlastBatchEnv` (numpy). PPO training loop'unda
`from src.env.jax_vec_wrapper import JaxBlockBlastVecEnv` ile geçilir.

Faz-1 (bu dosya): obs JAX device array → numpy host (`jax.device_get`).
Bu basit yol N=2048'de 100K+ ts/s hedefini tutar (random policy);
PPO loop'unda PyTorch policy zaten obs'u CUDA'ya yollar, host roundtrip
PyTorch'un device transfer'ine eşit maliyet.

Faz-2 (TODO): dlpack zero-copy `jax → torch.cuda` direkt köprü. Profile
sonrası %10+ kazanç varsa eklenir; yoksa karmaşıklığa değmez.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvIndices, VecEnvObs, VecEnvStepReturn

from src.env.encoding import (
    BOARD_SIZE,
    FRAME_STACK,
    HISTORY_LEN,
    HISTORY_SENTINEL,
    MAX_PIECE_DIM,
    META_DIM,
    NUM_ACTIONS,
    NUM_PIECE_TYPES,
    TRAY_SIZE,
)
from src.env.jax_env import (
    JaxState,
    _build_piece_tensor,
    cfg_from_yaml,
    make_reset_fn,
    make_step_fn,
)


class JaxBlockBlastVecEnv(VecEnv):
    """JAX-backed batched env with SB3 VecEnv interface."""

    def __init__(
        self,
        num_envs: int,
        config_path: str | Path,
        seed: int | None = None,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be > 0")

        # Config + piece tensor
        self.cfg = cfg_from_yaml(config_path)
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        pieces_path = Path(raw["pieces"]["config_file"])
        self.piece_cells_pad = _build_piece_tensor(pieces_path)

        # Observation / action spaces — numpy env ile birebir aynı
        observation_space = spaces.Dict(
            {
                "grid": spaces.Box(0.0, 1.0, (FRAME_STACK, BOARD_SIZE, BOARD_SIZE), dtype=np.float32),
                "placements": spaces.Box(0.0, 1.0, (TRAY_SIZE, BOARD_SIZE, BOARD_SIZE), dtype=np.float32),
                "tray": spaces.Box(0.0, 1.0, (TRAY_SIZE, NUM_PIECE_TYPES), dtype=np.float32),
                "tray_shapes": spaces.Box(0.0, 1.0, (TRAY_SIZE, MAX_PIECE_DIM, MAX_PIECE_DIM), dtype=np.float32),
                "history": spaces.Box(0, HISTORY_SENTINEL, (HISTORY_LEN,), dtype=np.int64),
                "meta": spaces.Box(0.0, 1.0, (META_DIM,), dtype=np.float32),
            }
        )
        action_space = spaces.Discrete(NUM_ACTIONS)
        super().__init__(num_envs, observation_space, action_space)

        # JIT'lenmiş step / reset
        self._step_fn = make_step_fn(self.cfg, self.piece_cells_pad)
        self._reset_fn = make_reset_fn(self.cfg, self.piece_cells_pad)

        # PRNG master key
        self._seed = int(seed) if seed is not None else 0
        master_key = jax.random.PRNGKey(self._seed)

        # Initial state
        self._state: JaxState = self._reset_fn(num_envs, master_key)
        # State'in device'a yerleştiğini garantile (jit cache warm)
        self._state.boards.block_until_ready()

        # Cached numpy mask (env_method('action_masks') ve action_masks() için)
        self._cached_mask_np: np.ndarray = np.asarray(self._state.cached_mask, dtype=bool)

        # step_async slot
        self._actions_jax: jnp.ndarray | None = None

    # ------------------------------------------------------------------
    # SB3 VecEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> VecEnvObs:
        master_key = self._state.master_key  # mevcut seed devam etsin
        # Yeni master key türet — reset deterministik olsun ama seed'den değil,
        # mevcut state'in master_key'inden ilerlesin.
        master_key = jax.random.fold_in(master_key, 0xC0FFEE)
        self._state = self._reset_fn(self.num_envs, master_key)
        self._state.boards.block_until_ready()
        self._cached_mask_np = np.asarray(self._state.cached_mask, dtype=bool)
        return self._build_obs_numpy(self._state)

    def step_async(self, actions: np.ndarray) -> None:
        self._actions_jax = jnp.asarray(actions, dtype=jnp.int32)

    def step_wait(self) -> VecEnvStepReturn:
        if self._actions_jax is None:
            raise RuntimeError("step_async() must be called before step_wait()")

        (
            new_state,
            post_obs_jax,
            pre_obs_jax,
            reward_jax,
            done_jax,
            terminated_jax,
            truncated_jax,
            ep_returns_pre,
            ep_lengths_pre,
        ) = self._step_fn(self._state, self._actions_jax)

        # Bekle (gerçekten async pipeline lazımsa kullanıcı dispatch edip
        # block_until_ready() çağırabilir, default eager).
        new_state.boards.block_until_ready()

        self._state = new_state
        self._cached_mask_np = np.asarray(new_state.cached_mask, dtype=bool)

        # Host'a aktarım
        post_obs_np = self._jax_obs_to_numpy(post_obs_jax)
        pre_obs_np = self._jax_obs_to_numpy(pre_obs_jax)
        reward = np.asarray(reward_jax, dtype=np.float32)
        done = np.asarray(done_jax, dtype=bool)
        terminated = np.asarray(terminated_jax, dtype=bool)
        truncated = np.asarray(truncated_jax, dtype=bool)
        ep_r = np.asarray(ep_returns_pre, dtype=np.float32)
        ep_l = np.asarray(ep_lengths_pre, dtype=np.int32)

        N = self.num_envs
        infos: list[dict[str, Any]] = [{} for _ in range(N)]
        for i in range(N):
            if truncated[i] and not terminated[i]:
                infos[i]["TimeLimit.truncated"] = True
            if done[i]:
                infos[i]["episode"] = {"r": float(ep_r[i]), "l": int(ep_l[i])}
                infos[i]["terminal_observation"] = {
                    k: pre_obs_np[k][i].copy() for k in pre_obs_np
                }

        return post_obs_np, reward, done, infos

    # ------------------------------------------------------------------
    # MaskablePPO hook
    # ------------------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        """Cached (N, 192) bool. step_wait sonrası güncel."""
        return self._cached_mask_np.copy()

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: VecEnvIndices = None,
        **method_kwargs,
    ):
        idx = self._get_indices(indices)
        if method_name == "action_masks":
            return [self._cached_mask_np[i].copy() for i in idx]
        raise NotImplementedError(
            f"JaxBlockBlastVecEnv.env_method('{method_name}') is not supported"
        )

    # ------------------------------------------------------------------
    # SB3 boilerplate
    # ------------------------------------------------------------------

    def seed(self, seed: int | None = None) -> Sequence[None]:
        self._seed = int(seed) if seed is not None else 0
        # Yeni state üret
        self._state = self._reset_fn(self.num_envs, jax.random.PRNGKey(self._seed))
        self._state.boards.block_until_ready()
        self._cached_mask_np = np.asarray(self._state.cached_mask, dtype=bool)
        return [None] * self.num_envs

    def close(self) -> None:
        return None

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        idx = self._get_indices(indices)
        if hasattr(self, attr_name):
            val = getattr(self, attr_name)
            return [val for _ in idx]
        raise AttributeError(f"JaxBlockBlastVecEnv has no attribute '{attr_name}'")

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        raise NotImplementedError(
            "JaxBlockBlastVecEnv.set_attr is not supported (state is JAX-batched)."
        )

    def env_is_wrapped(self, wrapper_class, indices: VecEnvIndices = None) -> list[bool]:
        idx = self._get_indices(indices)
        return [False for _ in idx]

    def render(self, mode: str = "rgb_array") -> Any:  # noqa: ARG002
        return np.asarray(self._state.boards)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_indices(self, indices: VecEnvIndices) -> Sequence[int]:
        if indices is None:
            return list(range(self.num_envs))
        if isinstance(indices, int):
            return [indices]
        return list(indices)

    @staticmethod
    def _jax_obs_to_numpy(obs_jax: dict) -> dict:
        """device → host. history int32 → int64 (SB3 embedding kontratı)."""
        out: dict[str, np.ndarray] = {}
        for k, v in obs_jax.items():
            arr = np.asarray(v)
            if k == "history":
                arr = arr.astype(np.int64, copy=False)
            out[k] = arr
        return out

    def _build_obs_numpy(self, state: JaxState) -> dict:
        """reset() için: state'ten obs üretir, host'a aktarır."""
        from src.env.jax_env import build_obs as _build_obs_jax
        obs = _build_obs_jax(
            boards=state.boards,
            frame_buf=state.frame_buf,
            cached_mask=state.cached_mask,
            tray_idx=state.tray_idx,
            history=state.history,
            step_count=state.step_count,
            piece_cells_pad=self.piece_cells_pad,
            max_steps=self.cfg.max_steps,
        )
        return self._jax_obs_to_numpy(obs)
