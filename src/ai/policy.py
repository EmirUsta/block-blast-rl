"""Custom features extractor — bottleneck'li CNN + early-fusion + history embedding + meta.

v3 mimari değişiklikleri (v1/v2'ye göre):

  1) **Early fusion:** grid (4 frame) ve placements (3 piece availability) CNN'den
     ÖNCE concat edilir → 7 kanal birlikte conv'a girer. CNN aynı uzamsal düzlemde
     "tahta dolu mu?" + "her parça nereye yerleşebilir?" sinyallerini görür.
  2) **Bottleneck çözümü:** ikinci conv stride=2 → feature haritası 8×8 → 4×4.
     Flatten 4096 → 1024 (4×). Linear(1024→128) ≈ 131K param (eski 524K'dan ↓).

Tray one-hot ve tray_shapes late-fusion'da kalır (semantik kimlik için sıvı).

Çıktı: features_dim=256 vektör. SB3 default mlp_extractor policy ve value head'lere
bağlar.
"""

from __future__ import annotations

import gymnasium as gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from src.env.encoding import (
    BOARD_SIZE,
    FRAME_STACK,
    HISTORY_LEN,
    HISTORY_SENTINEL,
    MAX_PIECE_DIM,
    META_DIM,
    NUM_PIECE_TYPES,
    TRAY_SIZE,
)


class BlockBlastFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 256,
        cnn_channels: tuple[int, int] = (32, 64),
        history_embed_dim: int = 16,
    ) -> None:
        super().__init__(observation_space, features_dim=features_dim)

        c1, c2 = cnn_channels

        # Early fusion: grid (FRAME_STACK) + placements (TRAY_SIZE) → toplam kanal
        in_channels = FRAME_STACK + TRAY_SIZE  # 4 + 3 = 7

        # Bottleneck CNN: ikinci conv stride=2 (8×8 → 4×4)
        self.grid_cnn = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        # Stride=2 ile (8+2*1-3)/2+1 = 4 → c2 × 4 × 4 = c2 * 16
        flat = c2 * 4 * 4
        self.grid_proj = nn.Sequential(
            nn.Linear(flat, 128),
            nn.ReLU(inplace=True),
        )

        # Tray late-fusion encoder (piece kimliği)
        tray_in = TRAY_SIZE * NUM_PIECE_TYPES + TRAY_SIZE * MAX_PIECE_DIM * MAX_PIECE_DIM
        self.tray_proj = nn.Sequential(
            nn.Linear(tray_in, 64),
            nn.ReLU(inplace=True),
        )

        # History encoder
        num_action_tokens = HISTORY_SENTINEL + 1  # 193
        self.history_embed = nn.Embedding(num_action_tokens, history_embed_dim)
        self.history_proj = nn.Sequential(
            nn.Linear(history_embed_dim, 32),
            nn.ReLU(inplace=True),
        )

        # Meta
        self.meta_proj = nn.Sequential(
            nn.Linear(META_DIM, 16),
            nn.ReLU(inplace=True),
        )

        concat_dim = 128 + 64 + 32 + 16  # 240
        self.head = nn.Sequential(
            nn.Linear(concat_dim, features_dim),
            nn.ReLU(inplace=True),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, obs: dict) -> th.Tensor:
        grid = obs["grid"]                         # (B, 4, 8, 8)
        placements = obs["placements"]             # (B, 3, 8, 8)
        tray = obs["tray"]                         # (B, 3, 31)
        tray_shapes = obs["tray_shapes"]           # (B, 3, 5, 5)
        history = obs["history"].long()            # (B, 8)
        meta = obs["meta"]                         # (B, 4)

        # Early fusion: grid + placements → (B, 7, 8, 8)
        spatial = th.cat([grid, placements], dim=1)
        g = self.grid_proj(self.grid_cnn(spatial))     # (B, 128)

        b = tray.shape[0]
        tray_flat = tray.reshape(b, -1)
        shapes_flat = tray_shapes.reshape(b, -1)
        t = self.tray_proj(th.cat([tray_flat, shapes_flat], dim=1))  # (B, 64)

        h_emb = self.history_embed(history)        # (B, 8, 16)
        h_mean = h_emb.mean(dim=1)                 # (B, 16)
        h = self.history_proj(h_mean)              # (B, 32)

        m = self.meta_proj(meta)                   # (B, 16)

        x = th.cat([g, t, h, m], dim=1)            # (B, 240)
        return self.head(x)                        # (B, 256)
