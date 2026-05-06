# Mimari

## Genel akış

```
Game (saf Python, np tabanlı)
   ▲
   │ step(piece_idx, x, y) → StepResult
   │
BlockBlastEnv (gymnasium.Env)
   ▲
   │ obs, reward, term, trunc, info
   │
DummyVecEnv / SubprocVecEnv (n_envs=8)
   ▲
   │ MaskablePPO.learn()
   │
MaskableActorCriticPolicy (custom features extractor)
```

## State (observation)

Dict observation:

| Anahtar | Şekil | Tip | Anlam |
|---|---|---|---|
| `grid` | (4, 8, 8) | float32 | Frame stack: en eskiden en yeniye 4 grid (binary fill) |
| `tray` | (3, NUM_PIECE_TYPES) | float32 | Her slot'taki parça type'ı için one-hot. Boş slot = sıfır vektör. |
| `tray_shapes` | (3, 5, 5) | float32 | Her slot'taki parçanın 5×5 maskesi (sol-üst origin, padded). Boş slot = sıfır. |
| `history` | (8,) | int64 | Son 8 action_id. Henüz alınmamış adımlar için sentinel = NUM_ACTIONS (192). |
| `meta` | (4,) | float32 | `[steps / max_steps, fill_ratio, n_legal / NUM_ACTIONS, slots_remaining / 3]` |

**NUM_PIECE_TYPES = 31** (tüm rotation varyantları dahil).

**Frame stack:** her step sonrası `grid` deque'sine güncel state itilir; en eski drop edilir. Reset'te 4 kopya doldurulur.

**Action history:** episode başında `[NUM_ACTIONS] * 8` ile başlar; her step sonrası action_id push, en eski pop.

## Action space

`Discrete(192)` = `piece_idx (0..2) × y (0..7) × x (0..7)`

Kodlama:
```
action_id = piece_idx * 64 + y * 8 + x
piece_idx = action_id // 64
y = (action_id % 64) // 8
x = action_id % 8
```

## Action masking

Her step `info["action_mask"]` 192-uzunluk bool array. `mask[a] = True` ⇔ `(piece_idx, x, y)` valid:
1. Slot dolu (`tray.has_piece(piece_idx)`).
2. `board.can_place(piece, x, y)` true (sınır içinde + çakışma yok).

MaskablePPO bu maskeyi softmax öncesi -∞ ile uygular.

## Reward

```python
def step(action):
    if action invalid:
        # mask zaten engelliyor; emniyet
        return obs, -1.0, terminated=True, truncated=False, info
    place success → +1.0
    if game_over: reward += -10.0; terminated=True
    if max_steps: truncated=True (ekstra reward yok)
```

**Hiçbir şey daha:** line clear, combo, holes, score, board clear, step_cost — hepsi 0 ekstra reward.

`info` içinde monitoring için tutulur:
- `score`: oyun-içi puan (combo / line / cell tabanlı, sadece logging)
- `cleared_lines`: bu step'teki line clear sayısı
- `board_was_cleared`: bool
- `episode_length`: o ana kadar kaç step
- `action_mask`: bool[192]

## Model: BlockBlastFeatureExtractor

`stable_baselines3.common.torch_layers.BaseFeaturesExtractor` alt sınıfı.

```
grid (B, 4, 8, 8)
  → Conv2d(4, 32, 3, padding=1) → ReLU
  → Conv2d(32, 64, 3, padding=1) → ReLU
  → Flatten → Linear(64*8*8, 128) → ReLU
  → grid_feat (B, 128)

tray (B, 3, 31) + tray_shapes (B, 3, 5, 5)
  → flatten → concat → Linear(3*31 + 3*25, 64) → ReLU
  → tray_feat (B, 64)

history (B, 8)
  → Embedding(193, 16) → mean pool
  → Linear(16, 32) → ReLU
  → hist_feat (B, 32)

meta (B, 4)
  → Linear(4, 16) → ReLU
  → meta_feat (B, 16)

concat (B, 128+64+32+16=240) → Linear(240, 256) → ReLU
                              → Linear(256, 256) → ReLU
                              → features (B, 256)
```

`features_dim = 256`. Policy ve value head'leri SB3'ün default `mlp_extractor` üzerinden 256→64 ile bağlanır.

Toplam parametre tahmini: ~200K. (1. denemede ~4M idi.)

## Curriculum / pieces

`configs/pieces.yaml` 31 parça içerir (1. denemeden taşındı). `env.yaml`'da `pieces.config_file: configs/pieces.yaml`. Curriculum filtreleme yok; tüm set baştan kullanılır.

## Tray davranışı

- 3 slot, hepsi boşaldığında 3 yeni parça gelir.
- `same_tray_unique=True`: aynı tray'de 3 farklı type.
- `anti_repeat_window=1`: bir önceki tray'in type'larını dışla.
- `anti_frustration=False`: kapalı (ajan zorlukla yüzleşmeli).
- `ensure_solvable=False`: kapalı (ajan game over riskini görmeli).

## Episode

- Reset → board boş, tray full, frame stack güncel state'in 4 kopyası, history `[192]*8`.
- `max_steps=5000` (truncation; pratikte ulaşılmaz).
- Termination: tray'deki hiçbir parça yerleştirilemez (game_over).
