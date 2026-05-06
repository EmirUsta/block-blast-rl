# Tasarım Kararları

Bu dosya projenin **neden** bu şekilde tasarlandığını anlatır. *Ne yaptığını* `docs/architecture.md` anlatır; *ne zaman ne değiştiğini* `docs/log.md` anlatır.

---

## 1. Reward: Sadece survival

### Karar
- **+1.0** her başarılı `place(piece, x, y)` adımı için
- **-10.0** game over olduğunda (terminal step)
- Diğer her şey: **0**

Yani:
- Line clear → 0 ekstra ödül
- Combo zinciri → 0 ekstra ödül
- Skor → 0 ekstra ödül (ama `info["score"]` üzerinden monitoring yapılır)
- Board clear → 0 ekstra ödül
- Holes / bumpiness / max_height → reward'a girmez

### Gerekçe
Birinci denemede 5+ shaping terimi aynı anda devredeydi. Ajan hangisine optimize edeceğini çözemeden episode'lar geçti. Burada hipotez:

> Hayatta kalmak = step başına +1 toplama → en uzun episode = en yüksek reward.
> Line clear yapan ajan zaten daha uzun yaşar (board boşalır → daha fazla yer = daha fazla step). Yani line clear'a *ekstra* ödül vermeye gerek yok; survival reward'ı line clear'ı **otomatik teşvik eder**.

### Beklenen davranış
- İlk yüz binlerce step'te ajan rastgele oynayıp ortalama ~10-20 step yaşar (random baseline).
- Reward sinyali: "uzun yaşa = çok ödül". Ajan board'ı dolu tutmamayı öğrenmeli.
- Line clear öğrenmesi *dolaylı* ortaya çıkmalı; cebinde line clear olmayan ajan da hayatta kalmaya çalışır → küçük parçalar tercih, fragment'tan kaçınma davranışları.

### Risk
- Line clear oluşmadan ajan tıkanırsa "hayatta kalmak için yer açmak gerek" sinyalini almaz. Bu durumda küçük parçalarla ilk faz curriculum gerekebilir — ama önce sade halini deneyelim. *Her şeyi aynı anda yapmamak* projenin baş kuralı.

---

## 2. Observation: Geçmişi gör, geleceği critic'e bırak

Kullanıcı talebi: "modelin eline gelen 3 şekli görsün, önceki hamlelerini hatırlasın, sonraki hamlelerini öngörsün, tahmin etsin."

### Karar
Observation şu Dict alanlarından oluşur:

| Alan | Şekil | Tip | Anlam |
|---|---|---|---|
| `grid` | (4, 8, 8) | float32 | **Frame stack 4**: şu anki + önceki 3 grid state |
| `tray` | (3, MAX_PIECE_TYPES) | float32 | 3 slot, her biri parça type'ı için one-hot |
| `tray_shapes` | (3, 5, 5) | float32 | 3 slot, her parça'nın 5×5 binary cell maskesi (padded) |
| `history` | (8,) | int64 | Son 8 action_id (yok ise sentinel = NUM_ACTIONS) |
| `meta` | (4,) | float32 | `[steps_norm, fill_ratio, n_legal_norm, slots_remaining_norm]` |

### Gerekçe — frame stack
Tek frame'de "ben az önce nereye koydum, board nasıl değişti" görünmez. Frame stack 4 ile son 4 grid state aynı anda kanal olarak verilir → CNN bunu doğal olarak işler. RNN/LSTM'e göre **çok daha basit** (bias=düşük, eğitimi=hızlı).

### Gerekçe — action history
Son 8 hamle action_id olarak verilir, embed katmanı (örneğin 16-dim) ile gömülür. Ajan "az önce piece 0'ı (3,4)'e koydum" bilgisini açıkça görür. Embedding boyutu 16, parametre maliyeti minimal.

### Gerekçe — sonrayı "öngörme"
PPO'nun **value head'i (critic)** tam olarak bunu yapar: V(s) = expected future return. Yani critic ağı zaten "sonraki hamlelere bağlı kazanç" tahminidir. Ekstra forward-model eklemiyoruz çünkü:
1. Block Blast'ta tray refill stokastik (random) → mükemmel öngörü matematiksel olarak imkânsız.
2. World-model eklemek pipeline'ı 3-4 katına çıkarır, kullanıcı "komplike olmasın" dedi.
3. Zaten critic'in V tahmini bu işi yapar.

### Gerekçe — meta minimal
Birinci denemede META_DIM=16'ya çıktı: col_heights, death_proximity, survival_norm, recent_clears... Hepsi gürültü ekledi, observation distribution shift'ini büyüttü. Burada **4 skaler** yeter: episode ilerlemesi, ne kadar dolu, kaç legal hamle var, kaç slot kaldı. Gerisi grid'den zaten okunabilir.

---

## 3. Action space ve masking

### Karar
- `Discrete(192)` = `piece_idx (0..2) × y (0..7) × x (0..7)`
- `action_id = piece_idx * 64 + y * 8 + x`
- Her step için bool action mask (192 uzunluk): sadece valid placement'lar 1.

### Gerekçe
Block Blast'te action space sabit ve küçük (192). MaskablePPO ile invalid action'lar baştan elenir → öğrenme hızlanır. Birinci denemede de işe yaradı, korunuyor.

---

## 4. Model mimarisi

### Karar — basit CNN + MLP
- **Grid encoder:** Conv2d(4 → 32, 3×3) → ReLU → Conv2d(32 → 64, 3×3) → Flatten → Linear → 128
- **Tray encoder:** tray (3, NUM_TYPES) flatten + tray_shapes (3, 25) flatten → Linear → 64
- **History encoder:** Embedding(NUM_ACTIONS+1, 16) → mean pool → Linear → 32
- **Meta:** Linear → 16
- **Concat → MLP(256) → MLP(256)** → policy + value head

### Gerekçe
Birinci denemede `features_dim=512`, `cnn_channels=[64,128]`, `pieces_embed_dim=128`, `meta_dim=96` ile gereksiz büyük bir ağ vardı. 8×8 grid için **çok daha küçük** ağ yeterli. Toplam parametre ~150K civarı (önceki 4M'den çok küçük). Daha hızlı eğitim, daha az overfit riski.

---

## 5. PPO hyperparameters — default ya da default'a yakın

### Karar
```yaml
learning_rate: 3.0e-4
n_steps: 2048
batch_size: 256
n_epochs: 10
gamma: 0.99
gae_lambda: 0.95
clip_range: 0.2
ent_coef: 0.01
vf_coef: 0.5
max_grad_norm: 0.5
target_kl: 0.03
n_envs: 8
total_timesteps: 2_000_000
```

### Gerekçe
- `gamma=0.99`: episode'lar kısa başlayacak (random baseline ~10-20 step), 0.999 abartı olur. Survival uzadıkça ileride 0.995'e çekilebilir.
- `ent_coef=0.01`: standart. Birinci denemede 0.02'ydi → keşif daha çok ama yakınsama daha zor. 0.01 makul başlangıç.
- `vf_coef=0.5`: standart. Birinci denemedeki 1.0 yapay shaping'in noisy reward'ını dengelemek içindi; burada reward sade, 0.5 yeter.
- `n_envs=8`: yeterli sample throughput, multi-process overhead düşük.
- `total_timesteps=2M`: ilk fazda yetmezse 5M-10M'e ölçeklenir. Önce 2M ile temel davranışın oluştuğunu gör, sonra ölçek.

VecNormalize **kullanmıyoruz** — reward ölçeği zaten [-10, +1] dar bir aralıkta, normalize gereksiz. Birinci denemede VecNormalize/eval senkronizasyonu hata kaynağı oldu.

---

## 6. Curriculum YOK (şimdilik)

### Karar
İlk eğitimde **31 parçanın tamamı** baştan kullanılır. Curriculum (kolaydan zora geçiş) eklenmiyor.

### Gerekçe
- Curriculum birinci denemede de v4'ten v5'e geçişin sebebiydi → işe yaramadı, modeli karıştırdı.
- "Önce sadeyi dene, gerekirse ekle" prensibi.
- Eğer 2M step sonunda ajan yine yetersizse curriculum ikinci faz olarak eklenir; şimdilik yok.

---

## 7. Birinci denemeden farklar — özet tablo

| Özellik | 1. Deneme (failed v5) | 2. Deneme (bu repo) |
|---|---|---|
| Reward | +score, +line, +combo^2, +survival_milestone, +board_clear, -holes, -bumpiness, -max_height, -game_over (-20), +step_cost | **+1 step, -10 game_over, geri kalan 0** |
| Meta features | 16 (col heights, death proximity, recent clears…) | **4 skaler** |
| Observation | grid + tray + meta=16 | grid (frame stack 4) + tray + tray_shapes + history + meta=4 |
| Frame stack | yok (ya da çevre dışı) | **4 frame** |
| Action history | yok | **son 8 action embedding** |
| Curriculum | 14 → 19 → 31 parça | **31 parça baştan** |
| VecNormalize | reward normalize | **kullanılmıyor** |
| CNN size | 64 → 128, features_dim 512 | **32 → 64, features_dim 256** |
| Hyperparams | gamma=0.999, vf_coef=1.0, batch=1024, ent=0.02 | **gamma=0.99, vf_coef=0.5, batch=256, ent=0.01** |
| Total steps | 10M | **2M (ilk fazda)** |

Tek mottosu var: **az değişken, sade hedef, hızlı geri bildirim**.
