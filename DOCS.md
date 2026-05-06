# Yaşayan Dökümantasyon

Bu dosya yapılan **her** önemli değişikliği, deneyimi ve kararı kaydeder. Amaç: 6 ay sonra geri dönen biri (insan veya AI) projenin nereden, nereye ve neden bu şekilde geldiğini anlayabilsin.

Yapı: en yeni en üstte. Her giriş tarihli. Format:

```
## YYYY-MM-DD — Kısa başlık
**Ne:** ...
**Neden:** ...
**Sonuç:** ...
```

---

## 2026-05-05 — Proje iskeleti kuruldu

**Ne:** `/home/emir/Emir/BlockBlast 2. deneme/` boş dizininden başlayarak şu yapı oluşturuldu:
- `configs/` — `pieces.yaml`, `env.yaml`, `ppo.yaml`
- `src/game/` — `pieces.py`, `board.py`, `tray.py`, `game.py` (saf oyun, headless)
- `src/env/` — `encoding.py`, `masking.py`, `env.py` (gymnasium wrapper)
- `src/ai/policy.py` — basit MaskableActorCritic + custom feature extractor
- `src/scripts/` — `train.py`, `eval.py`
- `tests/` — temel smoke testler
- `docs/` — `architecture.md`, `log.md`
- `README.md`, `DESIGN.md`, `DOCS.md`

**Neden:** Birinci denemenin (`/home/emir/Emir/block blast/`) başarısız v5 sonucu üzerine kullanıcı temiz başlangıç istedi. Birinci denemede çoklu reward shaping, geniş META, curriculum, VecNormalize ve büyük ağ aynı anda devredeydi → ajan hayatta kalmayı bile öğrenemedi. Bu projede tek bir hedef: **survival**. Diğer her şey sıfırlandı.

**Sonuç:**
- `pytest tests/ -q` → 10 passed in 3.67s
  - test_smoke (8 test): action bijection, game step, env shapes, survival reward, history & frame stack güncellenmesi, invalid → terminate, config dosyaları var.
  - test_policy_smoke (2 test): MaskablePPO build + 128 step learn(), predict döngüsü.
- Eğitim henüz başlatılmadı (kullanıcı "sonraki promptu vericem" dedi → bekleniyor).

## 2026-05-05 — v1_survival eğitimi başlatıldı

**Ne:** `python -m src.scripts.train --config configs/ppo.yaml --run-name v1_survival` arka planda başlatıldı.
- GPU: CUDA, 1 device (`device: auto` → cuda).
- Config: 2M step, n_envs=8, n_steps=2048, batch=256, γ=0.99, ent=0.01, vf=0.5.
- Logs: `logs/tensorboard/v1_survival_1`, `logs/train_v1.log`.
- TensorBoard: http://localhost:6006 (logdir = `logs/tensorboard`).
- Checkpoint: `models/checkpoints/v1_survival/` (her ~100K step), best: `models/best/v1_survival/`.

**Neden:** İskelet yeşil olduktan sonra ilk pratik ölçüm. Bu run "sade survival-only reward + frame stack + history" yaklaşımının çalışıp çalışmadığını gösterecek. Beklentiler:
- İlk ~50K step: ajan rastgele oynar, mean ep_length ~10-20.
- 200-500K aralığında ep_length artmalı (öğrenme sinyali).
- 2M sonunda ortalama 100-300 step civarı bir survival hedefliyoruz (1. denemenin v4 baseline'ı 20.78 step ortalama yaşıyordu — yani süreyi ~10x uzatmayı umuyoruz).

**Sonuç:** PID 31144 (process 30 dak+ sürebilir, GPU bound).

### Snapshot @ 20:45 (~4 dak çalıştı, ~589K step)

- fps = 2373, GPU %29 (env-bound, normal)
- Eval callback ep_len trend: 12.20 → 13.50 → 13.05 → 12.40 → **16.60**
- entropy_loss = -2.63 (keşif sürüyor), explained_variance = 0.41
- approx_kl = 0.033 (target_kl=0.05'e yakın, bazı update'lerde early-stop)
- Reward formülü kontrol: reward ≈ ep_len - 10 → ✓ (survival-only çalışıyor)
- Tahmini bitiş: ~10 dak sonra (1.4M step / 2373 fps).

### Snapshot @ 20:51 (~7 dak çalıştı, ~1.5M step / %75)

- Eval ep_len son 5 ölçüm: 15.85, 16.60, 16.35, 15.05, 16.30 → **15-17 aralığında plato**.
- Reward = ep_len - 10 → 5-7 aralığında.
- GPU %17, 706 MiB. Train sağlıklı, çökme yok.
- **Gözlem:** Ep_len 1.4M step boyunca ~16'da takılı; v4 baseline 20.78'e henüz ulaşılamadı. Survival-only reward'ın anlamlı yükseliş üretmemesi olası — eğitim bitince eval'la kesinleşecek.
- Tahmini bitiş: 3-4 dak sonra.

### v1_survival eğitim TAMAMLANDI (2M step, 20:54)

`models/checkpoints/v1_survival/final.zip` ve `models/best/v1_survival/best_model.zip` yazıldı.
Son eval (callback): ep_len = 20.00 ± 6.10 @ 2M step.

**1000-episode formal eval (best_model):**

| Metrik | Değer |
|---|---|
| ep_len mean | **18.4** |
| ep_len median | 17 |
| ep_len max | 62 |
| ep_len min | 8 |
| score mean | 103.5 |
| score median | 90 |
| score max | 515 |
| clears/ep mean | 3.27 |
| reward/ep mean | 8.36 |

**Karşılaştırma:**

| Model | mean ep_len | mean score | Yorum |
|---|---|---|---|
| 1.deneme v4_cont | 20.78 | 292.1 | Score-optimize |
| 1.deneme v5_immortal | 15.60 | 141.4 | Karmaşık shaping, başarısız |
| **2.deneme v1_survival** | **18.4** | **103.5** | Sade survival-only |

**Yorum:**
- v1_survival v5'ten **+18% daha uzun** survival, v4_cont'a **-11.5%** yakın.
- Score düşük (103 vs 292) ama beklenen — biz score'u optimize etmedik, sadece survival.
- Survival-only ajanın doğal olarak **3.27 line clear/episode** yapması zarif sonuç: clear yapan ajan zaten daha uzun yaşar.
- 2M step yetersiz olabilir — ep_len trend 1.4M'de plato 16, 2M sonunda 20. Daha uzun eğitimde daha çok kazanım olası.

## 2026-05-05 — v2_parallel başlatıldı (10M step, SubprocVecEnv n_envs=16)

**Ne:** v1 ile aynı hyperparams, **n_envs 8 → 16** + **DummyVecEnv → SubprocVecEnv** + **2M → 10M step**.
- Config: `configs/ppo_v2.yaml`.
- Run: `v2_parallel`, PID 45967.
- TB log: `logs/tensorboard/v2_parallel_1`.

**Neden:** İki gerekçe:
1. Paralel benchmark — 16-core CPU SubprocVecEnv ile fps gerçekten 2x+ artıyor mu?
2. Daha uzun eğitim (10M) — v1 son 500K'da hâlâ yükselişteydi, plato'nun aslında "yeterli zaman yok" olduğunu test et.

**Not:** SubprocVecEnv küçük env'lerde IPC overhead nedeniyle bazen yavaş olabilir; gerçek gözlem 3 dak sonraki snapshot'ta görünecek.

### Snapshot @ 21:01 (~5 dak, 924K / 10M = %9)

- **fps = 3566** (v1 fps 2370 ile karşılaştır → **1.50x speedup**).
- SubprocVecEnv n_envs=16 net olarak işe yaradı, ama 2x değil — küçük env IPC overhead'i hipotezi doğrulandı.
- GPU util %13 (CPU-bound; n_envs arttıkça GPU göreli olarak daha az meşgul olur, normal).
- Eval ep_len son 5: 13.60, 13.85, 14.70, 14.50, 12.50 → v1'in 1M civarı seviyesiyle uyumlu.
- Tahmini bitiş: ~42 dak sonra (kalan 9M / 3566 fps).

### Snapshot @ 21:05 (~9 dak, 1.87M / %18.7)

- fps = **3749** (ısındı, v1'in 2370'ine göre **1.58x speedup**)
- Eval ep_len son 5: 15.00, 16.65, 16.15, 13.50, 14.55 (13-17 plato; v1'in aynı timestep seviyesinde)
- GPU %47 (n_steps doldukça yoğunlaşıyor, 458 MiB)
- Tahmini bitiş: ~36 dak sonra

### Snapshot @ 21:09 (~13 dak, 2.75M / %27.5)

- fps = **3709** (stabil ~1.56x speedup)
- Eval ep_len son 5: 20.10, 20.20, 17.10, 20.45, 15.20 → **ilk 20+ pikleri**, varyans 15-20
- v1'in 2M sonu seviyesi (20.00) ile aynı yerde, beklendiği gibi
- GPU %43, 458 MiB stabil
- Tahmini bitiş: ~33 dak

### Snapshot @ 21:13 (~17 dak, 3.50M / %35)

- fps = **3586** (hafif yavaşladı)
- Eval ep_len son 5: 17.75, 20.65, 19.40, 19.50, 18.45 → ortalama **19.15**, önceki seviyenin üstü
- v1 2M = 20.00, v2 3.5M = 19.15 → benzer sample efficiency
- GPU %39, 458 MiB stabil
- Tahmini bitiş: ~30 dak

### Snapshot @ 21:17 (~21 dak, 4.22M / %42.2)

- fps = **3486** (~1.47x speedup, devam eden yavaşlama)
- Eval ep_len son 5: 20.65, 19.10, 20.65, **22.55** (yeni rekor), 17.60 → ortalama **20.11**
- v1 2M sonu seviyesi (20.00) geçildi → daha uzun eğitimin değer kattığını gösteren ilk net sinyal
- Tahmini bitiş: ~28 dak

### Snapshot @ 21:21 (~25 dak, 4.95M / %49.5 — yarısı geçildi)

- fps = **3403** (~1.44x speedup; yavaşlamaya devam)
- Eval ep_len son 5: 20.80, 20.40, 20.80, 18.95, 18.50 → ortalama **19.89**
- 4M-5M aralığında ep_len ~20'de **plato** sinyali → survival-only reward'ın olası doğal sınırı
- Tahmini bitiş: ~25 dak

### Snapshot @ 21:25 (~29 dak, 5.67M / %56.7)

- fps = **3348** (~1.41x speedup)
- Eval ep_len son 5: 19.55, **24.00** (yeni tek-eval rekoru), 21.25, 18.15, 17.60 → ort **20.11**
- Tepe noktaları yükseliyor (22.55 → 24.00) — plato olsa da yer kalmış
- Tahmini bitiş: ~22 dak

### Snapshot @ 21:29 (~33 dak, 6.40M / %64)

- fps = **3307**
- Eval ep_len son 5: **25.45** (yeni tek-eval rekoru), 20.25, 21.30, 20.15, 21.65 → ort **21.76**
- 4M ort 19.15 → 5M ort 19.89 → 6M ort 21.76 → **plato değil, yavaş yükseliş** (saniyede +0.5 ep_len gibi)
- 25.45 tek-eval rekoru: v4_cont baseline (20.78) eval-mean'ini geçti
- Tahmini bitiş: ~18 dak

### Snapshot @ 21:33 (~37 dak, 7.12M / %71.2)

- fps = **3279**
- Eval ep_len son 5: 15.30, 17.60, 22.50, 21.30, **28.75** (yeni rekor) → ort **21.09**
- Tepe artış zinciri: 22.55 → 24.00 → 25.45 → 28.75 (4M-7M boyunca)
- Varyans yüksek 15-28 ama trend yukarı
- Tahmini bitiş: ~15 dak

### Snapshot @ 21:37 (~41 dak, 7.86M / %78.6)

- fps = **3255**
- Eval ep_len son 5: 23.85, 23.65, 24.40, 24.20, 21.90 → ort **23.60**
- Hepsi 21+ üstünde, **stabil yüksek** seviye
- Trend: 19.15 → 19.89 → 21.76 → 21.09 → **23.60** (genel artış net)
- v1 final ort ~16; v2 7.8M = 23.60 ⇒ 1.5x daha iyi survival
- Tahmini bitiş: ~11 dak (≈ 21:48)

### Snapshot @ 21:41 (~45 dak, 8.58M / %85.8)

- fps = **3235**
- Eval ep_len son 5: 22.20, 19.30, 19.40, 20.20, 20.70 → ort **20.36**
- 23.60'tan hafif geri çekildi ama 20+ kararlı seviyede; tepe rekor 28.75 duruyor
- Tahmini bitiş: ~7 dak (sonraki wakeup'ta muhtemelen biter)

### Snapshot @ 21:45 (~49 dak, 9.14M / %91.4)

- fps = **3156**
- Eval ep_len son 5: 17.10, 21.50, 22.85, 20.70, 23.05 → ort **21.04**
- 20-23 aralığında stabil
- Tahmini bitiş: ~4-5 dak

### Snapshot @ 21:49 (~53 dak, 9.70M / %97)

- fps = **3098**
- Eval ep_len son 5: 22.15, 21.30, 20.30, 22.00, 26.05 → ort **22.36**
- ~1.5-2 dak kaldı; sonraki kontrolde kesin biter

### v2_parallel eğitim TAMAMLANDI (10M step, ~57 dak, 21:52)

`models/checkpoints/v2_parallel/final.zip` ve `models/best/v2_parallel/best_model.zip` yazıldı.

**1000-episode formal eval:**

| Metrik | v1_survival (2M) | **v2_parallel (10M)** | v4_cont (1.deneme) |
|---|---|---|---|
| ep_len mean | 18.4 | **21.3** | 20.78 |
| ep_len median | 17 | **20** | — |
| ep_len max | 62 | 62 | — |
| score mean | 103.5 | **129.1** | 292.1 |
| score median | 90 | 116 | 187 |
| clears/ep mean | 3.27 | **4.67** | 4.53 |
| reward/ep | 8.36 | 11.28 | — |

**Bulgular:**
- v2 v1'i tüm survival metriklerinde geçti (ep_len +%15.8, score +%24.7, clears/ep +%42.8).
- **v4_cont (1.denemenin şampiyonu) survival'da geçildi** — sade survival-only reward, 1. denemenin score-optimize ajanını survival açısından yendi.
- Score düşük (129 vs 292) ama beklenen — biz score'u eğitim sinyali olarak kullanmıyoruz.
- **clears/ep 3.27 → 4.67 sıçraması** ana sinyal: ajan uzun yaşamak için kendiliğinden 1.5x daha fazla line clear öğrendi. "Clear yapan daha uzun yaşar" hipotezi 1. denemenin score-optimize ajanından bile daha güçlü doğrulandı.

**Sonuç:** Survival-only reward + sade gözlem + paralel uzun eğitim, karmaşık 1. deneme yaklaşımından (multi-shaping + büyük META + curriculum + VecNormalize) **daha iyi** sonuç verdi. 5x daha uzun eğitim ile sample efficiency korundu (v1 2M ep_len 18.4 → v2 10M ep_len 21.3, log-linear iyileşme tipik).

**fps benchmark final:** v1 (DummyVecEnv n_envs=8) ~2370 → v2 (SubprocVecEnv n_envs=16) ~3100-3700 (1.4-1.6x speedup); 2x değil ama anlamlı, küçük env IPC overhead hipotezi doğrulandı.

## 2026-05-06 — v3 hazırlık: batch_size + hole_penalty

**Ne (henüz başlatılmadı):**
- `src/env/encoding.py` → `count_small_islands(grid, max_size=2)` BFS-tabanlı 4-connected component, küçük (size≤2) izole boş ada toplam hücre sayısı.
- `src/env/env.py` → step başında holes_before, sonra holes_after; `delta = max(0, after - before)`; `reward += hole_penalty * delta`. Asimetrik (sadece artış cezalandırılır).
- `configs/env.yaml` → `hole_penalty: 0.0, hole_max_size: 2` (default kapalı, geri uyumlu).
- `configs/env_v3.yaml` (yeni) → `hole_penalty: -0.5, hole_max_size: 2`.
- `configs/ppo_v3.yaml` → batch_size 256→1024 + env_config: configs/env_v3.yaml.
- `tests/test_holes.py` (yeni) → 9 test, hepsi yeşil. Toplam pytest 20/20 yeşil.

**Niye iki değişiklik aynı anda (v2→v3):**
- batch=1024: gradient noise hipotezi (32K buffer / 256 minibatch = 128 update × 10 epoch = 1280 noisy update; bunun yerine 320 stabil update).
- hole_penalty=-0.5: ajan dağınık yerleştirme yapmasın; izole 1-2 hücre boş ada oluşturma cezası ile "temiz tahta" davranışı doğsun.
- İkisi farklı problem alanlarına dokunuyor (optimizasyon stabilitesi vs reward sinyali) → birlikte denemek mantıklı; ablation gerekirse v3a/v3b ile ayrılabilir.

**Reward formülü (v3):**
```
reward = +1.0 (başarılı yerleştirme)
       + (-10.0 if game_over else 0)
       + (-0.5 * max(0, holes_after - holes_before))
       + (-1.0 if invalid_action else 0)
```

**Sentetik doğrulama:** 15 adımlık deterministic rollout'ta her hole-yaratan hamlede +0.5, double-hole'da 0.0 reward; hole azalmasında ceza yok (asimetri ✓).

**Sonraki adım:** kullanıcı eğitimi başlatmadan önce daha fazla değişiklik biriktirebilir.

### v3'e ek: bumpiness + line_clear (eksponansiyel)

**Ne:**
- `count_empty_components(grid)` — boş 4-connected component sayısı (1=ideal, çok=saçılmış).
- env.py: pre/post-step delta_components → `reward += bumpiness_penalty * Δ` (asimetrik).
- env.py: line_clear sonucuna göre `reward += line_clear_base * line_clear_exp ^ (n-1)`.
- env_v3.yaml: `bumpiness_penalty: -0.1`, `line_clear_base: 10.0`, `line_clear_exp: 3.0` (1=10, 2=30, 3=90, 4=270, ...).
- env.yaml defaults 0.0 (geri uyumlu).
- info dict'e: `comps_before/after/delta`, `line_clear_reward` eklendi.
- 7 yeni test, pytest 27/27 yeşil.

**Niye bumpiness için "empty component sayısı":**
Block Blast'ta yerçekimi yok → klasik Tetris column-heights bumpiness anlamlı değil. Boş hücre adası sayısı doğrudan saçılma sezgisini ölçer; tek büyük boş bölge ⇒ kümelenmiş yerleştirme, çok küçük adacık ⇒ saçılmış. Bu metrik kenar/köşe teşvikini doğal olarak içerir (kenarda yerleştirme yeni component yaratmaz, ortada yerleştirme yaratabilir).

**Reward formülü (v3 final):**
```
reward = +1.0  step
       + (-10.0  if game_over)
       + (-0.5   × max(0, Δhole))                # 1-2 cell izole adalar
       + (-0.1   × max(0, Δempty_components))    # saçılma (bumpiness)
       + (10 × 3^(n-1) if cleared_lines > 0)     # 1=10, 2=30, 3=90, 4=270
       + (-1.0   if invalid_action)
```

**Doğrulama (500 random adım, 40 ep):**
- step toplam +500, line_clear +340, hole -122, bumpiness -31, game_over -400 → net +286 ✓ tutarlı.
- Random ajanda %85 ep en az 1 clear yapıyor; eğitilmiş ajan bu oranı çok yukarı çekmeli.

**Reward dengesi:**
- Survival ana eksen kalıyor (her step +1).
- Line clear güçlü pozitif (+10), ajanı clear kurmaya doğal teşvik.
- Hole / bumpiness küçük negatif, dağınıklık caydırılır ama asla overshadow etmiyor.
- 1.denemenin hatasından kaçındık: tek bir mega-shaping yok, **dengeli birkaç küçük terim**.

### v3'e ek 2: hole penalty güçlendirme (reward hacking önlemi)

**Ne:** `hole_penalty -0.5 → -2.0` (4× sertleşti).

**Niye:** Önceki -0.5 değerinde 1 line clear (+10) başına 20 hole tolere ediliyordu → ajan "delikleri umursamadan clear yap" davranışına yöneltilir (reward hacking). -2.0 ile 1 clear başına 5 hole tolere edilir; line clear hâlâ güçlü pozitif ama hole gerçekten cezalandırılıyor.

**Doğrulama (random ajan 500 adım):** önceki dengede toplam reward +218 net (hole önemsiz), yeni dengede −81 net (hole çok ciddi). Dengeli ajanın gerçek davranışı bu iki extremin ortasında olur.

### v3'e ek 3: mimari — bottleneck + early fusion

**Ne:**
1. **Bottleneck (stride=2 conv):** ikinci Conv2d stride=2 → feature map 8×8 → 4×4. Flatten 4096 → 1024. `Linear(1024→128)` 131K param (eski 524K, **4× düşüş**). Spatial bilgi 4×4 hâlâ uzamsal, GAP gibi tek vektöre indirmedik.
2. **Early fusion:** grid (4 frame) + placements (3 piece availability) CNN'den ÖNCE concat → **7-kanal input**. CNN aynı uzamsal düzlemde "tahta dolu mu?" + "her parça nereye yerleşebilir?" sinyallerini görür. Bu sayede tahta-parça korelasyonu MLP head yerine CNN'de doğal öğreniliyor.
3. **Placement planes bedava:** action_mask zaten 192-bool → `mask.reshape(3, 8, 8)` ile placements elde edilir, ek hesap yok.

**Mimari özet:**
```
Eski: Conv(4→32, s=1) → Conv(32→64, s=1) → Flatten 4096 → Linear→128
Yeni: cat(grid+placements)=(7,8,8) → Conv(7→32, s=1) → Conv(32→64, s=2) → Flatten 1024 → Linear→128
```

**Param dökümü (yeni):**

| Modül | Param | Eski v2 | Δ |
|---|---:|---:|---:|
| `grid_cnn` | 20,544 | 19,680 | +0.9K (7→32 ek kanal) |
| `grid_proj` | **131,200** | 524,416 | **−393K** |
| `tray_proj` | 10,816 | 10,816 | 0 |
| `history_embed` + `history_proj` | 3,632 | 3,632 | 0 |
| `meta_proj` | 80 | 80 | 0 |
| `head` | 127,488 | 127,488 | 0 |
| `mlp_extractor` + heads | 53,761 | 53,761 | 0 |
| **TOPLAM** | **347,521** | **739,873** | **−53%** |

**Test:** pytest 27/27 yeşil (yeni `placements` shape doğrulandı, mask reshape eşitliği test edildi).

### v3 final özet

| # | Değişiklik | Niye |
|---|---|---|
| 1 | batch_size 256 → 1024 | Stabil gradient (32K rollout / 1024) |
| 2 | hole_penalty −2.0 | Reward hacking önle, 5-hole/clear dengesi |
| 3 | bumpiness_penalty −0.1 × Δ empty_components | Saçılma cezası (kenar/köşe doğal teşvik) |
| 4 | line_clear +10 × 3^(n-1) | Kombo motivasyonu |
| 5 | Mimari: 7-kanal early fusion + stride=2 bottleneck | %53 daha az param, spatial korelasyon doğal |

**Toplam farklar (v2 → v3):**
- Reward: 1 alanlı (sadece survival) → 5 alanlı (survival + 3 shaping + line bonus)
- Observation: 5 field → 6 field (placements eklendi)
- Param: 740K → 347K (yarısından az)
- Optimizasyon: batch 256 → 1024
- Eğitim süresi tahmini: param azalması nedeniyle v2'nin %57 dakikasından kısa olmalı.

## 2026-05-06 — v3_full eğitimi başladı

**Komut:** `python -m src.scripts.train --config configs/ppo_v3.yaml --run-name v3_full --use-subproc`
**PID:** 36337, GPU CUDA, 10M step.

### Snapshot @ 00:48 (~3 dak, 1.08M / %10.8)

- fps = **3824** (v2 aynı dönem ~3500-3700; %3-9 hızlı; net kazanç sınırlı çünkü hole/bump hesabı her step ek iş)
- Eval ep_len son 5: 13.45, 18.55, 13.75, 13.45, 18.25 → ort **15.49**
- Eval ep_reward varyansı yüksek: 6-39 (line clear bonusu çoklu sinyal yaratıyor — beklenen).
- GPU %8 — env-bound (CNN küçülmesi GPU yükünü düşürdü)
- Tahmini bitiş: ~39 dak.

### Snapshot @ 00:53 (~7 dak, 2.13M / %21.3)

- fps = **3734** (v2'nin aynı dönemiyle eşit)
- Eval ep_len son 5: 14.85, 15.65, 16.50, 15.95, 16.15 → ort **15.82** (v2 ~16-17)
- Eval ep_reward son 5: 15.99, 16.15, 24.14, 22.87, 24.64 → ort **20.76** (line clear bonusu)
- GPU %10, 244 MiB
- Tahmini bitiş: ~35 dak

### Snapshot @ 00:57 (~11 dak, 3.05M / %30.5)

- fps = **3745**
- Eval ep_len son 5: 17.25, 19.75, 20.40, 16.60, 15.10 → ort **17.82** (v2 aynı timestep ~19.15)
- Eval ep_reward son 5 ort = **29.61** (zirve 47.11) — line clear bonusu sayesinde
- ep_len -%7, ep_reward +5x → ajan survival yerine clear'a daha çok eğiliyor
- Tahmini bitiş: ~31 dak

### Snapshot @ 01:01 (~15 dak, 3.95M / %39.5)

- fps = **3757**
- ep_len son 5 ort = **17.37** (16.85-17.85 sıkışık) — v2 aynı timestep'te 19.15
- ep_reward son 5 ort = **27.70** (line clear bonusu sayılmaya devam)
- **v3 ep_len v2'den geri:** 1M→15.49, 2M→15.82, 3M→17.82, 4M→17.37 (yatay/hafif geri)
- Hipotezler:
  - (a) Çoklu shaping sinyali öğrenmek için zaman gerekiyor; 7-10M'de yakalama olası.
  - (b) hole=-2.0 fazla agresif, ajan risksiz oynamaya zorlanıyor → clear için gerekli geçici kümeleri bile inşa etmiyor.
- Karar erken; v2 6M'den sonra hızlı yükselişe geçmişti. Devam.
- Tahmini bitiş: ~27 dak

### Snapshot @ 01:05 (~19 dak, 4.82M / %48.2)

- fps = **3733**
- ep_len son 5 ort = **17.86** (16.45-19.70 dağılık) — v2 aynı timestep'te 19.89 (-%10)
- ep_reward son 5 ort = 26.35
- **Önemli gözlem:** v3 1-2M'de v2'den önde (15.5 vs 13) ama 3M sonra v2 atılım yapıyor, v3 yatay kalıyor. Hole=-2.0 fazla agresif olabilir.
- Tahmini bitiş: ~23 dak

### Snapshot @ 01:09 (~23 dak, 5.85M / %58.5)

- fps = **3704**
- ep_len son 5 ort = **19.06** (önceki 17.86'dan yükselişte!) tek-eval zirve **21.00**
- ep_reward son 5 ort = 37.13
- v2 ile fark kapanmaya başladı (-%4); geç patlama hipotezi doğrulanıyor
- Trend: 17.86 (5M) → 19.06 (5.85M) — son 850K'da +%6.7
- Tahmini bitiş: ~19 dak

### Snapshot @ 01:14 (~30 dak, 6.77M / %67.7) — UYARI

- fps = **3693**
- ep_len son 5 ort = **18.31** (önceki 19.06'dan **GERİ**!)
- v3 ep_len trend: 19.06 → 18.67 → **18.31** (düşüş)
- v2 aynı dönemde ~21+ → fark **−%14, açılıyor**.
- **Hipotez:** line_clear +10 cazibesi survival'ı bastırıyor → ajan kısa-ama-clear-yoğun oyunlara optimize. ep_reward yüksek ama survival değil.
- Tahmini bitiş: ~14.5 dak

### Snapshot @ 01:18 (~34 dak, 7.62M / %76.2)

- fps = **3683**
- ep_len son 5: 20.20, 20.65, **21.70**, 18.60, **21.00** → ort **20.43**
- v3 trend: 18.67 (6.13M) → 18.31 (6.77M) → **20.43** (7.62M) — düşüş bitti, ZIPLAYIŞ
- v2 (7M) ep_len 21.09 → v3 (7.62M) 20.43 → fark sadece **−%3**
- Önceki "düşüş trendi" uyarısını revize: yüksek varyanslı öğrenme, lokal dipler vardı
- Tahmini bitiş: ~10.7 dak

### Snapshot @ 01:22 (~38 dak, 8.55M / %85.5) — VOLATİL

- ep_len son 5: 19.90, 16.10, 19.35, 16.95, 20.35 → ort **18.53**
- v3 trend (volatil): 19.06↑ 18.67↓ 18.31↓ 20.43↑ 18.53↓
- v2 (8M) = 22.36 → fark **−%17**, açıldı
- Reward shaping ajanı stabil survival politikasına oturtmuyor; line clear cazibesi sürüyor
- Tahmini bitiş: ~6.5 dak

### Snapshot @ 01:26 (~42 dak, 9.50M / %95)

- ep_len son 5: 19.90, 21.00, 17.45, 19.50, 19.80 → ort **19.53**
- v3 final fazda 19-20 plato'sunda; v2 (9.5M) ~21 → fark −%6-8
- Tahmini bitiş: ~2.3 dak

### v3_full TAMAMLANDI (10M step, ~46 dak)

`models/best/v3_full/best_model.zip` yazıldı. Son callback evalları yükselişte (21.15 → 22.75) ama **1000-ep formal eval** dengeli bir resim verdi.

**1000-ep eval sonuçları (ep_len odaklı):**

| Model | ep_len mean | median | max | clears/ep | score mean |
|---|---:|---:|---:|---:|---:|
| v1_survival (2M) | 18.4 | 17 | 62 | 3.27 | 103.5 |
| **v2_parallel (10M)** ⭐ | **21.3** | **20** | **62** | **4.67** | **129.1** |
| v3_full (10M) | 19.3 | 18 | 56 | 4.11 | 115.2 |

**Karar:** v3 v2'yi GEÇEMEDİ. Tüm metriklerde v2 net şampiyon:
- ep_len −%9.4
- clears/ep −%12
- score −%11
- max ep_len −%10

**Tanı:** 5 değişikliği aynı anda paketlemek net negatif. En şüpheli: hole=−2.0 fazla cezalandırıcı + line_clear bonusu kısa-yoğun oyunlara yönlendirici. Mimari değişikliği (param 740K→347K) muhtemelen masum hatta yararlı, ama izole test edilmeli.

**Sonraki adım: ablation pipeline (v4 paketleri):**
- v4a: yalnız mimari (early fusion + bottleneck), reward v2 ile aynı (sade survival)
- v4b: v4a + sadece line_clear bonus
- v4c: v4a + tray empty-flag
- gamma annealing daha sonra

## 2026-05-06 — v4a başlatıldı (ablation: yalnız mimari)

**Komut:** `python -m src.scripts.train --config configs/ppo_v4a.yaml --run-name v4a_arch_only --use-subproc`
**PID:** 88566 (CUDA, batch=256, env.yaml=sade survival).
**Hedef:** kullanıcı talebi — 5M step'te durdur, v2 5M (19.89) ve v3 5M (17.86) ile karşılaştır.

### Snapshot @ 01:47 (~7 dak, 1.51M / 5M-hedef %30)

- fps = **3119** (v2 ~3500, v3 ~3824 — beklenenin altında, placement_planes overhead?)
- ep_len son 5 ort = **15.26** (v2/v3 1.5M seviyesi ile paralel)
- Henüz mimari pozitif/negatif sinyal yok; 5M'e ~19 dak kaldı.

### Snapshot @ 01:50 (~10 dak, 1.97M / 5M-hedef %39)

- fps = **2948** (düşmeye devam, placement_planes overhead hipotezi güçleniyor)
- ep_len son 5 ort = **16.75** — v2 (15) ve v3 (15.82) **üzerinde** 🟢
- v4a 2M'de hem v2'yi hem v3'ü geçiyor → mimari değişikliği sample efficiency POZITIF sinyal.
- 5M'e ~17 dak kaldı.

### Snapshot @ 01:52 (~16 dak, 2.59M / 5M-hedef %52)

- fps = **2879**
- ep_len son 5: 16.90, 18.10, 20.90, 18.70, **22.45** → ort **19.41**
- Trend: 1M=15.26, 2M=16.65, **2.59M=19.41** (son 590K'da +2.76)
- v1 2M (~18.97) ve v2 2M (~17.33) ŞIMDI GEÇİLDİ. Tek-eval zirvesi 22.45 v2 final'i (21.3) bile aştı.
- Mimari değişikliği geç-patlama eğrisi — önceki "kötü" yorumum revize edildi.
- 5M'e ~14 dak.

### v4a TAMAMLANDI 10M (kullanıcı 5M durdurmayı iptal edip 10M'e devam dedi)

**1000-ep eval:**
- ep_len mean=**21.9** (median 20, max **65**)
- clears/ep=**4.96**, clears/move=**0.2264**
- score mean=**134.3**

**Tüm modellerle karşılaştırma:**

| Model | ep_len | clears/ep | clears/move | score |
|---|---:|---:|---:|---:|
| v1 (2M) | 18.4 | 3.27 | 0.1777 | 103.5 |
| v2 (10M, eski şampiyon) | 21.3 | 4.67 | 0.2192 | 129.1 |
| v3 (10M, fail) | 19.3 | 4.11 | 0.2123 | 115.2 |
| **v4a (10M)** ⭐ | **21.9** | **4.96** | **0.2264** | **134.3** |

**v4a vs v2 kazanım:** ep_len +%2.8, clears/ep +%6.2, score +%4.0, max ep_len +%4.8.

### YORUM — Ablation çıkarımı

v3 5 değişikliği (batch + 3 reward shaping + mimari) birlikte denemiş, v2'den geri kalmıştı.
v4a sadece mimariyi (early fusion + stride=2 bottleneck) değiştirip v2 reward'ını korudu.

**Sonuç:**
- ✅ Mimari değişikliği POZİTİF (parametre %53 az, performans %2.8 fazla)
- ❌ Reward shaping (hole=-2.0, bumpiness=-0.1, line_clear=10×3^(n-1)) NET NEGATİF
- ❓ batch=1024 etkisi henüz izole edilmedi (v4a'da 256, v3'te 1024)

**Yeni şampiyon: v4a_arch_only (10M).**

## 2026-05-06 — v4b TAMAMLANDI (Glass Cannon tedavi başarısız)

**Komut:** `python -m src.scripts.train --config configs/ppo_v4b.yaml --run-name v4b_optimized_reward --use-subproc` (10M step, ~50 dak, fps ~3700)

**Reward formülü:** game_over=-50, line_clear_table=[0,2,10,40,120], hole_penalty=-0.2, bumpiness=-0.05.

**1000-ep eval:**

| Model | ep_len | clears/ep | clears/move | score |
|---|---:|---:|---:|---:|
| v1 (2M) | 18.4 | 3.27 | 0.1777 | 103.5 |
| v2 (10M) | 21.3 | 4.67 | 0.2192 | 129.1 |
| v3 (10M) | 19.3 | 4.11 | 0.2123 | 115.2 |
| **v4a (10M)** ⭐ | **21.9** | **4.96** | **0.2264** | **134.3** |
| v4b (10M) | 17.1 | 2.83 | 0.1648 | 93.9 |

**v4b v4a'ya göre:** ep_len -%21.9, clears/ep -%43, score -%30. v4b **v1'den bile düşük** (18.4 → 17.1).

**Ablation kararı:** Bu projede reward shaping (game_over güçlendirme + line_clear tablo + hole/bumpiness) **NET NEGATİF**. Sade survival (+1 step, -10 GO) + iyi mimari = en iyi formül. v4a şampiyon kalmaya devam ediyor.

### Eğitim sırasındaki teşhis (kanıt)

- Pearson r(ep_reward, ep_len) = +0.934 → reward hacking YOK, formül tutarlı.
- Son 2M ep_reward slope ≈ -0.06 (durmuş), ep_len slope = +0.68 (yavaş yükseliyor) → ajan plato'da 19-20'de saplandı.
- 8.45M sonrası ep_len düşüşe geçti (20.22 → 18.54 → final 17.1).
- Hipotez: game_over=-50 ajanı korumacı yaptı, kombo öğrenemedi (clears/ep 4.96 → 2.83).

**Tasarım kararları (DESIGN.md'den özet):**
- Reward: +1 her step, -10 game_over, başka hiçbir şey 0.
- Observation: grid (frame stack 4) + tray + tray_shapes + history(8) + meta(4).
- Model: CNN 32→64, features_dim 256, ~150K parametre.
- PPO default-ish: gamma=0.99, vf_coef=0.5, batch=256, ent=0.01, n_envs=8, 2M step.
- Curriculum yok, VecNormalize yok.

**Birinci denemeden taşınan kanıtlanmış parçalar:**
- 31 parçalık `pieces.yaml` set'i (rotation'lar dahil)
- `Board.placement_mask` ve `Board.clear_full_lines` mantığı
- `Tray.refill_if_all_empty` davranışı (ama anti-frustration / ensure_solvable kapalı — ajanın gerçek dünyayla tanışması için)

**İptal edilen mekanizmalar (1. denemeden):**
- `Scoring` modülü kaldırıldı — score tracking var ama reward'a girmez (sadece info).
- Combo, lines_table, board_clear_bonus, cell_score → hepsi info-only, eğitim sinyali değil.
- Anti-frustration tray — ajanın gerçek dağılımla başa çıkmasını öğrenmesi gerek.
- Ensure-solvable refill — ajan game-over olabilmeli ki hatalardan öğrensin.

**Sonraki adım (kullanıcı promptunu bekliyor):** muhtemelen `python -m src.scripts.train` ile 2M step eğitim.

---
