# Zaman Çizelgesi

En yeni en üstte. Her giriş tarihli ve kısa: bir cümle yetiyorsa bir cümle.

## 2026-05-05

- [done] Proje iskeleti: `configs/`, `src/{game,env,ai,scripts}/`, `tests/`, `docs/`, `models/`, `logs/`.
- [done] `pieces.yaml` 1. denemeden kopyalandı (31 parça, rotation'lar dahil).
- [done] Game core sade halde yazıldı (Scoring kaldırıldı, score sadece info için).
- [done] `BlockBlastEnv` survival-only reward ile yazıldı (+1 step, -10 game_over, başka bir şey 0).
- [done] Observation: grid frame stack 4 + tray + tray_shapes + history(8) + meta(4).
- [done] `BlockBlastFeatureExtractor` basit CNN+MLP + history embedding.
- [done] `train.py`, `eval.py` minimal.
- [done] `tests/test_smoke.py` — Game step, env reset/step/reward, masking smoke.
- [done] `tests/test_policy_smoke.py` — MaskablePPO build + kısa learn + predict döngüsü.
- [done] Pytest 10/10 yeşil (3.67s; 1. denemenin venv'si reuse edildi).
- [next] Eğitim — kullanıcı promptu bekleniyor.
- [done] v1_survival eğitimi başlatıldı (2M step, GPU, run-name=v1_survival, PID 31144).
- [done] TensorBoard 6006'da çalışıyor (logdir=logs/tensorboard).
- [next] Periyodik ilerleme kontrolü.
- [done] v1_survival 2M tamam (~14 dak, GPU). 1000 ep eval: ep_len mean=18.4, score=103.5, clears/ep=3.27.
- [done] v2_parallel 10M tamam (~57 dak, SubprocVecEnv n_envs=16, 1.5x throughput). 1000 ep eval: **ep_len mean=21.3, score=129.1, clears/ep=4.67** — v4_cont baseline (20.78) survival'da geçildi.

## 2026-05-06

- [done] v3 hazırlık: count_small_islands metric, env hole_penalty integration, env_v3.yaml (-0.5), ppo_v3.yaml (batch=1024 + env_v3). pytest 20/20 yeşil.
- [next] Kullanıcı v3'e başka değişiklik(ler) eklemek istiyor → biriktiriyoruz.
- [done] v3'e ek: count_empty_components, bumpiness_penalty -0.1, line_clear_base 10 + exp 3. pytest 27/27 yeşil. 500-step random sanity ✓.
- [done] v3'e ek: hole_penalty -0.5 → -2.0 (reward hacking dengesi).
- [done] v3 mimari: early fusion (grid+placements 7-kanal CNN) + bottleneck (stride=2, 4×4 flatten). Param 740K → **347K** (-%53). pytest 27/27 yeşil.
- [done] v3_full eğitim 10M tamam (~46 dak). 1000-ep eval: ep_len 19.3, clears 4.11, score 115.2. **v2'nin (21.3 / 4.67 / 129.1) altında kaldı**, tüm metriklerde -%9-12. Reward shaping fazla cezalandırıcı + line clear cazibesi → kısa yoğun oyun davranışı.
- [next] v4 ablation pipeline (mimari ↔ reward ↔ batch ayrı test).
- [done] v4a (sadece mimari, sade reward, 10M) eğitim tamam. 1000-ep eval: ep_len **21.9**, clears/ep **4.96**, score **134.3** — v2'yi tüm metriklerde geçti, **YENİ ŞAMPİYON**. Mimari değişikliği pozitif sample efficiency, reward shaping (v3) net negatif kanıtlandı.
- [done] v4b (mimari aynı + dengeli reward shaping: game_over=-50, line tablo, hole=-0.2, bump=-0.05) 10M tamam. 1000-ep eval: ep_len **17.1**, clears/ep **2.83**, score **93.9**. v4a'ya göre tüm metriklerde -%22 ile -%43 arası kayıp. v1'den (18.4) bile düşük. **Ablation kararı: bu projede reward shaping net negatif, v4a şampiyon.**
- [done] py-spy profil: SubprocVec n=16 IPC dominant (worker'lar %83 idle pipe.recv'de, env.step toplamın %1.2'si). Pickle marjinal (%0.04). Karar: vectorized batch-env yaz, Numba/Cython gerek yok.
- [done] `BlockBlastBatchEnv(VecEnv)` yazıldı (~370 satır): pad-true sliding window placement (12×12), 5×5 scratch scatter place, vectorized line clear/refill/game_over, action_masks (N,192). Behavioral parity: 10K random state → 0 mismatch; ep_length KS-test D=0.028 p=0.99. Strict identity tray refill RNG path farkı sebebiyle FAIL ama dağılım eşit (envpool/brax pattern).
- [done] Throughput sweep N∈{16,32,64,128,256} v4a, random policy 60s: fps 17.5K → 20.6K, plato N=128. PPO 200K: N=16 4,682 ts/s; N=64 6,925 ts/s + GPU avg %27 / p95 %60 (en doygun); N=128 7,305 ts/s ama horizon 256 → gradient noise. **Önerilen: N=64, n_steps=512, batch=1024 (rollout 32K SABİT)**.
- [done] v4b N=64 (BFS aktif) 6,381 fps — v4a N=64'in (19,320) **3× yavaşı**; reward shaping kullanılırsa hole/component vectorize edilmeli.
- [done] 500K sanity training KS-test pencereleri (50K boy, 10 pencere): batch16 vs subproc16 (pure backend) 8/10 PASS, |Δ|=0.01; batch64 vs subproc16 (önerilen config) 8/10 PASS, |Δ|=0.03. **Onay: backend + config eğitim eğrisini bozmuyor.**
- [done] **v5_batch64 10M (BatchEnv N=64, n_steps=512, batch=1024) wallclock 23.5 dk** (v4a SubprocVec ~46 dk → −%49). 1000-ep eval: ep_len **22.5**, clears/ep **5.20**, score **139.2**, clears/move **0.2315**. v4a'yı (21.9/4.96/134.3/0.2264) **her metrikte geçti**, **YENİ ŞAMPİYON**.
- [done] Repo GitHub'a yüklendi (private): `EmirUsta/block-blast-rl`, SSH deploy key + host alias.
- [done] **JAX GPU-resident env** yazıldı (`src/env/jax_env.py` + `src/env/jax_vec_wrapper.py`): jit-compiled step kernel, fancy-index sliding window action mask (12×12 pad-true scratch), vmap'li OR-scatter place, tek master key PRNG. Drop-in SB3 VecEnv, obs Dict layout v4a policy ile birebir aynı. Reward shaping (hole/component) build-time flag ile bypass'lı (v4a default'unda compile dışı). flax.struct.dataclass state. Auto-reset + terminal_observation pattern numpy env'in birebir kopyası.
- [done] Parity testleri (`tests/test_jax_env_parity.py`) 2/2 PASS — Test 5 (10K random state action_mask byte-equivalence CPU mode) 0 mismatch; Test 4 (KS distribution 500 ep N=64 random policy) 10/10 pencere PASS, |Δmean|=0.86 < 1.5.
- [done] **JAX throughput sweep** (`examples/jax_throughput.py`, RTX 4050 6 GB, random policy 30s/N): N=64 39K ts/s, N=256 104K, N=512 152K, **N=2048 208K (sweet spot)**, N=8192 147K (VRAM saturation). v5_batch64 numpy N=64 7,200 ts/s baseline'ına göre **~29×** hızlanma. GPU util %13 (random policy; PPO loop'unda forward+backward yük bindiğinde doygunluk artar). Hedefler: N=512 30K+ ✓, N=2048 100K+ ✓ (2×), N=8192 200K+ ✗ (6 GB VRAM saturation).
- [done] PPO 200K bench (3 config, rollout 32K SABİT): v5_batch64 numpy 5,720 ts/s (baseline) | v6_jax_512 9,729 ts/s (1.78×) | v6_jax_2048 9,795 ts/s (1.71×). Random policy 200K → PPO loop 9.8K = ~%95 kayıp; profile (`logs/jax_ppo_profile.txt`) host roundtrip toplam %0.7 → dlpack ÖLÜ. Asıl bottleneck: PPO update + sync barrier.
- [done] JAX wrapper optimizasyon (`src/env/jax_vec_wrapper.py`): `block_until_ready` __init__/reset/seed'den kaldırıldı (np.asarray implicit sync); step_wait'te tek stratejik block korundu (open-loop async memory pressure RSS 5.1 GB regresyona neden olmuş v2'de); **lazy info path** done.any()=False çoğu step için `[{} for _ in range(N)]` sabit-zaman, terminal_observation pre_obs full host'a sadece done varsa. v3 bench: v6_jax_2048 **10,256 ts/s = 1.87×** (v1: 1.71×), parity 2/2 korundu, profile "Other" %10→0%.
- [done] **v6_jax_2048 10M training (JAX env + N=2048 + n_steps=16) wallclock 14 dk 14 sn** (v5_batch64 23.5 dk → **−%40**). 1000-ep eval: ep_len **20.6** median **20** max **59**, score **123.4**, clears/ep **4.33**, clears/move **0.2097**, reward/ep **10.64**. v5_batch64 (22.5 / 20 / 56 / 139.2 / 5.20 / 0.2315 / 12.45) ile karşılaştırınca: ep_len -%8.4, score -%11.4, clears/ep -%16.7, clears/move -%9.4. **Trade-off**: wallclock −%40 ama sample efficiency düşük (n_steps=16 horizon kısa → GAE bootstrap zayıf). **Hipotez**: n_steps < mean_ep_length (16 < 22.5) horizon kalibrasyon hatası, backend bug değil.
- [done] **v6b_jax_512x64 10M training (JAX env + N=512 + n_steps=64, rollout 32K SABİT) wallclock 14 dk 56 sn** (v5_batch64 23.5 dk → **−%36.5**). n_steps/mean_ep_len = 64/22.5 ≈ 2.84× kalibre. 1000-ep eval: ep_len **22.6** median **21** max **68** min 8, score **140.7** median 127 max 485, clears/ep **5.28** max 23, clears/move **0.2338**, reward/ep **12.57**. v5_batch64 (22.5/20/56/139.2/5.20/0.2315/12.45) ile karşılaştırınca **HER METRİKTE GEÇTİ**: ep_len +%0.4, median +%5, max +%21, score +%1.1, clears/ep +%1.5, clears/move +%1.0, reward +%1.0. v6_jax_2048'i tüm metriklerde 10-21% farkla geçti. **Hipotez doğrulandı**: regresyon n_steps horizon kalibrasyon hatasıydı. **YENİ ŞAMPİYON v6b_jax_512x64** — JAX backend + kalibre rollout. v5 bench için saklı.
