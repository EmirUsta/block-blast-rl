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
