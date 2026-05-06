# Block Blast — Survival-First RL (2. Deneme)

Bu proje birinci denemeden çıkarılan derslere dayanır. Birinci denemede modeli "puan + combo + uzun yaşama"yı **aynı anda** öğrenmeye zorlamak, ajanı temel hayatta kalmayı bile kavrayamadan boğdu. Burada felsefe çok daha sade:

> **Önce hayatta kal. Puan, combo ve diğer her şey hayatta kalmanın doğal sonucu olsun.**

## Kısa Özet

- **Oyun:** 8×8 Block Blast, 3-slot tray. Headless / Pygame'siz çalışır.
- **Algoritma:** MaskablePPO (sb3-contrib).
- **Reward:** Sadece **+1.0 her başarılı yerleştirme**, **-10 game over**. Hiçbir oyun-içi puan, line clear, combo veya holes shaping yok.
- **Observation:** 8×8 grid + 3 parça (bool tensor) + son 4 frame stack + son 8 hamle history.
- **Hedef:** Modelin önce *hayatta kalma davranışı* edinmesi; karmaşık skor optimizasyonu sonraki fazlara bırakılır.

## Hızlı Başlangıç

```bash
cd "/home/emir/Emir/BlockBlast 2. deneme"
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -q                                     # smoke test
python -m src.scripts.train --config configs/ppo.yaml  # eğitim
python -m src.scripts.eval --model models/best/best_model.zip --episodes 200
```

## Dökümantasyon

- [`DESIGN.md`](DESIGN.md) — Tasarım kararları (neden sade reward, neden frame stack, neden curriculum yok)
- [`DOCS.md`](DOCS.md) — Yaşayan dökümantasyon: ne yaptık, ne işe yaradı, ne yaramadı
- [`docs/architecture.md`](docs/architecture.md) — State / action / reward formel tanımı
- [`docs/log.md`](docs/log.md) — Zaman çizelgesi (tüm değişiklikler tarihli)

## Birinci Denemenin Dersleri

Önceki proje `/home/emir/Emir/block blast/` altında duruyor. Oradaki başarısızlık özeti:
- Reward'a aynı anda 5+ shaping terimi yığdık (score, holes, bumpiness, max_height, survival, game_over).
- META_DIM 4 → 16 büyütüldü (col heights, death proximity, survival_norm, recent_clears…).
- 14 → 19 → 31 parça curriculum, üstel combo, VecNormalize → hepsi aynı anda.
- **Sonuç:** v5 modelimiz v4 baseline'dan %52 daha düşük mean skor, %25 daha kısa episode aldı.
- **Çıkarım:** Birden fazla zor değişikliği aynı anda yapmak öğrenmeyi *engeller*. Önce tek bir hedef → sonra üzerine ekle.

Bu repoda baştan başlıyoruz: **tek hedef = hayatta kalmak**.
