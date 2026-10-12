# EDA — air-raid alerts (national + per oblast)

All numbers below are produced by `python -m src.eda`; figures in `reports/figures/`.

## Data
- Window: 2022-02-25 → 2026-10-12 (UTC), 116,171 alert-starts, 24 regions.
- Censored: naive=5.02%; duration unusable (naive or >24h)=5.03%.

## Volume / regime
- Starts/day by year: 2022 49.1, 2023 52.2, 2024 69.2, 2025 67.5, 2026 112.1.
- 2026 vs 2022 volume ratio: **2.28×** (per-day, partial-year safe).
- Change point (2026-08-26, 28-day window): mean 117.5 → 151.0 starts/day (upward shift).

## Seasonality
- Peak local hour-of-week: **Mon 12:00** (Europe/Kyiv).
- Median gap between consecutive alerts: **3.61 h**; p90 gap: **17.13 h** (119 days max).

## Per-oblast burden (sorted by total known alert-minutes)
- Top 5:
  - Donetska oblast: 606,154 min, 8,549 starts, 506.6 starts/100d
  - Zaporizka oblast: 463,021 min, 8,973 starts, 531.2 starts/100d
  - Dnipropetrovska oblast: 452,213 min, 13,752 starts, 814.9 starts/100d
  - Kharkivska oblast: 422,936 min, 13,001 starts, 769.8 starts/100d
  - Sumska oblast: 359,061 min, 8,744 starts, 517.8 starts/100d
- Bottom 3:
  - Chernivetska oblast: 60,775 min, 917 starts, 54.4 starts/100d
  - Ivano-Frankivska oblast: 50,156 min, 895 starts, 53.1 starts/100d
  - Zakarpatska oblast: 46,136 min, 761 starts, 47.5 starts/100d

## Clustering
- 12% of gaps are <1 h; 6% are >24 h → repeated starts inside mass attacks plus long quiet tails.
- Longest current quiet stretch: 74.0 days.

## Figures
- `daily_starts.png`, `hour_of_week_heatmap.png`, `monthly_burden.png`, `gaps_and_quiet.png`, `region_burden.png`, `durations.png`
