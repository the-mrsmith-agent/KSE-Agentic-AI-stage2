# EDA — air-raid alerts (national + per oblast)

All numbers below are produced by `python -m src.eda`; figures in `reports/figures/`.

## Data
- Window: 2022-02-25 → 2026-10-11 (UTC), 115,986 alert-starts, 24 regions.
- Censored: naive=5.01%; duration unusable (naive or >24h)=5.02%.

## Volume / regime
- Starts/day by year: 2022 49.1, 2023 52.2, 2024 69.2, 2025 67.5, 2026 111.8.
- 2026 vs 2022 volume ratio: **2.28×** (per-day, partial-year safe).
- Change point (2026-08-26, 28-day window): mean 117.5 → 151.0 starts/day (upward shift).

## Seasonality
- Peak local hour-of-week: **Mon 12:00** (Europe/Kyiv).
- Median gap between consecutive alerts: **3.61 h**; p90 gap: **17.16 h** (119 days max).

## Per-oblast burden (sorted by total known alert-minutes)
- Top 5:
  - Donetska oblast: 605,958 min, 8,546 starts, 506.8 starts/100d
  - Zaporizka oblast: 462,884 min, 8,969 starts, 531.3 starts/100d
  - Dnipropetrovska oblast: 451,966 min, 13,728 starts, 813.9 starts/100d
  - Kharkivska oblast: 422,681 min, 12,980 starts, 769.0 starts/100d
  - Sumska oblast: 358,914 min, 8,737 starts, 517.7 starts/100d
- Bottom 3:
  - Chernivetska oblast: 60,643 min, 914 starts, 54.2 starts/100d
  - Ivano-Frankivska oblast: 50,156 min, 893 starts, 53.0 starts/100d
  - Zakarpatska oblast: 46,136 min, 761 starts, 47.5 starts/100d

## Clustering
- 12% of gaps are <1 h; 6% are >24 h → repeated starts inside mass attacks plus long quiet tails.
- Longest current quiet stretch: 73.0 days.

## Figures
- `daily_starts.png`, `hour_of_week_heatmap.png`, `monthly_burden.png`, `gaps_and_quiet.png`, `region_burden.png`, `durations.png`
