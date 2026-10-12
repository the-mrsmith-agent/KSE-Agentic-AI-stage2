# Air-raid alerts in Ukraine — the short version

116,171 air-raid alerts across 24 regions of Ukraine, 25 Feb 2022 → 12 Oct 2026. All times are Kyiv time.

## Life under sirens
- 2026 brings about 112 siren starts a day across the country; in 2022 it was 49 — **2.3× more**. Most of the rise is recent: the daily average climbed again in late August 2026.
- Half of all alerts are over within 31 minutes; 9 in 10 within 1.8 hours.
- Alerts come in waves: in a region, the next siren usually starts within 3.6 h of the previous one.
- About 3 in 10 alerts start at night (22:00–06:00).

![Alert starts per day since Feb 2022 — the level keeps rising](figures/daily_starts.png)

## When, and where, they happen
- Busiest slot of the week: **Monday around 12:00**. Quietest: Sunday around 07:00.
- Most siren time per day on average since 2022: Donetska oblast ~6.0 h, Zaporizka oblast ~4.6 h, Dnipropetrovska oblast ~4.5 h.
- Least: Chernivetska oblast ~0.6 h, Ivano-Frankivska oblast ~0.5 h, Zakarpatska oblast ~0.5 h.

![Alert starts by weekday and hour, Kyiv time — Monday noon is the busiest slot](figures/hour_of_week_heatmap.png)

## What we tried to predict, and how it went

Test: the last 14 days of data (28 Sep 2026 → 11 Oct 2026). One forecast per region per day, each made the evening before, then checked against what really happened. The forecast never saw these days while learning.

**How many alerts to expect — the forecast genuinely helps.**
- Off by about **1.5 alerts** per region per day; the simple "assume a typical recent day" guess is off by 2.2.
- Countrywide: about **11 alerts a day** wrong, against about 22 for the simple guess.

| Day | Sirens that really started | Model guessed | Simple guess |
|---|---|---|---|
| Mon 28 Sep | 118 | 139 | 149 |
| Tue 29 Sep | 143 | 134 | 147 |
| Wed 30 Sep | 133 | 136 | 167 |
| Thu 01 Oct | 124 | 137 | 153 |
| Fri 02 Oct | 132 | 134 | 158 |
| Sat 03 Oct | 117 | 133 | 124 |
| Sun 04 Oct | 120 | 132 | 112 |
| Mon 05 Oct | 126 | 135 | 118 |
| Tue 06 Oct | 134 | 134 | 143 |
| Wed 07 Oct | 141 | 138 | 133 |
| Thu 08 Oct | 132 | 144 | 124 |
| Fri 09 Oct | 166 | 148 | 132 |
| Sat 10 Oct | 171 | 145 | 117 |
| Sun 11 Oct | 164 | 157 | 120 |

![Last 14 days: what really happened vs the model's guess vs the simple guess](figures/eval_national_daily.png)

**Which hour an alert starts — about half right.**
- Of the hours flagged as risky, about **5 in 10** really had an alert; it named the right hour for about 6 in 10 of the alerts that really happened. The simple guesses manage about 4 in 10 either way.
- The exact minute is not predictable — nobody got close on that.

**How long an alert lasts — essentially unsolved.**
- Length guesses are off by about **19 minutes** on average; always answering "the usual length" is off by 20 minutes. The "between X and Y minutes" range held up about 77 times in 100.

**Where the burden falls in the test period** (the 8 busiest regions by siren time):

| Region | Hours under sirens per day (real) | Model guessed |
|---|---|---|
| Kharkivska oblast | 6.3 | 5.6 |
| Mykolaivska oblast | 4.9 | 5.0 |
| Dnipropetrovska oblast | 4.6 | 5.4 |
| Kirovohradska oblast | 4.5 | 4.6 |
| Kyivska oblast | 4.0 | 4.7 |
| Zaporizka oblast | 4.0 | 4.2 |
| Chernihivska oblast | 4.0 | 4.1 |
| Poltavska oblast | 3.8 | 4.5 |

## What this is not
- Not an early-warning system. It says a day and a region look busy; it cannot say "sirens at 14:03 in your street".
- Oblast-wide only — no city, no district.
- Volunteer-collected data: some alerts are missing, and Luhansk and Crimea are absent (their sirens are effectively permanent).

Technical version, with the full scoring rules and how we stopped the forecast peeking at the future: `reports/findings.md`. Figures: `reports/figures/`.
