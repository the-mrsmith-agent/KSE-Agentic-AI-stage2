# How the models evolved

Short version of the development story behind `findings.md` §4: the readable model came first,
and the second model is an iteration on the same inputs, built to be more accurate.

## 1. The bar to clear: "alerts happen at the usual times"

Before any model, two naive guesses set the bar: the average alert rate for that region,
weekday and hour ("hour-of-week profile"), and "same hour last week" (seasonal naive). Any
model that cannot beat these is useless, no matter how fancy.

## 2. First model: human-readable (Poisson + logistic)

Same feature matrix as everything after it, but deliberately simple learners: a Poisson
regression for daily counts and a logistic regression for "an alert starts in this hour".
Goal: numbers a human can explain — coefficients show what pushes risk up (many regions
alerting at once in the last 3 hours, high trailing alert-time) and down.

Result: already beats both naive guesses on counts (wrong by 2.05 alerts/region-day vs 2.23
and 2.49; countrywide 12.3 vs 21.7 and 33.7) and is competitive on timing (0.54 vs 0.49 and
0.40). The lesson from this step: *how many* and *roughly which hour* are learnable at all.

## 3. Second model: gradient boosting, iterated

Same inputs, more expressive learner (gradient boosting on decision trees). It was not
one-shot: on 3 rolling-origin folds we swept learning rate × tree count (0.05/0.1 × 100/300)
and took the setting that won consistently across folds — 0.05 with 100 trees for counts,
300 for duration. Different loss per task: Poisson-style loss for counts and alert-minutes,
quantile loss for durations.

Result vs the first model:

| Question | Model 1 (readable) | Model 2 (boosting) | What the iteration bought |
|---|---|---|---|
| How many alerts/day | 2.05 wrong | **1.53 wrong** | ~25% less error; countrywide 12.3 → 10.9 |
| Alert-minutes/day | 83 wrong | **79 wrong** | modest, same direction |
| Which hour starts | 0.54 | 0.54 | nothing — a tie |
| How long it lasts | not built | 0.80 vs 0.86 for the median guess | barely anything |

## 4. What we took away

- Use **Model 2 for the count** — the accuracy gain is real and it is the main deliverable.
- Use **Model 1 when the "why" matters** — it is just as good at hourly timing and its
  coefficients are readable; Model 2 cannot offer that.
- **Duration stays unsolved** for both approaches: neither beat answering "the usual length",
  and the promised ranges held up about 77 times in 100 instead of 80. Reported as a failure,
  not a feature.

Discipline that held across both iterations: one shared feature matrix (so the comparison is
about the learner, not the data), splits frozen in time order, folds used only for choosing
settings, and the last 14 days read exactly once.
