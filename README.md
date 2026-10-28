# Crypto Regime Model — "How likely is a trend over the next N hours?"

An XGBoost classifier that estimates, on every closed **4-hour** candle, the **probability
that the next 4 / 8 / 12 / 16 / 24 hours will be a *trending* rather than a *ranging*
regime**. It answers questions like:

> BTCUSDT · next 12h · **70.8 %** probability of a TRENDING regime → high confidence

and it is built to be *honest about its own skill*: purged walk-forward validation, an
untouched holdout period, five automatic leakage audits, overfitting diagnostics, calibration
that is discarded when it does not help, and confidence intervals that account for
overlapping labels.

| file | what it is |
|---|---|
| **`crypto_regime_xgboost.ipynb`** | the runnable notebook (open in Colab, `Runtime → Run all`) |
| `crypto_regime_xgboost.py` | same code as a plain module (`python crypto_regime_xgboost.py`) — the single source of truth |
| `build_notebook.py` | regenerates the `.ipynb` from the `.py` (`--check` to verify) |
| `smoke_test.py` | offline end-to-end self-test, no internet needed |
| `data/raw/` | small real Binance 4h CSV samples (for the `csv` data source / offline runs) |
| `results/validation_*` | metrics, decision tables and plots from the runs described below |

---

## Quick start

**Google Colab (recommended)**

1. Upload `crypto_regime_xgboost.ipynb` (or open it from GitHub via *File → Open notebook*).
2. `Runtime → Change runtime type → Python 3` — CPU is fine, a GPU gives nothing here.
3. `Runtime → Run all`. Klines are downloaded automatically from `data.binance.vision`
   (no API key, no rate limits). First run ≈ 5–12 min, later runs reuse `data/cache/`.
4. Read the two output blocks you care about:
   * **section 13** → the test-set summary table, the overfitting report, the CV fold table,
     the feature importances and the live probabilities;
   * **section 12** → the sentence-form answer ("…next 12h: 70.8 % probability of TRENDING").

**Locally**

```bash
pip install -r requirements.txt
python smoke_test.py --quick      # 1 min, offline, checks the whole pipeline
python crypto_regime_xgboost.py   # full run, needs internet for the kline download
```

---

## What you get (real output of this repo)

Live probabilities for the latest closed bar, every symbol × every horizon (this is the real
content of `results/validation_multisymbol/live_probabilities.csv`):

```
symbol                     as_of                       horizon_hours  p_trending   regime confidence
BTCUSDT 2025-04-30 20:00:00+00:00             24      0.613       TRENDING        low
ETHUSDT 2025-04-30 20:00:00+00:00             24      0.581       TRENDING        low
SOLUSDT 2025-04-30 20:00:00+00:00             24      0.552       TRENDING        low

2025-04-30 20:00 UTC | BTCUSDT | next 24h: 61.3% probability of a TRENDING regime
                       (-> TRENDING, low confidence)
```

`confidence` is `high` once the probability is at least 0.15 away from 0.5, and it is a
statement about the probability, not about the edge — check the decision table before acting.

Test-set summary (untouched last 20 % of history; `horizon_hours`, base rate, and the
honest error bars from a moving-block bootstrap):

```
 horizon_hours  n_test  base_rate  accuracy            roc_auc            log_loss   label
             4    1330     0.4992    0.6053 [0.580,0.627]  0.6589 [0.627,0.681]   0.6652  move_z
             8    1330     0.5038    0.5045 [0.482,0.531]  0.5171 [0.490,0.550]   0.6932  r2
            12    1330     0.5241    0.5218 [0.498,0.552]  0.5084 [0.480,0.550]   0.6961  efficiency
            16    1330     0.5045    0.5090 [0.479,0.539]  0.5327 [0.501,0.575]   0.6970  efficiency
            24    1329     0.4951    0.5222 [0.491,0.560]  0.5588 [0.524,0.594]   0.6915  efficiency
```

And the table that makes a probability *actionable* — "if I only act when the model says
≥ X %, how often was it right, and how often does that happen?":

```
 threshold  coverage  n_flagged  precision  lift_vs_base
      0.50    0.4942        729     0.5967        0.1958
      0.60    0.2292        338     0.6538        0.3104
      0.70    0.0712        105     0.6857        0.3742
      0.80    0.0203         30     0.8333        0.6701
```

**Read this before you get excited:** the 4h horizon carries real, repeatable signal
(AUC ≈ 0.64–0.66, +10 pts accuracy over the base rate). The 8–24h horizons were **not**
distinguishable from chance on the short samples I could validate with here (see
[Measured results](#measured-results-and-what-they-mean)). That is a data limitation, not a
bug — the same code learns the long horizons on synthetic regime-switching data
(AUC 0.60+ at 24h, see `smoke_test.py`), and the Colab default uses 2019→today × 3 symbols
(~40× more rows than the validations below). Report whatever *your* run prints, not what
you hope for: `logloss_skill_vs_base > 0` and `auc 95 % CI` excluding 0.5 are the pass marks.

---

## How the target is defined ("trending" vs "ranging")

For horizon `H` bars, look at the forward path `P[t … t+H]` and compute Kaufman's
**efficiency ratio**:

```
trend_score(t) = |log P[t+H] − log P[t]| / Σ |log P[i] − log P[i−1]|
```

* `≈ 1` → price moved in a straight line → **trending**
* `≈ 0` → price travelled far but ended where it started → **ranging**

It is scale-free and volatility-normalised, so BTC 2021 and SOL 2024 are comparable.
The label is `trend_score > threshold`, where the threshold is the **median of the training
region only** (per symbol), which keeps classes balanced (~50/50) and keeps the test period
out of the definition. Alternatives are built in: `label_method="r2"` (R² of a line fitted to
the forward path) and `label_method="sign_persistence"`.

**Why the label definition adapts at 4h/8h.** On a 4h timeframe the 4-hour horizon is one
bar, and the efficiency ratio is *identically 1.0* (numerator = denominator) — it cannot
separate anything. At 8 hours (2 bars) it can only take the values {0, 0.5, 1}. So the code
falls back automatically (and says so in the log):

| horizon | label used | why |
|---|---|---|
| 4h (1 bar) | `move_z` = \|net move\| / trailing σ | measures "a directional move bigger than normal" |
| 8h (2 bars) | `r2` = R² of the forward path | ER is too coarse with 2 steps |
| 12h / 16h / 24h | `efficiency` | the normal definition |

Want a stricter notion of "trending"? `Config(label_quantile=0.65)` labels only the most
directional 35 % of windows as trending; `Config(label_absolute_threshold=0.35)` fixes the ER
cut-off in advance so the label never depends on data at all.

---

## Features (≈75, all strictly causal)

Every feature at bar `t` uses bars `≤ t`; labels use bars `t+1 … t+H`, so the two windows
never overlap.

* **Momentum** — log returns over 1/2/3/6/12/24/48/96 bars, normalised by trailing volatility.
* **Volatility** — realised vol at 5 scales, Parkinson, Rogers–Satchell, ATR %, vol-of-vol,
  vol ratios, up/down vol asymmetry, volatility z-score.
* **Trend strength** — ADX/±DI, Aroon, MACD, EMA slope, **efficiency ratio at 2/3/4/6/12/24/48
  bars** (matching the label horizons), R² and t-stat of rolling linear fits, Hurst exponent,
  Lo–MacKinlay variance ratio, sign persistence, turn frequency, return autocorrelation.
  *Past trendiness is the strongest predictor of future trendiness — regimes cluster.*
* **Mean reversion / location** — z-scores, Bollinger %B and width, RSI, Stochastic,
  Donchian position, ATR-distance to 48-bar high/low.
* **Volume / order flow** — volume z-scores and ratios, taker-buy imbalance, OBV slope,
  Amihud illiquidity, VWAP distance, trade-count z-score.
* **Cross-asset** — rolling correlation and beta to BTC, relative momentum, BTC volatility /
  ADX / Hurst state, plus **market breadth**: share of the universe trending up, share
  trending down, mean and dispersion of trendiness. Symbol one-hot encodings let one pooled
  model learn per-asset behaviour.
* **Time** — cyclical hour-of-day and day-of-week encodings, weekend flag.
  Crypto trades 24/7 but liquidity has a daily cycle; the model uses them and they show up in
  the importances, so they stay. Delete them if you prefer a purely price-driven model.

Winsorising bounds, constant-column removal and NaN fills are all fitted on the training
region only.

---

## Data engineering

`data.binance.vision` monthly archives → cached CSV → de-duplicated → **reindexed onto the
exact 4h grid** → sanity checks (`high ≥ low`, positive prices) → gap flags.

Three bugs this catches that quietly ruin crypto models:

1. **Microsecond timestamps.** Binance switched kline timestamps from ms to µs in 2025. A
   file that mixes old and new months spans ~50,000 years if you assume one unit. Timestamps
   are normalised per value (`to_millis`), and an absurdly dense time grid raises an error
   instead of exploding memory.
2. **Exchange downtime.** Any bar inside a gap, and any row whose 96-bar lookback window
   crosses one, is dropped (`drop_gap_windows=True`) — features computed across an outage are
   fiction.
3. **Overlapping monthly files / header rows.** De-duplicated on `open_time`; both headered
   and header-less kline files parse.

Data sources: `binance_vision` (default), `binance_rest` (paginated fallback), `csv` (your own
files, e.g. `data/raw/`), `synthetic` (offline self-test with a 3-state regime-switching
market). A local archive mirror works too: `Config(vision_base="file:///path/to/data/spot")`.

---

## Leakage protection

Five controls, and four of them are **executed automatically** and fail loudly:

| # | protection | mechanism |
|---|---|---|
| 1 | Label threshold, winsorising, constants and NaN fills | fitted on the pre-test region only |
| 2 | Calibration | fitted on **out-of-fold** predictions, never on test; kept only if it wins on a time-held-out OOF slice |
| 3 | Early stopping | validation = the *next* time block, never a shuffled fold |
| 4 | Adjacent labels cannot straddle a split | purge = `H` bars + embargo |
| 5 | Feature selection | no target-based selection; constant columns dropped on train stats only |

Audits that run on every execution:

1. **Causality / time-shift audit** — features are recomputed on a dataset truncated at 70 %
   of the bars; every overlapping row must be *identical* (max abs diff < 1e-8). Catches any
   feature that peeks forward or uses whole-sample statistics.
2. **Label alignment audit** — labels are recomputed from raw closes for random rows and
   compared to the vectorised implementation. Catches off-by-one shifts, the classic
   "predict yesterday's move with today's feature" bug.
3. **Label-shuffle test** — labels are permuted in time; the purged-CV AUC must collapse to
   ≈ 0.50. If it does not, the folds are not separated and the model is memorising overlap.
4. **Reported contrasts** — `train_auc` vs `cv_auc` vs `test_auc`, and `spearman(prob, true
   trend score)` on the test set (a model that cannot rank *realised* trendiness is not
   detecting regimes, whatever its accuracy).

If a run reports a test AUC above ~0.75 on 4h crypto, assume leakage, not genius: check that
you kept the purged folds, kept the untouched holdout, and did not feed in centred windows.

---

## Overfitting controls

* purged, expanding-window **walk-forward CV** (4 folds) instead of random K-fold;
* early stopping (log loss) with the fold's *next* block as validation;
* shallow trees (depth ≤ 6), `min_child_weight` ≥ 5, row/column subsampling, L1 + L2;
* small random hyper-parameter search scored by **CV log loss (+ 0.1 × fold std)** — the
  penalty for unstable folds picks simpler models;
* `n_estimators` for the final model = 1.1 × mean best iteration, then refit on all
  pre-test data;
* per-fold report of `train_log_loss`, `cv_log_loss`, `test_log_loss`, the gaps between them,
  and the fold-to-fold AUC std, so you can *see* the generalisation gap rather than trust one
  number.

---

## Validation protocol

```
├──────────── development (80 % of time) ────────────┤├─ TEST (20 %) ─┤
 ├─ fold 0 train ─┤ purge
        ├─ fold 1 train ─┤ purge
               ├─ fold 2 train ─┤ purge
                      ├─ fold 3 train ─┤            ← thresholds, clamping,
                                                    ← calibration, HPO all fit here
                                                    ← TEST touched exactly once
```

Metrics: accuracy (against the base rate!), balanced accuracy, ROC AUC, **log loss** and
**Brier** (proper scoring rules — the numbers that matter for a probability model), MCC,
precision/recall, log-loss skill vs the base rate, plus moving-block-bootstrap 95 % CIs
(blocks of 2H bars, because overlapping labels are autocorrelated) and a per-symbol,
per-fold breakdown.

---

## Measured results and what they mean

Two validations were actually run for this repo (Binance data, 4h bars, real prices):

**A. BTCUSDT 2020–2021 + 2024–2025** (fragmented: a 2-year hole, ~6.6k usable rows)
→ `results/validation_btc/`

| horizon | test accuracy | base rate | AUC [95 % CI] | log-loss skill | verdict |
|---|---|---|---|---|---|
| 4h | 60.5 % | 49.9 % | **0.659 [0.627, 0.681]** | +4.0 % | usable signal |
| 8h | 50.5 % | 50.4 % | 0.517 [0.490, 0.550] | 0.0 % | no edge |
| 12h | 52.2 % | 52.4 % | 0.508 [0.480, 0.550] | −0.6 % | no edge |
| 16h | 50.9 % | 50.5 % | 0.533 [0.501, 0.575] | −0.6 % | marginal |
| 24h | 52.2 % | 49.5 % | 0.559 [0.524, 0.594] | +0.2 % | marginal |

**B. BTC + ETH + SOL, Mar 2024 – Apr 2025** (14 months, ~7.4k rows)
→ `results/validation_multisymbol/`

| horizon | test accuracy | base rate | AUC [95 % CI] | verdict |
|---|---|---|---|---|
| 4h | 59.7 % | 49.9 % | **0.638 [0.604, 0.671]** | usable signal |
| 8h | 48.0 % | 52.0 % | 0.478 [0.445, 0.508] | no edge |
| 12h | 49.9 % | 52.1 % | 0.496 [0.463, 0.533] | no edge |
| 16h | 49.6 % | 49.8 % | 0.482 [0.446, 0.538] | no edge |
| 24h | 50.1 % | 49.1 % | 0.499 [0.457, 0.562] | no edge |

**Synthetic 3-regime market** (drift/vol/no-drift states, 40k bars × 2 symbols) — a positive
control: the same pipeline reaches **AUC 0.60–0.64 with 55–60 % accuracy** at 4h/12h/24h, and
the shuffled-label test sits at 0.50. So the machinery *can* learn multi-bar regimes when they
exist; real BTC over 2024–2025 simply did not offer much at 8–24h.

Interpretation, in order of importance:

1. **Short-horizon regime prediction works; long-horizon is much harder.** 4h (one bar) is a
   volatility/participation problem the model solves well. Over 8–24h, price paths are close
   to a random walk in trend-shape terms — the efficiency ratio of the next 2–6 bars is
   dominated by noise.
2. **Always compare with the base rate.** By construction ~50 % of windows are "trending", so
   a 50 % accuracy is worthless. That is why every table shows `base_rate`, `logloss_skill`,
   `lift_over_majority`, `mcc` and CIs.
3. **A 0.70 probability is a statement about frequency, not a certainty.** The decision table
   shows that calls ≥ 0.70 happened on 7 % of bars and were right 69 % of the time — that is
   the honest meaning of "70 %".
4. **CV is optimistic relative to a regime-shifted test block.** In validation B the purged-CV
   AUCs were 0.638 / 0.518 / 0.533 / 0.528 / 0.551 (4h→24h) but the test AUCs were 0.638 /
   0.478 / 0.496 / 0.482 / 0.499. The last 14 months were a harder regime than 2024 for the
   long horizons. Trust the **test** column, and re-check the CV-vs-test column after every
   change — a widening gap is the first sign of a fragile model.
5. **Data volume is the binding constraint.** The models that look useless on 7k rows get
   ~5× more rows — and every 2019/2020/2021 regime — in the Colab default
   (`2019 → today × 3–8 symbols`). The 4h horizon should hold up at AUC ≈ 0.63–0.67; treat
   8–24h as "measure it on your own data before you believe it".
6. **Fold-to-fold spread matters more than the mean.** A horizon whose fold AUCs swing ±0.04
   is not a horizon you should trade.

---

## Tuning playbook (in order of expected payoff)

1. **More symbols** — `symbols=(...8–15 liquid USDT pairs...)`. Pooled rows are the cheapest
   accuracy win; the model already gets a per-symbol one-hot and cross-asset features.
2. **More history** — `start="2019-01-01"` (or `2017` for BTC). Include a chop year; a model
   trained only on a bull market calls everything trending.
3. **More HPO** — `hpo_iter=20..30`. Cheap, and it targets log loss (the right metric).
4. **Longer/shorter label definition** — `label_quantile` 0.6–0.7 if you only want the strong
   regimes; `min_abs_fwd_return_quantile=0.5` to drop dead windows; `label_method="r2"` to try
   the alternative target.
5. **Add a feature family** — funding rates, open interest, basis, spot-perp spread, and
   on-chain/ETF flows are the best-known complementary regime signals and plug in as extra
   columns in `build_features`.
6. **Different model** — the harness is model-agnostic: swap `xgb.XGBClassifier` for
   LightGBM/CatBoost, or average the three (the fold loop and reporting stay untouched).

Do **not** try to raise the numbers by shrinking the test set, switching to random K-fold, or
using the test period for early stopping. That is how the "70 % accurate regime model"
screenshots you see online are made.

---

## Limitations

* The label threshold is fitted on history — when market character shifts, "trending" is
  redefined relative to the past. Use `label_absolute_threshold` for a fixed definition.
* Overlapping labels mean the effective sample size is much smaller than `n_test`; CIs are
  bootstrap-based for that reason.
* This is a **research/education** tool, not investment advice. It estimates a probability of
  a *market-structure* label; it says nothing about trade sizing, costs, slippage or funding.
* Synthetic-data results prove the pipeline works, not that crypto is predictable.

---

## Reproducing everything here

```bash
python smoke_test.py            # offline, ~2 min: notebook cells + audits + synthetic run
python build_notebook.py --check # notebook in sync with the .py source
python crypto_regime_xgboost.py  # full run with the default CONFIG (needs internet)
```
