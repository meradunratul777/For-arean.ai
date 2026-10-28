# %% [markdown]
# ## 0. Setup

# %%
import io
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
import warnings
import zipfile
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 100)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb

try:
    import matplotlib.pyplot as plt

    HAS_MPL = True
except Exception:  # pragma: no cover
    HAS_MPL = False

RNG_SEED = 42
np.random.seed(RNG_SEED)

HOW_TO_READ_RESULTS = """
How to read these numbers honestly
----------------------------------
* ~50% of windows are 'trending' by construction -> always compare accuracy with base_rate.
  A 60% accuracy at a 50% base rate is real skill; 60% at a 55% base rate mostly is not.
* ROC AUC of 0.55-0.65 is a realistic range for this problem on 4h crypto bars. A test AUC
  above ~0.75 on financial time series is almost always leakage: random K-fold instead of
  time-ordered folds, centred (forward+backward) windows, labels normalised on the full
  sample, or features computed after the fact. The audits above exist to catch exactly that.
* log loss / Brier score are the numbers that matter for a probability model. Accuracy hides
  overconfidence; log loss does not. 'logloss_skill_vs_base' > 0 means you beat always
  predicting the base rate.
* the decision table answers "the model says 70% - so what?": it shows how often such a call
  was right (precision) and how often it happens (coverage).
* single years can dominate: a bull year makes 'trending' easy, a chop year makes it hard.
  Look at the per-fold table for regime-dependent swings before trusting the average.
* overlapping labels mean the effective sample size is smaller than 'n_test'. The block
  bootstrap confidence intervals are the honest error bars.
"""

# %% [markdown]
# ## 1. Configuration
#
# Everything a user normally wants to change lives here.
#
# **`label_method`** decides what "trending" means:
# * `efficiency` — Kaufman efficiency ratio of the *forward* path:
#   `ER = |log(P[t+H]/P[t])| / sum(|log(P[i]/P[i-1])|)`.
#   1.0 = perfectly straight move, 0.0 = pure chop. Scale-free, vol-normalised.
# * `r2` — R^2 of a straight line fitted to the forward log-price path. Also scale-free.
#
# The binarising threshold is a **quantile of the training region only**, so the
# test period never influences the label definition (no leakage, no lookahead).

# %%
@dataclass
class Config:
    # ---- data -------------------------------------------------------------
    symbols: Sequence[str] = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    market_symbol: str = "BTCUSDT"          # reference asset for cross-asset features
    interval: str = "4h"                    # base timeframe (bars are 4 hours)
    start: str = "2019-01-01"
    end: Optional[str] = None               # None -> up to the latest closed bar
    data_source: str = "binance_vision"     # "binance_vision" | "binance_rest" | "csv" | "synthetic"
    data_dir: str = "data/cache"            # download cache
    vision_base: str = "https://data.binance.vision/data/spot"  # kline archive root
    csv_dir: str = "data/raw"               # used when data_source="csv"
    csv_pattern: str = "{symbol}-4h.csv"    # used when data_source="csv"
    pool_symbols: bool = True               # True: one model on all symbols; False: per symbol
    drop_gap_windows: bool = True           # drop rows whose lookback window contains a data gap

    # ---- label ------------------------------------------------------------
    horizons_hours: Sequence[int] = (4, 8, 12, 16, 24)
    label_method: str = "efficiency"        # "efficiency" | "r2"
    label_quantile: float = 0.5             # 0.5 -> balanced classes (50% trending by construction)
    label_absolute_threshold: Optional[float] = None  # e.g. 0.35 to fix ER cut-off in advance
    min_abs_fwd_return_quantile: Optional[float] = None  # e.g. 0.9: ignore "dead" windows

    # ---- validation -------------------------------------------------------
    test_frac: float = 0.20                 # last 20% of time is never touched until the end
    n_cv_splits: int = 4                    # purged, expanding-window walk-forward folds
    min_train_frac: float = 0.40            # first fold trains on 40% of the dev period

    # ---- model ------------------------------------------------------------
    params: Dict = field(default_factory=lambda: dict(
        n_estimators=3000,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=20,
        subsample=0.8,
        colsample_bytree=0.6,
        colsample_bylevel=0.8,
        reg_lambda=2.0,
        reg_alpha=0.5,
        gamma=0.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        max_bin=256,
        n_jobs=-1,
        random_state=RNG_SEED,
    ))
    early_stopping_rounds: int = 100
    run_hpo: bool = True
    hpo_iter: int = 8
    hpo_horizon_hours: int = 12             # tune once here, reuse for every horizon
    calibration: str = "temperature"        # "temperature" | "isotonic" | "none"
    winsorize_quantiles: Tuple[float, float] = (0.001, 0.999)

    # ---- reporting --------------------------------------------------------
    out_dir: str = "results"
    run_permutation_test: bool = True
    bootstrap_reps: int = 400
    save_figures: bool = True
    verbose: bool = True

    @property
    def bar_hours(self) -> int:
        return {"1h": 1, "2h": 2, "4h": 4, "6h": 6, "8h": 8, "12h": 12, "1d": 24}[self.interval]


CONFIG = Config()

# Timeframe helpers -------------------------------------------------------
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "n_trades", "taker_buy_base", "taker_buy_quote", "ignore",
]


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


# %% [markdown]
# ## 2. Data engineering
#
# Order of operations, and why:
#
# | step | why it matters |
# |---|---|
# | download monthly zips from `data.binance.vision` | free, complete, versioned klines; no API rate limits |
# | de-duplicate on `open_time`, sort | merged monthly files overlap |
# | reindex onto the exact 4h grid | makes gap detection + time maths exact |
# | flag bars in/after a gap; drop rows whose **lookback window** crosses a gap | a feature computed across exchange downtime is fiction |
# | sanity checks (high>=low, positive prices) | bad ticks poison rolling vols |

# %%
def _http_get(url: str, retries: int = 4, timeout: int = 60) -> Optional[bytes]:
    """GET with retries. Also accepts file:// URLs, which makes local archive mirrors work."""
    if url.startswith("file://"):
        try:
            with urllib.request.urlopen(url) as r:
                return r.read()
        except Exception:
            return None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(1.5 * (attempt + 1))
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return None


def _parse_kline_bytes(raw: bytes) -> pd.DataFrame:
    """Parse a Binance klines CSV (some monthly files carry a header row, daily ones don't)."""
    if raw[:2] == b"PK":  # zip
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            name = zf.namelist()[0]
            raw = zf.read(name)
    head = raw[:200].decode("utf-8", errors="ignore")
    has_header = ("open_time" in head) or ("Open time" in head)
    df = pd.read_csv(io.BytesIO(raw), header=0 if has_header else None)
    if df.shape[1] < len(KLINE_COLS):
        df = df.iloc[:, : df.shape[1]]
    df = df.iloc[:, : len(KLINE_COLS)]
    df.columns = KLINE_COLS[: df.shape[1]]
    return df


def fetch_binance_vision(
    symbol: str,
    interval: str,
    start: str,
    end: Optional[str],
    cache_dir: str,
    verbose: bool = True,
    base: str = "https://data.binance.vision/data/spot",
) -> pd.DataFrame:
    """Download monthly (and daily for the current month) klines from data.binance.vision."""
    os.makedirs(cache_dir, exist_ok=True)
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") if end else pd.Timestamp.now(tz="UTC")
    end_ts = end_ts.floor("D") + pd.Timedelta(days=1)

    months = pd.period_range(start_ts.tz_localize(None).to_period("M"),
                             end_ts.tz_localize(None).to_period("M"), freq="M")
    frames: List[pd.DataFrame] = []
    now = pd.Timestamp.now(tz="UTC")
    for per in months:
        month_start = per.start_time.tz_localize("UTC")
        is_current = (month_start.year == now.year) and (month_start.month == now.month)
        if is_current:
            # current month has no monthly zip yet -> daily files
            day = month_start
            while day <= min(end_ts, now):
                f = os.path.join(cache_dir, f"{symbol}-{interval}-{day:%Y-%m-%d}.csv")
                if not os.path.exists(f):
                    url = (f"{base}/daily/klines/{symbol}/"
                           f"{interval}/{symbol}-{interval}-{day:%Y-%m-%d}.zip")
                    raw = _http_get(url)
                    if raw is None:
                        day += pd.Timedelta(days=1)
                        continue
                    df = _parse_kline_bytes(raw)
                    df.to_csv(f, index=False)
                else:
                    df = pd.read_csv(f)
                frames.append(df)
                day += pd.Timedelta(days=1)
            continue
        f = os.path.join(cache_dir, f"{symbol}-{interval}-{per}.csv")
        if os.path.exists(f):
            frames.append(pd.read_csv(f))
            continue
        url = (f"{base}/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{per}.zip")
        raw = _http_get(url)
        if raw is None:
            if verbose:
                log(f"  ! missing {symbol} {per}")
            continue
        df = _parse_kline_bytes(raw)
        df.to_csv(f, index=False)
        frames.append(df)

    if not frames:
        raise RuntimeError(
            f"No klines downloaded for {symbol}. Check the symbol/ticker or your internet access."
        )
    out = pd.concat(frames, ignore_index=True)
    if verbose:
        log(f"  {symbol}: {len(out):,} raw bars from binance.vision")
    return out


def fetch_binance_rest(symbol: str, interval: str, start: str, end: Optional[str],
                       cache_dir: str, verbose: bool = True) -> pd.DataFrame:
    """Paginated fallback via the public REST endpoint (max 1000 bars per call)."""
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"{symbol}-{interval}-rest.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache)
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int((pd.Timestamp(end, tz="UTC") if end else pd.Timestamp.now(tz="UTC")).timestamp() * 1000)
    rows: List[list] = []
    cursor = start_ms
    while cursor < end_ms:
        url = (f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}"
               f"&startTime={cursor}&limit=1000")
        raw = _http_get(url)
        if raw is None:
            break
        batch = json.loads(raw)
        if not batch:
            break
        rows.extend(batch)
        cursor = int(batch[-1][0]) + 1
        time.sleep(0.25)
    if not rows:
        raise RuntimeError(f"REST download failed for {symbol}")
    df = pd.DataFrame([r[:12] for r in rows], columns=KLINE_COLS)
    df.to_csv(cache, index=False)
    if verbose:
        log(f"  {symbol}: {len(df):,} bars from REST")
    return df


def fetch_local_csv(symbol: str, csv_dir: str, pattern: str) -> pd.DataFrame:
    path = os.path.join(csv_dir, pattern.format(symbol=symbol))
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return _parse_kline_bytes(open(path, "rb").read())


def make_synthetic(symbol: str = "SYNTHUSDT", start: str = "2019-01-01",
                   periods: int = 40000, interval: str = "4h", seed: int = 7) -> pd.DataFrame:
    """Offline self-test data: 3-state regime-switching GBM (trend-up / trend-down / range).

    A correct pipeline must recover usable signal here; if it cannot, the bug is ours.
    """
    rng = np.random.default_rng(seed)
    # Markov chain over regimes
    P = np.array([[0.975, 0.020, 0.005],
                  [0.020, 0.975, 0.005],
                  [0.012, 0.012, 0.976]])
    states = np.zeros(periods, dtype=int)
    for i in range(1, periods):
        states[i] = rng.choice(3, p=P[states[i - 1]])
    drift = np.array([0.0055, -0.0055, 0.0])[states]
    vol = np.array([0.009, 0.009, 0.005])[states] * (1 + 0.35 * np.sin(np.arange(periods) / 900))
    # vol clustering
    eps = np.zeros(periods)
    z = rng.standard_normal(periods)
    for i in range(1, periods):
        eps[i] = 0.92 * eps[i - 1] + np.sqrt(1 - 0.92**2) * 0.55 * z[i]
    ret = drift + vol * (0.9 * z + eps)
    close = 100 * np.exp(np.cumsum(ret))
    o = np.concatenate([[100.0], close[:-1]])
    span = np.abs(ret) + 0.002
    high = np.maximum(o, close) * (1 + span * rng.uniform(0.1, 0.8, periods))
    low = np.minimum(o, close) * (1 - span * rng.uniform(0.1, 0.8, periods))
    volume = 1000 * np.exp(rng.normal(0, 0.5, periods) + 3 * np.abs(ret) / vol[0])
    ts = pd.date_range(start, periods=periods, freq=interval, tz="UTC")
    # epoch milliseconds, resolved safely across pandas versions (the internal unit varies)
    open_ms = (ts.tz_localize(None).astype("datetime64[ns]").astype("int64") // 10**6)
    return pd.DataFrame({
        "open_time": open_ms,
        "open": o, "high": high, "low": low, "close": close,
        "volume": volume,
        "close_time": open_ms + 4 * 3600 * 1000 - 1,
        "quote_volume": volume * close,
        "n_trades": np.maximum(10, (volume / 8).astype(int)),
        "taker_buy_base": volume * rng.uniform(0.4, 0.6, periods),
        "taker_buy_quote": volume * close * rng.uniform(0.4, 0.6, periods),
        "ignore": 0,
    })


def to_millis(s: pd.Series) -> pd.Series:
    """Normalise an epoch column to milliseconds, row by row.

    Binance switched kline timestamps from milliseconds to **microseconds** in 2025, and
    older exports sometimes use seconds. A file that concatenates both silently spans tens
    of thousands of years if a single unit is assumed. Seconds / ms / us / ns are three
    orders of magnitude apart, so each value is snapped to whichever scale puts it closest
    to the column's median instant.
    """
    v = pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64")
    finite = v[np.isfinite(v) & (v > 0)]
    if finite.size == 0:
        return pd.Series(v, index=s.index)
    med = float(np.median(np.abs(finite)))
    scales = np.array([1e-6, 1e-3, 1.0, 1e3])          # divisor that produces milliseconds
    cand = np.abs(v)[:, None] / scales[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        dist = np.abs(np.log10(np.where(cand > 0, cand, np.nan)) - math.log10(med))
    best = np.nanargmin(np.nan_to_num(dist, nan=np.inf), axis=1)
    out = v / scales[best]
    out[~np.isfinite(v)] = np.nan
    return pd.Series(np.round(out), index=s.index)


def clean_ohlcv(df: pd.DataFrame, interval: str, verbose: bool = True) -> pd.DataFrame:
    """Normalise raw klines onto an exact, gap-flagged time grid."""
    d = df.copy()
    d = d[KLINE_COLS].apply(pd.to_numeric, errors="coerce")
    d = d.dropna(subset=["open_time", "open", "high", "low", "close"])
    d["open_time"] = to_millis(d["open_time"])
    d = d.drop_duplicates(subset="open_time").sort_values("open_time")

    d["time"] = pd.to_datetime(d["open_time"], unit="ms", utc=True)
    d = d.set_index("time")

    # sanity: positive prices, high >= low, high/low bound the body
    bad = ((d[["open", "high", "low", "close"]] <= 0).any(axis=1)) | (d["high"] < d["low"])
    if bad.any():
        if verbose:
            log(f"  dropped {int(bad.sum())} broken bars (non-positive price / high<low)")
        d = d[~bad]
    d["high"] = d[["open", "high", "low", "close"]].max(axis=1)
    d["low"] = d[["open", "high", "low", "close"]].min(axis=1)

    # exact resample grid -> detects exchange downtime
    grid = pd.date_range(d.index.min(), d.index.max(), freq=interval, tz="UTC")
    if len(grid) > 5 * len(d) + 1000:
        raise ValueError(
            f"Suspicious time grid: {len(grid):,} bars for {len(d):,} rows "
            f"({d.index.min()} -> {d.index.max()}). Check the timestamp unit / interval.")
    n_missing = len(grid) - len(d)
    d = d.reindex(grid)
    d.index.name = "time"
    d["gap_bar"] = d["close"].isna().astype(int)
    # a bar right after a gap has a distorted return -> flag it too
    d["gap_bar"] = d["gap_bar"].rolling(1).max().fillna(0)
    d["after_gap"] = d["gap_bar"].shift(1, fill_value=0)
    d["n_missing_bars_total"] = n_missing
    if verbose:
        log(f"  cleaned: {len(d):,} bars | missing bars filled: {n_missing} "
            f"({100 * n_missing / max(len(grid), 1):.2f}%)")

    # forward-fill price columns ONLY across single missing bars so that rolling
    # stats stay defined; the row is excluded later via the gap mask anyway.
    d[["open", "high", "low", "close"]] = d[["open", "high", "low", "close"]].ffill(limit=2)
    for c in ["volume", "quote_volume", "n_trades", "taker_buy_base", "taker_buy_quote"]:
        d[c] = d[c].fillna(0.0)
    return d


def load_symbol(symbol: str, cfg: Config) -> pd.DataFrame:
    if cfg.data_source == "synthetic":
        raw = make_synthetic(symbol, start=cfg.start, interval=cfg.interval,
                             periods=40000, seed=hash(symbol) % 10000)
    elif cfg.data_source == "csv":
        raw = fetch_local_csv(symbol, cfg.csv_dir, cfg.csv_pattern)
        start_ms = pd.Timestamp(cfg.start, tz="UTC").value // 10**6
        raw = raw[to_millis(raw["open_time"]) >= start_ms]
    elif cfg.data_source == "binance_rest":
        raw = fetch_binance_rest(symbol, cfg.interval, cfg.start, cfg.end, cfg.data_dir)
    else:
        raw = fetch_binance_vision(symbol, cfg.interval, cfg.start, cfg.end, cfg.data_dir,
                                   verbose=cfg.verbose, base=cfg.vision_base)
    return clean_ohlcv(raw, cfg.interval, verbose=cfg.verbose)


# %% [markdown]
# ## 3. Feature engineering (strictly causal)
#
# Every feature at bar `t` is a function of bars `<= t` only. Labels use bars `t+1 … t+H`,
# so feature and label windows are disjoint. Feature families:
#
# * **Momentum/returns** — vol-normalised so BTC 2021 and SOL 2024 are comparable.
# * **Volatility** — realised, Parkinson, Garman-Klass, ATR, vol-of-vol, vol ratios (regime proxies).
# * **Trend-strength** — ADX/DI, Aroon, MACD, EMA slopes, past efficiency ratio, past R^2 & t-stat of a
#   linear fit, Hurst exponent, variance ratio, return autocorrelation, sign persistence, direction changes.
#   *Past* trendiness is the single strongest predictor of *future* trendiness (regimes cluster).
# * **Mean-reversion / location** — z-scores, Bollinger %B/width, RSI, Stochastic, Donchian position.
# * **Volume / flow** — volume z-score, taker-buy imbalance, OBV slope, Amihud illiquidity, VWAP distance.
# * **Cross-asset** — rolling correlation & beta to BTC, relative momentum, market-wide vol/trend state.
# * **Time-of-day / day-of-week** — cyclical encodings (crypto runs 24/7 but liquidity has a daily cycle).

# %%
def _slide(x: np.ndarray, w: int) -> np.ndarray:
    return np.lib.stride_tricks.sliding_window_view(np.asarray(x, dtype=float), w)


def _pad(arr: np.ndarray, n_total: int) -> np.ndarray:
    """Left-pad a sliding-window result back to the original length with NaN."""
    out = np.full(n_total, np.nan)
    out[n_total - len(arr):] = arr
    return out


def rolling_corr(a: pd.Series, b: pd.Series, w: int) -> pd.Series:
    a = a.astype(float)
    b = b.astype(float)
    ma, mb = a.rolling(w).mean(), b.rolling(w).mean()
    cov = (a * b).rolling(w).mean() - ma * mb
    sa = a.rolling(w).std(ddof=0)
    sb = b.rolling(w).std(ddof=0)
    return (cov / (sa * sb).replace(0, np.nan)).clip(-1, 1)


def wilder(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def adx_di(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14):
    up = high.diff()
    dn = -low.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=high.index)
    pc = close.shift(1)
    tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    atr = wilder(tr, n)
    plus_di = 100 * wilder(plus_dm, n) / atr.replace(0, np.nan)
    minus_di = 100 * wilder(minus_dm, n) / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return wilder(dx, n), plus_di, minus_di, atr


def rolling_linreg(y: np.ndarray, w: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised OLS of y on t inside every window of length w -> (slope, R^2, t-stat)."""
    n = len(y)
    if n < w:
        return (np.full(n, np.nan),) * 3
    W = _slide(y, w)
    t = np.arange(w, dtype=float)
    tc = t - t.mean()
    Stt = float((tc ** 2).sum())
    Yc = W - np.nanmean(W, axis=1, keepdims=True)
    slope = (Yc @ tc) / Stt
    sst = (Yc ** 2).sum(axis=1)
    fitted = (slope ** 2) * Stt
    r2 = np.clip(fitted / np.where(sst > 0, sst, np.nan), 0.0, 1.0)
    resid = Yc - slope[:, None] * tc
    s2 = (resid ** 2).sum(axis=1) / max(w - 2, 1)
    se = np.sqrt(s2 / Stt)
    tstat = np.clip(slope / np.where(se > 0, se, np.nan), -60, 60)
    return _pad(slope, n), _pad(r2, n), _pad(tstat, n)


def hurst_rs(rets: np.ndarray, w: int = 64) -> np.ndarray:
    """Rolling Hurst exponent by rescaled range. >0.5 persistent/trending, <0.5 mean-reverting."""
    n = len(rets)
    if n < w:
        return np.full(n, np.nan)
    W = _slide(rets, w)
    dev = W - np.nanmean(W, axis=1, keepdims=True)
    cum = np.cumsum(dev, axis=1)
    R = np.nanmax(cum, axis=1) - np.nanmin(cum, axis=1)
    S = np.nanstd(W, axis=1, ddof=1)
    h = np.log(np.where(R > 0, R, np.nan) / np.where(S > 0, S, np.nan)) / math.log(w)
    return _pad(np.clip(h, -0.5, 1.5), n)


def variance_ratio(rets: np.ndarray, w: int = 48, k: int = 4) -> np.ndarray:
    """Lo-MacKinlay VR inside each window: >1 trending, <1 mean-reverting."""
    n = len(rets)
    if n < w:
        return np.full(n, np.nan)
    W = _slide(rets, w)
    nseg = w // k
    seg = W[:, : nseg * k].reshape(len(W), nseg, k).sum(axis=2)
    var_k = np.nanvar(seg, axis=1, ddof=1)
    var_1 = np.nanvar(W, axis=1, ddof=1)
    vr = var_k / np.where(var_1 > 0, k * var_1, np.nan)
    return _pad(np.log(np.clip(vr, 1e-3, 1e3)), n)


def aroon(high: pd.Series, low: pd.Series, n: int = 24) -> Tuple[pd.Series, pd.Series]:
    w = n + 1
    if len(high) < w:
        return pd.Series(np.nan, index=high.index), pd.Series(np.nan, index=high.index)
    Hi = _slide(high.to_numpy(), w)
    Lo = _slide(low.to_numpy(), w)
    since_hi = (w - 1) - np.argmax(Hi, axis=1)
    since_lo = (w - 1) - np.argmin(Lo, axis=1)
    up = _pad(100.0 * (n - since_hi) / n, len(high))
    dn = _pad(100.0 * (n - since_lo) / n, len(high))
    return pd.Series(up, index=high.index), pd.Series(dn, index=low.index)


def build_features(d: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Causal feature matrix for one symbol. Index = bar open time."""
    o, h, l, c = d["open"], d["high"], d["low"], d["close"]
    v, qv, nt = d["volume"], d["quote_volume"], d["n_trades"]
    tb = d["taker_buy_base"]
    lc = np.log(c.replace(0, np.nan))
    ret = lc.diff()
    absret = ret.abs()
    f = pd.DataFrame(index=d.index)

    atr14 = adx_di(h, l, c, 14)[3]
    rv = lambda k: ret.rolling(k).std(ddof=0)
    rv6, rv12, rv24, rv48 = rv(6), rv(12), rv(24), rv(48)
    rv96 = rv(96)
    vol_norm = rv48.replace(0, np.nan)

    # ---------------- momentum (vol-normalised) ----------------
    for k in (1, 2, 3, 6, 12, 24, 48, 96):
        f[f"mom{k}_z"] = lc.diff(k) / (vol_norm * math.sqrt(k))

    # ---------------- volatility ----------------
    for nm, s in (("rv6", rv6), ("rv12", rv12), ("rv24", rv24), ("rv48", rv48), ("rv96", rv96)):
        f[nm] = np.log(s.clip(lower=1e-8))
    f["vol_ratio_6_48"] = np.log((rv6 / vol_norm).clip(lower=1e-6))
    f["vol_ratio_12_48"] = np.log((rv12 / vol_norm).clip(lower=1e-6))
    f["vol_ratio_48_96"] = np.log((rv48 / rv96.replace(0, np.nan)).clip(lower=1e-6))
    f["rv_42"] = np.log(ret.rolling(42).std(ddof=0).clip(lower=1e-8))   # ~1 week
    f["vol_z_96"] = (rv24 - rv24.rolling(96).mean()) / rv24.rolling(96).std(ddof=0).replace(0, np.nan)
    f["vol_of_vol_48"] = rv12.rolling(48).std(ddof=0) / vol_norm
    hl = np.log((h / l).replace(0, np.nan))
    f["parkinson_24"] = np.log(np.sqrt((hl ** 2).rolling(24).mean() / (4 * math.log(2))).clip(lower=1e-8))
    rs = (np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)).clip(lower=0)
    f["rogers_satchell_24"] = np.log(np.sqrt(rs.rolling(24).mean()).clip(lower=1e-8))
    f["atr_pct_14"] = (atr14 / c).replace(0, np.nan)
    up_vol = ret.clip(lower=0).rolling(24).std(ddof=0)
    dn_vol = (-ret).clip(lower=0).rolling(24).std(ddof=0)
    f["down_up_vol_ratio_24"] = np.log((dn_vol / up_vol.replace(0, np.nan)).clip(lower=1e-6))

    # ---------------- trend strength ----------------
    adx14, plus_di, minus_di, _ = adx_di(h, l, c, 14)
    f["adx_14"] = adx14 / 100.0
    f["di_diff_14"] = (plus_di - minus_di) / 100.0
    f["adx_slope_6"] = adx14.diff(6) / 100.0
    aroon_up, aroon_dn = aroon(h, l, 24)
    f["aroon_osc_24"] = (aroon_up - aroon_dn) / 100.0
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    f["macd_hist_atr"] = (macd - macd_sig) / atr14.replace(0, np.nan)
    ema24 = c.ewm(span=24, adjust=False).mean()
    f["ema_slope_24_z"] = np.log(ema24 / ema24.shift(24)) / (vol_norm * math.sqrt(24))
    # past trend quality (the killer feature: regimes cluster in time)
    # efficiency ratio at several scales; 2/3/4 bars line up with the 8h/12h/16h labels,
    # so the model can read "trendiness at the horizon it is asked about"
    for k in (2, 3, 4, 6, 12, 24, 48):
        num = ret.rolling(k).sum().abs()
        den = absret.rolling(k).sum().replace(0, np.nan)
        f[f"er_{k}"] = num / den
    _, r2_12, tstat_12 = rolling_linreg(lc.to_numpy(), 12)
    slope24, r2_24, tstat_24 = rolling_linreg(lc.to_numpy(), 24)
    _, r2_48, tstat_48 = rolling_linreg(lc.to_numpy(), 48)
    f["trend_r2_12"] = r2_12
    f["trend_r2_24"] = r2_24
    f["trend_r2_48"] = r2_48
    f["slope_tstat_12"] = tstat_12
    f["slope_tstat_24"] = tstat_24
    f["slope_tstat_48"] = tstat_48
    f["hurst_64"] = hurst_rs(ret.to_numpy(), 64)
    f["vr_log_4_48"] = variance_ratio(ret.to_numpy(), 48, 4)
    sign = np.sign(ret.fillna(0.0))
    f["sign_persist_24"] = sign.rolling(24).sum().abs() / 24
    f["turns_24"] = (sign != sign.shift(1)).astype(float).rolling(24).mean()
    f["ac1_48"] = rolling_corr(ret, ret.shift(1), 48)
    f["ac_sum5_48"] = sum(rolling_corr(ret, ret.shift(k), 48).fillna(0) for k in range(1, 6))

    # ---------------- mean reversion / location ----------------
    sma24, sma96 = c.rolling(24).mean(), c.rolling(96).mean()
    f["zscore_24"] = (c - sma24) / c.rolling(24).std(ddof=0).replace(0, np.nan)
    f["zscore_96"] = (c - sma96) / c.rolling(96).std(ddof=0).replace(0, np.nan)
    bb_mid = c.rolling(20).mean()
    bb_sd = c.rolling(20).std(ddof=0)
    f["bb_pctb_20"] = ((c - (bb_mid - 2 * bb_sd)) / (4 * bb_sd).replace(0, np.nan)).clip(-1, 2)
    f["bb_width_20_atr"] = (4 * bb_sd / atr14.replace(0, np.nan)).clip(0, 20)
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs_ = gain / loss.replace(0, np.nan)
    f["rsi_14"] = (100 - 100 / (1 + rs_)) / 100.0
    ll14 = l.rolling(14).min()
    hh14 = h.rolling(14).max()
    f["stoch_14"] = ((c - ll14) / (hh14 - ll14).replace(0, np.nan)).clip(0, 1)
    hh48, ll48 = h.rolling(48).max(), l.rolling(48).min()
    f["donchian_pos_24"] = ((c - l.rolling(24).min()) / (h.rolling(24).max() - l.rolling(24).min()).replace(0, np.nan)).clip(0, 1)
    f["dist_high_48_atr"] = (hh48 - c) / atr14.replace(0, np.nan)
    f["dist_low_48_atr"] = (c - ll48) / atr14.replace(0, np.nan)

    # ---------------- volume / flow ----------------
    f["vol_z_24"] = (v - v.rolling(24).mean()) / v.rolling(24).std(ddof=0).replace(0, np.nan)
    f["vol_z_96"] = (v - v.rolling(96).mean()) / v.rolling(96).std(ddof=0).replace(0, np.nan)
    f["vol_ratio_6_48"] = np.log((v.rolling(6).mean() / v.rolling(48).mean().replace(0, np.nan)).clip(1e-6))
    f["taker_buy_ratio_24"] = (tb / v.replace(0, np.nan)).rolling(24).mean()
    obv = (sign * v).cumsum()
    f["obv_slope_24"] = (obv - obv.shift(24)) / v.rolling(24).sum().replace(0, np.nan)
    f["amihud_log_24"] = np.log(((absret / qv.replace(0, np.nan)).rolling(24).mean() * 1e9).clip(lower=1e-8))
    tp = (h + l + c) / 3
    vwap24 = (tp * v).rolling(24).sum() / v.rolling(24).sum().replace(0, np.nan)
    f["vwap_dist_24_atr"] = (c - vwap24) / atr14.replace(0, np.nan)
    f["trades_z_96"] = (nt - nt.rolling(96).mean()) / nt.rolling(96).std(ddof=0).replace(0, np.nan)

    # ---------------- time of day / week ----------------
    hour = d.index.hour + d.index.minute / 60.0
    dow = d.index.dayofweek
    f["hour_sin"] = np.sin(2 * math.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * math.pi * hour / 24)
    f["dow_sin"] = np.sin(2 * math.pi * dow / 7)
    f["dow_cos"] = np.cos(2 * math.pi * dow / 7)
    f["is_weekend"] = (dow >= 5).astype(float)

    # ---------------- bookkeeping ----------------
    f["_ret1"] = ret
    f["_gap_bar"] = d["gap_bar"].to_numpy()
    f["_after_gap"] = d["after_gap"].to_numpy()
    f["_logc"] = lc
    return f


MAX_LOOKBACK = 96  # longest rolling window used above (bars)


def add_cross_asset_features(feat: Dict[str, pd.DataFrame], cfg: Config) -> Dict[str, pd.DataFrame]:
    """Market context from the reference symbol + relative behaviour of each altcoin."""
    mkt = feat[cfg.market_symbol]
    mkt_block = pd.DataFrame({
        "mkt_rv24": mkt["rv24"],
        "mkt_er_24": mkt["er_24"],
        "mkt_adx_14": mkt["adx_14"],
        "mkt_mom24_z": mkt["mom24_z"],
        "mkt_vol_z_24": mkt["vol_z_24"],
        "mkt_hurst_64": mkt["hurst_64"],
        "mkt_downtrend": (mkt["di_diff_14"] < 0).astype(float),
    }, index=mkt.index)
    mkt_ret = mkt["_ret1"]

    # ---- market breadth: is the whole universe trending together? (causal, same-bar only) ----
    breadth = pd.DataFrame(index=mkt.index)
    if len(feat) > 1:
        dirs = pd.DataFrame({sym: np.sign(f["mom24_z"].fillna(0.0)) for sym, f in feat.items()})
        er12 = pd.DataFrame({sym: f["er_12"] for sym, f in feat.items()})
        di = pd.DataFrame({sym: f["di_diff_14"] for sym, f in feat.items()})
        breadth["xs_breadth_up"] = (dirs > 0).mean(axis=1)
        breadth["xs_breadth_trend_down"] = (di < 0).mean(axis=1)
        breadth["xs_mean_er12"] = er12.mean(axis=1)
        breadth["xs_er12_dispersion"] = er12.std(axis=1)
    else:
        for c in ("xs_breadth_up", "xs_breadth_trend_down", "xs_mean_er12",
                  "xs_er12_dispersion"):
            breadth[c] = np.nan

    out: Dict[str, pd.DataFrame] = {}
    for sym, f in feat.items():
        g = f.copy()
        # align market state onto this symbol's timestamps (reindex is causal: it only
        # takes the market value stamped at the same bar, never a future one)
        blk = pd.concat([mkt_block, breadth], axis=1).reindex(g.index)
        for col in blk.columns:
            g[col] = blk[col].to_numpy()
        if sym == cfg.market_symbol:
            g["corr_btc_96"] = 1.0
            g["beta_btc_96"] = 1.0
            g["rel_mom24_btc_z"] = 0.0
        else:
            r = g["_ret1"]
            mr = mkt_ret.reindex(g.index)
            g["corr_btc_96"] = rolling_corr(r, mr, 96).to_numpy()
            cov = (r * mr).rolling(96).mean() - r.rolling(96).mean() * mr.rolling(96).mean()
            var_m = mr.rolling(96).var(ddof=0).replace(0, np.nan)
            g["beta_btc_96"] = (cov / var_m).clip(-5, 5).to_numpy()
            g["rel_mom24_btc_z"] = ((g["_logc"].diff(24) - np.log(np.exp(mr)).rolling(24).sum())
                                    / (g["rv48"] * math.sqrt(24) + 1e-9)).clip(-10, 10).to_numpy()
        out[sym] = g
    return out


# %% [markdown]
# ## 4. Labelling
#
# For horizon `H` bars (H hours / 4), we look at the forward path `P[t] … P[t+H]`:
#
# ```
# trend_score(t) = |log P[t+H] - log P[t]| / Σ_{i=1..H} |log P[t+i] - log P[t+i-1]|     (efficiency ratio)
# ```
#
# * `trend_score ≈ 1` → price moved in a straight line → **trending**.
# * `trend_score ≈ 0` → price travelled a lot but ended where it started → **ranging**.
#
# The label is `trend_score > threshold`, where the threshold is a quantile of the
# **training region only** (computed later, in `make_xy`), per symbol. That keeps classes
# balanced and keeps the test period out of the definition.

# %%
def resolve_label_method(h_bars: int, cfg: "Config") -> str:
    """Pick the label definition for a horizon.

    With 1-bar horizons the efficiency ratio |net| / sum(|steps|) is *always* 1.0
    (numerator == denominator), so it cannot separate regimes. On a 4h timeframe that
    means the 4h horizon needs a different, still-causal definition: the size of the next
    move measured in trailing-volatility units ("move_z"). Every longer horizon uses the
    configured method unchanged.
    """
    if cfg.label_method != "efficiency":
        return cfg.label_method
    if h_bars == 1:
        return "move_z"        # ER is identically 1.0 -> use vol-normalised move size
    if h_bars == 2:
        return "r2"            # ER can only take {0, 0.5, 1} -> too coarse, use path R^2
    return "efficiency"


def forward_trend_score(logc: np.ndarray, h: int, method: str = "efficiency",
                        sigma: Optional[np.ndarray] = None) -> np.ndarray:
    """Trend score of the forward window [t, t+h]; aligned to t, NaN for the last h bars.

    method:
      * "efficiency"       — |net move| / path length. 1 = straight trend, 0 = pure chop.
      * "r2"               — R^2 of a line fitted to the forward log-price path.
      * "sign_persistence" — |sum of signs| / h.
      * "move_z"           — |net move| / sigma[t] (trailing volatility). Used for 1-bar
                             horizons where "efficiency" is degenerate.

    On a 4h timeframe the label definition is therefore chosen automatically per horizon:
    4h -> move_z, 8h -> r2, 12h/16h/24h -> efficiency ratio (see `resolve_label_method`).
    """
    n = len(logc)
    score = np.full(n, np.nan)
    if n < h + 2:
        return score
    W = _slide(logc, h + 1)                     # row i = logc[i : i+h+1]
    if method == "efficiency":
        net = np.abs(W[:, -1] - W[:, 0])
        path = np.abs(np.diff(W, axis=1)).sum(axis=1)
        s = net / np.where(path > 0, path, np.nan)
    elif method == "r2":
        t = np.arange(h + 1, dtype=float)
        tc = t - t.mean()
        Stt = (tc ** 2).sum()
        Yc = W - W.mean(axis=1, keepdims=True)
        slope = (Yc @ tc) / Stt
        sst = (Yc ** 2).sum(axis=1)
        s = np.clip((slope ** 2) * Stt / np.where(sst > 0, sst, np.nan), 0, 1)
    elif method == "sign_persistence":
        d = np.sign(np.diff(W, axis=1))
        s = np.abs(d.sum(axis=1)) / h
    elif method == "move_z":
        if sigma is None:
            raise ValueError("method='move_z' needs sigma (trailing volatility)")
        net = np.abs(W[:, -1] - W[:, 0])
        s = net / np.where(sigma[: len(net)] > 0, sigma[: len(net)], np.nan)
    else:
        raise ValueError(f"unknown label method {method}")
    score[: len(s)] = s
    return score


def forward_return(logc: np.ndarray, h: int) -> np.ndarray:
    n = len(logc)
    out = np.full(n, np.nan)
    out[: n - h] = logc[h:] - logc[: n - h]
    return out


# %% [markdown]
# ## 5. Leakage audit (run automatically, fails loudly)
#
# Three independent checks:
#
# 1. **Causality / time-shift test** — recompute features on a dataset truncated at 70% of
#    the bars. Rows that exist in both must be identical. Any feature that peeks forward
#    (or uses full-sample statistics) fails this.
# 2. **Label alignment test** — recompute the label for random rows straight from the raw
#    closes and compare; catches off-by-one shifts that silently leak the future.
# 3. **Label-shuffle test** — permute labels in time; the CV AUC must collapse to ~0.5.
#    If it does not, the split is not purged and the model is memorising overlap.

# %%
def audit_feature_causality(df: pd.DataFrame, cfg: Config, frac: float = 0.7) -> pd.DataFrame:
    """Rebuild features on truncated data; overlapping rows must match exactly."""
    feat_cols = [c for c in build_features(df, cfg).columns if not c.startswith("_")]
    full = build_features(df, cfg)[feat_cols]
    cut = int(len(df) * frac)
    trunc = build_features(df.iloc[:cut], cfg)[feat_cols]
    common = full.index.intersection(trunc.index)
    a, b = full.loc[common], trunc.loc[common]
    rows = []
    for col in feat_cols:
        x, y = a[col].to_numpy(dtype=float), b[col].to_numpy(dtype=float)
        both_nan = np.isnan(x) & np.isnan(y)
        diff = np.abs(x - y)
        max_diff = np.nanmax(np.where(both_nan, 0.0, diff)) if (~both_nan).any() else 0.0
        rows.append({"feature": col, "max_abs_diff": float(0.0 if np.isnan(max_diff) else max_diff),
                     "pass": bool(np.nan_to_num(max_diff, nan=0.0) < 1e-8)})
    audit = pd.DataFrame(rows).sort_values("max_abs_diff", ascending=False)
    n_fail = int((~audit["pass"]).sum())
    log(f"causality audit: {len(audit)} features checked, {n_fail} failed")
    if n_fail:
        print(audit[~audit["pass"]].head(20))
        raise AssertionError("Look-ahead detected: some features change when future bars are removed.")
    return audit


def audit_label_alignment(logc: pd.Series, h: int, method: str, score: np.ndarray,
                          n_checks: int = 200, seed: int = 0,
                          sigma: Optional[np.ndarray] = None) -> float:
    """Recompute the label from raw closes for random rows; return max abs difference."""
    if method == "move_z" and sigma is None:
        return float("nan")
    rng = np.random.default_rng(seed)
    lc = logc.to_numpy()
    idx = rng.choice(np.arange(len(lc) - h - 1), size=min(n_checks, len(lc) - h - 2), replace=False)
    worst = 0.0
    for i in idx:
        window = lc[i: i + h + 1]
        if method == "efficiency":
            ref = abs(window[-1] - window[0]) / np.abs(np.diff(window)).sum()
        elif method == "r2":
            t = np.arange(len(window), dtype=float)
            tc = t - t.mean()
            b = (tc @ (window - window.mean())) / (tc ** 2).sum()
            fit = window.mean() + b * tc
            ref = 1 - ((window - fit) ** 2).sum() / ((window - window.mean()) ** 2).sum()
        elif method == "move_z":
            ref = abs(window[-1] - window[0]) / sigma[i]
        else:
            ref = abs(np.sign(np.diff(window)).sum()) / h
        if not np.isnan(score[i]):
            worst = max(worst, abs(ref - score[i]))
    log(f"label alignment audit (h={h}): max abs diff = {worst:.2e}")
    if worst > 1e-8:
        raise AssertionError("Label misalignment detected (off-by-one leak).")
    return worst


# %% [markdown]
# ## 6. Purged walk-forward validation
#
# With a 24h horizon on 4h bars, consecutive labels share up to 5 bars of future path.
# A random K-fold would put nearly-identical rows in train and validation → inflated scores.
# So:
#
# * folds are **contiguous time blocks**, always validating on data *after* training;
# * the last `H` training bars are **purged** (their labels overlap the validation block);
# * the final test set is the last 20% of history and is used exactly once.
#
# Thresholds, winsorisation bounds and the isotonic calibrator are all fitted on
# pre-test data only.

# %%
class PurgedWalkForward:
    def __init__(self, n_splits: int = 4, embargo: int = 0, min_train_frac: float = 0.4):
        self.n_splits = n_splits
        self.embargo = embargo
        self.min_train_frac = min_train_frac

    def split(self, n: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        n0 = int(self.min_train_frac * n)
        step = max((n - n0) // self.n_splits, 1)
        folds = []
        for k in range(self.n_splits):
            va = n0 + k * step
            vb = n if k == self.n_splits - 1 else min(n0 + (k + 1) * step, n)
            tr_end = max(va - self.embargo, 1)
            if va >= n or va <= tr_end:
                continue
            folds.append((np.arange(0, tr_end), np.arange(va, vb)))
        return folds


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation that tolerates NaNs (scipy-free)."""
    a = pd.Series(np.asarray(a, dtype=float))
    b = pd.Series(np.asarray(b, dtype=float))
    ok = a.notna() & b.notna()
    if ok.sum() < 30:
        return np.nan
    ra, rb = a[ok].rank().to_numpy(), b[ok].rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def metrics_dict(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    pred = (p >= thr).astype(int)
    base = float(y.mean()) if len(y) else np.nan
    single_class = len(np.unique(y)) < 2
    out = {
        "n": int(len(y)),
        "base_rate": base,
        "accuracy": accuracy_score(y, pred),
        "balanced_acc": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "mcc": matthews_corrcoef(y, pred) if len(np.unique(pred)) > 1 else 0.0,
        "roc_auc": roc_auc_score(y, p) if not single_class else np.nan,
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "brier": brier_score_loss(y, p),
    }
    b = min(max(base, 1e-6), 1 - 1e-6)
    base_ll = -(b * math.log(b) + (1 - b) * math.log(1 - b))
    out["logloss_skill_vs_base"] = 1 - out["log_loss"] / base_ll if base_ll > 0 else np.nan
    out["acc_minus_base_rate"] = out["accuracy"] - max(base, 1 - base) if not np.isnan(base) else np.nan
    return out


def threshold_analysis(y: np.ndarray, p: np.ndarray,
                       thresholds: Sequence[float] = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
                       ) -> pd.DataFrame:
    """What does it mean to act only when P(trending) >= t? Coverage vs precision vs lift.

    This is the honest answer to "the model says 70%" — 0.70 is a probability, and this
    table shows how often that probability was actually right and how often it occurs.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    base = float(y.mean())
    rows = []
    for t in thresholds:
        sel = p >= t
        n = int(sel.sum())
        rows.append({
            "threshold": t,
            "coverage": n / len(y),
            "n_flagged": n,
            "precision": float(y[sel].mean()) if n else np.nan,
            "recall": float(y[sel].sum() / max(y.sum(), 1)) if n else np.nan,
            "lift_vs_base": (float(y[sel].mean()) / base - 1) if n and base > 0 else np.nan,
            "base_rate": base,
        })
    return pd.DataFrame(rows)


def block_bootstrap_ci(y: np.ndarray, p: np.ndarray, metric_fn, n_rep: int = 400,
                       block: int = 24, seed: int = 0) -> Tuple[float, float]:
    """Moving-block bootstrap (respects autocorrelation of overlapping labels)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    p = np.asarray(p)
    n = len(y)
    block = min(block, max(n // 3, 2))
    n_blocks = int(np.ceil(n / block))
    vals = []
    for _ in range(n_rep):
        starts = rng.integers(0, n - block, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        try:
            vals.append(metric_fn(y[idx], p[idx]))
        except Exception:
            continue
    if not vals:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


# %% [markdown]
# ## 7. Dataset assembly: features + labels + masks

# %%
def build_panel(cfg: Config, df_map: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, pd.DataFrame]:
    """Load + feature-engineer every symbol, add cross-asset context, apply gap masks."""
    if df_map is None:
        df_map = {}
        for sym in cfg.symbols:
            log(f"loading {sym} {cfg.interval} ({cfg.data_source})")
            df_map[sym] = load_symbol(sym, cfg)
    # restrict to a common time range across symbols
    start = max(d.index.min() for d in df_map.values())
    end = min(d.index.max() for d in df_map.values())
    df_map = {s: d.loc[(d.index >= start) & (d.index <= end)] for s, d in df_map.items()}

    feats = {s: build_features(d, cfg) for s, d in df_map.items()}
    feats = add_cross_asset_features(feats, cfg)

    if cfg.drop_gap_windows:
        for s, f in feats.items():
            gap = f["_gap_bar"].fillna(0).to_numpy()
            gap_recent = pd.Series(gap).rolling(MAX_LOOKBACK, min_periods=1).max().to_numpy()
            f["_clean"] = (gap_recent == 0).astype(float)
    else:
        for f in feats.values():
            f["_clean"] = 1.0
    return feats


def make_panel(feats: Dict[str, pd.DataFrame], horizon_bars: int, cfg: Config) -> pd.DataFrame:
    """Stack every symbol into one long panel: features + trend score + forward return."""
    method = resolve_label_method(horizon_bars, cfg)
    blocks = []
    for sym, f in feats.items():
        logc = f["_logc"].to_numpy(dtype=float)
        sigma = np.exp(f["rv48"].to_numpy(dtype=float))          # trailing per-bar volatility
        s = pd.DataFrame(f[[c for c in f.columns if not c.startswith("_")]].copy())
        s["symbol"] = sym
        s["trend_score"] = forward_trend_score(logc, horizon_bars, method, sigma=sigma)
        s["fwd_ret"] = forward_return(logc, horizon_bars)
        s["_clean"] = f["_clean"].to_numpy()
        s["_label_method"] = method
        s = s.iloc[MAX_LOOKBACK: len(s) - horizon_bars]      # drop warm-up and unlabelled tail
        blocks.append(s[s["_clean"] > 0])
    panel = pd.concat(blocks).sort_index()
    return panel


def assemble_X(panel: pd.DataFrame, cfg: Config, prep: Optional[Dict] = None) -> pd.DataFrame:
    """Feature matrix (+ symbol one-hots). With `prep`, columns are aligned and clamped
    exactly the way they were at training time."""
    drop = {"symbol", "trend_score", "fwd_ret", "_clean", "_label_method"}
    feat_cols = [c for c in panel.columns if c not in drop]
    X = panel[feat_cols].astype(float).reset_index(drop=True)
    if cfg.pool_symbols:
        dummies = pd.get_dummies(panel["symbol"].to_numpy(), prefix="sym").astype(float)
        if prep is not None:
            for c in prep["dummy_cols"]:
                if c not in dummies.columns:
                    dummies[c] = 0.0
            dummies = dummies[prep["dummy_cols"]]
        else:
            dummies = dummies[sorted(dummies.columns)]
        X = pd.concat([X, dummies.reset_index(drop=True)], axis=1)
    X.index = panel.index
    if prep is not None:
        X = X.reindex(columns=prep["feature_names"])
        X = X.clip(lower=prep["clip_lo"], upper=prep["clip_hi"], axis=1)
        X = X.fillna(prep["fill_values"])
    return X


def make_xy(feats: Dict[str, pd.DataFrame], horizon_bars: int, cfg: Config,
            fit_end: Optional[pd.Timestamp] = None
            ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, Dict]:
    """Assemble (X, y, meta, prep) for one horizon.

    `prep` holds everything a live prediction needs (thresholds, clamp bounds, column order)
    and is fitted on the **pre-test** region only, so the test set never shapes the data prep.
    """
    panel = make_panel(feats, horizon_bars, cfg)
    if fit_end is None:
        fit_end = panel.index[int(len(panel) * cfg.min_train_frac)]

    # ---- per-symbol threshold, fitted on the training region only ----
    thr_map: Dict[str, float] = {}
    for sym, grp in panel.groupby("symbol"):
        if cfg.label_absolute_threshold is not None:
            thr_map[sym] = float(cfg.label_absolute_threshold)
        else:
            fit_scores = grp.loc[grp.index < fit_end, "trend_score"].dropna()
            thr_map[sym] = float(np.quantile(fit_scores, cfg.label_quantile))
    y = pd.Series((panel["trend_score"] > panel["symbol"].map(thr_map)).astype(int),
                  index=panel.index, name="trending")

    # optional "ignore dead markets" filter (a flat window has no meaningful regime)
    if cfg.min_abs_fwd_return_quantile is not None:
        fit_rows = panel.index < fit_end
        cut = np.quantile(np.abs(panel.loc[fit_rows, "fwd_ret"]), cfg.min_abs_fwd_return_quantile)
        keep = (np.abs(panel["fwd_ret"]) >= cut).to_numpy()
        panel, y = panel[keep], y[keep]

    X_all = assemble_X(panel, cfg)
    train_rows = (X_all.index < fit_end)
    lo_q, hi_q = cfg.winsorize_quantiles
    clip_lo = X_all.loc[train_rows].quantile(lo_q)
    clip_hi = X_all.loc[train_rows].quantile(hi_q)
    X_c = X_all.clip(lower=clip_lo, upper=clip_hi, axis=1)

    nunique = X_c.loc[train_rows].nunique(dropna=True)   # drop constant/all-NaN columns
    feature_names = [c for c in X_c.columns if nunique.get(c, 0) > 1]
    fill_values = X_c.loc[train_rows, feature_names].median(numeric_only=True)

    prep = dict(thresholds=thr_map, clip_lo=clip_lo, clip_hi=clip_hi,
                feature_names=feature_names,
                dummy_cols=[c for c in X_all.columns if c.startswith("sym_")],
                fill_values=fill_values)
    X = X_c[feature_names].fillna(fill_values)

    meta = panel[["symbol", "trend_score", "fwd_ret"]].copy()
    meta["horizon_hours"] = horizon_bars * cfg.bar_hours
    meta["threshold"] = meta["symbol"].map(thr_map)
    return X.astype(float), y, meta, prep


def build_prediction_frame(feats: Dict[str, pd.DataFrame], prep: Dict, cfg: Config,
                           n_last: int = 1) -> pd.DataFrame:
    """Feature matrix for the most recent bar(s) of every symbol, ready for `model.predict_proba`."""
    blocks = []
    for sym, f in feats.items():
        cols = [c for c in f.columns if not c.startswith("_")]
        b = f[cols].iloc[-n_last:].copy()
        b["symbol"] = sym
        blocks.append(b)
    panel = pd.concat(blocks).sort_index()
    X = assemble_X(panel, cfg, prep=prep)
    X = X.copy()
    X["symbol"] = panel["symbol"].to_numpy()
    return X


# %% [markdown]
# ## 8. Training, calibration, multi-horizon comparison

# %%
def _make_model(params: Dict, n_estimators: Optional[int] = None,
                early_stopping: Optional[int] = None) -> xgb.XGBClassifier:
    p = dict(params)
    p.pop("n_estimators", None)
    p.pop("early_stopping_rounds", None)
    return xgb.XGBClassifier(
        n_estimators=n_estimators or params.get("n_estimators", 1000),
        early_stopping_rounds=early_stopping,
        **p,
    )


def fit_folds(X: pd.DataFrame, y: pd.Series, folds, params: Dict, cfg: Config):
    """Train one model per purged walk-forward fold; collect OOF preds + best iterations."""
    oof = np.full(len(X), np.nan)
    best_iters, fold_models, fold_stats = [], [], []
    for k, (tr, va) in enumerate(folds):
        model = _make_model(params, early_stopping=cfg.early_stopping_rounds)
        model.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])], verbose=False)
        p = model.predict_proba(X.iloc[va])[:, 1]
        oof[va] = p
        best_iters.append(int(getattr(model, "best_iteration", params["n_estimators"])))
        fold_models.append(model)
        m = metrics_dict(y.iloc[va].to_numpy(), p)
        tr_p = model.predict_proba(X.iloc[tr][: min(len(tr), 20000)])[:, 1]
        m_train = metrics_dict(y.iloc[tr][: len(tr_p)].to_numpy(), tr_p)
        m["fold"] = k
        m["train_log_loss"] = m_train["log_loss"]
        m["train_roc_auc"] = m_train["roc_auc"]
        m["overfit_gap_logloss"] = m["log_loss"] - m_train["log_loss"]
        m["best_iteration"] = best_iters[-1]
        fold_stats.append(m)
    return oof, best_iters, fold_models, pd.DataFrame(fold_stats)


def fit_final(X_tr, y_tr, n_estimators: int, params: Dict) -> xgb.XGBClassifier:
    model = _make_model(params, n_estimators=max(int(n_estimators), 50))
    model.fit(X_tr, y_tr, verbose=False)
    return model


class _GuardMixin:
    """Keeps a calibration map only if it improves log loss on a time-held-out OOF slice."""

    guard_frac: float = 0.3
    info: Dict[str, float]

    @staticmethod
    def _ll(p, y) -> float:
        return float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]))

    def _guard(self, p_fit, y_fit, p_chk, y_chk) -> bool:
        ll_raw = self._ll(p_chk, y_chk)
        ll_cal = self._ll(self._fit_map(p_fit, y_fit)(p_chk), y_chk)
        self.info = {"guard_logloss_raw": ll_raw, "guard_logloss_calibrated": ll_cal,
                     "applied": float(ll_cal < ll_raw - 1e-4)}
        if self.info["applied"]:
            log(f"  {type(self).__name__} applied (held-out OOF log loss "
                f"{ll_raw:.4f} -> {ll_cal:.4f})")
        else:
            log(f"  {type(self).__name__} REJECTED (held-out OOF log loss "
                f"{ll_raw:.4f} -> {ll_cal:.4f}); keeping raw probabilities")
        return bool(self.info["applied"])


class TemperatureCalibrator(_GuardMixin):
    """Two-parameter Platt scaling on the logit: p' = sigmoid(a * logit(p) + b).

    Far more robust than isotonic when data is scarce and the signal is weak (isotonic
    tends to push already-compressed probabilities towards 0/1 and can destroy log loss).
    """

    def __init__(self, enabled: bool = True, guard_frac: float = 0.3):
        self.enabled = enabled
        self.guard_frac = guard_frac
        self.info = {}
        self.lr = None

    @staticmethod
    def _logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def _fit_map(self, p, y):
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000).fit(self._logit(p), y)
        return lambda q: lr.predict_proba(self._logit(q))[:, 1]

    def fit(self, p_oof: np.ndarray, y: np.ndarray):
        ok = ~np.isnan(p_oof)
        p, yy = p_oof[ok], np.asarray(y)[ok].astype(int)
        if not self.enabled or len(p) < 300 or len(np.unique(yy)) < 2:
            return self
        n_guard = int(len(p) * self.guard_frac)
        if self._guard(p[:-n_guard], yy[:-n_guard], p[-n_guard:], yy[-n_guard:]):
            from sklearn.linear_model import LogisticRegression
            self.lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000).fit(
                self._logit(p), yy)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        if self.lr is None:
            return np.asarray(p, dtype=float)
        return np.clip(self.lr.predict_proba(self._logit(p))[:, 1], 1e-6, 1 - 1e-6)


class IsotonicCalibrator(_GuardMixin):
    """Monotone isotonic map, kept only if it wins on a held-out OOF slice."""

    def __init__(self, enabled: bool = True, guard_frac: float = 0.3):
        self.enabled = enabled
        self.guard_frac = guard_frac
        self.info = {}
        self.iso: Optional[IsotonicRegression] = None

    def _fit_map(self, p, y):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
        return lambda q: iso.predict(q)

    def fit(self, p_oof: np.ndarray, y: np.ndarray):
        ok = ~np.isnan(p_oof)
        p, yy = p_oof[ok], np.asarray(y)[ok].astype(int)
        if not self.enabled or len(p) < 300 or len(np.unique(yy)) < 2:
            return self
        n_guard = int(len(p) * self.guard_frac)
        if self._guard(p[:-n_guard], yy[:-n_guard], p[-n_guard:], yy[-n_guard:]):
            self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, yy)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        if self.iso is None:
            return np.asarray(p, dtype=float)
        return np.clip(self.iso.predict(p), 1e-6, 1 - 1e-6)


def make_calibrator(kind: str):
    """Factory: "temperature" (default), "isotonic", or "none"."""
    kind = (kind or "none").lower()
    if kind == "temperature":
        return TemperatureCalibrator()
    if kind == "isotonic":
        return IsotonicCalibrator()
    if kind == "none":
        return TemperatureCalibrator(enabled=False)
    raise ValueError(f"unknown calibration kind {kind!r}")


def hpo_search(X, y, folds, base_params: Dict, cfg: Config) -> Tuple[Dict, pd.DataFrame]:
    """Small random search scored by purged-CV log loss (proper scoring rule)."""
    rng = np.random.default_rng(RNG_SEED)
    space = dict(
        max_depth=[3, 4, 5, 6],
        learning_rate=[0.02, 0.03, 0.05],
        min_child_weight=[5, 10, 20, 40],
        subsample=[0.7, 0.8, 0.9],
        colsample_bytree=[0.4, 0.6, 0.8],
        reg_lambda=[0.5, 1.0, 2.0, 5.0],
        reg_alpha=[0.0, 0.5, 1.0],
    )
    trials, rows = [dict(base_params)], []
    for _ in range(max(cfg.hpo_iter - 1, 0)):
        p = dict(base_params)
        for k, v in space.items():
            p[k] = v[rng.integers(len(v))]
        trials.append(p)

    for i, p in enumerate(trials):
        p = dict(p)
        p.setdefault("n_estimators", 3000)
        ll, auc, acc = [], [], []
        for tr, va in folds:
            m = _make_model(p, early_stopping=cfg.early_stopping_rounds)
            m.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])], verbose=False)
            pr = m.predict_proba(X.iloc[va])[:, 1]
            md = metrics_dict(y.iloc[va].to_numpy(), pr)
            ll.append(md["log_loss"])
            auc.append(md["roc_auc"])
            acc.append(md["accuracy"])
        rows.append({**{k: p[k] for k in space}, "cv_logloss": np.mean(ll),
                     "cv_logloss_std": np.std(ll), "cv_auc": np.nanmean(auc),
                     "cv_accuracy": np.mean(acc)})
        log(f"    hpo {i + 1}/{len(trials)}: logloss={np.mean(ll):.4f} auc={np.nanmean(auc):.4f} "
            f"acc={np.mean(acc):.4f}")
    res = pd.DataFrame(rows)
    res["score"] = res["cv_logloss"] + 0.1 * res["cv_logloss_std"]
    best = res.sort_values("score").iloc[0]
    best_params = dict(base_params)
    for k in space:
        best_params[k] = best[k]
        best_params[k] = int(best_params[k]) if k in ("max_depth", "min_child_weight") else float(best_params[k])
    log(f"  best params: " + ", ".join(f"{k}={best_params[k]}" for k in space))
    return best_params, res.sort_values("score").reset_index(drop=True)


# %% [markdown]
# ## 9. One full run for a single horizon
#
# Steps: split → (optional) HPO on purged folds → fold models (early stopping) → OOF
# calibration → final model on all pre-test data → **one** evaluation on the untouched test set.

# %%
def run_horizon(feats: Dict[str, pd.DataFrame], h_hours: int, cfg: Config,
                params: Optional[Dict] = None, verbose: bool = True) -> Dict:
    """Train + validate + test one horizon. Returns a result dict with models and metrics.

    `params=None` triggers the (optional) purged-CV hyper-parameter search for this horizon.
    """
    h = max(int(round(h_hours / cfg.bar_hours)), 1)
    do_hpo = params is None and cfg.run_hpo
    params = dict(params or cfg.params)
    params.setdefault("n_estimators", 3000)

    X, y, meta, prep = make_xy(feats, h, cfg)
    n = len(X)
    n_test = int(n * cfg.test_frac)
    dev_end = n - n_test

    folds = PurgedWalkForward(cfg.n_cv_splits, embargo=h, min_train_frac=cfg.min_train_frac).split(dev_end)
    method = resolve_label_method(h, cfg)
    if verbose:
        note = "" if method == cfg.label_method else f" [label={method}: 1-bar efficiency is degenerate]"
        log(f"h={h_hours}h ({h} bars) | rows={n:,} dev={dev_end:,} test={n_test:,} "
            f"| folds={len(folds)} | trending share={y.mean():.3f}{note}")
    if y.nunique() < 2:
        raise ValueError(
            f"Label for the {h_hours}h horizon has a single class. Use more history, a longer "
            f"horizon, or cfg.label_quantile != 0/1.")
    if min(n_test, dev_end) < 500:
        raise ValueError(
            f"Only {n:,} usable rows for the {h_hours}h horizon - not enough to validate. "
            f"Load more history (cfg.start) or fewer/greater symbols.")

    hpo_table = None
    if do_hpo:
        params, hpo_table = hpo_search(X.iloc[:dev_end], y.iloc[:dev_end], folds, params, cfg)

    oof, best_iters, fold_models, fold_df = fit_folds(
        X.iloc[:dev_end], y.iloc[:dev_end], folds, params, cfg)
    n_est = int(np.mean(best_iters) * 1.1) if best_iters else params["n_estimators"]

    calib = make_calibrator(cfg.calibration).fit(oof, y.iloc[:dev_end].to_numpy())

    final = fit_final(X.iloc[:dev_end], y.iloc[:dev_end], n_est, params)

    # ---- untouched test set ----
    X_te, y_te = X.iloc[dev_end:], y.iloc[dev_end:]
    p_raw = final.predict_proba(X_te)[:, 1]
    p_cal = calib.transform(p_raw)
    test_raw = metrics_dict(y_te.to_numpy(), p_raw)
    test_cal = metrics_dict(y_te.to_numpy(), p_cal)

    oof_metrics = metrics_dict(y.iloc[:dev_end].to_numpy()[~np.isnan(oof)], oof[~np.isnan(oof)])

    res = dict(
        horizon_hours=h_hours,
        horizon_bars=h,
        params=params,
        n_estimators=n_est,
        best_iters=best_iters,
        thresholds=prep["thresholds"],
        prep=prep,
        fold_metrics=fold_df,
        oof_metrics=oof_metrics,
        test_metrics_raw=test_raw,
        test_metrics_cal=test_cal,
        hpo_table=hpo_table,
        model=final,
        calibrator=calib,
        fold_models=fold_models,
        feature_names=list(X.columns),
        X=X, y=y, meta=meta, dev_end=dev_end,
        p_test_raw=p_raw, p_test_cal=p_cal, y_test=y_te.to_numpy(),
        p_dev_raw=oof,
    )

    # ---- quality diagnostics on test ----
    te_meta = meta.iloc[dev_end:]
    res["score_spearman_test"] = spearman(p_cal, te_meta["trend_score"].to_numpy())
    res["absret_spearman_test"] = spearman(p_cal, te_meta["fwd_ret"].abs().to_numpy())
    # does a high probability actually precede bigger moves?
    hi = p_cal >= np.quantile(p_cal, 0.7)
    abs_fwd = te_meta["fwd_ret"].abs().to_numpy()
    res["mean_abs_fwd_ret_hi_prob"] = float(np.nanmean(abs_fwd[hi]))
    res["mean_abs_fwd_ret_lo_prob"] = float(np.nanmean(abs_fwd[~hi]))
    # bootstrap CIs (block bootstrap: overlapping labels are autocorrelated)
    res["test_ci_accuracy"] = block_bootstrap_ci(
        y_te.to_numpy(), p_cal, lambda a, b: accuracy_score(a, (b >= 0.5).astype(int)),
        cfg.bootstrap_reps, block=max(h * 2, 12))
    res["test_ci_auc"] = block_bootstrap_ci(
        y_te.to_numpy(), p_cal, lambda a, b: roc_auc_score(a, b) if len(np.unique(a)) > 1 else np.nan,
        cfg.bootstrap_reps, block=max(h * 2, 12))

    # per-symbol test breakdown (pooled models). Positions are resolved positionally because
    # the pooled index has duplicate timestamps across symbols.
    te_pd = pd.DataFrame({
        "symbol": te_meta["symbol"].to_numpy(),
        "y": y_te.to_numpy(),
        "p": p_cal,
    })
    rows = []
    for sym, g in te_pd.groupby("symbol"):
        rows.append({"symbol": sym, **metrics_dict(g["y"].to_numpy(), g["p"].to_numpy())})
    res["per_symbol_test"] = pd.DataFrame(rows)
    res["threshold_analysis"] = threshold_analysis(y_te.to_numpy(), p_cal)
    res["calibration_info"] = res["calibrator"].info

    if verbose:
        log(f"  test (raw):       acc={test_raw['accuracy']:.4f} auc={test_raw['roc_auc']:.4f} "
            f"logloss={test_raw['log_loss']:.4f} brier={test_raw['brier']:.4f} mcc={test_raw['mcc']:.3f}")
        log(f"  test (calibrated):acc={test_cal['accuracy']:.4f} auc={test_cal['roc_auc']:.4f} "
            f"logloss={test_cal['log_loss']:.4f} brier={test_cal['brier']:.4f}")
    return res

# %% [markdown]
# ## 10. Overfitting & leakage stress tests

# %%
def run_label_shuffle_test(feats: Dict[str, pd.DataFrame], h_hours: int, cfg: Config,
                           n_perm: int = 3) -> pd.DataFrame:
    """Shuffle labels in time -> AUC must fall to ~0.5. Otherwise the CV is leaking."""
    h = max(int(round(h_hours / cfg.bar_hours)), 1)
    X, y, meta, prep = make_xy(feats, h, cfg)
    n = len(X)
    dev_end = n - int(n * cfg.test_frac)
    folds = PurgedWalkForward(cfg.n_cv_splits, embargo=h, min_train_frac=cfg.min_train_frac).split(dev_end)
    tr, va = folds[-1]
    rows = []
    for i in range(n_perm):
        rng = np.random.default_rng(1000 + i)
        y_sh = pd.Series(rng.permutation(y.iloc[:dev_end].to_numpy()), index=y.iloc[:dev_end].index)
        m = _make_model({**cfg.params, "n_estimators": 300}, early_stopping=50)
        m.fit(X.iloc[:dev_end].iloc[tr], y_sh.iloc[tr], eval_set=[(X.iloc[:dev_end].iloc[va], y_sh.iloc[va])],
              verbose=False)
        p = m.predict_proba(X.iloc[:dev_end].iloc[va])[:, 1]
        md = metrics_dict(y_sh.iloc[va].to_numpy(), p)
        rows.append({"perm": i, "auc": md["roc_auc"], "accuracy": md["accuracy"]})
    df = pd.DataFrame(rows)
    log(f"label-shuffle test (h={h_hours}h): mean AUC={df['auc'].mean():.4f} "
        f"(expected ~0.50) acc={df['accuracy'].mean():.4f}")
    if df["auc"].mean() > 0.56:
        log("  !! WARNING: shuffled labels still score above 0.56 -> check purging/embargo")
    return df


def overfitting_report(res: Dict) -> pd.DataFrame:
    """Compare train vs CV vs test to expose memorisation."""
    fm = res["fold_metrics"]
    rows = [{
        "horizon_hours": res["horizon_hours"],
        "train_logloss": fm["train_log_loss"].mean(),
        "cv_logloss": fm["log_loss"].mean(),
        "test_logloss": res["test_metrics_cal"]["log_loss"],
        "cv_minus_train": (fm["log_loss"] - fm["train_log_loss"]).mean(),
        "test_minus_cv": res["test_metrics_cal"]["log_loss"] - fm["log_loss"].mean(),
        "train_auc": fm["train_roc_auc"].mean(),
        "cv_auc": fm["roc_auc"].mean(),
        "test_auc": res["test_metrics_cal"]["roc_auc"],
        "cv_auc_std": fm["roc_auc"].std(),
        "cv_acc_std": fm["accuracy"].std(),
        "mean_best_iteration": fm["best_iteration"].mean(),
        "n_features": len(res["feature_names"]),
        "test_n": res["test_metrics_cal"]["n"],
    }]
    return pd.DataFrame(rows)


def feature_importance_table(res: Dict, top: int = 25) -> pd.DataFrame:
    model = res["model"]
    gain = model.get_booster().get_score(importance_type="gain")
    total = max(sum(gain.values()), 1e-9)
    imp = pd.DataFrame({"feature": list(gain.keys()), "gain": list(gain.values())})
    imp["gain_pct"] = 100 * imp["gain"] / total
    imp = imp.sort_values("gain", ascending=False)
    return imp.head(top).reset_index(drop=True)


def permutation_importance_table(res: Dict, n_repeats: int = 5, top: int = 20) -> pd.DataFrame:
    from sklearn.inspection import permutation_importance
    X_te = res["X"].iloc[res["dev_end"]:]
    y_te = res["y_test"]
    r = permutation_importance(res["model"], X_te, y_te, scoring="roc_auc",
                              n_repeats=n_repeats, random_state=RNG_SEED, n_jobs=-1)
    df = pd.DataFrame({"feature": X_te.columns, "auc_drop_mean": r.importances_mean,
                       "auc_drop_std": r.importances_std})
    return df.sort_values("auc_drop_mean", ascending=False).head(top).reset_index(drop=True)


# %% [markdown]
# ## 11. Plots

# %%
def _savefig(fig, cfg: Config, name: str):
    if cfg.save_figures:
        os.makedirs(cfg.out_dir, exist_ok=True)
        path = os.path.join(cfg.out_dir, name)
        fig.savefig(path, dpi=130, bbox_inches="tight")
        log(f"  saved {path}")


def plot_regime_overlay(res: Dict, feats: Dict[str, pd.DataFrame], cfg: Config,
                        symbol: Optional[str] = None):
    """Price + model probability; background shaded where the model says 'trending'."""
    if not HAS_MPL:
        return None
    symbol = symbol or (cfg.market_symbol if cfg.market_symbol in feats else list(feats)[0])
    te_meta = res["meta"].iloc[res["dev_end"]:]
    mask = (te_meta["symbol"] == symbol).to_numpy()
    times = te_meta.index[mask]
    price = np.exp(feats[symbol]["_logc"].reindex(times).to_numpy())
    p = res["p_test_cal"][mask]
    y = res["y_test"][mask]

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1.2]})
    axes[0].plot(times, price, color="black", lw=1.1, label=f"{symbol} close")
    axis_vals = np.where(y == 1, price, np.nan)
    axes[0].scatter(times, axis_vals, s=6, color="tab:green", alpha=0.35, label="true: trending window")
    axes[0].set_title(f"{symbol} — test period, horizon {res['horizon_hours']}h "
                      f"(shaded = model P(trending) > 0.5)")
    axes[0].legend(loc="upper left", fontsize=8)
    for t0, pt in zip(times, p):
        if pt >= 0.5:
            axes[0].axvspan(t0, t0 + pd.Timedelta(hours=cfg.bar_hours), color="tab:green", alpha=0.08)
    axes[1].plot(times, p, color="tab:blue", lw=1.0, label="P(trending)")
    axes[1].axhline(0.5, color="grey", ls="--", lw=0.8)
    axes[1].axhline(0.7, color="tab:red", ls=":", lw=0.8, label="0.70 threshold")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_ylabel("probability")
    axes[1].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    _savefig(fig, cfg, f"regime_overlay_{res['horizon_hours']}h.png")
    return fig


def plot_reliability(res: Dict, cfg: Config, n_bins: int = 10):
    if not HAS_MPL:
        return None
    y, p_raw, p_cal = res["y_test"], res["p_test_raw"], res["p_test_cal"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, p, name in ((axes[0], p_raw, "raw XGBoost"), (axes[1], p_cal, "isotonic-calibrated")):
        bins = np.quantile(p, np.linspace(0, 1, n_bins + 1))
        bins = np.unique(bins)
        idx = np.clip(np.digitize(p, bins[1:-1]), 0, len(bins) - 2)
        xs, ys = [], []
        for b in range(len(bins) - 1):
            sel = idx == b
            if sel.sum() >= 30:
                xs.append(p[sel].mean())
                ys.append(y[sel].mean())
        ax.plot([0, 1], [0, 1], "--", color="grey", lw=0.9)
        ax.plot(xs, ys, "o-", color="tab:blue")
        ax.set_title(f"{name}\nBrier={brier_score_loss(y, p):.4f} "
                     f"logloss={log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)):.4f}")
        ax.set_xlabel("predicted P(trending)")
        ax.set_ylabel("observed frequency")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    fig.suptitle(f"Calibration on the untouched test set — horizon {res['horizon_hours']}h")
    fig.tight_layout()
    _savefig(fig, cfg, f"reliability_{res['horizon_hours']}h.png")
    return fig


def plot_prob_distribution(res: Dict, cfg: Config):
    if not HAS_MPL:
        return None
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.hist(res["p_test_cal"][res["y_test"] == 0], bins=30, alpha=0.6, label="true: ranging", color="tab:red")
    ax.hist(res["p_test_cal"][res["y_test"] == 1], bins=30, alpha=0.6, label="true: trending", color="tab:green")
    ax.set_xlabel("predicted P(trending)")
    ax.set_ylabel("bars")
    ax.set_title(f"Separation on test set — {res['horizon_hours']}h horizon")
    ax.legend()
    fig.tight_layout()
    _savefig(fig, cfg, f"prob_dist_{res['horizon_hours']}h.png")
    return fig


def plot_learning_curve(res: Dict, cfg: Config):
    """Train vs validation log loss per boosting round -> shows if/when trees start overfitting."""
    if not HAS_MPL or not res["fold_models"]:
        return None
    ev = getattr(res["fold_models"][-1], "evals_result_", None)
    if not ev:
        return None
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for k, v in ev.items():
        if "logloss" in v:
            ax.plot(v["logloss"], label=f"{k} logloss", lw=1.2)
    ax.axvline(res["fold_metrics"]["best_iteration"].mean(), color="grey", ls="--", lw=0.9,
               label="mean best_iteration")
    ax.set_xlabel("boosting round")
    ax.set_ylabel("log loss")
    ax.set_title(f"Boosting curve (last CV fold) — {res['horizon_hours']}h")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _savefig(fig, cfg, f"learning_curve_{res['horizon_hours']}h.png")
    return fig


def plot_horizon_comparison(summary: pd.DataFrame, cfg: Config):
    if not HAS_MPL:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, col, title in zip(axes, ["roc_auc", "accuracy", "log_loss"],
                              ["ROC AUC (test)", "Accuracy (test)", "Log loss (test, lower=better)"]):
        ax.plot(summary["horizon_hours"], summary[col], "o-", color="tab:blue")
        if col == "accuracy":
            ax.plot(summary["horizon_hours"], summary["base_rate"], "s--", color="grey",
                    label="majority-class baseline")
            ax.legend(fontsize=8)
        ax.set_xlabel("horizon (hours)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        if col == "roc_auc":
            ax.axhline(0.5, color="red", ls=":", lw=0.9)
    fig.suptitle("Horizon comparison (untouched test set)")
    fig.tight_layout()
    _savefig(fig, cfg, "horizon_comparison.png")
    return fig


# %% [markdown]
# ## 12. Prediction helper — "70% probability it will be trending over the next 12 hours"

# %%
def predict_regime(artifacts: Dict[int, Dict], feats: Dict[str, pd.DataFrame], cfg: Config,
                   symbol: Optional[str] = None, calibrated: bool = True) -> pd.DataFrame:
    """Live probabilities for the most recent *closed* bar, for every trained horizon."""
    rows = []
    symbol = symbol or (cfg.market_symbol if cfg.market_symbol in feats else list(feats)[0])
    for h_hours, res in sorted(artifacts.items()):
        X_live = build_prediction_frame(feats, res["prep"], cfg)
        if X_live.empty:
            continue
        row = X_live.loc[X_live["symbol"] == symbol]
        if row.empty:
            continue
        Xrow = row[res["feature_names"]].astype(float)
        p = res["model"].predict_proba(Xrow)[:, 1]
        if calibrated:
            p = res["calibrator"].transform(p)
        p = float(p[0])
        rows.append({
            "symbol": symbol,
            "as_of": row.index[0],
            "horizon_hours": h_hours,
            "p_trending": round(p, 4),
            "regime": "TRENDING" if p >= 0.5 else "RANGING",
            "confidence": "high" if abs(p - 0.5) >= 0.15 else "low",
        })
    return pd.DataFrame(rows)


def narrate(pred: pd.DataFrame) -> str:
    lines = []
    for _, r in pred.iterrows():
        lines.append(
            f"{r['as_of']:%Y-%m-%d %H:%M} UTC | {r['symbol']} | next {r['horizon_hours']}h: "
            f"{r['p_trending'] * 100:.1f}% probability of a TRENDING regime "
            f"(-> {r['regime']}, {r['confidence']} confidence)")
    return "\n".join(lines)


# %% [markdown]
# ## 13. Main driver

# %%
def main(cfg: Config = CONFIG, df_map: Optional[Dict[str, pd.DataFrame]] = None) -> Dict:
    os.makedirs(cfg.out_dir, exist_ok=True)
    t_start = time.time()

    # ---------------- data ----------------
    log("=" * 78)
    log("STEP 1/6  data loading & cleaning")
    if df_map is None:
        df_map = {}
        for sym in cfg.symbols:
            df_map[sym] = load_symbol(sym, cfg)
    for s, d in df_map.items():
        log(f"  {s}: {len(d):,} bars  {d.index.min():%Y-%m-%d} -> {d.index.max():%Y-%m-%d}")

    # ---------------- leakage audit ----------------
    log("STEP 2/6  leakage audit (causality + label alignment)")
    audit_sym = cfg.symbols[0] if cfg.symbols[0] in df_map else list(df_map)[0]
    audit_df = df_map[audit_sym]
    n_audit = min(len(audit_df), 6000)
    causality = audit_feature_causality(audit_df.tail(n_audit), cfg)
    lc = np.log(audit_df["close"].iloc[-n_audit:])
    if cfg.drop_gap_windows:
        _feat_audit = build_features(audit_df.tail(n_audit), cfg)
        sigma_audit = np.exp(_feat_audit["rv48"].to_numpy(dtype=float))
    else:
        sigma_audit = None
    for h_hours in cfg.horizons_hours:
        h = max(int(round(h_hours / cfg.bar_hours)), 1)
        method = resolve_label_method(h, cfg)
        sc = forward_trend_score(lc.to_numpy(dtype=float), h, method, sigma=sigma_audit)
        audit_label_alignment(lc, h, method, sc, sigma=sigma_audit)

    # ---------------- features ----------------
    log("STEP 3/6  feature engineering")
    feats = build_panel(cfg, df_map)
    log(f"  raw features per symbol: {len([c for c in feats[audit_sym].columns if not c.startswith('_')])}")

    if cfg.run_permutation_test:
        log("  leakage stress test (shuffled labels must score ~0.50 AUC)")
        run_label_shuffle_test(feats, cfg.hpo_horizon_hours, cfg, n_perm=2)

    # ---------------- train every horizon ----------------
    log("STEP 4/6  training")
    results: Dict[int, Dict] = {}
    tuned_params = None
    if cfg.run_hpo:
        log(f"  hyper-parameter search on the {cfg.hpo_horizon_hours}h horizon "
            f"(purged CV, {cfg.hpo_iter} candidates)")
        res_tune = run_horizon(feats, cfg.hpo_horizon_hours, cfg, params=None)
        tuned_params = res_tune["params"]
        results[cfg.hpo_horizon_hours] = res_tune
    for h_hours in cfg.horizons_hours:
        if h_hours in results:
            continue
        log("-" * 78)
        results[h_hours] = run_horizon(feats, h_hours, cfg, params=tuned_params)

    # ---------------- report ----------------
    log("STEP 5/6  results")
    summary_rows = []
    for h_hours, res in sorted(results.items()):
        m, ci_a, ci_u = res["test_metrics_cal"], res["test_ci_accuracy"], res["test_ci_auc"]
        summary_rows.append({
            "horizon_hours": h_hours,
            "n_test": m["n"],
            "base_rate": round(m["base_rate"], 4),
            "accuracy": round(m["accuracy"], 4),
            "acc_95ci": f"[{ci_a[0]:.3f}, {ci_a[1]:.3f}]",
            "balanced_acc": round(m["balanced_acc"], 4),
            "roc_auc": round(m["roc_auc"], 4),
            "auc_95ci": f"[{ci_u[0]:.3f}, {ci_u[1]:.3f}]",
            "log_loss": round(m["log_loss"], 4),
            "brier": round(m["brier"], 4),
            "mcc": round(m["mcc"], 4),
            "logloss_skill_vs_base": round(m["logloss_skill_vs_base"], 4),
            "lift_over_majority": round(m["acc_minus_base_rate"], 4),
            "label": resolve_label_method(res["horizon_bars"], cfg),
        "spearman_prob_vs_trend_score": round(res["score_spearman_test"], 4),
        })
    summary = pd.DataFrame(summary_rows)
    print("\n" + "=" * 78)
    print(f"TEST-SET PERFORMANCE (untouched last 20% of history, calibration={cfg.calibration})")
    print("=" * 78)
    print(summary.to_string(index=False))

    overfit = pd.concat([overfitting_report(r) for _, r in sorted(results.items())])
    print("\nOVERFITTING / STABILITY REPORT")
    print(overfit.to_string(index=False))

    for h_hours, res in sorted(results.items()):
        print(f"\nCV fold stability — {h_hours}h horizon (purged walk-forward, "
              f"embargo={res['horizon_bars']} bars)")
        fm = res["fold_metrics"][["fold", "n", "base_rate", "accuracy", "roc_auc",
                                  "log_loss", "train_log_loss", "overfit_gap_logloss", "best_iteration"]]
        print(fm.round(4).to_string(index=False))
        print(f"  CV mean±std: acc={fm['accuracy'].mean():.4f}±{fm['accuracy'].std():.4f} "
              f"auc={fm['roc_auc'].mean():.4f}±{fm['roc_auc'].std():.4f} "
              f"logloss={fm['log_loss'].mean():.4f}±{fm['log_loss'].std():.4f}")

    best_h = int(summary.sort_values("roc_auc", ascending=False).iloc[0]["horizon_hours"])
    res_best = results[best_h]
    print(f"\nTOP FEATURES — {best_h}h horizon (XGBoost gain, % of total)")
    print(feature_importance_table(res_best, top=20).to_string(index=False))
    try:
        print(f"\nPERMUTATION IMPORTANCE on test — {best_h}h (AUC drop when the feature is shuffled)")
        print(permutation_importance_table(res_best, n_repeats=4, top=15).round(4).to_string(index=False))
    except Exception as e:  # pragma: no cover
        log(f"  permutation importance skipped: {e}")

    if "per_symbol_test" in res_best:
        print(f"\nPER-SYMBOL BREAKDOWN on test — {best_h}h")
        print(res_best["per_symbol_test"].round(4).to_string(index=False))

    print(f"\nDECISION TABLE on test — {best_h}h "
          f"'act only when P(trending) >= threshold'")
    print(res_best["threshold_analysis"].round(4).to_string(index=False))

    # ---------------- plots ----------------
    log("STEP 6/6  plots & artefacts")
    for h_hours, res in sorted(results.items()):
        plot_regime_overlay(res, feats, cfg)
        plot_reliability(res, cfg)
        plot_prob_distribution(res, cfg)
        plot_learning_curve(res, cfg)
    plot_horizon_comparison(summary, cfg)

    # ---------------- live probabilities ----------------
    print("\n" + "=" * 78)
    print("LIVE REGIME PROBABILITIES (latest closed bar)")
    print("=" * 78)
    live = pd.concat([predict_regime({h: r}, feats, cfg, symbol=s)
                      for h, r in sorted(results.items()) for s in cfg.symbols
                      if s in feats], ignore_index=True)
    print(live.to_string(index=False))
    print("\n" + narrate(live))

    # ---------------- save ----------------
    summary.to_csv(os.path.join(cfg.out_dir, "summary_test_metrics.csv"), index=False)
    overfit.to_csv(os.path.join(cfg.out_dir, "overfitting_report.csv"), index=False)
    live.to_csv(os.path.join(cfg.out_dir, "live_probabilities.csv"), index=False)
    feature_importance_table(res_best, top=60).to_csv(
        os.path.join(cfg.out_dir, f"feature_importance_{best_h}h.csv"), index=False)
    for h_hours, res in sorted(results.items()):
        res["fold_metrics"].to_csv(os.path.join(cfg.out_dir, f"cv_folds_{h_hours}h.csv"), index=False)
        res["threshold_analysis"].to_csv(
            os.path.join(cfg.out_dir, f"decision_table_{h_hours}h.csv"), index=False)
        res["per_symbol_test"].to_csv(
            os.path.join(cfg.out_dir, f"per_symbol_{h_hours}h.csv"), index=False)
        res["model"].save_model(os.path.join(cfg.out_dir, f"xgb_regime_{h_hours}h.json"))
        json.dump({"horizon_hours": h_hours, "params": res["params"],
                   "n_estimators": res["n_estimators"], "thresholds": res["thresholds"],
                   "feature_names": res["feature_names"],
                   "test_metrics": res["test_metrics_cal"]},
                  open(os.path.join(cfg.out_dir, f"metadata_{h_hours}h.json"), "w"), indent=2)
    if cfg.verbose:
        print(HOW_TO_READ_RESULTS)
    log(f"done in {time.time() - t_start:.1f}s -> {cfg.out_dir}/")

    return {"results": results, "summary": summary, "overfitting": overfit,
            "live": live, "feats": feats, "df_map": df_map, "causality_audit": causality}


# %% [markdown]
# ## 14. Run it
#
# The default `CONFIG` downloads BTC/ETH/SOL 4h klines from 2019 and trains all five
# horizons. On a free Colab CPU runtime this takes roughly 5-12 minutes.
#
# Adjust the knobs below and re-run this cell to experiment.

# %%
def should_autorun() -> bool:
    """Run automatically in a notebook or as a script, but never when the module is imported
    (so `from crypto_regime_xgboost import run_horizon` stays instant and offline)."""
    return __name__ == "__main__" or "ipykernel" in sys.modules


if should_autorun():
    OUT = main(CONFIG)

# %% [markdown]
# ### Variants worth trying
#
# ```python
# # 1) stricter "trending" definition: only the top 35% most directional windows count
# cfg = Config(label_quantile=0.65, horizons_hours=(4, 8, 12, 16, 24))
# out = main(cfg)
#
# # 2) ignore flat windows (a move of at least the median size must happen)
# cfg = Config(min_abs_fwd_return_quantile=0.5)
#
# # 3) a single asset, so no symbol pooling
# cfg = Config(symbols=("BTCUSDT",), pool_symbols=False)
#
# # 4) more coins = more rows = usually a better model
# cfg = Config(symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
#                       "DOGEUSDT", "ADAUSDT", "LINKUSDT"), start="2020-01-01")
#
# # 5) no hyper-parameter search (much faster, slightly worse)
# cfg = Config(run_hpo=False)
# ```

# %%

# %%
if __name__ == "__main__":
    pass  # running as a script: `python crypto_regime_xgboost.py` executes main(CONFIG) above
