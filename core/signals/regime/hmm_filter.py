# core/signals/regime/hmm_filter.py
# D47 -- Regime filter: 3-state Gaussian HMM + ADX composite + interactions.
# Pure NumPy implementation (no hmmlearn dependency -- MSVC not available).
# ASCII-only output for cp1252 compatibility.
#
# Features produced:
#   HMM (3):  hmm_prob_bull, hmm_prob_bear, hmm_prob_calm
#   ADX (4):  adx_14, bb_width_normalized, atr_percentile_rank, regime_tag
#   Interaction (3): ob_bull_age_x_hmm_bull, fvg_bull_x_trending, ote_x_regime
#
# Total: 10 new features (adx_14 already in v3, so 9 truly new + 1 reused).

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Pure NumPy Gaussian HMM (1D, diagonal covariance)
# ---------------------------------------------------------------------------
class GaussianHMM1D:
    """Minimal 3-state Gaussian HMM for 1D observations (log-returns).

    Implements EM (Baum-Welch) for training and forward algorithm for
    causal state filtering. No external dependencies beyond NumPy.
    """

    def __init__(self, n_states: int = 3, n_iter: int = 50,
                 tol: float = 1e-4, random_state: int = 42):
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.rng = np.random.RandomState(random_state)

        # Parameters (initialized in fit())
        self.startprob_: np.ndarray | None = None
        self.transmat_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.vars_: np.ndarray | None = None

    def _init_params(self, X: np.ndarray) -> None:
        """K-means-ish initialization: sort data, split into n_states bins."""
        n = self.n_states
        self.startprob_ = np.ones(n) / n
        # Slight off-diagonal to avoid degenerate transitions
        self.transmat_ = np.full((n, n), 0.05 / (n - 1))
        np.fill_diagonal(self.transmat_, 0.95)

        # Initialize means by quantiles, vars by segment variance
        sorted_x = np.sort(X)
        splits = np.array_split(sorted_x, n)
        self.means_ = np.array([s.mean() for s in splits])
        self.vars_ = np.array([max(s.var(), 1e-10) for s in splits])

    def _log_emission(self, X: np.ndarray) -> np.ndarray:
        """Log P(x | state j) for each observation and state. Shape (T, n_states)."""
        T = len(X)
        n = self.n_states
        log_e = np.empty((T, n))
        for j in range(n):
            diff = X - self.means_[j]
            log_e[:, j] = -0.5 * (np.log(2 * np.pi * self.vars_[j])
                                   + diff ** 2 / self.vars_[j])
        return log_e

    def _forward(self, log_e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward algorithm. Returns (log_alpha, scales).

        log_alpha[t, j] = log P(x_0..x_t, s_t=j)
        Uses scaling to avoid underflow.
        Returns scaled alpha (not log) and scale factors.
        """
        T, n = log_e.shape
        alpha = np.empty((T, n))
        scales = np.empty(T)

        # t = 0
        alpha[0] = self.startprob_ * np.exp(log_e[0])
        scales[0] = alpha[0].sum()
        if scales[0] > 0:
            alpha[0] /= scales[0]
        else:
            alpha[0] = 1.0 / n
            scales[0] = 1e-300

        # t > 0
        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self.transmat_) * np.exp(log_e[t])
            scales[t] = alpha[t].sum()
            if scales[t] > 0:
                alpha[t] /= scales[t]
            else:
                alpha[t] = 1.0 / n
                scales[t] = 1e-300

        return alpha, scales

    def _backward(self, log_e: np.ndarray,
                  scales: np.ndarray) -> np.ndarray:
        """Backward algorithm (scaled). Returns beta."""
        T, n = log_e.shape
        beta = np.empty((T, n))
        beta[T - 1] = 1.0

        for t in range(T - 2, -1, -1):
            beta[t] = (self.transmat_ @ (np.exp(log_e[t + 1]) * beta[t + 1]))
            if scales[t + 1] > 0:
                beta[t] /= scales[t + 1]
            s = beta[t].sum()
            if s > 0:
                beta[t] /= s
                beta[t] *= n  # keep scale consistent

        return beta

    def fit(self, X: np.ndarray) -> "GaussianHMM1D":
        """Baum-Welch EM on 1D observations X (shape (T,))."""
        X = np.asarray(X, dtype=np.float64).ravel()
        T = len(X)
        if T < self.n_states + 1:
            return self

        if self.means_ is None:
            self._init_params(X)

        n = self.n_states
        prev_ll = -np.inf

        for iteration in range(self.n_iter):
            log_e = self._log_emission(X)
            alpha, scales = self._forward(log_e)
            beta = self._backward(log_e, scales)

            # Log-likelihood
            ll = np.sum(np.log(np.maximum(scales, 1e-300)))
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

            # E-step: posterior state probabilities gamma[t, j]
            gamma = alpha * beta
            gamma_sum = gamma.sum(axis=1, keepdims=True)
            gamma_sum = np.maximum(gamma_sum, 1e-300)
            gamma = gamma / gamma_sum

            # E-step: expected transitions xi[t, i, j]
            # xi[t, i, j] = alpha[t,i] * A[i,j] * e[t+1,j] * beta[t+1,j] / scale[t+1]
            xi_sum = np.zeros((n, n))
            for t in range(T - 1):
                numer = (alpha[t, :, None]
                         * self.transmat_
                         * (np.exp(log_e[t + 1]) * beta[t + 1])[None, :])
                denom = numer.sum()
                if denom > 0:
                    xi_sum += numer / denom

            # M-step
            self.startprob_ = gamma[0] / gamma[0].sum()

            trans_denom = gamma[:-1].sum(axis=0)
            for i in range(n):
                if trans_denom[i] > 0:
                    self.transmat_[i] = xi_sum[i] / trans_denom[i]
                else:
                    self.transmat_[i] = 1.0 / n
            # Normalize rows
            row_sums = self.transmat_.sum(axis=1, keepdims=True)
            self.transmat_ = self.transmat_ / np.maximum(row_sums, 1e-300)

            gamma_sum_states = gamma.sum(axis=0)
            for j in range(n):
                if gamma_sum_states[j] > 0:
                    self.means_[j] = (gamma[:, j] * X).sum() / gamma_sum_states[j]
                    diff = X - self.means_[j]
                    self.vars_[j] = ((gamma[:, j] * diff ** 2).sum()
                                     / gamma_sum_states[j])
                    self.vars_[j] = max(self.vars_[j], 1e-10)

        return self

    def filter_proba(self, X: np.ndarray) -> np.ndarray:
        """Causal forward filtering: P(state_t | x_0..x_t).

        Returns (T, n_states) array of state probabilities.
        Only uses past+current observations (no future lookahead).
        """
        X = np.asarray(X, dtype=np.float64).ravel()
        log_e = self._log_emission(X)
        alpha, _ = self._forward(log_e)
        # alpha is already normalized (scaled) -- each row sums to 1
        return alpha

    def label_states(self) -> dict[int, str]:
        """Assign semantic labels based on emission means.

        Returns {state_idx: 'bull'|'bear'|'calm'} mapping.
        Bull = highest mean, bear = lowest mean, calm = middle.
        """
        order = np.argsort(self.means_)
        labels = {}
        labels[order[0]] = "bear"
        labels[order[1]] = "calm"
        labels[order[2]] = "bull"
        return labels


# ---------------------------------------------------------------------------
# HMM regime probabilities (causal, walk-forward)
# ---------------------------------------------------------------------------
def compute_hmm_regime(
    df: pd.DataFrame,
    warmup_days: int = 252,
    retrain_days: int = 180,
) -> pd.DataFrame:
    """Compute causal HMM state probabilities on 5m data.

    Algorithm:
    1. Resample 5m close -> daily log-returns (using ONLY completed days)
    2. For each completed return d >= warmup_days:
       - Train HMM on rolling 1-year window (retrain every retrain_days)
       - Forward-filter to get P(state_d)
    3. Map daily probs to 5m bars via bar-index mapping (deterministic)

    Causality guarantee:
    - daily_returns[d] = log(close_{d+1}) - log(close_d)
    - close_d is at bar (d+1)*288-1 (last bar of completed day d)
    - We assign the filtered state from returns[0..d] to bars starting
      at (d+2)*288 (first bar of day d+2). This ensures that:
      close_{d+1} at bar (d+2)*288-1 is available before (d+2)*288.
    - The bar-to-day mapping uses absolute bar indices, so truncating
      the dataframe NEVER changes the value at earlier bars.

    Returns DataFrame with columns: hmm_prob_bull, hmm_prob_bear, hmm_prob_calm
    """
    close = df["close"].values.astype(np.float64)
    n_bars = len(close)
    bars_per_day = 288

    # Resample: use only COMPLETED days for daily close
    n_complete_days = n_bars // bars_per_day
    daily_close = np.array([
        close[(d + 1) * bars_per_day - 1] for d in range(n_complete_days)
    ])
    daily_returns = np.diff(np.log(daily_close))  # length n_complete_days - 1

    # Build bar-level probs array
    hmm_bull = np.full(n_bars, np.nan)
    hmm_bear = np.full(n_bars, np.nan)
    hmm_calm = np.full(n_bars, np.nan)

    model = GaussianHMM1D(n_states=3, n_iter=50, random_state=42)
    last_train_day = -1
    label_map = None

    for d in range(warmup_days, len(daily_returns)):
        # Retrain periodically
        if last_train_day < 0 or (d - last_train_day) >= retrain_days:
            train_start = max(0, d - 365)
            train_data = daily_returns[train_start:d]
            model = GaussianHMM1D(n_states=3, n_iter=50, random_state=42)
            model.fit(train_data)
            label_map = model.label_states()
            last_train_day = d

        # Forward-filter: causal P(state | returns[0..d])
        window_start = max(0, d - 60)
        window = daily_returns[window_start:d + 1]
        probs = model.filter_proba(window)
        day_prob = probs[-1]

        # Map state indices to semantic labels
        bull_idx = [k for k, v in label_map.items() if v == "bull"][0]
        bear_idx = [k for k, v in label_map.items() if v == "bear"][0]
        calm_idx = [k for k, v in label_map.items() if v == "calm"][0]

        # Assign to bars: returns[d] uses close at bar (d+2)*288-1.
        # Assign to bars starting at (d+2)*288 to ensure causality.
        bar_start = (d + 2) * bars_per_day
        bar_end = min((d + 3) * bars_per_day, n_bars)
        if bar_start < n_bars:
            hmm_bull[bar_start:bar_end] = day_prob[bull_idx]
            hmm_bear[bar_start:bar_end] = day_prob[bear_idx]
            hmm_calm[bar_start:bar_end] = day_prob[calm_idx]

    # Forward-fill any remaining bars after the last assigned day
    # (covers partial last day + any gap at the end)
    last_assigned = -1
    for i in range(n_bars - 1, -1, -1):
        if not np.isnan(hmm_bull[i]):
            last_assigned = i
            break
    if last_assigned >= 0 and last_assigned < n_bars - 1:
        hmm_bull[last_assigned + 1:] = hmm_bull[last_assigned]
        hmm_bear[last_assigned + 1:] = hmm_bear[last_assigned]
        hmm_calm[last_assigned + 1:] = hmm_calm[last_assigned]

    return pd.DataFrame({
        "hmm_prob_bull": hmm_bull,
        "hmm_prob_bear": hmm_bear,
        "hmm_prob_calm": hmm_calm,
    }, index=df.index)


# ---------------------------------------------------------------------------
# ADX composite features
# ---------------------------------------------------------------------------
def compute_adx_composite(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ADX composite regime features.

    Uses existing v3 columns where available (adx_14, bb_bandwidth),
    computes atr_percentile_rank and regime_tag fresh.

    Returns DataFrame with columns:
      adx_14             -- reused from v3 if present
      bb_width_normalized -- bb_bandwidth / close (normalized)
      atr_percentile_rank -- ATR rank over rolling 288-bar window (0-1)
      regime_tag          -- 0=ranging, 1=neutral, 2=trending (int-encoded)
    """
    out = {}

    # adx_14: reuse from v3 if available
    if "adx_14" in df.columns:
        adx = df["adx_14"].values.astype(np.float64)
    else:
        adx = _compute_adx(df, period=14)
    out["adx_14"] = adx

    # bb_width_normalized: Bollinger Band width / price
    if "bb_bandwidth" in df.columns:
        bb_bw = df["bb_bandwidth"].values.astype(np.float64)
        close = df["close"].values.astype(np.float64)
        bb_norm = bb_bw / np.maximum(close, 1e-10)
    else:
        close = df["close"].values.astype(np.float64)
        sma20 = _rolling_mean(close, 20)
        std20 = _rolling_std(close, 20)
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        bb_norm = (bb_upper - bb_lower) / np.maximum(close, 1e-10)
    out["bb_width_normalized"] = bb_norm

    # atr_percentile_rank: ATR rank over rolling 288-bar window
    if "ict_atr_14" in df.columns:
        atr = df["ict_atr_14"].values.astype(np.float64)
    else:
        atr = _compute_atr(df, period=14)
    out["atr_percentile_rank"] = _rolling_percentile_rank(atr, window=288)

    # regime_tag: 0=ranging, 1=neutral, 2=trending
    # "trending" if ADX>25 and bb_width > 75th percentile
    # "ranging" if ADX<20
    # "neutral" otherwise
    bb_75th = _rolling_quantile(bb_norm, window=288, q=0.75)
    regime = np.ones(len(df), dtype=np.float64)  # default: neutral (1)
    trending = (adx > 25) & (bb_norm > bb_75th)
    ranging = adx < 20
    regime[trending] = 2.0
    regime[ranging] = 0.0
    # NaN where inputs are NaN
    nan_mask = np.isnan(adx) | np.isnan(bb_norm)
    regime[nan_mask] = np.nan
    out["regime_tag"] = regime

    return pd.DataFrame(out, index=df.index)


# ---------------------------------------------------------------------------
# Interaction features
# ---------------------------------------------------------------------------
def compute_regime_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Compute interaction features between ICT signals and regime.

    Requires hmm_prob_bull, adx_14 to already exist in df.

    Returns DataFrame with columns:
      ob_bull_age_x_hmm_bull  -- ict_ob_bull_age * hmm_prob_bull
      fvg_bull_x_trending     -- (fvg_bull_recent_age not NaN) * (adx_14 > 25)
      ote_x_regime            -- ote_dist * hmm_prob_bull
    """
    out = {}

    # ob_bull_age * hmm_prob_bull
    ob_age = df["ict_ob_bull_age"].values.astype(np.float64)
    hmm_bull = df["hmm_prob_bull"].values.astype(np.float64)
    out["ob_bull_age_x_hmm_bull"] = ob_age * hmm_bull

    # fvg_bull * trending: (fvg_bull_recent_age is not NaN) * (adx_14 > 25)
    fvg_age = df["ict_fvg_bull_recent_age"].values.astype(np.float64)
    fvg_active = (~np.isnan(fvg_age)).astype(np.float64)
    adx = df["adx_14"].values.astype(np.float64)
    adx_trending = (adx > 25).astype(np.float64)
    # NaN where adx is NaN
    adx_trending[np.isnan(adx)] = np.nan
    out["fvg_bull_x_trending"] = fvg_active * adx_trending

    # ote_dist * hmm_prob_bull
    ote = df["ote_dist"].values.astype(np.float64)
    out["ote_x_regime"] = ote * hmm_bull

    return pd.DataFrame(out, index=df.index)


# ---------------------------------------------------------------------------
# Master function: compute all regime features
# ---------------------------------------------------------------------------
def compute_all_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all regime features (HMM + ADX composite + interactions).

    Returns DataFrame with 10 columns (9 new + adx_14 which may already exist).
    """
    print("  Regime: computing HMM probabilities ...")
    hmm = compute_hmm_regime(df)
    for col in hmm.columns:
        df[col] = hmm[col].values

    print("  Regime: computing ADX composite ...")
    adx = compute_adx_composite(df)
    for col in adx.columns:
        if col not in df.columns:
            df[col] = adx[col].values

    print("  Regime: computing interaction features ...")
    interact = compute_regime_interactions(df)
    for col in interact.columns:
        df[col] = interact[col].values

    # Collect all output columns
    all_cols = list(hmm.columns) + list(adx.columns) + list(interact.columns)
    # Deduplicate (adx_14 might already be in df)
    unique_cols = list(dict.fromkeys(all_cols))
    result = df[unique_cols].copy()
    return result


# ---------------------------------------------------------------------------
# Helper: rolling statistics (pandas rolling for C-backed speed)
# ---------------------------------------------------------------------------
def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean with NaN fill for warmup."""
    return pd.Series(arr).rolling(window, min_periods=window).mean().values


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling std with NaN fill for warmup."""
    return pd.Series(arr).rolling(window, min_periods=window).std(ddof=1).values


def _rolling_percentile_rank(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling percentile rank (0-1) over a window."""
    s = pd.Series(arr)
    return s.rolling(window, min_periods=window).rank(pct=True).values


def _rolling_quantile(arr: np.ndarray, window: int, q: float) -> np.ndarray:
    """Rolling quantile over a window."""
    return pd.Series(arr).rolling(window, min_periods=window).quantile(q).values


def _compute_atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """Compute ATR if not already in dataframe."""
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    close = df["close"].values.astype(np.float64)
    prev_close = np.concatenate(([close[0]], close[:-1]))
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - prev_close),
                               np.abs(low - prev_close)))
    atr = np.full_like(tr, np.nan)
    atr[period - 1] = tr[:period].mean()
    alpha = 1.0 / period
    for i in range(period, len(tr)):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]
    return atr


def _compute_adx(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """Compute ADX from OHLC data."""
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    close = df["close"].values.astype(np.float64)

    n = len(high)
    up_move = np.diff(high, prepend=high[0])
    down_move = np.diff(low, prepend=low[0]) * -1

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    prev_close = np.concatenate(([close[0]], close[:-1]))
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - prev_close),
                               np.abs(low - prev_close)))

    # Wilder smoothing
    alpha = 1.0 / period
    atr_s = np.full(n, np.nan)
    plus_s = np.full(n, np.nan)
    minus_s = np.full(n, np.nan)

    atr_s[period - 1] = tr[:period].mean()
    plus_s[period - 1] = plus_dm[:period].mean()
    minus_s[period - 1] = minus_dm[:period].mean()

    for i in range(period, n):
        atr_s[i] = alpha * tr[i] + (1 - alpha) * atr_s[i - 1]
        plus_s[i] = alpha * plus_dm[i] + (1 - alpha) * plus_s[i - 1]
        minus_s[i] = alpha * minus_dm[i] + (1 - alpha) * minus_s[i - 1]

    plus_di = 100 * plus_s / np.maximum(atr_s, 1e-10)
    minus_di = 100 * minus_s / np.maximum(atr_s, 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / np.maximum(plus_di + minus_di, 1e-10)

    adx = np.full(n, np.nan)
    start = 2 * period - 1
    if start < n:
        adx[start] = np.nanmean(dx[period:start + 1])
        for i in range(start + 1, n):
            if not np.isnan(adx[i - 1]):
                adx[i] = alpha * dx[i] + (1 - alpha) * adx[i - 1]

    return adx
