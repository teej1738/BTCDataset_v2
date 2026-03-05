"""D55b Holdout -- Supplementary Analysis.

Computes Lo (2002) autocorrelation-adjusted daily Sharpe and BTC trend context.
This is NOT a re-evaluation -- same seed, same config, same results.
Just computing additional diagnostics on the existing single holdout run.

Run from project root:
    python core/experiments/run_d55b_lo_supplement.py
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.engine.evaluator import (
    compute_auc,
    compute_ece,
    simulate,
    build_trade_returns,
)
from core.engine.sizing import kelly_fraction_array
from core.engine.calibrator import isotonic_calibrate
from core.engine.simulator import augment_features, select_features

# Paths
TRAIN_PATH = PROJECT_ROOT / "data" / "labeled" / "BTCUSDT_5m_labeled_v3_train.parquet"
HOLDOUT_PATH = PROJECT_ROOT / "data" / "holdout" / "BTCUSDT_5m_holdout_v3.parquet"
REGISTRY_PATH = PROJECT_ROOT / "core" / "experiments" / "registry.json"
D55B_OOS_PROBS_PATH = PROJECT_ROOT / "core" / "experiments" / "models" / "D55b_tier1_only_oos_probs.npy"
RESULTS_PATH = PROJECT_ROOT / "core" / "experiments" / "results" / "d55b_holdout.json"

# Config (identical to holdout ceremony)
SEED = 42
LABEL_COL = "label_long_hit_2r_48c"
THRESHOLD = 0.60
COOLDOWN = 576
R_TARGET = 2
COST_PER_R = 0.05
KELLY_DIVISOR = 40.0
KELLY_ODDS = 2.0
INITIAL_EQUITY = 10_000.0


def load_d55b_feature_exclude():
    with open(REGISTRY_PATH) as f:
        reg = json.load(f)
    for exp in reg["experiments"]:
        if exp["id"] == "D55b_tier1_only":
            return exp.get("feature_exclude", [])
    raise ValueError("D55b_tier1_only not found in registry")


def train_final_model(X_train, y_train, device="gpu"):
    """Identical to holdout ceremony -- same seed, same result."""
    import lightgbm as lgb

    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.01,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbosity": -1,
        "is_unbalance": True,
        "seed": SEED,
        "device": device,
    }

    n = len(X_train)
    val_size = max(int(n * 0.10), 1000)
    val_start = n - val_size

    X_tr = X_train.iloc[:val_start]
    y_tr = y_train[:val_start]
    X_val = X_train.iloc[val_start:]
    y_val = y_train[val_start:]

    dtrain = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
    dval = lgb.Dataset(X_val, label=y_val, free_raw_data=False, reference=dtrain)

    # GPU probe
    if device == "gpu":
        try:
            _probe = lgb.Dataset(X_train.iloc[:100], label=y_train[:100], free_raw_data=False)
            _m = lgb.train({**lgb_params, "num_iterations": 1, "verbosity": -1}, _probe)
            del _m, _probe
            print("  Device: gpu")
        except Exception:
            lgb_params["device"] = "cpu"
            print("  Device: cpu (fallback)")
    else:
        print("  Device: cpu")

    callbacks = [
        lgb.early_stopping(50, verbose=False),
        lgb.log_evaluation(period=200),
    ]

    model = lgb.train(
        lgb_params, dtrain,
        num_boost_round=1000,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )
    print(f"  Best iteration: {model.best_iteration}")
    return model


def compute_lo_2002_adjustment(daily_pnl, max_lag=10):
    """Lo (2002) autocorrelation adjustment for Sharpe ratio.

    The naive Sharpe ratio assumes i.i.d. returns. If daily returns are
    autocorrelated, the naive Sharpe overstates (positive AC) or understates
    (negative AC) the true risk-adjusted return.

    Lo (2002) adjustment factor:
        q = 1 + 2 * sum_{k=1}^{q} (1 - k/(q+1)) * rho_k
        SR_adjusted = SR_naive / sqrt(q)

    where rho_k is the autocorrelation at lag k.

    Returns dict with autocorrelations, adjustment factor, and adjusted Sharpe.
    """
    n = len(daily_pnl)
    if n < max_lag + 2:
        return {"error": "insufficient data", "n_days": n}

    mean_pnl = daily_pnl.mean()
    centered = daily_pnl - mean_pnl
    var = np.sum(centered ** 2) / n  # biased variance for AC computation

    if var == 0:
        return {"error": "zero variance", "n_days": n}

    # Compute autocorrelations at lags 1..max_lag
    autocorrs = {}
    for lag in range(1, max_lag + 1):
        ac = np.sum(centered[lag:] * centered[:-lag]) / (n * var)
        autocorrs[lag] = float(ac)

    # Lo (2002) adjustment factor (using Newey-West-style kernel)
    q_val = max_lag
    adjustment_sum = 0.0
    for k in range(1, q_val + 1):
        weight = 1.0 - k / (q_val + 1)
        adjustment_sum += weight * autocorrs[k]

    q_factor = 1.0 + 2.0 * adjustment_sum

    # Naive daily Sharpe
    std_pnl = daily_pnl.std(ddof=1)
    if std_pnl == 0:
        return {"error": "zero std", "n_days": n}
    naive_daily_sr = mean_pnl / std_pnl

    # Check significance of autocorrelations (Bartlett's formula: SE ~ 1/sqrt(n))
    se_ac = 1.0 / np.sqrt(n)
    significant_lags = []
    for lag, ac in autocorrs.items():
        if abs(ac) > 1.96 * se_ac:  # 95% CI
            significant_lags.append(lag)

    # Adjusted Sharpe
    if q_factor > 0:
        adjusted_daily_sr = naive_daily_sr / np.sqrt(q_factor)
    else:
        # Negative q_factor means extreme negative AC -- use naive
        adjusted_daily_sr = naive_daily_sr
        q_factor = 1.0  # fallback

    # Annualize both
    naive_annual_sr = naive_daily_sr * np.sqrt(365)
    adjusted_annual_sr = adjusted_daily_sr * np.sqrt(365)

    return {
        "n_days": n,
        "autocorrelations": autocorrs,
        "significant_lags": significant_lags,
        "bartlett_se": float(se_ac),
        "q_factor": float(q_factor),
        "naive_daily_sr": float(naive_daily_sr),
        "adjusted_daily_sr": float(adjusted_daily_sr),
        "naive_annual_sr": float(naive_annual_sr),
        "adjusted_annual_sr": float(adjusted_annual_sr),
        "adjustment_pct": float((1.0 - adjusted_annual_sr / naive_annual_sr) * 100)
            if naive_annual_sr != 0 else 0.0,
    }


def compute_btc_trend_context(df_hold):
    """Compute BTC price trend during holdout period."""
    close = df_hold["close"].values
    ts = df_hold["bar_start_ts_utc"]

    first_close = close[0]
    last_close = close[-1]
    max_close = close.max()
    min_close = close.min()
    pct_change = (last_close - first_close) / first_close * 100

    # 200-bar SMA (200 * 5min = ~16.7h)
    sma_200 = pd.Series(close).rolling(200).mean().values
    valid_sma = ~np.isnan(sma_200)
    above_sma_pct = (close[valid_sma] > sma_200[valid_sma]).mean() * 100

    # Daily returns for annualized metrics
    daily_close = df_hold.groupby(ts.dt.date)["close"].last()
    daily_log_ret = np.log(daily_close / daily_close.shift(1)).dropna()
    ann_return = daily_log_ret.mean() * 365 * 100
    ann_vol = daily_log_ret.std() * np.sqrt(365) * 100

    # Classify regime
    if pct_change > 20:
        regime = "BULL"
    elif pct_change < -20:
        regime = "BEAR"
    else:
        regime = "SIDEWAYS/CHOPPY"

    return {
        "first_close": float(first_close),
        "last_close": float(last_close),
        "change_pct": float(pct_change),
        "max_close": float(max_close),
        "min_close": float(min_close),
        "above_200sma_pct": float(above_sma_pct),
        "ann_return_pct": float(ann_return),
        "ann_vol_pct": float(ann_vol),
        "regime": regime,
        "start_date": str(ts.iloc[0].date()),
        "end_date": str(ts.iloc[-1].date()),
    }


def main():
    t0 = time.time()
    warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")

    print("=" * 60)
    print("  D55b Holdout -- Lo(2002) & BTC Trend Supplement")
    print("=" * 60)

    # Load existing results
    with open(RESULTS_PATH) as f:
        existing = json.load(f)
    print(f"\n  Existing verdict: {existing['verdict']}")
    print(f"  Existing daily Sharpe: {existing['holdout_metrics']['daily_sharpe']:.4f}")

    # Load feature exclude
    feature_exclude = load_d55b_feature_exclude()

    # Load and augment training data
    print("\n--- Loading Training Data ---")
    df_train = pd.read_parquet(TRAIN_PATH)
    df_train = augment_features(df_train)

    # Load and augment holdout data
    print("\n--- Loading Holdout Data ---")
    df_hold = pd.read_parquet(HOLDOUT_PATH)
    df_hold = augment_features(df_hold)

    # Select features
    features = select_features(df_train, "all", feature_exclude)
    missing_in_hold = [f for f in features if f not in df_hold.columns]
    if missing_in_hold:
        features = [f for f in features if f in df_hold.columns]
    print(f"  Features: {len(features)}")

    # Prepare training data
    raw_train_labels = df_train[LABEL_COL].values
    train_valid = ~np.isnan(raw_train_labels)
    train_labels = np.where(train_valid, raw_train_labels, 0).astype(int)
    valid_idx = np.where(train_valid)[0]
    X_train = df_train[features].iloc[valid_idx]
    y_train = train_labels[valid_idx]

    # Train model (same seed = same model)
    print("\n--- Training (same seed=42, reproducing identical model) ---")
    model = train_final_model(X_train, y_train, device="gpu")

    # Predict on holdout
    X_hold = df_hold[features]
    raw_hold_probs = model.predict(X_hold)

    # Calibrate
    d55b_oos_probs = np.load(D55B_OOS_PROBS_PATH)
    cal_valid = ~np.isnan(d55b_oos_probs) & train_valid
    cal_train_probs = d55b_oos_probs[cal_valid]
    cal_train_labels = train_labels[cal_valid]
    hold_probs_cal = isotonic_calibrate(cal_train_probs, cal_train_labels, raw_hold_probs)

    # Holdout labels
    raw_hold_labels = df_hold[LABEL_COL].values
    hold_valid = ~np.isnan(raw_hold_labels)
    hold_labels = np.where(hold_valid, raw_hold_labels, 0).astype(int)

    # Verify AUC matches (confirms identical model)
    auc_cal = compute_auc(hold_labels[hold_valid], hold_probs_cal[hold_valid])
    print(f"\n  Verification AUC: {auc_cal:.4f} (expected: {existing['holdout_metrics']['auc']:.4f})")
    if abs(auc_cal - existing["holdout_metrics"]["auc"]) > 0.001:
        print("  WARNING: AUC mismatch > 0.001 -- possible GPU non-determinism")

    # Simulate trades
    signal_mask = hold_valid & (hold_probs_cal >= THRESHOLD)
    trade_indices = simulate(signal_mask, hold_labels, COOLDOWN)
    r_returns = build_trade_returns(trade_indices, hold_labels, R_TARGET, COST_PER_R)
    print(f"  Trades: {len(trade_indices)} (expected: {existing['holdout_metrics']['n_trades']})")

    # -------------------------------------------------------------------
    # DAILY P&L
    # -------------------------------------------------------------------
    ts = df_hold["bar_start_ts_utc"]
    trade_dates = ts.iloc[trade_indices].dt.date.values

    all_dates = pd.date_range(
        start=ts.iloc[0].date(),
        end=ts.iloc[-1].date(),
        freq="D",
    )
    daily_pnl = np.zeros(len(all_dates))
    date_to_idx = {d.date(): i for i, d in enumerate(all_dates)}

    for trade_date, r_ret in zip(trade_dates, r_returns):
        idx = date_to_idx.get(trade_date)
        if idx is not None:
            daily_pnl[idx] += r_ret

    # -------------------------------------------------------------------
    # LO (2002) ADJUSTMENT
    # -------------------------------------------------------------------
    print("\n--- Lo (2002) Autocorrelation Adjustment ---")
    lo_result = compute_lo_2002_adjustment(daily_pnl, max_lag=10)

    if "error" not in lo_result:
        print(f"  Days in series: {lo_result['n_days']}")
        print(f"  Bartlett SE: {lo_result['bartlett_se']:.4f}")
        print()
        print("  Autocorrelations (lags 1-10):")
        for lag, ac in lo_result["autocorrelations"].items():
            sig = " *" if lag in lo_result["significant_lags"] else ""
            print(f"    Lag {lag:2d}: {ac:+.4f}{sig}")
        print()
        print(f"  Significant lags (p<0.05): {lo_result['significant_lags'] or 'NONE'}")
        print(f"  Lo q-factor: {lo_result['q_factor']:.4f}")
        print()
        print(f"  Naive daily Sharpe:    {lo_result['naive_daily_sr']:.4f}")
        print(f"  Lo-adj daily Sharpe:   {lo_result['adjusted_daily_sr']:.4f}")
        print(f"  Naive annual Sharpe:   {lo_result['naive_annual_sr']:.4f}")
        print(f"  Lo-adj annual Sharpe:  {lo_result['adjusted_annual_sr']:.4f}")
        print(f"  Adjustment: {lo_result['adjustment_pct']:+.1f}%")
    else:
        print(f"  ERROR: {lo_result['error']}")

    # -------------------------------------------------------------------
    # BTC TREND CONTEXT
    # -------------------------------------------------------------------
    print("\n--- BTC Trend Context (Holdout Period) ---")
    trend = compute_btc_trend_context(df_hold)
    print(f"  Period: {trend['start_date']} to {trend['end_date']}")
    print(f"  First close: ${trend['first_close']:,.0f}")
    print(f"  Last close:  ${trend['last_close']:,.0f}")
    print(f"  Change:      {trend['change_pct']:+.1f}%")
    print(f"  Max:         ${trend['max_close']:,.0f}")
    print(f"  Min:         ${trend['min_close']:,.0f}")
    print(f"  Above 200-bar SMA: {trend['above_200sma_pct']:.1f}%")
    print(f"  Ann. return: {trend['ann_return_pct']:+.1f}%")
    print(f"  Ann. vol:    {trend['ann_vol_pct']:.1f}%")
    print(f"  Regime: {trend['regime']}")

    # -------------------------------------------------------------------
    # UPDATE RESULTS JSON
    # -------------------------------------------------------------------
    existing["holdout_metrics"]["lo_2002"] = lo_result if "error" not in lo_result else {"error": lo_result["error"]}
    existing["holdout_metrics"]["btc_trend"] = trend

    # Add Lo-adjusted Sharpe to top level
    if "error" not in lo_result:
        existing["holdout_metrics"]["daily_sharpe_lo_adj"] = lo_result["adjusted_annual_sr"]
        existing["holdout_metrics"]["lo_q_factor"] = lo_result["q_factor"]

    with open(RESULTS_PATH, "w") as f:
        json.dump(existing, f, indent=2, default=str)
    print(f"\n  Updated: {RESULTS_PATH.name}")

    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed:.1f}s")
    print("=" * 60)

    return lo_result, trend


if __name__ == "__main__":
    lo, trend = main()
