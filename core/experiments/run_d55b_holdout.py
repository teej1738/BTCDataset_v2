"""D55b Holdout Evaluation -- ONE-TIME CEREMONY.

Train a SINGLE final LightGBM model on ALL 543K training rows.
Evaluate on the 105,121-row holdout (2025-03 to 2026-02).
Calibrate via isotonic regression (fit on D55b walk-forward OOS probs).

Run from project root:
    python core/experiments/run_d55b_holdout.py

Rules:
  - ONE run only. No parameter adjustment after seeing results.
  - Seed 42.
  - Report ALL metrics.
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
    compute_metrics,
    simulate,
    build_trade_returns,
)
from core.engine.sizing import equity_sim_variable, kelly_fraction_array
from core.engine.calibrator import isotonic_calibrate
from core.engine.simulator import augment_features, select_features, META_COLS, LABEL_PREFIX

# Paths
TRAIN_PATH = PROJECT_ROOT / "data" / "labeled" / "BTCUSDT_5m_labeled_v3_train.parquet"
HOLDOUT_PATH = PROJECT_ROOT / "data" / "holdout" / "BTCUSDT_5m_holdout_v3.parquet"
REGISTRY_PATH = PROJECT_ROOT / "core" / "experiments" / "registry.json"
D55B_OOS_PROBS_PATH = PROJECT_ROOT / "core" / "experiments" / "models" / "D55b_tier1_only_oos_probs.npy"
RESULTS_DIR = PROJECT_ROOT / "core" / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Config (from D55b registry entry)
SEED = 42
LABEL_COL = "label_long_hit_2r_48c"
THRESHOLD = 0.60
COOLDOWN = 576
R_TARGET = 2
COST_PER_R = 0.05
KELLY_DIVISOR = 40.0
KELLY_ODDS = 2.0
INITIAL_EQUITY = 10_000.0

# Holdout gates (pre-registered)
HOLDOUT_GATES = {
    "AUC": {"threshold": 0.780, "op": ">="},
    "WIN_RATE": {"threshold": 0.60, "op": ">="},
    "EV_R": {"threshold": 0.80, "op": ">="},
    "TRADES": {"threshold": 50, "op": ">="},
    "DAILY_SHARPE": {"threshold": 0.0, "op": ">"},
}


def load_d55b_feature_exclude():
    """Load the feature_exclude list from registry."""
    with open(REGISTRY_PATH) as f:
        reg = json.load(f)
    for exp in reg["experiments"]:
        if exp["id"] == "D55b_tier1_only":
            return exp.get("feature_exclude", [])
    raise ValueError("D55b_tier1_only not found in registry")


def train_final_model(X_train, y_train, device="gpu"):
    """Train a single LightGBM model on ALL training data."""
    import lightgbm as lgb

    # Same hyperparameters as walk-forward training in evaluator.py
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
    # 10% validation split (last 10% of training data, chronologically)
    val_size = max(int(n * 0.10), 1000)
    val_start = n - val_size

    X_tr = X_train.iloc[:val_start]
    y_tr = y_train[:val_start]
    X_val = X_train.iloc[val_start:]
    y_val = y_train[val_start:]

    print(f"  Train: {len(X_tr):,} rows")
    print(f"  Val:   {len(X_val):,} rows (last 10% for early stopping)")

    dtrain = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
    dval = lgb.Dataset(X_val, label=y_val, free_raw_data=False, reference=dtrain)

    # GPU probe
    gpu_ok = True
    if device == "gpu":
        try:
            _probe = lgb.Dataset(
                X_train.iloc[:100], label=y_train[:100], free_raw_data=False
            )
            _m = lgb.train(
                {**lgb_params, "num_iterations": 1, "verbosity": -1},
                _probe,
            )
            del _m, _probe
            print("  Training device: gpu")
        except Exception:
            gpu_ok = False
            lgb_params["device"] = "cpu"
            print("  Training device: cpu (fallback)")
    else:
        print("  Training device: cpu")

    callbacks = [
        lgb.early_stopping(50, verbose=False),
        lgb.log_evaluation(period=100),
    ]

    model = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    best_iter = model.best_iteration
    if best_iter <= 0:
        best_iter = 1000
    print(f"  Best iteration: {best_iter}")

    return model


def compute_daily_sharpe(trade_indices, r_returns, timestamps):
    """Compute daily Sharpe ratio from trade-level returns.

    Groups trades by calendar day and computes daily PnL.
    Days without trades have 0 PnL.
    Returns (daily_sharpe_annualized, daily_returns_array).
    """
    if len(trade_indices) == 0 or len(r_returns) == 0:
        return 0.0, np.array([])

    # Map each trade to its date
    trade_dates = timestamps.iloc[trade_indices].dt.date.values

    # Build daily returns (including zero-trade days)
    all_dates = pd.date_range(
        start=timestamps.iloc[0].date(),
        end=timestamps.iloc[-1].date(),
        freq="D",
    )

    daily_pnl = np.zeros(len(all_dates))
    date_to_idx = {d.date(): i for i, d in enumerate(all_dates)}

    for trade_date, r_ret in zip(trade_dates, r_returns):
        idx = date_to_idx.get(trade_date)
        if idx is not None:
            daily_pnl[idx] += r_ret

    # Daily Sharpe: mean / std * sqrt(365)
    if daily_pnl.std() == 0:
        return 0.0, daily_pnl

    daily_sr = daily_pnl.mean() / daily_pnl.std(ddof=1)
    annual_daily_sr = daily_sr * np.sqrt(365)
    return float(annual_daily_sr), daily_pnl


def compute_per_bar_sharpe(r_returns, n_bars, bars_per_year):
    """Compute per-bar annualized Sharpe (the inflated number for sparse traders)."""
    if len(r_returns) == 0:
        return 0.0
    mean_r = np.mean(r_returns)
    std_r = np.std(r_returns, ddof=1)
    if std_r == 0:
        return 0.0
    sr_pt = mean_r / std_r
    tpy = len(r_returns) / (n_bars / bars_per_year)
    return float(sr_pt * np.sqrt(tpy))


def evaluate_gates(metrics_dict):
    """Evaluate pre-registered holdout gates."""
    gate_results = {}
    for name, gate in HOLDOUT_GATES.items():
        if name == "AUC":
            value = metrics_dict["auc"]
        elif name == "WIN_RATE":
            value = metrics_dict["win_rate"]
        elif name == "EV_R":
            value = metrics_dict["ev_r"]
        elif name == "TRADES":
            value = metrics_dict["n_trades"]
        elif name == "DAILY_SHARPE":
            value = metrics_dict["daily_sharpe"]
        else:
            continue

        threshold = gate["threshold"]
        op = gate["op"]
        if op == ">=":
            passed = value >= threshold
        elif op == ">":
            passed = value > threshold
        elif op == "<=":
            passed = value <= threshold
        else:
            passed = False

        gate_results[name] = {
            "value": round(value, 4) if isinstance(value, float) else value,
            "threshold": threshold,
            "op": op,
            "pass": bool(passed),
        }
    return gate_results


def apply_decision_tree(gate_results, metrics_dict):
    """Apply pre-registered decision tree.

    - CONFIRMED CLEAN: All 5 gates pass AND AUC >= 0.78
    - MINOR DEGRADATION: All gates pass but Daily Sharpe < 1.0 or WR < 65%
    - INVESTIGATE: 1-2 gates fail marginally (within 10% of threshold)
    - FAIL: 2+ gates fail OR AUC < 0.72 OR EV < 0
    """
    n_pass = sum(1 for g in gate_results.values() if g["pass"])
    n_fail = len(gate_results) - n_pass
    auc = metrics_dict["auc"]
    ev = metrics_dict["ev_r"]
    daily_sr = metrics_dict["daily_sharpe"]
    wr = metrics_dict["win_rate"]

    # FAIL conditions
    if n_fail >= 2 or auc < 0.72 or ev < 0:
        return "FAIL"

    # INVESTIGATE: 1-2 gates fail but marginally
    if n_fail >= 1:
        return "INVESTIGATE"

    # All gates pass
    if auc >= 0.78:
        # Check for minor degradation signals
        if daily_sr < 1.0 or wr < 0.65:
            return "MINOR DEGRADATION"
        return "CONFIRMED CLEAN"

    return "MINOR DEGRADATION"


def main():
    t0 = time.time()
    warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")

    print("=" * 60)
    print("  D55b HOLDOUT EVALUATION -- ONE-TIME CEREMONY")
    print("=" * 60)
    print()

    # -----------------------------------------------------------------------
    # 1. Load feature exclude list
    # -----------------------------------------------------------------------
    feature_exclude = load_d55b_feature_exclude()
    print(f"Feature exclude: {len(feature_exclude)} features (Tier 2 + Tier 3)")

    # -----------------------------------------------------------------------
    # 2. Load and augment TRAINING data
    # -----------------------------------------------------------------------
    print("\n--- Loading Training Data ---")
    print(f"  Path: {TRAIN_PATH.name}")
    df_train = pd.read_parquet(TRAIN_PATH)
    print(f"  Shape: {df_train.shape[0]:,} x {df_train.shape[1]}")

    print("  Augmenting features (on-the-fly) ...")
    df_train = augment_features(df_train)
    print(f"  Shape after augment: {df_train.shape[0]:,} x {df_train.shape[1]}")

    # -----------------------------------------------------------------------
    # 3. Load and augment HOLDOUT data
    # -----------------------------------------------------------------------
    print("\n--- Loading Holdout Data ---")
    print(f"  Path: {HOLDOUT_PATH.name}")
    # Direct load -- bypasses load_data() guard intentionally for ceremony
    df_hold = pd.read_parquet(HOLDOUT_PATH)
    print(f"  Shape: {df_hold.shape[0]:,} x {df_hold.shape[1]}")

    print("  Augmenting features (on-the-fly) ...")
    df_hold = augment_features(df_hold)
    print(f"  Shape after augment: {df_hold.shape[0]:,} x {df_hold.shape[1]}")

    # -----------------------------------------------------------------------
    # 4. Select features (same logic as D55b)
    # -----------------------------------------------------------------------
    features = select_features(df_train, "all", feature_exclude)
    print(f"  Final feature count: {len(features)}")

    # Verify same features exist in holdout
    missing_in_hold = [f for f in features if f not in df_hold.columns]
    if missing_in_hold:
        print(f"  WARNING: {len(missing_in_hold)} features missing in holdout!")
        for f in missing_in_hold[:10]:
            print(f"    - {f}")
        # Remove missing features
        features = [f for f in features if f in df_hold.columns]
        print(f"  Adjusted feature count: {len(features)}")

    # -----------------------------------------------------------------------
    # 5. Prepare training labels
    # -----------------------------------------------------------------------
    raw_train_labels = df_train[LABEL_COL].values
    train_valid = ~np.isnan(raw_train_labels)
    train_labels = np.where(train_valid, raw_train_labels, 0).astype(int)

    # Filter to valid-label rows for training
    valid_idx = np.where(train_valid)[0]
    X_train = df_train[features].iloc[valid_idx]
    y_train = train_labels[valid_idx]

    print(f"\n  Training rows (valid labels): {len(valid_idx):,}")
    print(f"  Positive rate: {y_train.mean():.4f}")

    # -----------------------------------------------------------------------
    # 6. Train SINGLE final model
    # -----------------------------------------------------------------------
    print("\n--- Training Final Model ---")
    model = train_final_model(X_train, y_train, device="gpu")

    # -----------------------------------------------------------------------
    # 7. Predict on HOLDOUT
    # -----------------------------------------------------------------------
    print("\n--- Holdout Prediction ---")
    X_hold = df_hold[features]
    raw_hold_probs = model.predict(X_hold)
    print(f"  Holdout predictions: {len(raw_hold_probs):,}")
    print(f"  Mean prob: {raw_hold_probs.mean():.4f}")
    print(f"  Median prob: {np.median(raw_hold_probs):.4f}")
    print(f"  Prob >= 0.60: {(raw_hold_probs >= 0.60).sum():,} bars")

    # -----------------------------------------------------------------------
    # 8. Calibrate using D55b walk-forward OOS probs
    # -----------------------------------------------------------------------
    print("\n--- Isotonic Calibration ---")
    d55b_oos_probs = np.load(D55B_OOS_PROBS_PATH)
    # Valid OOS probs (not NaN) with valid labels
    cal_valid = ~np.isnan(d55b_oos_probs) & train_valid
    cal_train_probs = d55b_oos_probs[cal_valid]
    cal_train_labels = train_labels[cal_valid]
    print(f"  Calibration training set: {len(cal_train_probs):,} samples")

    hold_probs_cal = isotonic_calibrate(cal_train_probs, cal_train_labels, raw_hold_probs)
    print(f"  Calibrated holdout mean prob: {hold_probs_cal.mean():.4f}")
    print(f"  Calibrated prob >= 0.60: {(hold_probs_cal >= 0.60).sum():,} bars")

    # -----------------------------------------------------------------------
    # 9. Prepare holdout labels + compute AUC + ECE
    # -----------------------------------------------------------------------
    raw_hold_labels = df_hold[LABEL_COL].values
    hold_valid = ~np.isnan(raw_hold_labels)
    hold_labels = np.where(hold_valid, raw_hold_labels, 0).astype(int)

    # AUC (on all valid-label holdout bars)
    auc_raw = compute_auc(hold_labels[hold_valid], raw_hold_probs[hold_valid])
    auc_cal = compute_auc(hold_labels[hold_valid], hold_probs_cal[hold_valid])

    # ECE
    ece_raw = compute_ece(raw_hold_probs[hold_valid], hold_labels[hold_valid])
    ece_cal = compute_ece(hold_probs_cal[hold_valid], hold_labels[hold_valid])

    print(f"\n  Holdout AUC (raw):        {auc_raw:.4f}")
    print(f"  Holdout AUC (calibrated): {auc_cal:.4f}")
    print(f"  Holdout ECE (raw):        {ece_raw:.6f}")
    print(f"  Holdout ECE (calibrated): {ece_cal:.6f}")

    # -----------------------------------------------------------------------
    # 10. Signal generation + simulation
    # -----------------------------------------------------------------------
    print("\n--- Signal Generation & Simulation ---")
    signal_mask = hold_valid & (hold_probs_cal >= THRESHOLD)
    n_signals = int(signal_mask.sum())
    print(f"  Signals at t={THRESHOLD}: {n_signals:,}")

    trade_indices = simulate(signal_mask, hold_labels, COOLDOWN)
    n_trades = len(trade_indices)
    print(f"  Trades after CD={COOLDOWN}: {n_trades:,}")

    # -----------------------------------------------------------------------
    # 11. Build R returns + equity simulation
    # -----------------------------------------------------------------------
    r_returns = build_trade_returns(trade_indices, hold_labels, R_TARGET, COST_PER_R)

    # Kelly sizing
    trade_probs = hold_probs_cal[trade_indices]
    risk_pcts = kelly_fraction_array(trade_probs, odds=KELLY_ODDS, divisor=KELLY_DIVISOR)
    equity_path, max_dd = equity_sim_variable(r_returns, risk_pcts, INITIAL_EQUITY)
    final_equity = equity_path[-1] if len(equity_path) > 0 else INITIAL_EQUITY

    # -----------------------------------------------------------------------
    # 12. Compute all metrics
    # -----------------------------------------------------------------------
    # Date range for annualization
    if "bar_start_ts_utc" in df_hold.columns:
        ts = df_hold["bar_start_ts_utc"]
        years = (ts.iloc[-1] - ts.iloc[0]).total_seconds() / (365.25 * 86400)
    else:
        years = len(df_hold) * 5 / (365.25 * 24 * 60)
    years = max(years, 0.1)

    metrics = compute_metrics("D55b_holdout", r_returns, max_dd, final_equity, years)

    # Daily Sharpe (THE key metric)
    if "bar_start_ts_utc" in df_hold.columns:
        daily_sharpe, daily_pnl = compute_daily_sharpe(
            trade_indices, r_returns, df_hold["bar_start_ts_utc"]
        )
    else:
        daily_sharpe, daily_pnl = 0.0, np.array([])

    # Per-bar Sharpe (for comparison with walk-forward numbers)
    bars_per_year = 365.25 * 24 * 60 / 5  # 5-min bars
    per_bar_sharpe = compute_per_bar_sharpe(r_returns, len(df_hold), bars_per_year)

    # -----------------------------------------------------------------------
    # 13. Compile all results
    # -----------------------------------------------------------------------
    all_metrics = {
        "auc": auc_cal,
        "auc_raw": auc_raw,
        "ece": ece_cal,
        "ece_raw": ece_raw,
        "n_trades": n_trades,
        "n_signals": n_signals,
        "trades_per_yr": metrics["trades_per_yr"],
        "win_rate": metrics["win_rate"],
        "ev_r": metrics["ev_r"],
        "profit_factor": metrics["profit_factor"],
        "max_dd_pct": metrics["max_dd_pct"],
        "sharpe_ann": metrics["sharpe_ann"],
        "daily_sharpe": daily_sharpe,
        "per_bar_sharpe": per_bar_sharpe,
        "final_equity": metrics["final_equity"],
        "holdout_rows": len(df_hold),
        "holdout_years": round(years, 2),
        "mean_risk_pct": float(risk_pcts.mean()) if n_trades > 0 else 0.0,
    }

    # -----------------------------------------------------------------------
    # 14. Evaluate gates
    # -----------------------------------------------------------------------
    gate_results = evaluate_gates(all_metrics)

    # -----------------------------------------------------------------------
    # 15. Apply decision tree
    # -----------------------------------------------------------------------
    verdict = apply_decision_tree(gate_results, all_metrics)

    # -----------------------------------------------------------------------
    # 16. Print comprehensive report
    # -----------------------------------------------------------------------
    elapsed = time.time() - t0

    print()
    print("=" * 60)
    print("  D55b HOLDOUT RESULTS")
    print("=" * 60)
    print()
    print(f"  Holdout period:   2025-03 to 2026-02 ({years:.2f} years)")
    print(f"  Holdout rows:     {len(df_hold):,}")
    print(f"  Features used:    {len(features)}")
    print()
    print("  --- Discrimination ---")
    print(f"  AUC (calibrated): {auc_cal:.4f}   (train: 0.7938)")
    print(f"  AUC (raw):        {auc_raw:.4f}")
    print(f"  ECE (calibrated): {ece_cal:.6f} (train: 0.0197)")
    print(f"  ECE (raw):        {ece_raw:.6f}")
    print()
    print("  --- Trading Performance ---")
    print(f"  Trades:           {n_trades:,}  ({metrics['trades_per_yr']:.1f}/yr)")
    print(f"  Win Rate:         {metrics['win_rate'] * 100:.2f}%  (train: 74.93%)")
    print(f"  EV (R):           {metrics['ev_r']:+.4f}  (train: +1.198)")
    print(f"  Profit Factor:    {metrics['profit_factor']:.4f}")
    print(f"  Max Drawdown:     {metrics['max_dd_pct']:.2f}%  (train: 7.71%)")
    print()
    print("  --- Sharpe Ratios ---")
    print(f"  *** DAILY SHARPE:   {daily_sharpe:.4f} ***")
    print(f"  Per-bar Sharpe:     {per_bar_sharpe:.4f}  (train: 12.29)")
    print(f"  Trade-ann. Sharpe:  {metrics['sharpe_ann']:.4f}")
    print()
    print("  --- Equity ---")
    print(f"  Initial:          ${INITIAL_EQUITY:,.2f}")
    print(f"  Final:            ${final_equity:,.2f}")
    print(f"  Mean Kelly risk:  {all_metrics['mean_risk_pct'] * 100:.2f}%")
    print()
    print("  --- Pre-Registered Gates ---")
    n_pass = 0
    for name, g in gate_results.items():
        status = "PASS" if g["pass"] else "FAIL"
        if g["pass"]:
            n_pass += 1
        print(f"    {name:20s} {status}  "
              f"(value={g['value']}, threshold {g['op']} {g['threshold']})")
    print()
    print(f"  Gates: {n_pass}/{len(gate_results)}")
    print()
    print(f"  >>> VERDICT: {verdict} <<<")
    print()
    print(f"  Elapsed: {elapsed:.1f}s")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # 17. Save results JSON
    # -----------------------------------------------------------------------
    results_json = {
        "id": "D55b_holdout",
        "config": {
            "base_experiment": "D55b_tier1_only",
            "features": len(features),
            "feature_exclude": len(feature_exclude),
            "label": LABEL_COL,
            "threshold": THRESHOLD,
            "cooldown": COOLDOWN,
            "sizing": {"method": "kelly", "divisor": KELLY_DIVISOR, "odds": KELLY_ODDS},
            "seed": SEED,
        },
        "train_benchmarks": {
            "auc": 0.7938,
            "win_rate": 0.7493,
            "ev_r": 1.198,
            "sharpe_ann": 12.29,
            "max_dd_pct": 7.71,
            "ece": 0.0197,
        },
        "holdout_metrics": all_metrics,
        "gates": gate_results,
        "verdict": verdict,
        "elapsed_s": round(elapsed, 1),
    }

    out_path = RESULTS_DIR / "d55b_holdout.json"
    with open(out_path, "w") as f:
        json.dump(results_json, f, indent=2, default=str)
    print(f"\n  Results saved: {out_path.name}")

    # Return for potential programmatic use
    return results_json


if __name__ == "__main__":
    results = main()
    if results["verdict"] == "FAIL":
        sys.exit(1)
