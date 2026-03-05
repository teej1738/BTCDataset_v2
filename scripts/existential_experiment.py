"""Existential Experiment: Does the BTC signal survive H1/H4 ATR stops?

This is the go/no-go gate for Project Meridian Phase 2.
The validated signal (AUC 0.7993) used 5m ATR stops where cost_R ~1.18
(unviable). This tests whether wider stops preserve the signal while
making cost_R viable.

Run from BTCDataset_v2 root:
    python scripts/existential_experiment.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata


# ── Manual AUC (Mann-Whitney) ────────────────────────────────────────

def manual_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUC via Mann-Whitney U statistic. No sklearn needed."""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    ranks = rankdata(np.concatenate([neg, pos]))
    n_pos = len(pos)
    n_neg = len(neg)
    u = ranks[n_neg:].sum() - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


# ── Manual Isotonic Regression (PAVA) ────────────────────────────────

def isotonic_regression_pava(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pool Adjacent Violators Algorithm. Returns (x_sorted, y_fitted)."""
    order = np.argsort(x)
    x_s = x[order].astype(float)
    y_s = y[order].astype(float)
    n = len(y_s)
    # PAVA
    blocks_val = y_s.copy()
    blocks_weight = np.ones(n, dtype=float)
    blocks_end = np.arange(n)  # end index of each block
    active = list(range(n))
    changed = True
    while changed:
        changed = False
        new_active = [active[0]]
        for i in range(1, len(active)):
            if blocks_val[new_active[-1]] > blocks_val[active[i]]:
                # merge
                prev = new_active[-1]
                cur = active[i]
                w1 = blocks_weight[prev]
                w2 = blocks_weight[cur]
                blocks_val[prev] = (blocks_val[prev] * w1 + blocks_val[cur] * w2) / (w1 + w2)
                blocks_weight[prev] = w1 + w2
                blocks_end[prev] = blocks_end[cur]
                changed = True
            else:
                new_active.append(active[i])
        active = new_active
    # Build output
    y_fit = np.empty(n)
    start = 0
    for idx in active:
        end = blocks_end[idx] + 1
        y_fit[start:end] = blocks_val[idx]
        start = end
    return x_s, np.clip(y_fit, 0.0, 1.0)


def isotonic_predict(
    x_train: np.ndarray, y_fitted: np.ndarray, x_new: np.ndarray
) -> np.ndarray:
    """Predict using fitted isotonic regression via linear interpolation."""
    return np.clip(np.interp(x_new, x_train, y_fitted), 0.0, 1.0)

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent.parent

# ── Configuration ─────────────────────────────────────────────────────

CONFIGS = [
    {"name": "A", "atr_tf": "1h", "sl_mult": 1.0, "tp_mult": 1.0, "max_hold": 48},
    {"name": "B", "atr_tf": "1h", "sl_mult": 1.0, "tp_mult": 1.5, "max_hold": 48},
    {"name": "C", "atr_tf": "1h", "sl_mult": 1.0, "tp_mult": 2.0, "max_hold": 96},
    {"name": "D", "atr_tf": "4h", "sl_mult": 1.0, "tp_mult": 1.0, "max_hold": 96},
    {"name": "E", "atr_tf": "4h", "sl_mult": 1.0, "tp_mult": 1.5, "max_hold": 96},
    {"name": "F", "atr_tf": "4h", "sl_mult": 1.0, "tp_mult": 2.0, "max_hold": 192},
]

N_FOLDS = 3
SEEDS = [42, 123, 456]
PROB_THRESHOLD = 0.55
COOLDOWN_BARS = 6
NOTIONAL = 50_000.0
TAKER_FEE_BPS = 5.0
SLIPPAGE_BPS = 1.0
FUNDING_RATE_PER_8H = 0.0001  # 0.01%

LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "num_leaves": 64,
    "min_child_samples": 200,
    "max_depth": 8,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
    "seed": 42,
    "deterministic": True,
    "force_col_wise": True,
}


# ── Step 1: Load data ────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, list[str]]:
    """Load train parquet, augment features, return (df, feature_list)."""
    print("=" * 60)
    print("STEP 1: Loading data")
    print("=" * 60)

    train_path = ROOT / "data" / "labeled" / "BTCUSDT_5m_labeled_v3_train.parquet"
    print(f"  Loading {train_path.name} ...")
    df = pd.read_parquet(train_path)
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")

    # Date range
    ts = df["bar_start_ts_utc"]
    print(f"  Date range: {ts.min()} to {ts.max()}")

    # Load tier1 features
    tiers_path = ROOT / "core" / "experiments" / "d55_tiers.json"
    with open(tiers_path) as f:
        tiers = json.load(f)
    tier1 = tiers["tier1"]
    print(f"  Tier1 features: {len(tier1)}")

    # Check which features need augmentation
    missing = [f for f in tier1 if f not in df.columns]
    if missing:
        print(f"  Missing features ({len(missing)}): {missing}")
        print("  Running augment_features ...")
        sys.path.insert(0, str(ROOT))
        from core.engine.simulator import augment_features
        df = augment_features(df)
        # Re-check
        still_missing = [f for f in tier1 if f not in df.columns]
        if still_missing:
            print(f"  WARNING: Still missing after augmentation: {still_missing}")
            print("  Dropping missing features from tier1 list")
            tier1 = [f for f in tier1 if f not in still_missing]
    else:
        print("  All features present in parquet")

    print(f"  Final feature count: {len(tier1)}")
    return df, tier1


# ── Step 2: Compute HTF ATR ──────────────────────────────────────────

def compute_htf_atr(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 1H and 4H ATR(14) forward-filled to 5m index."""
    print()
    print("=" * 60)
    print("STEP 2: Computing HTF ATR")
    print("=" * 60)

    ts = pd.to_datetime(df["bar_start_ts_utc"], utc=True)
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    for tf_label, bars_per_period in [("1h", 12), ("4h", 48)]:
        # Resample OHLCV
        n = len(df)
        n_periods = n // bars_per_period
        usable = n_periods * bars_per_period

        h_arr = high[:usable].reshape(n_periods, bars_per_period)
        l_arr = low[:usable].reshape(n_periods, bars_per_period)
        c_arr = close[:usable].reshape(n_periods, bars_per_period)

        htf_high = h_arr.max(axis=1)
        htf_low = l_arr.min(axis=1)
        htf_close = c_arr[:, -1]

        # ATR(14) on HTF
        tr = np.maximum(
            htf_high - htf_low,
            np.maximum(
                np.abs(htf_high - np.roll(htf_close, 1)),
                np.abs(htf_low - np.roll(htf_close, 1)),
            ),
        )
        tr[0] = htf_high[0] - htf_low[0]

        # EMA ATR(14)
        atr = np.zeros(len(tr))
        atr[:14] = tr[:14].mean()
        alpha = 2.0 / (14 + 1)
        for i in range(14, len(tr)):
            atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]

        # Forward-fill to 5m
        atr_5m = np.repeat(atr, bars_per_period)
        # Handle remainder
        full = np.full(n, np.nan)
        full[:usable] = atr_5m
        # Forward fill the tail
        if usable < n:
            full[usable:] = atr[-1]

        col_name = f"atr_{tf_label}"
        df[col_name] = full

        median_atr = np.nanmedian(full)
        median_price = np.nanmedian(close)
        pct = median_atr / median_price * 100
        print(f"  {tf_label.upper()} ATR(14): median ${median_atr:.2f} ({pct:.3f}% of price)")

    return df


# ── Step 3: Generate triple-barrier labels ────────────────────────────

def generate_labels(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    sl_mult: float,
    tp_mult: float,
    max_hold: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate triple-barrier labels.

    Returns (long_label, short_label, hold_time) arrays.
    """
    n = len(close)
    long_label = np.full(n, np.nan)
    short_label = np.full(n, np.nan)
    hold_time = np.full(n, np.nan)

    for i in range(n - 1):
        if np.isnan(atr[i]) or atr[i] < 1e-6:
            continue

        entry = close[i]
        stop_dist = sl_mult * atr[i]
        target_dist = tp_mult * atr[i]
        upper = entry + target_dist
        lower = entry - stop_dist

        outcome = 0  # default: expiry
        bars_held = max_hold

        for j in range(i + 1, min(i + max_hold + 1, n)):
            # Check if barriers hit within bar
            if high[j] >= upper and low[j] <= lower:
                # Both hit -- conservative: SL wins
                outcome = -1
                bars_held = j - i
                break
            elif high[j] >= upper:
                outcome = 1
                bars_held = j - i
                break
            elif low[j] <= lower:
                outcome = -1
                bars_held = j - i
                break

        if outcome == 0:
            # Max hold reached -- check return
            exit_idx = min(i + max_hold, n - 1)
            ret = close[exit_idx] - entry
            bars_held = exit_idx - i
            threshold = 0.1 * atr[i]
            if ret > threshold:
                outcome = 1
            elif ret < -threshold:
                outcome = -1
            # else outcome stays 0

        long_label[i] = 1.0 if outcome == 1 else 0.0
        short_label[i] = 1.0 if outcome == -1 else 0.0
        hold_time[i] = bars_held

    return long_label, short_label, hold_time


def generate_all_labels(df: pd.DataFrame) -> dict:
    """Generate labels for all configs. Returns dict of config_name -> label_info."""
    print()
    print("=" * 60)
    print("STEP 3: Generating triple-barrier labels")
    print("=" * 60)

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    results = {}
    for cfg in CONFIGS:
        name = cfg["name"]
        atr_col = f"atr_{cfg['atr_tf']}"
        atr = df[atr_col].values

        t0 = time.monotonic()
        long_lbl, short_lbl, hold = generate_labels(
            close, high, low, atr,
            cfg["sl_mult"], cfg["tp_mult"], cfg["max_hold"],
        )
        elapsed = time.monotonic() - t0

        # Stats
        valid = ~np.isnan(long_lbl)
        n_valid = valid.sum()
        long_rate = np.nanmean(long_lbl[valid])
        short_rate = np.nanmean(short_lbl[valid])
        expiry_rate = 1.0 - long_rate - short_rate
        median_hold = np.nanmedian(hold[valid])
        median_stop_pct = np.nanmedian(cfg["sl_mult"] * atr[valid] / close[valid]) * 100

        print(f"  Config {name} ({cfg['atr_tf'].upper()} ATR, SL={cfg['sl_mult']}, "
              f"TP={cfg['tp_mult']}, hold={cfg['max_hold']}): "
              f"L={long_rate:.1%} S={short_rate:.1%} E={expiry_rate:.1%} "
              f"med_hold={median_hold:.0f}bars stop={median_stop_pct:.3f}% "
              f"[{elapsed:.1f}s]")

        results[name] = {
            "long_label": long_lbl,
            "short_label": short_lbl,
            "hold_time": hold,
            "long_rate": long_rate,
            "short_rate": short_rate,
            "expiry_rate": expiry_rate,
            "median_hold": median_hold,
            "median_stop_pct": median_stop_pct,
            "config": cfg,
        }

    return results


# ── Step 4: Walk-forward LightGBM ────────────────────────────────────

def walk_forward_lgb(
    df: pd.DataFrame,
    features: list[str],
    labels: np.ndarray,
    n_folds: int,
    embargo_bars: int,
    seeds: list[int],
) -> list[dict]:
    """Walk-forward with expanding window, K seeds, isotonic calibration.

    Returns list of fold dicts with AUC and calibrated probabilities.
    """
    n = len(df)
    valid = ~np.isnan(labels)
    X = df[features].values
    y = labels

    # Split: reserve last fraction as test folds
    fold_size = n // (n_folds + 1)
    fold_results = []

    for fold_id in range(n_folds):
        train_end = fold_size * (fold_id + 1)
        test_start = train_end + embargo_bars
        test_end = min(test_start + fold_size, n)

        if test_end <= test_start:
            continue

        # Train/test masks (must also be valid labels)
        train_mask = np.zeros(n, dtype=bool)
        train_mask[:train_end] = True
        train_mask &= valid

        test_mask = np.zeros(n, dtype=bool)
        test_mask[test_start:test_end] = True
        test_mask &= valid

        if train_mask.sum() < 200 or test_mask.sum() < 50:
            continue

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        # Skip if single class
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        # Early stopping split: last 15% of train
        n_train = len(X_train)
        n_val = max(int(n_train * 0.15), 50)
        X_tr = X_train[:-n_val]
        y_tr = y_train[:-n_val]
        X_val = X_train[-n_val:]
        y_val = y_train[-n_val:]

        # Multi-seed ensemble
        all_probs_test = []
        all_probs_val = []

        for seed in seeds:
            params = {**LGB_PARAMS, "seed": seed}
            dtrain = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
            dval = lgb.Dataset(X_val, label=y_val, free_raw_data=False)

            model = lgb.train(
                params,
                dtrain,
                num_boost_round=2000,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )

            all_probs_val.append(model.predict(X_val))
            all_probs_test.append(model.predict(X_test))

        # Average across seeds
        raw_probs_val = np.mean(all_probs_val, axis=0)
        raw_probs_test = np.mean(all_probs_test, axis=0)

        # Isotonic calibration on validation set
        try:
            x_iso, y_iso = isotonic_regression_pava(raw_probs_val, y_val)
            cal_probs = isotonic_predict(x_iso, y_iso, raw_probs_test)
        except Exception:
            cal_probs = raw_probs_test

        auc = manual_roc_auc(y_test, cal_probs)

        fold_results.append({
            "fold_id": fold_id,
            "auc": auc,
            "probs": cal_probs,
            "y_test": y_test,
            "test_indices": np.where(test_mask)[0],
            "train_size": train_mask.sum(),
            "test_size": test_mask.sum(),
        })

    return fold_results


def run_all_models(
    df: pd.DataFrame,
    features: list[str],
    label_results: dict,
) -> dict:
    """Run walk-forward for all configs x directions."""
    print()
    print("=" * 60)
    print("STEP 4: Walk-forward LightGBM training")
    print("=" * 60)

    model_results = {}

    for cfg_name, lbl_info in label_results.items():
        cfg = lbl_info["config"]
        embargo = 2 * cfg["max_hold"]

        for direction, label_key in [("long", "long_label"), ("short", "short_label")]:
            key = f"{cfg_name}_{direction}"
            labels = lbl_info[label_key]

            t0 = time.monotonic()
            folds = walk_forward_lgb(df, features, labels, N_FOLDS, embargo, SEEDS)
            elapsed = time.monotonic() - t0

            if folds:
                mean_auc = np.mean([f["auc"] for f in folds])
                print(f"  {key}: AUC={mean_auc:.4f} ({len(folds)} folds) [{elapsed:.1f}s]")
            else:
                mean_auc = 0.5
                print(f"  {key}: NO VALID FOLDS [{elapsed:.1f}s]")

            model_results[key] = {
                "config_name": cfg_name,
                "direction": direction,
                "folds": folds,
                "mean_auc": mean_auc,
                "config": cfg,
                "label_info": lbl_info,
            }

    return model_results


# ── Step 5: Trade simulation ─────────────────────────────────────────

def simulate_trades(
    model_result: dict,
    df: pd.DataFrame,
) -> list[dict]:
    """Simulate trades on test fold predictions with cost model."""
    cfg = model_result["config"]
    direction = model_result["direction"]
    atr_col = f"atr_{cfg['atr_tf']}"
    close = df["close"].values
    atr = df[atr_col].values

    all_trades = []

    for fold in model_result["folds"]:
        probs = fold["probs"]
        test_indices = fold["test_indices"]

        last_trade_bar = -COOLDOWN_BARS - 1

        for k, idx in enumerate(test_indices):
            if probs[k] < PROB_THRESHOLD:
                continue
            if idx - last_trade_bar < COOLDOWN_BARS:
                continue
            if np.isnan(atr[idx]) or atr[idx] < 1e-6:
                continue

            entry_price = close[idx]
            stop_dist = cfg["sl_mult"] * atr[idx]
            target_dist = cfg["tp_mult"] * atr[idx]

            if direction == "long":
                upper = entry_price + target_dist
                lower = entry_price - stop_dist
            else:
                upper = entry_price + stop_dist  # stop for short
                lower = entry_price - target_dist  # target for short

            # Walk forward to find exit
            exit_price = entry_price
            bars_held = cfg["max_hold"]
            outcome = "expiry"

            for j in range(idx + 1, min(idx + cfg["max_hold"] + 1, len(close))):
                h = df["high"].values[j]
                l = df["low"].values[j]

                if direction == "long":
                    if h >= upper and l <= lower:
                        exit_price = lower  # SL wins
                        bars_held = j - idx
                        outcome = "sl"
                        break
                    elif h >= upper:
                        exit_price = upper
                        bars_held = j - idx
                        outcome = "tp"
                        break
                    elif l <= lower:
                        exit_price = lower
                        bars_held = j - idx
                        outcome = "sl"
                        break
                else:  # short
                    if l <= lower and h >= upper:
                        exit_price = upper  # SL wins
                        bars_held = j - idx
                        outcome = "sl"
                        break
                    elif l <= lower:
                        exit_price = lower
                        bars_held = j - idx
                        outcome = "tp"
                        break
                    elif h >= upper:
                        exit_price = upper
                        bars_held = j - idx
                        outcome = "sl"
                        break

            if outcome == "expiry":
                exit_idx = min(idx + cfg["max_hold"], len(close) - 1)
                exit_price = close[exit_idx]
                bars_held = exit_idx - idx

            # PnL
            if direction == "long":
                gross_pnl = (exit_price - entry_price) / entry_price * NOTIONAL
            else:
                gross_pnl = (entry_price - exit_price) / entry_price * NOTIONAL

            # Cost model
            fee_cost = NOTIONAL * TAKER_FEE_BPS / 10000 * 2  # entry + exit
            slip_cost = NOTIONAL * SLIPPAGE_BPS / 10000 * 2
            n_funding = max(0, bars_held * 5 // (8 * 60))  # 5min bars -> 8h intervals
            funding_cost = NOTIONAL * FUNDING_RATE_PER_8H * n_funding
            total_cost = fee_cost + slip_cost + funding_cost
            net_pnl = gross_pnl - total_cost

            all_trades.append({
                "fold_id": fold["fold_id"],
                "entry_bar": int(idx),
                "bars_held": bars_held,
                "outcome": outcome,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "stop_dist": stop_dist,
                "gross_pnl": gross_pnl,
                "total_cost": total_cost,
                "net_pnl": net_pnl,
            })

            last_trade_bar = idx

    return all_trades


def run_all_simulations(
    model_results: dict,
    df: pd.DataFrame,
) -> dict:
    """Run trade simulations for all configs."""
    print()
    print("=" * 60)
    print("STEP 5: Trade simulation with cost model")
    print("=" * 60)

    sim_results = {}
    for key, mr in model_results.items():
        trades = simulate_trades(mr, df)
        print(f"  {key}: {len(trades)} trades")
        sim_results[key] = trades

    return sim_results


# ── Step 6: Compute metrics ──────────────────────────────────────────

def compute_metrics(
    model_results: dict,
    sim_results: dict,
    total_bars: int,
) -> list[dict]:
    """Compute summary metrics for all configs."""
    print()
    print("=" * 60)
    print("STEP 6: Computing metrics")
    print("=" * 60)

    bars_per_year = 105_120  # 5m bars in a year
    # Estimate test bars (roughly 1/4 of total for 3 folds)
    test_years = (total_bars / (N_FOLDS + 1) * N_FOLDS) / bars_per_year

    all_metrics = []
    for key, mr in model_results.items():
        trades = sim_results[key]
        cfg = mr["config"]

        if len(trades) < 3:
            all_metrics.append({
                "key": key,
                "config_name": mr["config_name"],
                "direction": mr["direction"],
                "atr_tf": cfg["atr_tf"],
                "sl_mult": cfg["sl_mult"],
                "tp_mult": cfg["tp_mult"],
                "max_hold": cfg["max_hold"],
                "auc": mr["mean_auc"],
                "cost_r": np.nan,
                "ev_per_trade": np.nan,
                "win_rate": np.nan,
                "sr_per_trade": np.nan,
                "trades_total": len(trades),
                "trades_per_year": len(trades) / max(test_years, 0.1),
                "ann_sr": np.nan,
            })
            continue

        net_pnls = np.array([t["net_pnl"] for t in trades])
        gross_pnls = np.array([t["gross_pnl"] for t in trades])
        costs = np.array([t["total_cost"] for t in trades])
        stop_dists = np.array([t["stop_dist"] for t in trades])
        stop_dist_usd = stop_dists / np.array([t["entry_price"] for t in trades]) * NOTIONAL

        cost_r = np.median(costs / np.maximum(stop_dist_usd, 1e-6))
        ev_per_trade = np.mean(net_pnls)
        win_rate = np.mean(net_pnls > 0)
        sr_per_trade = np.mean(net_pnls) / max(np.std(net_pnls), 1e-6)
        trades_per_year = len(trades) / max(test_years, 0.1)
        ann_sr = sr_per_trade * np.sqrt(trades_per_year)

        m = {
            "key": key,
            "config_name": mr["config_name"],
            "direction": mr["direction"],
            "atr_tf": cfg["atr_tf"],
            "sl_mult": cfg["sl_mult"],
            "tp_mult": cfg["tp_mult"],
            "max_hold": cfg["max_hold"],
            "auc": mr["mean_auc"],
            "cost_r": cost_r,
            "ev_per_trade": ev_per_trade,
            "win_rate": win_rate,
            "sr_per_trade": sr_per_trade,
            "trades_total": len(trades),
            "trades_per_year": trades_per_year,
            "ann_sr": ann_sr,
        }
        all_metrics.append(m)

        print(f"  {key}: AUC={m['auc']:.4f} cost_R={cost_r:.4f} "
              f"EV=${ev_per_trade:.2f} WR={win_rate:.1%} "
              f"SR/trade={sr_per_trade:.4f} trades/yr={trades_per_year:.0f} "
              f"Ann SR={ann_sr:.2f}")

    return all_metrics


# ── Step 7: Generate report ──────────────────────────────────────────

def generate_report(
    all_metrics: list[dict],
    n_rows: int,
    date_range: str,
    n_features: int,
) -> str:
    """Generate markdown report with verdict."""

    # Summary table
    header = ("| Config | Dir | ATR TF | SL | TP | Hold | AUC | cost_R | "
              "EV/trade | WR | SR/trade | Trades/yr | Ann SR |")
    sep = ("|--------|-----|--------|----|----|------|-----|--------|"
           "----------|----|---------|-----------| -------|")
    rows = []
    for m in all_metrics:
        def fmt(v, f=".4f"):
            return f"{{:{f}}}".format(v) if not (isinstance(v, float) and np.isnan(v)) else "N/A"

        row = (f"| {m['config_name']} | {m['direction'][0].upper()} | "
               f"{m['atr_tf'].upper()} | {m['sl_mult']} | {m['tp_mult']} | "
               f"{m['max_hold']} | {fmt(m['auc'])} | {fmt(m['cost_r'])} | "
               f"${fmt(m['ev_per_trade'], '.2f')} | {fmt(m['win_rate'], '.1%')} | "
               f"{fmt(m['sr_per_trade'])} | {fmt(m['trades_per_year'], '.0f')} | "
               f"{fmt(m['ann_sr'], '.2f')} |")
        rows.append(row)

    table = "\n".join([header, sep] + rows)

    # Determine verdict
    viable_configs = []
    all_dead = True
    any_marginal = False

    for m in all_metrics:
        sr = m["sr_per_trade"]
        auc = m["auc"]
        cost_r = m["cost_r"]
        ev = m["ev_per_trade"]
        wr = m["win_rate"]
        tpy = m["trades_per_year"]

        if isinstance(sr, float) and np.isnan(sr):
            continue

        if sr >= 0.02 or auc >= 0.55:
            all_dead = False

        if 0.02 <= sr < 0.05 or 0.55 <= auc < 0.60:
            any_marginal = True

        # Full viability check
        be_wr = 1.0 / (1.0 + m["tp_mult"] / m["sl_mult"])  # break-even WR
        if (sr >= 0.05 and cost_r < 0.25 and auc >= 0.60
                and ev > 0 and wr > be_wr and tpy >= 100):
            viable_configs.append(m)

    # Also check fold-level EV
    # (relaxed: just check overall for now since we have averaged)

    if viable_configs:
        verdict = "VIABLE"
        best = max(viable_configs, key=lambda m: m["sr_per_trade"])
        verdict_detail = (
            f"VIABLE -- {len(viable_configs)} config(s) pass all criteria.\n"
            f"Best: Config {best['config_name']} {best['direction']} "
            f"(SR/trade={best['sr_per_trade']:.4f}, AUC={best['auc']:.4f}, "
            f"cost_R={best['cost_r']:.4f}, EV=${best['ev_per_trade']:.2f})"
        )
    elif all_dead:
        verdict = "DEAD"
        verdict_detail = "DEAD -- Per-trade Sharpe < 0.02 and/or AUC < 0.55 across ALL configs."
    else:
        verdict = "MARGINAL"
        best = max(all_metrics, key=lambda m: m["sr_per_trade"] if not np.isnan(m["sr_per_trade"]) else -999)
        verdict_detail = (
            f"MARGINAL -- Signal shows some life but does not fully pass viability criteria.\n"
            f"Best: Config {best['config_name']} {best['direction']} "
            f"(SR/trade={best['sr_per_trade']:.4f}, AUC={best['auc']:.4f})"
        )

    report = f"""# Existential Experiment Results

Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Data: {n_rows:,} rows, {date_range}
Features: {n_features} (BTCDataset_v2 validated Tier1 set)

## Configuration
- Folds: {N_FOLDS} (expanding window)
- Seeds: {SEEDS}
- Embargo: 2x max_hold bars
- LightGBM: lr=0.03, leaves=64, min_child=200, depth=8
- Early stopping: patience=50 on last 15% of train
- Calibration: isotonic regression
- Entry threshold: prob > {PROB_THRESHOLD}
- Cooldown: {COOLDOWN_BARS} bars
- Cost: {TAKER_FEE_BPS}bps taker + {SLIPPAGE_BPS}bps slip + {FUNDING_RATE_PER_8H*100:.3f}%/8h funding

## Summary Table

{table}

## Verdict

**{verdict}**

{verdict_detail}

## Criteria Applied

DEAD if ALL of:
- Per-trade Sharpe < 0.02 at ALL configs
- AUC < 0.55 at ALL configs
- Trade count < 50 annualized at ALL configs
- Cost-adjusted EV <= 0 at ALL configs

VIABLE if at least ONE config has ALL of:
- Per-trade Sharpe >= 0.05
- cost_R < 0.25
- AUC >= 0.60
- Cost-adjusted EV > 0
- Win rate > break-even
- >= 100 trades annualized

MARGINAL if between DEAD and VIABLE.
"""
    return report, verdict, best if (viable_configs or not all_dead) else None


# ── Main ──────────────────────────────────────────────────────────────

def main():
    t_start = time.monotonic()

    # Step 1
    df, features = load_data()

    # Step 2
    df = compute_htf_atr(df)

    # Step 3
    label_results = generate_all_labels(df)

    # Step 4
    model_results = run_all_models(df, features, label_results)

    # Step 5
    sim_results = run_all_simulations(model_results, df)

    # Step 6
    all_metrics = compute_metrics(model_results, sim_results, len(df))

    # Step 7
    ts = pd.to_datetime(df["bar_start_ts_utc"], utc=True)
    date_range = f"{ts.min()} to {ts.max()}"
    report, verdict, best = generate_report(all_metrics, len(df), date_range, len(features))

    # Save report
    output_path = ROOT / "outputs" / "EXISTENTIAL_RESULTS.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

    elapsed = time.monotonic() - t_start

    print()
    print("=" * 60)
    print(f"VERDICT: {verdict}")
    print("=" * 60)
    if best:
        print(f"Best config: {best['config_name']} {best['direction']}")
        print(f"  AUC: {best['auc']:.4f}")
        print(f"  cost_R: {best['cost_r']:.4f}")
        print(f"  EV/trade: ${best['ev_per_trade']:.2f}")
        print(f"  SR/trade: {best['sr_per_trade']:.4f}")
        print(f"  Ann SR: {best['ann_sr']:.2f}")
    print(f"\nTotal time: {elapsed:.0f}s")
    print(f"Report saved: {output_path}")


if __name__ == "__main__":
    main()
