# core/engine/evaluator.py
# Walk-forward training, simulation, CSCV, gates, ECE, metrics.
# D38 -- ported from legacy ml_pipeline.py, ml_backtest.py, cscv_validation.py.
# ASCII-only output for cp1252 compatibility.

from __future__ import annotations

import traceback
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# AUC (no sklearn dependency)
# ---------------------------------------------------------------------------
def compute_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """ROC AUC via Mann-Whitney U statistic (vectorized)."""
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(-y_prob)
    y_sorted = y_true[order]
    cum_pos = np.cumsum(y_sorted)
    return float(cum_pos[y_sorted == 0].sum()) / (n_pos * n_neg)


# ---------------------------------------------------------------------------
# Walk-forward train
# ---------------------------------------------------------------------------
def walk_forward_train(
    df: pd.DataFrame,
    features: list[str],
    label: str,
    config: dict,
) -> np.ndarray:
    """
    Walk-forward expanding-window LightGBM training.

    config keys:
        n_folds (int): ignored -- folds are auto-computed from data size
        embargo_bars (int): gap between train and test
        device (str): "gpu" or "cpu"
        model (str): "lgbm" (only lgbm supported currently)
        min_train_bars (int, optional): default 105_000
        test_fold_bars (int, optional): default 52_500
        val_frac (float, optional): default 0.10
        n_estimators (int, optional): default 1000
        early_stop_rounds (int, optional): default 50
        lgb_params (dict, optional): override LGB hyperparameters

    Returns OOS probability array (same length as df, NaN where not covered).
    """
    import lightgbm as lgb

    n = len(df)
    raw_labels = df[label].values
    # NaN labels exist at the tail (forward-looking labels can't be computed
    # for the last horizon bars). Fill NaN with 0 for array indexing, but
    # mask them out during training.
    label_valid = ~np.isnan(raw_labels)
    label_arr = np.where(label_valid, raw_labels, 0).astype(int)
    X = df[features]

    # 288 bars = 24h = max feature lookback per AFML Ch.7
    embargo = config.get("embargo_bars", 288)
    min_train = config.get("min_train_bars", 105_000)
    test_fold = config.get("test_fold_bars", 52_500)
    val_frac = config.get("val_frac", 0.10)
    n_estimators = config.get("n_estimators", 1000)
    early_stop = config.get("early_stop_rounds", 50)

    # Device selection
    device = config.get("device", "gpu")
    lgb_params = config.get("lgb_params", {}).copy()

    # Default LGB params
    defaults = {
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
        "seed": 42,
    }
    for k, v in defaults.items():
        lgb_params.setdefault(k, v)

    lgb_params["device"] = device

    # Try GPU, fallback to CPU
    gpu_ok = True
    if device == "gpu":
        try:
            # Quick probe: train 1 iteration on tiny data
            _probe = lgb.Dataset(
                X.iloc[:100], label=label_arr[:100], free_raw_data=False
            )
            _m = lgb.train(
                {**lgb_params, "num_iterations": 1, "verbosity": -1},
                _probe,
            )
            del _m, _probe
            print("Training device: gpu")
        except Exception:
            gpu_ok = False
            lgb_params["device"] = "cpu"
            print("Training device: cpu (fallback)")
    else:
        gpu_ok = False
        print("Training device: cpu")

    oos_probs = np.full(n, np.nan)
    fold_results = []

    # Define fold boundaries
    folds = []
    t_start = min_train + embargo
    while t_start < n:
        t_end = min(t_start + test_fold, n)
        folds.append((t_start, t_end))
        t_start = t_end

    print(f"  Walk-forward: {len(folds)} folds, "
          f"min_train={min_train:,}, test_fold={test_fold:,}, "
          f"embargo={embargo}")

    for fold_i, (test_start, test_end) in enumerate(folds):
        avail_end = test_start - embargo

        val_size = max(int(avail_end * val_frac), 1000)
        val_start = avail_end - val_size
        pure_train_end = val_start - embargo

        train_idx = np.arange(0, pure_train_end)
        val_idx = np.arange(val_start, avail_end)
        test_idx = np.arange(test_start, test_end)

        # Exclude NaN-label rows from train and val
        train_idx = train_idx[label_valid[train_idx]]
        val_idx = val_idx[label_valid[val_idx]]

        # Purge: exclude samples whose label horizon extends past their
        # boundary. A sample at bar i has label computed from bars [i, i+H].
        # Train samples must not look into val zone; val must not look into
        # test zone. (D51: AFML Ch.7 label purging)
        horizon = config.get("label_horizon_bars", 48)
        train_idx = train_idx[train_idx + horizon < val_start]
        val_idx = val_idx[val_idx + horizon < test_start]

        X_train = X.iloc[train_idx]
        y_train = label_arr[train_idx]
        X_val = X.iloc[val_idx]
        y_val = label_arr[val_idx]
        X_test = X.iloc[test_idx]
        y_test = label_arr[test_idx]

        dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        dval = lgb.Dataset(
            X_val, label=y_val, free_raw_data=False, reference=dtrain
        )

        callbacks = [
            lgb.early_stopping(early_stop, verbose=False),
            lgb.log_evaluation(period=10000),
        ]

        model = lgb.train(
            lgb_params,
            dtrain,
            num_boost_round=n_estimators,
            valid_sets=[dtrain, dval],
            valid_names=["train", "val"],
            callbacks=callbacks,
        )

        probs = model.predict(X_test)
        oos_probs[test_idx] = probs

        # Fold metrics (exclude NaN-label rows from evaluation)
        test_valid = label_valid[test_idx]
        y_eval = y_test[test_valid]
        p_eval = probs[test_valid]

        pos_rate = float(y_eval.mean()) if len(y_eval) > 0 else 0.0
        logloss = float(-np.mean(
            y_eval * np.log(np.clip(p_eval, 1e-10, 1))
            + (1 - y_eval) * np.log(np.clip(1 - p_eval, 1e-10, 1))
        )) if len(y_eval) > 0 else 0.0
        auc = compute_auc(y_eval, p_eval)
        best_iter = model.best_iteration
        if best_iter <= 0:
            best_iter = n_estimators

        fold_info = {
            "fold": fold_i + 1,
            "train_bars": len(train_idx),
            "val_bars": len(val_idx),
            "test_bars": len(test_idx),
            "test_pos_rate": round(pos_rate, 4),
            "logloss": round(logloss, 6),
            "auc": round(auc, 4),
            "best_iteration": best_iter,
            "pred_mean": round(float(probs.mean()), 4),
        }
        fold_results.append(fold_info)

        print(f"    Fold {fold_i + 1}/{len(folds)}: "
              f"train={len(train_idx):,} val={len(val_idx):,} "
              f"test={len(test_idx):,}  "
              f"logloss={logloss:.4f} AUC={auc:.4f} "
              f"iter={best_iter}")

    return oos_probs


# ---------------------------------------------------------------------------
# Simulate (bar-by-bar with cooldown)
# ---------------------------------------------------------------------------
def simulate(
    signal_mask: np.ndarray,
    label_arr: np.ndarray,
    cooldown: int,
) -> list[int]:
    """
    Walk bars chronologically with cooldown.
    Returns list of bar indices where trades are taken.
    """
    trade_indices: list[int] = []
    bars_since = cooldown  # start ready to trade

    for i in range(len(signal_mask)):
        bars_since += 1
        if bars_since > cooldown and signal_mask[i]:
            trade_indices.append(i)
            bars_since = 0

    return trade_indices


def build_trade_returns(
    trade_indices: list[int],
    label_arr: np.ndarray,
    r_target: int = 2,
    cost_per_r: float = 0.05,
) -> np.ndarray:
    """Convert trade bar indices to R returns array."""
    r_win = r_target - cost_per_r
    r_loss = -(1 + cost_per_r)
    returns = []
    for i in trade_indices:
        returns.append(r_win if label_arr[i] == 1 else r_loss)
    return np.array(returns) if returns else np.array([], dtype=float)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(
    name: str,
    r_returns: np.ndarray,
    max_dd: float,
    final_equity: float,
    years: float,
) -> dict:
    """Standard metrics from R returns array."""
    n = len(r_returns)
    if n == 0:
        return {
            "name": name, "n_trades": 0, "trades_per_yr": 0.0,
            "win_rate": 0.0, "ev_r": 0.0, "profit_factor": 0.0,
            "max_dd_pct": 0.0, "sharpe_ann": 0.0,
            "final_equity": final_equity,
        }

    wins = int(np.sum(r_returns > 0))
    wr = wins / n
    ev = float(np.mean(r_returns))

    gross_win = float(np.sum(r_returns[r_returns > 0]))
    gross_loss = float(abs(np.sum(r_returns[r_returns < 0])))
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

    std_r = float(np.std(r_returns, ddof=1)) if n > 1 else 0.0
    sr_pt = ev / std_r if std_r > 0 else 0.0
    tpy = n / years
    ann_sr = sr_pt * np.sqrt(tpy) if tpy > 0 else 0.0

    return {
        "name": name,
        "n_trades": n,
        "trades_per_yr": round(tpy, 1),
        "win_rate": round(wr, 4),
        "ev_r": round(ev, 4),
        "profit_factor": round(pf, 4),
        "max_dd_pct": round(max_dd * 100, 2),
        "sharpe_ann": round(ann_sr, 4),
        "final_equity": round(final_equity, 2),
    }


# ---------------------------------------------------------------------------
# CSCV
# ---------------------------------------------------------------------------
def _pf(returns: np.ndarray) -> float:
    """Profit factor from return array."""
    gross_win = float(np.sum(returns[returns > 0]))
    gross_loss = float(abs(np.sum(returns[returns < 0])))
    return gross_win / gross_loss if gross_loss > 0 else float("inf")


def _compute_cscv_pbo(returns: np.ndarray, n_folds: int = 8) -> dict:
    """
    Combinatorially Symmetric Cross-Validation.
    PBO = fraction of C(n_folds, n_folds//2) splits where OOS mean R <= 0.
    """
    n = len(returns)
    half = n_folds // 2
    fold_size = n // n_folds

    fold_blocks = []
    for i in range(n_folds):
        start = i * fold_size
        end = (start + fold_size) if i < n_folds - 1 else n
        fold_blocks.append(returns[start:end])

    oos_means = []
    is_means = []

    for is_idx in combinations(range(n_folds), half):
        oos_idx = tuple(i for i in range(n_folds) if i not in is_idx)
        is_r = np.concatenate([fold_blocks[i] for i in is_idx])
        oos_r = np.concatenate([fold_blocks[i] for i in oos_idx])
        is_means.append(float(np.mean(is_r)))
        oos_means.append(float(np.mean(oos_r)))

    oos_arr = np.array(oos_means)
    is_arr = np.array(is_means)
    pbo = float(np.mean(oos_arr <= 0))

    corr = 0.0
    if len(is_arr) > 1:
        corr = float(np.corrcoef(is_arr, oos_arr)[0, 1])

    return {
        "pbo": round(pbo, 4),
        "n_combos": len(oos_arr),
        "n_negative_oos": int(np.sum(oos_arr <= 0)),
        "oos_mean": round(float(np.mean(oos_arr)), 4),
        "oos_median": round(float(np.median(oos_arr)), 4),
        "oos_std": round(float(np.std(oos_arr)), 4),
        "is_oos_correlation": round(corr, 4),
    }


def _compute_psr(returns: np.ndarray, benchmark: float = 0.0) -> dict:
    """
    Probabilistic Sharpe Ratio (Bailey & Lopez de Prado 2012).
    PSR(SR*) = Phi(z), tests P(true SR > benchmark | observed data).
    """
    n = len(returns)
    if n < 3:
        return {"psr": 0.0, "sharpe_ratio": 0.0, "z_score": 0.0, "n": n}

    mean_r = float(np.mean(returns))
    std_r = float(np.std(returns, ddof=1))

    if std_r == 0:
        sr = float("inf") if mean_r > 0 else 0.0
        return {
            "psr": 1.0 if mean_r > 0 else 0.0,
            "sharpe_ratio": sr, "z_score": 0.0, "n": n,
        }

    sr = mean_r / std_r
    gamma3 = float(stats.skew(returns, bias=False))
    gamma4 = float(stats.kurtosis(returns, fisher=False, bias=False))

    denom_sq = 1 - gamma3 * sr + (gamma4 - 1) / 4 * sr**2
    if denom_sq <= 0:
        denom_sq = 1.0

    z = (sr - benchmark) * np.sqrt(n - 1) / np.sqrt(denom_sq)
    psr = float(stats.norm.cdf(z))

    return {
        "psr": round(psr, 4),
        "sharpe_ratio": round(sr, 4),
        "z_score": round(z, 4),
        "n": n,
    }


def _block_bootstrap_ci(
    returns: np.ndarray,
    n_bootstrap: int = 10_000,
    block_size: int = 5,
    ci: float = 0.95,
) -> dict:
    """Block bootstrap confidence interval for mean R per trade."""
    n = len(returns)
    rng = np.random.default_rng(42)
    n_blocks = max(1, -(-n // block_size))
    max_start = max(1, n - block_size + 1)

    boot_means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        starts = rng.integers(0, max_start, size=n_blocks)
        sample = np.concatenate(
            [returns[s:s + block_size] for s in starts]
        )[:n]
        boot_means[b] = np.mean(sample)

    alpha = 1 - ci
    lo = float(np.percentile(boot_means, alpha / 2 * 100))
    hi = float(np.percentile(boot_means, (1 - alpha / 2) * 100))

    return {
        "ci_lower": round(lo, 4),
        "ci_upper": round(hi, 4),
        "ci_level": ci,
        "pct_positive": round(float(np.mean(boot_means > 0) * 100), 2),
    }


def _walk_forward_expanding(
    returns: np.ndarray,
    initial_train: int = 40,
    test_block: int = 25,
) -> list[dict]:
    """Expanding (anchored) walk-forward analysis on trade returns."""
    n = len(returns)
    windows = []
    train_end = initial_train

    while train_end < n:
        remaining = n - train_end
        if remaining < 5:
            break
        test_end = min(train_end + test_block, n)
        train_r = returns[:train_end]
        test_r = returns[train_end:test_end]

        windows.append({
            "window": len(windows) + 1,
            "train_n": train_end,
            "test_n": test_end - train_end,
            "train_mean_r": round(float(np.mean(train_r)), 4),
            "test_mean_r": round(float(np.mean(test_r)), 4),
            "train_pf": round(_pf(train_r), 2),
            "test_pf": round(_pf(test_r), 2),
            "train_wr": round(float(np.mean(train_r > 0)), 4),
            "test_wr": round(float(np.mean(test_r > 0)), 4),
        })
        train_end = test_end

    return windows


def run_cscv(
    r_returns: np.ndarray,
    n_combinations: int = 70,
) -> dict:
    """
    Full CSCV validation suite on trade returns.
    n_combinations controls the number of CSCV folds (C(8,4)=70 by default).
    """
    n = len(r_returns)
    if n < 16:
        return {
            "cscv": {"pbo": 0.0, "n_combos": 0, "error": "too few trades"},
            "psr": {"psr": 0.0},
            "bootstrap_ci": {"ci_lower": 0.0, "ci_upper": 0.0},
            "walk_forward": [],
        }

    # n_folds=8 gives C(8,4)=70 combos
    n_folds = 8

    cscv = _compute_cscv_pbo(r_returns, n_folds)
    psr = _compute_psr(r_returns, benchmark=0.0)
    bootstrap = _block_bootstrap_ci(r_returns)

    wf_init = max(20, n // 5)
    wf_block = max(10, n // 8)
    wf = _walk_forward_expanding(r_returns, wf_init, wf_block)

    return {
        "cscv": cscv,
        "psr": psr,
        "bootstrap_ci": bootstrap,
        "walk_forward": wf,
    }


# ---------------------------------------------------------------------------
# ECE (expected calibration error)
# ---------------------------------------------------------------------------
def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error.
    Measures how well predicted probabilities match actual frequencies.
    Lower is better. ECE < 0.05 is well-calibrated.
    """
    mask = ~np.isnan(probs)
    probs = probs[mask]
    labels = labels[mask]

    if len(probs) == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (probs >= lo) & (probs < hi)
        if i == n_bins - 1:
            in_bin = (probs >= lo) & (probs <= hi)

        n_in_bin = int(in_bin.sum())
        if n_in_bin == 0:
            continue

        avg_confidence = float(probs[in_bin].mean())
        avg_accuracy = float(labels[in_bin].mean())
        ece += (n_in_bin / len(probs)) * abs(avg_accuracy - avg_confidence)

    return round(ece, 6)


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------
DEFAULT_GATES = {
    "MIN_TRADES_PER_YEAR": 100,
    "MIN_OOS_AUC": 0.75,
    "MAX_PBO": 0.05,
    "MIN_PSR": 0.99,
    "MIN_WF_WINDOWS": "all",
    "MIN_SHARPE": 2.0,
    "MIN_WR": 0.55,
    "MIN_EV_R": 0.50,
    "MAX_DRAWDOWN": 0.20,
    "MAX_ECE": 0.05,
    "AUC_PROMOTION_DELTA": 0.005,
    "LOGLOSS_MUST_IMPROVE": True,
}


def compute_gates(results: dict, gates: dict | None = None) -> dict:
    """
    Evaluate pass/fail for each gate given experiment results.

    results keys expected:
        trades_per_yr, oos_auc, pbo, psr, wf_all_profitable,
        sharpe_ann, win_rate, ev_r, max_dd_pct, ece,
        auc_delta (optional), logloss_improved (optional)

    Returns dict of {gate_name: {"threshold": ..., "value": ..., "pass": bool}}.
    """
    g = {**DEFAULT_GATES, **(gates or {})}
    out = {}

    def _check(name, value, threshold, op):
        if op == ">=":
            passed = value >= threshold
        elif op == "<=":
            passed = value <= threshold
        elif op == "==":
            passed = value == threshold
        else:
            passed = False
        out[name] = {
            "threshold": threshold, "value": value, "pass": bool(passed),
        }

    _check("MIN_TRADES_PER_YEAR",
           results.get("trades_per_yr", 0), g["MIN_TRADES_PER_YEAR"], ">=")
    _check("MIN_OOS_AUC",
           results.get("oos_auc", 0), g["MIN_OOS_AUC"], ">=")
    _check("MAX_PBO",
           results.get("pbo", 1.0), g["MAX_PBO"], "<=")
    _check("MIN_PSR",
           results.get("psr", 0), g["MIN_PSR"], ">=")

    # Walk-forward: "all" means all windows must be profitable
    wf_val = results.get("wf_all_profitable", False)
    if g["MIN_WF_WINDOWS"] == "all":
        out["MIN_WF_WINDOWS"] = {
            "threshold": "all", "value": wf_val, "pass": bool(wf_val),
        }
    else:
        out["MIN_WF_WINDOWS"] = {
            "threshold": g["MIN_WF_WINDOWS"],
            "value": wf_val,
            "pass": bool(wf_val),
        }

    _check("MIN_SHARPE",
           results.get("sharpe_ann", 0), g["MIN_SHARPE"], ">=")
    _check("MIN_WR",
           results.get("win_rate", 0), g["MIN_WR"], ">=")
    _check("MIN_EV_R",
           results.get("ev_r", 0), g["MIN_EV_R"], ">=")
    _check("MAX_DRAWDOWN",
           results.get("max_dd_pct", 100) / 100, g["MAX_DRAWDOWN"], "<=")
    _check("MAX_ECE",
           results.get("ece", 1.0), g["MAX_ECE"], "<=")

    # Optional promotion gates
    if "auc_delta" in results:
        _check("AUC_PROMOTION_DELTA",
               results["auc_delta"], g["AUC_PROMOTION_DELTA"], ">=")
    if "logloss_improved" in results and g["LOGLOSS_MUST_IMPROVE"]:
        out["LOGLOSS_MUST_IMPROVE"] = {
            "threshold": True,
            "value": results["logloss_improved"],
            "pass": bool(results["logloss_improved"]),
        }

    return out
