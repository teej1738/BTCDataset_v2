# core/engine/shap_runner.py
# SHAP analysis on experiment models via LightGBM pred_contrib.
# D41b -- runs after a gate-passing experiment to update knowledge.md.
#
# Usage:
#   python -m core.engine.shap_runner --exp-id E001_rq1
#   python -m core.engine.shap_runner                     # latest experiment
#
# Since evaluator.walk_forward_train() does not save models, this script
# retrains the same walk-forward folds with pred_contrib=True to extract
# SHAP values. This is the same approach as legacy/scripts/shap_analysis_v2.py.
#
# ASCII-only output for cp1252 compatibility.

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.engine.evaluator import compute_auc
from core.engine.simulator import (
    DATA_FILES,
    REGISTRY_PATH,
    _read_registry,
    augment_features,
    load_data,
    select_features,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
KNOWLEDGE_PATH = PROJECT_ROOT / "core" / "signals" / "ict" / "knowledge.md"
SHAP_DIR = PROJECT_ROOT / "core" / "experiments" / "shap"
SHAP_BASELINE = PROJECT_ROOT / "legacy" / "scripts" / "results" / "shap_analysis_v2.json"

# Prune threshold: features below this mean |SHAP| are pruning candidates
PRUNE_THRESHOLD = 0.010

# ---------------------------------------------------------------------------
# Regime-dependent features (keep always regardless of SHAP rank)
# ---------------------------------------------------------------------------
REGIME_DEPENDENT = {
    "m30_cvd_zscore", "h4_close", "liq_nearest_below",
    "ict_dr_eq", "ict_fvg_bear_nearest_bot", "m15_ict_ob_bull_top",
}


# ---------------------------------------------------------------------------
# Walk-forward SHAP extraction
# ---------------------------------------------------------------------------
def walk_forward_shap(
    df: pd.DataFrame,
    features: list[str],
    label_col: str,
    config: dict,
) -> tuple[np.ndarray, list[dict], list[np.ndarray]]:
    """
    Walk-forward training with SHAP extraction per fold.

    Uses the same fold boundaries as evaluator.walk_forward_train() to ensure
    consistency with the experiment's OOS probs.

    Returns:
        oos_probs: array of shape (n,) with NaN where not covered
        fold_results: list of per-fold metric dicts
        shap_per_fold: list of arrays, each shape (n_test, n_features)
    """
    import lightgbm as lgb

    n = len(df)
    raw_labels = df[label_col].values
    label_valid = ~np.isnan(raw_labels)
    label_arr = np.where(label_valid, raw_labels, 0).astype(int)
    X = df[features]

    embargo = config.get("embargo_bars", 48)
    min_train = config.get("min_train_bars", 105_000)
    test_fold = config.get("test_fold_bars", 52_500)
    val_frac = config.get("val_frac", 0.10)
    n_estimators = config.get("n_estimators", 1000)
    early_stop = config.get("early_stop_rounds", 50)

    device = config.get("device", "gpu")
    lgb_params = config.get("lgb_params", {}).copy()

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

    # GPU probe
    if device == "gpu":
        try:
            _probe = lgb.Dataset(
                X.iloc[:100], label=label_arr[:100], free_raw_data=False
            )
            _m = lgb.train(
                {**lgb_params, "num_iterations": 1, "verbosity": -1},
                _probe,
            )
            del _m, _probe
            print("  SHAP training device: gpu")
        except Exception:
            lgb_params["device"] = "cpu"
            print("  SHAP training device: cpu (fallback)")
    else:
        print("  SHAP training device: cpu")

    oos_probs = np.full(n, np.nan)
    fold_results = []
    shap_per_fold = []

    # Define fold boundaries (same as evaluator.py)
    folds = []
    t_start = min_train + embargo
    while t_start < n:
        t_end = min(t_start + test_fold, n)
        folds.append((t_start, t_end))
        t_start = t_end

    print(f"  Walk-forward SHAP: {len(folds)} folds, {len(features)} features")

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

        # Predict
        probs = model.predict(X_test)
        oos_probs[test_idx] = probs

        # SHAP values via pred_contrib
        # Returns shape (n_test, n_features + 1) -- last col is bias
        shap_raw = model.predict(X_test, pred_contrib=True)
        shap_vals = shap_raw[:, :-1]  # drop bias column
        shap_per_fold.append(shap_vals)

        # Fold metrics (exclude NaN-label rows)
        test_valid = label_valid[test_idx]
        y_eval = y_test[test_valid]
        p_eval = probs[test_valid]

        auc = compute_auc(y_eval, p_eval) if len(y_eval) > 0 else 0.5
        best_iter = model.best_iteration
        if best_iter <= 0:
            best_iter = n_estimators

        fold_results.append({
            "fold": fold_i + 1,
            "train_bars": len(train_idx),
            "test_bars": len(test_idx),
            "auc": round(auc, 4),
            "best_iteration": best_iter,
            "shap_mean_abs": round(float(np.abs(shap_vals).mean()), 6),
        })

        print(f"    Fold {fold_i + 1}/{len(folds)}: "
              f"AUC={auc:.4f}  |SHAP| mean={np.abs(shap_vals).mean():.6f}")

    return oos_probs, fold_results, shap_per_fold


# ---------------------------------------------------------------------------
# SHAP aggregation
# ---------------------------------------------------------------------------
def aggregate_shap(
    features: list[str],
    shap_per_fold: list[np.ndarray],
) -> list[dict]:
    """
    Aggregate per-fold SHAP into per-feature records sorted by mean |SHAP|.

    Returns list of dicts with keys: feature, mean_abs_shap, cv, rank
    """
    n_features = len(features)
    n_folds = len(shap_per_fold)

    # Per-fold mean |SHAP| for each feature
    fold_mean_abs = np.zeros((n_folds, n_features))
    for fi, shap_vals in enumerate(shap_per_fold):
        fold_mean_abs[fi, :] = np.abs(shap_vals).mean(axis=0)

    # Global mean |SHAP| across all folds (weighted by test size)
    all_shap = np.vstack(shap_per_fold)
    global_mean_abs = np.abs(all_shap).mean(axis=0)

    # Stability: CV = std / mean across folds
    fold_stds = fold_mean_abs.std(axis=0)
    fold_means = fold_mean_abs.mean(axis=0)
    cv = np.where(fold_means > 0, fold_stds / fold_means, 0.0)

    records = []
    for i, feat in enumerate(features):
        records.append({
            "feature": feat,
            "mean_abs_shap": round(float(global_mean_abs[i]), 6),
            "cv": round(float(cv[i]), 4),
        })

    # Sort by mean |SHAP| descending
    records.sort(key=lambda r: r["mean_abs_shap"], reverse=True)

    # Assign rank
    for i, rec in enumerate(records):
        rec["rank"] = i + 1

    return records


# ---------------------------------------------------------------------------
# Compare with previous SHAP run
# ---------------------------------------------------------------------------
def load_previous_top30(exp_id: str | None = None) -> list[dict]:
    """
    Load previous top-30 SHAP features for delta comparison.

    Priority: most recent shap_[exp_id].json -> baseline shap_analysis_v2.json
    """
    # Check experiment-specific SHAP files
    if SHAP_DIR.exists():
        shap_files = sorted(SHAP_DIR.glob("shap_*.json"), reverse=True)
        for sf in shap_files:
            # Skip the current experiment
            if exp_id and exp_id in sf.stem:
                continue
            try:
                with open(sf, "r") as f:
                    data = json.load(f)
                return data.get("top30", [])
            except (json.JSONDecodeError, KeyError):
                continue

    # Fallback: baseline SHAP JSON
    if SHAP_BASELINE.exists():
        try:
            with open(SHAP_BASELINE, "r") as f:
                data = json.load(f)
            top50 = data.get("top50", [])
            return top50[:30]
        except (json.JSONDecodeError, KeyError):
            pass

    return []


def compute_deltas(
    current_top30: list[dict],
    previous_top30: list[dict],
) -> dict:
    """Compare current vs previous top-30 features."""
    current_names = {r["feature"] for r in current_top30}
    previous_names = {r.get("feature", "") for r in previous_top30}

    entered = current_names - previous_names
    left = previous_names - current_names

    return {
        "entered_top30": sorted(entered),
        "left_top30": sorted(left),
        "n_entered": len(entered),
        "n_left": len(left),
        "stable": len(current_names & previous_names),
    }


# ---------------------------------------------------------------------------
# Feature pruning
# ---------------------------------------------------------------------------
def compute_prune_list(
    records: list[dict],
    threshold: float = PRUNE_THRESHOLD,
    regime_keep: set[str] | None = None,
) -> list[str]:
    """
    Return feature names below threshold that are safe to prune.
    Excludes regime-dependent features.
    """
    keep = regime_keep or REGIME_DEPENDENT
    prune = []
    for rec in records:
        if rec["mean_abs_shap"] < threshold and rec["feature"] not in keep:
            prune.append(rec["feature"])
    return prune


# ---------------------------------------------------------------------------
# Save SHAP output
# ---------------------------------------------------------------------------
def save_shap_json(
    exp_id: str,
    records: list[dict],
    fold_results: list[dict],
    oos_auc: float,
    deltas: dict,
    prune_list: list[str],
) -> Path:
    """Save full SHAP output to core/experiments/shap/shap_[exp_id].json."""
    SHAP_DIR.mkdir(parents=True, exist_ok=True)

    top30 = records[:30]
    bottom30 = records[-30:]

    output = {
        "exp_id": exp_id,
        "n_features": len(records),
        "oos_auc": round(oos_auc, 4),
        "n_folds": len(fold_results),
        "fold_results": fold_results,
        "top30": [{k: v for k, v in r.items()} for r in top30],
        "bottom30": [{k: v for k, v in r.items()} for r in bottom30],
        "deltas": deltas,
        "prune_candidates": {
            "threshold": PRUNE_THRESHOLD,
            "n_prune": len(prune_list),
            "features": prune_list,
            "regime_protected": sorted(REGIME_DEPENDENT),
        },
        "all_features": [{
            "feature": r["feature"],
            "mean_abs_shap": r["mean_abs_shap"],
            "rank": r["rank"],
            "cv": r["cv"],
        } for r in records],
    }

    path = SHAP_DIR / f"shap_{exp_id}.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Print report
# ---------------------------------------------------------------------------
def print_report(
    exp_id: str,
    records: list[dict],
    oos_auc: float,
    deltas: dict,
    prune_list: list[str],
) -> None:
    """Print SHAP analysis report. ASCII only."""
    sep = "=" * 60
    rule = "-" * 60

    print()
    print(sep)
    print(f"  SHAP ANALYSIS: {exp_id}")
    print(sep)
    print(f"  Features:       {len(records)}")
    print(f"  OOS AUC:        {oos_auc:.4f}")
    print()

    # Top 30
    print("  TOP 30 FEATURES")
    print(rule)
    print(f"  {'Rank':>4}  {'Feature':<40}  {'|SHAP|':>10}  {'CV':>5}")
    for rec in records[:30]:
        print(f"  {rec['rank']:>4}  {rec['feature']:<40}  "
              f"{rec['mean_abs_shap']:>10.6f}  {rec['cv']:>5.2f}")
    print()

    # Deltas
    if deltas.get("n_entered", 0) > 0 or deltas.get("n_left", 0) > 0:
        print("  CHANGES vs PREVIOUS RUN")
        print(rule)
        if deltas["entered_top30"]:
            print(f"  ENTERED top 30: {deltas['entered_top30']}")
        if deltas["left_top30"]:
            print(f"  LEFT top 30:    {deltas['left_top30']}")
        print(f"  Stable:         {deltas['stable']}/30")
        print()
    else:
        print("  No previous SHAP run to compare against.")
        print()

    # Pruning
    print(f"  PRUNING CANDIDATES (|SHAP| < {PRUNE_THRESHOLD})")
    print(rule)
    print(f"  Total prune candidates: {len(prune_list)}")
    print(f"  Regime-protected:       {len(REGIME_DEPENDENT)} features")
    if prune_list[:10]:
        for feat in prune_list[:10]:
            rec = next((r for r in records if r["feature"] == feat), None)
            if rec:
                print(f"    #{rec['rank']:>3} {feat:<40}  "
                      f"|SHAP|={rec['mean_abs_shap']:.6f}")
        if len(prune_list) > 10:
            print(f"    ... and {len(prune_list) - 10} more")
    print()
    print(sep)


# ---------------------------------------------------------------------------
# Resolve experiment config
# ---------------------------------------------------------------------------
def get_experiment_config(exp_id: str | None) -> dict:
    """Get experiment config from registry, or use the latest."""
    registry = _read_registry()
    experiments = registry.get("experiments", [])

    if exp_id:
        for exp in experiments:
            if exp.get("id") == exp_id:
                return exp
        raise ValueError(f"Experiment '{exp_id}' not found in registry")

    # Use the most recent DONE experiment
    done = [e for e in experiments if e.get("status") == "DONE"]
    if not done:
        raise ValueError("No completed experiments in registry")
    return done[-1]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = sys.argv[1:]
    exp_id = None

    # Parse --exp-id
    for i, arg in enumerate(args):
        if arg == "--exp-id" and i + 1 < len(args):
            exp_id = args[i + 1]
            break

    print()
    print("=" * 60)
    print("  SHAP RUNNER (D41b)")
    print("=" * 60)

    # 1. Get experiment config from registry
    exp_config = get_experiment_config(exp_id)
    exp_id = exp_config.get("id", "unknown")
    print(f"\n  Experiment: {exp_id}")
    print(f"  Status:     {exp_config.get('status')}")
    print(f"  Label:      {exp_config.get('label')}")

    # 2. Load data + augment on-the-fly features
    df, data_version = load_data()
    df = augment_features(df)

    # 3. Resolve features (same as the experiment used)
    label_col = exp_config.get("label", "label_long_hit_2r_48c")
    feature_exclude = exp_config.get("feature_exclude")

    # If feature_exclude is not in registry, load from SHAP recommendations
    if feature_exclude is None:
        shap_json_path = (
            PROJECT_ROOT / "legacy" / "scripts" / "results" / "shap_analysis_v2.json"
        )
        if shap_json_path.exists():
            with open(shap_json_path, "r") as f:
                shap_data = json.load(f)
            feature_exclude = shap_data.get("recommendations", {}).get(
                "drop_features"
            )

    features_spec = exp_config.get("features", "all")
    # Handle n_features from registry (was already resolved at run time)
    features = select_features(df, features_spec, feature_exclude)

    # 4. Build ML config for retraining
    ml_config_raw = exp_config.get("ml_config", {})
    ml_config = {
        "embargo_bars": ml_config_raw.get("embargo_bars", 48),
        "device": ml_config_raw.get("device", "gpu"),
        "min_train_bars": ml_config_raw.get("min_train_bars", 105_000),
        "test_fold_bars": ml_config_raw.get("test_fold_bars", 52_500),
        "val_frac": ml_config_raw.get("val_frac", 0.10),
        "n_estimators": ml_config_raw.get("n_estimators", 1000),
        "early_stop_rounds": ml_config_raw.get("early_stop_rounds", 50),
        "lgb_params": ml_config_raw.get("lgb_params", {}),
    }

    # 5. Run walk-forward SHAP
    print(f"\n  Retraining {len(features)} features with pred_contrib=True ...")
    oos_probs, fold_results, shap_per_fold = walk_forward_shap(
        df, features, label_col, ml_config,
    )

    # 6. Compute OOS AUC
    covered = ~np.isnan(oos_probs)
    raw_labels = df[label_col].values
    label_valid = ~np.isnan(raw_labels)
    label_arr = np.where(label_valid, raw_labels, 0).astype(int)
    eval_mask = covered & label_valid
    oos_auc = compute_auc(label_arr[eval_mask], oos_probs[eval_mask])
    print(f"\n  OOS AUC: {oos_auc:.4f}")

    # 7. Aggregate SHAP
    records = aggregate_shap(features, shap_per_fold)

    # 8. Compare with previous
    previous_top30 = load_previous_top30(exp_id)
    deltas = compute_deltas(records[:30], previous_top30)

    # 9. Compute prune list
    prune_list = compute_prune_list(records)

    # 10. Print report
    print_report(exp_id, records, oos_auc, deltas, prune_list)

    # 11. Save SHAP JSON
    save_shap_json(exp_id, records, fold_results, oos_auc, deltas, prune_list)

    # 12. Summary for optimizer
    print(f"\n  Top 5 features:")
    for rec in records[:5]:
        print(f"    #{rec['rank']} {rec['feature']} "
              f"(|SHAP|={rec['mean_abs_shap']:.6f})")

    if deltas.get("entered_top30"):
        print(f"\n  NEW in top 30: {deltas['entered_top30']}")
    if deltas.get("left_top30"):
        print(f"  DROPPED from top 30: {deltas['left_top30']}")

    print(f"\n  Prune candidates: {len(prune_list)} features")
    print(f"  Next: update knowledge.md Section 1 with these findings.")
    print()


if __name__ == "__main__":
    main()
