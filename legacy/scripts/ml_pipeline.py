"""
ml_pipeline.py -- LightGBM Walk-Forward ML Pipeline (Step 6)
============================================================
Replaces binary Config B filters with continuous probability scores
using gradient-boosted trees. Walk-forward expanding window with
purging/embargo ensures no data leakage.

Features: all numeric master columns + regime_label (~390 features)
Label: label_long_hit_2r_48c (long-only, 2R target, 48-bar horizon)
Model: LightGBM native API (no sklearn)
Validation: walk-forward expanding window, 48-bar embargo

Output:
  - OOS probability scores per bar
  - Feature importance (gain, averaged across folds)
  - Precision/recall/WR/EV at various thresholds
  - Comparison with Config B binary filter
  - results/ml_pipeline.json
  - results/ml_pipeline.html
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from baseline_backtest_v2 import Config, load_labeled
from mtf_signals import TF_CONFIGS, config_b_filters

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"


# ---- config ----------------------------------------------------------------
@dataclass
class MLConfig:
    label_col: str = "label_long_hit_2r_48c"
    embargo_bars: int = 48            # 48 x 5m = 4h purge gap
    min_train_bars: int = 105_000     # ~1 year of 5m bars
    test_fold_bars: int = 52_500      # ~6 months of 5m bars
    val_frac: float = 0.10            # 10% of training for early stopping
    n_estimators: int = 1000          # max boosting rounds
    early_stop_rounds: int = 50       # patience for early stopping
    r_target: int = 2                 # for EV calculation

    lgb_params: dict = field(default_factory=lambda: {
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
    })


# ---- feature selection ------------------------------------------------------
def select_features(df: pd.DataFrame) -> list[str]:
    """Select numeric feature columns, excluding labels and metadata."""
    exclude_prefixes = ("label_", "bar_", "meta_")
    features = []
    for col in df.columns:
        if any(col.startswith(p) for p in exclude_prefixes):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if df[col].nunique(dropna=True) <= 1:
            continue
        features.append(col)
    return sorted(features)


# ---- AUC (no sklearn) ------------------------------------------------------
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


# ---- data preparation ------------------------------------------------------
def prepare_data(
    cfg: Config, ml_cfg: MLConfig
) -> tuple[pd.DataFrame, list[str]]:
    """Load data, select features, prepare for training."""
    df = load_labeled(cfg)
    df = df.reset_index(drop=True)

    if "regime_label" not in df.columns:
        print("  WARNING: regime_label not found -- run regime_classifier.py first")
        df["regime_label"] = -1

    features = select_features(df)
    print(f"  Selected {len(features)} features")

    n_before = len(df)
    df = df.dropna(subset=[ml_cfg.label_col])
    df = df.reset_index(drop=True)
    n_after = len(df)
    if n_before != n_after:
        print(f"  Dropped {n_before - n_after:,} rows with NaN in {ml_cfg.label_col}")

    df[ml_cfg.label_col] = df[ml_cfg.label_col].astype(int)
    pos_rate = df[ml_cfg.label_col].mean()
    print(f"  Rows: {len(df):,}  |  Positive rate: {pos_rate:.4f}")

    return df, features


# ---- walk-forward training --------------------------------------------------
def walk_forward_train(
    df: pd.DataFrame,
    features: list[str],
    ml_cfg: MLConfig,
) -> tuple[np.ndarray, list[dict], list[dict]]:
    """
    Walk-forward expanding window with purging/embargo.
    Early stopping uses a held-out 10% validation slice from training data
    (with its own embargo gap) so the test set is never seen during training.
    """
    n = len(df)
    label = df[ml_cfg.label_col].values
    X = df[features]

    oos_probs = np.full(n, np.nan)
    fold_results = []
    importance_list = []

    # Define fold boundaries
    folds = []
    t_start = ml_cfg.min_train_bars + ml_cfg.embargo_bars
    while t_start < n:
        t_end = min(t_start + ml_cfg.test_fold_bars, n)
        folds.append((t_start, t_end))
        t_start = t_end

    print(f"\n  Walk-forward: {len(folds)} folds")
    print(f"  Min train: {ml_cfg.min_train_bars:,} bars  "
          f"Test fold: {ml_cfg.test_fold_bars:,} bars  "
          f"Embargo: {ml_cfg.embargo_bars} bars")

    for fold_i, (test_start, test_end) in enumerate(folds):
        # All data before (test_start - embargo) is available for training
        avail_end = test_start - ml_cfg.embargo_bars

        # Split available data: last val_frac for early stopping validation
        val_size = max(int(avail_end * ml_cfg.val_frac), 1000)
        val_start = avail_end - val_size
        pure_train_end = val_start - ml_cfg.embargo_bars

        train_idx = np.arange(0, pure_train_end)
        val_idx = np.arange(val_start, avail_end)
        test_idx = np.arange(test_start, test_end)

        X_train = X.iloc[train_idx]
        y_train = label[train_idx]
        X_val = X.iloc[val_idx]
        y_val = label[val_idx]
        X_test = X.iloc[test_idx]
        y_test = label[test_idx]

        # Create LightGBM datasets
        dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        dval = lgb.Dataset(X_val, label=y_val, free_raw_data=False,
                           reference=dtrain)

        # Train with early stopping on validation set
        callbacks = [
            lgb.early_stopping(ml_cfg.early_stop_rounds, verbose=False),
            lgb.log_evaluation(period=10000),   # effectively suppress logs
        ]

        model = lgb.train(
            ml_cfg.lgb_params,
            dtrain,
            num_boost_round=ml_cfg.n_estimators,
            valid_sets=[dtrain, dval],
            valid_names=["train", "val"],
            callbacks=callbacks,
        )

        # Predict on test set (uses best_iteration automatically)
        probs = model.predict(X_test)
        oos_probs[test_idx] = probs

        # Fold metrics
        pos_rate = float(y_test.mean())
        logloss = float(-np.mean(
            y_test * np.log(np.clip(probs, 1e-10, 1))
            + (1 - y_test) * np.log(np.clip(1 - probs, 1e-10, 1))
        ))
        auc = compute_auc(y_test, probs)
        best_iter = model.best_iteration
        if best_iter <= 0:
            best_iter = ml_cfg.n_estimators

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

        # Feature importance (gain-based)
        imp = model.feature_importance(importance_type="gain")
        importance_list.append(dict(zip(features, imp.tolist())))

        print(f"    Fold {fold_i+1}/{len(folds)}: "
              f"train={len(train_idx):,} val={len(val_idx):,} "
              f"test={len(test_idx):,}  "
              f"logloss={logloss:.4f} AUC={auc:.4f} "
              f"iter={best_iter}")

    return oos_probs, fold_results, importance_list


# ---- threshold metrics ------------------------------------------------------
def compute_threshold_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    r_target: int = 2,
    years: float = 5.0,
) -> list[dict]:
    """Precision / recall / WR / EV at probability thresholds."""
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
                  0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    total_pos = int(y_true.sum())
    results = []

    for t in thresholds:
        pred = y_prob >= t
        n_pred = int(pred.sum())

        if n_pred == 0:
            results.append({
                "threshold": t, "n_signals": 0, "precision": 0,
                "recall": 0, "wr": 0, "ev_r": 0, "pf": 0,
                "signals_per_year": 0,
            })
            continue

        tp = int((pred & (y_true == 1)).sum())
        fp = n_pred - tp
        precision = tp / n_pred
        recall = tp / total_pos if total_pos > 0 else 0
        wr = precision
        ev = wr * r_target - (1 - wr) * 1.0
        pf = (tp * r_target) / (fp * 1.0) if fp > 0 else float("inf")

        results.append({
            "threshold": t,
            "n_signals": n_pred,
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "wr": round(float(wr), 4),
            "ev_r": round(float(ev), 4),
            "pf": round(float(pf), 4),
            "signals_per_year": round(n_pred / years, 1),
        })

    return results


# ---- feature importance aggregation -----------------------------------------
def aggregate_importance(
    importance_list: list[dict], top_n: int = 30
) -> list[tuple[str, float]]:
    """Average feature importance (gain) across folds, return top N."""
    all_feats: set[str] = set()
    for imp in importance_list:
        all_feats.update(imp.keys())

    avg = {}
    for f in all_feats:
        vals = [imp.get(f, 0) for imp in importance_list]
        avg[f] = float(np.mean(vals))

    return sorted(avg.items(), key=lambda x: x[1], reverse=True)[:top_n]


# ---- Config B baseline for comparison ---------------------------------------
def config_b_baseline(
    df: pd.DataFrame, cfg: Config, ml_cfg: MLConfig
) -> dict:
    """Config B MTF long-only results on the full dataset."""
    combined_long = pd.Series(False, index=df.index)
    for tf in TF_CONFIGS:
        long_mask, _ = config_b_filters(df, tf, cfg)
        combined_long = combined_long | long_mask

    n = int(combined_long.sum())
    if n == 0:
        return {"n": 0, "wr": 0, "ev_r": 0}

    wins = int(df.loc[combined_long, ml_cfg.label_col].sum())
    wr = wins / n
    ev = wr * ml_cfg.r_target - (1 - wr) * 1.0
    return {"n": n, "wr": round(wr, 4), "ev_r": round(ev, 4)}


# ---- plotly chart -----------------------------------------------------------
def save_chart(
    threshold_metrics: list[dict],
    config_b: dict,
    top_features: list[tuple[str, float]],
) -> Path | None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return None

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Win Rate vs Threshold",
            "Signals per Year vs Threshold",
            "EV (R) vs Threshold",
            "Top 20 Feature Importances",
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.12,
    )

    active = [m for m in threshold_metrics if m["n_signals"] > 0]
    ts = [m["threshold"] for m in active]
    wrs = [m["wr"] * 100 for m in active]
    sigs = [m["signals_per_year"] for m in active]
    evs = [m["ev_r"] for m in active]

    # Row 1 Col 1: WR
    fig.add_trace(
        go.Scatter(x=ts, y=wrs, mode="lines+markers",
                   line=dict(color="cyan"), name="ML WR"),
        row=1, col=1,
    )
    fig.add_hline(y=config_b["wr"] * 100, line_dash="dash", line_color="yellow",
                  annotation_text=f"Config B ({config_b['wr']:.1%})",
                  row=1, col=1)
    fig.add_hline(y=33.33, line_dash="dot", line_color="red",
                  annotation_text="BE", row=1, col=1)

    # Row 1 Col 2: Signals/yr
    fig.add_trace(
        go.Scatter(x=ts, y=sigs, mode="lines+markers",
                   line=dict(color="limegreen"), name="Signals/yr"),
        row=1, col=2,
    )
    fig.add_hline(y=config_b["n"] / 6.16, line_dash="dash", line_color="yellow",
                  annotation_text="Config B", row=1, col=2)

    # Row 2 Col 1: EV
    fig.add_trace(
        go.Scatter(x=ts, y=evs, mode="lines+markers",
                   line=dict(color="orange"), name="EV (R)"),
        row=2, col=1,
    )
    fig.add_hline(y=config_b["ev_r"], line_dash="dash", line_color="yellow",
                  annotation_text=f"Config B ({config_b['ev_r']:+.3f}R)",
                  row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="red", row=2, col=1)

    # Row 2 Col 2: Feature importance
    top20 = top_features[:20]
    names = [f[0][:30] for f in reversed(top20)]
    vals = [f[1] for f in reversed(top20)]
    fig.add_trace(
        go.Bar(y=names, x=vals, orientation="h",
               marker_color="dodgerblue", showlegend=False),
        row=2, col=2,
    )

    fig.update_layout(
        template="plotly_dark",
        title="ML Pipeline -- Walk-Forward LightGBM (OOS Results)",
        height=900,
        showlegend=False,
    )
    fig.update_xaxes(title_text="Threshold", row=1, col=1)
    fig.update_yaxes(title_text="Win Rate %", row=1, col=1)
    fig.update_xaxes(title_text="Threshold", row=1, col=2)
    fig.update_yaxes(title_text="Signals/Year", row=1, col=2)
    fig.update_xaxes(title_text="Threshold", row=2, col=1)
    fig.update_yaxes(title_text="EV (R)", row=2, col=1)
    fig.update_xaxes(title_text="Gain", row=2, col=2)

    path = RESULTS_DIR / "ml_pipeline.html"
    fig.write_html(str(path))
    return path


# ---- console report --------------------------------------------------------
def print_report(
    df: pd.DataFrame,
    features: list[str],
    oos_probs: np.ndarray,
    fold_results: list[dict],
    threshold_metrics: list[dict],
    top_features: list[tuple[str, float]],
    config_b: dict,
    ml_cfg: MLConfig,
) -> dict:
    """Print results and save outputs."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    sep = "=" * 70
    rule = "-" * 70

    print(f"\n{sep}")
    print("  ML PIPELINE -- Walk-Forward LightGBM (Step 6)")
    print(sep)
    print(f"  Label: {ml_cfg.label_col}")
    print(f"  Features: {len(features)}")
    print(f"  Folds: {len(fold_results)}")
    print(f"  Embargo: {ml_cfg.embargo_bars} bars (4h)")

    # -- fold summary --
    print(f"\n{rule}")
    print("  FOLD RESULTS")
    print(rule)
    print(f"  {'Fold':>4}  {'Train':>8}  {'Val':>7}  {'Test':>7}  "
          f"{'PosRate':>8}  {'LogLoss':>8}  {'AUC':>6}  {'Iter':>5}")
    for fr in fold_results:
        print(f"  {fr['fold']:>4}  {fr['train_bars']:>8,}  "
              f"{fr['val_bars']:>7,}  {fr['test_bars']:>7,}  "
              f"{fr['test_pos_rate']:>8.4f}  {fr['logloss']:>8.4f}  "
              f"{fr['auc']:>6.4f}  {fr['best_iteration']:>5}")

    avg_ll = np.mean([f["logloss"] for f in fold_results])
    avg_auc = np.mean([f["auc"] for f in fold_results])
    print(f"\n  Mean OOS logloss: {avg_ll:.4f}  |  Mean OOS AUC: {avg_auc:.4f}")

    # -- threshold analysis --
    print(f"\n{rule}")
    print("  THRESHOLD ANALYSIS (OOS predictions)")
    print(rule)
    print(f"  {'Thresh':>6}  {'Signals':>8}  {'Sig/Yr':>7}  {'WR':>7}  "
          f"{'EV(R)':>8}  {'PF':>6}  {'Recall':>7}")
    for tm in threshold_metrics:
        if tm["n_signals"] == 0:
            continue
        print(f"  {tm['threshold']:>6.2f}  {tm['n_signals']:>8,}  "
              f"{tm['signals_per_year']:>7.1f}  {tm['wr']:>6.2%}  "
              f"{tm['ev_r']:>+8.4f}  {tm['pf']:>6.2f}  "
              f"{tm['recall']:>6.2%}")

    # -- Config B comparison --
    print(f"\n{rule}")
    print("  CONFIG B COMPARISON")
    print(rule)
    cb_per_yr = config_b["n"] / 6.16
    print(f"  Config B MTF longs: n={config_b['n']}, "
          f"WR={config_b['wr']:.2%}, EV={config_b['ev_r']:+.4f}R, "
          f"{cb_per_yr:.0f}/yr")

    # Best threshold that beats Config B WR with >= 100 signals
    best_match = None
    for tm in threshold_metrics:
        if tm["n_signals"] >= 100 and tm["wr"] >= config_b["wr"]:
            if best_match is None or (
                tm["signals_per_year"] > best_match["signals_per_year"]
            ):
                best_match = tm
    if best_match:
        print(f"\n  ML >= Config B WR ({config_b['wr']:.1%}) "
              f"at t={best_match['threshold']:.2f}:")
        print(f"    n={best_match['n_signals']:,}, "
              f"{best_match['signals_per_year']:.0f}/yr, "
              f"WR={best_match['wr']:.2%}, "
              f"EV={best_match['ev_r']:+.4f}R")
    else:
        print(f"\n  No threshold beats Config B WR with >=100 signals")

    # Best EV with >= 50 signals/year
    viable = [tm for tm in threshold_metrics
              if tm["signals_per_year"] >= 50 and tm["ev_r"] > 0]
    if viable:
        best_ev = max(viable, key=lambda x: x["ev_r"])
        print(f"\n  Best EV (>=50 sig/yr): t={best_ev['threshold']:.2f}, "
              f"n={best_ev['n_signals']:,}, "
              f"{best_ev['signals_per_year']:.0f}/yr, "
              f"WR={best_ev['wr']:.2%}, EV={best_ev['ev_r']:+.4f}R")

    # ML probability on Config B signal bars
    valid_mask = ~np.isnan(oos_probs)
    if valid_mask.any():
        cb_long = pd.Series(False, index=df.index)
        for tf in TF_CONFIGS:
            lm, _ = config_b_filters(df, tf, Config())
            cb_long = cb_long | lm
        cb_oos = cb_long.values & valid_mask
        if cb_oos.any():
            cb_probs = oos_probs[cb_oos]
            print(f"\n  ML scores on Config B signal bars "
                  f"({int(cb_oos.sum())} in OOS):")
            print(f"    mean={cb_probs.mean():.3f}  "
                  f"median={np.median(cb_probs):.3f}  "
                  f"std={cb_probs.std():.3f}")
            print(f"    pct >= 0.30: {(cb_probs >= 0.30).mean():.1%}  "
                  f"pct >= 0.50: {(cb_probs >= 0.50).mean():.1%}")

    # -- feature importance --
    print(f"\n{rule}")
    print("  TOP 20 FEATURES (gain, averaged across folds)")
    print(rule)
    for i, (feat, imp) in enumerate(top_features[:20], 1):
        print(f"  {i:>2}. {feat:<40} {imp:>10.1f}")

    # -- assessment --
    print(f"\n{rule}")
    print("  ASSESSMENT")
    print(rule)

    checks = []
    checks.append((
        "OOS AUC > 0.50 (better than random)",
        bool(avg_auc > 0.50),
        f"{avg_auc:.4f}",
    ))
    checks.append((
        "Viable threshold (EV>0, >=50/yr)",
        bool(len(viable) > 0),
        f"{len(viable)} thresholds",
    ))
    checks.append((
        "ML matches Config B WR at some threshold",
        best_match is not None,
        "Yes" if best_match else "No",
    ))

    for label, passed, val in checks:
        status = "PASS" if passed else "MISS"
        print(f"  [{status}]  {label:<50}  {val}")

    if all(c[1] for c in checks):
        print(f"\n  >>> ML model shows predictive value over random")
    else:
        print(f"\n  >>> ML results mixed -- investigate further")

    print(sep)

    # -- save chart --
    chart_path = save_chart(threshold_metrics, config_b, top_features)
    if chart_path:
        print(f"\n  Saved: {chart_path}")

    # -- save JSON --
    summary = {
        "config": {
            "label": ml_cfg.label_col,
            "n_features": len(features),
            "n_folds": len(fold_results),
            "embargo_bars": ml_cfg.embargo_bars,
            "min_train_bars": ml_cfg.min_train_bars,
            "test_fold_bars": ml_cfg.test_fold_bars,
            "lgb_params": {k: v for k, v in ml_cfg.lgb_params.items()},
        },
        "fold_results": fold_results,
        "avg_oos_logloss": round(float(avg_ll), 6),
        "avg_oos_auc": round(float(avg_auc), 4),
        "threshold_metrics": threshold_metrics,
        "config_b_baseline": config_b,
        "top_30_features": [
            {"feature": f, "importance": round(v, 2)}
            for f, v in top_features[:30]
        ],
        "oos_coverage": {
            "total_bars": int(len(oos_probs)),
            "oos_bars": int(valid_mask.sum()),
            "pct": round(float(valid_mask.mean() * 100), 1),
        },
    }

    json_path = RESULTS_DIR / "ml_pipeline.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved: {json_path}")

    return summary


# ---- main ------------------------------------------------------------------
def main() -> None:
    cfg = Config()
    ml_cfg = MLConfig()

    # 1. Prepare data
    print("=" * 70)
    print("  STEP 1: Data Preparation")
    print("=" * 70)
    df, features = prepare_data(cfg, ml_cfg)

    # 2. Config B baseline for comparison
    print(f"\n  Computing Config B baseline...")
    config_b = config_b_baseline(df, cfg, ml_cfg)
    print(f"  Config B: n={config_b['n']}, WR={config_b['wr']:.2%}, "
          f"EV={config_b['ev_r']:+.4f}R")

    # 3. Walk-forward training
    print(f"\n{'=' * 70}")
    print("  STEP 2: Walk-Forward Training")
    print("=" * 70)
    oos_probs, fold_results, importance_list = walk_forward_train(
        df, features, ml_cfg
    )

    # 4. Threshold metrics on OOS predictions
    valid = ~np.isnan(oos_probs)
    y_oos = df[ml_cfg.label_col].values[valid]
    p_oos = oos_probs[valid]

    n_oos = int(valid.sum())
    oos_years = n_oos / (288 * 365.25)
    print(f"\n  OOS: {n_oos:,} bars ({oos_years:.2f} years, "
          f"{valid.mean()*100:.1f}% coverage)")

    threshold_metrics = compute_threshold_metrics(
        y_oos, p_oos, ml_cfg.r_target, years=oos_years
    )

    # 5. Feature importance
    top_features = aggregate_importance(importance_list, top_n=30)

    # 6. Report
    print_report(
        df, features, oos_probs, fold_results, threshold_metrics,
        top_features, config_b, ml_cfg,
    )


if __name__ == "__main__":
    main()
