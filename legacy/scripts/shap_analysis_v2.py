"""
shap_analysis_v2.py -- SHAP Feature Importance Analysis on v2 Model (D36)
=========================================================================
Retrains the walk-forward LightGBM folds, computes SHAP values per fold
using LightGBM's native pred_contrib=True, and produces:

A. Top 50 features by mean |SHAP|
B. Bottom 50 features (pruning candidates)
C. Stability analysis (CV across folds)
D. Family summary (aggregate by feature family)
E. Ablation: retrain on top-408 features, compare OOS AUC

Output:
  - Console report
  - results/shap_analysis_v2.json
  - results/shap_analysis_v2.html
  - results/shap_top50.csv
  - results/shap_bottom50.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import yaml

from baseline_backtest_v2 import Config
from ml_pipeline import MLConfig, select_features, compute_auc
from ml_pipeline_v2 import prepare_data_v2

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
PROJECT_DIR = SCRIPT_DIR.parent
CATALOG_PATH = PROJECT_DIR / "data" / "labeled" / "feature_catalog_v2.yaml"


# -- Feature family classification -------------------------------------------
def load_catalog() -> dict:
    """Load feature catalog YAML -> {feature_name: {family, evidence_tier}}."""
    if not CATALOG_PATH.exists():
        return {}
    with open(CATALOG_PATH, "r") as f:
        raw = yaml.safe_load(f)
    catalog = {}
    for name, info in raw.items():
        catalog[name] = {
            "family": info.get("family", "Unknown"),
            "evidence_tier": info.get("evidence_tier", 0),
        }
    return catalog


def classify_family(feature: str, catalog: dict) -> tuple[str, int]:
    """Return (family, evidence_tier) for a feature."""
    # Direct catalog lookup (v2 features)
    if feature in catalog:
        return catalog[feature]["family"], catalog[feature]["evidence_tier"]

    # Classify v1 features by prefix
    if feature.startswith(("m15_", "m30_", "h1_", "h4_", "d1_")):
        base = feature.split("_", 1)[1]  # strip TF prefix
        if base in catalog:
            return f"HTF/{catalog[base]['family']}", catalog[base]["evidence_tier"]
        if "ict_" in feature:
            return "HTF/ICT", 2
        return "HTF/Other", 3

    if feature.startswith("ict_"):
        return "ICT/Core", 2
    if feature.startswith("sess_"):
        return "Session", 2
    if feature.startswith("fund_"):
        return "Raw/Funding", 2
    if feature.startswith(("mark_", "basis_")):
        return "Raw/Market", 3
    if feature.startswith("cvd_"):
        return "Volume/CVD", 2
    if feature.startswith("htf_"):
        return "HTF/Derived", 3
    if feature.startswith("liq_"):
        return "ICT/Liquidity", 2
    if feature.startswith("regime_"):
        return "Regime", 3

    # Price/OHLCV
    if feature in ("open", "high", "low", "close", "volume",
                    "taker_buy_volume", "taker_sell_volume",
                    "quote_volume", "trade_count", "close_time_ms"):
        return "Raw/OHLCV", 3

    return "Other", 3


# -- Walk-forward SHAP extraction -------------------------------------------
def walk_forward_shap(
    df: pd.DataFrame,
    features: list[str],
    ml_cfg: MLConfig,
) -> tuple[np.ndarray, list[dict], list[np.ndarray]]:
    """
    Walk-forward training with SHAP extraction per fold.
    Returns: (oos_probs, fold_results, shap_per_fold)
    where shap_per_fold[i] is shape (n_test, n_features) of SHAP values.
    """
    n = len(df)
    label = df[ml_cfg.label_col].values
    X = df[features]

    oos_probs = np.full(n, np.nan)
    fold_results = []
    shap_per_fold = []

    # Define fold boundaries (same as ml_pipeline.py)
    folds = []
    t_start = ml_cfg.min_train_bars + ml_cfg.embargo_bars
    while t_start < n:
        t_end = min(t_start + ml_cfg.test_fold_bars, n)
        folds.append((t_start, t_end))
        t_start = t_end

    print(f"\n  Walk-forward SHAP: {len(folds)} folds, {len(features)} features")

    for fold_i, (test_start, test_end) in enumerate(folds):
        avail_end = test_start - ml_cfg.embargo_bars
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

        dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        dval = lgb.Dataset(X_val, label=y_val, free_raw_data=False,
                           reference=dtrain)

        callbacks = [
            lgb.early_stopping(ml_cfg.early_stop_rounds, verbose=False),
            lgb.log_evaluation(period=10000),
        ]

        model = lgb.train(
            ml_cfg.lgb_params,
            dtrain,
            num_boost_round=ml_cfg.n_estimators,
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

        # Fold metrics
        logloss = float(-np.mean(
            y_test * np.log(np.clip(probs, 1e-10, 1))
            + (1 - y_test) * np.log(np.clip(1 - probs, 1e-10, 1))
        ))
        auc = compute_auc(y_test, probs)
        best_iter = model.best_iteration
        if best_iter <= 0:
            best_iter = ml_cfg.n_estimators

        fold_results.append({
            "fold": fold_i + 1,
            "train_bars": len(train_idx),
            "test_bars": len(test_idx),
            "auc": round(auc, 4),
            "logloss": round(logloss, 6),
            "best_iteration": best_iter,
            "shap_mean_abs": round(float(np.abs(shap_vals).mean()), 6),
        })

        print(f"    Fold {fold_i+1}/{len(folds)}: "
              f"AUC={auc:.4f}  |SHAP| mean={np.abs(shap_vals).mean():.6f}")

    return oos_probs, fold_results, shap_per_fold


# -- Ablation: retrain on reduced feature set --------------------------------
def ablation_retrain(
    df: pd.DataFrame,
    features_full: list[str],
    features_reduced: list[str],
    ml_cfg: MLConfig,
) -> tuple[float, float]:
    """Quick retrain on reduced features, return (auc_full, auc_reduced)."""
    n = len(df)
    label = df[ml_cfg.label_col].values

    folds = []
    t_start = ml_cfg.min_train_bars + ml_cfg.embargo_bars
    while t_start < n:
        t_end = min(t_start + ml_cfg.test_fold_bars, n)
        folds.append((t_start, t_end))
        t_start = t_end

    aucs_full = []
    aucs_reduced = []

    print(f"\n  Ablation: {len(features_full)} -> {len(features_reduced)} features")

    for fold_i, (test_start, test_end) in enumerate(folds):
        avail_end = test_start - ml_cfg.embargo_bars
        val_size = max(int(avail_end * ml_cfg.val_frac), 1000)
        val_start = avail_end - val_size
        pure_train_end = val_start - ml_cfg.embargo_bars

        train_idx = np.arange(0, pure_train_end)
        val_idx = np.arange(val_start, avail_end)
        test_idx = np.arange(test_start, test_end)

        y_train = label[train_idx]
        y_val = label[val_idx]
        y_test = label[test_idx]

        for feat_set, auc_list in [
            (features_full, aucs_full),
            (features_reduced, aucs_reduced),
        ]:
            X_tr = df[feat_set].iloc[train_idx]
            X_va = df[feat_set].iloc[val_idx]
            X_te = df[feat_set].iloc[test_idx]

            dtrain = lgb.Dataset(X_tr, label=y_train, free_raw_data=False)
            dval = lgb.Dataset(X_va, label=y_val, free_raw_data=False,
                               reference=dtrain)
            callbacks = [
                lgb.early_stopping(ml_cfg.early_stop_rounds, verbose=False),
                lgb.log_evaluation(period=10000),
            ]
            model = lgb.train(
                ml_cfg.lgb_params, dtrain,
                num_boost_round=ml_cfg.n_estimators,
                valid_sets=[dtrain, dval],
                valid_names=["train", "val"],
                callbacks=callbacks,
            )
            probs = model.predict(X_te)
            auc_list.append(compute_auc(y_test, probs))

        print(f"    Fold {fold_i+1}/{len(folds)}: "
              f"full AUC={aucs_full[-1]:.4f}  "
              f"reduced AUC={aucs_reduced[-1]:.4f}  "
              f"delta={aucs_reduced[-1]-aucs_full[-1]:+.4f}")

    return float(np.mean(aucs_full)), float(np.mean(aucs_reduced))


# -- Chart -------------------------------------------------------------------
def save_chart(
    top50: list[dict],
    family_summary: list[dict],
    stability: list[dict],
) -> Path | None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return None

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Top 30 Features by Mean |SHAP|",
            "Feature Family Total SHAP Contribution",
            "SHAP Stability (CV < 1.0 = stable)",
            "v2 New Features in Top 50",
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.15,
    )

    # Panel 1: Top 30 horizontal bar
    top30 = top50[:30]
    names = [f["feature"][:35] for f in reversed(top30)]
    vals = [f["mean_abs_shap"] for f in reversed(top30)]
    colors = []
    for f in reversed(top30):
        fam = f.get("family", "")
        if fam.startswith("ICT"):
            colors.append("orange")
        elif fam.startswith("HTF"):
            colors.append("cyan")
        elif fam.startswith("Momentum"):
            colors.append("limegreen")
        elif fam.startswith("Trend"):
            colors.append("dodgerblue")
        elif fam.startswith("Volatility"):
            colors.append("magenta")
        elif fam.startswith("Volume"):
            colors.append("yellow")
        else:
            colors.append("gray")

    fig.add_trace(go.Bar(
        y=names, x=vals, orientation="h",
        marker_color=colors, showlegend=False,
    ), row=1, col=1)

    # Panel 2: Family contribution
    fam_names = [f["family"][:25] for f in family_summary[:15]]
    fam_shap = [f["total_shap"] for f in family_summary[:15]]
    fig.add_trace(go.Bar(
        x=fam_names, y=fam_shap, marker_color="steelblue",
        showlegend=False,
    ), row=1, col=2)

    # Panel 3: Stability scatter (mean SHAP vs CV)
    # Only show features with nonzero SHAP
    stable = [s for s in stability if s["mean_abs_shap"] > 0.0001]
    fig.add_trace(go.Scatter(
        x=[s["mean_abs_shap"] for s in stable],
        y=[min(s["cv"], 5.0) for s in stable],
        mode="markers",
        marker=dict(
            size=4,
            color=["red" if s["cv"] > 2.0 else "limegreen" for s in stable],
            opacity=0.6,
        ),
        text=[s["feature"][:30] for s in stable],
        showlegend=False,
    ), row=2, col=1)
    fig.add_hline(y=2.0, line_dash="dash", line_color="red",
                  annotation_text="CV=2.0 (regime-dependent)", row=2, col=1)

    # Panel 4: v2 new features bar chart
    v2_in_top = [f for f in top50 if f.get("is_v2_new", False)]
    if v2_in_top:
        v2_names = [f["feature"][:30] for f in reversed(v2_in_top[:20])]
        v2_vals = [f["mean_abs_shap"] for f in reversed(v2_in_top[:20])]
        fig.add_trace(go.Bar(
            y=v2_names, x=v2_vals, orientation="h",
            marker_color="limegreen", showlegend=False,
        ), row=2, col=2)
    else:
        fig.add_annotation(
            text="No v2 features in top 50",
            x=0.5, y=0.5, xref="x4 domain", yref="y4 domain",
            showarrow=False, font=dict(size=14, color="gray"),
            row=2, col=2,
        )

    fig.update_layout(
        template="plotly_dark",
        title="SHAP Feature Importance Analysis -- v2 Model (D36)",
        height=950,
    )
    fig.update_xaxes(title_text="Mean |SHAP|", row=1, col=1)
    fig.update_xaxes(title_text="Family", row=1, col=2)
    fig.update_yaxes(title_text="Total SHAP", row=1, col=2)
    fig.update_xaxes(title_text="Mean |SHAP|", row=2, col=1)
    fig.update_yaxes(title_text="CV (std/mean)", row=2, col=1)
    fig.update_xaxes(title_text="Mean |SHAP|", row=2, col=2)

    path = RESULTS_DIR / "shap_analysis_v2.html"
    fig.write_html(str(path))
    return path


# -- Main --------------------------------------------------------------------
def main() -> None:
    cfg = Config()
    ml_cfg = MLConfig()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    sep = "=" * 70
    rule = "-" * 70

    # 1. Load data
    print(sep)
    print("  STEP 1: Load v2 Data + Feature Catalog")
    print(sep)

    df, features = prepare_data_v2(cfg, ml_cfg)
    catalog = load_catalog()
    print(f"  Feature catalog: {len(catalog)} entries")

    # Load v1 gain importance for rank comparison
    v1_path = RESULTS_DIR / "ml_pipeline.json"
    with open(v1_path) as f:
        v1_data = json.load(f)
    v1_rank = {f["feature"]: i + 1 for i, f in
               enumerate(v1_data["top_30_features"])}

    # Load v2 gain importance for cross-reference
    v2_path = RESULTS_DIR / "ml_pipeline_v2.json"
    with open(v2_path) as f:
        v2_data = json.load(f)
    v2_gain_rank = {f["feature"]: i + 1 for i, f in
                    enumerate(v2_data["top_30_features"])}

    # Identify v2-new features (not in v1 feature set)
    v1_feature_count = v1_data["config"]["n_features"]  # 387
    # v2 features that are new = those in catalog (122 new features)
    v2_new_features = set(catalog.keys())

    # 2. Walk-forward SHAP
    print(f"\n{sep}")
    print("  STEP 2: Walk-Forward Training + SHAP Extraction")
    print(sep)

    oos_probs, fold_results, shap_per_fold = walk_forward_shap(
        df, features, ml_cfg
    )

    # Verify AUC matches D34
    valid = ~np.isnan(oos_probs)
    y_oos = df[ml_cfg.label_col].values[valid]
    p_oos = oos_probs[valid]
    overall_auc = compute_auc(y_oos, p_oos)
    print(f"\n  Overall OOS AUC: {overall_auc:.4f} (D34 was 0.7937)")

    # 3. Aggregate SHAP values
    print(f"\n{sep}")
    print("  STEP 3: SHAP Aggregation")
    print(sep)

    n_features = len(features)
    n_folds = len(shap_per_fold)

    # Per-fold mean |SHAP| for each feature
    fold_mean_abs = np.zeros((n_folds, n_features))
    for fi, shap_vals in enumerate(shap_per_fold):
        fold_mean_abs[fi, :] = np.abs(shap_vals).mean(axis=0)

    # Global mean |SHAP| across all folds (weighted by test size)
    all_shap = np.vstack(shap_per_fold)  # (total_test_bars, n_features)
    global_mean_abs = np.abs(all_shap).mean(axis=0)

    # Stability: CV = std / mean across folds
    fold_means = fold_mean_abs.mean(axis=0)  # should ~= global but per-fold
    fold_stds = fold_mean_abs.std(axis=0)
    cv = np.where(fold_means > 0, fold_stds / fold_means, 0.0)

    # Build per-feature records
    feature_records = []
    for i, feat in enumerate(features):
        family, tier = classify_family(feat, catalog)
        is_new = feat in v2_new_features
        rec = {
            "feature": feat,
            "mean_abs_shap": float(global_mean_abs[i]),
            "family": family,
            "evidence_tier": tier,
            "is_v2_new": is_new,
            "v2_gain_rank": v2_gain_rank.get(feat, None),
            "v1_gain_rank": v1_rank.get(feat, None),
            "cv": float(cv[i]),
            "fold_mean_abs": fold_mean_abs[:, i].tolist(),
        }
        feature_records.append(rec)

    # Sort by mean |SHAP| descending
    feature_records.sort(key=lambda r: r["mean_abs_shap"], reverse=True)

    # Assign SHAP rank
    for i, rec in enumerate(feature_records):
        rec["shap_rank"] = i + 1

    top50 = feature_records[:50]
    bottom50 = feature_records[-50:]

    print(f"  Total features: {n_features}")
    print(f"  SHAP computation: {all_shap.shape[0]:,} test bars x "
          f"{n_features} features")
    print(f"  Top feature: {top50[0]['feature']} "
          f"(|SHAP| = {top50[0]['mean_abs_shap']:.6f})")
    print(f"  Bottom feature: {bottom50[-1]['feature']} "
          f"(|SHAP| = {bottom50[-1]['mean_abs_shap']:.6f})")

    # 4. Reports
    print(f"\n{sep}")
    print("  A. TOP 50 FEATURES by Mean |SHAP|")
    print(sep)
    print(f"  {'Rank':>4}  {'Feature':<40}  {'|SHAP|':>10}  "
          f"{'Family':<25}  {'Tier':>4}  {'v2Gain':>6}  "
          f"{'v1Gain':>6}  {'New':>3}  {'CV':>5}")
    for rec in top50:
        v2g = f"{rec['v2_gain_rank']:>6}" if rec['v2_gain_rank'] else "     -"
        v1g = f"{rec['v1_gain_rank']:>6}" if rec['v1_gain_rank'] else "     -"
        new = "YES" if rec["is_v2_new"] else "  -"
        print(f"  {rec['shap_rank']:>4}  {rec['feature']:<40}  "
              f"{rec['mean_abs_shap']:>10.6f}  "
              f"{rec['family']:<25}  {rec['evidence_tier']:>4}  "
              f"{v2g}  {v1g}  {new}  {rec['cv']:>5.2f}")

    # Count v2 new in top 50
    n_v2_in_top50 = sum(1 for r in top50 if r["is_v2_new"])
    print(f"\n  v2 new features in top 50: {n_v2_in_top50}/50")

    print(f"\n{sep}")
    print("  B. BOTTOM 50 FEATURES (pruning candidates)")
    print(sep)
    print(f"  {'Rank':>4}  {'Feature':<40}  {'|SHAP|':>10}  "
          f"{'Family':<25}  {'Tier':>4}  {'Zero':>4}")
    n_near_zero = 0
    for rec in bottom50:
        is_zero = rec["mean_abs_shap"] < 0.001
        if is_zero:
            n_near_zero += 1
        zf = "<<<" if is_zero else ""
        print(f"  {rec['shap_rank']:>4}  {rec['feature']:<40}  "
              f"{rec['mean_abs_shap']:>10.6f}  "
              f"{rec['family']:<25}  {rec['evidence_tier']:>4}  {zf}")

    print(f"\n  Features with |SHAP| < 0.001: {n_near_zero}/50")

    # Total near-zero across all features
    total_near_zero = sum(1 for r in feature_records
                          if r["mean_abs_shap"] < 0.001)
    print(f"  Total features with |SHAP| < 0.001: "
          f"{total_near_zero}/{n_features}")

    print(f"\n{sep}")
    print("  C. STABILITY ANALYSIS (CV across folds)")
    print(sep)

    regime_dep = [r for r in feature_records
                  if r["cv"] > 2.0 and r["mean_abs_shap"] > 0.001]
    stable_high = [r for r in feature_records[:100] if r["cv"] < 0.5]

    print(f"  Regime-dependent (CV > 2.0, |SHAP| > 0.001): "
          f"{len(regime_dep)} features")
    if regime_dep:
        print(f"  {'Rank':>4}  {'Feature':<40}  {'|SHAP|':>10}  "
              f"{'CV':>5}  {'Family':<25}")
        for rec in regime_dep[:20]:
            print(f"  {rec['shap_rank']:>4}  {rec['feature']:<40}  "
                  f"{rec['mean_abs_shap']:>10.6f}  "
                  f"{rec['cv']:>5.2f}  {rec['family']:<25}")

    print(f"\n  Very stable top-100 features (CV < 0.5): "
          f"{len(stable_high)} features")
    if stable_high:
        for rec in stable_high[:10]:
            print(f"    #{rec['shap_rank']:>3} {rec['feature']:<40}  "
                  f"CV={rec['cv']:.2f}")

    print(f"\n{sep}")
    print("  D. FAMILY SUMMARY")
    print(sep)

    # Aggregate by family
    family_agg: dict[str, dict] = {}
    for rec in feature_records:
        fam = rec["family"]
        if fam not in family_agg:
            family_agg[fam] = {
                "family": fam,
                "n_features": 0,
                "total_shap": 0.0,
                "mean_shap": 0.0,
                "best_rank": 999,
                "n_v2_new": 0,
                "ranks": [],
            }
        fa = family_agg[fam]
        fa["n_features"] += 1
        fa["total_shap"] += rec["mean_abs_shap"]
        fa["ranks"].append(rec["shap_rank"])
        if rec["shap_rank"] < fa["best_rank"]:
            fa["best_rank"] = rec["shap_rank"]
        if rec["is_v2_new"]:
            fa["n_v2_new"] += 1

    for fa in family_agg.values():
        fa["mean_shap"] = fa["total_shap"] / fa["n_features"]
        fa["avg_rank"] = float(np.mean(fa["ranks"]))
        del fa["ranks"]

    family_summary = sorted(family_agg.values(),
                            key=lambda x: x["total_shap"], reverse=True)

    print(f"  {'Family':<25}  {'N':>3}  {'Total |SHAP|':>12}  "
          f"{'Mean |SHAP|':>11}  {'Best Rank':>9}  {'Avg Rank':>8}  "
          f"{'v2 New':>6}")
    for fa in family_summary:
        print(f"  {fa['family']:<25}  {fa['n_features']:>3}  "
              f"{fa['total_shap']:>12.6f}  {fa['mean_shap']:>11.6f}  "
              f"{fa['best_rank']:>9}  {fa['avg_rank']:>8.1f}  "
              f"{fa['n_v2_new']:>6}")

    # v2 new features group analysis
    v2_records = [r for r in feature_records if r["is_v2_new"]]
    v1_records = [r for r in feature_records if not r["is_v2_new"]]
    v2_total_shap = sum(r["mean_abs_shap"] for r in v2_records)
    v1_total_shap = sum(r["mean_abs_shap"] for r in v1_records)
    total_shap = v2_total_shap + v1_total_shap

    print(f"\n  v2 new features ({len(v2_records)}): "
          f"total |SHAP| = {v2_total_shap:.6f} "
          f"({v2_total_shap/total_shap*100:.1f}% of total)")
    print(f"  v1 features ({len(v1_records)}): "
          f"total |SHAP| = {v1_total_shap:.6f} "
          f"({v1_total_shap/total_shap*100:.1f}% of total)")

    # 5. Ablation
    print(f"\n{sep}")
    print("  E. ABLATION: Top 408 Features vs Full 508")
    print(sep)

    n_keep = len(features) - 100  # drop bottom 100
    top_features_list = [r["feature"] for r in feature_records[:n_keep]]
    dropped_features = [r["feature"] for r in feature_records[n_keep:]]

    print(f"  Keeping top {n_keep} features, dropping bottom 100")
    print(f"  Dropped features total |SHAP|: "
          f"{sum(r['mean_abs_shap'] for r in feature_records[n_keep:]):.6f}")

    auc_full, auc_reduced = ablation_retrain(
        df, features, top_features_list, ml_cfg
    )

    auc_delta = auc_reduced - auc_full
    print(f"\n  Full 508 features:   AUC = {auc_full:.4f}")
    print(f"  Top {n_keep} features:    AUC = {auc_reduced:.4f}")
    print(f"  Delta:               {auc_delta:+.4f}")

    # 6. Recommendations
    print(f"\n{sep}")
    print("  RECOMMENDATIONS")
    print(sep)

    # Candidates to DROP
    drop_candidates = [r for r in feature_records
                       if r["shap_rank"] > int(n_features * 0.8)
                       and r["cv"] < 2.0]
    print(f"\n  DROP candidates (bottom 20% by SHAP, CV < 2.0): "
          f"{len(drop_candidates)} features")
    for r in drop_candidates[:10]:
        print(f"    #{r['shap_rank']:>3} {r['feature']:<40}  "
              f"|SHAP|={r['mean_abs_shap']:.6f}  CV={r['cv']:.2f}")
    if len(drop_candidates) > 10:
        print(f"    ... and {len(drop_candidates) - 10} more")

    # Candidates to KEEP ALWAYS (regime features)
    keep_always = [r for r in feature_records
                   if r["cv"] > 2.0 and r["mean_abs_shap"] > 0.001]
    print(f"\n  KEEP ALWAYS candidates (regime-dependent, |SHAP| > 0.001): "
          f"{len(keep_always)} features")
    for r in keep_always[:10]:
        print(f"    #{r['shap_rank']:>3} {r['feature']:<40}  "
              f"|SHAP|={r['mean_abs_shap']:.6f}  CV={r['cv']:.2f}")

    print(f"\n  Ablation impact: dropping 100 features -> "
          f"AUC {auc_delta:+.4f} ({auc_reduced:.4f} vs {auc_full:.4f})")
    if abs(auc_delta) < 0.005:
        print(f"  -> SAFE to prune bottom 100 (AUC change < 0.005)")
    else:
        print(f"  -> CAUTION: AUC change >= 0.005, prune carefully")

    print(sep)

    # 7. Save outputs
    # CSV files
    top50_df = pd.DataFrame([{
        "rank": r["shap_rank"],
        "feature": r["feature"],
        "mean_abs_shap": round(r["mean_abs_shap"], 8),
        "family": r["family"],
        "evidence_tier": r["evidence_tier"],
        "is_v2_new": r["is_v2_new"],
        "v2_gain_rank": r["v2_gain_rank"],
        "v1_gain_rank": r["v1_gain_rank"],
        "cv": round(r["cv"], 4),
    } for r in top50])
    top50_df.to_csv(RESULTS_DIR / "shap_top50.csv", index=False)
    print(f"\n  Saved: {RESULTS_DIR / 'shap_top50.csv'}")

    bottom50_df = pd.DataFrame([{
        "rank": r["shap_rank"],
        "feature": r["feature"],
        "mean_abs_shap": round(r["mean_abs_shap"], 8),
        "family": r["family"],
        "evidence_tier": r["evidence_tier"],
        "is_v2_new": r["is_v2_new"],
        "cv": round(r["cv"], 4),
        "near_zero": r["mean_abs_shap"] < 0.001,
    } for r in bottom50])
    bottom50_df.to_csv(RESULTS_DIR / "shap_bottom50.csv", index=False)
    print(f"  Saved: {RESULTS_DIR / 'shap_bottom50.csv'}")

    # Chart
    stability_data = [{
        "feature": r["feature"],
        "mean_abs_shap": r["mean_abs_shap"],
        "cv": r["cv"],
    } for r in feature_records]

    chart_path = save_chart(top50, family_summary, stability_data)
    if chart_path:
        print(f"  Saved: {chart_path}")

    # JSON
    def _round(obj, dp=6):
        if isinstance(obj, dict):
            return {k: _round(v, dp) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_round(v, dp) for v in obj]
        if isinstance(obj, float) and not (obj != obj):
            return round(obj, dp)
        return obj

    # Strip fold_mean_abs arrays for JSON (too large)
    top50_clean = [{k: v for k, v in r.items() if k != "fold_mean_abs"}
                   for r in top50]
    bottom50_clean = [{k: v for k, v in r.items() if k != "fold_mean_abs"}
                      for r in bottom50]

    summary = {
        "n_features": n_features,
        "n_folds": n_folds,
        "overall_auc": round(overall_auc, 4),
        "total_test_bars": all_shap.shape[0],
        "fold_results": fold_results,
        "top50": top50_clean,
        "bottom50": bottom50_clean,
        "family_summary": family_summary,
        "stability": {
            "n_regime_dependent_cv2": len(regime_dep),
            "regime_dependent_features": [
                {"feature": r["feature"], "shap_rank": r["shap_rank"],
                 "mean_abs_shap": r["mean_abs_shap"], "cv": r["cv"]}
                for r in regime_dep
            ],
            "n_stable_top100_cv05": len(stable_high),
        },
        "v2_vs_v1": {
            "n_v2_new": len(v2_records),
            "n_v1": len(v1_records),
            "v2_total_shap": round(v2_total_shap, 6),
            "v1_total_shap": round(v1_total_shap, 6),
            "v2_pct_of_total": round(v2_total_shap / total_shap * 100, 1),
            "n_v2_in_top50": n_v2_in_top50,
        },
        "ablation": {
            "n_features_full": len(features),
            "n_features_reduced": n_keep,
            "auc_full": round(auc_full, 4),
            "auc_reduced": round(auc_reduced, 4),
            "auc_delta": round(auc_delta, 4),
            "safe_to_prune": abs(auc_delta) < 0.005,
        },
        "recommendations": {
            "n_drop_candidates": len(drop_candidates),
            "drop_features": [r["feature"] for r in drop_candidates],
            "n_keep_always": len(keep_always),
            "keep_always_features": [r["feature"] for r in keep_always],
        },
    }

    summary = _round(summary)

    json_path = RESULTS_DIR / "shap_analysis_v2.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved: {json_path}")


if __name__ == "__main__":
    main()
