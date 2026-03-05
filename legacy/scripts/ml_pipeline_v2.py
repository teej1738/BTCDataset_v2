"""
ml_pipeline_v2.py -- LightGBM Walk-Forward ML on v2 Dataset (D34)
=================================================================
Same walk-forward setup as D28 (ml_pipeline.py) but trained on the
v2 dataset (569 columns = 448 original + 122 new TA/ICT features).

Compares OOS AUC, WR, EV, PF against D28 baseline (387 features, AUC 0.78).

Output:
  - results/ml_pipeline_v2.json
  - results/ml_pipeline_v2.html
  - results/ml_oos_probs_v2.npy
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_backtest_v2 import Config
from ml_pipeline import (
    MLConfig,
    select_features,
    compute_auc,
    walk_forward_train,
    compute_threshold_metrics,
    aggregate_importance,
    config_b_baseline,
)
from mtf_signals import TF_CONFIGS, config_b_filters

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = SCRIPT_DIR / "results"
V2_PATH = PROJECT_DIR / "data" / "labeled" / "BTCUSDT_5m_labeled_v2.parquet"


# ---- data preparation (v2) -------------------------------------------------
def prepare_data_v2(cfg: Config, ml_cfg: MLConfig):
    """Load v2 parquet, select features, prepare for training."""
    print(f"Loading {V2_PATH.name} ...")
    df = pd.read_parquet(V2_PATH)
    print(f"  raw shape: {df.shape[0]:,} rows x {df.shape[1]} cols")

    df = df[df["bar_start_ts_utc"] >= pd.Timestamp(cfg.min_date, tz="UTC")].copy()
    print(f"  after date filter (>= {cfg.min_date}): {df.shape[0]:,} rows")

    before = len(df)
    df = df.dropna(subset=[cfg.long_label, cfg.short_label])
    dropped = before - len(df)
    if dropped:
        print(f"  dropped {dropped} rows with NaN labels")

    df = df.reset_index(drop=True)

    if "regime_label" not in df.columns:
        df["regime_label"] = -1

    features = select_features(df)
    print(f"  Selected {len(features)} features (v1 had 387)")

    df[ml_cfg.label_col] = df[ml_cfg.label_col].astype(int)
    pos_rate = df[ml_cfg.label_col].mean()
    print(f"  Rows: {len(df):,}  |  Positive rate: {pos_rate:.4f}")

    return df, features


# ---- chart ------------------------------------------------------------------
def save_chart_v2(threshold_metrics, config_b, top_features, d28_metrics):
    """Save comparison chart."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return None

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Win Rate: v2 vs v1 (D28)",
            "Signals/Year: v2 vs v1",
            "EV (R): v2 vs v1",
            "Top 20 Feature Importances (v2)",
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.12,
    )

    # v2 data
    active = [m for m in threshold_metrics if m["n_signals"] > 0]
    ts = [m["threshold"] for m in active]
    wrs = [m["wr"] * 100 for m in active]
    sigs = [m["signals_per_year"] for m in active]
    evs = [m["ev_r"] for m in active]

    # v1 data
    d28_active = [m for m in d28_metrics if m["n_signals"] > 0]
    ts1 = [m["threshold"] for m in d28_active]
    wrs1 = [m["wr"] * 100 for m in d28_active]
    sigs1 = [m["signals_per_year"] for m in d28_active]
    evs1 = [m["ev_r"] for m in d28_active]

    # WR
    fig.add_trace(go.Scatter(x=ts, y=wrs, mode="lines+markers",
                             line=dict(color="cyan"), name="v2"), row=1, col=1)
    fig.add_trace(go.Scatter(x=ts1, y=wrs1, mode="lines+markers",
                             line=dict(color="gray", dash="dash"), name="v1 (D28)"),
                  row=1, col=1)
    fig.add_hline(y=33.33, line_dash="dot", line_color="red",
                  annotation_text="BE", row=1, col=1)

    # Signals/yr
    fig.add_trace(go.Scatter(x=ts, y=sigs, mode="lines+markers",
                             line=dict(color="limegreen"), name="v2"), row=1, col=2)
    fig.add_trace(go.Scatter(x=ts1, y=sigs1, mode="lines+markers",
                             line=dict(color="gray", dash="dash"), name="v1"),
                  row=1, col=2)

    # EV
    fig.add_trace(go.Scatter(x=ts, y=evs, mode="lines+markers",
                             line=dict(color="orange"), name="v2"), row=2, col=1)
    fig.add_trace(go.Scatter(x=ts1, y=evs1, mode="lines+markers",
                             line=dict(color="gray", dash="dash"), name="v1"),
                  row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="red", row=2, col=1)

    # Feature importance
    top20 = top_features[:20]
    names = [f[0][:35] for f in reversed(top20)]
    vals = [f[1] for f in reversed(top20)]
    fig.add_trace(go.Bar(y=names, x=vals, orientation="h",
                         marker_color="dodgerblue", showlegend=False), row=2, col=2)

    fig.update_layout(
        template="plotly_dark",
        title="ML Pipeline v2 -- Walk-Forward LightGBM (v2 vs v1 Comparison)",
        height=950,
        showlegend=True,
    )
    fig.update_xaxes(title_text="Threshold", row=1, col=1)
    fig.update_yaxes(title_text="Win Rate %", row=1, col=1)
    fig.update_xaxes(title_text="Threshold", row=1, col=2)
    fig.update_yaxes(title_text="Signals/Year", row=1, col=2)
    fig.update_xaxes(title_text="Threshold", row=2, col=1)
    fig.update_yaxes(title_text="EV (R)", row=2, col=1)
    fig.update_xaxes(title_text="Gain", row=2, col=2)

    path = RESULTS_DIR / "ml_pipeline_v2.html"
    fig.write_html(str(path))
    return path


# ---- report -----------------------------------------------------------------
def print_report_v2(
    df, features, oos_probs, fold_results, threshold_metrics,
    top_features, config_b, ml_cfg, d28_summary,
):
    """Print v2 results with D28 comparison."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    sep = "=" * 70
    rule = "-" * 70

    print(f"\n{sep}")
    print("  ML PIPELINE v2 -- Walk-Forward LightGBM (D34)")
    print(sep)
    print(f"  Label: {ml_cfg.label_col}")
    print(f"  Features: {len(features)} (v1 had {d28_summary['config']['n_features']})")
    print(f"  Folds: {len(fold_results)}")
    print(f"  Embargo: {ml_cfg.embargo_bars} bars (4h)")

    # Fold summary
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

    avg_ll = float(np.mean([f["logloss"] for f in fold_results]))
    avg_auc = float(np.mean([f["auc"] for f in fold_results]))
    d28_auc = d28_summary["avg_oos_auc"]
    d28_ll = d28_summary["avg_oos_logloss"]
    auc_delta = avg_auc - d28_auc
    ll_delta = avg_ll - d28_ll

    print(f"\n  Mean OOS logloss: {avg_ll:.4f} (v1: {d28_ll:.4f}, delta: {ll_delta:+.4f})")
    print(f"  Mean OOS AUC:     {avg_auc:.4f} (v1: {d28_auc:.4f}, delta: {auc_delta:+.4f})")

    # Threshold analysis
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

    # v2 vs v1 comparison at key thresholds
    print(f"\n{rule}")
    print("  v2 vs v1 COMPARISON (key thresholds)")
    print(rule)
    d28_by_t = {m["threshold"]: m for m in d28_summary["threshold_metrics"]}
    v2_by_t = {m["threshold"]: m for m in threshold_metrics}
    print(f"  {'Thresh':>6}  {'v2 WR':>7}  {'v1 WR':>7}  {'dWR':>6}  "
          f"{'v2 EV':>7}  {'v1 EV':>7}  {'dEV':>6}  "
          f"{'v2 PF':>6}  {'v1 PF':>6}")
    for t in [0.50, 0.55, 0.60, 0.65, 0.70]:
        v2 = v2_by_t.get(t)
        v1 = d28_by_t.get(t)
        if v2 and v1 and v2["n_signals"] > 0 and v1["n_signals"] > 0:
            d_wr = (v2["wr"] - v1["wr"]) * 100
            d_ev = v2["ev_r"] - v1["ev_r"]
            print(f"  {t:>6.2f}  {v2['wr']:>6.2%}  {v1['wr']:>6.2%}  "
                  f"{d_wr:>+5.1f}pp  {v2['ev_r']:>+6.3f}  {v1['ev_r']:>+6.3f}  "
                  f"{d_ev:>+5.3f}  {v2['pf']:>6.2f}  {v1['pf']:>6.2f}")

    # Config B comparison
    print(f"\n{rule}")
    print("  CONFIG B COMPARISON")
    print(rule)
    cb_per_yr = config_b["n"] / 6.16
    print(f"  Config B MTF longs: n={config_b['n']}, "
          f"WR={config_b['wr']:.2%}, EV={config_b['ev_r']:+.4f}R, "
          f"{cb_per_yr:.0f}/yr")

    # ML on Config B bars
    valid_mask = ~np.isnan(oos_probs)
    if valid_mask.any():
        cb_long = pd.Series(False, index=df.index)
        for tf in TF_CONFIGS:
            lm, _ = config_b_filters(df, tf, Config())
            cb_long = cb_long | lm
        cb_oos = cb_long.values & valid_mask
        if cb_oos.any():
            cb_probs = oos_probs[cb_oos]
            print(f"\n  ML v2 scores on Config B bars ({int(cb_oos.sum())} in OOS):")
            print(f"    mean={cb_probs.mean():.3f}  "
                  f"median={np.median(cb_probs):.3f}  "
                  f"std={cb_probs.std():.3f}")

    # Feature importance
    print(f"\n{rule}")
    print("  TOP 30 FEATURES (gain, averaged across folds)")
    print(rule)
    # Mark new v2 features
    v1_features = set()
    for f_dict in d28_summary.get("top_30_features", []):
        v1_features.add(f_dict["feature"])
    # Actually, just load v1 feature list from json
    v1_feat_count = d28_summary["config"]["n_features"]

    for i, (feat, imp) in enumerate(top_features[:30], 1):
        tag = " [NEW]" if feat.startswith(("rsi_", "macd_", "stoch_", "roc_",
               "ema_", "vwap_", "super", "adx_", "di_", "ichi_", "mtf_ema",
               "clv", "mfi_", "obv_", "cvd_", "cmf_", "volume_", "taker_",
               "gk_", "parkinson_", "rs_", "hv_", "bb_", "squeeze_", "vol_",
               "sb_", "macro_", "kz_", "asia_", "po3_", "ote_", "cisd_",
               "ob_disp", "int_swing", "ict_confluence_v2", "spread_ar",
               "funding_regime", "funding_zscore_v2", "time_to_funding",
               "annualized_funding", "lunch_zone", "london_", "ny_")) else ""
        print(f"  {i:>2}. {feat:<40} {imp:>10.1f}{tag}")

    # New features in top 30
    new_in_top = sum(1 for f, _ in top_features[:30]
                     if f.startswith(("rsi_", "macd_", "stoch_", "roc_",
                        "ema_", "vwap_", "super", "adx_", "di_", "ichi_",
                        "mtf_ema", "clv", "mfi_", "obv_", "cvd_", "cmf_",
                        "volume_", "taker_", "gk_", "parkinson_", "rs_",
                        "hv_", "bb_", "squeeze_", "vol_", "sb_", "macro_",
                        "kz_", "asia_", "po3_", "ote_", "cisd_", "ob_disp",
                        "int_swing", "ict_confluence_v2", "spread_ar",
                        "funding_regime", "funding_zscore_v2",
                        "time_to_funding", "annualized_funding",
                        "lunch_zone", "london_", "ny_")))
    print(f"\n  New v2 features in top 30: {new_in_top}/30")

    # Assessment
    print(f"\n{rule}")
    print("  ASSESSMENT")
    print(rule)

    checks = [
        ("AUC improved over v1", avg_auc > d28_auc, f"{avg_auc:.4f} vs {d28_auc:.4f}"),
        ("Logloss improved over v1", avg_ll < d28_ll, f"{avg_ll:.4f} vs {d28_ll:.4f}"),
        ("AUC > 0.50", avg_auc > 0.50, f"{avg_auc:.4f}"),
        ("All folds AUC > 0.65", all(f["auc"] > 0.65 for f in fold_results),
         f"min={min(f['auc'] for f in fold_results):.4f}"),
    ]
    # Compare EV at t=0.60
    v2_t60 = v2_by_t.get(0.60, {})
    v1_t60 = d28_by_t.get(0.60, {})
    if v2_t60 and v1_t60:
        checks.append(("EV at t=0.60 improved",
                       v2_t60.get("ev_r", 0) > v1_t60.get("ev_r", 0),
                       f"{v2_t60.get('ev_r', 0):+.4f} vs {v1_t60.get('ev_r', 0):+.4f}"))

    for label, passed, val in checks:
        status = "PASS" if passed else "MISS"
        print(f"  [{status}]  {label:<45}  {val}")

    improved = avg_auc > d28_auc
    if improved:
        print(f"\n  >>> v2 features IMPROVE model (AUC +{auc_delta:.4f})")
    else:
        print(f"\n  >>> v2 features did not improve AUC (delta {auc_delta:+.4f})")
        print(f"      Check if logloss improved or if specific thresholds benefited")

    print(sep)

    # Save chart
    d28_tm = d28_summary.get("threshold_metrics", [])
    chart_path = save_chart_v2(threshold_metrics, config_b, top_features, d28_tm)
    if chart_path:
        print(f"\n  Saved: {chart_path}")

    # Save JSON
    summary = {
        "config": {
            "label": ml_cfg.label_col,
            "n_features": len(features),
            "n_features_v1": d28_summary["config"]["n_features"],
            "n_folds": len(fold_results),
            "embargo_bars": ml_cfg.embargo_bars,
            "dataset": "BTCUSDT_5m_labeled_v2.parquet",
            "lgb_params": {k: v for k, v in ml_cfg.lgb_params.items()},
        },
        "fold_results": fold_results,
        "avg_oos_logloss": round(avg_ll, 6),
        "avg_oos_auc": round(avg_auc, 4),
        "d28_baseline": {
            "avg_oos_auc": d28_auc,
            "avg_oos_logloss": d28_ll,
            "n_features": d28_summary["config"]["n_features"],
        },
        "auc_delta": round(auc_delta, 4),
        "logloss_delta": round(ll_delta, 6),
        "threshold_metrics": threshold_metrics,
        "config_b_baseline": config_b,
        "top_30_features": [
            {"feature": f, "importance": round(v, 2)}
            for f, v in top_features[:30]
        ],
        "oos_coverage": {
            "total_bars": len(oos_probs),
            "oos_bars": int(valid_mask.sum()),
            "pct": round(float(valid_mask.mean() * 100), 1),
        },
    }

    json_path = RESULTS_DIR / "ml_pipeline_v2.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved: {json_path}")

    return summary


# ---- main -------------------------------------------------------------------
def main():
    cfg = Config()
    ml_cfg = MLConfig()

    # Load D28 baseline for comparison
    d28_path = RESULTS_DIR / "ml_pipeline.json"
    with open(d28_path) as f:
        d28_summary = json.load(f)

    # 1. Prepare v2 data
    print("=" * 70)
    print("  STEP 1: Data Preparation (v2 dataset)")
    print("=" * 70)
    df, features = prepare_data_v2(cfg, ml_cfg)

    # 2. Config B baseline
    print(f"\n  Computing Config B baseline...")
    config_b = config_b_baseline(df, cfg, ml_cfg)
    print(f"  Config B: n={config_b['n']}, WR={config_b['wr']:.2%}, "
          f"EV={config_b['ev_r']:+.4f}R")

    # 3. Walk-forward training
    print(f"\n{'=' * 70}")
    print("  STEP 2: Walk-Forward Training (v2 features)")
    print("=" * 70)
    oos_probs, fold_results, importance_list = walk_forward_train(
        df, features, ml_cfg
    )

    # 4. Threshold metrics
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

    # 6. Save OOS probs
    np.save(RESULTS_DIR / "ml_oos_probs_v2.npy", oos_probs)
    print(f"  Saved OOS probs: {RESULTS_DIR / 'ml_oos_probs_v2.npy'}")

    # 7. Report
    print_report_v2(
        df, features, oos_probs, fold_results, threshold_metrics,
        top_features, config_b, ml_cfg, d28_summary,
    )


if __name__ == "__main__":
    main()
