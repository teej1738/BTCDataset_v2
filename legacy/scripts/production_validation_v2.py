"""
production_validation_v2.py -- Production Config Validation on v2 ML Probs (D35)
================================================================================
Re-runs the D29 (ml_backtest) and D31 (cooldown_sweep) logic on v2 OOS probs
to answer:
  1. Does ML>=0.60 still hold as the optimal threshold?
  2. Does CD=576 still hold as the optimal cooldown?
  3. What are the updated WR, EV, PF, MaxDD at the production config?

Compares directly against D29/D31 baselines.

Output:
  - Console report with side-by-side v1 vs v2 comparisons
  - results/production_validation_v2.json
  - results/production_validation_v2.html
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_backtest_v2 import Config
from ml_pipeline import MLConfig
from ml_pipeline_v2 import prepare_data_v2
from ml_backtest import (
    build_config_b_mask,
    simulate,
    build_trade_returns,
    equity_sim,
    compute_metrics,
    run_cscv,
)
from cooldown_sweep_t2 import per_year_breakdown

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
V2_PROBS_PATH = RESULTS_DIR / "ml_oos_probs_v2.npy"

# Cooldowns to test
COOLDOWNS = [48, 96, 144, 288, 576]

# Production target criteria (same as D31)
TARGET_MIN_TPY = 100
TARGET_MAX_TPY = 300
TARGET_MIN_WR = 0.65
TARGET_MIN_EV = 0.90


def load_baselines():
    """Load D29 and D31 baseline results for comparison."""
    d29_path = RESULTS_DIR / "ml_backtest.json"
    d31_path = RESULTS_DIR / "cooldown_sweep_t2.json"

    with open(d29_path) as f:
        d29 = json.load(f)
    with open(d31_path) as f:
        d31 = json.load(f)

    return d29, d31


def save_chart(
    v2_configs: list[dict],
    v1_configs: list[dict],
    v2_sweep: list[dict],
    v1_sweep: list[dict],
    v2_yearly: dict[int, list[dict]],
) -> Path | None:
    """4-panel comparison chart: threshold + cooldown, v2 vs v1."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return None

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Threshold Comparison: EV (R)",
            "Threshold Comparison: Win Rate",
            "Cooldown Sweep: EV (R)",
            "Cooldown Sweep: Per-Year WR",
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.12,
    )

    # Panel 1: Threshold EV
    thresh_names = ["T1 (ML>=0.50)", "T2 (ML>=0.60)", "T3 (CB+ML>=0.50)"]
    v2_evs = [c["ev_r"] for c in v2_configs if c["name"] in thresh_names]
    v1_evs = [c["ev_r"] for c in v1_configs if c["name"] in thresh_names]
    short_names = ["T1", "T2", "T3"]

    fig.add_trace(go.Bar(
        x=short_names, y=v2_evs, name="v2", marker_color="cyan",
        text=[f"{e:+.3f}" for e in v2_evs], textposition="outside",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=short_names, y=v1_evs, name="v1", marker_color="gray",
        text=[f"{e:+.3f}" for e in v1_evs], textposition="outside",
    ), row=1, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="red", row=1, col=1)

    # Panel 2: Threshold WR
    v2_wrs = [c["win_rate"] * 100 for c in v2_configs if c["name"] in thresh_names]
    v1_wrs = [c["win_rate"] * 100 for c in v1_configs if c["name"] in thresh_names]

    fig.add_trace(go.Bar(
        x=short_names, y=v2_wrs, name="v2", marker_color="orange",
        text=[f"{w:.1f}%" for w in v2_wrs], textposition="outside",
        showlegend=False,
    ), row=1, col=2)
    fig.add_trace(go.Bar(
        x=short_names, y=v1_wrs, name="v1", marker_color="gray",
        text=[f"{w:.1f}%" for w in v1_wrs], textposition="outside",
        showlegend=False,
    ), row=1, col=2)
    fig.add_hline(y=33.33, line_dash="dot", line_color="red",
                  annotation_text="BE", row=1, col=2)

    # Panel 3: Cooldown sweep EV
    v2_cds = [str(r["cooldown_bars"]) for r in v2_sweep]
    v2_cd_evs = [r["ev_r"] for r in v2_sweep]
    v1_cd_evs = [r["ev_r"] for r in v1_sweep]

    fig.add_trace(go.Bar(
        x=v2_cds, y=v2_cd_evs, name="v2", marker_color="limegreen",
        text=[f"{e:+.3f}" for e in v2_cd_evs], textposition="outside",
        showlegend=False,
    ), row=2, col=1)
    fig.add_trace(go.Bar(
        x=v2_cds, y=v1_cd_evs, name="v1", marker_color="gray",
        text=[f"{e:+.3f}" for e in v1_cd_evs], textposition="outside",
        showlegend=False,
    ), row=2, col=1)
    fig.add_hline(y=TARGET_MIN_EV, line_dash="dash", line_color="red",
                  annotation_text=f"+{TARGET_MIN_EV}R", row=2, col=1)

    # Panel 4: Per-year WR for key cooldowns
    colors = {48: "cyan", 288: "yellow", 576: "magenta"}
    for cd in [48, 288, 576]:
        yd = v2_yearly.get(cd, [])
        if not yd:
            continue
        yrs = [str(d["year"]) for d in yd if d["trades"] >= 10]
        yr_wrs = [d["win_rate"] * 100 for d in yd if d["trades"] >= 10]
        fig.add_trace(go.Scatter(
            x=yrs, y=yr_wrs, mode="lines+markers",
            name=f"v2 CD={cd}",
            line=dict(color=colors.get(cd, "white"), width=1.5),
        ), row=2, col=2)
    fig.add_hline(y=TARGET_MIN_WR * 100, line_dash="dash", line_color="red",
                  row=2, col=2)

    fig.update_layout(
        template="plotly_dark",
        title="Production Validation v2 -- D35 (v2 vs v1)",
        height=900,
        barmode="group",
    )
    fig.update_xaxes(title_text="Threshold Config", row=1, col=1)
    fig.update_yaxes(title_text="EV (R)", row=1, col=1)
    fig.update_xaxes(title_text="Threshold Config", row=1, col=2)
    fig.update_yaxes(title_text="Win Rate %", row=1, col=2)
    fig.update_xaxes(title_text="Cooldown (bars)", row=2, col=1)
    fig.update_yaxes(title_text="EV (R)", row=2, col=1)
    fig.update_xaxes(title_text="Year", row=2, col=2)
    fig.update_yaxes(title_text="Win Rate %", row=2, col=2)

    path = RESULTS_DIR / "production_validation_v2.html"
    fig.write_html(str(path))
    return path


def main() -> None:
    cfg = Config()
    ml_cfg = MLConfig()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 0. Load baselines
    d29, d31 = load_baselines()
    d29_configs = {c["name"]: c for c in d29["configs"]}
    d31_sweep = {r["cooldown_bars"]: r for r in d31["sweep_results"]}

    # 1. Load data + v2 OOS probs
    sep = "=" * 70
    rule = "-" * 70

    print(sep)
    print("  STEP 1: Load Data + v2 OOS Probabilities")
    print(sep)

    df, features = prepare_data_v2(cfg, ml_cfg)

    print(f"\n  Loading v2 OOS probs from {V2_PROBS_PATH.name}...")
    oos_probs = np.load(str(V2_PROBS_PATH))
    if len(oos_probs) != len(df):
        raise ValueError(
            f"Probs length ({len(oos_probs)}) != df length ({len(df)})"
        )
    print(f"  Loaded {len(oos_probs):,} probs")

    oos_valid = ~np.isnan(oos_probs)
    n_oos = int(oos_valid.sum())
    oos_years = n_oos / (288 * 365.25)
    print(f"  OOS bars: {n_oos:,} ({oos_years:.2f} years, "
          f"{oos_valid.mean()*100:.1f}% coverage)")

    label_arr = df[ml_cfg.label_col].values
    timestamps = df["bar_start_ts_utc"].values
    cb_mask = build_config_b_mask(df, cfg)

    # ================================================================
    # PART A: Threshold Comparison (CD=48, same as D29)
    # ================================================================
    print(f"\n{sep}")
    print("  PART A: Threshold Comparison (CD=48)")
    print(sep)

    configs_def = [
        ("T1 (ML>=0.50)", (oos_probs >= 0.50) & oos_valid),
        ("T2 (ML>=0.60)", (oos_probs >= 0.60) & oos_valid),
        ("T3 (CB+ML>=0.50)", cb_mask & (oos_probs >= 0.50) & oos_valid),
        ("Config B (OOS)", cb_mask & oos_valid),
    ]

    v2_configs: list[dict] = []
    v2_returns: dict[str, np.ndarray] = {}
    v2_indices: dict[str, list[int]] = {}
    v2_equity: list[list[float]] = []

    for name, mask in configs_def:
        raw_sigs = int(mask.sum())
        trade_idx = simulate(mask, label_arr, cfg.cooldown_bars)
        r_returns = build_trade_returns(
            trade_idx, label_arr, cfg.r_target, cfg.cost_per_r
        )
        eq_path, max_dd = equity_sim(
            r_returns, cfg.risk_pct, cfg.initial_equity
        )

        metrics = compute_metrics(
            name, r_returns, max_dd, eq_path[-1], oos_years
        )
        v2_configs.append(metrics)
        v2_returns[name] = r_returns
        v2_indices[name] = trade_idx
        v2_equity.append(eq_path)

        # Compare to D29
        v1 = d29_configs.get(name, {})
        d_wr = (metrics["win_rate"] - v1.get("win_rate", 0)) * 100
        d_ev = metrics["ev_r"] - v1.get("ev_r", 0)
        print(f"  {name}: {metrics['n_trades']} trades, "
              f"WR {metrics['win_rate']:.2%} ({d_wr:+.1f}pp), "
              f"EV {metrics['ev_r']:+.4f}R ({d_ev:+.4f}), "
              f"Sharpe {metrics['sharpe_ann']:.2f}")

    # Best threshold
    candidates = [cm for cm in v2_configs if cm["name"] != "Config B (OOS)"]
    best_thresh = max(candidates, key=lambda x: x["sharpe_ann"])
    print(f"\n  Best threshold: {best_thresh['name']} "
          f"(Sharpe {best_thresh['sharpe_ann']:.2f})")

    # Comparison table
    print(f"\n{rule}")
    print("  v2 vs v1 THRESHOLD COMPARISON (CD=48)")
    print(rule)
    print(f"  {'Config':<20}  {'v2 Tr':>6}  {'v1 Tr':>6}  "
          f"{'v2 WR':>7}  {'v1 WR':>7}  {'dWR':>6}  "
          f"{'v2 EV':>7}  {'v1 EV':>7}  {'dEV':>6}  "
          f"{'v2 PF':>6}  {'v1 PF':>6}")
    for cm in v2_configs:
        v1 = d29_configs.get(cm["name"], {})
        d_wr = (cm["win_rate"] - v1.get("win_rate", 0)) * 100
        d_ev = cm["ev_r"] - v1.get("ev_r", 0)
        print(f"  {cm['name']:<20}  {cm['n_trades']:>6}  "
              f"{v1.get('n_trades', 0):>6}  "
              f"{cm['win_rate']:>6.2%}  {v1.get('win_rate', 0):>6.2%}  "
              f"{d_wr:>+5.1f}pp  "
              f"{cm['ev_r']:>+6.3f}  {v1.get('ev_r', 0):>+6.3f}  "
              f"{d_ev:>+5.3f}  "
              f"{cm['profit_factor']:>6.2f}  "
              f"{v1.get('profit_factor', 0):>6.2f}")

    # ================================================================
    # PART B: Cooldown Sweep at ML>=0.60 (same as D31)
    # ================================================================
    print(f"\n{sep}")
    print("  PART B: Cooldown Sweep (ML>=0.60, v2 probs)")
    print(sep)

    t2_mask = (oos_probs >= 0.60) & oos_valid
    raw_signals = int(t2_mask.sum())
    print(f"  T2 raw signal bars: {raw_signals:,} "
          f"({raw_signals / oos_years:.0f}/yr)")
    print(f"  Cost model: flat {cfg.cost_per_r}R  |  "
          f"Risk: {cfg.risk_pct:.0%}  |  Label: {ml_cfg.label_col}")
    print(f"  Target: {TARGET_MIN_TPY}-{TARGET_MAX_TPY} trades/yr, "
          f"WR >= {TARGET_MIN_WR:.0%}, EV >= +{TARGET_MIN_EV}R")

    v2_sweep: list[dict] = []
    v2_yearly: dict[int, list[dict]] = {}

    for cd in COOLDOWNS:
        trade_idx = simulate(t2_mask, label_arr, cd)
        r_returns = build_trade_returns(
            trade_idx, label_arr, cfg.r_target, cfg.cost_per_r
        )
        eq_path, max_dd = equity_sim(
            r_returns, cfg.risk_pct, cfg.initial_equity
        )

        metrics = compute_metrics(
            f"CD={cd}", r_returns, max_dd, eq_path[-1], oos_years
        )
        metrics["cooldown_bars"] = cd
        metrics["cooldown_hours"] = cd * 5 / 60

        # Per-year
        yr_data = per_year_breakdown(
            trade_idx, label_arr, timestamps, cfg.r_target, cfg.cost_per_r
        )

        # Target check
        in_target = (
            TARGET_MIN_TPY <= metrics["trades_per_yr"] <= TARGET_MAX_TPY
            and metrics["win_rate"] >= TARGET_MIN_WR
            and metrics["ev_r"] >= TARGET_MIN_EV
        )
        metrics["meets_target"] = in_target

        # Temporal stability
        if yr_data:
            stable_yrs = [y for y in yr_data if y["trades"] >= 10]
            if stable_yrs:
                metrics["worst_yr_wr"] = round(
                    min(y["win_rate"] for y in stable_yrs), 4)
                metrics["worst_yr_ev"] = round(
                    min(y["ev_r"] for y in stable_yrs), 4)
                metrics["n_profitable_yrs"] = sum(
                    1 for y in stable_yrs if y["ev_r"] > 0)
                metrics["n_stable_yrs"] = len(stable_yrs)

        v2_sweep.append(metrics)
        v2_yearly[cd] = yr_data

        # Also store returns/indices for CSCV if this is CD=576
        if cd == 576:
            v2_returns[f"CD={cd}"] = r_returns
            v2_indices[f"CD={cd}"] = trade_idx

        v1_sw = d31_sweep.get(cd, {})
        d_wr = (metrics["win_rate"] - v1_sw.get("win_rate", 0)) * 100
        d_ev = metrics["ev_r"] - v1_sw.get("ev_r", 0)
        marker = " <<< TARGET" if in_target else ""
        print(f"  CD={cd:>3} ({cd*5/60:>4.0f}h): "
              f"{metrics['n_trades']:>6} trades, "
              f"{metrics['trades_per_yr']:>6.0f}/yr, "
              f"WR {metrics['win_rate']:>6.2%} ({d_wr:>+5.1f}pp), "
              f"EV {metrics['ev_r']:>+7.4f}R ({d_ev:>+5.3f}), "
              f"PF {metrics['profit_factor']:>5.2f}, "
              f"MaxDD {metrics['max_dd_pct']:>5.1f}%{marker}")

    # Sweep comparison table
    print(f"\n{rule}")
    print("  v2 vs v1 COOLDOWN COMPARISON")
    print(rule)
    print(f"  {'CD':>5}  {'v2 Tr':>6}  {'v1 Tr':>6}  "
          f"{'v2 WR':>7}  {'v1 WR':>7}  {'dWR':>6}  "
          f"{'v2 EV':>7}  {'v1 EV':>7}  {'dEV':>6}  "
          f"{'v2 DD':>6}  {'v1 DD':>6}  {'Tgt':>3}")
    for r in v2_sweep:
        cd = r["cooldown_bars"]
        v1 = d31_sweep.get(cd, {})
        d_wr = (r["win_rate"] - v1.get("win_rate", 0)) * 100
        d_ev = r["ev_r"] - v1.get("ev_r", 0)
        tgt = "YES" if r["meets_target"] else " no"
        print(f"  {cd:>5}  {r['n_trades']:>6}  "
              f"{v1.get('n_trades', 0):>6}  "
              f"{r['win_rate']:>6.2%}  {v1.get('win_rate', 0):>6.2%}  "
              f"{d_wr:>+5.1f}pp  "
              f"{r['ev_r']:>+6.3f}  {v1.get('ev_r', 0):>+6.3f}  "
              f"{d_ev:>+5.3f}  "
              f"{r['max_dd_pct']:>5.1f}%  "
              f"{v1.get('max_dd_pct', 0):>5.1f}%  "
              f"{tgt:>3}")

    # Per-year breakdown for key cooldowns
    print(f"\n{rule}")
    print("  PER-YEAR BREAKDOWN (v2, years with >= 10 trades)")
    print(rule)
    for cd in [48, 288, 576]:
        yr_data = v2_yearly.get(cd, [])
        stable = [y for y in yr_data if y["trades"] >= 10]
        if not stable:
            continue
        print(f"\n  CD={cd} ({cd*5//60}h cooldown):")
        print(f"    {'Year':>6}  {'Trades':>7}  {'WR':>7}  "
              f"{'EV(R)':>8}  {'PF':>6}")
        for y in stable:
            pf_str = f"{y['profit_factor']:>6.2f}"
            if y["profit_factor"] == float("inf"):
                pf_str = "   inf"
            print(f"    {y['year']:>6}  {y['trades']:>7}  "
                  f"{y['win_rate']:>6.2%}  {y['ev_r']:>+8.4f}  {pf_str}")

    # ================================================================
    # PART C: CSCV on best production config
    # ================================================================
    print(f"\n{sep}")
    print("  PART C: CSCV Validation")
    print(sep)

    # Find best production config from sweep (meeting target or closest)
    target_configs = [r for r in v2_sweep if r["meets_target"]]
    if target_configs:
        prod_config = min(target_configs,
                          key=lambda r: abs(r["trades_per_yr"] - 200))
    else:
        # Fall back to CD=576 (D31 production pick)
        prod_config = [r for r in v2_sweep if r["cooldown_bars"] == 576][0]

    prod_cd = prod_config["cooldown_bars"]
    print(f"  Production config: CD={prod_cd} "
          f"({prod_config['cooldown_hours']:.0f}h), "
          f"{prod_config['trades_per_yr']:.0f} trades/yr")

    # Run simulation for this cooldown to get returns + indices
    prod_idx = simulate(t2_mask, label_arr, prod_cd)
    prod_returns = build_trade_returns(
        prod_idx, label_arr, cfg.r_target, cfg.cost_per_r
    )
    prod_timestamps = timestamps[prod_idx]

    cscv_result = run_cscv(prod_returns, prod_timestamps,
                           f"T2+CD{prod_cd} (v2)")

    # Print CSCV results
    cscv = cscv_result["cscv"]
    psr_00 = cscv_result["psr_benchmark_0"]
    psr_05 = cscv_result["psr_benchmark_05"]
    bootstrap = cscv_result["bootstrap_ci"]
    wf = cscv_result["walk_forward"]

    pbo = cscv["pbo"]
    print(f"\n  PBO:            {pbo:.2%}  "
          f"({cscv['n_negative_oos']}/{cscv['n_combos']} negative OOS)")
    print(f"  OOS mean R:     {cscv['oos_mean']:+.4f}  "
          f"(range [{cscv['oos_min']:+.4f}, {cscv['oos_max']:+.4f}])")
    print(f"  IS-OOS corr:    {cscv['is_oos_correlation']:+.4f}")

    sr = psr_00["sharpe_ratio"]
    print(f"\n  Per-trade SR:   {sr:+.4f}")
    print(f"  PSR(SR > 0):    {psr_00['psr']:.4f}  "
          f"(z = {psr_00['z_score']:+.2f})")
    print(f"  PSR(SR > 0.5):  {psr_05['psr']:.4f}  "
          f"(z = {psr_05['z_score']:+.2f})")

    tpy = prod_config["trades_per_yr"]
    ann_sr = sr * np.sqrt(tpy) if tpy > 0 else 0
    print(f"  Annualized SR:  {ann_sr:+.4f}  (~{tpy:.0f} trades/yr)")

    ci_lo = bootstrap["ci_lower"]
    ci_hi = bootstrap["ci_upper"]
    print(f"\n  Bootstrap 95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"  P(mean R > 0):    {bootstrap['pct_positive']:.1f}%")

    if wf:
        n_profitable = sum(1 for w in wf if w["test_pf"] > 1.0)
        mean_pf = float(np.mean([w["test_pf"] for w in wf]))
        print(f"\n  Walk-forward: {n_profitable}/{len(wf)} "
              f"OOS windows profitable")
        print(f"  OOS mean PF:  {mean_pf:.2f}")

        print(f"\n  {'Win':>5}  {'Train':>7}  {'Test':>6}  "
              f"{'Tr PF':>7}  {'OOS PF':>7}  {'OOS WR':>7}  "
              f"{'OOS EV':>8}")
        for w in wf:
            print(f"  {w['window']:>5}  {w['train_n']:>7}  "
                  f"{w['test_n']:>6}  {w['train_pf']:>7.2f}  "
                  f"{w['test_pf']:>7.2f}  {w['test_wr']:>6.2%}  "
                  f"{w['test_mean_r']:>+8.4f}")

    # ================================================================
    # GO / NO-GO
    # ================================================================
    print(f"\n{sep}")
    print("  GO / NO-GO")
    print(sep)

    checks = [
        ("PBO <= 20%", bool(pbo <= 0.20), f"{pbo:.2%}"),
        ("Bootstrap 95% CI > 0", bool(ci_lo > 0),
         f"[{ci_lo:+.4f}, {ci_hi:+.4f}]"),
        ("PSR(SR > 0) >= 0.95", bool(psr_00["psr"] >= 0.95),
         f"{psr_00['psr']:.4f}"),
        ("T2 (ML>=0.60) still best threshold",
         best_thresh["name"] == "T2 (ML>=0.60)",
         best_thresh["name"]),
        (f"CD={prod_cd} meets production target",
         prod_config["meets_target"],
         f"{prod_config['trades_per_yr']:.0f}/yr, "
         f"WR {prod_config['win_rate']:.2%}, "
         f"EV {prod_config['ev_r']:+.4f}R"),
    ]

    for label, passed, val in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}]  {label:<40}  {val}")

    all_pass = all(c[1] for c in checks)
    if all_pass:
        print(f"\n  >>> GO -- v2 production config VALIDATED")
    else:
        print(f"\n  >>> Results mixed -- see details")

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{sep}")
    print("  D35 SUMMARY")
    print(sep)

    # Answer the 3 key questions
    print(f"\n  Q1: Does ML>=0.60 still hold as optimal threshold?")
    print(f"      -> {best_thresh['name']} is best by Sharpe "
          f"({best_thresh['sharpe_ann']:.2f})")

    target_cds = [r for r in v2_sweep if r["meets_target"]]
    if target_cds:
        best_cd = min(target_cds,
                      key=lambda r: abs(r["trades_per_yr"] - 200))
        print(f"\n  Q2: Does CD=576 still hold as optimal cooldown?")
        print(f"      -> CD={best_cd['cooldown_bars']} is production pick "
              f"({best_cd['trades_per_yr']:.0f}/yr)")
    else:
        print(f"\n  Q2: Does CD=576 still hold as optimal cooldown?")
        cd576 = [r for r in v2_sweep if r["cooldown_bars"] == 576][0]
        print(f"      -> CD=576: {cd576['trades_per_yr']:.0f}/yr, "
              f"WR {cd576['win_rate']:.2%}, EV {cd576['ev_r']:+.4f}R")

    print(f"\n  Q3: Updated production metrics (v2)?")
    print(f"      -> Trades/yr: {prod_config['trades_per_yr']:.0f}")
    print(f"      -> WR: {prod_config['win_rate']:.2%}")
    print(f"      -> EV: {prod_config['ev_r']:+.4f}R")
    print(f"      -> PF: {prod_config['profit_factor']:.2f}")
    print(f"      -> MaxDD: {prod_config['max_dd_pct']:.1f}%")
    print(f"      -> Sharpe: {prod_config['sharpe_ann']:.2f}")

    print(sep)

    # ================================================================
    # Save outputs
    # ================================================================

    # Chart
    v1_sweep_list = d31["sweep_results"]
    chart_path = save_chart(
        v2_configs, d29["configs"], v2_sweep, v1_sweep_list, v2_yearly
    )
    if chart_path:
        print(f"\n  Saved: {chart_path}")

    # JSON
    cscv_slim = {
        k: v for k, v in cscv.items()
        if k not in ("oos_values", "is_values")
    }

    def _round_floats(obj, dp=4):
        if isinstance(obj, dict):
            return {k: _round_floats(v, dp) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_round_floats(v, dp) for v in obj]
        if isinstance(obj, float) and not (obj != obj):
            return round(obj, dp)
        return obj

    summary = {
        "oos_years": round(oos_years, 2),
        "n_features_v2": len(features),
        "part_a_threshold_comparison": {
            "v2_configs": v2_configs,
            "v1_configs": d29["configs"],
            "best_threshold": best_thresh["name"],
        },
        "part_b_cooldown_sweep": {
            "raw_signal_bars": raw_signals,
            "v2_sweep": [
                {k: v for k, v in r.items() if not k.startswith("_")}
                for r in v2_sweep
            ],
            "v1_sweep": v1_sweep_list,
            "per_year": {str(cd): v2_yearly[cd] for cd in COOLDOWNS},
        },
        "part_c_cscv": {
            "production_config": f"T2+CD{prod_cd}",
            "cscv": cscv_slim,
            "psr_benchmark_0": psr_00,
            "psr_benchmark_05": psr_05,
            "bootstrap_ci": bootstrap,
            "walk_forward": wf,
        },
        "go_no_go": {
            "pbo_pass": bool(pbo <= 0.20),
            "ci_pass": bool(ci_lo > 0),
            "psr_pass": bool(psr_00["psr"] >= 0.95),
            "t2_still_best": best_thresh["name"] == "T2 (ML>=0.60)",
            "cd_meets_target": prod_config["meets_target"],
            "all_pass": bool(all_pass),
        },
        "production_config": {
            "threshold": 0.60,
            "cooldown_bars": prod_cd,
            "cooldown_hours": prod_cd * 5 / 60,
            "trades_per_yr": prod_config["trades_per_yr"],
            "win_rate": prod_config["win_rate"],
            "ev_r": prod_config["ev_r"],
            "profit_factor": prod_config["profit_factor"],
            "max_dd_pct": prod_config["max_dd_pct"],
            "sharpe_ann": prod_config["sharpe_ann"],
        },
    }

    summary = _round_floats(summary)

    json_path = RESULTS_DIR / "production_validation_v2.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved: {json_path}")


if __name__ == "__main__":
    main()
