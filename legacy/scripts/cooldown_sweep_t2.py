"""
cooldown_sweep_t2.py -- T2 Cooldown Sensitivity Analysis (D31)
==============================================================
Tests ML>=0.60 threshold at cooldowns: 48, 96, 144, 288, 576 bars
to find production-viable frequency (100-300 trades/yr) while
preserving WR >= 65% and EV >= +0.90R.

Uses ml_backtest.py logic (flat 0.05R cost model, consistent with D29).
Adds per-year temporal stability breakdown for each cooldown.

Output:
  - Console report with sweep table + per-year breakdown
  - results/cooldown_sweep_t2.json
  - results/cooldown_sweep_t2.html
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_backtest_v2 import Config
from ml_pipeline import MLConfig, prepare_data
from ml_backtest import (
    get_oos_probs,
    simulate,
    build_trade_returns,
    equity_sim,
    compute_metrics,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"

# Cooldowns to test (in 5m bars)
COOLDOWNS = [48, 96, 144, 288, 576]

# Production target criteria
TARGET_MIN_TPY = 100    # trades per year
TARGET_MAX_TPY = 300
TARGET_MIN_WR = 0.65    # 65%
TARGET_MIN_EV = 0.90    # +0.90R


# -- per-year breakdown -------------------------------------------------------
def per_year_breakdown(
    trade_indices: list[int],
    label_arr: np.ndarray,
    timestamps: np.ndarray,
    r_target: int,
    cost_per_r: float,
) -> list[dict]:
    """Compute metrics per calendar year for temporal stability."""
    if not trade_indices:
        return []

    r_win = r_target - cost_per_r
    r_loss = -(1 + cost_per_r)

    # Build per-trade data
    years_arr = pd.DatetimeIndex(timestamps[trade_indices]).year.values
    returns = np.array([
        r_win if label_arr[i] == 1 else r_loss for i in trade_indices
    ])

    results = []
    for yr in sorted(set(years_arr)):
        mask = years_arr == yr
        yr_ret = returns[mask]
        n = len(yr_ret)
        if n == 0:
            continue

        wins = int(np.sum(yr_ret > 0))
        wr = wins / n
        ev = float(np.mean(yr_ret))

        gross_win = float(np.sum(yr_ret[yr_ret > 0]))
        gross_loss = float(abs(np.sum(yr_ret[yr_ret < 0])))
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

        results.append({
            "year": int(yr),
            "trades": n,
            "win_rate": round(wr, 4),
            "ev_r": round(ev, 4),
            "profit_factor": round(pf, 2),
        })

    return results


# -- chart --------------------------------------------------------------------
def save_chart(
    sweep_results: list[dict],
    yearly_data: dict[int, list[dict]],
) -> Path | None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return None

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Trades/Year vs Cooldown",
            "Win Rate vs Cooldown",
            "EV (R) vs Cooldown",
            "Per-Year Win Rate by Cooldown",
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.12,
    )

    cds = [r["cooldown_bars"] for r in sweep_results]
    cd_labels = [str(c) for c in cds]

    # Panel 1: Trades/yr
    tpys = [r["trades_per_yr"] for r in sweep_results]
    fig.add_trace(go.Bar(
        x=cd_labels, y=tpys, marker_color="cyan",
        text=[f"{t:.0f}" for t in tpys], textposition="outside",
        showlegend=False,
    ), row=1, col=1)
    fig.add_hline(y=TARGET_MIN_TPY, line_dash="dash", line_color="lime",
                  annotation_text=f"{TARGET_MIN_TPY}", row=1, col=1)
    fig.add_hline(y=TARGET_MAX_TPY, line_dash="dash", line_color="lime",
                  annotation_text=f"{TARGET_MAX_TPY}", row=1, col=1)

    # Panel 2: Win Rate
    wrs = [r["win_rate"] * 100 for r in sweep_results]
    fig.add_trace(go.Bar(
        x=cd_labels, y=wrs, marker_color="orange",
        text=[f"{w:.1f}%" for w in wrs], textposition="outside",
        showlegend=False,
    ), row=1, col=2)
    fig.add_hline(y=TARGET_MIN_WR * 100, line_dash="dash", line_color="red",
                  annotation_text=f"{TARGET_MIN_WR:.0%} target", row=1, col=2)

    # Panel 3: EV
    evs = [r["ev_r"] for r in sweep_results]
    fig.add_trace(go.Bar(
        x=cd_labels, y=evs, marker_color="limegreen",
        text=[f"{e:+.3f}" for e in evs], textposition="outside",
        showlegend=False,
    ), row=2, col=1)
    fig.add_hline(y=TARGET_MIN_EV, line_dash="dash", line_color="red",
                  annotation_text=f"+{TARGET_MIN_EV}R target", row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)

    # Panel 4: Per-year WR by cooldown (line per cooldown)
    colors = {48: "cyan", 96: "orange", 144: "limegreen",
              288: "yellow", 576: "magenta"}
    for cd in cds:
        yd = yearly_data.get(cd, [])
        if not yd:
            continue
        yrs = [d["year"] for d in yd]
        yr_wrs = [d["win_rate"] * 100 for d in yd]
        fig.add_trace(go.Scatter(
            x=[str(y) for y in yrs], y=yr_wrs,
            mode="lines+markers", name=f"CD={cd}",
            line=dict(color=colors.get(cd, "white"), width=1.5),
        ), row=2, col=2)
    fig.add_hline(y=TARGET_MIN_WR * 100, line_dash="dash", line_color="red",
                  row=2, col=2)

    fig.update_layout(
        template="plotly_dark",
        title="T2 Cooldown Sweep -- Production Config Selection (D31)",
        height=900,
    )
    fig.update_xaxes(title_text="Cooldown (bars)", row=1, col=1)
    fig.update_yaxes(title_text="Trades/Year", row=1, col=1)
    fig.update_xaxes(title_text="Cooldown (bars)", row=1, col=2)
    fig.update_yaxes(title_text="Win Rate %", row=1, col=2)
    fig.update_xaxes(title_text="Cooldown (bars)", row=2, col=1)
    fig.update_yaxes(title_text="EV (R)", row=2, col=1)
    fig.update_xaxes(title_text="Year", row=2, col=2)
    fig.update_yaxes(title_text="Win Rate %", row=2, col=2)

    path = RESULTS_DIR / "cooldown_sweep_t2.html"
    fig.write_html(str(path))
    return path


# -- main ---------------------------------------------------------------------
def main() -> None:
    cfg = Config()
    ml_cfg = MLConfig()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data + OOS probs
    print("=" * 70)
    print("  STEP 1: Load Data + OOS Probabilities")
    print("=" * 70)
    df, features = prepare_data(cfg, ml_cfg)

    oos_probs = get_oos_probs(df, features, ml_cfg)
    oos_valid = ~np.isnan(oos_probs)
    n_oos = int(oos_valid.sum())
    oos_years = n_oos / (288 * 365.25)
    print(f"  OOS bars: {n_oos:,} ({oos_years:.2f} years, "
          f"{oos_valid.mean()*100:.1f}% coverage)")

    label_arr = df[ml_cfg.label_col].values
    timestamps = df["bar_start_ts_utc"].values
    t2_mask = (oos_probs >= 0.60) & oos_valid

    # Count raw signal bars (before any cooldown)
    raw_signals = int(t2_mask.sum())
    print(f"  T2 raw signal bars (ML>=0.60): {raw_signals:,} "
          f"({raw_signals / oos_years:.0f}/yr)")

    # 2. Sweep cooldowns
    print(f"\n{'=' * 70}")
    print("  STEP 2: Cooldown Sweep")
    print("=" * 70)
    print(f"  Cooldowns: {COOLDOWNS}")
    print(f"  Cost model: flat {cfg.cost_per_r}R per trade (consistent with D29)")
    print(f"  Risk: {cfg.risk_pct:.0%} per trade, compounding")
    print(f"  Target: {TARGET_MIN_TPY}-{TARGET_MAX_TPY} trades/yr, "
          f"WR >= {TARGET_MIN_WR:.0%}, EV >= +{TARGET_MIN_EV}R")

    sweep_results: list[dict] = []
    yearly_data: dict[int, list[dict]] = {}
    equity_paths: dict[int, list[float]] = {}

    for cd in COOLDOWNS:
        trade_idx = simulate(t2_mask, label_arr, cd)
        r_returns = build_trade_returns(
            trade_idx, label_arr, cfg.r_target, cfg.cost_per_r,
        )
        eq_path, max_dd = equity_sim(
            r_returns, cfg.risk_pct, cfg.initial_equity,
        )

        metrics = compute_metrics(
            f"CD={cd}", r_returns, max_dd, eq_path[-1], oos_years,
        )
        metrics["cooldown_bars"] = cd
        metrics["cooldown_hours"] = cd * 5 / 60

        # Per-year breakdown
        yr_data = per_year_breakdown(
            trade_idx, label_arr, timestamps, cfg.r_target, cfg.cost_per_r,
        )

        # Check target criteria
        in_target = (
            TARGET_MIN_TPY <= metrics["trades_per_yr"] <= TARGET_MAX_TPY
            and metrics["win_rate"] >= TARGET_MIN_WR
            and metrics["ev_r"] >= TARGET_MIN_EV
        )
        metrics["meets_target"] = in_target

        # Temporal stability: worst year WR
        if yr_data:
            # Only consider years with >= 10 trades for stability
            stable_yrs = [y for y in yr_data if y["trades"] >= 10]
            if stable_yrs:
                metrics["worst_yr_wr"] = round(
                    min(y["win_rate"] for y in stable_yrs), 4)
                metrics["worst_yr_ev"] = round(
                    min(y["ev_r"] for y in stable_yrs), 4)
                metrics["n_profitable_yrs"] = sum(
                    1 for y in stable_yrs if y["ev_r"] > 0)
                metrics["n_stable_yrs"] = len(stable_yrs)
            else:
                metrics["worst_yr_wr"] = 0.0
                metrics["worst_yr_ev"] = 0.0
                metrics["n_profitable_yrs"] = 0
                metrics["n_stable_yrs"] = 0

        sweep_results.append(metrics)
        yearly_data[cd] = yr_data
        equity_paths[cd] = eq_path

        marker = " <<< TARGET" if in_target else ""
        print(f"  CD={cd:>3} ({cd*5/60:>4.0f}h): "
              f"{metrics['n_trades']:>6} trades, "
              f"{metrics['trades_per_yr']:>6.0f}/yr, "
              f"WR {metrics['win_rate']:>6.2%}, "
              f"EV {metrics['ev_r']:>+7.4f}R, "
              f"PF {metrics['profit_factor']:>5.2f}, "
              f"MaxDD {metrics['max_dd_pct']:>5.1f}%{marker}")

    # 3. Report
    print(f"\n{'=' * 70}")
    print("  COOLDOWN SWEEP RESULTS -- T2 (ML>=0.60)")
    print("=" * 70)

    sep = "=" * 70
    rule = "-" * 70

    print(f"\n  OOS period: {oos_years:.2f} years  |  "
          f"Raw T2 signal bars: {raw_signals:,} ({raw_signals/oos_years:.0f}/yr)")
    print(f"  Cost: {cfg.cost_per_r}R flat  |  Risk: {cfg.risk_pct:.0%}  |  "
          f"Label: {ml_cfg.label_col}")

    # -- Main sweep table --
    print(f"\n{rule}")
    print("  SWEEP TABLE")
    print(rule)
    print(f"  {'CD':>5}  {'Hours':>5}  {'Trades':>7}  {'Tr/Yr':>6}  "
          f"{'WR':>7}  {'EV(R)':>8}  {'PF':>6}  {'MaxDD':>7}  "
          f"{'Sharpe':>7}  {'Target':>6}")
    print(f"  {'-----':>5}  {'-----':>5}  {'-------':>7}  {'------':>6}  "
          f"{'-------':>7}  {'--------':>8}  {'------':>6}  {'-------':>7}  "
          f"{'-------':>7}  {'------':>6}")
    for r in sweep_results:
        tgt = "YES" if r["meets_target"] else "no"
        pf_str = f"{r['profit_factor']:>6.2f}"
        if r["profit_factor"] == float("inf"):
            pf_str = "   inf"
        print(f"  {r['cooldown_bars']:>5}  {r['cooldown_hours']:>5.0f}  "
              f"{r['n_trades']:>7}  {r['trades_per_yr']:>6.0f}  "
              f"{r['win_rate']:>6.2%}  {r['ev_r']:>+8.4f}  {pf_str}  "
              f"{r['max_dd_pct']:>6.1f}%  {r['sharpe_ann']:>7.2f}  "
              f"{tgt:>6}")

    # -- Per-year breakdown for each cooldown --
    print(f"\n{rule}")
    print("  PER-YEAR BREAKDOWN (years with >= 10 trades)")
    print(rule)

    for cd in COOLDOWNS:
        yr_data = yearly_data[cd]
        stable = [y for y in yr_data if y["trades"] >= 10]
        if not stable:
            print(f"\n  CD={cd}: no years with >= 10 trades")
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

    # -- Temporal stability summary --
    print(f"\n{rule}")
    print("  TEMPORAL STABILITY (years with >= 10 trades)")
    print(rule)
    print(f"  {'CD':>5}  {'Stable Yrs':>10}  {'All Profit':>10}  "
          f"{'Worst WR':>9}  {'Worst EV':>9}")
    for r in sweep_results:
        n_sy = r.get("n_stable_yrs", 0)
        n_py = r.get("n_profitable_yrs", 0)
        w_wr = r.get("worst_yr_wr", 0)
        w_ev = r.get("worst_yr_ev", 0)
        all_p = "YES" if n_py == n_sy and n_sy > 0 else "no"
        print(f"  {r['cooldown_bars']:>5}  {n_sy:>10}  {all_p:>10}  "
              f"{w_wr:>8.2%}  {w_ev:>+9.4f}")

    # -- Recommendation --
    print(f"\n{sep}")
    print("  RECOMMENDATION")
    print(sep)

    # Find configs meeting target
    target_configs = [r for r in sweep_results if r["meets_target"]]
    # Also find configs that are close (relaxed criteria)
    relaxed = [r for r in sweep_results
               if r["win_rate"] >= 0.63 and r["ev_r"] >= 0.80
               and r["trades_per_yr"] >= 50]

    if target_configs:
        # Pick the one closest to center of target range
        best = min(target_configs,
                   key=lambda r: abs(r["trades_per_yr"] - 200))
        print(f"\n  TARGET MET: CD={best['cooldown_bars']} "
              f"({best['cooldown_hours']:.0f}h)")
        print(f"    {best['trades_per_yr']:.0f} trades/yr, "
              f"WR {best['win_rate']:.2%}, "
              f"EV {best['ev_r']:+.4f}R, "
              f"PF {best['profit_factor']:.2f}, "
              f"MaxDD {best['max_dd_pct']:.1f}%")
        print(f"    Worst year WR: {best.get('worst_yr_wr', 0):.2%}  |  "
              f"All years profitable: "
              f"{'YES' if best.get('n_profitable_yrs', 0) == best.get('n_stable_yrs', 0) else 'no'}")
    else:
        print("\n  No cooldown meets all target criteria exactly.")
        if relaxed:
            best = min(relaxed,
                       key=lambda r: abs(r["trades_per_yr"] - 200))
            print(f"\n  CLOSEST (relaxed): CD={best['cooldown_bars']} "
                  f"({best['cooldown_hours']:.0f}h)")
            print(f"    {best['trades_per_yr']:.0f} trades/yr, "
                  f"WR {best['win_rate']:.2%}, "
                  f"EV {best['ev_r']:+.4f}R, "
                  f"PF {best['profit_factor']:.2f}")
        else:
            print("  No viable configs found even with relaxed criteria.")

    # Execution cost note
    print(f"\n  Execution cost note (from D30):")
    print(f"    Actual per-trade cost: ~0.027R (funding-dominated)")
    print(f"    This analysis uses conservative 0.05R flat cost")
    print(f"    Real EV is ~0.023R higher than shown above")

    print(sep)

    # -- Save chart --
    chart_path = save_chart(sweep_results, yearly_data)
    if chart_path:
        print(f"\n  Saved: {chart_path}")
    else:
        print("\n  (plotly not installed -- skipping chart)")

    # -- Save JSON --
    def _round_floats(obj, dp=4):
        if isinstance(obj, dict):
            return {k: _round_floats(v, dp) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_round_floats(v, dp) for v in obj]
        if isinstance(obj, float) and not (obj != obj):
            return round(obj, dp)
        return obj

    # Clean sweep results for JSON (remove non-serialisable)
    clean_results = []
    for r in sweep_results:
        clean_results.append({
            k: v for k, v in r.items()
            if not k.startswith("_")
        })

    summary = {
        "oos_years": round(oos_years, 2),
        "raw_signal_bars": raw_signals,
        "cost_per_r": cfg.cost_per_r,
        "risk_pct": cfg.risk_pct,
        "target_criteria": {
            "min_trades_per_yr": TARGET_MIN_TPY,
            "max_trades_per_yr": TARGET_MAX_TPY,
            "min_win_rate": TARGET_MIN_WR,
            "min_ev_r": TARGET_MIN_EV,
        },
        "sweep_results": clean_results,
        "per_year": {str(cd): yearly_data[cd] for cd in COOLDOWNS},
        "recommendation": (
            {
                "cooldown_bars": best["cooldown_bars"],
                "cooldown_hours": best["cooldown_hours"],
                "trades_per_yr": best["trades_per_yr"],
                "win_rate": best["win_rate"],
                "ev_r": best["ev_r"],
                "profit_factor": best["profit_factor"],
                "max_dd_pct": best["max_dd_pct"],
                "meets_strict_target": best["meets_target"],
            }
            if target_configs or relaxed else None
        ),
    }

    summary = _round_floats(summary)

    json_path = RESULTS_DIR / "cooldown_sweep_t2.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved: {json_path}")


if __name__ == "__main__":
    main()
