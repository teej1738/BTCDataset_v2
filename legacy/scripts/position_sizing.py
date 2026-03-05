"""
position_sizing.py -- Position Sizing Models for Production Config (D32)
========================================================================
Compares three sizing models on the production trade set
(ML>=0.60, CD=576, ~179 trades/yr from D31):

  1. Fixed fractional: 1% risk per trade regardless of ML score
  2. Kelly fraction: fractional Kelly using per-trade ML probability
     as win-rate estimate, f = frac * (p - (1-p)/R), capped at 2%
  3. Volatility-adjusted: 1% base * (1 / ict_atr_ratio) so position
     shrinks in high-vol regimes, capped at [0.25%, 2%]

For each: CAGR, max DD, annualized Sharpe, equity curve.

Output:
  - Console report with side-by-side comparison
  - results/position_sizing.json
  - results/position_sizing.html
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_backtest_v2 import Config
from ml_pipeline import MLConfig, prepare_data
from ml_backtest import get_oos_probs, simulate, build_trade_returns

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"


# -- config -------------------------------------------------------------------
@dataclass
class SizingConfig:
    cooldown_bars: int = 576        # production cooldown (D31)
    ml_threshold: float = 0.60      # T2 threshold
    r_target: int = 2               # R-multiple target
    cost_per_r: float = 0.05        # flat cost (conservative, D30 actual ~0.027)
    initial_equity: float = 10_000.0
    # Fixed fractional
    fixed_risk: float = 0.01        # 1% per trade
    # Kelly
    kelly_frac: float = 1.0 / 40    # 2.5% Kelly -- calibrated so p=0.60 -> 1%
    kelly_floor: float = 0.0025     # 0.25% minimum
    kelly_cap: float = 0.02         # 2% maximum
    # Volatility-adjusted
    vol_base_risk: float = 0.01     # 1% base
    vol_floor: float = 0.0025       # 0.25% minimum
    vol_cap: float = 0.02           # 2% maximum


# -- variable-risk equity simulation -----------------------------------------
def equity_sim_variable(
    r_returns: np.ndarray,
    risk_pcts: np.ndarray,
    initial_equity: float,
) -> tuple[list[float], float]:
    """Equity path with per-trade variable risk. Returns (path, max_dd)."""
    equity = initial_equity
    peak = equity
    max_dd = 0.0
    path = [equity]

    for r, risk in zip(r_returns, risk_pcts):
        pnl = equity * risk * r
        equity += pnl
        if equity <= 0:
            equity = 0.0
            path.append(equity)
            max_dd = 1.0
            break
        peak = max(peak, equity)
        dd = (peak - equity) / peak
        max_dd = max(max_dd, dd)
        path.append(equity)

    return path, max_dd


# -- per-trade risk computation -----------------------------------------------
def compute_fixed_risk(n_trades: int, scfg: SizingConfig) -> np.ndarray:
    """Fixed fractional: constant risk per trade."""
    return np.full(n_trades, scfg.fixed_risk)


def compute_kelly_risk(
    ml_probs: np.ndarray, scfg: SizingConfig,
) -> np.ndarray:
    """
    Fractional Kelly using per-trade ML probability.
    f = kelly_frac * (p - (1-p) / R), clipped to [floor, cap].

    With kelly_frac=1/40 and R=2:
      p=0.60 -> 1.0%    p=0.70 -> 1.4%
      p=0.80 -> 1.8%    p=0.90 -> 2.0% (capped)
    """
    kelly_raw = ml_probs - (1.0 - ml_probs) / scfg.r_target
    kelly_raw = np.maximum(kelly_raw, 0.0)  # floor at 0 if negative edge
    f = scfg.kelly_frac * kelly_raw
    return np.clip(f, scfg.kelly_floor, scfg.kelly_cap)


def compute_vol_risk(
    atr_ratios: np.ndarray, scfg: SizingConfig,
) -> np.ndarray:
    """
    Volatility-adjusted: base_risk * (1 / atr_ratio).
    High vol = smaller position, low vol = larger position.
      atr_ratio=0.8 -> 1.25%    atr_ratio=1.0 -> 1.0%
      atr_ratio=1.5 -> 0.67%    atr_ratio=2.0 -> 0.50%
    """
    safe_atr = np.maximum(atr_ratios, 0.1)  # avoid division by zero
    f = scfg.vol_base_risk / safe_atr
    return np.clip(f, scfg.vol_floor, scfg.vol_cap)


# -- metrics ------------------------------------------------------------------
def compute_sizing_metrics(
    name: str,
    r_returns: np.ndarray,
    risk_pcts: np.ndarray,
    eq_path: list[float],
    max_dd: float,
    years: float,
) -> dict:
    """Compute CAGR, Sharpe, and other metrics for a sizing model."""
    n = len(r_returns)
    if n == 0:
        return {"name": name, "n_trades": 0}

    initial = eq_path[0]
    final = eq_path[-1]

    # CAGR
    if final > 0 and initial > 0 and years > 0:
        cagr = (final / initial) ** (1.0 / years) - 1.0
    else:
        cagr = -1.0

    # Per-trade portfolio returns (percentage of equity)
    pct_returns = risk_pcts * r_returns
    mean_pct = float(np.mean(pct_returns))
    std_pct = float(np.std(pct_returns, ddof=1)) if n > 1 else 0.0
    sharpe_pt = mean_pct / std_pct if std_pct > 0 else 0.0
    tpy = n / years if years > 0 else 0.0
    sharpe_ann = sharpe_pt * np.sqrt(tpy) if tpy > 0 else 0.0

    # Win/loss
    wins = int(np.sum(r_returns > 0))
    wr = wins / n
    ev = float(np.mean(r_returns))

    # PF
    gross_win = float(np.sum(r_returns[r_returns > 0]))
    gross_loss = float(abs(np.sum(r_returns[r_returns < 0])))
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

    # Risk distribution stats
    mean_risk = float(np.mean(risk_pcts))
    min_risk = float(np.min(risk_pcts))
    max_risk = float(np.max(risk_pcts))
    std_risk = float(np.std(risk_pcts))

    return {
        "name": name,
        "n_trades": n,
        "trades_per_yr": round(tpy, 1),
        "win_rate": round(wr, 4),
        "ev_r": round(ev, 4),
        "profit_factor": round(pf, 2),
        "cagr": round(cagr, 4),
        "max_dd_pct": round(max_dd * 100, 2),
        "sharpe_ann": round(sharpe_ann, 4),
        "initial_equity": round(initial, 2),
        "final_equity": round(final, 2),
        "mean_risk_pct": round(mean_risk * 100, 3),
        "min_risk_pct": round(min_risk * 100, 3),
        "max_risk_pct": round(max_risk * 100, 3),
        "std_risk_pct": round(std_risk * 100, 4),
    }


# -- chart --------------------------------------------------------------------
def save_chart(
    models: list[dict],
    equity_paths: dict[str, list[float]],
    risk_arrays: dict[str, np.ndarray],
) -> Path | None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return None

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Equity Curves",
            "Risk Distribution per Model",
            "CAGR / MaxDD / Sharpe Comparison",
            "Per-Trade Risk (Kelly model)",
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.12,
    )

    colors = {
        "Fixed 1%": "cyan",
        "Kelly": "orange",
        "Vol-Adjusted": "limegreen",
    }

    # Panel 1: Equity curves
    for m in models:
        name = m["name"]
        eq = equity_paths[name]
        fig.add_trace(go.Scatter(
            x=list(range(len(eq))), y=eq,
            mode="lines", name=name,
            line=dict(color=colors.get(name, "white"), width=1.5),
        ), row=1, col=1)

    # Panel 2: Risk distribution histograms
    for name, rarr in risk_arrays.items():
        fig.add_trace(go.Histogram(
            x=rarr * 100, name=f"{name} risk",
            opacity=0.5, nbinsx=30,
            marker_color=colors.get(name, "white"),
            showlegend=False,
        ), row=1, col=2)

    # Panel 3: Bar comparison
    names = [m["name"] for m in models]
    cagrs = [m["cagr"] * 100 for m in models]
    dds = [m["max_dd_pct"] for m in models]
    sharpes = [m["sharpe_ann"] for m in models]

    fig.add_trace(go.Bar(
        x=names, y=cagrs, name="CAGR %",
        marker_color="cyan", showlegend=False,
        text=[f"{c:.1f}%" for c in cagrs], textposition="outside",
    ), row=2, col=1)

    # Panel 4: Kelly per-trade risk scatter
    kelly_risk = risk_arrays.get("Kelly")
    if kelly_risk is not None:
        fig.add_trace(go.Scatter(
            x=list(range(len(kelly_risk))),
            y=kelly_risk * 100,
            mode="markers", name="Kelly risk",
            marker=dict(size=2, color="orange", opacity=0.5),
            showlegend=False,
        ), row=2, col=2)

    fig.update_layout(
        template="plotly_dark",
        title="Position Sizing Models -- Production Config (D32)",
        height=900,
    )
    fig.update_xaxes(title_text="Trade #", row=1, col=1)
    fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig.update_xaxes(title_text="Risk per Trade (%)", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_xaxes(title_text="Model", row=2, col=1)
    fig.update_yaxes(title_text="CAGR %", row=2, col=1)
    fig.update_xaxes(title_text="Trade #", row=2, col=2)
    fig.update_yaxes(title_text="Risk %", row=2, col=2)

    path = RESULTS_DIR / "position_sizing.html"
    fig.write_html(str(path))
    return path


# -- report -------------------------------------------------------------------
def print_report(
    models: list[dict],
    scfg: SizingConfig,
    oos_years: float,
    equity_paths: dict[str, list[float]],
    risk_arrays: dict[str, np.ndarray],
) -> dict:
    """Console report + JSON save. Returns summary dict."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    sep = "=" * 70
    rule = "-" * 70

    print(f"\n{sep}")
    print("  POSITION SIZING -- Production Config (D32)")
    print(sep)
    print(f"  Trade set: ML>=0.60, CD={scfg.cooldown_bars} "
          f"({scfg.cooldown_bars * 5 // 60}h)")
    print(f"  OOS period: {oos_years:.2f} years  |  "
          f"Cost: {scfg.cost_per_r}R flat")
    print(f"  Initial equity: ${scfg.initial_equity:,.0f}")

    # -- Sizing model descriptions --
    print(f"\n{rule}")
    print("  SIZING MODELS")
    print(rule)
    print(f"  1. Fixed 1%:     {scfg.fixed_risk:.1%} risk every trade")
    print(f"  2. Kelly:        {scfg.kelly_frac:.4f} * "
          f"(p - (1-p)/{scfg.r_target}), "
          f"clipped [{scfg.kelly_floor:.2%}, {scfg.kelly_cap:.1%}]")
    print(f"                   p=0.60 -> 1.0%,  p=0.70 -> 1.4%,  "
          f"p=0.80 -> 1.8%,  p>=0.86 -> 2.0%")
    print(f"  3. Vol-Adjusted: {scfg.vol_base_risk:.1%} / atr_ratio, "
          f"clipped [{scfg.vol_floor:.2%}, {scfg.vol_cap:.1%}]")
    print(f"                   atr=0.8 -> 1.25%,  atr=1.0 -> 1.0%,  "
          f"atr=1.5 -> 0.67%")

    # -- Side-by-side comparison --
    print(f"\n{rule}")
    print("  MODEL COMPARISON")
    print(rule)

    header = (f"  {'Metric':<22}")
    for m in models:
        header += f"  {m['name']:>14}"
    print(header)
    print(f"  {'-' * 22}" + f"  {'-' * 14}" * len(models))

    rows = [
        ("Trades", "n_trades", "{}"),
        ("Trades/yr", "trades_per_yr", "{:.0f}"),
        ("Win Rate", "win_rate", "{:.2%}"),
        ("EV (R)", "ev_r", "{:+.4f}"),
        ("Profit Factor", "profit_factor", "{:.2f}"),
        ("CAGR", "cagr", "{:.2%}"),
        ("Max Drawdown", "max_dd_pct", "{:.1f}%"),
        ("Sharpe (ann)", "sharpe_ann", "{:.2f}"),
        ("Final Equity", "final_equity", "${:,.0f}"),
        ("Mean Risk", "mean_risk_pct", "{:.3f}%"),
        ("Min Risk", "min_risk_pct", "{:.3f}%"),
        ("Max Risk", "max_risk_pct", "{:.3f}%"),
    ]

    for label, key, fmt in rows:
        line = f"  {label:<22}"
        for m in models:
            val = m.get(key, 0)
            line += f"  {fmt.format(val):>14}"
        print(line)

    # -- Risk distribution detail --
    print(f"\n{rule}")
    print("  RISK DISTRIBUTION DETAIL")
    print(rule)
    for name, rarr in risk_arrays.items():
        pcts = rarr * 100
        p25 = float(np.percentile(pcts, 25))
        p50 = float(np.percentile(pcts, 50))
        p75 = float(np.percentile(pcts, 75))
        print(f"  {name:<14}  "
              f"mean={np.mean(pcts):.3f}%  "
              f"median={p50:.3f}%  "
              f"p25={p25:.3f}%  p75={p75:.3f}%  "
              f"std={np.std(pcts):.4f}%")

    # -- Assessment --
    print(f"\n{sep}")
    print("  ASSESSMENT")
    print(sep)

    # Find best by Sharpe, best by CAGR, best by max DD
    best_sharpe = max(models, key=lambda m: m["sharpe_ann"])
    best_cagr = max(models, key=lambda m: m["cagr"])
    best_dd = min(models, key=lambda m: m["max_dd_pct"])

    print(f"\n  Best Sharpe:   {best_sharpe['name']:<14}  "
          f"({best_sharpe['sharpe_ann']:.2f})")
    print(f"  Best CAGR:     {best_cagr['name']:<14}  "
          f"({best_cagr['cagr']:.2%})")
    print(f"  Lowest MaxDD:  {best_dd['name']:<14}  "
          f"({best_dd['max_dd_pct']:.1f}%)")

    # Recommendation
    # Prefer the model with best risk-adjusted return (Sharpe)
    # unless another model has substantially better CAGR with acceptable DD
    rec = best_sharpe
    print(f"\n  Recommendation: {rec['name']}")
    print(f"    CAGR {rec['cagr']:.2%},  MaxDD {rec['max_dd_pct']:.1f}%,  "
          f"Sharpe {rec['sharpe_ann']:.2f}")
    print(f"    Risk range: {rec['min_risk_pct']:.3f}% - "
          f"{rec['max_risk_pct']:.3f}%  (mean {rec['mean_risk_pct']:.3f}%)")

    print(sep)

    # -- Save chart --
    chart_path = save_chart(models, equity_paths, risk_arrays)
    if chart_path:
        print(f"\n  Saved: {chart_path}")
    else:
        print("\n  (plotly not installed -- skipping chart)")

    # -- Save JSON --
    def _round_floats(obj, dp=6):
        if isinstance(obj, dict):
            return {k: _round_floats(v, dp) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_round_floats(v, dp) for v in obj]
        if isinstance(obj, float) and not (obj != obj):
            return round(obj, dp)
        return obj

    summary = {
        "config": {
            "cooldown_bars": scfg.cooldown_bars,
            "ml_threshold": scfg.ml_threshold,
            "r_target": scfg.r_target,
            "cost_per_r": scfg.cost_per_r,
            "initial_equity": scfg.initial_equity,
            "fixed_risk": scfg.fixed_risk,
            "kelly_frac": scfg.kelly_frac,
            "kelly_floor": scfg.kelly_floor,
            "kelly_cap": scfg.kelly_cap,
            "vol_base_risk": scfg.vol_base_risk,
            "vol_floor": scfg.vol_floor,
            "vol_cap": scfg.vol_cap,
        },
        "oos_years": round(oos_years, 2),
        "models": models,
        "recommendation": rec["name"],
    }

    summary = _round_floats(summary)

    json_path = RESULTS_DIR / "position_sizing.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved: {json_path}")

    return summary


# -- main ---------------------------------------------------------------------
def main() -> None:
    cfg = Config()
    ml_cfg = MLConfig()
    scfg = SizingConfig()

    # 1. Load data + OOS probs
    print("=" * 70)
    print("  STEP 1: Load Data + OOS Probabilities")
    print("=" * 70)
    df, features = prepare_data(cfg, ml_cfg)

    oos_probs = get_oos_probs(df, features, ml_cfg)
    oos_valid = ~np.isnan(oos_probs)
    n_oos = int(oos_valid.sum())
    oos_years = n_oos / (288 * 365.25)
    print(f"  OOS bars: {n_oos:,} ({oos_years:.2f} years)")

    # 2. Build T2 mask + simulate with CD=576
    print(f"\n{'=' * 70}")
    print("  STEP 2: Build Trade Set (ML>=0.60, CD=576)")
    print("=" * 70)

    label_arr = df[ml_cfg.label_col].values
    t2_mask = (oos_probs >= scfg.ml_threshold) & oos_valid

    trade_indices = simulate(t2_mask, label_arr, scfg.cooldown_bars)
    n_trades = len(trade_indices)
    print(f"  Trades: {n_trades} ({n_trades / oos_years:.0f}/yr)")

    # 3. Extract per-trade data
    trade_idx_arr = np.array(trade_indices)
    ml_probs = oos_probs[trade_idx_arr]
    atr_ratios = df["ict_atr_ratio"].values[trade_idx_arr]

    # Handle NaN atr_ratios (shouldn't happen, but be safe)
    nan_mask = np.isnan(atr_ratios)
    if nan_mask.any():
        atr_ratios[nan_mask] = 1.0
        print(f"  WARNING: {int(nan_mask.sum())} trades with NaN atr_ratio "
              f"(set to 1.0)")

    # Build R returns (same for all models -- win/loss doesn't change)
    r_returns = build_trade_returns(
        trade_indices, label_arr, scfg.r_target, scfg.cost_per_r,
    )

    print(f"  ML prob range: [{ml_probs.min():.3f}, {ml_probs.max():.3f}]  "
          f"mean={ml_probs.mean():.3f}")
    print(f"  ATR ratio range: [{atr_ratios.min():.3f}, "
          f"{atr_ratios.max():.3f}]  mean={atr_ratios.mean():.3f}")

    # 4. Compute per-trade risk for each model
    print(f"\n{'=' * 70}")
    print("  STEP 3: Compute Position Sizes + Equity Simulations")
    print("=" * 70)

    sizing_models = [
        ("Fixed 1%", compute_fixed_risk(n_trades, scfg)),
        ("Kelly", compute_kelly_risk(ml_probs, scfg)),
        ("Vol-Adjusted", compute_vol_risk(atr_ratios, scfg)),
    ]

    models_metrics: list[dict] = []
    equity_paths: dict[str, list[float]] = {}
    risk_arrays: dict[str, np.ndarray] = {}

    for name, risk_pcts in sizing_models:
        eq_path, max_dd = equity_sim_variable(
            r_returns, risk_pcts, scfg.initial_equity,
        )
        metrics = compute_sizing_metrics(
            name, r_returns, risk_pcts, eq_path, max_dd, oos_years,
        )
        models_metrics.append(metrics)
        equity_paths[name] = eq_path
        risk_arrays[name] = risk_pcts

        print(f"  {name:<14}: CAGR {metrics['cagr']:.2%},  "
              f"MaxDD {metrics['max_dd_pct']:.1f}%,  "
              f"Sharpe {metrics['sharpe_ann']:.2f},  "
              f"Final ${metrics['final_equity']:,.0f}  "
              f"(mean risk {metrics['mean_risk_pct']:.3f}%)")

    # 5. Report
    print_report(
        models_metrics, scfg, oos_years, equity_paths, risk_arrays,
    )


if __name__ == "__main__":
    main()
