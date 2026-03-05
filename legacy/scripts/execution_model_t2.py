"""
execution_model_t2.py -- Execution Cost Analysis for T2/T3 (D30)
================================================================
Applies realistic per-trade execution costs (latency, market impact,
funding) to the ML-scored trade sets from ml_backtest.py.

Key insight: with cooldown=48 and hold=48, at most 1 position is open
at a time, so the isolated-trade cost model from execution_model.py
applies directly. Per-trade cost is ~0.025R (same as D27), but annual
drag at ~1800 trades/year is the critical new metric.

Two configs:
  T2: ML prob >= 0.60, 48-bar cooldown (high frequency, ~1800 trades/yr)
  T3: Config B AND ML prob >= 0.50, 48-bar cooldown (low frequency)

Output:
  - Console report with T2 vs T3 side-by-side
  - results/execution_model_t2.json
  - results/execution_model_t2.html
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_backtest_v2 import Config, load_labeled
from mtf_signals import TF_CONFIGS, config_b_filters
from ml_pipeline import MLConfig, prepare_data
from ml_backtest import get_oos_probs, build_config_b_mask, simulate, equity_sim

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"


# -- config -------------------------------------------------------------------
@dataclass
class ExecConfigT2:
    impact_k: float = 0.10          # Kyle's lambda for BTC futures
    position_btc: float = 0.003     # ~$200 at BTC=$67k
    adv_window: int = 288           # 1 day of 5m bars for ADV
    regime_impact_mult: float = 1.5 # high-vol regime impact multiplier
    atr_ratio_high: float = 1.2     # threshold for regime multiplier
    hold_bars: int = 48             # hold window matching label horizon
    r_target: int = 2               # R-multiple target
    cooldown_bars: int = 48         # cooldown between trades
    risk_pct_t2: float = 0.005      # 0.5% per trade (high frequency)
    risk_pct_t3: float = 0.02       # 2% per trade (low frequency)
    initial_equity: float = 10_000.0


# -- build signal table from ML trade indices ---------------------------------
def build_signal_table_ml(
    df: pd.DataFrame,
    trade_indices: list[int],
    ecfg: ExecConfigT2,
    label_col: str,
) -> pd.DataFrame:
    """
    Build per-trade table from ML-generated trade indices.
    Extracts fields needed for execution cost computation.
    No TF source or FVG CE columns (not applicable to ML-scored trades).
    """
    n_rows = len(df)

    # Precompute ADV (rolling 288-bar sum of volume_base)
    adv = df["volume_base"].rolling(ecfg.adv_window, min_periods=1).sum().values

    # Precompute cumulative funding for fast window sums
    fund_rate = df["fund_rate_period"].fillna(0).values
    fund_cum = np.concatenate([[0.0], np.cumsum(fund_rate)])

    # Arrays for fast positional access
    close_arr = df["close"].values
    open_arr = df["open"].values
    atr_arr = df["ict_atr_14"].values
    atr_ratio_arr = df["ict_atr_ratio"].values
    vol_arr = df["ict_realized_vol_20"].values
    label_arr = df[label_col].values
    mfe_arr = df["label_max_up_pct_48c"].values
    ts_arr = df["bar_start_ts_utc"].values

    records = []
    for pos in trade_indices:
        # Need at least hold_bars of forward data + 1 bar for next_open
        if pos + ecfg.hold_bars >= n_rows:
            continue

        cl = close_arr[pos]
        atr = atr_arr[pos]

        if np.isnan(atr) or atr <= 0 or np.isnan(cl) or cl <= 0:
            continue

        # Funding sum over hold window: bars t+1 through t+hold_bars
        fund_end = min(pos + 1 + ecfg.hold_bars, n_rows)
        funding_sum = float(fund_cum[fund_end] - fund_cum[pos + 1])

        sigma = vol_arr[pos]

        records.append({
            "bar_idx": int(pos),
            "timestamp": ts_arr[pos],
            "close": float(cl),
            "atr": float(atr),
            "atr_ratio": float(atr_ratio_arr[pos]),
            "sigma": float(sigma) if not np.isnan(sigma) else 0.0,
            "next_open": float(open_arr[pos + 1]),
            "win": bool(label_arr[pos] == 1.0),
            "mfe_pct": float(mfe_arr[pos]) if not np.isnan(mfe_arr[pos]) else 0.0,
            "funding_sum": funding_sum,
            "adv": float(adv[pos]),
        })

    return pd.DataFrame(records)


# -- cost components (same formulas as execution_model.py) --------------------
def compute_latency(sig: pd.DataFrame) -> np.ndarray:
    """Latency cost in R: (open[t+1] - close[t]) / ATR. Positive = adverse."""
    return (sig["next_open"].values - sig["close"].values) / sig["atr"].values


def compute_impact(sig: pd.DataFrame, ecfg: ExecConfigT2) -> np.ndarray:
    """
    Square-root market impact in R.
    raw_pct = k * sigma * sqrt(Q / ADV), regime-conditional.
    impact_r = raw_pct / (atr / close).
    """
    sigma = sig["sigma"].values
    safe_adv = np.maximum(sig["adv"].values, 1e-10)
    raw_pct = ecfg.impact_k * sigma * np.sqrt(ecfg.position_btc / safe_adv)

    regime_mult = np.where(
        sig["atr_ratio"].values > ecfg.atr_ratio_high,
        ecfg.regime_impact_mult, 1.0,
    )
    raw_pct = raw_pct * regime_mult

    atr_frac = np.maximum(sig["atr"].values / sig["close"].values, 1e-10)
    return raw_pct / atr_frac


def compute_funding(sig: pd.DataFrame) -> np.ndarray:
    """
    Funding cost in R. Longs pay positive funding.
    funding_r = sum(rate over hold) * close / atr.
    """
    return sig["funding_sum"].values * sig["close"].values / sig["atr"].values


# -- win/loss flip detection (market entry only) ------------------------------
def flip_detection_market(
    sig: pd.DataFrame,
    latency_r: np.ndarray,
    impact_r: np.ndarray,
    ecfg: ExecConfigT2,
) -> np.ndarray:
    """
    Market entry flip detection (unidirectional, win->loss only).
    Flip win to loss if MFE_price < 2*ATR + latency_price + impact_price.
    Returns adjusted boolean win array.
    """
    atr = sig["atr"].values
    win = sig["win"].values.copy()
    mfe_price = sig["close"].values * sig["mfe_pct"].values / 100.0

    latency_price = latency_r * atr
    impact_price = impact_r * atr

    market_required = ecfg.r_target * atr + latency_price + impact_price
    market_win = win.copy()
    market_win[win & (mfe_price < market_required)] = False

    return market_win


# -- central execution cost analysis -----------------------------------------
def compute_execution_costs(
    sig: pd.DataFrame,
    ecfg: ExecConfigT2,
    name: str,
    oos_years: float,
    risk_pct: float,
) -> dict:
    """
    Compute execution costs for one trade set.
    Returns metrics dict with cost breakdown, adjusted performance,
    annual drag, equity simulation, and internal arrays for charting.
    """
    n_total = len(sig)
    if n_total == 0:
        return {
            "name": name, "n": 0, "trades_per_year": 0.0,
            "mean_r_adjusted": 0.0, "wr_adjusted": 0.0,
            "annualized_sharpe": 0.0, "risk_pct": risk_pct,
        }

    latency_r = compute_latency(sig)
    impact_r = compute_impact(sig, ecfg)
    funding_r = compute_funding(sig)

    market_win = flip_detection_market(sig, latency_r, impact_r, ecfg)

    # Adjusted R per trade: theoretical outcome - all costs
    market_theo = np.where(market_win, float(ecfg.r_target), -1.0)
    market_adj = market_theo - (latency_r + impact_r + funding_r)

    n_wins_th = int(sig["win"].sum())
    n_wins_flip = int(market_win.sum())
    # WR from adjusted R sign (consistent with execution_model.py)
    n_wins_adj = int(np.sum(market_adj > 0))

    # Profit factor from adjusted returns
    adj_win_vals = market_adj[market_adj > 0]
    adj_loss_vals = market_adj[market_adj <= 0]
    gross_win = float(np.sum(adj_win_vals)) if len(adj_win_vals) else 0.0
    gross_loss = float(np.sum(np.abs(adj_loss_vals))) if len(adj_loss_vals) else 0.0

    # Sharpe
    mean_adj = float(np.mean(market_adj))
    std_adj = float(np.std(market_adj, ddof=1)) if n_total > 1 else 0.0
    per_trade_sharpe = mean_adj / std_adj if std_adj > 0 else 0.0
    trades_per_year = n_total / oos_years if oos_years > 0 else 0.0
    ann_sharpe = (
        per_trade_sharpe * np.sqrt(trades_per_year)
        if trades_per_year > 0 else 0.0
    )

    # Equity simulation
    eq_path, max_dd = equity_sim(market_adj, risk_pct, ecfg.initial_equity)

    # Annual cost analysis
    mean_latency = float(np.mean(latency_r))
    mean_impact = float(np.mean(impact_r))
    mean_funding = float(np.mean(funding_r))
    mean_total_cost = mean_latency + mean_impact + mean_funding

    return {
        "name": name,
        "n": n_total,
        "n_wins_theoretical": n_wins_th,
        "n_wins_flipped": n_wins_flip,
        "n_flipped": abs(n_wins_th - n_wins_flip),
        "n_wins_adjusted": n_wins_adj,
        "wr_theoretical": n_wins_th / n_total,
        "wr_adjusted": n_wins_adj / n_total,
        "mean_latency_r": mean_latency,
        "median_latency_r": float(np.median(latency_r)),
        "std_latency_r": float(np.std(latency_r)),
        "mean_impact_r": mean_impact,
        "median_impact_r": float(np.median(impact_r)),
        "std_impact_r": float(np.std(impact_r)),
        "mean_funding_r": mean_funding,
        "median_funding_r": float(np.median(funding_r)),
        "std_funding_r": float(np.std(funding_r)),
        "mean_total_cost_r": mean_total_cost,
        "mean_r_theoretical": float(np.mean(
            np.where(sig["win"].values, ecfg.r_target, -1.0)
        )),
        "mean_r_adjusted": mean_adj,
        "pf": gross_win / gross_loss if gross_loss > 0 else float("inf"),
        "per_trade_sharpe": per_trade_sharpe,
        "annualized_sharpe": float(ann_sharpe),
        "trades_per_year": trades_per_year,
        "risk_pct": risk_pct,
        "equity_final": eq_path[-1],
        "max_dd_pct": max_dd * 100,
        "annual_latency_drag": mean_latency * trades_per_year,
        "annual_impact_drag": mean_impact * trades_per_year,
        "annual_funding_drag": mean_funding * trades_per_year,
        "annual_total_drag": mean_total_cost * trades_per_year,
        # Internal arrays for charting (not serialised)
        "_latency_r": latency_r,
        "_impact_r": impact_r,
        "_funding_r": funding_r,
        "_adjusted_returns": market_adj,
        "_equity_path": eq_path,
    }


# -- chart --------------------------------------------------------------------
def save_chart(t2: dict, t3: dict) -> Path | None:
    """3-panel Plotly chart: cost distribution, T2/T3 comparison, equity."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return None

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "T2 Cost Distribution (R per trade)",
            "T2 vs T3 Comparison",
            "T2 Equity Curve (0.5% risk)",
            "Annual Cost Drag (R/year)",
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.12,
    )

    # Panel 1: Cost distribution histograms (T2)
    fig.add_trace(go.Histogram(
        x=t2["_latency_r"], name="Latency", opacity=0.6,
        marker_color="#4a86c8", nbinsx=40,
    ), row=1, col=1)
    fig.add_trace(go.Histogram(
        x=t2["_impact_r"], name="Impact", opacity=0.6,
        marker_color="#5cb85c", nbinsx=40,
    ), row=1, col=1)
    fig.add_trace(go.Histogram(
        x=t2["_funding_r"], name="Funding", opacity=0.6,
        marker_color="#f0ad4e", nbinsx=40,
    ), row=1, col=1)
    fig.update_layout(barmode="overlay")

    # Panel 2: T2 vs T3 bar comparison
    metrics = ["EV (R)", "WR (%)", "Sharpe"]
    t2_vals = [t2["mean_r_adjusted"], t2["wr_adjusted"] * 100,
               t2["annualized_sharpe"]]
    t3_vals = [t3["mean_r_adjusted"], t3["wr_adjusted"] * 100,
               t3["annualized_sharpe"]]
    fig.add_trace(go.Bar(
        x=metrics, y=t2_vals, name="T2 (ML>=0.60)",
        marker_color="orange",
    ), row=1, col=2)
    fig.add_trace(go.Bar(
        x=metrics, y=t3_vals, name="T3 (CB+ML>=0.50)",
        marker_color="limegreen",
    ), row=1, col=2)

    # Panel 3: T2 equity curve
    eq = t2["_equity_path"]
    fig.add_trace(go.Scatter(
        x=list(range(len(eq))), y=eq,
        mode="lines", name="T2 Equity",
        line=dict(color="orange", width=1.5),
        showlegend=False,
    ), row=2, col=1)

    # Panel 4: Annual cost drag
    cats = ["Latency", "Impact", "Funding", "Total"]
    t2_drags = [t2["annual_latency_drag"], t2["annual_impact_drag"],
                t2["annual_funding_drag"], t2["annual_total_drag"]]
    t3_drags = [t3["annual_latency_drag"], t3["annual_impact_drag"],
                t3["annual_funding_drag"], t3["annual_total_drag"]]
    fig.add_trace(go.Bar(
        x=cats, y=t2_drags, name="T2 drag", marker_color="orange",
        showlegend=False,
    ), row=2, col=2)
    fig.add_trace(go.Bar(
        x=cats, y=t3_drags, name="T3 drag", marker_color="limegreen",
        showlegend=False,
    ), row=2, col=2)

    fig.update_layout(
        template="plotly_dark",
        title="Execution Cost Analysis -- T2 / T3 (D30)",
        height=900,
    )
    fig.update_xaxes(title_text="Cost (R)", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="Trade #", row=2, col=1)
    fig.update_yaxes(title_text="Equity ($)", row=2, col=1)
    fig.update_xaxes(title_text="Component", row=2, col=2)
    fig.update_yaxes(title_text="R / Year", row=2, col=2)

    path = RESULTS_DIR / "execution_model_t2.html"
    fig.write_html(str(path))
    return path


# -- report -------------------------------------------------------------------
def print_report(
    t2: dict, t3: dict, ecfg: ExecConfigT2, oos_years: float,
) -> dict:
    """Console report + JSON save. Returns summary dict."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    sep = "=" * 70
    rule = "-" * 70
    be_wr = 1.0 / (1 + ecfg.r_target)

    print(f"\n{sep}")
    print("  EXECUTION COST ANALYSIS -- T2 / T3 (D30)")
    print(sep)
    print(f"  Position size: {ecfg.position_btc} BTC  |  "
          f"Impact k: {ecfg.impact_k}")
    print(f"  Regime mult: {ecfg.regime_impact_mult}x above "
          f"ATR ratio {ecfg.atr_ratio_high}")
    print(f"  Hold: {ecfg.hold_bars} bars "
          f"({ecfg.hold_bars * 5 // 60}h)  |  "
          f"Cooldown: {ecfg.cooldown_bars} bars "
          f"({ecfg.cooldown_bars * 5 // 60}h)")
    print(f"  OOS period: {oos_years:.2f} years")

    # -- Cost breakdown (T2 as reference, same per-trade costs apply to T3) --
    print(f"\n{rule}")
    print("  EXECUTION COST BREAKDOWN -- T2 (per trade, in R)")
    print(rule)
    header = f"  {'Component':<20}  {'Mean':>8}  {'Median':>8}  {'Std':>8}"
    divider = f"  {'-' * 20}  {'-' * 8}  {'-' * 8}  {'-' * 8}"
    print(header)
    print(divider)
    for label, m, med, s in [
        ("Latency",
         t2["mean_latency_r"], t2["median_latency_r"], t2["std_latency_r"]),
        ("Market Impact",
         t2["mean_impact_r"], t2["median_impact_r"], t2["std_impact_r"]),
        ("Funding",
         t2["mean_funding_r"], t2["median_funding_r"], t2["std_funding_r"]),
    ]:
        print(f"  {label:<20}  {m:>+8.4f}  {med:>+8.4f}  {s:>8.4f}")
    print(divider)
    print(f"  {'Total Cost':<20}  {t2['mean_total_cost_r']:>+8.4f}")

    # -- Annual cost impact --
    print(f"\n{rule}")
    print("  ANNUAL COST IMPACT")
    print(rule)
    for name, res in [("T2 (ML>=0.60)", t2), ("T3 (CB+ML>=0.50)", t3)]:
        tpy = res["trades_per_year"]
        total = res["annual_total_drag"]
        print(f"\n  {name}  ({tpy:.0f} trades/yr)")
        if abs(total) > 1e-6:
            lat_pct = res["annual_latency_drag"] / total * 100
            imp_pct = res["annual_impact_drag"] / total * 100
            fund_pct = res["annual_funding_drag"] / total * 100
        else:
            lat_pct = imp_pct = fund_pct = 0.0
        print(f"    Latency:   {res['annual_latency_drag']:>+8.1f} R/yr"
              f"  ({lat_pct:>4.1f}%)")
        print(f"    Impact:    {res['annual_impact_drag']:>+8.1f} R/yr"
              f"  ({imp_pct:>4.1f}%)")
        print(f"    Funding:   {res['annual_funding_drag']:>+8.1f} R/yr"
              f"  ({fund_pct:>4.1f}%)")
        print(f"    Total:     {res['annual_total_drag']:>+8.1f} R/yr")

    # -- T2 vs T3 side-by-side --
    print(f"\n{rule}")
    print("  T2 vs T3 COMPARISON")
    print(rule)
    col_h = (f"  {'Metric':<25}  {'T2 (ML>=0.60)':>15}  "
             f"{'T3 (CB+ML>=0.50)':>17}")
    col_d = (f"  {'-' * 25}  {'-' * 15}  {'-' * 17}")
    print(col_h)
    print(col_d)

    rows = [
        ("Trades", f"{t2['n']}", f"{t3['n']}"),
        ("Trades/yr", f"{t2['trades_per_year']:.0f}",
         f"{t3['trades_per_year']:.0f}"),
        ("WR (theoretical)", f"{t2['wr_theoretical']:.2%}",
         f"{t3['wr_theoretical']:.2%}"),
        ("WR (adjusted)", f"{t2['wr_adjusted']:.2%}",
         f"{t3['wr_adjusted']:.2%}"),
        ("Wins flipped (MFE)", f"{t2['n_flipped']}",
         f"{t3['n_flipped']}"),
        ("Mean R (theoretical)", f"{t2['mean_r_theoretical']:+.4f}",
         f"{t3['mean_r_theoretical']:+.4f}"),
        ("Mean R (adjusted)", f"{t2['mean_r_adjusted']:+.4f}",
         f"{t3['mean_r_adjusted']:+.4f}"),
        ("Profit Factor", f"{t2['pf']:.2f}",
         f"{t3['pf']:.2f}"),
        ("Per-trade cost (R)", f"{t2['mean_total_cost_r']:+.4f}",
         f"{t3['mean_total_cost_r']:+.4f}"),
        ("Annual cost (R/yr)", f"{t2['annual_total_drag']:+.1f}",
         f"{t3['annual_total_drag']:+.1f}"),
        ("Per-trade Sharpe", f"{t2['per_trade_sharpe']:.4f}",
         f"{t3['per_trade_sharpe']:.4f}"),
        ("Annualized Sharpe", f"{t2['annualized_sharpe']:.4f}",
         f"{t3['annualized_sharpe']:.4f}"),
        ("Risk per trade", f"{t2['risk_pct']:.1%}",
         f"{t3['risk_pct']:.1%}"),
        ("Max DD", f"{t2['max_dd_pct']:.1f}%",
         f"{t3['max_dd_pct']:.1f}%"),
        ("Final equity", f"${t2['equity_final']:,.0f}",
         f"${t3['equity_final']:,.0f}"),
    ]
    for label, v2, v3 in rows:
        print(f"  {label:<25}  {v2:>15}  {v3:>17}")

    # -- GO/NO-GO --
    print(f"\n{sep}")
    print("  GO / NO-GO GATE")
    print(sep)

    all_pass = True
    for name, res in [("T2", t2), ("T3", t3)]:
        ev_pass = bool(res["mean_r_adjusted"] > 0)
        wr_pass = bool(res["wr_adjusted"] > be_wr)
        sharpe_pass = bool(res["annualized_sharpe"] > 0.5)

        checks = [
            (f"{name} Adjusted EV > 0", ev_pass,
             f"{res['mean_r_adjusted']:+.4f} R"),
            (f"{name} Adjusted WR > BE ({be_wr:.1%})", wr_pass,
             f"{res['wr_adjusted']:.2%}"),
            (f"{name} Ann Sharpe > 0.5", sharpe_pass,
             f"{res['annualized_sharpe']:.4f}"),
        ]
        for label, passed, val in checks:
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}]  {label:<35}  {val}")
            if not passed:
                all_pass = False

    if all_pass:
        verdict = "GO -- edge survives execution costs for both T2 and T3"
    else:
        verdict = "REVIEW -- some gates failed, see breakdown above"
    print(f"\n  >>> {verdict}")
    print(sep)

    # -- Save chart --
    chart_path = save_chart(t2, t3)
    if chart_path:
        print(f"\n  Saved: {chart_path}")
    else:
        print("\n  (plotly not installed -- skipping chart)")

    # -- Save JSON --
    def _clean(d: dict) -> dict:
        """Strip internal numpy arrays before serialisation."""
        return {k: v for k, v in d.items() if not k.startswith("_")}

    def _round_floats(obj, dp=4):
        if isinstance(obj, dict):
            return {k: _round_floats(v, dp) for k, v in obj.items()}
        if isinstance(obj, float) and not (obj != obj):  # not NaN
            return round(obj, dp)
        return obj

    summary = {
        "exec_config": {
            "impact_k": ecfg.impact_k,
            "position_btc": ecfg.position_btc,
            "adv_window": ecfg.adv_window,
            "regime_impact_mult": ecfg.regime_impact_mult,
            "atr_ratio_high": ecfg.atr_ratio_high,
            "hold_bars": ecfg.hold_bars,
            "cooldown_bars": ecfg.cooldown_bars,
            "risk_pct_t2": ecfg.risk_pct_t2,
            "risk_pct_t3": ecfg.risk_pct_t3,
        },
        "oos_years": oos_years,
        "t2": _clean(t2),
        "t3": _clean(t3),
        "cost_breakdown_t2": {
            "latency": {
                "mean": t2["mean_latency_r"],
                "median": t2["median_latency_r"],
                "std": t2["std_latency_r"],
            },
            "impact": {
                "mean": t2["mean_impact_r"],
                "median": t2["median_impact_r"],
                "std": t2["std_impact_r"],
            },
            "funding": {
                "mean": t2["mean_funding_r"],
                "median": t2["median_funding_r"],
                "std": t2["std_funding_r"],
            },
            "total_mean": t2["mean_total_cost_r"],
        },
        "annual_cost": {
            "t2": {
                "trades_per_year": t2["trades_per_year"],
                "latency_r_yr": t2["annual_latency_drag"],
                "impact_r_yr": t2["annual_impact_drag"],
                "funding_r_yr": t2["annual_funding_drag"],
                "total_r_yr": t2["annual_total_drag"],
            },
            "t3": {
                "trades_per_year": t3["trades_per_year"],
                "latency_r_yr": t3["annual_latency_drag"],
                "impact_r_yr": t3["annual_impact_drag"],
                "funding_r_yr": t3["annual_funding_drag"],
                "total_r_yr": t3["annual_total_drag"],
            },
        },
        "go_no_go": {
            "t2_ev_pass": bool(t2["mean_r_adjusted"] > 0),
            "t2_wr_pass": bool(t2["wr_adjusted"] > be_wr),
            "t2_sharpe_pass": bool(t2["annualized_sharpe"] > 0.5),
            "t3_ev_pass": bool(t3["mean_r_adjusted"] > 0),
            "t3_wr_pass": bool(t3["wr_adjusted"] > be_wr),
            "t3_sharpe_pass": bool(t3["annualized_sharpe"] > 0.5),
            "all_pass": all_pass,
            "verdict": verdict,
        },
    }

    summary = _round_floats(summary)

    json_path = RESULTS_DIR / "execution_model_t2.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved: {json_path}")

    return summary


# -- main ---------------------------------------------------------------------
def main() -> None:
    cfg = Config()
    ml_cfg = MLConfig()
    ecfg = ExecConfigT2()

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

    # 2. Build signal masks + simulate with cooldown
    print(f"\n{'=' * 70}")
    print("  STEP 2: Build Masks + Simulate")
    print("=" * 70)

    label_arr = df[ml_cfg.label_col].values
    cb_mask = build_config_b_mask(df, cfg)

    t2_mask = (oos_probs >= 0.60) & oos_valid
    t3_mask = cb_mask & (oos_probs >= 0.50) & oos_valid

    t2_indices = simulate(t2_mask, label_arr, ecfg.cooldown_bars)
    t3_indices = simulate(t3_mask, label_arr, ecfg.cooldown_bars)

    print(f"  T2 (ML>=0.60): {len(t2_indices)} trades "
          f"({len(t2_indices)/oos_years:.0f}/yr)")
    print(f"  T3 (CB+ML>=0.50): {len(t3_indices)} trades "
          f"({len(t3_indices)/oos_years:.0f}/yr)")

    # 3. Build signal tables
    print(f"\n{'=' * 70}")
    print("  STEP 3: Build Signal Tables + Compute Costs")
    print("=" * 70)

    sig_t2 = build_signal_table_ml(df, t2_indices, ecfg, ml_cfg.label_col)
    sig_t3 = build_signal_table_ml(df, t3_indices, ecfg, ml_cfg.label_col)

    print(f"  T2 signal table: {len(sig_t2)} trades")
    print(f"  T3 signal table: {len(sig_t3)} trades")

    # 4. Compute execution costs
    t2_results = compute_execution_costs(
        sig_t2, ecfg, "T2 (ML>=0.60)", oos_years, ecfg.risk_pct_t2,
    )
    t3_results = compute_execution_costs(
        sig_t3, ecfg, "T3 (CB+ML>=0.50)", oos_years, ecfg.risk_pct_t3,
    )

    # 5. Report
    print_report(t2_results, t3_results, ecfg, oos_years)


if __name__ == "__main__":
    main()
