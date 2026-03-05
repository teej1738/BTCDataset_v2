"""
execution_model.py -- Execution Reality Layer (Step 4)
======================================================
Applies realistic execution costs to the MTF long-only strategy
(176 signals, 47.2% WR, mean R +0.365, Sharpe 1.32) to determine
what the theoretical edge becomes under real trading conditions.

Four execution cost components:
  1. Latency (1-bar delay): entry at open[t+1] instead of close[t]
  2. Market impact (square-root model): Kyle's lambda, regime-conditional
  3. Funding (position-level): sum of fund_rate_period over hold window
  4. Fill probability (FVG CE limit): limit entry at FVG consequent encroachment

Two scenarios:
  A. Market Entry -- all signals, latency + impact + funding costs
  B. Limit at FVG CE -- probabilistic fill, better entry, fewer signals

Output:
  - Console report with GO/NO-GO
  - results/execution_model.json
  - results/execution_model.html
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_backtest_v2 import Config, load_labeled
from mtf_signals import TF_CONFIGS, config_b_filters

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"


# -- config ------------------------------------------------------------------
@dataclass
class ExecConfig:
    fill_window: int = 48           # full hold horizon for CE limit fill
    impact_k: float = 0.10          # Kyle's lambda for BTC futures
    position_btc: float = 0.003     # ~$200 at BTC=$67k (2% risk on $10k)
    adv_window: int = 288           # 1 day of 5m bars for ADV
    regime_impact_mult: float = 1.5 # high-vol regime impact multiplier
    atr_ratio_high: float = 1.2     # threshold for regime multiplier
    hold_bars: int = 48             # hold window matching label horizon
    r_target: int = 2               # R-multiple target


# -- FVG CE column mapping per TF --------------------------------------------
TF_FVG_COLS = {
    "H4": ("h4_ict_fvg_bull_nearest_top", "h4_ict_fvg_bull_nearest_bot"),
    "H1": ("h1_ict_fvg_bull_nearest_top", "h1_ict_fvg_bull_nearest_bot"),
    "M15": ("m15_ict_fvg_bull_nearest_top", "m15_ict_fvg_bull_nearest_bot"),
}


# -- build signal table ------------------------------------------------------
def build_signal_table(
    df: pd.DataFrame, cfg: Config, ecfg: ExecConfig
) -> pd.DataFrame:
    """
    Build per-signal table with all fields needed for execution cost calc.
    Union of H4+H1+M15 long signals (deduplicated at bar level).
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
    low_arr = df["low"].values
    atr_arr = df["ict_atr_14"].values
    atr_ratio_arr = df["ict_atr_ratio"].values
    vol_arr = df["ict_realized_vol_20"].values
    label_arr = df[cfg.long_label].values
    mfe_arr = df["label_max_up_pct_48c"].values
    ts_arr = df["bar_start_ts_utc"].values

    # FVG level arrays per TF (preload once)
    fvg_arrays = {}
    for tf_name, (top_col, bot_col) in TF_FVG_COLS.items():
        fvg_arrays[tf_name] = (df[top_col].values, df[bot_col].values)

    # Build union of long masks, tracking TF source
    combined_long = pd.Series(False, index=df.index)
    tf_source = pd.Series("", index=df.index, dtype="object")

    for tf in TF_CONFIGS:
        long_mask, _ = config_b_filters(df, tf, cfg)
        new_bars = long_mask & ~combined_long
        tf_source[new_bars] = tf.name
        combined_long = combined_long | long_mask

    signal_idx = np.where(combined_long.values)[0]

    records = []
    for pos in signal_idx:
        # Skip if not enough forward data for hold + 1 bar
        if pos + ecfg.hold_bars >= n_rows:
            continue

        tf_name = tf_source.iat[pos]
        cl = close_arr[pos]
        atr = atr_arr[pos]

        if np.isnan(atr) or atr <= 0 or np.isnan(cl) or cl <= 0:
            continue

        # FVG CE from signal's TF
        top_arr, bot_arr = fvg_arrays[tf_name]
        fvg_top = top_arr[pos]
        fvg_bot = bot_arr[pos]
        ce = (fvg_top + fvg_bot) / 2.0 if (
            not np.isnan(fvg_top) and not np.isnan(fvg_bot)
        ) else np.nan

        # Funding sum over hold window: bars t+1 through t+hold_bars
        fund_end = min(pos + 1 + ecfg.hold_bars, n_rows)
        funding_sum = float(fund_cum[fund_end] - fund_cum[pos + 1])

        # Fill window: min low from t+1 to t+1+fill_window
        fill_end = min(pos + 1 + ecfg.fill_window, n_rows)
        fill_min_low = float(np.nanmin(low_arr[pos + 1 : fill_end]))

        sigma = vol_arr[pos]

        records.append({
            "bar_idx": int(pos),
            "timestamp": ts_arr[pos],
            "tf": tf_name,
            "close": float(cl),
            "atr": float(atr),
            "atr_ratio": float(atr_ratio_arr[pos]),
            "sigma": float(sigma) if not np.isnan(sigma) else 0.0,
            "volume": float(df["volume_base"].iat[pos]),
            "next_open": float(open_arr[pos + 1]),
            "next_low": float(low_arr[pos + 1]),
            "win": bool(label_arr[pos] == 1.0),
            "mfe_pct": float(mfe_arr[pos]) if not np.isnan(mfe_arr[pos]) else 0.0,
            "ce": float(ce) if not np.isnan(ce) else np.nan,
            "funding_sum": funding_sum,
            "adv": float(adv[pos]),
            "fill_min_low": fill_min_low,
        })

    return pd.DataFrame(records)


# -- cost components ----------------------------------------------------------
def compute_latency(sig: pd.DataFrame) -> np.ndarray:
    """Latency cost in R: (open[t+1] - close[t]) / ATR.  Positive = adverse."""
    return (sig["next_open"].values - sig["close"].values) / sig["atr"].values


def compute_impact(sig: pd.DataFrame, ecfg: ExecConfig) -> np.ndarray:
    """
    Square-root market impact in R.
    raw_pct = k * sigma * sqrt(Q / ADV)
    Regime: x1.5 when atr_ratio > threshold.
    impact_r = raw_pct / (atr / close)
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
    Funding cost in R.  Longs pay positive funding.
    funding_r = sum(rate over hold) * close / atr
    Positive = cost, negative = benefit.
    """
    return sig["funding_sum"].values * sig["close"].values / sig["atr"].values


def compute_ce_fill(sig: pd.DataFrame) -> pd.DataFrame:
    """
    FVG CE limit fill model.
    Adds: ce_valid, ce_filled, ce_improvement_r.
    """
    ce = sig["ce"].values
    close = sig["close"].values
    atr = sig["atr"].values
    fill_min_low = sig["fill_min_low"].values

    ce_valid = ~np.isnan(ce) & (ce < close)
    ce_filled = ce_valid & (fill_min_low <= ce)
    improvement_r = np.where(ce_filled, (close - ce) / atr, 0.0)

    out = sig.copy()
    out["ce_valid"] = ce_valid
    out["ce_filled"] = ce_filled
    out["ce_improvement_r"] = improvement_r
    return out


# -- win/loss flip detection --------------------------------------------------
def flip_detection(
    sig: pd.DataFrame,
    latency_r: np.ndarray,
    impact_r: np.ndarray,
    ecfg: ExecConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Determine post-execution win/loss for each signal.

    Market entry (unidirectional, win->loss only):
      Flip win to loss if MFE_price < 2*ATR + latency_price + impact_price.
      Latency is tiny (~0.001R), so loss->win flips are negligible.

    CE entry (bidirectional):
      MFE from CE = MFE from close + (close - CE).
      Win from CE if MFE_from_CE >= 2*ATR + impact_price.
      This correctly handles both win->loss and loss->win flips
      (a label loss from close can be a win from a much better CE entry).

    Returns (market_win, ce_win) boolean arrays.
    """
    atr = sig["atr"].values
    win = sig["win"].values.copy()
    mfe_price = sig["close"].values * sig["mfe_pct"].values / 100.0

    latency_price = latency_r * atr
    impact_price = impact_r * atr

    # Market entry: win->loss only (latency too small for loss->win)
    market_required = ecfg.r_target * atr + latency_price + impact_price
    market_win = win.copy()
    market_win[win & (mfe_price < market_required)] = False

    # CE entry: bidirectional using MFE from CE entry point
    ce_improvement_price = sig["ce_improvement_r"].values * atr
    ce_mfe = mfe_price + ce_improvement_price
    ce_target = ecfg.r_target * atr + impact_price
    ce_win = ce_mfe >= ce_target

    return market_win, ce_win


# -- scenario computation ----------------------------------------------------
def compute_scenarios(
    sig: pd.DataFrame, ecfg: ExecConfig
) -> dict:
    """Compute Scenario A (Market Entry) and Scenario B (CE Limit)."""

    latency_r = compute_latency(sig)
    impact_r = compute_impact(sig, ecfg)
    funding_r = compute_funding(sig)
    sig = compute_ce_fill(sig)

    market_win, ce_win = flip_detection(sig, latency_r, impact_r, ecfg)

    # ---- Scenario A: Market Entry ----
    #   adjusted_r = theoretical_r - (latency_r + impact_r + funding_r)
    #   theoretical_r uses post-flip outcome (+2R win, -1R loss)
    market_theo = np.where(market_win, float(ecfg.r_target), -1.0)
    market_adj = market_theo - (latency_r + impact_r + funding_r)

    n_total = len(sig)
    n_wins_th = int(sig["win"].sum())
    # Derive WR from adjusted R sign (consistent across scenarios)
    n_wins_mkt = int(np.sum(market_adj > 0))

    mkt_win_vals = market_adj[market_adj > 0]
    mkt_loss_vals = market_adj[market_adj <= 0]
    gross_win = float(np.sum(mkt_win_vals)) if len(mkt_win_vals) else 0.0
    gross_loss = float(np.sum(np.abs(mkt_loss_vals))) if len(mkt_loss_vals) else 0.0

    scenario_a = {
        "n": n_total,
        "n_wins_theoretical": n_wins_th,
        "n_wins_adjusted": n_wins_mkt,
        "n_flipped": abs(n_wins_th - n_wins_mkt),
        "wr_theoretical": n_wins_th / n_total if n_total else 0.0,
        "wr_adjusted": n_wins_mkt / n_total if n_total else 0.0,
        "mean_latency_r": float(np.mean(latency_r)),
        "median_latency_r": float(np.median(latency_r)),
        "std_latency_r": float(np.std(latency_r)),
        "mean_impact_r": float(np.mean(impact_r)),
        "median_impact_r": float(np.median(impact_r)),
        "std_impact_r": float(np.std(impact_r)),
        "mean_funding_r": float(np.mean(funding_r)),
        "median_funding_r": float(np.median(funding_r)),
        "std_funding_r": float(np.std(funding_r)),
        "mean_total_cost_r": float(np.mean(latency_r + impact_r + funding_r)),
        "mean_r_theoretical": float(np.mean(
            np.where(sig["win"].values, ecfg.r_target, -1.0)
        )),
        "mean_r_adjusted": float(np.mean(market_adj)),
        "pf": gross_win / gross_loss if gross_loss > 0 else float("inf"),
    }

    std_adj = float(np.std(market_adj, ddof=1)) if n_total > 1 else 0.0
    scenario_a["per_trade_sharpe"] = (
        float(np.mean(market_adj)) / std_adj if std_adj > 0 else 0.0
    )
    trades_per_year = n_total / 6.0
    scenario_a["annualized_sharpe"] = (
        scenario_a["per_trade_sharpe"] * np.sqrt(trades_per_year)
    )

    # Keep arrays for chart (not serialised to JSON)
    scenario_a["_latency_r"] = latency_r
    scenario_a["_impact_r"] = impact_r
    scenario_a["_funding_r"] = funding_r
    scenario_a["_adjusted_returns"] = market_adj

    # ---- Scenario B: CE Limit Entry ----
    #   CE < close & filled   -> CE entry with R from MFE-based win/loss
    #   CE >= close or NaN    -> market fallback (same as Scenario A)
    #   CE < close & unfilled -> dropped
    ce_valid = sig["ce_valid"].values
    ce_filled = sig["ce_filled"].values
    ce_improvement_r = sig["ce_improvement_r"].values

    ce_entry = ce_valid & ce_filled
    market_fallback = ~ce_valid
    dropped = ce_valid & ~ce_filled
    included_b = market_fallback | ce_entry

    # Build per-trade R for Scenario B
    # CE entries: R = +2 or -1 from CE entry (bidirectional flip), minus costs
    # Market fallback: same formula as Scenario A
    ce_theo = np.where(ce_win, float(ecfg.r_target), -1.0)
    mkt_theo_b = np.where(market_win, float(ecfg.r_target), -1.0)

    b_adj = np.where(
        ce_entry,
        ce_theo - impact_r - funding_r,
        mkt_theo_b - (latency_r + impact_r + funding_r),
    )

    b_adj_inc = b_adj[included_b]
    n_b = int(included_b.sum())
    n_ce = int(ce_entry.sum())
    n_fb = int(market_fallback.sum())
    n_drop = int(dropped.sum())
    ce_valid_count = int(ce_valid.sum())

    # Derive WR from adjusted R sign
    n_wins_b = int(np.sum(b_adj_inc > 0))

    b_win_vals = b_adj_inc[b_adj_inc > 0]
    b_loss_vals = b_adj_inc[b_adj_inc <= 0]
    b_gw = float(np.sum(b_win_vals)) if len(b_win_vals) else 0.0
    b_gl = float(np.sum(np.abs(b_loss_vals))) if len(b_loss_vals) else 0.0

    # CE-only sub-stats for reporting
    ce_adj = b_adj[ce_entry]
    n_ce_wins = int(np.sum(ce_adj > 0)) if n_ce > 0 else 0

    scenario_b = {
        "n": n_b,
        "n_ce_entries": n_ce,
        "n_ce_wins": n_ce_wins,
        "n_market_fallback": n_fb,
        "n_dropped": n_drop,
        "ce_valid_count": ce_valid_count,
        "fill_rate": n_ce / ce_valid_count if ce_valid_count > 0 else 0.0,
        "mean_ce_improvement_r": (
            float(np.mean(ce_improvement_r[ce_entry])) if n_ce > 0 else 0.0
        ),
        "n_wins_adjusted": n_wins_b,
        "wr_adjusted": n_wins_b / n_b if n_b > 0 else 0.0,
        "mean_r_adjusted": float(np.mean(b_adj_inc)) if n_b > 0 else 0.0,
        "pf": b_gw / b_gl if b_gl > 0 else float("inf"),
    }
    b_std = float(np.std(b_adj_inc, ddof=1)) if n_b > 1 else 0.0
    scenario_b["per_trade_sharpe"] = (
        float(np.mean(b_adj_inc)) / b_std if b_std > 0 else 0.0
    )
    b_per_year = n_b / 6.0
    scenario_b["annualized_sharpe"] = (
        scenario_b["per_trade_sharpe"] * np.sqrt(b_per_year)
    )

    # ---- Per-TF breakdown (Scenario A) ----
    per_tf = {}
    for tf_name in ["H4", "H1", "M15"]:
        tf_mask = sig["tf"].values == tf_name
        tf_n = int(tf_mask.sum())
        if tf_n == 0:
            per_tf[tf_name] = {"n": 0}
            continue

        tf_win_th = int(sig["win"].values[tf_mask].sum())
        tf_adj = market_adj[tf_mask]
        tf_win_adj = int(np.sum(tf_adj > 0))
        tf_theo = np.where(sig["win"].values[tf_mask], ecfg.r_target, -1.0)

        per_tf[tf_name] = {
            "n": tf_n,
            "wr_theoretical": round(tf_win_th / tf_n, 4),
            "wr_adjusted": round(tf_win_adj / tf_n, 4),
            "n_flipped": abs(tf_win_th - tf_win_adj),
            "mean_r_theoretical": round(float(np.mean(tf_theo)), 4),
            "mean_r_adjusted": round(float(np.mean(tf_adj)), 4),
            "mean_latency_r": round(float(np.mean(latency_r[tf_mask])), 4),
            "mean_impact_r": round(float(np.mean(impact_r[tf_mask])), 4),
            "mean_funding_r": round(float(np.mean(funding_r[tf_mask])), 4),
        }

    return {
        "scenario_a": scenario_a,
        "scenario_b": scenario_b,
        "per_tf": per_tf,
    }


# -- chart --------------------------------------------------------------------
def save_chart(results: dict) -> Path | None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return None

    sa = results["scenario_a"]
    sb = results["scenario_b"]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "Execution Cost Distribution (Market Entry, R)",
            "Scenario Comparison: Mean R per Trade",
        ],
        horizontal_spacing=0.12,
    )

    # Left: cost distributions (overlaid histograms)
    fig.add_trace(go.Histogram(
        x=sa["_latency_r"], name="Latency", opacity=0.6,
        marker_color="#4a86c8", nbinsx=30,
    ), row=1, col=1)
    fig.add_trace(go.Histogram(
        x=sa["_impact_r"], name="Impact", opacity=0.6,
        marker_color="#5cb85c", nbinsx=30,
    ), row=1, col=1)
    fig.add_trace(go.Histogram(
        x=sa["_funding_r"], name="Funding", opacity=0.6,
        marker_color="#f0ad4e", nbinsx=30,
    ), row=1, col=1)
    fig.update_layout(barmode="overlay")

    # Right: scenario comparison bars
    scenarios = ["Theoretical", "A: Market", "B: CE Limit"]
    evs = [
        sa["mean_r_theoretical"],
        sa["mean_r_adjusted"],
        sb["mean_r_adjusted"],
    ]
    wrs = [
        sa["wr_theoretical"] * 100,
        sa["wr_adjusted"] * 100,
        sb["wr_adjusted"] * 100,
    ]
    bar_colors = ["#4a86c8", "#d9534f", "#5cb85c"]
    fig.add_trace(go.Bar(
        x=scenarios, y=evs, marker_color=bar_colors,
        text=[f"EV={e:+.3f}R\nWR={w:.1f}%" for e, w in zip(evs, wrs)],
        textposition="outside", name="Mean R", showlegend=False,
    ), row=1, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color="yellow", row=1, col=2)

    fig.update_layout(
        template="plotly_dark",
        title="Execution Reality Layer -- Step 4",
        height=500,
    )
    fig.update_xaxes(title_text="Cost (R)", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="Scenario", row=1, col=2)
    fig.update_yaxes(title_text="Mean R per Trade", row=1, col=2)

    path = RESULTS_DIR / "execution_model.html"
    fig.write_html(str(path))
    return path


# -- report -------------------------------------------------------------------
def print_report(results: dict, ecfg: ExecConfig) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    sa = results["scenario_a"]
    sb = results["scenario_b"]
    per_tf = results["per_tf"]

    sep = "=" * 70
    rule = "-" * 70
    be_wr = 1.0 / (1 + ecfg.r_target)

    print(f"\n{sep}")
    print("  EXECUTION REALITY LAYER (Step 4)")
    print(sep)
    print(f"  Position size: {ecfg.position_btc} BTC  |  "
          f"Impact k: {ecfg.impact_k}")
    print(f"  Regime mult: {ecfg.regime_impact_mult}x above "
          f"ATR ratio {ecfg.atr_ratio_high}")
    print(f"  Fill window: {ecfg.fill_window} bars "
          f"({ecfg.fill_window * 5}min)  |  "
          f"Hold: {ecfg.hold_bars} bars ({ecfg.hold_bars * 5 // 60}h)")

    # -- Cost breakdown --
    print(f"\n{rule}")
    print("  EXECUTION COST BREAKDOWN (per trade, in R)")
    print(rule)
    header = f"  {'Component':<20}  {'Mean':>8}  {'Median':>8}  {'Std':>8}"
    divider = f"  {'-' * 20}  {'-' * 8}  {'-' * 8}  {'-' * 8}"
    print(header)
    print(divider)
    for label, m, med, s in [
        ("Latency", sa["mean_latency_r"], sa["median_latency_r"],
         sa["std_latency_r"]),
        ("Market Impact", sa["mean_impact_r"], sa["median_impact_r"],
         sa["std_impact_r"]),
        ("Funding", sa["mean_funding_r"], sa["median_funding_r"],
         sa["std_funding_r"]),
    ]:
        print(f"  {label:<20}  {m:>+8.4f}  {med:>+8.4f}  {s:>8.4f}")
    print(divider)
    print(f"  {'Total Cost':<20}  {sa['mean_total_cost_r']:>+8.4f}")

    # -- Scenario comparison --
    print(f"\n{rule}")
    print("  SCENARIO COMPARISON")
    print(rule)
    col_h = (f"  {'Metric':<25}  {'Theoretical':>12}  "
             f"{'A: Market':>12}  {'B: CE Limit':>12}")
    col_d = (f"  {'-' * 25}  {'-' * 12}  {'-' * 12}  {'-' * 12}")
    print(col_h)
    print(col_d)

    print(f"  {'Signals':<25}  {sa['n']:>12}  "
          f"{sa['n']:>12}  {sb['n']:>12}")
    print(f"  {'Win Rate':<25}  {sa['wr_theoretical']:>11.2%}  "
          f" {sa['wr_adjusted']:>11.2%}   {sb['wr_adjusted']:>11.2%}")
    print(f"  {'Wins Flipped':<25}  {'--':>12}  "
          f"{sa['n_flipped']:>12}  {'--':>12}")
    print(f"  {'Mean R':<25}  {sa['mean_r_theoretical']:>+12.4f}  "
          f"{sa['mean_r_adjusted']:>+12.4f}  {sb['mean_r_adjusted']:>+12.4f}")
    print(f"  {'Profit Factor':<25}  {'--':>12}  "
          f"{sa['pf']:>12.2f}  {sb['pf']:>12.2f}")
    print(f"  {'Per-Trade Sharpe':<25}  {'--':>12}  "
          f"{sa['per_trade_sharpe']:>12.4f}  {sb['per_trade_sharpe']:>12.4f}")
    print(f"  {'Annualized Sharpe':<25}  {'--':>12}  "
          f"{sa['annualized_sharpe']:>12.4f}  {sb['annualized_sharpe']:>12.4f}")

    # CE fill details
    print(f"\n  CE Limit Details:")
    print(f"    CE valid (< close):   {sb['ce_valid_count']}")
    print(f"    CE filled:            {sb['n_ce_entries']}")
    print(f"    Fill rate:            {sb['fill_rate']:.1%}")
    print(f"    CE wins:              {sb['n_ce_wins']}")
    ce_wr = sb['n_ce_wins'] / sb['n_ce_entries'] if sb['n_ce_entries'] > 0 else 0
    print(f"    CE win rate:          {ce_wr:.1%}")
    print(f"    Market fallback:      {sb['n_market_fallback']}")
    print(f"    Dropped (unfilled):   {sb['n_dropped']}")
    if sb["n_ce_entries"] > 0:
        print(f"    Mean CE improvement:  {sb['mean_ce_improvement_r']:+.4f} R")

    # -- Per-TF breakdown --
    print(f"\n{rule}")
    print("  PER-TIMEFRAME BREAKDOWN (Scenario A: Market Entry)")
    print(rule)
    tf_h = (f"  {'TF':<5} {'N':>4}  {'WR Th':>6}  {'WR Ad':>6}  "
            f"{'Flip':>4}  {'EV Th':>7}  {'EV Ad':>7}  "
            f"{'Lat':>7}  {'Imp':>7}  {'Fund':>7}")
    tf_d = (f"  {'-----':<5} {'----':>4}  {'------':>6}  {'------':>6}  "
            f"{'----':>4}  {'-------':>7}  {'-------':>7}  "
            f"{'-------':>7}  {'-------':>7}  {'-------':>7}")
    print(tf_h)
    print(tf_d)
    for tf_name in ["H4", "H1", "M15"]:
        t = per_tf[tf_name]
        if t["n"] == 0:
            continue
        print(
            f"  {tf_name:<5} {t['n']:>4}  "
            f"{t['wr_theoretical']:>5.1%}  {t['wr_adjusted']:>5.1%}  "
            f"{t['n_flipped']:>4}  "
            f"{t['mean_r_theoretical']:>+7.3f}  {t['mean_r_adjusted']:>+7.3f}  "
            f"{t['mean_latency_r']:>+7.4f}  "
            f"{t['mean_impact_r']:>+7.4f}  "
            f"{t['mean_funding_r']:>+7.4f}"
        )

    # -- GO/NO-GO --
    print(f"\n{sep}")
    print("  GO / NO-GO GATE")
    print(sep)

    ev_pass = bool(sa["mean_r_adjusted"] > 0)
    wr_pass = bool(sa["wr_adjusted"] > be_wr)
    sharpe_pass = bool(sa["annualized_sharpe"] > 0.5)

    checks = [
        ("Adjusted EV > 0", ev_pass,
         f"{sa['mean_r_adjusted']:+.4f} R"),
        (f"Adjusted WR > BE ({be_wr:.1%})", wr_pass,
         f"{sa['wr_adjusted']:.2%}"),
        ("Annualized Sharpe > 0.5", sharpe_pass,
         f"{sa['annualized_sharpe']:.4f}"),
    ]

    all_pass = all(p for _, p, _ in checks)
    for label, passed, val in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}]  {label:<30}  {val}")

    if all_pass:
        verdict = "GO -- edge survives execution costs"
    else:
        verdict = "REVIEW -- edge degraded, see breakdown above"
    print(f"\n  >>> {verdict}")
    print(sep)

    # -- Save chart --
    chart_path = save_chart(results)
    if chart_path:
        print(f"\n  Saved: {chart_path}")
    else:
        print("\n  (plotly not installed -- skipping chart)")

    # -- Save JSON --
    def _clean(d: dict) -> dict:
        """Strip internal numpy arrays before serialisation."""
        return {k: v for k, v in d.items() if not k.startswith("_")}

    summary = {
        "exec_config": {
            "fill_window": ecfg.fill_window,
            "impact_k": ecfg.impact_k,
            "position_btc": ecfg.position_btc,
            "adv_window": ecfg.adv_window,
            "regime_impact_mult": ecfg.regime_impact_mult,
            "atr_ratio_high": ecfg.atr_ratio_high,
            "hold_bars": ecfg.hold_bars,
        },
        "cost_breakdown": {
            "latency": {
                "mean": round(sa["mean_latency_r"], 6),
                "median": round(sa["median_latency_r"], 6),
                "std": round(sa["std_latency_r"], 6),
            },
            "impact": {
                "mean": round(sa["mean_impact_r"], 6),
                "median": round(sa["median_impact_r"], 6),
                "std": round(sa["std_impact_r"], 6),
            },
            "funding": {
                "mean": round(sa["mean_funding_r"], 6),
                "median": round(sa["median_funding_r"], 6),
                "std": round(sa["std_funding_r"], 6),
            },
            "total_mean": round(sa["mean_total_cost_r"], 6),
        },
        "scenario_a": _clean(sa),
        "scenario_b": {k: v for k, v in sb.items()},
        "per_tf": per_tf,
        "go_no_go": {
            "ev_pass": ev_pass,
            "wr_pass": wr_pass,
            "sharpe_pass": sharpe_pass,
            "overall": all_pass,
            "verdict": verdict,
        },
    }

    # Round floats in nested dicts
    def _round_floats(obj, dp=4):
        if isinstance(obj, dict):
            return {k: _round_floats(v, dp) for k, v in obj.items()}
        if isinstance(obj, float) and not (obj != obj):  # not NaN
            return round(obj, dp)
        return obj

    summary = _round_floats(summary)

    json_path = RESULTS_DIR / "execution_model.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved: {json_path}")

    return summary


# -- main ---------------------------------------------------------------------
def main() -> None:
    cfg = Config()
    ecfg = ExecConfig()

    # 1. Load data
    df = load_labeled(cfg)

    # 2. Build signal table
    print("\n  Building signal table...")
    signals = build_signal_table(df, cfg, ecfg)
    print(f"  Signal table: {len(signals)} trades")
    for tf in ["H4", "H1", "M15"]:
        n_tf = int((signals["tf"] == tf).sum())
        print(f"    {tf}: {n_tf}")

    # 3. Compute scenarios
    print("\n  Computing execution costs...")
    results = compute_scenarios(signals, ecfg)

    # 4. Report
    print_report(results, ecfg)


if __name__ == "__main__":
    main()
