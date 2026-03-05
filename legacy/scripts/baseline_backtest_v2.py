"""
baseline_backtest_v2.py  –  Vectorized ICT-filter baseline backtest
====================================================================
Reads the pre-labeled dataset (648k x 447) and answers:
    "Do ICT binary filters provide edge above random entry?"

v3 filter stack (analytics-derived):
  - h4_fvg is the primary edge filter (+6.35pp solo)
  - Direction-session routing: longs -> London SB, shorts -> NY PM SB
  - ATR ratio band [0.8, 1.5] to filter extreme volatility
  - Exclude Monday/Tuesday (negative EV days)
  - h4_bear_trend required for shorts
  - d1_trend and h4_sweep available as optional toggles (off by default)

Two-layer evaluation:
  1. Signal-level analysis  (fully vectorized, no loop)
  2. Equity simulation      (bar-by-bar loop, cooldown + compounding)

Output:
  - Console report
  - results/trade_log_baseline_v3.csv
  - results/baseline_v3_summary.json
  - results/equity_baseline_v3.html  (plotly, optional)

See STRATEGY_LOG.md for column naming conventions.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ─── paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_PATH = PROJECT_DIR / "data" / "labeled" / "BTCUSDT_MASTER_labeled.parquet"
RESULTS_DIR = SCRIPT_DIR / "results"


# ─── CONFIG ─────────────────────────────────────────────────────────────────
@dataclass
class Config:
    # Primary label
    r_target: int = 2              # R-multiple target
    horizon: int = 48              # candle horizon
    # -> label columns: label_{long,short}_hit_2r_48c

    # Equity simulation
    initial_equity: float = 10_000.0
    risk_pct: float = 0.02          # 2% of equity per trade
    cooldown_bars: int = 48         # 4h cooldown (48 × 5m)
    cost_per_r: float = 0.05        # slippage + fees expressed in R

    # Optional ICT sub-filters (toggle on/off)
    require_discount: bool = False
    require_ote: bool = False
    require_ob: bool = False

    # Analytics-derived filters (v3)
    use_d1_trend: bool = False              # was required in v2, now optional (off)
    use_h4_sweep: bool = True               # h4 sweep gate (Config B optimal)
    require_h4_bear_trend: bool = True      # shorts require h4_ict_market_trend == -1
    atr_ratio_min: float = 0.8             # ATR ratio floor
    atr_ratio_max: float = 1.5             # ATR ratio ceiling
    exclude_mon_tue: bool = True            # drop Monday and Tuesday signals
    direction_session_routing: bool = True  # longs=London SB, shorts=NY PM SB

    # Date filter (D10)
    min_date: str = "2020-01-01"

    @property
    def long_label(self) -> str:
        return f"label_long_hit_{self.r_target}r_{self.horizon}c"

    @property
    def short_label(self) -> str:
        return f"label_short_hit_{self.r_target}r_{self.horizon}c"


CFG = Config()


# ─── helpers ────────────────────────────────────────────────────────────────
def _pf(wins_r: float, losses_r: float) -> float:
    """Profit factor: gross wins / gross losses."""
    return wins_r / losses_r if losses_r > 0 else float("inf")


# ─── load_labeled ───────────────────────────────────────────────────────────
def load_labeled(cfg: Config = CFG) -> pd.DataFrame:
    """Load the labeled parquet and apply the date filter (D10)."""
    print(f"Loading {DATA_PATH.name} …")
    df = pd.read_parquet(DATA_PATH)
    print(f"  raw shape: {df.shape[0]:,} rows × {df.shape[1]} cols")

    df = df[df["bar_start_ts_utc"] >= pd.Timestamp(cfg.min_date, tz="UTC")].copy()
    print(f"  after date filter (>= {cfg.min_date}): {df.shape[0]:,} rows")

    # Drop rows where primary labels are NaN (last N bars of dataset)
    before = len(df)
    df = df.dropna(subset=[cfg.long_label, cfg.short_label])
    dropped = before - len(df)
    if dropped:
        print(f"  dropped {dropped} rows with NaN labels")

    return df


# ─── old_baseline_filters (v2 reference) ───────────────────────────────────
def old_baseline_filters(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Reproduce the original v2 filter stack for side-by-side comparison.
    d1_trend + h4_sweep + h4_fvg + any Silver Bullet session.
    """
    in_session = (
        (df["sess_sb_london"] == 1)
        | (df["sess_sb_ny_am"] == 1)
        | (df["sess_sb_ny_pm"] == 1)
    )

    long_mask = (
        (df["d1_ict_market_trend"] == 1)
        & (df["h4_ict_bull_liq_sweep"] == 1)
        & (df["h4_ict_fvg_bull"] == 1)
        & in_session
    )
    short_mask = (
        (df["d1_ict_market_trend"] == -1)
        & (df["h4_ict_bear_liq_sweep"] == 1)
        & (df["h4_ict_fvg_bear"] == 1)
        & in_session
    )
    return long_mask, short_mask


# ─── ict_filters (v3 analytics-derived) ────────────────────────────────────
def ict_filters(df: pd.DataFrame, cfg: Config = CFG) -> tuple[pd.Series, pd.Series]:
    """
    Return boolean masks (long_signal, short_signal) using analytics-derived
    v3 filter stack.

    Long signals:
      REQUIRED: h4_ict_fvg_bull == 1
      REQUIRED: sess_sb_london == 1  (when direction_session_routing)
      REQUIRED: ict_atr_ratio in [atr_ratio_min, atr_ratio_max]
      REQUIRED: ict_day_of_week not in {0, 1}  (when exclude_mon_tue)
      OPTIONAL: d1_ict_market_trend == 1  (when use_d1_trend)
      OPTIONAL: h4_ict_bull_liq_sweep == 1  (when use_h4_sweep)

    Short signals:
      REQUIRED: h4_ict_fvg_bear == 1
      REQUIRED: sess_sb_ny_pm == 1  (when direction_session_routing)
      REQUIRED: h4_ict_market_trend == -1  (when require_h4_bear_trend)
      REQUIRED: ict_atr_ratio in [atr_ratio_min, atr_ratio_max]
      REQUIRED: ict_day_of_week not in {0, 1}  (when exclude_mon_tue)
      OPTIONAL: d1_ict_market_trend == -1  (when use_d1_trend)
      OPTIONAL: h4_ict_bear_liq_sweep == 1  (when use_h4_sweep)
    """
    # --- shared filters ---
    atr_ok = (
        (df["ict_atr_ratio"] >= cfg.atr_ratio_min)
        & (df["ict_atr_ratio"] <= cfg.atr_ratio_max)
    )
    if cfg.exclude_mon_tue:
        day_ok = ~df["ict_day_of_week"].isin([0, 1])
    else:
        day_ok = pd.Series(True, index=df.index)

    # --- LONG ---
    long_mask = (df["h4_ict_fvg_bull"] == 1) & atr_ok & day_ok

    if cfg.direction_session_routing:
        long_mask = long_mask & (df["sess_sb_london"] == 1)
    else:
        # fallback: any SB session
        long_mask = long_mask & (
            (df["sess_sb_london"] == 1)
            | (df["sess_sb_ny_am"] == 1)
            | (df["sess_sb_ny_pm"] == 1)
        )

    if cfg.use_d1_trend:
        long_mask = long_mask & (df["d1_ict_market_trend"] == 1)
    if cfg.use_h4_sweep:
        long_mask = long_mask & (df["h4_ict_bull_liq_sweep"] == 1)

    # legacy sub-filters still available
    if cfg.require_discount:
        long_mask = long_mask & (df["h4_ict_discount"] == 1)
    if cfg.require_ote:
        long_mask = long_mask & (df["h4_ict_ote_zone"] == 1)
    if cfg.require_ob:
        long_mask = long_mask & (df["h4_ict_ob_bull"] == 1)

    # --- SHORT ---
    short_mask = (df["h4_ict_fvg_bear"] == 1) & atr_ok & day_ok

    if cfg.direction_session_routing:
        short_mask = short_mask & (df["sess_sb_ny_pm"] == 1)
    else:
        short_mask = short_mask & (
            (df["sess_sb_london"] == 1)
            | (df["sess_sb_ny_am"] == 1)
            | (df["sess_sb_ny_pm"] == 1)
        )

    if cfg.require_h4_bear_trend:
        short_mask = short_mask & (df["h4_ict_market_trend"] == -1)

    if cfg.use_d1_trend:
        short_mask = short_mask & (df["d1_ict_market_trend"] == -1)
    if cfg.use_h4_sweep:
        short_mask = short_mask & (df["h4_ict_bear_liq_sweep"] == 1)

    # legacy sub-filters
    if cfg.require_discount:
        short_mask = short_mask & (df["h4_ict_premium"] == 1)
    if cfg.require_ote:
        short_mask = short_mask & (df["h4_ict_ote_zone"] == 1)
    if cfg.require_ob:
        short_mask = short_mask & (df["h4_ict_ob_bear"] == 1)

    return long_mask, short_mask


# ─── signal_analysis ────────────────────────────────────────────────────────
def signal_analysis(
    df: pd.DataFrame, long_mask: pd.Series, short_mask: pd.Series, cfg: Config = CFG
) -> dict:
    """
    Vectorized signal-level analysis.
    For every signal bar, look up its pre-computed label -> aggregate metrics.
    Compare against unconditional (random-entry) baselines.
    """
    results: dict = {}

    # --- unconditional baselines ---
    unc_long_wr = df[cfg.long_label].mean()
    unc_short_wr = df[cfg.short_label].mean()
    results["unconditional_long_wr"] = unc_long_wr
    results["unconditional_short_wr"] = unc_short_wr

    # --- filtered signals ---
    long_signals = df.loc[long_mask]
    short_signals = df.loc[short_mask]

    for direction, signals, label_col in [
        ("long", long_signals, cfg.long_label),
        ("short", short_signals, cfg.short_label),
    ]:
        n = len(signals)
        if n == 0:
            results[direction] = {"n": 0, "win_rate": 0, "ev_r": 0, "pf": 0}
            continue

        wins = signals[label_col].sum()
        losses = n - wins
        wr = wins / n
        # EV in R: win = +R_target, loss = -1R, minus costs both ways
        ev_r = wr * (cfg.r_target - cfg.cost_per_r) - (1 - wr) * (1 + cfg.cost_per_r)
        gross_win_r = wins * cfg.r_target
        gross_loss_r = losses * 1.0
        pf = _pf(gross_win_r, gross_loss_r)

        results[direction] = {
            "n": int(n),
            "wins": int(wins),
            "losses": int(losses),
            "win_rate": round(wr, 4),
            "ev_r": round(ev_r, 4),
            "profit_factor": round(pf, 4),
        }

    # Combined
    total_n = results.get("long", {}).get("n", 0) + results.get("short", {}).get("n", 0)
    total_wins = results.get("long", {}).get("wins", 0) + results.get("short", {}).get("wins", 0)
    if total_n > 0:
        combined_wr = total_wins / total_n
        combined_ev = combined_wr * (cfg.r_target - cfg.cost_per_r) - (1 - combined_wr) * (1 + cfg.cost_per_r)
        combined_gross_win = total_wins * cfg.r_target
        combined_gross_loss = (total_n - total_wins) * 1.0
        results["combined"] = {
            "n": total_n,
            "wins": total_wins,
            "losses": total_n - total_wins,
            "win_rate": round(combined_wr, 4),
            "ev_r": round(combined_ev, 4),
            "profit_factor": round(_pf(combined_gross_win, combined_gross_loss), 4),
        }
    else:
        results["combined"] = {"n": 0, "win_rate": 0, "ev_r": 0, "pf": 0}

    # --- session breakdown ---
    session_breakdown = {}
    for sess_col, sess_name in [
        ("sess_sb_london", "London SB"),
        ("sess_sb_ny_am", "NY AM SB"),
        ("sess_sb_ny_pm", "NY PM SB"),
    ]:
        sess_long = df.loc[long_mask & (df[sess_col] == 1)]
        sess_short = df.loc[short_mask & (df[sess_col] == 1)]
        sess_n = len(sess_long) + len(sess_short)
        if sess_n > 0:
            sess_wins = sess_long[cfg.long_label].sum() + sess_short[cfg.short_label].sum()
            session_breakdown[sess_name] = {
                "n": int(sess_n),
                "win_rate": round(sess_wins / sess_n, 4),
            }
        else:
            session_breakdown[sess_name] = {"n": 0, "win_rate": 0}
    results["session_breakdown"] = session_breakdown

    return results


# ─── simulate_equity ────────────────────────────────────────────────────────
def simulate_equity(
    df: pd.DataFrame, long_mask: pd.Series, short_mask: pd.Series, cfg: Config = CFG
) -> tuple[pd.DataFrame, dict]:
    """
    Bar-by-bar equity simulation with cooldown + compounding.

    Walk through bars chronologically. When a signal fires and we're flat
    + past cooldown -> enter. Outcome from label: win = +R*risk, loss = -1R*risk.
    Position sizing: risk_pct of current equity.

    Returns (trade_log_df, equity_stats).
    """
    equity = cfg.initial_equity
    peak_equity = equity
    max_dd = 0.0
    bars_since_trade = cfg.cooldown_bars  # start ready to trade

    long_arr = long_mask.values
    short_arr = short_mask.values
    long_lbl = df[cfg.long_label].values
    short_lbl = df[cfg.short_label].values
    timestamps = df["bar_start_ts_utc"].values
    closes = df["close"].values

    # Session info for trade log
    sess_london = df["sess_sb_london"].values
    sess_ny_am = df["sess_sb_ny_am"].values
    sess_ny_pm = df["sess_sb_ny_pm"].values

    trades: list[dict] = []
    equity_curve: list[dict] = []
    n = len(df)

    for i in range(n):
        bars_since_trade += 1

        # Check for signal
        direction = None
        if bars_since_trade > cfg.cooldown_bars:
            if long_arr[i]:
                direction = "long"
            elif short_arr[i]:
                direction = "short"

        if direction is not None:
            # Determine outcome from label
            lbl_val = long_lbl[i] if direction == "long" else short_lbl[i]
            win = lbl_val == 1.0

            # Position sizing
            risk_amount = equity * cfg.risk_pct
            if win:
                pnl = risk_amount * (cfg.r_target - cfg.cost_per_r)
            else:
                pnl = -risk_amount * (1 + cfg.cost_per_r)

            equity += pnl
            peak_equity = max(peak_equity, equity)
            dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
            max_dd = max(max_dd, dd)
            bars_since_trade = 0

            # Session label
            sess = "Off"
            if sess_london[i] == 1:
                sess = "London SB"
            elif sess_ny_am[i] == 1:
                sess = "NY AM SB"
            elif sess_ny_pm[i] == 1:
                sess = "NY PM SB"

            trades.append({
                "bar_idx": i,
                "timestamp": str(timestamps[i]),
                "close": round(float(closes[i]), 2),
                "direction": direction,
                "outcome": "win" if win else "loss",
                "risk_amount": round(risk_amount, 2),
                "pnl": round(pnl, 2),
                "equity_after": round(equity, 2),
                "drawdown_pct": round(dd * 100, 2),
                "session": sess,
            })

        # Sample equity curve every 288 bars (~1 day) for plotting
        if i % 288 == 0:
            equity_curve.append({
                "bar_idx": i,
                "timestamp": str(timestamps[i]),
                "equity": round(equity, 2),
            })

    # Final equity point
    equity_curve.append({
        "bar_idx": n - 1,
        "timestamp": str(timestamps[-1]),
        "equity": round(equity, 2),
    })

    trade_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve)

    # Compute stats
    if len(trades) > 0:
        wins = trade_df[trade_df["outcome"] == "win"]
        losses = trade_df[trade_df["outcome"] == "loss"]
        stats = {
            "total_trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / len(trades), 4),
            "total_pnl": round(trade_df["pnl"].sum(), 2),
            "final_equity": round(equity, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "avg_winner_pnl": round(wins["pnl"].mean(), 2) if len(wins) > 0 else 0,
            "avg_loser_pnl": round(losses["pnl"].mean(), 2) if len(losses) > 0 else 0,
            "profit_factor": round(
                wins["pnl"].sum() / abs(losses["pnl"].sum()), 4
            ) if len(losses) > 0 and losses["pnl"].sum() != 0 else float("inf"),
        }
    else:
        stats = {
            "total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
            "total_pnl": 0, "final_equity": cfg.initial_equity,
            "max_drawdown_pct": 0,
        }

    return trade_df, equity_df, stats


# ─── comparison_matrix ──────────────────────────────────────────────────────
def comparison_matrix(
    df: pd.DataFrame, long_mask: pd.Series, short_mask: pd.Series
) -> pd.DataFrame:
    """
    Build a matrix comparing random-entry vs ICT-filtered win rates
    across all R × horizon label combinations.
    """
    r_values = [1, 2, 3]
    horizons = [12, 24, 48, 96, 288]
    rows = []

    for r in r_values:
        for h in horizons:
            long_col = f"label_long_hit_{r}r_{h}c"
            short_col = f"label_short_hit_{r}r_{h}c"

            if long_col not in df.columns or short_col not in df.columns:
                continue

            # Random entry: unconditional win rate (average of long + short)
            rand_long = df[long_col].mean()
            rand_short = df[short_col].mean()
            rand_avg = (rand_long + rand_short) / 2

            # ICT filtered
            filt_long_signals = df.loc[long_mask]
            filt_short_signals = df.loc[short_mask]

            filt_long_n = len(filt_long_signals)
            filt_short_n = len(filt_short_signals)
            filt_total = filt_long_n + filt_short_n

            if filt_total > 0:
                filt_wins = (
                    filt_long_signals[long_col].sum()
                    + filt_short_signals[short_col].sum()
                )
                filt_wr = filt_wins / filt_total
            else:
                filt_wr = 0.0

            # Break-even win rate for this R
            be_wr = 1 / (1 + r)

            # Edge = filtered WR - random WR
            edge = filt_wr - rand_avg

            rows.append({
                "R_target": f"{r}R",
                "horizon": f"{h}c",
                "random_wr": round(rand_avg, 4),
                "ict_filtered_wr": round(filt_wr, 4),
                "edge_pp": round(edge * 100, 2),  # percentage points
                "breakeven_wr": round(be_wr, 4),
                "n_filtered": filt_total,
            })

    return pd.DataFrame(rows)


# ─── print_report ───────────────────────────────────────────────────────────
def print_report(
    sig_results: dict,
    eq_stats: dict,
    comp_df: pd.DataFrame,
    trade_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    cfg: Config = CFG,
    old_baseline_results: dict | None = None,
) -> None:
    """Print console report and save output files."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    sep = "=" * 70
    rule = "-" * 70

    print(f"\n{sep}")
    print("  BASELINE BACKTEST v3 -- Analytics-Derived Filter Stack")
    print(f"{sep}")
    print(f"  Label: {cfg.r_target}R target / 1R stop / {cfg.horizon}-bar horizon")
    print(f"  Date range: {cfg.min_date} onward")
    print(f"  v3 toggles: d1_trend={cfg.use_d1_trend}, h4_sweep={cfg.use_h4_sweep}, "
          f"h4_bear_trend={cfg.require_h4_bear_trend}")
    print(f"              routing={cfg.direction_session_routing}, "
          f"ATR=[{cfg.atr_ratio_min},{cfg.atr_ratio_max}], "
          f"no_mon_tue={cfg.exclude_mon_tue}")

    # --- unconditional baselines ---
    print(f"\n{rule}")
    print("  UNCONDITIONAL (RANDOM ENTRY) WIN RATES")
    print(rule)
    print(f"  Long  random WR: {sig_results['unconditional_long_wr']:.4f}")
    print(f"  Short random WR: {sig_results['unconditional_short_wr']:.4f}")
    be_wr = 1 / (1 + cfg.r_target)
    print(f"  Break-even WR for {cfg.r_target}R: {be_wr:.4f}")

    # --- signal-level results ---
    print(f"\n{rule}")
    print("  ICT-FILTERED SIGNAL ANALYSIS")
    print(rule)
    for direction in ["long", "short", "combined"]:
        d = sig_results.get(direction, {})
        label = direction.upper()
        print(f"\n  {label}:")
        print(f"    Trades:        {d.get('n', 0):>6}")
        print(f"    Win rate:      {d.get('win_rate', 0):>6.2%}")
        print(f"    EV (R):        {d.get('ev_r', 0):>+6.4f}")
        print(f"    Profit factor: {d.get('profit_factor', 0):>6.2f}")

    # --- session breakdown ---
    print(f"\n{rule}")
    print("  SESSION BREAKDOWN")
    print(rule)
    for sess, data in sig_results.get("session_breakdown", {}).items():
        n = data.get("n", 0)
        wr = data.get("win_rate", 0)
        print(f"  {sess:<12}  trades={n:<5}  WR={wr:.2%}")

    # --- equity simulation ---
    print(f"\n{rule}")
    print("  EQUITY SIMULATION (2% risk, 48-bar cooldown)")
    print(rule)
    print(f"  Initial equity:  ${cfg.initial_equity:>12,.2f}")
    print(f"  Final equity:    ${eq_stats.get('final_equity', 0):>12,.2f}")
    print(f"  Net P&L:         ${eq_stats.get('total_pnl', 0):>+12,.2f}")
    print(f"  Total trades:    {eq_stats.get('total_trades', 0):>6}")
    print(f"  Win rate:        {eq_stats.get('win_rate', 0):>6.2%}")
    print(f"  Max drawdown:    {eq_stats.get('max_drawdown_pct', 0):>6.2f}%")
    print(f"  Profit factor:   {eq_stats.get('profit_factor', 0):>6.2f}")

    # --- comparison matrix ---
    print(f"\n{rule}")
    print("  COMPARISON MATRIX: RANDOM vs ICT-FILTERED WIN RATES")
    print(rule)
    print(f"  {'R':>4}  {'Horizon':>8}  {'Random WR':>10}  {'ICT WR':>10}  "
          f"{'Edge (pp)':>10}  {'BE WR':>8}  {'N':>6}")
    print(f"  {'----':>4}  {'--------':>8}  {'----------':>10}  {'----------':>10}  "
          f"{'----------':>10}  {'--------':>8}  {'------':>6}")
    for _, row in comp_df.iterrows():
        print(f"  {row['R_target']:>4}  {row['horizon']:>8}  "
              f"{row['random_wr']:>10.4f}  {row['ict_filtered_wr']:>10.4f}  "
              f"{row['edge_pp']:>+10.2f}  {row['breakeven_wr']:>8.4f}  "
              f"{row['n_filtered']:>6}")

    # --- v2 vs v3 side-by-side comparison ---
    if old_baseline_results is not None:
        print(f"\n{rule}")
        print("  v2 vs v3 SIDE-BY-SIDE COMPARISON")
        print(rule)
        old_c = old_baseline_results.get("combined", {})
        new_c = sig_results.get("combined", {})
        print(f"  {'Metric':<20}  {'v2 (old)':>10}  {'v3 (new)':>10}  {'Delta':>10}")
        print(f"  {'-' * 20}  {'-' * 10}  {'-' * 10}  {'-' * 10}")
        # Signals
        old_n = old_c.get("n", 0)
        new_n = new_c.get("n", 0)
        print(f"  {'Signals':<20}  {old_n:>10}  {new_n:>10}  {new_n - old_n:>+10}")
        # Win rate
        old_wr = old_c.get("win_rate", 0)
        new_wr = new_c.get("win_rate", 0)
        print(f"  {'Win Rate':<20}  {old_wr:>10.2%}  {new_wr:>10.2%}  "
              f"{(new_wr - old_wr) * 100:>+10.2f}pp")
        # EV
        old_ev = old_c.get("ev_r", 0)
        new_ev = new_c.get("ev_r", 0)
        print(f"  {'EV (R)':<20}  {old_ev:>+10.4f}  {new_ev:>+10.4f}  "
              f"{new_ev - old_ev:>+10.4f}")
        # PF
        old_pf = old_c.get("profit_factor", 0)
        new_pf = new_c.get("profit_factor", 0)
        print(f"  {'Profit Factor':<20}  {old_pf:>10.2f}  {new_pf:>10.2f}  "
              f"{new_pf - old_pf:>+10.2f}")

        # Per-direction breakdown
        for d in ["long", "short"]:
            old_d = old_baseline_results.get(d, {})
            new_d = sig_results.get(d, {})
            print(f"\n  {d.upper()}:")
            o_n = old_d.get("n", 0)
            n_n = new_d.get("n", 0)
            o_wr = old_d.get("win_rate", 0)
            n_wr = new_d.get("win_rate", 0)
            print(f"    Signals:  {o_n:>6} -> {n_n:>6}   "
                  f"WR: {o_wr:.2%} -> {n_wr:.2%}  ({(n_wr - o_wr) * 100:+.2f}pp)")

    print(f"\n{sep}\n")

    # --- save files ---
    # Trade log
    trade_log_path = RESULTS_DIR / "trade_log_baseline_v3.csv"
    if len(trade_df) > 0:
        trade_df.to_csv(trade_log_path, index=False)
        print(f"  Saved: {trade_log_path}")
    else:
        print("  No trades to save.")

    # Summary JSON
    summary = {
        "config": {
            "r_target": cfg.r_target,
            "horizon": cfg.horizon,
            "risk_pct": cfg.risk_pct,
            "cooldown_bars": cfg.cooldown_bars,
            "cost_per_r": cfg.cost_per_r,
            "require_discount": cfg.require_discount,
            "require_ote": cfg.require_ote,
            "require_ob": cfg.require_ob,
            "use_d1_trend": cfg.use_d1_trend,
            "use_h4_sweep": cfg.use_h4_sweep,
            "require_h4_bear_trend": cfg.require_h4_bear_trend,
            "atr_ratio_min": cfg.atr_ratio_min,
            "atr_ratio_max": cfg.atr_ratio_max,
            "exclude_mon_tue": cfg.exclude_mon_tue,
            "direction_session_routing": cfg.direction_session_routing,
            "min_date": cfg.min_date,
        },
        "signal_analysis": {
            k: v for k, v in sig_results.items()
            if k not in ("unconditional_long_wr", "unconditional_short_wr")
        },
        "unconditional_long_wr": round(sig_results["unconditional_long_wr"], 6),
        "unconditional_short_wr": round(sig_results["unconditional_short_wr"], 6),
        "equity_simulation": eq_stats,
        "comparison_matrix": comp_df.to_dict(orient="records"),
    }
    if old_baseline_results is not None:
        summary["old_baseline_v2"] = {
            k: v for k, v in old_baseline_results.items()
            if k not in ("unconditional_long_wr", "unconditional_short_wr")
        }
    summary_path = RESULTS_DIR / "baseline_v3_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved: {summary_path}")

    # Equity curve (plotly)
    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(equity_df["timestamp"]),
            y=equity_df["equity"],
            mode="lines",
            name="Equity",
            line=dict(color="royalblue", width=1.5),
        ))
        fig.update_layout(
            title=f"Baseline v3 Equity Curve ({cfg.r_target}R / {cfg.horizon}c)",
            xaxis_title="Date",
            yaxis_title="Equity ($)",
            template="plotly_dark",
        )
        eq_path = RESULTS_DIR / "equity_baseline_v3.html"
        fig.write_html(str(eq_path))
        print(f"  Saved: {eq_path}")
    except ImportError:
        print("  (plotly not installed --skipping equity chart)")


# ─── main ───────────────────────────────────────────────────────────────────
def main() -> None:
    cfg = CFG

    # 1. Load data
    df = load_labeled(cfg)

    # 2. Compute OLD v2 baseline for comparison
    old_long, old_short = old_baseline_filters(df)
    print(f"\n  Old v2 signals: {old_long.sum():,} long + {old_short.sum():,} short "
          f"= {old_long.sum() + old_short.sum():,} total")
    old_baseline_results = signal_analysis(df, old_long, old_short, cfg)

    # 3. Compute NEW v3 filter masks
    long_mask, short_mask = ict_filters(df, cfg)
    print(f"  New v3 signals: {long_mask.sum():,} long + {short_mask.sum():,} short "
          f"= {long_mask.sum() + short_mask.sum():,} total")

    # 4. Vectorized signal analysis
    sig_results = signal_analysis(df, long_mask, short_mask, cfg)

    # 5. Equity simulation
    trade_df, equity_df, eq_stats = simulate_equity(df, long_mask, short_mask, cfg)

    # 6. Comparison matrix
    comp_df = comparison_matrix(df, long_mask, short_mask)

    # 7. Report + save
    print_report(sig_results, eq_stats, comp_df, trade_df, equity_df, cfg,
                 old_baseline_results=old_baseline_results)


if __name__ == "__main__":
    main()
