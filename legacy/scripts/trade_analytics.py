"""
trade_analytics.py -- Comprehensive diagnostic report on ICT signal quality.
=============================================================================
Loads the v2 labeled dataset, regenerates the 624 ICT-filtered signals from
baseline_backtest_v2.py, and produces detailed breakdowns:

  1. Component contribution analysis  (drop-one-filter ablation)
  2. Win rate breakdowns              (session, direction, regime, time, vol)
  3. Short-side deep dive             (find profitable short conditions or kill shorts)
  4. Signal decay over time           (quarterly WR trend + linear regression)
  5. R-multiple sensitivity           (heatmap across all R x horizon combos)
  6. Consecutive loss analysis        (streak distribution + mean-reversion check)

Output:
  - Console report
  - results/trade_analytics_report.json
  - results/signal_breakdown.csv

This is pure analysis -- no optimization, no strategy changes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_PATH = PROJECT_DIR / "data" / "labeled" / "BTCUSDT_MASTER_labeled.parquet"
RESULTS_DIR = SCRIPT_DIR / "results"

# ---------------------------------------------------------------------------
# Config (matches baseline_backtest_v2.py exactly)
# ---------------------------------------------------------------------------
R_TARGET = 2
HORIZON = 48
LONG_LABEL = f"label_long_hit_{R_TARGET}r_{HORIZON}c"
SHORT_LABEL = f"label_short_hit_{R_TARGET}r_{HORIZON}c"
COST_PER_R = 0.05
MIN_DATE = "2020-01-01"
BE_WR = 1 / (1 + R_TARGET)  # 0.3333 for 2R

LOW_SAMPLE = 10  # flag subgroups below this


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ev(wr: float) -> float:
    """Expected value in R for given win rate at 2R target / 1R stop."""
    return wr * (R_TARGET - COST_PER_R) - (1 - wr) * (1 + COST_PER_R)


def _pf(wins: int, losses: int) -> float:
    return (wins * R_TARGET) / (losses * 1.0) if losses > 0 else float("inf")


def _stats(wins_series: pd.Series) -> dict:
    """Compute standard stats from a boolean/0-1 win series."""
    n = len(wins_series)
    if n == 0:
        return {"n": 0, "wins": 0, "wr": 0.0, "ev_r": 0.0, "pf": 0.0, "flag": "NO DATA"}
    w = int(wins_series.sum())
    wr = w / n
    ev = _ev(wr)
    pf = _pf(w, n - w)
    flag = "LOW SAMPLE" if n < LOW_SAMPLE else ""
    return {"n": n, "wins": w, "wr": round(wr, 4), "ev_r": round(ev, 4),
            "pf": round(pf, 4), "flag": flag}


def _pr(label: str, s: dict, indent: int = 2) -> None:
    """Print a stats dict as one formatted line."""
    pad = " " * indent
    flag = f"  ** {s['flag']} **" if s.get("flag") else ""
    print(f"{pad}{label:<40}  N={s['n']:<6}  WR={s['wr']:.2%}  "
          f"EV={s['ev_r']:+.4f}R  PF={s['pf']:.2f}{flag}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    print(f"Loading {DATA_PATH.name} ...")
    df = pd.read_parquet(DATA_PATH)
    print(f"  raw: {df.shape[0]:,} x {df.shape[1]}")
    df = df[df["bar_start_ts_utc"] >= pd.Timestamp(MIN_DATE, tz="UTC")].copy()
    df = df.dropna(subset=[LONG_LABEL, SHORT_LABEL])
    print(f"  after date filter + NaN drop: {df.shape[0]:,}")
    return df


# ---------------------------------------------------------------------------
# ICT filter masks (same as baseline_backtest_v2.py)
# ---------------------------------------------------------------------------
def ict_masks(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    sb = (df["sess_sb_london"] == 1) | (df["sess_sb_ny_am"] == 1) | (df["sess_sb_ny_pm"] == 1)
    long_m = (
        (df["d1_ict_market_trend"] == 1)
        & (df["h4_ict_bull_liq_sweep"] == 1)
        & (df["h4_ict_fvg_bull"] == 1)
        & sb
    )
    short_m = (
        (df["d1_ict_market_trend"] == -1)
        & (df["h4_ict_bear_liq_sweep"] == 1)
        & (df["h4_ict_fvg_bear"] == 1)
        & sb
    )
    return long_m, short_m


# ---------------------------------------------------------------------------
# Build signal dataframe (one row per signal)
# ---------------------------------------------------------------------------
def build_signals(df: pd.DataFrame, long_m: pd.Series, short_m: pd.Series) -> pd.DataFrame:
    """Flatten long + short signals into a single DF with metadata columns."""
    parts = []
    if long_m.sum() > 0:
        lg = df.loc[long_m].copy()
        lg["direction"] = "long"
        lg["win"] = lg[LONG_LABEL].astype(int)
        parts.append(lg)
    if short_m.sum() > 0:
        sh = df.loc[short_m].copy()
        sh["direction"] = "short"
        sh["win"] = sh[SHORT_LABEL].astype(int)
        parts.append(sh)
    sig = pd.concat(parts).sort_index()

    # Enrich with breakdowns
    sig["year"] = sig["bar_start_ts_utc"].dt.year
    sig["quarter"] = sig["bar_start_ts_utc"].dt.to_period("Q").astype(str)
    sig["day_of_week"] = sig["ict_day_of_week"].astype(int)
    sig["hour"] = sig["ict_hour_of_day"].astype(int)
    sig["hour_bucket"] = (sig["hour"] // 4) * 4  # 0,4,8,12,16,20

    # Session label
    sig["session"] = "Off"
    sig.loc[sig["sess_sb_london"] == 1, "session"] = "London SB"
    sig.loc[sig["sess_sb_ny_am"] == 1, "session"] = "NY AM SB"
    sig.loc[sig["sess_sb_ny_pm"] == 1, "session"] = "NY PM SB"

    # ATR tercile
    atr = sig["ict_atr_14"]
    sig["atr_tercile"] = pd.qcut(atr, 3, labels=["low", "med", "high"], duplicates="drop")

    # Confluence score
    sig["confluence_score"] = sig["htf_confluence_score"]

    # Filter component values (for CSV export)
    sig["f_d1_trend"] = sig["d1_ict_market_trend"]
    sig["f_h4_sweep"] = np.where(
        sig["direction"] == "long",
        sig["h4_ict_bull_liq_sweep"],
        sig["h4_ict_bear_liq_sweep"],
    )
    sig["f_h4_fvg"] = np.where(
        sig["direction"] == "long",
        sig["h4_ict_fvg_bull"],
        sig["h4_ict_fvg_bear"],
    )
    sig["f_sb_session"] = 1  # always 1 by construction

    return sig


# ===================================================================
# ANALYSIS 1: Component Contribution
# ===================================================================
def analysis_component_contribution(df: pd.DataFrame) -> dict:
    sep = "=" * 70
    rule = "-" * 70
    print(f"\n{sep}")
    print("  1. COMPONENT CONTRIBUTION ANALYSIS")
    print(sep)

    sb = (df["sess_sb_london"] == 1) | (df["sess_sb_ny_am"] == 1) | (df["sess_sb_ny_pm"] == 1)

    # Define long/short components
    components = {
        "d1_trend":   {"long": df["d1_ict_market_trend"] == 1,
                       "short": df["d1_ict_market_trend"] == -1},
        "h4_sweep":   {"long": df["h4_ict_bull_liq_sweep"] == 1,
                       "short": df["h4_ict_bear_liq_sweep"] == 1},
        "h4_fvg":     {"long": df["h4_ict_fvg_bull"] == 1,
                       "short": df["h4_ict_fvg_bear"] == 1},
        "sb_session": {"long": sb, "short": sb},
    }

    def _combined_wr(long_mask, short_mask):
        n = long_mask.sum() + short_mask.sum()
        if n == 0:
            return 0.0, 0
        w = df.loc[long_mask, LONG_LABEL].sum() + df.loc[short_mask, SHORT_LABEL].sum()
        return w / n, int(n)

    # Full stack
    full_long = components["d1_trend"]["long"] & components["h4_sweep"]["long"] & components["h4_fvg"]["long"] & components["sb_session"]["long"]
    full_short = components["d1_trend"]["short"] & components["h4_sweep"]["short"] & components["h4_fvg"]["short"] & components["sb_session"]["short"]
    full_wr, full_n = _combined_wr(full_long, full_short)

    results = {"full_stack": {"wr": round(full_wr, 4), "n": full_n}}

    print(f"\n  {'Variant':<35}  {'N':>6}  {'WR':>8}  {'Delta':>8}")
    print(f"  {rule}")
    print(f"  {'Full stack (all 4)':<35}  {full_n:>6}  {full_wr:>8.2%}  {'---':>8}")

    # Drop-one analysis
    print(f"\n  --- Drop-one (remove single component, keep other 3) ---")
    for drop_name in ["d1_trend", "h4_sweep", "h4_fvg", "sb_session"]:
        keep = [k for k in components if k != drop_name]
        long_m = pd.Series(True, index=df.index)
        short_m = pd.Series(True, index=df.index)
        for k in keep:
            long_m = long_m & components[k]["long"]
            short_m = short_m & components[k]["short"]
        wr, n = _combined_wr(long_m, short_m)
        delta = wr - full_wr
        results[f"drop_{drop_name}"] = {"wr": round(wr, 4), "n": n, "delta_pp": round(delta * 100, 2)}
        flag = "  ** MOST VALUABLE **" if delta < -0.02 else ""
        print(f"  Drop {drop_name:<28}  {n:>6}  {wr:>8.2%}  {delta:>+8.2%}{flag}")

    # Solo analysis (only-one component vs random)
    unc_wr = (df[LONG_LABEL].mean() + df[SHORT_LABEL].mean()) / 2
    print(f"\n  --- Solo (ONLY this component vs random baseline {unc_wr:.4f}) ---")
    results["random_baseline_wr"] = round(unc_wr, 4)
    for comp_name, masks in components.items():
        long_m = masks["long"]
        short_m = masks["short"]
        wr, n = _combined_wr(long_m, short_m)
        edge = wr - unc_wr
        results[f"solo_{comp_name}"] = {"wr": round(wr, 4), "n": n, "edge_pp": round(edge * 100, 2)}
        print(f"  Only {comp_name:<29}  {n:>6}  {wr:>8.2%}  edge={edge:>+7.2%}")

    return results


# ===================================================================
# ANALYSIS 2: Win Rate Breakdowns
# ===================================================================
def analysis_breakdowns(sig: pd.DataFrame, df: pd.DataFrame) -> dict:
    sep = "=" * 70
    rule = "-" * 70
    print(f"\n{sep}")
    print("  2. WIN RATE BREAKDOWNS")
    print(sep)
    results = {}

    # --- By session ---
    print(f"\n  --- By Session ---")
    for sess in ["London SB", "NY AM SB", "NY PM SB"]:
        sub = sig[sig["session"] == sess]
        s = _stats(sub["win"])
        _pr(sess, s)
        results[f"session_{sess}"] = s

    # --- By direction ---
    print(f"\n  --- By Direction ---")
    for d in ["long", "short"]:
        sub = sig[sig["direction"] == d]
        s = _stats(sub["win"])
        _pr(d.capitalize(), s)
        results[f"dir_{d}"] = s

    # --- By direction x session ---
    print(f"\n  --- Direction x Session ---")
    for d in ["long", "short"]:
        for sess in ["London SB", "NY AM SB", "NY PM SB"]:
            sub = sig[(sig["direction"] == d) & (sig["session"] == sess)]
            s = _stats(sub["win"])
            _pr(f"{d.capitalize()} / {sess}", s)
            results[f"dir_sess_{d}_{sess}"] = s

    # --- By direction x regime (short side deep regime checks) ---
    print(f"\n  --- Short Regime Checks ---")
    shorts = sig[sig["direction"] == "short"]

    # h4 trend double confirm
    s_h4_bear = shorts[shorts["h4_ict_market_trend"] == -1]
    s = _stats(s_h4_bear["win"])
    _pr("Short + h4_trend==-1 (double bear)", s)
    results["short_h4_double_bear"] = s

    # confluence <= -3
    s_conf3 = shorts[shorts["confluence_score"] <= -3]
    s = _stats(s_conf3["win"])
    _pr("Short + confluence <= -3", s)
    results["short_confluence_le_neg3"] = s

    # --- By day of week ---
    print(f"\n  --- By Day of Week ---")
    dow_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    for d in range(7):
        sub = sig[sig["day_of_week"] == d]
        s = _stats(sub["win"])
        _pr(dow_names.get(d, str(d)), s)
        results[f"dow_{d}"] = s

    # --- By hour bucket ---
    print(f"\n  --- By Hour Bucket (UTC) ---")
    for hb in [0, 4, 8, 12, 16, 20]:
        sub = sig[sig["hour_bucket"] == hb]
        s = _stats(sub["win"])
        _pr(f"{hb:02d}:00-{hb+3:02d}:59 UTC", s)
        results[f"hour_{hb:02d}"] = s

    # --- By year ---
    print(f"\n  --- By Year (CRITICAL: signal decay check) ---")
    for yr in sorted(sig["year"].unique()):
        sub = sig[sig["year"] == yr]
        s = _stats(sub["win"])
        above = "ABOVE BE" if s["wr"] > BE_WR else "below BE"
        flag_str = f"  ** {s['flag']} **" if s.get("flag") else ""
        print(f"  {yr:<40}  N={s['n']:<6}  WR={s['wr']:.2%}  "
              f"EV={s['ev_r']:+.4f}R  PF={s['pf']:.2f}  [{above}]{flag_str}")
        results[f"year_{yr}"] = s

    # --- By ATR tercile ---
    print(f"\n  --- By Volatility (ATR tercile) ---")
    for t in ["low", "med", "high"]:
        sub = sig[sig["atr_tercile"] == t]
        s = _stats(sub["win"])
        _pr(f"ATR {t}", s)
        results[f"atr_{t}"] = s

    # --- By confluence score bucket ---
    print(f"\n  --- By HTF Confluence Score ---")
    longs = sig[sig["direction"] == "long"]
    for lo, hi, label in [(-99, -3, "conf <= -3"), (-3, -1, "-3 < conf <= -1"),
                          (-1, 1, "-1 < conf < +1"), (1, 3, "+1 <= conf < +3"),
                          (3, 5, "+3 <= conf < +5"), (5, 99, "conf >= +5")]:
        sub = longs[(longs["confluence_score"] > lo) & (longs["confluence_score"] <= hi)]
        if lo == -99:
            sub = longs[longs["confluence_score"] <= hi]
        if hi == 99:
            sub = longs[longs["confluence_score"] >= lo]
        s = _stats(sub["win"])
        _pr(f"Long / {label}", s)
        results[f"long_conf_{lo}_{hi}"] = s

    for lo, hi, label in [(3, 99, "conf >= +3"), (1, 3, "+1 <= conf < +3"),
                          (-1, 1, "-1 < conf < +1"), (-3, -1, "-3 < conf <= -1"),
                          (-5, -3, "-5 < conf <= -3"), (-99, -5, "conf <= -5")]:
        sub = shorts[(shorts["confluence_score"] > lo) & (shorts["confluence_score"] <= hi)]
        if lo == -99:
            sub = shorts[shorts["confluence_score"] <= hi]
        if hi == 99:
            sub = shorts[shorts["confluence_score"] >= lo]
        s = _stats(sub["win"])
        _pr(f"Short / {label}", s)
        results[f"short_conf_{lo}_{hi}"] = s

    return results


# ===================================================================
# ANALYSIS 3: Short-Side Deep Dive
# ===================================================================
def analysis_short_deep_dive(sig: pd.DataFrame, df: pd.DataFrame) -> dict:
    sep = "=" * 70
    print(f"\n{sep}")
    print("  3. SHORT-SIDE DEEP DIVE")
    print(f"     Overall short EV = {_ev(sig[sig['direction']=='short']['win'].mean()):+.4f}R")
    print(f"     Break-even WR = {BE_WR:.4f}")
    print(sep)

    shorts = sig[sig["direction"] == "short"]
    results = {}

    conditions = [
        ("All shorts (baseline)",        shorts),
        ("London SB only",               shorts[shorts["session"] == "London SB"]),
        ("NY AM SB only",                shorts[shorts["session"] == "NY AM SB"]),
        ("NY PM SB only",                shorts[shorts["session"] == "NY PM SB"]),
        ("h4_trend == -1 (double bear)", shorts[shorts["h4_ict_market_trend"] == -1]),
        ("confluence <= -3",             shorts[shorts["confluence_score"] <= -3]),
        ("confluence <= -5",             shorts[shorts["confluence_score"] <= -5]),
        ("Year 2022 only",              shorts[shorts["year"] == 2022]),
        ("Year 2022-2023 (bear+sideways)", shorts[shorts["year"].isin([2022, 2023])]),
        ("ATR top tercile (high vol)",   shorts[shorts["atr_tercile"] == "high"]),
        ("h4_trend==-1 + high vol",      shorts[(shorts["h4_ict_market_trend"] == -1) & (shorts["atr_tercile"] == "high")]),
        ("h4_trend==-1 + NY PM",         shorts[(shorts["h4_ict_market_trend"] == -1) & (shorts["session"] == "NY PM SB")]),
    ]

    above_be = []
    print(f"\n  {'Condition':<40}  {'N':>5}  {'WR':>8}  {'EV':>9}  {'vs BE':>7}  Note")
    print(f"  {'-'*40}  {'-'*5}  {'-'*8}  {'-'*9}  {'-'*7}  {'-'*15}")
    for name, sub in conditions:
        s = _stats(sub["win"])
        vs_be = "ABOVE" if s["wr"] > BE_WR else "below"
        flag = f"  ** {s['flag']} **" if s.get("flag") else ""
        print(f"  {name:<40}  {s['n']:>5}  {s['wr']:>8.2%}  {s['ev_r']:>+9.4f}  {vs_be:>7}{flag}")
        results[name] = s
        if s["ev_r"] > 0 and s["n"] >= LOW_SAMPLE:
            above_be.append((name, s))

    print()
    if above_be:
        print("  PROFITABLE SHORT CONDITIONS FOUND:")
        for name, s in above_be:
            print(f"    -> {name}: WR={s['wr']:.2%}, N={s['n']}, EV={s['ev_r']:+.4f}R")
    else:
        print("  ** NO short condition produces above break-even with sufficient sample **")
        print("  ** RECOMMENDATION: Drop shorts entirely from the strategy **")

    results["_above_be_conditions"] = [name for name, _ in above_be]
    return results


# ===================================================================
# ANALYSIS 4: Signal Decay Over Time
# ===================================================================
def analysis_signal_decay(sig: pd.DataFrame) -> dict:
    sep = "=" * 70
    print(f"\n{sep}")
    print("  4. SIGNAL DECAY OVER TIME (quarterly)")
    print(sep)

    quarterly = sig.groupby("quarter")["win"].agg(["count", "sum", "mean"])
    quarterly.columns = ["n", "wins", "wr"]
    quarterly["ev_r"] = quarterly["wr"].apply(_ev)

    results = {}
    print(f"\n  {'Quarter':<12}  {'N':>5}  {'WR':>8}  {'EV':>9}")
    print(f"  {'-'*12}  {'-'*5}  {'-'*8}  {'-'*9}")
    for q, row in quarterly.iterrows():
        flag = "  ** LOW SAMPLE **" if row["n"] < LOW_SAMPLE else ""
        print(f"  {q:<12}  {int(row['n']):>5}  {row['wr']:>8.2%}  {row['ev_r']:>+9.4f}{flag}")
        results[str(q)] = {"n": int(row["n"]), "wr": round(row["wr"], 4),
                           "ev_r": round(row["ev_r"], 4)}

    # Linear regression on WR over time
    from scipy import stats as sp_stats
    quarterly = quarterly.reset_index()
    x = np.arange(len(quarterly))
    y = quarterly["wr"].values
    if len(x) >= 3:
        slope, intercept, r_value, p_value, std_err = sp_stats.linregress(x, y)
        print(f"\n  Linear regression on quarterly WR:")
        print(f"    slope     = {slope:+.4f} per quarter")
        print(f"    R-squared = {r_value**2:.4f}")
        print(f"    p-value   = {p_value:.4f}")
        if p_value < 0.05:
            if slope > 0:
                print(f"    >> EDGE IS GROWING (statistically significant)")
            else:
                print(f"    >> EDGE IS DECAYING (statistically significant)")
        else:
            print(f"    >> No significant trend (p > 0.05) -- edge appears STABLE")
        results["_regression"] = {
            "slope": round(slope, 6), "r_squared": round(r_value**2, 4),
            "p_value": round(p_value, 4), "interpretation": "stable" if p_value >= 0.05 else ("growing" if slope > 0 else "decaying")
        }

    return results


# ===================================================================
# ANALYSIS 5: R-Multiple Sensitivity
# ===================================================================
def analysis_r_sensitivity(df: pd.DataFrame, long_m: pd.Series, short_m: pd.Series) -> dict:
    sep = "=" * 70
    print(f"\n{sep}")
    print("  5. R-MULTIPLE SENSITIVITY (heatmap)")
    print(sep)

    r_vals = [1, 2, 3]
    horizons = [12, 24, 48, 96, 288]
    results = {}

    # Header
    print(f"\n  {'':>8}", end="")
    for h in horizons:
        print(f"  {h}c", end="       ")
    print()
    print(f"  {'':>8}", end="")
    for _ in horizons:
        print(f"  {'--------':>10}", end="")
    print()

    for r in r_vals:
        print(f"  {r}R    ", end="")
        for h in horizons:
            lc = f"label_long_hit_{r}r_{h}c"
            sc = f"label_short_hit_{r}r_{h}c"
            if lc in df.columns and sc in df.columns:
                w = df.loc[long_m, lc].sum() + df.loc[short_m, sc].sum()
                n = long_m.sum() + short_m.sum()
                wr = w / n if n > 0 else 0
                be = 1 / (1 + r)
                marker = "+" if wr > be else " "
                print(f"  {marker}{wr:.4f}   ", end="")
                results[f"{r}R_{h}c"] = {"wr": round(wr, 4), "be": round(be, 4),
                                         "above_be": wr > be, "n": int(n)}
            else:
                print(f"  {'N/A':>8}   ", end="")
        print()

    print(f"\n  + = above break-even for that R target")
    return results


# ===================================================================
# ANALYSIS 6: Consecutive Loss Analysis
# ===================================================================
def analysis_consecutive_losses(sig: pd.DataFrame) -> dict:
    sep = "=" * 70
    print(f"\n{sep}")
    print("  6. CONSECUTIVE LOSS ANALYSIS")
    print(sep)

    wins = sig["win"].values
    n = len(wins)
    results = {}

    # Find loss streaks
    streaks = []
    current = 0
    for w in wins:
        if w == 0:
            current += 1
        else:
            if current > 0:
                streaks.append(current)
            current = 0
    if current > 0:
        streaks.append(current)

    if not streaks:
        print("  No losses found.")
        return {"max_streak": 0}

    streaks = np.array(streaks)
    max_streak = int(streaks.max())
    results["max_consecutive_losses"] = max_streak
    results["mean_loss_streak"] = round(streaks.mean(), 2)
    results["median_loss_streak"] = int(np.median(streaks))

    print(f"\n  Max consecutive losses: {max_streak}")
    print(f"  Mean loss streak:       {streaks.mean():.2f}")
    print(f"  Median loss streak:     {int(np.median(streaks))}")

    # Distribution
    print(f"\n  Streak length distribution:")
    for length in range(1, min(max_streak + 1, 8)):
        count = (streaks == length).sum()
        print(f"    {length} in a row: {count}")
        results[f"streak_{length}"] = int(count)
    if max_streak >= 8:
        count = (streaks >= 8).sum()
        print(f"    8+ in a row: {count}")
        results["streak_8plus"] = int(count)

    # Mean reversion check: after a 3+ loss streak ends (with a win),
    # look at the SECOND signal after the streak -- the one AFTER the
    # streak-breaking win. The win itself is tautological.
    print(f"\n  Mean reversion after 3+ loss streak:")
    after_streak_wins = []
    i = 0
    while i < n:
        if wins[i] == 0:
            streak_start = i
            while i < n and wins[i] == 0:
                i += 1
            streak_len = i - streak_start
            # i now points at the streak-breaking win (or end of array)
            # skip that win and look at i+1
            if streak_len >= 3 and (i + 1) < n:
                after_streak_wins.append(wins[i + 1])
            if i < n:
                i += 1  # advance past the streak-breaking win
        else:
            i += 1

    if after_streak_wins:
        after_wr = np.mean(after_streak_wins)
        overall_wr = sig["win"].mean()
        print(f"    Signals after 3+ loss streak: {len(after_streak_wins)}")
        print(f"    WR after streak:              {after_wr:.2%}")
        print(f"    Overall WR:                   {overall_wr:.2%}")
        diff = after_wr - overall_wr
        if abs(diff) < 0.03:
            print(f"    -> No meaningful mean reversion (delta = {diff:+.2%})")
        elif diff > 0:
            print(f"    -> Slight mean reversion detected (delta = {diff:+.2%})")
        else:
            print(f"    -> Anti-reversion: losses cluster (delta = {diff:+.2%})")
        results["after_3plus_streak_wr"] = round(after_wr, 4)
        results["after_3plus_streak_n"] = len(after_streak_wins)
    else:
        print(f"    No 3+ loss streaks found")

    return results


# ===================================================================
# Save signal breakdown CSV
# ===================================================================
def save_signal_csv(sig: pd.DataFrame) -> None:
    csv_cols = [
        "bar_start_ts_utc", "direction", "session", "win", "year", "quarter",
        "day_of_week", "hour", "atr_tercile", "confluence_score",
        "f_d1_trend", "f_h4_sweep", "f_h4_fvg", "f_sb_session",
        "close", "ict_atr_14", "h4_ict_market_trend",
    ]
    out = sig[[c for c in csv_cols if c in sig.columns]].copy()
    out.rename(columns={"bar_start_ts_utc": "timestamp"}, inplace=True)
    path = RESULTS_DIR / "signal_breakdown.csv"
    out.to_csv(path, index=False)
    print(f"\n  Saved: {path} ({len(out)} rows)")


# ===================================================================
# MAIN
# ===================================================================
def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    long_m, short_m = ict_masks(df)
    print(f"  Signals: {long_m.sum()} long + {short_m.sum()} short = {long_m.sum()+short_m.sum()} total")

    sig = build_signals(df, long_m, short_m)
    print(f"  Signal DF: {len(sig)} rows")

    report = {}

    # 1. Component contribution
    report["1_component_contribution"] = analysis_component_contribution(df)

    # 2. Win rate breakdowns
    report["2_breakdowns"] = analysis_breakdowns(sig, df)

    # 3. Short deep dive
    report["3_short_deep_dive"] = analysis_short_deep_dive(sig, df)

    # 4. Signal decay
    report["4_signal_decay"] = analysis_signal_decay(sig)

    # 5. R-multiple sensitivity
    report["5_r_sensitivity"] = analysis_r_sensitivity(df, long_m, short_m)

    # 6. Consecutive losses
    report["6_consecutive_losses"] = analysis_consecutive_losses(sig)

    # Save JSON
    json_path = RESULTS_DIR / "trade_analytics_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Saved: {json_path}")

    # Save CSV
    save_signal_csv(sig)

    print(f"\n{'='*70}")
    print("  TRADE ANALYTICS COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
