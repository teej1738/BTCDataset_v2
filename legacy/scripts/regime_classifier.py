"""
regime_classifier.py -- Volatility Regime Classification (Step 3)
=================================================================
Assigns a volatility regime label to every bar in the labeled dataset,
then tests whether Config B edge is regime-dependent.

Method:
  1. Compute 288-bar (1-day) rolling realized vol from 5m log returns
  2. Compute rolling percentile of rvol over 8640-bar (1-month) lookback
  3. Assign regime via rolling tertile thresholds (no lookahead):
       0 = low-vol / choppy
       1 = medium-vol / trending
       2 = high-vol / stressed
  4. Detect regime transitions as variance change points
  5. Re-run Config B per regime: WR, EV, PF, N

Output:
  - regime_label column added to labeled parquet (int8, -1 = warmup)
  - results/regime_analysis.json
  - results/regime_classification.html  (rvol time series + per-regime bars)

GO hypothesis:  regime 1 (medium) is where Config B edge lives,
                regime 0 (low-vol) is where it dies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_backtest_v2 import Config, ict_filters

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_PATH = PROJECT_DIR / "data" / "labeled" / "BTCUSDT_MASTER_labeled.parquet"
RESULTS_DIR = SCRIPT_DIR / "results"


# ── config ──────────────────────────────────────────────────────────────────
@dataclass
class RegimeConfig:
    rvol_window: int = 288         # 1 day of 5m bars for realized vol
    regime_lookback: int = 8640    # 1 month for rolling percentile thresholds
    min_warmup: int = 2016         # 1 week minimum before regime assignment
    low_pctile: float = 0.33      # below = low-vol regime
    high_pctile: float = 0.67     # above = high-vol regime
    min_date: str = "2020-01-01"  # date filter (D10)

REGIME_NAMES = {-1: "warmup", 0: "LOW", 1: "MED", 2: "HIGH"}


# ── 1. Compute realized vol ────────────────────────────────────────────────
def compute_realized_vol(close: pd.Series, window: int = 288) -> pd.Series:
    """
    Rolling realized volatility from 5m log returns.
    RVol = rolling std of log(close_t / close_{t-1}) over `window` bars.
    """
    log_ret = np.log(close / close.shift(1))
    rvol = log_ret.rolling(window, min_periods=window // 2).std()
    return rvol


# ── 2. Assign regime labels ────────────────────────────────────────────────
def assign_regime_labels(
    rvol: pd.Series, rcfg: RegimeConfig
) -> tuple[pd.Series, pd.Series]:
    """
    Assign regime labels using rolling percentile thresholds (no lookahead).

    For each bar, compute the 33rd and 67th percentile of rvol over the
    past `regime_lookback` bars. Compare current rvol to these thresholds.

    Returns (regime_label, rolling_pctile).
    """
    low_thresh = rvol.rolling(
        rcfg.regime_lookback, min_periods=rcfg.min_warmup
    ).quantile(rcfg.low_pctile)

    high_thresh = rvol.rolling(
        rcfg.regime_lookback, min_periods=rcfg.min_warmup
    ).quantile(rcfg.high_pctile)

    regime = pd.Series(
        np.full(len(rvol), -1, dtype=np.int8), index=rvol.index
    )
    valid = rvol.notna() & low_thresh.notna()

    regime[valid & (rvol < low_thresh)] = 0   # low vol
    regime[valid & (rvol >= low_thresh) & (rvol < high_thresh)] = 1  # medium
    regime[valid & (rvol >= high_thresh)] = 2  # high vol

    # Rolling percentile (for diagnostics / chart)
    # Approximate via: rank of current rvol within the rolling window
    # Use the thresholds: if below low_thresh -> ~0-33%, etc.
    pctile = pd.Series(np.nan, index=rvol.index)
    pctile[regime == 0] = 0.17  # midpoint of [0, 0.33]
    pctile[regime == 1] = 0.50  # midpoint of [0.33, 0.67]
    pctile[regime == 2] = 0.83  # midpoint of [0.67, 1.0]

    return regime, pctile


# ── 3. Transition / change-point analysis ───────────────────────────────────
def analyze_transitions(regime: pd.Series, timestamps: pd.Series) -> dict:
    """
    Detect regime transitions (change points) and compute statistics.
    A transition = consecutive bars with different regime labels.
    """
    valid = regime >= 0
    reg_valid = regime[valid].values
    ts_valid = timestamps[valid].values

    # Find transitions
    transitions = np.where(reg_valid[1:] != reg_valid[:-1])[0] + 1
    n_transitions = len(transitions)

    # Segment durations (in bars)
    seg_starts = np.concatenate([[0], transitions])
    seg_ends = np.concatenate([transitions, [len(reg_valid)]])
    seg_lengths = seg_ends - seg_starts
    seg_regimes = reg_valid[seg_starts]

    # Per-regime duration stats
    duration_stats = {}
    for r in [0, 1, 2]:
        mask = seg_regimes == r
        if mask.any():
            lens = seg_lengths[mask]
            duration_stats[REGIME_NAMES[r]] = {
                "n_segments": int(mask.sum()),
                "mean_duration_bars": round(float(np.mean(lens)), 1),
                "median_duration_bars": round(float(np.median(lens)), 1),
                "mean_duration_hours": round(float(np.mean(lens)) * 5 / 60, 1),
            }

    # Transitions per year
    if len(ts_valid) > 0:
        span_days = (ts_valid[-1] - ts_valid[0]) / np.timedelta64(1, "D")
        span_years = span_days / 365.25
        trans_per_year = n_transitions / span_years if span_years > 0 else 0
    else:
        trans_per_year = 0

    return {
        "n_transitions": n_transitions,
        "transitions_per_year": round(trans_per_year, 1),
        "n_segments": len(seg_starts),
        "duration_stats": duration_stats,
    }


# ── 4. Variance ratio diagnostic ───────────────────────────────────────────
def compute_variance_ratio(close: pd.Series) -> pd.Series:
    """
    Short-term / long-term variance ratio for regime shift detection.
    Ratio > 2.0 signals transition toward high-vol.
    Ratio < 0.5 signals transition toward low-vol.
    """
    log_ret = np.log(close / close.shift(1))
    var_short = log_ret.rolling(72, min_periods=36).var()    # 6 hours
    var_long = log_ret.rolling(576, min_periods=288).var()   # 2 days
    ratio = var_short / var_long
    return ratio


# ── 5. Config B per-regime analysis ─────────────────────────────────────────
def regime_conditional_analysis(
    df: pd.DataFrame, regime: pd.Series, cfg: Config
) -> dict:
    """
    Re-run Config B filters, then split by regime and compute WR/EV/PF.
    """
    long_mask, short_mask = ict_filters(df, cfg)
    r_win = cfg.r_target - cfg.cost_per_r
    r_loss = -(1 + cfg.cost_per_r)

    results = {}
    for r_val in [-1, 0, 1, 2]:
        r_name = REGIME_NAMES[r_val]
        in_regime = regime == r_val

        # Total bars in regime
        n_bars = int(in_regime.sum())

        # Signals in this regime
        long_in = long_mask & in_regime
        short_in = short_mask & in_regime
        n_long = int(long_in.sum())
        n_short = int(short_in.sum())
        n_total = n_long + n_short

        if n_total == 0:
            results[r_name] = {
                "n_bars": n_bars, "n_signals": 0,
                "n_long": 0, "n_short": 0,
                "win_rate": 0, "ev_r": 0, "profit_factor": 0,
            }
            continue

        # Compute wins/losses
        long_wins = int(df.loc[long_in, cfg.long_label].sum()) if n_long > 0 else 0
        short_wins = int(df.loc[short_in, cfg.short_label].sum()) if n_short > 0 else 0
        total_wins = long_wins + short_wins
        total_losses = n_total - total_wins

        wr = total_wins / n_total
        ev = wr * r_win + (1 - wr) * r_loss
        gross_win = total_wins * cfg.r_target
        gross_loss = total_losses * 1.0
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

        # Per-direction breakdown
        long_wr = long_wins / n_long if n_long > 0 else 0
        short_wr = short_wins / n_short if n_short > 0 else 0

        results[r_name] = {
            "n_bars": n_bars,
            "pct_bars": round(n_bars / len(df) * 100, 1),
            "n_signals": n_total,
            "n_long": n_long,
            "n_short": n_short,
            "wins": total_wins,
            "losses": total_losses,
            "win_rate": round(wr, 4),
            "ev_r": round(ev, 4),
            "profit_factor": round(pf, 4),
            "long_wr": round(long_wr, 4),
            "short_wr": round(short_wr, 4),
        }

    return results


# ── Plotly chart ────────────────────────────────────────────────────────────
def save_chart(
    timestamps: pd.Series,
    rvol: pd.Series,
    regime: pd.Series,
    regime_results: dict,
) -> Path | None:
    """Save regime visualization: rvol time series + per-regime WR bar chart."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return None

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            "Realized Volatility with Regime Classification",
            "Config B Win Rate by Regime",
        ],
        row_heights=[0.65, 0.35],
        vertical_spacing=0.12,
    )

    # --- Row 1: rvol time series colored by regime ---
    # Downsample to every 288 bars (~daily) for performance
    step = 288
    idx = np.arange(0, len(timestamps), step)
    ts_ds = timestamps.iloc[idx]
    rvol_ds = rvol.iloc[idx]
    reg_ds = regime.iloc[idx]

    colors_map = {-1: "gray", 0: "dodgerblue", 1: "limegreen", 2: "orangered"}
    color_arr = [colors_map.get(int(r), "gray") for r in reg_ds.values]

    fig.add_trace(
        go.Scatter(
            x=ts_ds, y=rvol_ds,
            mode="markers",
            marker=dict(color=color_arr, size=2, opacity=0.7),
            name="RVol (daily sample)",
            showlegend=False,
        ),
        row=1, col=1,
    )

    # Add regime legend entries
    for r_val, r_name, r_color in [
        (0, "LOW vol", "dodgerblue"),
        (1, "MED vol", "limegreen"),
        (2, "HIGH vol", "orangered"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(color=r_color, size=8),
                name=r_name,
            ),
            row=1, col=1,
        )

    # --- Row 2: WR bar chart per regime ---
    regime_order = ["LOW", "MED", "HIGH"]
    wrs = []
    evs = []
    ns = []
    bar_colors = ["dodgerblue", "limegreen", "orangered"]
    for r_name in regime_order:
        data = regime_results.get(r_name, {})
        wrs.append(data.get("win_rate", 0) * 100)
        evs.append(data.get("ev_r", 0))
        ns.append(data.get("n_signals", 0))

    fig.add_trace(
        go.Bar(
            x=regime_order, y=wrs,
            marker_color=bar_colors,
            text=[f"WR {wr:.1f}%<br>EV {ev:+.3f}R<br>n={n}"
                  for wr, ev, n in zip(wrs, evs, ns)],
            textposition="outside",
            name="Win Rate %",
            showlegend=False,
        ),
        row=2, col=1,
    )
    # Break-even line at 33.33%
    fig.add_hline(
        y=33.33, line_dash="dash", line_color="yellow",
        annotation_text="BE (33.3%)", row=2, col=1,
    )

    fig.update_layout(
        template="plotly_dark",
        title="Regime Classification -- Config B Edge by Volatility Regime",
        height=800,
    )
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Realized Vol (288-bar)", row=1, col=1)
    fig.update_xaxes(title_text="Regime", row=2, col=1)
    fig.update_yaxes(title_text="Win Rate %", row=2, col=1)

    path = RESULTS_DIR / "regime_classification.html"
    fig.write_html(str(path))
    return path


# ── Console report ──────────────────────────────────────────────────────────
def print_report(
    df: pd.DataFrame,
    rvol: pd.Series,
    regime: pd.Series,
    regime_results: dict,
    transition_stats: dict,
    rcfg: RegimeConfig,
) -> dict:
    """Print regime analysis report and save outputs."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    sep = "=" * 70
    rule = "-" * 70

    print(f"\n{sep}")
    print("  REGIME CLASSIFIER -- Volatility Regime Analysis (Step 3)")
    print(sep)
    print(f"  RVol window: {rcfg.rvol_window} bars (1 day)")
    print(f"  Regime lookback: {rcfg.regime_lookback} bars (1 month)")
    print(f"  Thresholds: low < P{rcfg.low_pctile:.0%}, "
          f"high >= P{rcfg.high_pctile:.0%}")

    # --- regime distribution ---
    print(f"\n{rule}")
    print("  REGIME DISTRIBUTION")
    print(rule)
    total = len(df)
    for r_val in [-1, 0, 1, 2]:
        r_name = REGIME_NAMES[r_val]
        n = int((regime == r_val).sum())
        pct = n / total * 100
        rvol_subset = rvol[regime == r_val]
        if len(rvol_subset) > 0 and rvol_subset.notna().any():
            mean_rv = rvol_subset.mean()
            print(f"  {r_name:<8}  {n:>8,} bars  ({pct:>5.1f}%)  "
                  f"mean rvol={mean_rv:.6f}")
        else:
            print(f"  {r_name:<8}  {n:>8,} bars  ({pct:>5.1f}%)")

    # --- transitions ---
    print(f"\n{rule}")
    print("  REGIME TRANSITIONS (Change Points)")
    print(rule)
    print(f"  Total transitions:     {transition_stats['n_transitions']:,}")
    print(f"  Transitions/year:      {transition_stats['transitions_per_year']:.1f}")
    print(f"  Total segments:        {transition_stats['n_segments']:,}")
    for r_name, dstats in transition_stats.get("duration_stats", {}).items():
        print(f"  {r_name:<6} segments: {dstats['n_segments']:>5}  "
              f"mean={dstats['mean_duration_bars']:.0f} bars "
              f"({dstats['mean_duration_hours']:.1f}h)  "
              f"median={dstats['median_duration_bars']:.0f} bars")

    # --- Config B per regime ---
    print(f"\n{rule}")
    print("  CONFIG B RESULTS BY REGIME")
    print(rule)
    print(f"  {'Regime':<8}  {'Bars':>8}  {'Signals':>8}  {'Long':>5}  {'Short':>6}  "
          f"{'WR':>7}  {'EV(R)':>8}  {'PF':>6}")
    print(f"  {'--------':<8}  {'--------':>8}  {'--------':>8}  {'-----':>5}  {'------':>6}  "
          f"{'-------':>7}  {'--------':>8}  {'------':>6}")

    for r_name in ["LOW", "MED", "HIGH", "warmup"]:
        data = regime_results.get(r_name, {})
        n_sig = data.get("n_signals", 0)
        if n_sig == 0 and r_name == "warmup":
            continue
        wr = data.get("win_rate", 0)
        ev = data.get("ev_r", 0)
        pf = data.get("profit_factor", 0)
        print(f"  {r_name:<8}  {data.get('n_bars', 0):>8,}  {n_sig:>8}  "
              f"{data.get('n_long', 0):>5}  {data.get('n_short', 0):>6}  "
              f"{wr:>6.2%}  {ev:>+8.4f}  {pf:>6.2f}")

    # --- per-direction within regime ---
    print(f"\n  Per-direction breakdown:")
    for r_name in ["LOW", "MED", "HIGH"]:
        data = regime_results.get(r_name, {})
        if data.get("n_signals", 0) == 0:
            continue
        l_wr = data.get("long_wr", 0)
        s_wr = data.get("short_wr", 0)
        print(f"    {r_name:<6}  Long WR: {l_wr:.2%} (n={data.get('n_long', 0)})  "
              f"Short WR: {s_wr:.2%} (n={data.get('n_short', 0)})")

    # --- hypothesis test ---
    print(f"\n{rule}")
    print("  HYPOTHESIS TEST")
    print(rule)
    med = regime_results.get("MED", {})
    low = regime_results.get("LOW", {})
    high = regime_results.get("HIGH", {})

    med_ev = med.get("ev_r", 0)
    low_ev = low.get("ev_r", 0)
    high_ev = high.get("ev_r", 0)
    med_n = med.get("n_signals", 0)
    low_n = low.get("n_signals", 0)

    h1 = med_ev > low_ev and med_ev > 0
    h2 = low_ev <= 0

    print(f"  H1: MED regime has positive EV?    "
          f"{'CONFIRMED' if h1 else 'REJECTED'}  "
          f"(EV={med_ev:+.4f}, n={med_n})")
    print(f"  H2: LOW regime has zero/neg EV?    "
          f"{'CONFIRMED' if h2 else 'REJECTED'}  "
          f"(EV={low_ev:+.4f}, n={low_n})")
    print(f"  HIGH regime:                        "
          f"EV={high_ev:+.4f}, n={high.get('n_signals', 0)}")

    if h1 and h2:
        print(f"\n  >>> CONFIRMED: edge is regime-dependent.")
        print(f"      MED regime is where Config B edge lives.")
        if med_n > 0:
            # What if we only traded MED regime?
            print(f"      MED-only strategy: {med_n} signals, "
                  f"WR {med.get('win_rate', 0):.2%}, "
                  f"PF {med.get('profit_factor', 0):.2f}")
    else:
        print(f"\n  >>> Hypothesis NOT fully confirmed. "
              f"Regime effect may be weaker than expected.")

    print(sep)

    # --- save chart ---
    chart_path = save_chart(
        df["bar_start_ts_utc"], rvol, regime, regime_results
    )
    if chart_path:
        print(f"\n  Saved: {chart_path}")
    else:
        print("\n  (plotly not installed -- skipping chart)")

    # --- save JSON ---
    summary = {
        "config": {
            "rvol_window": rcfg.rvol_window,
            "regime_lookback": rcfg.regime_lookback,
            "min_warmup": rcfg.min_warmup,
            "low_pctile": rcfg.low_pctile,
            "high_pctile": rcfg.high_pctile,
        },
        "regime_distribution": {
            REGIME_NAMES[r]: int((regime == r).sum())
            for r in [-1, 0, 1, 2]
        },
        "transitions": transition_stats,
        "config_b_by_regime": regime_results,
        "hypothesis": {
            "h1_med_positive_ev": h1,
            "h2_low_negative_ev": h2,
            "confirmed": h1 and h2,
        },
    }
    json_path = RESULTS_DIR / "regime_analysis.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved: {json_path}")

    return summary


# ── main ────────────────────────────────────────────────────────────────────
def main() -> None:
    cfg = Config()
    rcfg = RegimeConfig()

    # 1. Load full parquet (no date filter yet -- need regime for all rows)
    print(f"Loading {DATA_PATH.name} ...")
    df_full = pd.read_parquet(DATA_PATH)
    print(f"  Full shape: {df_full.shape[0]:,} rows x {df_full.shape[1]} cols")

    # 2. Compute realized vol
    print("  Computing realized volatility...")
    rvol = compute_realized_vol(df_full["close"], rcfg.rvol_window)
    rvol_valid = rvol.notna().sum()
    print(f"  RVol computed: {rvol_valid:,} valid values "
          f"(mean={rvol[rvol.notna()].mean():.6f})")

    # 3. Assign regime labels
    print("  Assigning regime labels...")
    regime, pctile = assign_regime_labels(rvol, rcfg)
    for r_val in [-1, 0, 1, 2]:
        n = int((regime == r_val).sum())
        print(f"    {REGIME_NAMES[r_val]:<8}: {n:>8,} bars "
              f"({n / len(df_full) * 100:.1f}%)")

    # 4. Save regime column to labeled parquet
    print("\n  Adding regime_label column to labeled parquet...")
    col_existed = "regime_label" in df_full.columns
    df_full["regime_label"] = regime.values
    df_full.to_parquet(DATA_PATH, index=False)
    action = "Updated" if col_existed else "Added"
    print(f"  {action} regime_label -> {DATA_PATH.name} "
          f"(now {df_full.shape[1]} cols)")

    # 5. Apply date filter + drop NaN labels for analysis
    df = df_full[
        df_full["bar_start_ts_utc"] >= pd.Timestamp(rcfg.min_date, tz="UTC")
    ].copy()
    before = len(df)
    df = df.dropna(subset=[cfg.long_label, cfg.short_label])
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} rows with NaN labels")
    print(f"  Analysis subset: {len(df):,} rows")

    # Recompute rvol/regime for the filtered subset (use same index)
    rvol_filtered = rvol.loc[df.index]
    regime_filtered = regime.loc[df.index]

    # 6. Transition analysis
    print("\n  Analyzing regime transitions...")
    transition_stats = analyze_transitions(
        regime_filtered, df["bar_start_ts_utc"]
    )

    # 7. Config B per-regime analysis
    print("  Running Config B per regime...")
    regime_results = regime_conditional_analysis(df, regime_filtered, cfg)

    # 8. Report
    print_report(
        df, rvol_filtered, regime_filtered,
        regime_results, transition_stats, rcfg,
    )


if __name__ == "__main__":
    main()
