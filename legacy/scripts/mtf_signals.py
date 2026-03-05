"""
mtf_signals.py -- Multi-Timeframe Signal Expansion (D24)
========================================================
Runs Config B filter logic on three timeframes simultaneously to solve
the signal frequency problem (116 signals/6yr is not tradeable).

Timeframes:
  H4:  h4_ict_fvg  + h4_ict_sweep  (current Config B -- 116 signals)
  H1:  h1_ict_fvg  + h1_ict_sweep  (same routing / ATR / day filters)
  M15: m15_ict_fvg + m15_ict_sweep (same routing / ATR / day filters)

Shared filters (applied at 5m base timeframe):
  - Session routing: longs = London SB, shorts = NY PM SB
  - ATR ratio: ict_atr_ratio in [0.8, 1.5]
  - Day-of-week: exclude Mon/Tue
  - Short trend: h4_ict_market_trend == -1 (H4 level, all TFs)

Target: 300+ signals with WR >= 38% and EV >= +0.10R

Output:
  - Console report: per-TF and combined metrics
  - results/mtf_signals.json
  - results/mtf_signals.html  (per-TF comparison chart)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_backtest_v2 import Config, load_labeled

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"


# ── timeframe config ────────────────────────────────────────────────────────
@dataclass
class TFConfig:
    """Column mapping for one timeframe's Config B variant."""
    name: str
    fvg_bull: str
    fvg_bear: str
    sweep_bull: str
    sweep_bear: str
    bear_trend: str        # short-side trend gate
    atr_ratio: str         # which ATR ratio column to use


TF_CONFIGS = [
    TFConfig(
        name="H4",
        fvg_bull="h4_ict_fvg_bull",
        fvg_bear="h4_ict_fvg_bear",
        sweep_bull="h4_ict_bull_liq_sweep",
        sweep_bear="h4_ict_bear_liq_sweep",
        bear_trend="h4_ict_market_trend",
        atr_ratio="ict_atr_ratio",          # 5m base ATR for all
    ),
    TFConfig(
        name="H1",
        fvg_bull="h1_ict_fvg_bull",
        fvg_bear="h1_ict_fvg_bear",
        sweep_bull="h1_ict_bull_liq_sweep",
        sweep_bear="h1_ict_bear_liq_sweep",
        bear_trend="h4_ict_market_trend",   # H4 trend for all TFs
        atr_ratio="ict_atr_ratio",
    ),
    TFConfig(
        name="M15",
        fvg_bull="m15_ict_fvg_bull",
        fvg_bear="m15_ict_fvg_bear",
        sweep_bull="m15_ict_bull_liq_sweep",
        sweep_bear="m15_ict_bear_liq_sweep",
        bear_trend="h4_ict_market_trend",   # H4 trend for all TFs
        atr_ratio="ict_atr_ratio",
    ),
]


# ── filter logic ────────────────────────────────────────────────────────────
def config_b_filters(
    df: pd.DataFrame, tf: TFConfig, cfg: Config
) -> tuple[pd.Series, pd.Series]:
    """
    Apply Config B filter logic using the given timeframe's columns.
    Session routing, ATR band, and day-of-week are always at the 5m base level.
    """
    # Shared base-level filters
    atr_ok = (
        (df[tf.atr_ratio] >= cfg.atr_ratio_min)
        & (df[tf.atr_ratio] <= cfg.atr_ratio_max)
    )
    day_ok = ~df["ict_day_of_week"].isin([0, 1])

    # --- LONG ---
    long_mask = (
        (df[tf.fvg_bull] == 1)
        & (df[tf.sweep_bull] == 1)
        & (df["sess_sb_london"] == 1)
        & atr_ok
        & day_ok
    )

    # --- SHORT ---
    short_mask = (
        (df[tf.fvg_bear] == 1)
        & (df[tf.sweep_bear] == 1)
        & (df[tf.bear_trend] == -1)
        & (df["sess_sb_ny_pm"] == 1)
        & atr_ok
        & day_ok
    )

    return long_mask, short_mask


# ── analysis per timeframe ──────────────────────────────────────────────────
def _pf(wins: int, losses: int, r_target: int) -> float:
    gross_win = wins * r_target
    gross_loss = losses * 1.0
    return gross_win / gross_loss if gross_loss > 0 else float("inf")


def analyze_signals(
    df: pd.DataFrame,
    long_mask: pd.Series,
    short_mask: pd.Series,
    cfg: Config,
    tf_name: str,
) -> dict:
    """Compute WR, EV, PF for one timeframe's signals."""
    r_win = cfg.r_target - cfg.cost_per_r
    r_loss = -(1 + cfg.cost_per_r)

    results = {}
    for direction, mask, label_col in [
        ("long", long_mask, cfg.long_label),
        ("short", short_mask, cfg.short_label),
    ]:
        n = int(mask.sum())
        if n == 0:
            results[direction] = {"n": 0, "wins": 0, "wr": 0, "ev_r": 0, "pf": 0}
            continue
        wins = int(df.loc[mask, label_col].sum())
        losses = n - wins
        wr = wins / n
        ev = wr * r_win + (1 - wr) * r_loss
        pf = _pf(wins, losses, cfg.r_target)
        results[direction] = {
            "n": n, "wins": wins, "losses": losses,
            "wr": round(wr, 4), "ev_r": round(ev, 4), "pf": round(pf, 4),
        }

    # Combined
    n_total = results["long"]["n"] + results["short"]["n"]
    wins_total = results["long"]["wins"] + results["short"]["wins"]
    if n_total > 0:
        wr_total = wins_total / n_total
        ev_total = wr_total * r_win + (1 - wr_total) * r_loss
        pf_total = _pf(wins_total, n_total - wins_total, cfg.r_target)
    else:
        wr_total = ev_total = pf_total = 0

    # Random baseline (unconditional)
    rand_long_wr = df[cfg.long_label].mean()
    rand_short_wr = df[cfg.short_label].mean()
    rand_avg_wr = (rand_long_wr + rand_short_wr) / 2

    return {
        "tf": tf_name,
        "long": results["long"],
        "short": results["short"],
        "combined": {
            "n": n_total, "wins": wins_total, "losses": n_total - wins_total,
            "wr": round(wr_total, 4), "ev_r": round(ev_total, 4),
            "pf": round(pf_total, 4),
        },
        "edge_pp": round((wr_total - rand_avg_wr) * 100, 2),
        "random_wr": round(rand_avg_wr, 4),
    }


# ── overlap analysis ────────────────────────────────────────────────────────
def compute_overlap(masks: dict[str, pd.Series]) -> dict:
    """Compute pairwise overlap counts between timeframe signal masks."""
    names = list(masks.keys())
    overlap = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            both = int((masks[a] & masks[b]).sum())
            overlap[f"{a} & {b}"] = both
    all_three = int(
        masks[names[0]] & masks[names[1]] & masks[names[2]]
    ).sum() if len(names) == 3 else 0
    # compute via reduce for safety
    combined = masks[names[0]]
    for n in names[1:]:
        combined = combined | masks[n]
    return {
        "pairwise": overlap,
        "all_three": int((masks[names[0]] & masks[names[1]] & masks[names[2]]).sum()),
        "union_total": int(combined.sum()),
    }


# ── chart ───────────────────────────────────────────────────────────────────
def save_chart(tf_results: list[dict], combined_result: dict) -> Path | None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return None

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Signal Count", "Win Rate %", "EV (R)"],
        horizontal_spacing=0.08,
    )

    names = [r["tf"] for r in tf_results] + ["COMBINED"]
    counts = [r["combined"]["n"] for r in tf_results] + [combined_result["combined"]["n"]]
    wrs = [r["combined"]["wr"] * 100 for r in tf_results] + [combined_result["combined"]["wr"] * 100]
    evs = [r["combined"]["ev_r"] for r in tf_results] + [combined_result["combined"]["ev_r"]]
    colors = ["#4a86c8", "#5cb85c", "#f0ad4e", "#d9534f"]

    fig.add_trace(go.Bar(x=names, y=counts, marker_color=colors,
                         text=counts, textposition="outside",
                         showlegend=False), row=1, col=1)

    fig.add_trace(go.Bar(x=names, y=wrs, marker_color=colors,
                         text=[f"{w:.1f}%" for w in wrs], textposition="outside",
                         showlegend=False), row=1, col=2)
    fig.add_hline(y=33.33, line_dash="dash", line_color="yellow",
                  annotation_text="BE", row=1, col=2)

    ev_colors = ["green" if e > 0 else "red" for e in evs]
    fig.add_trace(go.Bar(x=names, y=evs, marker_color=ev_colors,
                         text=[f"{e:+.3f}" for e in evs], textposition="outside",
                         showlegend=False), row=1, col=3)
    fig.add_hline(y=0, line_dash="dash", line_color="yellow", row=1, col=3)

    fig.update_layout(
        template="plotly_dark",
        title="Multi-Timeframe Signal Expansion -- Config B Logic",
        height=450,
    )
    fig.update_yaxes(title_text="Signals", row=1, col=1)
    fig.update_yaxes(title_text="Win Rate %", row=1, col=2)
    fig.update_yaxes(title_text="EV (R)", row=1, col=3)

    path = RESULTS_DIR / "mtf_signals.html"
    fig.write_html(str(path))
    return path


# ── console report ──────────────────────────────────────────────────────────
def print_report(
    tf_results: list[dict],
    combined_result: dict,
    overlap: dict,
    cfg: Config,
) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    sep = "=" * 70
    rule = "-" * 70

    print(f"\n{sep}")
    print("  MULTI-TIMEFRAME SIGNAL EXPANSION (D24)")
    print(sep)
    print(f"  Config B logic applied to H4, H1, M15 simultaneously")
    print(f"  Label: {cfg.r_target}R / 1R / {cfg.horizon}-bar horizon")
    print(f"  Shared: London SB longs, NY PM SB shorts, "
          f"ATR [{cfg.atr_ratio_min},{cfg.atr_ratio_max}], no Mon/Tue")
    print(f"  Target: 300+ signals, WR >= 38%, EV >= +0.10R")

    # --- per-TF table ---
    print(f"\n{rule}")
    print("  PER-TIMEFRAME RESULTS")
    print(rule)
    print(f"  {'TF':<6}  {'Signals':>8}  {'Long':>6}  {'Short':>6}  "
          f"{'WR':>7}  {'EV(R)':>8}  {'PF':>6}  {'Edge':>7}")
    print(f"  {'------':<6}  {'--------':>8}  {'------':>6}  {'------':>6}  "
          f"{'-------':>7}  {'--------':>8}  {'------':>6}  {'-------':>7}")

    for r in tf_results:
        c = r["combined"]
        print(f"  {r['tf']:<6}  {c['n']:>8}  "
              f"{r['long']['n']:>6}  {r['short']['n']:>6}  "
              f"{c['wr']:>6.2%}  {c['ev_r']:>+8.4f}  {c['pf']:>6.2f}  "
              f"{r['edge_pp']:>+6.2f}pp")

    # Combined row
    cc = combined_result["combined"]
    print(f"  {'------':<6}  {'--------':>8}  {'------':>6}  {'------':>6}  "
          f"{'-------':>7}  {'--------':>8}  {'------':>6}  {'-------':>7}")
    print(f"  {'COMB':<6}  {cc['n']:>8}  "
          f"{combined_result['long']['n']:>6}  {combined_result['short']['n']:>6}  "
          f"{cc['wr']:>6.2%}  {cc['ev_r']:>+8.4f}  {cc['pf']:>6.2f}  "
          f"{combined_result['edge_pp']:>+6.2f}pp")

    # --- per-direction within each TF ---
    print(f"\n{rule}")
    print("  PER-DIRECTION BREAKDOWN")
    print(rule)
    for r in tf_results:
        l = r["long"]
        s = r["short"]
        print(f"  {r['tf']:<6}  Long: n={l['n']:<5} WR={l['wr']:.2%} EV={l['ev_r']:+.3f}  "
              f"Short: n={s['n']:<5} WR={s['wr']:.2%} EV={s['ev_r']:+.3f}")

    # --- overlap ---
    print(f"\n{rule}")
    print("  SIGNAL OVERLAP (same 5m bar fires on multiple TFs)")
    print(rule)
    for pair, count in overlap["pairwise"].items():
        print(f"  {pair:<15}  {count:>5} bars")
    print(f"  {'All three':<15}  {overlap['all_three']:>5} bars")
    print(f"  Union (deduped): {overlap['union_total']:>5} bars")

    sum_raw = sum(r["combined"]["n"] for r in tf_results)
    dedup = overlap["union_total"]
    print(f"  Raw sum:         {sum_raw:>5}")
    print(f"  Overlap rate:    {(1 - dedup / sum_raw) * 100:.1f}%" if sum_raw > 0 else "")

    # --- signals per year ---
    print(f"\n{rule}")
    print("  SIGNAL FREQUENCY")
    print(rule)
    years = 6.16  # 2020-01 to 2026-02
    for r in tf_results:
        per_yr = r["combined"]["n"] / years
        print(f"  {r['tf']:<6}  {r['combined']['n']:>5} total  -> {per_yr:>5.1f} / year")
    per_yr_comb = cc["n"] / years
    print(f"  {'COMB':<6}  {cc['n']:>5} total  -> {per_yr_comb:>5.1f} / year")

    # --- GO/NO-GO vs targets ---
    print(f"\n{rule}")
    print("  TARGET CHECK")
    print(rule)
    n_pass = cc["n"] >= 300
    wr_pass = cc["wr"] >= 0.38
    ev_pass = cc["ev_r"] >= 0.10
    checks = [
        ("Signals >= 300", n_pass, f"{cc['n']}"),
        ("WR >= 38%", wr_pass, f"{cc['wr']:.2%}"),
        ("EV >= +0.10R", ev_pass, f"{cc['ev_r']:+.4f}"),
    ]
    for label, passed, val in checks:
        status = "PASS" if passed else "MISS"
        print(f"  [{status}]  {label:<25}  {val}")

    all_pass = n_pass and wr_pass and ev_pass
    if all_pass:
        print(f"\n  >>> ALL TARGETS MET")
    else:
        print(f"\n  >>> Some targets missed -- see breakdown above")
    print(sep)

    # --- save chart ---
    chart_path = save_chart(tf_results, combined_result)
    if chart_path:
        print(f"\n  Saved: {chart_path}")

    # --- save JSON ---
    summary = {
        "per_timeframe": {r["tf"]: r for r in tf_results},
        "combined_deduped": combined_result,
        "overlap": overlap,
        "signals_per_year": round(per_yr_comb, 1),
        "targets": {
            "n_pass": n_pass, "wr_pass": wr_pass, "ev_pass": ev_pass,
            "all_pass": all_pass,
        },
    }
    json_path = RESULTS_DIR / "mtf_signals.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved: {json_path}")

    return summary


# ── main ────────────────────────────────────────────────────────────────────
def main() -> None:
    cfg = Config()
    df = load_labeled(cfg)

    # Run Config B logic on each timeframe
    tf_results = []
    all_long_masks = {}
    all_short_masks = {}

    for tf in TF_CONFIGS:
        long_mask, short_mask = config_b_filters(df, tf, cfg)
        result = analyze_signals(df, long_mask, short_mask, cfg, tf.name)
        tf_results.append(result)
        all_long_masks[tf.name] = long_mask
        all_short_masks[tf.name] = short_mask
        print(f"  {tf.name}: {result['combined']['n']:>5} signals  "
              f"(L={result['long']['n']}, S={result['short']['n']})  "
              f"WR={result['combined']['wr']:.2%}")

    # Combined (union) masks -- deduplicated
    combined_long = all_long_masks["H4"].copy()
    combined_short = all_short_masks["H4"].copy()
    for name in ["H1", "M15"]:
        combined_long = combined_long | all_long_masks[name]
        combined_short = combined_short | all_short_masks[name]

    combined_result = analyze_signals(
        df, combined_long, combined_short, cfg, "COMBINED"
    )

    # Overlap analysis (on combined long|short per TF)
    any_signal = {}
    for name in ["H4", "H1", "M15"]:
        any_signal[name] = all_long_masks[name] | all_short_masks[name]
    overlap = {
        "pairwise": {},
        "all_three": int((any_signal["H4"] & any_signal["H1"] & any_signal["M15"]).sum()),
        "union_total": int((any_signal["H4"] | any_signal["H1"] | any_signal["M15"]).sum()),
    }
    for i, a in enumerate(["H4", "H1", "M15"]):
        for b in ["H4", "H1", "M15"][i + 1:]:
            overlap["pairwise"][f"{a} & {b}"] = int((any_signal[a] & any_signal[b]).sum())

    # Report
    print_report(tf_results, combined_result, overlap, cfg)


if __name__ == "__main__":
    main()
