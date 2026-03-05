"""FVG Edge Validation - Raw Statistical Analysis

Tests whether Fair Value Gaps have raw predictive edge before any ML.
Pure pandas + numpy + scipy. No Foundation imports, no LightGBM.

Run from BTCDataset_v2/:
    python scripts/fvg_edge_validation.py
"""
from __future__ import annotations

import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

# === CONFIG ===
DATA_PATH = Path("data/resampled/BTCUSDT_perp_5m.parquet")
OUTPUT_PATH = Path("outputs/FVG_EDGE_VALIDATION.md")
HORIZONS = {"5m": 1, "30m": 6, "1h": 12, "4h": 48, "8h": 96, "24h": 288}
FILL_WINDOW = 288  # 24h in 5m bars
BOOTSTRAP_ITERS = 1000
TF_MINUTES = {"5m": 5, "1H": 60, "4H": 240}

np.random.seed(42)


# ------------------------------------------------------------------ #
#  Step 1: Load data                                                  #
# ------------------------------------------------------------------ #
def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df = df[["bar_start_ts_utc", "open", "high", "low", "close", "volume_base"]].copy()
    df.rename(columns={"volume_base": "volume"}, inplace=True)
    df["bar_start_ts_utc"] = pd.to_datetime(df["bar_start_ts_utc"], utc=True)
    df.set_index("bar_start_ts_utc", inplace=True)
    df.sort_index(inplace=True)
    print(f"  Rows: {len(df):,}")
    print(f"  Range: {df.index.min()} to {df.index.max()}")
    return df


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #
def resample_ohlcv(df_5m: pd.DataFrame, rule: str) -> pd.DataFrame:
    return df_5m.resample(rule).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ------------------------------------------------------------------ #
#  Step 2: Detect FVGs                                                #
# ------------------------------------------------------------------ #
def detect_fvgs(df: pd.DataFrame, atr: pd.Series) -> pd.DataFrame:
    """Bullish FVG: bar[i+1].low > bar[i-1].high
    Bearish FVG: bar[i+1].high < bar[i-1].low
    confirmed_at = timestamp of bar[i+1] (the confirmation bar)
    """
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    timestamps = df.index
    atr_v = atr.values
    volumes = df["volume"].values

    records = []
    for i in range(1, len(df) - 1):
        a = atr_v[i + 1]
        if np.isnan(a) or a <= 0:
            continue
        prev_h = highs[i - 1]
        prev_l = lows[i - 1]
        next_h = highs[i + 1]
        next_l = lows[i + 1]

        if next_l > prev_h:  # bullish
            gap = next_l - prev_h
            records.append({
                "confirmed_at": timestamps[i + 1],
                "direction": "bullish",
                "gap_size": gap,
                "gap_size_atr": gap / a,
                "gap_top": next_l,
                "gap_bottom": prev_h,
                "volume": volumes[i],
                "atr": a,
                "close_price": closes[i + 1],
            })
        if next_h < prev_l:  # bearish
            gap = prev_l - next_h
            records.append({
                "confirmed_at": timestamps[i + 1],
                "direction": "bearish",
                "gap_size": gap,
                "gap_size_atr": gap / a,
                "gap_top": prev_l,
                "gap_bottom": next_h,
                "volume": volumes[i],
                "atr": a,
                "close_price": closes[i + 1],
            })
    return pd.DataFrame(records)


def map_to_5m_index(fvg_df: pd.DataFrame, df_5m: pd.DataFrame, tf_min: int) -> pd.DataFrame:
    """Map confirmed_at to the first 5m bar AFTER the HTF bar closes.

    For 5m: entry = next bar (confirmed_at + 5m).
    For 1H: entry = first 5m bar of next hour (confirmed_at + 60m).
    For 4H: entry = first 5m bar of next 4H period (confirmed_at + 240m).
    """
    fvg_df = fvg_df.copy()
    offset = pd.Timedelta(minutes=tf_min)
    entry_times = fvg_df["confirmed_at"] + offset
    entry_idx = df_5m.index.searchsorted(entry_times)

    max_horizon = max(HORIZONS.values())
    fvg_df["entry_5m_idx"] = entry_idx
    mask = (fvg_df["entry_5m_idx"] + max_horizon) < len(df_5m)
    mask &= fvg_df["entry_5m_idx"] < len(df_5m)
    fvg_df = fvg_df[mask].copy()
    fvg_df["entry_5m_idx"] = fvg_df["entry_5m_idx"].astype(int)
    return fvg_df


# ------------------------------------------------------------------ #
#  Step 3: Forward returns                                            #
# ------------------------------------------------------------------ #
def compute_forward_returns(fvg_df: pd.DataFrame, df_5m: pd.DataFrame) -> pd.DataFrame:
    closes = df_5m["close"].values
    idx = fvg_df["entry_5m_idx"].values
    for name, bars in HORIZONS.items():
        entry = closes[idx]
        exit_ = closes[idx + bars]
        fvg_df[f"fwd_{name}"] = (exit_ - entry) / entry
    return fvg_df


def return_stats(fvg_df: pd.DataFrame, direction: str, horizon: str) -> dict | None:
    sub = fvg_df[fvg_df["direction"] == direction]
    if len(sub) < 5:
        return None
    rets = sub[f"fwd_{horizon}"].values

    if direction == "bullish":
        t, p2 = stats.ttest_1samp(rets, 0)
        p = p2 / 2 if t > 0 else 1 - p2 / 2
        wr = np.mean(rets > 0)
    else:
        t, p2 = stats.ttest_1samp(rets, 0)
        p = p2 / 2 if t < 0 else 1 - p2 / 2
        wr = np.mean(rets < 0)

    return {
        "mean_bps": np.mean(rets) * 1e4,
        "median_bps": np.median(rets) * 1e4,
        "std_bps": np.std(rets, ddof=1) * 1e4,
        "t": t, "p": p, "n": len(rets), "wr": wr,
    }


# ------------------------------------------------------------------ #
#  Step 4: Fill rate analysis                                         #
# ------------------------------------------------------------------ #
def fill_rate_analysis(fvg_df: pd.DataFrame, df_5m: pd.DataFrame) -> pd.DataFrame:
    highs = df_5m["high"].values
    lows = df_5m["low"].values
    closes = df_5m["close"].values
    n = len(df_5m)

    filled = np.zeros(len(fvg_df), dtype=bool)
    bars_to_fill = np.full(len(fvg_df), np.nan)
    max_fav = np.zeros(len(fvg_df))

    dirs = fvg_df["direction"].values
    idxs = fvg_df["entry_5m_idx"].values
    g_tops = fvg_df["gap_top"].values
    g_bots = fvg_df["gap_bottom"].values

    for i in range(len(fvg_df)):
        idx = idxs[i]
        ep = closes[idx]
        end = min(idx + FILL_WINDOW, n)
        wh = highs[idx:end]
        wl = lows[idx:end]

        if dirs[i] == "bullish":
            fav = np.maximum.accumulate((wh - ep) / ep)
            fm = wl <= g_bots[i]
        else:
            fav = np.maximum.accumulate((ep - wl) / ep)
            fm = wh >= g_tops[i]

        if fm.any():
            fi = int(np.argmax(fm))
            filled[i] = True
            bars_to_fill[i] = fi
            max_fav[i] = fav[fi - 1] if fi > 0 else 0.0
        else:
            max_fav[i] = fav[-1] if len(fav) > 0 else 0.0

    fvg_df = fvg_df.copy()
    fvg_df["filled"] = filled
    fvg_df["bars_to_fill"] = bars_to_fill
    fvg_df["max_favorable"] = max_fav
    return fvg_df


# ------------------------------------------------------------------ #
#  Step 5: Contextual analysis                                        #
# ------------------------------------------------------------------ #
def contextual_analysis(fvg_df: pd.DataFrame) -> dict:
    ctx = {}
    H = "fwd_4h"

    # A) Gap size
    med = fvg_df["gap_size_atr"].median()
    for d in ("bullish", "bearish"):
        sign = 1 if d == "bullish" else -1
        s = fvg_df[(fvg_df["direction"] == d) & (fvg_df["gap_size_atr"] <= med)]
        l = fvg_df[(fvg_df["direction"] == d) & (fvg_df["gap_size_atr"] > med)]
        ctx[f"gap_{d}"] = {
            "small_bps": float(s[H].mean() * sign * 1e4) if len(s) else np.nan,
            "large_bps": float(l[H].mean() * sign * 1e4) if len(l) else np.nan,
            "small_n": len(s), "large_n": len(l),
        }

    # B) Volume
    v75, v25 = fvg_df["volume"].quantile(0.75), fvg_df["volume"].quantile(0.25)
    for d in ("bullish", "bearish"):
        sign = 1 if d == "bullish" else -1
        hv = fvg_df[(fvg_df["direction"] == d) & (fvg_df["volume"] >= v75)]
        lv = fvg_df[(fvg_df["direction"] == d) & (fvg_df["volume"] <= v25)]
        ctx[f"vol_{d}"] = {
            "high_bps": float(hv[H].mean() * sign * 1e4) if len(hv) else np.nan,
            "low_bps": float(lv[H].mean() * sign * 1e4) if len(lv) else np.nan,
            "high_n": len(hv), "low_n": len(lv),
        }

    # C) Time periods (6-month windows)
    fvg_df = fvg_df.copy()
    fvg_df["half"] = fvg_df["confirmed_at"].apply(
        lambda x: f"{x.year}H{'1' if x.month <= 6 else '2'}"
    )
    time_rows = []
    for half, grp in fvg_df.groupby("half"):
        for d in ("bullish", "bearish"):
            sub = grp[grp["direction"] == d]
            if len(sub) < 10:
                continue
            rets = sub[H].values
            sign = 1 if d == "bullish" else -1
            aligned = rets * sign
            t_s, p2 = stats.ttest_1samp(aligned, 0)
            p1 = p2 / 2 if t_s > 0 else 1 - p2 / 2
            time_rows.append({
                "period": half, "direction": d,
                "aligned_bps": float(np.mean(aligned) * 1e4),
                "t": t_s, "p": p1, "n": len(sub),
            })
    ctx["time_df"] = pd.DataFrame(time_rows)

    # Trend regression
    for d in ("bullish", "bearish"):
        tp = ctx["time_df"][ctx["time_df"]["direction"] == d]
        if len(tp) > 3:
            sl, ic, rv, pv, se = stats.linregress(np.arange(len(tp)), tp["aligned_bps"].values)
            ctx[f"trend_{d}"] = {"slope": sl, "r2": rv**2, "p": pv,
                                  "label": "decaying" if sl < 0 else "strengthening"}

    # D) ATR regime
    a75, a25 = fvg_df["atr"].quantile(0.75), fvg_df["atr"].quantile(0.25)
    for d in ("bullish", "bearish"):
        sign = 1 if d == "bullish" else -1
        ha = fvg_df[(fvg_df["direction"] == d) & (fvg_df["atr"] >= a75)]
        la = fvg_df[(fvg_df["direction"] == d) & (fvg_df["atr"] <= a25)]
        ctx[f"atr_{d}"] = {
            "high_bps": float(ha[H].mean() * sign * 1e4) if len(ha) else np.nan,
            "low_bps": float(la[H].mean() * sign * 1e4) if len(la) else np.nan,
            "high_n": len(ha), "low_n": len(la),
        }

    return ctx


# ------------------------------------------------------------------ #
#  Step 6: Bootstrap comparison                                       #
# ------------------------------------------------------------------ #
def bootstrap_comparison(fvg_df: pd.DataFrame, df_5m: pd.DataFrame) -> dict:
    closes = df_5m["close"].values
    hb = HORIZONS["4h"]
    max_start = len(closes) - hb - 1
    results = {}

    for d in ("bullish", "bearish"):
        sub = fvg_df[fvg_df["direction"] == d]
        if len(sub) < 5:
            continue
        n = len(sub)
        fvg_mean = float(sub["fwd_4h"].mean())
        rand_means = np.empty(BOOTSTRAP_ITERS)
        for it in range(BOOTSTRAP_ITERS):
            ri = np.random.randint(14, max_start, size=n)
            rand_means[it] = np.mean((closes[ri + hb] - closes[ri]) / closes[ri])

        if d == "bullish":
            p = float(np.mean(rand_means >= fvg_mean))
        else:
            p = float(np.mean(rand_means <= fvg_mean))

        results[d] = {
            "fvg_bps": fvg_mean * 1e4,
            "rand_bps": float(rand_means.mean() * 1e4),
            "p": p, "n": n,
        }
    return results


# ------------------------------------------------------------------ #
#  Step 7: Report                                                     #
# ------------------------------------------------------------------ #
def generate_report(det_all: dict, ctx_all: dict, bs_all: dict,
                    df_5m: pd.DataFrame) -> str:
    L = []
    total_days = (df_5m.index.max() - df_5m.index.min()).days

    L.append("# FVG Edge Validation Results")
    L.append(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    L.append(f"Data: {len(df_5m):,} rows, "
             f"{df_5m.index.min().strftime('%Y-%m-%d')} to "
             f"{df_5m.index.max().strftime('%Y-%m-%d')}")
    L.append("")

    # --- Detection summary ---
    L.append("## FVG Detection Summary")
    L.append("")
    L.append("| Timeframe | Total | Bullish | Bearish | Per Day | Median Size (ATR) |")
    L.append("|-----------|-------|---------|---------|---------|-------------------|")
    for tf in ("5m", "1H", "4H"):
        f = det_all[tf]["fvgs"]
        bu = int((f["direction"] == "bullish").sum())
        be = int((f["direction"] == "bearish").sum())
        L.append(f"| {tf} | {len(f):,} | {bu:,} | {be:,} | "
                 f"{len(f)/total_days:.1f} | {f['gap_size_atr'].median():.3f} |")
    L.append("")

    # --- Forward returns ---
    L.append("## Forward Returns (mean basis points)")
    L.append("")
    L.append("| TF | Direction | 5m | 30m | 1h | 4h | 8h | 24h | t-stat (4h) | p-value |")
    L.append("|-----|-----------|------|------|------|------|------|-------|-------------|---------|")
    for tf in ("5m", "1H", "4H"):
        rs = det_all[tf]["rstats"]
        for d in ("bullish", "bearish"):
            cells = [tf, d]
            for h in HORIZONS:
                s = rs.get((d, h))
                cells.append(f"{s['mean_bps']:+.1f}" if s else "--")
            s4 = rs.get((d, "4h"))
            cells.append(f"{s4['t']:.2f}" if s4 else "--")
            cells.append(f"{s4['p']:.4f}" if s4 else "--")
            L.append("| " + " | ".join(cells) + " |")
    L.append("")

    # --- Win rates ---
    L.append("## Win Rates (% in hypothesized direction)")
    L.append("")
    L.append("| TF | Direction | 5m | 30m | 1h | 4h | 8h | 24h | N |")
    L.append("|-----|-----------|------|------|------|------|------|------|-------|")
    for tf in ("5m", "1H", "4H"):
        rs = det_all[tf]["rstats"]
        for d in ("bullish", "bearish"):
            cells = [tf, d]
            for h in HORIZONS:
                s = rs.get((d, h))
                cells.append(f"{s['wr']:.1%}" if s else "--")
            s4 = rs.get((d, "4h"))
            cells.append(str(s4["n"]) if s4 else "--")
            L.append("| " + " | ".join(cells) + " |")
    L.append("")

    # --- Fill rate ---
    L.append("## Fill Rate Analysis")
    L.append("")
    L.append("| TF | Direction | Fill Rate (24h) | Median Fill Time (bars) | Mean Max Favorable Before Fill (bps) |")
    L.append("|-----|-----------|-----------------|-------------------------|--------------------------------------|")
    for tf in ("5m", "1H", "4H"):
        f = det_all[tf]["fvgs"]
        for d in ("bullish", "bearish"):
            sub = f[f["direction"] == d]
            if len(sub) == 0:
                continue
            fr = sub["filled"].mean()
            fsub = sub[sub["filled"]]
            mf = fsub["bars_to_fill"].median() if len(fsub) else np.nan
            mfe = sub["max_favorable"].mean() * 1e4
            mf_str = f"{mf:.0f}" if not np.isnan(mf) else "--"
            L.append(f"| {tf} | {d} | {fr:.1%} | {mf_str} | {mfe:+.1f} |")
    L.append("")

    # --- Contextual splits (use 5m for largest sample) ---
    ctx = ctx_all.get("5m", ctx_all.get("1H", {}))
    L.append("## Contextual Splits")
    L.append("(Using 5m FVGs for largest sample size; 4h forward return aligned to direction)")
    L.append("")

    L.append("### By Gap Size")
    L.append("")
    L.append("| Direction | Small Gap (bps) | Large Gap (bps) | Small N | Large N |")
    L.append("|-----------|-----------------|-----------------|---------|---------|")
    for d in ("bullish", "bearish"):
        g = ctx.get(f"gap_{d}", {})
        L.append(f"| {d} | {g.get('small_bps', 0):+.1f} | {g.get('large_bps', 0):+.1f} | "
                 f"{g.get('small_n', 0)} | {g.get('large_n', 0)} |")
    L.append("")

    L.append("### By Volume")
    L.append("")
    L.append("| Direction | High Vol (bps) | Low Vol (bps) | High N | Low N |")
    L.append("|-----------|----------------|---------------|--------|-------|")
    for d in ("bullish", "bearish"):
        v = ctx.get(f"vol_{d}", {})
        L.append(f"| {d} | {v.get('high_bps', 0):+.1f} | {v.get('low_bps', 0):+.1f} | "
                 f"{v.get('high_n', 0)} | {v.get('low_n', 0)} |")
    L.append("")

    L.append("### By Time Period (6-month windows)")
    L.append("")
    tdf = ctx.get("time_df", pd.DataFrame())
    if len(tdf) > 0:
        L.append("| Period | Direction | Aligned Mean (bps) | t-stat | p-value | N |")
        L.append("|--------|-----------|-------------------|--------|---------|---|")
        for _, r in tdf.iterrows():
            L.append(f"| {r['period']} | {r['direction']} | {r['aligned_bps']:+.1f} | "
                     f"{r['t']:.2f} | {r['p']:.4f} | {r['n']} |")
    L.append("")

    L.append("#### Trend Analysis")
    for d in ("bullish", "bearish"):
        tr = ctx.get(f"trend_{d}")
        if tr:
            L.append(f"- **{d.title()}**: {tr['label']} "
                     f"(slope={tr['slope']:+.2f} bps/window, R2={tr['r2']:.3f}, p={tr['p']:.4f})")
    L.append("")

    L.append("### By ATR Regime")
    L.append("")
    L.append("| Direction | High ATR (bps) | Low ATR (bps) | High N | Low N |")
    L.append("|-----------|----------------|---------------|--------|-------|")
    for d in ("bullish", "bearish"):
        a = ctx.get(f"atr_{d}", {})
        L.append(f"| {d} | {a.get('high_bps', 0):+.1f} | {a.get('low_bps', 0):+.1f} | "
                 f"{a.get('high_n', 0)} | {a.get('low_n', 0)} |")
    L.append("")

    # --- Bootstrap ---
    L.append("## FVG vs Random Entry (4h horizon)")
    L.append("")
    L.append("| TF | Direction | FVG Mean (bps) | Random Mean (bps) | Bootstrap p-value |")
    L.append("|-----|-----------|----------------|-------------------|-------------------|")
    for tf in ("5m", "1H", "4H"):
        bs = bs_all.get(tf, {})
        for d in ("bullish", "bearish"):
            v = bs.get(d)
            if v:
                L.append(f"| {tf} | {d} | {v['fvg_bps']:+.1f} | {v['rand_bps']:+.1f} | {v['p']:.4f} |")
    L.append("")

    # --- VERDICT ---
    COST_BPS = 12.0  # ~0.12% round-trip

    L.append("## VERDICT")
    L.append("")

    verdicts = {}
    for tf in ("5m", "1H", "4H"):
        L.append(f"### {tf}")
        rs = det_all[tf]["rstats"]
        bs = bs_all.get(tf, {})
        for d in ("bullish", "bearish"):
            s4 = rs.get((d, "4h"))
            if not s4:
                L.append(f"- **{d.title()}**: Insufficient data")
                continue
            m = abs(s4["mean_bps"])
            bp = bs.get(d, {}).get("p", 1.0)
            if s4["p"] < 0.01 and bp < 0.05 and m > COST_BPS * 2:
                rating = "STRONG"
            elif s4["p"] < 0.05 and m > COST_BPS:
                rating = "MODERATE"
            elif s4["p"] < 0.10:
                rating = "WEAK"
            else:
                rating = "NONE"
            verdicts[(tf, d)] = rating
            L.append(f"- **{d.title()}**: **{rating}** "
                     f"(mean={s4['mean_bps']:+.1f} bps, t={s4['t']:.2f}, p={s4['p']:.4f}, "
                     f"WR={s4['wr']:.1%}, bootstrap_p={bp:.4f})")
        L.append("")

    # Overall
    L.append("### Overall Assessment")
    L.append("")
    for d in ("bullish", "bearish"):
        tr = ctx.get(f"trend_{d}")
        if tr:
            if tr["p"] < 0.05:
                L.append(f"- {d.title()} FVG edge is **{tr['label']}** over time (p={tr['p']:.4f})")
            else:
                L.append(f"- {d.title()} FVG edge: no significant trend over time (p={tr['p']:.4f})")
    L.append("")

    L.append("### Cost Analysis")
    L.append(f"- Round-trip cost estimate: ~{COST_BPS:.0f}-15 bps")
    L.append("- Edge must exceed this to be tradeable standalone")
    L.append("")

    # Recommendation
    any_strong = any(v == "STRONG" for v in verdicts.values())
    any_moderate = any(v == "MODERATE" for v in verdicts.values())
    L.append("### Recommendation for Foundation Feature Engine")
    L.append("")
    if any_strong:
        L.append("FVGs show statistically significant predictive edge at one or more timeframes.")
        L.append("**Recommendation:** Prioritize FVG features in Foundation's feature engine.")
    elif any_moderate:
        L.append("FVGs show some predictive signal but not consistently strong.")
        L.append("**Recommendation:** Include FVG features as one family among many, "
                 "but do not prioritize over TA/microstructure features.")
    else:
        L.append("FVGs show no reliable standalone predictive edge after costs.")
        L.append("**Recommendation:** Treat FVGs as supplementary features only. "
                 "They may contribute via interaction effects in ML models "
                 "but should not be a primary feature family.")
    L.append("")

    return "\n".join(L)


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #
def main():
    print("=" * 60)
    print("FVG EDGE VALIDATION - Raw Statistical Analysis")
    print("=" * 60)

    # Step 1
    print("\n--- Step 1: Load data ---")
    df_5m = load_data()

    # Step 2
    print("\n--- Step 2: Detect FVGs ---")
    timeframes = {
        "5m": df_5m,
        "1H": resample_ohlcv(df_5m, "1h"),
        "4H": resample_ohlcv(df_5m, "4h"),
    }
    total_days = (df_5m.index.max() - df_5m.index.min()).days

    det_all = {}
    ctx_all = {}
    bs_all = {}

    for tf_label, tf_df in timeframes.items():
        print(f"\n  [{tf_label}] {len(tf_df):,} bars")
        atr = compute_atr(tf_df)
        fvgs = detect_fvgs(tf_df, atr)
        if len(fvgs) == 0:
            print(f"  No FVGs found at {tf_label}")
            det_all[tf_label] = {"fvgs": fvgs, "rstats": {}}
            continue

        bu = int((fvgs["direction"] == "bullish").sum())
        be = int((fvgs["direction"] == "bearish").sum())
        print(f"  Total: {len(fvgs):,} | Bullish: {bu:,} | Bearish: {be:,}")
        print(f"  Per day: {len(fvgs)/total_days:.1f} | Median gap/ATR: {fvgs['gap_size_atr'].median():.3f}")

        # Map to 5m
        fvgs = map_to_5m_index(fvgs, df_5m, TF_MINUTES[tf_label])
        print(f"  After 5m mapping: {len(fvgs):,}")

        # Step 3
        print(f"  Computing forward returns...")
        fvgs = compute_forward_returns(fvgs, df_5m)
        rstats = {}
        for d in ("bullish", "bearish"):
            for h in HORIZONS:
                s = return_stats(fvgs, d, h)
                if s:
                    rstats[(d, h)] = s
        for d in ("bullish", "bearish"):
            s = rstats.get((d, "4h"))
            if s:
                print(f"  {d} 4h: mean={s['mean_bps']:+.1f}bps t={s['t']:.2f} p={s['p']:.4f} WR={s['wr']:.1%}")

        # Step 4
        print(f"  Computing fill rates...")
        fvgs = fill_rate_analysis(fvgs, df_5m)
        for d in ("bullish", "bearish"):
            sub = fvgs[fvgs["direction"] == d]
            if len(sub):
                print(f"  {d} fill rate 24h: {sub['filled'].mean():.1%}")

        det_all[tf_label] = {"fvgs": fvgs, "rstats": rstats}

        # Step 5
        print(f"  Contextual analysis...")
        ctx_all[tf_label] = contextual_analysis(fvgs)

        # Step 6
        print(f"  Bootstrap comparison (1000 iter)...")
        bs = bootstrap_comparison(fvgs, df_5m)
        bs_all[tf_label] = bs
        for d, v in bs.items():
            print(f"  {d} vs random: FVG={v['fvg_bps']:+.1f} Random={v['rand_bps']:+.1f} p={v['p']:.4f}")

    # Step 7
    print("\n--- Step 7: Generate report ---")
    report = generate_report(det_all, ctx_all, bs_all, df_5m)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(report, encoding="utf-8")
    print(f"  Saved: {OUTPUT_PATH}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
