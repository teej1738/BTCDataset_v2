"""
cscv_validation.py -- CSCV Overfitting Validation
===================================================
Step 2 of Build Plan (CLAUDE.md).

Validates that the signal edge is real and not an artifact of overfitting.

Modes:
  Default:       Config B H4-only (116 trades, both directions)
  --mtf-longs:   MTF long-only (H4+H1+M15 combined longs, ~176 trades)

Methods:
  1. CSCV (Lopez de Prado): Probability of Backtest Overfitting (PBO)
  2. Probabilistic Sharpe Ratio (PSR)
  3. Block bootstrap confidence intervals for mean R per trade
  4. Walk-forward: expanding OOS windows with out-of-sample PF

GO/NO-GO gate:
  PBO <= 20% AND bootstrap 95% CI lower bound > 0

Output:
  - Console report with GO/NO-GO verdict
  - results/cscv_validation.json  (or cscv_mtf_longs_validation.json)
  - results/cscv_validation.html  (or cscv_mtf_longs_validation.html)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from baseline_backtest_v2 import Config, load_labeled, ict_filters
from mtf_signals import TF_CONFIGS, config_b_filters

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"


# ── config ──────────────────────────────────────────────────────────────────
@dataclass
class CSCVConfig:
    n_folds: int = 8                # S folds -> C(8,4)=70 IS/OOS combos
    n_bootstrap: int = 10_000       # bootstrap iterations
    bootstrap_block_size: int = 5   # contiguous block size for block bootstrap
    psr_benchmark: float = 0.5      # Sharpe ratio benchmark (per-trade)
    pbo_threshold: float = 0.20     # GO/NO-GO: PBO must be <= this
    wf_initial_train: int = 40      # expanding WF: initial training trades
    wf_test_block: int = 25         # expanding WF: test block size
    confidence_level: float = 0.95  # CI level


# ── extract trade returns ───────────────────────────────────────────────────
def extract_trade_returns(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Extract per-trade R returns from Config B filter masks.
    No cooldown -- every signal bar is a trade for statistical validation.

    Returns DataFrame with columns: timestamp, direction, win, r_return
    """
    long_mask, short_mask = ict_filters(df, cfg)

    r_win = cfg.r_target - cfg.cost_per_r
    r_loss = -(1 + cfg.cost_per_r)

    parts = []
    for mask, direction, label_col in [
        (long_mask, "long", cfg.long_label),
        (short_mask, "short", cfg.short_label),
    ]:
        chunk = df.loc[mask, ["bar_start_ts_utc", label_col]].copy()
        chunk["direction"] = direction
        chunk["win"] = chunk[label_col] == 1.0
        chunk["r_return"] = np.where(chunk["win"], r_win, r_loss)
        chunk = chunk.rename(columns={"bar_start_ts_utc": "timestamp"})
        parts.append(chunk[["timestamp", "direction", "win", "r_return"]])

    return pd.concat(parts).sort_values("timestamp").reset_index(drop=True)


# ── extract MTF long-only returns ──────────────────────────────────────────
def extract_mtf_long_returns(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Extract per-trade R returns from combined MTF long-only signals.
    Union of H4, H1, M15 long masks (deduplicated at bar level).
    No cooldown -- every signal bar is a trade for statistical validation.

    Returns DataFrame with columns: timestamp, direction, win, r_return, tf
    """
    r_win = cfg.r_target - cfg.cost_per_r
    r_loss = -(1 + cfg.cost_per_r)
    label_col = cfg.long_label

    # Build union of long masks across all TFs, track which TF fired
    combined_long = pd.Series(False, index=df.index)
    tf_source = pd.Series("", index=df.index, dtype="object")

    for tf in TF_CONFIGS:
        long_mask, _ = config_b_filters(df, tf, cfg)
        # For bars where multiple TFs fire, keep earliest TF name (H4 > H1 > M15)
        new_bars = long_mask & ~combined_long
        tf_source[new_bars] = tf.name
        combined_long = combined_long | long_mask

    chunk = df.loc[combined_long, ["bar_start_ts_utc", label_col]].copy()
    chunk["direction"] = "long"
    chunk["tf"] = tf_source[combined_long].values
    chunk["win"] = chunk[label_col] == 1.0
    chunk["r_return"] = np.where(chunk["win"], r_win, r_loss)
    chunk = chunk.rename(columns={"bar_start_ts_utc": "timestamp"})

    return chunk[["timestamp", "direction", "tf", "win", "r_return"]].sort_values(
        "timestamp"
    ).reset_index(drop=True)


# ── 1. CSCV PBO ────────────────────────────────────────────────────────────
def compute_cscv_pbo(returns: np.ndarray, n_folds: int) -> dict:
    """
    Combinatorially Symmetric Cross-Validation (Lopez de Prado adaptation
    for a single strategy).

    Split trade returns into n_folds chronological blocks.
    For each of C(n_folds, n_folds//2) IS/OOS splits:
      - Compute mean R on the OOS half
    PBO = fraction of splits where OOS mean R <= 0.
    """
    n = len(returns)
    half = n_folds // 2
    fold_size = n // n_folds

    # Build folds (last fold absorbs remainder)
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        end = (start + fold_size) if i < n_folds - 1 else n
        folds.append(returns[start:end])

    oos_means = []
    is_means = []

    for is_idx in combinations(range(n_folds), half):
        oos_idx = tuple(i for i in range(n_folds) if i not in is_idx)

        is_r = np.concatenate([folds[i] for i in is_idx])
        oos_r = np.concatenate([folds[i] for i in oos_idx])

        is_means.append(float(np.mean(is_r)))
        oos_means.append(float(np.mean(oos_r)))

    oos_arr = np.array(oos_means)
    is_arr = np.array(is_means)

    pbo = float(np.mean(oos_arr <= 0))

    # IS-OOS correlation: positive = edge is consistent across subsets
    corr = float(np.corrcoef(is_arr, oos_arr)[0, 1])

    return {
        "pbo": round(pbo, 4),
        "n_combos": len(oos_arr),
        "n_negative_oos": int(np.sum(oos_arr <= 0)),
        "oos_mean": round(float(np.mean(oos_arr)), 4),
        "oos_median": round(float(np.median(oos_arr)), 4),
        "oos_std": round(float(np.std(oos_arr)), 4),
        "oos_min": round(float(np.min(oos_arr)), 4),
        "oos_max": round(float(np.max(oos_arr)), 4),
        "is_mean": round(float(np.mean(is_arr)), 4),
        "is_oos_correlation": round(corr, 4),
        "oos_values": oos_arr.tolist(),
        "is_values": is_arr.tolist(),
    }


# ── 2. Probabilistic Sharpe Ratio ──────────────────────────────────────────
def compute_psr(returns: np.ndarray, benchmark: float = 0.5) -> dict:
    """
    Probabilistic Sharpe Ratio (Bailey & Lopez de Prado 2012).

    PSR(SR*) = Phi(z)
    z = (SR_hat - SR*) * sqrt(n-1) / sqrt(1 - gamma3*SR + (gamma4-1)/4 * SR^2)

    Where gamma3 = skewness, gamma4 = kurtosis (normal = 3).
    Tests: P(true SR > benchmark | observed data).
    """
    n = len(returns)
    if n < 3:
        return {"psr": 0.0, "sharpe_ratio": 0.0, "error": "insufficient data"}

    mean_r = float(np.mean(returns))
    std_r = float(np.std(returns, ddof=1))

    if std_r == 0:
        sr = float("inf") if mean_r > 0 else 0.0
        return {"psr": 1.0 if mean_r > 0 else 0.0, "sharpe_ratio": sr}

    sr = mean_r / std_r
    gamma3 = float(stats.skew(returns, bias=False))
    # Regular kurtosis (normal = 3), not excess kurtosis
    gamma4 = float(stats.kurtosis(returns, fisher=False, bias=False))

    denom_sq = 1 - gamma3 * sr + (gamma4 - 1) / 4 * sr**2
    if denom_sq <= 0:
        denom_sq = 1.0  # fallback to normal assumption

    z = (sr - benchmark) * np.sqrt(n - 1) / np.sqrt(denom_sq)
    psr = float(stats.norm.cdf(z))

    return {
        "psr": round(psr, 4),
        "sharpe_ratio": round(sr, 4),
        "benchmark": benchmark,
        "z_score": round(z, 4),
        "skewness": round(gamma3, 4),
        "kurtosis": round(gamma4, 4),
        "n_trades": n,
    }


# ── 3. Block bootstrap CI ──────────────────────────────────────────────────
def block_bootstrap_ci(
    returns: np.ndarray,
    n_bootstrap: int = 10_000,
    block_size: int = 5,
    ci: float = 0.95,
) -> dict:
    """
    Block bootstrap confidence interval for mean R per trade.
    Resamples contiguous blocks to preserve local autocorrelation structure.
    """
    n = len(returns)
    rng = np.random.default_rng(42)
    n_blocks = max(1, -(-n // block_size))  # ceil division

    boot_means = np.empty(n_bootstrap)
    max_start = max(1, n - block_size + 1)

    for b in range(n_bootstrap):
        starts = rng.integers(0, max_start, size=n_blocks)
        sample = np.concatenate([returns[s:s + block_size] for s in starts])[:n]
        boot_means[b] = np.mean(sample)

    alpha = 1 - ci
    lo = float(np.percentile(boot_means, alpha / 2 * 100))
    hi = float(np.percentile(boot_means, (1 - alpha / 2) * 100))

    return {
        "observed_mean": round(float(np.mean(returns)), 4),
        "boot_mean": round(float(np.mean(boot_means)), 4),
        "ci_lower": round(lo, 4),
        "ci_upper": round(hi, 4),
        "ci_level": ci,
        "pct_positive": round(float(np.mean(boot_means > 0) * 100), 2),
        "n_bootstrap": n_bootstrap,
        "block_size": block_size,
        "boot_std": round(float(np.std(boot_means)), 4),
    }


# ── 4. Walk-forward (expanding) ────────────────────────────────────────────
def _pf(returns: np.ndarray) -> float:
    """Profit factor from return array."""
    gross_win = float(np.sum(returns[returns > 0]))
    gross_loss = float(abs(np.sum(returns[returns < 0])))
    return gross_win / gross_loss if gross_loss > 0 else float("inf")


def walk_forward_expanding(
    returns: np.ndarray,
    timestamps: np.ndarray,
    initial_train: int = 40,
    test_block: int = 25,
) -> list[dict]:
    """
    Expanding (anchored) walk-forward analysis.

    Training set grows with each window; test block is ~constant size.
    Anchored at the start of the data to simulate real deployment.
    """
    n = len(returns)
    windows = []
    train_end = initial_train

    while train_end < n:
        remaining = n - train_end
        if remaining < 5:
            break

        test_end = min(train_end + test_block, n)
        train_r = returns[:train_end]
        test_r = returns[train_end:test_end]

        windows.append({
            "window": len(windows) + 1,
            "train_n": train_end,
            "test_n": test_end - train_end,
            "train_period": f"trade 1-{train_end}",
            "test_period": f"trade {train_end + 1}-{test_end}",
            "test_start_ts": str(timestamps[train_end]),
            "test_end_ts": str(timestamps[test_end - 1]),
            "train_mean_r": round(float(np.mean(train_r)), 4),
            "test_mean_r": round(float(np.mean(test_r)), 4),
            "train_pf": round(_pf(train_r), 2),
            "test_pf": round(_pf(test_r), 2),
            "train_wr": round(float(np.mean(train_r > 0)), 4),
            "test_wr": round(float(np.mean(test_r > 0)), 4),
        })

        train_end = test_end

    return windows


# ── Plotly chart ────────────────────────────────────────────────────────────
def save_chart(
    cscv_results: dict, wf_windows: list[dict], pbo: float, mode: str = "h4"
) -> Path | None:
    """Save combined CSCV distribution + walk-forward chart."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return None

    n_combos = cscv_results["n_combos"]
    n_rows = 2 if wf_windows else 1
    subtitles = [f"CSCV: OOS Mean R Distribution ({n_combos} IS/OOS Splits)"]
    if wf_windows:
        subtitles.append("Walk-Forward: Expanding OOS Profit Factor")

    fig = make_subplots(
        rows=n_rows, cols=1,
        subplot_titles=subtitles,
        vertical_spacing=0.15,
    )

    # -- Row 1: CSCV OOS histogram --
    oos_vals = cscv_results["oos_values"]
    fig.add_trace(
        go.Histogram(
            x=oos_vals,
            nbinsx=20,
            marker_color="steelblue",
            opacity=0.8,
            name="OOS Mean R",
        ),
        row=1, col=1,
    )
    # Zero reference line
    fig.add_vline(x=0, line_dash="dash", line_color="red", line_width=2, row=1, col=1)
    # Observed mean
    oos_mean = np.mean(oos_vals)
    fig.add_vline(
        x=oos_mean, line_dash="dot", line_color="lime", line_width=2, row=1, col=1,
    )
    fig.add_annotation(
        x=0.02, y=0.95, xref="x domain", yref="y domain",
        text=f"PBO = {pbo:.1%}",
        showarrow=False, font=dict(size=14, color="white"),
        bgcolor="rgba(255,0,0,0.5)" if pbo > 0.20 else "rgba(0,180,0,0.5)",
        row=1, col=1,
    )

    # -- Row 2: Walk-forward OOS PF --
    if wf_windows:
        labels = [f"W{w['window']}\n(n={w['test_n']})" for w in wf_windows]
        test_pfs = [w["test_pf"] for w in wf_windows]
        test_wrs = [w["test_wr"] * 100 for w in wf_windows]

        colors = ["green" if pf > 1.0 else "red" for pf in test_pfs]
        fig.add_trace(
            go.Bar(
                x=labels, y=test_pfs,
                marker_color=colors, opacity=0.8,
                name="OOS PF",
                text=[f"PF {pf:.2f}<br>WR {wr:.0f}%" for pf, wr in zip(test_pfs, test_wrs)],
                textposition="outside",
            ),
            row=2, col=1,
        )
        fig.add_hline(y=1.0, line_dash="dash", line_color="yellow", row=2, col=1)

    title_suffix = "MTF Long-Only (H4+H1+M15)" if mode == "mtf-longs" else "Config B"
    fig.update_layout(
        template="plotly_dark",
        title=f"CSCV Validation -- {title_suffix}",
        height=400 * n_rows,
        showlegend=False,
    )
    fig.update_xaxes(title_text="OOS Mean R per Trade", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    if wf_windows:
        fig.update_xaxes(title_text="Walk-Forward Window", row=2, col=1)
        fig.update_yaxes(title_text="Profit Factor", row=2, col=1)

    fname = "cscv_mtf_longs_validation.html" if mode == "mtf-longs" else "cscv_validation.html"
    path = RESULTS_DIR / fname
    fig.write_html(str(path))
    return path


# ── Console report + save ──────────────────────────────────────────────────
def print_report(
    trade_df: pd.DataFrame,
    cscv_results: dict,
    psr_05: dict,
    psr_00: dict,
    bootstrap: dict,
    wf_windows: list[dict],
    cscv_cfg: CSCVConfig,
    mode: str = "h4",
) -> dict:
    """Print full validation report and save output files."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    sep = "=" * 70
    rule = "-" * 70

    n = len(trade_df)
    n_wins = int(trade_df["win"].sum())
    mean_r = float(trade_df["r_return"].mean())

    title = ("MTF Long-Only (H4+H1+M15) Overfitting Test (Step 2)"
             if mode == "mtf-longs"
             else "Config B Overfitting Test (Step 2)")
    print(f"\n{sep}")
    print(f"  CSCV VALIDATION -- {title}")
    print(sep)
    print(f"  Trades: {n}  |  Wins: {n_wins}  |  Losses: {n - n_wins}  |  "
          f"WR: {n_wins / n:.2%}  |  Mean R: {mean_r:+.4f}")
    print(f"  CSCV folds: {cscv_cfg.n_folds} (C({cscv_cfg.n_folds},"
          f"{cscv_cfg.n_folds // 2})={comb(cscv_cfg.n_folds, cscv_cfg.n_folds // 2)} combos)"
          f"  |  Bootstrap: {cscv_cfg.n_bootstrap:,} iters")

    # ── 1. CSCV PBO ──
    print(f"\n{rule}")
    print("  1. CSCV -- Probability of Backtest Overfitting")
    print(rule)
    pbo = cscv_results["pbo"]
    pbo_pass = pbo <= cscv_cfg.pbo_threshold
    gate = "PASS" if pbo_pass else "FAIL"
    print(f"  PBO:              {pbo:.2%}  "
          f"({cscv_results['n_negative_oos']}/{cscv_results['n_combos']} combos "
          f"with OOS mean R <= 0)")
    print(f"  Threshold:        <= {cscv_cfg.pbo_threshold:.0%}")
    print(f"  Gate:             {gate}")
    print(f"  OOS mean R:       {cscv_results['oos_mean']:+.4f}  "
          f"(median {cscv_results['oos_median']:+.4f})")
    print(f"  OOS range:        [{cscv_results['oos_min']:+.4f}, "
          f"{cscv_results['oos_max']:+.4f}]")
    print(f"  OOS std:          {cscv_results['oos_std']:.4f}")
    print(f"  IS-OOS corr:      {cscv_results['is_oos_correlation']:+.4f}  "
          f"({'positive = consistent' if cscv_results['is_oos_correlation'] > 0 else 'negative = unstable'})")

    # ── 2. PSR ──
    print(f"\n{rule}")
    print("  2. Probabilistic Sharpe Ratio (PSR)")
    print(rule)
    sr = psr_05["sharpe_ratio"]
    print(f"  Observed SR:      {sr:+.4f}  (per-trade, not annualized)")
    print(f"  Skewness:         {psr_05['skewness']:+.4f}")
    print(f"  Kurtosis:         {psr_05['kurtosis']:+.4f}  (normal=3)")
    print(f"")
    print(f"  PSR(SR > 0.0):    {psr_00['psr']:.4f}  "
          f"(z = {psr_00['z_score']:+.2f})"
          f"  {'PASS' if psr_00['psr'] >= 0.95 else 'FAIL'}")
    print(f"  PSR(SR > 0.5):    {psr_05['psr']:.4f}  "
          f"(z = {psr_05['z_score']:+.2f})"
          f"  {'PASS' if psr_05['psr'] >= 0.95 else 'FAIL'}")
    print(f"")
    # Annualized Sharpe for context
    trades_per_year = n / 6.0  # ~6 years of data
    ann_sr = sr * np.sqrt(trades_per_year)
    print(f"  Annualized SR:    {ann_sr:+.4f}  "
          f"(~{trades_per_year:.0f} trades/yr, SR * sqrt(N/yr))")

    # ── 3. Bootstrap CI ──
    print(f"\n{rule}")
    print("  3. Block Bootstrap CI for Mean R per Trade")
    print(rule)
    ci_lo = bootstrap["ci_lower"]
    ci_hi = bootstrap["ci_upper"]
    ci_pass = ci_lo > 0
    print(f"  Observed mean R:  {bootstrap['observed_mean']:+.4f}")
    print(f"  Bootstrap mean:   {bootstrap['boot_mean']:+.4f}")
    print(f"  95% CI:           [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"  P(mean R > 0):    {bootstrap['pct_positive']:.1f}%")
    print(f"  Block size:       {bootstrap['block_size']} trades")
    print(f"  CI excludes 0:    {'PASS' if ci_pass else 'FAIL'}")

    # ── 4. Walk-forward ──
    print(f"\n{rule}")
    print("  4. Walk-Forward Analysis (Expanding, Anchored)")
    print(rule)
    if wf_windows:
        print(f"  {'Win':>5}  {'Train':>7}  {'Test':>6}  "
              f"{'Tr PF':>7}  {'OOS PF':>7}  {'OOS WR':>7}  {'OOS EV':>8}  Period")
        print(f"  {'-----':>5}  {'-------':>7}  {'------':>6}  "
              f"{'-------':>7}  {'-------':>7}  {'-------':>7}  {'--------':>8}  ------")
        for w in wf_windows:
            print(f"  {w['window']:>5}  {w['train_n']:>7}  {w['test_n']:>6}  "
                  f"{w['train_pf']:>7.2f}  {w['test_pf']:>7.2f}  "
                  f"{w['test_wr']:>6.2%}  {w['test_mean_r']:>+8.4f}  "
                  f"{w['test_period']}")

        oos_pfs = [w["test_pf"] for w in wf_windows]
        oos_evs = [w["test_mean_r"] for w in wf_windows]
        n_profitable = sum(1 for pf in oos_pfs if pf > 1.0)
        print(f"\n  OOS windows profitable:  {n_profitable}/{len(wf_windows)}")
        print(f"  OOS mean PF:             {np.mean(oos_pfs):.2f}")
        print(f"  OOS mean EV:             {np.mean(oos_evs):+.4f}")
    else:
        print("  Insufficient data for walk-forward windows.")

    # ── GO/NO-GO ──
    print(f"\n{sep}")
    print("  GO / NO-GO GATE")
    print(sep)

    checks = [
        ("PBO <= 20%", pbo_pass, f"{pbo:.2%}"),
        ("Bootstrap 95% CI > 0", ci_pass, f"[{ci_lo:+.4f}, {ci_hi:+.4f}]"),
        ("PSR(SR > 0) >= 0.95", psr_00["psr"] >= 0.95, f"{psr_00['psr']:.4f}"),
    ]

    all_pass = all(passed for _, passed, _ in checks)
    for label, passed, val in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}]  {label:<30}  {val}")

    if all_pass:
        verdict = "GO -- proceed to Step 3 (Regime Classification)"
    else:
        verdict = "NO-GO -- diagnose before proceeding"
    print(f"\n  >>> {verdict}")
    print(sep)

    # ── Save chart ──
    chart_path = save_chart(cscv_results, wf_windows, pbo, mode=mode)
    if chart_path:
        print(f"\n  Saved: {chart_path}")
    else:
        print("\n  (plotly not installed -- skipping chart)")

    # ── Save JSON ──
    # Strip large arrays from JSON for readability
    cscv_slim = {k: v for k, v in cscv_results.items()
                 if k not in ("oos_values", "is_values")}
    summary = {
        "mode": mode,
        "n_trades": n,
        "n_wins": n_wins,
        "win_rate": round(n_wins / n, 4),
        "mean_r": round(mean_r, 4),
        "cscv": cscv_slim,
        "psr_benchmark_0": psr_00,
        "psr_benchmark_05": psr_05,
        "bootstrap_ci": bootstrap,
        "walk_forward": wf_windows,
        "go_no_go": {
            "pbo_pass": pbo_pass,
            "ci_pass": ci_pass,
            "psr_pass": psr_00["psr"] >= 0.95,
            "overall": all_pass,
            "verdict": verdict,
        },
    }
    json_fname = ("cscv_mtf_longs_validation.json" if mode == "mtf-longs"
                  else "cscv_validation.json")
    json_path = RESULTS_DIR / json_fname
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved: {json_path}")

    return summary


# ── main ────────────────────────────────────────────────────────────────────
def main() -> None:
    mode = "mtf-longs" if "--mtf-longs" in sys.argv else "h4"

    cfg = Config()
    cscv_cfg = CSCVConfig()

    # 1. Load data + extract trade-level returns
    df = load_labeled(cfg)

    if mode == "mtf-longs":
        # MTF long-only: adjust walk-forward params for ~176 trades
        cscv_cfg.wf_initial_train = 50
        cscv_cfg.wf_test_block = 30

        trade_df = extract_mtf_long_returns(df, cfg)
        print(f"\n  MTF long-only trades extracted: {len(trade_df)}")
        if "tf" in trade_df.columns:
            for tf_name in ["H4", "H1", "M15"]:
                n_tf = (trade_df["tf"] == tf_name).sum()
                print(f"    {tf_name}: {n_tf}")
    else:
        trade_df = extract_trade_returns(df, cfg)
        print(f"\n  Config B trades extracted: {len(trade_df)}")
        print(f"  Long: {(trade_df['direction'] == 'long').sum()}  |  "
              f"Short: {(trade_df['direction'] == 'short').sum()}")

    returns = trade_df["r_return"].values
    timestamps = trade_df["timestamp"].values

    print(f"  Mean R: {np.mean(returns):+.4f}  |  Std R: {np.std(returns):.4f}")
    print(f"  Trades/fold (S={cscv_cfg.n_folds}): "
          f"{len(returns) // cscv_cfg.n_folds}-"
          f"{len(returns) // cscv_cfg.n_folds + 1}")

    # 2. CSCV PBO
    print("\n  Running CSCV...")
    cscv_results = compute_cscv_pbo(returns, cscv_cfg.n_folds)

    # 3. PSR at two benchmarks
    print("  Computing PSR...")
    psr_05 = compute_psr(returns, benchmark=0.5)
    psr_00 = compute_psr(returns, benchmark=0.0)

    # 4. Block bootstrap
    print(f"  Running block bootstrap ({cscv_cfg.n_bootstrap:,} iters)...")
    bootstrap = block_bootstrap_ci(
        returns, cscv_cfg.n_bootstrap,
        cscv_cfg.bootstrap_block_size, cscv_cfg.confidence_level,
    )

    # 5. Walk-forward (expanding, anchored)
    print("  Running walk-forward...")
    wf_windows = walk_forward_expanding(
        returns, timestamps,
        cscv_cfg.wf_initial_train, cscv_cfg.wf_test_block,
    )

    # 6. Report
    print_report(
        trade_df, cscv_results, psr_05, psr_00, bootstrap, wf_windows,
        cscv_cfg, mode=mode,
    )


if __name__ == "__main__":
    main()
