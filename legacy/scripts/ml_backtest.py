"""
ml_backtest.py -- ML-Scored Backtest with Cooldown + CSCV Validation
====================================================================
Tests three ML threshold configs with realistic 48-bar cooldown,
then validates the best with CSCV.

Configs:
  T1: ML prob >= 0.50 + 48-bar cooldown (ML standalone)
  T2: ML prob >= 0.60 + 48-bar cooldown (ML standalone, higher quality)
  T3: Config B signal AND ML prob >= 0.50 + 48-bar cooldown (overlay)
  BL: Config B only + 48-bar cooldown (baseline, OOS window only)

CSCV is run on the best config by annualized Sharpe.

Output:
  - Console report with side-by-side comparison
  - results/ml_backtest.json
  - results/ml_backtest.html
  - results/cscv_ml_validation.json / .html
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_backtest_v2 import Config
from mtf_signals import TF_CONFIGS, config_b_filters
from ml_pipeline import MLConfig, prepare_data, walk_forward_train
from cscv_validation import (
    CSCVConfig,
    compute_cscv_pbo,
    compute_psr,
    block_bootstrap_ci,
    walk_forward_expanding,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
CACHE_PATH = RESULTS_DIR / "ml_oos_probs.npy"


# ---- OOS probability loader ------------------------------------------------
def get_oos_probs(
    df: pd.DataFrame, features: list[str], ml_cfg: MLConfig
) -> np.ndarray:
    """Load cached OOS probs or re-run walk-forward training."""
    if CACHE_PATH.exists():
        probs = np.load(str(CACHE_PATH))
        if len(probs) == len(df):
            print(f"  Loaded cached OOS probs from {CACHE_PATH.name}")
            return probs
        print(f"  Cache length mismatch ({len(probs)} vs {len(df)}), "
              f"re-training...")

    print("  Running walk-forward training (takes ~10 min)...")
    oos_probs, _, _ = walk_forward_train(df, features, ml_cfg)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(CACHE_PATH), oos_probs)
    print(f"  Saved OOS probs to {CACHE_PATH.name}")
    return oos_probs


# ---- signal mask builders ---------------------------------------------------
def build_config_b_mask(df: pd.DataFrame, cfg: Config) -> np.ndarray:
    """MTF long-only Config B union mask (boolean array)."""
    combined = pd.Series(False, index=df.index)
    for tf in TF_CONFIGS:
        lm, _ = config_b_filters(df, tf, cfg)
        combined = combined | lm
    return combined.values


# ---- bar-by-bar simulation with cooldown ------------------------------------
def simulate(
    signal_mask: np.ndarray,
    label_arr: np.ndarray,
    cooldown: int,
) -> list[int]:
    """
    Walk bars chronologically with cooldown.
    Returns list of bar indices where trades are taken.
    """
    trade_indices: list[int] = []
    bars_since = cooldown  # start ready to trade

    for i in range(len(signal_mask)):
        bars_since += 1
        if bars_since > cooldown and signal_mask[i]:
            trade_indices.append(i)
            bars_since = 0

    return trade_indices


def build_trade_returns(
    trade_indices: list[int],
    label_arr: np.ndarray,
    r_target: int,
    cost_per_r: float,
) -> np.ndarray:
    """Convert trade bar indices to R returns array."""
    r_win = r_target - cost_per_r
    r_loss = -(1 + cost_per_r)
    returns = []
    for i in trade_indices:
        returns.append(r_win if label_arr[i] == 1 else r_loss)
    return np.array(returns) if returns else np.array([], dtype=float)


def equity_sim(
    r_returns: np.ndarray,
    risk_pct: float,
    initial_equity: float,
) -> tuple[list[float], float]:
    """Compute equity path and max drawdown from R returns."""
    equity = initial_equity
    peak = equity
    max_dd = 0.0
    path = [equity]

    for r in r_returns:
        pnl = equity * risk_pct * r
        equity += pnl
        peak = max(peak, equity)
        dd = (peak - equity) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)
        path.append(equity)

    return path, max_dd


def compute_metrics(
    name: str,
    r_returns: np.ndarray,
    max_dd: float,
    final_equity: float,
    years: float,
) -> dict:
    """Standard metrics from R returns array."""
    n = len(r_returns)
    if n == 0:
        return {
            "name": name, "n_trades": 0, "trades_per_yr": 0.0,
            "win_rate": 0.0, "ev_r": 0.0, "profit_factor": 0.0,
            "max_dd_pct": 0.0, "sharpe_ann": 0.0,
            "final_equity": final_equity,
        }

    wins = int(np.sum(r_returns > 0))
    wr = wins / n
    ev = float(np.mean(r_returns))

    gross_win = float(np.sum(r_returns[r_returns > 0]))
    gross_loss = float(abs(np.sum(r_returns[r_returns < 0])))
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

    std_r = float(np.std(r_returns, ddof=1)) if n > 1 else 0.0
    sr_pt = ev / std_r if std_r > 0 else 0.0
    tpy = n / years
    ann_sr = sr_pt * np.sqrt(tpy) if tpy > 0 else 0.0

    return {
        "name": name,
        "n_trades": n,
        "trades_per_yr": round(tpy, 1),
        "win_rate": round(wr, 4),
        "ev_r": round(ev, 4),
        "profit_factor": round(pf, 4),
        "max_dd_pct": round(max_dd * 100, 2),
        "sharpe_ann": round(ann_sr, 4),
        "final_equity": round(final_equity, 2),
    }


# ---- CSCV validation -------------------------------------------------------
def run_cscv(
    r_returns: np.ndarray, timestamps: np.ndarray, name: str
) -> dict:
    """Full CSCV validation suite on trade returns."""
    cscv_cfg = CSCVConfig()
    n = len(r_returns)

    # Adjust walk-forward params for trade count
    cscv_cfg.wf_initial_train = max(20, n // 5)
    cscv_cfg.wf_test_block = max(10, n // 8)

    print(f"\n  Running CSCV on {name} ({n} trades)...")

    cscv_results = compute_cscv_pbo(r_returns, cscv_cfg.n_folds)
    psr_00 = compute_psr(r_returns, benchmark=0.0)
    psr_05 = compute_psr(r_returns, benchmark=0.5)
    bootstrap = block_bootstrap_ci(
        r_returns, cscv_cfg.n_bootstrap,
        cscv_cfg.bootstrap_block_size, cscv_cfg.confidence_level,
    )
    wf_windows = walk_forward_expanding(
        r_returns, timestamps,
        cscv_cfg.wf_initial_train, cscv_cfg.wf_test_block,
    )

    return {
        "cscv": cscv_results,
        "psr_benchmark_0": psr_00,
        "psr_benchmark_05": psr_05,
        "bootstrap_ci": bootstrap,
        "walk_forward": wf_windows,
    }


# ---- plotly chart -----------------------------------------------------------
def save_chart(
    configs_metrics: list[dict],
    equity_paths: list[list[float]],
    cscv_result: dict | None,
) -> Path | None:
    """Equity curves + CSCV histogram."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return None

    has_cscv = cscv_result is not None
    n_rows = 2 if has_cscv else 1
    titles = ["Equity Curves (2% risk, 48-bar cooldown)"]
    if has_cscv:
        titles.append("CSCV: OOS Mean R Distribution")

    fig = make_subplots(
        rows=n_rows, cols=1,
        subplot_titles=titles,
        vertical_spacing=0.15,
    )

    colors = {
        "T1 (ML>=0.50)": "cyan",
        "T2 (ML>=0.60)": "orange",
        "T3 (CB+ML>=0.50)": "limegreen",
        "Config B (OOS)": "yellow",
    }

    for cm, eq_path in zip(configs_metrics, equity_paths):
        name = cm["name"]
        fig.add_trace(
            go.Scatter(
                x=list(range(len(eq_path))), y=eq_path,
                mode="lines", name=name,
                line=dict(color=colors.get(name, "white"), width=1.5),
            ),
            row=1, col=1,
        )

    fig.update_xaxes(title_text="Trade #", row=1, col=1)
    fig.update_yaxes(title_text="Equity ($)", row=1, col=1)

    if has_cscv:
        oos_vals = cscv_result["cscv"]["oos_values"]
        pbo = cscv_result["cscv"]["pbo"]
        fig.add_trace(
            go.Histogram(
                x=oos_vals, nbinsx=20,
                marker_color="steelblue", opacity=0.8,
                name="OOS Mean R", showlegend=False,
            ),
            row=2, col=1,
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red",
                      line_width=2, row=2, col=1)
        oos_mean = float(np.mean(oos_vals))
        fig.add_vline(x=oos_mean, line_dash="dot", line_color="lime",
                      line_width=2, row=2, col=1)
        fig.add_annotation(
            x=0.02, y=0.95, xref="x2 domain", yref="y2 domain",
            text=f"PBO = {pbo:.1%}",
            showarrow=False, font=dict(size=14, color="white"),
            bgcolor=("rgba(0,180,0,0.5)" if pbo <= 0.20
                     else "rgba(255,0,0,0.5)"),
        )
        fig.update_xaxes(title_text="OOS Mean R", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        title="ML Backtest -- Threshold Configs + CSCV Validation",
        height=400 * n_rows,
    )

    path = RESULTS_DIR / "ml_backtest.html"
    fig.write_html(str(path))
    return path


# ---- console report --------------------------------------------------------
def print_report(
    configs_metrics: list[dict],
    cscv_result: dict | None,
    best_name: str,
    oos_years: float,
) -> bool:
    """Print comparison table and CSCV results. Returns all_pass."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    sep = "=" * 70
    rule = "-" * 70

    print(f"\n{sep}")
    print("  ML BACKTEST -- Threshold Configs with 48-bar Cooldown")
    print(sep)
    print(f"  OOS window: ~{oos_years:.2f} years")
    print(f"  Cooldown: 48 bars (4h)")
    print(f"  Risk: 2% per trade, compounding")
    print(f"  R model: +1.95R win, -1.05R loss (2R target, 0.05R cost)")

    # -- side-by-side table --
    print(f"\n{rule}")
    print("  CONFIG COMPARISON")
    print(rule)
    print(f"  {'Config':<20}  {'Trades':>7}  {'Tr/Yr':>6}  {'WR':>7}  "
          f"{'EV(R)':>8}  {'PF':>6}  {'MaxDD':>7}  {'Sharpe':>7}  "
          f"{'Final$':>9}")

    for cm in configs_metrics:
        pf_str = f"{cm['profit_factor']:>6.2f}"
        if cm["profit_factor"] == float("inf"):
            pf_str = "   inf"
        print(f"  {cm['name']:<20}  {cm['n_trades']:>7}  "
              f"{cm['trades_per_yr']:>6.1f}  {cm['win_rate']:>6.2%}  "
              f"{cm['ev_r']:>+8.4f}  {pf_str}  "
              f"{cm['max_dd_pct']:>6.1f}%  {cm['sharpe_ann']:>7.2f}  "
              f"${cm['final_equity']:>8,.0f}")

    print(f"\n  Best by annualized Sharpe: {best_name}")

    # -- CSCV results --
    all_pass = False
    if cscv_result:
        cscv = cscv_result["cscv"]
        psr_00 = cscv_result["psr_benchmark_0"]
        psr_05 = cscv_result["psr_benchmark_05"]
        bootstrap = cscv_result["bootstrap_ci"]
        wf = cscv_result["walk_forward"]

        print(f"\n{rule}")
        print(f"  CSCV VALIDATION -- {best_name}")
        print(rule)

        pbo = cscv["pbo"]
        print(f"  PBO:            {pbo:.2%}  "
              f"({cscv['n_negative_oos']}/{cscv['n_combos']} negative OOS)")
        print(f"  OOS mean R:     {cscv['oos_mean']:+.4f}  "
              f"(range [{cscv['oos_min']:+.4f}, {cscv['oos_max']:+.4f}])")
        print(f"  IS-OOS corr:    {cscv['is_oos_correlation']:+.4f}")

        sr = psr_00["sharpe_ratio"]
        print(f"\n  Per-trade SR:   {sr:+.4f}")
        print(f"  PSR(SR > 0):    {psr_00['psr']:.4f}  "
              f"(z = {psr_00['z_score']:+.2f})")
        print(f"  PSR(SR > 0.5):  {psr_05['psr']:.4f}  "
              f"(z = {psr_05['z_score']:+.2f})")

        # Annualized
        best_cm = [cm for cm in configs_metrics if cm["name"] == best_name][0]
        tpy = best_cm["trades_per_yr"]
        ann_sr = sr * np.sqrt(tpy) if tpy > 0 else 0
        print(f"  Annualized SR:  {ann_sr:+.4f}  (~{tpy:.0f} trades/yr)")

        ci_lo = bootstrap["ci_lower"]
        ci_hi = bootstrap["ci_upper"]
        print(f"\n  Bootstrap 95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]")
        print(f"  P(mean R > 0):    {bootstrap['pct_positive']:.1f}%")

        if wf:
            n_profitable = sum(1 for w in wf if w["test_pf"] > 1.0)
            mean_pf = float(np.mean([w["test_pf"] for w in wf]))
            print(f"\n  Walk-forward: {n_profitable}/{len(wf)} "
                  f"OOS windows profitable")
            print(f"  OOS mean PF:  {mean_pf:.2f}")

            print(f"\n  {'Win':>5}  {'Train':>7}  {'Test':>6}  "
                  f"{'Tr PF':>7}  {'OOS PF':>7}  {'OOS WR':>7}  "
                  f"{'OOS EV':>8}")
            for w in wf:
                print(f"  {w['window']:>5}  {w['train_n']:>7}  "
                      f"{w['test_n']:>6}  {w['train_pf']:>7.2f}  "
                      f"{w['test_pf']:>7.2f}  {w['test_wr']:>6.2%}  "
                      f"{w['test_mean_r']:>+8.4f}")

        # -- GO/NO-GO --
        print(f"\n{rule}")
        print("  GO / NO-GO")
        print(rule)
        checks = [
            ("PBO <= 20%", bool(pbo <= 0.20), f"{pbo:.2%}"),
            ("Bootstrap 95% CI > 0", bool(ci_lo > 0),
             f"[{ci_lo:+.4f}, {ci_hi:+.4f}]"),
            ("PSR(SR > 0) >= 0.95", bool(psr_00["psr"] >= 0.95),
             f"{psr_00['psr']:.4f}"),
        ]
        for label, passed, val in checks:
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}]  {label:<30}  {val}")

        all_pass = all(c[1] for c in checks)
        if all_pass:
            print(f"\n  >>> GO -- ML-scored strategy validated")
        else:
            print(f"\n  >>> Results mixed -- see details above")

    print(sep)
    return all_pass


# ---- main ------------------------------------------------------------------
def main() -> None:
    cfg = Config()
    ml_cfg = MLConfig()

    # 1. Load data + OOS probabilities
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

    # 2. Build signal masks + simulate
    print(f"\n{'=' * 70}")
    print("  STEP 2: Build Signal Masks + Simulate")
    print("=" * 70)

    label_arr = df[ml_cfg.label_col].values
    cb_mask = build_config_b_mask(df, cfg)
    timestamps = df["bar_start_ts_utc"].values

    configs_def = [
        ("T1 (ML>=0.50)", (oos_probs >= 0.50) & oos_valid),
        ("T2 (ML>=0.60)", (oos_probs >= 0.60) & oos_valid),
        ("T3 (CB+ML>=0.50)", cb_mask & (oos_probs >= 0.50) & oos_valid),
        ("Config B (OOS)", cb_mask & oos_valid),
    ]

    configs_metrics: list[dict] = []
    configs_returns: dict[str, np.ndarray] = {}
    configs_indices: dict[str, list[int]] = {}
    equity_paths: list[list[float]] = []

    for name, mask in configs_def:
        trade_idx = simulate(mask, label_arr, cfg.cooldown_bars)
        r_returns = build_trade_returns(
            trade_idx, label_arr, cfg.r_target, cfg.cost_per_r
        )
        eq_path, max_dd = equity_sim(
            r_returns, cfg.risk_pct, cfg.initial_equity
        )

        metrics = compute_metrics(
            name, r_returns, max_dd, eq_path[-1], oos_years
        )
        configs_metrics.append(metrics)
        configs_returns[name] = r_returns
        configs_indices[name] = trade_idx
        equity_paths.append(eq_path)

        print(f"  {name}: {metrics['n_trades']} trades, "
              f"WR {metrics['win_rate']:.2%}, "
              f"EV {metrics['ev_r']:+.4f}R, "
              f"Sharpe {metrics['sharpe_ann']:.2f}")

    # 3. Select best config (by annualized Sharpe, excluding baseline)
    candidates = [cm for cm in configs_metrics
                  if cm["name"] != "Config B (OOS)"]
    best = max(candidates, key=lambda x: x["sharpe_ann"])
    best_name = best["name"]
    print(f"\n  Best config: {best_name} "
          f"(Sharpe {best['sharpe_ann']:.2f})")

    # 4. CSCV validation on best config
    print(f"\n{'=' * 70}")
    print("  STEP 3: CSCV Validation")
    print("=" * 70)

    best_returns = configs_returns[best_name]
    best_indices = configs_indices[best_name]
    best_timestamps = timestamps[best_indices]

    cscv_result = run_cscv(best_returns, best_timestamps, best_name)

    # 5. Report
    all_pass = print_report(
        configs_metrics, cscv_result, best_name, oos_years
    )

    # 6. Save chart
    chart_path = save_chart(configs_metrics, equity_paths, cscv_result)
    if chart_path:
        print(f"\n  Saved: {chart_path}")

    # 7. Save JSON
    cscv_slim = {
        k: v for k, v in cscv_result["cscv"].items()
        if k not in ("oos_values", "is_values")
    }
    summary = {
        "oos_years": round(oos_years, 2),
        "configs": configs_metrics,
        "best_config": best_name,
        "cscv_validation": {
            "cscv": cscv_slim,
            "psr_benchmark_0": cscv_result["psr_benchmark_0"],
            "psr_benchmark_05": cscv_result["psr_benchmark_05"],
            "bootstrap_ci": cscv_result["bootstrap_ci"],
            "walk_forward": cscv_result["walk_forward"],
            "go_no_go": {
                "pbo_pass": bool(cscv_result["cscv"]["pbo"] <= 0.20),
                "ci_pass": bool(
                    cscv_result["bootstrap_ci"]["ci_lower"] > 0
                ),
                "psr_pass": bool(
                    cscv_result["psr_benchmark_0"]["psr"] >= 0.95
                ),
                "all_pass": bool(all_pass),
            },
        },
    }

    json_path = RESULTS_DIR / "ml_backtest.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved: {json_path}")

    # Also save CSCV separately
    cscv_json_path = RESULTS_DIR / "cscv_ml_validation.json"
    with open(cscv_json_path, "w") as f:
        cscv_full = {
            "mode": "ml-backtest",
            "best_config": best_name,
            "n_trades": best["n_trades"],
            "win_rate": best["win_rate"],
            "ev_r": best["ev_r"],
            "cscv": cscv_slim,
            "psr_benchmark_0": cscv_result["psr_benchmark_0"],
            "psr_benchmark_05": cscv_result["psr_benchmark_05"],
            "bootstrap_ci": cscv_result["bootstrap_ci"],
            "walk_forward": cscv_result["walk_forward"],
        }
        json.dump(cscv_full, f, indent=2, default=str)
    print(f"  Saved: {cscv_json_path}")


if __name__ == "__main__":
    main()
