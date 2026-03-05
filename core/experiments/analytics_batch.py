"""
Analytics Batch Diagnostics for D55b/D54a validated baseline.
Computes 6 diagnostics, saves to analytics_batch.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from core.engine.simulator import (
    _read_registry, augment_features, load_data, select_features, REGISTRY_PATH,
)
from core.engine.evaluator import (
    build_trade_returns, compute_auc, simulate, walk_forward_train,
)
from core.engine.shap_runner import walk_forward_shap, aggregate_shap
from core.engine.sizing import equity_sim

OUTPUT_PATH = ROOT / "core" / "experiments" / "results" / "analytics_batch.json"
HOLDOUT_PATH = ROOT / "core" / "experiments" / "results" / "d55b_holdout.json"
ETF_DATE = pd.Timestamp("2024-01-10", tz="UTC")


# ===================================================================
# DIAGNOSTIC 1: DSR
# ===================================================================
def compute_dsr(experiments, holdout):
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014)."""
    # Collect annualized Sharpe from all experiments
    sharpes_ann = []
    for exp in experiments:
        m = exp.get("metrics", {})
        sr = m.get("sharpe_ann", 0)
        if sr != 0:
            sharpes_ann.append(sr)
    N_total = len(sharpes_ann)

    # Holdout per-trade stats
    hm = holdout["holdout_metrics"]
    T_trades = hm["n_trades"]  # 177
    wr = hm["win_rate"]
    ev = hm["ev_r"]
    r_win = 2.0 - 0.05   # 1.95
    r_loss = -(1.0 + 0.05)  # -1.05

    var_r = wr * (r_win - ev)**2 + (1 - wr) * (r_loss - ev)**2
    std_r = np.sqrt(var_r)
    sr_per_trade = ev / std_r if std_r > 0 else 0

    # Skewness and kurtosis of binary trade returns
    mu3 = wr * (r_win - ev)**3 + (1 - wr) * (r_loss - ev)**3
    mu4 = wr * (r_win - ev)**4 + (1 - wr) * (r_loss - ev)**4
    skew_val = mu3 / std_r**3 if std_r > 0 else 0
    kurt_val = mu4 / std_r**4 if std_r > 0 else 3

    gamma_em = 0.5772156649  # Euler-Mascheroni

    def e_max_z(n):
        if n <= 1:
            return 0.0
        z1 = stats.norm.ppf(1 - 1.0 / n)
        z2 = stats.norm.ppf(1 - 1.0 / (n * np.e))
        return (1 - gamma_em) * z1 + gamma_em * z2

    def dsr_value(sr_hat, sr_star, T, skew, kurt):
        denom_sq = 1 - skew * sr_hat + (kurt - 1) / 4 * sr_hat**2
        if denom_sq <= 0:
            return 0.0
        z = np.sqrt(T - 1) * (sr_hat - sr_star) / np.sqrt(denom_sq)
        return float(stats.norm.cdf(z))

    sensitivity = {}
    for n in sorted(set([N_total, 28, 55, 100])):
        if n < 1:
            continue
        e_max = e_max_z(n) / np.sqrt(T_trades)
        d = dsr_value(sr_per_trade, e_max, T_trades, skew_val, kurt_val)
        sensitivity[str(n)] = {
            "N": n,
            "E_max_SR_per_trade": round(float(e_max), 6),
            "DSR": round(d, 6),
        }

    result = {
        "method": "Bailey_LopezDePrado_2014",
        "N_experiments": N_total,
        "T_trades_holdout": T_trades,
        "sr_per_trade": round(float(sr_per_trade), 6),
        "std_r": round(float(std_r), 6),
        "skew": round(float(skew_val), 6),
        "kurtosis": round(float(kurt_val), 6),
        "sensitivity": sensitivity,
    }

    print(f"  N experiments: {N_total}")
    print(f"  Per-trade SR: {sr_per_trade:.4f}, Std(R): {std_r:.4f}")
    print(f"  Skew: {skew_val:.4f}, Kurt: {kurt_val:.4f}")
    for k, v in sensitivity.items():
        print(f"    N={v['N']:3d}: E[max]={v['E_max_SR_per_trade']:.4f}, DSR={v['DSR']:.4f}")
    return result


# ===================================================================
# DIAGNOSTIC 2: Treynor-Mazuy
# ===================================================================
def compute_treynor_mazuy(df, trade_indices, r_returns):
    """Treynor-Mazuy timing regression on WF OOS daily returns."""
    close = df["close"].values
    n_bars = len(df)

    # Build equity curve bar-by-bar
    equity = 100_000.0
    kelly_div = 40.0
    bar_eq = np.full(n_bars, np.nan)

    # Collect trade close events: (close_bar, r_return, entry_equity)
    events = {}
    eq = equity
    # Process trades in order -- no overlap since cooldown >> hold
    for idx, r in zip(trade_indices, r_returns):
        close_bar = idx + 48
        if close_bar < n_bars:
            risk = eq / kelly_div
            pnl = r * risk
            events[close_bar] = events.get(close_bar, 0.0) + pnl
            eq += pnl

    # Rebuild equity curve
    eq = equity
    # Find first OOS bar
    if len(trade_indices) == 0:
        print("  No trades -- skipping Treynor-Mazuy")
        return {"error": "no trades"}

    first_bar = max(0, trade_indices[0] - 100)
    for i in range(first_bar, n_bars):
        if i in events:
            eq += events[i]
            # Re-fix: events already added to eq above via the loop
            # Actually, we built events dict from a separate loop.
            # Let me redo this properly.
        pass

    # Redo properly: build equity curve from scratch
    eq = equity
    bar_eq = np.full(n_bars, np.nan)
    # Pre-compute all events
    event_map = {}
    running_eq = equity
    for idx, r in zip(trade_indices, r_returns):
        close_bar = idx + 48
        if close_bar < n_bars:
            risk = running_eq / kelly_div
            pnl = r * risk
            event_map[close_bar] = event_map.get(close_bar, 0.0) + pnl
            running_eq += pnl

    eq = equity
    first_trade = trade_indices[0]
    for i in range(first_trade, n_bars):
        if i in event_map:
            eq += event_map[i]
        bar_eq[i] = eq

    # Get dates
    ts_col = "bar_start_ts_utc"
    if ts_col not in df.columns:
        ts_col = "bar_end_ts_utc"
    dates = pd.to_datetime(df[ts_col])
    day_labels = dates.dt.date

    # Aggregate to daily: use end-of-day equity
    valid = ~np.isnan(bar_eq)
    daily_df = pd.DataFrame({
        "date": day_labels[valid],
        "equity": bar_eq[valid],
        "close": close[valid],
    })
    eod = daily_df.groupby("date").last()
    eod_eq = eod["equity"].values
    eod_btc = eod["close"].values

    # Daily returns (skip first day)
    r_strat = np.diff(eod_eq) / eod_eq[:-1]
    r_btc = np.diff(eod_btc) / eod_btc[:-1]

    # Remove NaN/inf
    mask = np.isfinite(r_strat) & np.isfinite(r_btc)
    r_s = r_strat[mask]
    r_b = r_btc[mask]

    # OLS: r_s = alpha + beta*r_b + gamma*r_b^2 + eps
    X = np.column_stack([np.ones(len(r_b)), r_b, r_b**2])
    try:
        betas, _, _, _ = np.linalg.lstsq(X, r_s, rcond=None)
        alpha, beta, gamma = betas

        y_hat = X @ betas
        resid = r_s - y_hat
        n = len(r_s)
        k = 3
        mse = np.sum(resid**2) / (n - k) if n > k else 0
        cov = mse * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(cov))

        t_stats = betas / se
        p_values = [float(2 * (1 - stats.t.cdf(abs(t), df=n - k))) for t in t_stats]

        ss_res = np.sum(resid**2)
        ss_tot = np.sum((r_s - r_s.mean())**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Simple CAPM
        X2 = np.column_stack([np.ones(len(r_b)), r_b])
        b2, _, _, _ = np.linalg.lstsq(X2, r_s, rcond=None)
        alpha_capm, beta_capm = b2

    except Exception as e:
        print(f"  OLS error: {e}")
        return {"error": str(e)}

    result = {
        "n_days": int(n),
        "alpha": round(float(alpha), 8),
        "beta": round(float(beta), 6),
        "gamma": round(float(gamma), 6),
        "alpha_se": round(float(se[0]), 8),
        "beta_se": round(float(se[1]), 6),
        "gamma_se": round(float(se[2]), 6),
        "alpha_t": round(float(t_stats[0]), 4),
        "beta_t": round(float(t_stats[1]), 4),
        "gamma_t": round(float(t_stats[2]), 4),
        "alpha_p": round(float(p_values[0]), 6),
        "beta_p": round(float(p_values[1]), 6),
        "gamma_p": round(float(p_values[2]), 6),
        "r_squared": round(float(r_squared), 6),
        "capm_alpha": round(float(alpha_capm), 8),
        "capm_beta": round(float(beta_capm), 6),
        "timing_significant_5pct": bool(p_values[2] < 0.05),
        "alpha_significant_5pct": bool(p_values[0] < 0.05),
    }

    print(f"  n_days = {n}")
    print(f"  alpha = {alpha:.6f} (t={t_stats[0]:.2f}, p={p_values[0]:.4f})")
    print(f"  beta  = {beta:.4f} (t={t_stats[1]:.2f}, p={p_values[1]:.4f})")
    print(f"  gamma = {gamma:.4f} (t={t_stats[2]:.2f}, p={p_values[2]:.4f})")
    print(f"  R^2   = {r_squared:.6f}")
    return result


# ===================================================================
# DIAGNOSTIC 3: Horizon Expiry
# ===================================================================
def compute_horizon_expiry(df, trade_indices, label_col):
    """Fraction of trades that expire at horizon without hitting barrier."""
    close = df["close"].values
    high_arr = df["high"].values
    low_arr = df["low"].values
    n_bars = len(df)
    is_long = "long" in label_col
    horizon = 48

    # Compute ATR(14) from OHLC
    h = high_arr.copy()
    l = low_arr.copy()
    c = close.copy()
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    # SMA ATR(14)
    atr = pd.Series(tr).rolling(14, min_periods=1).mean().values

    n_target = 0
    n_stop = 0
    n_expired = 0
    n_total = 0

    for idx in trade_indices:
        entry = close[idx]
        stop_dist = atr[idx]
        if stop_dist <= 0 or np.isnan(stop_dist):
            continue

        if is_long:
            target_px = entry + 2 * stop_dist
            stop_px = entry - 1 * stop_dist
        else:
            target_px = entry - 2 * stop_dist
            stop_px = entry + 1 * stop_dist

        n_total += 1
        hit = False
        end = min(idx + horizon + 1, n_bars)

        for j in range(idx + 1, end):
            if is_long:
                if low_arr[j] <= stop_px:
                    n_stop += 1
                    hit = True
                    break
                if high_arr[j] >= target_px:
                    n_target += 1
                    hit = True
                    break
            else:
                if high_arr[j] >= stop_px:
                    n_stop += 1
                    hit = True
                    break
                if low_arr[j] <= target_px:
                    n_target += 1
                    hit = True
                    break
        if not hit:
            n_expired += 1

    result = {
        "n_trades_analyzed": n_total,
        "n_target_hit": n_target,
        "n_stop_hit": n_stop,
        "n_expired": n_expired,
        "pct_target_hit": round(n_target / n_total * 100, 2) if n_total else 0,
        "pct_stop_hit": round(n_stop / n_total * 100, 2) if n_total else 0,
        "pct_expired": round(n_expired / n_total * 100, 2) if n_total else 0,
    }

    print(f"  Trades: {n_total}")
    print(f"  Target hit: {n_target} ({result['pct_target_hit']}%)")
    print(f"  Stop hit:   {n_stop} ({result['pct_stop_hit']}%)")
    print(f"  Expired:    {n_expired} ({result['pct_expired']}%)")
    return result


# ===================================================================
# DIAGNOSTIC 4: SHAP Rank Correlation
# ===================================================================
def compute_shap_rank_corr(features, shap_per_fold):
    """Spearman rank correlation of feature importance across folds."""
    n_folds = len(shap_per_fold)
    n_features = len(features)

    fold_mean_abs = np.zeros((n_folds, n_features))
    for fi, sv in enumerate(shap_per_fold):
        fold_mean_abs[fi] = np.abs(sv).mean(axis=0)

    # Per-fold rankings (1 = most important)
    fold_rankings = []
    for fi in range(n_folds):
        ranks = stats.rankdata(-fold_mean_abs[fi]).astype(int)
        fold_rankings.append(ranks)

    # Pairwise Spearman
    rho_matrix = np.eye(n_folds)
    for i in range(n_folds):
        for j in range(i + 1, n_folds):
            rho, _ = stats.spearmanr(fold_rankings[i], fold_rankings[j])
            rho_matrix[i, j] = rho
            rho_matrix[j, i] = rho

    off_diag = rho_matrix[np.triu_indices(n_folds, k=1)]

    # Most stable features (lowest CV across folds)
    fold_stds = fold_mean_abs.std(axis=0)
    fold_means = fold_mean_abs.mean(axis=0)
    cv = np.where(fold_means > 0, fold_stds / fold_means, 999.0)
    stable_idx = np.argsort(cv)[:10]
    stable = [
        {"feature": features[i], "cv": round(float(cv[i]), 4),
         "mean_abs_shap": round(float(fold_means[i]), 6)}
        for i in stable_idx
    ]

    result = {
        "n_folds": n_folds,
        "n_features": n_features,
        "mean_rho": round(float(off_diag.mean()), 4),
        "median_rho": round(float(np.median(off_diag)), 4),
        "min_rho": round(float(off_diag.min()), 4),
        "max_rho": round(float(off_diag.max()), 4),
        "std_rho": round(float(off_diag.std()), 4),
        "n_pairs": int(len(off_diag)),
        "most_stable_features": stable,
    }

    print(f"  Mean rho: {off_diag.mean():.4f} +/- {off_diag.std():.4f}")
    print(f"  Range: [{off_diag.min():.4f}, {off_diag.max():.4f}]")
    print(f"  Most stable: {stable[0]['feature']} (CV={stable[0]['cv']:.4f})")
    return result


# ===================================================================
# DIAGNOSTIC 5: Pre/Post ETF Split
# ===================================================================
def compute_etf_split(df, trade_indices, r_returns, config):
    """Split WF OOS performance by BTC spot ETF approval date."""
    ts_col = "bar_start_ts_utc"
    if ts_col not in df.columns:
        ts_col = "bar_end_ts_utc"
    dates = pd.to_datetime(df[ts_col])

    pre_idx = [i for i, idx in enumerate(trade_indices) if dates.iloc[idx] < ETF_DATE]
    post_idx = [i for i, idx in enumerate(trade_indices) if dates.iloc[idx] >= ETF_DATE]

    pre_r = r_returns[pre_idx] if pre_idx else np.array([])
    post_r = r_returns[post_idx] if post_idx else np.array([])

    def trade_metrics(r_arr):
        if len(r_arr) == 0:
            return {"n_trades": 0, "win_rate": 0, "ev_r": 0,
                    "profit_factor": 0, "per_trade_sr": 0}
        n = len(r_arr)
        wr = float((r_arr > 0).mean())
        ev = float(r_arr.mean())
        gw = float(r_arr[r_arr > 0].sum())
        gl = float(abs(r_arr[r_arr < 0].sum()))
        pf = gw / gl if gl > 0 else float("inf")
        std = float(r_arr.std(ddof=1)) if n > 1 else 0
        sr = ev / std if std > 0 else 0
        return {"n_trades": n, "win_rate": round(wr, 4), "ev_r": round(ev, 4),
                "profit_factor": round(pf, 4), "per_trade_sr": round(sr, 4)}

    # Map folds to periods
    embargo = config.get("embargo_bars", 288)
    min_train = config.get("min_train_bars", 105_000)
    test_fold_bars = config.get("test_fold_bars", 52_500)
    n = len(df)
    folds = []
    t_start = min_train + embargo
    fi = 0
    while t_start < n:
        t_end = min(t_start + test_fold_bars, n)
        sd = dates.iloc[t_start]
        ed = dates.iloc[min(t_end - 1, n - 1)]
        period = "pre_etf" if ed < ETF_DATE else (
            "post_etf" if sd >= ETF_DATE else "mixed")
        folds.append({
            "fold": fi + 1, "start_date": str(sd.date()),
            "end_date": str(ed.date()), "period": period,
        })
        t_start = t_end
        fi += 1

    result = {
        "etf_date": "2024-01-10",
        "pre_etf": trade_metrics(pre_r),
        "post_etf": trade_metrics(post_r),
        "fold_periods": folds,
    }

    print(f"  Pre-ETF:  {result['pre_etf']['n_trades']} trades, "
          f"WR={result['pre_etf']['win_rate']:.4f}, "
          f"EV={result['pre_etf']['ev_r']:.4f}")
    print(f"  Post-ETF: {result['post_etf']['n_trades']} trades, "
          f"WR={result['post_etf']['win_rate']:.4f}, "
          f"EV={result['post_etf']['ev_r']:.4f}")
    return result


# ===================================================================
# DIAGNOSTIC 6: Buy-and-Hold
# ===================================================================
def compute_buy_and_hold(holdout):
    """Compare strategy vs buy-and-hold on holdout period."""
    hm = holdout["holdout_metrics"]
    btc = hm["btc_trend"]

    strat_ret = (hm["final_equity"] - 100_000) / 100_000 * 100
    btc_ret = btc["change_pct"]
    years = hm["holdout_years"]
    strat_ann = ((1 + strat_ret / 100) ** (1 / years) - 1) * 100 if years > 0 else 0
    btc_ann = btc["ann_return_pct"]
    btc_vol = btc["ann_vol_pct"]
    btc_sr = btc_ann / btc_vol if btc_vol > 0 else 0

    result = {
        "holdout_period": f"{btc['start_date']} to {btc['end_date']}",
        "holdout_years": years,
        "btc_regime": btc["regime"],
        "strategy": {
            "return_pct": round(strat_ret, 2),
            "ann_return_pct": round(strat_ann, 2),
            "daily_sharpe_ann": round(hm["daily_sharpe"], 4),
            "lo_adj_sharpe_ann": round(hm.get("daily_sharpe_lo_adj", hm["daily_sharpe"]), 4),
            "max_dd_pct": hm["max_dd_pct"],
            "n_trades": hm["n_trades"],
        },
        "btc_buy_and_hold": {
            "return_pct": round(btc_ret, 2),
            "ann_return_pct": round(btc_ann, 2),
            "ann_vol_pct": round(btc_vol, 2),
            "sharpe_ann": round(btc_sr, 4),
        },
        "excess_return_pct": round(strat_ret - btc_ret, 2),
    }

    print(f"  Strategy: {strat_ret:+.1f}% ({strat_ann:+.1f}% ann)")
    print(f"  BTC B&H:  {btc_ret:+.1f}% ({btc_ann:+.1f}% ann)")
    print(f"  Excess:   {strat_ret - btc_ret:+.1f}pp")
    print(f"  Sharpe:   {hm['daily_sharpe']:.2f} vs {btc_sr:.4f}")
    return result


# ===================================================================
# MAIN
# ===================================================================
def main():
    t0 = time.time()
    results = {}

    print("=" * 60)
    print("  ANALYTICS BATCH DIAGNOSTICS")
    print("=" * 60)

    # --- Load registry + holdout ---
    reg = _read_registry()
    experiments = reg["experiments"]
    with open(HOLDOUT_PATH) as f:
        holdout = json.load(f)

    # Find D55b config
    d55b = None
    for exp in experiments:
        if exp["id"] == "D55b_tier1_only":
            d55b = exp
            break
    if d55b is None:
        raise RuntimeError("D55b_tier1_only not found in registry")

    feature_exclude = d55b.get("feature_exclude", [])
    label_col = d55b.get("label", "label_long_hit_2r_48c")
    threshold = d55b.get("threshold", 0.6)
    cooldown = d55b.get("cooldown_bars", 576)

    # --- DIAGNOSTIC 1: DSR (no data load needed) ---
    print("\n--- DIAGNOSTIC 1: DSR ---")
    results["dsr"] = compute_dsr(experiments, holdout)

    # --- DIAGNOSTIC 6: Buy-and-Hold (no data load needed) ---
    print("\n--- DIAGNOSTIC 6: Buy-and-Hold ---")
    results["buy_and_hold"] = compute_buy_and_hold(holdout)

    # --- Load train data ---
    print("\n--- Loading train data ---")
    df, version = load_data()
    df = augment_features(df)
    features = select_features(df, "all", feature_exclude)
    print(f"  Label: {label_col}, Features: {len(features)}")
    print(f"  Threshold: {threshold}, Cooldown: {cooldown}")

    # --- Walk-forward SHAP (for Diagnostics 2,3,4,5) ---
    print("\n--- Walk-forward SHAP ({} folds, {} features) ---".format(9, len(features)))
    wf_config = {
        "embargo_bars": 288,
        "min_train_bars": 105_000,
        "test_fold_bars": 52_500,
        "device": "gpu",
        "n_estimators": 1000,
        "early_stop_rounds": 50,
    }
    oos_probs, fold_results, shap_per_fold = walk_forward_shap(
        df, features, label_col, wf_config
    )

    # Simulate trades from OOS probs
    raw_labels = df[label_col].values
    label_valid = ~np.isnan(raw_labels)
    label_arr = np.where(label_valid, raw_labels, 0).astype(int)
    oos_covered = ~np.isnan(oos_probs)
    signal_mask = (oos_probs >= threshold) & oos_covered & label_valid
    trade_indices = simulate(signal_mask, label_arr, cooldown)
    r_returns = build_trade_returns(trade_indices, label_arr)

    print(f"\n  OOS trades: {len(trade_indices)}, "
          f"WR: {(r_returns > 0).mean():.4f}, "
          f"EV: {r_returns.mean():.4f}")

    # --- DIAGNOSTIC 2: Treynor-Mazuy ---
    print("\n--- DIAGNOSTIC 2: Treynor-Mazuy ---")
    results["treynor_mazuy"] = compute_treynor_mazuy(df, trade_indices, r_returns)

    # --- DIAGNOSTIC 3: Horizon Expiry ---
    print("\n--- DIAGNOSTIC 3: Horizon Expiry ---")
    results["horizon_expiry"] = compute_horizon_expiry(
        df, trade_indices, label_col
    )

    # --- DIAGNOSTIC 4: SHAP Rank Correlation ---
    print("\n--- DIAGNOSTIC 4: SHAP Rank Correlation ---")
    results["shap_rank_correlation"] = compute_shap_rank_corr(
        features, shap_per_fold
    )

    # --- DIAGNOSTIC 5: Pre/Post ETF ---
    print("\n--- DIAGNOSTIC 5: Pre/Post ETF ---")
    results["pre_post_etf"] = compute_etf_split(
        df, trade_indices, r_returns, wf_config
    )

    # --- Save ---
    results["metadata"] = {
        "generated": pd.Timestamp.now().isoformat(),
        "base_experiment": "D55b_tier1_only",
        "holdout_experiment": "D55b_holdout",
        "n_features": len(features),
        "n_oos_trades": len(trade_indices),
        "oos_wr": round(float((r_returns > 0).mean()), 4),
        "oos_ev": round(float(r_returns.mean()), 4),
        "elapsed_s": round(time.time() - t0, 1),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n  Saved: {OUTPUT_PATH}")
    print(f"  Elapsed: {elapsed:.1f}s")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    dsr = results["dsr"]
    n_key = str(dsr["N_experiments"])
    if n_key in dsr["sensitivity"]:
        print(f"  DSR (N={dsr['N_experiments']}): "
              f"{dsr['sensitivity'][n_key]['DSR']:.4f}")

    tm = results["treynor_mazuy"]
    if "alpha" in tm:
        print(f"  T-M alpha={tm['alpha']:.6f} (p={tm['alpha_p']:.4f}), "
              f"gamma={tm['gamma']:.4f} (p={tm['gamma_p']:.4f})")

    he = results["horizon_expiry"]
    print(f"  Expiry: {he['pct_expired']:.1f}% | "
          f"Target: {he['pct_target_hit']:.1f}% | "
          f"Stop: {he['pct_stop_hit']:.1f}%")

    sr = results["shap_rank_correlation"]
    print(f"  SHAP rho: {sr['mean_rho']:.4f} "
          f"[{sr['min_rho']:.4f}, {sr['max_rho']:.4f}]")

    etf = results["pre_post_etf"]
    print(f"  Pre-ETF WR: {etf['pre_etf']['win_rate']:.4f} | "
          f"Post-ETF WR: {etf['post_etf']['win_rate']:.4f}")

    bh = results["buy_and_hold"]
    print(f"  Strategy vs B&H: {bh['excess_return_pct']:+.1f}pp")


if __name__ == "__main__":
    main()
