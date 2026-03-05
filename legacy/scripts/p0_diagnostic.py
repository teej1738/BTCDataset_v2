# p0_diagnostic.py -- P0 Diagnostic: Daily Sharpe + Buy-and-Hold Comparison
# Uses saved D54a_baseline_long OOS probabilities (no retraining needed).
# ASCII-only output for cp1252 compatibility.

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths and Parameters (from D54a registry entry)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "data" / "labeled" / "BTCUSDT_5m_labeled_v3_train.parquet"
PROBS_PATH = ROOT / "core" / "experiments" / "models" / "D54a_baseline_long_oos_probs_cal.npy"

LABEL_COL = "label_long_hit_2r_48c"
THRESHOLD = 0.60
COOLDOWN = 576
R_TARGET = 2
COST_PER_R = 0.05
EMBARGO = 288
MIN_TRAIN = 105_000
TEST_FOLD = 52_500
RISK_PCT = 0.02  # Fixed 2% risk for system return comparison

R_WIN = R_TARGET - COST_PER_R     # +1.95
R_LOSS = -(1 + COST_PER_R)       # -1.05


def main():
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("Loading data...")
    df = pd.read_parquet(TRAIN_PATH)
    print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} cols")

    oos_probs = np.load(PROBS_PATH)
    n_covered = int(np.sum(~np.isnan(oos_probs)))
    print(f"  OOS probs: {len(oos_probs):,} total, {n_covered:,} covered")

    # Labels
    raw_labels = df[LABEL_COL].values
    label_valid = ~np.isnan(raw_labels)
    label_arr = np.where(label_valid, raw_labels, 0).astype(int)

    # Timestamps
    ts = df["bar_start_ts_utc"].values  # datetime64[ms, UTC]

    # ------------------------------------------------------------------
    # 2. Reconstruct fold boundaries (same logic as evaluator.py)
    # ------------------------------------------------------------------
    n = len(df)
    folds = []
    t_start = MIN_TRAIN + EMBARGO
    while t_start < n:
        t_end = min(t_start + TEST_FOLD, n)
        folds.append((t_start, t_end))
        t_start = t_end
    print(f"  Folds: {len(folds)}")

    # ------------------------------------------------------------------
    # 3. Simulate with cooldown (reproduce D54a exactly)
    # ------------------------------------------------------------------
    covered = ~np.isnan(oos_probs)
    signal_mask = covered & label_valid & (oos_probs >= THRESHOLD)

    trade_indices = []
    bars_since = COOLDOWN
    for i in range(len(signal_mask)):
        bars_since += 1
        if bars_since > COOLDOWN and signal_mask[i]:
            trade_indices.append(i)
            bars_since = 0

    trade_indices = np.array(trade_indices)
    n_trades = len(trade_indices)
    print(f"  Total trades: {n_trades}")

    # Build R-returns
    r_returns = np.array(
        [R_WIN if label_arr[i] == 1 else R_LOSS for i in trade_indices]
    )

    # ------------------------------------------------------------------
    # 4. Date mapping
    # ------------------------------------------------------------------
    trade_ts = pd.to_datetime(ts[trade_indices])
    trade_dates = trade_ts.normalize()  # midnight of each trade day

    # OOS coverage dates
    covered_idx = np.where(covered)[0]
    first_covered = covered_idx[0]
    last_covered = covered_idx[-1]
    first_date = pd.Timestamp(ts[first_covered]).normalize()
    last_date = pd.Timestamp(ts[last_covered]).normalize()
    all_dates = pd.date_range(first_date, last_date, freq="D")
    years_oos = (last_date - first_date).total_seconds() / (365.25 * 86400)

    print(f"  OOS period: {first_date.date()} to {last_date.date()} "
          f"({years_oos:.2f} years)")

    # ------------------------------------------------------------------
    # TASK 1: Daily Sharpe
    # ------------------------------------------------------------------
    # Daily R-PnL (sum of R-returns per calendar day, 0 on no-trade days)
    trade_r_series = pd.Series(r_returns, index=trade_dates)
    daily_r_pnl = trade_r_series.groupby(trade_r_series.index).sum()
    daily_r_full = daily_r_pnl.reindex(all_dates, fill_value=0.0)

    # Daily Sharpe (all calendar days, including zeros)
    d_mean = float(daily_r_full.mean())
    d_std = float(daily_r_full.std(ddof=1))
    daily_sharpe_all = (d_mean / d_std * np.sqrt(252)) if d_std > 0 else 0.0

    # Daily Sharpe (trading days only -- exclude zero-PnL days)
    trading_days = daily_r_full[daily_r_full != 0.0]
    n_trading_days = len(trading_days)
    if n_trading_days > 1:
        td_mean = float(trading_days.mean())
        td_std = float(trading_days.std(ddof=1))
        daily_sharpe_trading = (td_mean / td_std * np.sqrt(252)) if td_std > 0 else 0.0
    else:
        daily_sharpe_trading = 0.0

    # Per-trade Sharpe (original method from evaluator.py)
    if n_trades > 1:
        sr_pt = float(r_returns.mean() / np.std(r_returns, ddof=1))
        tpy = n_trades / years_oos
        sharpe_per_trade = sr_pt * np.sqrt(tpy)
    else:
        sharpe_per_trade = 0.0

    # ------------------------------------------------------------------
    # TASK 2: Per-fold analysis (system vs buy-and-hold)
    # ------------------------------------------------------------------
    fold_results = []

    for fold_i, (test_start, test_end) in enumerate(folds):
        # Trades in this fold
        in_fold = (trade_indices >= test_start) & (trade_indices < test_end)
        fold_trade_idx = np.where(in_fold)[0]
        fold_returns = r_returns[fold_trade_idx]
        fold_n = len(fold_returns)

        # Date range for this fold
        fold_first = pd.Timestamp(ts[test_start]).normalize()
        fold_last = pd.Timestamp(ts[test_end - 1]).normalize()
        fold_all_dates = pd.date_range(fold_first, fold_last, freq="D")

        # System daily R-PnL for this fold
        if fold_n > 0:
            fold_trade_ts = trade_dates[fold_trade_idx]
            fold_r_series = pd.Series(
                fold_returns, index=fold_trade_ts
            )
            fold_daily_r = fold_r_series.groupby(fold_r_series.index).sum()
            fold_daily_full = fold_daily_r.reindex(fold_all_dates, fill_value=0.0)
        else:
            fold_daily_full = pd.Series(0.0, index=fold_all_dates)

        # Fold daily Sharpe (system)
        fds = float(fold_daily_full.std(ddof=1))
        if fds > 0 and len(fold_daily_full) > 1:
            fold_sys_sharpe = float(fold_daily_full.mean()) / fds * np.sqrt(252)
        else:
            fold_sys_sharpe = 0.0

        # System compound return (fixed 2% risk per trade)
        sys_equity = 1.0
        for r in fold_returns:
            sys_equity *= (1.0 + r * RISK_PCT)
        sys_return = sys_equity - 1.0

        # BTC buy-and-hold return
        btc_start_price = float(df["close"].iloc[test_start])
        btc_end_price = float(df["close"].iloc[test_end - 1])
        bh_return = (btc_end_price - btc_start_price) / btc_start_price

        # BTC daily Sharpe for this fold
        btc_close_fold = df["close"].iloc[test_start:test_end]
        btc_daily_close = btc_close_fold.set_axis(
            pd.to_datetime(ts[test_start:test_end])
        ).resample("D").last().dropna()
        btc_daily_ret = btc_daily_close.pct_change().dropna()
        if len(btc_daily_ret) > 1 and float(btc_daily_ret.std()) > 0:
            bh_sharpe = float(btc_daily_ret.mean() / btc_daily_ret.std(ddof=1)
                              * np.sqrt(252))
        else:
            bh_sharpe = 0.0

        winner = "SYSTEM" if sys_return > bh_return else "BUY-HOLD"

        fold_results.append({
            "fold": fold_i + 1,
            "n_trades": fold_n,
            "sys_return": sys_return,
            "bh_return": bh_return,
            "winner": winner,
            "sys_sharpe": fold_sys_sharpe,
            "bh_sharpe": bh_sharpe,
            "date_range": f"{fold_first.date()} to {fold_last.date()}",
            "btc_start": btc_start_price,
            "btc_end": btc_end_price,
        })

    # BTC overall daily Sharpe
    btc_close_oos = df["close"].iloc[first_covered:last_covered + 1]
    btc_ts_oos = pd.to_datetime(ts[first_covered:last_covered + 1])
    btc_daily_oos = btc_close_oos.set_axis(btc_ts_oos).resample("D").last().dropna()
    btc_daily_ret_oos = btc_daily_oos.pct_change().dropna()
    if len(btc_daily_ret_oos) > 1:
        bh_daily_sharpe_overall = float(
            btc_daily_ret_oos.mean() / btc_daily_ret_oos.std(ddof=1) * np.sqrt(252)
        )
    else:
        bh_daily_sharpe_overall = 0.0

    # ------------------------------------------------------------------
    # Print Results
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("  P0 DIAGNOSTIC: D54a_baseline_long")
    print("=" * 70)

    print()
    print("--- TASK 1: Sharpe Ratio Comparison ---")
    print()
    print(f"  Per-trade Sharpe (current method):    {sharpe_per_trade:8.2f}")
    print(f"  Daily Sharpe (all days, incl zeros):  {daily_sharpe_all:8.2f}")
    print(f"  Daily Sharpe (trading days only):     {daily_sharpe_trading:8.2f}")
    print(f"  BTC Buy-Hold Daily Sharpe:            {bh_daily_sharpe_overall:8.2f}")
    print()
    print(f"  Total trades:       {n_trades}")
    print(f"  Trading days:       {n_trading_days}")
    print(f"  Calendar days:      {len(daily_r_full)}")
    print(f"  OOS span:           {years_oos:.2f} years")
    print(f"  Mean daily R-PnL:   {d_mean:+.4f}")
    print(f"  Std daily R-PnL:    {d_std:.4f}")

    print()
    print("--- Per-Fold Daily Sharpe ---")
    print()
    hdr = f"  {'Fold':>4s}  {'N':>4s}  {'Sys SR':>8s}  {'BH SR':>8s}  {'Date Range'}"
    print(hdr)
    print(f"  {'----':>4s}  {'----':>4s}  {'------':>8s}  {'------':>8s}  {'----------'}")
    for fr in fold_results:
        print(f"  {fr['fold']:4d}  {fr['n_trades']:4d}  "
              f"{fr['sys_sharpe']:8.2f}  {fr['bh_sharpe']:8.2f}  "
              f"{fr['date_range']}")

    print()
    print("--- TASK 2: Buy-and-Hold Comparison Per Fold (2% risk/trade) ---")
    print()
    hdr2 = (f"  {'Fold':>4s}  {'N':>4s}  {'Sys Ret':>9s}  "
            f"{'BH Ret':>9s}  {'Winner':>10s}  {'BTC Range'}")
    print(hdr2)
    sep2 = (f"  {'----':>4s}  {'----':>4s}  {'-------':>9s}  "
            f"{'------':>9s}  {'------':>10s}  {'---------'}")
    print(sep2)
    for fr in fold_results:
        btc_rng = f"${fr['btc_start']:,.0f} -> ${fr['btc_end']:,.0f}"
        print(f"  {fr['fold']:4d}  {fr['n_trades']:4d}  "
              f"{fr['sys_return']:+8.1%}  {fr['bh_return']:+8.1%}  "
              f"{fr['winner']:>10s}  {btc_rng}")

    n_system_wins = sum(1 for fr in fold_results if fr["winner"] == "SYSTEM")
    n_folds = len(fold_results)
    print()
    print(f"  System beats BH in {n_system_wins}/{n_folds} folds")

    # ------------------------------------------------------------------
    # Overall BH comparison
    # ------------------------------------------------------------------
    btc_first = float(df["close"].iloc[first_covered])
    btc_last = float(df["close"].iloc[last_covered])
    bh_total = (btc_last - btc_first) / btc_first

    sys_equity_total = 1.0
    for r in r_returns:
        sys_equity_total *= (1.0 + r * RISK_PCT)
    sys_total = sys_equity_total - 1.0

    print()
    print(f"  Overall (entire OOS period):")
    print(f"    System return (2% risk): {sys_total:+.1%}")
    print(f"    BTC buy-and-hold:        {bh_total:+.1%}")
    print(f"    BTC range: ${btc_first:,.0f} -> ${btc_last:,.0f}")

    # ------------------------------------------------------------------
    # TASK 3: Decision Table
    # ------------------------------------------------------------------
    ds = daily_sharpe_all
    print()
    print("=" * 70)

    if ds >= 1.0 and n_system_wins >= 6:
        print()
        print("  RESULT: VIABLE BASELINE")
        print(f"  Daily Sharpe: {ds:.2f}")
        print(f"  Beats BH in {n_system_wins}/{n_folds} folds")
        print("  Recommendation: Freeze current system. Proceed with redesign.")
        print("  The current system is the benchmark to beat.")
    elif (0.5 <= ds < 1.0) or (4 <= n_system_wins <= 5):
        print()
        print("  RESULT: MARGINAL")
        print(f"  Daily Sharpe: {ds:.2f}")
        print(f"  Beats BH in {n_system_wins}/{n_folds} folds")
        print("  Recommendation: Pause optimization. Redesign is a refinement.")
    else:
        print()
        print("  RESULT: NO DEMONSTRATED EDGE")
        print(f"  Daily Sharpe: {ds:.2f}")
        print(f"  Beats BH in {n_system_wins}/{n_folds} folds")
        print("  Recommendation: Pause everything. Redesign is a rebuild.")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
