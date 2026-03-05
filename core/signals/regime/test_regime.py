# core/signals/regime/test_regime.py
# D47 -- Causality tests + coverage stats for regime filter features.
# ASCII-only output for cp1252 compatibility.
#
# Usage:
#   python core/signals/regime/test_regime.py

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))


def causality_test(
    func,
    df: pd.DataFrame,
    T: int,
    feature_cols: list[str],
) -> bool:
    """Test that features at bar T are identical when computed on [0..T] vs [0..T+k].

    If feature[T] changes when future bars are added, the computation leaks
    future data (non-causal).
    """
    # Compute on [0..T]
    df_short = df.iloc[:T + 1].copy()
    result_short = func(df_short)

    # Compute on [0..T+500] (or full df if shorter)
    end = min(T + 500, len(df))
    df_long = df.iloc[:end].copy()
    result_long = func(df_long)

    # Compare feature values at bar T
    all_pass = True
    for col in feature_cols:
        if col not in result_short.columns or col not in result_long.columns:
            # Column might not exist in short slice (e.g. not enough warmup)
            continue
        val_short = result_short[col].iloc[T] if T < len(result_short) else np.nan
        val_long = result_long[col].iloc[T] if T < len(result_long) else np.nan

        # Both NaN = pass
        if np.isnan(val_short) and np.isnan(val_long):
            continue
        # One NaN = fail
        if np.isnan(val_short) != np.isnan(val_long):
            print(f"    FAIL: {col} at T={T}: short={val_short}, long={val_long}")
            all_pass = False
            continue
        # Value mismatch
        if abs(val_short - val_long) > 1e-8:
            print(f"    FAIL: {col} at T={T}: short={val_short:.8f}, long={val_long:.8f}")
            all_pass = False

    return all_pass


def test_hmm_causality(df: pd.DataFrame, test_points: list[int]) -> bool:
    """Test HMM regime probabilities for causality."""
    from core.signals.regime.hmm_filter import compute_hmm_regime

    hmm_cols = ["hmm_prob_bull", "hmm_prob_bear", "hmm_prob_calm"]
    all_pass = True

    for T in test_points:
        if T >= len(df):
            continue
        passed = causality_test(compute_hmm_regime, df, T, hmm_cols)
        status = "PASS" if passed else "FAIL"
        print(f"  HMM causality T={T:6d}: {status}")
        if not passed:
            all_pass = False

    return all_pass


def test_adx_composite_causality(df: pd.DataFrame, test_points: list[int]) -> bool:
    """Test ADX composite features for causality."""
    from core.signals.regime.hmm_filter import compute_adx_composite

    adx_cols = ["bb_width_normalized", "atr_percentile_rank", "regime_tag"]
    all_pass = True

    for T in test_points:
        if T >= len(df):
            continue
        passed = causality_test(compute_adx_composite, df, T, adx_cols)
        status = "PASS" if passed else "FAIL"
        print(f"  ADX causality T={T:6d}: {status}")
        if not passed:
            all_pass = False

    return all_pass


def test_interactions_causality(df: pd.DataFrame, test_points: list[int]) -> bool:
    """Test interaction features for causality.

    Interactions depend on HMM and ADX, so we need to compute those first.
    """
    from core.signals.regime.hmm_filter import (
        compute_adx_composite,
        compute_hmm_regime,
        compute_regime_interactions,
    )

    interact_cols = ["ob_bull_age_x_hmm_bull", "fvg_bull_x_trending", "ote_x_regime"]
    all_pass = True

    def compute_interactions_full(df_slice):
        """Compute HMM + ADX + interactions on a slice."""
        hmm = compute_hmm_regime(df_slice)
        for col in hmm.columns:
            df_slice[col] = hmm[col].values
        adx = compute_adx_composite(df_slice)
        for col in adx.columns:
            if col not in df_slice.columns:
                df_slice[col] = adx[col].values
        return compute_regime_interactions(df_slice)

    for T in test_points:
        if T >= len(df):
            continue
        passed = causality_test(compute_interactions_full, df, T, interact_cols)
        status = "PASS" if passed else "FAIL"
        print(f"  Interactions causality T={T:6d}: {status}")
        if not passed:
            all_pass = False

    return all_pass


def print_coverage(df: pd.DataFrame) -> None:
    """Print coverage statistics for all regime features."""
    from core.signals.regime.hmm_filter import compute_all_regime_features

    print("\n  Computing all regime features on full dataset ...")
    result = compute_all_regime_features(df)

    print(f"\n  Coverage ({len(df):,} bars):")
    print(f"  {'Feature':<30s} {'Non-NaN':>10s} {'Coverage':>10s} {'Mean':>12s} {'Std':>12s}")
    print("  " + "-" * 76)
    for col in result.columns:
        valid = result[col].notna()
        n_valid = valid.sum()
        pct = 100 * n_valid / len(df)
        mean_val = result[col][valid].mean() if n_valid > 0 else np.nan
        std_val = result[col][valid].std() if n_valid > 0 else np.nan
        print(f"  {col:<30s} {n_valid:>10,d} {pct:>9.1f}% {mean_val:>12.6f} {std_val:>12.6f}")


def main():
    print()
    print("=" * 60)
    print("  REGIME FILTER TESTS (D47)")
    print("=" * 60)

    # Load data
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    parquet = project_root / "data" / "labeled" / "BTCUSDT_5m_labeled_v3.parquet"
    cols_needed = [
        "close", "high", "low", "adx_14", "bb_bandwidth",
        "ict_atr_14", "ict_ob_bull_age",
        "ict_fvg_bull_recent_age", "ote_dist",
    ]
    print(f"\n  Loading {parquet.name} ...")
    df = pd.read_parquet(parquet, columns=cols_needed)
    print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} cols")

    # Causality test points -- must be large enough for HMM warmup (252 days = 72576 bars)
    # Also need bars_per_day * warmup_days worth of data minimum
    test_points = [1000, 5000, 10000, 50000]

    # For HMM causality, we need larger T values (HMM warmup = 252 * 288 = 72576)
    hmm_test_points = [80000, 100000, 200000, 400000]

    print("\n  --- ADX Composite Causality ---")
    adx_pass = test_adx_composite_causality(df, test_points)

    print("\n  --- HMM Causality ---")
    hmm_pass = test_hmm_causality(df, hmm_test_points)

    print("\n  --- Interaction Causality ---")
    interact_pass = test_interactions_causality(df, hmm_test_points)

    # Coverage on full dataset
    print_coverage(df)

    # Summary
    n_pass = sum([adx_pass, hmm_pass, interact_pass])
    n_total = 3
    print(f"\n  RESULT: {n_pass}/{n_total} test groups PASS")

    if n_pass == n_total:
        print("  ALL CAUSALITY TESTS PASS")
    else:
        print("  *** FAILURES DETECTED ***")
        if not adx_pass:
            print("    - ADX composite FAILED")
        if not hmm_pass:
            print("    - HMM FAILED")
        if not interact_pass:
            print("    - Interactions FAILED")

    print()
    print("=" * 60)
    return n_pass == n_total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
