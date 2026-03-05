# core/engine/test_d46b.py
# D46b smoke test: dynamic labeler integration + fill model + tier gates.
# Run from project root: python core/engine/test_d46b.py

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Test 1: Dynamic labeler produces valid labels matching parquet
# ---------------------------------------------------------------------------
def test_dynamic_labeler_match():
    """Compare dynamic ATR-stop market labels against parquet labels."""
    print("=== TEST 1: Dynamic labeler vs parquet label ===")
    from core.engine.labeler import compute_labels

    DATA = "data/labeled/BTCUSDT_5m_labeled_v3.parquet"
    COLS = ["open", "high", "low", "close", "ict_atr_14",
            "ict_swing_low_price", "ict_swing_high_price",
            "label_long_hit_2r_48c"]
    NROWS = 10000

    df = pd.read_parquet(DATA, columns=COLS)
    df = df.iloc[:NROWS].copy().reset_index(drop=True)

    # Dynamic label with ATR stop (should approximate parquet label)
    cfg = {
        "direction": "long",
        "target_r": 2.0,
        "stop_type": "atr",
        "stop_atr_mult": 1.5,
        "max_bars": 48,
        "entry_type": "market",
        "entry_delay": 0,
    }
    dyn = compute_labels(df, cfg)
    static = df["label_long_hit_2r_48c"].values

    # Both valid
    both_valid = ~np.isnan(dyn) & ~np.isnan(static)
    n_valid = int(both_valid.sum())
    agreement = int(np.sum(dyn[both_valid] == static[both_valid]))
    pct = agreement / n_valid * 100 if n_valid > 0 else 0

    print(f"  Both valid: {n_valid}")
    print(f"  Agreement:  {agreement} ({pct:.1f}%)")
    print(f"  Dynamic NaN: {int(np.isnan(dyn).sum())}")
    print(f"  Static NaN:  {int(np.isnan(static).sum())}")

    # They won't be 100% identical because the parquet uses a different
    # stop formula, but they should be correlated
    if pct > 50:
        print("  PASS: Dynamic labels reasonably correlated with parquet.")
    else:
        print("  WARNING: Low agreement -- check stop computation.")
    print()
    return pct > 50


# ---------------------------------------------------------------------------
# Test 2: Fill model produces fills for limit entries
# ---------------------------------------------------------------------------
def test_fill_model():
    """Test fill model on limit entries."""
    print("=== TEST 2: Fill model limit entries ===")
    from core.engine.fill_model import compute_entry_price

    DATA = "data/labeled/BTCUSDT_5m_labeled_v3.parquet"
    COLS = ["open", "high", "low", "close",
            "ict_ob_bull_top", "ict_ob_bull_bot",
            "ict_swing_high_price", "ict_swing_low_price"]
    NROWS = 10000

    df = pd.read_parquet(DATA, columns=COLS)
    df = df.iloc[:NROWS].copy().reset_index(drop=True)

    # Test market entry
    price, bar = compute_entry_price(df, 100, "market", 12, "long")
    print(f"  Market entry at bar 100: price={price:.2f}, bar={bar}")
    assert bar == 100, "Market entry should fill at signal bar"

    # Test limit_ob_mid
    n_fill = 0
    n_test = 0
    for i in range(500, 5000, 50):
        price, bar = compute_entry_price(df, i, "limit_ob_mid", 12, "long")
        if not np.isnan(price):
            n_test += 1
            if bar >= 0:
                n_fill += 1

    fill_rate = n_fill / n_test * 100 if n_test > 0 else 0
    print(f"  limit_ob_mid: {n_test} attempts, {n_fill} fills "
          f"({fill_rate:.1f}%)")

    # Test limit_ce
    n_fill_ce = 0
    n_test_ce = 0
    for i in range(500, 5000, 50):
        price, bar = compute_entry_price(df, i, "limit_ce", 12, "long")
        if not np.isnan(price):
            n_test_ce += 1
            if bar >= 0:
                n_fill_ce += 1

    fill_rate_ce = n_fill_ce / n_test_ce * 100 if n_test_ce > 0 else 0
    print(f"  limit_ce:     {n_test_ce} attempts, {n_fill_ce} fills "
          f"({fill_rate_ce:.1f}%)")

    passed = n_fill > 0 and n_fill_ce > 0
    print(f"  PASS" if passed else "  FAIL: no fills")
    print()
    return passed


# ---------------------------------------------------------------------------
# Test 3: Tier gate thresholds work
# ---------------------------------------------------------------------------
def test_tier_gates():
    """Test that tier-specific gates override defaults."""
    print("=== TEST 3: Tier-specific gate thresholds ===")
    from core.engine.simulator import TIER_GATES, _get_tier_gates
    from core.engine.evaluator import DEFAULT_GATES, compute_gates

    # Standard tier: no overrides
    std_gates = _get_tier_gates("standard")
    assert std_gates == {}, "Standard tier should have no overrides"

    # Monthly tier: relaxed MIN_TRADES_PER_YEAR
    monthly_gates = _get_tier_gates("monthly")
    assert monthly_gates["MIN_TRADES_PER_YEAR"] == 8, \
        f"Monthly MIN_TRADES should be 8, got {monthly_gates['MIN_TRADES_PER_YEAR']}"

    # Weekly tier
    weekly_gates = _get_tier_gates("weekly")
    assert weekly_gates["MIN_TRADES_PER_YEAR"] == 40, \
        f"Weekly MIN_TRADES should be 40, got {weekly_gates['MIN_TRADES_PER_YEAR']}"

    # Test gate computation with monthly tier (low trade count should PASS)
    gate_inputs = {
        "trades_per_yr": 12,
        "oos_auc": 0.80,
        "pbo": 0.0,
        "psr": 1.0,
        "wf_all_profitable": True,
        "sharpe_ann": 2.0,
        "win_rate": 0.65,
        "ev_r": 0.90,
        "max_dd_pct": 10.0,
        "ece": 0.03,
    }

    # With standard gates, 12 trades/yr fails MIN_TRADES_PER_YEAR=100
    std_result = compute_gates(gate_inputs, {})
    std_trades_pass = std_result["MIN_TRADES_PER_YEAR"]["pass"]
    assert not std_trades_pass, "Standard: 12 trades/yr should FAIL"

    # With monthly gates, 12 trades/yr passes MIN_TRADES_PER_YEAR=8
    monthly_result = compute_gates(gate_inputs, monthly_gates)
    monthly_trades_pass = monthly_result["MIN_TRADES_PER_YEAR"]["pass"]
    assert monthly_trades_pass, "Monthly: 12 trades/yr should PASS"

    print("  Standard gates: 12 trades/yr -> FAIL (correct)")
    print("  Monthly gates:  12 trades/yr -> PASS (correct)")
    print("  PASS")
    print()
    return True


# ---------------------------------------------------------------------------
# Test 4: Simulate with fills
# ---------------------------------------------------------------------------
def test_simulate_with_fills():
    """Test _simulate_with_fills function."""
    print("=== TEST 4: Simulate with fills ===")
    from core.engine.simulator import _simulate_with_fills

    DATA = "data/labeled/BTCUSDT_5m_labeled_v3.parquet"
    COLS = ["open", "high", "low", "close",
            "ict_ob_bull_top", "ict_ob_bull_bot",
            "ict_swing_high_price", "ict_swing_low_price",
            "label_long_hit_2r_48c"]
    NROWS = 10000

    df = pd.read_parquet(DATA, columns=COLS)
    df = df.iloc[:NROWS].copy().reset_index(drop=True)

    labels = df["label_long_hit_2r_48c"].values
    label_valid = ~np.isnan(labels)
    label_arr = np.where(label_valid, labels, 0).astype(int)

    # Create a simple signal mask (every 100 bars)
    signal_mask = np.zeros(NROWS, dtype=bool)
    signal_mask[np.arange(500, 9000, 100)] = True
    signal_mask &= label_valid

    trade_indices, fill_stats = _simulate_with_fills(
        signal_mask, label_arr, cooldown=48,
        df=df, entry_type="limit_ob_mid", fill_timeout=12,
        direction="long",
    )

    print(f"  Attempted:  {fill_stats['attempted']}")
    print(f"  Filled:     {fill_stats['filled']}")
    print(f"  Fill rate:  {fill_stats['fill_rate']:.1%}")
    print(f"  Trades:     {len(trade_indices)}")

    passed = fill_stats["attempted"] > 0
    print(f"  PASS" if passed else "  FAIL: no attempts")
    print()
    return passed


# ---------------------------------------------------------------------------
# Test 5: Optimizer tier proposal
# ---------------------------------------------------------------------------
def test_optimizer_tier_proposal():
    """Test that optimizer can propose tier experiments."""
    print("=== TEST 5: Optimizer tier proposal ===")
    from core.engine.optimizer import (
        TIER_CONFIGS, TIER_RESEARCH_QUESTIONS,
        propose_next_experiment,
    )

    # Verify tier configs exist
    assert "standard" in TIER_CONFIGS
    assert "weekly" in TIER_CONFIGS
    assert "monthly" in TIER_CONFIGS
    print(f"  Tier configs: {list(TIER_CONFIGS.keys())}")

    # Verify tier RQs exist
    print(f"  Tier RQs: {[rq['id'] for rq in TIER_RESEARCH_QUESTIONS]}")

    # Monthly tier should have label_config
    monthly = TIER_CONFIGS["monthly"]
    assert monthly["label_config"] is not None
    assert monthly["label_config"]["stop_type"] == "swing_low"
    assert monthly["label_config"]["target_r"] == 3.0
    print(f"  Monthly label_config: stop={monthly['label_config']['stop_type']}, "
          f"target={monthly['label_config']['target_r']}R")

    print("  PASS")
    print()
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    results = []
    results.append(("Dynamic labeler match", test_dynamic_labeler_match()))
    results.append(("Fill model", test_fill_model()))
    results.append(("Tier gates", test_tier_gates()))
    results.append(("Simulate with fills", test_simulate_with_fills()))
    results.append(("Optimizer tier proposal", test_optimizer_tier_proposal()))

    print()
    print("=" * 50)
    print("  D46b SMOKE TEST RESULTS")
    print("=" * 50)
    n_pass = sum(1 for _, r in results if r)
    n_total = len(results)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name:30s} {status}")
    print()
    print(f"  {n_pass}/{n_total} tests passed")
    print("=" * 50)


if __name__ == "__main__":
    main()
