# core/signals/ict/test_rules.py
# D53 causality test + smoke test for all ICT signal functions.
# Run from project root: python core/signals/ict/test_rules.py

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import numpy as np
import pandas as pd

from core.signals.ict.rules import (
    compute_swing_points,
    compute_swing_dual_layer,
    detect_displacement,
    detect_ob_bull,
    detect_ob_bear,
    detect_ob_anchored,
    detect_fvg_bull,
    detect_fvg_bear,
    detect_fvg_enhanced,
    compute_ote_dist,
    compute_liq_levels,
    detect_sweep,
    detect_sweep_sequence,
    compute_premium_discount,
    compute_cisd,
    compute_ob_quality,
    detect_breaker_blocks,
    detect_mss,
)

DATA_PATH = "data/labeled/BTCUSDT_5m_labeled_v3.parquet"
COLS = ["open", "high", "low", "close", "bar_start_ts_utc", "ict_atr_14", "volume_base"]
TEST_TS = [1000, 5000, 10000, 50000]
NROWS = max(TEST_TS) + 2  # 50002


def causality_check(name, a_val, b_val, T):
    """Compare last row of a with second-to-last row of b."""
    if isinstance(a_val, pd.Series):
        ok = a_val.equals(b_val)
    else:
        if pd.isna(a_val) and pd.isna(b_val):
            ok = True
        elif pd.isna(a_val) or pd.isna(b_val):
            ok = False
        else:
            ok = (a_val == b_val)
    assert ok, f"LOOKAHEAD at T={T} for {name}: {a_val} != {b_val}"


def test_causality_standalone(name, func, df_base):
    """Causality test for functions that only need OHLC + ATR."""
    for T in TEST_TS:
        a = func(df_base.iloc[:T])
        b = func(df_base.iloc[:T + 1])
        causality_check(name, a.iloc[-1], b.iloc[-2], T)
    print(f"CAUSALITY PASS: {name}")


def test_causality_dependent(name, func, df_with_deps):
    """Causality test for functions that need pre-merged columns."""
    for T in TEST_TS:
        a = func(df_with_deps.iloc[:T])
        b = func(df_with_deps.iloc[:T + 1])
        causality_check(name, a.iloc[-1], b.iloc[-2], T)
    print(f"CAUSALITY PASS: {name}")


def main():
    t0 = time.time()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"Loading {NROWS} rows from {DATA_PATH} ...")
    df = pd.read_parquet(DATA_PATH, columns=COLS)
    df = df.iloc[:NROWS].copy().reset_index(drop=True)
    print(f"  Loaded: {df.shape}")

    # ==================================================================
    # CAUSALITY TESTS
    # ==================================================================
    print("\n--- CAUSALITY TESTS ---\n")

    # 0. detect_displacement (standalone, D53 Task F)
    test_causality_standalone("detect_displacement", detect_displacement, df)

    # 1. compute_swing_points (standalone)
    test_causality_standalone("compute_swing_points", compute_swing_points, df)

    # 1b. compute_swing_dual_layer (standalone, D53 Task D)
    test_causality_standalone("compute_swing_dual_layer", compute_swing_dual_layer, df)

    # Pre-compute all prerequisite columns for dependent tests
    swings = compute_swing_points(df)
    dual = compute_swing_dual_layer(df)
    disp_full = detect_displacement(df)
    df_full = pd.concat([df, swings, dual, disp_full], axis=1)

    # 2. detect_ob_bull
    test_causality_dependent("detect_ob_bull", detect_ob_bull, df_full)

    # 3. detect_ob_bear
    test_causality_dependent("detect_ob_bear", detect_ob_bear, df_full)

    # 3b. detect_ob_anchored (D53 Task A)
    test_causality_dependent("detect_ob_anchored", detect_ob_anchored, df_full)

    # 4. detect_fvg_bull
    test_causality_dependent("detect_fvg_bull", detect_fvg_bull, df_full)

    # 5. detect_fvg_bear
    test_causality_dependent("detect_fvg_bear", detect_fvg_bear, df_full)

    # 5b. detect_fvg_enhanced (D53 Task C)
    test_causality_dependent("detect_fvg_enhanced", detect_fvg_enhanced, df_full)

    # 6. compute_ote_dist (now returns DataFrame)
    test_causality_dependent("compute_ote_dist", compute_ote_dist, df_full)

    # 7. compute_liq_levels
    test_causality_dependent("compute_liq_levels", compute_liq_levels, df_full)

    # 7b. detect_sweep (D53 Task B)
    test_causality_dependent("detect_sweep", detect_sweep, df_full)

    # Add sweep columns to df_full for dependent tests
    sweep_full = detect_sweep(df_full)
    fvg_b_full = detect_fvg_bull(df_full)
    fvg_r_full = detect_fvg_bear(df_full)
    df_full2 = pd.concat([df_full, sweep_full, fvg_b_full, fvg_r_full], axis=1)

    # 7c. detect_sweep_sequence (D53 Task B)
    test_causality_dependent("detect_sweep_sequence", detect_sweep_sequence, df_full2)

    # 8. compute_premium_discount (D53 Task G -- now returns DataFrame)
    test_causality_dependent("compute_premium_discount", compute_premium_discount, df_full)

    # 9. compute_cisd (D53 Task E -- corrected)
    test_causality_dependent("compute_cisd", compute_cisd, df_full2)

    # 10. compute_ob_quality
    test_causality_dependent("compute_ob_quality", compute_ob_quality, df_full)

    # 11. detect_breaker_blocks
    test_causality_dependent("detect_breaker_blocks", detect_breaker_blocks, df_full)

    # 12. detect_mss (D53 MSS)
    test_causality_dependent("detect_mss", detect_mss, df_full2)

    print("\nAll 18 causality tests PASSED.\n")

    # ==================================================================
    # SMOKE TEST on 10,000 bars
    # ==================================================================
    print("--- SMOKE TEST (10,000 bars) ---\n")

    df_smoke = df.iloc[:10000].copy().reset_index(drop=True)

    # 0. Displacement (D53 Task F)
    disp = detect_displacement(df_smoke)
    print(f"detect_displacement: {disp.shape}")
    bull_fires = disp["displacement_bull"].sum()
    bear_fires = disp["displacement_bear"].sum()
    print(f"  bull fired: {int(bull_fires)} ({bull_fires/len(disp)*100:.2f}%)")
    print(f"  bear fired: {int(bear_fires)} ({bear_fires/len(disp)*100:.2f}%)")
    bull_multi = disp.loc[disp["displacement_bull"] == 1, "displacement_bull_is_multi"]
    bear_multi = disp.loc[disp["displacement_bear"] == 1, "displacement_bear_is_multi"]
    if len(bull_multi) > 0:
        print(f"  bull multi-candle: {int(bull_multi.sum())}/{int(bull_fires)} "
              f"({bull_multi.mean()*100:.1f}%)")
    if len(bear_multi) > 0:
        print(f"  bear multi-candle: {int(bear_multi.sum())}/{int(bear_fires)} "
              f"({bear_multi.mean()*100:.1f}%)")
    print()

    # 1. Swing points
    sw = compute_swing_points(df_smoke)
    df_smoke = pd.concat([df_smoke, sw], axis=1)
    print(f"compute_swing_points: {sw.shape}")
    print(f"  swing_high sum: {sw['swing_high'].sum()}, swing_low sum: {sw['swing_low'].sum()}")
    print()

    # 1b. Dual-layer swings (D53 Task D)
    dual_s = compute_swing_dual_layer(df_smoke)
    df_smoke = pd.concat([df_smoke, dual_s, disp], axis=1)
    print(f"compute_swing_dual_layer: {dual_s.shape}")
    print(f"  int_swing_high sum: {dual_s['int_swing_high'].sum()}, "
          f"int_swing_low sum: {dual_s['int_swing_low'].sum()}")
    print(f"  ext_swing_high sum: {dual_s['ext_swing_high'].sum()}, "
          f"ext_swing_low sum: {dual_s['ext_swing_low'].sum()}")
    print()

    # 2. OB bull (legacy)
    ob_b = detect_ob_bull(df_smoke)
    print(f"detect_ob_bull: {ob_b.shape}")
    print(f"  ob_bull_in_zone sum: {ob_b['ob_bull_in_zone'].sum()}")
    print()

    # 3. OB bear (legacy)
    ob_br = detect_ob_bear(df_smoke)
    print(f"detect_ob_bear: {ob_br.shape}")
    print(f"  ob_bear_in_zone sum: {ob_br['ob_bear_in_zone'].sum()}")
    print()

    # 3b. OB anchored (D53 Task A)
    ob_a = detect_ob_anchored(df_smoke)
    print(f"detect_ob_anchored: {ob_a.shape}")
    print(f"  ob_bull_1_in_zone sum: {ob_a['ob_bull_1_in_zone'].sum()}")
    print(f"  ob_bear_1_in_zone sum: {ob_a['ob_bear_1_in_zone'].sum()}")
    cnt_bull = ob_a['count_active_ob_bull']
    cnt_bear = ob_a['count_active_ob_bear']
    print(f"  count_active_ob_bull: mean={cnt_bull.mean():.2f}, max={cnt_bull.max():.0f}")
    print(f"  count_active_ob_bear: mean={cnt_bear.mean():.2f}, max={cnt_bear.max():.0f}")
    print()

    # 4. FVG bull (legacy)
    fvg_b = detect_fvg_bull(df_smoke)
    print(f"detect_fvg_bull: {fvg_b.shape}")
    print(f"  fvg_bull_in_zone sum: {fvg_b['fvg_bull_in_zone'].sum()}")
    print()

    # 5. FVG bear (legacy)
    fvg_br = detect_fvg_bear(df_smoke)
    print(f"detect_fvg_bear: {fvg_br.shape}")
    print(f"  fvg_bear_in_zone sum: {fvg_br['fvg_bear_in_zone'].sum()}")
    df_smoke = pd.concat([df_smoke, fvg_b, fvg_br], axis=1)
    print()

    # 5b. FVG enhanced (D53 Task C)
    fvg_e = detect_fvg_enhanced(df_smoke)
    print(f"detect_fvg_enhanced: {fvg_e.shape}")
    print(f"  fvg_bull_count mean: {fvg_e['fvg_bull_count'].mean():.2f}")
    print(f"  fvg_bear_count mean: {fvg_e['fvg_bear_count'].mean():.2f}")
    b1_disp = fvg_e['fvg_bull_1_is_displacement'].sum()
    r1_disp = fvg_e['fvg_bear_1_is_displacement'].sum()
    print(f"  fvg_bull_1_is_displacement sum: {int(b1_disp)}")
    print(f"  fvg_bear_1_is_displacement sum: {int(r1_disp)}")
    b1_ce = fvg_e['fvg_bull_1_ce_touched'].sum()
    r1_ce = fvg_e['fvg_bear_1_ce_touched'].sum()
    print(f"  fvg_bull_1_ce_touched sum: {int(b1_ce)}")
    print(f"  fvg_bear_1_ce_touched sum: {int(r1_ce)}")
    print()

    # 6. OTE dist (now returns DataFrame)
    ote = compute_ote_dist(df_smoke)
    print(f"compute_ote_dist: {ote.shape}")
    ote_d = ote["ote_dist"].dropna()
    print(f"  ote_dist: {len(ote_d)} non-NaN, mean={ote_d.mean():.4f}")
    at705 = ote["ote_at_705"].dropna()
    print(f"  ote_at_705 sum: {at705.sum():.0f} ({at705.mean()*100:.1f}%)")
    print()

    # 7. Liq levels
    liq = compute_liq_levels(df_smoke)
    print(f"compute_liq_levels: {liq.shape}")
    print(f"  eq_high sum: {liq['liq_eq_high'].sum()}, eq_low sum: {liq['liq_eq_low'].sum()}")
    print()

    # 7b. Sweep detection (D53 Task B)
    swp = detect_sweep(df_smoke)
    df_smoke = pd.concat([df_smoke, swp], axis=1)
    print(f"detect_sweep: {swp.shape}")
    print(f"  sweep_bsl_fired sum: {swp['sweep_bsl_fired'].sum():.0f} "
          f"({swp['sweep_bsl_fired'].mean()*100:.2f}%)")
    print(f"  sweep_ssl_fired sum: {swp['sweep_ssl_fired'].sum():.0f} "
          f"({swp['sweep_ssl_fired'].mean()*100:.2f}%)")
    print(f"  dist_unswept_bsl_atr non-NaN: "
          f"{(~np.isnan(swp['dist_unswept_bsl_atr'].values)).sum()}")
    print(f"  dist_unswept_ssl_atr non-NaN: "
          f"{(~np.isnan(swp['dist_unswept_ssl_atr'].values)).sum()}")
    print()

    # 7c. Sweep sequence (D53 Task B)
    ssq = detect_sweep_sequence(df_smoke)
    print(f"detect_sweep_sequence: {ssq.shape}")
    print(f"  sweep_seq_bull_complete sum: {ssq['sweep_seq_bull_complete'].sum():.0f}")
    print(f"  sweep_seq_bear_complete sum: {ssq['sweep_seq_bear_complete'].sum():.0f}")
    print()

    # 8. Premium/discount (D53 Task G)
    pd_s = compute_premium_discount(df_smoke)
    print(f"compute_premium_discount: {pd_s.shape}")
    pp = pd_s["pd_position_5m"].dropna()
    print(f"  pd_position_5m: {len(pp)} non-NaN, mean={pp.mean():.4f}")
    print(f"  in_discount mean: {pd_s['in_discount'].mean():.4f}")
    print(f"  in_ote_bull mean: {pd_s['in_ote_bull'].mean():.4f}")
    print(f"  in_ote_bear mean: {pd_s['in_ote_bear'].mean():.4f}")
    print()

    # 9. CISD (D53 Task E)
    cisd = compute_cisd(df_smoke)
    print(f"compute_cisd: {cisd.shape}")
    print(f"  cisd_bull sum: {cisd['cisd_bull'].sum():.0f}, "
          f"cisd_bear sum: {cisd['cisd_bear'].sum():.0f}")
    print(f"  cisd_bull_with_sweep sum: {cisd['cisd_bull_with_sweep'].sum():.0f}")
    print(f"  cisd_bear_with_sweep sum: {cisd['cisd_bear_with_sweep'].sum():.0f}")
    print()

    # 10. OB quality
    obq = compute_ob_quality(df_smoke)
    print(f"compute_ob_quality: {obq.shape}")
    bull_valid = obq["ob_bull_quality"].dropna()
    bear_valid = obq["ob_bear_quality"].dropna()
    print(f"  ob_bull_quality: {len(bull_valid)} non-NaN, "
          f"mean={bull_valid.mean():.4f}, max={bull_valid.max():.4f}")
    print(f"  ob_bear_quality: {len(bear_valid)} non-NaN, "
          f"mean={bear_valid.mean():.4f}, max={bear_valid.max():.4f}")
    print()

    # 11. Breaker blocks
    brk = detect_breaker_blocks(df_smoke)
    print(f"detect_breaker_blocks: {brk.shape}")
    bb_age = brk["breaker_bull_age"].dropna()
    ba_age = brk["breaker_bear_age"].dropna()
    print(f"  breaker_bull_age: {len(bb_age)} non-NaN bars "
          f"(coverage {len(bb_age)/len(brk)*100:.1f}%)")
    print(f"  breaker_bear_age: {len(ba_age)} non-NaN bars "
          f"(coverage {len(ba_age)/len(brk)*100:.1f}%)")
    print(f"  breaker_bull_in_zone sum: {brk['breaker_bull_in_zone'].sum()}")
    print(f"  breaker_bear_in_zone sum: {brk['breaker_bear_in_zone'].sum()}")
    print()

    # 12. MSS (D53)
    mss = detect_mss(df_smoke)
    print(f"detect_mss: {mss.shape}")
    print(f"  mss_bull_fired sum: {mss['mss_bull_fired'].sum():.0f} "
          f"({mss['mss_bull_fired'].mean()*100:.2f}%)")
    print(f"  mss_bear_fired sum: {mss['mss_bear_fired'].sum():.0f} "
          f"({mss['mss_bear_fired'].mean()*100:.2f}%)")
    print(f"  mss_bull_with_sweep sum: {mss['mss_bull_with_sweep'].sum():.0f}")
    print(f"  mss_bear_with_sweep sum: {mss['mss_bear_with_sweep'].sum():.0f}")
    ms = mss["mss_strength"]
    ms_valid = ms[ms > 0]
    if len(ms_valid) > 0:
        print(f"  mss_strength (when fired): mean={ms_valid.mean():.4f}")
    print()

    elapsed = time.time() - t0
    print(f"--- ALL DONE in {elapsed:.1f}s ---")


if __name__ == "__main__":
    main()
