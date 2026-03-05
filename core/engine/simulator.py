# core/engine/simulator.py
# Experiment orchestrator: load data, train, simulate, evaluate, gate.
# D38 -- pure plumbing, no signal logic. ASCII-safe for cp1252.
# D42 -- added augment_features() for on-the-fly feature computation.
# D42 -- added walk-forward isotonic calibration (calibrator.py) + probs saving.
# D46b -- added label_config (dynamic labeler) and fill_config (fill model) support.
#          Added tier tracking (monthly/weekly/standard).
# D47b -- added signal_filter support for hard column-based signal gating.
# D51  -- embargo 48->288 bars, holdout guard, has_oi/has_liqs availability masks.

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from core.engine.evaluator import (
    build_trade_returns,
    compute_auc,
    compute_ece,
    compute_gates,
    compute_metrics,
    run_cscv,
    simulate,
    walk_forward_train,
)
from core.engine.labels import get_label, parse_label_col, validate_label_alignment
from core.engine.sizing import (
    equity_sim,
    equity_sim_variable,
    kelly_fraction_array,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "labeled"
REGISTRY_PATH = PROJECT_ROOT / "core" / "experiments" / "registry.json"

DATA_FILES = {
    "v3": DATA_DIR / "BTCUSDT_5m_labeled_v3_train.parquet",  # D51: train only
    "v3_full": DATA_DIR / "BTCUSDT_5m_labeled_v3.parquet",   # full (do not use)
    "v2": DATA_DIR / "BTCUSDT_5m_labeled_v2.parquet",
    "v1": DATA_DIR / "BTCUSDT_MASTER_labeled.parquet",
}

# Columns to exclude from features by default
META_COLS = {
    "bar_start_ts_utc", "bar_end_ts_utc", "bar_start_ts_ms", "bar_end_ts_ms",
    "open", "high", "low", "close", "volume",
    "perp_open", "perp_high", "perp_low", "perp_close", "perp_volume",
}
LABEL_PREFIX = "label_"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data() -> tuple[pd.DataFrame, str]:
    """
    Load the best available labeled parquet.
    Tries v3 (train) -> v2 -> v1 in order. Skips v3_full.
    Returns (dataframe, version_used).
    """
    for version, path in DATA_FILES.items():
        if version == "v3_full":
            continue  # D51: never load full dataset in experiments
        if path.exists():
            # D51: guard against accidentally loading holdout
            if "holdout" in str(path).lower():
                raise RuntimeError(
                    "Do not load holdout in experiments! "
                    f"Path: {path}"
                )
            print(f"  Loading dataset: {version} ({path.name})")
            df = pd.read_parquet(path)
            print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} cols")
            return df, version

    raise FileNotFoundError(
        f"No labeled parquet found. Checked: "
        f"{[str(p) for p in DATA_FILES.values()]}"
    )


# ---------------------------------------------------------------------------
# On-the-fly feature augmentation
# ---------------------------------------------------------------------------
# Features computed from rules.py functions that aren't in the parquet yet.
# Each entry: feature_name -> (module_path, function_name, required_cols).
# The optimizer uses ONTHEFLY_FEATURES to check availability.
# D53: massive expansion -- displacement, dual-layer swings, anchored OBs,
#       enhanced FVGs, sweeps, sweep sequence, CISD, premium/discount, OTE, MSS.
_D53_ICT = "core.signals.ict.rules"

def _d53_feature_names():
    """Generate all D53 on-the-fly feature names programmatically."""
    names = {}

    # detect_displacement: 14 cols
    for d in ("bull", "bear"):
        for s in ("", "_age", "_strength", "_close_loc", "_range_atr", "_has_fvg", "_is_multi"):
            names[f"displacement_{d}{s}"] = (_D53_ICT, "detect_displacement", [])

    # compute_swing_dual_layer: 22 cols
    for pfx in ("int_", "ext_"):
        for s in ("swing_high", "swing_low", "swing_high_price", "swing_low_price",
                  "dist_to_sh_atr", "dist_to_sl_atr", "trend",
                  "bos_bull", "bos_bear", "choch_bull", "choch_bear"):
            names[f"{pfx}{s}"] = (_D53_ICT, "compute_swing_dual_layer", [])

    # detect_ob_anchored: 82 cols
    for d in ("bull", "bear"):
        for k in (1, 2, 3):
            for s in ("state", "age", "top", "bot", "mid", "width_atr", "in_zone",
                      "penetration", "dist_top_atr", "dist_bot_atr", "strength",
                      "bars_to_bos", "has_fvg"):
                names[f"ob_{d}_{k}_{s}"] = (_D53_ICT, "detect_ob_anchored", ["bos_close"])
        names[f"count_active_ob_{d}"] = (_D53_ICT, "detect_ob_anchored", ["bos_close"])
        names[f"min_dist_ob_{d}_atr"] = (_D53_ICT, "detect_ob_anchored", ["bos_close"])

    # detect_fvg_enhanced: 52 cols
    for d in ("bull", "bear"):
        for k in (1, 2, 3):
            for s in ("ce", "dist_to_ce_atr", "ce_touched", "ce_rejected",
                      "fill_fraction", "fully_filled", "is_displacement", "is_ifvg"):
                names[f"fvg_{d}_{k}_{s}"] = (_D53_ICT, "detect_fvg_enhanced", [])
        names[f"fvg_{d}_count"] = (_D53_ICT, "detect_fvg_enhanced", [])
        names[f"fvg_{d}_recent_age"] = (_D53_ICT, "detect_fvg_enhanced", [])

    # detect_sweep: 8 cols
    for d in ("bsl", "ssl"):
        for s in ("fired", "age", "pen_atr"):
            names[f"sweep_{d}_{s}"] = (_D53_ICT, "detect_sweep", [])
    names["dist_unswept_bsl_atr"] = (_D53_ICT, "detect_sweep", [])
    names["dist_unswept_ssl_atr"] = (_D53_ICT, "detect_sweep", [])

    # detect_sweep_sequence: 4 cols
    for d in ("bull", "bear"):
        names[f"sweep_seq_{d}_complete"] = (_D53_ICT, "detect_sweep_sequence", [])
        names[f"sweep_seq_{d}_age"] = (_D53_ICT, "detect_sweep_sequence", [])

    # compute_premium_discount: 7 cols
    for s in ("pd_position_5m", "pd_dist_from_eq", "in_discount",
              "in_deep_discount", "in_deep_premium", "in_ote_bull", "in_ote_bear"):
        names[s] = (_D53_ICT, "compute_premium_discount", [])

    # compute_ote_dist: 3 cols (replaces old 1-col Series)
    for s in ("ote_dist", "ote_dist_from_705_atr", "ote_at_705"):
        names[s] = (_D53_ICT, "compute_ote_dist", [])

    # compute_cisd: 6 cols (replaces old 2-col)
    for d in ("bull", "bear"):
        for s in ("", "_age", "_with_sweep"):
            names[f"cisd_{d}{s}"] = (_D53_ICT, "compute_cisd", [])

    # detect_mss: 7 cols
    for d in ("bull", "bear"):
        names[f"mss_{d}_fired"] = (_D53_ICT, "detect_mss", [])
        names[f"mss_{d}_age"] = (_D53_ICT, "detect_mss", [])
        names[f"mss_{d}_with_sweep"] = (_D53_ICT, "detect_mss", [])
    names["mss_strength"] = (_D53_ICT, "detect_mss", [])

    return names

ONTHEFLY_FEATURES = {
    # Pre-D53 features
    "ob_bull_quality": (_D53_ICT, "compute_ob_quality", ["bos_close"]),
    "ob_bear_quality": (_D53_ICT, "compute_ob_quality", ["bos_close"]),
    "breaker_bull_age": (_D53_ICT, "detect_breaker_blocks", ["bos_close"]),
    "breaker_bull_dist": (_D53_ICT, "detect_breaker_blocks", ["bos_close"]),
    "breaker_bull_in_zone": (_D53_ICT, "detect_breaker_blocks", ["bos_close"]),
    "breaker_bear_age": (_D53_ICT, "detect_breaker_blocks", ["bos_close"]),
    "breaker_bear_dist": (_D53_ICT, "detect_breaker_blocks", ["bos_close"]),
    "breaker_bear_in_zone": (_D53_ICT, "detect_breaker_blocks", ["bos_close"]),
    # D47 regime features (hmm_filter.py)
    "hmm_prob_bull": ("core.signals.regime.hmm_filter", "compute_all_regime_features", ["close"]),
    "hmm_prob_bear": ("core.signals.regime.hmm_filter", "compute_all_regime_features", ["close"]),
    "hmm_prob_calm": ("core.signals.regime.hmm_filter", "compute_all_regime_features", ["close"]),
    "bb_width_normalized": ("core.signals.regime.hmm_filter", "compute_adx_composite", ["close"]),
    "atr_percentile_rank": ("core.signals.regime.hmm_filter", "compute_adx_composite", ["close"]),
    "regime_tag": ("core.signals.regime.hmm_filter", "compute_adx_composite", ["close"]),
    "ob_bull_age_x_hmm_bull": ("core.signals.regime.hmm_filter", "compute_regime_interactions", ["close"]),
    "fvg_bull_x_trending": ("core.signals.regime.hmm_filter", "compute_regime_interactions", ["close"]),
    "ote_x_regime": ("core.signals.regime.hmm_filter", "compute_regime_interactions", ["close"]),
    # D51 availability masks (trivially causal -- based on NaN pattern of existing cols)
    "has_oi": (None, None, ["oi_btc"]),
    "has_liqs": (None, None, ["liq_total_btc"]),
    # D53 ICT rules overhaul (~203 new features)
    **_d53_feature_names(),
}


def augment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute on-the-fly features if not already in df.

    D53 features are computed in dependency order:
      1. detect_displacement (standalone)
      2. compute_swing_points + compute_swing_dual_layer (standalone)
      3. detect_ob_anchored (needs bos_close + displacement)
      4. detect_fvg_bull/bear + detect_fvg_enhanced (needs displacement)
      5. compute_liq_levels (needs swing prices)
      6. detect_sweep (needs int_swing)
      7. detect_sweep_sequence (needs sweep + displacement + FVG)
      8. compute_premium_discount (needs ext_swing)
      9. compute_ote_dist (needs swing prices)
     10. compute_cisd (needs sweep ages)
     11. detect_mss (needs int_choch + displacement + sweep)
     12. detect_breaker_blocks (needs bos_close)
     13. compute_ob_quality (needs bos_close + volume)

    Returns df with new columns added in-place.
    """
    needed = [f for f in ONTHEFLY_FEATURES if f not in df.columns]
    if not needed:
        return df

    import warnings

    def _merge(result):
        """Merge a DataFrame/Series result into df, return column count."""
        if isinstance(result, pd.Series):
            df[result.name] = result.values
            return 1
        for col in result.columns:
            df[col] = result[col].values
        return len(result.columns)

    from core.signals.ict.rules import (
        detect_displacement,
        compute_swing_points,
        compute_swing_dual_layer,
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

    d53_count = 0

    # Suppress fragmentation warnings during column-by-column addition;
    # we defragment once at the end with df.copy().
    warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")

    # 1. Displacement (standalone, OHLC + ATR)
    if "displacement_bull" not in df.columns:
        print("  D53: computing displacement ...")
        d53_count += _merge(detect_displacement(df))

    # 2. Swing points (original + dual-layer)
    if "bos_close" not in df.columns:
        print("  D53: computing swing points ...")
        d53_count += _merge(compute_swing_points(df))

    # Guard: use int_bos_bull (D53-unique) not int_swing_high (collides with v2 parquet)
    if "int_bos_bull" not in df.columns:
        print("  D53: computing dual-layer swings ...")
        d53_count += _merge(compute_swing_dual_layer(df))

    # 3. OB anchored (needs bos_close + displacement)
    if "ob_bull_1_state" not in df.columns:
        print("  D53: computing anchored OBs (top-3) ...")
        d53_count += _merge(detect_ob_anchored(df))

    # 4. FVG (legacy + enhanced)
    if "fvg_bull_in_zone" not in df.columns:
        d53_count += _merge(detect_fvg_bull(df))
        d53_count += _merge(detect_fvg_bear(df))

    if "fvg_bull_1_ce" not in df.columns:
        print("  D53: computing enhanced FVGs (top-3, CE, IFVG) ...")
        d53_count += _merge(detect_fvg_enhanced(df))

    # 5. Liquidity levels (needs swing prices)
    if "liq_dist_above_pct" not in df.columns:
        d53_count += _merge(compute_liq_levels(df))

    # 6. Sweep (needs int_swing)
    if "sweep_bsl_fired" not in df.columns:
        print("  D53: computing sweeps ...")
        d53_count += _merge(detect_sweep(df))

    # 7. Sweep sequence (needs sweep + displacement + FVG counts)
    if "sweep_seq_bull_complete" not in df.columns:
        print("  D53: computing sweep sequences ...")
        d53_count += _merge(detect_sweep_sequence(df))

    # 8. Premium/discount (needs ext_swing prices)
    if "pd_position_5m" not in df.columns:
        print("  D53: computing premium/discount ...")
        d53_count += _merge(compute_premium_discount(df))

    # 9. OTE dist (needs swing prices)
    # Guard: use ote_at_705 (D53-unique) not ote_dist (collides with v2 parquet)
    if "ote_at_705" not in df.columns:
        print("  D53: computing OTE dist + 0.705 level ...")
        d53_count += _merge(compute_ote_dist(df))

    # 10. CISD (needs sweep ages)
    # Guard: use cisd_bull_age (D53-unique) not cisd_bull (collides with v2 parquet)
    if "cisd_bull_age" not in df.columns:
        print("  D53: computing CISD ...")
        d53_count += _merge(compute_cisd(df))

    # 11. MSS (needs int_choch + displacement + sweep)
    if "mss_bull_fired" not in df.columns:
        print("  D53: computing MSS ...")
        d53_count += _merge(detect_mss(df))

    # 12. Breaker blocks (needs bos_close)
    if "breaker_bull_age" not in df.columns:
        print("  D53: computing breaker blocks ...")
        d53_count += _merge(detect_breaker_blocks(df))

    # 13. OB quality (needs bos_close + volume)
    if "ob_bull_quality" not in df.columns or "ob_bear_quality" not in df.columns:
        print("  D53: computing OB quality ...")
        d53_count += _merge(compute_ob_quality(df))

    if d53_count > 0:
        print(f"  D53 total: +{d53_count} on-the-fly ICT features")
        # Defragment after bulk column additions
        df = df.copy()

    # Compute regime features (D47: HMM + ADX composite + interactions)
    if "hmm_prob_bull" not in df.columns:
        from core.signals.regime.hmm_filter import compute_all_regime_features
        pre_cols = set(df.columns)
        regime = compute_all_regime_features(df)
        new_cols = [c for c in df.columns if c not in pre_cols]
        print(f"  Augmented: +{len(new_cols)} regime features "
              f"(HMM + ADX + interactions)")

    # D51: Availability masks (trivially causal -- NaN-based)
    if "has_oi" not in df.columns and "oi_btc" in df.columns:
        df["has_oi"] = (~df["oi_btc"].isna()).astype(np.float64)
        n_has = int(df["has_oi"].sum())
        print(f"  Augmented: has_oi ({n_has:,}/{len(df):,} = "
              f"{n_has / len(df) * 100:.1f}%)")

    if "has_liqs" not in df.columns and "liq_total_btc" in df.columns:
        df["has_liqs"] = (~df["liq_total_btc"].isna()).astype(np.float64)
        n_has = int(df["has_liqs"].sum())
        print(f"  Augmented: has_liqs ({n_has:,}/{len(df):,} = "
              f"{n_has / len(df) * 100:.1f}%)")

    return df


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------
def select_features(
    df: pd.DataFrame,
    features: list[str] | str,
    feature_exclude: list[str] | None = None,
) -> list[str]:
    """
    Resolve feature list from experiment config.

    features="all"  -> all numeric columns minus labels and meta
    features=[...]  -> explicit list (validated against df columns)
    feature_exclude  -> columns to drop (only when features="all")
    """
    if features == "all":
        exclude = set(META_COLS)
        exclude.update(c for c in df.columns if c.startswith(LABEL_PREFIX))
        if feature_exclude:
            exclude.update(feature_exclude)

        feat_cols = [
            c for c in df.columns
            if c not in exclude and df[c].dtype in ("float64", "float32", "int64", "int32")
        ]
        print(f"  Features: {len(feat_cols)} (auto-selected, "
              f"{len(feature_exclude or [])} excluded)")
        return feat_cols

    # Explicit list -- validate
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(
            f"Features not found in dataframe: {missing[:10]}"
            f"{'...' if len(missing) > 10 else ''}"
        )
    print(f"  Features: {len(features)} (explicit list)")
    return list(features)


# ---------------------------------------------------------------------------
# Registry I/O
# ---------------------------------------------------------------------------
def _read_registry() -> dict:
    """Read registry.json or create it if missing."""
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, "r") as f:
            return json.load(f)
    return {"experiments": []}


def _write_registry(reg: dict) -> None:
    """Write registry.json atomically."""
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = REGISTRY_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(reg, f, indent=2, default=str)
    tmp.replace(REGISTRY_PATH)


def _append_result(result: dict) -> None:
    """Append an experiment result to registry.json."""
    reg = _read_registry()
    # Replace existing entry with same id, if any
    reg["experiments"] = [
        e for e in reg["experiments"] if e.get("id") != result.get("id")
    ]
    reg["experiments"].append(result)
    _write_registry(reg)
    print(f"  Registry updated: {REGISTRY_PATH.name} "
          f"({len(reg['experiments'])} experiments)")


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------
def _print_report(exp_id: str, results: dict) -> None:
    """Print a summary report to stdout. ASCII only."""
    print()
    print("=" * 60)
    print(f"  EXPERIMENT: {exp_id}")
    print("=" * 60)

    m = results.get("metrics", {})
    print(f"  Trades:          {m.get('n_trades', 0):,}")
    print(f"  Trades/yr:       {m.get('trades_per_yr', 0):.1f}")
    print(f"  Win Rate:        {m.get('win_rate', 0) * 100:.2f}%")
    print(f"  EV (R):          {m.get('ev_r', 0):+.4f}")
    print(f"  Profit Factor:   {m.get('profit_factor', 0):.4f}")
    print(f"  Max DD:          {m.get('max_dd_pct', 0):.2f}%")
    print(f"  Sharpe (ann.):   {m.get('sharpe_ann', 0):.4f}")
    print(f"  Final Equity:    {m.get('final_equity', 0):,.2f}")
    print()

    oos_auc = results.get("oos_auc", 0)
    ece = results.get("ece", 0)
    print(f"  OOS AUC:         {oos_auc:.4f}")
    print(f"  ECE:             {ece:.6f}")
    print()

    cscv = results.get("cscv", {}).get("cscv", {})
    psr = results.get("cscv", {}).get("psr", {})
    bootstrap = results.get("cscv", {}).get("bootstrap_ci", {})
    print(f"  CSCV PBO:        {cscv.get('pbo', 0):.4f}")
    print(f"  PSR:             {psr.get('psr', 0):.4f}")
    print(f"  Bootstrap CI:    [{bootstrap.get('ci_lower', 0):.4f}, "
          f"{bootstrap.get('ci_upper', 0):.4f}]")
    print()

    gates = results.get("gates", {})
    n_pass = sum(1 for v in gates.values() if v.get("pass"))
    n_total = len(gates)
    print(f"  Gates: {n_pass}/{n_total} PASS")
    for name, info in gates.items():
        status = "PASS" if info.get("pass") else "FAIL"
        print(f"    {name:30s} {status}  "
              f"(value={info.get('value')}, threshold={info.get('threshold')})")

    verdict = "ALL PASS" if n_pass == n_total else f"{n_total - n_pass} FAILED"
    print()
    print(f"  VERDICT: {verdict}")
    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# Tier-specific gate thresholds (D46b)
# ---------------------------------------------------------------------------
TIER_GATES = {
    "standard": {},  # use DEFAULT_GATES from evaluator.py
    "weekly": {
        "MIN_TRADES_PER_YEAR": 40,
        "MIN_WR": 0.55,
        "MIN_EV_R": 0.50,
        "MIN_SHARPE": 1.5,
    },
    "monthly": {
        "MIN_TRADES_PER_YEAR": 8,
        "MIN_WR": 0.60,
        "MIN_EV_R": 0.80,
        "MIN_SHARPE": 1.0,
        "MAX_DRAWDOWN": 0.15,
    },
}


def _get_tier_gates(tier: str) -> dict:
    """Return gate overrides for a given tier."""
    return TIER_GATES.get(tier, {})


# ---------------------------------------------------------------------------
# Limit-entry simulation with fill model (D46b)
# ---------------------------------------------------------------------------
def _simulate_with_fills(
    signal_mask: np.ndarray,
    label_arr: np.ndarray,
    cooldown: int,
    df: pd.DataFrame,
    entry_type: str,
    fill_timeout: int,
    direction: str,
) -> tuple[list[int], dict]:
    """
    Simulate trades with fill model for limit entries.

    For each signal bar where signal_mask is True, checks if a limit order
    would fill within fill_timeout bars. Only counts as a trade if filled.
    Cooldown starts from the signal bar (not the fill bar) to prevent
    piling up pending orders.

    Returns (trade_indices, fill_stats).
    trade_indices: list of signal bar indices where trades were taken.
    fill_stats: dict with attempted, filled, fill_rate.
    """
    from core.engine.fill_model import compute_entry_price

    trade_indices: list[int] = []
    bars_since = cooldown  # start ready to trade
    n_attempted = 0
    n_filled = 0

    for i in range(len(signal_mask)):
        bars_since += 1
        if bars_since > cooldown and signal_mask[i]:
            n_attempted += 1
            price, fill_bar = compute_entry_price(
                df, i, entry_type, fill_timeout, direction,
            )
            if not np.isnan(price) and fill_bar >= 0:
                # Fill succeeded -- take the trade
                trade_indices.append(i)
                n_filled += 1
            # Cooldown starts from signal bar regardless of fill
            bars_since = 0

    fill_rate = n_filled / n_attempted if n_attempted > 0 else 0.0
    return trade_indices, {
        "attempted": n_attempted,
        "filled": n_filled,
        "fill_rate": round(fill_rate, 4),
        "entry_type": entry_type,
    }


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------
def run_experiment(experiment: dict) -> dict:
    """
    Run a single experiment end-to-end.

    Experiment dict schema:
        id (str): unique experiment name
        signal_domain (str): "ml" | "structural" | "hybrid"
        features (list[str] | "all"): feature columns
        feature_exclude (list[str]): columns to exclude when features="all"
        label (str): label column name, e.g. "label_long_hit_2r_48c"
        label_config (dict, optional): dynamic label config (D46b).
            When present, labels are computed via labeler.compute_labels()
            instead of reading from parquet. Keys:
                direction (str): "long" or "short"
                target_r (float): R-multiple target (e.g. 2.0)
                stop_type (str): "atr", "swing_low", "swing_high", "fixed_pct"
                stop_atr_mult (float): ATR multiplier for stop (default 1.5)
                max_bars (int): max hold horizon in bars
                entry_type (str): "market", "limit_ob_mid", "limit_fvg_edge", "limit_ce"
                fill_timeout (int): max bars to wait for limit fill (default 12)
                entry_delay (int): bars after signal before entry (default 0)
            If entry_type != "market", the fill model is used during simulation.
        signal_filter (dict, optional): hard filters applied to signal mask (D47b).
            Each key is a dataframe column name, value is a dict of operators:
                {"min": val}  -> column >= val
                {"max": val}  -> column <= val
                {"eq": val}   -> column == val
            Example: {"hmm_prob_bull": {"min": 0.60}}
            Applied after ML threshold, before cooldown simulation.
        tier (str, optional): "standard" | "weekly" | "monthly" (D46b).
            Selects tier-specific gate thresholds. Default "standard".
        ml_config (dict):
            threshold (float): ML probability threshold for signals
            model (str): "lgbm"
            device (str): "gpu" or "cpu"
            n_folds (int): ignored -- auto-computed
            embargo_bars (int): gap between train/test
            min_train_bars (int, optional)
            test_fold_bars (int, optional)
        cooldown_bars (int): bars between trades
        sizing (dict):
            method (str): "kelly" or "fixed"
            risk_pct (float, optional): fixed risk pct (for method="fixed")
            divisor (float, optional): Kelly divisor (default 40)
            odds (float, optional): Kelly odds (default 2.0)
        gates (dict | None): override DEFAULT_GATES
        notes (str): free text

    Returns results dict with metrics, CSCV, gates, etc.
    """
    exp_id = experiment.get("id", "unnamed")
    t0 = time.time()
    print(f"\n--- Experiment: {exp_id} ---")

    # 1. Load data
    df, data_version = load_data()

    # 1b. Augment with on-the-fly features
    df = augment_features(df)

    # 2. Resolve label (static parquet column or dynamic labeler)
    label_config = experiment.get("label_config")
    use_dynamic_labels = label_config is not None

    if use_dynamic_labels:
        # D46b: dynamic label path
        from core.engine.labeler import compute_labels

        print("  Label mode: dynamic (labeler.py)")
        raw_labels = compute_labels(df, label_config)
        r_target = label_config.get("target_r", 2.0)
        horizon_bars = label_config.get("max_bars", 48)
        label_col = (
            f"dyn_{label_config.get('direction', 'long')}_"
            f"{label_config.get('stop_type', 'atr')}_"
            f"{int(r_target)}r_{horizon_bars}c"
        )
        print(f"  Dynamic label: {label_col}")
        valid_count = int(np.sum(~np.isnan(raw_labels)))
        win_count = int(np.sum(raw_labels == 1.0))
        if valid_count > 0:
            print(f"  Valid labels: {valid_count:,}, "
                  f"wins: {win_count:,} "
                  f"({win_count / valid_count * 100:.1f}%)")
        else:
            print("  Valid labels: 0 (no valid labels computed)")
        # Store dynamic labels as a temp column for walk-forward training
        df[label_col] = raw_labels
    else:
        # Static label path (existing behavior)
        label_col = experiment["label"]
        if label_col not in df.columns:
            raise ValueError(f"Label '{label_col}' not in dataframe columns")
        raw_labels = df[label_col].values
        label_info = parse_label_col(label_col)
        r_target = label_info["r_multiple"]
        horizon_bars = label_info["horizon_bars"]

    # Validate embargo alignment
    # 288 bars = 24h = max feature lookback per AFML Ch.7
    embargo = experiment.get("ml_config", {}).get("embargo_bars", 288)
    validate_label_alignment(embargo, horizon_bars)

    # 3. Select features
    features = select_features(
        df,
        experiment.get("features", "all"),
        experiment.get("feature_exclude"),
    )

    # 4. Walk-forward train
    ml_config = experiment.get("ml_config", {})
    oos_probs = walk_forward_train(df, features, label_col, ml_config)

    # 5. Prepare labels
    if not use_dynamic_labels:
        raw_labels = df[label_col].values
    # raw_labels already set for dynamic path
    label_valid = ~np.isnan(raw_labels)
    label_arr = np.where(label_valid, raw_labels, 0).astype(int)

    # 5b. Save raw OOS probs
    models_dir = PROJECT_ROOT / "core" / "experiments" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    np.save(models_dir / f"{exp_id}_oos_probs.npy", oos_probs)

    # 5c. Walk-forward isotonic calibration (calibrator.py D41a)
    from core.engine.calibrator import calibrate_walk_forward

    # 288 bars = 24h = max feature lookback per AFML Ch.7
    _emb = ml_config.get("embargo_bars", 288)
    _mtr = ml_config.get("min_train_bars", 105_000)
    _tfs = ml_config.get("test_fold_bars", 52_500)
    fold_bounds = []
    _t = _mtr + _emb
    while _t < len(df):
        fold_bounds.append(_t)
        _t = min(_t + _tfs, len(df))
    fold_bounds.append(len(df))

    ece_raw = compute_ece(oos_probs[label_valid], label_arr[label_valid])
    oos_probs = calibrate_walk_forward(
        oos_probs, raw_labels.astype(np.float64), fold_bounds,
    )
    ece = compute_ece(oos_probs[label_valid], label_arr[label_valid])
    print(f"  ECE raw:        {ece_raw:.6f}")
    print(f"  ECE calibrated: {ece:.6f}")
    np.save(models_dir / f"{exp_id}_oos_probs_cal.npy", oos_probs)

    # 6. Compute OOS AUC on covered bars
    covered = ~np.isnan(oos_probs)
    eval_mask = covered & label_valid
    oos_auc = compute_auc(label_arr[eval_mask], oos_probs[eval_mask])
    print(f"  OOS AUC: {oos_auc:.4f} ({int(eval_mask.sum()):,} bars evaluated)")

    # 7. Build signal mask (require valid label for trade)
    threshold = ml_config.get("threshold", 0.60)
    signal_mask = covered & label_valid & (oos_probs >= threshold)
    n_signals = int(signal_mask.sum())
    print(f"  Signals at t={threshold}: {n_signals:,}")

    # 7b. Apply signal_filter (D47b -- hard column filters on signal mask)
    signal_filter = experiment.get("signal_filter")
    if signal_filter:
        for col_name, ops in signal_filter.items():
            if col_name not in df.columns:
                print(f"  WARNING: signal_filter column '{col_name}' "
                      f"not in dataframe -- skipping")
                continue
            col_vals = df[col_name].values
            for op, val in ops.items():
                if op == "min":
                    filt = col_vals >= val
                    op_str = ">="
                elif op == "max":
                    filt = col_vals <= val
                    op_str = "<="
                elif op == "eq":
                    filt = col_vals == val
                    op_str = "=="
                else:
                    print(f"  WARNING: unknown signal_filter op '{op}' "
                          f"-- skipping")
                    continue
                # Handle NaN: NaN comparisons return False, which is correct
                # (NaN bars should not pass the filter)
                signal_mask = signal_mask & filt
                n_after = int(signal_mask.sum())
                print(f"  Signal filter: {col_name} {op_str} {val} "
                      f"-> {n_after:,} signals remaining")

    # 8. Simulate with cooldown (+ fill model for limit entries)
    cooldown = experiment.get("cooldown_bars", 48)
    entry_type = (label_config or {}).get("entry_type", "market")

    if use_dynamic_labels and entry_type != "market":
        # D46b: limit-entry simulation with fill model
        from core.engine.fill_model import compute_entry_price

        fill_timeout = (label_config or {}).get("fill_timeout", 12)
        direction = (label_config or {}).get("direction", "long")
        trade_indices, fill_rates = _simulate_with_fills(
            signal_mask, label_arr, cooldown,
            df, entry_type, fill_timeout, direction,
        )
        n_filled = len(trade_indices)
        n_attempted = fill_rates["attempted"]
        print(f"  Limit entry: {entry_type}, timeout={fill_timeout}")
        print(f"  Attempted: {n_attempted}, filled: {n_filled} "
              f"({n_filled / n_attempted * 100:.1f}%)" if n_attempted > 0 else "")
    else:
        trade_indices = simulate(signal_mask, label_arr, cooldown)
        fill_rates = None

    n_trades = len(trade_indices)
    print(f"  Trades after CD={cooldown}: {n_trades:,}")

    # 9. Build R returns
    cost_per_r = experiment.get("cost_per_r", 0.05)
    r_returns = build_trade_returns(trade_indices, label_arr, r_target, cost_per_r)

    # 10. Date range for annualization
    if "bar_start_ts_utc" in df.columns:
        ts = df["bar_start_ts_utc"]
        first_covered = ts[covered].iloc[0] if covered.any() else ts.iloc[0]
        last_covered = ts[covered].iloc[-1] if covered.any() else ts.iloc[-1]
        years = (last_covered - first_covered).total_seconds() / (365.25 * 86400)
    else:
        # Fallback: assume 5m bars
        years = covered.sum() * 5 / (365.25 * 24 * 60)
    years = max(years, 0.1)
    print(f"  OOS span: {years:.2f} years")

    # 11. Equity simulation
    sizing_cfg = experiment.get("sizing", {})
    sizing_method = sizing_cfg.get("method", "fixed")

    if sizing_method == "kelly" and n_trades > 0:
        trade_probs = oos_probs[trade_indices]
        divisor = sizing_cfg.get("divisor", 40.0)
        odds = sizing_cfg.get("odds", 2.0)
        risk_pcts = kelly_fraction_array(
            trade_probs, odds=odds, divisor=divisor,
        )
        equity_path, max_dd = equity_sim_variable(
            r_returns, risk_pcts, initial_equity=10_000.0,
        )
        mean_risk = float(risk_pcts.mean())
        print(f"  Sizing: Kelly 1/{int(divisor)} "
              f"(mean risk {mean_risk * 100:.2f}%)")
    else:
        risk_pct = sizing_cfg.get("risk_pct", 0.02)
        equity_path, max_dd = equity_sim(
            r_returns, risk_pct, initial_equity=10_000.0,
        )
        print(f"  Sizing: fixed {risk_pct * 100:.1f}%")

    final_equity = equity_path[-1] if equity_path else 10_000.0

    # 12. Compute metrics
    metrics = compute_metrics(exp_id, r_returns, max_dd, final_equity, years)

    # 13. CSCV
    cscv_results = run_cscv(r_returns)

    # 14. Walk-forward all profitable?
    wf = cscv_results.get("walk_forward", [])
    wf_all_profitable = all(w.get("test_mean_r", 0) > 0 for w in wf) if wf else False

    # 15. Assemble gate inputs
    gate_inputs = {
        "trades_per_yr": metrics["trades_per_yr"],
        "oos_auc": oos_auc,
        "pbo": cscv_results.get("cscv", {}).get("pbo", 1.0),
        "psr": cscv_results.get("psr", {}).get("psr", 0.0),
        "wf_all_profitable": wf_all_profitable,
        "sharpe_ann": metrics["sharpe_ann"],
        "win_rate": metrics["win_rate"],
        "ev_r": metrics["ev_r"],
        "max_dd_pct": metrics["max_dd_pct"],
        "ece": ece,
    }

    # 16. Compute gates (tier-aware: D46b)
    tier = experiment.get("tier", "standard")
    tier_gates = _get_tier_gates(tier)
    gate_overrides = experiment.get("gates") or {}
    merged_gates = {**tier_gates, **gate_overrides}
    gates = compute_gates(gate_inputs, merged_gates)

    elapsed = time.time() - t0

    # 17. Assemble results
    results = {
        "id": exp_id,
        "status": "DONE",
        "tier": tier,
        "data_version": data_version,
        "n_features": len(features),
        "label": label_col,
        "label_config": label_config,
        "threshold": threshold,
        "cooldown_bars": cooldown,
        "oos_auc": round(oos_auc, 4),
        "ece": round(ece, 6),
        "ece_raw": round(ece_raw, 6),
        "feature_exclude": experiment.get("feature_exclude"),
        "signal_filter": signal_filter,
        "metrics": metrics,
        "cscv": cscv_results,
        "gates": gates,
        "n_gates_pass": sum(1 for v in gates.values() if v.get("pass")),
        "n_gates_total": len(gates),
        "wf_all_profitable": wf_all_profitable,
        "sizing": sizing_cfg,
        "elapsed_sec": round(elapsed, 1),
        "notes": experiment.get("notes", ""),
    }
    if fill_rates is not None:
        results["fill_rates"] = fill_rates

    # 18. Print report
    _print_report(exp_id, results)

    return results


# ---------------------------------------------------------------------------
# Safe runner (auto-retry + registry write)
# ---------------------------------------------------------------------------
def run_safe(experiment: dict) -> dict:
    """
    Run experiment with auto-retry.
    On first exception: retry once.
    On second exception: write FAILED + traceback to registry, return error dict.
    Never crashes the session.
    """
    exp_id = experiment.get("id", "unnamed")

    for attempt in range(1, 3):
        try:
            results = run_experiment(experiment)
            _append_result(results)
            return results
        except Exception:
            tb = traceback.format_exc()
            if attempt == 1:
                print(f"\n  WARNING: Attempt 1 failed for '{exp_id}'. Retrying...")
                print(f"  Error: {tb.splitlines()[-1]}")
                continue
            else:
                print(f"\n  ERROR: Attempt 2 failed for '{exp_id}'. "
                      f"Recording FAILED in registry.")
                print(tb)
                failed = {
                    "id": exp_id,
                    "status": "FAILED",
                    "error": tb.splitlines()[-1],
                    "traceback": tb,
                    "notes": experiment.get("notes", ""),
                }
                _append_result(failed)
                return failed

    # Should not reach here, but just in case
    return {"id": exp_id, "status": "FAILED", "error": "unexpected"}
