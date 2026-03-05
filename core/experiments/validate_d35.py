# core/experiments/validate_d35.py
# D40 (Step 5A): Reproduce D35 production config through the new engine.
# Gate before any new experiments can run.
#
# Uses pre-computed OOS probs from baseline_d35.npy (no retraining).
# Differences from D35 reflect engine code, not model differences.
#
# Run from project root: python core/experiments/validate_d35.py

import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

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
)
from core.engine.labels import parse_label_col
from core.engine.sizing import equity_sim
from core.engine.simulator import (
    _append_result,
    _print_report,
    load_data,
)

# ---------------------------------------------------------------------------
# Config: D35 production config
# ---------------------------------------------------------------------------
LABEL_COL = "label_long_hit_2r_48c"
THRESHOLD = 0.60
COOLDOWN_BARS = 576
COST_PER_R = 0.05
RISK_PCT = 0.02  # D35 used fixed 2%, not variable Kelly
INITIAL_EQUITY = 10_000.0

PROBS_PATH = os.path.join(
    os.path.dirname(__file__), "models", "baseline_d35.npy"
)

# ---------------------------------------------------------------------------
# Expected values from D35 (with tolerances)
# ---------------------------------------------------------------------------
EXPECTED = {
    "win_rate":     {"value": 0.654,  "tol": 0.010},
    "ev_r":         {"value": 0.912,  "tol": 0.050},
    "pf":           {"value": 3.51,   "tol": 0.15},
    "max_dd_pct":   {"value": 12.0,   "tol": 1.5},
    "trades_per_yr":{"value": 180.0,  "tol": 15.0},
    "pbo":          {"value": 0.0,    "max": 0.01},
    "psr":          {"value": 1.0,    "min": 0.999},
}


def check_tolerance(name, actual, expected_cfg):
    """Check if actual is within tolerance of expected. Returns (pass, detail_str)."""
    if "tol" in expected_cfg:
        target = expected_cfg["value"]
        tol = expected_cfg["tol"]
        ok = abs(actual - target) <= tol
        return ok, f"{name:20s}  expected={target:>10.4f}  actual={actual:>10.4f}  tol=+/-{tol}  {'PASS' if ok else '** FAIL **'}"
    elif "max" in expected_cfg:
        threshold = expected_cfg["max"]
        ok = actual <= threshold
        return ok, f"{name:20s}  max={threshold:>10.4f}  actual={actual:>10.4f}  {'PASS' if ok else '** FAIL **'}"
    elif "min" in expected_cfg:
        threshold = expected_cfg["min"]
        ok = actual >= threshold
        return ok, f"{name:20s}  min={threshold:>10.4f}  actual={actual:>10.4f}  {'PASS' if ok else '** FAIL **'}"
    return True, f"{name:20s}  actual={actual}"


def main():
    t0 = time.time()
    print("=" * 60)
    print("  D40: Engine Validation Run (reproduce D35)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    df, data_version = load_data()

    # ------------------------------------------------------------------
    # 2. Load pre-computed OOS probs
    # ------------------------------------------------------------------
    print(f"  Loading probs: {PROBS_PATH}")
    loaded_probs = np.load(PROBS_PATH)
    print(f"  Probs shape: {loaded_probs.shape}")

    # ------------------------------------------------------------------
    # 3. Align probs to full dataframe
    # ------------------------------------------------------------------
    label_info = parse_label_col(LABEL_COL)
    r_target = label_info["r_multiple"]

    raw_labels = df[LABEL_COL].values
    label_valid = ~np.isnan(raw_labels)
    label_arr = np.where(label_valid, raw_labels, 0).astype(int)

    n_valid = int(label_valid.sum())
    if n_valid != len(loaded_probs):
        print(f"  ERROR: label valid count ({n_valid}) != probs length ({len(loaded_probs)})")
        print("  VALIDATION FAIL -- do not proceed")
        return

    # Map probs to full-length array (NaN where labels are NaN)
    oos_probs = np.full(len(df), np.nan)
    oos_probs[label_valid] = loaded_probs
    print(f"  Probs aligned: {len(oos_probs)} total, {(~np.isnan(oos_probs)).sum()} covered")

    # ------------------------------------------------------------------
    # 4. Compute OOS AUC
    # ------------------------------------------------------------------
    covered = ~np.isnan(oos_probs)
    eval_mask = covered & label_valid
    oos_auc = compute_auc(label_arr[eval_mask], oos_probs[eval_mask])
    print(f"  OOS AUC: {oos_auc:.4f} ({int(eval_mask.sum()):,} bars)")

    # ------------------------------------------------------------------
    # 5. Compute ECE
    # ------------------------------------------------------------------
    ece = compute_ece(oos_probs[label_valid], label_arr[label_valid])
    print(f"  ECE: {ece:.6f}")

    # ------------------------------------------------------------------
    # 6. Build signal mask
    # ------------------------------------------------------------------
    signal_mask = covered & label_valid & (oos_probs >= THRESHOLD)
    print(f"  Signals at t={THRESHOLD}: {int(signal_mask.sum()):,}")

    # ------------------------------------------------------------------
    # 7. Simulate with cooldown
    # ------------------------------------------------------------------
    trade_indices = simulate(signal_mask, label_arr, COOLDOWN_BARS)
    n_trades = len(trade_indices)
    print(f"  Trades after CD={COOLDOWN_BARS}: {n_trades:,}")

    # ------------------------------------------------------------------
    # 8. Build R returns
    # ------------------------------------------------------------------
    r_returns = build_trade_returns(trade_indices, label_arr, r_target, COST_PER_R)

    # ------------------------------------------------------------------
    # 9. Date range for annualization
    # ------------------------------------------------------------------
    if "bar_start_ts_utc" in df.columns:
        ts = df["bar_start_ts_utc"]
        first_covered = ts[covered].iloc[0]
        last_covered = ts[covered].iloc[-1]
        years = (last_covered - first_covered).total_seconds() / (365.25 * 86400)
    else:
        years = covered.sum() * 5 / (365.25 * 24 * 60)
    years = max(years, 0.1)
    print(f"  OOS span: {years:.2f} years")

    # ------------------------------------------------------------------
    # 10. Equity simulation (fixed 2% risk -- matches D35 legacy code)
    # ------------------------------------------------------------------
    equity_path, max_dd = equity_sim(
        r_returns, RISK_PCT, initial_equity=INITIAL_EQUITY,
    )
    final_equity = equity_path[-1] if equity_path else INITIAL_EQUITY
    print(f"  Sizing: fixed {RISK_PCT*100:.0f}% risk")

    # ------------------------------------------------------------------
    # 11. Compute metrics
    # ------------------------------------------------------------------
    metrics = compute_metrics("E000_validate_d35", r_returns, max_dd, final_equity, years)

    # ------------------------------------------------------------------
    # 12. CSCV
    # ------------------------------------------------------------------
    cscv_results = run_cscv(r_returns)

    # ------------------------------------------------------------------
    # 13. Walk-forward all profitable?
    # ------------------------------------------------------------------
    wf = cscv_results.get("walk_forward", [])
    wf_all_profitable = all(w.get("test_mean_r", 0) > 0 for w in wf) if wf else False

    # ------------------------------------------------------------------
    # 14. Gates
    # ------------------------------------------------------------------
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
    gates = compute_gates(gate_inputs)

    # ------------------------------------------------------------------
    # 15. Assemble results
    # ------------------------------------------------------------------
    elapsed = time.time() - t0

    results = {
        "id": "E000_validate_d35",
        "status": "DONE",
        "data_version": data_version,
        "label": LABEL_COL,
        "threshold": THRESHOLD,
        "cooldown_bars": COOLDOWN_BARS,
        "oos_auc": round(oos_auc, 4),
        "ece": round(ece, 6),
        "metrics": metrics,
        "cscv": cscv_results,
        "gates": gates,
        "n_gates_pass": sum(1 for v in gates.values() if v.get("pass")),
        "n_gates_total": len(gates),
        "wf_all_profitable": wf_all_profitable,
        "sizing": {
            "method": "fixed",
            "risk_pct": RISK_PCT,
        },
        "elapsed_sec": round(elapsed, 1),
        "notes": "D40 validation: reproduce D35 through engine (no retraining)",
    }

    # ------------------------------------------------------------------
    # 16. Print standard report
    # ------------------------------------------------------------------
    _print_report("E000_validate_d35", results)

    # ------------------------------------------------------------------
    # 17. Side-by-side comparison
    # ------------------------------------------------------------------
    pbo = cscv_results.get("cscv", {}).get("pbo", 1.0)
    psr = cscv_results.get("psr", {}).get("psr", 0.0)

    actual_values = {
        "win_rate": metrics["win_rate"],
        "ev_r": metrics["ev_r"],
        "pf": metrics["profit_factor"],
        "max_dd_pct": metrics["max_dd_pct"],
        "trades_per_yr": metrics["trades_per_yr"],
        "pbo": pbo,
        "psr": psr,
    }

    print()
    print("=" * 60)
    print("  SIDE-BY-SIDE: D35 Expected vs Engine Result")
    print("=" * 60)

    all_pass = True
    for name, cfg in EXPECTED.items():
        actual = actual_values[name]
        ok, line = check_tolerance(name, actual, cfg)
        print(f"  {line}")
        if not ok:
            all_pass = False

    print()
    print("-" * 60)

    if all_pass:
        print("  VALIDATION PASS")
        print("-" * 60)
        # Write E000 to registry
        _append_result(results)

        # Save JSON to results dir
        results_dir = os.path.join(
            os.path.dirname(__file__), "results"
        )
        os.makedirs(results_dir, exist_ok=True)
        json_path = os.path.join(results_dir, "E000_validate_d35.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  JSON saved: {json_path}")
    else:
        print("  VALIDATION FAIL -- do not proceed")
        print("-" * 60)
        print("  Do NOT write to registry.")

    print(f"\n  Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
