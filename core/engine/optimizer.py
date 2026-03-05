# core/engine/optimizer.py
# Experiment proposal engine + runner.
# D41b -- reads knowledge.md + registry.json to propose next experiment.
# D46b -- added dual-tier support (standard/weekly/monthly), tier RQs,
#          label_config for dynamic labeling, per-tier gate thresholds.
# ASCII-only output for cp1252 compatibility.
#
# Usage:
#   python -m core.engine.optimizer                  # checkpoint mode
#   python -m core.engine.optimizer --autonomous     # run up to 5

from __future__ import annotations

import json
import re
import sys
import os
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.engine.simulator import (
    _read_registry,
    load_data,
    run_safe,
    select_features,
    DATA_FILES,
    REGISTRY_PATH,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
KNOWLEDGE_PATH = PROJECT_ROOT / "core" / "signals" / "ict" / "knowledge.md"
PENDING_PATH = PROJECT_ROOT / "core" / "experiments" / "pending_approval.json"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODE = "checkpoint"  # "checkpoint" or "autonomous"
MAX_EXPERIMENTS_PER_SESSION = 5

# Production baseline (from E000 / D35)
PRODUCTION_AUC = 0.7937
PRODUCTION_LOGLOSS = 0.540  # approximate from D35

# ---------------------------------------------------------------------------
# Tier configurations (D46b)
# Each tier defines default experiment parameters and label_config.
# "standard" = current production (market entry, parquet labels).
# "weekly"   = ~52-156 trades/yr, medium conviction, market entry.
# "monthly"  = ~12 trades/yr, highest conviction, optional limit entries.
# ---------------------------------------------------------------------------
TIER_CONFIGS = {
    "standard": {
        "tier": "standard",
        "cooldown_bars": 576,
        "ml_threshold": 0.60,
        "label": "label_long_hit_2r_48c",
        "label_config": None,  # use parquet labels
    },
    "weekly": {
        "tier": "weekly",
        "cooldown_bars": 2016,  # 7 days
        "ml_threshold": 0.60,
        "label": "label_long_hit_2r_48c",
        "label_config": None,  # use parquet labels for now
    },
    "monthly": {
        "tier": "monthly",
        "cooldown_bars": 8640,  # 30 days
        "ml_threshold": 0.70,
        "label": "label_long_hit_2r_48c",
        "label_config": {
            "direction": "long",
            "target_r": 3.0,
            "stop_type": "swing_low",
            "max_bars": 96,
            "entry_type": "market",
            "entry_delay": 0,
            "fill_timeout": 12,
        },
    },
}

# Research question templates specific to tiers (D46b)
# These extend the main RESEARCH_QUESTIONS with tier-specific variants.
TIER_RESEARCH_QUESTIONS = [
    {
        "id": "TRQ1",
        "priority": 2,
        "tier": "monthly",
        "title": "Monthly: swing stop + 3R target vs ATR stop + 2R",
        "blocked": None,
        "requires_new_features": [],
        "feature_description": (
            "Test monthly tier with swing-based stops and 3R target. "
            "Uses dynamic labeler with stop_type=swing_low, target_r=3.0, "
            "max_bars=96. Expect fewer but higher quality trades."
        ),
        "mutation": {
            "notes": "TRQ1: Monthly tier with swing stops, 3R target.",
        },
    },
    {
        "id": "TRQ2",
        "priority": 2,
        "tier": "monthly",
        "title": "Monthly: limit_ob_mid entry vs market entry",
        "blocked": None,
        "requires_new_features": [],
        "feature_description": (
            "Test monthly tier with limit entry at OB midpoint. "
            "Uses fill model with entry_type=limit_ob_mid, fill_timeout=12. "
            "Expect better R:R but lower fill rate."
        ),
        "mutation": {
            "label_config_override": {
                "entry_type": "limit_ob_mid",
                "fill_timeout": 12,
            },
            "notes": "TRQ2: Monthly tier limit entry at OB midpoint.",
        },
    },
    {
        "id": "TRQ3",
        "priority": 2,
        "tier": "weekly",
        "title": "Weekly: CD=2016 (7d) with standard labels",
        "blocked": None,
        "requires_new_features": [],
        "feature_description": (
            "Test weekly tier with 7-day cooldown. Market entry, "
            "standard 2R/48c labels. Expect ~52-75 trades/yr."
        ),
        "mutation": {
            "notes": "TRQ3: Weekly tier, 7-day cooldown, standard labels.",
        },
    },
]

# ---------------------------------------------------------------------------
# Research question database
# Ordered by priority. Each has: id, title, blocked_reason (None = not blocked),
# requires_new_features (list of feature names not in v2),
# experiment_template (partial config dict).
# ---------------------------------------------------------------------------
RESEARCH_QUESTIONS = [
    # Priority 1
    {
        "id": "RQ1",
        "priority": 1,
        "title": "Does ob_quality_score beat raw ob_bull_age?",
        "blocked": None,
        "requires_new_features": ["ob_bull_quality", "ob_bear_quality"],
        "feature_description": (
            "ob_quality = recency_weight * displacement_strength * "
            "volume_surge. Computed by compute_ob_quality() in rules.py. "
            "On-the-fly augmentation in simulator.py."
        ),
        "mutation": {
            "notes": "RQ1: Add ob_quality features, retrain, "
                     "compare SHAP rank vs raw ob_bull_age (#1, 0.2057).",
        },
    },
    {
        "id": "RQ2",
        "priority": 1,
        "title": "Does H1 OTE distance score higher SHAP than 5m OTE?",
        "blocked": None,
        "requires_new_features": ["h1_ote_dist"],
        "feature_description": (
            "h1_ote_dist = OTE distance computed from H1-timeframe "
            "swing points instead of 5m swings. Requires H1 swing "
            "point data and Fibonacci retracement calculation."
        ),
        "mutation": {
            "notes": "RQ2: Add h1_ote_dist feature, retrain, "
                     "compare SHAP rank vs 5m ote_dist (#2, 0.1860).",
        },
    },
    {
        "id": "RQ3",
        "priority": 1,
        "title": "Does OTE + OB + FVG triple confluence produce WR > 70%?",
        "blocked": None,
        "requires_new_features": ["ict_triple_confluence"],
        "feature_description": (
            "ict_triple_confluence = binary flag when ote_dist is "
            "within OTE zone AND ob_bull_in_zone == 1 AND "
            "fvg_bull_in_zone == 1. Tests top-3 SHAP features "
            "aligning simultaneously."
        ),
        "mutation": {
            "notes": "RQ3: Add triple confluence feature, retrain. "
                     "Expect high WR but low signal count.",
        },
    },
    {
        "id": "RQ4",
        "priority": 1,
        "title": "Breaker blocks: reversal zone or price target?",
        "blocked": None,
        "requires_new_features": [
            "breaker_bull_age", "breaker_bear_age",
            "breaker_bull_dist", "breaker_bear_dist",
        ],
        "feature_description": (
            "Breaker block features: bars_since_mitigation and "
            "distance_from_mitigated_zone for both bull/bear. "
            "Mitigated OB flips to opposite direction signal. "
            "Requires tracking OB mitigation events in rules.py."
        ),
        "mutation": {
            "notes": "RQ4: Add breaker block features, retrain. "
                     "Expect breaker_age to rank in top 20.",
        },
    },
    {
        "id": "RQ5",
        "priority": 1,
        "title": "OI rate-of-change vs FVG formation events",
        "blocked": "Requires v3 parquet with OI data merged. "
                   "Do not propose until merge_v3.py complete.",
        "requires_new_features": [],  # v3 has oi_change_1h, oi_change_4h, oi_change_pct_*
        "feature_description": (
            "v3 includes oi_change_1h, oi_change_4h, oi_change_pct_1h, "
            "oi_change_pct_4h -- equivalent to oi_roc. Model learns "
            "OI-FVG interaction from existing columns."
        ),
        "mutation": {
            "notes": "RQ5: OI rate-of-change vs FVG. v3 OI columns included.",
        },
    },
    # RQ6-RQ7: Regime-conditional RQs (D47b, enabled by D47 HMM + ADX)
    {
        "id": "RQ6",
        "priority": 1,
        "title": "Regime-gated signals: HMM bull state as hard signal filter",
        "blocked": None,
        "requires_new_features": [],  # hmm_prob_bull in ONTHEFLY_FEATURES via D47
        "feature_description": (
            "Test whether gating signals on HMM bull state improves WR and EV. "
            "Add hmm_prob_bull > 0.60 as a hard filter on the signal mask in addition "
            "to the ML threshold. Hypothesis: ICT signals fire in all regimes but only "
            "have edge in trending bull conditions. Expect: fewer trades, higher WR, "
            "higher EV per trade, lower MaxDD. May fail MIN_TRADES_PER_YEAR gate -- "
            "if so, lower HMM threshold to 0.50 or use weekly tier gates."
        ),
        "mutation": {
            "signal_filter": {"hmm_prob_bull": {"min": 0.60}},
            "notes": "RQ6: HMM bull gate. Tests regime-conditional ICT edge.",
        },
    },
    {
        "id": "RQ7",
        "priority": 1,
        "title": "Regime features as soft ML inputs: HMM probs + ADX + interactions",
        "blocked": None,
        "requires_new_features": [],  # all in ONTHEFLY_FEATURES via D47
        "feature_description": (
            "Add all 10 D47 regime features as soft inputs to the ML model. "
            "No hard filter -- model decides how to weight them. Features: "
            "hmm_prob_bull, hmm_prob_bear, hmm_prob_calm, bb_width_normalized, "
            "atr_percentile_rank, regime_tag, ob_bull_age_x_hmm_bull, "
            "fvg_bull_x_trending, ote_x_regime. "
            "Hypothesis: model learns regime-conditional ICT patterns automatically. "
            "Compare SHAP rank of regime features vs ICT features post-experiment."
        ),
        "mutation": {
            "notes": "RQ7: Regime features as soft inputs. No hard filter.",
        },
    },
    # Priority 2
    {
        "id": "RQ8",
        "priority": 2,
        "title": "BPR: incremental signal above individual FVG?",
        "blocked": None,
        "requires_new_features": ["bpr_zone", "bpr_size", "bpr_age"],
        "feature_description": (
            "Balanced Price Range features. Detect overlapping "
            "bull+bear FVGs. bpr_zone=binary, bpr_size=overlap/ATR, "
            "bpr_age=bars since formation."
        ),
        "mutation": {
            "notes": "RQ8: Add BPR features, retrain, check SHAP delta.",
        },
    },
    {
        "id": "RQ9",
        "priority": 2,
        "title": "True tick CVD vs CLV: SHAP comparison",
        "blocked": "Requires v3 parquet with true tick CVD merged.",
        "requires_new_features": [],  # v3 has cvd_true_bar, cvd_true_zscore
        "feature_description": (
            "v3 includes cvd_true_bar, cvd_true_zscore from D37b. "
            "Compare SHAP rank vs CLV-based cvd_bar."
        ),
        "mutation": {
            "notes": "RQ9: True tick CVD vs CLV SHAP comparison. v3 columns included.",
        },
    },
    {
        "id": "RQ10",
        "priority": 2,
        "title": "Rising OI + FVG vs falling OI + FVG directional difference",
        "blocked": "Requires v3 parquet with OI data merged.",
        "requires_new_features": [],  # model learns interaction from oi_change + fvg columns
        "feature_description": (
            "v3 includes oi_change_* columns. LightGBM learns "
            "OI-FVG interaction natively from tree splits. "
            "No explicit interaction feature needed."
        ),
        "mutation": {
            "notes": "RQ10: Rising OI + FVG confluence. v3 OI columns included.",
        },
    },
    # Priority 3
    {
        "id": "RQ11",
        "priority": 3,
        "title": "Turtle Soup: sweep + immediate reversal WR",
        "blocked": None,
        "requires_new_features": ["turtle_soup_bull", "turtle_soup_bear"],
        "feature_description": (
            "Detect equal-high sweep followed by close below within "
            "3 bars. Binary event flags."
        ),
        "mutation": {
            "notes": "RQ11: Add Turtle Soup features, retrain.",
        },
    },
    {
        "id": "RQ12",
        "priority": 3,
        "title": "Judas Swing vs generic h4_sweep",
        "blocked": None,
        "requires_new_features": [
            "judas_sweep", "judas_magnitude", "judas_session",
        ],
        "feature_description": (
            "Session-open false breakout detection. Requires session "
            "boundary logic and sweep + reversal detection."
        ),
        "mutation": {
            "notes": "RQ12: Add Judas Swing features, compare vs h4_sweep.",
        },
    },
    {
        "id": "RQ13",
        "priority": 3,
        "title": "Internal (3-bar) vs external (20-bar) structure for 48-bar label",
        "blocked": None,
        "requires_new_features": [
            "int_swing_high_3", "int_swing_low_3",
            "ext_swing_high_20", "ext_swing_low_20",
        ],
        "feature_description": (
            "Compute swing points with pivot_n=3 and pivot_n=20. "
            "Compare SHAP contribution of each."
        ),
        "mutation": {
            "notes": "RQ13: Add multi-scale swing points, retrain.",
        },
    },
    {
        "id": "RQ14",
        "priority": 3,
        "title": "Short-side dedicated model with bear-specific features",
        "blocked": None,
        "requires_new_features": [],
        "feature_description": (
            "Train separate short model with label_short_hit_2r_48c. "
            "Not the same as symmetric rules (DE02)."
        ),
        "mutation": {
            "label": "label_short_hit_2r_48c",
            "notes": "RQ14: Dedicated short model. DE02 does NOT apply -- "
                     "this tests a bear-specific ML model, not symmetric rules.",
        },
    },
]

# Also: a special "pruned features" experiment that can always run
PRUNED_EXPERIMENT = {
    "id": "PRUNE",
    "priority": 0,
    "title": "Drop 81 dead features (D36 ablation confirmed AUC delta = 0)",
    "blocked": None,
    "requires_new_features": [],
    "feature_description": "No new features. Removes 81 lowest-SHAP features.",
    "mutation": {
        "feature_exclude": None,  # will be populated from knowledge
        "notes": "Feature pruning: remove 81 dead features (D36). "
                 "Expected AUC delta = 0.0000.",
    },
}


# ---------------------------------------------------------------------------
# Parse knowledge.md
# ---------------------------------------------------------------------------
def parse_knowledge() -> dict:
    """Extract structured data from knowledge.md."""
    if not KNOWLEDGE_PATH.exists():
        print(f"  WARNING: {KNOWLEDGE_PATH} not found")
        return {}

    text = KNOWLEDGE_PATH.read_text(encoding="utf-8")

    # Extract dead features list from Section 1
    dead_features = []
    # Look for known dead feature names from the 81 drop candidates
    # These are loaded from the SHAP JSON recommendations
    shap_json = PROJECT_ROOT / "legacy" / "scripts" / "results" / "shap_analysis_v2.json"
    if shap_json.exists():
        with open(shap_json, "r") as f:
            shap_data = json.load(f)
        dead_features = shap_data.get("recommendations", {}).get("drop_features", [])
        keep_always = shap_data.get("recommendations", {}).get("keep_always_features", [])
    else:
        keep_always = []

    # Extract experiment history (Section 4)
    completed_rqs = set()
    # Simple pattern: look for RQ mentions in experiment notes
    for rq in RESEARCH_QUESTIONS:
        rq_id = rq["id"]
        # Check if this RQ was already tested by searching experiment notes
        # in registry
        pass  # will be checked against registry in propose()

    return {
        "dead_features": dead_features,
        "keep_always": keep_always,
        "n_dead": len(dead_features),
    }


# ---------------------------------------------------------------------------
# Parse registry
# ---------------------------------------------------------------------------
def get_best_config(registry: dict, tier: str | None = None) -> dict | None:
    """Find the best experiment from registry (highest Sharpe among DONE).

    If tier is specified, returns best for that tier only.
    If tier is None, returns overall best.
    """
    experiments = registry.get("experiments", [])
    best = None
    best_sharpe = -float("inf")

    for exp in experiments:
        if exp.get("status") != "DONE":
            continue
        if tier is not None and exp.get("tier", "standard") != tier:
            continue
        sharpe = exp.get("metrics", {}).get("sharpe_ann", 0)
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best = exp

    return best


def get_tested_rqs(registry: dict) -> set[str]:
    """Find which RQs (standard + tier) have already been tested."""
    tested = set()
    all_rqs = list(RESEARCH_QUESTIONS) + TIER_RESEARCH_QUESTIONS
    for exp in registry.get("experiments", []):
        notes = exp.get("notes", "")
        for rq in all_rqs:
            if rq["id"] in notes:
                tested.add(rq["id"])
    return tested


def get_exhausted_params(registry: dict) -> dict[str, int]:
    """Track consecutive failures per parameter for mutation rules."""
    # Count consecutive failures at end of experiment list
    failures = {}
    experiments = registry.get("experiments", [])
    for exp in reversed(experiments):
        if exp.get("status") != "DONE":
            continue
        gates = exp.get("gates", {})
        all_pass = all(g.get("pass", False) for g in gates.values())
        notes = exp.get("notes", "")

        if not all_pass:
            # Extract RQ from notes
            for rq in RESEARCH_QUESTIONS:
                if rq["id"] in notes:
                    failures[rq["id"]] = failures.get(rq["id"], 0) + 1
        else:
            break  # stop at first success

    return failures


# ---------------------------------------------------------------------------
# Check data availability
# ---------------------------------------------------------------------------
def check_v3_exists() -> bool:
    """Check if v3 dataset exists."""
    return DATA_FILES["v3"].exists()


def check_features_available(feature_names: list[str]) -> list[str]:
    """Check which features are missing from the current dataset.

    Checks both the parquet schema AND on-the-fly features from
    simulator.ONTHEFLY_FEATURES (features computed at runtime by rules.py).
    """
    from core.engine.simulator import ONTHEFLY_FEATURES

    try:
        import pyarrow.parquet as pq
        # Read just the schema (fast, no data loaded)
        for version in ["v3", "v2", "v1"]:
            path = DATA_FILES.get(version)
            if path and path.exists():
                schema = pq.read_schema(path)
                existing = set(schema.names)
                # Also include on-the-fly features
                existing.update(ONTHEFLY_FEATURES.keys())
                missing = [f for f in feature_names if f not in existing]
                return missing
    except Exception:
        pass
    return feature_names  # assume all missing if we can't check


# ---------------------------------------------------------------------------
# Estimate runtime
# ---------------------------------------------------------------------------
def estimate_runtime(n_features: int, n_folds: int, device: str) -> float:
    """Estimate runtime in minutes. Rough heuristic."""
    # GPU: ~1.5 min per fold for 500 features
    # CPU: ~4 min per fold for 500 features
    per_fold_base = 1.5 if device == "gpu" else 4.0
    scale = n_features / 500.0
    return per_fold_base * n_folds * scale


# ---------------------------------------------------------------------------
# Propose next experiment
# ---------------------------------------------------------------------------
def propose_next_experiment(
    registry: dict,
    knowledge: dict,
) -> dict | None:
    """
    Propose the next experiment based on knowledge.md and registry.

    Returns experiment config dict, or None if nothing to propose.
    """
    best = get_best_config(registry)
    tested = get_tested_rqs(registry)
    exhausted = get_exhausted_params(registry)

    # Check if feature pruning has been done
    pruning_done = any(
        "prun" in exp.get("notes", "").lower() or
        "prun" in exp.get("id", "").lower()
        for exp in registry.get("experiments", [])
    )

    # Base config (from production or best experiment)
    base_threshold = 0.60
    base_cooldown = 576
    base_label = "label_long_hit_2r_48c"
    base_device = "gpu"

    if best:
        base_threshold = best.get("threshold", 0.60)
        base_cooldown = best.get("cooldown_bars", 576)
        base_label = best.get("label", base_label)

    # Feature exclusion list (from dead features)
    dead_features = knowledge.get("dead_features", [])

    # PRUNE has priority 0 (highest) -- check before RQs
    if not pruning_done and dead_features:
        n_experiments = len(registry.get("experiments", []))
        exp_id = f"E{n_experiments:03d}_prune"
        experiment = {
            "id": exp_id,
            "signal_domain": "ml",
            "features": "all",
            "feature_exclude": dead_features,
            "label": base_label,
            "ml_config": {
                "threshold": base_threshold,
                "model": "lgbm",
                "device": base_device,
                "embargo_bars": 48,
            },
            "cooldown_bars": base_cooldown,
            "sizing": {
                "method": "kelly",
                "divisor": 40.0,
                "odds": 2.0,
            },
            "cost_per_r": 0.05,
            "notes": "Feature pruning: remove 81 dead features (D36).",
        }
        est_features = 508 - len(dead_features)
        est_minutes = estimate_runtime(est_features, 11, base_device)
        return {
            "experiment": experiment,
            "rq": PRUNED_EXPERIMENT,
            "missing_features": [],
            "estimated_minutes": round(est_minutes, 1),
            "base_experiment": best.get("id") if best else "none",
            "base_auc": best.get("oos_auc") if best else None,
        }

    # Iterate through RQs in priority order
    for rq in RESEARCH_QUESTIONS:
        rq_id = rq["id"]

        # Skip if already tested
        if rq_id in tested:
            continue

        # Skip if blocked
        if rq["blocked"]:
            # Special check: if blocked on v3 and v3 exists, unblock
            if "v3" in str(rq["blocked"]) and check_v3_exists():
                pass  # unblocked
            else:
                continue

        # Skip if exhausted (2 consecutive failures)
        if exhausted.get(rq_id, 0) >= 2:
            continue

        # Check feature availability -- skip if features missing
        new_feats = rq.get("requires_new_features", [])
        missing = check_features_available(new_feats) if new_feats else []
        if missing:
            print(f"  Skipping {rq_id}: missing features {missing}")
            continue

        # Generate experiment ID
        n_experiments = len(registry.get("experiments", []))
        exp_id = f"E{n_experiments:03d}_{rq_id.lower()}"

        # Build experiment config
        mutation = rq.get("mutation", {})
        label = mutation.get("label", base_label)

        experiment = {
            "id": exp_id,
            "signal_domain": "ml",
            "features": "all",
            "feature_exclude": dead_features if dead_features else None,
            "label": label,
            "ml_config": {
                "threshold": base_threshold,
                "model": "lgbm",
                "device": base_device,
                "embargo_bars": 48,
            },
            "cooldown_bars": base_cooldown,
            "sizing": {
                "method": "kelly",
                "divisor": 40.0,
                "odds": 2.0,
            },
            "cost_per_r": 0.05,
            "notes": mutation.get("notes", f"{rq_id}: {rq['title']}"),
        }

        # Propagate signal_filter from mutation (D47b)
        if "signal_filter" in mutation:
            experiment["signal_filter"] = mutation["signal_filter"]

        # Estimate runtime
        est_features = 508 - len(dead_features) + len(new_feats)
        est_folds = 11
        est_minutes = estimate_runtime(est_features, est_folds, base_device)

        # Build proposal
        proposal = {
            "experiment": experiment,
            "rq": rq,
            "missing_features": missing,
            "estimated_minutes": round(est_minutes, 1),
            "base_experiment": best.get("id") if best else "none",
            "base_auc": best.get("oos_auc") if best else None,
        }

        return proposal

    # D46b: Check tier research questions after standard RQs
    for trq in TIER_RESEARCH_QUESTIONS:
        trq_id = trq["id"]

        # Skip if already tested
        if trq_id in tested:
            continue

        # Skip if blocked
        if trq.get("blocked"):
            continue

        tier_name = trq.get("tier", "standard")
        tier_cfg = TIER_CONFIGS.get(tier_name, TIER_CONFIGS["standard"])

        # Generate experiment ID
        n_experiments = len(registry.get("experiments", []))
        exp_id = f"E{n_experiments:03d}_{trq_id.lower()}"

        # Build label_config from tier defaults + mutation overrides
        tier_label_config = tier_cfg.get("label_config")
        if tier_label_config is not None:
            tier_label_config = dict(tier_label_config)
            override = trq.get("mutation", {}).get("label_config_override", {})
            tier_label_config.update(override)

        mutation = trq.get("mutation", {})
        experiment = {
            "id": exp_id,
            "signal_domain": "ml",
            "features": "all",
            "feature_exclude": dead_features if dead_features else None,
            "label": tier_cfg.get("label", base_label),
            "tier": tier_name,
            "ml_config": {
                "threshold": tier_cfg.get("ml_threshold", base_threshold),
                "model": "lgbm",
                "device": base_device,
                "embargo_bars": 48,
            },
            "cooldown_bars": tier_cfg.get("cooldown_bars", base_cooldown),
            "sizing": {
                "method": "kelly",
                "divisor": 40.0,
                "odds": 2.0,
            },
            "cost_per_r": 0.05,
            "notes": mutation.get("notes", f"{trq_id}: {trq['title']}"),
        }

        # Add label_config for dynamic labeling tiers
        if tier_label_config is not None:
            experiment["label_config"] = tier_label_config

        est_features = 508 - len(dead_features)
        est_minutes = estimate_runtime(est_features, 11, base_device)

        return {
            "experiment": experiment,
            "rq": trq,
            "missing_features": [],
            "estimated_minutes": round(est_minutes, 1),
            "base_experiment": best.get("id") if best else "none",
            "base_auc": best.get("oos_auc") if best else None,
        }

    print("  No experiments to propose. All RQs tested, blocked, or exhausted.")
    return None


# ---------------------------------------------------------------------------
# Print proposal
# ---------------------------------------------------------------------------
def print_proposal(proposal: dict) -> None:
    """Print a readable experiment proposal."""
    rq = proposal["rq"]
    exp = proposal["experiment"]
    missing = proposal["missing_features"]

    print()
    print("=" * 60)
    print("  PROPOSED EXPERIMENT")
    print("=" * 60)
    print()
    print(f"  ID:       {exp['id']}")
    print(f"  RQ:       {rq['id']} -- {rq['title']}")
    print(f"  Priority: {rq['priority']}")
    print(f"  Based on: {proposal['base_experiment']} "
          f"(AUC={proposal['base_auc']})")
    print()
    tier = exp.get("tier", "standard")
    print(f"  Tier:       {tier}")
    print(f"  Label:      {exp['label']}")
    lc = exp.get("label_config")
    if lc:
        print(f"  Label cfg:  {lc.get('stop_type', 'atr')} stop, "
              f"{lc.get('target_r', 2.0)}R, "
              f"{lc.get('max_bars', 48)}c, "
              f"entry={lc.get('entry_type', 'market')}")
    print(f"  Threshold:  {exp['ml_config']['threshold']}")
    print(f"  Cooldown:   {exp['cooldown_bars']}")
    print(f"  Sizing:     Kelly 1/{int(exp['sizing']['divisor'])}")
    print(f"  Device:     {exp['ml_config']['device']}")

    exclude = exp.get("feature_exclude")
    if exclude:
        print(f"  Excluded:   {len(exclude)} dead features (D36)")
    sig_filter = exp.get("signal_filter")
    if sig_filter:
        for col, ops in sig_filter.items():
            for op, val in ops.items():
                op_str = {"min": ">=", "max": "<=", "eq": "=="}.get(op, op)
                print(f"  Sig filter: {col} {op_str} {val}")
    print()

    if missing:
        print("  *** REQUIRES FEATURE ENGINEERING ***")
        print(f"  Missing features: {missing}")
        print(f"  Description: {rq['feature_description']}")
        print()
        print("  These features must be computed and added to the")
        print("  dataset (or computed on-the-fly) before this")
        print("  experiment can run. Approve to proceed with")
        print("  feature engineering, or reject to skip.")
        print()

    est = proposal["estimated_minutes"]
    print(f"  Estimated runtime: ~{est:.0f} min "
          f"({exp['ml_config']['device']})")

    if est > 90:
        print("  WARNING: Estimated runtime > 90 min!")

    print()
    print("-" * 60)


# ---------------------------------------------------------------------------
# Checkpoint mode
# ---------------------------------------------------------------------------
def checkpoint_approve(proposal: dict) -> bool:
    """Ask for approval in checkpoint mode."""
    try:
        # Check if running interactively
        if sys.stdin.isatty():
            response = input("  Run this experiment? [y/n]: ").strip().lower()
            return response in ("y", "yes")
        else:
            # Non-interactive: write to pending_approval.json
            PENDING_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(PENDING_PATH, "w") as f:
                json.dump(proposal["experiment"], f, indent=2, default=str)
            print(f"  CHECKPOINT: review {PENDING_PATH}")
            print("  Re-run with --approve to confirm.")
            return False
    except EOFError:
        # Non-interactive fallback
        PENDING_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PENDING_PATH, "w") as f:
            json.dump(proposal["experiment"], f, indent=2, default=str)
        print(f"  CHECKPOINT: review {PENDING_PATH}")
        print("  Re-run with --approve to confirm.")
        return False


# ---------------------------------------------------------------------------
# Post-run analysis
# ---------------------------------------------------------------------------
def diagnose_failure(results: dict) -> list[str]:
    """Diagnose why an experiment failed gates and suggest fixes."""
    suggestions = []
    gates = results.get("gates", {})
    metrics = results.get("metrics", {})

    for name, info in gates.items():
        if info.get("pass"):
            continue

        val = info.get("value")
        thresh = info.get("threshold")

        if name == "MIN_TRADES_PER_YEAR":
            suggestions.append(
                f"Low trade count ({val:.0f} < {thresh}): "
                f"reduce threshold by 0.02 OR reduce cooldown by 96 bars"
            )
        elif name == "MIN_WR":
            suggestions.append(
                f"Low WR ({val:.4f} < {thresh}): "
                f"raise threshold by 0.02 OR add OB quality filter"
            )
        elif name == "MAX_DRAWDOWN":
            suggestions.append(
                f"High drawdown ({val:.4f} > {thresh}): "
                f"reduce cooldown by 96 OR reduce Kelly fraction"
            )
        elif name == "MIN_OOS_AUC":
            suggestions.append(
                f"Low AUC ({val:.4f} < {thresh}): "
                f"add highest-SHAP unencoded feature from knowledge.md"
            )
        elif name == "MAX_ECE":
            suggestions.append(
                f"High ECE ({val:.4f} > {thresh}): "
                f"apply isotonic regression (core/engine/calibrator.py)"
            )
        elif name == "MIN_SHARPE":
            suggestions.append(
                f"Low Sharpe ({val:.4f} < {thresh}): "
                f"investigate EV or trade frequency"
            )
        elif name == "MIN_EV_R":
            suggestions.append(
                f"Low EV ({val:.4f} < {thresh}): "
                f"raise threshold or add quality filter"
            )

    return suggestions


def on_gate_pass(results: dict, experiment: dict) -> None:
    """Actions to take when all gates pass."""
    exp_id = results.get("id", "unknown")
    print(f"\n  GATE PASS: {exp_id}")
    print("  Post-pass actions:")
    print("    1. Run: python -m core.engine.shap_runner --exp-id " + exp_id)
    print("    2. SHAP runner will update knowledge.md Section 1")
    print("    3. Append experiment to knowledge.md Section 4")
    print("    4. Registry already updated by run_safe()")

    oos_auc = results.get("oos_auc", 0)
    auc_delta = oos_auc - PRODUCTION_AUC
    if auc_delta >= 0.005:
        print(f"    5. AUC delta = {auc_delta:+.4f} >= 0.005 -- "
              f"UPDATE core/reports/best_configs.json")
    else:
        print(f"    5. AUC delta = {auc_delta:+.4f} < 0.005 -- "
              f"no promotion (informational only)")

    ece = results.get("ece", 0)
    if ece > 0.05:
        print(f"    6. ECE = {ece:.4f} > 0.05 -- "
              f"run calibrator.py before promotion")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = sys.argv[1:]
    mode = MODE

    if "--autonomous" in args:
        mode = "autonomous"
    if "--approve" in args:
        # Load pending experiment and run it
        if PENDING_PATH.exists():
            with open(PENDING_PATH, "r") as f:
                experiment = json.load(f)
            print(f"  Running approved experiment: {experiment.get('id')}")
            results = run_safe(experiment)

            gates = results.get("gates", {})
            all_pass = all(g.get("pass", False) for g in gates.values())

            if all_pass:
                on_gate_pass(results, experiment)
            else:
                suggestions = diagnose_failure(results)
                if suggestions:
                    print("\n  FAILURE DIAGNOSIS:")
                    for s in suggestions:
                        print(f"    - {s}")

            PENDING_PATH.unlink(missing_ok=True)
            return
        else:
            print("  No pending experiment found.")
            return

    print()
    print("=" * 60)
    print("  OPTIMIZER (D41b)")
    print(f"  Mode: {mode}")
    print(f"  Max experiments: {MAX_EXPERIMENTS_PER_SESSION}")
    print("=" * 60)

    # Load state
    registry = _read_registry()
    knowledge = parse_knowledge()
    n_done = len([e for e in registry.get("experiments", [])
                  if e.get("status") == "DONE"])
    print(f"\n  Registry: {n_done} completed experiments")
    print(f"  Knowledge: {knowledge.get('n_dead', 0)} dead features identified")

    best = get_best_config(registry)
    if best:
        print(f"  Best config: {best['id']} "
              f"(AUC={best.get('oos_auc')}, "
              f"Sharpe={best.get('metrics', {}).get('sharpe_ann', 0):.2f})")
    else:
        print("  No completed experiments in registry.")

    # Run experiment loop
    n_run = 0
    n_passed = 0
    best_auc = PRODUCTION_AUC

    while n_run < MAX_EXPERIMENTS_PER_SESSION:
        # Refresh registry each iteration
        registry = _read_registry()

        proposal = propose_next_experiment(registry, knowledge)
        if proposal is None:
            break

        print_proposal(proposal)

        # Check if features are missing
        missing = proposal.get("missing_features", [])
        if missing:
            print("  Cannot run automatically -- feature engineering required.")
            if mode == "checkpoint":
                print("  Write feature engineering code first, then re-run.")
                # Write proposal to pending for reference
                PENDING_PATH.parent.mkdir(parents=True, exist_ok=True)
                with open(PENDING_PATH, "w") as f:
                    json.dump({
                        "experiment": proposal["experiment"],
                        "missing_features": missing,
                        "feature_description": proposal["rq"]["feature_description"],
                    }, f, indent=2, default=str)
                print(f"  Proposal saved: {PENDING_PATH}")
            break

        if mode == "checkpoint":
            approved = checkpoint_approve(proposal)
            if not approved:
                print("  Experiment not approved. Stopping.")
                break
        else:
            print("  [autonomous] Running...")

        # Run the experiment
        experiment = proposal["experiment"]
        t0 = time.time()
        results = run_safe(experiment)
        elapsed = time.time() - t0
        n_run += 1

        # Check gates
        gates = results.get("gates", {})
        all_pass = all(g.get("pass", False) for g in gates.values())

        if all_pass:
            n_passed += 1
            on_gate_pass(results, experiment)

            auc = results.get("oos_auc", 0)
            if auc > best_auc:
                best_auc = auc
        else:
            suggestions = diagnose_failure(results)
            if suggestions:
                print("\n  FAILURE DIAGNOSIS:")
                for s in suggestions:
                    print(f"    - {s}")

        print(f"\n  Experiment {n_run} complete in {elapsed:.1f}s")
        print()

    # Session summary
    print()
    print("=" * 60)
    print("  SESSION SUMMARY")
    print("=" * 60)
    print(f"  Experiments run: {n_run}")
    print(f"  Gates passed:    {n_passed}")
    print(f"  Best AUC:        {best_auc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
