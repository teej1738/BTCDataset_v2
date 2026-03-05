"""D55: Aggressive feature prune cycle.
  D55a: Drop Tier 3 (keep Tier 1 + Tier 2 = 110 features)
  D55b: Tier 1 only (64 features)
Run from project root: python core/experiments/run_d55.py [a|b]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.engine.simulator import run_safe

TIERS_PATH = Path(__file__).resolve().parent / "d55_tiers.json"


def load_tiers():
    with open(TIERS_PATH) as f:
        return json.load(f)


def make_d55a():
    """Drop Tier 3 features (keep Tier 1 + Tier 2 = 110 features)."""
    tiers = load_tiers()
    return {
        "id": "D55a_drop_tier3",
        "signal_domain": "ml",
        "features": "all",
        "feature_exclude": tiers["tier3"],
        "label": "label_long_hit_2r_48c",
        "label_config": None,
        "ml_config": {
            "threshold": 0.60,
            "model": "lgbm",
            "device": "gpu",
            "embargo_bars": 288,
        },
        "cooldown_bars": 576,
        "sizing": {
            "method": "kelly",
            "divisor": 40.0,
            "odds": 2.0,
        },
        "tier": "standard",
        "signal_filter": None,
        "notes": f"D55a: Drop Tier 3 ({tiers['tier3_count']} features with "
                 f"|SHAP| < 0.003). Keep Tier 1 ({tiers['tier1_count']}) + "
                 f"Tier 2 ({tiers['tier2_count']}) = "
                 f"{tiers['tier1_count'] + tiers['tier2_count']} features. "
                 f"Baseline AUC 0.7933 (D54a). Target: within 0.002.",
    }


def make_d55b():
    """Tier 1 only (64 features)."""
    tiers = load_tiers()
    exclude = tiers["tier2"] + tiers["tier3"]
    return {
        "id": "D55b_tier1_only",
        "signal_domain": "ml",
        "features": "all",
        "feature_exclude": exclude,
        "label": "label_long_hit_2r_48c",
        "label_config": None,
        "ml_config": {
            "threshold": 0.60,
            "model": "lgbm",
            "device": "gpu",
            "embargo_bars": 288,
        },
        "cooldown_bars": 576,
        "sizing": {
            "method": "kelly",
            "divisor": 40.0,
            "odds": 2.0,
        },
        "tier": "standard",
        "signal_filter": None,
        "notes": f"D55b: Tier 1 only ({tiers['tier1_count']} features with "
                 f"|SHAP| >= 0.010). Dropped Tier 2 ({tiers['tier2_count']}) "
                 f"and Tier 3 ({tiers['tier3_count']}). "
                 f"Baseline AUC 0.7933 (D54a). Target: within 0.002.",
    }


if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else "a"

    if variant == "a":
        experiment = make_d55a()
        label = "D55a: Drop Tier 3 (keep 110 features)"
    elif variant == "b":
        experiment = make_d55b()
        label = "D55b: Tier 1 Only (64 features)"
    else:
        print(f"Unknown variant: {variant}. Use 'a' or 'b'.")
        sys.exit(1)

    print("=" * 60)
    print(label)
    print("=" * 60)
    result = run_safe(experiment)
    if result.get("status") == "FAILED":
        print(f"\nFAILED: {result.get('error')}")
        sys.exit(1)
    print(f"\n{experiment['id']} complete.")
