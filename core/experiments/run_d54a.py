"""D54a: Clean post-D51/D53 baseline using E002_prune parameters.
Run from project root: python core/experiments/run_d54a.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.engine.simulator import run_safe

experiment = {
    "id": "D54a_baseline_long",
    "signal_domain": "ml",
    "features": "all",
    "feature_exclude": [],
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
    "notes": "D54a: clean baseline on v3_train with D53 ICT features. "
             "E002_prune equivalent. 288-bar embargo, label purging. "
             "All features (no prune list -- fresh evaluation).",
}

if __name__ == "__main__":
    print("=" * 60)
    print("D54a: Post-D51/D53 Baseline (E002_prune equivalent)")
    print("=" * 60)
    result = run_safe(experiment)
    if result.get("status") == "FAILED":
        print(f"\nFAILED: {result.get('error')}")
        sys.exit(1)
    print("\nD54a complete.")
