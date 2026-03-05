# core/config/parameters.py
# D52 -- Full search space for Optuna hyperparameter optimization.
# Research-validated defaults from D53 spec + GPT synthesis.
# ASCII-only output for cp1252 compatibility.
#
# Usage:
#   from core.config.parameters import SEARCH_SPACE, PARAMETER_GROUPS
#   from core.config.parameters import get_untested_options, record_tested

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Search space -- each entry: default, options, description
# ---------------------------------------------------------------------------
SEARCH_SPACE: dict[str, dict[str, Any]] = {
    # -- Quick wins (trade-level) --
    "ml_threshold": {
        "default": 0.60,
        "options": [0.55, 0.57, 0.60, 0.62, 0.65, 0.70, 0.75, 0.80],
        "description": "ML probability threshold for signal generation",
    },
    "cooldown_bars": {
        "default": 576,
        "options": [144, 288, 576, 864, 1152, 2016, 4032, 8640],
        "description": "Minimum bars between trades (576=48h, 2016=7d, 8640=30d)",
    },

    # -- Label config --
    "target_r": {
        "default": 2.0,
        "options": [1.0, 1.5, 2.0, 2.5, 3.0],
        "description": "Risk-reward target multiple",
    },
    "max_bars": {
        "default": 36,
        "options": [12, 24, 36, 48, 96, 144, 288],
        "description": "Maximum bars for label horizon",
    },
    "stop_atr_mult": {
        "default": 1.0,
        "options": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
        "description": "ATR multiplier for stop-loss distance",
    },
    "entry_type": {
        "default": "market",
        "options": ["market", "limit_ob_mid", "limit_fvg_edge"],
        "description": "Order entry type (market or limit variants)",
    },
    "direction": {
        "default": "long",
        "options": ["long", "short"],
        "description": "Trade direction",
    },

    # -- ICT structure parameters --
    "ob_age_cap": {
        "default": 200,
        "options": [96, 144, 200, 288, 576],
        "description": "Max age in bars for order block validity (~16.7h at 200)",
    },
    "ob_mitigation": {
        "default": "50pct",
        "options": ["wick_touch", "50pct", "close_through"],
        "description": "OB mitigation trigger: wick touch, 50% penetration, or close through",
    },
    "fvg_min_size_atr": {
        "default": 0.50,
        "options": [0.10, 0.20, 0.35, 0.50, 0.75],
        "description": "Minimum FVG size as ATR fraction (0.1 encodes noise)",
    },
    "fvg_age_cap": {
        "default": 100,
        "options": [24, 48, 100, 144, 288],
        "description": "Max age in bars for FVG validity (~8h at 100)",
    },
    "swing_n_internal": {
        "default": 5,
        "options": [3, 5, 7, 10],
        "description": "Internal swing pivot_n (25min lag at 5; entry timing, CHoCH)",
    },
    "swing_n_external": {
        "default": 10,
        "options": [7, 10, 15, 20],
        "description": "External swing pivot_n (50min lag at 10; dealing range, major structure)",
    },
    "displacement_atr_mult": {
        "default": 1.5,
        "options": [1.0, 1.5, 2.0, 3.0],
        "description": "ATR multiplier for displacement detection threshold",
    },
    "liq_tolerance_atr": {
        "default": 0.10,
        "options": [0.05, 0.10, 0.20, 0.30],
        "description": "ATR tolerance for EQH/EQL clustering",
    },
    "breaker_age_cap": {
        "default": 200,
        "options": [96, 144, 200, 288, 576],
        "description": "Max age for breaker block validity (~16.7h at 200)",
    },
    "pd_lookback": {
        "default": 96,
        "options": [48, 96, 144, 288],
        "description": "Lookback for premium/discount calculation (~8h at 96)",
    },
    "ote_fib_low": {
        "default": 0.618,
        "options": [0.50, 0.618, 0.65],
        "description": "OTE zone lower Fibonacci level",
    },
    "ote_fib_high": {
        "default": 0.786,
        "options": [0.786, 0.79, 0.85],
        "description": "OTE zone upper Fibonacci level",
    },

    # -- Regime --
    "hmm_bull_threshold": {
        "default": 0.60,
        "options": [0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
        "description": "HMM bull probability threshold for signal_filter gate",
    },

    # -- Sizing --
    "kelly_divisor": {
        "default": 40.0,
        "options": [20.0, 30.0, 40.0, 50.0],
        "description": "Kelly fraction divisor (40 = 2.5% risk per trade)",
    },

    # -- Tier --
    "tier": {
        "default": "standard",
        "options": ["standard", "weekly", "monthly"],
        "description": "Trading tier (standard ~178/yr, weekly ~52/yr, monthly ~12/yr)",
    },
}

# ---------------------------------------------------------------------------
# Parameter groups -- for grouped optimization sweeps
# ---------------------------------------------------------------------------
PARAMETER_GROUPS: dict[str, list[str]] = {
    "quick_wins": ["ml_threshold", "cooldown_bars"],
    "label_config": [
        "target_r", "max_bars", "stop_atr_mult", "entry_type", "direction",
    ],
    "ict_structure": [
        "ob_age_cap", "ob_mitigation", "fvg_min_size_atr", "fvg_age_cap",
        "swing_n_internal", "swing_n_external", "displacement_atr_mult",
        "breaker_age_cap", "pd_lookback",
    ],
    "regime": ["hmm_bull_threshold"],
    "sizing": ["kelly_divisor"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_untested_options(
    param_name: str,
    registry: dict,
) -> list[Any]:
    """Return options for a parameter that haven't been tested in any experiment.

    Checks experiment notes and config fields in the registry for mentions of
    each option value.

    Args:
        param_name: Key in SEARCH_SPACE.
        registry: Registry dict with "experiments" list.

    Returns:
        List of untested option values.
    """
    if param_name not in SEARCH_SPACE:
        return []

    spec = SEARCH_SPACE[param_name]
    all_options = spec["options"]
    tested_values: set[str] = set()

    for exp in registry.get("experiments", []):
        # Check direct config fields
        val = exp.get(param_name)
        if val is not None:
            tested_values.add(str(val))

        # Check ml_config sub-dict
        ml_cfg = exp.get("ml_config", {})
        val = ml_cfg.get(param_name)
        if val is not None:
            tested_values.add(str(val))

        # Check label_config sub-dict
        lbl_cfg = exp.get("label_config", {})
        if lbl_cfg:
            val = lbl_cfg.get(param_name)
            if val is not None:
                tested_values.add(str(val))

        # Check sizing sub-dict
        sizing = exp.get("sizing", {})
        val = sizing.get(param_name)
        if val is not None:
            tested_values.add(str(val))

        # Check notes for explicit mentions
        notes = exp.get("notes", "")
        for opt in all_options:
            if f"{param_name}={opt}" in notes:
                tested_values.add(str(opt))

    untested = [opt for opt in all_options if str(opt) not in tested_values]
    return untested


def record_tested(
    param_name: str,
    value: Any,
    registry_path: str | Path | None = None,
) -> None:
    """Record a tested parameter value in the registry metadata.

    Appends a tested_params entry to the registry JSON so future
    get_untested_options calls exclude it.

    Args:
        param_name: Key in SEARCH_SPACE.
        value: The tested value.
        registry_path: Path to registry.json (defaults to standard location).
    """
    if registry_path is None:
        registry_path = (
            Path(__file__).resolve().parent.parent
            / "experiments" / "registry.json"
        )
    registry_path = Path(registry_path)

    if not registry_path.exists():
        return

    with open(registry_path, "r") as f:
        registry = json.load(f)

    # Append to tested_params metadata section
    tested = registry.setdefault("tested_params", {})
    param_list = tested.setdefault(param_name, [])
    if value not in param_list:
        param_list.append(value)

    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2, default=str)


def get_default_config() -> dict[str, Any]:
    """Return a config dict with all default values from SEARCH_SPACE."""
    return {k: v["default"] for k, v in SEARCH_SPACE.items()}


def get_group_params(group_name: str) -> dict[str, dict]:
    """Return SEARCH_SPACE entries for a parameter group."""
    if group_name not in PARAMETER_GROUPS:
        raise ValueError(
            f"Unknown group '{group_name}'. "
            f"Available: {list(PARAMETER_GROUPS.keys())}"
        )
    return {
        name: SEARCH_SPACE[name]
        for name in PARAMETER_GROUPS[group_name]
        if name in SEARCH_SPACE
    }


def print_search_space() -> None:
    """Print the full search space in a readable table."""
    print()
    print("=" * 80)
    print("  SEARCH SPACE (D52)")
    print("=" * 80)
    print()
    for group_name, params in PARAMETER_GROUPS.items():
        print(f"  [{group_name}]")
        for p in params:
            spec = SEARCH_SPACE.get(p, {})
            default = spec.get("default", "?")
            options = spec.get("options", [])
            n_opts = len(options)
            print(f"    {p:30s}  default={default!s:8s}  ({n_opts} options)")
        print()
    # Print ungrouped params
    grouped = set()
    for params in PARAMETER_GROUPS.values():
        grouped.update(params)
    ungrouped = [k for k in SEARCH_SPACE if k not in grouped]
    if ungrouped:
        print("  [ungrouped]")
        for p in ungrouped:
            spec = SEARCH_SPACE[p]
            print(f"    {p:30s}  default={spec['default']!s:8s}  "
                  f"({len(spec['options'])} options)")
        print()
    total_combos = 1
    for spec in SEARCH_SPACE.values():
        total_combos *= len(spec["options"])
    print(f"  Total parameters: {len(SEARCH_SPACE)}")
    print(f"  Total grid combinations: {total_combos:,.0f}")
    print("=" * 80)


# ---------------------------------------------------------------------------
# Main (for quick verification)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print_search_space()
    defaults = get_default_config()
    print(f"\n  Default config: {len(defaults)} params")
    for k, v in defaults.items():
        print(f"    {k}: {v}")
