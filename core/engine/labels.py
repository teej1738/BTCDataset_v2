# core/engine/labels.py
# Label utilities for triple-barrier label lookup and validation.
# D38 -- ported from legacy conventions.

from __future__ import annotations


# label naming convention: label_{direction}_hit_{r}r_{horizon}c
VALID_DIRECTIONS = ("long", "short")
VALID_R_MULTIPLES = (1, 2, 3)
VALID_HORIZONS = (12, 24, 48, 96, 288)


def get_label(
    df,
    direction: str = "long",
    r_multiple: int = 2,
    horizon_bars: int = 48,
) -> str:
    """
    Return the label column name for the given parameters.
    Raises ValueError if the column does not exist in df.
    """
    if direction not in VALID_DIRECTIONS:
        raise ValueError(
            f"direction must be one of {VALID_DIRECTIONS}, got '{direction}'"
        )
    if r_multiple not in VALID_R_MULTIPLES:
        raise ValueError(
            f"r_multiple must be one of {VALID_R_MULTIPLES}, got {r_multiple}"
        )
    if horizon_bars not in VALID_HORIZONS:
        raise ValueError(
            f"horizon_bars must be one of {VALID_HORIZONS}, got {horizon_bars}"
        )

    col = f"label_{direction}_hit_{r_multiple}r_{horizon_bars}c"

    if col not in df.columns:
        raise ValueError(
            f"Label column '{col}' not found in dataframe. "
            f"Available label columns: "
            f"{[c for c in df.columns if c.startswith('label_')]}"
        )

    return col


def validate_label_alignment(embargo_bars: int, label_horizon: int) -> bool:
    """
    Check that embargo_bars matches label_horizon_bars.
    If they differ, print a loud warning and return False.
    The embargo must equal the label horizon to prevent forward leakage:
    a 48-bar label looks 48 bars ahead, so the embargo gap must be >= 48.
    """
    if embargo_bars != label_horizon:
        print(
            f"WARNING: embargo_bars ({embargo_bars}) != "
            f"label_horizon ({label_horizon}). "
            f"This may cause forward leakage in walk-forward training. "
            f"Set embargo_bars = label_horizon to be safe."
        )
        return False
    return True


def parse_label_col(label_col: str) -> dict:
    """
    Parse a label column name into its components.
    Example: 'label_long_hit_2r_48c' -> {direction: 'long', r: 2, horizon: 48}
    """
    parts = label_col.split("_")
    # label_{direction}_hit_{r}r_{horizon}c
    if len(parts) != 5 or parts[0] != "label" or parts[2] != "hit":
        raise ValueError(f"Cannot parse label column: '{label_col}'")

    direction = parts[1]
    r_multiple = int(parts[3].rstrip("r"))
    horizon = int(parts[4].rstrip("c"))

    return {
        "direction": direction,
        "r_multiple": r_multiple,
        "horizon_bars": horizon,
    }
