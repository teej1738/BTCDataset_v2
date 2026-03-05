# core/engine/calibrator.py
# Isotonic regression calibration for ML probability outputs.
# D41a -- fixes ECE = 0.125 (gate threshold 0.05) found in D40.
#
# Usage:
#   from core.engine.calibrator import isotonic_calibrate, compute_ece
#   cal_probs = isotonic_calibrate(train_probs, train_labels, test_probs)
#
# No sklearn dependency -- uses numpy PAVA (Pool Adjacent Violators).
# ASCII-only output for cp1252 compatibility.

from __future__ import annotations

import numpy as np


def _pava(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Pool Adjacent Violators Algorithm (isotonic, non-decreasing).

    Parameters
    ----------
    y : array of shape (n,) -- target values (sorted by predicted prob)
    w : array of shape (n,) -- sample weights (usually all 1s)

    Returns
    -------
    result : array of shape (n,) -- isotonic fit values
    """
    n = len(y)
    result = y.astype(np.float64).copy()
    weight = w.astype(np.float64).copy()

    # Forward pass: merge violating blocks
    i = 0
    while i < n - 1:
        if result[i] > result[i + 1]:
            # Pool blocks
            merged_w = weight[i] + weight[i + 1]
            merged_val = (result[i] * weight[i] + result[i + 1] * weight[i + 1]) / merged_w
            result[i] = merged_val
            weight[i] = merged_w
            # Remove i+1
            result = np.delete(result, i + 1)
            weight = np.delete(weight, i + 1)
            n -= 1
            # Step back to check previous block
            if i > 0:
                i -= 1
        else:
            i += 1

    # Expand blocks back to original length
    expanded = np.empty(len(y), dtype=np.float64)
    idx = 0
    for k in range(len(result)):
        block_len = int(round(weight[k]))
        expanded[idx : idx + block_len] = result[k]
        idx += block_len

    # Handle rounding edge case
    if idx < len(y):
        expanded[idx:] = result[-1]

    return expanded


def isotonic_calibrate(
    train_probs: np.ndarray,
    train_labels: np.ndarray,
    test_probs: np.ndarray,
) -> np.ndarray:
    """Calibrate test probabilities using isotonic regression fit on training data.

    Parameters
    ----------
    train_probs : array of shape (n_train,) -- predicted probabilities on training set
    train_labels : array of shape (n_train,) -- binary labels (0/1)
    test_probs : array of shape (n_test,) -- predicted probabilities to calibrate

    Returns
    -------
    calibrated : array of shape (n_test,) -- calibrated probabilities, clipped to [0, 1]
    """
    # Sort training data by predicted probability
    order = np.argsort(train_probs)
    sorted_probs = train_probs[order]
    sorted_labels = train_labels[order].astype(np.float64)
    weights = np.ones(len(sorted_labels), dtype=np.float64)

    # Fit isotonic regression
    iso_values = _pava(sorted_labels, weights)

    # Map test probabilities via interpolation
    calibrated = np.interp(test_probs, sorted_probs, iso_values)
    return np.clip(calibrated, 0.0, 1.0)


def calibrate_walk_forward(
    oos_probs: np.ndarray,
    labels: np.ndarray,
    fold_boundaries: list[int],
) -> np.ndarray:
    """Walk-forward isotonic calibration using only past folds to calibrate each fold.

    Parameters
    ----------
    oos_probs : array of shape (n,) -- OOS predicted probabilities (NaN where no label)
    labels : array of shape (n,) -- binary labels (0/1), NaN where invalid
    fold_boundaries : list of fold start indices (ascending), last element is end of data

    Returns
    -------
    calibrated : array of shape (n,) -- calibrated probabilities (NaN preserved)
    """
    calibrated = oos_probs.copy()
    n_folds = len(fold_boundaries) - 1

    for i in range(1, n_folds):
        # Training data: all previous folds
        train_start = fold_boundaries[0]
        train_end = fold_boundaries[i]
        test_start = fold_boundaries[i]
        test_end = fold_boundaries[i + 1]

        # Valid training mask
        train_mask = np.zeros(len(oos_probs), dtype=bool)
        train_mask[train_start:train_end] = True
        train_valid = train_mask & ~np.isnan(oos_probs) & ~np.isnan(labels)

        # Valid test mask
        test_mask = np.zeros(len(oos_probs), dtype=bool)
        test_mask[test_start:test_end] = True
        test_valid = test_mask & ~np.isnan(oos_probs)

        if train_valid.sum() < 100 or test_valid.sum() == 0:
            continue

        calibrated[test_valid] = isotonic_calibrate(
            oos_probs[train_valid],
            labels[train_valid],
            oos_probs[test_valid],
        )

    return calibrated


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error.

    Parameters
    ----------
    probs : array of predicted probabilities
    labels : array of binary labels (0/1)
    n_bins : number of equal-width bins

    Returns
    -------
    ece : float in [0, 1]
    """
    valid = ~np.isnan(probs) & ~np.isnan(labels)
    p = probs[valid]
    y = labels[valid].astype(np.float64)
    n = len(p)
    if n == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        avg_conf = p[mask].mean()
        avg_acc = y[mask].mean()
        ece += (n_bin / n) * abs(avg_acc - avg_conf)

    return float(ece)
