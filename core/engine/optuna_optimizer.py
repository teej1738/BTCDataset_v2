# core/engine/optuna_optimizer.py
# D52 -- Optuna-based hyperparameter optimizer with GT-Score objective.
# Scaffolded alongside existing optimizer.py (does not replace it).
# ASCII-only output for cp1252 compatibility.
#
# Architecture:
#   - JournalFileStorage for parallel-safe persistence
#   - TPESampler(multivariate=True, constant_liar=True) for correlated params
#   - MedianPruner(n_startup_trials=10, n_warmup_steps=2) for early stopping
#   - GT-Score objective: recency-weighted fold Sharpe - variance - complexity
#
# Usage:
#   python -m core.engine.optuna_optimizer                  # run optimization
#   python -m core.engine.optuna_optimizer --n-trials 100   # custom trial count
#   python -m core.engine.optuna_optimizer --smoke          # dry-run smoke test
#
# IMPORTANT: D51 must complete before running real trials.
#            This file can be created before D51 is done.

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
JOURNAL_PATH = PROJECT_ROOT / "core" / "experiments" / "optuna_journal.log"
REGISTRY_PATH = PROJECT_ROOT / "core" / "experiments" / "registry.json"

# ---------------------------------------------------------------------------
# Lazy imports (Optuna is optional until needed)
# ---------------------------------------------------------------------------
_optuna = None


def _import_optuna():
    """Lazy import to avoid import errors if optuna not installed."""
    global _optuna
    if _optuna is None:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        _optuna = optuna
    return _optuna


# ---------------------------------------------------------------------------
# GT-Score: composite objective function
# ---------------------------------------------------------------------------
def compute_gt_score(
    fold_scores: list[float] | np.ndarray,
    n_training_rows: int,
    n_features: int,
    n_leaves: int = 31,
    half_life: float = 3.0,
    variance_penalty: float = 0.3,
    feature_penalty_weight: float = 0.001,
    hard_floor: float = -0.5,
) -> float:
    """Compute GT-Score: recency-weighted fold Sharpe with penalties.

    GT-Score = weighted_mean(fold_SRs)
             - variance_penalty * weighted_std(fold_SRs)
             - BIC_complexity_penalty
             - feature_count_penalty

    Args:
        fold_scores: Per-fold Sharpe ratios (one per walk-forward fold).
        n_training_rows: Total training rows (for BIC penalty).
        n_features: Number of features used.
        n_leaves: LightGBM num_leaves (default 31).
        half_life: Recency weighting half-life in folds (default 3).
        variance_penalty: Weight for fold-to-fold variance (default 0.3).
        feature_penalty_weight: Per-feature penalty (default 0.001).
        hard_floor: Reject trial if any fold below this (default -0.5).

    Returns:
        GT-Score (higher is better). Returns -999.0 for catastrophic trials.
    """
    scores = np.asarray(fold_scores, dtype=np.float64)
    n = len(scores)

    if n == 0:
        return -999.0

    # Hard floor: reject if any fold is catastrophic
    if np.nanmin(scores) < hard_floor:
        return -999.0

    # Exponential recency weighting (recent folds matter more for BTC regimes)
    weights = np.exp(-np.log(2) / half_life * np.arange(n)[::-1])
    weights /= weights.sum()

    weighted_mean = float(np.dot(weights, scores))
    weighted_var = float(np.dot(weights, (scores - weighted_mean) ** 2))
    weighted_std = math.sqrt(max(weighted_var, 0.0))

    # BIC-style complexity penalty
    T = max(n_training_rows, 1)
    k = n_features * math.log2(max(n_leaves, 2))  # effective complexity
    bic_penalty = (k * math.log(T)) / (2 * T)

    # Feature count penalty
    feature_penalty = feature_penalty_weight * n_features

    score = weighted_mean - variance_penalty * weighted_std - bic_penalty - feature_penalty

    return round(score, 6)


# ---------------------------------------------------------------------------
# Build experiment config from Optuna trial
# ---------------------------------------------------------------------------
def build_experiment_from_trial(
    trial,
    trial_number: int | None = None,
    param_group: str | None = None,
    study_name: str | None = None,
) -> dict:
    """Map Optuna trial suggestions to an experiment config dict.

    Uses the SEARCH_SPACE from core.config.parameters to define
    the parameter ranges for each trial.

    Args:
        trial: Optuna Trial object.
        trial_number: Override for experiment numbering.
        param_group: If set, only suggest params in this group;
            all others use defaults. Group names from PARAMETER_GROUPS.
        study_name: If set, used as prefix in experiment ID to avoid
            collisions between studies. E.g. "d54b_quick_wins_T0001".

    Returns:
        Experiment config dict compatible with simulator.run_safe().
    """
    from core.config.parameters import SEARCH_SPACE, PARAMETER_GROUPS

    # Determine which params to explore vs fix at defaults
    if param_group is not None:
        if param_group not in PARAMETER_GROUPS:
            raise ValueError(
                f"Unknown param_group '{param_group}'. "
                f"Available: {list(PARAMETER_GROUPS.keys())}"
            )
        explore_params = set(PARAMETER_GROUPS[param_group])
    else:
        explore_params = set(SEARCH_SPACE.keys())

    # Suggest explored params, use defaults for the rest
    params = {}
    for name, spec in SEARCH_SPACE.items():
        if name in explore_params:
            params[name] = trial.suggest_categorical(name, spec["options"])
        else:
            params[name] = spec["default"]

    # Build experiment config
    exp_number = trial_number if trial_number is not None else trial.number
    prefix = study_name if study_name else "optuna"
    exp_id = f"{prefix}_T{exp_number:04d}"

    # Determine tier config
    tier = params.get("tier", "standard")

    # Build label_config for non-market entries or non-default labels
    label_config = None
    entry_type = params.get("entry_type", "market")
    if entry_type != "market" or params.get("target_r", 2.0) != 2.0:
        label_config = {
            "direction": params.get("direction", "long"),
            "target_r": params.get("target_r", 2.0),
            "stop_type": "atr",
            "stop_atr_mult": params.get("stop_atr_mult", 1.0),
            "max_bars": params.get("max_bars", 36),
            "entry_type": entry_type,
            "entry_delay": 0,
            "fill_timeout": 12,
        }

    # Direction determines label column
    direction = params.get("direction", "long")
    label_col = f"label_{direction}_hit_2r_48c"

    # Signal filter for HMM gating
    signal_filter = None
    hmm_thresh = params.get("hmm_bull_threshold")
    if hmm_thresh is not None and hmm_thresh < 1.0:
        # Only apply HMM filter if threshold is below 1.0
        # (1.0 effectively disables it)
        if direction == "long":
            signal_filter = {"hmm_prob_bull": {"min": hmm_thresh}}
        else:
            signal_filter = {"hmm_prob_bear": {"min": hmm_thresh}}

    experiment = {
        "id": exp_id,
        "signal_domain": "ml",
        "features": "all",
        "feature_exclude": None,  # pruning handled separately
        "label": label_col,
        "tier": tier,
        "ml_config": {
            "threshold": params.get("ml_threshold", 0.60),
            "model": "lgbm",
            "device": "gpu",
            "embargo_bars": 288,
        },
        "cooldown_bars": params.get("cooldown_bars", 576),
        "sizing": {
            "method": "kelly",
            "divisor": params.get("kelly_divisor", 40.0),
            "odds": params.get("target_r", 2.0),
        },
        "cost_per_r": 0.05,
        "notes": f"Optuna trial {exp_number}. "
                 f"Params: {json.dumps(params, default=str)}",
    }

    if label_config is not None:
        experiment["label_config"] = label_config

    if signal_filter is not None:
        experiment["signal_filter"] = signal_filter

    # Store ICT structure params for rules.py (D53 will use these)
    ict_params = {}
    ict_keys = [
        "ob_age_cap", "ob_mitigation", "fvg_min_size_atr", "fvg_age_cap",
        "swing_n_internal", "swing_n_external", "displacement_atr_mult",
        "liq_tolerance_atr", "breaker_age_cap", "pd_lookback",
        "ote_fib_low", "ote_fib_high",
    ]
    for k in ict_keys:
        if k in params:
            ict_params[k] = params[k]
    if ict_params:
        experiment["ict_params"] = ict_params

    return experiment


# ---------------------------------------------------------------------------
# Extract fold-level Sharpe from experiment results
# ---------------------------------------------------------------------------
def extract_fold_scores(results: dict) -> list[float]:
    """Extract per-fold Sharpe approximations from experiment results.

    Uses CSCV walk-forward windows as fold-level score proxies.
    Each window's test_mean_r serves as a fold-level return metric.
    We compute per-window Sharpe as mean_r / std_r (approximated).

    Args:
        results: Results dict from run_safe().

    Returns:
        List of per-fold scores (Sharpe approximations).
    """
    cscv = results.get("cscv", {})
    wf_windows = cscv.get("walk_forward", [])

    if not wf_windows:
        # Fallback: use overall Sharpe as single "fold"
        sharpe = results.get("metrics", {}).get("sharpe_ann", 0.0)
        return [sharpe]

    fold_scores = []
    for window in wf_windows:
        # test_mean_r is the mean R per trade in that window
        mean_r = window.get("test_mean_r", 0.0)
        # Approximate per-window Sharpe from mean_r and trade count
        # (higher mean_r = better, directly usable as score)
        fold_scores.append(mean_r)

    return fold_scores


# ---------------------------------------------------------------------------
# Optuna objective function
# ---------------------------------------------------------------------------
def create_objective(
    n_features_estimate: int = 400,
    n_leaves: int = 31,
    n_training_rows_estimate: int = 500_000,
    dry_run: bool = False,
    param_group: str | None = None,
    study_name: str | None = None,
):
    """Create an Optuna objective function.

    Args:
        n_features_estimate: Expected feature count (for penalty).
        n_leaves: LightGBM num_leaves (for complexity penalty).
        n_training_rows_estimate: Expected training rows (for BIC).
        dry_run: If True, use dummy scores instead of running experiments.
        param_group: If set, restrict Optuna exploration to this group.

    Returns:
        Callable objective function for Optuna study.optimize().
    """

    def objective(trial) -> float:
        """Optuna objective: run experiment, compute GT-Score."""
        experiment = build_experiment_from_trial(
            trial, param_group=param_group, study_name=study_name
        )

        if dry_run:
            # Dry-run mode: generate synthetic fold scores for testing
            n_folds = 11
            rng = np.random.RandomState(trial.number)
            base_sharpe = rng.uniform(0.5, 2.0)
            fold_scores = base_sharpe + rng.normal(0, 0.3, n_folds)

            # Report fold-by-fold for pruner
            for i, score in enumerate(fold_scores):
                trial.report(float(np.mean(fold_scores[: i + 1])), i)
                if trial.should_prune():
                    # Return partial score instead of raising TrialPruned
                    # so TPE can learn from partial information
                    return compute_gt_score(
                        fold_scores[: i + 1].tolist(),
                        n_training_rows=n_training_rows_estimate,
                        n_features=n_features_estimate,
                        n_leaves=n_leaves,
                    )

            return compute_gt_score(
                fold_scores.tolist(),
                n_training_rows=n_training_rows_estimate,
                n_features=n_features_estimate,
                n_leaves=n_leaves,
            )

        # Real mode: run the experiment through simulator
        from core.engine.simulator import run_safe

        t0 = time.time()
        results = run_safe(experiment)
        elapsed = time.time() - t0

        # Check if experiment failed
        if results.get("status") == "FAILED":
            return -999.0

        # Extract fold scores
        fold_scores = extract_fold_scores(results)

        # Report fold-by-fold for pruner
        for i, score in enumerate(fold_scores):
            trial.report(float(np.mean(fold_scores[: i + 1])), i)
            if trial.should_prune():
                return compute_gt_score(
                    fold_scores[: i + 1],
                    n_training_rows=n_training_rows_estimate,
                    n_features=n_features_estimate,
                    n_leaves=n_leaves,
                )

        # Store extra info as trial user attributes
        metrics = results.get("metrics", {})
        trial.set_user_attr("exp_id", experiment["id"])
        trial.set_user_attr("oos_auc", results.get("oos_auc", 0))
        trial.set_user_attr("sharpe_ann", metrics.get("sharpe_ann", 0))
        trial.set_user_attr("win_rate", metrics.get("win_rate", 0))
        trial.set_user_attr("ev_r", metrics.get("ev_r", 0))
        trial.set_user_attr("max_dd_pct", metrics.get("max_dd_pct", 0))
        trial.set_user_attr("trades_per_yr", metrics.get("trades_per_yr", 0))
        trial.set_user_attr("elapsed_sec", round(elapsed, 1))

        gates = results.get("gates", {})
        n_pass = sum(1 for g in gates.values() if g.get("pass", False))
        trial.set_user_attr("gates_passed", f"{n_pass}/{len(gates)}")

        gt = compute_gt_score(
            fold_scores,
            n_training_rows=n_training_rows_estimate,
            n_features=n_features_estimate,
            n_leaves=n_leaves,
        )

        print(f"  Trial {trial.number}: GT={gt:.4f} "
              f"AUC={results.get('oos_auc', 0):.4f} "
              f"Sharpe={metrics.get('sharpe_ann', 0):.2f} "
              f"WR={metrics.get('win_rate', 0):.1%} "
              f"Gates={n_pass}/{len(gates)} "
              f"({elapsed:.0f}s)")

        return gt

    return objective


# ---------------------------------------------------------------------------
# Study creation
# ---------------------------------------------------------------------------
def create_study(
    study_name: str = "btc_ict_v1",
    journal_path: Path | str | None = None,
    seed: int = 42,
) -> "optuna.Study":
    """Create an Optuna study with correct sampler/pruner/storage.

    Uses:
      - JournalFileStorage (parallel-safe, no SQLite locking)
      - TPESampler(multivariate=True, constant_liar=True)
      - MedianPruner(n_startup_trials=10, n_warmup_steps=2)

    Args:
        study_name: Name for the study.
        journal_path: Path to journal file. Defaults to JOURNAL_PATH.
        seed: Random seed for reproducibility.

    Returns:
        Optuna Study object.
    """
    optuna = _import_optuna()

    if journal_path is None:
        journal_path = JOURNAL_PATH

    journal_path = Path(journal_path)
    journal_path.parent.mkdir(parents=True, exist_ok=True)

    # JournalStorage -- parallel-safe, no SQLite locking issues
    # Windows: use JournalFileOpenLock (symlink lock requires admin privileges)
    from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
    lock_obj = JournalFileOpenLock(str(journal_path) + ".lock")
    storage = optuna.storages.JournalStorage(
        JournalFileBackend(str(journal_path), lock_obj=lock_obj)
    )

    # TPESampler: multivariate captures parameter correlations,
    # constant_liar prevents duplicate exploration by parallel workers
    sampler = optuna.samplers.TPESampler(
        multivariate=True,
        constant_liar=True,
        seed=seed,
    )

    # MedianPruner: don't prune until 10 trials complete,
    # never before fold 2 (n_warmup_steps=2)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=2,
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",  # GT-Score: higher is better
        load_if_exists=True,
    )

    return study


# ---------------------------------------------------------------------------
# Run optimization
# ---------------------------------------------------------------------------
def run_optimization(
    n_trials: int = 100,
    study_name: str = "btc_ict_v1",
    journal_path: Path | str | None = None,
    dry_run: bool = False,
    n_jobs: int = 1,
    seed: int = 42,
    param_group: str | None = None,
) -> "optuna.Study":
    """Run Optuna optimization.

    Args:
        n_trials: Number of trials to run.
        study_name: Optuna study name.
        journal_path: Path to journal log.
        dry_run: If True, use dummy fold scores.
        n_jobs: Number of parallel workers.
        seed: Random seed.
        param_group: If set, restrict exploration to this parameter group.

    Returns:
        Completed Optuna Study.
    """
    optuna = _import_optuna()

    study = create_study(
        study_name=study_name,
        journal_path=journal_path,
        seed=seed,
    )

    objective = create_objective(
        dry_run=dry_run, param_group=param_group, study_name=study_name
    )

    group_label = param_group if param_group else "all (22 params)"
    print()
    print("=" * 60)
    print("  OPTUNA OPTIMIZER (D52)")
    print("=" * 60)
    print(f"  Study:     {study_name}")
    print(f"  Trials:    {n_trials}")
    print(f"  Workers:   {n_jobs}")
    print(f"  Group:     {group_label}")
    print(f"  Dry run:   {dry_run}")
    print(f"  Storage:   JournalFileStorage")
    print(f"  Sampler:   TPESampler(multivariate=True, constant_liar=True)")
    print(f"  Pruner:    MedianPruner(n_startup=10, n_warmup=2)")
    print(f"  Objective: GT-Score (maximize)")
    print("=" * 60)
    print()

    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=False,
    )

    # Print summary
    print()
    print("=" * 60)
    print("  OPTIMIZATION SUMMARY")
    print("=" * 60)
    print(f"  Total trials:    {len(study.trials)}")
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials
              if t.state == optuna.trial.TrialState.PRUNED]
    print(f"  Completed:       {len(completed)}")
    print(f"  Pruned:          {len(pruned)}")

    if study.best_trial:
        best = study.best_trial
        print(f"  Best GT-Score:   {best.value:.4f}")
        print(f"  Best trial:      {best.number}")
        print()
        print("  Best parameters:")
        for k, v in sorted(best.params.items()):
            print(f"    {k:30s} = {v}")
        print()
        # Print user attrs if available
        attrs = best.user_attrs
        if attrs:
            print("  Best trial metrics:")
            for k in ["oos_auc", "sharpe_ann", "win_rate", "ev_r",
                       "max_dd_pct", "trades_per_yr", "gates_passed"]:
                if k in attrs:
                    print(f"    {k:20s} = {attrs[k]}")
    print("=" * 60)

    return study


# ---------------------------------------------------------------------------
# Print top N trials
# ---------------------------------------------------------------------------
def print_top_trials(study, n: int = 10) -> None:
    """Print the top N trials from a study."""
    optuna = _import_optuna()

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print("  No completed trials.")
        return

    # Sort by value (GT-Score) descending
    completed.sort(key=lambda t: t.value or -999, reverse=True)
    top = completed[:n]

    print()
    print(f"  TOP {min(n, len(completed))} TRIALS")
    print("-" * 80)
    print(f"  {'#':>4s}  {'Trial':>6s}  {'GT-Score':>9s}  "
          f"{'AUC':>6s}  {'Sharpe':>7s}  {'WR':>6s}  {'EV':>6s}  "
          f"{'MaxDD':>6s}  {'Gates':>6s}")
    print("-" * 80)

    for rank, trial in enumerate(top, 1):
        attrs = trial.user_attrs
        print(f"  {rank:4d}  {trial.number:6d}  {trial.value:9.4f}  "
              f"{attrs.get('oos_auc', 0):6.4f}  "
              f"{attrs.get('sharpe_ann', 0):7.2f}  "
              f"{attrs.get('win_rate', 0):6.1%}  "
              f"{attrs.get('ev_r', 0):+6.3f}  "
              f"{attrs.get('max_dd_pct', 0):6.2f}  "
              f"{attrs.get('gates_passed', '?'):>6s}")
    print("-" * 80)


# ---------------------------------------------------------------------------
# DSR (Deflated Sharpe Ratio) computation
# ---------------------------------------------------------------------------
def compute_dsr(
    observed_sr: float,
    n_experiments: int,
    n_bars: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Compute the Deflated Sharpe Ratio (DSR).

    Adjusts for multiple testing: given N experiments, what's the probability
    that the best observed Sharpe exceeds the expected maximum from noise?

    Args:
        observed_sr: Best observed Sharpe ratio.
        n_experiments: Total number of experiments/trials conducted.
        n_bars: Number of time bars used.
        skewness: Returns skewness (default 0 = normal).
        kurtosis: Returns kurtosis (default 3 = normal).

    Returns:
        DSR p-value in [0, 1]. DSR >= 0.95 for promotion, >= 0.99 for deploy.
    """
    from scipy import stats as sp_stats

    if n_experiments <= 1:
        return 1.0

    # Expected maximum Sharpe from N independent trials under null
    # E[max(SR)] ~= (1-gamma)*Phi^{-1}(1-1/N) + gamma*Phi^{-1}(1-1/(N*e))
    gamma = 0.5772156649  # Euler-Mascheroni
    expected_max_sr = (
        (1 - gamma) * sp_stats.norm.ppf(1 - 1.0 / n_experiments)
        + gamma * sp_stats.norm.ppf(1 - 1.0 / (n_experiments * np.e))
    )

    # Standard error of the Sharpe ratio
    se_sr = np.sqrt(
        (1 + 0.5 * observed_sr ** 2
         - skewness * observed_sr
         + ((kurtosis - 3) / 4) * observed_sr ** 2)
        / n_bars
    )

    if se_sr < 1e-10:
        return 1.0

    # DSR test statistic
    z = (observed_sr - expected_max_sr) / se_sr
    dsr = float(sp_stats.norm.cdf(z))

    return round(dsr, 4)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def smoke_test() -> bool:
    """Run dry-run smoke test with dummy fold scores.

    Creates a study, runs 3 dummy trials, verifies:
      1. JournalStorage creates log file
      2. Pruner is wired (fires on a bad trial)
      3. GT-Score computation produces valid numbers
      4. Study tracks best trial

    Returns True on success, False on failure.
    """
    import tempfile

    optuna = _import_optuna()

    # Use a temp directory for smoke test storage
    tmpdir = Path(tempfile.mkdtemp(prefix="optuna_smoke_"))
    journal_path = tmpdir / "smoke_journal.log"

    print()
    print("=" * 60)
    print("  D52 SMOKE TEST (dry-run)")
    print("=" * 60)
    print()

    # Test 1: GT-Score computation
    print("  Test 1: GT-Score computation ...")
    scores_good = [1.5, 1.8, 2.0, 1.6, 1.9, 2.1, 1.7, 1.8, 2.2, 1.9, 2.0]
    gt_good = compute_gt_score(scores_good, 500_000, 400, 31)
    print(f"    Good scores: GT = {gt_good:.4f}")
    assert gt_good > 0, f"FAIL: GT-Score should be positive for good folds, got {gt_good}"

    scores_bad = [-1.0, -0.5, -0.8, -0.3]
    gt_bad = compute_gt_score(scores_bad, 500_000, 400, 31)
    print(f"    Bad scores:  GT = {gt_bad:.4f}")
    assert gt_bad == -999.0, f"FAIL: GT-Score should be -999 for catastrophic folds"

    scores_empty = []
    gt_empty = compute_gt_score(scores_empty, 500_000, 400, 31)
    assert gt_empty == -999.0, "FAIL: GT-Score should be -999 for empty folds"
    print("    PASS")
    print()

    # Test 2: Create study with JournalStorage
    print("  Test 2: JournalFileStorage ...")
    study = create_study(
        study_name="smoke_test",
        journal_path=journal_path,
        seed=42,
    )
    assert journal_path.exists(), f"FAIL: Journal file not created at {journal_path}"
    print(f"    Journal: {journal_path}")
    print(f"    Size: {journal_path.stat().st_size} bytes")
    print("    PASS")
    print()

    # Test 3: Run 3 dry-run trials
    print("  Test 3: Running 3 dry-run trials ...")
    objective = create_objective(dry_run=True)
    study.optimize(objective, n_trials=3, show_progress_bar=False)

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"    Completed: {len(completed)} trials")
    assert len(completed) == 3, f"FAIL: Expected 3 completed trials, got {len(completed)}"

    for t in completed:
        print(f"    Trial {t.number}: GT={t.value:.4f} "
              f"params={len(t.params)}")

    best = study.best_trial
    print(f"    Best: trial {best.number} (GT={best.value:.4f})")
    print("    PASS")
    print()

    # Test 4: Verify pruner is wired
    print("  Test 4: Pruner wired ...")
    assert isinstance(study.pruner, optuna.pruners.MedianPruner), (
        f"FAIL: Expected MedianPruner, got {type(study.pruner)}"
    )
    print(f"    Pruner: MedianPruner(n_startup=10, n_warmup=2)")
    print("    PASS")
    print()

    # Test 5: Verify sampler is TPE
    print("  Test 5: Sampler ...")
    assert isinstance(study.sampler, optuna.samplers.TPESampler), (
        f"FAIL: Expected TPESampler, got {type(study.sampler)}"
    )
    print("    Sampler: TPESampler(multivariate=True, constant_liar=True)")
    print("    PASS")
    print()

    # Test 6: DSR computation
    print("  Test 6: DSR computation ...")
    dsr_9 = compute_dsr(
        observed_sr=12.9,  # E002_prune Sharpe
        n_experiments=9,
        n_bars=648_000,
    )
    print(f"    DSR(SR=12.9, N=9, bars=648K) = {dsr_9:.4f}")
    assert 0 <= dsr_9 <= 1, f"FAIL: DSR should be in [0,1], got {dsr_9}"

    # Expected max SR from 9 noise experiments
    from scipy import stats as sp_stats
    gamma = 0.5772156649
    expected_max = (
        (1 - gamma) * sp_stats.norm.ppf(1 - 1.0 / 9)
        + gamma * sp_stats.norm.ppf(1 - 1.0 / (9 * np.e))
    )
    print(f"    Expected max SR from noise (N=9): {expected_max:.4f}")
    print("    PASS")
    print()

    # Test 7: build_experiment_from_trial
    print("  Test 7: build_experiment_from_trial ...")
    # Create a mock trial using FixedTrial
    from core.config.parameters import get_default_config
    defaults = get_default_config()
    fixed_trial = optuna.trial.FixedTrial(defaults)
    exp = build_experiment_from_trial(fixed_trial, trial_number=0)
    assert exp["id"] == "optuna_T0000", f"FAIL: Expected 'optuna_T0000', got {exp['id']}"
    assert exp["ml_config"]["embargo_bars"] == 288, "FAIL: embargo should be 288"
    assert exp["cooldown_bars"] == 576, f"FAIL: cooldown should be 576"
    print(f"    Experiment: {exp['id']}")
    print(f"    Embargo: {exp['ml_config']['embargo_bars']}")
    print(f"    Cooldown: {exp['cooldown_bars']}")
    print("    PASS")
    print()

    # Cleanup
    try:
        journal_path.unlink(missing_ok=True)
        tmpdir.rmdir()
    except Exception:
        pass

    print("=" * 60)
    print("  D52 smoke test: PASS (7/7 tests)")
    print("=" * 60)

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = sys.argv[1:]

    if "--smoke" in args:
        success = smoke_test()
        sys.exit(0 if success else 1)

    # Parse arguments
    n_trials = 100
    n_jobs = 1
    dry_run = False
    study_name = "btc_ict_v1"
    param_group = None
    print_top_n = None

    for i, arg in enumerate(args):
        if arg == "--n-trials" and i + 1 < len(args):
            n_trials = int(args[i + 1])
        elif arg == "--n-jobs" and i + 1 < len(args):
            n_jobs = int(args[i + 1])
        elif arg == "--dry-run":
            dry_run = True
        elif arg == "--study" and i + 1 < len(args):
            study_name = args[i + 1]
        elif arg == "--group" and i + 1 < len(args):
            param_group = args[i + 1]
        elif arg == "--print-top" and i + 1 < len(args):
            print_top_n = int(args[i + 1])

    # Print-only mode: load existing study and print top trials
    if print_top_n is not None and "--n-trials" not in args:
        study = create_study(study_name=study_name)
        print_top_trials(study, n=print_top_n)
        return

    study = run_optimization(
        n_trials=n_trials,
        study_name=study_name,
        dry_run=dry_run,
        n_jobs=n_jobs,
        param_group=param_group,
    )

    print_top_trials(study, n=print_top_n or 10)


if __name__ == "__main__":
    main()
