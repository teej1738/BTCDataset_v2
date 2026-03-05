# Foundation Build Plan: Step-by-Step

*Generated: 2026-03-04*
*Reference: FOUNDATION_DESIGN.md for architecture, KNOWLEDGE_BASE_DESIGN.md
for file protocols*

---

## MAINTENANCE RULES (for Claude Code)

- Read this file at the start of every session alongside CLAUDE.md
- After completing any step: update that step's status and result only
- Update the CURRENT STATE table at the bottom
- Do not rewrite other sections unless a design decision has changed
- Append completed steps to DECISIONS.md as needed

---

## STATUS LEGEND

- [x] COMPLETE
- [ ] PENDING
- [>] IN PROGRESS
- [!] BLOCKED
- [-] SKIPPED (with reason)

---

## PRE-FLIGHT CHECKS

### [ ] Verify BTCDataset_v2 is FROZEN
1. Confirm P0 diagnostic ran: Daily Sharpe 10.71, BH 9/9, VIABLE BASELINE
2. Confirm no experiments running in BTCDataset_v2
3. Confirm the SEARCH_SPACE wiring bug fix is DEFERRED (fix in new project)
4. BTCDataset_v2 holdout: UNTOUCHED. Will remain untouched.

### [ ] Choose project location
Decision needed: Where does the Foundation project live?
Option A: C:\Users\tjall\Desktop\Trading\Foundation\ (sibling to BTCDataset_v2)
Option B: C:\Users\tjall\Desktop\Trading\BTCDataset_v3\ (successor naming)
Recommended: Option A (clean break, not a "v3" of the same thing)

### [ ] Verify Python environment
- Python 3.12+ installed and on PATH
- pip works: `pip --version`
- git initialized: `git init` in project root

---

## PHASE 0: PROJECT SCAFFOLD

**Goal:** Empty project structure with config, tests, and knowledge files.
No feature code, no ML code, no data. Just the skeleton.

### Step 0.1 -- Create directory structure
**Status:** [ ] PENDING
**CC Prompt:**
```
Create the Foundation project directory structure exactly as specified in
FOUNDATION_DESIGN.md Section 2. Create all directories, __init__.py files,
and placeholder files. Do not write implementation code yet.

Include:
- pyproject.toml with all dependencies from AD-2
- .gitignore (Python standard + .env, data/, holdout/)
- .env.example (template with empty values)
- Empty CLAUDE.md, STATUS.md, KNOWLEDGE.md, DECISIONS.md
- config/experiments/_template.toml (fully documented template)
- tests/conftest.py (empty)
- src/foundation/__init__.py
- src/foundation/cli.py (stub: print "foundation CLI not yet implemented")
```
**Success:** `python -m foundation` prints the stub message. `pytest` runs
(0 tests collected, no errors).

### Step 0.2 -- Write CLAUDE.md for new project
**Status:** [ ] PENDING
**Depends on:** Step 0.1
**CC Prompt:**
```
Write CLAUDE.md for the Foundation project. Follow the format specified in
KNOWLEDGE_BASE_DESIGN.md Section 1.1.

Include:
- Self-maintenance instructions
- Project overview (1 paragraph: what this project is)
- Current state table (all values: "not yet computed")
- File structure (from Step 0.1)
- Instrument configs (BTC placeholder)
- Environment notes (Python version, OS, encoding)
- What NOT to do (from FOUNDATION_DESIGN.md Section 12)

Keep under 300 lines. This is the foundation version, not BTCDataset_v2.
```
**Success:** CLAUDE.md exists and is readable. All sections present.

### Step 0.3 -- Write STATUS.md and KNOWLEDGE.md
**Status:** [ ] PENDING
**Depends on:** Step 0.2
**CC Prompt:**
```
Write STATUS.md and KNOWLEDGE.md for the Foundation project.

STATUS.md: Follow KNOWLEDGE_BASE_DESIGN.md Section 1.2 format.
Current task: "Phase 0 scaffold in progress"
Next steps: "Phase 1: Data foundation"
Session checklist: include all items

KNOWLEDGE.md: Follow KNOWLEDGE_BASE_DESIGN.md Section 1.3 format.
Seed with BTCDataset_v2 proven findings from KNOWLEDGE_BASE_DESIGN.md
Section 8:
- Feature evidence: D55 pruning results (64 features, top 10 SHAP)
- Dead ends: DE01-DE06 from BTCDataset_v2
- Proven findings: FVG edge, session routing, OTE predictive, Sharpe 10.71
- Research questions: migrate open RQs from BTCDataset_v2
- Parameter evidence: key resolved values from D53

Keep KNOWLEDGE.md under 300 lines (initial seed).
```
**Success:** Both files exist. Tables are properly formatted.

### Step 0.4 -- Implement config schema
**Status:** [ ] PENDING
**Depends on:** Step 0.1
**CC Prompt:**
```
Implement src/foundation/config/schema.py with Pydantic models for:
- InstrumentConfig (from FOUNDATION_DESIGN.md AD-4 example)
- ExperimentConfig (from FOUNDATION_DESIGN.md AD-4 example)
- EnvironmentConfig (dev/staging/prod)
- All nested models (LabelConfig, ModelConfig, WalkForwardConfig, etc.)

Implement src/foundation/config/loader.py:
- load_instrument(path) -> InstrumentConfig
- load_experiment(path) -> ExperimentConfig
- load_environment(path) -> EnvironmentConfig

Write tests in tests/test_config/:
- test_schema.py: validate that example TOMLs parse correctly
- test_loader.py: validate that invalid TOMLs raise ValidationError

Create config/instruments/btcusdt_5m.toml (from FOUNDATION_DESIGN.md).
Create config/environments/dev.toml (minimal: log_level = "DEBUG").
```
**Success:** `pytest tests/test_config/ -v` passes. Config loads without error.

### Step 0.5 -- Implement data contracts
**Status:** [ ] PENDING
**Depends on:** Step 0.4
**CC Prompt:**
```
Implement src/foundation/data/contracts.py:
- ColumnContract and DataContract Pydantic models
- BTCUSDT_5M_RAW contract (from FOUNDATION_DESIGN.md Section 4)
- validate_contract(df, contract) function
- Raises ContractViolationError on schema mismatch

Implement src/foundation/data/loaders.py:
- load_train(instrument_config) -> pd.DataFrame
- Includes holdout guard (raise HoldoutViolationError)
- Includes contract validation on load
- Stub: just validate path exists and raise NotImplementedError

Write tests in tests/test_data/:
- test_contracts.py: validate contract checking with synthetic DataFrames
- test_loaders.py: verify holdout guard raises exception
```
**Success:** `pytest tests/test_data/ -v` passes. Holdout guard test passes.

---

## PHASE 1: DATA FOUNDATION

**Goal:** Raw BTC data downloaded, validated, split into train/holdout,
with contracts enforced. Equivalent to BTCDataset_v2's data/ directory.

### Step 1.1a -- Document data source and resampling logic
**Status:** [ ] PENDING
**Depends on:** Phase 0 complete
**CC Prompt:**
```
Read BTCDataset_v2/data_pipeline/ scripts and identify:
1. The exact source of raw BTC candle data (Binance archive URL,
   which endpoint, what timeframe was downloaded)
2. Whether data was downloaded as 1m bars and resampled to 5m,
   or downloaded directly as 5m bars
3. If resampled: the exact resampling logic (open=first, high=max,
   low=min, close=last, volume=sum -- any edge cases?)
4. The date range covered and known gaps
5. Any exchange-specific artifacts noted in BTCDataset_v2 comments
Write findings to knowledge/domain/btc_data_source.md with sections:
Source, Raw resolution, Resampling, Date range, Known gaps,
Known artifacts, Validated against (648,288 bars)
```
**Success:** btc_data_source.md written. All fields populated.
No "unknown" entries without explanation.

### Step 1.1b -- Port candle data pipeline
**Status:** [ ] PENDING
**Depends on:** Step 1.1a
**CC Prompt:**
```
Read knowledge/domain/btc_data_source.md before writing any code.
Implement src/foundation/data/downloaders/binance_perp.py:
- Download BTCUSDT perpetual candles from documented source
- If raw resolution is 1m: resample to 5m using documented logic
- Save as monthly parquet in data/raw/btcusdt/candles/
- SHA256 verification per file
- Resume support (skip existing files)
- Validate output row count matches 648,288 total bars
CLI: python -m foundation download --instrument btcusdt_5m --type candles
```
**Success:** Monthly parquet files exist. Total rows 648,288.
SHA256 manifest written.

### Step 1.2 -- Port supplementary data pipelines
**Status:** [ ] PENDING
**Depends on:** Step 1.1b
**CC Prompt:**
```
Port the three supplementary data downloaders from BTCDataset_v2:

1. src/foundation/data/downloaders/binance_oi.py
   (port from data_pipeline/download_oi.py)
   Output: data/raw/btcusdt/oi_metrics/

2. src/foundation/data/downloaders/binance_aggtrades.py
   (port from data_pipeline/download_aggtrades.py)
   Output: data/raw/btcusdt/aggtrades/

3. src/foundation/data/downloaders/coinalyze_liq.py
   (port from data_pipeline/download_liquidations.py)
   Output: data/raw/btcusdt/liquidations/

CLI: python -m foundation download --instrument btcusdt_5m --type all
```
**Success:** All three data sources downloaded. Row counts match BTCDataset_v2.

### Step 1.3 -- Build data processing pipeline
**Status:** [ ] PENDING
**Depends on:** Steps 1.1b, 1.2
**CC Prompt:**
```
Implement the data processing pipeline:
1. Merge raw candles + OI + CVD + liquidations (port merge_v3.py logic)
2. Apply data contract validation
3. Split into train and holdout (using instrument config dates)
4. Compute SHA256 manifest
5. Save processed parquet + manifest JSON

Output:
  data/processed/btcusdt_5m_train.parquet
  data/processed/btcusdt_5m_manifest.json
  data/holdout/btcusdt_5m_holdout.parquet

CLI: python -m foundation build --instrument btcusdt_5m

Validate: row counts, column counts, holdout date range, SHA256 hashes.
Cross-check against BTCDataset_v2 manifest_v3.json.
```
**Success:** Train and holdout parquets created. Row counts match.
Contract validation passes. Holdout guard tested.

---

## PHASE 2: FEATURE FRAMEWORK

**Goal:** Feature protocol implemented, surviving D55 features ported,
causality test suite passing for all features.

### Step 2.0 -- Implement HTF enrichment layer
**Status:** [ ] PENDING
**Depends on:** Phase 1 complete
**CC Prompt:**
```
Before any feature functions run, the raw 5m DataFrame must be enriched
with higher-timeframe columns. BTCDataset_v2 uses m15_, m30_, h1_, h4_,
d1_ prefixed columns extensively.
Implement src/foundation/data/enrichment.py:
  enrich_htf(df, timeframes) -> pd.DataFrame
  - For each timeframe in [m15, m30, h1, h4, d1]:
    Resample (open=first, high=max, low=min, close=last, volume=sum)
    Forward-fill back to 5m index using ffill()
    Add columns: {tf}_open, {tf}_high, {tf}_low, {tf}_close, {tf}_volume
  - CRITICAL: Use only COMPLETED bars. Apply shift(1) on resampled
    data before forward-filling.
  - Validate output with data contract
Write tests in tests/test_data/test_enrichment.py:
  - test_htf_no_lookahead: at bar T, h1_ columns reflect only the
    h1 bar that CLOSED before T, not the in-progress h1 bar
  - test_htf_columns_present: all expected columns exist
  - test_htf_no_nan: no NaN after warmup period
Feature Protocol compute(df, params) receives df already containing
HTF columns. Features do not resample themselves.
```
**Success:** pytest tests/test_data/test_enrichment.py passes.
No-lookahead test passes specifically. HTF columns visible in output.

### Step 2.1 -- Implement feature protocol and registry
**Status:** [ ] PENDING
**Depends on:** Phase 1 complete
**CC Prompt:**
```
Implement the feature framework from FOUNDATION_DESIGN.md Section 5:
- src/foundation/features/protocol.py (Feature Protocol)
- src/foundation/features/registry.py (@register decorator, compute_features)
- tests/test_features/test_causality_all.py (parametrized causality tests)
- tests/fixtures/ (create sample_btc_1000.parquet from first 1000 train rows)

The registry must:
1. Run quick causality test on registration (T=500)
2. Track dependencies between features
3. Compute features in topological order
4. Pass params from config to each feature.compute()
5. Raise ConfigError if params are missing (the wiring bug fix)

Write a trivial example feature (e.g., "simple_return") to validate the
framework works end-to-end before porting real features.
```
**Success:** `pytest tests/test_features/ -v` passes. Example feature
registered, computed, and causality-tested.

### Step 2.2 -- Port ICT features (survivors only)
**Status:** [ ] PENDING
**Depends on:** Step 2.1
**CC Prompt:**
```
Port the 3 surviving ICT feature families from BTCDataset_v2's
core/signals/ict/rules.py:

1. src/foundation/features/ict/ote.py
   - compute_ote_dist (with corrected fib 0.618/0.786)
   - ote_dist_from_705_atr
   - Params: fib_low, fib_high, swing_lookback

2. src/foundation/features/ict/premium_discount.py
   - compute_premium_discount (continuous 0-1 encoding)
   - pd_position_5m, pd_dist_from_eq, in_discount, in_deep_discount, etc.
   - Params: swing_lookback

3. src/foundation/features/ict/structure.py
   - compute_swing_dual_layer (internal N=5, external N=10)
   - int_swing_high/low, ext_swing_high/low, BOS, CHoCH
   - Params: pivot_n_internal, pivot_n_external

Each feature must:
- Implement the Feature protocol
- Be decorated with @register
- Pass causality tests at T in [500, 1000, 5000, 10000]
- Accept all params from experiment config (no hardcoded defaults)
```
**Success:** All 3 features registered. Causality tests pass at all T values.

### Step 2.3a -- Port proven D55 survivors: OTE-705 + Premium/Discount
**Status:** [ ] PENDING
**Depends on:** Step 2.2
**CC Prompt:**
```
Port two D53 families that survived D55 pruning:
1. src/foundation/features/ict/ote.py
   Features: ote_dist, ote_dist_from_705_atr
   ote_dist_from_705_atr was rank #14 long, #8 short after D55.
2. src/foundation/features/ict/premium_discount.py
   Feature: pd_position_5m (continuous 0-1 P/D encoding)
For each: register as Feature per protocol.py, all params from
params dict (no hardcoded defaults), run causality test,
document params in feature TOML section.
Write tests: tests/test_features/test_ict_ote.py and
tests/test_features/test_ict_premium_discount.py
```
**Success:** Both families registered. Causality tests pass.

### Step 2.3b -- Port dual-swing structure features
**Status:** [ ] PENDING
**Depends on:** Step 2.3a
**CC Prompt:**
```
Port dual-layer swing structure from BTCDataset_v2.
Implement src/foundation/features/ict/structure.py:
  compute_swing_dual_layer(df, params)
  Parameters: pivot_n_internal, pivot_n_external (from params dict)
  Outputs: swing_high_price, swing_low_price,
    int_dist_to_sh_atr, ext_dist_to_sh_atr,
    m15_ict_swing_high, m30_ict_swing_high,
    m30_ict_swing_low_price, h1_ict_swing_low_price
    and all other swing columns from BTCDataset_v2
CRITICAL: swing_high_price = SHAP rank #1 long.
swing_low_price = SHAP rank #1 short. Highest importance features.
Run causality test. Verify swing lag correlation < 0.005.
```
**Success:** Swing features registered. Causality pass.
Swing lag correlation < 0.005.

### Step 2.3c -- Port ICT families confirmed dead in D55
**Status:** [ ] PENDING
**Depends on:** Step 2.3b
**CC Prompt:**
```
Port ICT families with zero Tier 1 features in D55.
Register so autonomous engine can confirm they are still dead.
1. src/foundation/features/ict/displacement.py (d55_verdict=dead)
2. src/foundation/features/ict/sweeps.py including sweep-sequence
   (d55_verdict=dead)
3. src/foundation/features/ict/cisd.py (d55_verdict=dead)
4. src/foundation/features/ict/fvg.py
   Port both basic FVG and FVG-enhanced variants.
   Basic FVG: d55_verdict=untested (pre-D53 version, unknown)
   FVG-enhanced: d55_verdict=dead (D53 version, pruned)
For each: register, implement, causality test.
Mark all features with d55_verdict metadata field.
```
**Success:** All families registered. Causality tests pass.
d55_verdict metadata present on all features.

### Step 2.3d -- Port order blocks and MSS
**Status:** [ ] PENDING
**Depends on:** Step 2.3c
**CC Prompt:**
```
Port remaining ICT families:
1. src/foundation/features/ict/order_blocks.py
   Core: ict_ob_bull_age (SHAP rank #2), ict_ob_bear_age (rank #5)
   These survived D55. Parameter: ob_refresh_bars from params dict.
   Also port anchored OB variants (d55_verdict=dead).
2. src/foundation/features/ict/mss.py
   MSS features (d55_verdict=dead, 0 Tier 1 in D55)
Run causality tests on all new features.
Update KNOWLEDGE.md FEATURE EVIDENCE HIERARCHY table with
d55_verdict for each new feature family.
```
**Success:** All features registered. Causality pass.
ict_ob_bull_age and ict_ob_bear_age present and causal.
KNOWLEDGE.md FEATURE EVIDENCE updated.

---

## PHASE 3: TRAINING ENGINE

**Goal:** Walk-forward training, simulation, and evaluation pipeline
working end-to-end. Equivalent to BTCDataset_v2's core/engine/.

### Step 3.1 -- Port walk-forward training
**Status:** [ ] PENDING
**Depends on:** Phase 2 complete
**CC Prompt:**
```
Port the walk-forward training engine from core/engine/evaluator.py:
- src/foundation/engine/walk_forward.py
  - walk_forward_train(df, features, label, config)
  - Expanding and rolling window support
  - Embargo and label purging (AFML Ch.7)
  - LightGBM with GPU probe + CPU fallback
  - Early stopping with embargoed validation set
  - Returns: OOS probabilities, fold metrics

Params come from ExperimentConfig (Pydantic validated).
No hardcoded defaults for min_train, test_fold, embargo.

CLI: python -m foundation train --config <path>
Output: experiments/models/<name>_oos_probs.npy
        experiments/results/<name>_train.json
```
**Success:** Training runs end-to-end on dev config (subset data).
OOS probs produced.

### Step 3.2 -- Port simulation and evaluation
**Status:** [ ] PENDING
**Depends on:** Step 3.1
**CC Prompt:**
```
Port simulation and evaluation:
- src/foundation/engine/simulation.py
  - simulate(probs, labels, threshold, cooldown)
  - build_trade_returns(trade_indices, labels, r_target, cost_model)  # cost_model is dynamic per AD-19, not flat cost_per_r
- src/foundation/engine/metrics.py
  - compute_metrics(name, r_returns, years)
  - Including: daily Sharpe, per-trade Sharpe, BH comparison
- src/foundation/engine/sizing.py
  - kelly_fraction, equity_sim, equity_sim_variable
- src/foundation/engine/calibration.py
  - platt_calibrate (sigmoid, 2 params; default). isotonic_calibrate for fold N > 500

CLI: python -m foundation evaluate --config <path>
Output: experiments/results/<name>_eval.json
```
**Success:** Evaluation JSON produced with all metrics and gate results.

### Step 3.3 -- Port validation suite
**Status:** [ ] PENDING
**Depends on:** Step 3.2
**CC Prompt:**
```
Port the validation suite:
- src/foundation/validation/gates.py
  - compute_gates(results, gate_config) -> dict of pass/fail
  - 10 default gates from FOUNDATION_DESIGN.md Section 7
- src/foundation/validation/cscv.py
  - run_cscv(r_returns) -> PBO, PSR, bootstrap CI, walk-forward
- src/foundation/validation/dsr.py
  - deflated_sharpe_ratio(observed_sr, n_trials, ...)
- src/foundation/validation/holdout.py
  - holdout_ceremony(config, confirm=False)
  - Logs access to data/holdout/holdout_access.log
  - Raises if confirm is not True

CLI: python -m foundation validate --config <path>
```
**Success:** Validation suite produces correct gate results on known inputs.
Holdout ceremony raises without --confirm flag.

### Step 3.4 -- SHAP Rank Stability Gate
**Status:** [ ] PENDING
**Depends on:** Step 3.3
**CC Prompt:**
```
Compute Spearman rank correlation of SHAP feature importance
across all walk-forward folds. Report mean rho and min rho.
Gate: mean rho >= 0.60
Failure: SHAP rankings are unstable across time periods,
indicating regime-specific overfitting.
Source: AD-17 (Perplexity V2).
```
**Success:** Gate computes and reports correctly on known inputs.

### Step 3.5 -- Horizon Expiry Fraction
**Status:** [ ] PENDING
**Depends on:** Step 3.2
**CC Prompt:**
```
Compute what fraction of labels hit the time barrier (48 bars)
vs stop (1R) vs target (2R).
Gate: time-barrier fraction <= 40%
Failure: barriers are miscalibrated for the volatility regime.
Optimal: < 20% time-expired labels.
Note: binary label (expired = miss) discards signal if
expiry fraction is high. Consider continuous labeling if
expiry > 40%.
```
**Success:** Fraction computed. Reported in experiment JSON.

### Step 3.6 -- Monte Carlo Permutation Test
**Status:** [ ] PENDING
**Depends on:** Step 3.1
**CC Prompt:**
```
Shuffle labels 1000+ times, rerun full pipeline each time.
Check if real performance exceeds 95th percentile of
shuffled-label distribution.
Gate: real AUC > 95th percentile of permuted AUC
Failure: model is fitting noise, not signal.
Note: computationally expensive. Run on reduced fold count
(3 folds) for permutation test to keep runtime manageable.
```
**Success:** Permutation test produces distribution. Real AUC compared.

---

## PHASE 4: EQUIVALENCE TEST

**Goal:** Reproduce D54a baseline results through the new foundation.
This is the gate between "new architecture" and "trusted replacement."

### Step 4.1 -- Run D54a equivalent experiment
**Status:** [ ] PENDING
**Depends on:** Phase 3 complete
**CC Prompt:**
```
Create config/experiments/btc_long_d54a_equiv.toml:
- Before writing the config, read:
  BTCDataset_v2/core/experiments/registry.json
  Find entry with exp_id D54a_baseline_long and extract:
  label_config, ml_threshold, cooldown_bars, feature_set,
  walk_forward settings, sizing config.
  Use these exact values in btc_long_d54a_equiv.toml.
  If registry.json does not have all fields, also read
  BTCDataset_v2/core/experiments/run_d54a.py for the config.
- label_long_hit_2r_48c, threshold 0.60, cooldown 576
- All 670 features (before D55 pruning) -- or as many as registered
- Walk-forward: min_train=105000, test_fold=52500, embargo=288

Run:
  python -m foundation train --config config/experiments/btc_long_d54a_equiv.toml
  python -m foundation evaluate --config config/experiments/btc_long_d54a_equiv.toml

Compare against BTCDataset_v2 D54a results:
  AUC: 0.7933 (tolerance: +/- 0.005)
  WR: 76.0% (tolerance: +/- 2pp)
  Daily Sharpe: 10.71 (tolerance: +/- 1.0)
  Gates: 10/10 PASS

If within tolerance: EQUIVALENCE PASS. Foundation is viable.
If not: debug. Do not proceed until equivalent.
```
**Success:** "EQUIVALENCE PASS" printed. All metrics within tolerance.
Minimum 3 random seed runs completed. Mean AUC within 0.005,
mean daily Sharpe within 1.5 (widened from 1.0 to account for
confirmed seed variance). Post-cost Sharpe computed and recorded.

**Daily Sharpe target: >= 5.0**
Rationale: D55b holdout naive daily Sharpe 10.71 on BTCDataset_v2.
Foundation target is 50% of this value, representing conservative
live expectation after costs. Primary equivalence gate remains
AUC (>= 0.780) and train-split daily Sharpe >= 5.0. Per-bar
Sharpe is reported for comparison only.

### Step 4.2 -- Run D55 equivalent (pruned features)
**Status:** [ ] PENDING
**Depends on:** Step 4.1
**CC Prompt:**
```
Create config/experiments/btc_long_d55_equiv.toml:
- Same as D54a but with only 64 D55-surviving features
- Verify AUC 0.7938 +/- 0.005

This confirms that the feature pruning result transfers correctly.
```
**Success:** Pruned model metrics within tolerance of D55b results.

---

## PHASE 5: AUTONOMOUS TRAINING

**Goal:** Optuna integration with correct parameter passthrough.
The wiring bug from BTCDataset_v2 is impossible in this architecture.

### Step 5.1 -- Implement experiment runner and registry
**Status:** [ ] PENDING
**Depends on:** Phase 4 complete
**CC Prompt:**
```
Implement the experiment lifecycle:
- src/foundation/experiment/registry.py
  - load_registry() -> dict
  - save_experiment(result) -- atomic write

  IMPORTANT: The experiment registry must be TOML format,
  not JSON. CC sessions read and write this file directly.
  JSON is brittle for agent writes (missing commas, trailing
  commas, encoding issues cause silent corruption).

  registry.toml format:
    [[experiments]]
    id = "btc_long_d54a_equiv"
    date = "2026-03-04"
    branch = "experiment/D54a-equiv"
    git_hash = "abc123"
    data_hash = "sha256:..."
    auc = 0.7933
    wr = 0.760
    ev_r = 1.23
    daily_sharpe = 10.71
    gates = "10/10"
    verdict = "PASS"
    notes = "Equivalence test passed"

  Each experiment appends a new [[experiments]] block.
  Python reads with tomllib (stdlib 3.11+).
  Python writes by appending raw TOML string, not by
  serializing a dict (avoids formatting issues).

  Individual experiment result files in
  experiments/results/*.json remain JSON -- these are
  written by Python code only, never by CC directly,
  so JSON is acceptable there.

- src/foundation/experiment/runner.py
  - run_experiment(config_path) -- full pipeline
  - Returns: ExperimentResult with metrics, gates, fold details
- src/foundation/experiment/shap_analysis.py
  - run_shap(config_path) -- LightGBM pred_contrib across folds
  - Returns: feature importance rankings

CLI: python -m foundation run --config <path>
(combines train + evaluate + record)
```
**Success:** Full experiment lifecycle works. Registry updated correctly.

### Step 5.2 -- Implement Optuna integration
**Status:** [ ] PENDING
**Depends on:** Step 5.1
**CC Prompt:**
```
Implement Optuna-based optimization:
- src/foundation/experiment/optimizer.py
  - Reads KNOWLEDGE.md for open research questions
  - Proposes experiment configs with search space from TOML
  - TPESampler(multivariate=True, constant_liar=True)
  - MedianPruner(n_startup_trials=10, n_warmup_steps=2)
  - JournalFileStorage (not SQLite)
  - GT-Score objective (from BTCDataset_v2 D52)
  - DSR correction after N trials
  - Parameter passthrough: Optuna value -> TOML -> Pydantic -> feature.compute()

CRITICAL: Verify the wiring fix works:
  1. Optuna suggests fvg_age_cap = 48
  2. Value appears in generated TOML
  3. Pydantic validates it
  4. Feature receives it in compute(df, params)
  5. Feature USES it (not a hardcoded default)

VARIANCE CHECK: After confirming wiring is correct, run the
same config 3 times with different seeds:
  lgbm_seed = [42, 123, 999]
Record mean and std of Sharpe across runs. If std/mean > 0.3
(30% coefficient of variation), flag in KNOWLEDGE.md as
HIGH_VARIANCE and investigate before trusting point estimates.

CLI: python -m foundation optimize --max-trials 10 --mode checkpoint
```
**Success:** Optuna trial runs with custom params. Feature receives correct
param values. Params visible in experiment TOML and result JSON.

### Step 5.3 -- Treynor-Mazuy Timing Regression
**Status:** [ ] PENDING
**Depends on:** Step 5.1
**CC Prompt:**
```
Run regression: R_system = alpha + beta*R_BTC + gamma*R_BTC^2 + epsilon
on the train-split daily equity curve.
Gate: positive gamma (statistically significant market timing)
OR MaxDD compression vs buy-and-hold > 50%
Context: D55b holdout occurred during BTC -20.6% with
system producing +1.26R EV (pre-cost; cost-adjusted EV ~+0.28R at current 5-min ATR stops, see AD-19). Qualitative evidence of timing
alpha is strong. This test quantifies it.
```
**Success:** Regression coefficients computed. gamma reported with p-value.

### Step 5.4 -- Time-in-Market Alpha Adjustment
**Status:** [ ] PENDING
**Depends on:** Step 5.1
**CC Prompt:**
```
Compute system time-in-market fraction (trades x avg hold).
Compare system daily returns vs BTC buy-and-hold scaled
to same average exposure.
Report: alpha above exposure-adjusted benchmark.
Gate: positive alpha above adjusted benchmark.
```
**Success:** Time-in-market fraction and adjusted alpha computed.

### Step 5.5 -- MinTRL Computation
**Status:** [ ] PENDING
**Depends on:** Step 5.1
**CC Prompt:**
```
Compute Minimum Track Record Length (Bailey & Lopez de Prado):
how many months of live trading are required to statistically
confirm the backtested daily Sharpe at 95% confidence.
Report this number before paper trading begins.
It sets the minimum paper trading duration.
```
**Success:** MinTRL months computed. Reported in STATUS.md.

---

## PHASE 6: NQ/ES EXTENSION

**Goal:** Add NQ and ES futures data. Run baseline experiments.
This is out-of-distribution validation, not portfolio construction.

### Step 6.1 -- CME data pipeline
**Status:** [ ] PENDING
**Depends on:** Phase 5 complete
**Note:** Requires identifying a data source for NQ/ES 5m bars.
Options: Databento, Polygon.io, FirstRate Data, or IQFeed.
Research needed before implementation.

### Step 6.2 -- NQ instrument config and baseline
**Status:** [ ] PENDING
**Depends on:** Step 6.1

### Step 6.3 -- ES instrument config and baseline
**Status:** [ ] PENDING
**Depends on:** Step 6.1

---

## PHASE 7: PAPER TRADING

**Goal:** Live signal generation without execution. 3-6 month minimum
before any capital deployment.

### Step 7.1 -- Live data feed
**Status:** [ ] PENDING
**Depends on:** Phase 4 complete (does not need Phases 5-6)

### Step 7.2 -- Signal generation on new bars
**Status:** [ ] PENDING
**Depends on:** Step 7.1

### Step 7.3 -- Paper trade logging and monitoring
**Status:** [ ] PENDING
**Depends on:** Step 7.2

---

## PHASE 8: LIVE DEPLOYMENT

**Goal:** Automated trading with position management and risk controls.

### Step 8.1 -- Order execution
**Status:** [ ] PENDING
**Depends on:** Phase 7 complete (3-6 months paper trading)

### Step 8.2 -- Position monitoring and kill switch
**Status:** [ ] PENDING
**Depends on:** Step 8.1

### Step 8.3 -- Drawdown circuit breaker
**Status:** [ ] PENDING
**Depends on:** Step 8.2

---

## CURRENT STATE

| Property | Value |
|----------|-------|
| Phase | 0 (not started) |
| Steps complete | 0 |
| BTCDataset_v2 status | FROZEN (viable baseline) |
| Holdout status | UNTOUCHED |
| Best experiment | (none yet) |
| Last updated | 2026-03-04 (W2 findings + Perplexity Foundation integrated. Awaiting Perplexity V2 trading research results.) |

---

## ESTIMATED EFFORT PER PHASE

These are CC session estimates, not calendar time estimates.

| Phase | Steps | CC sessions (est.) | Dependency |
|-------|-------|--------------------|------------|
| 0: Scaffold | 5 | 2-3 | None |
| 1: Data | 3 | 3-4 | Phase 0 |
| 2: Features | 3 | 4-6 | Phase 1 |
| 3: Engine | 3 | 3-4 | Phase 2 |
| 4: Equivalence | 2 | 1-2 | Phase 3 |
| 5: Autonomous | 2 | 2-3 | Phase 4 |
| 6: NQ/ES | 3 | 3-5 | Phase 5 |
| 7: Paper trade | 3 | 2-3 + 3-6 months | Phase 4 |
| 8: Live | 3 | 2-3 | Phase 7 |

**Critical path:** Phases 0-4 (equivalence test) is the minimum viable
foundation. Everything after Phase 4 builds on a proven base.

---

*End of MASTER_PLAN_TEMPLATE.md*
