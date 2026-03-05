# Trading Research Foundation: Technical Design Specification

*Generated: 2026-03-04*
*Input corpus: CLAUDE.md (793 lines), STRATEGY_LOG.md D01-D55 (3,923 lines),
THE_PLAN.md (~1,330 lines), D53_IMPLEMENTATION_SPEC.md (851 lines),
REDESIGN_V2.md, P0 diagnostic results*

---

## 0. Design Philosophy

### Why a new project

BTCDataset_v2 achieved its goal: AUC 0.7933, WR 76.0%, 10/10 gates PASS,
daily Sharpe 10.71 (D55b holdout, BEAR market). It is a VIABLE BASELINE.
It is also a research prototype with 55 ad-hoc decisions, a fatal wiring bug
(SEARCH_SPACE never reaches rules.py), and no structural enforcement of any
validation discipline.

The new foundation is not a refactor. It is a clean-room design informed by
everything D01-D55 proved and disproved. The prototype stays frozen. The
foundation replaces it when (and only when) it reproduces equivalent results
through a more rigorous architecture.

### Central constraint

Claude Code (CC) is the primary developer. CC has zero session memory. Every
session starts from scratch. If context is not in a file, it does not exist.
This means:

1. All state must live in files with known paths and standardized formats.
2. All decisions must be documented with reasoning (not just outcomes).
3. All experiments must be reproducible from config alone (no "run it like
   last time" instructions).
4. Vibe coding must be structurally impossible -- not just discouraged but
   physically prevented by the architecture.

### What "structurally impossible" means

A vibe-coding-proof architecture has these properties:

- You cannot run an experiment without a validated config file.
- You cannot add a feature without registering it and passing causality tests.
- You cannot train a model without embargo and purging enforced by the engine.
- You cannot access holdout data without an explicit ceremony that logs permanently.
- You cannot promote a model without passing all statistical gates.
- You cannot deploy without a paper-trading record.

These are not guidelines. They are runtime checks that raise exceptions.

### Priority order (non-invertible)

1. Correct architecture and design
2. Software engineering rigor -- eliminate vibe coding structurally
3. Robust, validated data foundation
4. Strategy framework that plugs into the foundation
5. Autonomous training engine on top
6. Live data and day trading (last)

---

## 1. Architecture Overview

### Layer model

```
Layer 5: Deployment        paper trade -> live -> monitoring
Layer 4: Experiment Loop   propose -> config -> train -> evaluate -> record
Layer 3: Training Engine   walk-forward, calibration, CSCV, gates
Layer 2: Feature + Label   registered features, triple-barrier labels, causality
Layer 1: Data Foundation   raw -> validated -> processed (immutable pipeline)
Layer 0: Infrastructure    config, logging, testing, knowledge base
```

Each layer depends only on layers below it. No upward dependencies.

### Data flow

```
Exchange APIs                Instrument Config (TOML)
     |                              |
     v                              v
  Raw Parquet  ----------->  Data Loader + Contract Validation
     |                              |
     v                              v
  Feature Registry  ------>  Feature Computation (params from config)
     |                              |
     v                              v
  Label Generator  -------->  Labeled Dataset (train / holdout split)
     |                              |
     v                              v
  Walk-Forward Engine  ---->  OOS Probabilities + Fold Metrics
     |                              |
     v                              v
  Validation Suite  ------->  Gate Results (PASS/FAIL)
     |                              |
     v                              v
  Experiment Registry  ---->  Knowledge Base Update
```

---

## 2. Project Structure

```
Foundation/
  pyproject.toml              -- single source of truth for dependencies
  .env                        -- secrets only (API keys), gitignored
  .gitignore
  CLAUDE.md                   -- project context (CC reads first, always)
  STATUS.md                   -- current state, next steps, blockers
  KNOWLEDGE.md                -- evidence base, experiment history, dead ends
  DECISIONS.md                -- architectural decisions (append-only)

  config/
    instruments/
      btcusdt_5m.toml         -- BTC perpetual futures, 5-minute bars
      nq_5m.toml              -- NQ futures (placeholder, Phase 6)
      es_5m.toml              -- ES futures (placeholder, Phase 6)
    environments/
      dev.toml                -- development settings (subset data, fast)
      staging.toml            -- full data, train split only
      prod.toml               -- includes holdout access ceremony
    experiments/
      btc_long_baseline.toml  -- example experiment config
      _template.toml          -- blank template with all fields documented

  src/
    foundation/
      __init__.py
      cli.py                  -- entry point: python -m foundation <command>
      config/
        __init__.py
        schema.py             -- Pydantic models for all config types
        loader.py             -- TOML loader with validation
      data/
        __init__.py
        contracts.py          -- column schema, NaN policy, type enforcement
        loaders.py            -- Parquet loader with holdout guard
        downloaders/
          __init__.py
          binance_perp.py     -- BTCUSDT perpetual data
          binance_oi.py       -- Open interest metrics
          binance_aggtrades.py -- True tick CVD
          coinalyze_liq.py    -- Liquidation data
          cme.py              -- NQ/ES futures (placeholder)
      features/
        __init__.py
        registry.py           -- feature registration, discovery, metadata
        protocol.py           -- Feature protocol (interface contract)
        ict/
          __init__.py
          displacement.py
          fvg.py
          sweeps.py
          order_blocks.py
          structure.py        -- swings, BOS, CHoCH, MSS
          ote.py
          premium_discount.py
          cisd.py
        ta/
          __init__.py
          momentum.py         -- RSI, MACD, Stochastic
          trend.py            -- EMA, VWAP, Supertrend
          volatility.py       -- GK, Parkinson, ATR, Bollinger
        regime/
          __init__.py
          hmm.py              -- 3-state Gaussian HMM
          adx_composite.py
        microstructure/
          __init__.py
          oi_features.py      -- Open interest derived features
          cvd_features.py     -- True tick CVD
          liquidations.py     -- Liquidation cascades
      labels/
        __init__.py
        triple_barrier.py     -- dynamic ATR-based labeling
        purging.py            -- label purging (AFML Ch.7)
      engine/
        __init__.py
        walk_forward.py       -- expanding/rolling window training
        simulation.py         -- signal -> trade with cooldown
        sizing.py             -- Kelly, fixed, vol-adjusted
        calibration.py        -- Platt scaling (AD-21, Lin/Lin/Weng 2007 numerics)
        metrics.py            -- Sharpe, PF, MaxDD, ECE, etc.
      validation/
        __init__.py
        causality.py          -- automated causality testing framework
        gates.py              -- statistical gates (10 default + promotion)
        cscv.py               -- combinatorial symmetric cross-validation
        dsr.py                -- deflated Sharpe ratio
        holdout.py            -- holdout access ceremony + guard
      experiment/
        __init__.py
        registry.py           -- TOML experiment registry I/O (AD-9)
        runner.py             -- orchestrator: config -> result
        shap_analysis.py      -- SHAP via pred_contrib
        optimizer.py          -- experiment proposal engine
      logging/
        __init__.py
        structured.py         -- structlog configuration

  tests/
    conftest.py               -- shared fixtures (sample data, configs)
    test_config/
      test_schema.py
      test_loader.py
    test_data/
      test_contracts.py
      test_loaders.py
    test_features/
      test_causality_all.py   -- parametrized causality for every feature
      test_ict/
      test_ta/
      test_regime/
    test_engine/
      test_walk_forward.py
      test_simulation.py
      test_sizing.py
    test_validation/
      test_gates.py
      test_cscv.py
    test_experiment/
      test_registry.py
      test_runner.py
    fixtures/
      sample_btc_1000.parquet -- 1000 rows for fast testing

  data/
    raw/                      -- immutable downloads
      btcusdt/
        candles/              -- monthly parquet (5m bars)
        oi_metrics/           -- monthly parquet (OI)
        aggtrades/            -- monthly parquet (CVD)
        liquidations/         -- monthly parquet (liq)
      nq/                     -- placeholder
      es/                     -- placeholder
    processed/
      btcusdt_5m_train.parquet    -- labeled, feature-enriched, train split
      btcusdt_5m_manifest.json    -- SHA256, row counts, column schema
    holdout/
      btcusdt_5m_holdout.parquet  -- NEVER TOUCH without ceremony
      holdout_access.log          -- permanent log of any holdout access

  experiments/
    registry.toml             -- all experiment results (AD-9)
    configs/                  -- archived experiment TOML files
    models/                   -- saved model artifacts (.npy, .txt)
    shap/                     -- SHAP analysis outputs
    results/                  -- per-experiment JSON results

  outputs/                    -- design docs, research prompts, reports
```

### Naming conventions

- Source files: `snake_case.py`
- Config files: `snake_case.toml`
- Data files: `{instrument}_{timeframe}_{stage}.parquet`
- Experiment configs: `{name}.toml` (descriptive, no numbering)
- Test files: `test_{module}.py` mirroring `src/` structure

---

## 3. Architectural Decisions

### AD-1: Storage -- Parquet on disk, DuckDB for queries

**Decision:** Parquet files as the source of truth. DuckDB as an optional
analytical layer (read-only queries across parquet files).

**Rationale:**
- Parquet is the standard columnar format. Portable, compressed, fast I/O.
- BTCDataset_v2 uses parquet and it works (648K rows, 594 cols, 676 MB).
- DuckDB reads parquet natively without ETL. Zero admin, embedded, pip install.
- No server processes. No Docker. No database maintenance.
- BTC 5m data grows at ~105K rows/year. At this scale, pandas + parquet is
  sufficient for all operations. DuckDB provides SQL when needed for ad-hoc
  analysis but is not required for the pipeline.

**What this means in practice:**
- All pipeline steps read and write parquet files.
- Feature computation reads raw parquet, writes processed parquet.
- Data contracts validate parquet schema on read.
- DuckDB is imported only in analysis scripts, not in the core pipeline.
- Migration to a database (Postgres, TimescaleDB) is possible later by
  changing the loader layer only.

**Rejected alternatives:**
- TimescaleDB: Server process. Overkill for <1M rows. Adds ops burden.
- Arctic/MongoDB: Server process. Same problem.
- SQLite: Not designed for analytical workloads. Slow on wide tables.
- HDF5: Less portable than parquet, weaker ecosystem.

---

### AD-2: Dependencies -- pyproject.toml with pinned versions

**Decision:** Standard pyproject.toml with pinned major.minor versions.
pip for installation. No conda, no poetry.

**Rationale:**
- pyproject.toml is the Python standard (PEP 621).
- Pinned versions ensure reproducibility across CC sessions.
- pip is universally available, including on Windows.
- No exotic tooling that CC needs to learn or troubleshoot.

**Core dependencies:**
```toml
[project]
name = "trading-foundation"
requires-python = ">=3.12"
dependencies = [
    "pandas>=2.2,<3",
    "numpy>=1.26,<2",
    "pyarrow>=15,<17",
    "lightgbm>=4.3,<5",
    "scipy>=1.12,<2",
    "pydantic>=2.6,<3",
    "tomli>=2.0,<3",           # TOML parser (stdlib in 3.11+ but explicit)
    "structlog>=24.1,<25",
    "python-dotenv>=1.0,<2",
    "arch>=7.0,<8",            # Hansen's SPA, bootstrap tests (AD-20)
    "hmmlearn>=0.3,<1",        # HMM regime filter (refitted per fold, AD-18)
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0,<9",
    "pytest-cov>=5.0,<6",
    "duckdb>=0.10,<1",
    "optuna>=4.0,<5",          # hyperparameter optimization (Phase 5)
]
```

**What this means in practice:**
- `pip install -e .` installs everything.
- `pip install -e ".[dev]"` adds test and analysis tools.
- No lock file (pip-tools or uv can generate one if needed later).
- CC can install dependencies in a single command.

---

### AD-3: Testing -- pytest with causality as first-class citizen

**Decision:** pytest for all testing. Causality tests are parametrized and
run automatically. Coverage target: 100% for data contracts, 90% for
feature computation, 80% for engine.

**Rationale:**
- pytest is the Python testing standard.
- Causality testing is THE most important validation in this project.
  The BTCDataset_v2 causality test pattern (compare df[:T] vs df[:T+1])
  is proven and must be automated for every registered feature.
- Property-based testing (hypothesis) is valuable for mathematical functions
  but not a launch requirement.

**Causality test framework:**
```python
# tests/test_features/test_causality_all.py

import pytest
from foundation.features.registry import get_all_features

CAUSALITY_T_VALUES = [500, 1000, 5000, 10000]

@pytest.fixture(scope="session")
def sample_df():
    """Load 50,000 rows of BTC 5m data for causality testing."""
    ...

@pytest.mark.parametrize("feature_cls", get_all_features())
@pytest.mark.parametrize("T", CAUSALITY_T_VALUES)
def test_causality(feature_cls, T, sample_df):
    """At bar T, feature value must depend only on bars 0..T."""
    feature = feature_cls()
    result_short = feature.compute(sample_df.iloc[:T+1], feature.default_params)
    result_long = feature.compute(sample_df.iloc[:T+2], feature.default_params)
    # Row T in short result must equal row T in long result
    pd.testing.assert_series_equal(
        result_short.iloc[-1],
        result_long.iloc[-2],
        check_names=False,
    )
```

**Test organization:**
- `tests/test_features/test_causality_all.py` -- parametrized over ALL features
- `tests/test_data/test_contracts.py` -- schema validation
- `tests/test_engine/` -- walk-forward, simulation, sizing correctness
- `tests/test_validation/` -- gates produce expected results on known inputs
- `tests/fixtures/sample_btc_1000.parquet` -- small dataset for fast tests

**Enforcement:**
- `pytest tests/ -x --tb=short` must pass before any experiment runs.
- Feature registration function calls `test_causality` internally on a
  small sample. If it fails, the feature is not registered.
- CC prompt template includes: "Run pytest before proceeding."

---

### AD-4: Configuration -- TOML files validated by Pydantic

**Decision:** All configuration in TOML files. Pydantic models validate
every config on load. No hardcoded constants in source code.

**Rationale:**
- TOML is human-readable and Python-native (stdlib in 3.11+).
- Pydantic provides runtime type checking, default values, and clear errors.
- This directly solves the BTCDataset_v2 problem where constants were
  scattered across evaluator.py, simulator.py, and sizing.py.
- Every experiment is fully defined by its TOML file. Reproducibility is
  guaranteed: same config = same result.

**Config hierarchy:**
```
Instrument config (btcusdt_5m.toml)
  |-- defines: data paths, column names, timeframe, holdout dates
  |
Experiment config (btc_long_baseline.toml)
  |-- references: instrument config
  |-- defines: label, features, model params, walk-forward params,
  |            threshold, cooldown, costs, gates
  |
Environment config (dev.toml / staging.toml / prod.toml)
  |-- defines: data subset, log level, holdout access policy
```

**Example instrument config:**
```toml
# config/instruments/btcusdt_5m.toml
[instrument]
name = "BTCUSDT"
exchange = "binance"
type = "perpetual"
timeframe = "5m"
bars_per_day = 288
bars_per_year = 105120

[data]
raw_dir = "data/raw/btcusdt"
train_path = "data/processed/btcusdt_5m_train.parquet"
holdout_path = "data/holdout/btcusdt_5m_holdout.parquet"
manifest_path = "data/processed/btcusdt_5m_manifest.json"

[holdout]
start_date = "2025-03-01"
end_date = "2026-02-28"
embargo_bars = 288

[columns]
timestamp = "bar_start_ts_utc"
open = "open"
high = "high"
low = "low"
close = "close"
volume = "volume_base"
```

**Example experiment config:**
```toml
# config/experiments/btc_long_baseline.toml
[experiment]
name = "btc_long_baseline"
instrument = "btcusdt_5m"
direction = "long"
description = "Reproduce D54a baseline from BTCDataset_v2"

[label]
type = "triple_barrier"
direction = "long"
r_target = 2
horizon_bars = 48
# Generates column name: label_long_hit_2r_48c

[features]
mode = "explicit"  # "explicit" or "all_registered"
include = [
    "ict.displacement",
    "ict.fvg",
    "ict.structure",
    "ict.ote",
    "ict.premium_discount",
    "ta.momentum",
    "regime.hmm",
]
# Each feature family has its own params below

[features.ict.displacement]
disp_k = 1.5
disp_close_frac = 0.75
age_cap = 48

[features.ict.fvg]
age_cap = 100
min_size_atr = 0.50

[features.ict.structure]
pivot_n_internal = 5
pivot_n_external = 10

[features.ict.ote]
fib_low = 0.618
fib_high = 0.786

[model]
type = "lightgbm"
threshold = 0.60
cooldown_bars = 576

[model.lightgbm]
learning_rate = 0.01
num_leaves = 31
max_depth = 6
min_child_samples = 50
subsample = 0.8
colsample_bytree = 0.8
reg_alpha = 0.1
reg_lambda = 1.0
is_unbalance = true

[walk_forward]
mode = "rolling"  # "rolling" (default, AD-13) or "expanding"
min_train = 105000
test_fold = 52500
embargo_bars = 288
# Embargo derived at runtime from label params (AD-28):
# embargo = horizon_bars (label overlap) + cooldown_bars (trade overlap)

# Sample uniqueness weighting (AD-29): Triple-barrier labels on adjacent bars
# overlap by (horizon-1) bars. Instead of within-train purging (which destroys
# sample size), compute label concurrency and weight each sample by 1/concurrency
# (AFML Ch4). Pass uniqueness array as LightGBM sample_weight. Reduces IS/OOS gap
# by 15-25%.
sample_weight = "uniqueness"  # "uniqueness" (AFML Ch4) or "uniform"

[costs]
# Dynamic cost model (AD-31). Full formula:
# cost_R = 2*(effective_spread_bps + commission_bps)/10000/stop_pct
#        + funding_8h*(holding_hours/8)/stop_pct
#        + latency_slippage_bps/stop_pct
# Effective spread: EDGE estimator + ATR-conditioned widening (base=3.0, beta=2.5)
# Latency slippage: 3-10bps calm, 10-30bps displacement (bar-close delay 0.5-3s)
spread_model = {type = "edge", base_bps = 3.0, atr_beta = 2.5}
commission_bps = 5.0  # Binance VIP 0 taker (corrected from 4.0)
funding_source = "binance_api"
latency_slippage_bps = 5.0  # conservative default, higher during displacement
cost_filter_max = 0.8  # skip trades where cost_R > 0.8
r_target = 2

[sizing]
method = "kelly"
divisor = 40
floor = 0.01
cap = 0.02

[gates]
min_trades_per_year = 100
min_oos_auc = 0.75
# max_pbo = 0.05  # DEFERRED to Phase 5 (AD-23). Requires N>=100 Optuna configs.
min_psr = 0.99
min_sharpe = 2.0
min_wr = 0.55
min_ev_r = 0.50
max_drawdown = 0.20
max_ece = 0.05
```

**The wiring bug fix:**
In BTCDataset_v2, SEARCH_SPACE defined 22 parameters but augment_features()
called rules.py functions with hardcoded defaults. Optuna searched a space
that never reached the functions.

In the new foundation, this is structurally impossible:
1. Feature params are defined in the experiment TOML.
2. The experiment runner passes params from TOML to each feature's compute().
3. Feature.compute() REQUIRES a params dict (no default fallback).
4. If a param is missing from TOML, Pydantic raises a ValidationError
   before the experiment starts.

The pipeline is: TOML -> Pydantic validation -> feature.compute(df, params).
There is no path where params are silently dropped.

---

### AD-5: Secrets -- .env with python-dotenv

**Decision:** API keys and credentials in `.env` file only. Loaded via
python-dotenv. Never in config files. Never in source code.

**Rationale:**
- Simple, proven, universally understood.
- .env is in .gitignore (template .env.example is committed).
- No vault, no keyring, no encrypted config -- overkill for personal use.
- If the user moves to a team setup, migrate to environment variables or
  a secrets manager at that point.

**Implementation:**
```
# .env.example (committed)
BINANCE_API_KEY=
BINANCE_API_SECRET=
COINALYZE_API_KEY=
```

```python
# src/foundation/config/loader.py
from dotenv import load_dotenv
load_dotenv()  # loads .env into os.environ
```

---

### AD-6: Logging -- structlog with file rotation

**Decision:** structlog for structured logging. Console output is
human-readable. File output is JSON for machine parsing. One log file
per experiment run. Daily rotation for system logs.

**Rationale:**
- BTCDataset_v2 uses print() everywhere. This works for interactive sessions
  but produces no persistent record and no structured data.
- structlog provides structured key-value logging that is both human-readable
  in the console and machine-parseable in files.
- Per-experiment log files enable post-mortem analysis of failed runs.
- JSON logs enable CC to parse experiment outcomes programmatically.

**Configuration:**
```python
# src/foundation/logging/structured.py
import structlog
import logging
from pathlib import Path

def setup_logging(experiment_name: str = None, level: str = "INFO"):
    """Configure structlog for console + file output."""
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),  # human-readable console
    ]
    structlog.configure(processors=processors)

    # File handler for experiment logs
    if experiment_name:
        log_path = Path("experiments/logs") / f"{experiment_name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(getattr(logging, level))
        logging.getLogger().addHandler(file_handler)
```

**What gets logged:**
- Experiment start/end with full config
- Each walk-forward fold: train size, test size, AUC, duration
- Gate results (pass/fail with values)
- Feature registration events
- Holdout access attempts (always logged, even denials)
- Errors with full tracebacks

---

### AD-7: Knowledge Base -- Layered files with structured update protocol

**Decision:** Four knowledge files, each with a specific purpose and
mandatory update rules. Detailed spec in KNOWLEDGE_BASE_DESIGN.md.

**File inventory:**

| File | Purpose | Update frequency |
|------|---------|-----------------|
| CLAUDE.md | Project overview, file structure, current state | Every session that changes state |
| STATUS.md | Current task, blockers, next steps | Start and end of every session |
| KNOWLEDGE.md | Evidence base, experiments, dead ends, RQs | After every experiment |
| DECISIONS.md | Architectural decisions (append-only) | When a design choice is made |

**Critical rules:**
- CC reads CLAUDE.md first in every session. If CLAUDE.md is stale, CC
  reads STATUS.md to find what changed.
- KNOWLEDGE.md uses structured tables (not prose) so CC can parse it
  mechanically, not interpretively.
- DECISIONS.md is append-only. Decisions are never edited, only superseded
  with a new entry referencing the old one.
- STATUS.md is ephemeral -- it reflects the NOW, not history.

**CC session startup protocol:**
1. Read CLAUDE.md (always)
2. Read STATUS.md (always)
3. Read KNOWLEDGE.md if running experiments
4. Read DECISIONS.md if making architectural choices

### Foundation Session Protocol

The Foundation CLAUDE.md must include the same session
start/end protocol as BTCDataset_v2/CLAUDE.md.
trading-brain is shared across both projects.
All Foundation sessions must:
- Read trading-brain/STATUS.md at start
- Read trading-brain/HANDOFF.md last entry at start
- Write session end updates to trading-brain
- Commit trading-brain after every session

The Foundation is a separate git repo but shares
trading-brain as the organizational backbone.
See trading-brain/SESSION_PROTOCOL.md for the full protocol.
5. Read the specific experiment TOML if modifying an experiment

---

### AD-8: Autonomous Training Interface -- CLI commands with JSON contracts

**Decision:** Every pipeline step is a CLI command that reads TOML config,
produces JSON output, and returns an exit code. CC orchestrates by calling
CLI commands in sequence.

**Rationale:**
- CC interacts with the system through shell commands. CLI is the natural
  interface.
- JSON output is machine-readable. CC can parse results and decide next steps.
- No daemon processes, no message queues, no orchestration frameworks.
- Each command is idempotent: same config = same result.

**CLI commands:**
```
python -m foundation download   --instrument btcusdt_5m
python -m foundation build      --instrument btcusdt_5m
python -m foundation train      --config experiments/configs/exp.toml
python -m foundation evaluate   --config experiments/configs/exp.toml
python -m foundation calibrate  --config experiments/configs/exp.toml
python -m foundation shap       --config experiments/configs/exp.toml
python -m foundation validate   --config experiments/configs/exp.toml
python -m foundation holdout    --config experiments/configs/exp.toml --confirm
python -m foundation status
python -m foundation report     --experiment exp_name
```

**Experiment lifecycle (CC perspective):**
```
1. CC writes experiment TOML to config/experiments/
2. CC runs: python -m foundation train --config <path>
   -> Produces: experiments/models/<name>_oos_probs.npy
   -> Produces: experiments/results/<name>_train.json
3. CC runs: python -m foundation evaluate --config <path>
   -> Produces: experiments/results/<name>_eval.json
   -> JSON includes: all metrics, gate results (PASS/FAIL), fold details
4. CC reads eval JSON, decides whether to:
   a. Run SHAP analysis (if gates pass)
   b. Modify config and re-run (if gates fail)
   c. Record as dead end (if hypothesis is wrong)
5. CC updates KNOWLEDGE.md with results
6. CC updates STATUS.md with next steps
```

**Autonomous mode:**
The optimizer (foundation.experiment.optimizer) reads KNOWLEDGE.md and
experiments/registry.toml, proposes the next experiment config, and can
run in a loop:
```
python -m foundation optimize --max-experiments 5 --mode checkpoint
```
In checkpoint mode: proposes, prints, waits for approval.
In autonomous mode: proposes, runs, evaluates, records, repeats.

### AD-9: Registry as TOML Not JSON
**Date:** 2026-03-04
**Context:** BTCDataset_v2 uses registry.json for experiment results. JSON is hard
for CC to write manually (trailing commas, escaping). TOML is simpler.
**Decision:** Foundation uses TOML for experiment registry. CC can append entries directly.
**Rationale:** TOML is human-writable, supports comments, no trailing comma issues.
CC writes TOML more reliably than JSON.
**Alternatives:** JSON (rejected: write-error-prone), YAML (rejected: indent-sensitive),
SQLite (rejected: can't read with text tools).
**Consequences:** Registry is a TOML file. Existing JSON registry in BTCDataset_v2 unchanged.
**Source:** MASTER_PLAN_TEMPLATE.md Step 5.1

### AD-10: Archival Threshold 300 Lines (Context Saturation)
**Date:** 2026-03-04
**Context:** CC context saturates at ~20,000 tokens (~300 lines of dense content).
Files loaded fully by CC must stay under this limit.
**Decision:** Any file CC loads fully is archived when it exceeds 300 lines.
Old sections moved to archive files. Active file stays lean.
CLAUDE.md: 80-100 lines. STATUS.md: 40-50 lines.
KNOWLEDGE.md: 200-300 lines. research/synthesis.md: 200-250 lines.
EVENT_LOG.md and DECISIONS.md: unlimited (grep-accessed, never loaded fully).
**Rationale:** Exceeding context saturation causes CC to miss information or
produce lower-quality responses. 300 lines is the empirically observed limit.
**Alternatives:** No limit (rejected: CC quality degrades), chunked loading
(rejected: CC doesn't support pagination).
**Consequences:** Files stay lean. Archived content preserved but not auto-loaded.
**Source:** KNOWLEDGE_BASE_DESIGN.md Section 5 (line 359)

### AD-11: Daily Sharpe as Primary Gate (Lo 2002 Correction)
**Date:** 2026-03-04
**Context:** Per-bar Sharpe uses sqrt(N) annualization where N=105,120 bars.
This produces inflated values (Sharpe 12-21) because it assumes i.i.d. returns.
Lo (2002) showed autocorrelation invalidates this assumption. ~50% of trading
days have zero P&L, further inflating the metric.
**Decision:** Daily Sharpe is the primary performance gate. Per-bar Sharpe reported
for comparison only. Realistic daily Sharpe target: 1.5-3.0.
**Rationale:** Daily Sharpe is the industry standard and handles autocorrelation
more naturally than per-bar. Per-bar Sharpe 12-21 is NOT a deployment target.
**Alternatives:** Per-bar Sharpe with Lo correction (rejected: correction factor
uncertain without empirical autocorrelation measurement), monthly Sharpe
(rejected: too few data points for significance).
**Consequences:** All experiments report daily Sharpe alongside per-bar.
Promotion gates use daily Sharpe. D55b holdout reported Daily Sharpe 10.71.
**Review finding (2026-03-04):** Three external reviewers flagged that daily Sharpe is inflated by zero-return days for sparse-trading systems. Foundation should report BOTH daily Sharpe (with zero days included) AND per-trade Sharpe annualized by sqrt(trades/year) as complementary metrics. Neither is the "correct" Sharpe -- they answer different questions.
**Source:** REDESIGN_V2.md, PERPLEXITY_V2.md Domain 1

### AD-12: 3-Seed Minimum for Experiments (76% Seed Variance)
**Date:** 2026-03-04
**Context:** D54b revealed extreme seed variance in LightGBM. Same config
(t=0.70, cd=288) produced Sharpe 12.09 and 21.34 on different random seeds.
This is a 76% swing, making all single-seed point estimates unreliable.
**Decision:** All promoted experiments must run 3+ seeds. Report mean +/- std.
Single-seed results are preliminary only.
**Rationale:** Root cause: feature_fraction + bagging_fraction in LightGBM
introduce stochastic variation. The 76% swing means a single run can over- or
under-estimate true performance by nearly 2x.
**Alternatives:** Single seed (rejected: unreliable), 10+ seeds (rejected: compute
cost for 9-fold walk-forward), fixed seed only (rejected: masks true variance).
**Consequences:** Experiment promotion requires 3 seeds. Compute cost ~3x per
promoted config. Preliminary exploration can use single seed.
**Source:** REDESIGN_V2.md, STRATEGY_LOG.md D54b

### AD-13: Rolling 12-18mo Window Over Expanding from 2020
**Date:** 2026-03-04
**Context:** Current walk-forward uses expanding window from 2020-01. ETF approval
in January 2024 was a structural break that changed BTC market behavior (Stevens
Institute research). Pre-2024 data may represent a different market structure.
**Decision:** Foundation should use rolling 12-18 month training windows instead of
expanding from 2020. Earlier data used for validation only, not training.
**Rationale:** Non-stationarity in crypto markets means older data can hurt more than
help. Rolling windows adapt to regime changes. ETF structural break makes
pre-2024 data potentially misleading.
**Alternatives:** Expanding from 2020 (rejected: includes pre-ETF regime),
rolling 6mo (rejected: too little training data), weighted (rejected: adds
hyperparameter for window weighting).
**Consequences:** Training data is 12-18 months (130K-260K bars). More folds
possible (~24 folds). Earlier data still used for out-of-sample validation.
**Source:** PERPLEXITY_V2.md Domain 5, FOUNDATION_DESIGN.md anti-pattern #13

### AD-14: HMM Must Use Online Forward Filtering

**Date:** 2026-03-04
**Status:** ACCEPTED

**Decision:** Foundation HMM must use forward-only filtering.
Never use hmmlearn predict() (Viterbi) or any batch decoder.

**Rationale:** Viterbi backward pass leaks future state into
past labels. BTCDataset_v2 avoided this via custom NumPy
GaussianHMM1D with filter_proba() (forward pass only).
BTCDataset_v2 audit: CLEAN. Foundation must match.

**Implementation:** Use hmmlearn score_samples() or custom
forward pass. Add HMM causality test to test suite.

**Update (AD-18):** In addition to forward-only filtering, the HMM must be
refitted on training data only per WF fold. Full-sample parameter estimation
is parameter snooping -- HMM parameters encode future statistical properties.
hmmlearn has no public causal-only API (confirmed from base.py source).
Custom log-domain forward filter required. Compute cost: ~10ms per fold,
~4 seconds total. State labels aligned across folds by sorting by emission mean.

### AD-15: LightGBM min_child_samples Minimum 100

**Date:** 2026-03-04
**Status:** ACCEPTED

**Decision:** Foundation LightGBM default min_child_samples=200.
Acceptable range: 100-500. Never use default (20).

**Rationale:** Default 20 too low for noisy financial data.
Contributes to seed instability (76% Sharpe variance in D54b).
Source: Opus V2 research, confirmed by LightGBM practitioners.

### AD-16: Triple-Barrier AUC Not Comparable to Fixed-Horizon

**Date:** 2026-03-04
**Status:** ACCEPTED

**Decision:** All experiment registry entries must include
label_type field. AUC on triple-barrier labels must be
labeled as such and never compared to fixed-horizon benchmarks.

**Rationale:** Triple-barrier inflates AUC by 0.10-0.20 vs
fixed-horizon on same features (Opus V2, Perplexity V2,
MQL5 study). AUC 0.79 triple-barrier ~ 0.59-0.69 fixed-horizon.

### AD-17: SHAP Rank Correlation Across Folds Required

**Date:** 2026-03-04
**Status:** ACCEPTED

**Decision:** Compute Spearman rank correlation of SHAP
importance across all walk-forward folds. Flag if mean rho < 0.6.
Add as gate to validation framework.

**Rationale:** Unstable SHAP rankings signal regime-specific
overfitting. Low-cost diagnostic with high signal value.
Source: Perplexity V2.

---

## 4. Data Contracts

### Schema definition

Every parquet file has a schema contract defined in code. The contract
specifies: column name, dtype, NaN policy, value range, and description.

```python
# src/foundation/data/contracts.py
from pydantic import BaseModel, Field
from typing import Optional

class ColumnContract(BaseModel):
    name: str
    dtype: str                        # "float32", "float64", "int64", "datetime64[ms, UTC]"
    nullable: bool = False            # True if NaN is allowed
    min_val: Optional[float] = None   # value range (None = no bound)
    max_val: Optional[float] = None
    description: str = ""

class DataContract(BaseModel):
    name: str
    version: str
    row_count_range: tuple[int, int]  # (min, max) expected rows
    columns: list[ColumnContract]
    timestamp_col: str = "bar_start_ts_utc"
    must_be_sorted: bool = True       # by timestamp
    sha256: Optional[str] = None      # for immutable datasets

# Example
BTCUSDT_5M_RAW = DataContract(
    name="btcusdt_5m_raw",
    version="1.0",
    row_count_range=(600_000, 700_000),
    columns=[
        ColumnContract(name="bar_start_ts_utc", dtype="datetime64[ms, UTC]",
                      description="Bar open timestamp"),
        ColumnContract(name="open", dtype="float64",
                      min_val=0, description="Open price"),
        ColumnContract(name="high", dtype="float64",
                      min_val=0, description="High price"),
        ColumnContract(name="low", dtype="float64",
                      min_val=0, description="Low price"),
        ColumnContract(name="close", dtype="float64",
                      min_val=0, description="Close price"),
        ColumnContract(name="volume_base", dtype="float64",
                      min_val=0, nullable=True,
                      description="Volume in base currency (BTC)"),
    ],
)
```

### Runtime validation

```python
# src/foundation/data/loaders.py
def load_train(instrument_config: InstrumentConfig) -> pd.DataFrame:
    """Load training data with contract validation and holdout guard."""
    path = Path(instrument_config.data.train_path)
    df = pd.read_parquet(path)

    # Contract validation
    contract = get_contract(instrument_config.instrument.name)
    validate_contract(df, contract)

    # Holdout guard: verify no rows from holdout period
    holdout_start = pd.Timestamp(instrument_config.holdout.start_date)
    ts = df[instrument_config.columns.timestamp]
    if (ts >= holdout_start).any():
        raise HoldoutViolationError(
            f"Train data contains {(ts >= holdout_start).sum()} rows "
            f"from holdout period (>= {holdout_start}). "
            f"This is a FATAL error. Rebuild the train split."
        )

    return df
```

### NaN policy

| Context | NaN policy |
|---------|-----------|
| Raw OHLCV | Not allowed (gaps = data quality issue) |
| Feature warmup | Allowed for first N bars (documented per feature) |
| OI/CVD/Liq | Allowed where data source doesn't cover (has_oi mask) |
| Labels | Allowed at tail (forward-looking labels end before data ends) |
| Model input | LightGBM handles NaN natively (as missing) |
| Holdout features | Same as train (no special treatment) |

### Column naming convention

```
{source}_{metric}[_{qualifier}]

Source prefixes:
  (none)    Raw OHLCV from exchange
  ict_      ICT-derived structural features
  ta_       Technical analysis indicators
  regime_   Market regime features
  micro_    Microstructure (OI, CVD, liquidations)
  label_    Forward-looking ML labels

HTF prefixes (prepended to base name):
  m15_      15-minute
  m30_      30-minute
  h1_       1-hour
  h4_       4-hour
  d1_       Daily

Examples:
  ict_fvg_bull_1_age
  ta_rsi_14
  regime_hmm_state
  micro_oi_zscore_20
  label_long_hit_2r_48c
```

---

## 5. Feature Framework

### Feature protocol

Every feature family implements this interface:

```python
# src/foundation/features/protocol.py
from typing import Protocol
import pandas as pd

class Feature(Protocol):
    """Interface that all feature families must implement."""

    @property
    def name(self) -> str:
        """Unique name for this feature family (e.g., 'ict.fvg')."""
        ...

    @property
    def output_columns(self) -> list[str]:
        """List of column names this feature produces."""
        ...

    @property
    def default_params(self) -> dict:
        """Default parameter values (used for testing only, not production)."""
        ...

    @property
    def warmup_bars(self) -> int:
        """Number of bars needed before output is valid."""
        ...

    @property
    def dependencies(self) -> list[str]:
        """Other feature families that must run before this one."""
        ...

    def compute(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """
        Compute features and return as DataFrame with self.output_columns.

        Args:
            df: Input DataFrame with OHLCV + any dependency columns.
            params: Parameters from experiment TOML config.
                    MUST be passed explicitly. No fallback to defaults.

        Returns:
            DataFrame with one column per output_columns entry.
        """
        ...
```

### Feature registration

```python
# src/foundation/features/registry.py
_REGISTRY: dict[str, type] = {}

def register(cls):
    """Decorator to register a feature family."""
    instance = cls()
    name = instance.name

    # Validate the feature implements the protocol
    assert hasattr(instance, 'compute'), f"{name}: missing compute()"
    assert hasattr(instance, 'output_columns'), f"{name}: missing output_columns"
    assert hasattr(instance, 'default_params'), f"{name}: missing default_params"

    # Run quick causality test on registration
    from tests.fixtures import load_sample_1000
    sample = load_sample_1000()
    _run_quick_causality(instance, sample, T=500)

    _REGISTRY[name] = cls
    return cls

def get_all_features() -> list[type]:
    return list(_REGISTRY.values())

def compute_features(df, feature_names, params_by_feature):
    """
    Compute requested features in dependency order.
    params_by_feature: dict mapping feature name -> params dict
    """
    # Topological sort by dependencies
    ordered = _topo_sort(feature_names)

    result = df.copy()
    for name in ordered:
        cls = _REGISTRY[name]
        feature = cls()
        params = params_by_feature.get(name, {})
        if not params:
            raise ConfigError(
                f"No params provided for feature '{name}'. "
                f"Every feature requires explicit params from config."
            )
        feature_df = feature.compute(result, params)
        result = pd.concat([result, feature_df], axis=1)

    return result
```

### Parameter passthrough (the wiring bug fix, detailed)

The BTCDataset_v2 wiring bug:
```
parameters.py:  SEARCH_SPACE = {"fvg_age_cap": [48, 96, 288], ...}
simulator.py:   augment_features(df)  # calls rules.py with NO params
rules.py:       detect_fvg_bull(df, age_cap=288)  # hardcoded default
```

Optuna searches SEARCH_SPACE but the values never reach rules.py.

The foundation fix:
```
experiment.toml:  [features.ict.fvg]  age_cap = 100
schema.py:        FvgParams(age_cap=100)  # Pydantic validates
runner.py:        compute_features(df, names, params_by_feature)
registry.py:      feature.compute(df, params)  # params passed through
fvg.py:           def compute(self, df, params):
                      age_cap = params["age_cap"]  # KeyError if missing
```

There is no code path where params are silently dropped. If Optuna searches
`fvg_age_cap = 48`, the value goes into the TOML, through Pydantic, through
the registry, into fvg.compute(). If the value is missing, Pydantic raises
a ValidationError before the experiment starts.

---

## 6. Label Framework

### Triple-barrier labels

The label system from BTCDataset_v2 is proven and carries forward:
- Stop loss: 1R (1 x ATR at entry)
- Take profit: configurable R-multiple (default 2R)
- Time horizon: configurable bars (default 48 = 4 hours at 5m)
- Direction: separate long and short labels

**Configuration in TOML:**
```toml
[label]
type = "triple_barrier"
direction = "long"
r_target = 2
horizon_bars = 48
```

### Purging and embargo

AFML Chapter 7 purging and embargo are mandatory:
- Embargo: configurable bars between train and test (default 288 = 24h)
- Purging: remove training samples whose labels overlap the test period

These are enforced by the walk-forward engine, not optional flags.

---

## 7. Validation Suite

### Causality tests

Every feature must pass causality at T in [500, 1000, 5000, 10000]:
```
result_short = feature.compute(df[:T+1], params)
result_long = feature.compute(df[:T+2], params)
assert result_short.iloc[-1] == result_long.iloc[-2]
```

This runs automatically on feature registration and as part of pytest.

### Planted Signal Test (AD-22)

Pipeline integrity gate. Must pass before any real experiments run.

**Null test:** Uniform random labels independent of features. Expected AUC = 0.50.
Must be within [0.497, 0.503] for n=543K. Any deviation indicates pipeline leakage.

**Feature-dependent test:** Labels generated via prob = Phi(d' * z_j) for a chosen
feature j. Verify recovered AUC exceeds 85% of theoretical AUC (Green & Swets 1966:
AUC = Phi(d'/sqrt(2))). Run 20 trials per condition.

Both tests use the real feature matrix. Combined, they verify labels, features,
model, walk-forward splitting, and evaluation -- the full pipeline.

### Statistical gates (10 default)

Carried forward from BTCDataset_v2 evaluator.py:

| Gate | Threshold | Rationale |
|------|-----------|-----------|
| MIN_TRADES_PER_YEAR | 100 | Statistical significance |
| MIN_OOS_AUC | 0.75 | Discriminative power |
| ~~MAX_PBO~~ | ~~0.05~~ | Deferred to Phase 5 (AD-23). Replaced by SPA (AD-20). |
| MIN_PSR | 0.99 | Probabilistic Sharpe ratio |
| MIN_WF_WINDOWS | "all" | Every fold must be profitable |
| MIN_SHARPE | 2.0 | Annualized Sharpe (daily) |
| MIN_WR | 0.55 | Win rate above break-even |
| MIN_EV_R | 0.50 | Expected value per trade in R |
| MAX_DRAWDOWN | 0.20 | Maximum drawdown |
| MAX_ECE | 0.05 | Expected calibration error |

**New gates (AD-43, Phase 3+):**

| Gate | Metric | Threshold | Phase |
|------|--------|-----------|-------|
| BINOMIAL_TEST | p_win > 1/3 | p < 0.01 (Holm-corrected) | 3 |
| COST_ADJ_EV | Cost-adjusted EV per trade | >= +0.25R, LCB > 0 | 3 |
| PER_TRADE_SHARPE | Per-trade Sharpe x sqrt(N) | >= 2.0 | 3 |
| WORST_FOLD_EV | Worst-fold cost-adjusted EV | >= +0.05R | 3 |
| SEED_CV | Coefficient of variation across seeds | < 0.30 | 3 |
| SPIEGELHALTER_Z | Calibration diagnostic | p > 0.01 (diagnostic, not hard gate) | 3 |

Promotion gates (optional, for model upgrades):
- AUC_PROMOTION_DELTA >= 0.005
- LOGLOSS_MUST_IMPROVE = true

### CSCV / PBO -- Deferred to Phase 5 (AD-23)

CSCV operates on a T x N matrix where N = number of strategy configurations
(Optuna trials), NOT walk-forward fold boundaries. Bailey et al. (2015) recommend
N >= 100 for robust PBO estimates. With fewer configs, PBO is noisy and unreliable.

CSCV is applied in Phase 5 after Optuna generates N >= 100 distinct configurations.
Use S = 16 temporal blocks on concatenated OOS returns, yielding C(16,8) = 12,870
combinations. Self-contained implementation (pypbo broken on pandas >= 2.0).

Phase 3 uses SPA (AD-20) as the primary multiple-testing correction instead.

### Hansen's SPA -- Primary Multiple-Testing Correction (AD-20)

Replaces CSCV/PBO as the Phase 3 multiple-testing correction. Uses the `arch`
package (arch.bootstrap.SPA). Tests whether the best strategy significantly
outperforms a zero-benchmark using stationary bootstrap, handling temporal
dependence and cross-correlation between trials.

DSR (Bailey & Lopez de Prado 2014) remains as a complementary parametric check.
If both SPA and DSR reject the null, strong evidence. If they disagree, trust SPA
(handles correlated trials better).

Romano-Wolf StepM (also in arch) identifies which specific trials beat benchmark.

### OOS Permutation Test (Monte Carlo)

OOS-only permutation: keep trained models fixed, block-shuffle OOS returns within
each fold (block size = embargo length, 288 bars). 10,000 permutations for iterative
development (~100 seconds). 200 full-retrain permutations as final gate (~5 days,
parallelizable). P-value formula (Phipson & Smyth 2010): p = (b+1)/(m+1).

### DSR (Deflated Sharpe Ratio)

Multiple testing correction. Critical for Optuna optimization where hundreds
of trials inflate the expected maximum Sharpe from noise.

After N experiments: expected max SR from noise = sqrt(2 * ln(N)).
DSR >= 0.95 for promotion. DSR >= 0.99 for deployment.

### Holdout ceremony

Holdout data is NEVER accessed in normal operation. When the time comes:

1. CC writes a pre-registration document: HOLDOUT_PREREGISTRATION.md
   - Lists exact experiment config that will be tested
   - Lists exact pass/fail thresholds (committed before seeing results)
   - Lists what action follows each outcome
2. CC runs: `python -m foundation holdout --config <path> --confirm`
3. The command logs the access to `data/holdout/holdout_access.log` with
   timestamp, config hash, and user confirmation
4. Results are computed and saved
5. This log entry is permanent and cannot be deleted
6. After first holdout access, STATUS.md is updated: "HOLDOUT TOUCHED"

---

## 8. Experiment Lifecycle

### Workflow

```
1. PROPOSE: CC reads KNOWLEDGE.md, identifies next research question
2. CONFIG:  CC writes experiment TOML to config/experiments/
3. VALIDATE CONFIG: python -m foundation validate --config <path>
   -> Checks: all features registered, all params present, gates defined
4. TRAIN:   python -m foundation train --config <path>
   -> Walk-forward training, produces OOS probs and fold metrics
5. EVALUATE: python -m foundation evaluate --config <path>
   -> Simulates trades, computes metrics, checks gates
6. RECORD:  CC reads eval JSON, updates experiments/registry.toml
7. ANALYZE: python -m foundation shap --config <path> (if gates pass)
8. UPDATE:  CC updates KNOWLEDGE.md with experiment results
9. DECIDE:  CC determines next action based on results
```

### Registry format

```json
{
  "experiments": [
    {
      "name": "btc_long_baseline",
      "timestamp": "2026-03-15T10:30:00Z",
      "config_hash": "sha256:abc123...",
      "config_path": "config/experiments/btc_long_baseline.toml",
      "status": "complete",
      "metrics": {
        "trades": 742,
        "trades_per_year": 178.3,
        "win_rate": 0.7601,
        "ev_r": 1.2303,
        "sharpe_daily": 10.71,
        "sharpe_per_trade": 12.81,
        "max_drawdown": 0.0547,
        "auc": 0.7933,
        "ece": 0.016,
        "spa_pvalue": null,  // computed in Phase 5 after N>=100 trials
        "pbo": null,         // computed in Phase 5 (AD-23)
        "psr": 1.0
      },
      "gates": {
        "MIN_TRADES_PER_YEAR": {"value": 178.3, "threshold": 100, "pass": true},
        "MIN_OOS_AUC": {"value": 0.7933, "threshold": 0.75, "pass": true}
      },
      "gate_summary": "10/10 PASS",
      "fold_details": [ ... ],
      "notes": "Reproduces D54a from BTCDataset_v2"
    }
  ]
}
```

---

## 9. Extension Model

### Adding a new instrument (NQ, ES)

1. Create instrument config: `config/instruments/nq_5m.toml`
   - Define data paths, column names, trading hours, holdout dates
   - Define instrument-specific parameters (no funding rate, session times)
2. Create data downloader: `src/foundation/data/downloaders/cme.py`
3. Run: `python -m foundation download --instrument nq_5m`
4. All feature computation, training, and evaluation use the same engine
   with the new instrument config.

Features that are BTC-specific (funding rate, liquidation cascades) will
produce NaN columns for NQ/ES. LightGBM handles this natively.

### Adding a new signal family

1. Create feature module: `src/foundation/features/new_family/signal.py`
2. Implement the Feature protocol (name, output_columns, compute, etc.)
3. Decorate with @register
4. Add params to experiment TOML
5. Run pytest -- causality tests run automatically
6. Feature is now available for experiments

### Adding a new label type

1. Add label function to `src/foundation/labels/`
2. Add label type to experiment config schema
3. The walk-forward engine uses whatever label column the config specifies

---

## 10. Self-Critique

### [R1] Simplicity -- can a solo developer maintain this?

**Assessment: YES, with caveats.**
- Pydantic adds a learning curve but prevents entire classes of config bugs.
- structlog adds a dependency but replaces ad-hoc print() statements.
- The project structure has more files than BTCDataset_v2, but each file has
  a single clear purpose.
- **Risk:** Over-abstraction of the feature protocol. If the protocol is too
  rigid, CC will waste time fighting the interface instead of doing research.
- **Mitigation:** Start with concrete implementations, extract protocol later.
  The protocol is a goal, not a starting requirement.

### [R2] CC interface -- is everything file-reconstructable?

**Assessment: YES.**
- All state is in files: TOML configs, TOML registry (AD-9), markdown knowledge.
- CC reads CLAUDE.md + STATUS.md at session start and has full context.
- No hidden state in databases, environment variables, or running processes.
- **Risk:** KNOWLEDGE.md grows unwieldy (BTCDataset_v2's STRATEGY_LOG hit
  3,923 lines and became hard to parse).
- **Mitigation:** Structured tables, not prose. Archive old experiments to
  a separate file. Keep KNOWLEDGE.md under 500 lines.

### [R3] Vibe coding -- where can someone skip validation?

**Identified gaps and fixes:**

| Gap | How to skip | Structural fix |
|-----|-------------|---------------|
| Add feature without tests | Write .py, don't register | @register decorator runs causality test |
| Run experiment without config | Call functions directly | Runner requires config path (no defaults) |
| Access holdout | Load parquet directly | Holdout guard in loader raises exception |
| Skip gates | Read metrics, ignore gates | Gates are part of eval JSON (always computed) |
| Hardcode params | Pass dict instead of TOML | Pydantic rejects unregistered params |
| Skip KNOWLEDGE update | Just don't update it | STATUS.md checklist: "Updated KNOWLEDGE.md? [y/n]" |

The last gap (KNOWLEDGE update) is the weakest link. It requires CC
discipline, not structural enforcement. The CC prompt template includes
the update step, but CC can skip it.

**Remaining manual discipline required:**
- Updating KNOWLEDGE.md after experiments (structural enforcement is hard)
- Writing meaningful experiment descriptions (quality is subjective)
- Choosing the right next research question (requires judgment)

### [R4] 12-month -- will this age well?

**Assessment: YES for core, WATCH for tooling.**
- Parquet: Apache standard, not going anywhere. SAFE.
- Pydantic v2: Stable, widely adopted, 5+ year track record. SAFE.
- LightGBM: Best-in-class for tabular data. No serious challenger. SAFE.
- structlog: Niche but stable. If it dies, stdlib logging is the fallback. LOW RISK.
- TOML: Python stdlib since 3.11. SAFE.
- pytest: Industry standard. SAFE.
- **Risk:** Python 3.14 is bleeding edge. Some packages may lag.
- **Mitigation:** requires-python >= 3.12 (not pinned to 3.14).

### [R5] Failure modes -- what breaks first?

| Failure mode | Likelihood | Impact | Mitigation |
|-------------|------------|--------|------------|
| Binance API changes | MEDIUM | Data pipeline breaks | Cache raw data locally, retry logic |
| Feature lookahead bug | LOW (with causality tests) | Invalid results | Automated causality testing |
| Config drift | MEDIUM | Unreproducible results | Config hash in registry, Pydantic validation |
| KNOWLEDGE.md staleness | HIGH | CC proposes redundant experiments | Structured format, mandatory update step |
| Holdout contamination | LOW (with guard) | Invalid final validation | Runtime guard + permanent access log |
| Overfitting to Optuna | MEDIUM | Inflated metrics | DSR, trial counting, pre-registration |
| NQ/ES data quality | UNKNOWN | Feature NaN, model degradation | Data contracts validate on load |
| CC context loss mid-task | HIGH (inherent) | Incomplete work | STATUS.md tracks current task state |

**Most likely first failure: KNOWLEDGE.md staleness.** CC will eventually
skip an update, and the knowledge base will drift from reality. The
structured format and mandatory update step in the prompt template are the
best available mitigation, but this requires ongoing discipline.

---

## 11. What Transfers from BTCDataset_v2

### Direct transfer (proven, port with minor adaptation)

- Triple-barrier labeling logic (labeler.py)
- Walk-forward training with embargo and purging (evaluator.py)
- DSR, SPA (AD-20). CSCV deferred to Phase 5 (AD-23, requires N>=100 configs).
- Kelly sizing with fractional divisor
- Calibration (see below -- default changed from isotonic to Platt scaling)
- ICT feature functions that survived D55 pruning:
  - OTE-705 distance
  - Premium/discount continuous encoding
  - Dual-layer swing structure
- 10-gate validation framework
- D54a/D54c experiment configs (as reference baselines)

### Rebuild from scratch

- Feature registration and parameter passthrough (fixes wiring bug)
- Configuration management (TOML replaces hardcoded constants)
- Data contracts (new, did not exist)
- Holdout ceremony (strengthened from D51)
- Logging (structured replaces print())
- Knowledge base (restructured for CC parsing)

### Equivalence test

Before the foundation is considered viable, it must reproduce D54a results:
- AUC within 0.005 of 0.7933
- WR within 2pp of 76.0%
- Daily Sharpe within 1.0 of 10.71
- 10/10 gates PASS

This is the gate between "new architecture" and "trusted replacement."

---

## 12. Anti-Patterns (What NOT to Do)

Carried forward from BTCDataset_v2 CLAUDE.md "What NOT to Do" section,
plus new structural lessons:

1. **Do NOT hardcode feature parameters.** All params come from TOML config.
   If you find a magic number in a feature function, it is a bug.

2. **Do NOT run experiments without a config file.** The runner will refuse.
   Jupyter notebooks that call engine functions directly bypass all validation.

3. **Do NOT treat Sharpe 12.8 as a live target.** Per-trade annualization
   inflates Sharpe by sqrt(trades_per_year). Per-trade Sharpe x sqrt(N) is the
   meaningful metric. Realistic live target: 1.5-2.5. SR > 3.0 would be
   global top tier. Expect 40-60% haircut from backtest to live.

4. **Do NOT skip causality tests after modifying a feature.** Run
   `pytest tests/test_features/test_causality_all.py -x` before anything else.

5. **Do NOT add features without registering them.** Unregistered features
   cannot be used in experiments and will not be causality-tested.

6. **Do NOT touch holdout data without the ceremony.** The loader will
   raise HoldoutViolationError.

7. **Do NOT re-add d1_ict_market_trend as a mandatory filter.** Confirmed
   dead weight across D16, D21, ablation analysis.

8. **Do NOT use Silver Bullet or Power of 3 as first-class ML features.**
   Both are overfit magnets at 5m resolution. Test at 1m before reconsideration.

9. **Do NOT optimize multiple parameters simultaneously without DSR.** Each
   Optuna trial counts toward the multiple testing correction.

10. **Do NOT commit .env, holdout data, or API keys.** The .gitignore
    handles this, but verify before every push.

11. **Do NOT interpret CAGR figures as live return forecasts.** Backtest
    compounding over hundreds of trades produces astronomic numbers that
    have no relationship to live performance.

12. **Do NOT use symmetric session rules for long and short.** Direction-
    session routing is essential (D17, D25). Shorts need dedicated features.

13. **Do NOT use expanding window with training data starting before Q1 2024
    without explicit regime justification.** The BTC ETF approval in January
    2024 is a confirmed structural break (validated via Chow tests and
    DCC-GARCH analysis). Data from 2020-2023 reflects a different market
    structure. If using expanding window with pre-2024 data, document the
    justification explicitly in the experiment config. A rolling window of
    12-18 months is the safer default for post-2024 predictions.

14. **Do NOT use isotonic calibration as the default with fewer than 500 trades
    per fold.** Isotonic regression requires ~1,000+ samples for reliable
    calibration. With ~90 trades per fold, isotonic is unreliable (Niculescu-Mizil
    & Caruana, ICML 2005). Use Platt scaling (sigmoid, 2 parameters) as default.
    Note: BTCDataset_v2 used isotonic and ECE improved to 0.006 on holdout --
    however ECE 0.0063 is not statistically meaningful at N=177 trades (finite-sample
    noise floor ~0.28 is 44x larger than reported value; Opus review, Futami et al. 2024).
    Foundation default: Platt scaling. Test isotonic if fold trade count exceeds 500.

15. **Do NOT allow training-serving skew.** Feature computation in the research
    pipeline must match the production pipeline exactly. Off-by-one errors,
    different rolling window implementations, or timezone handling differences
    between backtest and live will silently degrade performance. The live system
    does not fail -- it just trades differently than the backtest. Mitigation:
    single feature computation codebase used in both research and production.
    Add integration test: compute features on same data in both pipelines,
    assert identical output.

---

## 13. Sharpe Ratio Policy

BTCDataset_v2 reported per-trade Sharpe of 12.8 using:
`mean(R) / std(R) * sqrt(trades_per_year)`

D55b holdout measured daily Sharpe of 10.71 using:
`mean(daily_R_pnl) / std(daily_R_pnl) * sqrt(252)`

The foundation reports BOTH, clearly labeled:
- **Daily Sharpe (all days):** Annualized from daily P&L including zero days.
  This is the industry standard and the primary metric for evaluation.
- **Per-trade Sharpe:** Annualized from per-trade R-returns. Useful for
  comparing strategies with different frequencies but inflated for sparse
  trading systems.
- **BTC buy-and-hold Sharpe:** Computed over the same period as a benchmark.

The evaluation JSON always includes all three. Gates use daily Sharpe.

### Equivalence Test Cost Requirement

The Phase 4 equivalence test (reproducing D54a results) must include a
realistic cost model before the result is considered valid. Required cost
components:
- Maker vs taker fees (Binance/Bybit specific rates)
- Slippage as a function of position size and time-of-day
- Funding rate payments across 8h intervals
- Bid-ask spread impact on fill quality

A daily Sharpe of 10.71 without cost modeling is NOT a valid equivalence
pass. The post-cost Sharpe is the meaningful number.

**Cost reality (AD-19):** Empirical analysis of D55b holdout shows median stop
distance = 0.142% (14.2 bps, from 1x ATR(14) on 5-min bars). Roundtrip
spread+commission = ~14 bps. At current stop widths, execution costs are ~1.0R
per trade — the same magnitude as the stop itself. EV drops from +1.26R to
~+0.28R at realistic costs. The Foundation must explore wider stops (H1/H4 ATR,
swing-based) to improve the cost/stop ratio. Phase 4 equivalence test targets
must use cost-adjusted metrics.

### Re-Run Variance Warning

W2 Optuna results (D54b) showed that identical parameters with different
LightGBM random seeds produced Sharpe 12.09 vs 21.34 (76% swing) for the
same configuration. This means:
- All historical Sharpe numbers from BTCDataset_v2 are single-seed point
  estimates with ±30-50% variance
- The equivalence test tolerance (±1.0 Sharpe) may be too tight
- Every experiment in the new foundation must run with at least 3 different
  random seeds and report mean ± std, not a point estimate
- The equivalence test passes if the mean result (across 3 seeds) is within
  tolerance, not any single run

### D55b Holdout Reference Values (2026-03-04, validated)

- Daily Sharpe (naive): 10.71
- ~~Daily Sharpe (Lo-adjusted): 14.05~~ (WITHDRAWN -- invalid for sparse-trading systems per external review. Alternating AC from 576-bar cooldown is mechanical artifact, not genuine return property)
- Lo q-factor: 0.5815 -- Note: this adjustment is invalid when autocorrelation is mechanically induced by cooldown structure rather than genuine return dynamics
- Daily Sharpe 10.71 is inflated by ~90% zero-return days suppressing daily volatility. Per-trade Sharpe ~1.03 x sqrt(177) = 13.7 is the comparable annualization. Both exceed all known live benchmarks
- BTC during holdout: -20.6% (BEAR market)

These are the honest baseline numbers for Foundation equivalence testing.
Foundation live target: per-trade Sharpe x sqrt(N) >= 2.0 (revised from daily
Sharpe 5.0). Realistic live Sharpe: 1.5-2.5. SR > 3.0 would place the system
in the global top tier of systematic crypto strategies.

---

## 14. Migration Roadmap

### Phase 0: Project scaffold (this design)
- Create directory structure
- Set up pyproject.toml, .gitignore, .env.example
- Write CLAUDE.md, STATUS.md, KNOWLEDGE.md, DECISIONS.md
- Implement config schema (Pydantic)
- Implement data contracts

### Phase 1: Data foundation
- Port data downloaders from BTCDataset_v2
- Build data processing pipeline (raw -> processed)
- Implement holdout split with guard
- Implement planted signal test (AD-22) as pipeline integrity gate
- Implement label-only screening (AD-27 Stage 0) for label sweep: compute
  median cost_R, expiry fraction, class balance for any label config without
  ML training
- Validate: row counts, schemas, SHA256 hashes match BTCDataset_v2

### Phase 2: Feature framework
- Implement feature protocol and registry
- Port surviving ICT features (OTE-705, P/D, dual-swing)
- Port TA features (RSI, MACD, Stochastic)
- Port regime features (HMM, ADX)
- Run full causality suite

### Phase 3: Training engine
- Port walk-forward (rolling 12-month, 3-month steps, 16 folds)
- 70/15/15 fold split: train / early-stopping val / Platt calibration (AD-21)
- Sample uniqueness weighting from AFML Ch4 (AD-29)
- Dynamic cost model: EDGE spread + latency slippage + funding (AD-31)
- Platt calibration with Lin/Lin/Weng label smoothing on independent held-out set
- Multi-seed ensemble: K=5, average before calibrate (AD-32)
- Embargo derived at runtime from label params (AD-28)
- Horizon auto-computed from diffusion scaling (AD-30)
- SPA (AD-20) as primary multiple-testing correction, DSR complementary
- Promotion gates: binomial test + cost-adjusted EV + per-trade Sharpe (AD-43)
- Brier Score + Spiegelhalter Z for calibration diagnostic (AD-44)
- CSCV deferred to Phase 5 (AD-23)

### Phase 4: Equivalence test + label sweep
- Run D54a-equivalent experiment: AUC within 0.005, 3-seed minimum
- CRITICAL: systematic stop-width exploration (AD-27)
  - Stage 0: label-only screening (400 configs, minutes)
  - Stage 1: cheap ML screening (80 configs, 4 folds, 1-2 seeds, ~7h parallel)
  - Stage 2: full confirmation (15 configs, 16 folds, 3-5 seeds, ~16h parallel)
- Cost-adjusted metrics with realistic cost model (AD-31)
- Horizon auto-computed from diffusion scaling (AD-30)
- Live Sharpe target: 1.5-2.5 (revised from earlier estimates per execution research)
- Phase 4 equivalence uses per-trade Sharpe x sqrt(N) >= 2.0, not daily Sharpe

### Phase 5: Autonomous training
- Three-stage hierarchical optimizer (AD-37):
  - Stage 0: SHAP group screening (600->400-500 features)
  - Stage 1: Boruta/BorutaSHAP (400->100-200 features, 15-25 groups)
  - Stage 2: Optuna TPE on <=35 dimensions (multivariate=True, group=True)
- WilcoxonPruner for walk-forward folds (not MedianPruner)
- Warm-start from proven feature set via enqueue_trial
- Dead-end pruning after 15 trials with SHAP < 0.001 (AD-38)
- CSCV/PBO on N>=100 Optuna configs (AD-23)
- K=5 multi-seed ensemble with hyperparameter perturbation (AD-32)
- Optimizer objective: post-cost EV per calendar day (AD-33)
- Optuna SQLite persistence for cross-session resume (AD-41)
- Long+short dual model support (AD-35, AD-36)
- Per-direction cooldowns, EU decision framework, conflict resolution

### Phase 6: NQ/ES extension
- Add CME data downloader
- Create NQ/ES instrument configs
- Run baseline experiments (out-of-distribution validation)

### Phase 7: Paper trading
- Internal simulator on live WebSocket data (not Binance testnet)
- ccxt for exchange connectivity (AD-31 execution research)
- Minimum 100 trades (~7 months) for preliminary validation
- Sequential 3-month OOS windows, sealed immutably (AD-42)
- Track: prediction-to-execution latency, implementation shortfall, slippage distribution
- Statistical completion: KS test on return distributions, chi-squared on WR, bootstrap CI
- Capital ramp: 25% -> 50% -> 75% -> 100% with statistical triggers (F-33)

### Phase 8: Live deployment
- Server-side STOP_MARKET on Binance for every position (F-30)
- Kill switch: 2x MaxDD shutdown, 1.5x flatten, 1x reduce size
- VPS recommended over Windows desktop for 24/7 operation
- Leverage cap 3-5x regardless of Kelly (AD-39)
- Exchange adapter via ccxt for venue portability
- Telegram alerts for trades, errors, heartbeat
- Model retraining: monthly calendar + event-triggered on drift detection

---

*End of FOUNDATION_DESIGN.md*
