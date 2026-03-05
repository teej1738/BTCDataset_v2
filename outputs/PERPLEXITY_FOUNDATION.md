# Research Prompt for Perplexity: Trading Foundation Architecture Validation

## Context for Perplexity

I'm designing a production-grade algorithmic trading research foundation
from first principles. The system will be built primarily by Claude Code
(an AI coding assistant with zero session memory) for an individual trader
on a personal laptop (Windows, Python, <$50K account).

The foundation must support:
- Walk-forward ML on BTC 5-minute bars (initial), extending to NQ/ES futures
- Feature engineering with mandatory causality testing
- Automated experiment lifecycle (propose -> train -> evaluate -> record)
- Structural enforcement preventing "vibe coding" (no validation shortcuts)

I have a frozen prototype (BTCDataset_v2: AUC 0.7933, WR 76%, daily Sharpe
8.77, 10/10 validation gates PASS) that informs the design. The new system
must reproduce these results through better architecture before expanding.

Below are specific technical questions where I need current research and
practitioner evidence. Please answer with citations where possible.

---

## Question 1: DuckDB for Financial Time-Series Analytics

I plan to use Parquet files as the storage format with DuckDB as an optional
analytical layer (reading parquet directly, no ETL). The dataset is ~650K
rows x 600 columns of 5-minute BTC OHLCV data plus derived features.

**Research questions:**
- What is the current (2024-2026) state of DuckDB adoption in quantitative
  finance? Are hedge funds or proprietary trading firms using it?
- DuckDB vs Polars vs pandas for financial time-series workloads at the
  100K-1M row scale. What are the performance benchmarks?
- Are there published architectures combining DuckDB with Parquet for
  ML feature stores? How do they handle feature versioning?
- What are the limitations of DuckDB for real-time / streaming data
  (relevant when the system moves to live trading)?
- Are there alternatives to DuckDB that are better suited for a single-user
  financial research system on Windows?

## Question 2: Pydantic for ML Pipeline Configuration

I plan to use TOML config files validated by Pydantic v2 models. Every
experiment is fully defined by its TOML file (model params, feature params,
walk-forward settings, cost model, gates).

**Research questions:**
- What are best practices for ML experiment configuration management
  as of 2025-2026? How do tools like MLflow, Weights & Biases, and
  Hydra handle config validation?
- Is Pydantic v2 the right choice for config validation in ML pipelines?
  Are there performance concerns at scale (hundreds of Optuna trials)?
- How do quantitative finance firms manage experiment reproducibility?
  Is config-as-code (committing configs to git) the standard?
- TOML vs YAML vs JSON for ML experiment configs. What does the Python
  ML ecosystem prefer? Are there strong arguments for one format?
- Are there published examples of Pydantic-based ML pipeline configurations?
  What patterns emerge?

## Question 3: Feature Store Patterns for Solo Researchers

I'm designing a feature registration system where each feature family
implements a protocol (interface) with compute(), causality_test(), and
output_columns. Features are registered via decorator and automatically
tested for lookahead bias.

**Research questions:**
- What feature store patterns exist for small-scale financial ML research
  (not Feast, Tecton, or enterprise solutions -- single user, local)?
- How do academic quantitative finance researchers manage feature pipelines?
  Are there published frameworks or conventions?
- What is the state of the art for automated lookahead bias detection in
  feature engineering? Beyond manual df[:T] vs df[:T+1] comparison.
- Are there Python libraries specifically for financial feature engineering
  with built-in causality guarantees? (e.g., tsfresh, featuretools)
- How do practitioners handle feature versioning when feature logic changes?
  Is recomputation from scratch the standard, or is there incremental
  computation that maintains consistency?

## Question 4: Walk-Forward Validation Best Practices (2024-2026)

My system uses expanding-window walk-forward with 9 folds, 288-bar embargo
(24h at 5-min bars), and AFML Ch.7 label purging. I'm considering adding
rolling windows.

**Research questions:**
- What is the current consensus (2024-2026) on expanding vs rolling window
  for financial ML walk-forward validation? Has the recommendation changed
  since Marcos Lopez de Prado's AFML (2018)?
- For BTC specifically: given the ETF approval in January 2024 and the
  potential structural regime change, is expanding window still appropriate?
  Should training data from 2020-2022 be used to predict 2025 behavior?
- What embargo length is recommended for 5-minute bar data? My 288 bars
  (24 hours) was chosen to exceed the label horizon (48 bars = 4 hours).
  Is this overly conservative?
- Are there published benchmarks for minimum trades per fold for
  statistically reliable walk-forward results at the 5-minute timeframe?
- What is the state of the art for fold selection in financial ML?
  Are there methods beyond fixed-size expanding/rolling windows?

## Question 5: Structural Enforcement in ML Research Workflows

I want "vibe coding" to be structurally impossible. My approach is:
- Feature registration with mandatory causality test
- Experiment runner that requires validated TOML config (no defaults)
- Holdout data loader that raises exceptions if accessed without ceremony
- Statistical gates that are always computed (cannot be skipped)
- Parameter passthrough enforced by Pydantic (fixing a real bug in my prototype)

**Research questions:**
- How do professional quantitative research teams enforce research discipline?
  Are there published frameworks for preventing common backtesting mistakes?
- Is there academic literature on "defensive ML engineering" -- designing
  ML systems that prevent the researcher from fooling themselves?
- What are the most common ways backtesting validation gets bypassed
  in practice, even in disciplined teams? (Published post-mortems welcome)
- Are there Python tools or frameworks designed specifically for enforcing
  financial ML research discipline? (Beyond generic ML tools like MLflow)
- How do firms handle the "holdout contamination" problem? Is a one-time
  ceremony the standard, or are there more sophisticated approaches?

## Question 6: AI-Assisted Development Knowledge Management

My primary developer (Claude Code) has zero session memory. Every session
starts fresh. Context must be reconstructed from files. I'm designing a
layered knowledge system: CLAUDE.md (project context), STATUS.md (current
task), KNOWLEDGE.md (evidence base), DECISIONS.md (architectural log).

**Research questions:**
- Are there published patterns for managing AI-assisted development where
  the AI agent has no persistent memory? How do teams handle this?
- What documentation structures work best for providing context to LLM
  coding assistants? Is there research on optimal document formats?
- How do teams maintain knowledge bases that are both human-readable and
  machine-parseable? Are structured tables better than prose for LLM context?
- What is the maximum effective context size for an LLM coding assistant?
  At what point does more context degrade performance?
- Are there tools or frameworks designed for "AI-developer-first"
  project management? How is this space evolving in 2025-2026?

## Question 7: Multi-Asset Extension (BTC -> NQ/ES)

My system currently trades BTC perpetual futures. I plan to extend to
NQ1! (Nasdaq) and ES1! (S&P 500) futures as out-of-distribution validation.

**Research questions:**
- What is the current (2025) correlation between BTC and NQ/ES? Is BTC
  still correlated enough to NQ that NQ extension is meaningful?
- How does feature engineering differ between crypto perpetuals (24/7, funding
  rate) and equity index futures (limited hours, no funding)? What
  adaptations are needed?
- What is the standard data source and resolution for systematic NQ/ES
  trading? Is 5-minute appropriate, or do most use 1-minute or tick data?
- Are there published examples of trading systems that successfully
  transferred from crypto to equity futures (or vice versa)?
- What are the execution characteristics of NQ/ES vs BTC perpetuals
  for a retail trader ($10K-$100K position sizes)?

## Question 8: Autonomous ML Experiment Management

I want the system to support autonomous experiment loops: the optimizer
proposes experiments, runs them, evaluates results, and updates the knowledge
base. The human reviews periodically but does not need to approve each trial.

**Research questions:**
- How do Optuna's TPE sampler and MedianPruner perform for financial ML
  optimization? Are there published benchmarks specific to time-series?
- What is the correct way to count trials for multiple testing correction
  (DSR) when using Optuna? Does each trial count, or does the structured
  search reduce the effective N?
- Are there published frameworks for autonomous ML research loops
  (propose -> train -> evaluate -> learn -> propose again)?
- How do firms handle the exploration-exploitation tradeoff in automated
  strategy research? When should the system explore new feature families
  vs refine known good features?
- What monitoring metrics detect when an automated search is overfitting
  the search space itself (meta-overfitting)?

## Question 9: Position Sizing for Uncertain Edge

My system uses Kelly fraction with 1/40 divisor, clipped [1%, 2%]. The
edge estimate comes from backtesting only -- no live validation yet.

**Research questions:**
- What does recent (2023-2026) literature recommend for position sizing
  when transitioning from backtest to live? Is there a consensus ramp-up
  schedule?
- For a system with estimated daily Sharpe 8.77 (likely inflated by
  backtest conditions), what position sizing would a practitioner
  recommend for the first 6 months of live trading?
- Are there adaptive position sizing methods that start conservative and
  increase as live performance validates the backtest estimates?
- What is the relationship between daily Sharpe and probability of ruin
  for different position sizing approaches?

## Question 10: What Am I Missing?

Given the system described above:
- Single individual, Claude Code as primary developer, Windows laptop
- BTC 5m perpetual futures with LightGBM walk-forward ML
- ICT-derived features (displacement, FVG, swing structure, OTE)
- TOML configs + Pydantic + causality testing + 10 statistical gates
- Extending to NQ/ES futures as OOD validation

**Research questions:**
- What are the most common failure modes for solo quant traders who build
  their own ML systems? Published accounts welcome.
- What validation step am I most likely to have missed?
- Are there regulatory or legal considerations for running an automated
  BTC + NQ/ES trading system on a personal account (US-based)?
- What operational risks (beyond market risk) should I plan for?
  (Exchange outages, API changes, data corruption, etc.)
- What would a professional quant's first concern be upon reviewing
  this architecture?

---

## Instructions for Perplexity

For each question:
1. Provide specific citations (author, title, year, journal/source)
2. Distinguish between peer-reviewed research and practitioner opinions
3. If the answer is "no evidence exists," say so clearly
4. Prioritize recent sources (2024-2026) over older ones
5. If you find contradictory evidence, present both sides
6. For quantitative claims, provide the source and sample size

---

*End of PERPLEXITY_FOUNDATION.md*
