# Research Prompt for Perplexity: Follow-Up Questions (Architecture + Trading)

## Context Update

This is a follow-up to a previous research prompt on trading foundation
architecture. The first round of answers confirmed several architectural
choices (DuckDB + Parquet, Pydantic + TOML, structural enforcement patterns)
but left gaps in specific areas. This prompt asks targeted follow-up questions
where the first round was vague, lacked citations, or where new information
has emerged.

**New developments since the first prompt:**
- Optuna search completed (D54b): 40 trials on threshold/cooldown parameters.
  Best config: threshold=0.70, cooldown=288, producing Sharpe 21.34, WR 80.8%,
  10/10 gates PASS. This is a major improvement over baseline (Sharpe 12.8).
- Re-run variance discovered: same parameters with different LightGBM random
  seed produced Sharpe 12 vs Sharpe 21. This instability is concerning.
- Limit entry types (limit_fvg_edge, limit_ob_mid) in Optuna search produced
  degenerate results: WR=100%, Sharpe overflow. The backtest fill model
  (assumes fill if price touches limit) creates survivorship bias.
- BTC-equity correlation confirmed unstable: positive post-ETF (Jan 2024),
  then dropped to -0.299 with S&P 500 by late 2025.
- Coinbase launched CFTC-regulated BTC perpetual futures in July 2025.

**System recap** (unchanged): BTC 5m perpetual futures, LightGBM walk-forward
(9 folds, 288-bar embargo, label purging), 64 features after pruning from 670,
triple-barrier labels (1R stop / 2R target / 48-bar horizon), solo trader on
Windows laptop, Claude Code as primary developer.

---

## Question 1: Citation Chase -- Papers Referenced Without Full Citations

The first research round mentioned several findings without proper citations.
I need the actual papers to verify claims.

**Specific requests:**
- A study analyzing optimal rolling window sizes across "1,455 assets" that
  found "600-800 day clusters" as optimal. What is this paper? (Author, title,
  year, journal.) If no such single paper exists, what is the actual source
  for optimal window size recommendations?
- The DCC-GARCH analysis confirming BTC-equity structural break at ETF
  approval (January 2024). Cite the specific paper(s) using Chow tests and/or
  DCC-GARCH on BTC post-ETF correlation structure.
- "Look-Ahead-Bench" (described as January 2026): A standardized benchmark
  for measuring lookahead bias in financial LLMs using alpha decay across
  temporal market regimes. Full citation, methodology summary, and whether
  this is applicable to feature-level causality testing (not just LLM evaluation).
- "QuantEvolve" (described as 2025): A multi-agent evolutionary framework
  for automated strategy discovery. Full citation, whether code is open-source,
  and whether it has been validated on crypto or only equities/futures.
- Lopez de Prado's "10 Reasons Most Machine Learning Funds Fail" -- is this
  a published paper or a presentation? Full citation with venue.

---

## Question 2: Sharpe Ratio Computation for Infrequent Traders

My system trades 178 times per year (~1 trade every 2 days). I computed
daily Sharpe as: `mean(daily_pnl) / std(daily_pnl) * sqrt(252)` and got
8.77. The first research round flagged this as "implausibly high" and
equivalent to "annualized Sharpe above 100."

**The specific issue:** With ~178 trades/year and a 48h cooldown, approximately
50% of trading days have zero P&L (no open position, no trade). These zero
days pull std(daily_pnl) down relative to mean(daily_pnl), inflating the
ratio.

**Research questions:**
- Is this a documented statistical artifact? Is there a name for this
  inflation effect when computing Sharpe on a series with many exact zeros?
- What is the CORRECT way to compute and report daily Sharpe for a strategy
  that trades every 2 days on average? Should zero-P&L days be:
  (a) included as zeros, (b) excluded entirely, (c) treated differently?
- Lo (2002, "The Statistics of Sharpe Ratios") discusses autocorrelation
  adjustment. Does Lo's framework address the zero-return-day problem, or
  is this a separate issue?
- How do hedge funds and fund-of-funds report Sharpe for strategies with
  low trade frequency (100-300 trades/year)? Is per-trade Sharpe with
  sqrt(N_trades) annualization actually more appropriate than daily Sharpe
  for this frequency?
- What is the mathematical relationship between per-trade Sharpe and daily
  Sharpe for a system with known trade frequency and average holding period?
  Derive or cite the formula.

---

## Question 3: LightGBM Re-Run Variance and Seed Sensitivity

Running the same LightGBM walk-forward pipeline with identical parameters
but different random seeds produced dramatically different results:
Sharpe 12.8 (seed A) vs Sharpe 21.34 (seed B). Same data, same features,
same walk-forward folds, same everything except the random seed.

**Research questions:**
- What causes this variance in LightGBM specifically? Is it from:
  (a) feature subsampling (colsample_bytree), (b) data subsampling
  (bagging_fraction), (c) random tie-breaking in splits, (d) something else?
- What is the expected variance of Sharpe ratio estimates from LightGBM
  random seed alone, for a dataset with ~540K rows and 64 features?
  Are there published benchmarks?
- How do practitioners handle seed sensitivity in financial ML?
  Options: (a) average metrics over K seeds, (b) use the median seed,
  (c) report the worst seed, (d) fix the seed and accept path dependence.
  What is the consensus?
- Does seed sensitivity indicate overfitting? If a model's Sharpe ranges
  from 12 to 21 across seeds, is the "true" Sharpe closer to 12 (lower bound),
  17 (mean), or unknowable?
- Are there LightGBM settings that REDUCE seed sensitivity without hurting
  OOS performance? (e.g., increasing n_estimators, reducing learning_rate,
  increasing min_child_samples, using deterministic mode?)

---

## Question 4: Optimal Walk-Forward Window Size for Intraday Financial Data

The first research round cited optimal windows of "600-800 days" for daily
data, but my system uses 5-minute bars (~105,000 bars/year). The current
setup uses expanding windows (all data from 2020 to fold boundary).

**Research questions:**
- Is there published research on optimal walk-forward window sizes
  specifically for INTRADAY (1-min, 5-min, 15-min) financial data?
  Not daily or monthly -- intraday.
- For BTC specifically: given the confirmed structural break at ETF
  approval (Jan 2024), what is the empirical evidence for how far back
  training data should extend? Has anyone tested rolling vs expanding
  windows on BTC 5-min data with pre/post-ETF splits?
- Is there a principled method for choosing window size, beyond
  cross-validation? (e.g., half-life of autocorrelation, regime detection,
  information ratio decay curves)
- Sample size requirements: with 64 features and LightGBM, what is the
  minimum number of training bars needed for reliable walk-forward results?
  Is there a features-to-samples ratio guideline for gradient boosting
  on financial time series?
- For a system that has already confirmed a structural break: what is the
  recommended approach? (a) Drop all pre-break data. (b) Use pre-break
  data but down-weight. (c) Add a regime feature. (d) Run separate
  models for each regime. Published evidence for each approach?

---

## Question 5: Fill Model Bias in Limit Order Backtesting

Our Optuna search included limit entry types (enter at FVG edge price,
enter at order block midpoint). These produced WR=100% and Sharpe overflow
in backtest. The fill model assumes: if price reaches the limit price at
any point during the bar, the order is filled at the limit price.

This creates survivorship bias: only cases where price reached the limit
AND subsequently hit the 2R target are counted as fills, because if price
reached the limit but reversed, the trade hits the stop -- but the fill
model doesn't account for queue position or partial fills.

**Research questions:**
- Is this specific bias documented in academic or practitioner literature?
  What is the standard name for it? (I've seen "fill-or-kill bias",
  "touch-means-fill bias", "limit order survivorship bias.")
- What is the state of the art for backtesting limit orders realistically?
  Published models that account for: queue position, partial fills,
  adverse selection (price reaching your limit often means it's about
  to reverse), and time-priority.
- Are there published estimates of the backtest-to-live degradation ratio
  specifically for limit order strategies? (Not market order strategies.)
  How much worse do limit strategies perform live vs backtest?
- For a system that primarily uses market orders but wants to TEST limit
  entries: what fill probability model should be used? Is there a simple
  heuristic (e.g., "assume 50% fill rate when price touches") that
  is empirically grounded?
- Avellaneda & Stoikov (2008) and subsequent market-making literature
  model adverse selection in limit orders. Is this framework applicable
  to a directional strategy using limit entries (not market-making)?

---

## Question 6: mlfinlab and AFML Implementation Libraries (2025-2026)

The first research round mentioned Hudson & Thames' mlfinlab as the closest
tool for enforcing financial ML research discipline.

**Research questions:**
- What is the current state (2025-2026) of mlfinlab? Is it actively
  maintained? What version? What is the license (it was previously
  restrictive -- has this changed)?
- Does mlfinlab implement: CSCV, DSR, purged k-fold, triple-barrier
  labels, fractional differentiation, meta-labeling? Which of these
  work correctly and which have known issues?
- Are there ALTERNATIVES to mlfinlab that implement AFML concepts?
  Check: vectorbt, skfolio, qstrader, finml-utils, or any new libraries
  published 2024-2026.
- Specifically: is there an open-source, maintained Python implementation
  of the Deflated Sharpe Ratio (DSR) that I can use directly? One that
  correctly handles non-normal returns, autocorrelation, and multiple
  testing correction?
- For CSCV specifically: is there an implementation that handles
  non-uniform fold sizes (my walk-forward folds grow with expanding
  window, so early folds are smaller)?

---

## Question 7: US Regulatory Framework for Automated BTC + NQ/ES Trading

Coinbase launched CFTC-regulated BTC perpetual futures in July 2025.
I plan to trade BTC perpetuals + NQ + ES futures from a personal US account.

**Research questions:**
- Coinbase Financial Markets BTC perpetuals: What are the contract specs?
  (tick size, minimum order, maximum leverage for retail, trading hours,
  margin requirements, funding rate mechanism vs offshore perps)
- Can a US individual legally run a fully automated trading system on
  both BTC perpetuals (Coinbase) and equity index futures (CME via
  Interactive Brokers) from a personal account? Any restrictions on
  API trading, order frequency, or automation?
- Tax classification: Do CFTC-regulated BTC perpetual futures qualify
  for Section 1256 (60/40 tax treatment) like NQ/ES futures? Or are
  they taxed as regular crypto (capital gains)?
- Are there reporting requirements (CFTC Large Trader Reporting, etc.)
  that apply to a retail trader at the $10K-$100K position size?
- What is the counterparty risk difference between Coinbase perpetuals
  (CFTC-regulated, US entity) and offshore exchanges (Bybit, Binance)?
  Published analysis of exchange risk, not opinions.

---

## Question 8: BTC Correlation Instability and Cross-Asset Implications

The first research round confirmed that BTC-equity correlation is unstable:
positive post-ETF (Jan 2024), then dropped to -0.299 with S&P 500 by
late 2025. This has direct implications for using NQ/ES as OOD validation.

**Research questions:**
- What drives BTC-equity correlation regime shifts? Published research on
  the mechanisms (institutional flows, macro regime, risk appetite, BTC
  supply dynamics). Cite specific papers using DCC-GARCH or similar
  time-varying correlation models on BTC.
- If BTC-NQ correlation was ~0.5 in 2024 but ~-0.3 in 2025, does
  cross-instrument validation (train on BTC, test on NQ) have any
  statistical meaning? Or does the correlation instability make it
  meaningless?
- Are there published frameworks for cross-asset strategy validation
  that account for time-varying correlation? Not just "test on another
  asset" but a principled methodology.
- What is the current (late 2025 / early 2026) consensus on whether
  BTC is a "risk-on tech proxy" or a "digital gold" uncorrelated asset?
  Has the narrative shifted since the ETF approval?
- For the specific features that survive in my system (momentum, gaps,
  pivots): are these features that would be expected to transfer across
  correlated assets, or are they instrument-specific? Published evidence
  on feature transferability across asset classes.

---

## Instructions for Perplexity

For each question:
1. Provide specific citations (author, title, year, journal/source)
2. Distinguish between peer-reviewed research and practitioner opinions
3. If the answer is "no evidence exists," say so clearly
4. Prioritize 2024-2026 sources
5. If a claim from the first research round cannot be verified, flag it
6. For quantitative claims, provide source and sample size

IMPORTANT: Several answers in the first round cited findings without
proper citations (the "1,455 assets" study, the DCC-GARCH BTC analysis,
QuantEvolve, Look-Ahead-Bench). If you referenced these in a previous
response, please provide the ACTUAL paper citations now, or acknowledge
that no such paper exists and the claim was synthesized.

---

*End of PERPLEXITY_FOLLOWUP.md*
