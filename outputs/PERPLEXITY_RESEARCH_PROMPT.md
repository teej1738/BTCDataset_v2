# Research Prompt for Perplexity: BTC Algorithmic Trading System Validation

## Context for Perplexity

I'm building an algorithmic trading system for BTC perpetual futures using LightGBM walk-forward ML with features derived from ICT (Inner Circle Trader) methodology. The system shows AUC 0.79, WR 74.9%, and passes 10 independent validation gates (CSCV PBO, PSR, etc.). However, I have specific technical questions where I need current research and practitioner evidence to validate my approach. Please answer each question with citations to academic papers, practitioner blogs, or quantitative finance resources.

---

## Question 1: Sharpe Ratio Annualization from Intraday Bars

My system trades on 5-minute bars (105,120 bars/year) and reports Sharpe 12.3 using per-bar annualization: `mean(per_bar_returns) / std(per_bar_returns) * sqrt(105120)`.

**Research questions:**
- What is the correct method for annualizing Sharpe ratios for intraday trading systems that trade infrequently (178 trades/year on 5-minute bars)?
- Should Sharpe be computed on per-trade returns, daily P&L, or per-bar returns? What do Lo (2002) and Bailey & Lopez de Prado recommend?
- What adjustment is needed for autocorrelation in intraday returns? (Reference: Andrew Lo, "The Statistics of Sharpe Ratios," Financial Analysts Journal, 2002)
- What Sharpe ratio range is typical for profitable systematic crypto trading strategies in live deployment (not backtest)?
- Are there documented cases of Sharpe inflation from high-frequency annualization, and what correction factors are used in practice?

## Question 2: Deflated Sharpe Ratio - Practical Implementation

I've run 55+ experiments on the same BTC 2020-2026 dataset, including 8 contaminated early experiments and hundreds of Optuna hyperparameter trials. I need to apply the Deflated Sharpe Ratio (DSR) from Bailey & Lopez de Prado.

**Research questions:**
- How should N_trials be counted when using Optuna (TPE sampler) for hyperparameter optimization? Does each Optuna trial count as a separate trial, or does the structured search reduce the effective N?
- Bailey & Lopez de Prado (2014) "The Deflated Sharpe Ratio" - what is the practical threshold for DSR significance? Is DSR > 2.0 sufficient, or do practitioners use different cutoffs?
- When using walk-forward validation with expanding windows, does each fold count as a separate test, or is the walk-forward considered a single test?
- Are there more recent alternatives to DSR for multiple testing correction in quantitative finance (published 2022-2026)?
- How do firms like Two Sigma, AQR, or Citadel handle multiple testing in strategy development? Any public information on their practices?

## Question 3: ICT (Inner Circle Trader) Methodology - Evidence Base

My system uses ICT concepts: Fair Value Gaps (FVGs), Order Blocks, liquidity sweeps, market structure (BOS/CHoCH/MSS), premium/discount zones, and displacement. SHAP analysis shows these contribute 72.3% of model signal, but aggressive pruning eliminated 7/10 ICT feature families.

**Research questions:**
- Is there ANY peer-reviewed academic research validating ICT trading concepts (Fair Value Gaps, Order Blocks, liquidity sweeps, smart money concepts)?
- Are there quantitative studies measuring the predictive power of price gaps (the statistical equivalent of FVGs) in crypto or equity markets? What do they find?
- The concept of "liquidity sweeps" (price briefly piercing a previous high/low before reversing) - is there evidence this pattern has predictive value? Is this related to stop-hunting research?
- Has anyone published backtested results of systematic ICT-based strategies? What were the results?
- What does the academic literature say about "order blocks" and "institutional order flow" concepts that ICT teaches? Is there any connection to real institutional market microstructure?
- Are there structural reasons why price gaps (FVGs) might have predictive power in BTC perpetual futures specifically (e.g., related to funding rates, liquidation cascades, market maker behavior)?

## Question 4: LightGBM Walk-Forward Validation Best Practices

My system uses expanding-window walk-forward with 9 folds, 288-bar embargo (24h at 5-min bars), label purging (AFML Ch.7), and CSCV (Combinatorial Symmetric Cross-Validation) with PBO < 0.05.

**Research questions:**
- In walk-forward validation for financial ML, is expanding window or sliding window preferred? What does recent research (2023-2026) recommend?
- What is the minimum recommended number of trades per fold for statistically reliable walk-forward results? Is 100 trades/fold sufficient?
- CSCV with C(8,4)=70 combinations and PBO threshold < 0.05 - is this sufficient for financial ML validation? Are there known limitations of CSCV?
- Label purging (removing training samples whose labels depend on test period data) - are there cases where standard purging is insufficient? What about information leakage through feature construction?
- What is the state-of-the-art for temporal cross-validation in financial ML as of 2025-2026? Any significant advances beyond AFML methodology?
- Are there published benchmarks for what AUC/accuracy is achievable with gradient boosting on financial time series? Is AUC 0.79 on 5-minute BTC bars unusually high?

## Question 5: Crypto Perpetual Futures - Execution Reality

My backtest assumes entry at signal bar close, fixed slippage, and historical funding rates. I need to understand execution reality.

**Research questions:**
- What is typical slippage for BTC perpetual futures on Bybit/Binance for orders of $10K-$100K notional? Does this vary significantly by time of day or volatility regime?
- How much does execution latency matter for a system that trades 178 times per year on 5-minute signals? Is sub-second execution necessary?
- Funding rate dynamics: how predictable are funding rates? Can a backtest accurately model funding costs using historical rates, or is there a significant live-vs-backtest gap?
- What percentage of backtested alpha is typically lost to execution costs in live crypto trading? Are there published studies or practitioner accounts?
- Market impact: at what position size does market impact become significant for BTC perpetual futures? ($100K? $1M? $10M?)
- Are there known systematic biases in crypto exchange data that could affect backtesting (e.g., exchange-specific price spikes, data gaps, survivorship bias in available trading pairs)?

## Question 6: Feature Lookahead Bias Detection

I have 64 features computed from 5-minute OHLCV data including multi-timeframe resampling (H4, H1, M15) and rolling computations. I need to verify no lookahead bias exists.

**Research questions:**
- What are the most common sources of lookahead bias in financial ML feature engineering? Provide a comprehensive checklist.
- When resampling from 5-minute to H4/H1/M15 bars, what is the correct way to avoid including the current incomplete bar?
- For rolling window features (e.g., rolling mean, rolling std), does pandas `rolling()` with default settings introduce any lookahead? What about `ewm()`?
- Is there a systematic method or tool for automatically detecting lookahead bias in feature pipelines? Any open-source implementations?
- What about more subtle forms of leakage: using the same data for feature selection (SHAP) and model evaluation? How should this be handled?
- The "triple peeking" problem (using test data for feature engineering, model selection, AND evaluation) - how does this manifest in walk-forward setups?

## Question 7: Position Sizing for Uncertain Edge

My system uses Kelly fraction with 1/40 divisor (fractional Kelly), clipped [1%, 2%]. The edge estimate comes from backtesting only - no live validation yet.

**Research questions:**
- What does the academic literature recommend for position sizing when edge magnitude is uncertain? Is fractional Kelly (1/2 Kelly or less) sufficient, or are there better approaches?
- Kelly criterion assumes known win rate and payoff ratio. In practice, these are estimated with error. How does estimation error affect optimal Kelly fraction? (Reference: Ed Thorp's work)
- For a system transitioning from backtest to live, what position sizing ramp-up schedule do practitioners recommend?
- What is the probability of ruin for a system with estimated Sharpe 2.0 using Kelly 1/40 vs Kelly 1/20 vs fixed 0.25% risk?
- Are there adaptive position sizing methods that adjust based on recent live performance (not backtest)? How do they compare to fixed fractional Kelly?
- What position sizing approach do systematic crypto funds use? Any public information from firms like Alameda (pre-collapse), Jump Crypto, or Wintermute?

## Question 8: BTC Market Regime and Stationarity

My training data spans 2020-2026, covering multiple BTC market regimes (COVID crash, 2021 bull run, 2022 bear market, 2024-2025 ETF-driven rally). The system is long-only.

**Research questions:**
- How stationary is BTC price behavior across regimes? Is there evidence that patterns from 2020-2022 persist in 2024-2026?
- Has BTC market microstructure changed significantly since ETF approval (January 2024)? Are pre-ETF patterns still relevant?
- For a long-only BTC system, how much of backtested performance is explained by the secular bull trend? What methods separate alpha from beta in crypto?
- Hidden Markov Models for regime detection in crypto - do they work in practice? What are the failure modes? (My system uses a 3-state Gaussian HMM)
- What is the typical "half-life" of a quantitative trading strategy in crypto? How quickly do edges decay?
- Are there known seasonal or cyclical patterns in BTC that could explain backtested performance without implying a generalizable edge (e.g., halving cycles, quarterly funding patterns)?

## Question 9: Multi-Instrument Extension (BTC → NQ → ES)

I plan to extend the system from BTC to NQ1! (Nasdaq futures) and ES1! (S&P futures) as out-of-distribution validation and diversification.

**Research questions:**
- What is the typical correlation between BTC and NQ/ES? Has this changed over time (2020 vs 2025)?
- Can features designed for crypto (24/7 market, funding rates, etc.) transfer to equity index futures (limited hours, no funding)? What adaptations are needed?
- ICT methodology was originally taught for forex and index futures. Are there backtested results of ICT on NQ/ES?
- What data resolution is standard for systematic NQ/ES trading? Is 5-minute bars appropriate, or do most systematic strategies use different timeframes?
- What are the execution characteristics of NQ/ES vs BTC perps (slippage, market impact, session effects)?
- For portfolio construction across BTC + NQ + ES: what correlation-adjusted position sizing methods work best? Is there research on systematic cross-asset portfolio construction combining crypto and equity futures?

## Question 10: What Am I Missing?

Given the system described above (LightGBM walk-forward on BTC 5-min bars with ICT features, triple-barrier labels, isotonic calibration, 10 validation gates):

**Research questions:**
- What are the most common failure modes for ML-based trading systems that pass backtesting validation but fail in live deployment?
- Are there published post-mortems of ML trading systems that looked good in backtest but failed live? What were the root causes?
- What validation tests am I NOT running that I should be? (Beyond walk-forward, CSCV, PBO, PSR, DSR, ECE)
- Is there a standard "deployment readiness" checklist for ML trading systems in the quantitative finance literature?
- What monitoring metrics should I track in live trading to detect model degradation early?
- Are there regulatory or legal considerations for running an automated BTC trading system (US-based, personal account)?

---

## Instructions for Perplexity

For each question:
1. Provide specific citations (author, title, year, journal/source) where possible
2. Distinguish between peer-reviewed research and practitioner blog posts/opinions
3. If the answer is "no evidence exists," say so clearly rather than speculating
4. Prioritize recent sources (2022-2026) over older ones when both exist
5. If you find contradictory evidence, present both sides
6. For quantitative claims (e.g., "typical Sharpe for live crypto strategies is X"), provide the source and sample size
