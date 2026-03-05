# Research Prompt for Perplexity: BTC Algorithmic Trading System -- Deep Validation (V2)

## System Description (Read This First -- All Questions Reference This Context)

I am building an algorithmic trading system for BTC perpetual futures (with planned extension to NQ and ES equity index futures). Here are the specifics:

**Data:** 648,288 rows of 5-minute BTCUSDT perpetual data from 2020-01-01 to 2026-02-28. Train set: 543,167 rows (2020-01 to 2025-02). Holdout: 105,121 rows (2025-03 to 2026-02, untouched).

**Model:** LightGBM gradient boosting with walk-forward validation (expanding window, 9 folds, 288-bar embargo between train/val and val/test, label purging per AFML Ch.7).

**Features:** 64 features surviving aggressive pruning (from 670). The signal is primarily: momentum displacement (price move > 1.5× ATR with directional close), price gaps (Fair Value Gaps from ICT methodology), and dual-layer swing structure (pivot points at 5-bar and 10-bar windows). Originally framed as "ICT methodology," but 7/10 ICT feature families were pruned with zero AUC loss -- the surviving signal is essentially momentum + gap + pivots.

**Labels:** Triple-barrier: 1R ATR stop / 2R ATR target / 48-bar (4-hour) horizon. Primary label: "did this long setup hit 2R profit before 1R loss within 48 bars?"

**Reported metrics:** AUC 0.79, Win Rate 74.9%, Expected Value +1.20R per trade, 178 trades/year, MaxDD 7.2%. Sharpe reported as 12.3 (per-trade annualized via √trades_per_year, NOT daily).

**Validation:** 10-gate framework (AUC, CSCV PBO, PSR, Sharpe, WR, EV, MaxDD, ECE, trade count, fold consistency). CSCV C(8,4)=70 combinations, PBO = 0%. PSR > 0.99.

**Position sizing:** Kelly criterion with 1/40 divisor, clipped [1%, 2%] risk per trade. Long-only (shorts viable at AUC 0.7966 but not yet deployed).

**Known issues I've identified:**
1. Daily Sharpe has never been computed (Sharpe 12.3 is per-trade annualized, not industry-standard daily)
2. No buy-and-hold comparison (alpha vs beta unmeasured for a long-only BTC system during 2020-2026)
3. 28+ experiments on the same dataset without formal DSR correction on the final model
4. Optuna parameter search space for ICT signal parameters is defined but never actually wired to the feature functions (they run on hardcoded defaults)
5. No out-of-distribution testing on other instruments
6. Holdout period (2025-03 to 2026-02) is a BTC bull market -- weak validation for a long-only system

**For each question below, I need:** specific citations (author, title, year, source), distinction between peer-reviewed and practitioner sources, honest "no evidence exists" when applicable, priority to 2022-2026 sources, and both sides of contradictory evidence.

---

## Domain 1: Sharpe Ratio Computation -- Per-Trade vs. Daily Annualization

My system computes Sharpe as: `mean(per_trade_returns) / std(per_trade_returns) * sqrt(trades_per_year)`. With 178 trades/year and per-trade SR ≈ 0.92, this gives Sharpe 12.3. I have never computed the daily Sharpe (daily P&L → annualize via √252).

1. **What is the industry-standard method for computing and reporting Sharpe ratios for systematic trading strategies?** Specifically: should I use per-trade returns, daily P&L, or per-bar returns as the base unit? What do Lo (2002, "The Statistics of Sharpe Ratios"), Bailey & Lopez de Prado, and standard quantitative finance textbooks recommend?

2. **What happens to Sharpe when you have many zero-return days?** My system trades 178 times per year on 5-minute bars. Many calendar days will have zero trades (and thus zero P&L). How does this affect daily Sharpe computation? Should zero-P&L days be included or excluded? What is the mathematical relationship between per-trade Sharpe and daily Sharpe for an infrequently trading system?

3. **What is the typical range of daily-annualized Sharpe ratios for profitable systematic crypto strategies in actual live deployment (not backtest)?** Provide source, sample size, and whether the data is self-reported. Is there a published survey or database of live crypto fund Sharpes?

4. **Are there documented cases of Sharpe inflation from using non-standard annualization methods?** Specifically, cases where per-trade or per-bar Sharpe was dramatically higher than daily Sharpe for the same strategy? What correction factors exist?

5. **How should Sharpe ratios be adjusted for autocorrelation in returns?** My system has a 576-bar (48-hour) cooldown between trades, which may introduce autocorrelation structure. What is the Lo (2002) autocorrelation adjustment, and when is it material?

---

## Domain 2: Deflated Sharpe Ratio and Multiple Testing in Strategy Development

I have run 55+ documented decisions and at least 28 distinct experiments on the same BTC 2020-2026 dataset, plus planned Optuna hyperparameter optimization (potentially 500+ trials). The DSR framework was scaffolded but never applied to the final model.

1. **Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio" -- what is the practical implementation?** Specifically: how should N_trials be counted? Does each Optuna trial (TPE sampler) count as a separate trial, or does the structured search reduce effective N? Does each walk-forward fold count separately?

2. **With N=28 experiments, what is the expected maximum Sharpe from pure noise?** I've seen the approximation E[max(SR)] ≈ √(2 × ln(N)). For N=28, this gives ≈2.56. Is this the right formula? What assumptions does it rely on?

3. **What DSR threshold indicates genuine out-of-sample significance?** Is DSR > 2.0 a standard cutoff, or do practitioners use different thresholds? How does the threshold change with N?

4. **Are there alternatives to DSR for multiple testing correction in quantitative finance published between 2022-2026?** Any improvements or replacements for the Bailey-Lopez de Prado framework?

5. **How do quantitative firms handle the multiple testing problem in practice?** Any public information from Two Sigma, AQR, Citadel, DE Shaw, or similar firms about their strategy validation frameworks? Published talks, papers, or interviews?

6. **Benjamini-Hochberg FDR (False Discovery Rate) vs. Bonferroni vs. DSR -- which is most appropriate for trading strategy development?** My system uses a single "best" strategy selected from 28+ experiments. Is FDR even applicable (it's designed for multiple simultaneous tests, not sequential)?

---

## Domain 3: ICT (Inner Circle Trader) Methodology -- Academic Evidence Base

My system uses features derived from ICT concepts, but aggressive pruning eliminated 7/10 ICT feature families. The surviving features are: (1) displacement detection (momentum thrust > 1.5× ATR), (2) Fair Value Gaps (3-bar price gaps), and (3) dual-layer swing structure (pivot points). I need to know whether these have any academic basis.

1. **Is there ANY peer-reviewed academic research specifically studying ICT trading concepts (Fair Value Gaps, Order Blocks, liquidity sweeps, Smart Money Concepts)?** Search for papers published 2018-2026 that explicitly reference "ICT," "Inner Circle Trader," "Smart Money Concepts," "Fair Value Gap," or "Order Block" in the context of trading.

2. **What does the academic market microstructure literature say about price gaps as predictive features?** FVGs are 3-bar patterns where bar 1's high is below bar 3's low (or vice versa). This is a specific type of price gap. Is there research on gap predictiveness in crypto, equities, or futures? (Look for: Hasbrouck, O'Hara, Bouchaud, Cont -- microstructure researchers.)

3. **Is there academic evidence for "liquidity sweep" or "stop hunt" patterns?** The concept: price briefly pierces a previous high/low (triggering stops) before reversing. Is there research documenting this pattern? Is it related to stop-loss clustering research (Osler 2003, 2005)?

4. **Momentum thrust detection (displacement) -- what does the literature say about large-move continuation vs. reversal?** My displacement feature identifies bars where price moves > 1.5× ATR with close in the top/bottom 25% of the bar range. Is there evidence that such moves predict continuation? Or reversal? In crypto specifically?

5. **Pivot points / swing detection -- any evidence of predictive value at specific lookback windows?** My system uses 5-bar and 10-bar pivots. Is there research on optimal pivot lookback for different timeframes or instruments?

6. **Are there structural reasons why price gaps might have predictive power specifically in BTC perpetual futures?** Consider: funding rate dynamics, liquidation cascades, market maker behavior, exchange-specific mechanics.

---

## Domain 4: Triple-Barrier Labeling -- Evaluation and Alternatives

My system uses triple-barrier labels (1R stop / 2R target / 48-bar horizon). The AUC is 0.79, which is unusually high for financial ML. I suspect the label design may contribute to this high AUC.

1. **Triple-barrier labeling (Lopez de Prado, AFML Chapter 3) -- what are its known limitations?** Specifically: does the label design make the classification "easier" compared to fixed-horizon return prediction? What is the expected AUC difference between triple-barrier and fixed-horizon labels on the same features?

2. **What AUC is achievable with gradient boosting (LightGBM/XGBoost) on financial time series prediction tasks?** Provide published benchmarks, specifying: instrument, timeframe, label type, feature count, and reported AUC. Is AUC 0.79 on 5-minute BTC bars unusually high? What range is typical?

3. **Meta-labeling (Lopez de Prado, AFML Chapter 3.6) -- does it improve live trading performance vs. direct signal ML?** Has anyone published live results comparing meta-labeling (ML as filter) vs. direct labeling (ML as signal generator)?

4. **Fixed-horizon vs. triple-barrier vs. trend-scanning labels -- comparative studies?** Are there published comparisons of different labeling methods for financial ML? What do they find in terms of backtest performance vs. live performance?

5. **What is the "horizon expiry" problem in triple-barrier labels?** If a large fraction of labels hit the time horizon (neither stop nor target), how does this affect model training and reported metrics? Is there guidance on the acceptable fraction of horizon-expired labels?

---

## Domain 5: Walk-Forward Validation -- Expanding vs. Sliding Windows

My system uses expanding-window walk-forward: the first fold trains on 2020-2021, later folds train on all data from 2020 to the fold boundary. Each fold has a 288-bar embargo and label purging.

1. **The BTC ETF approval in January 2024 is confirmed as a structural break via Chow tests and DCC-GARCH analysis.** Given this confirmed break, is an expanding window with training data starting 2020 still defensible for predicting 2025 behavior, or is a rolling window of 12-18 months now the correct choice? What does recent literature (2023-2026) say about handling confirmed structural breaks in walk-forward validation -- specifically, should pre-break data be excluded entirely, down-weighted, or used with explicit regime conditioning?

2. **Has BTC market microstructure changed significantly since ETF approval (January 2024)?** If so, an expanding window that includes 2020-2023 data may dilute the signal with stale patterns. Is there evidence of a structural break in BTC price behavior around ETF approval?

3. **CSCV (Combinatorial Symmetric Cross-Validation) with C(8,4)=70 combinations and PBO threshold 0.05 -- is this sufficient?** What are the known limitations of CSCV? Are there cases where CSCV passes but the strategy still fails live?

4. **What is the minimum recommended number of trades per walk-forward fold for statistically reliable results?** My folds have 100-200+ trades each. Is this sufficient? What sample size does the academic literature recommend?

5. **What is the state-of-the-art for temporal cross-validation in financial ML as of 2025-2026?** Any advances beyond the AFML methodology (purged k-fold, CSCV, embargo)?

---

## Domain 6: Alpha vs. Beta Separation in Long-Only BTC Systems

My system is long-only on BTC from 2020-2026. During this period, BTC appreciated from ~$7,000 to ~$90,000+ (approximately 12×). I have never separated the system's returns from buy-and-hold returns.

1. **What methods exist for separating alpha from beta in a single-instrument, long-only trading system?** Standard approaches use factor models (CAPM, Fama-French), but for a single crypto asset with no established factor model, what alternatives exist?

2. **How much of a long-only BTC system's backtested performance from 2020-2026 is explained by the secular bull trend?** Has anyone published analysis of how much alpha typical BTC trading strategies add above buy-and-hold?

3. **What is the appropriate benchmark for a long-only BTC perpetual futures strategy?** Is it BTC spot return? BTC-adjusted for funding costs? A volatility-matched benchmark? What do crypto fund evaluators use?

4. **Conditional alpha analysis: does the system add value in down markets?** For a long-only system, alpha should be most visible when BTC is falling (the system should trade less or avoid losses). What methods test for conditional alpha across market regimes?

5. **Has BTC provided positive expected returns purely from being long over any multi-year period?** If BTC has a positive drift, any long-only system will show positive backtest returns without any predictive signal. What is the expected Sharpe of a naive BTC buy-and-hold from 2020-2026?

---

## Domain 7: Crypto Perpetual Futures Execution Reality

My backtest assumes entry at signal bar close with fixed slippage and historical funding rates. I need to understand execution reality for BTC perpetual futures.

1. **What is typical slippage for BTC perpetual futures on Bybit/Binance for $10K-$100K market orders?** Does this vary by time of day, volatility regime, or exchange? Provide data sources or practitioner accounts.

2. **Funding rate costs for BTC perpetual longs -- what is the actual annualized drag?** My data shows positive funding on ~322/365 days in 2024. Is 6-8% annualized a reasonable estimate? Has the funding regime changed post-ETF?

3. **What percentage of backtested alpha is typically lost in live crypto trading due to execution costs?** Published accounts from systematic crypto traders or funds. What is the expected backtest-to-live degradation ratio?

4. **Market impact thresholds for BTC perpetual futures: at what notional size does market impact become significant?** $100K? $500K? $1M? On Bybit vs. Binance? How does this affect a system that scales from $10K to larger positions?

5. **Are there known systematic biases in historical BTC exchange data that affect backtesting?** Exchange-specific price spikes, data gaps, survivorship bias in trading pairs, stale data during outages? How should these be handled?

6. **For a system that trades 178 times per year (approximately every 2 days) on 5-minute signals, how much does execution latency matter?** Is sub-second execution necessary, or is a 5-10 second execution window acceptable given the 5-minute bar resolution?

---

## Domain 8: LightGBM vs. Simpler Models for Financial Prediction

My system uses LightGBM but I have never compared it to logistic regression or other simpler models on the same features.

1. **Is there evidence that gradient boosting (LightGBM/XGBoost) outperforms logistic regression for financial time series prediction in out-of-sample tests?** Published comparisons, not just Kaggle competitions. Specifically for return prediction or trading signal generation.

2. **When a gradient boosting model achieves AUC 0.79 but a logistic regression achieves AUC 0.74+ on the same features, what does this imply?** Is the 0.05 gap typically from genuine non-linear interactions, or from overfitting?

3. **Monotonicity constraints in LightGBM for financial applications -- do they improve out-of-sample performance?** If I know that feature X should have a monotonically positive relationship with the target, does constraining this in LightGBM reduce overfitting?

4. **What regularization settings for LightGBM are recommended for financial time series?** Typical values for num_leaves, max_depth, min_child_samples, learning_rate, n_estimators for datasets with 500K-1M rows and 50-100 features?

5. **Are there published benchmarks comparing tree-based models to deep learning (transformers, temporal CNNs) for financial prediction tasks?** What is the current state-of-the-art for 5-minute bar financial prediction as of 2025-2026?

---

## Domain 9: NQ and ES Equity Index Futures -- Systematic Trading Considerations

I plan to extend my system from BTC to NQ1! (Nasdaq 100 futures) and ES1! (S&P 500 futures) using the same signal concept (momentum + gap + pivots).

1. **What data resolution is standard for systematic NQ/ES trading?** Is 5-minute appropriate, or do most systematic strategies use 1-minute, 15-minute, or daily bars? What resolution do published systematic NQ/ES studies use?

2. **What features are known to be predictive for intraday NQ/ES trading?** Opening range breakout, VWAP deviation, overnight gap, initial balance -- which of these have published evidence? Provide citations.

3. **What is the typical correlation between BTC and NQ/ES, and how has it changed over time (2020 vs. 2025)?** Specifically: is BTC correlated enough with NQ that cross-instrument validation is meaningless (same signal appearing twice), or different enough that it provides genuine out-of-distribution evidence?

4. **Execution characteristics of NQ/ES vs. BTC perps.** Typical slippage, market impact thresholds, session effects (RTH vs. ETH), and cost structures. How do these compare?

5. **ICT methodology was originally taught for forex and index futures. Are there backtested results of systematic ICT-based strategies on NQ/ES?** Published or well-documented backtest results (not just YouTube videos).

6. **For portfolio construction across BTC + NQ + ES, what correlation-adjusted position sizing methods are recommended?** Published research on cross-asset systematic portfolio construction combining crypto and equity futures?

---

## Domain 10: ML Trading System Failure Modes -- Post-Mortems and Deployment Reality

My system passes 10 validation gates in backtest. I need to understand how systems like this fail in practice.

1. **What are the most common failure modes for ML-based trading systems that pass backtesting but fail live?** Provide a ranked list with examples and citations. How often is the failure from overfitting vs. execution vs. regime change vs. other causes?

2. **Are there published post-mortems of ML trading systems that looked good in backtest but failed live?** Specific cases with root cause analysis. Academic papers, blog posts from practitioners, or fund letters.

3. **What validation tests am I NOT running that I should be?** Given my current framework (walk-forward, CSCV, PBO, PSR, DSR, ECE, isotonic calibration, SHAP analysis, causality testing), what else should I add? Check: Monte Carlo permutation tests, feature importance stability across folds, prediction calibration drift, regime-conditional performance, white-reality check (Romano & Wolf)?

4. **Is there a standard "deployment readiness" checklist for ML trading systems?** Published frameworks from academia or industry for deciding when a backtested strategy is ready for live capital?

5. **What monitoring metrics detect model degradation earliest?** For a live ML trading system, what should I track daily/weekly/monthly to detect when the model is failing before it costs significant money?

6. **What is the typical "half-life" of a quantitative trading strategy?** How quickly do edges decay in (a) crypto and (b) equity index futures? Published estimates of strategy half-life by asset class.

---

## Domain 11: Position Sizing Under Edge Uncertainty

My system uses Kelly criterion with 1/40 divisor (fractional Kelly), clipped [1%, 2%] risk per trade. The edge estimate comes entirely from backtesting -- no live validation.

1. **What does the academic literature recommend for position sizing when edge magnitude is uncertain?** Is fractional Kelly (1/2 Kelly or smaller) optimal, or are there better approaches? Reference Ed Thorp's work on Kelly criterion with estimation error.

2. **For a system transitioning from backtest to live, what position sizing ramp-up schedules do practitioners recommend?** Published approaches for gradually increasing position size as live data accumulates.

3. **What is the probability of ruin for different Kelly fractions?** Specifically: for a system with estimated daily Sharpe 2.0 (possibly optimistic), what is P(ruin) at Kelly 1/40 vs. 1/20 vs. 1/10?

4. **Are there adaptive position sizing methods that adjust based on recent live performance?** Methods that increase sizing when recent performance matches backtest and decrease when it diverges?

5. **What position sizing approaches do systematic crypto funds use?** Any public information from firms like Jump Crypto, Wintermute, Galaxy Digital, or others about their risk management frameworks?

---

## Domain 12: Feature Lookahead Bias -- Comprehensive Detection

I have 64 features computed from 5-minute OHLCV data with multi-timeframe resampling (H4, H1, M15) and rolling computations. I passed 18 causality tests but need to verify no subtle leakage remains.

1. **What are ALL known sources of lookahead bias in financial ML feature engineering?** Provide a comprehensive, prioritized checklist. Include subtle sources that are commonly missed.

2. **Multi-timeframe resampling (5min → H4/H1/M15): what is the correct method to avoid including the current incomplete bar?** If I'm at 5-minute bar 10:25, and I resample to H1, does the H1 bar for 10:00-11:00 exist yet? How should this be handled?

3. **Rolling window features: does pandas `rolling()` with default settings introduce lookahead?** What about `ewm()`? What about when rolling windows are applied to resampled (H4) data that is then merged back to 5-minute?

4. **HMM (Hidden Markov Model) regime detection: does batch Viterbi decoding on the full training window introduce lookahead?** My system uses a 3-state Gaussian HMM. If the HMM is fitted on data from T=0 to T=N, does the regime label at T=k (k < N) use information from T>k?

5. **SHAP-based feature selection using the same data as model evaluation -- does this create a subtle form of information leakage?** If I use SHAP importance to prune features and then evaluate the pruned model on the same walk-forward structure, is this "triple peeking"?

6. **Are there automated tools or systematic methods for detecting lookahead bias in feature pipelines?** Open-source Python implementations? Published methodologies beyond manual code review?

---

## Deliverable Format

For each domain, please provide:

1. **Direct answers** with specific citations (author, title, year, journal/venue)
2. **Peer-reviewed vs. practitioner** distinction for each source
3. **"No evidence exists"** where appropriate -- do not speculate
4. **Contradictory evidence** presented on both sides when it exists
5. **Quantitative claims** with source and sample size (e.g., "typical live crypto Sharpe is X, based on [source] studying N funds")
6. **Recency priority**: prefer 2022-2026 sources, but include foundational older work where it's the definitive reference

This research will inform real financial deployment decisions. Accuracy and source quality matter more than comprehensiveness.
