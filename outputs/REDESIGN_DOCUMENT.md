# BTC Algorithmic Trading System: Independent Review & Redesign

## Executive Summary

This document is an independent, first-principles review of the BTC perpetual futures trading system built over 55 documented decision iterations (D01–D55). The system uses ICT (Inner Circle Trader) methodology features fed into a LightGBM walk-forward ML pipeline with isotonic calibration, 10-gate CSCV validation, and triple-barrier labeling.

**Bottom line:** The system demonstrates genuine engineering rigor—walk-forward validation, label purging, CSCV, 10-gate framework, dataset hardening, and aggressive feature pruning are all sound. However, several critical issues must be resolved before any capital deployment:

1. **Sharpe 12.3 is an annualization artifact, not a real Sharpe ratio.** The true daily-annualized Sharpe is likely 1.5–3.0.
2. **Multiple testing contamination is unquantified.** 55+ experiments on the same dataset without rigorous trial counting invalidates naive statistical significance.
3. **The ICT framework is mostly noise.** D55 proved 7/10 D53 feature families are dead weight. The 64 surviving features likely capture simple price dynamics that happen to align with ICT nomenclature.
4. **SEARCH_SPACE parameters are not wired to rules.py.** Optuna suggests ICT parameter values that are never actually used—all rules functions run with hardcoded defaults.
5. **No live validation exists.** The entire evidence base is backtested. Execution assumptions (slippage, fill rates, funding timing) are untested.

The system has a plausible but unproven edge. The path to deployment requires fixing the Sharpe computation, running honest multiple-testing corrections, touching the holdout exactly once with pre-registered thresholds, and completing a mandatory paper trading phase.

---

## Part 1: Honest Assessment of Current State

### 1.1 What's Actually Good

**Methodical Decision Framework.** 55 documented decisions with full reasoning, data, and willingness to reverse (D25 dropped shorts, D48 brought them back with ML scoring). This discipline is rare and valuable.

**Walk-Forward with Embargo + Purging.** Expanding window, 9 folds, 288-bar embargo, label purging via `train_idx = train_idx[train_idx + horizon < val_start]`. This is textbook AFML Ch.7 implementation. The double-embargo pattern (train→val and val→test gaps) adds further protection.

**10-Gate Validation Framework.** Independent statistical tests: AUC≥0.75, PBO≤0.05, PSR≥0.99, WR≥0.55, EV≥0.50R, Sharpe≥2.0, MaxDD≤0.20, ECE≤0.05, plus trade count and window consistency. This multi-dimensional validation is far beyond what most retail quant systems implement.

**Dataset Hardening (D51).** Holdout carve (2025-03 to 2026-02), embargo increase 48→288, AFML label purging, availability masks. This retroactively fixed real methodological problems.

**Feature Pruning Discipline (D55).** 670→64 features with zero AUC loss. This is strong evidence that the surviving features capture genuine signal, not noise. The willingness to delete 90% of features shows intellectual honesty.

**Isotonic Calibration.** PAVA algorithm with ECE≤0.05 gate. Well-calibrated probabilities are essential for Kelly-based sizing and were correctly identified as the key to passing all 10 gates (D43).

**Clean Baseline Methodology.** D54a/D54c establish regression-safe baselines post-hardening. Every future experiment must beat these baselines or be rejected.

### 1.2 What's Actually Wrong

#### 1.2.1 Sharpe 12.3 Is Not a Real Sharpe Ratio

This is the single most important issue. The reported Sharpe appears to be computed as:

```
per_bar_mean / per_bar_std * sqrt(bars_per_year)
```

With 5-minute bars: `bars_per_year ≈ 105,120`. A per-bar Sharpe of 0.038 becomes `0.038 * sqrt(105120) ≈ 12.3`.

This is mathematically correct but practically meaningless. It assumes:
- Returns are i.i.d. (they are not—autocorrelation exists in intraday returns)
- No execution friction at the bar level (unrealistic for 5-minute bars)
- The system actually trades at every bar (it doesn't—178 trades/year)

**The correct computation:** Aggregate to daily P&L, compute daily Sharpe, annualize with `sqrt(252)`. The likely result: Sharpe 1.5–3.0. Still potentially good, but a completely different risk profile.

This is not a minor quibble. A Sharpe of 12.3 implies near-zero probability of losing money in any given month. A Sharpe of 2.5 implies a ~5% monthly loss probability. These demand entirely different position sizing and risk management.

#### 1.2.2 Multiple Testing Contamination

The strategy log documents 55+ decision iterations on the same BTC 2020-2026 dataset. Even with walk-forward validation, each experiment that informs the next experiment creates a dependency chain. The effective number of independent hypotheses tested is at least 10-15 (conservative count of materially different configurations).

The Deflated Sharpe Ratio (DSR) is mentioned but it's unclear whether `N_trials` honestly counts ALL experiments, including:
- E000-E008 (explicitly contaminated—saw holdout, inadequate embargo)
- All Optuna trials within each experiment
- All manual configuration variants (Config A/B/C, various thresholds)
- The D53 ICT overhaul (which introduced 204 new features, most of which were noise)

With honest N_trials ≥ 50, the DSR threshold rises substantially. A reported Sharpe of 2.5 (daily-annualized) with N=50 trials has a DSR that may or may not clear 2.0.

#### 1.2.3 The ICT Framework Is Mostly Noise

This finding emerged clearly from the project's own analysis:

- **D36 (SHAP):** ICT structural features = 72.3% of signal. But this includes simple price-level features (swing highs/lows, premium/discount zones) that any price-action system would use.
- **D53:** Added 204 new features from 10 ICT function families (displacement, anchored OBs, dual-layer swings, sweep detection, FVG enhancements, OTE, CISD, MSS, premium/discount, sweep sequences).
- **D55:** Pruned 670→64 features. **7 out of 10 D53 feature families were dead weight.** Only 3 families survived.

The surviving 64 features likely capture:
1. Simple price dynamics (momentum, mean reversion at support/resistance)
2. Volatility regime indicators
3. FVG-related gap features (the original edge generator from D16-D21)

ICT nomenclature is a retail-trader marketing layer on top of real price phenomena. The ML doesn't care about "order blocks" or "CISD"—it cares about statistical regularities in price. The D55 result proves this: you can strip away most of the ICT taxonomy and lose nothing.

**Implication:** Future feature engineering should focus on the statistical properties of the surviving 64 features, not on adding more ICT concepts.

#### 1.2.4 SEARCH_SPACE → rules.py Wiring Gap

This is a critical implementation bug. The Optuna search space in `parameters.py` defines 9 ICT structure parameters:

```
ict_structure: swing_n_internal, swing_n_external, fvg_min_pct,
               ob_refresh_bars, sweep_lb, cisd_lb, displacement_mult,
               displacement_lb, mss_lb
```

But `simulator.py:augment_features()` calls all `rules.py` functions with **hardcoded defaults**. The SEARCH_SPACE values are never passed through. This means:
- Every Optuna trial that varied ICT parameters was testing the same feature values
- The "optimization" of ICT parameters was illusory
- All reported results use default ICT parameters regardless of what Optuna suggested

This doesn't invalidate the results (they're all from the same default parameters), but it means the ICT parameter space is entirely unexplored. Fixing this wiring could improve or worsen performance—we don't know.

#### 1.2.5 Survivorship Bias in Instrument Selection

The system has been developed exclusively on BTC perpetual futures during 2020-2026, a period characterized by:
- Secular bull trend (BTC ~$7K → $80K+)
- Increasing institutional adoption and liquidity
- Growing derivatives market infrastructure
- Multiple strong trend phases interspersed with sharp corrections

A long-only system on BTC during this period captures substantial beta. The question is whether the ML-filtered long signals add alpha above buy-and-hold. This requires computing:
- Buy-and-hold return for the same period
- System return (daily P&L, not per-bar)
- Alpha = System return - Beta * Market return

If the system doesn't meaningfully beat risk-adjusted buy-and-hold, the complexity is not justified.

#### 1.2.6 Feature Construction Audit Not Done

The 224 on-the-fly features are computed in Python with pandas operations on 543K bars. No formal audit has been conducted to verify:
- No lookahead bias in any feature computation
- Rolling/expanding window operations use correct `min_periods`
- Multi-timeframe features (H4, H1, M15) are resampled correctly without future bars
- Edge effects at series boundaries are handled

This is not theoretical. Common sources of lookahead bias in financial ML:
- Using `df['close'].shift(-1)` anywhere in feature computation
- Resampling to higher timeframes that include the current incomplete bar
- Filling NaN with forward-fill that uses future values
- Computing features on the full series before train/test split (information leakage through statistics)

#### 1.2.7 No Execution Model Validation

The backtester assumes:
- Entry at the close of the signal bar (or next bar open?)
- Fixed slippage (if any)
- Funding rate formula from historical data
- No market impact
- Fills always occur at specified prices

In live BTC perpetual trading:
- Slippage on 5-minute entries can be 0.01-0.05% depending on size and liquidity
- Funding rates are variable and can be extreme during volatile periods
- Market impact exists for positions above ~$100K notional
- Liquidation engine interactions can cause unexpected fills
- Exchange API latency means signal-to-execution delay is 1-10 seconds

### 1.3 The Core Question: Genuine Edge or Data Mining?

**Assessment: The system likely has a small genuine edge, heavily polluted by data mining artifacts that make it appear much larger than it is.**

Evidence supporting a genuine edge:
- AUC 0.79 is stable across multiple retrains and validation configurations
- Feature pruning maintained AUC (robust, not overfit to specific features)
- FVG-based features consistently appear as top SHAP contributors
- Walk-forward with embargo/purging is sound methodology
- 10/10 gates pass consistently

Evidence suggesting the edge magnitude is overstated:
- Sharpe inflation from per-bar annualization (12.3 → likely 1.5-3.0)
- Multiple testing on same dataset (55+ experiments)
- BTC secular bull trend flatters long-only system
- No out-of-distribution validation (different instruments or regimes)
- No live trading evidence

**Realistic expectation if the edge is real:** After proper Sharpe correction, multiple testing adjustment, and execution cost modeling, the system might deliver Sharpe 1.0-2.0 with 15-40% annual return at moderate leverage. This is worth pursuing, but it's not the money printer that Sharpe 12.3 implies.

---

## Part 2: Proposed Architecture (Redesign from First Principles)

### 2.1 Philosophy

The redesign preserves what works (walk-forward, gate framework, triple-barrier, LightGBM, isotonic calibration) and fixes what doesn't (Sharpe computation, multiple testing, ICT noise, parameter wiring, execution model). It does NOT throw away the existing codebase—it refactors and extends it.

Guiding principles:
1. **Measure correctly before optimizing.** Fix Sharpe computation and multiple testing before any new experiments.
2. **Features should be statistical, not theological.** Keep features that survive pruning; don't add more ICT concepts.
3. **Validate out-of-distribution before trusting in-distribution.** Multi-instrument testing is the strongest evidence of genuine edge.
4. **Deploy incrementally.** Paper → micro → small → target size.

### 2.2 Signal Architecture

#### 2.2.1 Feature Framework

**Tier 1: Proven Features (Keep)**
The 64 features surviving D55 pruning. These have demonstrated predictive power through aggressive elimination. Do not modify or remove these.

**Tier 2: Price Action Primitives (Audit & Refine)**
Reframe surviving ICT features in statistical terms:
- "FVG" → Price gap features: magnitude, fill rate, time since formation
- "Order Block" → Rejection zone features: volume at reversal, price displacement
- "Sweep" → Liquidity probe features: wick ratio, recovery speed
- "Premium/Discount" → Mean reversion features: distance from rolling median

This reframing isn't cosmetic—it clarifies what the ML actually uses and guides future feature engineering toward statistical properties rather than ICT taxonomy.

**Tier 3: Cross-Instrument Features (New)**
When extending to NQ/ES:
- BTC-NQ rolling correlation (21-period, 63-period)
- ES realized volatility as regime indicator
- Cross-instrument momentum divergence
- US market session alignment features

**Tier 4: Microstructure Features (Future)**
- Funding rate momentum and extremes
- Open interest changes
- Liquidation clustering
- Bid-ask spread (if available)

#### 2.2.2 Feature Audit Protocol

Before any new experiments, audit the surviving 64 features:

```
For each feature f in surviving_64:
  1. Trace computation from raw OHLCV to final value
  2. Verify: no future data used at any step
  3. Verify: rolling windows use min_periods correctly
  4. Verify: multi-TF resampling doesn't include current incomplete bar
  5. Compute: feature stability across folds (rank correlation of SHAP)
  6. Document: what price phenomenon this feature captures
```

Priority: Top 20 features by SHAP importance get full audit. Remaining 44 get spot-check audit.

#### 2.2.3 Fix the SEARCH_SPACE → rules.py Wiring

This is a blocking bug. The fix:

```python
# In simulator.py:augment_features()
# BEFORE (broken):
df = detect_fvg_enhanced(df)  # uses hardcoded defaults

# AFTER (fixed):
df = detect_fvg_enhanced(
    df,
    fvg_min_pct=params.get('fvg_min_pct', 0.001),
    # ... other params from trial
)
```

Wire ALL SEARCH_SPACE ICT parameters through `augment_features()` to their respective `rules.py` functions. Then re-run optimization to see if non-default ICT parameters improve performance.

### 2.3 Labeling

#### 2.3.1 Keep Triple-Barrier, Fix Edge Cases

The triple-barrier framework (1R stop / 2R target / N-bar horizon) is sound. Retain it. But audit:

1. **Label distribution:** What % hit target, hit stop, or expire?
   - If >30% expire: horizon may be too short or barriers too wide
   - If stop-hit rate > 40%: stop may be too tight
   - Healthy distribution: ~35% target, ~25% stop, ~40% expire (for 2R target)

2. **ATR computation:** Is the ATR for stop/target calculation point-in-time? Is it computed on the training set only, or does it see test data?

3. **Label stability test:** Perturb stop by ±5%, target by ±5%. If >10% of labels flip, the labeling is fragile and the model is learning noise.

#### 2.3.2 Consider Side Labels

Currently: binary (long entry / no entry). Consider:
- **Three-class:** Strong long / Weak long / No trade
- This gives the model a "soft no" option that captures borderline setups
- Requires ordinal calibration but may improve trade selection

### 2.4 ML Architecture

#### 2.4.1 LightGBM Configuration Audit

Current configuration should be verified against overfitting best practices:

| Parameter | Recommended Range | Purpose |
|-----------|------------------|---------|
| `num_leaves` | 31-127 | Lower = less overfit |
| `max_depth` | 6-10 | Hard depth limit |
| `min_child_samples` | 50-200 | Minimum samples per leaf |
| `feature_fraction` | 0.6-0.8 | Random feature subset per tree |
| `bagging_fraction` | 0.7-0.9 | Random sample subset per tree |
| `lambda_l1` | 0.1-10 | L1 regularization |
| `lambda_l2` | 0.1-10 | L2 regularization |
| `learning_rate` | 0.01-0.05 | Lower with more trees |
| `n_estimators` | 500-2000 | With early stopping |

**Key check:** Is early stopping based on validation AUC? If not, it should be.

#### 2.4.2 Walk-Forward Refinements

**Current:** Expanding window, 9 folds, 288-bar embargo, label purging.

**Refinements:**
1. **Monitor fold-by-fold performance.** If later folds (more training data) perform worse than earlier folds, the market is non-stationary and expanding window may hurt.
2. **Add sliding window variant.** Train on most recent N bars only. Compare expanding vs sliding. If sliding wins, the older data is hurting.
3. **Increase minimum fold trades.** MIN_TRADES_PER_YEAR=100 is low. A fold with 100 trades has wide confidence intervals. Consider 150-200.

#### 2.4.3 Calibration

Isotonic calibration with PAVA is correct. No changes needed. But verify:
- Calibration is fit on validation data only (not training data)
- Calibration stability: does the isotonic mapping change drastically between folds?
- Extreme predictions: what does the model output for P(win)>0.90? Are there enough samples in that range for reliable calibration?

### 2.5 Validation Architecture (Critical Fixes)

#### 2.5.1 Fix Sharpe Ratio Computation

```python
# WRONG (current):
per_bar_returns = ...  # 5-minute returns
sharpe = per_bar_returns.mean() / per_bar_returns.std() * np.sqrt(105120)

# CORRECT:
daily_pnl = per_bar_returns.resample('D').sum()  # or groupby date
daily_sharpe = daily_pnl.mean() / daily_pnl.std()
annual_sharpe = daily_sharpe * np.sqrt(252)
```

This is priority #1. It changes the entire risk assessment.

#### 2.5.2 Honest Multiple Testing Correction

Implement Bailey-Lopez de Prado DSR with honest trial counting:

```
N_trials = (
    8   # E000-E008 contaminated experiments
  + 5   # Major config variants (A/B/C, threshold sweeps)
  + 3   # Retrains (D34 v2, D43 prune, D55 prune)
  + 2   # Short model variants
  + 5   # Regime filter variants
  + ???  # All Optuna trials (could be hundreds)
)
```

Conservative estimate: N_trials ≥ 25. With Optuna: N_trials ≥ 100.

Compute DSR for the final model. If DSR < 2.0 with honest N, the result is not statistically significant at conventional levels. This doesn't mean the edge doesn't exist—it means we can't prove it from backtesting alone and live validation becomes essential.

#### 2.5.3 Holdout Protocol

The holdout (2025-03 to 2026-02, 105,121 bars) is reportedly untouched. This is the single most valuable piece of evidence remaining. The protocol:

1. **Pre-register thresholds BEFORE touching holdout:**
   - Minimum AUC: 0.72 (lower than train to account for regime shift)
   - Minimum daily-annualized Sharpe: 1.0
   - Minimum WR: 60%
   - Minimum EV: 0.30R
   - Maximum drawdown: 25%
   - Minimum trades: 40 (for ~11 months of data)

2. **Run ONCE with the final production model.** No parameter changes, no feature changes, no threshold adjustments.

3. **Report all metrics.** Do not cherry-pick. If AUC is 0.74 but Sharpe is 0.8, report both.

4. **If holdout fails: STOP.** Do not re-optimize and re-test. The system does not have a deployable edge. Go back to fundamental research.

5. **If holdout passes: Proceed to paper trading.** Holdout pass is necessary but not sufficient.

#### 2.5.4 Out-of-Distribution Validation

The strongest evidence of genuine edge is performance on unseen instruments. When extending to NQ/ES:
- Same feature framework, same model architecture
- Independent training and validation
- If the methodology works on NQ/ES without BTC-specific tuning, it's much more likely to be real

### 2.6 Execution Model

#### 2.6.1 Realistic Cost Model

Build a comprehensive cost model:

| Cost Component | BTC Perp Estimate | Source |
|---------------|-------------------|--------|
| Maker fee | 0.02% | Exchange fee schedule |
| Taker fee | 0.05% | Exchange fee schedule |
| Slippage (entry) | 0.01-0.03% | Measure from paper trading |
| Slippage (exit) | 0.01-0.03% | Measure from paper trading |
| Funding rate | Variable (~0.01%/8h) | Historical average |
| Signal delay | 1-5 seconds | Measure from paper trading |

**Total round-trip cost estimate: 0.10-0.20% per trade.**

With 178 trades/year and 2R target (meaning target is ~2x the stop-loss distance), each trade's expected gross return at WR 74.9%:
```
EV = 0.749 * 2R - 0.251 * 1R = 1.247R (matches reported 1.20R)
```

If R ≈ 0.5% (typical for 5-min ATR-based stop), gross return per trade ≈ 0.6%. Net of costs (0.15%): ~0.45% per trade. Annual: 0.45% * 178 ≈ 80% gross before sizing.

**But this assumes WR 74.9% survives in live trading.** A realistic degradation to WR 65% changes everything:
```
EV = 0.65 * 2R - 0.35 * 1R = 0.95R
Net per trade: 0.95 * 0.5% - 0.15% ≈ 0.325%
Annual: 0.325% * 178 ≈ 58%
```

Still attractive if real. But with Kelly 1/40 sizing (2.5% risk per trade max), the actual portfolio return depends on position sizing and compounding effects.

#### 2.6.2 Position Sizing

**Current:** Kelly fraction with 1/40 divisor, clipped [1%, 2%].

**Recommendation:** Start more conservatively.
- Phase 1 (Paper): Full Kelly 1/40 to measure hypothetical performance
- Phase 2 (Micro-live): 0.25% fixed risk per trade (no Kelly)
- Phase 3 (Scale): If paper matches live within tolerance, scale to Kelly 1/60 (more conservative)
- Phase 4 (Target): After 6+ months, consider Kelly 1/40

Rationale: Kelly assumes known edge. In early live trading, edge is uncertain. Fractional Kelly (1/60 to 1/40) limits ruin probability while still capitalizing on edge.

#### 2.6.3 Risk Management (Not Optional)

Hard circuit breakers that override all signals:

| Rule | Threshold | Action |
|------|-----------|--------|
| Daily loss limit | 2% of equity | Stop trading for 24h |
| Weekly loss limit | 5% of equity | Stop trading until manual review |
| Drawdown from peak | 15% | Stop trading, full system review |
| Consecutive losses | 8 in a row | Reduce size by 50% |
| Model staleness | No retrain in 30 days | Stop trading |
| Execution divergence | >0.1% avg slippage | Reduce size, investigate |

### 2.7 Multi-Instrument Extension

#### 2.7.1 Instrument Selection

| Instrument | Rationale | Data Source | Priority |
|-----------|-----------|-------------|----------|
| BTCUSDT Perp | Existing system | Bybit/Binance | Primary |
| NQ1! (Nasdaq Futures) | ICT originally taught on indices; high liquidity | CME | Secondary |
| ES1! (S&P 500 Futures) | Most liquid futures globally; different vol profile | CME | Tertiary |

#### 2.7.2 Shared vs Instrument-Specific Components

**Shared:**
- Feature computation framework (same code, different data)
- Walk-forward validation pipeline
- Gate framework and thresholds
- Calibration methodology
- Risk management rules

**Instrument-Specific:**
- Trained LightGBM models (separate per instrument)
- Triple-barrier parameters (R-multiple calibrated to instrument ATR)
- Session timing (crypto 24/7 vs equity RTH)
- Cost model parameters
- Funding rate logic (crypto-specific)

#### 2.7.3 Cross-Instrument Features

When models for multiple instruments exist:
- Portfolio-level correlation monitoring
- Cross-instrument signal confirmation (if BTC and NQ both signal long, higher confidence)
- Correlation-adjusted position sizing (reduce size when BTC-NQ correlation spikes)
- Regime features: VIX as equity regime indicator, BTC dominance as crypto regime indicator

### 2.8 What This Redesign Does NOT Change

Some things in the current system are correct and should be preserved:

1. **LightGBM as primary model.** Gradient boosting is proven for tabular financial data. No need for neural networks, transformers, or other architectures at this stage.
2. **Triple-barrier labeling.** Sound framework for defining trade outcomes.
3. **Walk-forward with embargo.** Correct temporal cross-validation approach.
4. **Isotonic calibration.** Best practice for probability calibration with tree models.
5. **10-gate validation.** Comprehensive multi-dimensional validation.
6. **Longs as primary direction.** In trending markets, long-biased systems have structural advantage.
7. **FVG as core feature family.** Consistently top SHAP contributor; survived all pruning rounds.

---

## Part 3: Implementation Roadmap

### Phase A: Measurement Fixes (1-2 weeks)
**Goal: Know what you actually have before changing anything.**

| Step | Task | Effort | Dependency |
|------|------|--------|------------|
| A1 | Fix Sharpe to daily-annualized | 2h | None |
| A2 | Compute buy-and-hold benchmark for same period | 2h | None |
| A3 | Calculate alpha over buy-and-hold | 1h | A1, A2 |
| A4 | Honest DSR with N_trials ≥ 25 | 4h | A1 |
| A5 | Label distribution audit (target/stop/expire %) | 2h | None |
| A6 | Feature lookahead audit (top 20 by SHAP) | 8h | None |
| A7 | Wire SEARCH_SPACE to rules.py augment_features | 4h | None |

**Exit criterion:** Daily-annualized Sharpe > 1.0 AND DSR > 1.5. If either fails, do not proceed—go back to research.

### Phase B: Holdout Validation (1 day)
**Goal: Touch the holdout exactly once.**

| Step | Task | Effort | Dependency |
|------|------|--------|------------|
| B1 | Pre-register holdout thresholds (write to file before running) | 1h | Phase A complete |
| B2 | Run production model on holdout | 1h | B1 |
| B3 | Report all metrics, compare to pre-registered thresholds | 1h | B2 |

**Exit criterion:** Pass pre-registered thresholds. If fail, STOP. System is not deployable.

### Phase C: Paper Trading (1-3 months)
**Goal: Validate execution assumptions and measure live-vs-backtest divergence.**

| Step | Task | Effort | Dependency |
|------|------|--------|------------|
| C1 | Set up Bybit testnet or paper trading account | 4h | Phase B pass |
| C2 | Implement signal generation → order execution pipeline | 16-24h | C1 |
| C3 | Implement monitoring dashboard (P&L, fills, slippage) | 8h | C2 |
| C4 | Run paper trading for minimum 4 weeks | Ongoing | C3 |
| C5 | Compare paper results to backtest expectations | 4h | C4 |

**Exit criterion:** Paper Sharpe within 50% of backtested daily Sharpe AND execution slippage < 0.05% average.

### Phase D: Micro-Live Deployment (3-6 months)
**Goal: Validate with real capital at minimal risk.**

| Step | Task | Effort | Dependency |
|------|------|--------|------------|
| D1 | Deploy with 0.25% risk per trade (fixed, no Kelly) | 4h | Phase C pass |
| D2 | Implement all circuit breakers | 8h | D1 |
| D3 | Run for minimum 3 months | Ongoing | D2 |
| D4 | Monthly review: compare live to backtest | 2h/month | D3 |

**Exit criterion:** 3 months of live results within 40% of backtested daily Sharpe. No circuit breaker triggers.

### Phase E: Scale & Multi-Instrument (6+ months)
**Goal: Scale position sizing and extend to new instruments.**

| Step | Task | Effort | Dependency |
|------|------|--------|------------|
| E1 | Scale to Kelly 1/60 | 1h | Phase D pass |
| E2 | Source NQ/ES data | 8h | Any time |
| E3 | Adapt feature framework for NQ/ES | 16-24h | E2 |
| E4 | Train and validate NQ/ES models | 16-24h | E3 |
| E5 | Paper trade NQ/ES | 4 weeks | E4 |
| E6 | Deploy NQ/ES micro-live | Ongoing | E5 |

---

## Part 4: Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| R1 | Sharpe collapses after daily annualization | HIGH | CRITICAL | Phase A fixes this before any deployment |
| R2 | Holdout fails | MEDIUM | CRITICAL | Do not deploy. Return to research. |
| R3 | Paper trading diverges >50% from backtest | MEDIUM | HIGH | Investigate execution model. Do not go live until resolved. |
| R4 | BTC regime shift invalidates model | MEDIUM | HIGH | Monthly retraining. Drawdown circuit breaker. Multi-instrument diversification. |
| R5 | Feature lookahead bias discovered | LOW-MED | CRITICAL | Audit in Phase A. If found, re-run entire validation from scratch. |
| R6 | Exchange API issues (latency, downtime) | MEDIUM | MEDIUM | Redundant exchange connections. Position size limits. |
| R7 | Multiple testing renders DSR insignificant | MEDIUM | HIGH | Honest N-counting in Phase A. If DSR < 1.5, edge is unproven. |
| R8 | Overfitting to 2020-2026 BTC market structure | MEDIUM | HIGH | Multi-instrument validation. Out-of-sample testing on pre-2020 data if available. |
| R9 | Kelly sizing too aggressive for live volatility | LOW | MEDIUM | Start with fixed 0.25%, scale slowly. |
| R10 | ICT parameter space unexplored (wiring gap) | CONFIRMED | MEDIUM | Fix in Phase A (A7). May improve or worsen results. |

---

## Part 5: Deployment Decision Criteria

### Go/No-Go Checklist

The system is deployable if and only if ALL of the following are true:

- [ ] Daily-annualized Sharpe > 1.0 (Phase A)
- [ ] DSR > 1.5 with honest N_trials (Phase A)
- [ ] Alpha over buy-and-hold > 0% annualized (Phase A)
- [ ] No lookahead bias in top 20 features (Phase A)
- [ ] Holdout passes pre-registered thresholds (Phase B)
- [ ] Paper trading Sharpe within 50% of backtested daily Sharpe (Phase C)
- [ ] Average execution slippage < 0.05% (Phase C)
- [ ] 3 months live with no circuit breaker triggers (Phase D)
- [ ] Monthly live P&L within 40% of backtested expectation (Phase D)

### Kill Criteria

Stop the project immediately if:
- Daily-annualized Sharpe < 0.5
- DSR < 1.0
- Holdout AUC < 0.65
- Lookahead bias found in any top-10 SHAP feature
- Paper trading produces negative P&L over 4+ weeks
- Live drawdown exceeds 15% from peak
- 3+ consecutive months of negative live P&L

---

## Part 6: Honest Conclusion

This system represents serious, disciplined quantitative work. The walk-forward validation, gate framework, isotonic calibration, dataset hardening, and aggressive feature pruning are all best-practice. Most retail quant projects never reach this level of rigor.

However, the system has not yet proven it has a deployable edge. The inflated Sharpe ratio, unquantified multiple testing burden, untested execution model, and single-instrument in-sample-only validation leave the core question unanswered.

The good news: answering that question is straightforward. Fix the Sharpe computation (1 day). Run honest DSR (1 day). Touch holdout (1 day). Paper trade (1-3 months). These are not research problems—they're engineering tasks with clear deliverables.

The most likely outcome: the system has a small but real edge (Sharpe 1.0-2.5 daily-annualized) that can generate 15-40% annual returns with moderate risk, after proper position sizing and execution costs. This is not spectacular by backtest standards but is genuinely good by live-trading standards. Most profitable systematic trading strategies operate in this range.

The worst-case outcome: the edge is entirely data-mined, the Sharpe is below 1.0 after correction, and the holdout fails. In this case, the infrastructure (walk-forward pipeline, gate framework, calibration) is still valuable and can be applied to new signal research.

Either way, you'll know within 2 weeks of executing Phase A.
