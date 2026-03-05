# BTC Algorithmic Trading System: Independent Redesign (V2)

*Generated: 2026-03-04*
*Input corpus: CLAUDE.md (793 lines), STRATEGY_LOG.md D01-D55 (3,923 lines), THE_PLAN.md (~1,330 lines), D53_IMPLEMENTATION_SPEC.md (851 lines), core/engine/evaluator.py, core/engine/simulator.py, core/config/parameters.py, core/signals/ict/rules.py*

---

## Preface: What the First Attempt Got Wrong (and Why)

The first redesign document committed the cardinal sin of institutional consulting: it validated the existing architecture while repackaging it with better labels. It said "your Sharpe is inflated" but kept the Sharpe computation framework. It said "ICT features need validation" but kept the ICT feature set intact. It said "you need NQ/ES" but provided a two-paragraph sketch.

Three specific failures:

1. **It never confronted the ICT collapse.** D55 pruning eliminated 7/10 ICT feature families with zero AUC loss. The surviving signal is displacement + FVG + swing structure -- that's gap-and-momentum with pivot points, not "ICT methodology." The first attempt didn't confront the implication: the elaborate ICT taxonomy may be intellectual scaffolding around three standard technical analysis concepts.

2. **It never computed what matters.** The daily Sharpe ratio is the number that determines whether this system is tradeable. The first attempt discussed Sharpe inflation abstractly but never estimated the daily Sharpe, never compared to buy-and-hold, and never asked whether the system's returns are alpha or beta.

3. **It treated NQ/ES as diversification.** NQ/ES is not a portfolio extension -- it is the only available out-of-distribution validation before risking real capital. The first attempt's NQ/ES section was a placeholder, not a design.

**Root cause:** The first attempt treated the codebase as a constraint and the redesign as optimization within those constraints. This attempt treats the trading thesis as a hypothesis and the codebase as one possible -- potentially wrong -- implementation.

---

## Part 1: Honest Assessment

### 1.1 What Is Genuinely Sound

**1. Decision discipline (D01-D55).** 55 documented decisions with explicit rationale, alternatives considered, and test results. Each decision has a GO/NO-GO outcome and feeds forward into subsequent decisions. This level of documentation is rare in quantitative trading development and creates an auditable trail that prevents revisionism.

**2. Walk-forward with double embargo and label purging.** The expanding-window walk-forward in `evaluator.py` implements 288-bar gaps between both train→val AND val→test splits, plus label purging that removes training samples whose labels depend on test period data (AFML Ch.7). The implementation is correct: `train_idx = train_idx[train_idx + horizon < val_start]`. Most retail ML trading projects skip temporal splitting entirely; this one implements it properly.

**3. 10-gate validation framework.** Requiring simultaneous passage of AUC ≥ 0.75, PBO ≤ 0.05, PSR ≥ 0.99, Sharpe ≥ 2.0, WR ≥ 55%, EV ≥ 0.50R, MaxDD ≤ 20%, ECE ≤ 0.05, trades ≥ 100/year, and all-windows-profitable creates a high bar that protects against cherry-picking. The framework's structure is sound even where specific thresholds need recalibration.

**4. Isotonic calibration (D43).** Discovering that ECE collapsed from 0.124 to 0.016 with PAVA isotonic calibration was a genuine finding. Well-calibrated probabilities are prerequisite for any Kelly-based position sizing. The implementation in `calibrator.py` is clean and sklearn-free.

**5. Dataset hardening (D51).** Carving a true holdout of 105,121 rows (2025-03 to 2026-02), widening embargo from 48 to 288 bars, adding label purging, creating availability masks (has_oi, has_liqs), and marking experiments E000-E008 as COMPROMISED shows genuine intellectual honesty. Most projects would quietly continue with the contaminated results.

**6. Feature pruning (D55).** Reducing 670 features to 64 with zero AUC loss demonstrates that the predictive signal is highly concentrated, not diffusely spread across hundreds of features. This is actually good news -- concentrated signal is harder to overfit than diffuse signal. The pruning also correctly used SHAP importance with a 0.010 threshold and protected 6 regime-dependent features.

**7. Short side investigation (D48, D54c).** Testing shorts separately and finding that ML-scored shorts are viable (AUC 0.7966, 10/10 PASS) despite structural shorts being negative EV (D25: 25-33% WR) is an interesting finding. It means the ML model is adding genuine value on the short side, even against the positive-funding headwind. The 7/10 top feature overlap with longs suggests a common underlying signal.

**8. Causality testing.** Running 18 causality checks on all on-the-fly features at multiple test points (T = 1K, 5K, 10K, 50K bars) provides meaningful protection against one class of lookahead bias. The swing lag correlation check (all < 0.005) is a specific, targeted test for the highest-risk leakage vector.

**9. GT-Score objective (D52).** The composite optimization objective -- recency-weighted fold Sharpe minus variance minus BIC minus feature penalty -- with a hard floor rejecting any trial where worst fold Sharpe < -0.5, is a well-designed objective that resists common optimizer pathologies (overfitting recent data, accepting unstable strategies, rewarding complexity).

**10. On-the-fly feature augmentation.** Computing 224 features at experiment runtime via the ONTHEFLY_FEATURES registry in `simulator.py`, rather than storing them in the parquet, prevents subtle leakage from feature pre-computation and makes the feature pipeline fully reproducible from raw data.

### 1.2 What Is Wrong

#### FATAL

**F1. SEARCH_SPACE parameters are not wired to rules.py functions.**

`core/config/parameters.py` defines a 22-parameter search space including 9 ICT structure parameters: `swing_n_internal`, `swing_n_external`, `fvg_min_pct`, `ob_refresh_bars`, `sweep_lb`, `cisd_lb`, `displacement_mult`, `displacement_lb`, `mss_lb`. These parameters exist in Optuna's search space but are **never passed** to the actual feature computation functions.

In `simulator.py`, `augment_features()` calls every rules.py function with **no arguments**:
```python
d53_count += _merge(detect_displacement(df))    # no params passed
d53_count += _merge(compute_swing_dual_layer(df))  # no params passed
```

Meanwhile, the functions in `rules.py` **accept** these parameters:
```python
def detect_displacement(df, disp_k=1.5, disp_close_frac=0.75, age_cap=48, ...)
def compute_swing_dual_layer(df, pivot_n_internal=5, pivot_n_external=10)
```

This means:
- Every Optuna trial computes **identical** ICT features regardless of sampled parameters
- The D53 ICT rules overhaul is running on hardcoded defaults only -- never optimized
- Optuna is optimizing label config + regime + sizing, but NOT signal parameters
- The entire "100 trillion grid combinations" claim in CLAUDE.md is misleading -- the ICT parameter dimensions are inert
- **Any "best" Optuna result reflects zero ICT parameter optimization**

This is not a cosmetic bug. It means the ICT signal has been tested at exactly one parameter setting, and the system's claimed search space is substantially smaller than believed.

#### SERIOUS

**S1. Sharpe 12.3 is not comparable to any published benchmark.**

The system computes per-trade Sharpe annualized via `sr_pt * sqrt(trades_per_year)`. With ~178 trades/year and per-trade SR ≈ 0.92, this gives 0.92 × √178 ≈ 12.3.

This computation is mathematically defensible within a narrow interpretation but:
- No published fund, strategy, or benchmark reports Sharpe this way
- Industry standard: aggregate to daily P&L, annualize via √252
- The daily Sharpe ratio has **never been computed** for this system
- Many trading days have zero trades, producing zero-return days that compress Sharpe
- Estimated daily Sharpe: likely **1.5-3.0** (good, but a different universe from 12.3)
- The MIN_SHARPE ≥ 2.0 gate is passing on an incomparable metric -- it should use daily Sharpe
- CLAUDE.md warns about this ("Do NOT interpret Sharpe 12.9 as a live target") but the codebase still uses per-trade Sharpe in all gates and comparisons

**[R1]** The per-trade Sharpe is not technically wrong, but it makes every performance comparison in the project misleading. The daily Sharpe is THE number that determines whether this system is viable in practice.

**S2. Multiple testing without formal correction on the final model.**

The STRATEGY_LOG documents at least 55 decisions across experiments E000-E055+, with at least 28 distinct experiments on the same BTC 2020-2026 dataset. DSR was scaffolded in D52 but **never computed** on the final model. The implications:

- With N ≥ 28 trials, the expected maximum Sharpe from pure noise is approximately √(2 × ln(28)) ≈ 2.56 (per standard order statistics)
- Even if most trials were structural (different architectures, not parameter sweeps), the statistical penalty applies to any "best result" selection
- The holdout (2025-03 to 2026-02) has not been touched, which is the correct mitigation
- But the holdout is a single contiguous period -- it provides one data point, not a distribution
- Optuna trials (D52) add to the trial count: if 500 Optuna trials were planned, the total N could reach 528+

**[R2]** The holdout period covers 2025-03 to 2026-02 -- a period of BTC strength post-ETF approval. A long-only system validated on a bull market holdout proves much less than validation on a mixed or bear period.

**S3. No alpha vs. beta separation.**

BTC went from ~$7,000 in January 2020 to ~$90,000+ by early 2026 -- approximately 12× appreciation over the training + holdout period. For a **long-only** system tested exclusively during this period:

- What fraction of total PnL comes from simply being long BTC?
- What is the system's return vs. buy-and-hold for each walk-forward fold?
- Does the system add alpha during the 2022 bear market, or only during bull phases?
- What happens to system performance if you subtract the risk-free BTC beta?

Zero analysis in the project addresses these questions. A per-fold buy-and-hold comparison is the single most important missing computation.

**[R3]** This is arguably the most important unanswered question in the entire project. A long-only BTC system from 2020-2026 **must** demonstrate it outperforms buy-and-hold to claim any edge whatsoever. The entire validation framework is incomplete without this comparison.

**S4. AUC 0.79 on 5-minute BTC bars warrants scrutiny.**

Published financial ML papers typically report AUC 0.51-0.55 for return prediction on liquid markets. AUC 0.79 is in the range of credit scoring or medical diagnostics, not financial time series. Possible explanations:

- **Label design effect:** Triple-barrier with 2R target and 48-bar horizon creates a binary classification where many bars are "obviously" not going to hit 2R within 48 bars (e.g., during flat, low-volatility periods). The classifier may be primarily learning "is this a high-volatility setup?" rather than "will this setup produce alpha."
- **Feature richness:** 64 features on 5-minute bars with multi-timeframe context is a rich input relative to the classification task.
- **Genuine predictive power:** The signal may actually be this strong, especially if the classification task is narrower than "predict return direction."
- **Subtle leakage:** Despite 18 causality tests passing, there could be leakage vectors not tested (e.g., through HMM regime labels, through multi-timeframe resampling boundaries, through rolling window edge effects).

The AUC is not necessarily wrong, but it demands investigation: specifically, what AUC does a fixed-horizon label (same 48-bar window, simple positive/negative return) achieve with the same features? If that AUC is 0.55-0.60, the gap quantifies how much the triple-barrier label design contributes vs. how much the features contribute.

**S5. Survivorship/selection bias in instrument choice.**

The system has been developed, tested, and validated exclusively on BTC perpetual futures. BTC is the most liquid, most volatile, and most momentum-driven cryptocurrency. There is no evidence that the signal generalizes to:
- Other crypto pairs (ETH, SOL, which were mentioned in planning but never tested)
- Other asset classes (equity index futures, planned but not implemented)
- Different market regimes (the 2020-2026 period is dominated by a single macro cycle)

A signal that works on BTC during a 12× bull run is the lowest possible bar for validation.

#### MODERATE

**M1. ICT methodology contribution is overstated.**

SHAP analysis (D33) claimed 72.3% ICT structural contribution. But D55 pruning eliminated 7/10 D53 ICT feature families with zero AUC loss. The three surviving families are:

| Surviving Family | What It Actually Is |
|-----------------|-------------------|
| Displacement detection | Momentum thrust identification (price moves > k × ATR with directional close) |
| FVG detection | Gap identification (3-bar pattern where bar 1 high < bar 3 low or vice versa) |
| Swing structure (dual-layer) | Pivot point detection at two timescales (5-bar and 10-bar lookback) |

These are standard technical analysis concepts -- momentum, gaps, and pivots -- with ICT terminology. The claim that "ICT methodology drives the edge" is more accurately stated as "momentum + gap + pivot structure drives the edge." This is not a criticism of the features' predictive power; it's a criticism of the attribution. The seven pruned families (breaker blocks, CISD refinements, sweep sequences, MSS, OB anchored, etc.) added no predictive value despite significant implementation effort (D53).

**[R2]** The surviving features should be understood for what they statistically are, not what ICT theory says they are. This reframing matters for NQ/ES extension: "momentum + gap + pivots" is a more transferable concept than "ICT methodology."

**M2. Execution model uses fixed slippage.**

The backtest assumes fixed slippage per trade (0.027R per D30). In reality:
- Slippage varies with volatility, time of day, and order size
- Asian session liquidity is significantly lower than US/EU session
- Slippage should scale with ATR -- volatile periods mean wider spreads
- Market impact becomes non-linear above certain position sizes
- The 0.027R cost was estimated from analysis, not measured from live fills

**[R1]** Should also model slippage as a function of position size for any scaling analysis.

**M3. HMM regime features risk leakage.**

The 3-state Gaussian HMM (D47) is fitted to what data and when? If the HMM is fitted once to the full training window, then at prediction time T, the regime label at T reflects information from bars after T (since the HMM's Viterbi decoding uses the full sequence). Proper implementation requires:
- Online HMM: refit at each prediction point using only data up to T
- Or: walk-forward HMM fitting within each fold, with embargo

The D47 regime filter passed 12/12 causality tests, but the test methodology checks correlation at fixed points -- it may not catch full-sequence leakage from batch Viterbi decoding.

**M4. Expanding window may cause concept drift blindness.**

The expanding window grows monotonically: the first fold trains on 2020-2021, the last fold trains on 2020-2025. This means:
- 2020 COVID crash data has equal weight in all folds
- Pre-ETF market microstructure (2020-2023) dominates the training data even in later folds
- If BTC market behavior changed post-ETF approval (January 2024), the expanding window dilutes post-ETF patterns with 4 years of pre-ETF data
- No comparison with a sliding window has been performed

**[R1]** A sliding window comparison is not optional -- it's a necessary stationarity check.

#### MINOR

**m1. Holdout guard is path-level only.** The simulator checks for "holdout" in the file path but has no row-level guard preventing holdout rows from being included if the full DataFrame is loaded and subseted programmatically.

**m2. No model diversity.** The system uses a single LightGBM model. No comparison with logistic regression, random forest, neural network, or ensemble has been attempted. Single-model systems are more fragile than ensembles.

**m3. Breakeven cost analysis never computed.** At what per-trade cost does the system become unprofitable? This number determines how much execution degradation the system can absorb before failing.

### 1.3 What Is Missing

1. **Daily Sharpe ratio computation.** The single most important missing metric. Must aggregate per-bar or per-trade returns to daily P&L and annualize via √252. Estimated range: 1.5-3.0. This is the number that makes the system comparable to every other systematic strategy in existence.

2. **Buy-and-hold benchmark.** Per fold and overall. Compute: (a) system return vs. BTC return for each walk-forward test fold, (b) system Sharpe vs. BTC buy-and-hold Sharpe, (c) fraction of folds where system outperforms BTC. Without this, the system's "alpha" is entirely unquantified.

3. **Logistic regression baseline.** Train `sklearn.linear_model.LogisticRegression` (L2 regularization, default C) on the same 64 features with identical walk-forward. This answers: "Is the edge in the features or in the model complexity?" If LR achieves AUC ≥ 0.74, the signal is essentially linear and LightGBM's added complexity is a liability.

4. **Fixed-horizon label comparison.** Compute AUC with 48-bar forward returns (binary: positive/negative) using the same features. If triple-barrier AUC >> fixed-horizon AUC, the label design is doing significant work. This isn't necessarily bad, but it must be understood.

5. **DSR computation on the final model.** With honest N ≥ 28 trial count. If DSR < 2.0, the statistical significance of the observed edge is questionable.

6. **NQ/ES out-of-distribution validation.** The only way to test whether the signal generalizes before risking real capital. This must be designed and executed, not sketched.

7. **Sliding window comparison.** Run the identical pipeline with a sliding (fixed-length) training window instead of expanding. If sliding window performance is materially different, this reveals stationarity assumptions.

8. **Cost sensitivity analysis.** Compute system performance at 1×, 1.5×, 2×, 3×, and 5× the estimated execution cost. Identify the breakeven multiplier. If the system becomes unprofitable at 2× costs, the margin of safety is thin.

### 1.4 The Core Question: Is There an Edge?

**Assessment: Probably yes, but significantly smaller than reported, and possibly not distinguishable from buy-and-hold beta.**

**Evidence FOR an edge:**
- AUC 0.79 is well above random (0.50), even accounting for potential label design effects
- 10/10 gate passage including CSCV PBO < 0.05 (0/70 negative OOS) and PSR > 0.99
- Feature pruning to 64 features with no AUC loss -- concentrated, non-random signal
- Short side also achieves AUC 0.7966 -- bidirectional predictive power reduces the "just long bias" explanation
- Signal survived D51 hardening (wider embargo, label purging, contamination marking)
- 18/18 causality tests pass at multiple test points

**Evidence AGAINST (or for a smaller edge):**
- Daily Sharpe never computed -- the industry-comparable metric is unknown
- No buy-and-hold comparison -- alpha vs. beta is entirely unmeasured
- Long-only system on BTC 2020-2026 benefits enormously from secular bull trend
- 7/10 ICT feature families pruned -- the "ICT edge" is narrower than claimed
- AUC 0.79 is unusually high for financial ML -- may partially reflect label design
- 28+ experiments on same data without DSR correction on the final model
- Holdout period is another bull market phase
- SEARCH_SPACE wiring gap means only one parameter setting has ever been tested
- No out-of-distribution validation on any other instrument

**[R3] The honest bottom line:** The system likely captures some predictive signal for BTC price movement on 5-minute bars. Whether that signal exceeds transaction costs and buy-and-hold after proper risk-adjustment -- measured in daily Sharpe on out-of-sample data -- is unproven. The daily Sharpe ratio and buy-and-hold comparison will answer this question definitively. Everything else is secondary.

---

## Part 2: Trading Thesis

BTC and equity index futures exhibit short-term price inefficiencies driven by institutional order flow patterns. Specifically, liquidity-seeking behavior creates predictable sequences: large directional moves (displacement) leave price gaps (FVGs) that act as future support/resistance zones, while swing structure at multiple timeframes defines the directional context. A gradient-boosted classifier trained on these features -- which are essentially momentum, gap, and pivot-point signals -- can identify high-probability entries with favorable risk-reward. This thesis is viable if and only if: (a) the edge is measured in daily-rebalanced, transaction-cost-adjusted terms, (b) the system demonstrably outperforms buy-and-hold, (c) the signal validates out-of-distribution on at least one equity index futures instrument, and (d) position sizing is conservative enough to survive the expected 3-5× degradation from backtest to live performance.

---

## Part 3: Redesigned Architecture

### 3.1 Signal Architecture

**Keep:** The three surviving ICT feature families (displacement, FVG, swing structure) plus the non-ICT features that survived D55 pruning (64 total features).

**Reframe:** Stop calling this "ICT methodology." Call it what it statistically is: a momentum-gap-structure signal. This reframing is important for three reasons:
1. It removes the intellectual overhead of maintaining 7 dead ICT feature families
2. It makes the signal concept transferable to NQ/ES without forcing an "ICT on equities" narrative
3. It enables clearer communication with any future collaborators or reviewers

**Restructure as a 4-tier signal hierarchy:**

| Tier | Function | Features | Source |
|------|----------|----------|--------|
| 0 - Structure | Market context: trend direction, dealing range, premium/discount position | Dual-layer swing structure (N=5 internal, N=10 external), pd_position continuous [0,1] | D53 swing + P/D |
| 1 - Zones | Actionable price zones: FVG zones with age/fill tracking | FVG bull/bear with min_size_atr=0.50, age_cap=100, mitigation tracking | D53 FVG enhanced |
| 2 - Triggers | Entry signals: momentum displacement, OTE proximity | Displacement (k=1.5 ATR, close_frac=0.75), ote_dist_from_705_atr | D53 displacement + OTE |
| 3 - Filters | Regime/volatility/session context | HMM regime, ADX composite, volatility features, session indicators | D47 regime + existing TA |

**[R1] Mandatory tier ablation test:** Train models using (a) only Tier 0+1 features, (b) only Tier 2+3 features, (c) all tiers. This reveals which tiers carry the actual signal and whether the tiers are complementary or redundant. If a single tier carries >80% of signal, the other tiers are noise.

**[R2] Remove dead code:** The 7 pruned D53 families (breaker blocks, CISD refinements, sweep sequences, MSS, OB anchored, etc.) should be removed from the `ONTHEFLY_FEATURES` registry in `simulator.py`. Dead code creates false confidence in system completeness and increases maintenance burden. Archive in a `deprecated/` directory if needed for reference.

**Immediate fix required:** Wire the SEARCH_SPACE parameters through `augment_features()` to the respective `rules.py` functions. This is prerequisite to any meaningful Optuna optimization. The fix requires:
1. `augment_features()` to accept an `ict_params` dict
2. Each rules.py call to unpack relevant parameters from the dict
3. `optuna_optimizer.py` to pass sampled parameters through the experiment config
4. Verification that different parameter values produce different features (add a test)

### 3.2 Labeling

**Decision: Keep triple-barrier labeling, with mandatory comparison tests.**

Why keep it:
- Triple-barrier directly encodes the risk-reward structure (1R stop / 2R target / 48-bar horizon)
- It produces labels that correspond to tradeable outcomes: "did this setup hit 2R profit before 1R loss within 4 hours?"
- The system's entire gate framework (EV, win rate, Sharpe, profit factor) is computed on triple-barrier-defined trades
- Switching to a fundamentally different labeling scheme would require rebuilding all gates, backtesting infrastructure, and accumulated validation results

Why the label design needs scrutiny:
- The 48-bar horizon means the "neither stopped out nor target hit" outcome is common -- what fraction of labels expire at the horizon? If >40%, the triple-barrier is effectively a fixed-horizon label with extra complexity
- The 2R target creates an asymmetric payoff that may make the classification "easier" in AUC terms (the model needs to identify high-volatility-favorable-direction moments, not predict returns)
- The 1R stop is expressed in what units? If ATR-based, the stop adapts to volatility; if fixed-pip, it doesn't. This matters for regime robustness

**Mandatory comparison tests before ANY further development:**

1. **Fixed-horizon AUC comparison:** Compute AUC with 48-bar forward returns (binary: positive/negative) using the same 64 features and identical walk-forward. Record the AUC gap: `triple_barrier_AUC - fixed_horizon_AUC`. If gap > 0.05, the label design is contributing materially to the reported AUC.

2. **Label stability test:** Vary the R-multiple: test at 1.5R/1R, 2R/1R (current), 2.5R/1R, and 3R/1R with the same 48-bar horizon. If AUC varies by > 0.03 across these, the system's performance is fragile to the exact label specification.

3. **Horizon expiry fraction:** Compute what percentage of labels expire at the 48-bar horizon (neither stop nor target hit). If this fraction exceeds 40%, the horizon is doing most of the classification, and the stop/target structure is secondary.

4. **[R1] Meta-labeling test:** Use a simple rule-based entry (FVG active + displacement detected within 5 bars) and have the ML model predict only the probability of success given the entry. This explicitly separates signal generation from signal filtering and tests whether the ML adds value beyond the base signal.

5. **[R2] Horizon sensitivity:** Test at 24, 48, 96, and 192 bars. If optimal AUC shifts substantially with horizon, the system's edge may be horizon-specific rather than direction-specific.

**[R3] What would change the decision:** If the fixed-horizon label achieves AUC ≥ 0.72 (within 0.07 of triple-barrier), consider switching to fixed-horizon for simplicity. If the horizon expiry fraction is >60%, the triple-barrier is adding complexity without value.

### 3.3 ML Pipeline

**Decision: Keep LightGBM as primary, but mandate a logistic regression comparison.**

**Mandatory LR baseline test:**
- Train `sklearn.linear_model.LogisticRegression` with L2 regularization (default C=1.0) on the same 64 features
- Run through identical walk-forward with the same fold structure, embargo, and label purging
- Apply the same 10-gate framework

Interpretation matrix:

| LR AUC | Implication | Action |
|--------|-------------|--------|
| ≥ 0.74 (within 0.05 of LightGBM) | Edge is in features, not model complexity | Strongly consider deploying LR. Linear models are more robust to distribution shift, easier to interpret, and less prone to overfitting. LightGBM's 0.05 AUC advantage is likely overfitting. |
| 0.65-0.73 | LightGBM captures non-linear interactions that matter | Keep LightGBM but investigate which interactions matter via SHAP interaction values. Add monotonicity constraints where directional relationship is known. |
| < 0.65 | LightGBM is doing heavy lifting with non-linearities | Keep LightGBM but increase regularization concerns. The larger the LR→LGB gap, the more the system depends on complex patterns that may not persist. |

**[R1] Sanity check insight:** If LR achieves AUC 0.79 (matching LightGBM), either the features are extraordinarily predictive or there's leakage. LR is much less able to exploit subtle leakage patterns than tree-based models. A matching LR AUC would be both reassuring (robust signal) and suspicious (too good for financial ML).

**LightGBM modifications if retained:**
- Cap `num_leaves` at 15 and `max_depth` at 4 to limit interaction order
- Increase `min_child_samples` to 50 (currently default ~20) to further regularize
- **[R2]** Add monotonicity constraints on features where directional relationship is known:
  - `displacement_bull_strength`: should monotonically increase P(long success)
  - `fvg_bull_active`: presence should increase P(long success)
  - `pd_position`: higher (premium) should decrease P(long success) for mean-reversion setups
- **[R3]** SHAP rank correlation across walk-forward folds is a hard validation gate (AD-17): Spearman rho >= 0.60 required. D55c measured rho=0.82 (PASS). Additionally, run permutation importance on holdout to verify consistency with in-sample SHAP. If rankings diverge significantly, the in-sample SHAP may reflect overfitting.

### 3.4 Validation Framework

**Redesigned validation requirements:**

**Step 1: Compute honest statistics (P0 -- do immediately)**

| Metric | Computation | Pass Criterion |
|--------|------------|----------------|
| Daily Sharpe | Aggregate per-trade returns to daily P&L, annualize × √252 | ≥ 1.0 |
| Buy-and-hold alpha | Per-fold: system return minus BTC return | Positive in ≥ 6/9 folds |
| DSR | Bailey & Lopez de Prado with N = 28 trials | ≥ 2.0 |
| Breakeven cost | Maximum per-trade cost where system remains profitable | ≥ 2× estimated actual cost |

If daily Sharpe < 1.0 or the system loses to buy-and-hold in >3 folds: **stop all development and investigate**. These are fatal findings that cannot be patched.

**Step 2: Walk-forward comparison (P1)**

Run a sliding-window walk-forward alongside the existing expanding-window:
- Sliding window: 24 months training, 6 months testing, rolling forward
- Same features, same gates, same label

| Outcome | Implication |
|---------|-------------|
| Sliding Sharpe >> Expanding Sharpe | Market is non-stationary; expanding window dilutes signal with stale data. Switch to sliding. |
| Expanding Sharpe >> Sliding Sharpe | Historical data helps; signal is relatively stable. Keep expanding but monitor. |
| Similar performance | Stationarity is not a major issue. Keep expanding for simplicity. |

**Step 3: Gate recalibration**

Replace the per-trade Sharpe gate with a daily Sharpe gate:

| Gate | Current | Proposed |
|------|---------|----------|
| MIN_SHARPE | 2.0 (per-trade) | 1.0 (daily) |
| MIN_ALPHA_VS_BH | (not present) | 0.0 (must beat buy-and-hold) |
| MIN_DSR | (not present) | 2.0 (with honest N) |
| **[R2]** MIN_CALMAR | (not present) | 0.5 (return / max drawdown) |

Keep all other gates unchanged.

**Step 4: CSCV reconfirmation**

Rerun CSCV C(8,4)=70 combinations using **daily Sharpe** as the performance metric instead of per-trade Sharpe. If PBO increases significantly, the system's apparent consistency was inflated by the Sharpe computation method.

**[R1]** This is a critical retest. The current PBO = 0% (0/70 negative OOS) was computed on per-trade metrics. It must be reconfirmed on daily metrics.

### 3.5 Execution Model

**Replace fixed slippage with ATR-adaptive model:**

```
slippage_bps = base_bps + volatility_mult * (ATR_5min / close * 10000)
```

Default parameters:
- `base_bps` = 2.0 (exchange fee component)
- `volatility_mult` = 0.5

**Additional execution model improvements:**

| Improvement | Rationale | Effort |
|------------|-----------|--------|
| ATR-adaptive slippage | Volatile periods have wider spreads | 2h |
| Time-of-day adjustment | Asian session: +50% slippage; London/NY overlap: -25% | 1h |
| **[R1]** Position-size impact | $10K-50K: +0 bps; $50K-200K: +1-2 bps; $200K+: +3-5 bps | 2h |
| Funding rate modeling | Apply actual 8-hour funding from historical data per position | 2h |
| Breakeven cost computation | Max cost per trade where system stays profitable | 1h |
| **[R2]** NQ/ES: RTH vs ETH liquidity | ETH: +100% slippage; RTH open/close: +50% | 2h |

**Cost sensitivity analysis (mandatory):**

Run system performance at cost multipliers of 1×, 1.5×, 2×, 3×, and 5× baseline. Report:
- At what multiplier does daily Sharpe drop below 1.0?
- At what multiplier does the system become unprofitable (EV < 0)?
- What is the safety margin between estimated actual costs and breakeven?

If the system becomes unprofitable at 2× costs, the margin of safety is unacceptably thin.

### 3.6 Position Sizing and Risk

**Keep fractional Kelly, recalibrate for honest edge estimate.**

Current: Kelly 1/40, clipped [1%, 2%]. This is extremely conservative, which is appropriate given the unvalidated edge.

**Proposed live ramp-up schedule:**

| Phase | Kelly Fraction | Max Risk/Trade | Required Condition |
|-------|---------------|----------------|-------------------|
| Paper trading | N/A | $0 | Minimum 3 months, 100+ paper trades |
| Live Phase 1 | 1/40 | 0.5% equity | First 50 live trades |
| Live Phase 2 | 1/20 | 1.0% equity | Live daily Sharpe ≥ 50% of backtest daily Sharpe, 100+ live trades |
| Live Phase 3 | 1/10 | 2.0% equity | Live daily Sharpe ≥ 70% of backtest daily Sharpe, 200+ live trades |
| Full | 1/5 | 3.0% equity | 6+ months live, daily Sharpe ≥ 50% of backtest, consistent profitability |

Each phase-up requires DAILY Sharpe measured on LIVE trades, not per-trade Sharpe from backtest.

**Drawdown-based throttle:**
- If live drawdown exceeds 50% of backtest max drawdown → reduce position to Phase 1 sizing
- If live drawdown exceeds 100% of backtest max drawdown → **halt all trading**, investigate
- If drawdown exceeds 150% of backtest max drawdown → assume model failure, do not restart without full re-evaluation

**[R1]** The ramp-up conditions must use daily Sharpe from live trades. Per-trade Sharpe from backtest is not acceptable as a comparison metric at any phase.

**[R3]** Add a "degradation detector": compute a 30-day rolling daily Sharpe on live trades. If this drops below 0.0 (negative), trigger investigation. If it stays below 0.0 for 60 consecutive calendar days, halt trading.

### 3.7 NQ/ES Full Design

**This section is not a future extension. It is the primary out-of-distribution validation for the trading thesis.**

If the signal is "momentum + gap + pivot structure," it should appear in any liquid market with sufficient volatility and institutional participation. NQ and ES are the ideal test cases: liquid, well-studied, with decades of data, and with structural differences from BTC that make them genuine out-of-distribution tests.

#### NQ1! (Nasdaq 100 Futures)

**Data requirements:**
- Source: Databento, Polygon.io, or CME direct (via broker data feed)
- Resolution: 5-minute bars (matching BTC)
- Minimum history: 4 years (2022-2026), must include 2022 bear market
- Sessions:
  - RTH (Regular Trading Hours): 09:30-16:00 ET -- primary analysis window
  - ETH (Extended Trading Hours): 18:00-09:30 ET -- supplementary, lower liquidity
  - Globex overnight: 18:00-09:30 ET (same as ETH)
- Volume: RTH volume only for volume-based features (ETH volume is unreliable)
- Fields needed: OHLCV + number of trades per bar (if available)

**Feature adaptation matrix:**

| BTC Feature | NQ Adaptation | Rationale |
|-------------|--------------|-----------|
| Displacement (k=1.5 ATR) | Recalibrate k -- NQ has different ATR distribution | NQ mean reversion may be stronger; test k=1.0, 1.5, 2.0 |
| FVG detection (min_pct=0.50 ATR) | Likely reduce threshold -- NQ has tighter spreads, smaller gaps | Test min_pct=0.25, 0.50, 0.75 ATR |
| Swing dual-layer (N=5, N=10) | May need larger N -- NQ trends more slowly than BTC | Test N_internal=5,8,13; N_external=10,15,21 |
| pd_position (96-bar lookback) | Likely increase -- NQ trends last longer | Test 96, 192, 384 bars |
| HMM regime (3-state) | Refit entirely -- BTC regimes ≠ NQ regimes | BTC regimes are crash/sideways/rally; NQ regimes are risk-on/risk-off/rotation |
| Funding rate features | **REMOVE entirely** | NQ has no funding mechanism |
| Session features | **REPLACE with RTH/ETH structure** | London SB / NY SB not applicable; replace with opening range, VWAP anchor, etc. |
| 24/7 continuity features | **REMOVE** | NQ has daily gaps; model overnight risk explicitly |

**NQ-specific features to ADD:**

| Feature | Description | Rationale |
|---------|-------------|-----------|
| Opening range (15min, 30min) | High-low of first 15/30 minutes of RTH | Strong predictive feature for NQ; well-documented in academic literature |
| Prior day H/L/C relative | Current price relative to yesterday's high, low, close | Overnight gap context |
| Overnight gap size/direction | (RTH open - prior RTH close) / ATR | Gap fills are tradeable events in NQ |
| Initial Balance range | First hour high minus low of RTH | Narrow IB → breakout; wide IB → reversion |
| **[R2]** VWAP deviation | (price - VWAP) / ATR, cumulative from RTH open | NQ institutional traders anchor to VWAP; deviation is mean-reverting |
| **[R2]** Weekly options expiry effect | Day-of-week indicator for Wed/Fri expiry | Significant gamma exposure affects NQ pinning behavior |
| VIX level/change | VIX close, VIX 1-day change (if data available) | NQ-VIX relationship is structural |

**Label adaptation for NQ:**
- Keep triple-barrier structure but recalibrate:
  - NQ ATR is proportionally smaller relative to price than BTC ATR
  - Stop/target in ATR units: test 1R/2R (same as BTC) and 0.5R/1R (tighter for lower volatility)
  - **[R2]** Horizon: NQ may need longer horizon. Test 48, 96, 192 bars (RTH only, excluding ETH bars)
  - Consider RTH-only bar counting (exclude ETH bars from horizon count) to match the 4-hour effective window

#### ES1! (S&P 500 Futures)

ES is structurally similar to NQ but with key differences:

| Dimension | NQ | ES |
|-----------|----|----|
| Daily ATR range | ~200-400 pts ($4K-$8K/contract) | ~40-80 pts ($2K-$4K/contract) |
| Volatility | Higher | Lower |
| Trend behavior | Momentum-driven (tech leadership) | More mean-reverting |
| Liquidity | Deep | Deepest (most liquid futures globally) |
| Options influence | Significant (weekly expiries) | Dominant (0DTE, gamma exposure) |
| Institutional activity | Tech-focused, growth-sensitive | Broad-market, VWAP + quarter-points |

**ES-specific adaptations beyond NQ:**
- Wider swing detection windows (ES trends are slower and smoother)
- **[R1]** Consider cross-market features if data available: SPX vs VIX divergence, put/call ratio changes
- ES may benefit from quarter-point level features (price relative to round numbers: xx00, xx25, xx50, xx75)
- ES 0DTE options create intraday pinning effects on expiry days -- model as a feature if data available

#### Cross-Instrument Validation Protocol

This is the most important validation in the project. The protocol:

**Step 1: BTC baseline confirmation**
- Confirm BTC daily Sharpe ≥ 1.0 and positive alpha vs buy-and-hold (from Section 3.4)
- If BTC fails these criteria, cross-instrument validation is moot

**Step 2: NQ model development**
- Adapt feature pipeline for NQ (same feature TYPES, recalibrated PARAMETERS)
- Train NQ model on NQ 2022-2025 data, validate on NQ 2025-2026
- Run full 10-gate framework (recalibrated for NQ: different trade frequency, different ATR)

**Step 3: The key test**
- Compare top feature families between BTC and NQ models (by SHAP importance)
- **If the same feature families (displacement, FVG, swing structure) are top features for both BTC and NQ:** This strongly supports a generalizable edge based on market microstructure
- **If completely different features drive NQ:** The BTC edge may be instrument-specific and the "momentum + gap + pivots" thesis needs revision

**Step 4: ES as third confirmation**
- Repeat Step 2 for ES
- Three-instrument confirmation with consistent feature importance would be strong evidence

**Step 5: [R3] Cross-prediction test**
- Train model on BTC data, predict NQ using the same features (renamed appropriately)
- If cross-prediction AUC > 0.52 (above random), there is genuine signal transfer across markets
- This is the strongest possible test of signal generalization

**NQ/ES execution model:**

| Parameter | NQ | ES | BTC Perps |
|-----------|----|----|-----------|
| Commission | ~$4.50/RT per contract | ~$4.50/RT per contract | % based (0.02-0.06%) |
| Tick size | 0.25 pts ($5/tick) | 0.25 pts ($12.50/tick) | $0.10 (perp) |
| Typical slippage | 1 tick ($5) | 0.5-1 tick ($6.25-$12.50) | Variable, Vol-dependent |
| Funding | None | None | ~6-8% annualized on longs |
| Session gaps | Yes (overnight, weekend) | Yes (overnight, weekend) | None (24/7) |
| Margin | SPAN margin (~$16K/contract) | SPAN margin (~$14K/contract) | Exchange-specific (1-100× leverage) |
| Market impact threshold | ~$1M+ notional | ~$5M+ notional | ~$500K+ notional |

### 3.8 Deployment Roadmap

**Phase A: Critical Fixes (Week 1-2)**

| Task | Effort | Success Criterion |
|------|--------|-------------------|
| Fix SEARCH_SPACE → rules.py wiring | 2-3h | Different Optuna parameter samples produce different features (verified by test) |
| Compute daily Sharpe ratio | 1h | Number exists. Target: ≥ 1.0 |
| Compute buy-and-hold comparison per fold | 2h | System beats BH in ≥ 6/9 folds |
| Run DSR with N=28 | 2h | DSR ≥ 2.0 |

**Gate:** If daily Sharpe < 1.0 OR system loses to buy-and-hold in >3/9 folds OR DSR < 2.0 → STOP. Investigate root cause before proceeding. These are potentially fatal findings.

**Phase B: Validation Deepening (Week 3-4)**

| Task | Effort | Success Criterion |
|------|--------|-------------------|
| LR baseline comparison | 3h | Result documented (any outcome is informative) |
| Fixed-horizon label comparison | 3h | AUC gap quantified |
| Sliding window comparison | 4h | Stationarity assessment complete |
| Label stability tests (1.5R, 2R, 2.5R) | 2h | AUC variation < 0.03 across R-multiples |
| Horizon expiry fraction | 1h | Fraction documented |
| Tier-level feature ablation | 4h | Per-tier signal contribution quantified |
| ATR-based slippage model | 3h | Implemented and integrated |
| Cost sensitivity analysis | 2h | Breakeven multiplier documented |

**Gate:** Phase B results may change the ML pipeline (switch to LR), labeling scheme, or window type. Proceed to Phase C with the best validated configuration, not necessarily the current one.

**Phase C: NQ/ES Development (Week 5-8)**

| Task | Effort | Success Criterion |
|------|--------|-------------------|
| NQ data acquisition (4+ years, 5-min) | 2-4h | Data loaded, quality-checked |
| NQ feature adaptation | 1-2 weeks | Feature pipeline running, causality tests pass |
| NQ model training + full gate suite | 3-5 days | Model trained, gates evaluated |
| NQ SHAP analysis | 4h | Top feature families compared to BTC |
| ES data + model (if NQ succeeds) | 1-2 weeks | Second OOD confirmation |
| Cross-instrument feature comparison | 2h | Same/different family analysis |
| **[R3]** Cross-prediction test (BTC→NQ) | 4h | AUC documented |

**Gate:** NQ model passes ≥ 8/12 gates. Same top 3 feature families as BTC. If NQ fails completely, reconsider whether the signal is BTC-specific.

**Phase D: Holdout Test (Week 9)**

This is a ONE-SHOT test. No parameter adjustments after seeing results.

| Metric | Required |
|--------|----------|
| Holdout daily Sharpe | ≥ 50% of walk-forward daily Sharpe |
| Holdout alpha vs buy-and-hold | > 0 (positive) |
| Holdout max drawdown | < 1.5× walk-forward max drawdown |
| Holdout trade count | Within 50-150% of walk-forward expected |

**[R2]** The holdout period (2025-03 to 2026-02) is a BTC bull market period. The buy-and-hold comparison is especially critical here: the system MUST beat buy-and-hold during this period to prove it's not just a leveraged BTC bet.

**Gate:** If any holdout criterion fails → the backtest overfitting hypothesis is confirmed. Do NOT re-optimize on the holdout. Accept the result.

**Phase E: Paper Trading (Week 10-22, minimum 3 months)**

| Monitor | Expected Range | Alarm Trigger |
|---------|---------------|---------------|
| Daily Sharpe (30-day rolling) | ≥ 40% of backtest daily Sharpe | < 0.0 for 30+ consecutive days |
| Trade frequency | 50-150% of backtest frequency | < 50% or > 200% |
| Win rate | Within 5pp of backtest | > 10pp below backtest |
| Max drawdown | < 2× backtest max drawdown | > 1.5× → warning; > 2× → halt |
| Signal-backtest agreement | Signals should match historical patterns | Systematic divergence detected |

**[R3]** Paper trading success criteria (3-month minimum):
- Rolling 30-day daily Sharpe has been positive for ≥ 60 of 90 days
- 100+ paper trades completed
- No halt triggers activated
- Execution characteristics (slippage, timing, fill rates) are documented

**Phase F: Live Capital (Week 23+)**

Start with 1/40 Kelly, $10K-$25K notional. Follow the ramp-up schedule in Section 3.6.

**Kill switches (automated):**
- Max drawdown > 2× backtest max drawdown → halt immediately
- Trade frequency < 50% of expected for 30+ days → halt, investigate signal degradation
- Daily Sharpe < 0 after 100+ live trades → halt
- **[R1]** Any exchange regulatory action affecting BTC perps access → halt
- **[R3]** Rolling 60-day daily Sharpe drops below 0.0 → halt, assume model degradation

**Monitoring cadence:**
- Daily: check P&L, drawdown, trade count
- Weekly: compute weekly Sharpe, compare to backtest expectations
- Monthly: full system review, feature importance drift check, regime analysis
- **[R3]** Quarterly: re-run walk-forward on most recent data to detect edge decay. If new walk-forward daily Sharpe < 50% of original, consider retraining or halting.

---

## Part 4: Implementation Roadmap

| Priority | Task | Dependencies | Effort | Impact |
|----------|------|-------------|--------|--------|
| P0 | Fix SEARCH_SPACE → rules.py wiring | None | 2-3h | Unblocks all Optuna optimization |
| P0 | Compute daily Sharpe ratio | None | 1h | THE critical number |
| P0 | Compute buy-and-hold comparison per fold | None | 2h | Alpha vs beta answer |
| P0 | Run DSR with N=28 | None | 2h | Multiple testing answer |
| P1 | LR baseline comparison | None | 3h | Is model complexity justified? |
| P1 | Fixed-horizon label comparison | None | 3h | Label design contribution |
| P1 | Sliding window comparison | None | 4h | Stationarity check |
| P1 | Label stability tests | None | 2h | Robustness assessment |
| P1 | Tier-level feature ablation | None | 4h | Signal attribution |
| P2 | ATR-based slippage model | None | 3h | Execution realism |
| P2 | Cost sensitivity analysis | ATR slippage | 2h | Margin of safety |
| P2 | NQ data acquisition | None | 2-4h | Data dependency for Phase C |
| P3 | NQ feature adaptation | NQ data | 1-2 weeks | OOD feature pipeline |
| P3 | NQ model training + full gate suite | NQ features | 3-5 days | OOD validation |
| P3 | ES feature adaptation + model | ES data | 1-2 weeks | Second OOD validation |
| P3 | Cross-instrument feature analysis | NQ + ES models | 4h | Signal generalization test |
| P4 | Holdout test (ONE SHOT) | P0-P3 all complete | 2h | Final validation gate |
| P5 | Paper trading deployment | Holdout PASS | 3 months minimum | Live validation |
| P6 | Live deployment | Paper PASS | Ongoing | Capital deployment |

**Critical path:** P0 tasks → P1 tasks → NQ data → NQ model → Holdout → Paper → Live
**Estimated time to live capital:** ~6 months (if no fatal findings)
**Estimated time to first fatal/proceed decision:** ~2 hours (daily Sharpe computation)

---

## Part 5: Risk Register

| ID | Risk | Probability | Impact | Mitigation |
|----|------|------------|--------|------------|
| R1 | Daily Sharpe < 1.0 | 35% | **Fatal** -- system may not be viable | Compute immediately. If confirmed, investigate whether any edge exists after costs. |
| R2 | System doesn't beat buy-and-hold | 30% | **Fatal** for alpha claim | Per-fold comparison. If BH wins >50% of folds, system is a leveraged BTC bet. |
| R3 | DSR < 2.0 (multiple testing) | 25% | **Serious** -- edge may be statistical noise | Accept result honestly. Consider constraining search space to reduce effective N. |
| R4 | LR matches LightGBM AUC (within 0.05) | 30% | **Moderate** -- switch to LR, actually good for robustness | Not fatal; a linear signal is more robust than a non-linear one. Redeploy with LR. |
| R5 | NQ features don't transfer | 40% | **Serious** -- edge may be BTC-specific | Signals may be instrument-specific. Reduce conviction in generalizability. |
| R6 | Holdout fails (daily Sharpe < 50% of WF) | 30% | **Serious** -- backtest overfitting confirmed | Accept result. Do NOT re-optimize on holdout. |
| R7 | Live slippage 2-3× backtest estimate | 25% | **Moderate** -- reduced edge | Paper trade first, measure actual fills, calibrate slippage model. |
| R8 | BTC market regime shift post-deployment | 20%/yr | **Serious** -- model degradation | Regime monitoring, quarterly re-evaluation, automatic halt on degradation. |
| R9 | **[R2]** Holdout is bull market only | 100% | **Moderate** -- validation is one-sided | BH comparison in holdout is essential. NQ validation (which includes 2022 bear) partially compensates. |
| R10 | **[R2]** Fixed-horizon AUC much lower than triple-barrier | 40% | **Moderate** -- AUC is partly label artifact | Not fatal but means reported AUC overstates predictive power. Recalibrate expectations. |
| R11 | **[R3]** Edge decay faster than monitoring detects | 15%/yr | **Serious** -- silent failure | Quarterly walk-forward re-evaluation, rolling Sharpe monitoring, multiple halt triggers. |
| R12 | **[R3]** Exchange risk (hack, insolvency, regulatory) | 5%/yr | **Catastrophic** | Multi-exchange deployment, never keep >30% of capital on any single exchange. |

---

## Part 6: Deployment Decision Criteria

### Must-Pass Gates (ALL required before live capital)

1. **Daily Sharpe ≥ 1.0** (walk-forward, out-of-sample, annualized via √252)
2. **System beats buy-and-hold** in ≥ 6/9 walk-forward folds
3. **DSR ≥ 2.0** with honest N ≥ 28 trial count
4. **Holdout daily Sharpe ≥ 50%** of walk-forward daily Sharpe
5. **Holdout beats buy-and-hold** (positive alpha during holdout period)
6. **NQ model passes ≥ 8/12 gates** (out-of-distribution confirmation)
7. **Paper trading daily Sharpe ≥ 40%** of backtest daily Sharpe after 100+ trades
8. **Paper trading max drawdown < 2×** backtest max drawdown
9. **Breakeven cost multiplier ≥ 2.0×** (50% margin of safety on execution costs)

### Halt Conditions (ANY triggers immediate halt during live trading)

1. Drawdown exceeds 2× backtest max drawdown
2. Daily Sharpe < 0 after 100+ live trades
3. Trade frequency drops below 50% of expected for 30+ consecutive days
4. Rolling 60-day daily Sharpe drops below 0.0
5. Any exchange regulatory action affecting BTC perpetual futures access
6. **[R3]** Quarterly walk-forward re-evaluation shows daily Sharpe < 50% of original

### Hard No-Go Criteria (if ANY is true, do not deploy)

1. Daily Sharpe (walk-forward) < 0.5
2. System loses to buy-and-hold in >5/9 folds
3. DSR < 1.0
4. Holdout daily Sharpe < 0
5. NQ model fails >6/12 gates with completely different top features

---

## Part 7: What Three Rounds of Self-Critique Changed

### Round 1 [R1] -- "Am I Being Too Nice?"

Changes prompted by asking: "Where am I implicitly accepting the system's framing instead of challenging it?"

1. **Made LR comparison mandatory.** The original design kept LightGBM without proving a simpler model couldn't match it. This is the classic ML sin: assuming complexity adds value without testing. The LR comparison is now P1 priority, and the interpretation matrix explicitly states that LR matching LightGBM is a GOOD outcome (more robust signal).

2. **Added label stability test and meta-labeling test.** Changed labeling from "keep triple-barrier" to "keep triple-barrier, but test four mandatory comparison dimensions." The meta-labeling test is particularly important: it tests whether the ML model adds value as a signal filter vs. a signal generator.

3. **Made sliding window comparison mandatory.** Originally listed as "nice to have." Elevated to P1 because the 2020-2026 BTC dataset spans fundamentally different market regimes (pre-ETF, post-ETF, pre-halving, post-halving), and non-stationarity is the default assumption, not an edge case.

4. **Added position-size-dependent slippage.** The original execution model treated slippage as constant regardless of order size, making any scaling analysis meaningless.

5. **Added mandatory tier-level feature ablation.** Without knowing which tiers carry signal, the 4-tier architecture is a taxonomy, not a design.

### Round 2 [R2] -- "Am I Avoiding Hard Truths?"

Changes prompted by asking: "What am I avoiding because it's uncomfortable?"

1. **Confronted holdout contamination.** The holdout period (2025-03 to 2026-02) covers a BTC bull market. A long-only system tested on a bull market holdout is a weak validation. Added explicit buy-and-hold comparison requirement for the holdout test and noted this as a 100% probability risk.

2. **Made fixed-horizon label comparison mandatory.** Originally mentioned as a possibility. Elevated to P1 because the AUC 0.79 might be partially an artifact of label design. If fixed-horizon AUC is 0.60, the gap is important to understand even if it doesn't change the architecture.

3. **Expanded NQ/ES from sketch to full design.** Added: specific feature adaptation tables, session handling, VWAP deviation features, weekly options expiry effects, RTH/ETH bar counting for horizons, and a complete cross-instrument validation protocol with 5 steps.

4. **Added Calmar ratio gate.** Sharpe alone can be gamed by infrequent trading with large wins. Calmar (return / max drawdown) penalizes strategies that achieve returns through excessive risk.

5. **Renamed "ICT methodology" to "momentum + gap + structure."** The surviving features are standard technical analysis concepts with ICT terminology. Honest naming improves transferability to NQ/ES and removes intellectual dependency on ICT theory.

6. **Added horizon sensitivity testing.** The 48-bar horizon was chosen once and never varied. Performance at 24, 96, and 192 bars would reveal whether the edge is horizon-specific.

### Round 3 [R3] -- "What Would Make Me Not Trade This?"

Changes prompted by asking: "If I had to bet my own money, what would I need to see first?"

1. **Identified daily Sharpe as THE critical number.** Everything else in this document is secondary to this one number. The system could pass 10/10 gates and still be unviable if daily Sharpe < 1.0. This realization reorganized the entire priority structure: P0 tasks are now purely about computing this number and the buy-and-hold comparison.

2. **Elevated buy-and-hold comparison to S3 (SERIOUS) severity.** A long-only BTC system from 2020-2026 MUST demonstrate it outperforms buy-and-hold. This was discussed but buried in the first two rounds. It is now the second most important unanswered question after daily Sharpe.

3. **Added cross-prediction test for NQ/ES.** Train on BTC, predict NQ. If AUC > 0.52, there's genuine cross-market signal transfer -- the strongest possible evidence for a generalizable edge. This was missing from the first two rounds entirely.

4. **Added quarterly walk-forward re-evaluation.** The original deployment plan monitored metrics but never re-evaluated the underlying model. Quarterly re-running of walk-forward on the most recent data detects edge decay that daily monitoring might miss.

5. **Added explicit probability estimates to the risk register.** 35% chance that daily Sharpe < 1.0. 30% chance the system doesn't beat buy-and-hold. These are honest estimates, not worst-case scenarios. Most backtested strategies fail these tests.

6. **Added "Hard No-Go" criteria.** Separate from halt conditions, these are findings that should prevent deployment entirely. Daily Sharpe < 0.5 or system losing to buy-and-hold in 5+ folds means the edge doesn't exist, not that it's small.

7. **Confronted the single-instrument fragility.** The entire validation edifice rests on one instrument (BTC) during one macro cycle (2020-2026). NQ/ES isn't diversification -- it's the minimum viable evidence that the signal concept is real. Made this the explicit framing for Part 3.7.

---

*This document was written with the understanding that someone will make real financial decisions based on its content. Every recommendation assumes the edge is smaller than measured and the risks are larger than estimated. The single most important action is computing the daily Sharpe ratio. That one number -- obtainable in approximately one hour of work -- will determine whether the rest of this document is an implementation roadmap or a post-mortem.*
