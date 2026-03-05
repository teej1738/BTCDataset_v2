# ICT Knowledge Base
# core/signals/ict/knowledge.md
# D41a -- Created as Step 6A of THE_PLAN.md
# All SHAP values sourced from core/experiments/shap/ (E002_prune = current best)
# Legacy baseline: legacy/scripts/results/shap_analysis_v2.json (D36, 508 features)
# Last updated: 2026-03-03 (comprehensive update through D50)

---

## SECTION 1: SHAP EVIDENCE HIERARCHY

### Current Best: E002_prune (D43, long-only)

**Model:** LightGBM walk-forward, 11 folds, 48-bar embargo, isotonic calibration
**Dataset:** ~400 features (81 dead dropped), 648,288 bars, OOS AUC = 0.7942
**SHAP method:** LightGBM native pred_contrib, averaged across 11 folds
**Source:** core/experiments/shap/shap_E002_prune.json

### Top 15 Features (E002_prune, long)

| Rank | Feature | Mean |SHAP| | Family | Verdict |
|------|---------|------|--------|---------|
| 1 | ote_dist | 0.2417 | ICT/OTE | KEEP -- #1 alpha, up from #2 in D36 |
| 2 | swing_high_price | 0.2083 | ICT/Core | KEEP -- price level, major jump |
| 3 | ict_ob_bull_age | 0.1802 | ICT/Core | KEEP -- was #1 in D36 |
| 4 | ict_ob_bear_age | 0.1570 | ICT/Core | KEEP -- directional pair |
| 5 | swing_low_price | 0.1488 | ICT/Core | KEEP -- structural support |
| 6 | clv | 0.0999 | Volume/CLV | KEEP -- jumped from #23 in D36 |
| 7 | liq_dist_above_pct | 0.0805 | ICT/Liq | KEEP -- liquidity distance |
| 8 | m15_ict_swing_high | 0.0654 | HTF/ICT | KEEP -- M15 structural |
| 9 | m30_ict_swing_high | 0.0639 | HTF/ICT | KEEP -- M30 structural |
| 10 | ict_fvg_bear_recent_age | 0.0638 | ICT/Core | KEEP -- FVG timing |
| 11 | ict_fvg_bull_recent_age | 0.0594 | ICT/Core | KEEP -- FVG timing |
| 12 | m15_ict_ob_bull | 0.0571 | HTF/ICT | KEEP -- M15 OB presence |
| 13 | m30_ict_ob_bear | 0.0518 | HTF/ICT | KEEP -- M30 OB directional |
| 14 | m30_ict_ob_bull | 0.0502 | HTF/ICT | KEEP -- M30 OB presence |
| 15 | liq_dist_below_pct | 0.0474 | ICT/Liq | KEEP -- liquidity distance |

**Key changes from D36 -> E002_prune:**
- ote_dist: #2 -> #1 (now dominant alpha feature)
- swing_high_price / swing_low_price: entered top 5 (were #14/#12 in D36)
- clv: #23 -> #6 (volume proxy much more important after dead feature removal)
- ict_ob_bull_age: #1 -> #3 (still critical, relatively smaller after pruning)
- stoch_k, ict_market_trend, ict_bos_wick: dropped from top 10

### Short Direction SHAP (E004_short_baseline, D48)

**Source:** core/experiments/shap/shap_E004_short_baseline.json

| Rank | SHORT feature | |SHAP| | vs Long rank |
|------|--------------|-------|-------------|
| 1 | swing_low_price | 0.2134 | #5 long |
| 2 | ote_dist | 0.2115 | #1 long |
| 3 | ict_ob_bull_age | 0.1848 | #3 long |
| 4 | ict_ob_bear_age | 0.1490 | #4 long |
| 5 | swing_high_price | 0.1423 | #2 long |
| 6 | liq_dist_above_pct | 0.1291 | #7 long |
| 7 | clv | 0.0821 | #6 long |
| 8 | m30_swing_low_price | 0.0710 | #29 long |

Key: 8/10 features shared in top 10. Directional swap: shorts prioritize
swing_low (#1) over swing_high (#5) -- structurally intuitive (each direction
cares about levels on its loss side). liq_dist_above_pct rises for shorts
(0.079 long -> 0.129 short). cvd_bar: #26 long -> #11 short.

### Regime Features SHAP (E008_rq7, D50)

7 of 9 regime features are prune candidates. HMM probabilities not useful
as soft ML inputs -- model already captures regime via ATR/momentum/structure.

| Rank | Feature | |SHAP| | Prune? |
|------|---------|--------|--------|
| #51 | atr_percentile_rank | 0.0149 | No (borderline) |
| #61 | ob_bull_age_x_hmm_bull | 0.0109 | No (borderline) |
| #77 | ote_x_regime | 0.0066 | Yes |
| #93 | hmm_prob_bear | 0.0044 | Yes |
| #153 | hmm_prob_bull | 0.0018 | Yes |
| #184 | hmm_prob_calm | 0.0013 | Yes |
| #234 | bb_width_normalized | 0.0007 | Yes |
| #305 | regime_tag | 0.0003 | Yes |
| #385 | fvg_bull_x_trending | 0.00004 | Yes |

### Prune Candidates

- E002_prune: 328/400 features below 0.01 threshold
- E004_short: 339/409 features below threshold
- E008_rq7: 339 features below threshold (includes 7/9 regime features)
- Aggressive prune would retain ~70 features. Safe: ablation confirmed
  AUC delta = 0.000 when dropping 100 lowest (D36).

### Dead Feature Families (stable across all experiments)

- ICT/SilverBullet: avg rank 409, total SHAP ~0 (4 features)
- ICT/KillZones: avg rank 370, total SHAP ~0 (4 features)
- ICT/PO3: avg rank 315, total SHAP ~0.004 (8 features)
- ICT/Macros: avg rank 480, total SHAP ~0 (9 features)
- All `*_mitigated` flags: zero SHAP (14 features)
- All `fund_*`: near-zero SHAP (4 features)
- All `macro_*`: zero SHAP (7 features)

**EXCEPTION:** Silver Bullet and Macro windows NOT final dead until tested
at 1m resolution (current 5m encoding too coarse).

---

## SECTION 2: ICT RULES LIBRARY

### 2.1 Order Blocks (OB)

**Status:** ENCODED + TESTED (rules.py: detect_ob_bull, detect_ob_bear)
**SHAP ranks:** ict_ob_bull_age #1 (0.205700), ict_ob_bear_age #3 (0.147422),
m15_ict_ob_bull #11 (0.061619), m30_ict_ob_bull #15 (0.050651),
m30_ict_ob_bear #18 (0.047979), h1_ict_ob_bull #26 (0.040375),
h1_ict_ob_bear #28 (0.039644)

**Definition:** Last opposite-direction candle before a displacement move.
Bull OB = last bearish candle before bullish displacement.
Bear OB = last bullish candle before bearish displacement.

**Encoded params:**
- displacement_atr: 1.0 (displacement = body > 1.0 * ATR14)
- lookback: 10 bars (search window for OB candle)
- zone: OB high to OB low
- in_zone: 1 when current close is within [ob_low, ob_high]
- age: bars since OB formed (continuous, uncapped)

**Variants to test:**
- OB quality score: weight by displacement magnitude + volume spike
- OB with FVG confluence: OB that also contains an FVG
- Multi-touch OB: number of times price has returned to OB zone
- OB size filter: skip OBs smaller than X * ATR

**Parameter search space:**
- age_threshold: [5, 10, 20, 50, 100, 200]
- min_displacement_atr: [0.5, 1.0, 1.5, 2.0]

**Open questions:** RQ1 ANSWERED: NO. ob_quality_score ranks #99 (SHAP 0.004),
far below raw ob_bull_age #3 (SHAP 0.180). Raw age is sufficient. See D43.

### 2.2 Fair Value Gaps (FVG)

**Status:** ENCODED + TESTED (rules.py: detect_fvg_bull, detect_fvg_bear)
**SHAP ranks:** ict_fvg_bull #17 (0.049119), ict_fvg_bear #19 (0.047653),
h1_ict_fvg_bear #30 (0.037445), m30_ict_fvg_bear #37 (0.030019),
h1_ict_fvg_bull #42 (0.025327), m30_ict_fvg_bull #45 (0.024168)

**Definition:** 3-candle pattern where candle 1 high < candle 3 low (bull) or
candle 1 low > candle 3 high (bear). The gap between candle 1 and candle 3
is the FVG zone.

**Encoded params:**
- gap_threshold: 0.0 (any gap qualifies)
- age_cap: 288 bars (FVG expires after 288 bars)
- mitigation: bull FVG dies when close <= fvg_top (D14 fix)
- in_zone: 1 when close is within [fvg_bot, fvg_top]

**Variants to test:**
- FVG with minimum size: skip gaps smaller than X * ATR
- FVG consequent encroachment (CE): entry at 50% of gap
- FVG age decay: weight inversely by age
- Stacked FVGs: count of overlapping FVGs in same direction

**Parameter search space:**
- min_size_atr: [0.1, 0.2, 0.5, 1.0]
- age_cap: [48, 96, 144, 288, 576]
- entry_level: ["top", "mid", "62pct", "bot"]

**Open questions:** None currently -- FVG is the validated edge generator.

### 2.3 Optimal Trade Entry (OTE)

**Status:** ENCODED + TESTED (rules.py: compute_ote_dist)
**SHAP ranks:** ote_dist #1 (0.2417, E002_prune). Was #2 in D36, now dominant alpha.

**Definition:** Fibonacci retracement zone (61.8%-78.6%) of the most recent
swing high to swing low range. OTE distance measures how far current price
is from the OTE zone center.

**Encoded params:**
- fib_low: 0.618
- fib_high: 0.786
- swing_lookback: inherits from compute_swing_points (pivot_n=5)
- output: distance as fraction of swing range (continuous, signed)

**Variants to test:**
- H1 OTE distance (currently only 5m swings)
- Extended OTE zone (50%-88.6%)
- OTE + OB confluence (OTE zone contains an active OB)

**Parameter search space:**
- fib_low: [0.50, 0.618, 0.62]
- fib_high: [0.786, 0.79, 0.886]
- swing_lookback: [3, 5, 10, 20]
- timeframe_for_swing: [5m, 15m, 30m, 1h, 4h]

**Open questions:** RQ2 (does H1 OTE > 5m OTE?), RQ3 (triple confluence WR > 70%?)

### 2.4 Liquidity Levels

**Status:** ENCODED + TESTED (rules.py: compute_liq_levels)
**SHAP ranks:** liq_dist_above_pct #4 (0.081261), liq_dist_below_pct #16 (0.049314)

**Definition:** Equal highs / equal lows within tolerance (0.1% of price).
These are liquidity pools where stop orders cluster.

**Encoded params:**
- eq_tolerance: 0.001 (0.1% of price)
- lookback: 20 bars for equal high/low detection
- output: binary flags (liq_eq_high, liq_eq_low) + distance columns

**Variants to test:**
- Dynamic tolerance based on ATR
- Liquidity sweep + reversal (Turtle Soup pattern)
- Distance-to-nearest as continuous feature vs binary flag

**Parameter search space:**
- eq_tolerance: [0.0005, 0.001, 0.002, 0.005]
- lookback: [10, 20, 50, 100]

**Open questions:** RQ5 ANSWERED: NO (D49, OI features do not improve model).
RQ10: OI+FVG interaction -- low priority given RQ5 result.
RQ11 (Turtle Soup WR) still open.

### 2.5 Market Structure (Swing Points + BOS + CHoCH)

**Status:** ENCODED + TESTED (rules.py: compute_swing_points, compute_cisd)
**SHAP ranks:** ict_swing_high #5 (0.070042), m30_ict_swing_high #6 (0.068826),
ict_market_trend #7 (0.067806), m15_ict_swing_high #9 (0.066267),
ict_bos_wick #10 (0.061814), ict_swing_low #12 (0.055642),
ict_swing_high_price #14 (0.052767), ict_bos #50 (0.020925)

**Definition:** Swing points = local pivots (n-bar high/low). BOS = break of
structure (price exceeds prior swing). CHoCH = change of character (BOS in
opposite direction to trend). CISD = change in state of delivery.

**Encoded params:**
- pivot_n: 5 (require 5 bars each side for swing confirmation)
- bos: binary (wick or close break)
- bos_wick: binary (break by wick only)
- market_trend: -1/0/+1 derived from BOS sequence
- cisd_bull/cisd_bear: state-of-delivery change signals

**Variants to test:**
- Internal structure (3-bar pivots) vs external (20-bar)
- BOS close-only vs wick-only
- Trend strength: count of consecutive same-direction BOS

**Parameter search space:**
- swing_pivot_n: [3, 5, 8, 13, 20]
- bos_type: ["wick", "close", "both"]

**Open questions:** RQ13 (internal vs external structure)

### 2.6 Premium / Discount

**Status:** ENCODED + TESTED (rules.py: compute_premium_discount)
**SHAP ranks:** ict_discount #13 (0.053334), m30_ict_premium #22 (0.044872),
h1_ict_premium #25 (0.041627), ict_premium #29 (0.039601)

**Definition:** Price position relative to the 50% level of the current
swing range. Premium = above 50% (overvalued), Discount = below 50%.

**Encoded params:**
- swing range from compute_swing_points output
- output: binary flags (ict_premium, ict_discount) + continuous zone measure

**Variants to test:**
- Continuous premium/discount score (distance from equilibrium)
- Multi-timeframe premium/discount alignment

**Parameter search space:** None -- derived from swing points.

**Open questions:** None currently.

### 2.7 Silver Bullet Sessions

**Status:** ENCODED (in dataset, not in rules.py)
**SHAP ranks:** sb_ny_am_et #471 (0.000004), sb_london_et in drop list.
Session family: avg rank 375, total SHAP 0.006278.
ICT/SilverBullet family: avg rank 409, total SHAP 0.000296.

**Definition:** Specific intraday time windows where FVGs form with higher
probability. London SB: 03:00-04:00 ET. NY AM SB: 10:00-11:00 ET.
NY PM SB: 14:00-15:00 ET.

**Encoded params:**
- Binary gate (1 during window, 0 outside)
- Elapsed time within window (sb_*_et)

**Variants to test:**
- 1m resolution encoding (60 features per window vs 1 binary gate)
- Session-specific FVG quality score
- Session + direction interaction features

**Parameter search space:** N/A (time windows are fixed by definition)

**Open questions:** Silver Bullet at 1m (blocked, needs 1m data pipeline)
**NOTE:** Current 5m encoding is too coarse. Do NOT declare dead until 1m test.

### 2.8 ICT Macros (Kill Zones / Time Windows)

**Status:** ENCODED (in dataset, not in rules.py)
**SHAP ranks:** All zero or near-zero. ICT/Macros family avg rank 480,
total SHAP 0.000058 (9 features). ICT/KillZones: avg rank 370, total SHAP 0.000666.

**Definition:** Specific 15-20 minute windows around key times:
- London Open macro: 02:33-03:00, 04:03-04:30 ET
- NY Open macro: 08:50-09:10 ET
- NY Continuation: 10:10-10:50 ET
- Late AM: 11:10-12:00 ET, Late Afternoon: 13:10-14:00 ET

**Encoded params:**
- Binary gate per window
- macro_any: union of all macro windows

**Variants to test:**
- 1m resolution encoding
- Macro + FVG formation event interaction

**Parameter search space:** N/A (time windows are fixed by definition)

**Open questions:** Same as Silver Bullet (blocked, needs 1m data pipeline). Do NOT declare dead until 1m test.

### 2.9 PO3 / AMD (Power of 3 / Accumulation-Manipulation-Distribution)

**Status:** ENCODED (in dataset, not in rules.py)
**SHAP ranks:** ICT/PO3 family avg rank 315, total SHAP 0.003808 (8 features).
Best feature in family: rank 168.

**Definition:** Session-level pattern: accumulation (range), manipulation (false
breakout), distribution (trend move). Encoded as session-relative price position
and phase markers.

**Encoded params:**
- po3_phase: session phase marker
- po3_range: session range relative to prior session
- Various session-level derived metrics

**Variants to test:**
- Cross-session PO3 (Asia accumulation -> London manipulation -> NY distribution)
- PO3 with volume confirmation

**Parameter search space:** Discretionary -- requires manual pattern definition.

**Open questions:** None (low priority, too discretionary for automated encoding).

### 2.10 Market Trend (5m + HTF)

**Status:** ENCODED + TESTED (in rules.py via compute_swing_points + BOS logic)
**SHAP ranks:** ict_market_trend #7 (0.067806).
d1_ict_market_trend: rank ~260s (in drop candidates list).

**Definition:** Directional regime derived from BOS sequence.
+1 = bullish (higher highs, higher lows), -1 = bearish, 0 = neutral.

**Encoded params:**
- Derived from swing point break sequence
- Available at 5m, M15, M30, H1, H4, D1 timeframes

**IMPORTANT:** d1_ict_market_trend is confirmed dead weight as a HARD FILTER
(Config A: 33.3% WR, D16). Valid as an ML feature input (rank ~25-30 range
across HTFs). Do NOT re-add as a mandatory filter. See DE01.

---

## SECTION 3: UNENCODED RULES (priority order)

### 3.1 Breaker Block (TESTED -- NO VALUE)

**Definition:** An order block that gets mitigated (price breaks through it),
then the OB zone flips to act as support/resistance in the opposite direction.

**Status:** ENCODED + TESTED (rules.py: detect_breaker_blocks, D44).
6 features: breaker_{bull,bear}_{age,dist,in_zone}.
All on-the-fly via ONTHEFLY_FEATURES in simulator.py.

**RQ4 ANSWERED: NO (D46).** All 4 breaker features rank #125-#178 in E003_rq4
SHAP analysis. All below prune threshold (|SHAP| < 0.01). Breaker blocks have
no predictive value above raw ob_age. The continuous "time since mitigation"
encoding does not extract signal that the model doesn't already have from
ob_bull_age (#3) and ob_bear_age (#4).

**Recommendation:** Prune breaker features in next aggressive prune cycle.

### 3.2 Balanced Price Range / BPR (MEDIUM PRIORITY)

**Definition:** Overlapping bull FVG + bear FVG in the same price zone.
This creates a "balanced" zone where both sides have unfilled gaps.

**Why medium priority:** FVG bull (#17) and FVG bear (#19) are both in top 20.
BPR is their intersection -- may indicate stronger zones than individual FVGs.

**Encoding plan:**
- Detect overlapping bull + bear FVGs
- bpr_zone: binary (1 when both FVGs overlap at current price)
- bpr_size: size of overlap as fraction of ATR
- bpr_age: bars since BPR formed

**Open questions:** RQ8 (BPR incremental signal above individual FVG?)

### 3.3 Judas Swing (MEDIUM PRIORITY)

**Definition:** False breakout at session open. Price sweeps liquidity in one
direction then reverses. Specifically: NY open false rally before sell-off,
or London open false dip before rally.

**Why medium priority:** h4_sweep is the quality gate for Config B.
Judas Swing is a session-specific variant that may capture higher-quality
sweep events with better timing.

**Encoding plan:**
- Detect session-open sweep: price exceeds prior session high/low within
  first N bars of session, then reverses
- judas_sweep: binary event
- judas_magnitude: size of false move as fraction of ATR
- judas_session: which session (London, NY AM, NY PM)

**Open questions:** RQ12 (Judas Swing vs generic h4_sweep)

### 3.4 Liquidity Void (LOW PRIORITY)

**Definition:** Large single-candle moves that leave no opposing orders.
Similar to FVG but single-candle rather than 3-candle pattern.

**Why low priority:** FVG already captures most of this. Single-candle voids
may be too noisy. Defer until SHAP evidence shows FVG alone is insufficient.

**Encoding plan:** TBD -- defer until SHAP evidence justifies.

### 3.5 Time/Price Theory (LOW PRIORITY)

**Definition:** ICT's framework for when specific price moves are "due" based
on time symmetry and cycle analysis.

**Why low priority:** Too discretionary for automated encoding. No clear
parameterization. Would require extensive manual pattern labeling.

**Encoding plan:** Do not encode. If future manual analysis identifies specific
testable patterns, revisit.

---

## SECTION 4: EXPERIMENT HISTORY

### Legacy experiments (D21-D36, pre-engine)

| Ref | Decision | Config | Key Metrics |
|-----|----------|--------|-------------|
| -- | D21 | Config B baseline | 116 sig, WR 40.52%, EV +0.166R, PF 1.36 |
| -- | D24 | MTF expansion | 176 longs, WR 47.16%, EV +0.365R. H1 best: 58.18% |
| -- | D26 | CSCV validation | PBO 0%, PSR 0.9994, CI [+0.024, +0.706] |
| -- | D28 | ML v1 | AUC 0.7819. Top: ob_age, swings, liq_dist |
| -- | D29 | T2 ML>=0.60 CD=48 | 9,180 trades, WR 67.3%, EV +0.97R |
| -- | D34 | ML v2 retrain | AUC 0.7937 (+0.012). 508 features |
| -- | D35 | Production v2 | WR 65.4%, EV +0.912R, PF 3.51. CSCV ALL PASS |
| -- | D36 | SHAP analysis | 81 dead features identified. AUC delta 0.000 on drop |

### Engine experiments (D40-D50, core/experiments/registry.json)

| ID | Decision | Tier | AUC | WR | EV(R) | MaxDD | Sharpe | Gates | RQ | Verdict |
|----|----------|------|-----|-----|-------|-------|--------|-------|-----|---------|
| E000 | D40 | std | 0.7935 | 65.4% | +0.912 | 12.0% | 8.57 | 9/10 | -- | Validate D35 (ECE fail, expected) |
| E001 | D42 | std | 0.7991 | 64.7% | +0.893 | 10.2% | 8.86 | 9/10 | RQ1 | ob_quality #99, NO |
| E002* | D43 | std | 0.7942 | 76.2% | +1.237 | 7.1% | 12.92 | 10/10 | prune | BEST LONG. 81 dead dropped + calibration |
| E003 | D46 | std | 0.7949 | 75.0% | +1.198 | 7.5% | 12.31 | 10/10 | RQ4 | Breakers #125-178, NO |
| E004_rq5 | D49 | std | 0.7949 | 74.3% | +1.179 | 7.5% | 12.0 | 10/10 | RQ5 | OI not helpful, NO |
| E004_short* | D48 | std | 0.7981 | 71.9% | +1.107 | 6.3% | 10.98 | 10/10 | -- | BEST SHORT. Promoted |
| E005_trq3 | D49 | weekly | 0.7941 | 73.9% | +1.166 | 7.7% | 6.4 | 10/10 | TRQ3 | Weekly viable, YES |
| E007_rq6 | D50 | std | 0.7944 | 75.5% | +1.214 | 6.2% | 7.41 | 9/10 | RQ6 | HMM gate: YES quality, NO freq |
| E008_rq7 | D50 | std | 0.7944 | 75.9% | +1.227 | 7.7% | 12.76 | 10/10 | RQ7 | Regime soft inputs, NO |

*Starred = current best for that direction.

**Key insight (D50):** HMM hard gate (E007) and soft inputs (E008) produce
identical AUC (0.7944). E007's quality improvement comes entirely from
post-model signal filtering, not better probability estimates. Regime is
useful for TRADE SELECTION but not for PROBABILITY ESTIMATION.

**SHAP stability:** Top 5 features are identical across E002/E003/E004_rq5/
E005_trq3/E008_rq7 (long experiments): ote_dist, swing_high_price,
ict_ob_bull_age, ict_ob_bear_age, swing_low_price. Core alpha is robust.

---

## SECTION 5: OPEN RESEARCH QUESTIONS

### Answered (closed)

**RQ1: ANSWERED: NO (D42, E001)**
ob_quality_score ranks #99 (SHAP 0.004), far below raw ob_bull_age #3 (0.180).
Raw age is sufficient. Quality weighting adds no signal. See DE07.

**RQ4: ANSWERED: NO (D46, E003)**
Breaker blocks rank #125-#178. All below prune threshold. Mitigated OBs have
no predictive value above raw ob_age. See DE08.

**RQ5: ANSWERED: NO (D49, E004_rq5)**
OI features do not improve model. AUC identical (0.7949 vs 0.7942 baseline).
OI was merged and tested -- no value. See DE09.

**RQ6: ANSWERED: YES quality / NO frequency (D50, E007_rq6)**
HMM bull gate (hmm_prob_bull >= 0.60): WR 75.5%, MaxDD 6.2%, PF 5.71.
Quality improvement is real (best MaxDD of any experiment). BUT only 62 trades/yr,
below 100/yr minimum gate -> 9/10 PASS (freq fail). Regime is useful for
TRADE SELECTION but not as a primary strategy filter at standard tier.
Note: identical AUC to E008 (0.7944) -- quality comes from post-model filtering,
not better probability estimates.

**RQ7: ANSWERED: NO (D50, E008_rq7)**
7/9 regime features are prune candidates. hmm_prob_bull ranks #153 (SHAP 0.0018),
far below rank 50 diagnostic threshold. LightGBM already captures regime via
ATR/momentum/structure features. Soft regime inputs add no signal. See DE10.

### Open -- Priority 1

**RQ2:** Does H1 OTE distance score higher SHAP than 5m OTE?
- Hypothesis: H1 swings produce more meaningful retracement zones.
- Test: Compute h1_ote_dist from H1 swing points, add as feature, compare.
- Expected: H1 OTE may rank #1-3 due to structural significance.

**RQ3:** Does OTE + OB + FVG triple confluence produce WR > 70%?
- Hypothesis: All three #1/#2/#17 features aligning = strongest signal.
- Test: Filter for bars where all three are active, compute label WR.
- Expected: Very high WR but very low signal count.

### Open -- Priority 2

**RQ8:** BPR: incremental signal above individual FVG?
- Test: Encode BPR (Section 3.2), add as feature, check SHAP delta.

**RQ9:** True tick CVD vs CLV: SHAP comparison.
- Test: Requires tick-level trade data (aggtrades).
- Blocking: Tick data not yet downloaded.

**RQ10:** Rising OI + FVG vs falling OI + FVG directional difference.
- Test: Interaction feature oi_change * fvg_active.
- Deprioritized: RQ5 showed OI alone adds nothing. Interaction may differ
  but evidence is weak. Only pursue if RQ8/RQ9 are exhausted first.

### Open -- Priority 3

**RQ11:** Turtle Soup: sweep + immediate reversal WR.
- Test: Detect equal-high sweep followed by close below within 3 bars.

**RQ12:** Judas Swing vs generic h4_sweep.
- Test: Encode Judas Swing (Section 3.3), compare SHAP vs h4_sweep.

**RQ13:** Internal (3-bar) vs external (20-bar) structure for 48-bar label.
- Test: Compute swing points with pivot_n=3 and pivot_n=20, compare.

**RQ14:** Short-side dedicated model with bear-specific features.
- Status: PARTIALLY ANSWERED by D48 (E004_short).
- D48 showed a full ML short model works: AUC 0.7981, WR 71.9%, 10/10 PASS.
  This used the same feature set as longs (not bear-specific features).
- Remaining question: do bear-SPECIFIC features (inverted premium/discount,
  bear-weighted session rules) further improve short AUC beyond E004_short?
- Priority lowered: E004_short already achieves 10/10 PASS without custom features.

---

## SECTION 6: DEAD ENDS

**DE01: d1_ict_market_trend as mandatory hard filter.**
Config A: 33.3% WR (D16). Ablation: dropping d1_trend improved WR by +3.26pp.
Valid as ML input feature (rank ~25-30 across HTFs). Dead as hard filter ONLY.

**DE02: Short-taking with SYMMETRIC SESSION RULES.**
D17/D25: 25-33% WR across all TFs. Symmetric rules (same filters for both
directions) are dead. UPDATE (D48): ML short model E004_short achieves
AUC 0.7981, WR 71.9%, 10/10 PASS -- proving shorts work via ML even with
the same feature set. The issue was never "shorts don't work" but rather
"symmetric hard-coded session rules don't capture the short edge." ML does.

**DE03: Regime classification as mandatory filter.**
D23-D24: 81% of Config B signals are in HIGH-vol regime (FVG+sweep events
inherently high-vol). UPDATE (D50): HMM hard gate (E007_rq6) improves
quality (MaxDD 6.2%, PF 5.71) but fails frequency (62/yr < 100 min gate).
Regime soft inputs (E008_rq7) add no signal (7/9 prunable). Regime is
useful for TRADE SELECTION (post-model filter) but dead as both hard
pre-model filter AND soft ML input. See D50.

**DE04: Fixed 48-bar cooldown at low signal frequency.**
14 trades/6yr. Untradeable. Only valid at high-frequency ML signal rates.
CD=576 at ML frequency = 180/yr (tradeable).

**DE05: All-time cumulative CVD.**
No predictive meaning per bar at 5m. Use session-reset CVD only.
cvd_bar (session-level) ranks #44 (0.024209). Cumulative CVD would be noise.

**DE06: H4 CE limit entries.**
Median 7.1 ATR below close at signal time. 27% fill rate. Dead.
H1 CE (1.9 ATR, 42%) and M15 CE (1.3 ATR, 50%) are viable for future
limit-entry strategies but not the current market-entry system.

**DE07: ob_quality_score (RQ1: NO, D42, E001).**
Composite OB quality (displacement magnitude + volume spike weighting) ranks
#99 (SHAP 0.004). Raw ob_bull_age at #3 (0.180) is 45x more informative.
Quality scoring adds complexity without signal. Dead.

**DE08: Breaker blocks (RQ4: NO, D46, E003).**
All 6 breaker features rank #125-#178 (all < 0.01 SHAP). Time-since-mitigation
encoding captures nothing that ob_age doesn't already provide. Dead.
Prune in next aggressive prune cycle.

**DE09: OI features as ML inputs (RQ5: NO, D49, E004_rq5).**
OI rate-of-change and OI interaction features produce AUC 0.7949, identical
to baseline 0.7942 within noise. Open interest adds no predictive value
for ICT-style signals. Dead. Deprioritizes RQ10 (OI+FVG interaction).

**DE10: Regime soft inputs (RQ7: NO, D50, E008_rq7).**
All 10 D47 regime features (HMM probs, ADX composite, interactions) tested
as soft ML inputs. 7/9 are prune candidates. hmm_prob_bull ranks #153.
LightGBM already captures regime information through ATR, momentum, and
structural features. Adding explicit regime features is redundant. Dead.

---

## SECTION 7: IMPLEMENTATION NOTES

**Causality:** Every new feature must pass `df[0:T]` vs `df[0:T+1]` test
at T in [1000, 5000, 10000, 50000] before inclusion. Use test_rules.py pattern.

**Annualization:** sqrt(105120) = 324.22 for 5-minute bars (24/7 crypto).
There are 105,120 5-minute bars per year (365.25 * 24 * 12).

**Embargo:** 48 bars minimum between train and test folds. Recalculate if
label horizon changes (currently 48 bars = 4 hours).

**Naming convention:**
- Features: `ict_{rule}_{direction}_{metric}` (e.g., ict_ob_bull_age)
- HTF features: `{tf}_{base_name}` (e.g., h1_ict_ob_bull)
- New v2+ features: follow existing naming, add to feature_catalog_v2.yaml

**AUC promotion threshold:** +0.005 minimum improvement over current production
model. Logloss must also improve (or not worsen by more than 0.001).

**Calibration:** Run reliability diagram after every retrain. If ECE > 0.05,
apply isotonic regression via `core/engine/calibrator.py`.

**Model:** LightGBM default (device="gpu", CPU fallback). CatBoost if
categorical-heavy features dominate. Logistic regression always run as
sanity-check baseline.

**AUC > 0.85 on financial series:** Suspicious. Check for lookahead leak.
Financial data rarely exceeds AUC 0.85 without information leakage.

**CAGR 1711% (D32):** Backtest-compounded at 2% reinvestment. Not a live
forecast. Live performance will be materially lower due to drawdown sequences,
execution variance, and periodic capital withdrawals.

**CD=576 rationale:** 300/yr monitoring cap for manual oversight. CD=288 is
the natural upgrade path if monitoring becomes automated. Retest CD=288 first.

**cp1252 encoding:** NEVER use Unicode box-drawing chars, em-dashes, or arrows
in print statements. Use ASCII equivalents (-, |, ->, --).

---

## SECTION 8: OPTIMIZER BEHAVIOR RULES

### Pre-proposal checklist

Before proposing any new experiment:

1. Read full experiment history (Section 4)
2. Identify highest-SHAP unencoded rule (Section 3)
3. Check open research questions (Section 5, Priority 1 first)
4. Verify no dead end repeated (Section 6) without new hypothesis
5. Always run calibrator.py after retrain. Do not promote any experiment
   with ECE > 0.05 without first attempting isotonic calibration.

### Failure diagnosis

| Symptom | Action |
|---------|--------|
| Low trade count | Reduce threshold by 0.02 OR reduce cooldown by 96 bars |
| Low WR | Raise threshold by 0.02 OR add signal_filter gate (e.g., HMM bull) |
| High drawdown | Reduce cooldown by 96 OR reduce Kelly fraction |
| Low AUC | Add highest-SHAP unencoded feature from Section 3 |
| Logloss worsened | Remove 50 lowest-SHAP features |
| High ECE | Apply isotonic regression calibration (core/engine/calibrator.py) |

### Mutation rules

1. One parameter at a time.
   Exception: new ICT rule = all params set simultaneously.
2. Max 3 variants of same parameter per session.
3. Two consecutive failures on same parameter: mark exhausted, move to next RQ.

### On gate pass

1. Run shap_runner.py. Append results to Section 1.
2. Append experiment to Section 4.
3. Update experiments/registry.json.
4. If AUC delta >= 0.005 vs production: update reports/best_configs.json.
5. If ECE > 0.05: run calibrator.py before promotion.
