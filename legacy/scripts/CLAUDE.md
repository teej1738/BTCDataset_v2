# SELF-MAINTENANCE INSTRUCTIONS
*Read this section before every task. Follow these rules automatically.*

## When to update this file

Update CLAUDE.md at the end of any task where ONE OR MORE of the
following is true:

1. A new script was created and successfully validated
2. A production config value changed (AUC, WR, EV, DD, threshold,
   cooldown, sizing parameters)
3. A new dataset file was created or a new file became the primary
4. A GO/NO-GO decision was made (pass or fail)
5. A new feature family or column group was added to the dataset
6. The NEXT STEPS section needs updating
7. A D-numbered entry was logged in STRATEGY_LOG.md

## How to update this file

- Do NOT rewrite the entire file
- Only update the specific values that changed
- Add new entries to the decision log table (do not edit old entries)
- Update the NEXT STEPS section to reflect what is now pending
- Update the production config section if any metric changed
- Update the dataset section if paths, row counts, or columns changed
- Update the scripts inventory if a new script was created

## When NOT to update this file

- Mid-run script errors that get fixed (wait until task completes)
- Exploratory analyses with inconclusive results
- Helper/utility scripts that do not affect the pipeline
- Any task that fails validation and produces no state change

---

# CLAUDE.md -- BTCDataset_v2 Project Context
# Read this file at the start of every session before writing any code.
# Last updated: 2026-03-03

---

## Project Overview

A Python backtesting engine for BTC perpetual futures using ICT-style structural
signals (FVGs, liquidity sweeps, market structure, order blocks), evaluated on
5-minute BTCUSDT perpetual data from 2020-2026 with multi-timeframe context and
triple-barrier labels. The goal is a deployable, statistically validated strategy
with realistic execution modeling.

**Current stage:** All build steps DONE through D35. Production config validated.
LONG-ONLY, MTF (H4+H1+M15 combined), execution-validated via market entry.
ML pipeline v2: LightGBM walk-forward (11 folds, 508 features), OOS AUC 0.7937.
Production config T2 (ML>=0.60) + CD=576 (48h cooldown):
928 trades, 180/yr, WR 65.4%, EV +0.912R, PF 3.51, MaxDD 12.0%.
CSCV on production config: PBO 0%, PSR 1.0, 7/7 WF windows profitable.
Position sizing: Kelly 1/40, Sharpe 9.85. See D28-D35.

---

## File Structure

```
BTCDataset_v2/
  data/
    raw/          5 files, 893 MB  (1m candles, funding, mark, index)
    resampled/   12 files, 251 MB  (spot+perp x 6 TFs, 22 cols each)
    enriched/    12 files, 692 MB  (spot+perp x 6 TFs, 155 cols each)
    master/       1 parquet + json (648,288 x 402 cols, 250 MB)
    labeled/
      BTCUSDT_5m_labeled_v2.parquet   648,288 x 569 cols, 612 MB (PRIMARY)
      BTCUSDT_MASTER_labeled.parquet  648,288 x 448 cols, 312 MB (v1 fallback)
      BTCUSDT_MASTER_labeled_metadata.json
      feature_catalog_v2.yaml
  scripts/
    # --- Pipeline (data prep) ---
    download_v2.py
    resample_v2.py
    enrich_ict_v4.py          -- FVG mitigation fix (D14)
    build_master_v4.py        -- HTF merge, no-lookahead verified
    generate_labels_v2.py     -- triple-barrier labels (45 label cols)
    enrich_features_v2.py     -- v2 dataset: 122 new TA/ICT cols (D33)
    # --- Backtesting + analysis ---
    baseline_backtest_v2.py   -- Config B filter stack (current baseline)
    trade_analytics.py        -- ablation, decay, session/direction breakdown
    cscv_validation.py        -- CSCV overfitting validation (Step 2)
    regime_classifier.py      -- volatility regime classification (Step 3)
    mtf_signals.py            -- multi-timeframe signal expansion (D24)
    execution_model.py        -- execution reality layer, Config B (Step 4 / D27)
    # --- ML pipeline ---
    ml_pipeline.py            -- LightGBM walk-forward ML, v1 (D28)
    ml_pipeline_v2.py         -- LightGBM walk-forward ML, v2 (D34)
    ml_backtest.py            -- ML trade-level backtest + CSCV (D29)
    # --- Production config ---
    execution_model_t2.py     -- execution costs on ML trade sets (D30)
    cooldown_sweep_t2.py      -- cooldown optimization (D31)
    position_sizing.py        -- Kelly/Fixed/Vol sizing comparison (D32)
    production_validation_v2.py -- v2 production config validation (D35)
    shap_analysis_v2.py       -- SHAP feature importance analysis (D36)
    # --- Docs ---
    STRATEGY_LOG.md           -- full decision log D01-D36
    CLAUDE.md                 -- this file
  results/
    # Baseline + structural
    trade_log_baseline_v3.csv / baseline_v3_summary.json / equity_baseline_v3.html
    cscv_mtf_longs_validation.json / .html
    regime_analysis.json / regime_classification.html
    mtf_signals.json / mtf_signals.html
    # Execution + ML (v1)
    execution_model.json / .html
    ml_pipeline.json / .html
    ml_backtest.json / .html
    ml_oos_probs.npy
    cscv_ml_validation.json
    # Production config (D30-D32)
    execution_model_t2.json / .html
    cooldown_sweep_t2.json / .html
    position_sizing.json / .html
    # ML v2 + production validation (D34-D36)
    ml_pipeline_v2.json / .html
    ml_oos_probs_v2.npy           -- PRIMARY OOS probability file
    production_validation_v2.json / .html
    shap_analysis_v2.json / .html
    shap_top50.csv / shap_bottom50.csv
```

---

## Dataset

| Property | Value |
|----------|-------|
| Base timeframe | 5m perp BTCUSDT |
| Date range | 2020-01-01 to 2026-02-28 |
| Rows | 648,288 |
| Master columns | 402 (155 base + 49x5 HTF + 2 derived) |
| v1 labeled columns | 448 (402 + 45 labels + 1 derived) |
| v2 labeled columns | 569 (448 + 122 new TA/ICT features from D33) |
| Primary file | data/labeled/BTCUSDT_5m_labeled_v2.parquet (612 MB) |
| v1 fallback | data/labeled/BTCUSDT_MASTER_labeled.parquet (312 MB) |
| Feature catalog | data/labeled/feature_catalog_v2.yaml |

### Column naming convention
- Base 5m cols: `ict_*`, `sess_*`, `cvd_*`, `fund_*`, `mark_*`, `basis_*`
- HTF prefixes: `m15_`, `m30_`, `h1_`, `h4_`, `d1_`
- Labels: `label_{long,short}_hit_{1,2,3}r_{12,24,48,96,288}c`
- Derived: `htf_confluence_score` [-5,+5], `htf_d1_spot_trend`

### Key columns used by the strategy
- `h4_ict_fvg_bull` / `h4_ict_fvg_bear` -- H4 FVG active flag
- `h4_ict_bull_liq_sweep` / `h4_ict_bear_liq_sweep` -- H4 liquidity sweep
- `h4_ict_market_trend` -- H4 trend (-1/0/+1)
- `d1_ict_market_trend` -- D1 trend (-1/0/+1)
- `sess_sb_london` / `sess_sb_ny_am` / `sess_sb_ny_pm` -- Silver Bullet windows
- `ict_atr_ratio` -- ATR ratio (current ATR / 20-period mean ATR)
- `ict_day_of_week` -- 0=Mon, 1=Tue, ..., 6=Sun
- `label_long_hit_2r_48c` / `label_short_hit_2r_48c` -- primary labels

### v2 feature families (122 columns, D33)
- `spread_ar`, `funding_regime`, `funding_zscore_v2`, `time_to_funding`, `annualized_funding`
- `rsi_*`, `macd_*`, `stoch_*`, `roc_*` -- momentum (23 cols)
- `ema_*`, `vwap_*`, `supertrend_*`, `adx_*`, `di_*`, `ichi_*`, `mtf_ema_score` -- trend (26 cols)
- `clv`, `mfi_*`, `obv_*`, `cvd_*`, `cmf_20`, `volume_*`, `taker_buy_ratio` -- volume (13 cols)
- `gk_*`, `parkinson_*`, `rs_*`, `hv_*`, `bb_*`, `squeeze_*`, `vol_*` -- volatility (18 cols)
- `sb_*_et`, `macro_*`, `kz_*`, `asia_*`, `po3_*`, `ote_*`, `cisd_*`, `ob_disp_*`, `int_swing_*`, `ict_confluence_v2` -- ICT session/structural (37 cols)

### Known bugs (low urgency, nothing depends on these)
- D15: `bar_start_ts_ms` and `bar_end_ts_ms` are truncated integers. All scripts
  use `bar_start_ts_utc` / `bar_end_ts_utc` (datetime64[ms, UTC]) which are correct.

---

## Production Config: T2 + CD=576 (D29-D35)

The validated production configuration uses ML probability scoring with a cooldown
filter. All metrics are from v2 OOS walk-forward evaluation (D35).

**ML model:**
- LightGBM walk-forward, 11 folds, 48-bar embargo
- v2 dataset: 508 features (v1 had 387)
- OOS AUC: 0.7937 (v1: 0.7819, +0.012)
- OOS probabilities: results/ml_oos_probs_v2.npy

**Signal generation:**
- Threshold: ML prob >= 0.60 (T2)
- Cooldown: 576 bars (48h) between trades
  > CD=576 rationale: The 300 trades/yr cap was set to limit
  > monitoring overhead and maintain capacity for discretionary
  > oversight of each signal. CD=288 (353/yr) showed better v2
  > metrics (+2.0pp WR, +0.061 EV) but was rejected for exceeding
  > this cap. If monitoring becomes automated, CD=288 is the
  > natural upgrade path and should be retested first.
- Hold: 48 bars (4h), 2R target, 1R stop
- Cost model: flat 0.05R per trade (conservative; actual ~0.027R per D30)
- Long-only

**Production metrics (v2, D35):**

| Metric | v2 (D35) | v1 (D31) |
|--------|----------|----------|
| Trades | 928 | 926 |
| Trades/yr | 180 | 179 |
| Win Rate | 65.4% | 66.9% |
| EV (R) | +0.912 | +0.955 |
| Profit Factor | 3.51 | 3.74 |
| Max Drawdown | 12.0% | 10.1% |
| Sharpe (ann.) | 8.57 | 9.05 |

**Position sizing (D32):** Kelly 1/40 (2.5% risk per trade)
- Sharpe 9.85, CAGR 1711%, MaxDD 12.2%
  > Note: CAGR figure is backtest-compounded at 2% risk per trade
  > with full reinvestment. Live performance will be materially
  > lower due to drawdown sequences, execution variance, and
  > periodic capital withdrawals.

**CSCV validation (D35, on v2 production config):**
- PBO: 0% (0/70 negative OOS)
- PSR(SR > 0): 1.0000 (z = +16.12)
- Bootstrap 95% CI: [+0.819, +1.003]
- Walk-forward: 7/7 windows profitable, mean OOS PF 3.55

**Execution costs (D30, at ~1800 trades/yr CD=48 frequency):**
- Per-trade: ~0.027R (funding 0.019R, latency 0.003R, impact 0.005R)
- 0 win/loss flips from execution costs
- At CD=576 (180/yr): annual drag ~4.9R vs ~316R gross profit = 1.5% drag

---

## Config B Filter Stack (D21-D25)

The original structural filter stack, now superseded by ML scoring but still used
as the T3 overlay mask and as a reference baseline.

**Filter stack (applied per-TF to H4, H1, M15 columns):**

```python
# LONG signals (all three TFs)
{tf}_ict_fvg_bull == 1
AND {tf}_ict_{bull}_liq_sweep == 1   # quality gate
AND sess_sb_london == 1              # direction-session routing
AND ict_atr_ratio in [0.8, 1.5]     # medium ATR only
AND ict_day_of_week not in [0, 1]   # exclude Mon/Tue
```

Shorts dropped entirely (D25): 25-33% WR across all TFs, structurally negative EV.

**MTF long-only results (signal-level, 2020-2026):**

| Metric | Combined | H4 | H1 | M15 |
|--------|----------|-----|-----|------|
| Signals | 176 | 77 | 55 | 56 |
| Win Rate | 47.16% | 48.05% | 58.18% | 46.43% |
| EV (R) | +0.365 | +0.456 | +0.698 | +0.346 |
| PF | 1.85 | 1.85 | 2.79 | 1.73 |

Primary label: `label_long_hit_2r_48c` (2R target, 1R stop, 4h horizon)
Break-even WR for 2R: 33.33%
Signals per year: ~29
H1 is the primary TF (58.2% WR, best single cohort)

---

## Key Research Findings (trade_analytics.py + GPT research)

### What works
- **FVG is the sole edge generator** (+6.35pp solo over random)
- **h4_sweep is the quality gate** -- cuts signals 12x but triples EV
- **Direction-session routing** is essential -- wrong session for wrong direction
  flips to negative edge (London SB longs 55% WR, NY PM SB shorts 41% WR)
- **Medium ATR regime** (+0.48R EV) -- low and high ATR are both negative EV
- **Wed-Sun only** -- Mon 28% WR, Tue 17% WR (structural, not noise)
- **FVG microstructure is academically validated** -- liquidity voids and
  stop clustering are documented market microstructure phenomena
- **OTE distance** is 4th most important ML feature (D34) -- validates
  Fibonacci retracement as genuinely predictive for 2R label hits
- **Stochastic K** (#6 importance) adds complementary momentum signal

### What does NOT work
- `d1_ict_market_trend` as a mandatory filter -- confirmed dead weight, reduces
  WR when added to FVG stack (Config A: 33.3% WR, worse than FVG alone)
- Blanket short-taking -- structurally harder in crypto due to positive funding
  (longs pay shorts). Shorts need h4_trend == -1 AND regime gate
- Silver Bullet session applied to both directions -- destroys short edge
- Fixed cooldown with only 116 signals -- produces 14 trades/6yr, not tradeable

### Ablation results (drop-one-filter from full v2 stack)
| Action | N | WR | Delta |
|--------|---|----|-------|
| Full stack | 624 | 36.70% | baseline |
| Drop d1_trend | 1,584 | 39.96% | +3.26pp |
| Drop h4_sweep | 6,144 | 39.42% | +2.72pp |
| Drop h4_fvg | 3,348 | 34.62% | -2.08pp |
| Drop session | 5,232 | 36.93% | +0.23pp |

### Short-side regime conditions (only profitable combos)
- h4_trend == -1 + high vol: 55% WR, +0.60R EV (best)
- NY PM SB only: 41% WR, +0.175R EV
- h4_trend == -1: 39% WR, +0.109R EV

### Signal decay
- STABLE -- slope +0.013/quarter, p=0.128 (not significant)
- 2020-2021 weak (21-31% WR), 2022-2024 strong (37-50%), 2025 settled 37.5%

---

## Validation Status

| Check | Status | Notes |
|-------|--------|-------|
| Timestamp continuity | PASS | Zero gaps in 648k 5m bars |
| Row count consistency | PASS | All stages match exactly |
| HTF no-lookahead | PASS | Verified on FTX crash week Nov 2022 |
| FVG tracking | PASS | Persistence capped, in_zone at H4 boundaries |
| bar_start_ts_ms precision | FAIL | Known bug D15, nothing depends on it |
| Label correctness | PASS | Manual verified at 2 test rows |
| Label win rates | PASS | 1R ~50%, scaling correct across R/horizon |
| Execution reality (D27) | PASS | Sharpe 1.39 after costs, 0 flips |
| ML pipeline v1 OOS (D28) | PASS | AUC 0.78, positive EV at all thresholds |
| ML backtest CSCV (D29) | PASS | T2: PBO 0%, PSR 1.0, CI [+0.94, +1.00], 7/7 WF |
| Execution costs T2 (D30) | PASS | +0.027R/trade, 0 flips, ALL GO |
| Cooldown sweep (D31) | PASS | CD=576 meets all production targets |
| Position sizing (D32) | PASS | Kelly 1/40: Sharpe 9.85, controlled DD |
| ML pipeline v2 OOS (D34) | PASS | AUC 0.794 (+0.012), all thresholds improved |
| Production v2 CSCV (D35) | PASS | PBO 0%, PSR 1.0, CI [+0.82, +1.00], 7/7 WF |
| SHAP analysis (D36) | PASS | 297/508 features prunable, ablation AUC delta=0.0000 |

---

## What NOT to Do

- Do NOT re-add `d1_ict_market_trend` as a mandatory filter -- confirmed dead weight
- Do NOT use a single session rule for both long and short directions
- Do NOT take short signals -- structurally negative EV confirmed across all TFs (D25)
- Do NOT rebuild the pipeline from scratch -- all pipeline scripts are final
  unless a specific bug is found
- Do NOT optimize more filter parameters without multiple-testing controls --
  adding filters inflates in-sample metrics
- Do NOT treat "touched = filled" as a valid execution assumption for FVG entries
- Do NOT model funding as a fixed average -- it must be applied at position level
- Do NOT add more filters to increase quality -- use diversification instead
  (more assets or timeframes) to solve the signal frequency problem
- Do NOT use CE limit orders for H4 signals -- H4 FVG CE is median 7.1 ATR below
  close at signal time, making fills unreliable (27% fill rate). H1 (1.9 ATR, 42%)
  and M15 (1.3 ATR, 50%) CE limits are viable for future limit-entry strategies
- Do NOT use ml_oos_probs.npy (v1) for new work -- use ml_oos_probs_v2.npy
- Do NOT modify v1 parquet or v1 scripts -- they are preserved as reference

---

## Build Plan (completed)

### Step 2: CSCV Overfitting Validation -- DONE
`cscv_validation.py` -- PASSED on MTF long-only (176 trades).
PBO 0%, PSR 0.9994, bootstrap CI [+0.024, +0.706]. See D26.

### Step 3: Regime Classification -- DONE
`regime_classifier.py` -- hypothesis REJECTED. 81% of Config B signals are
in HIGH-vol regime (FVG+sweep events are inherently high-vol). Regime gating
would collapse signal count without improving WR. See D23-D24.

### Step 4: Execution Reality Layer -- DONE
`execution_model.py` -- PASSED on MTF long-only (176 trades).
Market entry: mean R +0.390, PF 1.71, annualized Sharpe 1.39.
Total cost: +0.025R/trade (funding-dominated). 0 win/loss flips. See D27.

### Step 5: Signal Frequency -- SKIPPED
ML model (Step 6) solves signal frequency internally by scoring every bar.
Asset diversification (ETH/SOL) remains an option for future portfolio expansion.

### Step 6: ML Training -- DONE (v1: D28, v2: D34)
`ml_pipeline.py` (v1) -- 387 features, AUC 0.78. See D28.
`ml_pipeline_v2.py` (v2) -- 508 features, AUC 0.7937 (+0.012). See D34.
Top features: order block age, swing points, OTE distance, stochastic K.

### ML Backtest -- DONE (D29)
`ml_backtest.py` -- T2 (ML>=0.60) best: 9,180 trades, WR 67.3%, EV +0.97R, PF 3.82.
CSCV: PBO 0%, PSR 1.0, 7/7 WF windows profitable. See D29.

### Execution Costs on ML -- DONE (D30)
`execution_model_t2.py` -- +0.027R/trade at T2 frequency.
Annual drag ~47R at CD=48 (1800/yr). ALL GO gates passed. See D30.

### Cooldown Optimization -- DONE (D31)
`cooldown_sweep_t2.py` -- CD=576 (48h) selected: 179/yr, WR 66.9%, EV +0.955R.
Only config meeting strict targets (100-300/yr, WR>=65%, EV>=+0.90R). See D31.

### Position Sizing -- DONE (D32)
`position_sizing.py` -- Kelly 1/40 (2.5% risk) selected.
Sharpe 9.85, CAGR 1711%, MaxDD 12.2%. See D32.
> Note: CAGR figure is backtest-compounded at 2% risk per trade
> with full reinvestment. Live performance will be materially
> lower due to drawdown sequences, execution variance, and
> periodic capital withdrawals.

### Dataset Enrichment -- DONE (D33)
`enrich_features_v2.py` -- 122 new columns (momentum, trend, volume, volatility,
ICT session/structural). v2 parquet: 569 cols, 612 MB. See D33.

### ML v2 Retrain -- DONE (D34)
`ml_pipeline_v2.py` -- 508 features, AUC 0.7937 (+0.012 vs v1).
All thresholds improved +1.5-1.9pp WR. See D34.

### Production Validation v2 -- DONE (D35)
`production_validation_v2.py` -- T2+CD576 confirmed on v2 probs.
WR 65.4%, EV +0.912R, PF 3.51, MaxDD 12.0%. CSCV ALL PASS. See D35.

### SHAP Feature Importance -- DONE (D36)
`shap_analysis_v2.py` -- LightGBM native SHAP (pred_contrib) across 11 folds.
Top: ict_ob_bull_age (#1), ote_dist (#2), ict_swing_low (#3). 297/508 features
have |SHAP| < 0.001 (prunable). Ablation: drop 100 -> AUC delta = 0.0000. See D36.

---

## Next Steps

- **D37:** Multi-strategy architecture restructure (see Architecture below)
- **D38:** Strategy B setup (RSI/EMA/Fib in projects/rsi_ema_fib/)
- Future: ETH/SOL perpetuals for portfolio diversification
- Future: Limit-entry strategies using H1/M15 CE fill paths

---

## Architecture (Planned)

```
core/
  data/          shared v2 parquet -- read only
  shared/        evaluation harness, position sizing, data loader
projects/
  ict/           current work migrates here in D37
  rsi_ema_fib/   Strategy B -- separate CC window
legacy/          D01-D35 scripts preserved for reference
```

---

## Decisions Log Summary (D01-D36)

Full details in STRATEGY_LOG.md. Key decisions:

| D | Decision |
|---|----------|
| D08 | Triple-barrier labels: 1/2/3R x 5 horizons (12/24/48/96/288 bars) |
| D10 | Date filter: perp data starts 2020-01-01 |
| D13 | FVG time-based lookback cap (replaces CHoCH reset) |
| D14 | FVG gap-return mitigation: bull dies when close <= fvg_top (not >= ) |
| D15 | bar_start_ts_ms truncation -- known bug, low urgency |
| D16 | h4_fvg is sole edge generator; d1_trend confirmed dead weight |
| D17 | Direction-session routing: longs=London SB, shorts=NY PM SB |
| D18 | Shorts require h4_ict_market_trend == -1 mandatory |
| D19 | ATR regime filter: ict_atr_ratio in [0.8, 1.5] only |
| D20 | Mon/Tue excluded (16-28% WR vs 41-44% Wed-Sun) |
| D21 | Config B selected: FVG + sweep + routing + ATR + no Mon/Tue |
| D22 | Liquidation/OI data gap identified (pre-ML requirement) |
| D23 | Additional labels needed before ML (regime, MAE/MFE, funding-adj) |
| D24 | MTF expansion: H4+H1+M15 combined 290 signals, H1 best TF |
| D25 | Shorts dropped: long-only, H1 primary TF, 176 MTF longs |
| D26 | CSCV passed on MTF long-only: PBO 0%, PSR 0.9994 -- GO |
| D27 | Execution model passed: Sharpe 1.39, cost +0.025R/trade -- GO |
| D28 | ML pipeline v1: LightGBM walk-forward, AUC 0.78, all thresholds positive EV |
| D29 | ML backtest: T2 (ML>=0.60) best, WR 67.3%, EV +0.97R, CSCV ALL PASS |
| D30 | Execution costs on T2: +0.027R/trade, annual drag 47.2 R/yr, ALL GO |
| D31 | Cooldown sweep: CD=576 (48h) selected, 179/yr, WR 66.85%, EV +0.955R |
| D32 | Position sizing: Kelly (1/40) selected, Sharpe 9.85, CAGR 1711% |

> Note: CAGR figure is backtest-compounded at 2% risk per trade
> with full reinvestment. Live performance will be materially
> lower due to drawdown sequences, execution variance, and
> periodic capital withdrawals.

| D33 | Dataset enrichment: v2 parquet, 122 new TA + ICT session features |
| D34 | ML v2 retrain: AUC 0.7937 (+0.012), 508 features, all thresholds improved |
| D35 | Production validation v2: T2+CD576 confirmed, WR 65.4%, EV +0.91R, CSCV ALL PASS |
| D36 | SHAP analysis: ote_dist #2, 297/508 features prunable, ablation AUC delta=0 |

---

## Academic Grounding

ICT concepts map to documented microstructure phenomena:
- FVGs = liquidity voids (empirically documented as predictive)
- Liquidity sweeps = stop clustering + price cascades (Osler; BTC liquidation cascades)
- Session windows = intraday predictability variation (documented in BTC)
- Short underperformance = structural: positive funding regime in BTC perpetuals

What is NOT academically validated: the exact ICT taxonomy. Frame research as
testing microstructure hypotheses, not "proving ICT."

---

## Environment Notes

- Python 3.14, Windows, cp1252 encoding
- NEVER use Unicode box-drawing chars, em-dashes, or arrows in
  print statements -- use ASCII equivalents (-, |, ->, --) to avoid cp1252 errors
- pip packages: pandas, numpy, scipy, plotly, pyarrow, lightgbm installed
- Numba NOT installed -- use vectorized NumPy instead
- All scripts run from: C:\Users\tjall\Desktop\Trading\BTCDataset_v2\scripts\
- Working data dir: C:\Users\tjall\Desktop\Trading\BTCDataset_v2\data\
- Results dir: C:\Users\tjall\Desktop\Trading\BTCDataset_v2\scripts\results\
