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

## Session Protocol (READ EVERY SESSION)

Before starting any task, read:
1. trading-brain/CLAUDE.md
2. trading-brain/STATUS.md
3. trading-brain/TODO.md -- check for CRITICAL/IN PROGRESS items
4. trading-brain/HANDOFF.md (last entry only)

After completing any task, run the session end protocol
defined in trading-brain/SESSION_PROTOCOL.md.
This is mandatory. Not optional. Every session.

trading-brain is the source of truth.
BTCDataset_v2/STRATEGY_LOG.md is the experiment detail.
Both must stay in sync.

---

# CLAUDE.md -- BTCDataset_v2 Project Context
# Read this file at the start of every session before writing any code.
# Last updated: 2026-03-04 (D55b holdout CONFIRMED CLEAN)

---

## Project Overview

A Python backtesting engine for BTC perpetual futures using ICT-style structural
signals (FVGs, liquidity sweeps, market structure, order blocks), evaluated on
5-minute BTCUSDT perpetual data from 2020-2026 with multi-timeframe context and
triple-barrier labels. The goal is a deployable, statistically validated strategy
with realistic execution modeling.

**Current stage:** D54b Optuna search complete. Best config: t=0.70, cd=288.
D54b quick_wins: Sharpe 21.34, WR 80.8%, 10/10 gates. Major improvement over D54a.
D55b_tier1_only: AUC 0.7938, WR 74.9%, EV +1.20R, Sharpe 12.3, 64 features, 10/10 PASS.
D54c_baseline_short: AUC 0.7966, WR 73.2%, EV +1.14R, Sharpe 11.5, MaxDD 5.47%.
670 -> 64 features (90.4% pruned) with zero AUC loss. 7/10 D53 families dead weight.
D51 DONE: Holdout carved (105,121 rows), embargo 288 bars, label purging, has_oi/has_liqs.
D52 DONE: Optuna integration + GT-Score objective + parameters.py search space.
D53 DONE: ICT rules overhaul. 3/10 families survived prune: OTE-705, P/D, dual-swing.
D54b DONE: Optuna search (40+30 trials). t=0.70/cd=288 optimal. Label_config non-viable.
Train: 543,167 rows (2020-01 to 2025-02). Holdout: 105,121 rows (2025-03 to 2026-02).
Embargo: 288 bars (24h) with label purging (AFML Ch.7).
Regime filter: 3-state Gaussian HMM + ADX composite + interactions (10 features, D47).
ML pipeline: LightGBM walk-forward (9 folds), AUC 0.7938. Lean model: 64 features (D55).
Position sizing: Kelly 1/40. See D28-D35 for legacy production config.

---

## File Structure

```
BTCDataset_v2/
  .env                        -- API keys (COINALYZE_API_KEY)
  CLAUDE.md                   -- this file (root copy, primary)
  core/
    __init__.py
    data/
      raw/
        aggtrades/            -- 74 monthly parquet files, 641k bars, 22 MB (D37b)
        oi_metrics/           -- 52 monthly parquet files, 447k rows, 39.9 MB (D37a)
        liquidations/         -- 75 monthly parquet files, 2,229 daily rows (D37c)
        funding/              -- future: granular funding rate data
    engine/                   -- shared backtest engine (D38)
      __init__.py
      labels.py               -- label lookup, validation, parsing
      labeler.py              -- dynamic triple-barrier labeler (D46a)
      fill_model.py           -- fill price computation for limit/market entries (D46a)
      sizing.py               -- Kelly fraction, equity sim
      evaluator.py            -- walk-forward train, CSCV, gates, ECE
      simulator.py            -- experiment orchestrator, auto-retry, registry, tier gates (D46b)
      calibrator.py           -- isotonic regression calibration (D41a)
      optimizer.py            -- experiment proposal engine (D41b, D46b dual-tier)
      optuna_optimizer.py     -- Optuna-based optimizer with GT-Score objective (D52)
      test_d46b.py            -- D46b smoke test (5 tests)
      shap_runner.py          -- SHAP analysis via pred_contrib (D41b)
    signals/
      __init__.py
      ict/
        __init__.py
        rules.py              -- 18 causal ICT signal functions (D39, D42, D44, D53)
        test_rules.py         -- causality + smoke tests for rules.py
        knowledge.md          -- 8-section ICT knowledge base (D41a, updated D50, 696 lines)
        variants.py           -- T1/T2/T3 threshold configs (placeholder)
      regime/
        __init__.py
        hmm_filter.py         -- 3-state Gaussian HMM + ADX composite + interactions (D47)
        test_regime.py        -- causality tests for regime features (D47)
      ta/
        __init__.py
        rules.py              -- TA signal rules (placeholder)
    config/
      __init__.py
      parameters.py           -- SEARCH_SPACE (22 params), PARAMETER_GROUPS (5 groups) (D52)
    experiments/
      __init__.py
      validate_d35.py         -- D40 engine validation (reproduces D35)
      run_d55b_holdout.py     -- D55b holdout evaluation ceremony
      run_d55b_lo_supplement.py -- D55b Lo(2002) autocorrelation supplement
      analytics_batch.py      -- D55c analytics batch diagnostics (6 tests)
      results/                -- experiment output JSONs
      models/
        baseline_d35.npy      -- v2 OOS probs (copied from legacy)
      shap/                   -- SHAP analysis outputs
    reports/
      __init__.py
      best_configs.json       -- production config registry (placeholder)
  data/
    raw/          5 files, 893 MB  (1m candles, funding, mark, index)
    resampled/   12 files, 251 MB  (spot+perp x 6 TFs, 22 cols each)
    enriched/    12 files, 692 MB  (spot+perp x 6 TFs, 155 cols each)
    master/       1 parquet + json (648,288 x 402 cols, 250 MB)
    labeled/
      BTCUSDT_5m_labeled_v3.parquet   648,288 x 594 cols, 676 MB (PRIMARY, D37d)
      BTCUSDT_5m_labeled_v2.parquet   648,288 x 569 cols, 612 MB (v2 fallback)
      BTCUSDT_MASTER_labeled.parquet  648,288 x 448 cols, 312 MB (v1 fallback)
      BTCUSDT_MASTER_labeled_metadata.json
      feature_catalog_v2.yaml
      feature_catalog_v3.yaml         -- 371 features: 147 static + 224 on-the-fly (D37d, updated D53)
  data_pipeline/
    download_oi.py              -- Binance OI metrics downloader (D37a)
    download_aggtrades.py       -- Binance aggTrades -> true tick CVD (D37b)
    download_liquidations.py    -- Coinalyze liquidation data downloader (D37c)
    merge_v3.py                 -- v2 + OI + CVD + liq -> v3 merge (D37d)
  tradingview/
    export_to_pine.py           -- SHAP-to-Pine generator (D45)
    ict_strategy_v1.pine        -- Pine Script v5 indicator (388 lines, D45)
  scripts/                    -- empty, safe to delete manually (VS Code lock)
  STRATEGY_LOG.md             -- full decision log D01-D53 (moved to root D39)
  D53_IMPLEMENTATION_SPEC.md -- full ICT rules overhaul spec (two GPT responses reconciled, 2026-03-03)
  legacy/
    scripts/                  -- all D01-D36 scripts + results (moved from scripts/)
      results/                -- all historical result JSONs, HTMLs, CSVs, NPYs
      CLAUDE.md               -- legacy copy (frozen at D36)
      *.py                    -- 20 Python scripts (D01-D36)
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
| v3 labeled columns | 594 (569 + 25 OI/CVD/liq features from D37d) |
| Primary file | data/labeled/BTCUSDT_5m_labeled_v3.parquet (676 MB) -- FULL (do not use in experiments) |
| Train file | data/labeled/BTCUSDT_5m_labeled_v3_train.parquet -- 543,167 rows (D51) |
| Holdout file | data/holdout/BTCUSDT_5m_holdout_v3.parquet -- 105,121 rows, NEVER TOUCH (D51) |
| Manifest | data/labeled/manifest_v3.json -- SHA256 hashes, row counts, known issues (D51) |
| v2 fallback | data/labeled/BTCUSDT_5m_labeled_v2.parquet (612 MB) |
| v1 fallback | data/labeled/BTCUSDT_MASTER_labeled.parquet (312 MB) |
| Feature catalog | data/labeled/feature_catalog_v3.yaml (371 features: 147 static + 224 on-the-fly) |

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

### v3 feature families (25 columns, D37d)
- `oi_btc`, `oi_usdt`, `toptrader_ls_ratio_*`, `global_ls_ratio`, `taker_ls_vol_ratio` -- OI raw (10 cols)
- `oi_change_1h`, `oi_change_4h`, `oi_change_pct_1h`, `oi_zscore_20` -- OI derived
- `cvd_true_bar`, `cvd_true_daily`, `cvd_true_session`, `cvd_true_zscore` -- true tick CVD (4 cols)
- `liq_long_btc`, `liq_short_btc`, `liq_total_btc`, `liq_ratio`, `liq_cascade_flag` -- liq raw (8 cols)
- `liq_zscore_7d`, `liq_change_1d`, `liq_change_pct_1d` -- liq derived
- `liq_long_usd`, `liq_short_usd`, `liq_total_usd` -- liq USD conversion (3 cols)
- OI NaN before 2021-12-01 (31.6%); CVD NaN 1.2%; Liq NaN 1.1%
- Liq is daily data forward-filled to 5m with +1 day causality shift

### Known issues (D51 FIXED)
- HOLDOUT CARVED (D51): 105,121 rows reserved (2025-03 to 2026-02). NEVER load in experiments.
- EMBARGO FIXED (D51): 48 -> 288 bars (24h) with label purging (AFML Ch.7).
- AVAILABILITY MASKS ADDED (D51): has_oi, has_liqs binary features in augment_features().
- E000-E008 COMPROMISED: All used full data + 48-bar embargo. Must re-run.

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
- OOS probabilities: core/experiments/models/baseline_d35.npy (also legacy/scripts/results/ml_oos_probs_v2.npy)

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
| Engine smoke test (D38) | PASS | 4 modules, GPU training, full pipeline end-to-end |
| ICT signal causality (D39+D44+D53) | PASS | 18/18 functions causal at T in [1k, 5k, 10k, 50k] |
| Engine validation (D40) | PASS | Reproduces D35: 7/7 metrics within tolerance |
| Knowledge base (D41a) | PASS | 643 lines, 8 sections, all SHAP values verified |
| Optimizer + SHAP (D41b) | PASS | Proposes RQ1, imports clean, all functions verified |
| OB quality causality (D42) | PASS | 10/10 functions causal, on-the-fly augmentation works |
| E002_prune + calibration (D43) | PASS | 10/10 gates, ECE 0.124->0.016, WR 76.2%, EV +1.24R |
| SHAP E002_prune (D43) | PASS | ote_dist #1, ob_quality #99 (prune candidate), 326/396 prunable |
| Dataset merge v3 (D37d) | PASS | 594 cols, 25 new (OI+CVD+liq), causality gate ALL PASS |
| TradingView export (D45) | PASS | Pine v5, 388 lines, 5 SHAP conditions, no lookahead |
| E003_rq4 breaker blocks (D46) | PASS | 10/10 gates, AUC 0.7949, breaker SHAP #125-#178 (prune). RQ4: NO |
| Dynamic labeler (D46a) | PASS | ATR stop 100% match (long+short), fill model 83% fill rate, 3/3 smoke PASS |
| Engine integration (D46b) | PASS | 5/5 smoke: labeler 84.3% match, fills work, tier gates correct, simulate_with_fills OK |
| E004_rq5 OI features (D49) | PASS | 10/10 gates, AUC 0.7949, WR 74.3%, EV +1.18R. RQ5: NO (OI not helpful) |
| E005_trq3 weekly tier (D49) | PASS | 10/10 gates (weekly), AUC 0.7941, WR 73.9%, 52/yr. TRQ3: YES (viable) |
| Short baseline (D48) | PASS | E004: AUC 0.7981, WR 71.9%, EV +1.107R, MaxDD 6.28%, 10/10 gates, SHAP done |
| Regime filter (D47) | PASS | 12/12 causality PASS (HMM+ADX+interactions), 10 features, pure NumPy HMM |
| Regime RQs + signal_filter (D47b) | PASS | RQ6/RQ7 added, signal_filter mechanism in simulator.py, backward compatible |
| E007_rq6 HMM hard gate (D50) | PASS* | 9/10 gates (62.4/yr < 100 min). WR 75.5%, MaxDD 6.2%. Passes weekly tier. RQ6: YES quality, NO frequency |
| E008_rq7 regime soft inputs (D50) | PASS | 10/10 gates, AUC 0.7944, WR 75.9%, EV +1.23R. 7/9 regime features prunable. RQ7: NO |
| Dataset hardening (D51) | PASS | Holdout 105,121 rows, embargo 288, label purging, has_oi/has_liqs, manifest SHA256 verified |
| Optuna smoke test (D52) | PASS | 7/7 tests: GT-Score, JournalStorage, 3 dry-run trials, pruner, sampler, DSR, build_experiment |
| ICT rules overhaul (D53) | PASS | 18/18 causality, swing lag PASS, 224 on-the-fly features, 6 new + 4 modified functions |
| D54a long baseline | PASS | 10/10 gates, AUC 0.7933, WR 76.0%, EV +1.23R, Sharpe 12.8. Regression-safe vs E002_prune |
| D54c short baseline | PASS | 10/10 gates, AUC 0.7966, WR 73.2%, EV +1.14R, Sharpe 11.5, MaxDD 5.47%. E004 superseded |
| D54b Optuna search | PASS | quick_wins: 7/40 pass 10/10, best t=0.70/cd=288 Sharpe 21.34. label_config: 67% fail rate, defaults optimal. ID collision bug fixed |
| D55 feature prune | PASS | 670->64 features (90.4% pruned), AUC 0.7938, 10/10 PASS. 7/10 D53 families dead weight |
| **D55b holdout** | **PASS** | **CONFIRMED CLEAN: 5/5 gates. AUC 0.7993 (train 0.7938), WR 76.8%, EV +1.26R (pre-cost), Daily Sharpe 10.71 (Lo-adj 14.05 WITHDRAWN -- invalid for sparse trading), MaxDD 4.65%. BTC -20.6% (BEAR). Signal validated; trade structure not validated under correct costs** |
| D55c analytics batch | PASS | 6 diagnostics: DSR=1.0 (N=100), T-M alpha significant (p<0.0001)/gamma n.s., expiry 0.4%, SHAP rho=0.82, post-ETF WR +5.5pp, +273pp vs B&H |

---

## What NOT to Do

- Do NOT use SQLite for Optuna parallel optimization -- use JournalFileStorage
- Do NOT use raw OOS Sharpe as optimization objective -- use GT-Score composite
- Do NOT optimize ICT parameters without correcting the defaults first (see D53)
- Do NOT treat Silver Bullet or Power of 3 as first-class ML features -- both are
  overfit magnets. Time-window strategies are the highest-risk ICT concepts.
- Do NOT ignore funding rate drag. Positive funding ~6-8% annualized (322/365 days
  in 2024). Must deduct actual 8-hour funding from live EV projections.
- Do NOT interpret Sharpe 12.9 as a live target. Annualization is sqrt(105,120)=324x.
  Per-bar SR of 0.04 produces Sharpe 12.9. Realistic live: 40-60% haircut.
  After funding drag, expect Sharpe 4-6 live. Compute DSR with N=9 experiments.
- Do NOT re-add `d1_ict_market_trend` as a mandatory filter -- confirmed dead weight
- Do NOT use a single session rule for both long and short directions
- SHORT-SIDE UPDATE (D54c): D54c_baseline_short 10/10 PASS, AUC 0.7966, WR 73.2%.
  ML-scored shorts are VIABLE (unlike structural Config B shorts which were -EV in D25).
  E004 superseded (compromised: full data + 48-bar embargo). 7/10 SHAP features shared
  with long -- short underperformance is structural (market), not signal quality.
- Do NOT trust Optuna results with limit entry types (limit_fvg_edge, limit_ob_mid) --
  the fill model creates survivorship bias (WR=100%, Sharpe overflow). Guard against
  degenerate results before running label_config studies.
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

### Project Restructure -- DONE (D37)
Folder restructure: legacy/scripts/ (all D01-D36 work), core/ (new development),
data_pipeline/, tradingview/. Baseline probs copied to core/experiments/models/.
No code changes, no metric changes. See D37.

### Shared Backtest Engine -- DONE (D38)
`core/engine/` -- 4 modules: labels.py, sizing.py, evaluator.py, simulator.py.
Walk-forward LightGBM (GPU+fallback), CSCV, 10 gates, Kelly sizing, auto-retry.
Smoke tested: 10-feature toy model, full pipeline end-to-end on GPU. See D38.

### ICT Signal Migration -- DONE (D39)
`core/signals/ict/rules.py` -- 9 causal functions ported from legacy enrich_ict_v4.py.
compute_swing_points, detect_ob_{bull,bear}, detect_fvg_{bull,bear}, compute_ote_dist,
compute_liq_levels, compute_premium_discount, compute_cisd.
All 9/9 causality tests PASSED at T in [1000, 5000, 10000, 50000]. See D39.

### OI Metrics Download -- DONE (D37a)
`data_pipeline/download_oi.py` -- 1,553 days from data.binance.vision, SHA256 verified.
447,124 rows (2021-12 to 2026-03), 11 cols (OI, ratios, 4 derived features).
52 monthly parquets in core/data/raw/oi_metrics/, 39.9 MB total. See D37a.

### Engine Validation -- DONE (D40)
`core/experiments/validate_d35.py` -- reproduces D35 production config using pre-computed
OOS probs (no retraining). All 7 metrics within tolerance (WR 65.41%, EV +0.912R,
PF 3.51, MaxDD 11.96%, 180/yr, PBO 0%, PSR 1.0). E000 written to registry. See D40.

### ICT Knowledge Base -- DONE (D41a)
`core/signals/ict/knowledge.md` -- 8-section knowledge base (643 lines).
SHAP hierarchy, rules library, unencoded rules, experiment history, research questions,
dead ends, implementation notes, optimizer behavior rules.
`core/engine/calibrator.py` -- isotonic regression calibration (PAVA, no sklearn).
ECE gate: 0.05. Production model ECE = 0.125 requires calibration before promotion.

### Optimizer + SHAP Runner -- DONE (D41b)
`core/engine/optimizer.py` -- experiment proposal engine (~850 lines). Checkpoint mode
(wait for y/n) and autonomous mode (up to 5 experiments). Reads knowledge.md + registry.json.
14 research questions (RQ1-RQ14) in 3 priority tiers. Mutation rules, failure diagnosis,
feature availability checking. First proposal: E001_rq1 (OB Quality Score).
`core/engine/shap_runner.py` -- SHAP analysis via LightGBM pred_contrib (~350 lines).
Retrains walk-forward folds, aggregates SHAP across folds, compares top 30 vs previous run,
prune threshold 0.010 with 6 regime-dependent features protected. See D41b.

### True Tick CVD Pipeline -- DONE (D37b)
`data_pipeline/download_aggtrades.py` -- streams monthly aggTrades from data.binance.vision.
74/74 months (2020-01 to 2026-02), 3.1B trades -> 640,664 bars, SHA256 verified.
Output: core/data/raw/aggtrades/aggtrades_cvd_YYYY-MM.parquet (5 cols per file).
Pearson(cvd_bar, cvd_true_bar) = 0.5330 (expected range). 53 min runtime. See D37b.

### Liquidation Data Download -- DONE (D37c)
`data_pipeline/download_liquidations.py` -- Coinalyze API daily liquidation data.
2,229 daily rows (2020-01-25 to 2026-03-02), 9 cols, 75 monthly parquets.
Values in BTC (USD conversion deferred to merge -- convert_to_usd only works from 2022-01).
Binance um liquidationSnapshot: does not exist (404 all dates). Skipped.
35 cascade events detected (1.6% of days). COVID crash (2020-03-12) validated. See D37c.

### Dataset Merge v2 -> v3 -- DONE (D37d)
`data_pipeline/merge_v3.py` -- left join v2 + OI + CVD + liq data.
648,288 rows x 594 cols (569 + 25 new), 676 MB. v2 preserved unchanged.
OI (10 cols, 68.4% coverage), CVD (4 cols, 98.8%), Liq (11 cols, 98.9%).
Liq: daily forward-fill with +1 day causality shift, BTC->USD via close price.
Causality gate: ALL PASS (4 tests). Feature catalog v3: 1,064 lines.
RQ5/RQ10 now UNBLOCKED (OI features in v3). See D37d.

---

## Next Steps

**BTCDataset_v2 is FROZEN as of 2026-03-04.**

D51 (holdout), D52 (Optuna), D53 (ICT overhaul), D55b (holdout eval),
D55c (analytics batch) all complete. No further development in this repo.

**All new work happens in Foundation** (C:/Users/tjall/Desktop/Trading/foundation).
See trading-brain STATUS.md for current task.

---

## Architecture (D37)

Project restructured in D37. See File Structure section above for full layout.

- `core/` -- new development: signals, engine, experiments, reports
- `data/` -- unchanged: all parquet/raw data stays in place
- `legacy/scripts/` -- all D01-D36 scripts preserved for reference
- `data_pipeline/` -- download_oi.py (D37a), download_aggtrades.py (D37b), download_liquidations.py (D37c), merge_v3.py (D37d)
- `tradingview/` -- future: Pine Script indicators

---

## Decisions Log Summary (D01-D53)

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
| D37 | Project restructure: legacy/scripts/, core/, data_pipeline/, tradingview/ |
| D38 | Shared engine: core/engine/ (labels, sizing, evaluator, simulator), smoke tested |
| D39 | ICT signal migration: 9 causal functions in core/signals/ict/rules.py, all causality PASS |
| D40 | Engine validation: reproduces D35 (7/7 metrics match), E000 written to registry |
| D41a | ICT knowledge base (643 lines, 8 sections) + calibrator.py (isotonic regression) |
| D37a | OI metrics downloaded: 1,553 days, 447k rows, 52 monthly parquets, 2021-12 to 2026-03 |
| D37b | True tick CVD pipeline: download_aggtrades.py, 74 monthly files, Pearson vs CLV = 0.50 |
| D37c | Liquidations pipeline: Coinalyze daily, 2,229 rows (2020-01 to 2026-03), BTC values, Binance um=404 |
| D37d | Dataset merge v2->v3: 594 cols (+25 OI/CVD/liq), 676 MB, causality ALL PASS, RQ5/RQ10 unblocked |
| D41b | Optimizer + SHAP runner: optimizer.py (850 lines), shap_runner.py (350 lines), RQ1 first |
| D42 | Feature engineering: compute_ob_quality() in rules.py, on-the-fly augmentation, E001_rq1 ready |
| D43 | E002_prune: 81 dead dropped + calibration, 10/10 PASS. SHAP: ote_dist #1, ob_quality #99 (prune). RQ1: NO |
| D44 | Breaker block encoding: detect_breaker_blocks() in rules.py, 6 on-the-fly features, 11/11 causality PASS |
| D45 | TradingView export: Pine v5 indicator (388 lines), 5 SHAP confluence conditions, export_to_pine.py generator |
| D46 | E003_rq4: breaker blocks tested, all below prune threshold. RQ4: NO. 10/10 PASS but no promotion over E002 |
| D46a | Dynamic labeler + fill model: labeler.py (350 lines), fill_model.py (142 lines). ATR 100% match, 3/3 smoke PASS |
| D46b | Engine integration: label_config + fill model wired into simulator, dual-tier (standard/weekly/monthly), 3 TRQs, 5/5 smoke PASS |
| D49 | Weekly tier + RQ5: E004_rq5 (OI, 10/10 PASS, RQ5: NO), E005_trq3 (weekly, 10/10 PASS, 52/yr, TRQ3: YES) |
| D48 | E004_short_baseline: AUC 0.7981, WR 71.9%, EV +1.107R, MaxDD 6.28%, 10/10 PASS. SHAP: 8/10 top features shared, swing_low #1. PROMOTED |
| D47 | Regime filter: 3-state Gaussian HMM + ADX composite + 3 interactions. 10 features, pure NumPy, 12/12 causality PASS |
| D47b | Regime RQs: RQ6 (HMM hard gate) + RQ7 (soft inputs). signal_filter support in simulator.py. Old RQ6/RQ7 replaced |
| D50 | Regime experiments: E007_rq6 (HMM gate, 9/10 PASS, RQ6: YES quality/NO frequency), E008_rq7 (soft inputs, 10/10 PASS, RQ7: NO). 7/9 regime features prunable |
| D51 | Dataset hardening: holdout 105,121 rows (2025-03 to 2026-02), embargo 288 bars, label purging, has_oi/has_liqs masks, manifest_v3.json, holdout guard. E000-E008 COMPROMISED |
| D52 | Optuna integration: optuna_optimizer.py (814 lines), parameters.py (22 params), GT-Score objective, DSR, 7/7 smoke PASS |
| D53 | DONE: ICT rules overhaul per D53_IMPLEMENTATION_SPEC.md. 6 new functions + parameter fixes. 205 new cols. ONTHEFLY 17->224. 18/18 causality PASS. Swing lag check PASS. Ready for Optuna search |
| D54a | DONE: Post-D51/D53 long baseline. 670 features, AUC 0.7933, WR 76.0%, EV +1.23R, Sharpe 12.8, 10/10 PASS. Regression-safe vs E002_prune. SHAP: ote_dist_from_705_atr NEW #15. E000-E008 superseded |
| D54c | DONE: Post-D51/D53 short baseline. AUC 0.7966, WR 73.2%, EV +1.14R, Sharpe 11.5, MaxDD 5.47%, 10/10 PASS. E004 superseded. SHAP: 7/10 shared with long, ote_dist_from_705_atr #8 |
| D55 | DONE: Aggressive feature prune. 670->64 features (90.4% pruned), AUC 0.7938 (within 0.002). D55b_tier1_only 10/10 PASS. 7/10 D53 families dead weight. 3 survived: OTE-705, P/D, dual-swing |
| D54b | DONE: First Optuna search. quick_wins (40 trials): t=0.70/cd=288 best (Sharpe 21.34, WR 80.8%, 10/10). label_config (30 trials): 67% failure rate, defaults confirmed optimal. Registry ID collision bug found+fixed |
| D55b-HO | DONE: Holdout evaluation CONFIRMED CLEAN. 5/5 gates PASS. AUC 0.7993 (train 0.7938), WR 76.8%, EV +1.26R (pre-cost), Daily Sharpe 10.71 (Lo-adj 14.05 WITHDRAWN), MaxDD 4.65%. BTC -20.6% (BEAR). Signal validated; trade structure not validated under correct costs |
| D55c | DONE: Analytics batch (6 diagnostics). DSR=1.0 at N=100. T-M alpha=+0.0107 (p<0.0001), gamma n.s. (pure skill, no timing). Expiry 0.4%. SHAP rho=0.82. Post-ETF WR 67.3% > pre-ETF 61.8%. Strategy +273pp vs B&H in bear |

---

## ICT Parameter Corrections (D53 -- full spec in D53_IMPLEMENTATION_SPEC.md)

Two independent GPT research responses were reviewed and reconciled (2026-03-03).
The table below reflects the final authoritative resolved values.

| Parameter | Old / Missing | Resolved | Notes |
|-----------|--------------|----------|-------|
| displacement_k | 2.0x ATR | 1.5x ATR | close must be in top/bottom 25% of range |
| ob_disp_search | not implemented | 30 bars | window backward from BOS to find displacement |
| ob_anchor_search | not implemented | 20 bars | window backward from displacement to find OB candle |
| ob_zone | unknown | hybrid | bull OB: [low, max(o,c)]; bear OB: [min(o,c), high] |
| ob_mitigation | wick-touch | 50% penetration | MIT_P=0.5; fresh->mitigated state |
| ob_age_cap | 864 bars | 200 bars | ~16.7h -- one session cycle |
| ob_state | not tracked | 3-state | 1=fresh, 2=mitigated; remove on close-through far edge |
| fvg_min_size_atr | 0.10 | 0.50 | current value encodes noise as signal |
| fvg_age_cap | 288 bars | 100 bars | ~8h |
| fvg_ifvg_age_cap | not implemented | 144 bars | ~12h; inverted FVGs are weaker |
| swing_n_internal | 3 | 5 | 25 min confirmation lag; used for entry timing, CHoCH |
| swing_n_external | not implemented | 10 | 50 min lag; used for dealing range, major structure |
| sweep_m | not implemented | 2 bars | close-back confirmation window after wick-through |
| sweep_eq_tolerance_atr | 0.20 | 0.10 | tighter EQH/EQL clustering |
| mss_k | not implemented | 3 bars | displacement look-back window for CHoCH -> MSS |
| cisd_min_run | 1 candle | 2 candles | prevents single doji triggering CISD |
| breaker_age_cap | 576 bars | 200 bars | ~16.7h |
| ote_fib_low | 0.62 | 0.618 | correct Fibonacci level |
| ote_fib_high | 0.79 | 0.786 | correct Fibonacci level |
| ote_705 | not encoded | explicit feature | ote_dist_from_705_atr = abs(pd_pos - 0.705) / atr |
| pd_encoding | +1/0/-1 discrete | continuous 0-1 | pd_position = (close-swing_low)/(swing_high-swing_low) |
| pd_lookback | 50 bars | 96 bars | ~8h; uses ext_swing_high/low_price |

**New functions to add (not currently in rules.py):**
- detect_displacement()       14 cols per direction; prerequisite for OB anchor, FVG tag, MSS
- detect_sweep()              BSL/SSL events, unswept level distances (8 cols)
- detect_sweep_sequence()     3-step ICT composite: sweep -> displacement -> FVG (4 cols)
- compute_swing_dual_layer()  internal N=5 + external N=10, independently (22 cols)
- detect_mss()                CHoCH + displacement within k=3 bars (7 cols)

**D53 implementation priority order (F before A -- displacement is prerequisite):**
  F -> A -> G -> H -> D -> B -> C -> E -> MSS

**Expected feature count after D53:**
  ONTHEFLY_FEATURES: 17 -> ~199
  New columns in rules.py output: ~182

**Swing confirmation lag (CRITICAL anti-leakage check):**
A swing at candidate bar j is confirmed at bar T = j + pivot_n.
All downstream features (OB, FVG, BOS, CHoCH, liq levels) must use T, not j.
Verify with: corr(int_swing_high[lag=0], target) ~= 0
             corr(int_swing_high[lag=-5], target) >> 0
If lag-0 correlation is non-trivial, there is a lookahead bug -- stop and fix it.

**ICT evidence ranking (most to least robust for BTC):**
1. Liquidity sweeps + close-based BOS/CHoCH/MSS (best; academic backing via stop cascades)
2. Displacement + FVG (60%+ hold rate as S/R -- only ICT concept with academic validation)
3. HTF context zones (premium/discount, dealing range)
4. Order blocks (moderate; definition-sensitive)
5. OTE (Fibonacci -- overfit-prone)
6. CISD (usually redundant with sweep + MSS)
7. Power of 3 / Silver Bullet (overfit magnets -- do not use as primary features)

---

## Optimizer Architecture (D52 target)

**Replace JSON registry + custom optimizer with Optuna:**
- Sampler: TPESampler(multivariate=True, constant_liar=True)
- Pruner: MedianPruner(n_startup_trials=10, n_warmup_steps=2) -- prune after fold 3
- Storage: JournalFileStorage("optuna_journal.log") -- NEVER SQLite
- Objective (GT-Score):
    weights = exp(-ln(2)/3 * arange(n_folds)[::-1]), normalized
    score = weighted_mean(fold_SRs) - 0.3*weighted_std(fold_SRs)
            - (k * log(T)) / (2*T) - 0.001*n_features
- Hard floor: reject any trial where worst fold Sharpe < -0.5
- Target: 500-1000 trials for ~15-dimensional search space
- 3 parallel workers: each sets CUDA_VISIBLE_DEVICES, shares same study

**Multiple testing / DSR:**
- After 9 experiments: expected max SR from noise = ~2.1. Track carefully.
- After 100 experiments: expected max SR from noise = 2.51.
- Bonferroni t-stat: 50 exps -> t>3.29, 100 -> t>3.48, 200 -> t>3.65.
- DSR >= 0.95 for promotion. DSR >= 0.99 for "deploy small".
- PBO < 5% per promoted strategy.
- Use Benjamini-Hochberg FDR (Q=5%) over Bonferroni.

**Rolling walk-forward (preferred over anchored for BTC):**
- Train: 3-6 months (~130K-260K bars). Test: 1-2 months (~43K-86K bars).
- ~24 folds from remaining ~504K training rows (after holdout).
- Early stopping: fold 3 of 11 correlates 0.85-0.92 with final result.
  Prune 50% of bad trials at 27% compute cost.

**Realistic expectations:**
- 40-60% Sharpe degradation backtest -> live for crypto perp strategies.
- Funding rate: ~6-8% annualized drag on longs (322/365 days positive in 2024).
- Paper trade 3-6 months minimum before capital deployment.

---



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
- Project root: C:\Users\tjall\Desktop\Trading\BTCDataset_v2\
- New code in: core/ (signals, engine, experiments)
- Legacy scripts: legacy/scripts/ (read-only reference)
- Working data dir: C:\Users\tjall\Desktop\Trading\BTCDataset_v2\data\
- Legacy results: legacy/scripts/results/
