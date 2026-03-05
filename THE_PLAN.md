---
MAINTENANCE RULES (for Claude Code):
- Read this file at the start of every session alongside CLAUDE.md
- After completing any step: update that step's status and result only
- Update the CURRENT STATE table at the bottom
- Do not rewrite other sections unless a decision has changed
- Append completed steps to STRATEGY_LOG.md as usual
---

# THE PLAN
# BTC Algo Trading -- Step by Step Build
# Last updated: Research synthesis complete, D51-D53 next (architecture sprint)
# Rule: only edit completed steps and next step. no full rewrites.

---

## STATUS LEGEND
- [x] COMPLETE
- [ ] PENDING
- [>] IN PROGRESS
- [!] BLOCKED

---

## PRE-STEPS

### [x] Get Coinalyze API Key (5 min)
1. Go to https://coinalyze.net
2. Sign Up — free, no credit card
3. After login: API menu or https://coinalyze.net/api/
4. Copy API key
5. Open BTCDataset_v2/.env and add:
   COINALYZE_API_KEY=your_key_here
6. Done.

### [x] How to Open a Second CC Window
1. Open a NEW terminal window (not a new tab)
2. cd C:\Users\tjall\Desktop\Trading\BTCDataset_v2
3. Run: claude
4. Two independent CC sessions, same filesystem.
   Window 1 owns: core/engine/, core/signals/, core/experiments/
   Window 2 owns: data_pipeline/, core/data/raw/

---

## STEP 1 — D37: Folder Restructure
**Window:** 1 (do before opening Window 2)
**Status:** [x] COMPLETE (2026-03-03)
**Result:** 14 new directories created. All D01-D36 scripts moved to
legacy/scripts/. ml_oos_probs_v2.npy copied to
core/experiments/models/baseline_d35.npy. CLAUDE.md updated at root.
Note: empty scripts/ folder remains (VS Code lock) — safe to delete manually.

**CC Prompt used:**
```
Read CLAUDE.md, STRATEGY_LOG.md, and THE_PLAN.md before starting.

D37: Create the new project folder structure.

1. Create these folders if they don't exist:
   core/data/
   core/data/raw/aggtrades/
   core/data/raw/oi_metrics/
   core/data/raw/liquidations/
   core/data/raw/funding/
   core/engine/
   core/signals/ict/
   core/signals/ta/
   core/experiments/results/
   core/experiments/models/
   core/experiments/shap/
   core/reports/
   data_pipeline/
   tradingview/
   legacy/scripts/

2. Move (do not copy) all existing scripts/ contents into legacy/scripts/
   Exception: BTCUSDT_5m_labeled_v2.parquet and feature_catalog_v2.yaml
   stay where they are. Do not move the parquet file (612MB).

3. Copy (not move) these specific files:
   scripts/results/ml_oos_probs_v2.npy
     -> core/experiments/models/baseline_d35.npy
   scripts/CLAUDE.md -> keep in legacy/scripts/ AND copy to root

4. Create empty placeholder files:
   core/signals/ict/rules.py      (header comment only)
   core/signals/ict/variants.py   (header comment only)
   core/signals/ta/rules.py       (header comment only)
   core/experiments/registry.json (empty JSON: {"experiments": []})
   core/reports/best_configs.json (empty JSON: {"configs": []})

5. Update CLAUDE.md root copy:
   - Dataset primary path: core/data/BTCUSDT_5m_labeled_v2.parquet
   - Add NEXT STEPS section listing D38-D42
   - Add D37 to decision log

6. Print full folder tree when done.

Log D37 in STRATEGY_LOG.md.
```

---

## STEP 2 — Open Second CC Window
**Status:** [x] COMPLETE — open when needed for Step 3B

**CMD:**
```
cd C:\Users\tjall\Desktop\Trading\BTCDataset_v2
claude
```

---

## STEP 3A — D38: Engine Scaffold
**Window:** 1
**Status:** [x] COMPLETE (2026-03-03)
**Result:** 4 modules: labels.py, sizing.py, evaluator.py, simulator.py.
Walk-forward LightGBM (GPU+fallback), CSCV, 10 gates, Kelly sizing, auto-retry.
Smoke tested end-to-end on GPU. NaN label bug found and fixed. Gate bool bug fixed.
**Depends on:** Step 1

**CC Prompt:**
```
Read CLAUDE.md, STRATEGY_LOG.md, and THE_PLAN.md before starting.

TASK: Build core/engine/ -- simulator.py, evaluator.py, sizing.py, labels.py.
No signals yet. Pure plumbing.

ENVIRONMENT:
Python 3.14, Windows, cp1252 encoding.
ASCII only in all print statements. No Unicode box-drawing chars,
em-dashes, arrows, or non-ASCII symbols anywhere in output or comments.
Packages: pandas, numpy, requests, pyarrow, lightgbm, shap, zipfile.
Paths relative to BTCDataset_v2/.
LightGBM GPU: params["device"] = "gpu" for RTX 5080.
Always print which device is being used at start of every training run:
  print("Training device: gpu") or print("Training device: cpu (fallback)")
GPU fallback: if LightGBM raises DeviceError, catch it, set device="cpu",
print warning, and continue. Never crash on GPU unavailability.

SIMULATOR.PY:
- Takes experiment dict, runs end-to-end
- Loads v3 if exists, v2 fallback. Print which is used.
- Auto-retries once on exception. On second failure: write FAILED +
  traceback to registry, continue. Never crash the session.

Experiment schema:
{
  "id": str,
  "signal_domain": "ict" | "ta",
  "features": list or "all",
  "feature_exclude": list,
  "label": str,
  "ml_config": {
    "threshold": float,
    "model": "lgbm" or "catboost",
    "device": "gpu" or "cpu",
    "n_folds": int,
    "embargo_bars": int
  },
  "cooldown_bars": int,
  "sizing": {"method": "kelly", "fraction": float},
  "gates": dict,
  "notes": str
}

EVALUATOR.PY methods:
  walk_forward_train(df, features, label, config) -> np.ndarray (OOS probs)
  compute_gates(results) -> dict of pass/fail per gate
  run_cscv(returns, n_combinations=70) -> dict (PBO, PSR, CI)
  compute_ece(probs, labels, n_bins=10) -> float (expected calibration error)
  run_stress_tests(returns, costs) -> dict of stress scenario results

STRESS TEST SUITE (run automatically on every gate-passing experiment):
  Scenario 1 -- fees doubled:     multiply all costs x2, recompute EV/Sharpe/WR
  Scenario 2 -- spread widened:   add 0.5x ATR to entry slippage per trade
  Scenario 3 -- fat-tail slippage: randomly apply 3x slippage to 5% of trades
  Scenario 4 -- missing data:     drop random 10% of signals, recompute metrics
  Scenario 5 -- bear regime only: evaluate on 2022-01-01 to 2022-12-31 only
  For each scenario: print EV, WR, Sharpe, MaxDD vs baseline
  Stress test does NOT block promotion -- it is informational only.
  But: if EV goes negative in ANY scenario, print a loud warning:
    "WARNING: EV negative under [scenario]. Review before live deployment."

Default gates (10):
  MIN_TRADES_PER_YEAR=100, MIN_OOS_AUC=0.75, MAX_PBO=0.05,
  MIN_PSR=0.99, MIN_WF_WINDOWS="all", MIN_SHARPE=2.0,
  MIN_WR=0.55, MIN_EV_R=0.50, MAX_DRAWDOWN=0.20,
  MAX_ECE=0.05,
  AUC_PROMOTION_DELTA=0.005, LOGLOSS_MUST_IMPROVE=True

SIZING.PY:
  kelly_fraction(p, odds=2.0, divisor=40) -> float, clipped [0.01, 0.02]
  Port directly from legacy/scripts/position_sizing.py Kelly section.

LABELS.PY:
  get_label(df, direction, r_multiple, horizon_bars) -> str
  validate_label_alignment(embargo_bars, label_horizon) -> bool
  If embargo_bars != label_horizon_bars: warn loudly.

SMOKE TEST when complete:
  Load v2 parquet. Create minimal experiment (10 features, 3 folds).
  Run through simulator.py. Verify results dict produced without crash.
  Print the results dict.

Log D38 in STRATEGY_LOG.md. Update CLAUDE.md.
```

**Success check:** CC prints smoke test results dict without crash.

---

## STEP 3B — D37a: OI Data Pipeline
**Window:** 2
**Status:** [ ] PENDING
**Depends on:** Step 1

**CC Prompt:**
```
Read CLAUDE.md, STRATEGY_LOG.md, and THE_PLAN.md before starting.

TASK: Build data_pipeline/download_oi.py

ENVIRONMENT:
Python 3.14, Windows, cp1252 encoding.
ASCII only in all print statements. No Unicode box-drawing chars,
em-dashes, arrows, or non-ASCII symbols.
Packages: pandas, numpy, requests, pyarrow, zipfile, hashlib.
Paths relative to BTCDataset_v2/.

DOWNLOAD_OI.PY:
- Download Binance OI metrics from data.binance.vision
- URL pattern: https://data.binance.vision/data/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-YYYY-MM-DD.zip
- Date range: 2020-02-11 to yesterday
- Verify SHA256 checksum for each file (.CHECKSUM file at same URL)
- Align to 5m bars via bar_start_ts_utc
- Compute: oi_change_1h, oi_change_4h, oi_change_pct_1h, oi_zscore_20
- Save to core/data/raw/oi_metrics/ as monthly parquet
- Missing dates: log and continue, do not crash

SUCCESS CRITERIA:
- Print total rows downloaded and date range covered
- Print count of missing dates
- Print sample of first 5 rows of output parquet

Log D37a in STRATEGY_LOG.md. Update CLAUDE.md.
```

**Success check:** Rows downloaded, date range printed, sample rows shown.
**Expected:** ~2,200 daily files, 2020-02-11 to present. Takes 1-2 hours.

---

## STEP 4A — D39: ICT Signal Migration
**Window:** 1
**Status:** [x] COMPLETE (2026-03-03)
**Result:** 9 causal functions in core/signals/ict/rules.py (671 lines).
All 9/9 causality tests PASSED at T in [1000, 5000, 10000, 50000].
Smoke test on 10k bars: all shapes/NaN counts sensible. Runtime 5.6s.
Functions: compute_swing_points, detect_ob_{bull,bear}, detect_fvg_{bull,bear},
compute_ote_dist, compute_liq_levels, compute_premium_discount, compute_cisd.
**Depends on:** Step 3A complete

**CC Prompt:**
```
Read CLAUDE.md, STRATEGY_LOG.md, and THE_PLAN.md before starting.

TASK: Decompose legacy/scripts/enrich_ict_v4.py into standalone causal
functions in core/signals/ict/rules.py.

ENVIRONMENT:
Python 3.14, Windows, cp1252 encoding.
ASCII only in all print statements. No Unicode symbols.
Paths relative to BTCDataset_v2/.

RULES:
- Each function takes df (DataFrame) + optional parameters
- Returns pd.Series or pd.DataFrame (one value per bar)
- Strictly causal: at bar T, uses only df.iloc[:T+1]
- No global state. No side effects. Pure functions.
- PORT logic from enrich_ict_v4.py. Do NOT rewrite from scratch.

FUNCTIONS (priority order by SHAP rank):

detect_ob_bull(df, lookback=200) -> DataFrame:
  columns: ob_bull_age, ob_bull_top, ob_bull_bot, ob_bull_mid, ob_bull_in_zone

detect_ob_bear(df, lookback=200) -> DataFrame (same, bear direction)

detect_fvg_bull(df, age_cap=288, min_size_atr=0.0) -> DataFrame:
  columns: fvg_bull_in_zone, fvg_bull_near_top, fvg_bull_near_bot,
           fvg_bull_age, fvg_bull_count

detect_fvg_bear(df, age_cap=288, min_size_atr=0.0) -> DataFrame

compute_ote_dist(df, fib_low=0.62, fib_high=0.79, swing_lookback=20) -> Series

compute_swing_points(df, pivot_n=3) -> DataFrame:
  columns: swing_high, swing_low, market_trend, bos_wick, bos_close, choch_close

compute_liq_levels(df, eq_tolerance_atr=0.2, lookback=50) -> DataFrame:
  columns: liq_dist_above_pct, liq_dist_below_pct, liq_eq_high, liq_eq_low,
           liq_pdh, liq_pdl, liq_pwh, liq_pwl

compute_premium_discount(df, swing_lookback=20) -> Series

compute_cisd(df) -> DataFrame: columns: cisd_bull, cisd_bear

MANDATORY CAUSALITY TEST for every function (fail loudly):
  for T in [1000, 5000, 10000, 50000]:
      a = func(df.iloc[:T])
      b = func(df.iloc[:T+1])
      assert a.iloc[-1].equals(b.iloc[-2]), f"LOOKAHEAD at T={T}"
Never suppress assertions.

SUCCESS CRITERIA:
- Print "CAUSALITY PASS: [function_name]" for each function
- Smoke test on first 10,000 bars: print shape and NaN counts

Log D39 in STRATEGY_LOG.md. Update CLAUDE.md.
```

**Success check:** `CAUSALITY PASS` printed for every function. No failures.

---

## STEP 4B — D37b: True Tick CVD Pipeline
**Window:** 2
**Status:** [x] COMPLETE (2026-03-03)
**Result:** 74/74 months, 3.1 billion trades -> 640,664 bars, 53 minutes.
Pearson(cvd_bar, cvd_true_bar) = 0.5330 (in expected range). Zero failures.
7,624 bars missing (1.2% zero-trade windows, mostly early 2020).
**Depends on:** Step 3B complete (or run in parallel -- different output folder)

**CC Prompt:**
```
Read CLAUDE.md, STRATEGY_LOG.md, and THE_PLAN.md before starting.

TASK: Build data_pipeline/download_aggtrades.py

ENVIRONMENT:
Python 3.14, Windows, cp1252 encoding.
ASCII only in all print statements. No Unicode symbols.
Packages: pandas, numpy, requests, pyarrow, zipfile.
Paths relative to BTCDataset_v2/.

SOURCE: https://data.binance.vision/data/futures/um/monthly/aggTrades/BTCUSDT/
Columns: agg_trade_id, price, quantity, first_trade_id, last_trade_id,
         timestamp, is_buyer_maker
Date range: 2020-01 to 2026-02

WARNING: files are 1-5GB compressed. NEVER load full file into memory.
Use pandas read_csv with chunksize=100000. Streaming is mandatory.

ALGORITHM:
1. Download one monthly ZIP at a time
2. Stream CSV in chunks of 100,000 rows
3. delta = quantity * (1 if not is_buyer_maker else -1)
4. Group by 5m bar (floor timestamp to 5m UTC = bar_start_ts_utc)
5. Sum delta per bar -> cvd_true_bar
6. Accumulate:
   cvd_true_daily: reset at 00:00 UTC each day
   cvd_true_session: reset at London/NY session opens
   cvd_true_zscore: (cvd_true_daily - rolling_mean(20)) / rolling_std(20)
7. Discard raw trades after each chunk

OUTPUT: core/data/raw/aggtrades/ monthly parquet with:
  bar_start_ts_utc, cvd_true_bar, cvd_true_daily, cvd_true_session, cvd_true_zscore

SUCCESS CRITERIA:
- Print rows processed per month
- Print Pearson correlation vs existing cvd_zscore (expected 0.5-0.7)
- Print 5 rows at midnight UTC rollover to confirm reset

Log D37b in STRATEGY_LOG.md. Update CLAUDE.md.
```

**Success check:** Pearson correlation 0.5-0.7. If below 0.5 there is a bug — stop and report.

---

## STEP 5A — D40: Engine Validation Run
**Window:** 1
**Status:** [x] COMPLETE (2026-03-03)
**Result:** VALIDATION PASS. All 7 D35 metrics reproduced within tolerance.
WR 65.41%, EV +0.912R, PF 3.51, MaxDD 11.96%, 180/yr, PBO 0%, PSR 1.0.
E000 written to registry.json. Note: D35 used fixed 2% risk, not variable Kelly.
ECE gate fails (0.125 > 0.05) but ECE was not part of D35 validation spec.
**Depends on:** Steps 3A and 4A complete

**CC Prompt:**
```
Read CLAUDE.md, STRATEGY_LOG.md, and THE_PLAN.md before starting.

TASK: Reproduce D35 production config through the new engine.
This is the gate before any new experiments can run.

USE: core/experiments/models/baseline_d35.npy
(same OOS probs as D35 -- differences reflect code, not retraining)

Config:
  threshold = 0.60
  cooldown_bars = 576
  sizing = Kelly 1/40
  label = label_long_hit_2r_48c

EXPECTED (from D35):
  WR:        65.4%  +/- 1.0pp
  EV:        +0.912R +/- 0.05R
  PF:        3.51   +/- 0.15
  MaxDD:     12.0%  +/- 1.5pp
  Trades/yr: ~180   +/- 15
  PBO <= 0.01
  PSR >= 0.999

OUTPUT: side-by-side comparison table (D35 expected vs engine result)
PASS: write as E000 to core/experiments/registry.json.
      Print "VALIDATION PASS"
FAIL: print full discrepancy table.
      Print "VALIDATION FAIL -- do not proceed"
      Do not write to registry if FAIL.

Log D40 in STRATEGY_LOG.md. Update CLAUDE.md.
```

**Success check:** `VALIDATION PASS` printed and E000 written to registry.
**If FAIL:** Paste full output here before doing anything else.

---

## STEP 5B — D37c: Liquidations Pipeline
**Window:** 2
**Status:** [!] BLOCKED until Coinalyze API key obtained
**Depends on:** Step 1 + API key in .env

**CC Prompt:**
```
Read CLAUDE.md, STRATEGY_LOG.md, and THE_PLAN.md before starting.

TASK: Build data_pipeline/download_liquidations.py

ENVIRONMENT:
Python 3.14, Windows, cp1252 encoding.
ASCII only in all print statements. No Unicode symbols.
Packages: pandas, numpy, requests, pyarrow, zipfile, os.
Load API key: import os; api_key = os.getenv("COINALYZE_API_KEY")
or read from .env manually if dotenv not available.
Paths relative to BTCDataset_v2/.

SOURCE A: data.binance.vision liquidationSnapshot
URL: https://data.binance.vision/data/futures/um/daily/liquidationSnapshot/BTCUSDT/BTCUSDT-liquidationSnapshot-YYYY-MM-DD.zip
Date range: 2020-07-01 to 2024-03-31
Note: Binance throttled to 1 liq/second after April 2021.
Data is inherently incomplete after that date from any source.

SOURCE B: Coinalyze free API (post-2024 + daily aggregates)
Endpoint: https://api.coinalyze.net/v1/liquidation-history
Symbol: BTCUSDT_PERP.A
Rate limit: 40 req/min -- respect this with time.sleep(1.5) between calls

COMPUTE per 5m bar:
  liq_buy_usd_1h:   rolling 1h sum of buy liquidations in USD
  liq_sell_usd_1h:  rolling 1h sum of sell liquidations in USD
  liq_total_1h:     liq_buy_usd_1h + liq_sell_usd_1h
  liq_ratio_1h:     liq_buy_usd_1h / (liq_total_1h + 1e-9)
  liq_cascade_flag: 1 when liq_total_1h > 3x rolling_mean_liq_total_24h

OUTPUT: core/data/raw/liquidations/ as parquet

SUCCESS CRITERIA:
- Print rows per source (Binance vs Coinalyze)
- Print date coverage and NaN rate for post-April-2021 period
- Print sample of 5 rows including a cascade event if any exist

Log D37c in STRATEGY_LOG.md. Update CLAUDE.md.
```

**Success check:** Both sources loaded, NaN rate reported, cascade events found.

---

## STEP 6A — D41a: ICT Knowledge Base
**Window:** 1
**Status:** [x] COMPLETE
**Depends on:** Step 5A (VALIDATION PASS required)

**CC Prompt:**
```
Read CLAUDE.md, STRATEGY_LOG.md, and THE_PLAN.md before starting.

TASK: Create core/signals/ict/knowledge.md

CRITICAL: Read ALL SHAP values directly from
legacy/scripts/results/shap_analysis_v2.json
Do not approximate or infer. Use exact mean_abs_shap values from the JSON.

Build 8 sections:

SECTION 1: SHAP EVIDENCE HIERARCHY (D36)
- Header: OOS AUC 0.7937, 11-fold, 508 features, 648,288 bars
- Top 30 table: rank, feature, exact SHAP from JSON, category, verdict
- Regime-dependent (keep always regardless of rank):
  m30_cvd_zscore, h4_close, liq_nearest_below, ict_dr_eq,
  ict_fvg_bear_nearest_bot, m15_ict_ob_bull_top
- Dead features (81 total, read from JSON bottom50 + recommendations):
  All ict_macro_* (zero SHAP), Silver Bullet gates (avg rank 409),
  Kill Zone gates (avg rank 370), PO3/AMD (avg rank 315),
  all fund_* columns, all ob_mitigated flags, gk_* volatility,
  rsi_14/rsi_21 (avg rank 249), ema_cross_* (avg rank 292)
  EXCEPTION NOTE: Silver Bullet and Macro windows not final dead.
  Test at 1m timeframe (60 bars/window) before final verdict.

SECTION 2: ICT RULES LIBRARY
For each rule: Status | SHAP ranks | Definition | Encoded params |
Variants to test | Parameter search space | Open questions
Rules: OB, FVG, OTE, Liquidity, Structure, Premium/Discount, CISD,
       Silver Bullet, Macros, PO3/AMD

Parameter search spaces:
  OB: age_threshold [5,10,20,50,100,200], min_displacement_atr [0.5,1.0,1.5,2.0]
  FVG: min_size_atr [0.1,0.2,0.5,1.0], age_cap [48,96,144,288,576],
       entry_level ["top","mid","62pct","bot"]
  OTE: fib_low [0.50,0.618,0.62], fib_high [0.786,0.79,0.886],
       swing_lookback [3,5,10,20], timeframe_for_swing [5m,15m,30m,1h,4h]
  Structure: swing_pivot_n [3,5,8,13,20], bos_type ["wick","close","both"]

SECTION 3: UNENCODED RULES (priority order)
1. Breaker Block (HIGH): mitigated OB flips to opposite direction
2. BPR (MEDIUM): overlapping bull+bear FVG
3. Judas Swing (MEDIUM): false session-open liquidity sweep + reversal
4. Liquidity Void (LOW): defer until SHAP evidence
5. Time/Price Theory (LOW): too discretionary, do not encode yet

SECTION 4: EXPERIMENT HISTORY
E001  D21  Config B baseline: 116 signals, WR 40.52%, EV +0.166R, PF 1.36
E002  D24  MTF expansion: 176 longs, WR 47.16%, EV +0.365R. H1 best: 58.18%
E003  D26  CSCV: PBO 0%, PSR 0.9994, CI [+0.024,+0.706]. PASS.
E004  D27  Execution: R +0.390, PF 1.71, Sharpe 1.39, cost +0.025R/trade
E005  D28  ML v1: AUC 0.7819. Top: ob_age, swings, liq_dist.
E006  D29  T2 ML>=0.60 CD=48: 9,180 trades, WR 67.3%, EV +0.97R. CSCV ALL PASS.
E007  D31  Cooldown: CD=576 179/yr WR 66.85% EV +0.955R
E008  D32  Kelly 1/40: Sharpe 9.85, MaxDD 12.2%. CAGR 1711% = backtest only.
E009  D33  v2 enrichment: +122 features. Key: ote_dist, stoch_k, macd_fast_norm
E010  D34  ML v2: AUC 0.7937 (+0.012). ote_dist = #2 SHAP.
E011  D35  Production v2: WR 65.4%, EV +0.912R, PF 3.51. CSCV ALL PASS.
E012  D36  SHAP: ICT structural = alpha. 81 dead features identified.

SECTION 5: OPEN RESEARCH QUESTIONS
Priority 1:
  RQ1: Does ob_quality_score beat raw ob_bull_age?
  RQ2: Does H1 OTE distance score higher SHAP than 5m OTE?
  RQ3: Does OTE + OB + FVG triple confluence produce WR > 70%?
  RQ4: Breaker blocks: reversal zone or price target?
  RQ5: [BLOCKED] OI rate-of-change vs FVG formation events.
       STATUS: BLOCKED. Requires v3 parquet with OI data.
       Do not propose until download_oi.py complete and v3 exists.
Priority 2:
  RQ6: Regime-conditional models: 2022-bear vs 2023-recovery vs 2024-bull
  RQ7: Silver Bullet at 1m: do time gates emerge as predictive?
  RQ8: BPR: incremental signal above individual FVG?
  RQ9: True tick CVD vs CLV: SHAP comparison
  RQ10: Rising OI + FVG vs falling OI + FVG directional difference
Priority 3:
  RQ11: Turtle Soup: sweep + immediate reversal WR
  RQ12: Judas Swing vs generic h4_sweep
  RQ13: Internal (3-bar) vs external (20-bar) structure for 48-bar label
  RQ14: Short-side dedicated model with bear-specific features

SECTION 6: DEAD ENDS
DE01: d1_ict_market_trend as mandatory hard filter. Config A: 33.3% WR.
      Valid as ML input (rank ~25-30). Dead as hard filter only.
DE02: Short-taking with SYMMETRIC SESSION RULES. D17/D25. 25-33% WR all TFs.
      Symmetric rules are dead. NOT the same as a bear-specific ML model.
      A dedicated short model using bear OB age, bear FVG in_zone,
      h4_trend==-1 has NOT been tested. See RQ14. Do not conflate.
DE03: Regime classification as mandatory filter. 81% signals in HIGH-vol anyway.
      Regime features valid as ML inputs. Not valid as hard filters.
DE04: Fixed 48-bar cooldown at low signal frequency. 14 trades/6yr. Untradeable.
DE05: All-time cumulative CVD. No predictive meaning per bar. Use session-reset only.
DE06: H4 CE limit entries. Median 7.1 ATR below close. 27% fill rate. Dead.
      H1 CE (1.9 ATR, 42%) and M15 CE (1.3 ATR, 50%) viable for future testing.

SECTION 7: IMPLEMENTATION NOTES
Causality: df[0:T] vs df[0:T+1] test mandatory for every feature.
Annualization: sqrt(105120) = 324.22 (5m, 24/7 crypto).
Embargo: 48 bars minimum. Recalculate if label horizon changes.
Naming: ict_{rule}_{direction}_{metric} / HTF: {tf}_{base_name}
AUC promotion threshold: +0.005 minimum, logloss must also improve.
Calibration: reliability diagram after every retrain. ECE > 0.05 = isotonic.
Model: LightGBM default (device="gpu"). CatBoost if categorical-heavy.
Logistic regression always run as sanity check baseline.
AUC > 0.85 on financial series: suspicious. Check for lookahead.
CAGR 1711% (D32): backtest-compounded at 2% reinvestment. Not a live forecast.
CD=576 rationale: 300/yr monitoring cap. CD=288 is natural upgrade if automated.

SECTION 8: OPTIMIZER BEHAVIOR RULES
Pre-proposal checklist:
  1. Read full experiment history (Section 4)
  2. Identify highest-SHAP unencoded rule (Section 3)
  3. Check open research questions (Section 5, Priority 1 first)
  4. Verify no dead end repeated (Section 6) without new hypothesis

Failure diagnosis:
  Low trade count  -> reduce threshold 0.02 OR reduce cooldown 96 bars
  Low WR           -> raise threshold 0.02 OR add OB quality filter
  High drawdown    -> reduce cooldown 96 OR reduce Kelly fraction
  Low AUC          -> add highest-SHAP unencoded feature from Section 3
  Logloss worsened -> remove 50 lowest-SHAP features
  High ECE         -> apply isotonic regression calibration

Mutation rules:
  One parameter at a time.
  Exception: new ICT rule = all params set simultaneously.
  Max 3 variants of same parameter per session.
  Two consecutive failures on same parameter: mark exhausted, next RQ.

On gate pass:
  1. Run shap_runner.py. Append to Section 1.
  2. Append experiment to Section 4.
  3. Update experiments/registry.json.
  4. If AUC delta >= 0.005 vs production: update reports/best_configs.json.

SUCCESS CRITERIA:
- File written to core/signals/ict/knowledge.md
- All Section 1 SHAP values verified against shap_analysis_v2.json
- Print "KNOWLEDGE BASE WRITTEN: N lines"

Log D41a in STRATEGY_LOG.md. Update CLAUDE.md.
```

**Success check:** `KNOWLEDGE BASE WRITTEN: N lines` printed. Verify top 5 SHAP values manually against JSON.

---

## STEP 6B — D37d: Dataset Merge v2 -> v3
**Window:** 2
**Status:** [ ] PENDING
**Depends on:** Steps 3B, 4B, and 5B ALL complete

**CC Prompt:**
```
Read CLAUDE.md, STRATEGY_LOG.md, and THE_PLAN.md before starting.

TASK: Build data_pipeline/merge_v3.py

PURPOSE: Read v2 as source. Merge new columns. Write new v3. Never overwrite v2.

INPUT:
  Source: core/data/BTCUSDT_5m_labeled_v2.parquet (READ ONLY)
  OI:     core/data/raw/oi_metrics/ (monthly parquets)
  CVD:    core/data/raw/aggtrades/ (monthly parquets)
  Liq:    core/data/raw/liquidations/ (parquets)

OUTPUT:
  core/data/BTCUSDT_5m_labeled_v3.parquet (new file)
  core/data/feature_catalog_v3.yaml (extend v2 catalog)

MERGE: left join on bar_start_ts_utc. Keep all v2 rows. NaN for missing new data.

CAUSALITY GATE (mandatory before writing v3):
  For every new column added:
    for T in [1000, 5000, 50000]:
      recalculate using only df.iloc[:T+1]
      assert value at T matches
  OI rate-of-change: use shift(1) for all lags. Especially prone to 1-bar lookahead.
  If any column fails: halt, print which column, do not write v3.

VALIDATION:
  Row count v3 must equal v2 exactly (648,288)
  v2 modification timestamp must be unchanged after script runs
  Print: new column count, NaN rates per column, file sizes

SUCCESS CRITERIA:
  "CAUSALITY GATE: ALL PASS" printed
  Row count matches
  v2 timestamp unchanged
  v3 written

Log D37d in STRATEGY_LOG.md. Update CLAUDE.md.
```

**Success check:** `CAUSALITY GATE: ALL PASS`, row count 648,288, v2 unchanged.

---

## STEP 7 — D41b: Optimizer + SHAP Runner
**Window:** 1
**Status:** [x] COMPLETE
**Depends on:** Steps 5A and 6A complete. Window 2 can still run.

**CC Prompt:**
```
Read CLAUDE.md, STRATEGY_LOG.md, THE_PLAN.md, and core/signals/ict/knowledge.md before starting.

TASK: Build core/engine/optimizer.py and core/engine/shap_runner.py

ENVIRONMENT:
Python 3.14, Windows, cp1252 encoding.
ASCII only in all print statements. No Unicode symbols.
Paths relative to BTCDataset_v2/.
GPU: device="gpu", fallback to cpu with warning.

OPTIMIZER.PY:
MODE = "checkpoint"    # default. change to "autonomous" to run hands-free.
MAX_EXPERIMENTS_PER_SESSION = 5

propose_next_experiment(registry, knowledge) -> dict:
  1. Parse all past experiments from registry.json
  2. Find best config (highest OOS Sharpe among gate-passing experiments)
  3. Read knowledge.md: unencoded rules (Section 3), RQs (Section 5),
     dead ends (Section 6)
  4. Check if experiment requires data not yet available
     (OI features need v3 -- check v3 exists before proposing RQ5/Exp D)
  5. Apply mutation rules (one parameter at a time)
  6. Estimate runtime. Print: "Estimated: ~N min (M features x K folds on GPU/CPU)"
     If session total > 90 min: warn before starting.

CHECKPOINT MODE:
  Print proposed experiment in readable format.
  Call input("Run this experiment? [y/n]: ") and wait for response.
  If non-interactive: write to core/experiments/pending_approval.json and print:
    "CHECKPOINT: review core/experiments/pending_approval.json, re-run to confirm."
  Never proceed automatically without confirmation in checkpoint mode.

AUTONOMOUS MODE:
  Run up to MAX_EXPERIMENTS_PER_SESSION without pausing.
  Print session summary: N run, N passed gates, best AUC achieved.

Failure diagnosis map:
  Low trade count  -> reduce threshold 0.02 OR reduce cooldown 96 bars
  Low WR           -> raise threshold 0.02 OR add OB quality filter
  High drawdown    -> reduce cooldown 96 OR reduce Kelly fraction
  Low AUC          -> add highest-SHAP unencoded feature from knowledge.md
  Logloss worsened -> remove 50 lowest-SHAP features
  High ECE         -> apply calibrator.py isotonic regression before next run

Mutation rules:
  One parameter at a time.
  Exception: new ICT rule = all params simultaneously.
  Max 3 variants same parameter per session.
  Two consecutive failures same parameter: mark exhausted, move to next RQ.

On gate pass:
  1. Run shap_runner.py on saved model
  2. Append SHAP findings to knowledge.md Section 1
  3. Append to knowledge.md Section 4
  4. Update core/experiments/registry.json
  5. If AUC delta >= 0.005 vs production: update core/reports/best_configs.json

SHAP_RUNNER.PY:
  - Load saved LightGBM model from core/experiments/models/
  - Run SHAP TreeExplainer on OOS test set only (not full dataset)
  - Append top 30 findings to knowledge.md Section 1
  - Flag features entering or leaving top 30 vs previous run
  - Prune features with mean abs SHAP < 0.010 (except regime-dependent list)
    from feature set for next experiment
  - Save full SHAP output to core/experiments/shap/shap_[exp_id].json

SUCCESS CRITERIA:
  - optimizer.py proposes first experiment (should be RQ1: OB Quality Score)
  - Prints proposed config in readable format
  - Waits for [y/n] in checkpoint mode

Log D41b in STRATEGY_LOG.md. Update CLAUDE.md.
```

**Success check:** Optimizer proposes RQ1 (OB Quality Score) as first experiment and waits for approval. If it proposes something else, paste it here for review.

---

## STEP 8 — Switch Engine to v3
**Window:** 2
**Status:** [x] DONE (2026-03-03) — v3 confirmed default, smoke test PASS
**Result:** simulator.py already had v3 as first entry in DATA_FILES. Smoke test
confirmed 594 cols, 648,288 rows, all 25 new features (OI/CVD/liq) working in
LightGBM. Full pipeline end-to-end PASS. Registry cleaned.
**Depends on:** Step 6B complete (v3 exists)

**CC Prompt:**
```
Read CLAUDE.md, STRATEGY_LOG.md, and THE_PLAN.md before starting.

Update core/engine/simulator.py:
Confirm v3 is now the default dataset.
Run a smoke test experiment (3 folds, 20 features).
Print: "Dataset: v3 (N cols)" and confirm row count is 648,288.

Log v3 switch in CLAUDE.md.
```

**Success check:** `Dataset: v3` printed, 648,288 rows confirmed.

---

## STEP 9 — D42+: First Experiments
**Window:** 1
**Status:** [x] DONE (2026-03-03) — E000-E003 complete, E002_prune is best
**Result:**
  E000 baseline: WR 65.4%, EV +0.91R, AUC 0.7935, ECE FAIL, 9/10 gates
  E001 rq1 (ob_quality): WR 64.7%, EV +0.89R, AUC 0.7991, ECE FAIL, 9/10
  E002_prune: WR 76.2%, EV +1.237R, Sharpe 12.9, AUC 0.7942, ECE 0.016, 10/10 PASS
    CURRENT BEST. PBO 0%, PSR 1.0, CI [+1.15,+1.32]. Calibration was the key.
  E003_rq4 (breakers): WR 74.95%, EV +1.198R, 10/10 PASS, no promotion.
    RQ1 (ob_quality): NO. RQ4 (breakers): NO. Both below prune threshold.
**Depends on:** Step 7 complete. v3 preferred but v2 works.

**CC Prompt:**
```
Read CLAUDE.md, STRATEGY_LOG.md, THE_PLAN.md, and core/signals/ict/knowledge.md before starting.

Run the optimizer in checkpoint mode.
MAX_EXPERIMENTS_PER_SESSION = 5
MODE = "checkpoint"

Propose the first experiment based on knowledge.md Section 5 Priority 1.
Print the proposed config and wait for approval before running.
```

**After each experiment:** Paste results here. We review together before approving next.

---

## STEP 10 — TradingView Export
**Window:** 1
**Status:** [x] DONE (D45) — Pine Script v5 generated from E002_prune SHAP
**Gate conditions (all must be true):**
- All 10 evaluation gates pass (including max_ece <= 0.05) -- PASS (E002_prune: 10/10)
- Minimum 200 OOS trades -- PASS (921 trades)
- Minimum 2 full years of profitable walk-forward windows -- PASS (7/7 windows, 5.16 yr span)

**When earned:** Paste Prompt 9 from btc_algo_complete_plan_v2.md

---

## STEP 11 — D47: Regime Filter
**Window:** 1
**Status:** [x] COMPLETE (2026-03-03)
**Result:** 3-state Gaussian HMM (pure NumPy, no hmmlearn) + ADX composite + interactions.
10 features total: hmm_prob_{bull,bear,calm}, bb_width_normalized, atr_percentile_rank,
regime_tag, ob_bull_age_x_hmm_bull, fvg_bull_x_trending, ote_x_regime.
All 12/12 causality tests PASS (ADX 4/4, HMM 4/4, interactions 4/4).
HMM coverage 88.7% (warmup 252 days), ADX ~100%, interactions 88.7-100%.
Registered in simulator.py ONTHEFLY_FEATURES. On-the-fly augmentation verified.
**Depends on:** D46b complete (DONE)
**Why:** Research confirms ICT signals are regime-dependent (Kim et al. 2025,
JFM). RQ1/RQ4 answering NO is consistent with model averaging across
trending/ranging/calm regimes. Regime filter is the highest-leverage structural
addition remaining. HMMs outperform ADX for BTC regime detection (Koki et al. 2022).

**CC Prompt:**
```
Read CLAUDE.md, STRATEGY_LOG.md, THE_PLAN.md, and core/signals/ict/knowledge.md before starting.

D47: Build the regime filter. Two components: HMM (accurate) + ADX composite (fast).

TASK 1: Build core/signals/regime/hmm_filter.py
- 3-state Gaussian HMM on log-returns (resample 5m data to daily for HMM training)
- States: trending_bull, trending_bear, ranging/calm
- Output soft probabilities per bar: hmm_prob_bull, hmm_prob_bear, hmm_prob_calm
- Retrain on rolling 1-year window at each walk-forward fold boundary (no lookahead)
- At bar T: only use data from bars [0..T]. Causality required.
- Use hmmlearn (pip install hmmlearn --break-system-packages)
- Upsample daily HMM probabilities back to 5m bars (forward-fill within day)

TASK 2: Add ADX composite features to existing pipeline
- adx_14: Average Directional Index (14-period)
- bb_width_normalized: Bollinger Band width / price (normalized)
- atr_percentile_rank: ATR rank over rolling 288-bar window (0-1)
- regime_tag: "trending" if ADX>25 and bb_width > 75th pct, "ranging" if ADX<20, else "neutral"

TASK 3: Add interaction features
- ob_bull_age_x_hmm_bull: ict_ob_bull_age * hmm_prob_bull
- fvg_bull_x_trending: (ict_fvg_bull_recent_age is not NaN).astype(int) * (adx_14 > 25).astype(int)
- ote_x_regime: ote_dist * hmm_prob_bull

TASK 4: Register all new features in simulator.py ONTHEFLY_FEATURES
- All regime features computed on-the-fly via augment_features()
- Causality test at T=[1000,5000,10000,50000] for all features

TASK 5: Smoke test
python core/signals/regime/test_regime.py
All causality tests must PASS. Print coverage stats for each feature.

TASK 6: Update all docs. Log as D47.
```

**Success check:** All causality tests PASS. Features registered in ONTHEFLY_FEATURES.
Then run: python -m core.engine.optimizer --approve (should propose RQ with regime features)

---

## STEP 12 — D48: First Short Experiment
**Window:** 2
**Status:** [x] DONE (2026-03-03) -- E004_short_baseline 10/10 PASS, SHAP done
**Result:** E004: AUC 0.7981, WR 71.89%, EV +1.107R, MaxDD 6.28%, Sharpe 10.98.
SHAP: 8/10 top features shared with long. swing_low_price rises to #1 (from #5 long).
Kim et al. prediction partially confirmed: same core features, different order.
Short baseline PROMOTED.
**Depends on:** D46b complete (DONE). label_short_hit_2r_48c confirmed in v3.
**Why:** System is currently long-only. Short label exists in v3 parquet (~648k
non-null). Kim et al. (2025) predicts short SHAP rankings will differ from long.
This is the first step toward a balanced long/short system.

**CC Prompt:**
```
Read CLAUDE.md, STRATEGY_LOG.md, and THE_PLAN.md before starting.

D48: Run first short baseline experiment.

TASK 1: Verify short label exists
python -c "
import pandas as pd
df = pd.read_parquet('data/labeled/BTCUSDT_5m_labeled_v3.parquet',
                     columns=['label_short_hit_2r_48c'])
print(f'label_short_hit_2r_48c: {df.iloc[:,0].notna().sum():,} non-null')
print(f'Win rate: {df.iloc[:,0].mean():.4f}')
"

TASK 2: Queue and run short baseline experiment
Experiment config:
{
  "id": "E004_short_baseline",
  "status": "approved",
  "tier": "standard",
  "label": "label_short_hit_2r_48c",
  "features": "all",
  "feature_exclude": None,
  "ml_config": {
    "threshold": 0.60,
    "cooldown_bars": 576,
    "model": "lgbm",
    "device": "gpu",
    "embargo_bars": 48,
  },
  "sizing": {"method": "kelly", "divisor": 40.0, "odds": 2.0},
  "notes": "E004: short baseline, label_short_hit_2r_48c, standard tier. D48.",
}

python -m core.engine.optimizer --approve

TASK 3: After experiment completes, run SHAP
python -m core.engine.shap_runner --exp-id E004_short_baseline

TASK 4: Compare short vs long SHAP rankings
- Which features are shared in top 10?
- Which are different? (Kim et al. 2025 predicts significant differences)
- Is ote_dist still #1 for shorts? Or does a different ICT feature dominate?

TASK 5: Update all docs. Log as D48.
If 10/10 gates PASS: note "short baseline PROMOTED" in STRATEGY_LOG.md
```

**Success check:** E004 runs to completion, SHAP done, long vs short comparison printed.

---

## STEP 13 — D49: Weekly Tier Experiment (TRQ3)
**Window:** 3
**Status:** [x] DONE (2026-03-03)
**Result:** E004_rq5 (OI, standard) ran first: 10/10 PASS, AUC 0.7949, WR 74.3%, RQ5: NO.
E005_trq3 (weekly tier): 10/10 PASS (weekly gates), AUC 0.7941, 52 trades/yr, WR 73.9%,
EV +1.17R, Sharpe 6.4, MaxDD 7.7%. SHAP: top 5 identical to standard (ote_dist #1).
Weekly tier validated end-to-end. Dual-tier framework works.
**Depends on:** D46b complete (DONE). Optimizer has TRQ3 ready.
**Why:** Validates the dual-tier framework end-to-end. Weekly model targets
1-3 trades/week (52-156/yr) vs current 178/yr. Different cooldown changes
which setups qualify. First test of tier-aware gate thresholds.

**CC Prompt:**
```
Read CLAUDE.md, STRATEGY_LOG.md, and THE_PLAN.md before starting.

D49: Run weekly tier experiment (TRQ3).

TASK 1: Check optimizer proposal
python -m core.engine.optimizer
Should propose TRQ3 (weekly, CD=2016, 7-day cooldown, standard labels).
If it proposes a standard RQ first, run that first then re-run for TRQ3.

TASK 2: Run the experiment
python -m core.engine.optimizer --approve

TASK 3: After completion, check:
- trades/yr (expect 52-75 with 7-day cooldown vs 178 standard)
- EV per trade vs standard (expect higher conviction per trade)
- Which gates pass/fail under weekly tier thresholds
  (MIN_TRADES_PER_YEAR=40, MIN_SHARPE=1.5)

TASK 4: Run SHAP on weekly experiment
python -m core.engine.shap_runner --exp-id [weekly_exp_id]

TASK 5: Update all docs. Log as D49.
```

**Success check:** Weekly experiment runs with CD=2016, tier="weekly" in results,
tier gates applied correctly, SHAP done.

---

## STEP 14 -- D51: Dataset Hardening (BLOCKS ALL NEW EXPERIMENTS)
**Window:** 1
**Status:** [x] COMPLETE (2026-03-03)
**Depends on:** D50 complete (DONE)
**Why:** We have run 9 experiments on 100% of 648K rows. No holdout exists.
Embargo is 48 bars (should be 288). Availability masks missing. These are
critical integrity issues that must be fixed before autonomous grinding begins.

**CC Prompt:**
```
Read CLAUDE.md, STRATEGY_LOG.md, and THE_PLAN.md before starting.

D51: Dataset hardening. Four tasks. Do them in order. Do not skip any.

TASK 1: Carve out final holdout
  Load data/labeled/BTCUSDT_5m_labeled_v3.parquet.
  Identify the last 12 months of bars by timestamp.
  Write holdout to: data/holdout/BTCUSDT_5m_holdout_v3.parquet (create folder)
  Write training set to: data/labeled/BTCUSDT_5m_labeled_v3_train.parquet
  Print: holdout rows, training rows, holdout date range, training date range.
  Assert: train.index.max() < holdout.index.min()
  Assert: len(holdout) > 100000
  NEVER load holdout in any future experiment. It exists only for final validation.

TASK 2: Fix embargo in simulator.py
  Change embargo_bars from 48 to 288 everywhere in core/engine/simulator.py.
  Add comment: "288 bars = 24h = max feature lookback per AFML Ch.7"
  Verify change propagates through all fold construction logic.
  Also: add purging of training samples whose labels extend into test period.

TASK 3: Add availability masks to augment_features()
  In core/engine/simulator.py augment_features():
  has_oi: 1 where OI columns (oi_btc, oi_usdt, etc.) are non-NaN, else 0
  has_liqs: 1 where liquidation columns are non-NaN, else 0
  Add both to ONTHEFLY_FEATURES. Both are trivially causal (based on timestamps).
  Do NOT impute NaN with zeros -- LightGBM handles NaN natively, which is correct.

TASK 4: Dataset manifest
  Create data/labeled/manifest_v3.json:
  {
    "version": "3.0.0",
    "created": "2026-03-03",
    "rows_total": 648288,
    "rows_train": <computed>,
    "rows_holdout": <computed>,
    "holdout_start": "<timestamp>",
    "cols": 594,
    "train_hash": "<SHA256>",
    "holdout_hash": "<SHA256>",
    "known_issues": [
      "embargo was 48 bars in E000-E008 (correct is 288)",
      "no holdout in E000-E008 experiments",
      "swing confirmation lag unverified in rules.py"
    ]
  }

TASK 5: Update simulator.py data path
  Default data path: data/labeled/BTCUSDT_5m_labeled_v3_train.parquet
  Add guard: if "holdout" in path: raise RuntimeError("Do not load holdout in experiments!")

TASK 6: Smoke test
  python -c "
  import pandas as pd, json
  train = pd.read_parquet('data/labeled/BTCUSDT_5m_labeled_v3_train.parquet')
  holdout = pd.read_parquet('data/holdout/BTCUSDT_5m_holdout_v3.parquet')
  assert train.index.max() < holdout.index.min(), 'FAIL: holdout leaks into train'
  assert len(holdout) > 100000, 'FAIL: holdout too small'
  print(f'Train: {len(train):,} rows through {train.index.max()}')
  print(f'Holdout: {len(holdout):,} rows {holdout.index.min()} to {holdout.index.max()}')
  print('D51 smoke test: PASS')
  "

Log as D51 in STRATEGY_LOG.md. Update CLAUDE.md. Update THE_PLAN.md.
```

**Success check:** "D51 smoke test: PASS" printed. manifest_v3.json written.

---

## STEP 15 -- D52: Optuna Integration + Parameter Search Space
**Window:** 2
**Status:** [x] COMPLETE (2026-03-03)
**Result:** Optuna 4.7.0. core/config/parameters.py (22 params, 5 groups, ~100T combos).
core/engine/optuna_optimizer.py (814 lines): GT-Score objective, TPESampler(multivariate+
constant_liar), MedianPruner, JournalFileStorage+OpenLock (Windows), DSR computation.
Smoke test 7/7 PASS. No real data loaded. Ready for autonomous trials after D53.
**Depends on:** D51 Task 6 PASS (simulator.py updated to use _train.parquet)
**Why:** JSON registry is fragile for parallel. Raw Sharpe is wrong objective.
No systematic parameter search. Optuna + GT-Score unlocks autonomous grinding.

**CC Prompt:**
```
Read CLAUDE.md, STRATEGY_LOG.md, and THE_PLAN.md before starting.

D52: Optuna integration. Three tasks.

TASK 1: Install and verify Optuna
  pip install optuna --break-system-packages
  python -c "import optuna; print(optuna.__version__)"

TASK 2: core/config/parameters.py -- full search space

Create SEARCH_SPACE dict. Each entry: {default, options, description, experiments_tested}.
Use these research-validated values:

SEARCH_SPACE = {
  "ml_threshold":   {"default": 0.60, "options": [0.55, 0.57, 0.60, 0.62, 0.65, 0.70, 0.75, 0.80]},
  "cooldown_bars":  {"default": 576,  "options": [144, 288, 576, 864, 1152, 2016, 4032, 8640]},
  "target_r":       {"default": 2.0,  "options": [1.0, 1.5, 2.0, 2.5, 3.0]},
  "max_bars":       {"default": 36,   "options": [12, 24, 36, 48, 96, 144, 288]},
  "stop_atr_mult":  {"default": 1.0,  "options": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]},
  "entry_type":     {"default": "market", "options": ["market", "limit_ob_mid", "limit_fvg_edge"]},
  "direction":      {"default": "long",   "options": ["long", "short"]},
  "ob_lookback":    {"default": 15,   "options": [5, 10, 15, 20, 30]},
  "ob_age_cap":     {"default": 192,  "options": [96, 144, 192, 288, 576]},
  "ob_mitigation":  {"default": "50pct", "options": ["wick_touch", "50pct", "close_through"]},
  "fvg_min_size_atr": {"default": 0.50, "options": [0.10, 0.20, 0.35, 0.50, 0.75]},
  "fvg_age_cap":    {"default": 100,  "options": [24, 48, 100, 144, 288]},
  "swing_n_internal": {"default": 5,  "options": [3, 5, 7, 10]},
  "swing_n_external": {"default": 10, "options": [7, 10, 15, 20]},
  "liq_tolerance_atr": {"default": 0.10, "options": [0.05, 0.10, 0.20, 0.30]},
  "displacement_atr_mult": {"default": 1.5, "options": [1.0, 1.5, 2.0, 3.0]},
  "breaker_age_cap": {"default": 200, "options": [96, 144, 200, 288, 576]},
  "pd_lookback":    {"default": 96,   "options": [48, 96, 144, 288]},
  "ote_fib_low":    {"default": 0.618, "options": [0.50, 0.618, 0.65]},
  "ote_fib_high":   {"default": 0.786, "options": [0.786, 0.79, 0.85]},
  "hmm_bull_threshold": {"default": 0.60, "options": [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]},
  "kelly_divisor":  {"default": 40.0, "options": [20.0, 30.0, 40.0, 50.0]},
  "tier":           {"default": "standard", "options": ["standard", "weekly", "monthly"]},
}

PARAMETER_GROUPS = {
  "quick_wins":    ["ml_threshold", "cooldown_bars"],
  "label_config":  ["target_r", "max_bars", "stop_atr_mult", "entry_type", "direction"],
  "ict_structure": ["ob_lookback", "ob_age_cap", "ob_mitigation", "fvg_min_size_atr",
                    "fvg_age_cap", "swing_n_internal", "swing_n_external",
                    "displacement_atr_mult", "breaker_age_cap", "pd_lookback"],
  "regime":        ["hmm_bull_threshold"],
  "sizing":        ["kelly_divisor"],
}

Helper functions:
  get_untested_options(param_name, registry) -> list of untested values
  record_tested(param_name, value, registry) -> None

TASK 3: core/engine/optuna_optimizer.py -- Optuna wrapper

Create alongside existing optimizer.py (do not break it).

Storage: JournalFileStorage (never SQLite -- locking issues with parallel workers)
  storage = optuna.storages.JournalStorage(
    optuna.storages.JournalFileBackend("optuna_journal.log")
  )

Sampler: TPESampler(multivariate=True, constant_liar=True, seed=42)
  multivariate=True: captures parameter correlations
  constant_liar=True: prevents duplicate exploration by parallel workers

Pruner: MedianPruner(n_startup_trials=10, n_warmup_steps=2)
  Don't prune until 10 trials complete and never before fold 2.
  After each fold: trial.report(np.mean(fold_scores), fold_idx)
  For pruned trials: return mean of completed folds (not raise TrialPruned())
  so TPE can learn from partial information.

Objective (GT-Score composite):
  # Exponential recency weighting (recent folds matter more for BTC regimes)
  half_life = 3  # folds
  n = len(fold_scores)
  weights = np.exp(-np.log(2)/half_life * np.arange(n)[::-1])
  weights /= weights.sum()
  weighted_mean = np.dot(weights, fold_scores)
  weighted_std = np.sqrt(np.dot(weights, (fold_scores - weighted_mean)**2))

  # BIC-style complexity penalty
  T = n_training_rows
  k = n_features * np.log2(n_leaves)  # effective complexity
  bic_penalty = (k * np.log(T)) / (2 * T)

  # Feature count penalty
  feature_penalty = 0.001 * n_features

  # Hard floor: reject if any fold is catastrophic
  if min(fold_scores) < -0.5:
    return -999.0

  score = weighted_mean - 0.3 * weighted_std - bic_penalty - feature_penalty
  return score

Smoke test: run 3 trials manually. Verify JournalStorage creates log file.
  Verify pruner fires after fold 3 on a deliberately bad trial.
  Verify 2 parallel processes don't collide.

Log as D52 in STRATEGY_LOG.md. Update CLAUDE.md. Update THE_PLAN.md.
```

**Success check:** Optuna study created, 3 test trials run, JournalStorage log exists.

---

## STEP 16 -- D53: ICT Rules Overhaul (rules.py)
**Window:** 3 (or Window 1 after D51 completes)
**Status:** [x] COMPLETE (2026-03-03)
**Result:** 6 new functions + 4 enhanced + param corrections. 205 new on-the-fly
columns. ONTHEFLY_FEATURES: 19 -> 224. 18/18 causality tests PASS. 15.2s runtime.
**Depends on:** D51 complete (holdout + embargo fix before running new experiments)
**Spec:** D53_IMPLEMENTATION_SPEC.md -- READ THIS BEFORE WRITING ANY CODE.
**Why:** Two GPT research responses on ICT rules reconciled (2026-03-03). Current
rules.py has wrong parameter defaults, OBs anchored incorrectly (to BOS, not to
displacement), no sweep detection, no standalone displacement, wrong CISD algorithm,
and coarse P/D encoding. Correcting these is the highest-AUC-impact work remaining.
Expected uplift from displacement anchor fix alone (ob_age = #3/#4 SHAP): > 0.01 AUC.

**CC Prompt:**
```
Read CLAUDE.md, MEMORY.md, STRATEGY_LOG.md, THE_PLAN.md, and
D53_IMPLEMENTATION_SPEC.md before writing any code.
D53_IMPLEMENTATION_SPEC.md is the authoritative spec for this task.
All parameter values and algorithms there override anything in CLAUDE.md or THE_PLAN.md.

TASK: Implement D53 -- ICT rules overhaul in core/signals/ict/rules.py.

Execute tasks in priority order: F -> A -> G -> H -> D -> B -> C -> E -> MSS.
After EACH task: run causality test at T=10000 before proceeding to next task.
Do not proceed past any function that fails causality.

After ALL tasks complete:
  1. Run full causality suite at T in [1000, 5000, 10000, 50000] for all functions.
  2. Print swing confirmation lag correlation for internal layer:
       corr(int_swing_high, target) at lag-0  -- must be near 0
       corr(int_swing_high, target) at lag-5  -- must be materially higher
     If lag-0 is non-trivial: STOP, find and fix the lookahead bug first.
  3. Run smoke test on 10,000 bars. Print coverage % for every new feature family.
     Flag anything outside the target ranges in D53_IMPLEMENTATION_SPEC.md.
  4. Update simulator.py ONTHEFLY_FEATURES with all new column names (17 -> ~199).
  5. Update data/labeled/feature_catalog_v3.yaml with all new features.
     Column names must exactly match ONTHEFLY_FEATURES keys.
  6. Log as D53 in STRATEGY_LOG.md. Update CLAUDE.md and THE_PLAN.md.

ENVIRONMENT:
Python 3.14, Windows, cp1252 encoding.
ASCII only in print statements. No Unicode symbols.
No numba. Use stateful loops for OB/FVG/sweep state machines.
Do not vectorise where per-bar mutable state is needed.
All functions pure and strictly causal (at bar T, use only df.iloc[:T+1]).

DO NOT:
  - Run experiments (that is D54+)
  - Touch holdout data (D51 may not be complete)
  - Modify embargo or walk-forward logic
  - Delete or rename any existing function (add new ones alongside)
```

**Success check:**
- "CAUSALITY PASS" at all four T values for every function
- Swing lag-0 correlation printed and near 0
- Coverage stats printed for all feature families, within spec targets
- ONTHEFLY_FEATURES count grows from 17 to ~199
- Feature catalog names exactly match ONTHEFLY_FEATURES keys


---

## QUICK REFERENCE — CURRENT STATE

| Metric | Value |
|--------|-------|
| Dataset | v3 (594 cols). Train: 543,167 rows. Holdout: 105,121 rows (D51) |
| Embargo | 288 bars (24h) with label purging (D51, AFML Ch.7) |
| ML AUC | 0.7938 (D55b_tier1_only, 64 features) |
| Production WR | 74.93% (D55b, calibrated) |
| Production EV | +1.198R (D55b) |
| Production MaxDD | 7.71% (D55b) |
| Sharpe | 12.29 (D55b, annualized) |
| Trades/yr | 178.3 |
| Features | 64 (pruned from 670 in D55, zero AUC loss) |
| CSCV | PBO 0%, PSR 1.0 -- ALL 10/10 PASS |
| Short AUC | 0.7966 (D54c_baseline_short) |
| Short WR | 73.15% (D54c, calibrated) |
| Short EV | +1.1446R (D54c) |
| Short MaxDD | 5.47% (D54c) |
| Registry | 49 experiments (E000-E008 + D54a + D54c + 37 Optuna), D54a/D54c = baselines |
| ONTHEFLY_FEATURES | 224 (19 pre-D53 + 205 D53 ICT overhaul) |
| SHAP top D53 | ote_dist_from_705_atr: #8 short, #15 long (NEW in top 30) |
| SHAP overlap | 7/10 top features shared long/short (was 8/10 pre-D51) |
| Last completed | D55: feature prune 670->64, AUC 0.7938, 10/10 PASS |
| Next step | Optuna autonomous search with lean 64-feature model |

---

## DECISIONS LOG (quick ref)
Full detail in STRATEGY_LOG.md (project root)

D47: [x] DONE -- Regime filter: 3-state HMM (pure NumPy) + ADX composite + 3 interactions. 10 features, 12/12 causality PASS. Registered in ONTHEFLY_FEATURES.
D47b: [x] DONE -- Regime RQs: RQ6 (HMM hard gate, signal_filter) + RQ7 (soft inputs). signal_filter support in simulator.py. Old RQ6/RQ7 replaced.
D48: [x] DONE -- E004_short_baseline: AUC 0.7981, WR 71.9%, EV +1.107R, MaxDD 6.28%, 10/10 PASS. SHAP: 8/10 top features shared with long, swing_low rises to #1. PROMOTED.
D50: [x] DONE -- Regime experiments: E007_rq6 (HMM gate, 9/10, RQ6: YES quality/NO freq), E008_rq7 (soft inputs, 10/10, RQ7: NO). 7/9 regime features prunable. SHAP: hmm_prob_bull #153.
Research synthesis: [x] DONE -- 6 reports reviewed. D51-D53 architecture decisions made.
D51: [x] COMPLETE -- Holdout 105,121 rows, embargo 288, label purging, has_oi/has_liqs, manifest, holdout guard
D52: [x] COMPLETE -- Optuna 4.7.0, optuna_optimizer.py (814 lines), parameters.py (22 params, 5 groups), GT-Score, DSR, 7/7 smoke PASS
D53: [x] COMPLETE -- ICT rules overhaul: 6 new functions + 4 enhanced + param corrections. 205 new cols. ONTHEFLY 19->224. 18/18 causality PASS.
D54a: [x] COMPLETE -- Post-D51/D53 long baseline. 670 features, AUC 0.7933, WR 76.0%, EV +1.23R, Sharpe 12.8, 10/10 PASS. Regression-safe vs E002_prune. SHAP: ote_dist_from_705_atr NEW #15.
D54c: [x] COMPLETE -- Post-D51/D53 short baseline. AUC 0.7966, WR 73.2%, EV +1.14R, Sharpe 11.5, MaxDD 5.47%, 10/10 PASS. E004 superseded. SHAP: 7/10 shared, ote_dist_from_705_atr #8.
D55: [x] COMPLETE -- Feature prune: 670->64 (90.4% pruned), AUC 0.7938 (within 0.002 of D54a). D55b_tier1_only 10/10 PASS. 7/10 D53 families dead. 3 survived: OTE-705, P/D, dual-swing.
D49: [x] DONE -- E004_rq5 (OI, 10/10, RQ5: NO) + E005_trq3 (weekly, 10/10, 52/yr, TRQ3: YES)
D46b: Engine integration: label_config + fill model wired into simulator, dual-tier (standard/weekly/monthly), TIER_CONFIGS + 3 TRQs in optimizer, tier-aware gates, 5/5 smoke PASS.
D46a: Dynamic labeler + fill model built. labeler.py (350 lines), fill_model.py (142 lines). ATR stop: 100% match both directions. Fill model: 83% fill rate. 3/3 smoke tests PASS.
D46: E003_rq4: breaker blocks tested. All 4 features below prune threshold (#125-#178). RQ4: NO. 10/10 PASS but no promotion over E002. cvd_true_daily #45 is only useful v3 feature.
D45: TradingView Pine Script export: ict_strategy_v1.pine (388 lines), 5 SHAP conditions, export_to_pine.py generator. Step 10 DONE.
D44: Breaker block encoding: detect_breaker_blocks() in rules.py, 6 on-the-fly features, 11/11 causality PASS. RQ4 unblocked.
D43: E002_prune + SHAP: 10/10 PASS, ECE 0.016. SHAP: ote_dist #1, ob_quality #99. RQ1 answered NO.
D42: Feature engineering for RQ1: compute_ob_quality() in rules.py, on-the-fly augmentation, E001_rq1 ready.
D41b: Optimizer + SHAP runner built. First proposal: RQ1 (OB Quality Score).
D37b: True tick CVD pipeline: 74 months, Pearson vs CLV = 0.50, downloading.
D41a: ICT knowledge base (643 lines, 8 sections) + calibrator.py.
D40: Engine validation: reproduces D35 (7/7 metrics match).
D39: ICT signal migration: 9 causal functions, all causality PASS.
D38: Shared engine: 4 modules, smoke tested end-to-end.
D37a: OI metrics downloaded: 447k rows, 52 monthly parquets.
D37: Folder restructure complete. New architecture in place.
D36: SHAP — ICT structural = alpha. OB age #1, OTE dist #2. 81 dead features.
D35: Production v2 validated. T2+CD576 confirmed. CSCV all pass.
D34: ML v2 retrain: AUC 0.7937 (+0.012), 508 features.
D33: Dataset enrichment: v2 parquet, 122 new TA + ICT session features.
D32: Kelly (1/40) selected, Sharpe 9.85, CAGR 1711% (backtest only).
D31: Cooldown sweep: CD=576 (48h) selected, 179/yr.
D30: Execution costs: +0.027R/trade, all GO.
D29: ML backtest T2: WR 67.3%, EV +0.97R, CSCV all pass.
D28: LightGBM 11-fold walk-forward, AUC 0.7819.
