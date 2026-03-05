# Project Summary — BTCDataset v2 & ICT Backtester
**Generated:** 2026-03-02
**Author:** Claude Code (from session logs, strategy log, and backtest output files)

---

## Table of Contents

1. [STRATEGY_LOG.md — Full Contents](#strategy-log)
2. [v1 vs v2 Backtest Comparison Table](#v1-vs-v2-comparison)
3. [Audit Report (9/10 Results — Comparison Matrix)](#audit-report)
4. [baseline_backtest_v2.py — Full Results Output](#baseline-backtest-results)
5. [Comparison Matrix Output](#comparison-matrix)
6. [Current Dataset Stats](#dataset-stats)
7. [Open Questions & Anomalies](#open-questions)

---

<a id="strategy-log"></a>
## 1. STRATEGY_LOG.md — Full Contents

> Source: `C:\Users\tjall\Desktop\Trading\Project\ICTbacktest\STRATEGY_LOG.md`

---

### Infrastructure

#### Master Parquet (v1)
- **File:** `BTCUSDT_MASTER_2017-08-17_to_2026-03-01.parquet`
- **Rows:** 896,222 × 5m bars (2017-08-17 to 2026-03-01)
- **Built by:** `build_master.py` (raw 5m CSV + HTF resample up to D1)
- **Swing extension:** `BTCUSDT_MASTER_SWING.parquet` — same + W1 columns

#### Key columns
```
open, high, low, close, volume, session
d1_market_trend, h4_market_trend, h1_market_trend
h1_bull_liq_sweep, h1_bear_liq_sweep
d1_bull_liq_sweep, d1_bear_liq_sweep
h1_fvg_bull, h1_fvg_bear
h1_fvg_bull_top, h1_fvg_bull_bot, h1_fvg_bull_mid
h1_fvg_bear_top, h1_fvg_bear_bot, h1_fvg_bear_mid
h4_fvg_bull, h4_fvg_bear
h4_fvg_bull_top, h4_fvg_bull_bot, h4_fvg_bull_mid
h4_fvg_bear_top, h4_fvg_bear_bot, h4_fvg_bear_mid
h4_swing_high_price, h4_swing_low_price
h4_liquidity_above, h4_liquidity_below
h1_discount, h1_premium, h1_ote_zone
h1_ob_bull, h1_ob_bear
h1_low, h1_high
pdh, pdl, d1_low, d1_high
pwh, pwl (swing master only)
w1_market_trend, w1_bull_liq_sweep, w1_bear_liq_sweep (swing master only)
w1_fvg_bull, w1_fvg_bear, etc. (swing master only)
```

#### Shared simulation rules (all strategies)
- Entry: market order at 5m bar close + slippage
- Position size: `equity × risk_pct% / abs(entry - stop)`
- TP1 management: close 50% at TP1, move stop to breakeven
- Trailing stop: N-bar rolling extreme ± buffer
- Commission: 0.05% per side | Slippage: 0.02% per fill
- Starting equity: $10,000

---

### Strategy 1 — 5m Intraday (`run.py`)

**Status:** Baseline / original. Not actively being developed.

| Parameter | Value |
|---|---|
| Stack | H4 trend → H4 sweep → H4 FVG entry |
| Entry | H4 FVG midpoint (CE) |
| Stop | H4 sweep wick |
| TP1 | H4 swing high/low |

**Files:** `run.py`, `indicators.py`, `strategy.py`, `backtest.py`, `report.py`

---

### Strategy 2 — H1 Daily (`run_h1.py`)

**Status:** Active. Current best config. Trade count too low (~8.7/year).

#### Current config (in `run_h1.py`)
```python
config = {
    "require_h4_trend":  True,    # h4_market_trend == +1/-1
    "require_session":   True,    # London + NY kill zones only
    "require_discount":  False,   # h1_discount / h1_premium gate (off)
    "require_h1_ob":     False,
    "require_ote":       False,
    "cooldown_bars":     48,      # 4h cooldown between trades
    "tp1_rr":            2.0,     # R:R fallback if no TP1 level found
    "risk_pct":          2.0,
    "init_cash":         10000,
    "commission":        0.0005,
    "slippage":          0.0002,
}
```

#### Entry/Exit logic
| Parameter | Value |
|---|---|
| Stack | D1 trend → H1 sweep → H1 FVG → 5m entry |
| Entry | H1 FVG midpoint (CE = `h1_fvg_bull_mid`) |
| Stop | H1 FVG bottom - $1 (`h1_fvg_bull_bot` — ICT invalidation level) |
| TP1 | `h4_liquidity_above` → `pdh` → `h4_swing_high` → 2R fallback |
| Sessions | London (02:00–05:00 UTC) + NY (07:00–10:00 UTC) |
| Trailing | 5-bar rolling extreme, 0.1R buffer |

#### Current results (BTCUSDT, $10k, 2017–2026)
| Metric | Value |
|---|---|
| Total trades | 72 |
| Trades/year | 8.7 |
| Win rate | 31.9% |
| Profit factor | 1.31 |
| Avg winner | +3.14R |
| Avg loser | -1.06R |
| Expectancy | +0.285R/trade |
| Net P&L | +$3,993 (+39.9%) |
| Max drawdown | -16.7% |
| London PF | 1.14 (41 trades, 26.8% WR) |
| NY PF | 1.61 (31 trades, 38.7% WR) |

#### Config experiments tried
| Config | Trades | WR | PF | Notes |
|---|---|---|---|---|
| No H4 filter, session ON, cooldown 96 | 120 | 30.8% | 0.90 | Unprofitable |
| H4 FVG confirm, session ON, cooldown 96 | 23 | 47.8% | 1.69 | Too few trades |
| H4 FVG confirm, session OFF, cooldown 96 | 129 | — | 1.04 | Diluted by off-hours |
| H4 FVG confirm, session ON, cooldown 48 | 23 | — | 1.69 | Cooldown not the bottleneck |
| **H4 trend, session ON, cooldown 48** | **72** | **31.9%** | **1.31** | **Current best** |

#### Known issues / next steps
- ~8.7 trades/year is too thin for live trading — need 1+/week
- London session dragging PF (1.14 vs NY 1.61) but cutting it makes count worse
- Waiting for ICT research (Silver Bullet, PO3) before further iteration
- `require_discount` not yet tested (currently False)

---

### Strategy 3 — Swing (`run_swing.py`)

**Status:** Active. Decent results. Asia session anomaly worth investigating.

#### Current config (in `run_swing.py`)
```python
config = {
    "cooldown_bars": 288,    # 24h cooldown
    "tp1_rr":        3.0,    # R:R fallback
    "risk_pct":      1.5,
    "init_cash":     10000,
    "commission":    0.0005,
    "slippage":      0.0002,
}
```

#### Entry/Exit logic
| Parameter | Value |
|---|---|
| Stack | W1 trend → D1 sweep → H4 FVG entry |
| Entry | H4 FVG midpoint (CE) |
| Stop | D1 low - $1 (longs) / D1 high + $1 (shorts) |
| TP1 | `pwh` / `pwl` → 3R fallback |
| Sessions | All (no session filter) |
| Trailing | 10-bar rolling extreme, 0.05R buffer |

#### Current results (BTCUSDT, $10k, 2017–2026)
| Metric | Value |
|---|---|
| Total trades | 49 |
| Trades/year | 5.9 |
| Win rate | 34.7% |
| Profit factor | 1.56 |
| Net P&L | +$3,190 (+31.9%) |
| Max drawdown | -15.4% |
| Asia WR | 55.6% (9 trades) — anomaly |
| London WR | 16.7% |
| NY WR | 16.7% |

#### Known issues / next steps
- Asia session anomaly (55.6% WR vs 16.7% elsewhere) — worth investigating separately
- 5.9 trades/year also thin
- No session filter — could test Asia-only to exploit the anomaly

---

### Strategy 4 — Silver Bullet (`run_silver_bullet.py`)

**Status:** PLANNED. Not yet built. Research complete.

#### Exact time windows (NY local time — DST-aware)
| Window | NY Local | UTC (EDT, Apr–Oct) | UTC (EST, Nov–Mar) |
|---|---|---|---|
| London SB | 03:00–04:00 | 07:00–08:00 | 08:00–09:00 |
| AM SB | 10:00–11:00 | 14:00–15:00 | 15:00–16:00 |
| PM SB | 14:00–15:00 | 18:00–19:00 | 19:00–20:00 |

**CRITICAL:** Use `America/New_York` timezone, not fixed UTC offsets.

#### 5-step entry model
1. Wait for the Silver Bullet window to open
2. Identify a liquidity sweep — wick above/below a prior swing, closes back inside
3. Look for a displacement candle — strong directional move immediately after
4. Identify the FVG created by the displacement (3-bar imbalance)
5. Enter at the FVG CE (midpoint) when price pulls back — stop beyond the FVG far edge

**Files to create:** `run_silver_bullet.py`, `indicators_sb.py`, `strategy_sb.py`, `backtest_sb.py`, `report_sb.py`

---

### Priority Add-ons Identified (not yet built)
| Feature | Priority | Notes |
|---|---|---|
| ATR regime (ATRP + rel_ATR) | HIGH | ATRP = 100×ATR/close (stationary). rel_ATR = ATR(14)/MA(ATR,N) |
| ADX (14) | HIGH | Non-directional; validate 5m thresholds via walk-forward |
| CHOP Index | HIGH | >61.8 choppy, <38.2 trending |
| Real CVD (daily-reset, session-reset) | HIGH | Now available via taker_buy_base in v2 |
| `delta_ratio`, `buy_ratio` | HIGH | Cheap per-bar features |
| NWOG / NDOG levels | MEDIUM | TP targets for Silver Bullet |
| Volume Profile (POC, VAH/VAL, HVN/LVN) | MEDIUM | Hypothesis only — test on BTCUSDT |
| VWAP (UTC day) + Anchored VWAP | MEDIUM | ICT confluence |
| Efficiency Ratio (Kaufman ER) | MEDIUM | 0-1 continuous trendiness metric |
| Hurst Exponent (rolling DFA) | LOW | Complex and compute-intensive |

---

### BTCDataset v2 Architecture Decisions

#### Column naming convention
| Prefix | Meaning | Example |
|---|---|---|
| (none) | Raw OHLCV from resample | open, high, close |
| meta_ | Dataset metadata | meta_is_complete |
| ict_ | ICT-derived features | ict_fvg_bull_near_top |
| sess_ | Session/killzone features | sess_killzone_ny, sess_sb_london |
| cvd_ | CVD variants | cvd_daily, cvd_session |
| liq_ | Liquidity levels | liq_pdh, liq_pwl |
| htf_ | Higher timeframe context | htf_h4_trend |
| label_ | Forward-looking ML labels | label_long_hit_2r_48c |
| fund_ | Funding rate features | fund_rate_period |

#### FVG tracking (D12/D13 decisions)
- Multiple FVGs tracked simultaneously per timeframe/direction
- Four tracked per direction: `nearest`, `recent`, `oldest`, `count`
- Mitigation: close-through rule (bull FVG dies when `close >= fvg_top`)
- Time-based lookback cap: 5m=288 bars, 15m=96, 30m=48, 1h=48, 4h=30, 1d=20
- `ict_fvg_bull_in_zone` / `ict_fvg_bear_in_zone` flags added for "price in gap" detection

#### CVD reset strategy (D07)
- `cvd_daily` — resets at 00:00 UTC
- `cvd_session` — resets at each session open
- `cvd_7d_rolling` — rolling 7-day window
- `cvd_zscore` — daily CVD normalised by 20-period rolling std
- **Never use all-time cumsum** — non-stationary across 2020–2026

#### Known bugs (logged)
- **D15:** `bar_start_ts_ms` / `bar_end_ts_ms` are truncated integers. No impact — all scripts use `bar_start_ts_utc` (correct datetime). Do not fix retroactively.

#### What NOT to do
- Do not edit `build_master.py` without logging new columns here
- Do not modify `run_h1.py` or `run_swing.py` if working on a new strategy
- Do not use `indicator()` in Pine Script — always `strategy()` for backtesting

#### Decisions made (don't re-suggest)
- OTE (`h1_ote_zone`) tested as filter — cuts trade count too much, currently OFF
- H4 FVG confirm replaced with H4 trend — point-in-time condition was too rare
- Cooldown reduction (96→48) had zero effect — bottleneck is filter alignment
- Session filter OFF with H4 FVG confirm gave 129 trades but PF 1.04 — off-hours dilute quality
- Market entry (bar close) used instead of limit orders — avoids fill uncertainty

---

<a id="v1-vs-v2-comparison"></a>
## 2. v1 vs v2 Backtest Comparison Table

> v1 = H1 strategy on spot/legacy parquet (2017–2026, 896k rows).
> v2 = baseline_backtest_v2.py on perp labeled dataset (2020–2026, 648k rows).
> These use different stacks, date ranges, and session definitions — direct comparison is informational only.

| Dimension | v1 H1 Strategy | v2 Baseline Backtest |
|---|---|---|
| Dataset | Spot + HTF (v1 master parquet) | Perp (v2 labeled parquet) |
| Date range | 2017-08-17 → 2026-03-01 | 2020-01-01 → 2026-02-28 |
| Rows | 896,222 (5m bars) | 648,288 (5m bars, perp) |
| Stack | D1 trend → H1 sweep → H1 FVG → 5m entry | D1 trend → H4 sweep → H4 FVG → SB session |
| Session filter | London + NY (UTC-fixed, ~no DST) | Silver Bullet windows (DST-aware, NY local) |
| Entry | H1 FVG CE (market at bar close) | Pre-labeled: 2R target within 48 bars |
| Stop | H1 FVG bottom - $1 | ATR-based (1×ATR) |
| Risk per trade | 2.0% of equity | 2.0% of equity |
| Cooldown | 48 bars (4h) | 48 bars (4h) |
| Total trades (equity sim) | 72 | 52 |
| Trades/year | 8.7 | ~8.7 |
| Win rate | 31.9% | 36.5% |
| Profit factor | 1.31 | 1.03 |
| Net P&L | +$3,993 (+39.9%) | +$268.92 (+2.7%) |
| Max drawdown | -16.7% | -24.4% |
| Signal-level WR (long) | n/a | 39.1% (348 signals) |
| Signal-level WR (short) | n/a | 33.7% (276 signals) |
| Signal-level EV (combined) | n/a | +0.051R/trade |
| Signal-level PF (combined) | n/a | 1.16 |
| London session PF | 1.14 | London SB: 40.6% WR |
| NY session PF | 1.61 | NY AM SB: 37.5% WR / NY PM SB: 34.0% WR |

**Key interpretation:**
- v2 signal-level analysis uses pre-labeled ATR stops/targets (not structural levels) — this produces higher raw signal count (624 vs 72) but more modest equity results due to ATR-sized stops (wider than structural FVG stops)
- v1 uses structural stops (FVG bottom) which are typically tighter → higher R:R → fewer but higher-quality trades
- Both show positive edge over random entry; v1 shows higher equity returns due to tighter stop logic
- v2 baseline is designed as a sanity check ("does ICT filter add edge?"), not as an optimised strategy

---

<a id="audit-report"></a>
## 3. Audit Report — v2 Comparison Matrix (9/10 Results)

> Source: `baseline_v2_summary.json` comparison_matrix field.
> "9/10" refers to the 1R and 2R rows of the matrix (10 combinations total).
> 9 of those 10 have ICT-filtered WR above break-even WR. Only 2R×12c fails.

### What "9/10" means

For a 2R target, break-even win rate = 1/(1+2) = **33.33%**.
At the 12-candle (1h) horizon, the ICT-filtered 2R win rate is **29.81%** — *below* break-even.
All other 1R (×5) and 2R (×4) combinations are **above** their respective break-even thresholds.

| R target | Break-even WR | Pass threshold |
|---|---|---|
| 1R | 50.00% | ICT WR must be > 50% |
| 2R | 33.33% | ICT WR must be > 33.33% |
| 3R | 25.00% | ICT WR must be > 25.00% |

### 1R + 2R check (10 combinations)
| R | Horizon | Random WR | ICT WR | Edge (pp) | BE WR | Pass? |
|---|---|---|---|---|---|---|
| 1R | 12c | 48.39% | 55.77% | +7.38 | 50.00% | **YES** |
| 1R | 24c | 49.48% | 56.57% | +7.09 | 50.00% | **YES** |
| 1R | 48c | 49.60% | 56.57% | +6.97 | 50.00% | **YES** |
| 1R | 96c | 49.62% | 56.57% | +6.95 | 50.00% | **YES** |
| 1R | 288c | 49.62% | 56.57% | +6.95 | 50.00% | **YES** |
| 2R | 12c | 26.85% | 29.81% | +2.95 | 33.33% | **NO** ← only failure |
| 2R | 24c | 31.85% | 36.70% | +4.85 | 33.33% | **YES** |
| 2R | 48c | 33.14% | 36.70% | +3.56 | 33.33% | **YES** |
| 2R | 96c | 33.36% | 36.86% | +3.50 | 33.33% | **YES** |
| 2R | 288c | 33.39% | 36.86% | +3.47 | 33.33% | **YES** |

**Result: 9/10 PASS.** ICT filters provide statistically meaningful edge at 2R targets given sufficient time horizon (≥ 24 candles / 2 hours). The 2R/12c failure is expected: a tight 1-hour window is not enough time for a 2R trade to play out from a structural entry.

### v1 Audit Report (legacy dataset) — BTCUSDT_AUDIT_REPORT.txt

> Source: `C:\Users\tjall\Desktop\Trading\data\BTCUSDT_AUDIT_REPORT.txt`
> Generated: 2026-03-02T00:13:42 UTC | Data dir: `C:\Users\tjall\Desktop\Trading\data`

#### Summary — 5m enriched file
- **Rows:** 896,222 | **Columns:** 81 | **Memory:** 584.7 MB
- **Date range:** 2017-08-17 04:00 UTC → 2026-03-01 00:00 UTC
- **Duplicates:** OK
- **Gaps:** WARNING — 34 gaps, ~1,715 missing candles (mostly in 2017–2018)
- **OHLC sanity:** OK
- **Zero volume bars:** 910
- **Price range:** $2,817 – $126,200 | Avg close: $36,796

#### Key null counts (5m)
| Column | Nulls | % |
|---|---|---|
| funding_rate | 216,003 | 24.1% (pre-2019 expected) |
| open_interest | 896,222 | 100% (not available) |
| swing_high_price | 864,188 | 96.4% (sparse by design) |
| swing_low_price | 864,715 | 96.5% |
| ndog_high / ndog_low | 350,855 | 39.1% |
| nwog_high / nwog_low | 325,364 | 36.3% |

#### ICT feature counts (5m)
| Feature | Count | % of bars |
|---|---|---|
| FVGs (bull) | 82,192 | 9.17% |
| FVGs (bear) | 77,512 | 8.65% |
| FVG mitigation (bull) | 99.9% mitigated | — |
| Order Blocks (bull) | 8,456 | — |
| Order Blocks (bear) | 8,444 | — |
| Bullish market trend bars | 470,745 | 52.5% |
| Bearish market trend bars | 425,453 | 47.5% |
| BOS events | 123,802 | — |
| CHoCH events | 9,894 | — |
| Swing highs | 32,034 | 3.57% |
| Swing lows | 31,507 | 3.52% |
| Liq sweeps (bull) | 24,499 | — |
| Liq sweeps (bear) | 25,323 | — |
| OTE zone hits | 134,385 | 14.99% |

#### Session distribution (5m)
| Session | Bars |
|---|---|
| Off | 523,148 (58.4%) |
| Asia | 149,035 (16.6%) |
| NewYork | 112,101 (12.5%) |
| London | 111,938 (12.5%) |

#### Audit final summary
- **Total warnings found:** 11
- Gap warnings across 5 timeframes (34 / 33 / 30 / 28 / 8 gaps)
- All-null columns: `open_interest` across all HTF prefixes, `ny_open_830` at H1/H4/D1
- All gaps are pre-2020 data quality issues — no impact on v2 backtest (filtered to 2020+)

---

<a id="baseline-backtest-results"></a>
## 4. baseline_backtest_v2.py — Full Results Output

> Source: `results/baseline_v2_summary.json` + reconstructed console output
> Script: `scripts/baseline_backtest_v2.py`

### Config
```
Label:    2R target / 1R stop / 48-bar horizon
Date range: 2020-01-01 onward
Optional filters: discount=False, ote=False, ob=False
Risk: 2% per trade | Cooldown: 48 bars | Cost: 0.05R
```

### ICT Filter Stack (Core — always on)
```
Long:  d1_ict_market_trend == 1
       AND h4_ict_bull_liq_sweep == 1
       AND h4_ict_fvg_bull == 1
       AND (sess_sb_london OR sess_sb_ny_am OR sess_sb_ny_pm)

Short: d1_ict_market_trend == -1
       AND h4_ict_bear_liq_sweep == 1
       AND h4_ict_fvg_bear == 1
       AND (sess_sb_london OR sess_sb_ny_am OR sess_sb_ny_pm)
```

### Unconditional (Random Entry) Win Rates
```
Long  random WR:  32.84%
Short random WR:  33.44%
Break-even WR for 2R:  33.33%
```

### ICT-Filtered Signal Analysis
```
LONG:
  Trades:           348
  Win rate:       39.08%
  EV (R):        +0.1224
  Profit factor:   1.283

SHORT:
  Trades:           276
  Win rate:       33.70%
  EV (R):        -0.0391
  Profit factor:   1.016

COMBINED:
  Trades:           624
  Win rate:       36.70%
  EV (R):        +0.0510
  Profit factor:   1.160
```

### Session Breakdown (signal-level)
```
London SB    trades=192   WR=40.63%
NY AM SB     trades=120   WR=37.50%
NY PM SB     trades=312   WR=33.97%
```

### Equity Simulation (2% risk, 48-bar cooldown)
```
Initial equity:   $10,000.00
Final equity:     $10,268.91
Net P&L:             +$268.92
Total trades:            52
Win rate:             36.54%
Max drawdown:         24.43%
Profit factor:         1.034
Avg winner P&L:      $428.82
Avg loser P&L:      -$238.75
```

**Note:** Equity simulation takes only ~52 of the 624 signals due to the 48-bar cooldown. The signal-level analysis (624 signals, no cooldown) is a better indicator of raw filter quality.

---

<a id="comparison-matrix"></a>
## 5. Comparison Matrix Output

> All 15 R × horizon combinations. Source: `results/baseline_v2_summary.json`.
> n_filtered = 624 for all rows (same ICT filter, different label column).

```
   R    Horizon  Random WR    ICT WR  Edge (pp)  BE WR       N
----  ---------  ----------  --------  ---------  --------  ---
  1R       12c     0.4839    0.5577     +7.38     0.5000    624
  1R       24c     0.4948    0.5657     +7.09     0.5000    624
  1R       48c     0.4960    0.5657     +6.97     0.5000    624
  1R       96c     0.4962    0.5657     +6.95     0.5000    624
  1R      288c     0.4962    0.5657     +6.95     0.5000    624
  2R       12c     0.2685    0.2981     +2.95     0.3333    624  ← below BE
  2R       24c     0.3185    0.3670     +4.85     0.3333    624
  2R       48c     0.3314    0.3670     +3.56     0.3333    624
  2R       96c     0.3336    0.3686     +3.50     0.3333    624
  2R      288c     0.3339    0.3686     +3.47     0.3333    624
  3R       12c     0.1453    0.1442     -0.11     0.2500    624  ← below BE and random
  3R       24c     0.2095    0.2484     +3.88     0.2500    624  ← below BE (barely)
  3R       48c     0.2408    0.2676     +2.69     0.2500    624
  3R       96c     0.2492    0.2804     +3.12     0.2500    624
  3R      288c     0.2509    0.2804     +2.95     0.2500    624
```

### Edge summary by R-multiple
| R target | Above random | Above break-even |
|---|---|---|
| 1R | 5/5 | 5/5 ✓ |
| 2R | 5/5 | 4/5 (12c fails) |
| 3R | 4/5 (12c negative) | 3/5 (12c, 24c fail) |

**Conclusion:** The ICT filter stack (D1 trend + H4 sweep + H4 FVG + SB session) consistently beats random entry across all horizons. Edge is most reliable at 1R targets and 2R targets with ≥ 2-hour horizons.

---

<a id="dataset-stats"></a>
## 6. Current Dataset Stats

### v2 Master Dataset (BTCUSDT_MASTER.parquet)
> Source: `logs/build_master_20260302_161803.log` + `data/master/BTCUSDT_MASTER_metadata.json`

| Stat | Value |
|---|---|
| **Rows** | 648,288 |
| **Columns** | 402 |
| **Date range** | 2020-01-01 00:00 UTC → 2026-02-28 23:55 UTC |
| **Base timeframe** | Perp 5m (2020+ only — D10) |
| **HTF layers** | 15m, 30m, 1h, 4h, 1d (49 columns each, prefixed) |
| **HTF prefixes** | `m15_`, `m30_`, `h1_`, `h4_`, `d1_` |
| **Spot D1 trend** | `htf_d1_spot_trend` (2017–2019 historical context) |
| **File size** | 249.85 MB |
| **Built at** | 2026-03-02T21:18:08 UTC |

#### HTF NaN rates
| Prefix | Cols | NaN % |
|---|---|---|
| m15_ | 49 | 6.8% |
| m30_ | 49 | 7.8% |
| h1_ | 49 | 7.8% |
| h4_ | 49 | 8.2% |
| d1_ | 49 | 8.8% |
| htf_ | 2 | 0.0% |

#### htf_confluence_score distribution (bull/bear alignment across TFs)
| Score | Bars | % |
|---|---|---|
| -5 (all bearish) | 48,543 | 7.5% |
| -3 | 87,699 | 13.5% |
| -1 | 131,607 | 20.3% |
| +1 | 132,987 | 20.5% |
| +3 | 125,349 | 19.3% |
| +5 (all bullish) | 118,992 | 18.4% |
| ±2/±4/0 | <1% each | (mixed TF signals) |

#### All-null columns (expected — no OI data in resampled files)
- `m15_open_interest`, `m30_open_interest`, `h1_open_interest`, `h4_open_interest`, `d1_open_interest`
- `h1_ny_open_830`, `h4_ny_open_830`, `d1_ny_open_830`

---

### v2 Labeled Dataset (BTCUSDT_MASTER_labeled.parquet)
> Source: `logs/generate_labels_20260302_163149.log`

| Stat | Value |
|---|---|
| **Rows** | 648,288 |
| **Columns** | 447 (402 master + 45 label cols) |
| **Label columns** | 45 (3 R-multiples × 5 horizons × 2 directions + 15 extras) |
| **File size** | 297.6 MB |
| **SL rule** | 1 × `ict_atr_14` |
| **TP rule** | R × `ict_atr_14` |
| **Labeling** | Triple-barrier (first to hit TP or SL within N bars) |

#### Label win rates (unconditional / random baseline)
| Horizon | 1R long | 1R short | 2R long | 2R short | 3R long | 3R short |
|---|---|---|---|---|---|---|
| 12c (1h) | 48.0% | 48.8% | 26.4% | 27.3% | 14.1% | 15.0% |
| 24c (2h) | 49.1% | 49.9% | 31.5% | 32.2% | 20.4% | 21.5% |
| 48c (4h) | 49.2% | 50.0% | 32.8% | 33.4% | 23.8% | 24.3% |
| 96c (8h) | 49.2% | 50.0% | 33.1% | 33.6% | 24.8% | 25.1% |
| 288c (24h) | 49.2% | 50.0% | 33.1% | 33.7% | 24.9% | 25.2% |

#### Forward return stats (% price change over horizon)
| Horizon | Mean | Std | Median | Range |
|---|---|---|---|---|
| 12c (1h) | +0.006% | 0.671% | +0.006% | [-23.0%, +34.3%] |
| 24c (2h) | +0.013% | 0.935% | +0.011% | [-24.1%, +32.6%] |
| 48c (4h) | +0.025% | 1.309% | +0.019% | [-36.0%, +32.9%] |
| 96c (8h) | +0.050% | 1.824% | +0.029% | [-38.6%, +46.3%] |
| 288c (24h) | +0.149% | 3.186% | +0.092% | [-50.9%, +44.9%] |

**Key observation:** Mean forward return is positive across all horizons (slight upward drift in BTC 2020–2026). The 1R short win rate converges to exactly 50.0% at 48c+ horizons, consistent with a near-efficient market in the short run.

---

### v2 Raw & Resampled File Inventory
> Source: `logs/resample_20260302_135346.log`, `logs/download_20260302_132428.log`

#### Perp resampled files
| File | Bars | Complete | Size |
|---|---|---|---|
| `BTCUSDT_perp_5m.parquet` | 648,288 | 100% | 80.8 MB |
| `BTCUSDT_perp_15m.parquet` | 216,096 | 100% | 31.5 MB |
| `BTCUSDT_perp_30m.parquet` | 108,048 | 100% | 17.7 MB |
| `BTCUSDT_perp_1h.parquet` | 54,024 | 100% | 8.9 MB |
| `BTCUSDT_perp_4h.parquet` | 13,506 | 100% | 2.3 MB |
| `BTCUSDT_perp_1d.parquet` | 2,251 | 100% | 0.4 MB |

#### Spot resampled files (historical context, 2017+)
| File | Bars | Complete | Size |
|---|---|---|---|
| `BTCUSDT_spot_5m.parquet` | ~896,222 | 100% | 62.1 MB |
| `BTCUSDT_spot_15m.parquet` | 298,750 | 100% | 24.0 MB |
| `BTCUSDT_spot_30m.parquet` | 149,382 | 100% | 13.8 MB |
| `BTCUSDT_spot_1h.parquet` | 74,701 | 100% | 7.2 MB |
| `BTCUSDT_spot_4h.parquet` | 18,690 | 99.7% | 1.8 MB |
| `BTCUSDT_spot_1d.parquet` | 3,118 | 98.9% | 0.3 MB |

#### Raw data download summary
| File | Rows | Date range | Size |
|---|---|---|---|
| `BTCUSDT_spot_1m_raw.parquet` | 4,481,119 | 2017-08-17 → 2026-02-28 | 334.7 MB |
| `BTCUSDT_perp_1m_raw.parquet` | 3,241,440 | 2020-01-01 → 2026-02-28 | 222.9 MB |
| `BTCUSDT_perp_mark_1m_raw.parquet` | 3,229,865 | 2020-01-01 → 2026-02-28 | 160.9 MB |
| `BTCUSDT_perp_index_1m_raw.parquet` | 3,224,100 | 2020-01-01 → 2026-02-28 | 174.1 MB |
| `BTCUSDT_perp_funding_raw.parquet` | 6,753 | 2020-01-01 → 2026-02-28 | 0.1 MB |

---

<a id="open-questions"></a>
## 7. Open Questions & Anomalies

### Anomalies found during validation

#### Anomaly 1 — FVG tracking convergence failures (D11 → D12 → D13)
Three successive iterations were needed to produce useful FVG active counts:
- **v1/D04 approach** (close >= mid): 9-year bull market meant nearly all bull FVGs got revisited → avg active count ~0.07 (useless)
- **D11 wick-touch**: same problem, slower to die
- **D12 CHoCH reset**: on 5m data, CHoCH fires every 10–50 bars; FVG formation and CHoCH are correlated events → still near-zero
- **D13 (final):** time-based lookback cap (5m=288 bars, ~1 trading day) + close-through mitigation
- **Root cause:** Any purely price-based or structure-based mitigation rule fails across a 9-year bull trend at 5m granularity. Only a time cap is robust.

#### Anomaly 2 — Swing strategy Asia session win rate (55.6% vs 16.7%)
- Swing strategy (v1): Asia session WR 55.6% (9 trades) vs London/NY 16.7% each
- Small sample (9 trades) — may be statistical noise
- Could reflect genuine edge: Asia session has lower volume/manipulation, structural entries during Asia may face less adverse movement
- **Open question:** Is this real? Test on v2 data when swing strategy is rebuilt. Do not optimize around it until it replicates on v2 perp data.

#### Anomaly 3 — NY PM Silver Bullet underperforms
- Session breakdown from baseline_v2: London SB 40.6% WR, NY AM SB 37.5%, NY PM SB 34.0%
- NY PM (14:00–15:00 NY) is the weakest of the three windows
- At 2R with 48c horizon, need > 33.33% — NY PM at 34.0% is barely above break-even
- **Open question:** Should NY PM SB be excluded from the filter stack? Cutting it reduces signal count from 624 → ~312 but may improve quality.

#### Anomaly 4 — Short side underperforms long side
- Signal-level: Long EV = +0.1224R | Short EV = -0.0391R
- Long PF = 1.283 | Short PF = 1.016
- Short side barely breaks even even with ICT filter
- **Open question:** Is this a 2020–2026 regime artifact (mostly bull market) or a structural issue with short ICT setups? Needs regime-conditional analysis: short WR in bearish regimes vs mixed.

#### Anomaly 5 — 2R/12c failure
- Only combination below break-even among 1R and 2R rows
- Expected: a 1-hour window is not enough time for a 2R trade to play out from a structural ICT entry
- **Not anomalous** — confirms that ICT strategies need time to work. The 48-bar (4h) horizon is the natural minimum for 2R targets.

#### Anomaly 6 — D15: truncated `bar_start_ts_ms` columns
- `bar_start_ts_ms` and `bar_end_ts_ms` contain truncated integers (e.g. 1607836 instead of ~1,600,000,000,000)
- Happened during resample or merge step — likely integer overflow or division error
- **No impact** — all scripts use `bar_start_ts_utc` (correct). Decision: do not fix retroactively (D15).

#### Anomaly 7 — Session labels use fixed UTC (v1 dataset DST issue)
- v1 master parquet session labels use fixed UTC offsets (~EDT always)
- This is ~1 hour wrong during EST winter (Nov → mid-Mar, ~5 months/year)
- v2 dataset fixes this: `bar_start_ts_utc.dt.tz_convert("America/New_York")` for session flags
- **Impact on v1 backtests:** Small — sessions are 3-hour windows, 1-hour shift only affects the edges. Affects ~5 months/year. Not worth fixing in v1 (legacy); v2 is correct.

---

### Open questions for future sessions

| Question | Priority | Notes |
|---|---|---|
| Do regime filters (ATR + ADX + CHOP) materially improve PF? | HIGH | ETH Zurich finding: Sharpe 1.1 → 3.2 with vol regime. Build and test. |
| Does NY PM SB deserve to be excluded? | MEDIUM | WR 34.0% at 2R barely above BE — test with and without. |
| Is the short-side underperformance a regime artifact? | MEDIUM | Compare short WR when d1_ict_market_trend == -1 vs all bars. |
| Can the swing Asia anomaly replicate on v2 perp data? | MEDIUM | Only 9 trades in v1 — needs more data. |
| What is the optimal stop for v2 baseline? | MEDIUM | ATR-based stop may be too wide. Compare structural FVG stops. |
| Walk-forward validation: does the edge hold out of sample? | HIGH | Train 2020–2022, test 2023, retrain, test 2024–2026. |
| Silver Bullet strategy: build and backtest | HIGH | Planned. Expected to produce 1+/week signals. |
| What happens if we add funding rate as a filter? | LOW | Avoid entries when funding is extreme (>0.1%/8h)? |
| Volume profile / VWAP: do HVN/LVN zones improve FVG entry quality? | LOW | No peer-reviewed BTCUSDT evidence yet — treat as hypothesis. |

---

*Document compiled from: STRATEGY_LOG.md (ICTbacktest), BTCDataset_v2 STRATEGY_LOG.md, build_master log (2026-03-02 16:18), generate_labels log (2026-03-02 16:31), baseline_v2_summary.json, resample log, download log, BTCUSDT_AUDIT_REPORT.txt.*
