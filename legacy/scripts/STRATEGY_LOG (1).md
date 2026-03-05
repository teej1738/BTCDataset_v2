# BTCDataset v2 - Strategy and Design Log

This file records design decisions made during dataset construction.
It exists so Claude Code and future contributors understand WHY choices
were made, not just what they are. Never delete entries - append only.

---

## Decision Log

### 2026-03-02 - Architecture decisions locked before enrich_ict_v4.py

These decisions were reviewed and confirmed by both Claude (claude.ai)
and Claude Code before the enrichment script was written. Do not change
these without updating this log.

---

**D01 - Primary instrument: Perp only for ICT enrichment**

Decision: Enrich perp as primary instrument. Spot 2017-2019 used only
for D1/W1 higher timeframe trend detection in build_master_v4.py.

Reasoning: You would actually be trading perp, not spot. Perp has
funding costs, a mark price, and different microstructure. Spot data
from 2017-2019 has no funding, no OI, and different price behavior
during high-volatility events (e.g. 2021 bull run liquidations moved
perp much harder than spot). Using spot for execution labels would
produce unrealistic backtest results.

Implementation:
- enrich_ict_v4.py processes BTCUSDT_perp_{interval}.parquet files
- Spot files used only for D1/W1 columns tagged htf_d1_spot_trend
- Any backtest or ML training must filter bar_start_ts_utc >= 2020-01-01

---

**D02 - All timeframes derived from 1m, never downloaded separately**

Decision: Download 1m only from Binance archive, resample up to all
higher timeframes in resample_v2.py.

Reasoning: Independently downloaded timeframes can disagree at gap
periods. A gap in the 5m file might not appear in the 4h file because
each API call returns slightly different data. Building everything from
1m guarantees mathematical consistency - the same gap appears at all
timeframes because they all come from the same source rows.

Implementation: resample_v2.py -- COMPLETE

---

**D03 - Silver Bullet: Three separate boolean flags**

Decision: Three separate flags, never combined.
  sess_sb_london  -- 03:00-04:00 AM NY time
  sess_sb_ny_am   -- 10:00-11:00 AM NY time
  sess_sb_ny_pm   -- 14:00-15:00 PM NY time

Reasoning: These are three different setups with different confluence
requirements. London SB trades during London session momentum. NY AM SB
trades the NY open reversal. NY PM SB trades the afternoon session.
Combining them into one flag destroys this information. You can always
OR them later; you cannot split them after the fact.

Implementation: DST-aware via tz_convert("America/New_York") on
bar_start_ts_utc. No offset tables needed.

---

**D04 - FVG mitigation direction**

Decision:
  Bull FVG (gap sits above price): mitigated when close >= fvg_top
    (price rallied UP through the gap from below)
  Bear FVG (gap sits below price): mitigated when close <= fvg_bot
    (price dropped DOWN through the gap from above)

Reasoning: This is the most common FVG implementation bug. Bull FVGs
are not invalidated by price dropping - they are filled by price
rallying back up into them. Checking lows against a bull FVG is wrong.

History: This bug existed in enrich_ict_v2.py and was fixed in v3.
Port the v3 fix exactly. Do not rewrite this logic from scratch.

---

**D05 - Multi-FVG tracking: nearest, oldest, recent + count**

Decision: Per direction (bull/bear), track four things simultaneously:
  ict_fvg_{dir}_near_{top/bot/mid}   -- nearest to current price
  ict_fvg_{dir}_recent_{top/bot/mid} -- most recently formed
  ict_fvg_{dir}_oldest_{top/bot/mid} -- longest standing (strongest)
  ict_fvg_{dir}_count                -- total active unmitigated count

Reasoning: "Most recent FVG" and "nearest FVG" are different things
and give different signals. The oldest unmitigated FVG acts as the
strongest magnet because more liquidity has accumulated around it.
Tracking only the most recent (as v1 did) loses this information.

---

**D06 - Order block deduplication: one OB per BOS**

Decision: Each BOS event produces exactly one order block. If multiple
candles precede a BOS, only the last opposing candle is the OB.
Deduplicate by tracking which BOS events have already been assigned an OB.

Reasoning: Without deduplication, a fast-moving market stacks 5+ OBs
from a single structural move. These are all redundant and represent the
same liquidity event. Stacking inflates OB counts and produces false
confluence signals.

History: This bug existed in enrich_ict_v2.py and was fixed in v3 using
assigned_bull_bos / assigned_bear_bos tracking sets. Port the v3 fix.

---

**D07 - CVD: reset variants only, never all-time cumsum**

Decision: Compute four CVD variants:
  cvd_daily       -- resets at 00:00 UTC each day
  cvd_session     -- resets at each London/NY session open
  cvd_7d_rolling  -- rolling 7-day window
  cvd_zscore      -- daily CVD normalised by 20-period rolling std

Reasoning: All-time cumsum is an ever-growing number from 2020 to 2026
with no predictive meaning per bar. A cvd value of +50,000 in 2021 vs
+50,000 in 2024 means completely different things. Reset variants
provide meaningful relative context.

---

**D08 - Label symmetry: explicit long and short labels**

Decision: Generate both long and short label variants separately.
  label_long_hit_1r_{n}c   -- long trade hit 1R before stop in N candles
  label_long_hit_2r_{n}c   -- long trade hit 2R before stop
  label_long_hit_3r_{n}c
  label_short_hit_1r_{n}c  -- short trade equivalent
  label_short_hit_2r_{n}c
  label_short_hit_3r_{n}c

Reasoning: ICT trades both directions. A long-only label set biases any
ML model toward the long side. Inferring short labels from the sign of
long labels is wrong because stops and targets are placed differently
for longs vs shorts (below/above OB or FVG respectively).

---

**D09 - Funding cost formula: mark price at event**

Decision: Use mark price at each funding event, not a fixed notional.
  funding_cost = fund_rate * mark_price_at_event * qty

Reasoning: sum(fund_rate) * fixed_notional is wrong when mark price
moves significantly during a multi-day hold. For a position held across
8+ funding events during a volatile period, the error compounds.
Since BTCUSDT_perp_mark_1m_raw.parquet has full mark price history,
we can compute exact cost by joining funding events to nearest mark bar.

---

**D10 - Backtest date restriction**

Decision: All strategy backtests and ML training must filter:
  bar_start_ts_utc >= 2020-01-01

Reasoning: Perp data starts January 2020 (Sep-Dec 2019 archive files
do not exist on Binance). Rows before 2020 are spot-only and have null
funding, mark price, index, basis, and CVD columns. Training a model
on rows with null perp features would produce a model that implicitly
learns to ignore those features, degrading performance on perp trades.

Exception: D1/W1 trend columns derived from spot 2017-2019 are valid
features for rows after 2020 where they provide long-term context.
These are tagged htf_d1_spot_trend to distinguish them.

---

## Column Naming Convention

Prefix     Meaning                          Example
------     -------                          -------
(none)     Raw OHLCV from resample          open, high, close, volume_base
meta_      Dataset metadata                 meta_is_complete, meta_interval
ict_       ICT-derived features             ict_fvg_bull_near_top, ict_ob_bear_bot
sess_      Session and killzone features    sess_killzone_ny, sess_sb_london
cvd_       CVD variants                     cvd_daily, cvd_session, cvd_zscore
liq_       Liquidity levels                 liq_pdh, liq_pwl, liq_eq_high
htf_       Higher timeframe context         htf_h4_trend, htf_d1_spot_trend
label_     Forward-looking ML labels        label_long_hit_2r_48c
fund_      Funding rate features            fund_rate_period, fund_rate_cum_24h

---

## File naming convention

data/raw/         BTCUSDT_{instrument}_{type}_raw.parquet
data/resampled/   BTCUSDT_{instrument}_{interval}.parquet
data/enriched/    BTCUSDT_{instrument}_{interval}_enriched.parquet
data/master/      BTCUSDT_MASTER.parquet
data/master/      BTCUSDT_MASTER_metadata.json
data/labeled/     BTCUSDT_MASTER_labeled.parquet

---

### 2026-03-02 - FVG mitigation threshold updated after first enrichment run

**D11 - FVG mitigation: wick-touch rule (replaces D04 mid-point rule)**

Initial run with `close >= fvg_mid` (50% fill) produced avg active FVGs ~0.07
across all timeframes. In a 9-year bull market, price spends almost all time above
prior bull FVG mids so they mitigated almost immediately. FVG tracking was useless.

Decision: change to wick-touch rule:
  - Bull FVG mitigated when `high >= fvg_top`  (wick touches top of gap)
  - Bear FVG mitigated when `low  <= fvg_bot`  (wick touches bottom of gap)

Rationale: wick touch = price has entered the imbalance zone, which is when ICT
practitioners consider the FVG "tapped". More FVGs remain active as structural
context. Stricter than mid-point, looser than close-through.

This supersedes D04. D04's directional logic is preserved — only threshold changes.

---

### 2026-03-02 - FVG tracking rebuilt (D12)

**D12 - FVG: structure-scoped + close-through + in_zone flags**

Three changes implemented together:

1. STRUCTURE-SCOPED RESET
   FVG list resets at each CHoCH (ict_choch_close column).
   - Bullish CHoCH (+1): clear active bear FVGs (new uptrend, bear gaps irrelevant)
   - Bearish CHoCH (-1): clear active bull FVGs (new downtrend, bull gaps irrelevant)
   Root cause of 0.07 avg: 9-year bull market meant all old FVGs were visited.
   Scoping to current leg means only recent, contextually relevant FVGs survive.

2. CLOSE-THROUGH MITIGATION
   Bull FVG: close >= fvg_top (price closes fully above gap)
   Bear FVG: close <= fvg_bot (price closes fully below gap)
   A wick into the zone is a "tap" (entry signal), not a fill.
   Price must accept (close) beyond the zone to truly mitigate.
   Supersedes D11 (wick-touch) and D04 (mid-point).

3. IN-ZONE FLAGS (new columns)
   ict_fvg_bull_in_zone: 1 when price bar overlaps any active bull FVG
   ict_fvg_bear_in_zone: 1 when price bar overlaps any active bear FVG
   Condition: low <= fvg_top AND high >= fvg_bot
   This is the highest-signal moment: the retrace INTO the imbalance.
   Expected to fire ~1-3% of bars -- rare but high-quality signal.

New columns added (2): ict_fvg_bull_in_zone, ict_fvg_bear_in_zone
Total columns after this change: spot 144, perp 155.

---

### 2026-03-02 - FVG lookback cap replaces CHoCH reset (D13)

**D13 - FVG: time-based lookback cap (supersedes D12 structure reset)**

Root cause analysis of persistent 0.07 avg active FVGs across all attempts:

  Attempt 1 (D04): close >= mid mitigation
    -> 9yr bull market means all old bull FVGs get tapped during retracements
    -> Result: near-zero count

  Attempt 2 (D11): wick-touch mitigation
    -> Same problem, just slightly slower to die
    -> Result: near-zero count

  Attempt 3 (D12): CHoCH reset + close-through mitigation
    -> On 5m data, CHoCH fires every 10-50 bars (structure breaks constantly)
    -> FVGs form during displacement -> CHoCH fires shortly after -> FVGs wiped
    -> FVG formation and CHoCH are correlated events on fine timeframes
    -> Result: still near-zero count

FINAL DIAGNOSIS: Both mitigation-based and structure-based approaches fail on
a 9yr dataset at fine granularity. The only reliable solution is a time-based
lookback cap that gives FVGs a fixed shelf life independent of price/structure.

Solution: drop any FVG older than N bars (timeframe-adaptive, ~1 trading day):
  5m:  288 bars  |  15m: 96  |  30m: 48  |  1h: 48  |  4h: 30  |  1d: 20

Close-through mitigation (D12) retained: FVG still dies if price closes through it.
In-zone flags (D12) retained: ict_fvg_bull_in_zone / ict_fvg_bear_in_zone.
CHoCH reset from D12 removed entirely.

Expected avg active FVGs after this change: 3-8 on 5m, scaling down with timeframe.
