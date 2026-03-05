# D53 Implementation Spec
# ICT Rules Overhaul -- core/signals/ict/rules.py
# Source: Two independent GPT research responses, reconciled
# Last updated: 2026-03-03
# Priority order: F -> A -> G -> H -> D -> B -> C -> E -> MSS

---

## RECONCILIATION NOTES (two GPT responses compared)

Where the two GPT responses disagreed, the resolved value is shown below.
All other parameters had full agreement between both responses.

| Parameter | GPT-1 | GPT-2 | RESOLVED |
|-----------|-------|-------|----------|
| Displacement k (body/ATR) | 1.0 | 1.5 | 1.5 (make searchable) |
| OB zone mode | body-only | hybrid | hybrid |
| OB mitigation trigger | wick-touch | 50% penetration | 50% penetration (MIT_P=0.5) |
| OB age cap | 200 bars | 864 bars | 200 bars (one session cycle) |
| OB anchor search depth | 10 bars | 20 bars | 20 bars |
| Sweep close-back m | 3 bars | 2 bars | 2 bars |
| Sweep sequence N/k | N=10, k=5 | N=12, k=6 | N=12, k=6 |
| Internal pivot_n | 5 | 3 (5 optional) | 5 |
| MSS k window | 5 bars | 3 bars | 3 bars |
| CISD min consecutive run | 1 | 2 | 2 (min_run=2) |
| IFVG age cap | 100 bars | 144 bars | 144 bars |
| Priority #1 | G (P/D) | F (displacement) | F (displacement is prerequisite for A, C, H) |

---

## READING ORDER FOR CC

Read this file top-to-bottom before writing any code.
Each task has: WHAT changes, EXACT algorithm (pseudocode or formula), and
WHY (AUC impact and SHAP rationale).

After completing ALL tasks:
  1. Run causality tests at T in [1000, 5000, 10000, 50000] for every modified/new function
  2. Run smoke test on 10,000 bars, print coverage stats for all new features
  3. Register all new features in simulator.py ONTHEFLY_FEATURES
  4. Add all new features to data/labeled/feature_catalog_v3.yaml
  5. Log as D53 in STRATEGY_LOG.md, update CLAUDE.md and THE_PLAN.md

Environment: Python 3.14, Windows, cp1252 encoding.
ASCII only in all print statements. No Unicode symbols.
No numba. Vectorized NumPy where possible; stateful loop for OB/FVG/sweep tracking.
All functions: pure (no side effects, no global state).
All functions: strictly causal -- at bar T, uses only df.iloc[:T+1].

---

## TASK F -- Displacement Detection (DO THIS FIRST -- prerequisite for A, C, H, MSS)
**Expected AUC: > 0.01 -- displacement underpins OB anchoring, FVG quality tagging, MSS.**
**Without this function, Tasks A/C/H cannot be correctly implemented.**

### New helper: _is_displacement_candle()

Used by every other function. Add as private helper at top of rules.py.

```python
DISP_K = 1.5                # body >= 1.5x ATR (searchable: range 1.0-2.5)
DISP_CLOSE_FRAC = 0.75      # close in top/bottom 25% of range

def _is_displacement_candle(o, h, l, c, atr, direction):
    """
    Returns (occurred: bool, strength: float, close_loc: float).
    strength = body / ATR (continuous, used as quality score).
    close_loc = (c-l)/(h-l), 0..1.
    direction: "bull" or "bear".
    """
    rng = max(h - l, 1e-9)
    body = abs(c - o)
    strength = body / max(atr, 1e-9)
    close_loc = (c - l) / rng

    if direction == "bull":
        occurred = (c > o) and (strength >= DISP_K) and (close_loc >= DISP_CLOSE_FRAC)
    else:
        occurred = (c < o) and (strength >= DISP_K) and (close_loc <= (1.0 - DISP_CLOSE_FRAC))

    return occurred, strength, close_loc
```

### New function: detect_displacement()

```python
def detect_displacement(df, disp_k=1.5, disp_close_frac=0.75,
                        age_cap=48, multi_k_total=2.0):
    """
    Detect single and multi-candle displacement events at every bar.

    Single-candle: body >= disp_k * ATR, close in top/bottom disp_close_frac of range.
    Multi-candle: 3 consecutive same-direction candles,
                  combined move (close[t]-open[t-2]) >= multi_k_total * ATR.

    age_cap: bars until displacement_age goes NaN.
    """
```

#### Output columns (per direction bull/bear):

```
displacement_bull          float32  1.0 at the exact bar displacement occurs
displacement_bull_age      float32  bars since last bull displacement (NaN if > age_cap)
displacement_bull_strength float32  body / ATR of the displacement candle
displacement_bull_close_loc float32 close_loc (0..1); how extreme the close was
displacement_bull_range_atr float32 (high-low) / ATR of the displacement candle
displacement_bull_has_fvg  float32  1.0 if displacement created a same-direction FVG
displacement_bull_is_multi float32  1.0 if multi-candle, 0.0 if single-candle

displacement_bear          float32  (mirror)
displacement_bear_age      float32
displacement_bear_strength float32
displacement_bear_close_loc float32
displacement_bear_range_atr float32
displacement_bear_has_fvg  float32
displacement_bear_is_multi float32
```

#### Multi-candle implementation:

```python
def _is_multi_candle_displacement(df, t, direction, k_total=2.0):
    if t < 2:
        return 0
    atr = df['atr_14'].iloc[t]
    if direction == "bull":
        all_same = (df['close'].iloc[t]   > df['open'].iloc[t]   and
                    df['close'].iloc[t-1] > df['open'].iloc[t-1] and
                    df['close'].iloc[t-2] > df['open'].iloc[t-2])
        move = abs(df['close'].iloc[t] - df['open'].iloc[t-2]) / max(atr, 1e-9)
    else:
        all_same = (df['close'].iloc[t]   < df['open'].iloc[t]   and
                    df['close'].iloc[t-1] < df['open'].iloc[t-1] and
                    df['close'].iloc[t-2] < df['open'].iloc[t-2])
        move = abs(df['close'].iloc[t] - df['open'].iloc[t-2]) / max(atr, 1e-9)
    return 1 if (all_same and move >= k_total) else 0
```

---

## TASK A -- Order Block Detection: Full Algorithm Overhaul
**Expected AUC: > 0.01 -- ob_bull_age and ob_bear_age are #3/#4 SHAP.**
**Requires Task F (displacement) to be completed first.**

### Parameters

```python
OB_DISP_SEARCH   = 30    # bars to search backward from BOS for displacement candle
OB_ANCHOR_SEARCH = 20    # bars to search backward from displacement for OB candle
OB_AGE_CAP       = 200   # bars; auto-expire OBs older than this
MIT_P            = 0.5   # 50% penetration triggers mitigation
ZONE_MODE        = "hybrid"  # "wick" | "body" | "hybrid"
```

### Zone definition (hybrid -- RESOLVED choice):

```python
def _ob_zone(o, h, l, c, direction, zone_mode="hybrid"):
    """
    hybrid (default for BTC 5m):
      Bull OB (bearish candle): zone = [low, max(open, close)]
      Bear OB (bullish candle): zone = [min(open, close), high]
    body: zone = [min(o,c), max(o,c)]
    wick: zone = [l, h]
    """
    if zone_mode == "wick":
        return l, h
    if zone_mode == "body":
        return min(o, c), max(o, c)
    # hybrid
    if direction == "bull":   # OB candle is bearish
        return l, max(o, c)
    else:                     # OB candle is bullish
        return min(o, c), h
```

### OB creation (at BOS event):

```python
def _create_ob_on_bos(df, t, direction):
    """
    Called when BOS_close fires at bar t.
    Returns OB dict or None if no valid displacement found.
    """
    # Step 1: find most recent displacement candle within OB_DISP_SEARCH bars ending at t
    disp_idx, disp_strength = None, None
    for j in range(t, max(-1, t - OB_DISP_SEARCH), -1):
        ok, strength, _ = _is_displacement_candle(
            df['open'].iloc[j], df['high'].iloc[j],
            df['low'].iloc[j],  df['close'].iloc[j],
            df['atr_14'].iloc[j], direction
        )
        if ok:
            disp_idx, disp_strength = j, strength
            break

    if disp_idx is None:
        return None  # BOS not powered by displacement -> skip OB

    # Step 2: find last opposite-colored candle before displacement
    ob_idx = None
    for j in range(disp_idx - 1, max(-1, disp_idx - OB_ANCHOR_SEARCH), -1):
        if direction == "bull" and df['close'].iloc[j] < df['open'].iloc[j]:
            ob_idx = j
            break
        if direction == "bear" and df['close'].iloc[j] > df['open'].iloc[j]:
            ob_idx = j
            break

    if ob_idx is None:
        return None

    bot, top = _ob_zone(
        df['open'].iloc[ob_idx], df['high'].iloc[ob_idx],
        df['low'].iloc[ob_idx],  df['close'].iloc[ob_idx],
        direction, ZONE_MODE
    )

    return {
        "ob_idx":        ob_idx,
        "disp_idx":      disp_idx,
        "disp_strength": disp_strength,
        "bot":           bot,
        "top":           top,
        "mid":           (bot + top) / 2.0,
        "width":         top - bot,
        "state":         1,   # 1=fresh
        "age":           0,
        "direction":     direction,
        "bos_bar":       t,
        "bars_to_bos":   t - disp_idx,
        "has_fvg":       0,   # set externally after FVG detection
    }
```

### Three-state tracking (called each bar for each active OB):

```python
def _update_ob_state(ob, h, l, c):
    """
    Returns True if OB remains active, False if it should be removed.
    fresh (1) -> mitigated (2): 50% penetration from entry side.
    mitigated (2) -> invalid: close through far edge (remove from list).
    Expired: age > OB_AGE_CAP (remove from list).
    """
    bot, top = ob['bot'], ob['top']
    width = max(ob['width'], 1e-9)

    if ob['state'] == 1:  # fresh -> check mitigation
        if ob['direction'] == "bull":
            penetration = (top - min(l, top)) / width
        else:
            penetration = (max(h, bot) - bot) / width
        penetration = max(0.0, min(1.0, penetration))
        if penetration >= MIT_P:
            ob['state'] = 2  # mitigated

    if ob['state'] == 2:  # mitigated -> check invalidation
        if ob['direction'] == "bull" and c < bot:
            return False  # invalid, remove
        if ob['direction'] == "bear" and c > top:
            return False  # invalid, remove

    ob['age'] += 1
    if ob['age'] > OB_AGE_CAP:
        return False  # expired

    return True  # still active
```

### Output columns (top-3 per direction):

For each of ob_bull_1, ob_bull_2, ob_bull_3 (and mirror ob_bear_1/2/3):

```
ob_bull_1_state        float32  0=none, 1=fresh, 2=mitigated (ordinal numeric)
ob_bull_1_age          float32  min(bars, OB_AGE_CAP) / OB_AGE_CAP
ob_bull_1_top          float32  zone top price
ob_bull_1_bot          float32  zone bot price
ob_bull_1_mid          float32  zone midpoint
ob_bull_1_width_atr    float32  (top - bot) / atr_14
ob_bull_1_in_zone      float32  1.0 if high >= bot AND low <= top
ob_bull_1_penetration  float32  current penetration fraction (0..1)
ob_bull_1_dist_top_atr float32  (top - close) / atr_14
ob_bull_1_dist_bot_atr float32  (close - bot) / atr_14
ob_bull_1_strength     float32  displacement body / ATR at formation
ob_bull_1_bars_to_bos  float32  displacement to BOS distance (bars)
ob_bull_1_has_fvg      float32  1.0 if displacement created FVG
```

Aggregates:
```
count_active_ob_bull   float32
count_active_ob_bear   float32
min_dist_ob_bull_atr   float32
min_dist_ob_bear_atr   float32
```

### Quality score for top-3 ranking:

```python
import math

def _ob_quality(disp_strength, bars_to_bos, width_atr, has_fvg, state, age):
    def clip01(x): return max(0.0, min(1.0, x))
    disp_term  = clip01((disp_strength - 1.0) / 2.0)
    bos_term   = clip01(1.0 - (bars_to_bos / 12.0))
    width_term = clip01(width_atr / 1.0)
    fvg_boost  = 1.0 + 0.25 * float(has_fvg)
    fresh_mult = 1.0 if state == 1 else 0.85
    age_mult   = math.exp(-age / 200.0)
    score = (0.45*disp_term + 0.35*bos_term + 0.20*width_term)
    return score * fvg_boost * fresh_mult * age_mult
# Rank: quality desc, then dist_atr asc as tiebreaker.
```

---

## TASK G -- Premium/Discount: Replace +1/0/-1 with continuous 0-1
**Expected AUC: 0.001-0.005 -- ote_dist is #1 SHAP. Requires Task D (dual swings) first.**

```python
def compute_premium_discount(df, swing_lookback=96):
    """
    Uses EXTERNAL swing layer prices (ext_swing_high_price, ext_swing_low_price).
    Requires compute_swing_dual_layer() to have run first.
    """
    swing_high = df['ext_swing_high_price']
    swing_low  = df['ext_swing_low_price']
    rng = (swing_high - swing_low).replace(0, np.nan)

    pd_pos = ((df['close'] - swing_low) / rng).clip(0.0, 1.0).astype('float32')

    results = pd.DataFrame(index=df.index)
    results['pd_position_5m']   = pd_pos
    results['pd_dist_from_eq']  = (pd_pos - 0.5).abs().astype('float32')
    results['in_discount']      = (pd_pos < 0.5).astype('float32')
    results['in_deep_discount'] = (pd_pos <= 0.25).astype('float32')
    results['in_deep_premium']  = (pd_pos >= 0.75).astype('float32')
    # Bull OTE: discount-side retracement 61.8-78.6% => pd_pos in [0.214, 0.382]
    results['in_ote_bull']      = ((pd_pos >= 0.214) & (pd_pos <= 0.382)).astype('float32')
    # Bear OTE: premium-side => pd_pos in [0.618, 0.786]
    results['in_ote_bear']      = ((pd_pos >= 0.618) & (pd_pos <= 0.786)).astype('float32')
    return results
```

---

## TASK H -- OTE 0.705 Explicit Level
**Expected AUC: ~0.001. Trivial add to compute_ote_dist().**

```python
# Add to compute_ote_dist() output:
results['ote_dist_from_705_atr'] = (pd_position - 0.705).abs() / atr_14
results['ote_at_705'] = ((pd_position - 0.705).abs() < 0.5).astype('float32')
# Binary: price within 0.5 ATR of the 0.705 ICT sweet spot level
```

---

## TASK D -- Dual-Layer Swings: Internal (N=5) + External (N=10)
**Expected AUC: ~0.001-0.005 -- swing prices are #2/#5 SHAP.**
**Required by Tasks G and B.**

### New function: compute_swing_dual_layer()

Keep compute_swing_points() for backward compat. Add new function alongside it.

```python
def compute_swing_dual_layer(df, pivot_n_internal=5, pivot_n_external=10):
    """
    Internal (N=5): entry timing. Swing at j confirmed at j+5 (25 min lag).
    External (N=10): structure, dealing range. Confirmed at j+10 (50 min lag).
    BOS and CHoCH computed independently per layer.
    """
```

#### Output columns per layer (prefix int_ and ext_):

```
int_swing_high           float32  1.0 at bar when internal swing high confirmed
int_swing_low            float32  1.0 at bar when internal swing low confirmed
int_swing_high_price     float32  price of most recent confirmed int swing high (ffill)
int_swing_low_price      float32  price of most recent confirmed int swing low (ffill)
int_dist_to_sh_atr       float32  (int_swing_high_price - close) / atr_14
int_dist_to_sl_atr       float32  (close - int_swing_low_price) / atr_14
int_trend                float32  +1 bullish, -1 bearish
int_bos_bull             float32  1.0 if close > most recent int swing high
int_bos_bear             float32  1.0 if close < most recent int swing low
int_choch_bull           float32  1.0 if bullish CHoCH on internal layer
int_choch_bear           float32  1.0 if bearish CHoCH on internal layer
```

Same 11 columns with ext_ prefix for external layer.

#### Causality pattern:

```python
# Swing at candidate bar j confirmed at bar T when j + pivot_n <= T.
# candidate = T - pivot_n
# Window: df['high'].iloc[candidate - pivot_n : candidate + pivot_n + 1]
# Forward-fill prices from confirmation bar onward.
```

#### CRITICAL verification:

```python
# After implementation, print:
# corr(int_swing_high at lag 0 vs target)  -- should be near 0
# corr(int_swing_high at lag -5 vs target) -- should be higher
# If lag-0 corr is non-trivial, there is a lookahead bug.
```

---

## TASK B -- Sweep Detection
**Expected AUC: 0.005-0.01 -- liq_dist_above_pct is #6 SHAP with no sweep events.**
**Requires Task D (dual swings) for confirmed swing highs/lows.**

### Parameters

```python
EQ_TOLERANCE_ATR = 0.10
LOOKBACK_BARS    = 50
SWEEP_M          = 2      # close-back window (bars) -- RESOLVED: 2 not 3
VALIDITY_BARS    = 6
MIN_TOUCHES      = 2
```

### Full implementation:

```python
def detect_sweep(df, swing_high_bool, swing_high_price, swing_low_bool, swing_low_price,
                 eq_tolerance_atr=0.10, lookback_bars=50, m=2, validity_bars=6):
    """
    BSL sweep (bearish): wick above EQH, close back below within m bars.
    SSL sweep (bullish): wick below EQL, close back above within m bars.
    Uses confirmed swing highs/lows from compute_swing_dual_layer (internal layer).
    """
    n = len(df)
    bsl_fired = [0]*n; ssl_fired = [0]*n
    bsl_age   = [float('nan')]*n; ssl_age = [float('nan')]*n
    bsl_pen   = [0.0]*n; ssl_pen = [0.0]*n
    dist_unswept_bsl = [float('nan')]*n
    dist_unswept_ssl = [float('nan')]*n

    pending_bsl = None; pending_ssl = None
    last_bsl = None;    last_ssl = None

    def cluster_levels(prices, tol):
        if not prices: return []
        prices = sorted(prices)
        clusters = []; cur = [prices[0]]
        for p in prices[1:]:
            if abs(p - cur[-1]) <= tol:
                cur.append(p)
            else:
                clusters.append(cur); cur = [p]
        clusters.append(cur)
        return [(sum(c)/len(c), len(c)) for c in clusters if len(c) >= 2]

    for t in range(n):
        atr = df['atr_14'].iloc[t]
        tol = eq_tolerance_atr * atr
        close = df['close'].iloc[t]

        hi = [swing_high_price[i] for i in range(max(0,t-lookback_bars), t) if swing_high_bool[i]]
        lo = [swing_low_price[i]  for i in range(max(0,t-lookback_bars), t) if swing_low_bool[i]]
        eqh = cluster_levels(hi, tol)
        eql = cluster_levels(lo, tol)

        # Nearest unswept levels
        bsl_lvls = [lvl for (lvl,_) in eqh if lvl > close]
        ssl_lvls = [lvl for (lvl,_) in eql if lvl < close]
        if bsl_lvls: dist_unswept_bsl[t] = (min(bsl_lvls) - close) / max(atr, 1e-9)
        if ssl_lvls: dist_unswept_ssl[t] = (close - max(ssl_lvls)) / max(atr, 1e-9)

        # Confirm pending BSL
        if pending_bsl is not None:
            lvl, breach_bar, pen = pending_bsl
            if t - breach_bar > m:
                pending_bsl = None
            elif df['close'].iloc[t] < lvl:
                bsl_fired[t] = 1; bsl_pen[t] = pen
                last_bsl = t; pending_bsl = None

        # Confirm pending SSL
        if pending_ssl is not None:
            lvl, breach_bar, pen = pending_ssl
            if t - breach_bar > m:
                pending_ssl = None
            elif df['close'].iloc[t] > lvl:
                ssl_fired[t] = 1; ssl_pen[t] = pen
                last_ssl = t; pending_ssl = None

        # New BSL breach
        nearest_eqh = min([lvl for (lvl,_) in eqh if lvl > close], default=None)
        if nearest_eqh is not None and pending_bsl is None and df['high'].iloc[t] > nearest_eqh:
            pending_bsl = (nearest_eqh, t, (df['high'].iloc[t]-nearest_eqh)/max(atr,1e-9))

        # New SSL breach
        nearest_eql = max([lvl for (lvl,_) in eql if lvl < close], default=None)
        if nearest_eql is not None and pending_ssl is None and df['low'].iloc[t] < nearest_eql:
            pending_ssl = (nearest_eql, t, (nearest_eql-df['low'].iloc[t])/max(atr,1e-9))

        # Freshness ages
        if last_bsl is not None:
            age = t - last_bsl
            if 0 <= age <= validity_bars: bsl_age[t] = age
        if last_ssl is not None:
            age = t - last_ssl
            if 0 <= age <= validity_bars: ssl_age[t] = age

    return dict(sweep_bsl_fired=bsl_fired, sweep_bsl_age=bsl_age, sweep_bsl_pen_atr=bsl_pen,
                sweep_ssl_fired=ssl_fired, sweep_ssl_age=ssl_age, sweep_ssl_pen_atr=ssl_pen,
                dist_unswept_bsl_atr=dist_unswept_bsl, dist_unswept_ssl_atr=dist_unswept_ssl)
```

### Sweep sequence composite:

```python
def detect_sweep_sequence(df, ssl_fired, bsl_fired, disp_bull, disp_bear,
                          fvg_bull_formed, fvg_bear_formed, k=6, N=12):
    """
    Bull: SSL sweep -> bull disp within k bars -> bull FVG within k bars.
          Entire sequence within N bars. Fires at FVG bar.
    Bear: BSL sweep -> bear disp -> bear FVG. Same windows.
    """
```

Output:
```
sweep_seq_bull_complete  float32
sweep_seq_bull_age       float32  (0..N else NaN)
sweep_seq_bear_complete  float32
sweep_seq_bear_age       float32
```

---

## TASK C -- FVG Enhancements
**Expected AUC: ~0.001-0.005 -- fvg_bear_recent_age is #9/#10 SHAP.**

### Parameter corrections first:

```python
detect_fvg_bull(df, age_cap=100, min_size_atr=0.50)  # was: 288, 0.0
detect_fvg_bear(df, age_cap=100, min_size_atr=0.50)
```

### CE formula (confirmed by both GPTs):

CE = (fvg_top + fvg_bot) / 2.0

CE touched (bull FVG): low[t] <= CE
Fully filled (bull FVG): low[t] <= fvg_bot

```python
def _fvg_fill_fraction(direction, bot, top, high_t, low_t):
    width = max(top - bot, 1e-9)
    if direction == "bull":
        return max(0.0, min(1.0, (top - min(low_t, top)) / width))
    else:
        return max(0.0, min(1.0, (max(high_t, bot) - bot) / width))
```

### New columns per active FVG (top-3 per direction):

```
fvg_bull_1_ce              float32
fvg_bull_1_dist_to_ce_atr  float32  abs(close - CE) / atr_14
fvg_bull_1_ce_touched      float32
fvg_bull_1_ce_rejected     float32  1.0 if CE touched then closed back out bull direction
fvg_bull_1_fill_fraction   float32
fvg_bull_1_fully_filled    float32
fvg_bull_1_is_displacement float32  1.0 if middle candle qualifies as displacement
fvg_bull_1_is_ifvg         float32  0.0 normal, 1.0 inverted
```

Mirror for ranks 2, 3, and bear direction.

### Displacement FVG tag:

```python
def _is_displacement_fvg(df, i, direction):
    """Middle candle i-1 qualifies as displacement candle."""
    if direction == "bull":
        if i < 2 or not (df['high'].iloc[i-2] < df['low'].iloc[i]):
            return False
    else:
        if i < 2 or not (df['low'].iloc[i-2] > df['high'].iloc[i]):
            return False
    ok, _, _ = _is_displacement_candle(
        df['open'].iloc[i-1], df['high'].iloc[i-1],
        df['low'].iloc[i-1],  df['close'].iloc[i-1],
        df['atr_14'].iloc[i-1], direction
    )
    return ok
```

### IFVG:

```python
IFVG_AGE_CAP = 144  # bars (~12h) -- RESOLVED: 144 not 100

# Bull FVG -> bear IFVG: close[t] < fvg_bot  (close-through, not wick)
# Bear FVG -> bull IFVG: close[t] > fvg_top
# IFVG invalid: close crosses opposite far edge again
```

### FVG ranking score:

```python
import math

def _fvg_rank_score(dist_atr, age, width_atr, is_disp_fvg):
    dist_term  = 1.0 / (1.0 + dist_atr)
    age_term   = math.exp(-age / 288.0)
    size_term  = max(0.0, min(1.0, width_atr / 1.0))
    disp_boost = 1.0 + 0.35 * float(is_disp_fvg)
    return disp_boost * (0.55*dist_term + 0.30*age_term + 0.15*size_term)
# Higher score = higher priority. Displacement gets 35% boost.
```

---

## TASK E -- CISD: Correct Algorithm
**Expected AUC: ~0.001-0.003. Current implementation is likely wrong.**

```python
def compute_cisd(df, L=20, min_run=2,
                 ssl_sweep_age=None, bsl_sweep_age=None):
    """
    Bullish CISD:
      1. Find most recent run of >= min_run consecutive bearish candles in last L bars.
      2. delivery_open = open of FIRST candle in that run.
      3. Fires when: close[t] > delivery_open AND close[t-1] <= delivery_open.

    min_run=2: require at least 2 consecutive bearish candles (prevents single doji).
    cisd_with_sweep: also requires sweep within 6 bars (strongest signal).
    """
    n = len(df)
    cisd_bull = [0]*n; cisd_bear = [0]*n
    cisd_bull_ws = [0]*n; cisd_bear_ws = [0]*n

    for t in range(1, n):
        # Bullish CISD
        run, start = 0, None
        for j in range(t-1, max(-1, t-L), -1):
            if df['close'].iloc[j] < df['open'].iloc[j]:
                run += 1; start = j
            else:
                if run >= min_run: break
                run, start = 0, None
        if start is not None and run >= min_run:
            level = df['open'].iloc[start]
            if df['close'].iloc[t] > level and df['close'].iloc[t-1] <= level:
                cisd_bull[t] = 1
                if (ssl_sweep_age is not None and
                    not (ssl_sweep_age[t] != ssl_sweep_age[t]) and  # not NaN
                    ssl_sweep_age[t] <= 6):
                    cisd_bull_ws[t] = 1

        # Bearish CISD
        run, start = 0, None
        for j in range(t-1, max(-1, t-L), -1):
            if df['close'].iloc[j] > df['open'].iloc[j]:
                run += 1; start = j
            else:
                if run >= min_run: break
                run, start = 0, None
        if start is not None and run >= min_run:
            level = df['open'].iloc[start]
            if df['close'].iloc[t] < level and df['close'].iloc[t-1] >= level:
                cisd_bear[t] = 1
                if (bsl_sweep_age is not None and
                    not (bsl_sweep_age[t] != bsl_sweep_age[t]) and
                    bsl_sweep_age[t] <= 6):
                    cisd_bear_ws[t] = 1

    return cisd_bull, cisd_bear, cisd_bull_ws, cisd_bear_ws
```

Output columns:
```
cisd_bull              float32
cisd_bear              float32
cisd_bull_age          float32  bars since last (NaN if > 20)
cisd_bear_age          float32
cisd_bull_with_sweep   float32
cisd_bear_with_sweep   float32
```

---

## TASK: MSS -- New Function

```python
MSS_K = 3  # RESOLVED: 3 not 5

def detect_mss(df, int_choch_bull, int_choch_bear, disp_bull, disp_bear,
               disp_strength, sweep_ssl_age=None, sweep_bsl_age=None):
    """
    MSS_bull[t] = int_choch_bull[t] AND any(disp_bull[t-MSS_K:t+1])
    MSS_bear[t] = int_choch_bear[t] AND any(disp_bear[t-MSS_K:t+1])
    mss_with_sweep: also requires sweep within 10 bars.
    """
    n = len(int_choch_bull)
    mss_bull=[0]*n; mss_bear=[0]*n
    mss_bull_ws=[0]*n; mss_bear_ws=[0]*n
    mss_str=[0.0]*n

    for t in range(n):
        lo = max(0, t - MSS_K)
        if int_choch_bull[t] and any(disp_bull[lo:t+1]):
            mss_bull[t] = 1
            mss_str[t] = max(disp_strength[lo:t+1])
            if (sweep_ssl_age is not None and
                not (sweep_ssl_age[t] != sweep_ssl_age[t]) and
                sweep_ssl_age[t] <= 10):
                mss_bull_ws[t] = 1
        if int_choch_bear[t] and any(disp_bear[lo:t+1]):
            mss_bear[t] = 1
            mss_str[t] = max(disp_strength[lo:t+1])
            if (sweep_bsl_age is not None and
                not (sweep_bsl_age[t] != sweep_bsl_age[t]) and
                sweep_bsl_age[t] <= 10):
                mss_bear_ws[t] = 1

    return mss_bull, mss_bear, mss_str, mss_bull_ws, mss_bear_ws
```

Output:
```
mss_bull_fired       float32
mss_bear_fired       float32
mss_bull_age         float32  (cap 24)
mss_bear_age         float32
mss_strength         float32
mss_bull_with_sweep  float32
mss_bear_with_sweep  float32
```

---

## PARAMETER CORRECTIONS TO EXISTING FUNCTIONS

| Function | Parameter | Old | New |
|----------|-----------|-----|-----|
| detect_fvg_bull/bear | age_cap | 288 | 100 |
| detect_fvg_bull/bear | min_size_atr | 0.10 | 0.50 |
| compute_liq_levels | eq_tolerance_atr | 0.20 | 0.10 |
| detect_breaker_blocks | age_cap | 576 | 200 |
| compute_ote_dist | fib_low | 0.62 | 0.618 |
| compute_ote_dist | fib_high | 0.79 | 0.786 |

---

## DEPENDENCY ORDER (augment_features() must run in this order)

1.  detect_displacement()           -- standalone, OHLC + ATR only
2.  compute_swing_dual_layer()      -- provides int_* and ext_* swing columns
3.  detect_ob_bull()                -- needs bos_close from step 2 + displacement from step 1
4.  detect_ob_bear()
5.  detect_fvg_bull()               -- standalone (uses displacement tag from step 1)
6.  detect_fvg_bear()
7.  compute_liq_levels()            -- uses swing prices for PDH/PDL/PWH/PWL
8.  detect_sweep()                  -- needs confirmed swing highs/lows from step 2
9.  compute_premium_discount()      -- needs ext_swing_high_price, ext_swing_low_price
10. compute_ote_dist()              -- needs swing prices
11. compute_cisd()                  -- with_sweep variant needs step 8
12. detect_mss()                    -- needs int_choch from step 2 + displacement step 1
13. detect_sweep_sequence()         -- needs steps 8 + 1 + 5/6
14. detect_breaker_blocks()         -- needs OB state from steps 3/4
15. compute_ob_quality()            -- needs OB + displacement + FVG

---

## EXPECTED NEW FEATURE COUNT

Task F (displacement):      14 columns
Task A (OB overhaul):      ~82 columns (13 per OB x 3 ranks x 2 dirs + 4 aggregates)
Task G (premium/discount):   7 columns
Task H (OTE 0.705):          2 columns
Task D (dual-layer swings): 22 columns
Task B (sweeps):            12 columns
Task C (FVG enhancements): ~30 columns
Task E (CISD fix):           6 columns
MSS:                         7 columns

TOTAL NEW: ~182 columns
ONTHEFLY_FEATURES: grows from 17 to ~199

---

## SMOKE TEST EXPECTED COVERAGE (10,000 bars)

| Feature | Expected | Flag if outside |
|---------|----------|----------------|
| displacement_bull | 5-15% | <2% or >25% |
| sweep_ssl_fired | 1-4% | <0.5% |
| sweep_seq_bull_complete | 0.5-2% | <0.2% |
| mss_bull_fired | 2-6% | >15% |
| cisd_bull | 3-8% | >15% |
| cisd_bull_with_sweep | 0.5-3% | -- |
| pd_position_5m | 100% | NaN > 0.1% |
| in_ote_bull | 15-20% | -- |
| ob_bull_1_in_zone | 60-85% | <40% = age cap too short |
| fvg_bull_1_fill_fraction | 15-25% | -- |

---

## CC EXECUTION PROMPT

```
Read CLAUDE.md, MEMORY.md, and D53_IMPLEMENTATION_SPEC.md before starting.
D53_IMPLEMENTATION_SPEC.md is the authoritative spec. Follow it exactly.

TASK: Implement D53 -- ICT rules overhaul in core/signals/ict/rules.py.

Execute tasks in this order: F -> A -> G -> H -> D -> B -> C -> E -> MSS.
After each task: run causality test at T=10000 before proceeding.

After ALL tasks:
  1. Run full causality suite at T in [1000, 5000, 10000, 50000] for all functions.
  2. Print swing confirmation lag correlation for int layer: lag-0 vs lag-5.
  3. Run smoke test on 10,000 bars. Print coverage % for every new feature family.
  4. Update simulator.py ONTHEFLY_FEATURES with all new column names.
  5. Update data/labeled/feature_catalog_v3.yaml with all new features.
  6. Log as D53 in STRATEGY_LOG.md. Update CLAUDE.md and THE_PLAN.md.

ENVIRONMENT:
Python 3.14, Windows, cp1252 encoding.
ASCII only in print statements. No Unicode.
No numba. Stateful loop for OB/FVG/sweep state machines.
Do not use vectorized pandas where per-bar state is needed.

CRITICAL CHECKS:
  - int_swing_high lag-0 correlation must be near 0 (if not, lookahead bug)
  - All causality tests PASS before moving to next function
  - ONTHEFLY_FEATURES count grows from 17 to ~199
  - Feature catalog names must exactly match ONTHEFLY_FEATURES keys

DO NOT:
  - Run experiments (that is D54+)
  - Touch holdout data (D51 not complete yet)
  - Modify embargo or WF logic
  - Retrain any existing models
```
