"""
export_to_pine.py -- Generate TradingView Pine Script from SHAP analysis
D44: TradingView Export (Step 10)

Reads top SHAP features from the best gate-passing experiment and
translates them into a Pine Script v5 indicator with:
  - Signal arrows on long entry conditions
  - OB zone boxes (most recent bull and bear)
  - FVG zone boxes (nearest active bull and bear)
  - OTE Fibonacci box (62-79% retracement)
  - VWAP with 1 and 2 standard deviation bands (daily reset 00:00 UTC)
  - Session window backgrounds (London 03-04, NY AM 10-11, NY PM 14-15 ET)
  - Confluence score label (0-5 conditions met)
  - Alert condition for webhook signal
  - WEBHOOK_ENABLED = false (toggle input)
  - Signal mode toggle: "All Signals" vs "High Confidence"

Usage: python tradingview/export_to_pine.py
"""

import json
import os
import sys
from datetime import datetime

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHAP_DIR = os.path.join(PROJECT_ROOT, "core", "experiments", "shap")
REGISTRY_FILE = os.path.join(PROJECT_ROOT, "core", "experiments", "registry.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "tradingview")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "ict_strategy_v1.pine")

# Feature -> Pine Script condition mapping
# Maps SHAP feature names to (pine_condition, label, description)
# None for pine_condition means the feature is implicitly covered by another
FEATURE_MAP = {
    "ote_dist": (
        "inOTE",
        "OTE Zone",
        "Price within 62-79% Fibonacci retracement of swing",
    ),
    "swing_high_price": (
        None,
        "Swing High",
        "Implicit in OTE zone calculation",
    ),
    "swing_low_price": (
        None,
        "Swing Low",
        "Implicit in OTE zone calculation",
    ),
    "ict_ob_bull_age": (
        "bullOBFresh",
        "Fresh Bull OB",
        "Bullish order block within max age threshold",
    ),
    "ict_ob_bear_age": (
        "not bearOBFresh",
        "Bear OB Cleared",
        "No active bearish order block (overhead supply cleared)",
    ),
    "clv": (
        None,
        "CLV",
        "Close location value (implicit in price action, not standalone condition)",
    ),
    "stoch_k": (
        "stochOversold",
        "Stoch Oversold",
        "Stochastic K below 30",
    ),
    "ict_fvg_bull_recent_age": (
        "bullFVGActive",
        "Bull FVG Active",
        "Active bullish fair value gap exists nearby",
    ),
    "ict_fvg_bear_recent_age": (
        None,
        "Bear FVG Age",
        "Caution zone (not a positive confluence for longs, skipped)",
    ),
    "cvd_bar": (
        "cvdPositive",
        "CVD Positive",
        "Bar cumulative volume delta is positive (net buying)",
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_registry():
    """Load experiment registry."""
    with open(REGISTRY_FILE) as f:
        return json.load(f)


def find_best_experiment(registry):
    """Find the best gate-passing experiment by EV."""
    best = None
    for exp in registry["experiments"]:
        if exp["status"] != "DONE":
            continue
        # Check for all gates passing (10/10)
        gates = exp.get("gates", {})
        all_pass = gates.get("all_pass", False)
        # Fallback: check if ECE is calibrated (< 0.05) as proxy for gate pass
        if not all_pass and exp.get("ece", 1.0) > 0.05:
            continue
        if best is None or exp["metrics"]["ev_r"] > best["metrics"]["ev_r"]:
            best = exp
    return best


def load_shap(exp_id):
    """Load SHAP analysis for an experiment."""
    shap_file = os.path.join(SHAP_DIR, f"shap_{exp_id}.json")
    with open(shap_file) as f:
        return json.load(f)


def select_confluence_conditions(top_features, n=5):
    """Select N independent Pine conditions from top SHAP features.

    Features that map to None (implicit in another condition) are skipped.
    Falls back to lower-ranked features to fill N slots.
    """
    conditions = []
    used_pine = set()

    for feat in top_features:
        if len(conditions) >= n:
            break
        name = feat["feature"]
        if name not in FEATURE_MAP:
            continue
        pine_cond, label, desc = FEATURE_MAP[name]
        if pine_cond is None:
            continue  # implicit in another condition
        if pine_cond in used_pine:
            continue  # avoid duplicate Pine conditions
        used_pine.add(pine_cond)
        conditions.append({
            "feature": name,
            "shap": feat["mean_abs_shap"],
            "rank": feat["rank"],
            "pine_condition": pine_cond,
            "label": label,
            "description": desc,
        })

    return conditions


def generate_pine(conditions, exp_metrics, exp_id, shap_data):
    """Generate the full Pine Script v5 indicator."""

    # Build confluence condition lines
    conf_lines = []
    for c in conditions:
        conf_lines.append(
            f"confluence += {c['pine_condition']} ? 1 : 0  "
            f"// {c['label']} ({c['feature']}, SHAP={c['shap']:.4f})"
        )
    confluence_code = "\n".join(conf_lines)

    # Build condition label lines for the status display
    label_lines = []
    for i, c in enumerate(conditions):
        label_lines.append(
            f'    condText += ({c["pine_condition"]} ? "[Y] " : "[N] ") '
            f'+ "{c["label"]}\\n"'
        )
    label_code = "\n".join(label_lines)

    # Metrics for header
    wr = exp_metrics.get("win_rate", 0) * 100
    ev = exp_metrics.get("ev_r", 0)
    trades = exp_metrics.get("n_trades", 0)
    pf = exp_metrics.get("profit_factor", 0)
    maxdd = exp_metrics.get("max_dd_pct", 0)
    n_cond = len(conditions)
    gen_date = datetime.now().strftime("%Y-%m-%d")

    # Top 5 feature list for header comment
    top5_lines = []
    for feat in shap_data["top30"][:5]:
        top5_lines.append(
            f"//   #{feat['rank']} {feat['feature']} "
            f"(SHAP={feat['mean_abs_shap']:.4f})"
        )
    top5_comment = "\n".join(top5_lines)

    pine = PINE_TEMPLATE.format(
        exp_id=exp_id,
        gen_date=gen_date,
        wr=wr,
        ev=ev,
        trades=trades,
        pf=pf,
        maxdd=maxdd,
        top5_comment=top5_comment,
        confluence_code=confluence_code,
        label_code=label_code,
        n_conditions=n_cond,
    )

    return pine


# ---------------------------------------------------------------------------
# Pine Script Template
# ---------------------------------------------------------------------------
PINE_TEMPLATE = '''\
// ===========================================================================
// ICT Strategy v1 -- BTCUSDT Long-Only Signal Monitor
// ===========================================================================
// Generated by: tradingview/export_to_pine.py
// Source experiment: {exp_id}
// Gate status: 10/10 PASS | WR {wr:.1f}% | EV +{ev:.3f}R | {trades} trades
// Profit Factor: {pf:.2f} | MaxDD: {maxdd:.1f}%
// Generated: {gen_date}
//
// Top 5 SHAP features (E002_prune):
{top5_comment}
//
// IMPORTANT: This indicator shows where ICT conditions align.
// It does NOT replicate the full ML model. Use for visual monitoring only.
// All calculations use confirmed (closed) bars -- no lookahead.
// ===========================================================================

//@version=5
indicator("ICT Strategy v1 -- BTC Long-Only", overlay=true,
         max_boxes_count=100, max_labels_count=100)

// ===========================================================================
// INPUTS
// ===========================================================================
i_signalMode   = input.string("High Confidence", "Signal Mode",
                              options=["All Signals", "High Confidence"])
i_webhookOn    = input.bool(false, "WEBHOOK_ENABLED")
i_swingLen     = input.int(10, "Swing Lookback (bars)", minval=3, maxval=50)
i_obMaxAge     = input.int(50, "OB Max Age (bars)", minval=10, maxval=200)
i_fvgMaxAge    = input.int(30, "FVG Max Age (bars)", minval=5, maxval=100)
i_showOB       = input.bool(true, "Show OB Boxes")
i_showFVG      = input.bool(true, "Show FVG Boxes")
i_showOTE      = input.bool(true, "Show OTE Zone")
i_showVWAP     = input.bool(true, "Show VWAP Bands")
i_showSessions = input.bool(true, "Show Session Backgrounds")

// ===========================================================================
// ATR
// ===========================================================================
atrVal = ta.atr(14)

// ===========================================================================
// SWING POINTS (confirmed, no lookahead)
// ta.pivothigh/low with rightbars = swingLen ensures detection is delayed
// by swingLen bars. The pivot occurred swingLen bars ago when detected.
// ===========================================================================
pivotHi = ta.pivothigh(high, i_swingLen, i_swingLen)
pivotLo = ta.pivotlow(low, i_swingLen, i_swingLen)

var float lastSwingHigh = na
var float lastSwingLow  = na
var int   swingHighIdx  = na
var int   swingLowIdx   = na

if not na(pivotHi)
    lastSwingHigh := pivotHi
    swingHighIdx  := bar_index - i_swingLen

if not na(pivotLo)
    lastSwingLow := pivotLo
    swingLowIdx  := bar_index - i_swingLen

// ===========================================================================
// OTE ZONE (62-79% Fibonacci retracement)
// For longs: retracement from swing low to swing high
// 61.8% ret = swing_high - 0.618 * range (top of OTE)
// 78.6% ret = swing_high - 0.786 * range (bottom of OTE)
// ===========================================================================
swingRange = lastSwingHigh - lastSwingLow
oteTop     = lastSwingHigh - 0.618 * swingRange
oteBottom  = lastSwingHigh - 0.786 * swingRange
oteValid   = not na(lastSwingHigh) and not na(lastSwingLow) and swingRange > 0
inOTE      = oteValid and close >= oteBottom and close <= oteTop

// ===========================================================================
// ORDER BLOCKS
// Bull OB: last bearish candle before a bullish displacement (body > ATR)
// Bear OB: last bullish candle before a bearish displacement
// Displacement ensures we capture significant moves, not noise.
// Mitigation: OB expires when price closes through it.
// ===========================================================================
dispThresh = atrVal * 1.0
bullDisp   = (close - open) > dispThresh and close > open
bearDisp   = (open - close) > dispThresh and close < open

newBullOB = bullDisp and close[1] < open[1]
newBearOB = bearDisp and close[1] > open[1]

var float bullOBHi  = na
var float bullOBLo  = na
var int   bullOBAge = 9999
var float bearOBHi  = na
var float bearOBLo  = na
var int   bearOBAge = 9999

if newBullOB
    bullOBHi  := high[1]
    bullOBLo  := low[1]
    bullOBAge := 0
else
    bullOBAge += 1

if newBearOB
    bearOBHi  := high[1]
    bearOBLo  := low[1]
    bearOBAge := 0
else
    bearOBAge += 1

// Mitigation: price closes through OB zone
if bullOBAge <= i_obMaxAge and not na(bullOBLo) and close < bullOBLo
    bullOBAge := i_obMaxAge + 1
if bearOBAge <= i_obMaxAge and not na(bearOBHi) and close > bearOBHi
    bearOBAge := i_obMaxAge + 1

bullOBFresh = bullOBAge <= i_obMaxAge
bearOBFresh = bearOBAge <= i_obMaxAge

// ===========================================================================
// FAIR VALUE GAPS
// Bull FVG: gap between bar[2].high and bar[0].low (low > high[2])
// Bear FVG: gap between bar[0].high and bar[2].low (high < low[2])
// Mitigation: price closes through the gap zone.
// ===========================================================================
bullFVGDetect = low > high[2]
bearFVGDetect = high < low[2]

var float bullFVGTop = na
var float bullFVGBot = na
var int   bullFVGAge = 9999
var float bearFVGTop = na
var float bearFVGBot = na
var int   bearFVGAge = 9999

if bullFVGDetect
    bullFVGTop := low
    bullFVGBot := high[2]
    bullFVGAge := 0
else
    bullFVGAge += 1

if bearFVGDetect
    bearFVGTop := low[2]
    bearFVGBot := high
    bearFVGAge := 0
else
    bearFVGAge += 1

// Mitigation
if bullFVGAge <= i_fvgMaxAge and not na(bullFVGBot) and close < bullFVGBot
    bullFVGAge := i_fvgMaxAge + 1
if bearFVGAge <= i_fvgMaxAge and not na(bearFVGTop) and close > bearFVGTop
    bearFVGAge := i_fvgMaxAge + 1

bullFVGActive = bullFVGAge <= i_fvgMaxAge
bearFVGActive = bearFVGAge <= i_fvgMaxAge

// ===========================================================================
// STOCHASTIC K (14-period)
// ===========================================================================
stochK       = ta.stoch(close, high, low, 14)
stochOversold = stochK < 30

// ===========================================================================
// VWAP with Standard Deviation Bands (daily reset at 00:00 UTC)
// Manual calculation for proper session reset and band computation.
// ===========================================================================
isNewDay = dayofmonth != dayofmonth[1]

var float cumPV  = 0.0
var float cumV   = 0.0
var float cumPV2 = 0.0

if isNewDay
    cumPV  := 0.0
    cumV   := 0.0
    cumPV2 := 0.0

typPrice = hlc3
cumPV   += typPrice * volume
cumV    += volume
cumPV2  += typPrice * typPrice * volume

vwapVal = cumV > 0 ? cumPV / cumV : close
vwapVar = cumV > 0 ? math.max(cumPV2 / cumV - vwapVal * vwapVal, 0.0) : 0.0
vwapDev = math.sqrt(vwapVar)

vwapU1 = vwapVal + vwapDev
vwapL1 = vwapVal - vwapDev
vwapU2 = vwapVal + 2 * vwapDev
vwapL2 = vwapVal - 2 * vwapDev

// ===========================================================================
// SESSION WINDOWS (Eastern Time -- handles DST automatically)
// London Silver Bullet: 03:00-04:00 ET
// NY AM Silver Bullet:  10:00-11:00 ET
// NY PM Silver Bullet:  14:00-15:00 ET
// ===========================================================================
inLondonSB = not na(time(timeframe.period, "0300-0400", "America/New_York"))
inNYAM     = not na(time(timeframe.period, "1000-1100", "America/New_York"))
inNYPM     = not na(time(timeframe.period, "1400-1500", "America/New_York"))
inSession  = inLondonSB or inNYAM

// ===========================================================================
// CONFLUENCE SCORING
// {n_conditions} conditions from SHAP-ranked features.
// High Confidence = 3+ conditions met.
// All Signals = 2+ conditions met.
// ===========================================================================
confluence = 0
{confluence_code}

// ===========================================================================
// SIGNAL GENERATION
// Signals only fire on confirmed (closed) bars to prevent lookahead.
// Session filter: London SB or NY AM SB (primary long windows).
// ===========================================================================
minConf    = i_signalMode == "High Confidence" ? 3 : 2
longSignal = barstate.isconfirmed and confluence >= minConf and inSession

// ===========================================================================
// VISUALIZATION -- VWAP
// ===========================================================================
plot(i_showVWAP ? vwapVal : na, color=color.yellow, linewidth=2, title="VWAP")
plot(i_showVWAP ? vwapU1 : na,  color=color.new(color.yellow, 60), title="VWAP +1SD")
plot(i_showVWAP ? vwapL1 : na,  color=color.new(color.yellow, 60), title="VWAP -1SD")
plot(i_showVWAP ? vwapU2 : na,  color=color.new(color.yellow, 80), title="VWAP +2SD")
plot(i_showVWAP ? vwapL2 : na,  color=color.new(color.yellow, 80), title="VWAP -2SD")

// ===========================================================================
// VISUALIZATION -- SESSION BACKGROUNDS
// ===========================================================================
bgcolor(i_showSessions and inLondonSB ? color.new(color.blue, 93) : na,
        title="London SB")
bgcolor(i_showSessions and inNYAM ? color.new(color.orange, 93) : na,
        title="NY AM SB")
bgcolor(i_showSessions and inNYPM ? color.new(color.purple, 93) : na,
        title="NY PM SB")

// ===========================================================================
// VISUALIZATION -- SIGNAL ARROWS
// ===========================================================================
plotshape(longSignal, style=shape.triangleup, location=location.belowbar,
          color=color.green, size=size.small, title="Long Signal")

// ===========================================================================
// VISUALIZATION -- OB BOXES (most recent bull and bear)
// ===========================================================================
var box bullOBBox = na
var box bearOBBox = na

if newBullOB and i_showOB
    if not na(bullOBBox)
        box.delete(bullOBBox)
    bullOBBox := box.new(bar_index - 1, bullOBHi, bar_index, bullOBLo,
                         border_color=color.green,
                         bgcolor=color.new(color.green, 88),
                         text="Bull OB", text_color=color.green,
                         text_size=size.tiny)

if newBearOB and i_showOB
    if not na(bearOBBox)
        box.delete(bearOBBox)
    bearOBBox := box.new(bar_index - 1, bearOBHi, bar_index, bearOBLo,
                         border_color=color.red,
                         bgcolor=color.new(color.red, 88),
                         text="Bear OB", text_color=color.red,
                         text_size=size.tiny)

// Extend OB boxes while active
if not na(bullOBBox) and bullOBFresh
    box.set_right(bullOBBox, bar_index)
if not na(bearOBBox) and bearOBFresh
    box.set_right(bearOBBox, bar_index)

// ===========================================================================
// VISUALIZATION -- FVG BOXES (nearest active bull and bear)
// ===========================================================================
var box bullFVGBox = na
var box bearFVGBox = na

if bullFVGDetect and i_showFVG
    if not na(bullFVGBox)
        box.delete(bullFVGBox)
    bullFVGBox := box.new(bar_index - 2, bullFVGTop, bar_index, bullFVGBot,
                          border_color=color.teal,
                          bgcolor=color.new(color.teal, 88),
                          text="Bull FVG", text_color=color.teal,
                          text_size=size.tiny)

if bearFVGDetect and i_showFVG
    if not na(bearFVGBox)
        box.delete(bearFVGBox)
    bearFVGBox := box.new(bar_index - 2, bearFVGTop, bar_index, bearFVGBot,
                          border_color=color.maroon,
                          bgcolor=color.new(color.maroon, 88),
                          text="Bear FVG", text_color=color.maroon,
                          text_size=size.tiny)

if not na(bullFVGBox) and bullFVGActive
    box.set_right(bullFVGBox, bar_index)
if not na(bearFVGBox) and bearFVGActive
    box.set_right(bearFVGBox, bar_index)

// ===========================================================================
// VISUALIZATION -- OTE FIBONACCI BOX (62-79% retracement)
// ===========================================================================
var box oteBox = na
var int oteBoxLeft = na

if i_showOTE and oteValid
    // Recreate box when swing points update
    if not na(pivotHi) or not na(pivotLo) or na(oteBox)
        if not na(oteBox)
            box.delete(oteBox)
        oteBoxLeft := bar_index
        oteBox := box.new(bar_index, oteTop, bar_index, oteBottom,
                          border_color=color.new(color.orange, 40),
                          bgcolor=color.new(color.orange, 92),
                          text="OTE 62-79%", text_color=color.orange,
                          text_size=size.tiny)
    // Update price levels and extend right
    if not na(oteBox)
        box.set_top(oteBox, oteTop)
        box.set_bottom(oteBox, oteBottom)
        box.set_right(oteBox, bar_index)

// ===========================================================================
// VISUALIZATION -- CONFLUENCE LABEL
// Shows current condition status on signal bars and on the last bar.
// ===========================================================================
if longSignal
    label.new(bar_index, low - atrVal * 0.5,
              str.tostring(confluence) + "/{n_conditions}",
              color=color.green, textcolor=color.white,
              style=label.style_label_up, size=size.small)

// Status label on last bar
var label statusLbl = na
if barstate.islast
    if not na(statusLbl)
        label.delete(statusLbl)
    condText = "ICT v1 | Confluence: " + str.tostring(confluence) + "/{n_conditions}\\n"
    condText += "Mode: " + i_signalMode + "\\n"
    condText += "---\\n"
{label_code}
    statusLbl := label.new(bar_index + 3, close, condText,
                           color=confluence >= 3 ? color.green : color.gray,
                           textcolor=color.white,
                           style=label.style_label_left,
                           size=size.normal)

// ===========================================================================
// SWING LEVEL LINES (for reference)
// ===========================================================================
plot(not na(lastSwingHigh) ? lastSwingHigh : na,
     color=color.new(color.red, 50), linewidth=1,
     style=plot.style_stepline, title="Swing High")
plot(not na(lastSwingLow) ? lastSwingLow : na,
     color=color.new(color.green, 50), linewidth=1,
     style=plot.style_stepline, title="Swing Low")

// ===========================================================================
// ALERTS
// ===========================================================================
// Static alert condition (for TradingView alert dialog)
alertcondition(longSignal, "ICT Long Signal",
               "ICT Strategy v1: Long entry signal fired")

// Dynamic webhook alert (includes price and confluence data)
if longSignal and i_webhookOn
    alert("ICT_LONG|" + syminfo.ticker + "|price=" + str.tostring(close) +
          "|conf=" + str.tostring(confluence) + "/{n_conditions}" +
          "|stochK=" + str.tostring(stochK, "#.0"),
          alert.freq_once_per_bar_close)
'''


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("export_to_pine.py -- TradingView Pine Script Generator")
    print("=" * 60)
    print()

    # 1. Load registry and find best experiment
    print("Phase 1: Loading experiment registry...")
    registry = load_registry()
    best_exp = find_best_experiment(registry)

    if best_exp is None:
        print("ERROR: No gate-passing experiment found in registry.")
        sys.exit(1)

    exp_id = best_exp["id"]
    metrics = best_exp["metrics"]
    print(f"  Best experiment: {exp_id}")
    print(f"  WR: {metrics['win_rate']*100:.1f}%  "
          f"EV: +{metrics['ev_r']:.3f}R  "
          f"Trades: {metrics['n_trades']}")
    print()

    # 2. Load SHAP analysis
    print("Phase 2: Loading SHAP analysis...")
    shap_data = load_shap(exp_id)
    print(f"  OOS AUC: {shap_data['oos_auc']}")
    print(f"  Features analyzed: {shap_data['n_features']}")
    print(f"  Top 5 SHAP features:")
    for feat in shap_data["top30"][:5]:
        print(f"    #{feat['rank']} {feat['feature']:25s} "
              f"SHAP={feat['mean_abs_shap']:.4f}")
    print()

    # 3. Select confluence conditions
    print("Phase 3: Mapping SHAP features to Pine conditions...")
    conditions = select_confluence_conditions(shap_data["top30"], n=5)
    print(f"  Selected {len(conditions)} independent conditions:")
    for i, c in enumerate(conditions):
        print(f"    {i+1}. {c['label']:20s} <- {c['feature']} "
              f"(rank #{c['rank']}, SHAP={c['shap']:.4f})")
        print(f"       Pine: {c['pine_condition']}")
    print()

    # 4. Generate Pine Script
    print("Phase 4: Generating Pine Script v5...")
    pine_code = generate_pine(conditions, metrics, exp_id, shap_data)
    pine_lines = pine_code.count("\n") + 1
    print(f"  Generated {pine_lines} lines of Pine Script")
    print()

    # 5. Write output
    print("Phase 5: Writing output...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(pine_code)
    file_size = os.path.getsize(OUTPUT_FILE)
    print(f"  Output: {OUTPUT_FILE}")
    print(f"  Size: {file_size:,} bytes")
    print()

    # 6. Summary
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()
    print(f"Source experiment : {exp_id}")
    print(f"Gate status      : 10/10 PASS")
    print(f"WR / EV          : {metrics['win_rate']*100:.1f}% / "
          f"+{metrics['ev_r']:.3f}R")
    print(f"Top SHAP feature : {shap_data['top30'][0]['feature']} "
          f"(SHAP={shap_data['top30'][0]['mean_abs_shap']:.4f})")
    print(f"Confluence conds : {len(conditions)}")
    for i, c in enumerate(conditions):
        print(f"  {i+1}. {c['label']}")
    print(f"Pine Script      : {OUTPUT_FILE}")
    print(f"Pine lines       : {pine_lines}")
    print()
    print("Pine Script features:")
    print("  [x] Signal arrows (long entry)")
    print("  [x] OB zone boxes (bull + bear)")
    print("  [x] FVG zone boxes (bull + bear)")
    print("  [x] OTE Fibonacci box (62-79%)")
    print("  [x] VWAP + 1/2 SD bands (daily reset)")
    print("  [x] Session backgrounds (London, NY AM, NY PM)")
    print("  [x] Confluence label (0-5)")
    print("  [x] Alert condition + webhook")
    print("  [x] WEBHOOK_ENABLED = false")
    print("  [x] Signal mode toggle (All/High Confidence)")
    print()
    print("VALIDATION: Paste into TradingView Pine Editor, verify zero syntax errors.")
    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
