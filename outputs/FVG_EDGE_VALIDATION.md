# FVG Edge Validation Results
Date: 2026-03-05
Data: 648,288 rows, 2020-01-01 to 2026-02-28

## FVG Detection Summary

| Timeframe | Total | Bullish | Bearish | Per Day | Median Size (ATR) |
|-----------|-------|---------|---------|---------|-------------------|
| 5m | 145,889 | 74,753 | 71,136 | 64.8 | 0.274 |
| 1H | 8,935 | 4,687 | 4,248 | 4.0 | 0.250 |
| 4H | 2,303 | 1,248 | 1,055 | 1.0 | 0.304 |

## Forward Returns (mean basis points)

| TF | Direction | 5m | 30m | 1h | 4h | 8h | 24h | t-stat (4h) | p-value |
|-----|-----------|------|------|------|------|------|-------|-------------|---------|
| 5m | bullish | +0.1 | +0.1 | +0.6 | +2.9 | +5.2 | +16.1 | 6.33 | 0.0000 |
| 5m | bearish | -0.0 | +0.4 | +0.7 | +1.8 | +4.3 | +12.2 | 3.78 | 0.9999 |
| 1H | bullish | -0.2 | -0.2 | +1.1 | +4.7 | +7.3 | +18.3 | 2.62 | 0.0044 |
| 1H | bearish | +0.8 | +0.8 | -0.5 | -0.6 | +3.7 | +13.9 | -0.28 | 0.3914 |
| 4H | bullish | -1.5 | -1.5 | +0.7 | +7.9 | +12.6 | +17.7 | 2.12 | 0.0170 |
| 4H | bearish | +1.5 | +3.2 | +4.3 | +2.8 | -5.4 | +11.5 | 0.55 | 0.7086 |

## Win Rates (% in hypothesized direction)

| TF | Direction | 5m | 30m | 1h | 4h | 8h | 24h | N |
|-----|-----------|------|------|------|------|------|------|-------|
| 5m | bullish | 47.4% | 47.9% | 48.7% | 49.9% | 50.2% | 51.7% | 74753 |
| 5m | bearish | 47.3% | 47.0% | 47.1% | 47.3% | 47.2% | 47.1% | 71136 |
| 1H | bullish | 48.8% | 46.4% | 47.8% | 47.8% | 48.8% | 50.8% | 4687 |
| 1H | bearish | 47.6% | 45.2% | 46.9% | 46.0% | 45.9% | 45.6% | 4248 |
| 4H | bullish | 45.8% | 46.5% | 46.4% | 49.0% | 51.9% | 51.6% | 1248 |
| 4H | bearish | 49.0% | 44.5% | 44.0% | 44.4% | 47.0% | 46.1% | 1055 |

## Fill Rate Analysis

| TF | Direction | Fill Rate (24h) | Median Fill Time (bars) | Mean Max Favorable Before Fill (bps) |
|-----|-----------|-----------------|-------------------------|--------------------------------------|
| 5m | bullish | 93.3% | 3 | +63.4 |
| 5m | bearish | 94.1% | 3 | +63.5 |
| 1H | bullish | 75.1% | 28 | +154.1 |
| 1H | bearish | 76.4% | 27 | +170.1 |
| 4H | bullish | 53.3% | 76 | +216.7 |
| 4H | bearish | 53.6% | 66 | +239.1 |

## Contextual Splits
(Using 5m FVGs for largest sample size; 4h forward return aligned to direction)

### By Gap Size

| Direction | Small Gap (bps) | Large Gap (bps) | Small N | Large N |
|-----------|-----------------|-----------------|---------|---------|
| bullish | +2.7 | +3.0 | 36913 | 37840 |
| bearish | -1.0 | -2.7 | 36032 | 35104 |

### By Volume

| Direction | High Vol (bps) | Low Vol (bps) | High N | Low N |
|-----------|----------------|---------------|--------|-------|
| bullish | +2.9 | +1.4 | 18521 | 18526 |
| bearish | -3.6 | -0.9 | 17952 | 17947 |

### By Time Period (6-month windows)

| Period | Direction | Aligned Mean (bps) | t-stat | p-value | N |
|--------|-----------|-------------------|--------|---------|---|
| 2020H1 | bullish | +5.3 | 2.41 | 0.0080 | 5538 |
| 2020H1 | bearish | +3.4 | 1.25 | 0.1048 | 5058 |
| 2020H2 | bullish | +10.5 | 7.75 | 0.0000 | 6285 |
| 2020H2 | bearish | -12.6 | -9.00 | 1.0000 | 5448 |
| 2021H1 | bullish | +1.8 | 0.60 | 0.2752 | 5202 |
| 2021H1 | bearish | -7.2 | -2.13 | 0.9833 | 4796 |
| 2021H2 | bullish | +3.5 | 1.73 | 0.0418 | 5188 |
| 2021H2 | bearish | -1.5 | -0.77 | 0.7801 | 5132 |
| 2022H1 | bullish | -5.2 | -2.50 | 0.9938 | 5060 |
| 2022H1 | bearish | +9.1 | 4.08 | 0.0000 | 5189 |
| 2022H2 | bullish | -1.4 | -0.91 | 0.8179 | 5873 |
| 2022H2 | bearish | +1.2 | 0.84 | 0.2016 | 5685 |
| 2023H1 | bullish | +6.8 | 4.97 | 0.0000 | 5756 |
| 2023H1 | bearish | -5.9 | -4.66 | 1.0000 | 5439 |
| 2023H2 | bullish | +4.5 | 4.99 | 0.0000 | 7036 |
| 2023H2 | bearish | -2.3 | -2.69 | 0.9964 | 6712 |
| 2024H1 | bullish | +4.4 | 3.21 | 0.0007 | 6368 |
| 2024H1 | bearish | -4.8 | -3.31 | 0.9995 | 5903 |
| 2024H2 | bullish | +3.9 | 2.87 | 0.0020 | 6536 |
| 2024H2 | bearish | -4.5 | -3.13 | 0.9991 | 6258 |
| 2025H1 | bullish | +1.3 | 1.09 | 0.1377 | 6735 |
| 2025H1 | bearish | -1.3 | -1.06 | 0.8565 | 6458 |
| 2025H2 | bullish | +0.9 | 1.00 | 0.1596 | 7153 |
| 2025H2 | bearish | +1.1 | 1.14 | 0.1281 | 6957 |
| 2026H1 | bullish | -7.1 | -3.02 | 0.9987 | 2023 |
| 2026H1 | bearish | +6.9 | 2.80 | 0.0026 | 2101 |

#### Trend Analysis
- **Bullish**: decaying (slope=-0.54 bps/window, R2=0.196, p=0.1302)
- **Bearish**: strengthening (slope=+0.40 bps/window, R2=0.071, p=0.3779)

### By ATR Regime

| Direction | High ATR (bps) | Low ATR (bps) | High N | Low N |
|-----------|----------------|---------------|--------|-------|
| bullish | +0.1 | +6.2 | 18684 | 18899 |
| bearish | -3.0 | -2.9 | 17789 | 17575 |

## FVG vs Random Entry (4h horizon)

| TF | Direction | FVG Mean (bps) | Random Mean (bps) | Bootstrap p-value |
|-----|-----------|----------------|-------------------|-------------------|
| 5m | bullish | +2.9 | +2.5 | 0.2380 |
| 5m | bearish | +1.8 | +2.5 | 0.0720 |
| 1H | bullish | +4.7 | +2.7 | 0.1440 |
| 1H | bearish | -0.6 | +2.6 | 0.0490 |
| 4H | bullish | +7.9 | +2.5 | 0.0730 |
| 4H | bearish | +2.8 | +2.5 | 0.5270 |

## VERDICT

### 5m
- **Bullish**: **WEAK** (mean=+2.9 bps, t=6.33, p=0.0000, WR=49.9%, bootstrap_p=0.2380)
- **Bearish**: **NONE** (mean=+1.8 bps, t=3.78, p=0.9999, WR=47.3%, bootstrap_p=0.0720)

### 1H
- **Bullish**: **WEAK** (mean=+4.7 bps, t=2.62, p=0.0044, WR=47.8%, bootstrap_p=0.1440)
- **Bearish**: **NONE** (mean=-0.6 bps, t=-0.28, p=0.3914, WR=46.0%, bootstrap_p=0.0490)

### 4H
- **Bullish**: **WEAK** (mean=+7.9 bps, t=2.12, p=0.0170, WR=49.0%, bootstrap_p=0.0730)
- **Bearish**: **NONE** (mean=+2.8 bps, t=0.55, p=0.7086, WR=44.4%, bootstrap_p=0.5270)

### Overall Assessment

- Bullish FVG edge: no significant trend over time (p=0.1302)
- Bearish FVG edge: no significant trend over time (p=0.3779)

### Cost Analysis
- Round-trip cost estimate: ~12-15 bps
- Edge must exceed this to be tradeable standalone

### Recommendation for Foundation Feature Engine

FVGs show no reliable standalone predictive edge after costs.
**Recommendation:** Treat FVGs as supplementary features only. They may contribute via interaction effects in ML models but should not be a primary feature family.
