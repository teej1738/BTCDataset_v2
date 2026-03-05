# Existential Experiment Results

Date: 2026-03-05 11:43
Data: 543,167 rows, 2020-01-01 00:00:00+00:00 to 2025-02-28 23:50:00+00:00
Features: 64 (BTCDataset_v2 validated Tier1 set)

## Configuration
- Folds: 3 (expanding window)
- Seeds: [42, 123, 456]
- Embargo: 2x max_hold bars
- LightGBM: lr=0.03, leaves=64, min_child=200, depth=8
- Early stopping: patience=50 on last 15% of train
- Calibration: isotonic regression
- Entry threshold: prob > 0.55
- Cooldown: 6 bars
- Cost: 5.0bps taker + 1.0bps slip + 0.010%/8h funding

## Summary Table

| Config | Dir | ATR TF | SL | TP | Hold | AUC | cost_R | EV/trade | WR | SR/trade | Trades/yr | Ann SR |
|--------|-----|--------|----|----|------|-----|--------|----------|----|---------|-----------| -------|
| A | L | 1H | 1.0 | 1.0 | 48 | 0.7419 | 0.1489 | $92.99 | 65.0% | 0.2233 | 8964 | 21.14 |
| A | S | 1H | 1.0 | 1.0 | 48 | 0.7418 | 0.1582 | $116.35 | 66.5% | 0.2999 | 8041 | 26.90 |
| B | L | 1H | 1.0 | 1.5 | 48 | 0.7286 | 0.1445 | $125.58 | 61.2% | 0.2500 | 7479 | 21.62 |
| B | S | 1H | 1.0 | 1.5 | 48 | 0.7319 | 0.1598 | $108.09 | 57.3% | 0.2312 | 8736 | 21.60 |
| C | L | 1H | 1.0 | 2.0 | 96 | 0.7095 | 0.1421 | $174.12 | 58.5% | 0.2548 | 5446 | 18.81 |
| C | S | 1H | 1.0 | 2.0 | 96 | 0.7102 | 0.1696 | $54.91 | 45.8% | 0.0934 | 11370 | 9.96 |
| D | L | 4H | 1.0 | 1.0 | 96 | 0.6791 | 0.0683 | $144.52 | 60.8% | 0.1841 | 6551 | 14.90 |
| D | S | 4H | 1.0 | 1.0 | 96 | 0.6810 | 0.0815 | $214.01 | 64.4% | 0.3338 | 4897 | 23.36 |
| E | L | 4H | 1.0 | 1.5 | 96 | 0.6722 | 0.0668 | $189.75 | 60.9% | 0.2149 | 5106 | 15.35 |
| E | S | 4H | 1.0 | 1.5 | 96 | 0.6750 | 0.0837 | $210.03 | 61.1% | 0.2928 | 4880 | 20.46 |
| F | L | 4H | 1.0 | 2.0 | 192 | 0.6552 | 0.0699 | $277.99 | 59.2% | 0.2364 | 4415 | 15.70 |
| F | S | 4H | 1.0 | 2.0 | 192 | 0.6610 | 0.0881 | $138.73 | 51.6% | 0.1440 | 7544 | 12.51 |

## Verdict

**VIABLE**

VIABLE -- 12 config(s) pass all criteria.
Best: Config D short (SR/trade=0.3338, AUC=0.6810, cost_R=0.0815, EV=$214.01)

## Criteria Applied

DEAD if ALL of:
- Per-trade Sharpe < 0.02 at ALL configs
- AUC < 0.55 at ALL configs
- Trade count < 50 annualized at ALL configs
- Cost-adjusted EV <= 0 at ALL configs

VIABLE if at least ONE config has ALL of:
- Per-trade Sharpe >= 0.05
- cost_R < 0.25
- AUC >= 0.60
- Cost-adjusted EV > 0
- Win rate > break-even
- >= 100 trades annualized

MARGINAL if between DEAD and VIABLE.
