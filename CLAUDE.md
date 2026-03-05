# BTCDataset_v2

**Status: FROZEN.** Do NOT modify existing source code.
Active development is in Foundation/ (Prism engine).
See trading-brain/VISION.md and trading-brain/SYSTEM_DESIGN.md.

---

## What This Repo Proved

- AUC 0.7993 on genuine out-of-sample BEAR holdout (BTC 5m, D55b)
- 64 features after pruning from 670 (SHAP rho=0.82 across folds)
- LightGBM binary classifier, walk-forward, 5 folds
- Signal survives at H1/H4 ATR stops (existential experiment, AD-50)
- 89% of signal importance from ICT structural conditions
- Best config: t=0.70, cd=288, Sharpe 21.34 (but 76% seed variance)
- Realistic live Sharpe target: 1.5-2.5 (not the backtest 10.71)

---

## Existential Experiment Results (2026-03-05)

12/12 configurations VIABLE:
- 1H ATR: AUC 0.71-0.74, cost_R 0.14-0.17, per-trade Sharpe 0.18-0.27
- 4H ATR: AUC 0.66-0.68, cost_R 0.07-0.09, per-trade Sharpe 0.25-0.33
- Best: 4H ATR short, per-trade Sharpe 0.3338 (6x viability threshold)
- 5m ATR baseline: AUC 0.7993, cost_R ~1.18 (UNVIABLE -- confirms the problem)
- Script: scripts/existential_experiment.py
- Results: outputs/EXISTENTIAL_RESULTS.md

---

## FVG Edge Validation (2026-03-05)

FVGs have WEAK/NONE standalone edge. Valuable only as conditioning
features in combination with other structural conditions.
- Bullish 5m FVGs: +2.9 bps at 4h (t=6.33, p<0.001) -- statistically significant but tiny
- Bearish FVGs: NO directional edge at any timeframe
- Win rates cluster at 47-50% (near random)
- Script: scripts/fvg_edge_validation.py
- Results: outputs/FVG_EDGE_VALIDATION.md

---

## Feature Families in D55b (Prism porting reference)

| Family | Features | SHAP % |
|--------|----------|--------|
| Swing Points | 19 | 33.0% |
| Order Blocks | 11 | 20.5% |
| Liquidity | 4 | 12.0% |
| Non-ICT (TA/micro) | 9 | 11.1% |
| FVG | 7 | 8.5% |
| Premium/Discount | 7 | 7.7% |
| OTE/Fibonacci | 2 | 4.3% |
| ATR Ratio | 2 | 1.4% |
| BOS | 2 | 1.0% |
| Calendar | 1 | 0.6% |

Full inventory: trading-brain/reviews/ICT_COVERAGE_AUDIT_2026-03-05.md

---

## Key Files for Porting

| File | Purpose |
|------|---------|
| core/signals/ict/rules.py | Swing detection, OB, FVG, BOS (18 functions) |
| core/engine/simulator.py | augment_features() for on-the-fly features |
| data/labeled/feature_catalog_v3.yaml | 371 feature definitions |
| outputs/FOUNDATION_DESIGN.md | DEPRECATED (see trading-brain/SYSTEM_DESIGN.md) |

---

## Dataset

- 648,288 rows x 594 columns (5m bars, 2020-01 to 2026-02)
- Primary: data/labeled/BTCUSDT_5m_labeled_v3.parquet (676 MB)
- Holdout: data/holdout/BTCUSDT_5m_holdout_v3.parquet (105,121 rows)
- Feature catalog: data/labeled/feature_catalog_v3.yaml

---

## Environment

- Python 3.14, Windows 11, cp1252 encoding
- Never use Unicode box-drawing, em-dashes, or arrows in print statements
- Data files are local only (gitignored), ~4.5 GB total
