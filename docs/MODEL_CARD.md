# Model Card - Used Car Price Predictor

## Overview

| Field | Value |
| --- | --- |
| Model | Stacked Ensemble |
| Task | Regression on selling price, in rupees |
| Target transform | log1p on fit, expm1 on predict |
| Training records | 2,861 |
| Test records | 716 |
| Reference year | 2020 |
| Random seed | 42 |
| Created | 2026-09-11T21:17:41 |

## Measured performance (held-out test set)

| metric | value |
| --- | --- |
| R2 | 0.7229 |
| MAE | 99279.6204 |
| RMSE | 298742.5185 |
| MAPE | 20.8362 |
| R2_log | 0.8881 |
| R2_excl_top1pct | 0.8590 |

## Intended use

- Indicative resale estimates for Indian used cars of the kind listed on CarDekho between 1992 and 2020.
- Coursework demonstration of a regression workflow.

## Out of scope

- Any binding valuation, insurance settlement or loan decision.
- Vehicles above roughly Rs 15 lakh, where the training data is thin and the error is largest.
- Markets outside India, and model years after 2020.

## Known limitations

- The dataset carries no condition, service history, accident record or city, all of which move real prices.
- Electric (1 record) and LPG vehicles are severely under-represented, so predictions for them are unreliable.
- Prices are historical; the model does not track market drift and will need retraining on newer listings.
- Measured R2 of 0.7229 is below the documented 0.90 objective; see section 14.6 of the final report for why.
