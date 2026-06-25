# MBS Prepayment Prediction with Machine Learning

A side project exploring how an XGBoost model trained on Freddie Mac loan-level data compares to the industry-standard 100 PSA prepayment curve. The goal isn't to replace PSA — it's to understand where ML adds value, where it breaks, and what the right features look like for mortgage-prepayment modeling.

## What this project actually does

Predicts the probability that a US 30-year fixed-rate mortgage will prepay (in full) within the next 12 months, using:
- Loan-level attributes (FICO, LTV, original rate, balance, geography)
- Time-varying signals (loan age, current market rate, rate incentive, burnout)
- Housing market dynamics (HPI growth since origination, current LTV)

All sourced from public datasets: Freddie Mac Single-Family Loan-Level Dataset, FRED MORTGAGE30US, and Freddie Mac FMHPI.

## Why prepayment matters

A US mortgage is a callable bond from the investor's perspective. The borrower has a free option to refinance when rates drop, repaying the loan early. This makes MBS pools harder to price and hedge than non-callable bonds — and makes prepayment forecasting the central problem in MBS modeling. Banks, agencies, and FHLBs maintain dedicated prepayment models. Standard industry baseline is the 1985-era 100 PSA curve, which models prepayment purely as a function of loan age — explicitly ignoring rate incentive, borrower quality, and housing market conditions.

## Approach

I followed an honest temporal evaluation protocol: train on earlier data, test on later data with no leakage from the future. Two evaluation periods were chosen deliberately:

1. **2021 (peak refi-wave era):** trained on 2019-2020 originations, tested on 2021 observations. This is the rate-driven regime where ML should shine.
2. **2023-2024 (rate-spike era):** trained on 2019-2022, tested on 2023-2024. This tests whether the model generalizes when prepayment becomes turnover-driven rather than rate-driven.

The target is a binary indicator: did this loan prepay in the next 12 months? For each loan-month observation, I computed the indicator from the actual zero-balance-code field in the Freddie Mac servicing data, taking care to exclude post-prepayment rows and observations where the 12-month forward window wasn't yet observable.

## Features

- **Loan attributes:** credit_score, original_interest_rate, original_upb, ltv, dti
- **Time-varying:** loan_age, market_rate (FRED MORTGAGE30US), rate_incentive = original_rate − market_rate
- **Engineered:** burnout (max rate_incentive the loan has historically experienced), month (seasonality), hpi_growth_since_origination, current_ltv

## Models

- **PSA baseline:** standard 100 PSA curve (linear ramp from 0% to 6% CPR over 30 months, then flat)
- **XGBoost classifier:** 200 trees, max_depth 6, learning_rate 0.1, evaluated via AUC and RMSE

## What I found

The XGBoost model beats PSA on out-of-sample test data, with the largest improvement during the refi-wave regime. PSA's AUC drops below 0.5 (anti-predictive) in the refi wave because it has no awareness of rate incentive. XGBoost's probabilities are reasonably calibrated — a "30% predicted" group has roughly 30% actual prepay rate.

In the rate-spike regime (2023-2024), both models struggle. Prepayment in this era is dominated by housing turnover and cash-out activity rather than rate-driven refi. State-level HPI features helped modestly but not dramatically — production prepayment models address this by using mode-of-prepayment frameworks with separate sub-models for refi, turnover, and default.

## Limitations and future work

This is a learning project with public data, so a few honest caveats:

**1. Coarse housing-price data.** I used state-level home price indices, but homes don't all appreciate at the state average — a downtown Chicago condo grows differently from a rural Illinois property, but both share the same "Illinois" HPI here. Production models at banks use MSA or zip-level HPI for finer geographic granularity.

**2. Single model for all prepay types.** Prepayment happens for very different reasons — rate refis, cash-out HELOCs, home sales, distressed exits. Each behaves differently. Production models typically train separate sub-models per prepay mode; my single-model approach loses signal when turnover and cash-out dominate (as in 2023-2024).

**3. Only one origination cohort used.** I loaded the 2019 sample (50,000 loans). Adding the 2020 and 2021 cohorts (already downloaded) would give the model more diverse rate environments to learn from.

**4. Binary target instead of hazard model.** "Will it prepay in next 12 months?" is simpler than a proper survival model that captures the timing of prepayment month-by-month. A hazard model would also handle still-alive loans (right-censored data) more rigorously.

## Files

- `mbs_prepayment_model` — full exploration: data loading, feature engineering, train/test split, model training, evaluation
- `data/` — Freddie Mac sample data and FMHPI (not committed; download instructions below)

## Reproducing

1. Register at freddiemac.com/research/datasets and download `sample_2019.zip`. Extract to `data/sample_2019/`
2. Download FMHPI Master File CSV from freddiemac.com/research/indices/house-price-index.html. Save to `data/fmhpi_master_file.csv`
3. Install: `pip install pandas numpy scikit-learn xgboost matplotlib jupyter`
4. Open and run `mbs_prepayment_model.ipynb`

## Author

Built as part of a self-directed quant skill-building project. Author: Bhavin Dixit ([github.com/dixitbhavin](https://github.com/dixitbhavin))
