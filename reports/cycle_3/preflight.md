# Preflight Check - Cycle 3

## 1. Data Boundary Table

| Item | Value |
|---|---|
| Data Acquisition End Date | 2015-01-01 (UCI dataset, well before today 2026-03-30) |
| Train Period | 2011-01-01 ~ 2013-10-19 |
| Validation Period | 2013-10-19 ~ 2014-05-26 |
| Test Period | 2014-05-26 ~ 2015-01-01 |
| No Overlap Confirmed | Yes |
| No Future Dates Confirmed | Yes |

## 2. Feature Timestamp Contract

- All features use data at t-1 or earlier for prediction at time t? -> Yes (sliding window: past_inputs from [t-168, t), targets from [t, t+24))
- Scaler/Imputer fit on train data only? -> Yes (EntityScaler.fit() called on train split only)
- No centered rolling windows used? -> Yes (no rolling windows used; features are point-in-time calendar features)

## 3. Paper Spec Diff Table

| Parameter | Paper Value | Current Implementation | Match? |
|---|---|---|---|
| Universe | 370 electricity customers | 370 entities from UCI dataset | Yes |
| Lookback Period | 168 hours (7 days) | 168 hours | Yes |
| Forecast Horizon | 24 hours | 24 hours | Yes |
| Features (observed) | Power consumption | power_usage (1 feature) | Yes |
| Features (known future) | Hour, day_of_week, holiday | hour_of_day, day_of_week, month, day_of_month, is_holiday (5 features) | Yes (superset) |
| Features (static) | Entity ID | entity_id (1 feature for embedding) | Yes |
| Normalization | Per-entity standardization | EntityScaler (per-entity StandardScaler, fit on train) | Yes |
| Data Frequency | Hourly (resampled from 15-min) | Hourly (mean resampling) | Yes |
| Loss Function | Quantile Loss (P10, P50, P90) | Quantile Loss (P50, P90) - implementing this cycle | Partial |
| Optimizer | Adam with LR schedule | Adam with cosine annealing - implementing this cycle | Yes |
