# Preflight Check — Cycle 2 (Phase 2: Electricity Data Pipeline)

## 1. Data Boundary Table

| Item | Value |
|---|---|
| Data Source | UCI Electricity Load Diagrams 2011-2014 (370 entities) |
| Data Acquisition End Date | 2015-01-01 (historical, well before today 2026-03-30) |
| Train Period | 2011-01-01 ~ 2013-10-19 |
| Validation Period | 2013-10-19 ~ 2014-05-26 |
| Test Period | 2014-05-26 ~ 2015-01-01 |
| No Overlap Confirmed | Yes |
| No Future Dates Confirmed | Yes |

## 2. Feature Timestamp Contract

- All features at time t use only data from t-1 or earlier? → **Yes**
  - Lookback window uses past data only
  - Known future inputs (calendar features, holidays) are deterministic and known in advance
- Scaler / Imputer fit on train data only? → **Yes**
  - EntityScaler.fit() called only on train_df
- No centered rolling windows used? → **Yes** (not used)

## 3. Paper Spec Difference Table

| Parameter | Paper Value | Implementation | Match? |
|---|---|---|---|
| Dataset | UCI Electricity (370 customers, 15-min) | UCI Electricity (370 customers, 15-min resampled to 1h) | Yes |
| Lookback Period | 168 (7 days x 24h) | 168 | Yes |
| Forecast Horizon | 24 (24 hours) | 24 | Yes |
| Static Features | Customer ID | Entity ID (integer for embedding) | Yes |
| Temporal Features | hour, day_of_week, month, day_of_month, holiday | hour, day_of_week, month, day_of_month, holiday | Yes |
| Normalization | Per-entity standardization | Per-entity StandardScaler (train only) | Yes |
| Quantile Loss | P10, P50, P90 | P10, P50, P90 (to be implemented in Phase 3) | Yes |
