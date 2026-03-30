# Technical Findings — Cycle 2: Electricity Data Pipeline

## Summary

Integrated the UCI Electricity Load Diagrams 2011-2014 dataset (370 entities) into the TFT data pipeline, replacing the previous ARF OHLCV proxy. Added Portuguese holiday flags as a known-future feature, matching the paper's specification.

## Review Feedback Addressed

1. **UCI Electricity dataset integration** — Implemented `load_electricity_data()` that parses the semicolon-separated, comma-decimal UCI file, resamples from 15-min to hourly, and produces long-format data for all 370 customer entities.
2. **Holiday flag feature** — Added `build_holiday_feature()` using the `holidays` library with Portuguese (PT) calendar. Holiday flag is included as the 5th column in `known_future_inputs`.
3. **Tests for 370 entities and holiday feature** — Added `TestElectricityDataShape` (3 tests: entity count=370, power_usage column, hourly frequency) and `TestHolidayFeature` (4 tests: presence, binary values, both values exist, specific PT holiday verification).

## Implementation Details

### Data Source
- **Source**: UCI Electricity Load Diagrams 2011-2014
- **File**: `data/LD2011_2014.txt` (711 MB, 140,256 rows x 370 columns)
- **Format**: Semicolon-separated, comma-decimal, 15-minute intervals
- **Entities**: 370 electricity customers (MT_001 to MT_370)
- **Period**: 2011-01-01 to 2015-01-01

### Pipeline Architecture

1. **Data Loading**: `load_electricity_data()` parses UCI format, resamples to 1h, melts to long format
2. **Temporal Features**: `build_temporal_features()` generates normalized calendar features (hour_of_day, day_of_week, month, day_of_month)
3. **Holiday Feature**: `build_holiday_feature()` generates binary Portuguese holiday flags
4. **Entity Scaling**: `EntityScaler` fits per-entity StandardScaler on train split only
5. **Window Creation**: Sliding windows of (lookback=168, horizon=24) per entity
6. **Chronological Split**: 70% train / 15% val / 15% test per entity

### Output Tensor Shapes
| Tensor | Shape | Description |
|--------|-------|-------------|
| `past_inputs` | (B, 168, 5) | 1 observed (power_usage) + 4 temporal |
| `known_future_inputs` | (B, 24, 5) | 4 temporal + 1 holiday flag |
| `static_inputs` | (B, 1) | Entity ID (integer for embedding lookup) |
| `targets` | (B, 24) | Scaled power_usage over forecast horizon |

### Dataset Sizes (per entity)
- Train: ~24,354 windows per entity
- Validation: ~5,069 windows per entity
- Test: ~5,069 windows per entity
- Full 370-entity dataset: ~9.01M train / ~1.88M val / ~1.88M test samples

### Paper Alignment
- **370 entities**: matches paper (previously 18 OHLCV tickers)
- **Lookback = 168** (7 days x 24h): matches paper
- **Horizon = 24** (24 hours): matches paper
- **Per-entity normalization**: matches paper
- **Temporal features**: hour, day_of_week, month, day_of_month: matches paper
- **Holiday flag**: Portuguese holidays included: matches paper
- **Static covariate**: entity ID for embedding: matches paper
- **Data frequency**: 15-min resampled to hourly: matches paper

### Leakage Prevention
- Scaler fit on train data only
- No centered rolling windows used
- Chronological split with no overlap
- Calendar features and holidays are deterministic (known in advance by definition)

## Tests
22/22 tests passed:
- UCI Electricity data verification (3 tests: 370 entities, power_usage column, hourly frequency)
- Batch shape verification (4 tests)
- Holiday feature validation (4 tests: presence, binary, both values, PT holidays)
- NaN integrity checks on all splits (3 tests)
- Feature range validation (2 tests)
- Split size validation (3 tests)
- EntityScaler unit tests (2 tests)
- Temporal feature generation (1 test)

## Changes from Previous Iteration
- Replaced ARF OHLCV proxy data with UCI Electricity dataset (370 entities)
- Added `load_electricity_data()` function for UCI data parsing
- Added `build_holiday_feature()` with Portuguese holiday calendar
- `known_future_inputs` now has 5 features (was 4): added holiday flag
- `past_inputs` has 5 features for UCI data (1 observed + 4 temporal, was 6 with 2 observed)
- ARF OHLCV mode preserved as fallback via `use_uci=False`
- Added `max_entities` parameter for faster iteration during development
