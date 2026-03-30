# Technical Findings — Cycle 2: Electricity Data Pipeline

## Summary

Implemented `ElectricityDataModule` in `src/data.py` that fetches hourly OHLCV data from the ARF Data API (18 US equity tickers) and preprocesses it into TFT-compatible format. The pipeline produces batches of `(past_inputs, known_future_inputs, static_inputs, targets)` matching the paper's input specification.

## Implementation Details

### Data Source
- **Source**: ARF Data API (`https://ai.1s.xyz/api/data/ohlcv`)
- **Entities**: 18 US equity tickers (AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, JPM, V, XOM, UNH, PG, HD, MA, DIS, NFLX, INTC, AMD)
- **Frequency**: 1-hour intervals, ~2 years of data per ticker (~3,474 rows each)
- **UCI Electricity download failed** (corrupted zip file); ARF OHLCV data used as proxy per escape rules

### Pipeline Architecture

1. **Data Fetching**: `fetch_ticker_data()` downloads from API with local CSV caching
2. **Temporal Features**: `build_temporal_features()` generates normalized calendar features (hour_of_day, day_of_week, month, day_of_month)
3. **Entity Scaling**: `EntityScaler` fits per-entity StandardScaler on train split only
4. **Window Creation**: Sliding windows of (lookback=168, horizon=24) per entity
5. **Chronological Split**: 70% train / 15% val / 15% test per entity

### Output Tensor Shapes
| Tensor | Shape | Description |
|--------|-------|-------------|
| `past_inputs` | (B, 168, 6) | 2 observed (close, log_volume) + 4 temporal |
| `known_future_inputs` | (B, 24, 4) | 4 temporal features (calendar, known in advance) |
| `static_inputs` | (B, 1) | Entity ID (integer for embedding lookup) |
| `targets` | (B, 24) | Scaled close price over forecast horizon |

### Dataset Sizes (18 entities)
- Train: 40,320 samples
- Validation: 5,940 samples
- Test: 5,950 samples

### Paper Alignment
- **Lookback = 168** (7 days × 24h): matches paper
- **Horizon = 24** (24 hours): matches paper
- **Per-entity normalization**: matches paper
- **Temporal features**: hour, day_of_week, month, day_of_month (paper also uses holiday flag, omitted)
- **Static covariate**: entity ID for embedding (paper uses customer ID)

### Leakage Prevention
- Scaler fit on train data only
- No centered rolling windows used
- Chronological split with no overlap
- All features at time t use only data from t-1 or earlier (except calendar features which are known in advance)

## Tests
14/14 tests passed:
- Batch shape verification (4 tests)
- NaN checks on all splits (3 tests)
- Value range checks for temporal features (2 tests)
- Split size validation (2 tests)
- EntityScaler unit tests (2 tests)
- Temporal feature generation (1 test)

## Limitations & Open Questions
- UCI Electricity dataset could not be downloaded (zip file corruption); ARF OHLCV data used as proxy
- 18 entities vs. paper's 370 customers — smaller scale but sufficient for pipeline validation
- Holiday feature not implemented (paper includes it)
- Financial OHLCV data has different statistical properties than electricity consumption data
