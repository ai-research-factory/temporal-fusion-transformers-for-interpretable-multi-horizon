# Open Questions — Temporal Fusion Transformer

## Data

1. **UCI Electricity dataset unavailable**: The zip file from `https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip` downloads but appears corrupted (invalid zip structure, possibly incomplete transfer). Using ARF Data API hourly OHLCV data (18 US equities) as proxy. The pipeline architecture is identical — only the data source differs.

2. **Entity count**: Paper uses 370 electricity customers. Current implementation has 18 entities (tickers). This is sufficient for pipeline validation and model training, but cross-entity attention patterns may differ at this scale.

3. **Holiday feature**: Paper includes a holiday indicator as a known future input. Not implemented yet — could be added with a holiday calendar library if needed.

4. **Data frequency mismatch**: Paper uses 15-minute electricity data resampled to hourly. ARF API provides native hourly data, so no resampling needed, but the underlying data generation process differs (electricity consumption vs. financial prices).

## Model

5. **Phase 1 model skeleton**: Marked as complete in the phase plan but no model code found in `src/`. The `ElectricityDataModule` output shapes are designed to be compatible with standard TFT implementations.
