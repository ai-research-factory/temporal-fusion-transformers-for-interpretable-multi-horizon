# Open Questions — Temporal Fusion Transformer

## Data

1. **UCI Electricity dataset integrated**: The dataset (`data/LD2011_2014.txt`, 711 MB) is now loaded and processed. All 370 entities are active with non-zero readings.

2. **Entity count**: Paper uses 370 electricity customers. Implementation now matches with all 370 entities from UCI data.

3. **Holiday feature**: Implemented using the `holidays` Python library with Portuguese (PT) calendar, matching the paper's known-future holiday flag.

4. **Data frequency**: Paper uses 15-minute electricity data resampled to hourly. Implementation resamples using mean aggregation, matching the paper's protocol.

## Model

5. **Phase 1 model skeleton**: Marked as complete in the phase plan but no model code found in `src/`. The `ElectricityDataModule` output shapes are designed to be compatible with standard TFT implementations.

## Performance

6. **Full dataset memory**: Processing all 370 entities with sliding windows generates ~9M training samples. The `max_entities` parameter allows limiting entity count for faster iteration during development.
