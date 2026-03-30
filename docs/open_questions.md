# Open Questions — Temporal Fusion Transformer

## Data

1. **UCI Electricity dataset integrated**: The dataset (`data/LD2011_2014.txt`, 711 MB) is now loaded and processed. All 370 entities are active with non-zero readings.

2. **Entity count**: Paper uses 370 electricity customers. Implementation now matches with all 370 entities from UCI data.

3. **Holiday feature**: Implemented using the `holidays` Python library with Portuguese (PT) calendar, matching the paper's known-future holiday flag.

4. **Data frequency**: Paper uses 15-minute electricity data resampled to hourly. Implementation resamples using mean aggregation, matching the paper's protocol.

## Model

5. **TFT architecture implemented**: Full TFT model in `src/models/tft.py` with all paper components (GRN, VSN, static encoders, LSTM, interpretable attention). 346K parameters with hidden_dim=64.

6. **Entity subset for training**: Current results use 10 of 370 entities due to CPU-only training environment. Architecture supports full 370 entities; scaling requires GPU resources.

7. **Window stride trade-off**: Using stride=24 (one window per day per entity) reduces dataset ~24x for feasible CPU training. Full stride=1 would provide denser temporal coverage but requires more compute.

## Performance

8. **Quantile Loss baseline**: Current P50 QL=0.153, P90 QL=0.088 on test set. These should improve with more entities and epochs. Paper reports significantly lower values on the full dataset.

9. **Early stopping behavior**: Model reached best validation loss at epoch 0, suggesting learning rate or architecture may benefit from tuning in future cycles.

## Environment

10. **No GPU available**: Training runs on CPU only, limiting feasible dataset size and training duration.
