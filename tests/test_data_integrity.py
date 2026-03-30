"""
Data integrity and leakage tests for the TFT data pipeline.
"""

import numpy as np
import pytest
from src.data import (
    ElectricityDataModule,
    EntityScaler,
    build_temporal_features,
)
import pandas as pd


# Use a small subset for fast testing
TEST_TICKERS = ["AAPL", "MSFT"]


@pytest.fixture(scope="module")
def data_module():
    dm = ElectricityDataModule(
        tickers=TEST_TICKERS,
        lookback=168,
        horizon=24,
        batch_size=32,
    )
    dm.setup()
    return dm


class TestBatchShapes:
    def test_past_inputs_shape(self, data_module):
        batch = data_module.get_sample_batch()
        past, future, static, targets = batch
        assert past.ndim == 3
        assert past.shape[1] == 168  # lookback
        assert past.shape[2] == 6   # 2 observed + 4 temporal

    def test_future_inputs_shape(self, data_module):
        batch = data_module.get_sample_batch()
        _, future, _, _ = batch
        assert future.ndim == 3
        assert future.shape[1] == 24  # horizon
        assert future.shape[2] == 4   # 4 temporal features

    def test_static_inputs_shape(self, data_module):
        batch = data_module.get_sample_batch()
        _, _, static, _ = batch
        assert static.ndim == 2
        assert static.shape[1] == 1

    def test_targets_shape(self, data_module):
        batch = data_module.get_sample_batch()
        _, _, _, targets = batch
        assert targets.ndim == 2
        assert targets.shape[1] == 24  # horizon


class TestDataIntegrity:
    def test_no_nan_in_train(self, data_module):
        ds = data_module.train_dataset
        assert not ds.past_inputs.isnan().any()
        assert not ds.known_future_inputs.isnan().any()
        assert not ds.targets.isnan().any()

    def test_no_nan_in_val(self, data_module):
        ds = data_module.val_dataset
        assert not ds.past_inputs.isnan().any()
        assert not ds.known_future_inputs.isnan().any()
        assert not ds.targets.isnan().any()

    def test_no_nan_in_test(self, data_module):
        ds = data_module.test_dataset
        assert not ds.past_inputs.isnan().any()
        assert not ds.known_future_inputs.isnan().any()
        assert not ds.targets.isnan().any()

    def test_temporal_features_in_range(self, data_module):
        """Temporal features should be normalized to [0, 1]."""
        ds = data_module.train_dataset
        # Temporal features are last 4 columns of past_inputs
        temporal = ds.past_inputs[:, :, 2:]
        assert temporal.min() >= 0.0 - 1e-6
        assert temporal.max() <= 1.0 + 1e-6

    def test_known_future_in_range(self, data_module):
        """Known future inputs should be normalized to [0, 1]."""
        ds = data_module.train_dataset
        assert ds.known_future_inputs.min() >= 0.0 - 1e-6
        assert ds.known_future_inputs.max() <= 1.0 + 1e-6


class TestNoLeakage:
    def test_split_sizes_nonzero(self, data_module):
        assert len(data_module.train_dataset) > 0
        assert len(data_module.val_dataset) > 0
        assert len(data_module.test_dataset) > 0

    def test_train_larger_than_val_and_test(self, data_module):
        assert len(data_module.train_dataset) > len(data_module.val_dataset)
        assert len(data_module.train_dataset) > len(data_module.test_dataset)


class TestEntityScaler:
    def test_scaler_fit_transform(self):
        scaler = EntityScaler()
        ids = np.array([0, 0, 0, 1, 1, 1])
        values = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0],
                           [10.0, 20.0], [20.0, 40.0], [30.0, 60.0]])
        scaler.fit(ids, values)
        transformed = scaler.transform(ids, values)

        # Each entity should be roughly zero-mean
        for eid in [0, 1]:
            mask = ids == eid
            entity_mean = transformed[mask].mean(axis=0)
            np.testing.assert_allclose(entity_mean, 0.0, atol=1e-5)

    def test_scaler_inverse(self):
        scaler = EntityScaler()
        ids = np.array([0, 0, 0])
        values = np.array([[10.0], [20.0], [30.0]])
        scaler.fit(ids, values)
        transformed = scaler.transform(ids, values)
        recovered = scaler.inverse_transform(0, transformed)
        np.testing.assert_allclose(recovered, values, atol=1e-5)


class TestTemporalFeatures:
    def test_build_temporal_features(self):
        timestamps = pd.Series(pd.date_range("2024-01-01", periods=48, freq="h"))
        features = build_temporal_features(timestamps)
        assert len(features) == 48
        assert set(features.columns) == {"hour_of_day", "day_of_week", "month", "day_of_month"}
        assert features["hour_of_day"].min() >= 0
        assert features["hour_of_day"].max() <= 1
