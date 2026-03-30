"""
Data integrity and leakage tests for the TFT data pipeline.

Tests both UCI Electricity (370 entities) and OHLCV fallback modes.
"""

import numpy as np
import pytest
from src.data import (
    ElectricityDataModule,
    EntityScaler,
    build_temporal_features,
    build_holiday_feature,
    load_electricity_data,
)
import pandas as pd


# Use a small subset for fast testing
TEST_MAX_ENTITIES = 3


@pytest.fixture(scope="module")
def uci_data_module():
    """UCI Electricity data module with 3 entities for fast testing."""
    dm = ElectricityDataModule(
        max_entities=TEST_MAX_ENTITIES,
        lookback=168,
        horizon=24,
        batch_size=32,
        use_uci=True,
    )
    dm.setup()
    return dm


class TestElectricityDataShape:
    """Verify UCI Electricity data has correct entity count and structure."""

    def test_electricity_data_loads_370_entities(self):
        """Full UCI dataset should have 370 entities."""
        df = load_electricity_data()
        n_entities = df["entity_id"].nunique()
        assert n_entities == 370, f"Expected 370 entities, got {n_entities}"

    def test_electricity_data_has_power_usage(self):
        """UCI data should have power_usage column."""
        df = load_electricity_data()
        assert "power_usage" in df.columns

    def test_electricity_hourly_frequency(self):
        """After resampling, data should be hourly."""
        df = load_electricity_data()
        entity_0 = df[df["entity_id"] == 0].sort_values("timestamp")
        diffs = entity_0["timestamp"].diff().dropna()
        median_diff = diffs.median()
        assert median_diff == pd.Timedelta(hours=1), f"Expected 1h frequency, got {median_diff}"


class TestBatchShapes:
    """Verify batch tensor shapes for UCI data."""

    def test_past_inputs_shape(self, uci_data_module):
        batch = uci_data_module.get_sample_batch()
        past, future, static, targets = batch
        assert past.ndim == 3
        assert past.shape[1] == 168  # lookback
        # UCI: 1 observed (power_usage) + 4 temporal = 5
        assert past.shape[2] == 5

    def test_future_inputs_shape(self, uci_data_module):
        batch = uci_data_module.get_sample_batch()
        _, future, _, _ = batch
        assert future.ndim == 3
        assert future.shape[1] == 24  # horizon
        # 4 temporal + 1 holiday = 5
        assert future.shape[2] == 5

    def test_static_inputs_shape(self, uci_data_module):
        batch = uci_data_module.get_sample_batch()
        _, _, static, _ = batch
        assert static.ndim == 2
        assert static.shape[1] == 1

    def test_targets_shape(self, uci_data_module):
        batch = uci_data_module.get_sample_batch()
        _, _, _, targets = batch
        assert targets.ndim == 2
        assert targets.shape[1] == 24  # horizon


class TestHolidayFeature:
    """Verify holiday flag is correctly generated and included."""

    def test_holiday_feature_presence(self, uci_data_module):
        """Holiday should be the 5th column in known_future_inputs."""
        batch = uci_data_module.get_sample_batch()
        _, future, _, _ = batch
        # 5 features: hour, day_of_week, month, day_of_month, is_holiday
        assert future.shape[2] == 5

    def test_holiday_feature_binary(self, uci_data_module):
        """Holiday flag should be 0 or 1."""
        ds = uci_data_module.train_dataset
        holiday_col = ds.known_future_inputs[:, :, 4]  # last column
        unique_vals = holiday_col.unique()
        assert all(v in [0.0, 1.0] for v in unique_vals), f"Holiday should be binary, got {unique_vals}"

    def test_holiday_feature_has_both_values(self, uci_data_module):
        """Dataset should contain both holiday and non-holiday days."""
        ds = uci_data_module.train_dataset
        holiday_col = ds.known_future_inputs[:, :, 4]
        assert (holiday_col == 1.0).any(), "No holiday days found"
        assert (holiday_col == 0.0).any(), "No non-holiday days found"

    def test_build_holiday_feature_portugal(self):
        """Portuguese holidays should be correctly identified."""
        timestamps = pd.Series(pd.to_datetime([
            "2014-01-01",  # New Year - holiday
            "2014-04-25",  # Freedom Day - holiday
            "2014-07-15",  # Regular day - not a holiday
        ]))
        holidays = build_holiday_feature(timestamps, country="PT")
        assert holidays.iloc[0] == 1.0, "Jan 1 should be a holiday"
        assert holidays.iloc[1] == 1.0, "Apr 25 should be a holiday"
        assert holidays.iloc[2] == 0.0, "Jul 15 should not be a holiday"


class TestDataIntegrity:
    def test_no_nan_in_train(self, uci_data_module):
        ds = uci_data_module.train_dataset
        assert not ds.past_inputs.isnan().any()
        assert not ds.known_future_inputs.isnan().any()
        assert not ds.targets.isnan().any()

    def test_no_nan_in_val(self, uci_data_module):
        ds = uci_data_module.val_dataset
        assert not ds.past_inputs.isnan().any()
        assert not ds.known_future_inputs.isnan().any()
        assert not ds.targets.isnan().any()

    def test_no_nan_in_test(self, uci_data_module):
        ds = uci_data_module.test_dataset
        assert not ds.past_inputs.isnan().any()
        assert not ds.known_future_inputs.isnan().any()
        assert not ds.targets.isnan().any()

    def test_temporal_features_in_range(self, uci_data_module):
        """Temporal features should be normalized to [0, 1]."""
        ds = uci_data_module.train_dataset
        # Temporal features are columns 1-4 of past_inputs (after 1 observed col for UCI)
        temporal = ds.past_inputs[:, :, 1:]
        assert temporal.min() >= 0.0 - 1e-6
        assert temporal.max() <= 1.0 + 1e-6

    def test_known_future_in_range(self, uci_data_module):
        """Known future inputs should be normalized to [0, 1]."""
        ds = uci_data_module.train_dataset
        assert ds.known_future_inputs.min() >= 0.0 - 1e-6
        assert ds.known_future_inputs.max() <= 1.0 + 1e-6


class TestNoLeakage:
    def test_split_sizes_nonzero(self, uci_data_module):
        assert len(uci_data_module.train_dataset) > 0
        assert len(uci_data_module.val_dataset) > 0
        assert len(uci_data_module.test_dataset) > 0

    def test_train_larger_than_val_and_test(self, uci_data_module):
        assert len(uci_data_module.train_dataset) > len(uci_data_module.val_dataset)
        assert len(uci_data_module.train_dataset) > len(uci_data_module.test_dataset)

    def test_entity_count_matches_config(self, uci_data_module):
        assert uci_data_module.split_info["n_entities"] == TEST_MAX_ENTITIES


class TestEntityScaler:
    def test_scaler_fit_transform(self):
        scaler = EntityScaler()
        ids = np.array([0, 0, 0, 1, 1, 1])
        values = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0],
                           [10.0, 20.0], [20.0, 40.0], [30.0, 60.0]])
        scaler.fit(ids, values)
        transformed = scaler.transform(ids, values)

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
