"""
Electricity Data Pipeline for Temporal Fusion Transformer.

Loads the UCI Electricity Load Diagrams 2011-2014 dataset (370 entities),
preprocesses it into TFT-compatible format with:
  - past_inputs: observed values over the lookback window + temporal features
  - known_future_inputs: calendar features + holiday flag known in advance
  - static_inputs: entity (customer) embeddings
  - targets: values to predict over the forecast horizon

Falls back to ARF Data API (OHLCV tickers) if UCI data is unavailable.
"""

import os
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

# Paper-aligned defaults
DEFAULT_LOOKBACK = 168      # 7 days * 24 hours
DEFAULT_HORIZON = 24        # 24 hours ahead
DATA_DIR = Path("data")
UCI_FILENAME = "LD2011_2014.txt"

# ARF API fallback
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V",
    "XOM", "UNH", "PG", "HD", "MA", "DIS", "NFLX", "INTC", "AMD",
]
ARF_API_URL = "https://ai.1s.xyz/api/data/ohlcv"


def load_electricity_data(data_dir: Path = DATA_DIR,
                          resample_freq: str = "1h") -> pd.DataFrame:
    """Load and preprocess the UCI Electricity Load Diagrams dataset.

    The raw data is 15-minute interval electricity consumption for 370 customers.
    We resample to hourly and melt into long format (entity_id, timestamp, power_usage).

    Returns:
        DataFrame with columns: timestamp, entity_id, power_usage
    """
    filepath = data_dir / UCI_FILENAME
    if not filepath.exists():
        raise FileNotFoundError(
            f"UCI Electricity data not found at {filepath}. "
            "Download from https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014"
        )

    logger.info(f"Loading UCI Electricity data from {filepath}")
    df = pd.read_csv(
        filepath, sep=";", decimal=",",
        index_col=0, parse_dates=True,
    )

    # Resample from 15-min to hourly (mean aggregation, matching paper)
    logger.info(f"Resampling from 15-min to {resample_freq}")
    df = df.resample(resample_freq).mean()

    # Filter entities with sufficient non-zero data (paper uses all 370)
    nonzero_counts = (df > 0).sum()
    active_cols = nonzero_counts[nonzero_counts > 100].index.tolist()
    logger.info(f"Active entities: {len(active_cols)} / {len(df.columns)}")
    df = df[active_cols]

    # Melt to long format
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "timestamp"})
    melted = df.melt(id_vars=["timestamp"], var_name="customer_id", value_name="power_usage")

    # Create integer entity_id from customer_id
    customers = sorted(melted["customer_id"].unique())
    customer_to_id = {c: i for i, c in enumerate(customers)}
    melted["entity_id"] = melted["customer_id"].map(customer_to_id)

    melted = melted.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)
    logger.info(
        f"Loaded {len(customers)} entities, {len(melted)} total rows, "
        f"date range: {melted['timestamp'].min()} to {melted['timestamp'].max()}"
    )
    return melted


def fetch_ticker_data(ticker: str, interval: str = "1h", period: str = "2y",
                      cache_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Fetch OHLCV data from ARF Data API with local caching (fallback mode)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{ticker.lower()}_{interval}.csv"

    if cache_file.exists():
        logger.info(f"Loading cached data for {ticker} from {cache_file}")
        df = pd.read_csv(cache_file)
    else:
        url = f"{ARF_API_URL}?ticker={ticker}&interval={interval}&period={period}"
        logger.info(f"Fetching {ticker} from {url}")
        df = pd.read_csv(url)
        df.to_csv(cache_file, index=False)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def build_temporal_features(timestamps: pd.Series) -> pd.DataFrame:
    """Generate calendar-based temporal features (known in future)."""
    ts = pd.to_datetime(timestamps)
    return pd.DataFrame({
        "hour_of_day": ts.dt.hour / 23.0,           # normalized [0, 1]
        "day_of_week": ts.dt.dayofweek / 6.0,       # normalized [0, 1]
        "month": (ts.dt.month - 1) / 11.0,          # normalized [0, 1]
        "day_of_month": (ts.dt.day - 1) / 30.0,     # normalized [0, 1]
    }, index=timestamps.index)


def build_holiday_feature(timestamps: pd.Series, country: str = "PT") -> pd.Series:
    """Generate a binary holiday flag for each timestamp.

    The UCI Electricity dataset is from Portugal, so we use PT holidays by default.
    This is a known-future feature (holidays are deterministic).

    Args:
        timestamps: Series of datetime timestamps
        country: ISO country code for holiday calendar (default: PT for Portugal)

    Returns:
        Series of float32 holiday flags (0.0 or 1.0)
    """
    import holidays as holidays_lib
    ts = pd.to_datetime(timestamps)
    years = range(ts.dt.year.min(), ts.dt.year.max() + 1)
    holiday_dates = holidays_lib.country_holidays(country, years=years)
    is_holiday = ts.dt.date.isin(holiday_dates).astype(np.float32)
    return pd.Series(is_holiday, index=timestamps.index, name="is_holiday")


class EntityScaler:
    """Per-entity StandardScaler. Fits on train data only."""

    def __init__(self):
        self.means: dict[int, np.ndarray] = {}
        self.stds: dict[int, np.ndarray] = {}

    def fit(self, entity_ids: np.ndarray, values: np.ndarray):
        """Fit scaler per entity. values shape: (n_samples, n_features)."""
        for eid in np.unique(entity_ids):
            mask = entity_ids == eid
            self.means[eid] = values[mask].mean(axis=0)
            self.stds[eid] = values[mask].std(axis=0)
            self.stds[eid][self.stds[eid] < 1e-8] = 1.0
        return self

    def transform(self, entity_ids: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Transform values using fitted parameters."""
        result = np.empty_like(values, dtype=np.float32)
        for eid in np.unique(entity_ids):
            mask = entity_ids == eid
            if eid in self.means:
                result[mask] = (values[mask] - self.means[eid]) / self.stds[eid]
            else:
                logger.warning(f"Entity {eid} not seen during fit, using global stats")
                result[mask] = (values[mask] - values[mask].mean(axis=0)) / (values[mask].std(axis=0) + 1e-8)
        return result

    def inverse_transform(self, entity_id: int, values: np.ndarray) -> np.ndarray:
        """Inverse transform for a single entity."""
        return values * self.stds[entity_id] + self.means[entity_id]


class TFTDataset(Dataset):
    """PyTorch Dataset producing TFT-compatible samples.

    Each sample is a tuple of:
        past_inputs:         (lookback, n_past_features)    - observed values + temporal features
        known_future_inputs: (horizon, n_known_features)    - calendar features + holiday for forecast
        static_inputs:       (n_static,)                    - entity embedding index
        targets:             (horizon,)                     - values to predict
    """

    def __init__(self, past_inputs: np.ndarray, known_future_inputs: np.ndarray,
                 static_inputs: np.ndarray, targets: np.ndarray):
        self.past_inputs = torch.tensor(past_inputs, dtype=torch.float32)
        self.known_future_inputs = torch.tensor(known_future_inputs, dtype=torch.float32)
        self.static_inputs = torch.tensor(static_inputs, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (
            self.past_inputs[idx],
            self.known_future_inputs[idx],
            self.static_inputs[idx],
            self.targets[idx],
        )


class ElectricityDataModule:
    """Data module that loads UCI Electricity data and serves TFT-compatible batches.

    Follows the TFT paper's Electricity dataset protocol:
    - 370 customer entities from UCI Electricity Load Diagrams 2011-2014
    - Hourly data (resampled from 15-min) with temporal + holiday features
    - Per-entity normalization (fit on train only)
    - Chronological train/val/test split

    Falls back to ARF Data API (OHLCV tickers) if UCI data is unavailable.
    """

    # Number of known future features: hour, day_of_week, month, day_of_month, is_holiday
    N_KNOWN_FUTURE = 5
    # Number of observed features in past: 1 (power_usage) for UCI, 2 for OHLCV
    N_OBSERVED_UCI = 1
    N_OBSERVED_OHLCV = 2

    def __init__(
        self,
        lookback: int = DEFAULT_LOOKBACK,
        horizon: int = DEFAULT_HORIZON,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        batch_size: int = 64,
        data_dir: Path = DATA_DIR,
        use_uci: Optional[bool] = None,
        tickers: Optional[list[str]] = None,
        max_entities: Optional[int] = None,
        holiday_country: str = "PT",
        stride: int = 1,
    ):
        self.lookback = lookback
        self.horizon = horizon
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.batch_size = batch_size
        self.data_dir = Path(data_dir)
        self.tickers = tickers or DEFAULT_TICKERS
        self.max_entities = max_entities
        self.holiday_country = holiday_country
        self.stride = stride
        self.scaler = EntityScaler()
        self.entity_map: dict = {}
        self._prepared = False

        # Auto-detect UCI data availability
        if use_uci is None:
            self.use_uci = (self.data_dir / UCI_FILENAME).exists()
        else:
            self.use_uci = use_uci

    def prepare_data(self) -> pd.DataFrame:
        """Load data from UCI Electricity dataset or ARF API fallback."""
        if self.use_uci:
            return self._prepare_uci_data()
        else:
            return self._prepare_ohlcv_data()

    def _prepare_uci_data(self) -> pd.DataFrame:
        """Load UCI Electricity dataset (370 entities)."""
        df = load_electricity_data(self.data_dir)

        # Optionally limit entities for faster iteration
        if self.max_entities is not None:
            unique_entities = sorted(df["entity_id"].unique())
            keep = unique_entities[:self.max_entities]
            df = df[df["entity_id"].isin(keep)].reset_index(drop=True)
            # Re-map entity IDs to be contiguous
            eid_map = {old: new for new, old in enumerate(sorted(df["entity_id"].unique()))}
            df["entity_id"] = df["entity_id"].map(eid_map)

        self.entity_map = {
            f"MT_{eid+1:03d}": eid
            for eid in df["entity_id"].unique()
        }
        self.raw_data = df
        return df

    def _prepare_ohlcv_data(self) -> pd.DataFrame:
        """Fallback: load OHLCV data from ARF API."""
        frames = []
        for i, ticker in enumerate(self.tickers):
            self.entity_map[ticker] = i
            df = fetch_ticker_data(ticker, cache_dir=self.data_dir)
            df["entity_id"] = i
            df["ticker"] = ticker
            frames.append(df)
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)
        self.raw_data = combined
        return combined

    def setup(self):
        """Full pipeline: fetch data, build features, split, scale, create windows."""
        if self._prepared:
            return

        # Step 1: Load data
        df = self.prepare_data()

        # Step 2: Build temporal features
        temporal = build_temporal_features(df["timestamp"])
        df = pd.concat([df, temporal], axis=1)

        # Step 3: Build holiday feature
        holiday = build_holiday_feature(df["timestamp"], country=self.holiday_country)
        df["is_holiday"] = holiday.values

        # Step 4: Determine observed columns based on data source
        if self.use_uci:
            observed_cols = ["power_usage"]
        else:
            df["log_volume"] = np.log1p(df["volume"].values)
            observed_cols = ["close", "log_volume"]

        self._observed_cols = observed_cols
        temporal_cols = ["hour_of_day", "day_of_week", "month", "day_of_month"]
        known_future_cols = temporal_cols + ["is_holiday"]
        self._temporal_cols = temporal_cols
        self._known_future_cols = known_future_cols

        # Step 5: Chronological split per entity
        train_dfs, val_dfs, test_dfs = [], [], []
        for eid in sorted(df["entity_id"].unique()):
            edf = df[df["entity_id"] == eid].sort_values("timestamp").reset_index(drop=True)
            n = len(edf)
            train_end = int(n * self.train_frac)
            val_end = int(n * (self.train_frac + self.val_frac))
            train_dfs.append(edf.iloc[:train_end])
            val_dfs.append(edf.iloc[train_end:val_end])
            test_dfs.append(edf.iloc[val_end:])

        train_df = pd.concat(train_dfs, ignore_index=True)
        val_df = pd.concat(val_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)

        # Store date boundaries for preflight
        self.date_boundaries = {
            "train_start": str(train_df["timestamp"].min()),
            "train_end": str(train_df["timestamp"].max()),
            "val_start": str(val_df["timestamp"].min()),
            "val_end": str(val_df["timestamp"].max()),
            "test_start": str(test_df["timestamp"].min()),
            "test_end": str(test_df["timestamp"].max()),
        }

        # Step 6: Fit scaler on train only
        self.scaler.fit(
            train_df["entity_id"].values,
            train_df[observed_cols].values,
        )

        # Step 7: Create windowed samples for each split
        self.train_dataset = self._create_dataset(train_df, observed_cols, known_future_cols)
        self.val_dataset = self._create_dataset(val_df, observed_cols, known_future_cols)
        self.test_dataset = self._create_dataset(test_df, observed_cols, known_future_cols)

        # Store split info for reporting
        self.split_info = {
            "train_size": len(self.train_dataset),
            "val_size": len(self.val_dataset),
            "test_size": len(self.test_dataset),
            "n_entities": len(df["entity_id"].unique()),
            "lookback": self.lookback,
            "horizon": self.horizon,
            "n_past_features": len(observed_cols) + len(temporal_cols),
            "n_known_future_features": len(known_future_cols),
            "data_source": "UCI Electricity" if self.use_uci else "ARF Data API",
        }

        self._prepared = True
        logger.info(
            f"Data prepared ({self.split_info['data_source']}): "
            f"train={len(self.train_dataset)}, "
            f"val={len(self.val_dataset)}, test={len(self.test_dataset)}, "
            f"entities={self.split_info['n_entities']}"
        )

    def _create_dataset(self, df: pd.DataFrame, observed_cols: list[str],
                        known_future_cols: list[str]) -> TFTDataset:
        """Create windowed TFT dataset from a split DataFrame."""
        temporal_cols = self._temporal_cols
        all_past = []
        all_future = []
        all_static = []
        all_targets = []

        for eid in sorted(df["entity_id"].unique()):
            edf = df[df["entity_id"] == eid].sort_values("timestamp").reset_index(drop=True)
            n = len(edf)

            if n < self.lookback + self.horizon:
                logger.warning(
                    f"Entity {eid} has only {n} samples, need {self.lookback + self.horizon}. Skipping."
                )
                continue

            # Scale observed values
            obs_values = self.scaler.transform(
                np.full(n, eid), edf[observed_cols].values
            )
            temporal_values = edf[temporal_cols].values.astype(np.float32)
            known_future_values = edf[known_future_cols].values.astype(np.float32)

            # Create sliding windows
            for start in range(0, n - self.lookback - self.horizon + 1, self.stride):
                lb_end = start + self.lookback
                hz_end = lb_end + self.horizon

                # Past: scaled observed + temporal features over lookback
                past_obs = obs_values[start:lb_end]
                past_temp = temporal_values[start:lb_end]
                past = np.concatenate([past_obs, past_temp], axis=1)

                # Known future: temporal + holiday features over horizon
                future = known_future_values[lb_end:hz_end]

                # Static: entity id
                static = np.array([eid])

                # Target: scaled target over horizon (first observed col)
                target = obs_values[lb_end:hz_end, 0]

                all_past.append(past)
                all_future.append(future)
                all_static.append(static)
                all_targets.append(target)

        if not all_past:
            n_past = len(observed_cols) + len(temporal_cols)
            n_future = len(known_future_cols)
            return TFTDataset(
                np.zeros((0, self.lookback, n_past)),
                np.zeros((0, self.horizon, n_future)),
                np.zeros((0, 1), dtype=np.int64),
                np.zeros((0, self.horizon)),
            )

        return TFTDataset(
            np.stack(all_past),
            np.stack(all_future),
            np.stack(all_static),
            np.stack(all_targets),
        )

    def train_dataloader(self) -> DataLoader:
        self.setup()
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=0, pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        self.setup()
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=0, pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        self.setup()
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=0, pin_memory=True,
        )

    def get_sample_batch(self) -> tuple:
        """Get a single batch for inspection/debugging."""
        loader = self.train_dataloader()
        return next(iter(loader))
