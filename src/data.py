"""
Electricity-style Data Pipeline for Temporal Fusion Transformer.

Fetches hourly OHLCV data from the ARF Data API (multiple tickers as entity proxies),
preprocesses it into TFT-compatible format with:
  - past_inputs: observed values over the lookback window
  - known_future_inputs: calendar features known in advance (hour, day_of_week, month)
  - static_inputs: entity (ticker) embeddings
  - targets: values to predict over the forecast horizon
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
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V",
    "XOM", "UNH", "PG", "HD", "MA", "DIS", "NFLX", "INTC", "AMD",
]
ARF_API_URL = "https://ai.1s.xyz/api/data/ohlcv"
DATA_DIR = Path("data")


def fetch_ticker_data(ticker: str, interval: str = "1h", period: str = "2y",
                      cache_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Fetch OHLCV data from ARF Data API with local caching."""
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
        known_future_inputs: (horizon, n_known_features)    - calendar features for forecast period
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
    """Data module that fetches, preprocesses, and serves TFT-compatible batches.

    Follows the TFT paper's Electricity dataset protocol:
    - Multiple entities (tickers as proxy for customers)
    - Hourly data with temporal features
    - Per-entity normalization (fit on train only)
    - Chronological train/val/test split
    """

    def __init__(
        self,
        tickers: Optional[list[str]] = None,
        lookback: int = DEFAULT_LOOKBACK,
        horizon: int = DEFAULT_HORIZON,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        batch_size: int = 64,
        target_col: str = "close",
        data_dir: Path = DATA_DIR,
    ):
        self.tickers = tickers or DEFAULT_TICKERS
        self.lookback = lookback
        self.horizon = horizon
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.batch_size = batch_size
        self.target_col = target_col
        self.data_dir = Path(data_dir)
        self.scaler = EntityScaler()
        self.entity_map: dict[str, int] = {}
        self._prepared = False

    def prepare_data(self) -> pd.DataFrame:
        """Download and combine data from all tickers."""
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

        # Step 1: Fetch data
        df = self.prepare_data()

        # Step 2: Build temporal features
        temporal = build_temporal_features(df["timestamp"])
        df = pd.concat([df, temporal], axis=1)

        # Step 3: Observed features (past only): close, volume (log-transformed)
        df["log_volume"] = np.log1p(df["volume"].values)

        # Step 4: Chronological split per entity
        train_dfs, val_dfs, test_dfs = [], [], []
        for eid in df["entity_id"].unique():
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

        # Step 5: Fit scaler on train only
        observed_cols = [self.target_col, "log_volume"]
        self.scaler.fit(
            train_df["entity_id"].values,
            train_df[observed_cols].values,
        )

        # Step 6: Create windowed samples for each split
        self.train_dataset = self._create_dataset(train_df, observed_cols, fit_split=True)
        self.val_dataset = self._create_dataset(val_df, observed_cols)
        self.test_dataset = self._create_dataset(test_df, observed_cols)

        # Store split info for reporting
        self.split_info = {
            "train_size": len(self.train_dataset),
            "val_size": len(self.val_dataset),
            "test_size": len(self.test_dataset),
            "n_entities": len(self.tickers),
            "lookback": self.lookback,
            "horizon": self.horizon,
        }

        self._prepared = True
        logger.info(
            f"Data prepared: train={len(self.train_dataset)}, "
            f"val={len(self.val_dataset)}, test={len(self.test_dataset)}"
        )

    def _create_dataset(self, df: pd.DataFrame, observed_cols: list[str],
                        fit_split: bool = False) -> TFTDataset:
        """Create windowed TFT dataset from a split DataFrame."""
        temporal_cols = ["hour_of_day", "day_of_week", "month", "day_of_month"]
        all_past = []
        all_future = []
        all_static = []
        all_targets = []

        for eid in df["entity_id"].unique():
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

            # Create sliding windows
            for start in range(0, n - self.lookback - self.horizon + 1, 1):
                lb_end = start + self.lookback
                hz_end = lb_end + self.horizon

                # Past: scaled observed + temporal features over lookback
                past_obs = obs_values[start:lb_end]           # (lookback, 2)
                past_temp = temporal_values[start:lb_end]      # (lookback, 4)
                past = np.concatenate([past_obs, past_temp], axis=1)  # (lookback, 6)

                # Known future: temporal features over horizon
                future_temp = temporal_values[lb_end:hz_end]   # (horizon, 4)

                # Static: entity id
                static = np.array([eid])                        # (1,)

                # Target: scaled target over horizon
                target = obs_values[lb_end:hz_end, 0]          # (horizon,) — first col is target

                all_past.append(past)
                all_future.append(future_temp)
                all_static.append(static)
                all_targets.append(target)

        if not all_past:
            # Return empty dataset
            return TFTDataset(
                np.zeros((0, self.lookback, len(observed_cols) + len(temporal_cols))),
                np.zeros((0, self.horizon, len(temporal_cols))),
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
