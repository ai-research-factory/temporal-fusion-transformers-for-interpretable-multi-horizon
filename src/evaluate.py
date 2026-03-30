"""
Evaluation script for trained TFT model.

Loads the best checkpoint, evaluates on the test set, and saves
P50/P90 Quantile Loss to reports/cycle_3/metrics.json.

Usage:
    python -m src.evaluate [--max_entities N]
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl

from src.data import ElectricityDataModule
from src.evaluation import calculate_quantile_loss
from src.models.tft import TFTModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports/cycle_3")


def evaluate():
    parser = argparse.ArgumentParser(description="Evaluate trained TFT model")
    parser.add_argument("--max_entities", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None,
                        help="Window stride for test data (default: from train_info)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Explicit checkpoint path (otherwise uses train_info.json)")
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load training info
    info_path = MODELS_DIR / "train_info.json"
    if info_path.exists():
        with open(info_path) as f:
            train_info = json.load(f)
        logger.info(f"Loaded training info from {info_path}")
    else:
        train_info = {}

    # Determine checkpoint path
    ckpt_path = args.checkpoint or train_info.get("best_model_path")
    if not ckpt_path or not Path(ckpt_path).exists():
        raise FileNotFoundError(
            f"No checkpoint found. Train the model first with: python -m src.train"
        )

    # Use max_entities from training if not overridden
    max_entities = args.max_entities or train_info.get("max_entities")

    # Load data with same settings as training
    logger.info("Setting up data module...")
    stride = args.stride or train_info.get("stride", 24)
    dm = ElectricityDataModule(max_entities=max_entities, batch_size=128, stride=stride)
    dm.setup()

    # Load model from checkpoint
    logger.info(f"Loading model from {ckpt_path}")
    model = TFTModel.load_from_checkpoint(ckpt_path)
    model.eval()

    # Run test set evaluation via Trainer
    trainer = pl.Trainer(accelerator="auto", devices=1, enable_progress_bar=True)
    test_results = trainer.test(model, dataloaders=dm.test_dataloader())
    logger.info(f"Trainer test results: {test_results}")

    # Detailed per-quantile evaluation with numpy
    logger.info("Computing detailed quantile losses...")
    device = next(model.parameters()).device
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dm.test_dataloader():
            past, future, static, targets = batch
            past = past.to(device)
            future = future.to(device)
            static = static.to(device)
            preds = model(past, future, static)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())

    all_preds = np.concatenate(all_preds, axis=0)    # (N, horizon, n_quantiles)
    all_targets = np.concatenate(all_targets, axis=0)  # (N, horizon)

    # Calculate quantile losses
    ql_p50 = calculate_quantile_loss(all_targets, all_preds[:, :, 0], quantile=0.5)
    ql_p90 = calculate_quantile_loss(all_targets, all_preds[:, :, 1], quantile=0.9)

    logger.info(f"Test Quantile Loss P50: {ql_p50:.6f}")
    logger.info(f"Test Quantile Loss P90: {ql_p90:.6f}")

    # Generate metrics.json (ARF standard schema + custom TFT metrics)
    metrics = {
        "sharpeRatio": 0.0,
        "annualReturn": 0.0,
        "maxDrawdown": 0.0,
        "hitRate": 0.0,
        "totalTrades": 0,
        "transactionCosts": {"feeBps": 10, "slippageBps": 5, "netSharpe": 0.0},
        "walkForward": {"windows": 0, "positiveWindows": 0, "avgOosSharpe": 0.0},
        "customMetrics": {
            "phase": "3-training-evaluation",
            "model": "Temporal Fusion Transformer",
            "quantile_loss_p50": round(ql_p50, 6),
            "quantile_loss_p90": round(ql_p90, 6),
            "quantile_loss_total": round(ql_p50 + ql_p90, 6),
            "test_samples": int(all_targets.shape[0]),
            "horizon": int(all_targets.shape[1]),
            "n_entities": dm.split_info["n_entities"],
            "epochs_trained": train_info.get("epochs_trained", 0),
            "best_val_loss": train_info.get("best_val_loss"),
            "hidden_dim": train_info.get("hidden_dim", 64),
            "data_source": dm.split_info["data_source"],
            "train_period": dm.date_boundaries.get("train_start", "") + " to " + dm.date_boundaries.get("train_end", ""),
            "test_period": dm.date_boundaries.get("test_start", "") + " to " + dm.date_boundaries.get("test_end", ""),
        },
    }

    metrics_path = REPORTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")

    return metrics


if __name__ == "__main__":
    evaluate()
