"""
Training script for Temporal Fusion Transformer on Electricity dataset.

Usage:
    python -m src.train [--max_entities N] [--epochs E] [--batch_size B]
"""

import argparse
import json
import logging
import os
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.data import ElectricityDataModule
from src.models.tft import TFTModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports/cycle_3")


def main():
    parser = argparse.ArgumentParser(description="Train TFT on Electricity dataset")
    parser.add_argument("--max_entities", type=int, default=None,
                        help="Limit entities for faster dev iteration (None=all 370)")
    parser.add_argument("--epochs", type=int, default=20, help="Max training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--hidden_dim", type=int, default=64, help="TFT hidden dimension")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--n_heads", type=int, default=4, help="Attention heads")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--stride", type=int, default=1, help="Window stride (>1 to reduce dataset size)")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Data setup
    logger.info("Setting up data module...")
    dm = ElectricityDataModule(
        max_entities=args.max_entities,
        batch_size=args.batch_size,
        stride=args.stride,
    )
    dm.setup()

    n_entities = dm.split_info["n_entities"]
    n_past = dm.split_info["n_past_features"]
    n_future = dm.split_info["n_known_future_features"]
    logger.info(
        f"Data ready: {n_entities} entities, "
        f"{dm.split_info['train_size']} train / "
        f"{dm.split_info['val_size']} val / "
        f"{dm.split_info['test_size']} test samples"
    )

    # Model setup
    model = TFTModel(
        n_past_features=n_past,
        n_future_features=n_future,
        n_entities=n_entities,
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        dropout=args.dropout,
        learning_rate=args.lr,
        max_epochs=args.epochs,
        quantiles=[0.5, 0.9],
    )
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(MODELS_DIR),
        filename="tft-best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        patience=args.patience,
        mode="min",
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=[checkpoint_cb, early_stop_cb],
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
        log_every_n_steps=50,
    )

    # Train
    logger.info("Starting training...")
    trainer.fit(
        model,
        train_dataloaders=dm.train_dataloader(),
        val_dataloaders=dm.val_dataloader(),
    )

    logger.info(f"Best model: {checkpoint_cb.best_model_path}")
    logger.info(f"Best val_loss: {checkpoint_cb.best_model_score:.6f}")

    # Save training info for evaluate.py
    train_info = {
        "best_model_path": checkpoint_cb.best_model_path,
        "best_val_loss": float(checkpoint_cb.best_model_score) if checkpoint_cb.best_model_score else None,
        "n_entities": n_entities,
        "n_past_features": n_past,
        "n_future_features": n_future,
        "hidden_dim": args.hidden_dim,
        "n_heads": args.n_heads,
        "dropout": args.dropout,
        "max_entities": args.max_entities,
        "stride": args.stride,
        "epochs_trained": trainer.current_epoch,
    }
    info_path = MODELS_DIR / "train_info.json"
    with open(info_path, "w") as f:
        json.dump(train_info, f, indent=2)
    logger.info(f"Training info saved to {info_path}")

    return trainer, model


if __name__ == "__main__":
    main()
