"""
Quantile Loss evaluation for Temporal Fusion Transformer.

Implements the quantile loss from the TFT paper (Equation 3):
    QL(y, y_hat, q) = max(q * (y - y_hat), (q - 1) * (y - y_hat))

This is equivalent to the pinball loss used in quantile regression.
"""

import numpy as np
import torch


def calculate_quantile_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """Calculate quantile loss (pinball loss) per the TFT paper Equation (3).

    Args:
        y_true: Ground truth values, shape (n_samples,) or (n_samples, horizon)
        y_pred: Predicted quantile values, same shape as y_true
        quantile: Target quantile in (0, 1), e.g. 0.5 for P50, 0.9 for P90

    Returns:
        Mean quantile loss (scalar).
    """
    if not 0 < quantile < 1:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    errors = y_true - y_pred
    loss = np.maximum(quantile * errors, (quantile - 1.0) * errors)
    return float(loss.mean())


class QuantileLoss(torch.nn.Module):
    """PyTorch quantile loss for training.

    Computes the sum of quantile losses across all specified quantiles.
    Used as the training objective for the TFT model.
    """

    def __init__(self, quantiles: list[float] = None):
        super().__init__()
        self.quantiles = quantiles or [0.5, 0.9]

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute combined quantile loss.

        Args:
            y_pred: Predictions, shape (batch, horizon, n_quantiles)
            y_true: Targets, shape (batch, horizon)

        Returns:
            Scalar loss (sum over quantiles, mean over batch and horizon).
        """
        losses = []
        for i, q in enumerate(self.quantiles):
            pred_q = y_pred[:, :, i]
            errors = y_true - pred_q
            loss = torch.max(q * errors, (q - 1.0) * errors)
            losses.append(loss.mean())
        return torch.stack(losses).sum()
