"""Tests for quantile loss evaluation functions."""

import numpy as np
import pytest
import torch
from src.evaluation import calculate_quantile_loss, QuantileLoss


class TestCalculateQuantileLoss:
    def test_perfect_prediction(self):
        """Loss should be zero when prediction equals ground truth."""
        y = np.array([1.0, 2.0, 3.0])
        loss = calculate_quantile_loss(y, y, quantile=0.5)
        assert loss == pytest.approx(0.0, abs=1e-10)

    def test_p50_symmetric(self):
        """P50 loss should be symmetric for equal over/under predictions."""
        y_true = np.array([0.0, 0.0])
        y_over = np.array([1.0, 1.0])   # over-predict by 1
        y_under = np.array([-1.0, -1.0])  # under-predict by 1
        loss_over = calculate_quantile_loss(y_true, y_over, quantile=0.5)
        loss_under = calculate_quantile_loss(y_true, y_under, quantile=0.5)
        assert loss_over == pytest.approx(loss_under, abs=1e-10)

    def test_p90_penalizes_underprediction_more(self):
        """P90 should penalize underprediction more than overprediction."""
        y_true = np.array([0.0])
        y_over = np.array([1.0])
        y_under = np.array([-1.0])
        loss_over = calculate_quantile_loss(y_true, y_over, quantile=0.9)
        loss_under = calculate_quantile_loss(y_true, y_under, quantile=0.9)
        assert loss_under > loss_over

    def test_known_value_p50(self):
        """Manual calculation for P50."""
        y_true = np.array([3.0])
        y_pred = np.array([1.0])
        # error = 3 - 1 = 2; loss = max(0.5*2, -0.5*2) = 1.0
        loss = calculate_quantile_loss(y_true, y_pred, quantile=0.5)
        assert loss == pytest.approx(1.0, abs=1e-10)

    def test_known_value_p90(self):
        """Manual calculation for P90."""
        y_true = np.array([3.0])
        y_pred = np.array([1.0])
        # error = 2; loss = max(0.9*2, -0.1*2) = 1.8
        loss = calculate_quantile_loss(y_true, y_pred, quantile=0.9)
        assert loss == pytest.approx(1.8, abs=1e-10)

    def test_2d_input(self):
        """Should work with 2D arrays (batch, horizon)."""
        y_true = np.ones((10, 24))
        y_pred = np.zeros((10, 24))
        loss = calculate_quantile_loss(y_true, y_pred, quantile=0.5)
        assert loss == pytest.approx(0.5, abs=1e-10)

    def test_invalid_quantile(self):
        """Should raise for quantiles outside (0,1)."""
        y = np.array([1.0])
        with pytest.raises(ValueError):
            calculate_quantile_loss(y, y, quantile=0.0)
        with pytest.raises(ValueError):
            calculate_quantile_loss(y, y, quantile=1.0)


class TestQuantileLossTorch:
    def test_zero_loss_on_perfect_prediction(self):
        """Torch loss should be zero when predictions match targets."""
        criterion = QuantileLoss(quantiles=[0.5, 0.9])
        y_true = torch.ones(4, 24)
        y_pred = torch.ones(4, 24, 2)
        loss = criterion(y_pred, y_true)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_loss_is_positive(self):
        """Loss should always be non-negative."""
        criterion = QuantileLoss(quantiles=[0.5, 0.9])
        y_true = torch.randn(8, 24)
        y_pred = torch.randn(8, 24, 2)
        loss = criterion(y_pred, y_true)
        assert loss.item() >= 0

    def test_gradient_flows(self):
        """Loss should be differentiable."""
        criterion = QuantileLoss(quantiles=[0.5, 0.9])
        y_pred = torch.randn(4, 24, 2, requires_grad=True)
        y_true = torch.randn(4, 24)
        loss = criterion(y_pred, y_true)
        loss.backward()
        assert y_pred.grad is not None
        assert not torch.isnan(y_pred.grad).any()

    def test_consistency_with_numpy(self):
        """Torch and numpy implementations should agree."""
        np.random.seed(42)
        y_true_np = np.random.randn(16, 24)
        y_pred_np = np.random.randn(16, 24)

        # Numpy P50
        np_loss_p50 = calculate_quantile_loss(y_true_np, y_pred_np, 0.5)
        np_loss_p90 = calculate_quantile_loss(y_true_np, y_pred_np, 0.9)

        # Torch
        criterion = QuantileLoss(quantiles=[0.5, 0.9])
        y_true_t = torch.tensor(y_true_np, dtype=torch.float32)
        y_pred_t = torch.stack([
            torch.tensor(y_pred_np, dtype=torch.float32),
            torch.tensor(y_pred_np, dtype=torch.float32),
        ], dim=-1)
        torch_loss = criterion(y_pred_t, y_true_t)

        assert torch_loss.item() == pytest.approx(np_loss_p50 + np_loss_p90, abs=1e-4)
