"""
Temporal Fusion Transformer (TFT) - PyTorch Lightning implementation.

Architecture components (from the paper):
1. Gated Residual Networks (GRN) for adaptive nonlinear processing
2. Variable Selection Networks for input feature selection
3. Static covariate encoders for time-invariant metadata
4. LSTM encoder-decoder for local temporal processing
5. Interpretable multi-head attention for long-range dependencies

Input shapes (from ElectricityDataModule):
    past_inputs:         (B, 168, 5)  - 1 observed + 4 temporal
    known_future_inputs: (B, 24, 5)   - 4 temporal + 1 holiday
    static_inputs:       (B, 1)       - entity_id (long)
    targets:             (B, 24)      - scaled power_usage
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl

from src.evaluation import QuantileLoss


class GatedLinearUnit(nn.Module):
    """GLU activation: sigmoid gate * linear transform."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.gate = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.gate(x)) * self.fc(x)


class GatedResidualNetwork(nn.Module):
    """GRN: the primary building block of TFT.

    Applies ELU -> Linear -> GLU with skip connection and layer norm.
    Optionally accepts a context vector (static enrichment).
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 context_dim: int = 0, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.context_proj = nn.Linear(context_dim, hidden_dim, bias=False) if context_dim > 0 else None
        self.glu = GatedLinearUnit(hidden_dim, output_dim)
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        residual = self.skip(x) if self.skip is not None else x
        hidden = self.fc1(x)
        if self.context_proj is not None and context is not None:
            hidden = hidden + self.context_proj(context)
        hidden = self.elu(hidden)
        hidden = self.fc2(hidden)
        hidden = self.dropout(hidden)
        hidden = self.glu(hidden)
        return self.layer_norm(hidden + residual)


class VariableSelectionNetwork(nn.Module):
    """Selects the most salient input variables at each time step."""

    def __init__(self, input_dim: int, n_vars: int, hidden_dim: int,
                 context_dim: int = 0, dropout: float = 0.1):
        super().__init__()
        self.n_vars = n_vars
        self.per_var_dim = input_dim // n_vars

        # One GRN per input variable
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(self.per_var_dim, hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(n_vars)
        ])

        # Softmax selection weights
        self.selection_grn = GatedResidualNetwork(
            input_dim, hidden_dim, n_vars, context_dim=context_dim, dropout=dropout
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (batch, time, input_dim) or (batch, input_dim)
            context: optional static context (batch, context_dim)
        Returns:
            (batch, time, hidden_dim) or (batch, hidden_dim)
        """
        # Expand static context to match temporal dimension if needed
        if context is not None and x.ndim == 3 and context.ndim == 2:
            context = context.unsqueeze(1).expand(-1, x.size(1), -1)

        # Split input into per-variable chunks
        var_inputs = torch.chunk(x, self.n_vars, dim=-1)

        # Process each variable independently
        var_outputs = []
        for i, var_grn in enumerate(self.var_grns):
            var_outputs.append(var_grn(var_inputs[i]))

        var_outputs = torch.stack(var_outputs, dim=-2)  # (..., n_vars, hidden)

        # Compute selection weights
        weights = self.softmax(self.selection_grn(x, context))  # (..., n_vars)
        weights = weights.unsqueeze(-1)  # (..., n_vars, 1)

        # Weighted combination
        selected = (var_outputs * weights).sum(dim=-2)  # (..., hidden)
        return selected


class InterpretableMultiHeadAttention(nn.Module):
    """Multi-head attention with shared value weights for interpretability."""

    def __init__(self, hidden_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = hidden_dim // n_heads

        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # Shared value weights (interpretability)
        self.W_v = nn.Linear(hidden_dim, self.d_k, bias=False)
        self.W_o = nn.Linear(self.d_k, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = query.shape

        Q = self.W_q(query).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value)  # (B, T_k, d_k) - shared across heads

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Average attention across heads, apply to shared values
        avg_attn = attn_weights.mean(dim=1)  # (B, T_q, T_k)
        attn_output = torch.matmul(avg_attn, V)  # (B, T_q, d_k)
        output = self.W_o(attn_output)  # (B, T_q, hidden)

        return output, avg_attn


class TFTModel(pl.LightningModule):
    """Temporal Fusion Transformer for multi-horizon time series forecasting.

    Paper: "Temporal Fusion Transformers for Interpretable Multi-horizon
    Time Series Forecasting" (Lim et al., 2021)
    """

    def __init__(
        self,
        n_past_features: int = 5,
        n_future_features: int = 5,
        n_static_features: int = 1,
        n_entities: int = 370,
        hidden_dim: int = 64,
        lstm_layers: int = 1,
        n_heads: int = 4,
        dropout: float = 0.1,
        horizon: int = 24,
        quantiles: list[float] = None,
        learning_rate: float = 1e-3,
        max_epochs: int = 50,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.quantiles = quantiles or [0.5, 0.9]
        self.horizon = horizon
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.n_quantiles = len(self.quantiles)

        # Static embedding
        self.entity_embedding = nn.Embedding(n_entities, hidden_dim)

        # Static covariate encoders (4 context vectors for different uses)
        self.static_grns = nn.ModuleList([
            GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(4)
        ])

        # Variable selection for past (observed) and future (known) inputs
        self.past_vsn = VariableSelectionNetwork(
            n_past_features, n_past_features, hidden_dim,
            context_dim=hidden_dim, dropout=dropout,
        )
        self.future_vsn = VariableSelectionNetwork(
            n_future_features, n_future_features, hidden_dim,
            context_dim=hidden_dim, dropout=dropout,
        )

        # LSTM encoder-decoder
        self.encoder_lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers=lstm_layers,
            batch_first=True, dropout=dropout if lstm_layers > 1 else 0,
        )
        self.decoder_lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers=lstm_layers,
            batch_first=True, dropout=dropout if lstm_layers > 1 else 0,
        )

        # Gated skip connection from LSTM
        self.lstm_glu = GatedLinearUnit(hidden_dim, hidden_dim)
        self.lstm_layer_norm = nn.LayerNorm(hidden_dim)

        # Static enrichment GRN
        self.enrichment_grn = GatedResidualNetwork(
            hidden_dim, hidden_dim, hidden_dim, context_dim=hidden_dim, dropout=dropout,
        )

        # Interpretable multi-head attention
        self.attention = InterpretableMultiHeadAttention(hidden_dim, n_heads, dropout=dropout)
        self.attn_glu = GatedLinearUnit(hidden_dim, hidden_dim)
        self.attn_layer_norm = nn.LayerNorm(hidden_dim)

        # Position-wise feed-forward
        self.output_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout=dropout)

        # Quantile output projection
        self.output_proj = nn.Linear(hidden_dim, self.n_quantiles)

        # Loss
        self.criterion = QuantileLoss(self.quantiles)

    def forward(self, past_inputs: torch.Tensor, known_future_inputs: torch.Tensor,
                static_inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through TFT.

        Args:
            past_inputs: (B, lookback, n_past_features)
            known_future_inputs: (B, horizon, n_future_features)
            static_inputs: (B, 1) entity IDs (long)

        Returns:
            Quantile predictions: (B, horizon, n_quantiles)
        """
        B = past_inputs.size(0)

        # 1. Static covariate encoding
        static_emb = self.entity_embedding(static_inputs.squeeze(-1))  # (B, hidden)
        cs, ce, ch, cc = [grn(static_emb) for grn in self.static_grns]

        # 2. Variable selection
        past_selected = self.past_vsn(past_inputs, cs)        # (B, lookback, hidden)
        future_selected = self.future_vsn(known_future_inputs, cs)  # (B, horizon, hidden)

        # 3. LSTM encoder-decoder
        h0 = ch.unsqueeze(0).expand(self.hparams.lstm_layers, -1, -1).contiguous()
        c0 = cc.unsqueeze(0).expand(self.hparams.lstm_layers, -1, -1).contiguous()

        encoder_out, (hn, cn) = self.encoder_lstm(past_selected, (h0, c0))
        decoder_out, _ = self.decoder_lstm(future_selected, (hn, cn))

        # Combine encoder and decoder outputs for the full sequence
        # We only need decoder outputs for the forecast horizon
        lstm_out = decoder_out  # (B, horizon, hidden)

        # Gated skip connection over LSTM
        lstm_skip = self.lstm_glu(lstm_out)
        lstm_skip = self.lstm_layer_norm(lstm_skip + future_selected)

        # 4. Static enrichment
        enriched = self.enrichment_grn(lstm_skip, ce.unsqueeze(1).expand(-1, self.horizon, -1))

        # 5. Self-attention (decoder only for forecasting)
        # Create causal mask for the decoder
        mask = torch.triu(torch.ones(self.horizon, self.horizon, device=past_inputs.device), diagonal=1)
        mask = (mask == 0).unsqueeze(0).unsqueeze(0)  # (1, 1, H, H)

        attn_out, self._attn_weights = self.attention(enriched, enriched, enriched, mask)
        attn_skip = self.attn_glu(attn_out)
        attn_skip = self.attn_layer_norm(attn_skip + enriched)

        # 6. Position-wise output
        output = self.output_grn(attn_skip)

        # 7. Quantile projections
        quantile_preds = self.output_proj(output)  # (B, horizon, n_quantiles)
        return quantile_preds

    def training_step(self, batch, batch_idx):
        past, future, static, targets = batch
        preds = self(past, future, static)
        loss = self.criterion(preds, targets)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        past, future, static, targets = batch
        preds = self(past, future, static)
        loss = self.criterion(preds, targets)

        # Per-quantile losses for monitoring
        for i, q in enumerate(self.quantiles):
            errors = targets - preds[:, :, i]
            ql = torch.max(q * errors, (q - 1) * errors).mean()
            self.log(f"val_ql_p{int(q*100)}", ql, prog_bar=True)

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        past, future, static, targets = batch
        preds = self(past, future, static)
        loss = self.criterion(preds, targets)

        for i, q in enumerate(self.quantiles):
            errors = targets - preds[:, :, i]
            ql = torch.max(q * errors, (q - 1) * errors).mean()
            self.log(f"test_ql_p{int(q*100)}", ql)

        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs, eta_min=1e-6
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
