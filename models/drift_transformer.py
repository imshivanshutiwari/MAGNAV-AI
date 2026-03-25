"""Transformer-based drift correction model."""
import os
import math
import numpy as np
import torch
import torch.nn as nn
from typing import Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class DriftTransformer(nn.Module):
    """Transformer encoder for drift correction.

    Input: (batch, seq_len, input_dim) residual sequence
    Output: (batch, output_dim) drift correction Δx, Δy, Δz
    """

    def __init__(self,
                 input_dim: int = 3,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 output_dim: int = 3):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.output_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            (batch, output_dim) drift correction
        """
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        x = self.encoder(x)
        # Use mean pooling over sequence
        x = x.mean(dim=1)
        return self.output_head(x)

    def train_step(self, optimizer: torch.optim.Optimizer,
                   x_batch: torch.Tensor, y_batch: torch.Tensor) -> float:
        """Single training step.

        Args:
            optimizer: PyTorch optimizer
            x_batch: (batch, seq_len, input_dim)
            y_batch: (batch, output_dim) ground truth drift

        Returns:
            MSE loss value
        """
        self.train()
        optimizer.zero_grad()
        pred = self(x_batch)
        loss = nn.functional.mse_loss(pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()
        return float(loss.item())

    def predict(self, residual_sequence: np.ndarray) -> np.ndarray:
        """Predict drift correction from residual sequence.

        Args:
            residual_sequence: (seq_len, input_dim) or (batch, seq_len, input_dim)

        Returns:
            Drift correction (output_dim,)
        """
        self.eval()
        x = np.asarray(residual_sequence, dtype=np.float32)
        if x.ndim == 2:
            x = x[np.newaxis]
        tensor = torch.from_numpy(x)
        with torch.no_grad():
            out = self(tensor)
        return out.squeeze(0).numpy()

    def save(self, path: str) -> None:
        """Save model weights."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, **kwargs) -> "DriftTransformer":
        """Load model from file."""
        model = cls(**kwargs)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        return model
