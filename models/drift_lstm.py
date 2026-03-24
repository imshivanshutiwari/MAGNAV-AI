"""3-layer LSTM with attention for drift correction."""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """Soft attention over LSTM output sequence."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        """Compute attention-weighted context vector.

        Args:
            lstm_out: (batch, seq_len, hidden_dim)

        Returns:
            context: (batch, hidden_dim)
        """
        scores = self.attn(lstm_out).squeeze(-1)  # (batch, seq_len)
        weights = F.softmax(scores, dim=-1)        # (batch, seq_len)
        context = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)  # (batch, hidden_dim)
        return context, weights


class DriftLSTM(nn.Module):
    """3-layer LSTM + attention for drift correction.

    Input: (batch, seq_len, input_dim)
    Output: (batch, output_dim) drift correction Δx, Δy, Δz
    """

    def __init__(self,
                 input_dim: int = 3,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 output_dim: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = AttentionLayer(hidden_dim)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        for module in self.output_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            (batch, output_dim)
        """
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)
        context, _ = self.attention(lstm_out)  # (batch, hidden_dim)
        return self.output_head(context)

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return attention weights for visualization.

        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            (batch, seq_len) attention weights
        """
        with torch.no_grad():
            lstm_out, _ = self.lstm(x)
            _, weights = self.attention(lstm_out)
        return weights

    def train_step(self, optimizer: torch.optim.Optimizer,
                   x_batch: torch.Tensor, y_batch: torch.Tensor) -> float:
        """Single training step."""
        self.train()
        optimizer.zero_grad()
        pred = self(x_batch)
        loss = F.mse_loss(pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()
        return float(loss.item())

    def predict(self, residual_sequence: np.ndarray) -> np.ndarray:
        """Predict drift correction.

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
    def load(cls, path: str, **kwargs) -> "DriftLSTM":
        """Load model from file."""
        model = cls(**kwargs)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        return model
