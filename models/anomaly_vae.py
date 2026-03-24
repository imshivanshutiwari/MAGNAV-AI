"""Variational Autoencoder for magnetic anomaly detection."""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class AnomalyVAE(nn.Module):
    """VAE for detecting anomalous magnetic field readings.

    Architecture:
        Encoder: input_dim → 128 → 64 → (mu: 32, log_var: 32)
        Decoder: 32 → 64 → 128 → input_dim
    """

    def __init__(self,
                 input_dim: int = 3,
                 latent_dim: int = 32,
                 hidden_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_log_var = nn.Linear(64, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

        self._threshold: Optional[float] = None
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters."""
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterize(self, mu: torch.Tensor,
                        log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + eps * std."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            (reconstruction, mu, log_var)
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var

    @staticmethod
    def loss_function(recon_x: torch.Tensor, x: torch.Tensor,
                      mu: torch.Tensor, log_var: torch.Tensor,
                      beta: float = 1.0) -> torch.Tensor:
        """ELBO loss = reconstruction loss + beta * KL divergence."""
        recon_loss = F.mse_loss(recon_x, x, reduction="sum")
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + beta * kl_loss

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-sample reconstruction error as anomaly score."""
        self.eval()
        with torch.no_grad():
            recon, mu, log_var = self(x)
            score = F.mse_loss(recon, x, reduction="none").mean(dim=-1)
        return score

    def detect_anomaly(self, x: np.ndarray,
                        threshold: float = None) -> Tuple[bool, float]:
        """Detect if input is anomalous.

        Args:
            x: (input_dim,) or (batch, input_dim)
            threshold: Override stored threshold

        Returns:
            (is_anomaly, anomaly_score)
        """
        if threshold is None:
            threshold = self._threshold if self._threshold is not None else 1000.0

        x_t = torch.from_numpy(np.asarray(x, dtype=np.float32))
        if x_t.ndim == 1:
            x_t = x_t.unsqueeze(0)

        score = float(self.anomaly_score(x_t).mean().item())
        return score > threshold, score

    def compute_threshold(self, dataset: np.ndarray,
                           percentile: float = 95) -> float:
        """Compute anomaly threshold from a dataset.

        Args:
            dataset: (N, input_dim) normal samples
            percentile: Percentile for threshold

        Returns:
            Threshold value
        """
        self.eval()
        x_t = torch.from_numpy(np.asarray(dataset, dtype=np.float32))
        with torch.no_grad():
            scores = self.anomaly_score(x_t).numpy()
        self._threshold = float(np.percentile(scores, percentile))
        return self._threshold

    def train_step(self, optimizer: torch.optim.Optimizer,
                   x_batch: torch.Tensor, beta: float = 1.0) -> float:
        """Single training step."""
        self.train()
        optimizer.zero_grad()
        recon, mu, log_var = self(x_batch)
        loss = self.loss_function(recon, x_batch, mu, log_var, beta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()
        return float(loss.item())

    def save(self, path: str) -> None:
        """Save model weights and threshold."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "threshold": self._threshold}, path)

    @classmethod
    def load(cls, path: str, **kwargs) -> "AnomalyVAE":
        """Load model from file."""
        model = cls(**kwargs)
        data = torch.load(path, map_location="cpu")
        model.load_state_dict(data["state_dict"])
        model._threshold = data.get("threshold")
        return model
