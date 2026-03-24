"""Tests for ML models: DriftTransformer, DriftLSTM, AnomalyVAE."""
import numpy as np
import pytest
import torch


class TestDriftTransformer:
    def _make_model(self, d_model=64, nhead=4, nlayers=2):
        from models.drift_transformer import DriftTransformer
        return DriftTransformer(
            input_dim=3, d_model=d_model, nhead=nhead,
            num_encoder_layers=nlayers, dim_feedforward=128,
            dropout=0.0, output_dim=3,
        )

    def test_output_shape(self):
        model = self._make_model()
        x = torch.randn(4, 20, 3)
        out = model(x)
        assert out.shape == (4, 3)

    def test_predict_numpy(self):
        model = self._make_model()
        seq = np.random.randn(20, 3).astype(np.float32)
        out = model.predict(seq)
        assert out.shape == (3,)

    def test_predict_batch_numpy(self):
        model = self._make_model()
        seq = np.random.randn(2, 20, 3).astype(np.float32)
        # Predict single from 3D input; should return (batch, 3) from forward
        out = model.predict(seq[0])
        assert out.shape == (3,)

    def test_train_step_reduces_loss(self):
        model = self._make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        x = torch.randn(8, 10, 3)
        y = torch.randn(8, 3)
        losses = [model.train_step(optimizer, x, y) for _ in range(5)]
        # Loss should be finite
        assert all(np.isfinite(l) for l in losses)

    def test_save_load(self, tmp_path):
        model = self._make_model()
        path = str(tmp_path / "transformer.pt")
        model.save(path)
        from models.drift_transformer import DriftTransformer
        loaded = DriftTransformer.load(
            path, input_dim=3, d_model=64, nhead=4,
            num_encoder_layers=2, dim_feedforward=128,
            dropout=0.0, output_dim=3,
        )
        x = torch.randn(1, 10, 3)
        model.eval()
        loaded.eval()
        with torch.no_grad():
            out1 = model(x)
            out2 = loaded(x)
        assert torch.allclose(out1, out2, atol=1e-5)


class TestDriftLSTM:
    def _make_model(self):
        from models.drift_lstm import DriftLSTM
        return DriftLSTM(input_dim=3, hidden_dim=64, num_layers=2, output_dim=3, dropout=0.0)

    def test_output_shape(self):
        model = self._make_model()
        x = torch.randn(4, 20, 3)
        out = model(x)
        assert out.shape == (4, 3)

    def test_attention_weights_shape(self):
        model = self._make_model()
        x = torch.randn(2, 15, 3)
        weights = model.get_attention_weights(x)
        assert weights.shape == (2, 15)

    def test_attention_weights_sum_to_one(self):
        model = self._make_model()
        x = torch.randn(3, 10, 3)
        weights = model.get_attention_weights(x)
        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(3), atol=1e-5)

    def test_predict_numpy(self):
        model = self._make_model()
        seq = np.random.randn(20, 3).astype(np.float32)
        out = model.predict(seq)
        assert out.shape == (3,)

    def test_train_step(self):
        model = self._make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        x = torch.randn(4, 10, 3)
        y = torch.randn(4, 3)
        loss = model.train_step(optimizer, x, y)
        assert np.isfinite(loss)

    def test_save_load(self, tmp_path):
        model = self._make_model()
        path = str(tmp_path / "lstm.pt")
        model.save(path)
        from models.drift_lstm import DriftLSTM
        loaded = DriftLSTM.load(
            path, input_dim=3, hidden_dim=64, num_layers=2, output_dim=3, dropout=0.0
        )
        x = torch.randn(1, 10, 3)
        model.eval()
        loaded.eval()
        with torch.no_grad():
            assert torch.allclose(model(x), loaded(x), atol=1e-5)


class TestAnomalyVAE:
    def _make_model(self):
        from models.anomaly_vae import AnomalyVAE
        return AnomalyVAE(input_dim=3, latent_dim=8, hidden_dim=32)

    def test_output_shapes(self):
        model = self._make_model()
        x = torch.randn(4, 3)
        recon, mu, log_var = model(x)
        assert recon.shape == (4, 3)
        assert mu.shape == (4, 8)
        assert log_var.shape == (4, 8)

    def test_loss_finite(self):
        model = self._make_model()
        from models.anomaly_vae import AnomalyVAE
        x = torch.randn(4, 3)
        recon, mu, log_var = model(x)
        loss = AnomalyVAE.loss_function(recon, x, mu, log_var)
        assert torch.isfinite(loss)

    def test_reparameterize_shape(self):
        model = self._make_model()
        mu = torch.zeros(4, 8)
        lv = torch.zeros(4, 8)
        z = model.reparameterize(mu, lv)
        assert z.shape == (4, 8)

    def test_detect_anomaly_normal(self):
        model = self._make_model()
        x = np.zeros(3, dtype=np.float32)
        is_anomaly, score = model.detect_anomaly(x, threshold=1e9)
        assert not is_anomaly
        assert score >= 0.0

    def test_compute_threshold(self):
        model = self._make_model()
        dataset = np.random.randn(100, 3).astype(np.float32)
        threshold = model.compute_threshold(dataset, percentile=95)
        assert np.isfinite(threshold)
        assert threshold > 0.0

    def test_train_step(self):
        model = self._make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        x = torch.randn(8, 3)
        loss = model.train_step(optimizer, x)
        assert np.isfinite(loss)

    def test_save_load(self, tmp_path):
        model = self._make_model()
        model._threshold = 42.0
        path = str(tmp_path / "vae.pt")
        model.save(path)
        from models.anomaly_vae import AnomalyVAE
        loaded = AnomalyVAE.load(path, input_dim=3, latent_dim=8, hidden_dim=32)
        assert loaded._threshold == 42.0
