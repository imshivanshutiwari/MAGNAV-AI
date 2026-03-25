"""Tests for the NavigationPipeline (10-step integration test)."""
import numpy as np
import pytest


class TestNavigationPipeline:
    def _make_pipeline(self, tmp_path):
        """Create a pipeline with a minimal test config."""
        import yaml
        import os
        config = {
            "data": {"emag2": {"cache_dir": str(tmp_path / "emag2")}},
            "sensors": {
                "imu": {"accel_noise_std": 0.001, "gyro_noise_std": 0.0001,
                         "update_rate_hz": 100.0},
                "magnetometer": {"noise_std": 5.0},
                "barometer": {"noise_std": 0.5},
            },
            "filters": {
                "ekf": {"state_dim": 15, "obs_dim": 3},
                "ukf": {"state_dim": 15, "obs_dim": 3},
                "particle_filter": {"n_particles": 20, "state_dim": 9},
            },
            "models": {
                "drift_transformer": {"input_dim": 3, "d_model": 64, "nhead": 4,
                                       "num_encoder_layers": 2, "dim_feedforward": 128,
                                       "output_dim": 3},
                "drift_lstm": {"input_dim": 3, "hidden_dim": 64, "num_layers": 2, "output_dim": 3},
                "anomaly_vae": {"input_dim": 3, "latent_dim": 8},
            },
            "pipeline": {"initial_lat": 40.0, "initial_lon": -74.0, "initial_alt": 0.0},
        }
        config_path = str(tmp_path / "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        from pipeline.main import NavigationPipeline
        pipeline = NavigationPipeline(config_path=config_path)
        pipeline.setup()
        return pipeline

    def test_setup(self, tmp_path):
        pipeline = self._make_pipeline(tmp_path)
        assert pipeline._initialized

    def test_step_returns_nav_state(self, tmp_path):
        pipeline = self._make_pipeline(tmp_path)
        from pipeline.main import NavigationState
        accel = np.array([0.0, 0.0, 9.81])
        gyro = np.zeros(3)
        mag = np.array([-1450.0, 4650.0, -29000.0])
        baro = 100.0
        state = pipeline.step(accel, gyro, mag, baro)
        assert isinstance(state, NavigationState)

    def test_step_position_shape(self, tmp_path):
        pipeline = self._make_pipeline(tmp_path)
        state = pipeline.step(np.array([0., 0., 9.81]), np.zeros(3),
                               np.array([-1450., 4650., -29000.]), 100.0)
        assert state.position.shape == (3,)

    def test_step_velocity_shape(self, tmp_path):
        pipeline = self._make_pipeline(tmp_path)
        state = pipeline.step(np.array([0., 0., 9.81]), np.zeros(3),
                               np.array([-1450., 4650., -29000.]), 100.0)
        assert state.velocity.shape == (3,)

    def test_step_latency_under_200ms(self, tmp_path):
        pipeline = self._make_pipeline(tmp_path)
        state = pipeline.step(np.array([0., 0., 9.81]), np.zeros(3),
                               np.array([-1450., 4650., -29000.]), 100.0)
        assert state.latency_ms < 200.0  # Should be well under threshold

    def test_step_magnetic_residual_shape(self, tmp_path):
        pipeline = self._make_pipeline(tmp_path)
        state = pipeline.step(np.array([0., 0., 9.81]), np.zeros(3),
                               np.array([-1450., 4650., -29000.]), 100.0)
        assert state.magnetic_residual.shape == (3,)

    def test_step_filter_weights_sum_to_one(self, tmp_path):
        pipeline = self._make_pipeline(tmp_path)
        state = pipeline.step(np.array([0., 0., 9.81]), np.zeros(3),
                               np.array([-1450., 4650., -29000.]), 100.0)
        assert abs(state.filter_weights.sum() - 1.0) < 0.01

    def test_multiple_steps_build_history(self, tmp_path):
        pipeline = self._make_pipeline(tmp_path)
        accel = np.array([0., 0., 9.81])
        gyro = np.zeros(3)
        mag = np.array([-1450., 4650., -29000.])
        for _ in range(5):
            pipeline.step(accel, gyro, mag, 100.0)
        assert len(pipeline.get_history()) == 5

    def test_get_trajectory_array_shape(self, tmp_path):
        pipeline = self._make_pipeline(tmp_path)
        accel = np.array([0., 0., 9.81])
        gyro = np.zeros(3)
        mag = np.array([-1450., 4650., -29000.])
        for _ in range(3):
            pipeline.step(accel, gyro, mag, 100.0)
        traj = pipeline.get_trajectory_array()
        assert traj.shape == (3, 3)

    def test_to_dict(self, tmp_path):
        pipeline = self._make_pipeline(tmp_path)
        state = pipeline.step(np.array([0., 0., 9.81]), np.zeros(3),
                               np.array([-1450., 4650., -29000.]), 100.0)
        d = state.to_dict()
        assert "position" in d
        assert "velocity" in d
        assert "orientation" in d
        assert "filter_weights" in d

    def test_run_simulation(self, tmp_path):
        pipeline = self._make_pipeline(tmp_path)
        states = pipeline.run_simulation(n_steps=10, dt=0.01)
        assert len(states) == 10

    def test_all_states_finite(self, tmp_path):
        pipeline = self._make_pipeline(tmp_path)
        rng = np.random.default_rng(42)
        for _ in range(10):
            accel = np.array([0., 0., 9.81]) + rng.standard_normal(3) * 0.001
            gyro = rng.standard_normal(3) * 0.001
            mag = np.array([-1450., 4650., -29000.]) + rng.standard_normal(3) * 5
            state = pipeline.step(accel, gyro, mag, 100.0)
            assert np.all(np.isfinite(state.position))
            assert np.all(np.isfinite(state.velocity))
