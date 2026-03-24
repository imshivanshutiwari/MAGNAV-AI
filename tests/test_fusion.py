"""Tests for sensor fusion filters: EKF, UKF, Particle Filter, StateEstimator."""
import numpy as np
import pytest

from fusion.ekf import ExtendedKalmanFilter
from fusion.ukf import UnscentedKalmanFilter
from fusion.particle_filter import ParticleFilter
from fusion.state_estimator import StateEstimator


# ---------------------------------------------------------------------------
# EKF Tests
# ---------------------------------------------------------------------------

class TestEKF:
    def test_init_state_dim(self):
        ekf = ExtendedKalmanFilter(state_dim=15)
        assert ekf.get_state().shape == (15,)

    def test_init_covariance_positive_definite(self):
        ekf = ExtendedKalmanFilter()
        P = ekf.get_covariance()
        eigvals = np.linalg.eigvals(P)
        assert np.all(eigvals > 0)

    def test_predict_changes_state(self):
        ekf = ExtendedKalmanFilter()
        x0 = ekf.get_state().copy()
        accel = np.array([0.0, 0.0, 9.81])
        gyro = np.array([0.0, 0.0, 0.1])
        ekf.predict(dt=0.01, imu_accel=accel, imu_gyro=gyro)
        x1 = ekf.get_state()
        # Some elements should change
        assert not np.allclose(x0, x1)

    def test_predict_covariance_increases(self):
        ekf = ExtendedKalmanFilter()
        P0_trace = np.trace(ekf.get_covariance())
        accel = np.zeros(3)
        gyro = np.zeros(3)
        ekf.predict(dt=0.1, imu_accel=accel, imu_gyro=gyro)
        P1_trace = np.trace(ekf.get_covariance())
        assert P1_trace >= P0_trace * 0.99  # Should not decrease much

    def test_update_reduces_uncertainty(self):
        ekf = ExtendedKalmanFilter()
        H = np.zeros((3, 15))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        P0_trace = np.trace(ekf.get_covariance())
        meas = np.array([1.0, 2.0, 3.0])
        ekf.update(meas, H)
        P1_trace = np.trace(ekf.get_covariance())
        assert P1_trace < P0_trace

    def test_set_state(self):
        ekf = ExtendedKalmanFilter()
        x = np.arange(15, dtype=float)
        ekf.set_state(x)
        assert np.allclose(ekf.get_state(), x)

    def test_predict_update_cycle(self):
        ekf = ExtendedKalmanFilter()
        accel = np.array([0.0, 0.0, 9.81])
        gyro = np.zeros(3)
        H = np.eye(3, 15)
        for _ in range(10):
            ekf.predict(dt=0.01, imu_accel=accel, imu_gyro=gyro)
            ekf.update(np.zeros(3), H)
        P = ekf.get_covariance()
        assert np.all(np.isfinite(P))


# ---------------------------------------------------------------------------
# UKF Tests
# ---------------------------------------------------------------------------

class TestUKF:
    def test_init(self):
        ukf = UnscentedKalmanFilter(state_dim=15)
        assert ukf.get_state().shape == (15,)

    def test_sigma_points_count(self):
        ukf = UnscentedKalmanFilter(state_dim=15)
        sigma = ukf._compute_sigma_points()
        assert sigma.shape == (31, 15)

    def test_predict_changes_state(self):
        ukf = UnscentedKalmanFilter()
        x0 = ukf.get_state().copy()
        accel = np.array([0.0, 0.0, 9.81])
        gyro = np.array([0.0, 0.0, 0.05])
        ukf.predict(dt=0.01, imu_accel=accel, imu_gyro=gyro)
        x1 = ukf.get_state()
        assert not np.allclose(x0, x1)

    def test_weights_sum_to_one(self):
        ukf = UnscentedKalmanFilter(state_dim=15)
        assert abs(ukf._Wm.sum() - 1.0) < 1e-9

    def test_update_cycle(self):
        ukf = UnscentedKalmanFilter()
        accel = np.zeros(3)
        gyro = np.zeros(3)
        for _ in range(5):
            ukf.predict(dt=0.01, imu_accel=accel, imu_gyro=gyro)
            ukf.update(np.array([0.0, 0.0, 0.0]))
        assert np.all(np.isfinite(ukf.get_state()))


# ---------------------------------------------------------------------------
# Particle Filter Tests
# ---------------------------------------------------------------------------

class TestParticleFilter:
    def test_init(self):
        pf = ParticleFilter(n_particles=100, state_dim=9)
        assert pf.n_particles == 100

    def test_initialize_particles_shape(self):
        pf = ParticleFilter(n_particles=100)
        pf.initialize_particles(np.zeros(9), np.eye(9) * 0.1)
        assert pf._particles.shape == (100, 9)

    def test_predict_changes_particles(self):
        pf = ParticleFilter(n_particles=50)
        pf.initialize_particles(np.zeros(9), np.eye(9) * 0.01)
        particles_before = pf._particles.copy()
        accel = np.array([0.0, 0.0, 9.81])
        gyro = np.zeros(3)
        pf.predict(dt=0.01, imu_accel=accel, imu_gyro=gyro)
        assert not np.allclose(particles_before, pf._particles)

    def test_get_estimate_shape(self):
        pf = ParticleFilter(n_particles=50)
        pf.initialize_particles(np.zeros(9), np.eye(9) * 0.1)
        estimate = pf.get_estimate()
        assert estimate.shape == (9,)

    def test_effective_sample_size_max_at_uniform(self):
        pf = ParticleFilter(n_particles=50)
        pf.initialize_particles(np.zeros(9), np.eye(9) * 0.1)
        ess = pf._effective_sample_size()
        assert abs(ess - 50.0) < 1.0  # Should be close to n_particles

    def test_resample_preserves_count(self):
        pf = ParticleFilter(n_particles=50)
        pf.initialize_particles(np.zeros(9), np.eye(9) * 0.1)
        pf.resample()
        assert pf._particles.shape[0] == 50

    def test_update_normalizes_weights(self):
        pf = ParticleFilter(n_particles=50)
        pf.initialize_particles(np.zeros(9), np.eye(9) * 0.1)
        pf.update(np.zeros(3))
        log_w = pf._log_weights
        # log sum exp should be ≈ 0 (normalized)
        log_sum = np.logaddexp.reduce(log_w)
        assert abs(log_sum) < 1e-6


# ---------------------------------------------------------------------------
# State Estimator Tests
# ---------------------------------------------------------------------------

class TestStateEstimator:
    def test_covariance_intersection_basic(self):
        se = StateEstimator()
        states = [np.array([1.0, 2.0, 3.0]), np.array([1.1, 1.9, 3.1])]
        covs = [np.eye(3) * 2.0, np.eye(3) * 3.0]
        x_fused, P_fused = se.fuse_estimates(states, covs)
        assert x_fused.shape == (3,)
        assert P_fused.shape == (3, 3)

    def test_weights_sum_to_one(self):
        se = StateEstimator()
        covs = [np.eye(3) * 1.0, np.eye(3) * 2.0, np.eye(3) * 4.0]
        w = se.compute_filter_weights(covs)
        assert abs(w.sum() - 1.0) < 1e-9

    def test_lower_covariance_gets_higher_weight(self):
        se = StateEstimator()
        covs = [np.eye(3) * 1.0, np.eye(3) * 10.0]
        w = se.compute_filter_weights(covs)
        assert w[0] > w[1]

    def test_fused_state_between_estimates(self):
        se = StateEstimator()
        states = [np.array([0.0, 0.0, 0.0]), np.array([10.0, 10.0, 10.0])]
        covs = [np.eye(3), np.eye(3)]
        x_fused, _ = se.fuse_estimates(states, covs)
        # Fused should be between the two
        assert np.all(x_fused >= -0.1)
        assert np.all(x_fused <= 10.1)

    def test_step_with_all_filters(self):
        ekf = ExtendedKalmanFilter()
        ukf = UnscentedKalmanFilter()
        pf = ParticleFilter(n_particles=20)
        pf.initialize_particles(np.zeros(9), np.eye(9) * 0.1)
        se = StateEstimator(ekf, ukf, pf)
        x_fused, P_fused = se.step()
        assert x_fused is not None
        assert P_fused is not None
