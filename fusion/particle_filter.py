"""Particle Filter (Bootstrap SIR) for magnetic navigation."""
import numpy as np
from typing import Callable


class ParticleFilter:
    """Bootstrap SIR Particle Filter for magnetic navigation.

    State: [x, y, z, vx, vy, vz, roll, pitch, yaw]
    """

    def __init__(self,
                 n_particles: int = 500,
                 state_dim: int = 9,
                 process_noise_pos: float = 0.1,
                 process_noise_vel: float = 0.5,
                 process_noise_att: float = 0.01,
                 measurement_noise: float = 25.0,
                 resample_threshold: float = 0.5):
        self.n_particles = n_particles
        self.state_dim = state_dim
        self.resample_threshold = resample_threshold
        self.measurement_noise = measurement_noise

        self._process_noise = np.array(
            [process_noise_pos] * 3 +
            [process_noise_vel] * 3 +
            [process_noise_att] * 3
        )

        # Particles: (n_particles, state_dim)
        self._particles = np.zeros((n_particles, state_dim))
        # Log-weights
        self._log_weights = np.full(n_particles, -np.log(n_particles))
        self._rng = np.random.default_rng(3)
        self._initialized = False

    def initialize_particles(self, initial_state: np.ndarray,
                              initial_cov: np.ndarray) -> None:
        """Initialize particles from a Gaussian around the initial state.

        Args:
            initial_state: (state_dim,) mean state
            initial_cov: (state_dim, state_dim) covariance
        """
        s = min(self.state_dim, len(initial_state))
        cov = initial_cov[:s, :s]
        cov = 0.5 * (cov + cov.T) + np.eye(s) * 1e-9
        samples = self._rng.multivariate_normal(initial_state[:s], cov, self.n_particles)
        self._particles[:, :s] = samples
        self._log_weights = np.full(self.n_particles, -np.log(self.n_particles))
        self._initialized = True

    def _state_transition(self, particles: np.ndarray, dt: float,
                           accel: np.ndarray, gyro: np.ndarray) -> np.ndarray:
        """Propagate all particles through dynamics model."""
        new_p = particles.copy()
        roll = particles[:, 6]
        pitch = particles[:, 7]
        yaw = particles[:, 8]

        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        # Rotation matrix per particle (vectorized)
        gravity = np.array([0.0, 0.0, 9.81])

        # Simple flat-earth approximation for particle propagation
        accel_world = np.zeros((self.n_particles, 3))
        accel_world[:, 0] = cp * cy * accel[0] + (sr*sp*cy - cr*sy) * accel[1] + (cr*sp*cy + sr*sy) * accel[2] - gravity[0]
        accel_world[:, 1] = cp * sy * accel[0] + (sr*sp*sy + cr*cy) * accel[1] + (cr*sp*sy - sr*cy) * accel[2] - gravity[1]
        accel_world[:, 2] = -sp * accel[0] + sr*cp * accel[1] + cr*cp * accel[2] - gravity[2]

        new_p[:, 0:3] = particles[:, 0:3] + particles[:, 3:6] * dt + 0.5 * accel_world * dt**2
        new_p[:, 3:6] = particles[:, 3:6] + accel_world * dt
        new_p[:, 6] = roll + gyro[0] * dt
        new_p[:, 7] = pitch + gyro[1] * dt
        new_p[:, 8] = yaw + gyro[2] * dt
        return new_p

    def predict(self, dt: float, imu_accel: np.ndarray,
                imu_gyro: np.ndarray, process_noise: np.ndarray = None) -> None:
        """Particle filter prediction step with process noise injection."""
        if not self._initialized:
            self.initialize_particles(np.zeros(self.state_dim), np.eye(self.state_dim))

        self._particles = self._state_transition(self._particles, dt, imu_accel, imu_gyro)

        # Add process noise
        noise_std = process_noise if process_noise is not None else self._process_noise
        noise = self._rng.standard_normal((self.n_particles, self.state_dim)) * noise_std
        self._particles += noise

    def update(self, measurement: np.ndarray,
               measurement_fn: Callable = None,
               R: np.ndarray = None) -> None:
        """Particle filter update: compute likelihoods and update log-weights.

        Args:
            measurement: Observation vector (obs_dim,)
            measurement_fn: h(particle) -> predicted measurement
            R: Measurement noise covariance
        """
        if not self._initialized:
            return

        obs_dim = len(measurement)
        if R is None:
            R = np.eye(obs_dim) * self.measurement_noise
        if measurement_fn is None:
            def measurement_fn(p):
                return p[:obs_dim]

        try:
            R_inv = np.linalg.inv(R)
            log_det_R = np.log(np.linalg.det(R + np.eye(obs_dim) * 1e-9))
        except np.linalg.LinAlgError:
            R_inv = np.eye(obs_dim) / self.measurement_noise
            log_det_R = obs_dim * np.log(self.measurement_noise)

        # Compute log-likelihood for each particle
        log_likelihoods = np.zeros(self.n_particles)
        for i, particle in enumerate(self._particles):
            predicted = measurement_fn(particle)
            residual = measurement - predicted
            log_likelihoods[i] = -0.5 * (residual @ R_inv @ residual + log_det_R)

        # Update log-weights
        self._log_weights = self._log_weights + log_likelihoods
        # Normalize in log domain
        log_sum = np.logaddexp.reduce(self._log_weights)
        self._log_weights -= log_sum

        # Resample if effective sample size is too low
        if self._effective_sample_size() < self.resample_threshold * self.n_particles:
            self.resample()

    def resample(self) -> None:
        """Systematic resampling."""
        weights = np.exp(self._log_weights)
        weights /= weights.sum()

        cumsum = np.cumsum(weights)
        positions = (self._rng.random() + np.arange(self.n_particles)) / self.n_particles
        indices = np.searchsorted(cumsum, positions)
        indices = np.clip(indices, 0, self.n_particles - 1)

        self._particles = self._particles[indices]
        self._log_weights = np.full(self.n_particles, -np.log(self.n_particles))

    def _effective_sample_size(self) -> float:
        """Compute effective sample size."""
        log_w = self._log_weights - np.max(self._log_weights)
        w = np.exp(log_w)
        w /= w.sum()
        return float(1.0 / np.sum(w**2))

    def get_estimate(self) -> np.ndarray:
        """Return weighted mean state estimate."""
        weights = np.exp(self._log_weights)
        weights /= weights.sum()
        return np.sum(weights[:, np.newaxis] * self._particles, axis=0)

    def get_covariance(self) -> np.ndarray:
        """Return weighted covariance of particle distribution."""
        weights = np.exp(self._log_weights)
        weights /= weights.sum()
        mean = self.get_estimate()
        diff = self._particles - mean
        return np.einsum("i,ij,ik->jk", weights, diff, diff)
