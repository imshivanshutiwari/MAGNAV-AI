"""Unscented Kalman Filter for magnetic navigation."""
import numpy as np
from typing import Callable, Tuple


class UnscentedKalmanFilter:
    """15-state UKF using scaled unscented transform.

    State: [x, y, z, vx, vy, vz, roll, pitch, yaw, bgx, bgy, bgz, bax, bay, baz]
    """

    def __init__(self,
                 state_dim: int = 15,
                 obs_dim: int = 3,
                 alpha: float = 0.001,
                 beta: float = 2.0,
                 kappa: float = 0.0,
                 process_noise_pos: float = 0.01,
                 process_noise_vel: float = 0.1,
                 process_noise_att: float = 0.001,
                 process_noise_bias: float = 1e-5,
                 measurement_noise: float = 25.0):
        self.n = state_dim
        self.obs_dim = obs_dim
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        lam = alpha**2 * (state_dim + kappa) - state_dim
        self._lambda = lam

        # Sigma point weights
        n2 = 2 * state_dim + 1
        self._Wm = np.full(n2, 1.0 / (2 * (state_dim + lam)))
        self._Wm[0] = lam / (state_dim + lam)
        self._Wc = self._Wm.copy()
        self._Wc[0] += 1.0 - alpha**2 + beta

        # State and covariance
        self._x = np.zeros(state_dim)
        self._P = np.eye(state_dim) * 100.0

        # Process noise
        self._Q = np.diag(
            [process_noise_pos] * 3 +
            [process_noise_vel] * 3 +
            [process_noise_att] * 3 +
            [process_noise_bias] * 6
        )

        # Measurement noise
        self._R = np.eye(obs_dim) * measurement_noise
        self._gravity = np.array([0.0, 0.0, 9.81])

    def _compute_sigma_points(self) -> np.ndarray:
        """Compute 2n+1 sigma points from current state and covariance."""
        n = self.n
        lam = self._lambda
        sigma_pts = np.zeros((2 * n + 1, n))
        sigma_pts[0] = self._x

        # Cholesky of (n + lambda) * P with regularization
        M = (n + lam) * self._P
        M = 0.5 * (M + M.T) + np.eye(n) * 1e-9
        try:
            L = np.linalg.cholesky(M)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(M + np.eye(n) * 1e-6)

        for i in range(n):
            sigma_pts[i + 1] = self._x + L[:, i]
            sigma_pts[i + 1 + n] = self._x - L[:, i]
        return sigma_pts

    def _state_transition(self, x: np.ndarray, dt: float,
                           accel: np.ndarray, gyro: np.ndarray) -> np.ndarray:
        """Nonlinear state transition (same as EKF)."""
        x_new = x.copy()
        roll, pitch, yaw = x[6:9]
        bg = x[9:12]
        ba = x[12:15]

        gyro_c = gyro - bg
        accel_c = accel - ba

        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        R = np.array([
            [cp*cy, sr*sp*cy - cr*sy, cr*sp*cy + sr*sy],
            [cp*sy, sr*sp*sy + cr*cy, cr*sp*sy - sr*cy],
            [-sp, sr*cp, cr*cp],
        ])

        accel_world = R @ accel_c - self._gravity
        x_new[0:3] = x[0:3] + x[3:6] * dt + 0.5 * accel_world * dt**2
        x_new[3:6] = x[3:6] + accel_world * dt

        T_inv = np.array([
            [1, sr * np.tan(pitch), cr * np.tan(pitch)],
            [0, cr, -sr],
            [0, sr / (cp + 1e-9), cr / (cp + 1e-9)],
        ])
        euler_dot = T_inv @ gyro_c
        x_new[6:9] = x[6:9] + euler_dot * dt
        x_new[9:15] = x[9:15]
        return x_new

    def _unscented_transform(self, sigma_pts: np.ndarray,
                              Wm: np.ndarray, Wc: np.ndarray,
                              noise_cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute weighted mean and covariance from sigma points."""
        x_mean = np.sum(Wm[:, np.newaxis] * sigma_pts, axis=0)
        diff = sigma_pts - x_mean
        P = np.einsum("i,ij,ik->jk", Wc, diff, diff) + noise_cov
        return x_mean, P

    def predict(self, dt: float, imu_accel: np.ndarray,
                imu_gyro: np.ndarray) -> None:
        """UKF prediction step."""
        sigma_pts = self._compute_sigma_points()
        prop_pts = np.array([
            self._state_transition(sp, dt, imu_accel, imu_gyro)
            for sp in sigma_pts
        ])
        self._x, self._P = self._unscented_transform(
            prop_pts, self._Wm, self._Wc, self._Q * dt
        )

    def _observation_fn(self, x: np.ndarray) -> np.ndarray:
        """Default observation: return first 3 elements (position)."""
        return x[:3]

    def update(self, measurement: np.ndarray, R: np.ndarray = None,
               obs_fn: Callable = None) -> None:
        """UKF measurement update.

        Args:
            measurement: Observation vector
            R: Measurement noise covariance; uses stored R if None
            obs_fn: Observation function h(x); defaults to first 3 states
        """
        if R is None:
            R = self._R
        if obs_fn is None:
            obs_fn = self._observation_fn

        sigma_pts = self._compute_sigma_points()
        obs_sigma = np.array([obs_fn(sp) for sp in sigma_pts])

        z_pred = np.sum(self._Wm[:, np.newaxis] * obs_sigma, axis=0)
        dz = obs_sigma - z_pred
        dx = sigma_pts - self._x

        S = np.einsum("i,ij,ik->jk", self._Wc, dz, dz) + R
        Pxz = np.einsum("i,ij,ik->jk", self._Wc, dx, dz)

        K = Pxz @ np.linalg.inv(S)
        self._x = self._x + K @ (measurement - z_pred)
        self._P = self._P - K @ S @ K.T
        self._P = 0.5 * (self._P + self._P.T)

    def get_state(self) -> np.ndarray:
        """Return current state vector."""
        return self._x.copy()

    def get_covariance(self) -> np.ndarray:
        """Return current covariance matrix."""
        return self._P.copy()

    def set_state(self, x: np.ndarray, P: np.ndarray = None) -> None:
        """Set state and optionally covariance."""
        self._x = x.copy()
        if P is not None:
            self._P = P.copy()
