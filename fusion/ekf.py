"""Extended Kalman Filter for magnetic navigation."""
import numpy as np
from typing import Tuple


class ExtendedKalmanFilter:
    """15-state Extended Kalman Filter for INS/magnetic navigation.

    State: [x, y, z, vx, vy, vz, roll, pitch, yaw, bgx, bgy, bgz, bax, bay, baz]
    """

    def __init__(self,
                 state_dim: int = 15,
                 obs_dim: int = 3,
                 process_noise_pos: float = 0.01,
                 process_noise_vel: float = 0.1,
                 process_noise_att: float = 0.001,
                 process_noise_bias: float = 1e-5,
                 measurement_noise: float = 25.0):
        self.state_dim = state_dim
        self.obs_dim = obs_dim

        # State vector [x,y,z,vx,vy,vz,roll,pitch,yaw,bgx,bgy,bgz,bax,bay,baz]
        self._x = np.zeros(state_dim)
        # Covariance matrix
        self._P = np.eye(state_dim) * 100.0

        # Process noise covariance
        self._Q = np.diag(
            [process_noise_pos] * 3 +
            [process_noise_vel] * 3 +
            [process_noise_att] * 3 +
            [process_noise_bias] * 6
        )

        # Measurement noise covariance
        self._R = np.eye(obs_dim) * measurement_noise

        self._gravity = np.array([0.0, 0.0, 9.81])

    def _f(self, x: np.ndarray, dt: float,
           accel: np.ndarray, gyro: np.ndarray) -> np.ndarray:
        """State transition function f(x, u)."""
        x_new = x.copy()
        roll, pitch, yaw = x[6:9]
        bg = x[9:12]
        ba = x[12:15]

        # Correct for biases
        gyro_c = gyro - bg
        accel_c = accel - ba

        # Rotation matrix body -> world
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        R = np.array([
            [cp*cy, sr*sp*cy - cr*sy, cr*sp*cy + sr*sy],
            [cp*sy, sr*sp*sy + cr*cy, cr*sp*sy - sr*cy],
            [-sp, sr*cp, cr*cp],
        ])

        # Acceleration in world frame
        accel_world = R @ accel_c - self._gravity

        # Velocity and position
        x_new[0:3] = x[0:3] + x[3:6] * dt + 0.5 * accel_world * dt**2
        x_new[3:6] = x[3:6] + accel_world * dt

        # Attitude (Euler angle integration - approximate)
        T_inv = np.array([
            [1, sr*np.tan(pitch), cr*np.tan(pitch)],
            [0, cr, -sr],
            [0, sr/(cp + 1e-9), cr/(cp + 1e-9)],
        ])
        euler_dot = T_inv @ gyro_c
        x_new[6:9] = x[6:9] + euler_dot * dt

        # Biases: random walk (no dynamics)
        x_new[9:15] = x[9:15]
        return x_new

    def _compute_jacobian(self, x: np.ndarray, dt: float,
                           accel: np.ndarray, gyro: np.ndarray) -> np.ndarray:
        """Numerical Jacobian via central differences."""
        eps = 1e-5
        F = np.zeros((self.state_dim, self.state_dim))
        f0 = self._f(x, dt, accel, gyro)
        for i in range(self.state_dim):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            f_plus = self._f(x_plus, dt, accel, gyro)
            f_minus = self._f(x_minus, dt, accel, gyro)
            F[:, i] = (f_plus - f_minus) / (2 * eps)
        return F

    def predict(self, dt: float, imu_accel: np.ndarray,
                imu_gyro: np.ndarray) -> None:
        """EKF prediction step.

        Args:
            dt: Time step (s)
            imu_accel: Accelerometer reading (3,) m/s^2
            imu_gyro: Gyroscope reading (3,) rad/s
        """
        F = self._compute_jacobian(self._x, dt, imu_accel, imu_gyro)
        self._x = self._f(self._x, dt, imu_accel, imu_gyro)
        self._P = F @ self._P @ F.T + self._Q * dt

    def update(self, measurement: np.ndarray,
               H: np.ndarray, R: np.ndarray = None) -> None:
        """EKF measurement update step (Joseph form).

        Args:
            measurement: Observation vector (obs_dim,)
            H: Observation matrix (obs_dim, state_dim)
            R: Measurement noise covariance; uses stored R if None
        """
        if R is None:
            R = self._R

        innovation = measurement - H @ self._x
        S = H @ self._P @ H.T + R
        K = self._P @ H.T @ np.linalg.inv(S)
        self._x = self._x + K @ innovation

        # Joseph form for numerical stability
        I_KH = np.eye(self.state_dim) - K @ H
        self._P = I_KH @ self._P @ I_KH.T + K @ R @ K.T

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
