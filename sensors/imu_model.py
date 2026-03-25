"""IMU sensor model: 15-state strapdown INS with RK4 quaternion integration."""
import numpy as np
from typing import Optional


def _skew(v: np.ndarray) -> np.ndarray:
    """Return 3x3 skew-symmetric matrix for vector v."""
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])


def _quat_mult(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion multiplication q1 * q2 (scalar-last convention)."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Quaternion (scalar-last) to rotation matrix."""
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def _rot_to_euler(R: np.ndarray) -> np.ndarray:
    """Rotation matrix to Euler angles [roll, pitch, yaw] in radians."""
    pitch = np.arcsin(-R[2, 0])
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw])


class IMUModel:
    """15-state strapdown IMU model.

    State: [px, py, pz, vx, vy, vz, qx, qy, qz, qw, bgx, bgy, bgz, bax, bay]
    (position, velocity, quaternion, gyro bias, accel bias x/y only for compactness)
    """

    GRAVITY = np.array([0.0, 0.0, 9.81])

    def __init__(self,
                 accel_noise_std: float = 0.001,
                 gyro_noise_std: float = 0.0001,
                 accel_bias_std: float = 0.0001,
                 gyro_bias_std: float = 0.00001,
                 update_rate_hz: float = 100.0):
        self.accel_noise_std = accel_noise_std
        self.gyro_noise_std = gyro_noise_std
        self.accel_bias_std = accel_bias_std
        self.gyro_bias_std = gyro_bias_std
        self.dt = 1.0 / update_rate_hz

        # State: [px, py, pz, vx, vy, vz, qx, qy, qz, qw, bgx, bgy, bgz, bax, bay]
        self._state = np.zeros(15)
        self._state[9] = 1.0  # qw = 1 (identity quaternion, scalar-last)

        self._gyro_bias = np.zeros(3)
        self._accel_bias = np.zeros(2)
        self._rng = np.random.default_rng(0)

        # Calibration offsets
        self._accel_cal_offset = np.zeros(3)
        self._gyro_cal_offset = np.zeros(3)

    def reset(self, initial_pos: np.ndarray = None):
        """Reset IMU state."""
        self._state = np.zeros(15)
        self._state[9] = 1.0
        if initial_pos is not None:
            self._state[:3] = initial_pos

    def propagate_state(self, accel_body: np.ndarray, gyro_body: np.ndarray,
                        dt: float = None) -> np.ndarray:
        """Propagate state using noisy IMU measurements.

        Args:
            accel_body: True acceleration in body frame (m/s^2)
            gyro_body: True angular rate in body frame (rad/s)
            dt: Time step; uses default if None

        Returns:
            Updated state vector (15,)
        """
        if dt is None:
            dt = self.dt

        noisy_accel, noisy_gyro = self.add_noise(accel_body, gyro_body)

        pos = self._state[:3].copy()
        vel = self._state[3:6].copy()
        q = self._state[6:10].copy()
        bg = self._state[10:13].copy()
        ba = self._state[13:15].copy()

        # Correct for bias
        gyro_corrected = noisy_gyro - bg
        accel_corrected = noisy_accel.copy()
        accel_corrected[:2] -= ba

        # RK4 quaternion integration
        def q_dot(q_, omega):
            ox, oy, oz = omega
            Omega = np.array([
                [0, oz, -oy, ox],
                [-oz, 0, ox, oy],
                [oy, -ox, 0, oz],
                [-ox, -oy, -oz, 0],
            ])
            return 0.5 * Omega @ q_

        k1 = q_dot(q, gyro_corrected)
        k2 = q_dot(q + 0.5 * dt * k1, gyro_corrected)
        k3 = q_dot(q + 0.5 * dt * k2, gyro_corrected)
        k4 = q_dot(q + dt * k3, gyro_corrected)
        q_new = q + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        q_new /= np.linalg.norm(q_new)

        # Rotate accel to world frame and remove gravity
        R = _quat_to_rot(q)
        accel_world = R @ accel_corrected - self.GRAVITY

        # Integrate position and velocity
        vel_new = vel + accel_world * dt
        pos_new = pos + vel * dt + 0.5 * accel_world * dt ** 2

        # Random-walk bias evolution
        bg_new = bg + self._rng.standard_normal(3) * self.gyro_bias_std * np.sqrt(dt)
        ba_new = ba + self._rng.standard_normal(2) * self.accel_bias_std * np.sqrt(dt)

        self._state[:3] = pos_new
        self._state[3:6] = vel_new
        self._state[6:10] = q_new
        self._state[10:13] = bg_new
        self._state[13:15] = ba_new
        return self._state.copy()

    def add_noise(self, accel: np.ndarray, gyro: np.ndarray) -> tuple:
        """Add realistic IMU noise to measurements."""
        noisy_accel = accel + self._rng.standard_normal(3) * self.accel_noise_std
        noisy_gyro = gyro + self._rng.standard_normal(3) * self.gyro_noise_std
        return noisy_accel, noisy_gyro

    def calibrate(self, static_accel_samples: np.ndarray,
                  static_gyro_samples: np.ndarray):
        """One-position static calibration."""
        self._accel_cal_offset = np.mean(static_accel_samples, axis=0) - np.array([0, 0, 9.81])
        self._gyro_cal_offset = np.mean(static_gyro_samples, axis=0)

    def get_state(self) -> np.ndarray:
        """Return current state vector."""
        return self._state.copy()

    def get_euler_angles(self) -> np.ndarray:
        """Return [roll, pitch, yaw] from current quaternion."""
        R = _quat_to_rot(self._state[6:10])
        return _rot_to_euler(R)
