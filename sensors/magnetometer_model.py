"""Magnetometer sensor model with hard/soft iron calibration."""
import numpy as np
from typing import Optional


class MagnetometerModel:
    """Simulates a 3-axis magnetometer with calibration support."""

    def __init__(self,
                 noise_std: float = 5.0,
                 hard_iron: Optional[np.ndarray] = None,
                 soft_iron: Optional[np.ndarray] = None,
                 update_rate_hz: float = 10.0):
        self.noise_std = noise_std
        self.hard_iron = np.array(hard_iron) if hard_iron is not None else np.array([20.0, -15.0, 10.0])
        self.soft_iron = np.array(soft_iron) if soft_iron is not None else np.eye(3)
        self.update_rate_hz = update_rate_hz
        self._rng = np.random.default_rng(1)

        # Calibration matrices (initialized to identity / zero)
        self._cal_hard_iron = np.zeros(3)
        self._cal_soft_iron = np.eye(3)

    def measure(self, Bx: float, By: float, Bz: float,
                attitude_rpy: np.ndarray = None) -> np.ndarray:
        """Simulate magnetometer measurement in sensor frame.

        Args:
            Bx, By, Bz: True field in NED frame (nT)
            attitude_rpy: [roll, pitch, yaw] in radians

        Returns:
            Noisy measurement vector (3,) in nT
        """
        B_ned = np.array([Bx, By, Bz])

        if attitude_rpy is not None:
            roll, pitch, yaw = attitude_rpy
            # Rotation matrix NED -> body
            cr, sr = np.cos(roll), np.sin(roll)
            cp, sp = np.cos(pitch), np.sin(pitch)
            cy, sy = np.cos(yaw), np.sin(yaw)
            R_body = np.array([
                [cp*cy, cp*sy, -sp],
                [sr*sp*cy - cr*sy, sr*sp*sy + cr*cy, sr*cp],
                [cr*sp*cy + sr*sy, cr*sp*sy - sr*cy, cr*cp],
            ])
            B_body = R_body @ B_ned
        else:
            B_body = B_ned.copy()

        # Apply hard/soft iron distortions
        B_distorted = self.soft_iron @ B_body + self.hard_iron

        # Add white noise
        noise = self._rng.standard_normal(3) * self.noise_std
        return B_distorted + noise

    def calibrate(self, samples: np.ndarray) -> tuple:
        """Estimate and remove hard/soft iron effects from samples.

        Uses sphere-fitting for hard iron offset, then ellipsoid correction.

        Args:
            samples: (N, 3) array of uncalibrated magnetometer readings

        Returns:
            (hard_iron_offset, soft_iron_matrix)
        """
        if samples.shape[0] < 10:
            return np.zeros(3), np.eye(3)

        # Hard iron: center of the measurement ellipsoid
        hard_iron_est = np.mean(samples, axis=0)

        # Soft iron: covariance-based scaling
        centered = samples - hard_iron_est
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        eigenvalues = np.maximum(eigenvalues, 1e-9)
        scale = np.mean(np.sqrt(eigenvalues))
        soft_iron_est = eigenvectors @ np.diag(scale / np.sqrt(eigenvalues)) @ eigenvectors.T

        self._cal_hard_iron = hard_iron_est
        self._cal_soft_iron = soft_iron_est
        return hard_iron_est, soft_iron_est

    def apply_calibration(self, raw: np.ndarray) -> np.ndarray:
        """Apply stored calibration to raw measurement."""
        return self._cal_soft_iron @ (raw - self._cal_hard_iron)

    def estimate_attitude_from_field(self, B_body: np.ndarray,
                                     B_ref_ned: np.ndarray) -> tuple:
        """Estimate pitch and roll from magnetometer and known reference field.

        Returns:
            (pitch_rad, roll_rad) using tilt-compensated method
        """
        B_norm = np.linalg.norm(B_body)
        if B_norm < 1e-9:
            return 0.0, 0.0
        b = B_body / B_norm

        pitch = float(np.arcsin(-b[0]))
        roll = float(np.arctan2(b[1], b[2]))
        return pitch, roll

    def compute_heading(self, B_body: np.ndarray, roll: float, pitch: float) -> float:
        """Compute tilt-compensated magnetic heading.

        Args:
            B_body: Magnetometer reading in body frame (nT)
            roll: Roll angle in radians
            pitch: Pitch angle in radians

        Returns:
            Magnetic heading in radians [0, 2*pi)
        """
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)

        # Project B onto horizontal plane
        Bh = B_body[0] * cp + B_body[1] * sr * sp + B_body[2] * cr * sp
        Bv = B_body[1] * cr - B_body[2] * sr

        heading = np.arctan2(-Bv, Bh)
        return float(heading % (2 * np.pi))
