"""Magnetic feature extraction from field components."""
import numpy as np
from typing import Dict


class MagneticFeatureExtractor:
    """Extracts scalar and derived features from 3-axis magnetic field data."""

    def compute_intensity(self, Bx: float, By: float, Bz: float) -> float:
        """Compute total field intensity |B|."""
        return float(np.sqrt(Bx**2 + By**2 + Bz**2))

    def compute_inclination(self, Bx: float, By: float, Bz: float) -> float:
        """Compute field inclination angle (dip angle) in radians.

        Positive downward (positive in northern hemisphere).
        """
        Bh = np.sqrt(Bx**2 + By**2)
        return float(np.arctan2(Bz, Bh))

    def compute_declination(self, Bx: float, By: float, Bz: float) -> float:
        """Compute magnetic declination angle in radians.

        Angle between geographic north and magnetic north.
        """
        return float(np.arctan2(By, Bx))

    def compute_horizontal_intensity(self, Bx: float, By: float, Bz: float) -> float:
        """Compute horizontal component magnitude."""
        return float(np.sqrt(Bx**2 + By**2))

    def compute_vertical_intensity(self, Bx: float, By: float, Bz: float) -> float:
        """Compute vertical component (Bz)."""
        return float(abs(Bz))

    def extract_all(self, Bx: float, By: float, Bz: float) -> Dict[str, float]:
        """Extract all magnetic features.

        Returns:
            Dict with keys: Bx, By, Bz, intensity, inclination, declination,
                            horizontal_intensity, vertical_intensity
        """
        return {
            "Bx": float(Bx),
            "By": float(By),
            "Bz": float(Bz),
            "intensity": self.compute_intensity(Bx, By, Bz),
            "inclination": self.compute_inclination(Bx, By, Bz),
            "declination": self.compute_declination(Bx, By, Bz),
            "horizontal_intensity": self.compute_horizontal_intensity(Bx, By, Bz),
            "vertical_intensity": self.compute_vertical_intensity(Bx, By, Bz),
        }

    def extract_batch(self, B: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract features for a batch of measurements.

        Args:
            B: (N, 3) array of [Bx, By, Bz]

        Returns:
            Dict of feature arrays each of length N
        """
        Bx, By, Bz = B[:, 0], B[:, 1], B[:, 2]
        intensity = np.sqrt(Bx**2 + By**2 + Bz**2)
        Bh = np.sqrt(Bx**2 + By**2)
        return {
            "Bx": Bx,
            "By": By,
            "Bz": Bz,
            "intensity": intensity,
            "inclination": np.arctan2(Bz, Bh),
            "declination": np.arctan2(By, Bx),
            "horizontal_intensity": Bh,
            "vertical_intensity": np.abs(Bz),
        }
