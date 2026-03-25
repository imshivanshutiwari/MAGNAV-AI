"""Residual feature extraction for sensor fusion."""
import numpy as np
from scipy.stats import kurtosis


class ResidualFeatureExtractor:
    """Computes residuals and statistical features for sensor fusion."""

    def compute_residual(self, measured: np.ndarray,
                          expected: np.ndarray) -> np.ndarray:
        """Compute residual vector: measured - expected."""
        return np.asarray(measured, dtype=float) - np.asarray(expected, dtype=float)

    def compute_normalized_residual(self, measured: np.ndarray,
                                     expected: np.ndarray,
                                     covariance: np.ndarray) -> np.ndarray:
        """Compute normalized innovation residual S^{-1/2} * (z - h(x)).

        Args:
            measured: Measurement vector (m,)
            expected: Expected measurement (m,)
            covariance: Innovation covariance (m, m)

        Returns:
            Normalized residual (m,)
        """
        residual = self.compute_residual(measured, expected)
        try:
            # Cholesky decomposition for numerical stability
            L = np.linalg.cholesky(covariance + np.eye(len(residual)) * 1e-9)
            return np.linalg.solve(L, residual)
        except np.linalg.LinAlgError:
            # Fallback: eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(covariance)
            eigvals = np.maximum(eigvals, 1e-9)
            S_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            return S_inv_sqrt @ residual

    def compute_mahalanobis_distance(self, residual: np.ndarray,
                                      covariance: np.ndarray) -> float:
        """Compute Mahalanobis distance: sqrt(r^T S^{-1} r).

        Args:
            residual: Innovation vector (m,)
            covariance: Innovation covariance (m, m)

        Returns:
            Scalar Mahalanobis distance
        """
        try:
            S_inv = np.linalg.inv(covariance + np.eye(len(residual)) * 1e-9)
            dist_sq = float(residual @ S_inv @ residual)
        except np.linalg.LinAlgError:
            dist_sq = float(np.dot(residual, residual))
        return float(np.sqrt(max(dist_sq, 0.0)))

    def sliding_window_stats(self, residuals: np.ndarray,
                              window: int = 50) -> dict:
        """Compute sliding window statistics on residuals.

        Args:
            residuals: (N, D) or (N,) residual array
            window: Window size

        Returns:
            Dict with 'mean', 'std', 'kurtosis' arrays of length N
        """
        r = np.atleast_2d(residuals)
        if residuals.ndim == 1:
            r = r.T
        N, D = r.shape
        mean_out = np.zeros((N, D))
        std_out = np.zeros((N, D))
        kurt_out = np.zeros((N, D))

        for i in range(N):
            start = max(0, i - window + 1)
            window_data = r[start:i+1, :]
            mean_out[i] = np.mean(window_data, axis=0)
            std_out[i] = np.std(window_data, axis=0)
            if len(window_data) >= 4:
                kurt_out[i] = kurtosis(window_data, axis=0, bias=False)
            else:
                kurt_out[i] = 0.0

        if residuals.ndim == 1:
            return {"mean": mean_out[:, 0], "std": std_out[:, 0], "kurtosis": kurt_out[:, 0]}
        return {"mean": mean_out, "std": std_out, "kurtosis": kurt_out}
