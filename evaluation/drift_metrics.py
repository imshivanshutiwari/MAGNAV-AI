"""Drift and navigation quality metrics."""
import numpy as np
from typing import List, Dict, Optional


class DriftMetrics:
    """Computes heading error, drift rates, consistency, and NEES."""

    def compute_heading_error(self, estimated_heading: np.ndarray,
                               true_heading: np.ndarray) -> np.ndarray:
        """Compute wrapped heading error in degrees.

        Args:
            estimated_heading: (N,) estimated headings in degrees
            true_heading: (N,) true headings in degrees

        Returns:
            (N,) heading errors in degrees, wrapped to [-180, 180]
        """
        diff = np.asarray(estimated_heading) - np.asarray(true_heading)
        return ((diff + 180.0) % 360.0) - 180.0

    def compute_position_drift(self, positions: np.ndarray,
                                times: np.ndarray) -> float:
        """Compute average drift rate (m/s) as total drift / total time.

        Args:
            positions: (N, 3) estimated positions
            times: (N,) timestamps in seconds

        Returns:
            Drift rate in m/s
        """
        total_time = float(times[-1] - times[0])
        if total_time < 1e-9:
            return 0.0
        total_displacement = float(np.linalg.norm(positions[-1] - positions[0]))
        return total_displacement / total_time

    def compute_drift_per_km(self, positions: np.ndarray,
                               distances: np.ndarray) -> float:
        """Compute drift per kilometer traveled.

        Args:
            positions: (N, 3) estimated positions
            distances: (N,) cumulative distances traveled (m)

        Returns:
            Drift in m/km
        """
        total_km = distances[-1] / 1000.0
        if total_km < 1e-6:
            return 0.0
        final_drift = float(np.linalg.norm(positions[-1] - positions[0]))
        return final_drift / total_km

    def compute_consistency_score(self, filter_estimates: List[np.ndarray],
                                   fused_estimate: np.ndarray) -> float:
        """Compute consistency score [0, 1] between individual filters and fused estimate.

        Score of 1 means all filters agree with the fused estimate.
        """
        if not filter_estimates:
            return 0.0
        dim = len(fused_estimate)
        dists = []
        for est in filter_estimates:
            n = min(len(est), dim)
            diff = est[:n] - fused_estimate[:n]
            dists.append(float(np.linalg.norm(diff)))
        mean_dist = np.mean(dists)
        fused_norm = float(np.linalg.norm(fused_estimate)) + 1e-9
        score = 1.0 / (1.0 + mean_dist / fused_norm)
        return float(np.clip(score, 0.0, 1.0))

    def compute_nees(self, state_error: np.ndarray,
                      covariance: np.ndarray) -> float:
        """Normalized Estimation Error Squared (NEES).

        NEES = error^T * P^{-1} * error / n

        A consistent filter should have NEES ≈ 1.

        Args:
            state_error: (n,) estimation error vector
            covariance: (n, n) estimated covariance

        Returns:
            NEES scalar
        """
        n = len(state_error)
        try:
            P_inv = np.linalg.inv(covariance + np.eye(n) * 1e-9)
            nees = float(state_error @ P_inv @ state_error) / n
        except np.linalg.LinAlgError:
            nees = float(np.dot(state_error, state_error)) / n
        return max(nees, 0.0)

    def generate_report(self, all_metrics: Dict) -> Dict:
        """Generate a summary report from collected metrics.

        Args:
            all_metrics: Dict with metric arrays over time

        Returns:
            Summary dict with mean/std/min/max for each metric
        """
        report = {}
        for key, values in all_metrics.items():
            arr = np.asarray(values, dtype=float)
            arr = arr[np.isfinite(arr)]
            if len(arr) == 0:
                continue
            report[key] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "median": float(np.median(arr)),
            }
        return report
