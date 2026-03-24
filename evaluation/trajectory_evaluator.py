"""Trajectory evaluation metrics."""
import numpy as np
from typing import Dict


class TrajectoryEvaluator:
    """Computes trajectory accuracy metrics for navigation filter comparison."""

    def compute_rmse(self, estimated: np.ndarray,
                      ground_truth: np.ndarray) -> float:
        """Root Mean Square Error over trajectory.

        Args:
            estimated: (N, 3) estimated positions [x, y, z]
            ground_truth: (N, 3) true positions

        Returns:
            RMSE in meters
        """
        errors = estimated - ground_truth
        return float(np.sqrt(np.mean(np.sum(errors**2, axis=-1))))

    def compute_mae(self, estimated: np.ndarray,
                     ground_truth: np.ndarray) -> float:
        """Mean Absolute Error over trajectory."""
        errors = np.linalg.norm(estimated - ground_truth, axis=-1)
        return float(np.mean(errors))

    def compute_3d_error(self, estimated: np.ndarray,
                          ground_truth: np.ndarray) -> Dict[str, np.ndarray]:
        """Per-axis and total 3D error.

        Returns:
            Dict with 'x', 'y', 'z', 'total' error arrays
        """
        diff = estimated - ground_truth
        return {
            "x": np.abs(diff[:, 0]),
            "y": np.abs(diff[:, 1]),
            "z": np.abs(diff[:, 2]) if diff.shape[1] > 2 else np.zeros(len(diff)),
            "total": np.linalg.norm(diff, axis=-1),
        }

    def compute_drift_per_km(self, estimated: np.ndarray,
                               ground_truth: np.ndarray) -> float:
        """Compute drift normalized by distance traveled (m/km).

        Args:
            estimated: (N, 3) estimated positions
            ground_truth: (N, 3) true positions

        Returns:
            Drift in meters per kilometer
        """
        total_distance_m = float(np.sum(np.linalg.norm(np.diff(ground_truth, axis=0), axis=-1)))
        final_error_m = float(np.linalg.norm(estimated[-1] - ground_truth[-1]))
        if total_distance_m < 1.0:
            return 0.0
        return final_error_m / (total_distance_m / 1000.0)

    def compare_trajectories(self,
                               ekf_traj: np.ndarray,
                               ukf_traj: np.ndarray,
                               pf_traj: np.ndarray,
                               fused_traj: np.ndarray,
                               ground_truth: np.ndarray) -> Dict[str, Dict]:
        """Compare all filter trajectories against ground truth.

        Returns:
            Nested dict {filter_name: {metric: value}}
        """
        results = {}
        filters = {"EKF": ekf_traj, "UKF": ukf_traj, "PF": pf_traj, "Fused": fused_traj}
        for name, traj in filters.items():
            n = min(len(traj), len(ground_truth))
            gt = ground_truth[:n, :3]
            est = traj[:n, :3]
            results[name] = {
                "rmse_m": self.compute_rmse(est, gt),
                "mae_m": self.compute_mae(est, gt),
                "drift_m_per_km": self.compute_drift_per_km(est, gt),
            }
        return results
