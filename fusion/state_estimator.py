"""State estimator: fuse EKF, UKF, and PF via Covariance Intersection."""
import numpy as np
from typing import List, Optional


class StateEstimator:
    """Fuses estimates from EKF, UKF, and Particle Filter using Covariance Intersection."""

    def __init__(self, ekf=None, ukf=None, pf=None):
        self._ekf = ekf
        self._ukf = ukf
        self._pf = pf
        self._fused_state: Optional[np.ndarray] = None
        self._fused_cov: Optional[np.ndarray] = None
        self._weights: Optional[np.ndarray] = None

    def compute_filter_weights(self, covs: List[np.ndarray]) -> np.ndarray:
        """Compute fusion weights based on inverse trace of each covariance.

        Args:
            covs: List of covariance matrices

        Returns:
            Normalized weights array
        """
        traces = np.array([np.trace(c) for c in covs])
        # Avoid division by zero
        traces = np.maximum(traces, 1e-12)
        inv_traces = 1.0 / traces
        return inv_traces / inv_traces.sum()

    def fuse_estimates(self,
                       states: List[np.ndarray],
                       covs: List[np.ndarray]) -> tuple:
        """Covariance Intersection fusion of multiple estimates.

        Args:
            states: List of state vectors (possibly different dimensions)
            covs: List of corresponding covariance matrices

        Returns:
            (fused_state, fused_covariance)
        """
        if not states:
            raise ValueError("No estimates to fuse.")

        # Use minimum common state dimension
        min_dim = min(s.shape[0] for s in states)
        states_common = [s[:min_dim] for s in states]
        covs_common = [c[:min_dim, :min_dim] for c in covs]

        weights = self.compute_filter_weights(covs_common)
        self._weights = weights

        # Covariance Intersection
        P_fused_inv = np.zeros((min_dim, min_dim))
        x_fused_accum = np.zeros(min_dim)

        for w, x, P in zip(weights, states_common, covs_common):
            P_reg = P + np.eye(min_dim) * 1e-9
            try:
                P_inv = np.linalg.inv(P_reg)
            except np.linalg.LinAlgError:
                P_inv = np.eye(min_dim) / (np.trace(P_reg) / min_dim + 1e-9)
            P_fused_inv += w * P_inv
            x_fused_accum += w * P_inv @ x

        try:
            P_fused = np.linalg.inv(P_fused_inv + np.eye(min_dim) * 1e-12)
        except np.linalg.LinAlgError:
            P_fused = np.eye(min_dim) * 1e4

        x_fused = P_fused @ x_fused_accum

        # Pad to max dimension if needed
        max_dim = max(s.shape[0] for s in states)
        if max_dim > min_dim:
            x_full = np.zeros(max_dim)
            P_full = np.eye(max_dim) * 1e4
            x_full[:min_dim] = x_fused
            P_full[:min_dim, :min_dim] = P_fused
            # Fill remaining dimensions from highest-weight estimate
            best_idx = int(np.argmax(weights))
            best_state = states[best_idx]
            if len(best_state) > min_dim:
                x_full[min_dim:max_dim] = best_state[min_dim:max_dim]
            self._fused_state = x_full
            self._fused_cov = P_full
        else:
            self._fused_state = x_fused
            self._fused_cov = P_fused

        return self._fused_state.copy(), self._fused_cov.copy()

    def step(self) -> tuple:
        """Collect states from all filters and return fused estimate."""
        states = []
        covs = []

        if self._ekf is not None:
            states.append(self._ekf.get_state())
            covs.append(self._ekf.get_covariance())
        if self._ukf is not None:
            states.append(self._ukf.get_state())
            covs.append(self._ukf.get_covariance())
        if self._pf is not None:
            states.append(self._pf.get_estimate())
            covs.append(self._pf.get_covariance())

        if not states:
            raise RuntimeError("No filters attached to StateEstimator.")

        return self.fuse_estimates(states, covs)

    def get_fused_state(self) -> Optional[np.ndarray]:
        """Return last fused state."""
        return self._fused_state.copy() if self._fused_state is not None else None

    def get_fused_covariance(self) -> Optional[np.ndarray]:
        """Return last fused covariance."""
        return self._fused_cov.copy() if self._fused_cov is not None else None

    def get_filter_weights(self) -> Optional[np.ndarray]:
        """Return last computed filter weights."""
        return self._weights.copy() if self._weights is not None else None
