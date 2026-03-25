"""Data preprocessing utilities."""
import numpy as np
from scipy.signal import savgol_filter
from typing import Optional, Dict, List


class DataPreprocessor:
    """Preprocessing pipeline: normalize, outlier rejection, smoothing, alignment."""

    def __init__(self):
        self._means: Optional[np.ndarray] = None
        self._stds: Optional[np.ndarray] = None

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Z-score normalize along axis 0. Stores params for denormalization."""
        self._means = np.mean(data, axis=0)
        self._stds = np.std(data, axis=0)
        self._stds[self._stds < 1e-9] = 1.0
        return (data - self._means) / self._stds

    def denormalize(self, data: np.ndarray) -> np.ndarray:
        """Reverse z-score normalization using stored params."""
        if self._means is None or self._stds is None:
            raise RuntimeError("Call normalize() before denormalize().")
        return data * self._stds + self._means

    def filter_outliers(self, data: np.ndarray, iqr_factor: float = 3.0,
                        strategy: str = "nan") -> np.ndarray:
        """Remove outliers using IQR method.

        Args:
            data: Input array (N, D) or (N,)
            iqr_factor: Multiplier for IQR bounds
            strategy: 'nan' | 'clip' | 'median'

        Returns:
            Array with outliers handled
        """
        out = data.copy().astype(float)
        if data.ndim == 1:
            out = out[:, np.newaxis]
        for col in range(out.shape[1]):
            col_data = out[:, col]
            q1 = np.nanpercentile(col_data, 25)
            q3 = np.nanpercentile(col_data, 75)
            iqr = q3 - q1
            lo = q1 - iqr_factor * iqr
            hi = q3 + iqr_factor * iqr
            mask = (col_data < lo) | (col_data > hi)
            if strategy == "nan":
                col_data[mask] = np.nan
            elif strategy == "clip":
                col_data[mask] = np.clip(col_data[mask], lo, hi)
            elif strategy == "median":
                col_data[mask] = np.nanmedian(col_data)
            out[:, col] = col_data
        if data.ndim == 1:
            return out[:, 0]
        return out

    def smooth(self, data: np.ndarray, window_length: int = 11,
               polyorder: int = 3) -> np.ndarray:
        """Apply Savitzky-Golay smoothing with NaN gap-fill."""
        out = data.copy().astype(float)
        if out.ndim == 1:
            out = out[:, np.newaxis]
        wl = min(window_length, len(out) - 1)
        if wl % 2 == 0:
            wl -= 1
        wl = max(wl, polyorder + 2)
        for col in range(out.shape[1]):
            col_data = out[:, col]
            nan_mask = np.isnan(col_data)
            if nan_mask.any():
                indices = np.arange(len(col_data))
                valid = ~nan_mask
                col_data[nan_mask] = np.interp(indices[nan_mask], indices[valid], col_data[valid])
            out[:, col] = savgol_filter(col_data, wl, polyorder)
        if data.ndim == 1:
            return out[:, 0]
        return out

    def align_timestamps(self, streams: Dict[str, dict],
                         target_rate_hz: float = 100.0) -> Dict[str, np.ndarray]:
        """Align multiple streams to a common uniform time grid.

        Args:
            streams: Dict of {name: {'timestamps': np.ndarray, 'data': np.ndarray}}
            target_rate_hz: Target sample rate for output

        Returns:
            Dict of {name: resampled_data}
        """
        # Find common time range
        t_starts = [s["timestamps"][0] for s in streams.values()]
        t_ends = [s["timestamps"][-1] for s in streams.values()]
        t_start = max(t_starts)
        t_end = min(t_ends)
        dt = 1.0 / target_rate_hz
        t_common = np.arange(t_start, t_end, dt)

        result = {}
        for name, stream in streams.items():
            ts = stream["timestamps"]
            data = stream["data"]
            if data.ndim == 1:
                result[name] = np.interp(t_common, ts, data)
            else:
                aligned = np.zeros((len(t_common), data.shape[1]))
                for col in range(data.shape[1]):
                    aligned[:, col] = np.interp(t_common, ts, data[:, col])
                result[name] = aligned
        result["_timestamps"] = t_common
        return result
