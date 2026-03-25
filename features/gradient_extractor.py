"""Spatial and temporal gradient extraction for magnetic field maps."""
import numpy as np


class GradientExtractor:
    """Computes spatial and temporal gradients of magnetic field data."""

    def compute_spatial_gradient(self, field_map: np.ndarray,
                                  lat_grid: np.ndarray,
                                  lon_grid: np.ndarray) -> tuple:
        """Compute spatial gradients of a 2D field map.

        Args:
            field_map: (nlat, nlon) 2D array of field values
            lat_grid: (nlat,) latitude values in degrees
            lon_grid: (nlon,) longitude values in degrees

        Returns:
            (dB_dlat, dB_dlon) gradient arrays, each (nlat, nlon)
        """
        # Convert degree spacing to meters
        lat_spacing_m = np.mean(np.diff(lat_grid)) * 111320.0
        lon_spacing_m = np.mean(np.diff(lon_grid)) * 111320.0 * np.cos(np.deg2rad(np.mean(lat_grid)))

        dB_dlat = np.gradient(field_map, lat_spacing_m, axis=0)
        dB_dlon = np.gradient(field_map, lon_spacing_m, axis=1)
        return dB_dlat, dB_dlon

    def compute_temporal_gradient(self, B_history: np.ndarray, dt: float) -> np.ndarray:
        """Compute temporal gradient dB/dt from time series.

        Args:
            B_history: (N, 3) or (N,) array of field values over time
            dt: Time step in seconds

        Returns:
            (N, 3) or (N,) gradient array dB/dt
        """
        return np.gradient(B_history, dt, axis=0)

    def compute_gradient_magnitude(self, dB_dlat: np.ndarray,
                                    dB_dlon: np.ndarray) -> np.ndarray:
        """Compute total gradient magnitude from lat/lon components.

        Args:
            dB_dlat: Gradient in latitude direction
            dB_dlon: Gradient in longitude direction

        Returns:
            Total gradient magnitude array
        """
        return np.sqrt(dB_dlat**2 + dB_dlon**2)

    def compute_laplacian(self, field_map: np.ndarray,
                           lat_grid: np.ndarray,
                           lon_grid: np.ndarray) -> np.ndarray:
        """Compute 2D Laplacian of the field map."""
        dB_dlat, dB_dlon = self.compute_spatial_gradient(field_map, lat_grid, lon_grid)
        d2B_dlat2, _ = self.compute_spatial_gradient(dB_dlat, lat_grid, lon_grid)
        _, d2B_dlon2 = self.compute_spatial_gradient(dB_dlon, lat_grid, lon_grid)
        return d2B_dlat2 + d2B_dlon2
