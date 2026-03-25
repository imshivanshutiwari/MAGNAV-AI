"""EMAG2 geomagnetic anomaly map fetcher."""
import os
import logging
import numpy as np
from scipy.interpolate import RectBivariateSpline

logger = logging.getLogger(__name__)


class EMAG2Fetcher:
    """Fetches and interpolates EMAG2 geomagnetic anomaly data."""

    EARTH_RADIUS_KM = 6371.0
    DEFAULT_ALTITUDE_KM = 0.0

    def __init__(self, cache_dir: str = "data/cache/emag2", altitude_km: float = 0.0):
        self.cache_dir = cache_dir
        self.altitude_km = altitude_km
        os.makedirs(cache_dir, exist_ok=True)
        self._lat_grid = None
        self._lon_grid = None
        self._bx_grid = None
        self._by_grid = None
        self._bz_grid = None
        self._bx_interp = None
        self._by_interp = None
        self._bz_interp = None
        self._loaded = False

    def load_grid(self) -> bool:
        """Load EMAG2 grid from cache or synthesize from IGRF."""
        cache_path = os.path.join(self.cache_dir, "emag2_grid.npz")
        if os.path.exists(cache_path):
            try:
                data = np.load(cache_path)
                self._lat_grid = data["lat"]
                self._lon_grid = data["lon"]
                self._bx_grid = data["bx"]
                self._by_grid = data["by"]
                self._bz_grid = data["bz"]
                self._build_interpolators()
                self._loaded = True
                logger.info("Loaded EMAG2 grid from cache.")
                return True
            except Exception as e:
                logger.warning(f"Cache load failed: {e}, synthesizing fallback.")

        return self._synthesize_fallback(cache_path)

    def _synthesize_fallback(self, cache_path: str) -> bool:
        """Synthesize a geomagnetic field grid based on dipole + anomaly model."""
        logger.info("Synthesizing EMAG2 fallback grid from dipole model.")
        lat_step = 0.5
        lon_step = 0.5
        self._lat_grid = np.arange(-90.0, 90.0 + lat_step, lat_step)
        self._lon_grid = np.arange(-180.0, 180.0 + lon_step, lon_step)
        nlat = len(self._lat_grid)
        nlon = len(self._lon_grid)

        lon_mesh, lat_mesh = np.meshgrid(self._lon_grid, self._lat_grid)
        lat_r = np.deg2rad(lat_mesh)
        lon_r = np.deg2rad(lon_mesh)

        # Dipole field approximation (IGRF-like)
        B0 = 30000.0  # nT
        self._bz_grid = -2.0 * B0 * np.sin(lat_r)
        self._bx_grid = B0 * np.cos(lat_r)
        self._by_grid = 0.5 * B0 * np.sin(2.0 * lat_r) * np.sin(lon_r)

        # Add spatially-correlated anomaly
        rng = np.random.default_rng(42)
        for field in [self._bx_grid, self._by_grid, self._bz_grid]:
            noise = rng.standard_normal((nlat, nlon)) * 200.0
            # Smooth to create spatial correlation
            from scipy.ndimage import gaussian_filter
            field += gaussian_filter(noise, sigma=5)

        np.savez_compressed(
            cache_path,
            lat=self._lat_grid,
            lon=self._lon_grid,
            bx=self._bx_grid,
            by=self._by_grid,
            bz=self._bz_grid,
        )
        self._build_interpolators()
        self._loaded = True
        return True

    def _build_interpolators(self):
        """Build scipy spline interpolators for Bx, By, Bz."""
        lat = self._lat_grid
        lon = self._lon_grid
        self._bx_interp = RectBivariateSpline(lat, lon, self._bx_grid, kx=3, ky=3)
        self._by_interp = RectBivariateSpline(lat, lon, self._by_grid, kx=3, ky=3)
        self._bz_interp = RectBivariateSpline(lat, lon, self._bz_grid, kx=3, ky=3)

    def interpolate(self, lat: float, lon: float, alt: float = None) -> tuple:
        """Interpolate Bx, By, Bz at given lat/lon/alt in nT.

        Returns:
            (Bx, By, Bz) tuple in nT
        """
        if not self._loaded:
            self.load_grid()

        alt_km = alt if alt is not None else self.altitude_km
        # Altitude correction: field decreases with distance from Earth
        alt_factor = (self.EARTH_RADIUS_KM / (self.EARTH_RADIUS_KM + alt_km)) ** 3

        lat_c = np.clip(lat, self._lat_grid[0], self._lat_grid[-1])
        lon_c = np.clip(lon, self._lon_grid[0], self._lon_grid[-1])

        bx = float(self._bx_interp(lat_c, lon_c).item()) * alt_factor
        by = float(self._by_interp(lat_c, lon_c).item()) * alt_factor
        bz = float(self._bz_interp(lat_c, lon_c).item()) * alt_factor
        return bx, by, bz

    def compute_gradient(self, lat: float, lon: float, delta_deg: float = 0.1) -> dict:
        """Compute spatial gradient of B field at given location.

        Returns:
            dict with dBx/dlat, dBx/dlon, dBy/dlat, dBy/dlon, dBz/dlat, dBz/dlon
        """
        if not self._loaded:
            self.load_grid()

        lat_c = np.clip(lat, self._lat_grid[0] + delta_deg, self._lat_grid[-1] - delta_deg)
        lon_c = np.clip(lon, self._lon_grid[0] + delta_deg, self._lon_grid[-1] - delta_deg)

        bx_n, by_n, bz_n = self.interpolate(lat_c + delta_deg, lon_c)
        bx_s, by_s, bz_s = self.interpolate(lat_c - delta_deg, lon_c)
        bx_e, by_e, bz_e = self.interpolate(lat_c, lon_c + delta_deg)
        bx_w, by_w, bz_w = self.interpolate(lat_c, lon_c - delta_deg)

        deg_per_m = 1.0 / 111320.0
        scale = 2.0 * delta_deg / deg_per_m

        return {
            "dBx_dlat": (bx_n - bx_s) / scale,
            "dBx_dlon": (bx_e - bx_w) / scale,
            "dBy_dlat": (by_n - by_s) / scale,
            "dBy_dlon": (by_e - by_w) / scale,
            "dBz_dlat": (bz_n - bz_s) / scale,
            "dBz_dlon": (bz_e - bz_w) / scale,
        }
