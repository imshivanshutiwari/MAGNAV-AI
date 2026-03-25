"""IGRF-13 geomagnetic reference field fetcher."""
import logging
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

# IGRF-13 dipole coefficients (simplified; g10, g11, h11 for main dipole terms in nT)
# Full IGRF-13 coefficients up to degree 13 at epoch 2020.0
_IGRF13_COEFFS = {
    # g(n,m), h(n,m) for n=1,m=0; n=1,m=1; etc.
    # Format: (n, m): (g_nm, h_nm, g_sv, h_sv)  sv = secular variation per year
    (1, 0): (-29404.5, 0.0, 6.7, 0.0),
    (1, 1): (-1450.7, 4652.9, 7.7, -25.1),
    (2, 0): (-2500.0, 0.0, -11.5, 0.0),
    (2, 1): (2982.0, -2991.6, -7.1, -30.2),
    (2, 2): (1676.8, 1764.2, 2.8, -7.2),
    (3, 0): (1363.8, 0.0, 2.8, 0.0),
    (3, 1): (-2381.0, -82.2, -6.2, 5.9),
    (3, 2): (1236.2, 241.8, 3.4, -0.9),
    (3, 3): (525.7, -543.4, -12.2, 1.4),
}

_IGRF13_EPOCH = 2020.0


class IGRFFetcher:
    """Computes IGRF-13 geomagnetic reference field values."""

    def __init__(self):
        self._coeffs = _IGRF13_COEFFS
        self._epoch = _IGRF13_EPOCH

    def load(self) -> bool:
        """Load/initialize IGRF coefficients."""
        logger.info("IGRF-13 coefficients loaded (degree 1-3 terms).")
        return True

    @staticmethod
    def _date_to_decimal_year(date: datetime) -> float:
        """Convert datetime to decimal year."""
        year = date.year
        start = datetime(year, 1, 1)
        end = datetime(year + 1, 1, 1)
        fraction = (date - start).total_seconds() / (end - start).total_seconds()
        return year + fraction

    @staticmethod
    def _schmidt_quasi_normal(n: int, m: int, theta: float) -> float:
        """Compute Schmidt quasi-normal associated Legendre function P_n^m(cos theta)."""
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Use recurrence relation for associated Legendre polynomials
        # P_0^0 = 1, P_1^0 = cos(theta), P_1^1 = sin(theta)
        if n == 1 and m == 0:
            return cos_t
        if n == 1 and m == 1:
            return sin_t
        if n == 2 and m == 0:
            return 0.5 * (3 * cos_t ** 2 - 1)
        if n == 2 and m == 1:
            return np.sqrt(3.0) * sin_t * cos_t
        if n == 2 and m == 2:
            return np.sqrt(3.0 / 4.0) * sin_t ** 2
        if n == 3 and m == 0:
            return 0.5 * cos_t * (5 * cos_t ** 2 - 3)
        if n == 3 and m == 1:
            return np.sqrt(6.0 / 4.0) * sin_t * (5 * cos_t ** 2 - 1)
        if n == 3 and m == 2:
            return np.sqrt(15.0 / 4.0) * sin_t ** 2 * cos_t
        if n == 3 and m == 3:
            return np.sqrt(10.0 / 8.0) * sin_t ** 3
        return 0.0

    def compute_field(self, lat: float, lon: float, alt: float = 0.0,
                      date: datetime = None) -> tuple:
        """Compute IGRF magnetic field components at given location.

        Args:
            lat: Geodetic latitude in degrees
            lon: Longitude in degrees
            alt: Altitude in km above WGS84 ellipsoid
            date: Date for field computation (defaults to 2020.0)

        Returns:
            (Bx, By, Bz) in nT (North, East, Down)
        """
        if date is None:
            decimal_year = self._epoch
        else:
            decimal_year = self._date_to_decimal_year(date)

        dt = decimal_year - self._epoch

        # Spherical coordinates
        theta = np.deg2rad(90.0 - lat)  # colatitude
        phi = np.deg2rad(lon)
        r = 6371.2 + alt  # geocentric radius in km
        a = 6371.2  # reference radius

        Br = 0.0
        Bt = 0.0
        Bp = 0.0

        for (n, m), (g, h, g_sv, h_sv) in self._coeffs.items():
            # Update coefficients with secular variation
            g_curr = g + g_sv * dt
            h_curr = h + h_sv * dt

            P = self._schmidt_quasi_normal(n, m, theta)
            # Derivative dP/dtheta (numerical)
            dtheta = 1e-4
            dP = (self._schmidt_quasi_normal(n, m, theta + dtheta) -
                  self._schmidt_quasi_normal(n, m, theta - dtheta)) / (2 * dtheta)

            ratio = (a / r) ** (n + 2)

            cos_mp = np.cos(m * phi)
            sin_mp = np.sin(m * phi)
            cos_term = g_curr * cos_mp + h_curr * sin_mp

            Br -= (n + 1) * ratio * P * cos_term
            Bt -= ratio * dP * cos_term
            if m > 0:
                Bp += m * ratio * P * (-g_curr * sin_mp + h_curr * cos_mp) / np.sin(theta + 1e-9)

        # Convert from geocentric (r, theta, phi) to NED
        Bx = -Bt  # North
        By = Bp   # East
        Bz = -Br  # Down

        return float(Bx), float(By), float(Bz)
