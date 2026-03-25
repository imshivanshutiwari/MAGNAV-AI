"""Barometer model with full ISA atmosphere."""
import numpy as np

# ISA constants
_T0 = 288.15       # Sea-level temperature (K)
_P0 = 101325.0     # Sea-level pressure (Pa)
_L = 0.0065        # Temperature lapse rate (K/m)
_g = 9.80665       # Gravitational acceleration (m/s^2)
_R = 287.05287     # Specific gas constant for air (J/(kg·K))
_TROPOPAUSE_ALT = 11000.0  # Tropopause altitude (m)
_T_TROP = 216.65           # Tropopause temperature (K)
_P_TROP = 22632.1          # Tropopause pressure (Pa)


class BarometerModel:
    """Simulates a barometric altimeter using ISA atmosphere."""

    def __init__(self,
                 noise_std: float = 0.5,
                 bias: float = 0.0,
                 update_rate_hz: float = 50.0):
        self.noise_std = noise_std
        self.bias = bias
        self.update_rate_hz = update_rate_hz
        self._rng = np.random.default_rng(2)

    def measure(self, altitude_m: float) -> float:
        """Simulate barometric altitude measurement.

        Args:
            altitude_m: True altitude in meters

        Returns:
            Noisy altitude measurement in meters
        """
        pressure = self.altitude_to_pressure(altitude_m)
        # Add pressure noise
        pressure_noise = self._rng.standard_normal() * self.noise_std * 12.0  # ~12 Pa/m at sea level
        noisy_pressure = pressure + pressure_noise
        noisy_altitude = self.pressure_to_altitude(noisy_pressure) + self.bias
        return float(noisy_altitude)

    @staticmethod
    def altitude_to_pressure(altitude_m: float) -> float:
        """Convert altitude (m) to pressure (Pa) using ISA model."""
        if altitude_m <= _TROPOPAUSE_ALT:
            T = _T0 - _L * altitude_m
            p = _P0 * (T / _T0) ** (_g / (_L * _R))
        else:
            # Stratosphere: isothermal layer
            dh = altitude_m - _TROPOPAUSE_ALT
            p = _P_TROP * np.exp(-_g * dh / (_R * _T_TROP))
        return float(p)

    @staticmethod
    def pressure_to_altitude(pressure_pa: float) -> float:
        """Convert pressure (Pa) to altitude (m) using ISA model."""
        pressure_pa = max(pressure_pa, 1.0)  # Avoid log(0)
        if pressure_pa >= _P_TROP:
            # Troposphere
            T_ratio = (pressure_pa / _P0) ** (_L * _R / _g)
            altitude = (_T0 - _T0 * T_ratio) / _L
        else:
            # Stratosphere
            altitude = _TROPOPAUSE_ALT - _R * _T_TROP / _g * np.log(pressure_pa / _P_TROP)
        return float(altitude)

    def get_temperature(self, altitude_m: float) -> float:
        """Get ISA temperature at altitude (K)."""
        if altitude_m <= _TROPOPAUSE_ALT:
            return _T0 - _L * altitude_m
        return _T_TROP

    def get_air_density(self, altitude_m: float) -> float:
        """Get ISA air density at altitude (kg/m^3)."""
        p = self.altitude_to_pressure(altitude_m)
        T = self.get_temperature(altitude_m)
        return p / (_R * T)
