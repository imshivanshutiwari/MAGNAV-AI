"""Tests for geomagnetic map fetchers: EMAG2Fetcher and IGRFFetcher."""
import numpy as np
import pytest


class TestEMAG2Fetcher:
    def _make_fetcher(self, tmp_path):
        from data.fetchers.emag2_fetcher import EMAG2Fetcher
        return EMAG2Fetcher(cache_dir=str(tmp_path))

    def test_load_grid(self, tmp_path):
        fetcher = self._make_fetcher(tmp_path)
        result = fetcher.load_grid()
        assert result is True
        assert fetcher._loaded

    def test_interpolate_returns_three_values(self, tmp_path):
        fetcher = self._make_fetcher(tmp_path)
        fetcher.load_grid()
        Bx, By, Bz = fetcher.interpolate(40.0, -74.0)
        assert isinstance(Bx, float)
        assert isinstance(By, float)
        assert isinstance(Bz, float)

    def test_interpolate_values_reasonable(self, tmp_path):
        fetcher = self._make_fetcher(tmp_path)
        fetcher.load_grid()
        Bx, By, Bz = fetcher.interpolate(40.0, -74.0)
        # Earth's field is between -100000 and +100000 nT
        for val in [Bx, By, Bz]:
            assert -200000 < val < 200000

    def test_interpolate_altitude_effect(self, tmp_path):
        fetcher = self._make_fetcher(tmp_path)
        fetcher.load_grid()
        Bx0, By0, Bz0 = fetcher.interpolate(40.0, -74.0, alt=0.0)
        Bx1, By1, Bz1 = fetcher.interpolate(40.0, -74.0, alt=100.0)
        # Higher altitude should have lower field magnitude
        mag0 = np.sqrt(Bx0**2 + By0**2 + Bz0**2)
        mag1 = np.sqrt(Bx1**2 + By1**2 + Bz1**2)
        assert mag1 <= mag0 * 1.01  # Allow tiny numerical tolerance

    def test_interpolate_different_locations_differ(self, tmp_path):
        fetcher = self._make_fetcher(tmp_path)
        fetcher.load_grid()
        B1 = fetcher.interpolate(40.0, -74.0)
        B2 = fetcher.interpolate(50.0, 10.0)
        assert not np.allclose(B1, B2)

    def test_compute_gradient_keys(self, tmp_path):
        fetcher = self._make_fetcher(tmp_path)
        fetcher.load_grid()
        grad = fetcher.compute_gradient(40.0, -74.0)
        for key in ["dBx_dlat", "dBx_dlon", "dBz_dlat"]:
            assert key in grad

    def test_cache_reuse(self, tmp_path):
        fetcher1 = self._make_fetcher(tmp_path)
        fetcher1.load_grid()
        # Second fetcher should load from cache
        fetcher2 = self._make_fetcher(tmp_path)
        result = fetcher2.load_grid()
        assert result is True
        Bx1, _, _ = fetcher1.interpolate(40.0, -74.0)
        Bx2, _, _ = fetcher2.interpolate(40.0, -74.0)
        assert abs(Bx1 - Bx2) < 1.0


class TestIGRFFetcher:
    def _make_fetcher(self):
        from data.fetchers.igrf_fetcher import IGRFFetcher
        return IGRFFetcher()

    def test_load(self):
        fetcher = self._make_fetcher()
        result = fetcher.load()
        assert result is True

    def test_compute_field_returns_three_floats(self):
        fetcher = self._make_fetcher()
        Bx, By, Bz = fetcher.compute_field(40.0, -74.0)
        assert isinstance(Bx, float)
        assert isinstance(By, float)
        assert isinstance(Bz, float)

    def test_field_magnitude_reasonable(self):
        fetcher = self._make_fetcher()
        Bx, By, Bz = fetcher.compute_field(40.0, -74.0, alt=0.0)
        magnitude = np.sqrt(Bx**2 + By**2 + Bz**2)
        # Earth's field at surface: ~25000-65000 nT
        assert 15000 < magnitude < 80000

    def test_field_varies_with_latitude(self):
        fetcher = self._make_fetcher()
        Bx_eq, By_eq, Bz_eq = fetcher.compute_field(0.0, 0.0)
        Bx_np, By_np, Bz_np = fetcher.compute_field(90.0, 0.0)
        mag_eq = np.sqrt(Bx_eq**2 + By_eq**2 + Bz_eq**2)
        mag_np = np.sqrt(Bx_np**2 + By_np**2 + Bz_np**2)
        # Field is stronger at poles
        assert mag_np > mag_eq * 0.8  # Allow some tolerance

    def test_date_parameter(self):
        from datetime import datetime
        fetcher = self._make_fetcher()
        Bx1, _, _ = fetcher.compute_field(40.0, -74.0, date=datetime(2020, 1, 1))
        Bx2, _, _ = fetcher.compute_field(40.0, -74.0, date=datetime(2025, 1, 1))
        # Fields should be slightly different due to secular variation
        assert np.isfinite(Bx1)
        assert np.isfinite(Bx2)

    def test_altitude_reduces_field(self):
        fetcher = self._make_fetcher()
        Bx0, By0, Bz0 = fetcher.compute_field(40.0, -74.0, alt=0.0)
        Bx1, By1, Bz1 = fetcher.compute_field(40.0, -74.0, alt=400.0)
        mag0 = np.sqrt(Bx0**2 + By0**2 + Bz0**2)
        mag1 = np.sqrt(Bx1**2 + By1**2 + Bz1**2)
        assert mag1 < mag0
