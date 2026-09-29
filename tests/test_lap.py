"""Unit tests for single-lap telemetry analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lapbox.telemetry import (
    ANALYSIS_CHANNELS,
    TelemetryAnalyzer,
    channel_summary,
    compute_gforces,
    corner_analysis,
    detect_braking_zones,
    detect_corners,
    detect_full_throttle_zones,
    resample_by_distance,
)

pytestmark = pytest.mark.unit


class TestResampleByDistance:
    def test_shape_and_grid(self, lap_tel_a: pd.DataFrame) -> None:
        out = resample_by_distance(lap_tel_a, num_points=500)
        assert len(out) == 500
        assert out["Distance"].is_monotonic_increasing
        assert {"Speed", "Throttle", "Brake"}.issubset(out.columns)

    def test_requires_distance(self) -> None:
        with pytest.raises(ValueError):
            resample_by_distance(pd.DataFrame({"Speed": [1, 2, 3]}))

    def test_needs_two_points(self) -> None:
        with pytest.raises(ValueError):
            resample_by_distance(pd.DataFrame({"Distance": [0.0], "Speed": [100.0]}))


class TestBrakingZones:
    def test_finds_two_zones(self, lap_tel_a: pd.DataFrame) -> None:
        zones = detect_braking_zones(lap_tel_a)
        assert len(zones) == 2
        # Zones should sit on the approach to each corner (~1000 m, ~2000 m).
        assert zones.iloc[0]["end_distance"] <= 1000
        assert 1800 <= zones.iloc[1]["end_distance"] <= 2000

    def test_min_length_filter(self, lap_tel_a: pd.DataFrame) -> None:
        assert len(detect_braking_zones(lap_tel_a, min_length=10_000)) == 0

    def test_no_brake_column(self) -> None:
        out = detect_braking_zones(pd.DataFrame({"Distance": [0.0, 1.0]}))
        assert out.empty


class TestFullThrottleZones:
    def test_detects_wot(self, lap_tel_a: pd.DataFrame) -> None:
        zones = detect_full_throttle_zones(lap_tel_a, threshold=99)
        assert len(zones) >= 1
        assert (zones["length"] > 0).all()


class TestCorners:
    @staticmethod
    def _lap(speed: np.ndarray, step: float = 5.0) -> pd.DataFrame:
        return pd.DataFrame({"Distance": np.arange(len(speed), dtype=float) * step, "Speed": speed})

    def test_detects_two_corners(self, lap_tel_a: pd.DataFrame) -> None:
        corners = detect_corners(lap_tel_a)
        assert len(corners) == 2
        apexes = corners["apex_distance"].to_list()
        assert any(abs(a - 1000) < 60 for a in apexes)
        assert any(abs(a - 2000) < 60 for a in apexes)

    def test_apex_speed_is_low(self, lap_tel_a: pd.DataFrame) -> None:
        corners = detect_corners(lap_tel_a)
        assert (corners["apex_speed"] < 150).all()

    def test_detects_a_high_speed_corner(self) -> None:
        """A fast corner is a shallow dip at high speed — the regression.

        The old rule only accepted minima below ``0.6 * vmax``, which at every
        real circuit sits below the 210 km/h ``fast`` band, so a corner like
        Copse could never be found. Prominence has no ceiling.
        """
        distance = np.arange(0.0, 3000.0, 5.0)
        speed = 320.0 - 30.0 * np.exp(-(((distance - 1500.0) / 100.0) ** 2))
        corners = detect_corners(self._lap(speed))

        assert len(corners) == 1
        apex_speed = float(corners["apex_speed"].iloc[0])
        assert abs(float(corners["apex_distance"].iloc[0]) - 1500.0) < 30.0
        # Above the old ceiling: this is exactly what used to be discarded.
        assert apex_speed > 0.6 * speed.max()

    def test_high_speed_corner_classifies_as_fast(self) -> None:
        """Detection and classification must agree that fast corners exist."""
        distance = np.arange(0.0, 3000.0, 5.0)
        speed = 320.0 - 30.0 * np.exp(-(((distance - 1500.0) / 100.0) ** 2))
        apexes = detect_corners(self._lap(speed))["apex_distance"].to_list()
        rows = corner_analysis(distance, speed, apexes)

        assert [r["speed_class"] for r in rows] == ["fast"]

    def test_ignores_ripple_along_a_straight(self) -> None:
        """Small speed wobble at constant throttle is not a corner."""
        distance = np.arange(0.0, 3000.0, 5.0)
        speed = 320.0 + 3.0 * np.sin(distance / 40.0)
        assert detect_corners(self._lap(speed)).empty

    def test_merges_apexes_closer_than_min_separation(self) -> None:
        """Two minima inside one corner collapse to the slower of the pair."""
        distance = np.arange(0.0, 2000.0, 2.0)
        speed = (
            300.0
            - 100.0 * np.exp(-(((distance - 1000.0) / 8.0) ** 2))
            - 120.0 * np.exp(-(((distance - 1030.0) / 8.0) ** 2))
        )
        corners = detect_corners(self._lap(speed, step=2.0), min_separation=60.0)

        assert len(corners) == 1
        assert float(corners["apex_speed"].iloc[0]) == pytest.approx(180.0, abs=5.0)

    def test_empty_when_no_speed_channel(self) -> None:
        frame = pd.DataFrame({"Distance": [0.0, 10.0, 20.0]})
        out = detect_corners(frame)
        assert out.empty
        assert list(out.columns) == ["apex_distance", "apex_speed"]


class TestChannelSummary:
    def test_keys_and_values(self, lap_tel_a: pd.DataFrame) -> None:
        summary = channel_summary(lap_tel_a)
        assert summary["max_speed"] == pytest.approx(300.0, abs=1.0)
        assert 0 <= summary["full_throttle_pct"] <= 100
        assert 0 <= summary["braking_pct"] <= 100
        assert summary["max_gear"] >= 1


class TestTelemetryAnalyzer:
    def test_analyze_bundles_everything(self, lap_tel_a: pd.DataFrame) -> None:
        analysis = TelemetryAnalyzer().analyze(lap_tel_a, driver="VER")
        assert analysis.driver == "VER"
        assert len(analysis.corners) == 2
        assert len(analysis.braking_zones) == 2
        assert "max_speed" in analysis.summary


class TestCornerAnalysis:
    @staticmethod
    def _lap():
        # A straight (fast) into a slow hairpin apex at 1000 m, back up to speed.
        distance = np.linspace(0.0, 2000.0, 400)
        speed = 300.0 - 240.0 * np.exp(-(((distance - 1000.0) / 120.0) ** 2))
        return distance, speed

    def test_entry_min_exit_and_class(self):
        distance, speed = self._lap()
        rows = corner_analysis(distance, speed, [1000.0])
        assert len(rows) == 1
        c = rows[0]
        assert c["corner"] == 1
        assert c["min_speed"] < c["entry_speed"]  # slower at the apex than approach
        assert c["min_speed"] < c["exit_speed"]
        assert c["min_speed"] == pytest.approx(60.0, abs=5.0)  # ~300-240
        assert c["speed_class"] == "slow"

    def test_classes_span_slow_medium_fast(self):
        distance = np.linspace(0.0, 3000.0, 600)
        speed = np.full_like(distance, 250.0)
        # Carve three dips of different depth at 500 / 1500 / 2500 m.
        for apex, depth in ((500.0, 160.0), (1500.0, 80.0), (2500.0, 20.0)):
            speed -= depth * np.exp(-(((distance - apex) / 60.0) ** 2))
        rows = corner_analysis(distance, speed, [500.0, 1500.0, 2500.0])
        assert [r["speed_class"] for r in rows] == ["slow", "medium", "fast"]

    def test_empty_and_degenerate(self):
        assert corner_analysis([0, 1, 2], [100, 100, 100], []) == []
        assert corner_analysis([0.0], [100.0], [0.0]) == []  # too few points


class TestComputeGForces:
    """G-force derivation from speed + racing-line geometry."""

    @staticmethod
    def _circle(radius: float, speed_ms: float, n: int = 400):
        """Constant-speed circular path sampled on a uniform distance grid."""
        distance = np.linspace(0.0, 2 * np.pi * radius, n, endpoint=False)
        angle = distance / radius
        return (
            distance,
            np.full(n, speed_ms * 3.6),  # km/h
            radius * np.cos(angle),
            radius * np.sin(angle),
        )

    def test_constant_speed_circle_is_pure_lateral(self) -> None:
        radius, speed = 100.0, 30.0  # m, m/s
        a_long, a_lat = compute_gforces(*self._circle(radius, speed))
        expected = speed**2 / (radius * 9.81)  # v^2/r in g
        # Interior points avoid gradient edge effects / the unwrap seam.
        assert np.allclose(np.abs(a_lat[5:-5]), expected, rtol=1e-3)
        assert np.allclose(a_long[5:-5], 0.0, atol=1e-6)

    def test_lateral_g_is_invariant_to_xy_units(self) -> None:
        """Heading-angle method must not care about FastF1's X/Y scaling."""
        distance, speed, x, y = self._circle(100.0, 30.0)
        _, a_lat = compute_gforces(distance, speed, x, y)
        _, a_lat_scaled = compute_gforces(distance, speed, x * 10.0, y * 10.0)
        assert np.allclose(a_lat, a_lat_scaled)

    def test_straight_line_acceleration_is_pure_longitudinal(self) -> None:
        distance = np.linspace(0.0, 500.0, 300)
        # v^2 = 2*a*s with a = 4 m/s^2 -> v = sqrt(8*s); avoids v=0 at the start.
        v_ms = np.sqrt(8.0 * distance + 100.0)
        a_long, a_lat = compute_gforces(distance, v_ms * 3.6, distance, np.zeros_like(distance))
        # Away from the boundaries (finite-difference + smoothing edge padding)
        # the constant 4 m/s^2 is recovered exactly.
        assert np.allclose(a_long[20:-5], 4.0 / 9.81, rtol=1e-3)
        assert np.allclose(a_long[5:-5], 4.0 / 9.81, rtol=5e-3)
        assert np.allclose(a_lat[5:-5], 0.0, atol=1e-6)

    def test_smoothing_suppresses_position_noise_spikes(self) -> None:
        """Curvature is a 2nd derivative of noisy position — spikes must not survive."""
        radius, speed = 100.0, 30.0
        distance, speed_kmh, x, y = self._circle(radius, speed)
        rng = np.random.default_rng(42)
        # Noise scaled to the sample spacing at the same ratio as real telemetry
        # (~0.5 m jitter on a ~9 m resampled grid).
        spacing = float(distance[1] - distance[0])
        sigma = 0.055 * spacing
        noisy_x = x + rng.normal(0.0, sigma, size=len(x))
        noisy_y = y + rng.normal(0.0, sigma, size=len(y))

        _, raw = compute_gforces(distance, speed_kmh, noisy_x, noisy_y, smooth_window=0)
        _, smoothed = compute_gforces(distance, speed_kmh, noisy_x, noisy_y)
        truth = speed**2 / (radius * 9.81)

        # Unsmoothed noise blows well past the real value; smoothing reins it in.
        assert np.abs(raw).max() > np.abs(smoothed).max()
        assert np.abs(smoothed[5:-5]).max() < 3 * truth

    def test_degenerate_input_returns_zeros(self) -> None:
        a_long, a_lat = compute_gforces([0.0, 1.0], [100.0, 100.0], [0.0, 1.0], [0.0, 0.0])
        assert np.all(a_long == 0) and np.all(a_lat == 0)
        # Mismatched lengths also degrade instead of raising.
        a_long, a_lat = compute_gforces([0.0, 1.0, 2.0], [100.0], [0.0], [0.0])
        assert len(a_long) == 3 and np.all(a_long == 0)


class TestRawFastF1Telemetry:
    """FastF1's own ``Telemetry`` object goes straight in, untouched."""

    def test_same_analysis_as_the_plain_frame(
        self, fastf1_tel_a: pd.DataFrame, lap_tel_a: pd.DataFrame
    ) -> None:
        raw = TelemetryAnalyzer().analyze(fastf1_tel_a, driver="VER")
        plain = TelemetryAnalyzer().analyze(lap_tel_a, driver="VER")
        pd.testing.assert_frame_equal(raw.corners, plain.corners)
        pd.testing.assert_frame_equal(raw.braking_zones, plain.braking_zones)
        pd.testing.assert_frame_equal(raw.full_throttle_zones, plain.full_throttle_zones)
        assert raw.summary == pytest.approx(plain.summary)

    def test_resample_skips_non_numeric_columns(self, fastf1_tel_a: pd.DataFrame) -> None:
        out = resample_by_distance(fastf1_tel_a, num_points=200)
        assert list(out.columns) == ["Distance", *ANALYSIS_CHANNELS]

    def test_input_is_not_modified(self, fastf1_tel_a: pd.DataFrame) -> None:
        before = fastf1_tel_a.copy()
        TelemetryAnalyzer().analyze(fastf1_tel_a)
        resample_by_distance(fastf1_tel_a)
        pd.testing.assert_frame_equal(fastf1_tel_a, before)
