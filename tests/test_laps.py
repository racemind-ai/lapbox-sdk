"""Unit tests for session-level lap data: seconds columns, weather, pit stops, stints, gaps."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from lapbox.data import (
    derive_pit_stops,
    derive_tyre_stints,
    gaps_to_car_ahead,
    normalize_laps,
    normalize_weather,
)

pytestmark = pytest.mark.unit


class TestNormalizeLaps:
    def test_adds_seconds_columns(self, laps_df: pd.DataFrame) -> None:
        out = normalize_laps(laps_df)
        assert "LapTimeSeconds" in out.columns
        assert out.loc[0, "LapTimeSeconds"] == pytest.approx(90.5)
        # sector seconds present too
        assert "Sector1TimeSeconds" in out.columns

    def test_does_not_mutate_input(self, laps_df: pd.DataFrame) -> None:
        before = set(laps_df.columns)
        normalize_laps(laps_df)
        assert set(laps_df.columns) == before


class TestNormalizeWeather:
    def test_adds_time_seconds(self, weather_df: pd.DataFrame) -> None:
        out = normalize_weather(weather_df)
        assert list(out["TimeSeconds"]) == [0.0, 60.0]

    def test_missing_time_column_is_tolerated(self) -> None:
        out = normalize_weather(pd.DataFrame({"AirTemp": [20.0]}))
        assert "TimeSeconds" not in out.columns


class TestDerivePitStops:
    def test_pairs_in_and_out_lap(self, laps_df: pd.DataFrame) -> None:
        stops = derive_pit_stops(laps_df)
        # Only VER made a stop (in on lap 2, out on lap 3).
        assert len(stops) == 1
        row = stops.iloc[0]
        assert row["Driver"] == "VER"
        assert row["LapNumber"] == 2
        assert row["PitInTimeSeconds"] == pytest.approx(185.0)
        assert row["PitOutTimeSeconds"] == pytest.approx(208.0)
        assert row["PitStopDurationSeconds"] == pytest.approx(23.0)

    def test_no_pit_columns_returns_empty_frame(self) -> None:
        out = derive_pit_stops(pd.DataFrame({"Driver": ["VER"], "LapNumber": [1]}))
        assert out.empty
        assert "PitStopDurationSeconds" in out.columns

    def test_missing_out_lap_time_gives_nan_duration(self) -> None:
        laps = pd.DataFrame(
            {
                "Driver": ["VER", "VER"],
                "LapNumber": [1, 2],
                "Stint": [1, 1],
                "Compound": ["SOFT", "SOFT"],
                "PitInTime": [pd.Timedelta("0:03:00"), pd.NaT],
                "PitOutTime": [pd.NaT, pd.NaT],
            }
        )
        stops = derive_pit_stops(laps)
        assert len(stops) == 1
        assert math.isnan(stops.iloc[0]["PitStopDurationSeconds"])


class TestDeriveTyreStints:
    def test_aggregates_stints(self, laps_df: pd.DataFrame) -> None:
        stints = derive_tyre_stints(laps_df)
        ver = stints[stints["Driver"] == "VER"].sort_values("StintStartLap")
        assert len(ver) == 2
        first = ver.iloc[0]
        assert first["Compound"] == "SOFT"
        assert first["StintStartLap"] == 1
        assert first["StintEndLap"] == 2
        assert first["StintLength"] == 2
        # HAM ran a single stint of length 2
        ham = stints[stints["Driver"] == "HAM"]
        assert len(ham) == 1
        assert ham.iloc[0]["StintLength"] == 2

    def test_missing_columns_returns_empty(self) -> None:
        out = derive_tyre_stints(pd.DataFrame({"Driver": ["VER"]}))
        assert out.empty


class TestGapsToCarAhead:
    def test_gap_is_time_behind_the_car_ahead_on_the_same_lap(self, laps_df: pd.DataFrame) -> None:
        gaps = gaps_to_car_ahead(laps_df)
        assert gaps[("HAM", 1)] == pytest.approx(0.700, abs=1e-3)
        assert gaps[("HAM", 2)] == pytest.approx(0.800, abs=1e-3)

    def test_the_leader_of_a_lap_has_no_entry(self, laps_df: pd.DataFrame) -> None:
        gaps = gaps_to_car_ahead(laps_df)
        assert ("VER", 1) not in gaps
        assert ("VER", 2) not in gaps
        assert ("VER", 3) not in gaps  # alone on lap 3

    def test_missing_time_column_returns_nothing(self) -> None:
        assert gaps_to_car_ahead(pd.DataFrame({"Driver": ["VER"], "LapNumber": [1]})) == {}


class TestRawFastF1Laps:
    """FastF1's own ``Laps`` object (what ``session.laps`` is) goes straight in."""

    def test_same_results_as_the_plain_frame(
        self, fastf1_laps: pd.DataFrame, laps_df: pd.DataFrame
    ) -> None:
        pd.testing.assert_frame_equal(
            pd.DataFrame(normalize_laps(fastf1_laps)), normalize_laps(laps_df)
        )
        pd.testing.assert_frame_equal(derive_pit_stops(fastf1_laps), derive_pit_stops(laps_df))
        pd.testing.assert_frame_equal(
            pd.DataFrame(derive_tyre_stints(fastf1_laps)), derive_tyre_stints(laps_df)
        )
        assert gaps_to_car_ahead(fastf1_laps) == gaps_to_car_ahead(laps_df)

    def test_input_is_not_modified(self, fastf1_laps: pd.DataFrame) -> None:
        before = fastf1_laps.copy()
        normalize_laps(fastf1_laps)
        derive_pit_stops(fastf1_laps)
        derive_tyre_stints(fastf1_laps)
        gaps_to_car_ahead(fastf1_laps)
        pd.testing.assert_frame_equal(fastf1_laps, before)
