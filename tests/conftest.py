"""Shared pytest fixtures for lapbox.

Everything here is synthetic, so the suite runs fully offline: no FastF1
downloads, no network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def build_synthetic_lap(*, speed_scale: float = 1.0) -> pd.DataFrame:
    """Build a deterministic single-lap telemetry frame with two corners.

    The speed profile has two Gaussian dips (corners) at 1000 m and 2000 m; the
    brake is applied on the approach to each; throttle is wide open on the
    straights between them. Used to exercise the analysis/comparison functions.
    """
    distance = np.arange(0, 3000, 10, dtype=float)
    speed = np.full_like(distance, 300.0)
    for center in (1000.0, 2000.0):
        speed -= 200.0 * np.exp(-((distance - center) ** 2) / (2 * 80.0**2))
    speed *= speed_scale

    brake = ((distance > 820) & (distance < 1000)) | ((distance > 1820) & (distance < 2000))
    throttle = np.where(speed > 250 * speed_scale, 100.0, 40.0)
    throttle = np.where(brake, 0.0, throttle)
    gear = np.clip((speed / 40).astype(int), 1, 8)

    return pd.DataFrame(
        {
            "Distance": distance,
            "Speed": speed,
            "Throttle": throttle,
            "Brake": brake,
            "RPM": speed * 45.0,
            "nGear": gear,
            "DRS": np.zeros_like(distance),
            "X": distance,
            "Y": 50.0 * np.sin(distance / 500.0),
        }
    )


def build_fastf1_telemetry(*, speed_scale: float = 1.0) -> pd.DataFrame:
    """The synthetic lap as ``lap.get_telemetry()`` hands it over.

    FastF1's own ``Telemetry`` class with its full column set, in its order and
    dtypes -- including the datetime, timedelta and string columns (``Date``,
    ``Time``, ``Source``, ``Status``, ...) that the analysis has to ignore.
    """
    from fastf1.core import Telemetry  # imported here: FastF1 is slow to import

    lap = build_synthetic_lap(speed_scale=speed_scale)
    distance = lap["Distance"].to_numpy()
    speed_ms = lap["Speed"].to_numpy() / 3.6
    seconds = np.concatenate([[0.0], np.cumsum(np.diff(distance) / speed_ms[1:])])
    time = pd.to_timedelta(seconds, unit="s")
    frame = pd.DataFrame(
        {
            "Date": pd.Timestamp("2026-07-05 14:03:00") + time,
            "SessionTime": pd.Timedelta(minutes=63) + time,
            "DriverAhead": "16",
            "DistanceToDriverAhead": 42.0,
            "Time": time,
            "RPM": lap["RPM"],
            "Speed": lap["Speed"],
            "nGear": lap["nGear"],
            "Throttle": lap["Throttle"],
            "Brake": lap["Brake"],
            "DRS": lap["DRS"].astype(int),
            "Source": np.where(np.arange(len(lap)) % 2 == 0, "car", "pos"),
            "Distance": distance,
            "RelativeDistance": distance / distance.max(),
            "Status": "OnTrack",
            "X": lap["X"],
            "Y": lap["Y"],
            "Z": 0.0,
        }
    )
    return Telemetry(frame)


@pytest.fixture
def laps_df() -> pd.DataFrame:
    """A minimal but realistic two-driver laps frame.

    VER runs a 2-stop-shaped set of laps (SOFT -> MEDIUM) with one pit stop;
    HAM stays out on a single MEDIUM stint. Times are ``Timedelta`` as FastF1
    produces them.
    """
    td = pd.to_timedelta
    rows = [
        # VER stint 1 (SOFT), pits at end of lap 2
        {
            "Driver": "VER",
            "LapNumber": 1,
            "Stint": 1,
            "Compound": "SOFT",
            "TyreLife": 1,
            "LapTime": td("0:01:30.500"),
            "Sector1Time": td("0:00:30.100"),
            "Sector2Time": td("0:00:30.200"),
            "Sector3Time": td("0:00:30.200"),
            "PitInTime": pd.NaT,
            "PitOutTime": pd.NaT,
        },
        {
            "Driver": "VER",
            "LapNumber": 2,
            "Stint": 1,
            "Compound": "SOFT",
            "TyreLife": 2,
            "LapTime": td("0:01:31.000"),
            "Sector1Time": td("0:00:30.300"),
            "Sector2Time": td("0:00:30.300"),
            "Sector3Time": td("0:00:30.400"),
            "PitInTime": td("0:03:05.000"),
            "PitOutTime": pd.NaT,
        },
        # VER stint 2 (MEDIUM), out-lap on lap 3
        {
            "Driver": "VER",
            "LapNumber": 3,
            "Stint": 2,
            "Compound": "MEDIUM",
            "TyreLife": 1,
            "LapTime": td("0:01:33.000"),
            "Sector1Time": td("0:00:31.000"),
            "Sector2Time": td("0:00:31.000"),
            "Sector3Time": td("0:00:31.000"),
            "PitInTime": pd.NaT,
            "PitOutTime": td("0:03:28.000"),
        },
        # HAM single MEDIUM stint, no stops
        {
            "Driver": "HAM",
            "LapNumber": 1,
            "Stint": 1,
            "Compound": "MEDIUM",
            "TyreLife": 5,
            "LapTime": td("0:01:31.200"),
            "Sector1Time": td("0:00:30.400"),
            "Sector2Time": td("0:00:30.400"),
            "Sector3Time": td("0:00:30.400"),
            "PitInTime": pd.NaT,
            "PitOutTime": pd.NaT,
        },
        {
            "Driver": "HAM",
            "LapNumber": 2,
            "Stint": 1,
            "Compound": "MEDIUM",
            "TyreLife": 6,
            "LapTime": td("0:01:31.100"),
            "Sector1Time": td("0:00:30.300"),
            "Sector2Time": td("0:00:30.400"),
            "Sector3Time": td("0:00:30.400"),
            "PitInTime": pd.NaT,
            "PitOutTime": pd.NaT,
        },
    ]
    frame = pd.DataFrame(rows)
    # Elapsed session time at the line, as FastF1 provides it. VER leads lap 1
    # by 0.7s and lap 2 by 0.2s; on lap 3 only VER is running.
    frame["Time"] = pd.to_timedelta(
        [
            "0:01:30.500",  # VER lap 1
            "0:03:01.500",  # VER lap 2
            "0:04:34.500",  # VER lap 3
            "0:01:31.200",  # HAM lap 1  -> 0.700s behind VER
            "0:03:02.300",  # HAM lap 2  -> 0.800s behind VER
        ]
    )
    return frame


@pytest.fixture
def fastf1_laps(laps_df: pd.DataFrame) -> pd.DataFrame:
    """``laps_df`` as FastF1's own ``Laps`` object, the type ``session.laps`` is."""
    from fastf1.core import Laps  # imported here: FastF1 is slow to import

    return Laps(laps_df)


@pytest.fixture
def weather_df() -> pd.DataFrame:
    """A minimal weather timeline."""
    return pd.DataFrame(
        {
            "Time": pd.to_timedelta(["0:00:00", "0:01:00"]),
            "AirTemp": [24.5, 24.7],
            "TrackTemp": [38.1, 38.4],
            "Humidity": [40.0, 41.0],
            "Rainfall": [False, False],
            "WindSpeed": [1.2, 1.4],
        }
    )


@pytest.fixture
def lap_tel_a() -> pd.DataFrame:
    """Reference lap telemetry (the faster driver)."""
    return build_synthetic_lap(speed_scale=1.0)


@pytest.fixture
def lap_tel_b() -> pd.DataFrame:
    """A slightly slower lap for comparison tests."""
    return build_synthetic_lap(speed_scale=0.98)


@pytest.fixture
def fastf1_tel_a() -> pd.DataFrame:
    """``lap_tel_a`` as a FastF1 ``Telemetry`` object."""
    return build_fastf1_telemetry(speed_scale=1.0)


@pytest.fixture
def fastf1_tel_b() -> pd.DataFrame:
    """``lap_tel_b`` as a FastF1 ``Telemetry`` object."""
    return build_fastf1_telemetry(speed_scale=0.98)
