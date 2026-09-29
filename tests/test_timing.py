"""Unit tests for time-conversion helpers."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from lapbox.data import (
    format_laptime,
    seconds_to_timedelta,
    timedelta_to_seconds,
)

pytestmark = pytest.mark.unit


class TestTimedeltaToSeconds:
    def test_converts_timedelta(self) -> None:
        assert timedelta_to_seconds(pd.Timedelta("0:01:23.245")) == pytest.approx(83.245)

    def test_none_returns_none(self) -> None:
        assert timedelta_to_seconds(None) is None

    def test_nat_returns_none(self) -> None:
        assert timedelta_to_seconds(pd.NaT) is None

    def test_zero(self) -> None:
        assert timedelta_to_seconds(pd.Timedelta(0)) == 0.0


class TestSecondsToTimedelta:
    def test_roundtrip(self) -> None:
        assert seconds_to_timedelta(83.245) == pd.Timedelta("0:01:23.245")

    def test_none_returns_none(self) -> None:
        assert seconds_to_timedelta(None) is None

    def test_nan_returns_none(self) -> None:
        assert seconds_to_timedelta(math.nan) is None


class TestFormatLaptime:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (83.245, "1:23.245"),
            (59.999, "0:59.999"),
            (0.0, "0:00.000"),
            (125.5, "2:05.500"),
        ],
    )
    def test_formats_seconds(self, value: float, expected: str) -> None:
        assert format_laptime(value) == expected

    def test_accepts_timedelta(self) -> None:
        assert format_laptime(pd.Timedelta("0:01:23.245")) == "1:23.245"

    def test_missing_renders_dash(self) -> None:
        assert format_laptime(None) == "-"
        assert format_laptime(pd.NaT) == "-"

    def test_negative_delta(self) -> None:
        assert format_laptime(-1.234) == "-0:01.234"
