"""Unit tests for a circuit's representative lap time."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from lapbox.pace import representative_base_time

pytestmark = pytest.mark.unit


class TestRepresentativeBaseTime:
    """The per-track anchor, and the contamination that made it lie.

    A circuit's anchor is the target offset a lap-time model trains against, so
    an anchor that is wrong by N seconds moves the unseen-circuit error by about
    N seconds — the two tracked almost one-for-one on the real 2026 season.
    """

    def test_a_wet_race_does_not_become_the_circuits_normal_pace(self) -> None:
        """The 2025 Australian GP ran 7% of its laps at representative pace and
        Silverstone 25%. Taking a plain median of those read a safety-car crawl
        as the circuit's pace, and scored ~7s in leave-one-track-out."""
        dry = [90.0 + i * 0.01 for i in range(40)]  # representative running
        soaked = [125.0 + i * 0.05 for i in range(160)]  # behind the safety car
        laps = pd.Series(dry + soaked)

        assert laps.median() > 120  # what the old anchor would have used
        assert representative_base_time(laps) < 91  # what the circuit actually runs at

    def test_a_clean_race_is_left_alone(self) -> None:
        laps = pd.Series([90.0 + i * 0.02 for i in range(60)])
        assert representative_base_time(laps) == pytest.approx(float(laps.median()), abs=0.05)

    def test_too_few_representative_laps_falls_back_to_the_median(self) -> None:
        """A handful of laps is a worse estimator than a contaminated many, so
        the filter must not fire when it would be measuring noise."""
        laps = pd.Series([90.0, 90.1, 90.2] + [130.0] * 50)
        assert representative_base_time(laps) == pytest.approx(float(laps.median()), abs=0.05)

    def test_an_empty_session_is_not_an_anchor(self) -> None:
        assert math.isnan(representative_base_time(pd.Series(dtype=float)))
