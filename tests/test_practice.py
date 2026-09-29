"""Unit tests for practice run detection and long-run pace."""

from __future__ import annotations

import pandas as pd
import pytest

from lapbox.practice import (
    FUEL_EFFECT_S_PER_LAP,
    MIN_LONG_RUN,
    classify_run,
    detect_runs,
    long_run_pace,
    longest_run_length,
    session_runs,
)

pytestmark = pytest.mark.unit


def frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def lap(driver: str, n: int, t: float | None, compound: str = "MEDIUM", stint: float = 1.0) -> dict:
    return {
        "driver": driver,
        "lap_number": n,
        "lap_time_seconds": t,
        "compound": compound,
        "stint": stint,
    }


class TestDetectRuns:
    def test_untimed_lap_ends_a_run(self) -> None:
        """An in-lap/out-lap has no time; the running either side is separate."""
        laps = frame(
            [
                lap("VER", 1, 95.0),
                lap("VER", 2, 94.8),
                lap("VER", 3, None),  # pits
                lap("VER", 4, 94.5),
            ]
        )
        runs = detect_runs(laps)
        assert [r.length for r in runs] == [2, 1]
        assert runs[0].start_lap == 1 and runs[0].end_lap == 2
        assert runs[1].start_lap == 4

    def test_gap_in_lap_numbers_breaks_a_run(self) -> None:
        laps = frame([lap("VER", 1, 95.0), lap("VER", 2, 94.9), lap("VER", 8, 94.7)])
        assert [r.length for r in detect_runs(laps)] == [2, 1]

    def test_compound_change_breaks_a_run(self) -> None:
        laps = frame(
            [
                lap("VER", 1, 95.0, "MEDIUM"),
                lap("VER", 2, 94.9, "MEDIUM"),
                lap("VER", 3, 93.0, "SOFT"),
            ]
        )
        runs = detect_runs(laps)
        assert [r.compound for r in runs] == ["MEDIUM", "SOFT"]

    def test_stint_change_breaks_a_run_on_the_same_compound(self) -> None:
        """Two runs on the same tyre spec are still two separate runs."""
        laps = frame(
            [
                lap("VER", 1, 95.0, "MEDIUM", stint=1),
                lap("VER", 2, 94.9, "MEDIUM", stint=1),
                lap("VER", 3, 94.8, "MEDIUM", stint=2),
            ]
        )
        assert [r.length for r in detect_runs(laps)] == [2, 1]

    def test_drivers_are_kept_separate(self) -> None:
        laps = frame([lap("VER", 1, 95.0), lap("HAM", 1, 96.0), lap("VER", 2, 94.9)])
        runs = detect_runs(laps)
        assert {r.driver for r in runs} == {"VER", "HAM"}
        assert next(r for r in runs if r.driver == "VER").length == 2

    def test_accepts_raw_loader_column_names(self) -> None:
        laps = frame(
            [
                {"Driver": "VER", "LapNumber": 1, "LapTimeSeconds": 95.0, "Compound": "MEDIUM"},
                {"Driver": "VER", "LapNumber": 2, "LapTimeSeconds": 94.9, "Compound": "MEDIUM"},
            ]
        )
        assert [r.length for r in detect_runs(laps)] == [2]

    def test_missing_columns_returns_nothing(self) -> None:
        assert detect_runs(pd.DataFrame({"foo": [1]})) == []


class TestClassifyRun:
    def _run(self, times: list[float]):
        laps = frame([lap("VER", i + 1, t) for i, t in enumerate(times)])
        return detect_runs(laps)[0]

    def test_five_or_more_laps_is_a_long_run(self) -> None:
        run = self._run([95.0] * MIN_LONG_RUN)
        assert classify_run(run, session_best=90.0) == "long_run"

    def test_short_and_fast_is_a_quali_sim(self) -> None:
        run = self._run([90.2])
        assert classify_run(run, session_best=90.0) == "quali_sim"

    def test_short_and_slow_is_neither(self) -> None:
        """An aborted run is short too — brevity alone must not mean quali sim."""
        run = self._run([99.0])
        assert classify_run(run, session_best=90.0) == "other"

    def test_short_run_without_a_session_best_is_unclassifiable(self) -> None:
        assert classify_run(self._run([90.0]), session_best=None) == "other"


class TestLongRunPace:
    def test_degradation_adds_the_fuel_effect_back(self) -> None:
        """A flat run is not a zero-degradation run — the car got lighter."""
        laps = frame([lap("VER", i + 1, 95.0) for i in range(6)])
        pace = long_run_pace(detect_runs(laps)[0])
        assert pace.raw_slope_s_per_lap == pytest.approx(0.0, abs=1e-9)
        assert pace.deg_slope_s_per_lap == pytest.approx(FUEL_EFFECT_S_PER_LAP)

    def test_rising_lap_times_are_degradation(self) -> None:
        times = [95.0, 95.1, 95.2, 95.3, 95.4, 95.5]
        pace = long_run_pace(
            detect_runs(frame([lap("VER", i + 1, t) for i, t in enumerate(times)]))[0]
        )
        assert pace.raw_slope_s_per_lap == pytest.approx(0.1, abs=1e-6)
        assert pace.deg_slope_s_per_lap == pytest.approx(0.1 + FUEL_EFFECT_S_PER_LAP)
        assert pace.median_s == pytest.approx(95.25)
        assert pace.laps == 6

    def test_consistency_is_the_spread(self) -> None:
        laps = frame([lap("VER", i + 1, t) for i, t in enumerate([95.0, 95.0, 95.0, 95.0, 95.0])])
        assert long_run_pace(detect_runs(laps)[0]).consistency_s == pytest.approx(0.0)


class TestUnrepresentativeLaps:
    """A cruised or traffic-bound lap inside a run is not race pace.

    Left in, these produce degradation of several seconds per lap — physically
    impossible, and exactly what real practice data contained before filtering.
    """

    def _pace(self, times: list[float]):
        laps = frame([lap("VER", i + 1, t) for i, t in enumerate(times)])
        return long_run_pace(detect_runs(laps)[0])

    def test_a_cruised_lap_is_dropped(self) -> None:
        pace = self._pace([95.0, 95.1, 130.0, 95.2, 95.3, 95.4])
        assert pace.laps_dropped == 1
        assert pace.laps == 5
        assert pace.consistency_s < 1.0  # the 130s lap no longer dominates

    def test_degradation_stays_physical_despite_an_outlier(self) -> None:
        clean = self._pace([95.0, 95.1, 95.2, 95.3, 95.4, 95.5])
        noisy = self._pace([95.0, 95.1, 95.2, 130.0, 95.4, 95.5])
        assert noisy.deg_slope_s_per_lap == pytest.approx(clean.deg_slope_s_per_lap, abs=0.05)

    def test_dropped_laps_do_not_compress_the_slope(self) -> None:
        """Survivors keep their ORIGINAL positions, so the rate stays per-lap."""
        pace = self._pace([95.0, 130.0, 95.2, 95.3, 95.4, 95.5])
        # 0.1 s/lap across the real laps, not 0.125 from renumbering 5 survivors.
        assert pace.raw_slope_s_per_lap == pytest.approx(0.1, abs=0.02)

    def test_anchored_to_the_best_lap_not_the_median(self) -> None:
        """When most of a run is compromised, the median is contaminated too."""
        pace = self._pace([95.0, 130.0, 131.0, 132.0, 133.0, 134.0])
        assert pace.laps == 1  # only the genuine lap survives
        assert pace.median_s == pytest.approx(95.0)


class TestSessionSummary:
    def test_groups_and_classifies_per_driver(self) -> None:
        laps = frame(
            # VER: a 6-lap long run, then a one-lap quali sim on softs.
            [lap("VER", i + 1, 95.0) for i in range(6)]
            + [lap("VER", 8, 90.0, "SOFT", stint=2)]
        )
        out = session_runs(laps)
        kinds = [kind for _, kind in out["VER"]]
        assert "long_run" in kinds and "quali_sim" in kinds

    def test_longest_run_reports_what_a_session_can_support(self) -> None:
        """The gate for the whole feature: a disrupted session has no long run."""
        sparse = frame(
            [lap("VER", 1, 95.0), lap("VER", 2, 94.9), lap("VER", 4, None), lap("VER", 5, 94.0)]
        )
        assert longest_run_length(sparse) == 2  # too short for race-pace analysis
        healthy = frame([lap("VER", i + 1, 95.0) for i in range(14)])
        assert longest_run_length(healthy) == 14

    def test_longest_run_of_an_empty_session_is_zero(self) -> None:
        assert longest_run_length(pd.DataFrame()) == 0


class TestRawFastF1Laps:
    """``session.laps`` carries the lap time only as a ``LapTime`` Timedelta."""

    @staticmethod
    def _laps(times: list[float | None]) -> pd.DataFrame:
        from fastf1.core import Laps  # imported here: FastF1 is slow to import

        return Laps(
            pd.DataFrame(
                {
                    "Driver": "VER",
                    "LapNumber": [float(i + 1) for i in range(len(times))],
                    "LapTime": pd.to_timedelta(times, unit="s"),
                    "Compound": "MEDIUM",
                    "Stint": 1.0,
                }
            )
        )

    def test_lap_time_timedelta_is_read(self) -> None:
        laps = self._laps([95.0, 95.1, None, 94.5, 94.6, 94.7, 94.8, 94.9])
        runs = detect_runs(laps)
        assert [r.length for r in runs] == [2, 5]
        assert runs[1].lap_times[0] == pytest.approx(94.5)

    def test_same_runs_as_seconds_columns(self) -> None:
        times = [95.0, 95.1, 95.2, 95.3, 95.4, 95.5]
        raw = long_run_pace(detect_runs(self._laps(times))[0])
        plain = long_run_pace(
            detect_runs(frame([lap("VER", i + 1, t) for i, t in enumerate(times)]))[0]
        )
        assert raw == plain

    def test_input_is_not_modified(self) -> None:
        laps = self._laps([95.0, 95.1, 95.2])
        before = laps.copy()
        session_runs(laps)
        pd.testing.assert_frame_equal(laps, before)
