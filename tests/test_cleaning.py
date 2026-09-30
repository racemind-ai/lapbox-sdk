"""Unit tests for the lap-cleaning pipeline."""

from __future__ import annotations

import pandas as pd
import pytest

from lapbox.pace import LapCleaningConfig, LapCleaningPipeline

pytestmark = pytest.mark.unit


@pytest.fixture
def dirty_laps() -> pd.DataFrame:
    """A frame exercising every cleaning step."""
    return pd.DataFrame(
        {
            "Driver": ["VER", "VER", "VER", "VER", "VER", "HAM"],
            "LapNumber": [1, 1, 2, 3, 4, 1],  # lap 1 duplicated
            "LapTimeSeconds": [90.5, 90.5, None, 500.0, 91.0, 30.0],
            #                    ok   dup  missing implausible ok  too-fast
            "PitInTime": [pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.Timedelta("0:05:00"), pd.NaT],
            "IsAccurate": [True, True, True, True, True, True],
            "Compound": ["SOFT"] * 6,
            "Stint": [1, 1, 1, 1, 1, 1],
        }
    )


def test_full_pipeline_removes_bad_rows(dirty_laps: pd.DataFrame) -> None:
    result = LapCleaningPipeline().clean(dirty_laps)
    # Survivors: VER lap1 (one copy) only.
    #   - duplicate lap1 removed
    #   - lap2 missing laptime removed
    #   - lap3 implausible (500s) removed
    #   - lap4 pit lap removed
    #   - HAM lap1 too fast (30s) removed
    assert result.final_rows == 1
    survivor = result.frame.iloc[0]
    assert survivor["Driver"] == "VER"
    assert survivor["LapNumber"] == 1


def test_report_accounts_for_each_step(dirty_laps: pd.DataFrame) -> None:
    result = LapCleaningPipeline().clean(dirty_laps)
    assert result.initial_rows == 6
    assert result.removed["duplicates"] == 1
    assert result.removed["missing_laptime"] == 1
    assert result.removed["pit_laps"] == 1
    # Both the 500s lap and HAM's 30s lap fall outside the plausible window.
    assert result.removed["implausible_time"] == 2
    assert result.total_removed == 5


def test_keep_pit_laps_when_configured(dirty_laps: pd.DataFrame) -> None:
    cfg = LapCleaningConfig(drop_pit_laps=False)
    result = LapCleaningPipeline(cfg).clean(dirty_laps)
    assert result.removed["pit_laps"] == 0


def test_is_accurate_filter() -> None:
    laps = pd.DataFrame(
        {
            "Driver": ["VER", "VER"],
            "LapNumber": [1, 2],
            "LapTimeSeconds": [90.0, 91.0],
            "IsAccurate": [True, False],
        }
    )
    result = LapCleaningPipeline().clean(laps)
    assert result.removed["inaccurate"] == 1
    assert result.final_rows == 1


def test_deleted_filter() -> None:
    laps = pd.DataFrame(
        {
            "Driver": ["VER", "VER"],
            "LapNumber": [1, 2],
            "LapTimeSeconds": [90.0, 91.0],
            "Deleted": [False, True],
        }
    )
    result = LapCleaningPipeline().clean(laps)
    assert result.removed["deleted"] == 1


def test_outlier_removal_optional() -> None:
    # One wild lap among consistent ones; MAD outlier detection removes it.
    laps = pd.DataFrame(
        {
            "Driver": ["VER"] * 6,
            "LapNumber": [1, 2, 3, 4, 5, 6],
            "LapTimeSeconds": [90.0, 90.1, 89.9, 90.2, 90.0, 130.0],
        }
    )
    off = LapCleaningPipeline(LapCleaningConfig(outlier_mad_threshold=None)).clean(laps)
    on = LapCleaningPipeline(LapCleaningConfig(outlier_mad_threshold=3.5)).clean(laps)
    assert off.removed["outliers"] == 0
    assert on.removed["outliers"] == 1


def test_does_not_mutate_input(dirty_laps: pd.DataFrame) -> None:
    original = dirty_laps.copy()
    LapCleaningPipeline().clean(dirty_laps)
    pd.testing.assert_frame_equal(dirty_laps, original)


def test_empty_frame_is_safe() -> None:
    result = LapCleaningPipeline().clean(pd.DataFrame(columns=["Driver", "LapNumber"]))
    assert result.final_rows == 0


class TestRawFastF1Laps:
    """``session.laps`` has the lap time only as a ``LapTime`` Timedelta."""

    @staticmethod
    def _raw(dirty: pd.DataFrame) -> pd.DataFrame:
        from fastf1.core import Laps  # imported here: FastF1 is slow to import

        raw = dirty.drop(columns=["LapTimeSeconds"])
        raw["LapTime"] = pd.to_timedelta(dirty["LapTimeSeconds"], unit="s")
        return Laps(raw)

    def test_lap_time_checks_run_on_raw_laps(self, dirty_laps: pd.DataFrame) -> None:
        """Before, both steps skipped silently: there was no LapTimeSeconds."""
        result = LapCleaningPipeline().clean(self._raw(dirty_laps))
        assert result.removed["missing_laptime"] == 1
        assert result.removed["implausible_time"] == 2
        assert result.final_rows == 1

    def test_same_report_as_the_seconds_column(self, dirty_laps: pd.DataFrame) -> None:
        raw = LapCleaningPipeline().clean(self._raw(dirty_laps))
        plain = LapCleaningPipeline().clean(dirty_laps)
        assert raw.removed == plain.removed
        assert list(raw.frame["LapNumber"]) == list(plain.frame["LapNumber"])

    def test_input_is_not_modified(self, dirty_laps: pd.DataFrame) -> None:
        raw = self._raw(dirty_laps)
        before = raw.copy()
        LapCleaningPipeline().clean(raw)
        pd.testing.assert_frame_equal(raw, before)
