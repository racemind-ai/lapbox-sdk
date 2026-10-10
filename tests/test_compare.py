"""Unit tests for two-driver telemetry comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lapbox.telemetry import (
    MAX_ALIGNMENT_RESIDUAL,
    ComparisonResult,
    align_by_distance,
    alignment_residual,
    channel_delta,
    compare_drivers,
    corner_speeds,
    cumulative_time_delta,
    minisector_dominance,
    segment_gaps,
)
from lapbox.telemetry.compare import _best_circular_shift

pytestmark = pytest.mark.unit


def _lap(speed: np.ndarray, step: float = 5.0) -> pd.DataFrame:
    """A minimal single-channel lap on a uniform distance axis."""
    return pd.DataFrame({"Distance": np.arange(len(speed), dtype=float) * step, "Speed": speed})


def _two_corner_lap(n: int = 600, shift: int = 0, scale: float = 1.0) -> pd.DataFrame:
    """A synthetic lap with two sharp corners, optionally rotated and stretched.

    ``shift`` rotates the speed profile around the lap the way a different
    lap-start point does; ``scale`` stretches the distance axis the way
    integration drift does.
    """
    idx = np.arange(n, dtype=float)
    speed = np.full(n, 300.0)
    for centre in (n * 0.3, n * 0.7):
        speed -= 220.0 * np.exp(-(((idx - centre) / (n * 0.03)) ** 2))
    return _lap(np.roll(speed, shift), step=5.0 * scale)


def _with_clock(lap: pd.DataFrame, *, timedelta: bool = False) -> pd.DataFrame:
    """The lap with its own timing: seconds since its start, from its speed trace."""
    distance = lap["Distance"].to_numpy(dtype=float)
    speed_ms = lap["Speed"].to_numpy(dtype=float) / 3.6
    seconds = np.concatenate([[0.0], np.cumsum(np.diff(distance) / speed_ms[1:])])
    if timedelta:
        return lap.assign(Time=pd.to_timedelta(seconds, unit="s"))
    return lap.assign(TimeSeconds=seconds)


def _misplaced_corner_lap() -> pd.DataFrame:
    """A lap whose corners sit in the wrong places, and not by one constant offset.

    What a telemetry dropout does to a lap: the distance axis is stretched
    locally, so no global alignment can match it to :func:`_two_corner_lap`.
    """
    idx = np.arange(600, dtype=float)
    speed = np.full(600, 300.0)
    for centre in (600 * 0.3, 600 * 0.55):
        speed -= 220.0 * np.exp(-(((idx - centre) / 18.0) ** 2))
    return _lap(speed)


class TestAlignByDistance:
    def test_shared_grid(self, lap_tel_a: pd.DataFrame, lap_tel_b: pd.DataFrame) -> None:
        aligned = align_by_distance(lap_tel_a, lap_tel_b, num_points=400)
        assert len(aligned) == 400
        assert {"Distance", "Speed_a", "Speed_b"}.issubset(aligned.columns)
        assert aligned["Distance"].is_monotonic_increasing

    def test_matches_a_lap_that_starts_further_round_the_circuit(self) -> None:
        """The regression: equal distance is not the same place on track.

        Two identical laps, one rotated 25 samples (125 m, within the 5 %
        search bound) around the circuit. Sampling at equal absolute distances
        lines a corner up against a straight; matching by track position must
        recover the identity.
        """
        a = _two_corner_lap()
        b = _two_corner_lap(shift=25)

        aligned = align_by_distance(a, b, num_points=600, channels=("Speed",))
        assert alignment_residual(aligned) < 5.0

    def test_offset_beyond_the_search_bound_is_reported_not_guessed(self) -> None:
        """Past 5 % of a lap, say the laps don't match rather than force them.

        Searching further would let a circuit's repeated sections lock onto the
        wrong match, which is worse than declining to compare.
        """
        a = _two_corner_lap()
        b = _two_corner_lap(shift=120)  # 20 % of the lap

        aligned = align_by_distance(a, b, num_points=600, channels=("Speed",))
        assert alignment_residual(aligned) > MAX_ALIGNMENT_RESIDUAL

    def test_matches_a_lap_whose_distance_axis_drifted(self) -> None:
        """A lap that integrated 3 % long still describes the same corners."""
        a = _two_corner_lap()
        b = _two_corner_lap(scale=1.03)

        aligned = align_by_distance(a, b, num_points=600, channels=("Speed",))
        assert alignment_residual(aligned) < 5.0

    def test_identical_laps_are_left_alone(self) -> None:
        a = _two_corner_lap()
        aligned = align_by_distance(a, a.copy(), num_points=600, channels=("Speed",))
        assert alignment_residual(aligned) == pytest.approx(0.0, abs=1e-6)

    def test_real_speed_difference_survives_alignment(self) -> None:
        """Alignment must move laps, not erase the gap between them."""
        a = _two_corner_lap()
        b = _two_corner_lap()
        b["Speed"] = b["Speed"] * 0.9  # B genuinely 10 % slower everywhere

        aligned = align_by_distance(a, b, num_points=600, channels=("Speed",))
        assert (aligned["Speed_a"] > aligned["Speed_b"]).mean() > 0.95

    def test_a_fraction_of_a_sample_is_recovered(self) -> None:
        def trace(i: np.ndarray) -> np.ndarray:
            speed = np.full(len(i), 300.0)
            for centre in (180.0, 420.0):
                speed -= 220.0 * np.exp(-(((i - centre) / 18.0) ** 2))
            return speed

        idx = np.arange(600, dtype=float)
        # ``other`` is ``ref`` read 3.4 samples later: rolling it by 3.4 lines them up.
        assert _best_circular_shift(trace(idx), trace(idx + 3.4), 30) == pytest.approx(3.4, abs=0.1)

    @pytest.mark.parametrize("num_points", [300, 1200])
    def test_the_match_does_not_depend_on_the_grid(self, num_points: int) -> None:
        # 125 m round the circuit: 12.5 samples of a 300-point grid, which a shift
        # by whole samples cannot reach.
        aligned = align_by_distance(
            _two_corner_lap(), _two_corner_lap(shift=25), num_points=num_points, channels=("Speed",)
        )
        assert alignment_residual(aligned) < 1.0

    def test_laps_with_timing_get_their_clocks(self) -> None:
        aligned = align_by_distance(_with_clock(_two_corner_lap()), _with_clock(_two_corner_lap()))
        assert {"Time_a", "Time_b"} <= set(aligned.columns)
        assert aligned["Time_a"].iloc[0] == pytest.approx(0.0)
        assert aligned["Time_b"].to_numpy() == pytest.approx(aligned["Time_a"].to_numpy())

    def test_time_and_time_seconds_read_the_same(self) -> None:
        a, b = _two_corner_lap(), _two_corner_lap(scale=1.01)
        seconds = align_by_distance(_with_clock(a), _with_clock(b))
        timedelta = align_by_distance(
            _with_clock(a, timedelta=True), _with_clock(b, timedelta=True)
        )
        pd.testing.assert_frame_equal(seconds, timedelta)


class TestAlignmentResidual:
    def test_unmatchable_laps_are_reported(self) -> None:
        """A lap stretched only in its middle cannot be globally aligned."""
        a = _two_corner_lap()
        b = _misplaced_corner_lap()

        aligned = align_by_distance(a, b, num_points=600, channels=("Speed",))
        assert alignment_residual(aligned) > MAX_ALIGNMENT_RESIDUAL

    def test_missing_speed_is_infinite(self) -> None:
        frame = pd.DataFrame({"Distance": [0.0, 1.0, 2.0]})
        assert alignment_residual(frame) == float("inf")


class TestChannelDelta:
    def test_delta_positive_when_a_faster(
        self, lap_tel_a: pd.DataFrame, lap_tel_b: pd.DataFrame
    ) -> None:
        aligned = align_by_distance(lap_tel_a, lap_tel_b)
        delta = channel_delta(aligned, "Speed")
        # A (scale 1.0) is faster than B (0.98) almost everywhere.
        assert delta.mean() > 0

    def test_missing_channel_raises(self, lap_tel_a: pd.DataFrame, lap_tel_b: pd.DataFrame) -> None:
        aligned = align_by_distance(lap_tel_a, lap_tel_b)
        with pytest.raises(KeyError):
            channel_delta(aligned, "Nonexistent")


class TestCumulativeTimeDelta:
    def test_faster_driver_gains_time(
        self, lap_tel_a: pd.DataFrame, lap_tel_b: pd.DataFrame
    ) -> None:
        aligned = align_by_distance(lap_tel_a, lap_tel_b)
        delta = cumulative_time_delta(aligned)
        assert delta.iloc[0] == pytest.approx(0.0)
        # A (scale 1.0) is quicker than B (0.98) everywhere, so the gap grows
        # in A's favour for the whole lap.
        assert delta.is_monotonic_increasing
        assert delta.iloc[-1] > 0

    def test_missing_speed_raises(self) -> None:
        with pytest.raises(KeyError):
            cumulative_time_delta(pd.DataFrame({"Distance": [0.0, 1.0]}))

    @pytest.mark.parametrize("num_points", [500, 2000])
    def test_with_clocks_it_ends_at_the_lap_time_gap(self, num_points: int) -> None:
        # B is 3 % slower and its lap starts 125 m further round, so matching it to
        # A carries part of B's lap across the line.
        a = _with_clock(_two_corner_lap())
        slower = _two_corner_lap(shift=25)
        b = _with_clock(slower.assign(Speed=slower["Speed"] * 0.97))
        lap_a, lap_b = a["TimeSeconds"].iloc[-1], b["TimeSeconds"].iloc[-1]

        delta = cumulative_time_delta(align_by_distance(a, b, num_points=num_points))
        assert delta.iloc[0] == 0.0
        assert delta.iloc[-1] == pytest.approx(lap_b - lap_a, abs=1e-9)
        assert delta.is_monotonic_increasing  # B loses time all the way round


class TestCornerSpeeds:
    def test_min_speed_per_corner(self, lap_tel_a: pd.DataFrame, lap_tel_b: pd.DataFrame) -> None:
        aligned = align_by_distance(lap_tel_a, lap_tel_b)
        cs = corner_speeds(aligned, [1000.0, 2000.0])
        assert list(cs["corner"]) == [1, 2]
        assert list(cs["distance"]) == [1000.0, 2000.0]
        # Apex troughs sit near 100/98 km/h, far below straight-line speed.
        assert (cs["speed_a"] < 150).all()
        assert (cs["speed_a"] > cs["speed_b"]).all()

    def test_apex_outside_grid_is_skipped(
        self, lap_tel_a: pd.DataFrame, lap_tel_b: pd.DataFrame
    ) -> None:
        aligned = align_by_distance(lap_tel_a, lap_tel_b)
        cs = corner_speeds(aligned, [99_999.0])
        assert cs.empty

    def test_missing_speed_raises(self) -> None:
        with pytest.raises(KeyError):
            corner_speeds(pd.DataFrame({"Distance": [0.0, 1.0]}), [0.5])


class TestMinisectorDominance:
    def test_row_count_and_columns(self, lap_tel_a: pd.DataFrame, lap_tel_b: pd.DataFrame) -> None:
        ms = minisector_dominance(
            lap_tel_a, lap_tel_b, driver_a="VER", driver_b="HAM", num_minisectors=21
        )
        assert len(ms) == 21
        assert {
            "minisector",
            "fastest",
            "mean_speed_a",
            "mean_speed_b",
            "time_a",
            "time_b",
        }.issubset(ms.columns)

    def test_faster_driver_dominates(
        self, lap_tel_a: pd.DataFrame, lap_tel_b: pd.DataFrame
    ) -> None:
        ms = minisector_dominance(lap_tel_a, lap_tel_b, driver_a="VER", driver_b="HAM")
        assert (ms["fastest"] == "VER").sum() > (ms["fastest"] == "HAM").sum()

    def test_time_decides_not_mean_speed(self) -> None:
        # A crawls through a corner then flies down a straight: the higher mean
        # speed (200 vs 180 km/h), but far more time than B's steady 180.
        slow_fast = _lap(np.r_[np.full(100, 50.0), np.full(100, 350.0)])
        steady = _lap(np.full(200, 180.0))
        [row] = minisector_dominance(
            slow_fast, steady, driver_a="A", driver_b="B", num_minisectors=1
        ).itertuples()
        assert row.mean_speed_a > row.mean_speed_b
        assert row.time_a > row.time_b
        assert row.fastest == "B"

    def test_minisector_times_add_up_to_each_laps_clock(self) -> None:
        # Same speeds, but B's clock says its lap took half a second longer: the
        # speed trace gives the shape of each lap, its clock the total.
        a = _with_clock(_two_corner_lap())
        b = a.copy()
        b.loc[300:, "TimeSeconds"] += 0.5
        ms = minisector_dominance(a, b, driver_a="A", driver_b="B")
        assert ms["time_a"].sum() == pytest.approx(a["TimeSeconds"].iloc[-1], abs=1e-9)
        assert ms["time_b"].sum() == pytest.approx(b["TimeSeconds"].iloc[-1], abs=1e-9)
        assert (ms["fastest"] == "A").all()


class TestSegmentGaps:
    def test_segments_end_at_the_speed_peaks(self) -> None:
        aligned = align_by_distance(_two_corner_lap(), _two_corner_lap(scale=0.98))
        seg = segment_gaps(aligned)
        distance = aligned["Distance"]
        assert list(seg["segment"]) == [1, 2]
        assert seg["start_distance"].iloc[0] == distance.iloc[0]
        assert seg["end_distance"].iloc[-1] == distance.iloc[-1]
        # The straight between the corners (apexes at 30 % and 70 % of the lap).
        assert seg["end_distance"].iloc[0] == pytest.approx(distance.iloc[-1] / 2, rel=0.01)

    def test_segments_add_up_to_the_gap_and_each_laps_clock(self) -> None:
        a = _with_clock(_two_corner_lap())
        slower = _two_corner_lap(shift=25)
        b = _with_clock(slower.assign(Speed=slower["Speed"] * 0.97))
        aligned = align_by_distance(a, b, num_points=500)
        seg = segment_gaps(aligned)
        gap = cumulative_time_delta(aligned).iloc[-1]
        assert seg["time_delta"].sum() == pytest.approx(gap, abs=1e-9)
        assert seg["time_a"].sum() == pytest.approx(a["TimeSeconds"].iloc[-1], abs=1e-9)
        assert seg["time_b"].sum() == pytest.approx(b["TimeSeconds"].iloc[-1], abs=1e-9)
        assert (seg["time_delta"] > 0).all()  # B is slower everywhere

    def test_swapping_the_drivers_keeps_the_segments(self) -> None:
        aligned = align_by_distance(
            _with_clock(_two_corner_lap()), _with_clock(_two_corner_lap(scale=0.98))
        )
        swapped = aligned.rename(
            columns={
                "Speed_a": "Speed_b",
                "Speed_b": "Speed_a",
                "Time_a": "Time_b",
                "Time_b": "Time_a",
            }
        )
        seg, back = segment_gaps(aligned), segment_gaps(swapped)
        pd.testing.assert_frame_equal(
            back[["segment", "start_distance", "end_distance"]],
            seg[["segment", "start_distance", "end_distance"]],
        )
        np.testing.assert_allclose(back["time_delta"], -seg["time_delta"], atol=1e-12)

    def test_a_misplaced_corner_moves_time_only_within_its_segment(self) -> None:
        # The same lap twice, B's distance 5 m out around the first corner: no time
        # changes hands anywhere, but the gap curve swings around that corner.
        a = _with_clock(_two_corner_lap())
        distance = a["Distance"].to_numpy(dtype=float)
        apex = distance[np.argmin(a["Speed"].to_numpy())]
        b = a.assign(Distance=distance + 5.0 * np.clip(1 - np.abs(distance - apex) / 150, 0, 1))
        aligned = align_by_distance(a, b, num_points=500)
        curve = cumulative_time_delta(aligned)
        around = (aligned["Distance"] - apex).abs() < 300
        assert curve[around].max() - curve[around].min() > 0.05
        assert segment_gaps(aligned)["time_delta"].abs().max() < 0.001

    def test_a_lap_without_peaks_is_one_segment(self) -> None:
        aligned = align_by_distance(_lap(np.full(200, 250.0)), _lap(np.full(200, 240.0)))
        [row] = segment_gaps(aligned).itertuples()
        assert (row.start_distance, row.end_distance) == (0.0, aligned["Distance"].iloc[-1])
        assert row.time_delta > 0

    def test_missing_speed_raises(self) -> None:
        with pytest.raises(KeyError):
            segment_gaps(pd.DataFrame({"Distance": [0.0, 1.0]}))


class TestCompareDrivers:
    def test_summary(self, lap_tel_a: pd.DataFrame, lap_tel_b: pd.DataFrame) -> None:
        result = compare_drivers(lap_tel_a, lap_tel_b, driver_a="VER", driver_b="HAM")
        assert result.summary["more_minisectors"] == "VER"
        assert "dominant_driver" not in result.summary  # a count, not a verdict: renamed
        assert result.summary["minisectors_a"] >= result.summary["minisectors_b"]
        assert result.summary["max_speed_a"] >= result.summary["max_speed_b"]
        assert not result.aligned.empty

    def test_minisectors_do_not_follow_the_display_grid(
        self, lap_tel_a: pd.DataFrame, lap_tel_b: pd.DataFrame
    ) -> None:
        # A coarse grid for the charts must not make the minisectors coarser.
        fine = compare_drivers(lap_tel_a, lap_tel_b, driver_a="VER", driver_b="HAM")
        coarse = compare_drivers(
            lap_tel_a, lap_tel_b, driver_a="VER", driver_b="HAM", num_points=400
        )
        assert len(coarse.aligned) == 400
        pd.testing.assert_frame_equal(coarse.minisectors, fine.minisectors)

    def test_minisector_times_add_up_to_the_gap_on_their_grid(
        self, lap_tel_a: pd.DataFrame, lap_tel_b: pd.DataFrame
    ) -> None:
        ms = minisector_dominance(lap_tel_a, lap_tel_b, driver_a="VER", driver_b="HAM")
        own_grid = align_by_distance(lap_tel_a, lap_tel_b, num_points=4000, channels=("Speed",))
        gap = cumulative_time_delta(own_grid).iloc[-1]
        assert (ms["time_b"] - ms["time_a"]).sum() == pytest.approx(gap, abs=1e-9)

    def test_segments_are_on_the_aligned_grid(
        self, lap_tel_a: pd.DataFrame, lap_tel_b: pd.DataFrame
    ) -> None:
        result = compare_drivers(
            lap_tel_a, lap_tel_b, driver_a="VER", driver_b="HAM", num_points=400
        )
        pd.testing.assert_frame_equal(result.segments, segment_gaps(result.aligned))


class TestComparisonMatched:
    """Whether two laps can be compared is decided here, not by each caller."""

    def test_sound_pair_is_matched(self, lap_tel_a: pd.DataFrame, lap_tel_b: pd.DataFrame) -> None:
        result = compare_drivers(lap_tel_a, lap_tel_b, driver_a="VER", driver_b="HAM")
        assert result.residual == pytest.approx(alignment_residual(result.aligned))
        assert result.residual < MAX_ALIGNMENT_RESIDUAL
        assert result.matched

    def test_dropout_pair_is_not_matched(self) -> None:
        result = compare_drivers(
            _two_corner_lap(), _misplaced_corner_lap(), driver_a="VER", driver_b="HAM"
        )
        assert result.residual > MAX_ALIGNMENT_RESIDUAL
        assert not result.matched

    @pytest.mark.parametrize(
        ("residual", "matched"),
        [
            (MAX_ALIGNMENT_RESIDUAL, True),  # inclusive, as the website has always had it
            (MAX_ALIGNMENT_RESIDUAL + 0.01, False),
            (float("inf"), False),  # alignment_residual's "nothing to compare"
            (float("nan"), False),
        ],
    )
    def test_threshold(self, residual: float, matched: bool) -> None:
        empty = pd.DataFrame()
        result = ComparisonResult("VER", "HAM", empty, empty, {}, residual=residual)
        assert result.matched is matched


class TestRawFastF1Telemetry:
    """FastF1's own ``Telemetry`` objects go straight in, untouched."""

    def test_same_result_as_the_plain_frames(
        self,
        fastf1_tel_a: pd.DataFrame,
        fastf1_tel_b: pd.DataFrame,
        lap_tel_a: pd.DataFrame,
        lap_tel_b: pd.DataFrame,
    ) -> None:
        raw = compare_drivers(fastf1_tel_a, fastf1_tel_b, driver_a="VER", driver_b="HAM")
        plain = compare_drivers(lap_tel_a, lap_tel_b, driver_a="VER", driver_b="HAM")
        # FastF1's own clock comes through; the plain frames have none.
        assert {"Time_a", "Time_b"} <= set(raw.aligned.columns)
        assert not {"Time_a", "Time_b"} & set(plain.aligned.columns)
        pd.testing.assert_frame_equal(raw.aligned[plain.aligned.columns], plain.aligned)
        same = ["minisector", "start_distance", "end_distance", "mean_speed_a", "mean_speed_b"]
        pd.testing.assert_frame_equal(raw.minisectors[same], plain.minisectors[same])
        assert raw.minisectors["fastest"].tolist() == plain.minisectors["fastest"].tolist()
        ends = ["segment", "start_distance", "end_distance"]
        pd.testing.assert_frame_equal(raw.segments[ends], plain.segments[ends])
        assert raw.summary == plain.summary
        assert raw.matched
        assert raw.residual == pytest.approx(plain.residual)

    def test_inputs_are_not_modified(
        self, fastf1_tel_a: pd.DataFrame, fastf1_tel_b: pd.DataFrame
    ) -> None:
        before_a, before_b = fastf1_tel_a.copy(), fastf1_tel_b.copy()
        compare_drivers(fastf1_tel_a, fastf1_tel_b, driver_a="VER", driver_b="HAM")
        pd.testing.assert_frame_equal(fastf1_tel_a, before_a)
        pd.testing.assert_frame_equal(fastf1_tel_b, before_b)
