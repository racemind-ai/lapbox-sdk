"""Unit tests for race pace: fuel correction, true pace, clean air, consistency, ideal lap.

The cases are ported one-for-one from the TypeScript tests of LapBox's Race Analysis
page (truePace, cleanAir, consistency and idealLap .test.ts), where the same maths
runs in the browser.
"""

from __future__ import annotations

import pandas as pd
import pytest

from lapbox.pace import (
    DIRTY_AIR_S,
    biggest_loss,
    clean_air_ranking,
    clean_air_split,
    consistency_ranking,
    driver_consistency,
    fuel_correct,
    ideal_lap,
    true_pace_ranking,
)
from lapbox.pace.race import _drivers, _Lap, _slope
from lapbox.practice import FUEL_EFFECT_S_PER_LAP

pytestmark = pytest.mark.unit


def lap(
    n: int,
    seconds: float | None,
    *,
    driver: str = "VER",
    pit: bool = False,
    compound: str = "MEDIUM",
    s1: float | None = None,
    s2: float | None = None,
    s3: float | None = None,
    gap: float | None = None,
) -> dict:
    return {
        "driver": driver,
        "lap": n,
        "seconds": seconds,
        "pit": pit,
        "compound": compound,
        "s": (s1, s2, s3),
        "gap": gap,
    }


def laps_frame(rows: list[dict]) -> pd.DataFrame:
    """Rows as FastF1 has them: Timedelta times, pit-in time on a pit lap."""
    td = lambda s: pd.NaT if s is None else pd.Timedelta(seconds=s)  # noqa: E731
    return pd.DataFrame(
        {
            "Driver": [r["driver"] for r in rows],
            "Team": ["Team " + r["driver"] for r in rows],
            "LapNumber": [float(r["lap"]) for r in rows],
            "LapTime": [td(r["seconds"]) for r in rows],
            "Compound": [r["compound"] for r in rows],
            "PitInTime": [pd.Timedelta(minutes=30) if r["pit"] else pd.NaT for r in rows],
            "PitOutTime": [pd.NaT for _ in rows],
            "Sector1Time": [td(r["s"][0]) for r in rows],
            "Sector2Time": [td(r["s"][1]) for r in rows],
            "Sector3Time": [td(r["s"][2]) for r in rows],
        },
        columns=[
            "Driver",
            "Team",
            "LapNumber",
            "LapTime",
            "Compound",
            "PitInTime",
            "PitOutTime",
            "Sector1Time",
            "Sector2Time",
            "Sector3Time",
        ],
    )


def gaps_of(rows: list[dict]) -> dict[tuple[str, int], float]:
    return {(r["driver"], r["lap"]): r["gap"] for r in rows if r["gap"] is not None}


def run(start: int, n: int, seconds: float, gap: float | None, driver: str = "VER") -> list[dict]:
    """n laps at ``seconds``, all with the same gap, starting at lap ``start``."""
    return [lap(start + i, seconds, driver=driver, gap=gap) for i in range(n)]


# --------------------------------------------------------------------------- #
# fuelCorrect
# --------------------------------------------------------------------------- #
class TestFuelCorrect:
    def test_removes_the_fuel_advantage_of_later_laps(self) -> None:
        # Two identical raw laps at opposite ends of a 50-lap race are NOT equal
        # pace: the later one was run on a far lighter car.
        out = fuel_correct(laps_frame([lap(1, 95), lap(50, 95)]), 50)
        first, last = out["corrected"].tolist()
        assert first < last
        assert last == pytest.approx(95)  # final lap needs no correction
        assert first == pytest.approx(95 - 49 * FUEL_EFFECT_S_PER_LAP)

    def test_makes_a_car_with_constant_true_pace_look_constant(self) -> None:
        # A car getting steadily faster purely from fuel burn has flat true pace.
        rows = [lap(n, 95 - (n - 1) * FUEL_EFFECT_S_PER_LAP) for n in (1, 2, 3, 4)]
        corrected = fuel_correct(laps_frame(rows), 4)["corrected"].tolist()
        for c in corrected:
            assert c == pytest.approx(corrected[0], abs=1e-6)

    def test_skips_pit_and_untimed_laps(self) -> None:
        out = fuel_correct(laps_frame([lap(1, 95), lap(2, None), lap(3, 99, pit=True)]), 10)
        assert len(out) == 1


class TestSlope:
    def test_measures_the_per_lap_trend(self) -> None:
        assert _slope([1, 2, 3, 4], [10, 10.1, 10.2, 10.3]) == pytest.approx(0.1)

    def test_is_zero_when_undefined(self) -> None:
        assert _slope([1], [10]) == 0
        assert _slope([2, 2], [10, 11]) == 0  # no spread in x


# --------------------------------------------------------------------------- #
# truePaceRanking
# --------------------------------------------------------------------------- #
class TestTruePaceRanking:
    def test_reorders_the_field_once_fuel_is_accounted_for(self) -> None:
        # SLOW ran only late (light) laps; FAST ran only early (heavy) laps.
        # Raw times say SLOW was quicker; corrected says the opposite.
        rows = [lap(n, 94.0, driver="SLOW") for n in (40, 41, 42, 43)] + [
            lap(n, 94.5, driver="FAST") for n in (1, 2, 3, 4)
        ]
        ranked = true_pace_ranking(laps_frame(rows), 50)
        top = ranked.iloc[0]
        assert top["driver"] == "FAST"
        # FAST was 2nd on raw times and 1st once corrected.
        assert top["raw_rank"] == 2
        assert top["corrected_rank"] == 1
        assert top["rank_change"] == 1

    def test_reports_the_gap_to_the_quickest_corrected_pace(self) -> None:
        rows = [lap(n, 95, driver="A") for n in (1, 2, 3)] + [
            lap(n, 95.5, driver="B") for n in (1, 2, 3)
        ]
        ranked = true_pace_ranking(laps_frame(rows), 10)
        assert ranked["gap"].iloc[0] == 0
        assert ranked["gap"].iloc[1] == pytest.approx(0.5)

    def test_excludes_compromised_laps_from_the_median(self) -> None:
        rows = [lap(1, 95), lap(2, 95.1), lap(3, 95.2), lap(4, 140)]  # safety car
        assert true_pace_ranking(laps_frame(rows), 4)["laps"].iloc[0] == 3

    def test_ignores_drivers_with_too_few_laps_to_judge(self) -> None:
        assert true_pace_ranking(laps_frame([lap(1, 95), lap(2, 95.1)]), 10).empty


# --------------------------------------------------------------------------- #
# cleanAirSplit / cleanAirRanking
# --------------------------------------------------------------------------- #
def split(rows: list[dict], total: int = 60):
    return clean_air_split(laps_frame(rows), "VER", total, gaps=gaps_of(rows))


class TestCleanAirSplit:
    def test_measures_what_following_another_car_costs(self) -> None:
        # Clean and traffic laps INTERLEAVED, so both sides sit at the same
        # average fuel load and the only difference left is the traffic itself.
        rows = []
        for n in range(2, 22, 2):
            rows.append(lap(n, 95.0, gap=8.0))  # clear air
            rows.append(lap(n + 1, 95.5, gap=1.0))  # following a car
        s = split(rows)
        assert s.clean_laps == 10
        assert s.traffic_laps == 10
        assert s.delta == pytest.approx(0.5, abs=0.05)

    def test_counts_leading_a_lap_as_the_cleanest_air(self) -> None:
        s = split(run(2, 4, 95.0, None) + run(6, 4, 95.6, 1.2))
        assert s.led_laps == 4
        assert s.clean_laps == 4
        assert s.delta > 0

    def test_splits_exactly_at_the_dirty_air_threshold(self) -> None:
        # A gap of exactly 2.0s counts as traffic; just over it does not.
        s = split(run(2, 4, 95.0, DIRTY_AIR_S + 0.01) + run(6, 4, 95.0, DIRTY_AIR_S))
        assert s.clean_laps == 4
        assert s.traffic_laps == 4

    def test_ignores_lap_1_which_is_decided_by_the_grid_and_the_start(self) -> None:
        s = split([lap(1, 120, gap=0.3)] + run(2, 4, 95.0, 8.0) + run(6, 4, 95.5, 1.0))
        assert s.clean_laps + s.traffic_laps == 8

    def test_excludes_pit_and_compromised_laps(self) -> None:
        s = split(
            run(2, 4, 95.0, 8.0)
            + [lap(6, 95.2, gap=8.0, pit=True), lap(7, 140, gap=8.0)]  # pit; safety car
            + run(8, 4, 95.5, 1.0)
        )
        assert s.clean_laps == 4
        assert s.traffic_laps == 4

    def test_fuel_corrects_before_comparing(self) -> None:
        # Traffic laps run EARLY (heavy) and clean laps LATE (light). Identical raw
        # times at different fuel loads mean the early laps were genuinely
        # quicker, so traffic shows as a gain, not a loss.
        s = split(run(2, 6, 95.0, 1.0) + run(40, 6, 95.0, 9.0))
        assert s.delta < 0

    def test_is_none_when_either_side_is_too_small_for_a_median(self) -> None:
        assert split(run(2, 10, 95, 8.0)) is None  # never in traffic
        assert split(run(2, 10, 95, 1.0)) is None  # never clear
        assert split([]) is None

    def test_reports_the_share_of_racing_laps_spent_in_traffic(self) -> None:
        s = split(run(2, 3, 95.0, 8.0) + run(5, 9, 95.5, 1.0))
        assert s.traffic_share == pytest.approx(75, abs=0.5)


class TestCleanAirRanking:
    def test_ranks_the_biggest_traffic_penalty_first(self) -> None:
        fine = run(2, 5, 95.0, 8.0, "B") + run(7, 5, 95.1, 1.0, "B")
        hurt = run(2, 5, 95.0, 8.0, "A") + run(7, 5, 96.0, 1.0, "A")
        rows = fine + hurt
        ranked = clean_air_ranking(laps_frame(rows), 60, gaps=gaps_of(rows))
        assert ranked["driver"].tolist() == ["A", "B"]

    def test_drops_drivers_without_enough_of_both_kinds_of_lap(self) -> None:
        rows = run(2, 10, 95, 8.0, "LEADER")
        assert clean_air_ranking(laps_frame(rows), 60, gaps=gaps_of(rows)).empty


# --------------------------------------------------------------------------- #
# driverConsistency / consistencyRanking
# --------------------------------------------------------------------------- #
def timed(driver: str, seconds: list[float | None], pit: tuple[int, ...] = ()) -> list[dict]:
    return [lap(i + 1, s, driver=driver, pit=(i + 1) in pit) for i, s in enumerate(seconds)]


class TestDriverConsistency:
    def test_computes_cv_best_and_gap_to_best_over_representative_laps(self) -> None:
        c = driver_consistency(laps_frame(timed("VER", [90.0, 90.2, 90.1, 90.3])), "VER")
        assert c.best == pytest.approx(90.0)
        assert c.laps == 4
        assert c.series[0] == (1, 0)
        assert 0 < c.cv < 1  # very consistent -> tiny CV

    def test_excludes_pit_laps_and_compromised_laps(self) -> None:
        # A safety-car lap (120s) and a pit lap must not count.
        rows = timed("HAM", [90.0, 90.2, 120.0, 90.1, 95.0], pit=(5,))
        c = driver_consistency(laps_frame(rows), "HAM")
        assert c.laps == 3  # 90.0, 90.2, 90.1
        assert c.best == pytest.approx(90.0)

    def test_is_none_with_too_few_timed_laps(self) -> None:
        assert driver_consistency(laps_frame(timed("BOT", [90.0, None])), "BOT") is None


class TestConsistencyRanking:
    def test_orders_the_named_drivers_most_consistent_first(self) -> None:
        rows = timed("VER", [90.0, 90.05, 90.02, 90.03]) + timed("LEC", [90.0, 90.6, 90.1, 90.5])
        ranked = consistency_ranking(laps_frame(rows), ["VER", "LEC"])
        assert ranked["driver"].tolist() == ["VER", "LEC"]
        assert ranked["cv"].iloc[0] < ranked["cv"].iloc[1]


# --------------------------------------------------------------------------- #
# idealLap / biggestLoss
# --------------------------------------------------------------------------- #
def ideal(rows: list[dict]):
    return ideal_lap(laps_frame(rows), "VER")


THREE_LAPS = [
    lap(1, 91.5, s1=30.0, s2=30.7, s3=30.8),  # best S1 here
    lap(2, 91.0, s1=30.4, s2=30.4, s3=30.2),  # fastest actual lap, best only in S3
    lap(3, 91.4, s1=30.5, s2=30.1, s3=30.8),  # best S2 here
]


class TestIdealLap:
    def test_sums_the_best_sector_from_each_lap_and_reports_the_gain(self) -> None:
        r = ideal(THREE_LAPS)
        assert [s.seconds for s in r.sectors] == pytest.approx([30.0, 30.1, 30.2])
        assert [s.lap for s in r.sectors] == [1, 3, 2]
        assert r.ideal == pytest.approx(90.3, abs=1e-6)
        assert r.fastest == pytest.approx(91.0)
        assert r.fastest_lap == 2
        assert r.gain == pytest.approx(0.7, abs=1e-6)
        assert r.fastest_sectors == pytest.approx((30.4, 30.4, 30.2))
        assert r.laps == 3

    def test_ignores_pit_and_untimed_laps_even_when_their_sectors_are_quick(self) -> None:
        r = ideal(
            [
                lap(1, 91.0, s1=30.3, s2=30.3, s3=30.4),
                lap(2, 96.0, s1=29.0, s2=30.5, s3=36.5, pit=True),  # in-lap
                lap(3, None, s1=29.1, s2=30.2, s3=30.2),  # no lap time at all
                lap(4, 91.2, s1=30.4, s2=30.2, s3=30.6),
            ]
        )
        assert r.laps == 2
        assert [s.seconds for s in r.sectors] == pytest.approx([30.3, 30.2, 30.4])
        assert r.ideal == pytest.approx(90.9, abs=1e-6)

    def test_takes_each_sector_independently_when_a_lap_is_missing_one(self) -> None:
        r = ideal(
            [
                lap(1, 91.5, s1=29.9, s2=None, s3=30.9),  # S2 untimed, S1 still counts
                lap(2, 91.0, s1=30.4, s2=30.3, s3=30.3),
            ]
        )
        assert [s.seconds for s in r.sectors] == pytest.approx([29.9, 30.3, 30.3])
        assert r.sectors[0].lap == 1

    def test_is_none_when_a_sector_was_never_timed_on_a_clean_lap(self) -> None:
        assert (
            ideal(
                [
                    lap(1, 91.0, s1=30.3, s2=30.3, s3=None),
                    lap(2, 91.2, s1=30.4, s2=30.2, s3=None),
                ]
            )
            is None
        )

    def test_is_none_without_clean_laps(self) -> None:
        assert ideal([]) is None
        assert ideal([lap(1, 95.0, s1=30, s2=30, s3=35, pit=True)]) is None

    def test_is_none_rather_than_a_fake_zero_gain_when_the_fastest_lap_lacks_a_split(
        self,
    ) -> None:
        # FastF1 leaves lap 1's S1 blank after a standing start, so the best S1
        # can only come from the slower lap: the "ideal" lap would be slower
        # than a lap actually driven.
        r = ideal(
            [
                lap(1, 90.0, s1=None, s2=30.0, s3=30.0),
                lap(2, 93.0, s1=31.5, s2=30.5, s3=31.0),
            ]
        )
        assert r is None

    def test_reports_a_zero_gain_when_the_fastest_lap_already_is_the_ideal_lap(self) -> None:
        r = ideal(
            [
                lap(1, 90.0, s1=30.1, s2=29.9, s3=30.0),
                lap(2, 91.0, s1=30.4, s2=30.3, s3=30.3),
            ]
        )
        assert r is not None
        assert r.gain == 0
        assert biggest_loss(r) is None


class TestBiggestLoss:
    def test_picks_the_sector_costing_the_most_on_the_fastest_lap(self) -> None:
        worst = biggest_loss(ideal(THREE_LAPS))
        # The fastest lap lost 0.4 in S1 and 0.3 in S2 against the best sectors.
        assert worst.sector == 1
        assert worst.loss_on_fastest == pytest.approx(0.4, abs=1e-6)


# --------------------------------------------------------------------------- #
# FastF1 input
# --------------------------------------------------------------------------- #
class TestFastF1Laps:
    ROWS = (
        run(1, 12, 95.0, None, "VER")
        + run(1, 12, 95.4, None, "HAM")
        + [lap(13, 99.0, driver="VER", pit=True)]
    )

    def test_a_fastf1_laps_object_gives_the_same_answers(self) -> None:
        from fastf1.core import Laps  # imported here: FastF1 is slow to import

        plain = laps_frame(self.ROWS)
        ff1 = Laps(plain)
        pd.testing.assert_frame_equal(true_pace_ranking(ff1), true_pace_ranking(plain))
        pd.testing.assert_frame_equal(consistency_ranking(ff1), consistency_ranking(plain))
        assert fuel_correct(ff1).equals(fuel_correct(plain))

    def test_seconds_columns_are_read_the_same_as_timedeltas(self) -> None:
        plain = laps_frame(self.ROWS)
        seconds = plain.assign(
            LapTimeSeconds=plain["LapTime"].dt.total_seconds(),
            PitInTimeSeconds=plain["PitInTime"].dt.total_seconds(),
        ).drop(columns=["LapTime", "PitInTime"])
        pd.testing.assert_frame_equal(true_pace_ranking(seconds), true_pace_ranking(plain))

    def test_total_laps_defaults_to_the_last_lap_run(self) -> None:
        plain = laps_frame(self.ROWS)
        pd.testing.assert_frame_equal(fuel_correct(plain), fuel_correct(plain, 13))

    def test_gaps_come_from_the_time_at_the_line(self) -> None:
        """Without ``gaps=``, the gap is read from FastF1's ``Time`` column."""
        rows = []
        for n in range(2, 12):
            behind = 1.0 if n <= 6 else 5.0  # in traffic for five laps, then clear
            rows += [lap(n, 95.0, driver="LEAD"), lap(n, 95.3 if n <= 6 else 95.0, driver="VER")]
            rows[-2]["time"] = 100.0 * n
            rows[-1]["time"] = 100.0 * n + behind
        frame = laps_frame(rows).assign(Time=[pd.Timedelta(seconds=r["time"]) for r in rows])
        s = clean_air_split(frame, "VER")
        assert s.traffic_laps == 5
        assert s.clean_laps == 5
        assert s.led_laps == 0
        assert clean_air_split(frame, "LEAD") is None  # always led: never in traffic

    def test_input_is_not_modified(self) -> None:
        frame = laps_frame(self.ROWS)
        before = frame.copy()
        true_pace_ranking(frame)
        clean_air_ranking(frame)
        consistency_ranking(frame)
        ideal_lap(frame, "VER")
        pd.testing.assert_frame_equal(frame, before)


# --------------------------------------------------------------------------- #
# Per-driver calls on a whole session
# --------------------------------------------------------------------------- #
class TestPerDriverCalls:
    """``driver_consistency`` and ``ideal_lap`` are called once per driver, so each
    reads only that driver's rows."""

    ROWS = (
        [
            lap(n, 90.0 + 0.1 * (n % 3), driver="VER", s1=30.0, s2=30.0 + 0.1 * (n % 2), s3=30.0)
            for n in range(1, 9)
        ]
        + [
            lap(n, 90.5 + 0.2 * (n % 2), driver="LEC", s1=30.2, s2=30.1, s3=30.2 + 0.1 * (n % 3))
            for n in range(1, 9)
        ]
        + [lap(9, 110.0, driver="VER", pit=True)]
    )

    def test_the_whole_session_gives_the_same_answer_as_the_drivers_own_laps(self) -> None:
        frame = laps_frame(self.ROWS)
        for driver in ("VER", "LEC"):
            own = frame[frame["Driver"] == driver]
            assert driver_consistency(frame, driver) is not None
            assert driver_consistency(frame, driver) == driver_consistency(own, driver)
            assert ideal_lap(frame, driver) is not None
            assert ideal_lap(frame, driver) == ideal_lap(own, driver)

    def test_a_driver_not_in_the_session_gets_none(self) -> None:
        frame = laps_frame(self.ROWS)
        assert driver_consistency(frame, "XXX") is None
        assert ideal_lap(frame, "XXX") is None

    def test_a_categorical_driver_column_gives_the_same_answers(self) -> None:
        frame = laps_frame(self.ROWS)
        cat = frame.assign(Driver=frame["Driver"].astype("category"))
        assert driver_consistency(cat, "LEC") == driver_consistency(frame, "LEC")
        assert ideal_lap(cat, "LEC") == ideal_lap(frame, "LEC")
        pd.testing.assert_frame_equal(consistency_ranking(cat), consistency_ranking(frame))


class TestReader:
    """``_drivers``, which every function here reads its laps through."""

    ROWS = [
        lap(3, 91.0, driver="LEC", s1=30.0, s2=30.5, s3=30.5),
        lap(2, 90.5, driver="VER", compound="HARD", s1=30.25),
        lap(1, 90.0, driver="VER"),
        lap(1, None, driver="LEC", pit=True),
        lap(4, 92.0, driver="LEC"),  # LapNumber removed below: ignored
    ]

    def frame(self) -> pd.DataFrame:
        frame = laps_frame(self.ROWS)
        frame.index = [40, 10, 30, 20, 50]  # FastF1 keeps its own index after filtering
        frame.loc[50, "LapNumber"] = float("nan")
        return frame

    def test_reads_every_field_per_driver_in_lap_order(self) -> None:
        got = _drivers(self.frame(), gaps={("VER", 2): 1.5})
        assert list(got) == ["LEC", "VER"]  # the order drivers first appear in
        assert [lp.lap for lp in got["LEC"]] == [1, 3]
        assert [lp.lap for lp in got["VER"]] == [1, 2]
        assert got["LEC"][0] == _Lap(
            lap=1, seconds=None, compound="MEDIUM", pit=True, sectors=(None, None, None),
            gap_ahead=None,
        )  # fmt: skip
        assert got["LEC"][1] == _Lap(
            lap=3, seconds=91.0, compound="MEDIUM", pit=False, sectors=(30.0, 30.5, 30.5),
            gap_ahead=None,
        )  # fmt: skip
        assert got["VER"][1] == _Lap(
            lap=2, seconds=90.5, compound="HARD", pit=False, sectors=(30.25, None, None),
            gap_ahead=1.5,
        )  # fmt: skip

    def test_without_a_compound_column_every_compound_is_none(self) -> None:
        got = _drivers(self.frame().drop(columns=["Compound"]), gaps={})
        assert all(lp.compound is None for driver_laps in got.values() for lp in driver_laps)

    def test_categorical_columns_read_like_plain_ones(self) -> None:
        frame = self.frame()
        cat = frame.assign(
            Driver=frame["Driver"].astype("category"),
            Compound=frame["Compound"].astype("category"),
        )
        assert _drivers(cat, gaps={}) == _drivers(frame, gaps={})
