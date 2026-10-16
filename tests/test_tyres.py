"""Unit tests for tyres: the per-compound degradation model and stint-by-stint degradation.

The model tests are LapBox's own (``tests/unit/test_tyre_degradation.py``), on the
same synthetic engineered laps. The stint cases are ported one-for-one from the
TypeScript tests of the Race Analysis page's tyre-life panel (``truePace.test.ts``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lapbox.practice import FUEL_EFFECT_S_PER_LAP
from lapbox.tyres import TyreDegradationModel, compound_summary, stint_degradation

pytestmark = pytest.mark.unit


def engineered_laps(*, n_drivers: int = 8, laps_per_driver: int = 25) -> pd.DataFrame:
    """LapBox's engineered laps with a known signal (``build_ml_laps`` in its tests).

    ``lap_time = 90 + driver_offset + event_offset + 0.08·tyre_age + fuel effect + noise``,
    the fuel effect ``FUEL_EFFECT_S_PER_LAP`` per lap still to run, as the model
    assumes. ``fuel_load_est`` burns down from 110 kg like LapBox's (4.4 kg a lap).
    Pit lap and starting compound vary per driver.
    """
    rng = np.random.default_rng(0)
    events = (("Monza GP", -1.5), ("Silverstone GP", 1.5))
    rows: list[dict] = []
    for d in range(n_drivers):
        offset = float(rng.normal(0, 0.3))
        pit_lap = int(rng.integers(9, 17))
        start_soft = d % 2 == 0
        for event_name, event_offset in events:
            for lap in range(1, laps_per_driver + 1):
                stint = 1 if lap <= pit_lap else 2
                tyre_age = lap if stint == 1 else lap - pit_lap
                first, second = ("SOFT", "MEDIUM") if start_soft else ("MEDIUM", "SOFT")
                fuel = 110.0 * (laps_per_driver - lap) / laps_per_driver
                rows.append(
                    {
                        "driver": f"DR{d:02d}",
                        "compound": first if stint == 1 else second,
                        "event_name": event_name,
                        "lap_number": lap,
                        "stint": stint,
                        "stint_lap": tyre_age,
                        "tyre_age": tyre_age,
                        "fuel_load_est": fuel,
                        "lap_time_seconds": 90.0
                        + offset
                        + event_offset
                        + 0.08 * tyre_age
                        + FUEL_EFFECT_S_PER_LAP * (laps_per_driver - lap)
                        + float(rng.normal(0, 0.05)),
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def ml_laps() -> pd.DataFrame:
    return engineered_laps()


def as_fastf1(laps: pd.DataFrame) -> pd.DataFrame:
    """The same laps in FastF1's own column names (Timedelta lap times)."""
    return pd.DataFrame(
        {
            "Driver": laps["driver"],
            "LapNumber": laps["lap_number"].astype(float),
            "LapTime": pd.to_timedelta(laps["lap_time_seconds"], unit="s"),
            "Compound": laps["compound"],
            "TyreLife": laps["tyre_age"].astype(float),
            "PitInTime": pd.Series(pd.NaT, index=laps.index, dtype="timedelta64[ns]"),
            "PitOutTime": pd.Series(pd.NaT, index=laps.index, dtype="timedelta64[ns]"),
        }
    )


# --------------------------------------------------------------------------- #
# TyreDegradationModel (LapBox's tests)
# --------------------------------------------------------------------------- #
class TestTyreDegradationModel:
    def test_recovers_degradation_rate(self, ml_laps: pd.DataFrame) -> None:
        # Degradation is fitted on ONE session's laps in LapBox, so fit on a single
        # event: the two-event fixture would otherwise inject a track offset.
        model = TyreDegradationModel().fit(ml_laps[ml_laps["event_name"] == "Monza GP"])
        assert {"SOFT", "MEDIUM"}.issubset(model.compounds.keys())
        # The synthetic data has 0.08 s/lap; recovered after fuel correction
        # (driver-to-driver offsets keep R² near 0.5).
        for compound in ("SOFT", "MEDIUM"):
            assert model.degradation_rate(compound) == pytest.approx(0.08, abs=0.02)
            assert model.compounds[compound].r2 > 0.4

    def test_predict_time_loss_scales_with_age(self, ml_laps: pd.DataFrame) -> None:
        model = TyreDegradationModel().fit(ml_laps)
        loss5 = model.predict_time_loss("SOFT", 5)
        loss10 = model.predict_time_loss("SOFT", 10)
        assert loss10 == pytest.approx(2 * loss5, rel=1e-6)
        assert loss10 > 0

    def test_summary_table(self, ml_laps: pd.DataFrame) -> None:
        summary = TyreDegradationModel().fit(ml_laps).summary()
        assert set(summary["compound"]) == {"SOFT", "MEDIUM"}
        assert "degradation_s_per_lap" in summary.columns

    def test_unknown_compound_raises(self, ml_laps: pd.DataFrame) -> None:
        model = TyreDegradationModel().fit(ml_laps)
        with pytest.raises(KeyError):
            model.predict_time_loss("WET", 5)

    def test_missing_columns_raise(self) -> None:
        with pytest.raises(ValueError):
            TyreDegradationModel().fit(pd.DataFrame({"compound": ["SOFT"]}))


def one_stint(*, start: int = 1, laps: int = 10, total: int = 50, kg: float = 2.0) -> pd.DataFrame:
    """One driver's stint on new hards with no noise: tyre age and fuel both move
    with the lap number. ``lap_time = 92 + 0.1·age + fuel effect``."""
    lap = np.arange(start, start + laps, dtype=float)
    age = lap - start + 1
    return pd.DataFrame(
        {
            "driver": "DR00",
            "compound": "HARD",
            "lap_number": lap,
            "tyre_age": age,
            "fuel_load_est": kg * (total - lap),
            "lap_time_seconds": 92.0 + 0.1 * age + FUEL_EFFECT_S_PER_LAP * (total - lap),
        }
    )


class TestFuelAndRepresentativeLaps:
    """The fuel effect is fixed, not fitted; only laps within 107 % are fitted."""

    def test_a_stint_where_age_and_fuel_move_together_is_recovered(self) -> None:
        # A fitted fuel coefficient cannot be told apart from tyre age here.
        fit = TyreDegradationModel().fit(one_stint()).compounds["HARD"]
        assert fit.degradation_s_per_lap == pytest.approx(0.1, abs=1e-9)
        assert fit.fuel_coef == pytest.approx(FUEL_EFFECT_S_PER_LAP / 2.0)
        assert fit.intercept_s == pytest.approx(92.0, abs=1e-9)
        assert fit.r2 == pytest.approx(1.0)
        assert fit.laps == 10

    def test_the_fuel_unit_is_read_from_the_burn_per_lap(self) -> None:
        heavy = TyreDegradationModel().fit(one_stint(kg=1.6)).compounds["HARD"]
        assert heavy.fuel_coef == pytest.approx(FUEL_EFFECT_S_PER_LAP / 1.6)
        assert heavy.degradation_s_per_lap == pytest.approx(0.1, abs=1e-9)

    def test_a_constant_fuel_load_gets_a_zero_coefficient(self) -> None:
        # Practice and qualifying: LapBox sets one fuel load for every lap.
        laps = pd.DataFrame(
            {
                "compound": "SOFT",
                "tyre_age": [1.0, 2.0, 3.0, 4.0, 5.0],
                "fuel_load_est": 10.0,
                "lap_time_seconds": [80.0, 80.1, 80.2, 80.3, 80.4],
            }
        )
        fit = TyreDegradationModel().fit(laps).compounds["SOFT"]
        assert fit.fuel_coef == 0.0
        assert fit.degradation_s_per_lap == pytest.approx(0.1)

    def test_a_falling_fuel_load_without_lap_numbers_raises(self) -> None:
        laps = one_stint().drop(columns=["lap_number"])
        with pytest.raises(ValueError, match="lap number"):
            TyreDegradationModel().fit(laps)
        fit = TyreDegradationModel(use_fuel_correction=False).fit(laps).compounds["HARD"]
        assert fit.fuel_coef is None
        assert fit.degradation_s_per_lap == pytest.approx(0.1 - FUEL_EFFECT_S_PER_LAP)

    def test_missing_fuel_loads_mean_no_correction(self) -> None:
        laps = one_stint().assign(fuel_load_est=np.nan)
        fit = TyreDegradationModel().fit(laps).compounds["HARD"]
        assert fit.fuel_coef is None
        assert fit.laps == 10

    def test_laps_beyond_107_percent_of_the_drivers_best_are_left_out(self) -> None:
        laps = one_stint(laps=12)
        slow = laps.copy()
        slow.loc[[4, 5], "lap_time_seconds"] *= 1.3  # a safety car
        fit = TyreDegradationModel().fit(slow).compounds["HARD"]
        assert fit.degradation_s_per_lap == pytest.approx(0.1, abs=1e-9)
        assert fit.r2 == pytest.approx(1.0)
        assert fit.laps == 10

    def test_107_percent_is_measured_against_each_drivers_own_best(self) -> None:
        # A driver 8 % off the fastest keeps every lap.
        quick = one_stint()
        slow = one_stint().assign(
            driver="DR01", lap_time_seconds=lambda d: d["lap_time_seconds"] * 1.08
        )
        fit = TyreDegradationModel().fit(pd.concat([quick, slow], ignore_index=True))
        assert fit.compounds["HARD"].laps == 20

    def test_a_compound_without_age_spread_is_not_fitted(self) -> None:
        laps = pd.DataFrame(
            {
                "compound": "SOFT",
                "tyre_age": 3.0,
                "lap_time_seconds": [80.0, 80.1, 80.2, 80.3, 80.4],
            }
        )
        assert TyreDegradationModel().fit(laps).compounds == {}


class TestFastF1Laps:
    def test_fastf1_columns_give_the_same_degradation(self, ml_laps: pd.DataFrame) -> None:
        monza = ml_laps[ml_laps["event_name"] == "Monza GP"]
        engineered = TyreDegradationModel().fit(monza).compounds
        fastf1 = TyreDegradationModel().fit(as_fastf1(monza)).compounds
        assert engineered.keys() == fastf1.keys()
        for compound, fit in engineered.items():
            assert fastf1[compound].degradation_s_per_lap == pytest.approx(
                fit.degradation_s_per_lap, rel=1e-6
            )
            assert fastf1[compound].r2 == pytest.approx(fit.r2, rel=1e-6)
            # Laps still to run instead of kg: the same correction, in its own unit.
            assert fastf1[compound].fuel_coef == pytest.approx(FUEL_EFFECT_S_PER_LAP)
            assert fit.fuel_coef == pytest.approx(FUEL_EFFECT_S_PER_LAP * 25 / 110)

    def test_pit_laps_are_dropped(self, ml_laps: pd.DataFrame) -> None:
        laps = as_fastf1(ml_laps[ml_laps["event_name"] == "Monza GP"]).reset_index(drop=True)
        with_pit = laps.copy()
        # 2 s slow: inside 107 %, so only the pit rule can drop it.
        with_pit.loc[0, "LapTime"] += pd.Timedelta(seconds=2)
        with_pit.loc[0, "PitInTime"] = pd.Timedelta(minutes=30)
        expected = TyreDegradationModel().fit(laps.drop(index=0)).compounds
        assert TyreDegradationModel().fit(with_pit).compounds == expected

    def test_tyre_age_falls_back_to_stint_lap(self, ml_laps: pd.DataFrame) -> None:
        a = TyreDegradationModel().fit(ml_laps).compounds
        b = TyreDegradationModel().fit(ml_laps.drop(columns=["tyre_age"])).compounds
        assert a == b  # the fixture's stint_lap equals its tyre_age

    def test_input_is_not_modified(self, ml_laps: pd.DataFrame) -> None:
        laps = as_fastf1(ml_laps)
        before = laps.copy()
        TyreDegradationModel().fit(laps)
        pd.testing.assert_frame_equal(laps, before)


# --------------------------------------------------------------------------- #
# stint_degradation / compound_summary (the tyre-life panel's TypeScript tests)
# --------------------------------------------------------------------------- #
def stint_laps(rows: list[tuple]) -> pd.DataFrame:
    """``(driver, lap, seconds, stint, compound)`` rows as FastF1 laps."""
    return pd.DataFrame(
        {
            "Driver": [r[0] for r in rows],
            "Team": ["Team " + r[0] for r in rows],
            "LapNumber": [float(r[1]) for r in rows],
            "LapTime": [pd.Timedelta(seconds=r[2]) for r in rows],
            "Stint": [float(r[3]) for r in rows],
            "Compound": [r[4] for r in rows],
            "PitInTime": pd.Series([pd.NaT] * len(rows), dtype="timedelta64[ns]"),
            "PitOutTime": pd.Series([pd.NaT] * len(rows), dtype="timedelta64[ns]"),
        }
    )


class TestStintDegradation:
    def test_measures_the_tyre_trend_not_the_fuel_trend(self) -> None:
        # Raw times FLAT across the stint. That is not zero degradation: the car was
        # getting lighter, so the tyre must have been going off to cancel it.
        laps = stint_laps([("VER", n, 95.0, 1, "SOFT") for n in range(10, 16)])
        [stint] = stint_degradation(laps, 60).itertuples()
        assert stint.deg_per_lap == pytest.approx(FUEL_EFFECT_S_PER_LAP, abs=1e-3)

    def test_reports_zero_degradation_when_pace_truly_holds(self) -> None:
        # Times improving exactly at the fuel-burn rate = flat true pace.
        laps = stint_laps(
            [("VER", n, 95.0 - n * FUEL_EFFECT_S_PER_LAP, 1, "HARD") for n in range(10, 15)]
        )
        assert stint_degradation(laps, 60)["deg_per_lap"].iloc[0] == pytest.approx(0, abs=1e-6)

    def test_splits_stints_and_keeps_their_compounds(self) -> None:
        laps = stint_laps(
            [("VER", n, 95.0, 1, "SOFT") for n in range(1, 5)]
            + [("VER", n, 96.0, 2, "HARD") for n in range(5, 9)]
        )
        assert stint_degradation(laps, 20)["compound"].tolist() == ["SOFT", "HARD"]

    def test_skips_stints_too_short_to_fit_a_trend(self) -> None:
        laps = stint_laps([("VER", 1, 95.0, 1, "SOFT"), ("VER", 2, 95.1, 1, "SOFT")])
        assert stint_degradation(laps, 20).empty

    def test_reads_the_team_and_needs_fastf1s_stint_column(self) -> None:
        laps = stint_laps([("VER", n, 95.0, 1, "SOFT") for n in range(1, 6)])
        assert stint_degradation(laps)["team"].tolist() == ["Team VER"]
        assert stint_degradation(laps.drop(columns=["Stint"])).empty


class TestCompoundSummary:
    def test_aggregates_degradation_per_compound_kindest_first(self) -> None:
        laps = stint_laps(
            [("A", n, 95.0 + n * 0.3, 1, "SOFT") for n in range(1, 5)]
            + [("A", n, 96.0, 2, "HARD") for n in range(5, 9)]
        )
        summary = compound_summary(stint_degradation(laps, 20)).set_index("compound")
        assert summary.loc["SOFT", "median_deg"] > summary.loc["HARD", "median_deg"]
        assert summary.loc["SOFT", "stints"] == 1
        assert summary.index[0] == "HARD"  # kindest first

    def test_ignores_stints_with_no_compound_recorded(self) -> None:
        stints = pd.DataFrame(
            [
                {
                    "driver": "A",
                    "team": None,
                    "stint": 1,
                    "compound": None,
                    "laps": 5,
                    "median_corrected": 95.0,
                    "deg_per_lap": 0.1,
                }
            ]
        )
        assert compound_summary(stints).empty
