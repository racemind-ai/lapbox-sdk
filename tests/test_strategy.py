"""Unit tests for the strategy engine: plans, race time lap by lap, track condition, cautions.

Moved with the engine from LapBox (``tests/unit/test_strategy_engine.py`` and the engine-physics
cases of ``tests/unit/test_strategy_room.py``; its two engine review fixes followed in P12),
unchanged but for the imports.
"""

from __future__ import annotations

import numpy as np
import pytest

from lapbox.strategy import (
    RaceModel,
    Stint,
    Strategy,
    StrategyEngine,
    condition_pace_multiplier,
    temp_degradation_factor,
    wrong_compound_penalty,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def model() -> RaceModel:
    return RaceModel(
        total_laps=20,
        base_lap_time=90.0,
        compound_offsets={"SOFT": -0.5, "MEDIUM": 0.0, "HARD": 0.6},
        degradation={"SOFT": 0.10, "MEDIUM": 0.05, "HARD": 0.03},
        pit_loss=20.0,
        fuel_effect_per_lap=0.05,
    )


class TestStrategy:
    def test_from_pairs_and_properties(self) -> None:
        s = Strategy.from_pairs([("SOFT", 8), ("MEDIUM", 12)])
        assert s.total_laps == 20
        assert s.n_stops == 1
        assert s.pit_laps == [8]
        assert s.compounds == ["SOFT", "MEDIUM"]

    def test_two_stop_pit_laps(self) -> None:
        s = Strategy.from_pairs([("SOFT", 7), ("MEDIUM", 7), ("HARD", 6)])
        assert s.pit_laps == [7, 14]
        assert s.n_stops == 2

    def test_invalid_stint(self) -> None:
        with pytest.raises(ValueError):
            Stint(compound="SOFT", laps=0)


class TestStrategyEngine:
    def test_per_lap_length_and_monotone_fuel(self, model: RaceModel) -> None:
        engine = StrategyEngine(model)
        s = Strategy.from_pairs([("MEDIUM", 20)])
        times = engine.per_lap_times(s)
        assert len(times) == 20
        # With no compound change, degradation rises but fuel falls; check the
        # fuel term specifically by zeroing degradation.
        flat = StrategyEngine(
            RaceModel(
                total_laps=20,
                base_lap_time=90.0,
                degradation={"MEDIUM": 0.0},
                fuel_effect_per_lap=0.05,
            )
        ).per_lap_times(s)
        assert flat[0] > flat[-1]  # heavier (more fuel) early -> slower

    def test_race_time_includes_pit_loss(self, model: RaceModel) -> None:
        engine = StrategyEngine(model)
        one_stop = Strategy.from_pairs([("SOFT", 10), ("MEDIUM", 10)])
        no_stop = Strategy.from_pairs([("MEDIUM", 20)])
        result = engine.simulate(one_stop)
        assert result.n_stops == 1
        assert result.pit_loss_total == pytest.approx(20.0)
        assert result.total_time == pytest.approx(result.green_lap_time + 20.0)
        # The no-stop race pays no pit loss.
        assert engine.simulate(no_stop).pit_loss_total == 0.0

    def test_lap_count_mismatch_raises(self, model: RaceModel) -> None:
        engine = StrategyEngine(model)
        with pytest.raises(ValueError):
            engine.per_lap_times(Strategy.from_pairs([("SOFT", 5)]))

    def test_degradation_makes_long_stints_costly(self, model: RaceModel) -> None:
        engine = StrategyEngine(model)
        # Same compounds, but a very long SOFT stint should degrade more.
        early = engine.race_time(Strategy.from_pairs([("SOFT", 5), ("MEDIUM", 15)]))
        late = engine.race_time(Strategy.from_pairs([("SOFT", 15), ("MEDIUM", 5)]))
        assert late > early  # SOFT degrades faster -> running it longer is slower

    def test_average_lap_time(self, model: RaceModel) -> None:
        result = StrategyEngine(model).simulate(Strategy.from_pairs([("MEDIUM", 20)]))
        assert result.average_lap_time == pytest.approx(result.green_lap_time / 20)
        assert isinstance(result.per_lap_times, np.ndarray)


# --- engine physics: cliff, condition, temperature, cautions, lap trace ---
def _model(**overrides) -> RaceModel:
    defaults: dict = {
        "total_laps": 50,
        "base_lap_time": 90.0,
        "compound_offsets": {"SOFT": 0.0, "MEDIUM": 0.3, "HARD": 0.7},
        "degradation": {"SOFT": 0.10, "MEDIUM": 0.06, "HARD": 0.04},
        "pit_loss": 22.0,
    }
    defaults.update(overrides)
    return RaceModel(**defaults)


def _strategy(pit: int = 22, total: int = 50) -> Strategy:
    return Strategy.from_pairs([("SOFT", pit), ("MEDIUM", total - pit)])


class TestEnginePhysics:
    def test_defaults_reproduce_legacy_behaviour(self) -> None:
        # No cliff/SC/weather: formula identical to the original engine.
        engine = StrategyEngine(_model())
        times = engine.per_lap_times(_strategy())
        expected_lap1 = 90.0 + 0.0 + 0.10 * 1 + 0.06 * 49
        assert times[0] == pytest.approx(expected_lap1)

    def test_cliff_adds_extra_slope_past_the_knee(self) -> None:
        base = StrategyEngine(_model()).per_lap_times(_strategy())
        cliffed = StrategyEngine(_model(cliff_lap={"SOFT": 18})).per_lap_times(_strategy())
        # Identical up to the cliff, then +0.5 s per lap past it.
        assert cliffed[17] == pytest.approx(base[17])  # age 18 = at the knee
        assert cliffed[18] == pytest.approx(base[18] + 0.5)  # age 19
        assert cliffed[21] == pytest.approx(base[21] + 0.5 * 4)  # age 22

    def test_condition_multiplier_and_wrong_compound_penalty(self) -> None:
        assert condition_pace_multiplier("dry") == 1.0
        assert condition_pace_multiplier("wet") == pytest.approx(1.18)
        assert wrong_compound_penalty("wet", "SOFT") == pytest.approx(8.0)
        assert wrong_compound_penalty("damp", "INTERMEDIATE") == 0.0
        assert wrong_compound_penalty("dry", "INTERMEDIATE") == pytest.approx(5.5)

    def test_temperature_scales_degradation(self) -> None:
        assert temp_degradation_factor(30.0) == pytest.approx(1.0)
        assert temp_degradation_factor(40.0) == pytest.approx(1.3)
        assert temp_degradation_factor(0.0) == pytest.approx(0.7)  # clamped

        hot = StrategyEngine(_model(track_temp_c=40.0)).per_lap_times(_strategy())
        cool = StrategyEngine(_model(track_temp_c=30.0)).per_lap_times(_strategy())
        # Same lap, older tyre => hotter track degrades faster.
        assert hot[15] > cool[15]

    def test_sc_window_slows_laps_and_halves_pit_loss(self) -> None:
        model = _model(sc_windows=((20, 23, "SC"),))
        engine = StrategyEngine(model)
        strategy = _strategy(pit=22)  # pit lap inside the window
        times = engine.per_lap_times(strategy)
        assert times[19] == pytest.approx(times[18] / 1.0 * 1.6, rel=0.05)
        result = engine.simulate(strategy)
        assert result.pit_loss_total == pytest.approx(22.0 * 0.5)

        vsc = StrategyEngine(_model(sc_windows=((22, 23, "VSC"),))).simulate(strategy)
        assert vsc.pit_loss_total == pytest.approx(22.0 * 0.65)

    def test_lap_trace_shape_and_flags(self) -> None:
        model = _model(cliff_lap={"SOFT": 18}, sc_windows=((20, 23, "SC"),))
        trace = StrategyEngine(model).lap_trace(_strategy(pit=22))
        assert len(trace["lap"]) == 50
        assert trace["compound"][0] == "SOFT" and trace["compound"][-1] == "MEDIUM"
        assert trace["sc_state"][19] == "SC" and trace["sc_state"][24] is None
        assert sum(trace["in_cliff"]) == 4  # ages 19..22 on the soft
        # Cumulative time includes the (discounted) pit loss at the in-lap.
        jump = trace["cum_time"][21] - trace["cum_time"][20]
        assert jump == pytest.approx(trace["lap_time"][21] + 11.0)


class TestReviewFixes:
    """Regression tests for the adversarial-review findings (the engine ones)."""

    def test_wet_physics_not_inverted(self) -> None:
        # THE big one: rain tyres must beat slicks in the rain.
        offsets = {"SOFT": 0.0, "MEDIUM": 0.3, "INTERMEDIATE": 8.0, "WET": 14.0}
        deg = {"SOFT": 0.10, "MEDIUM": 0.06, "INTERMEDIATE": 0.05, "WET": 0.03}

        def fresh_lap(condition: str, compound: str) -> float:
            model = _model(
                total_laps=10, compound_offsets=offsets, degradation=deg, condition=condition
            )
            engine = StrategyEngine(model)
            return float(engine.per_lap_times(Strategy.from_pairs([(compound, 10)]))[0])

        # Wet: full wets fastest, inters second, slicks nowhere.
        assert fresh_lap("wet", "WET") < fresh_lap("wet", "INTERMEDIATE") < fresh_lap("wet", "SOFT")
        # Damp: inters fastest.
        assert fresh_lap("damp", "INTERMEDIATE") < fresh_lap("damp", "WET")
        assert fresh_lap("damp", "INTERMEDIATE") < fresh_lap("damp", "SOFT")
        # Dry unchanged: slicks fastest, rain tyres penalised.
        assert fresh_lap("dry", "SOFT") < fresh_lap("dry", "INTERMEDIATE")

    def test_sc_lap_time_is_pace_limited_not_tyre_dependent(self) -> None:
        # Under SC, an old-tyre car and a fresh-tyre car lap at the same speed.
        model = _model(sc_windows=((30, 33, "SC"),))
        engine = StrategyEngine(model)
        old_tyres = engine.per_lap_times(Strategy.from_pairs([("SOFT", 40), ("MEDIUM", 10)]))
        fresh_tyres = engine.per_lap_times(Strategy.from_pairs([("SOFT", 29), ("MEDIUM", 21)]))
        assert old_tyres[30] == pytest.approx(fresh_tyres[30])  # both SC laps
        assert old_tyres[30] == pytest.approx(90.0 * 1.6)
