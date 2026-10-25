"""Unit tests for planning on the strategy engine: Monte Carlo, pit-stop plans, pit window, undercut.

Moved from LapBox (``tests/unit/test_monte_carlo.py``, ``tests/unit/test_pit_stop.py`` and the
planning cases of ``tests/unit/test_strategy_room.py``), unchanged but for the imports and the
classes that now hold each module's ``engine`` fixture.
"""

from __future__ import annotations

import pytest

from lapbox.strategy import (
    MonteCarloConfig,
    MonteCarloSimulator,
    RaceModel,
    Strategy,
    StrategyEngine,
    generate_candidate_strategies,
    pit_window_sweep,
    recommend_strategy,
    shift_pit_lap,
    undercut_report,
)

pytestmark = pytest.mark.unit


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


# --- Monte Carlo (from test_monte_carlo.py) ---------------------------------------------------


class TestMonteCarlo:
    @pytest.fixture
    def engine(self) -> StrategyEngine:
        return StrategyEngine(
            RaceModel(
                total_laps=20,
                base_lap_time=90.0,
                compound_offsets={"SOFT": -0.5, "MEDIUM": 0.0},
                degradation={"SOFT": 0.10, "MEDIUM": 0.05},
                pit_loss=20.0,
            )
        )

    def test_deterministic_with_seed(self, engine: StrategyEngine) -> None:
        strat = Strategy.from_pairs([("SOFT", 10), ("MEDIUM", 10)])
        r1 = MonteCarloSimulator(engine, seed=7).simulate(strat)
        r2 = MonteCarloSimulator(engine, seed=7).simulate(strat)
        assert r1.mean == r2.mean
        assert (r1.samples == r2.samples).all()

    def test_distribution_shape(self, engine: StrategyEngine) -> None:
        cfg = MonteCarloConfig(n_sims=1000)
        result = MonteCarloSimulator(engine, cfg, seed=1).simulate(
            Strategy.from_pairs([("SOFT", 10), ("MEDIUM", 10)])
        )
        assert result.n_sims == 1000
        assert result.p05 < result.p50 < result.p95
        # Mean should be close to the deterministic time plus positive event penalties.
        deterministic = engine.race_time(result.strategy)
        assert result.mean >= deterministic - 1.0

    def test_mean_tracks_deterministic_ranking(self, engine: StrategyEngine) -> None:
        # A clearly worse strategy (long SOFT stint) should have a higher mean.
        good = Strategy.from_pairs([("SOFT", 8), ("MEDIUM", 12)])
        bad = Strategy.from_pairs([("SOFT", 16), ("MEDIUM", 4)])
        sim = MonteCarloSimulator(engine, MonteCarloConfig(n_sims=1500), seed=3)
        assert sim.simulate(good).mean < sim.simulate(bad).mean

    def test_compare_win_probabilities_sum_to_one(self, engine: StrategyEngine) -> None:
        strategies = [
            Strategy.from_pairs([("SOFT", 8), ("MEDIUM", 12)]),
            Strategy.from_pairs([("SOFT", 12), ("MEDIUM", 8)]),
            Strategy.from_pairs([("MEDIUM", 20)]),
        ]
        comparison = MonteCarloSimulator(engine, MonteCarloConfig(n_sims=1500), seed=5).compare(
            strategies
        )
        total = sum(r["win_probability"] for r in comparison.rows)
        assert total == pytest.approx(1.0)
        # The best row has the highest win probability.
        assert comparison.best["win_probability"] == max(
            r["win_probability"] for r in comparison.rows
        )

    def test_compare_requires_strategies(self, engine: StrategyEngine) -> None:
        with pytest.raises(ValueError):
            MonteCarloSimulator(engine).compare([])

    def test_safety_car_makes_stops_cheaper(self) -> None:
        # With a high SC probability and big pit saving, a one-stop's mean should be
        # pulled down relative to no SC. Compare two configs on the same strategy.
        engine = StrategyEngine(
            RaceModel(
                total_laps=30,
                base_lap_time=90.0,
                degradation={"MEDIUM": 0.05, "SOFT": 0.1},
                pit_loss=25.0,
            )
        )
        strat = Strategy.from_pairs([("SOFT", 15), ("MEDIUM", 15)])
        no_sc = MonteCarloConfig(n_sims=2000, safety_car_probability=0.0, rain_probability=0.0)
        with_sc = MonteCarloConfig(
            n_sims=2000,
            safety_car_probability=1.0,
            safety_car_lap_penalty_s=0.0,  # isolate the pit-saving effect
            safety_car_pit_saving_s=15.0,
            safety_car_laps=30,  # SC always covers the pit lap
            rain_probability=0.0,
        )
        mean_no_sc = MonteCarloSimulator(engine, no_sc, seed=2).simulate(strat).mean
        mean_with_sc = MonteCarloSimulator(engine, with_sc, seed=2).simulate(strat).mean
        assert mean_with_sc < mean_no_sc


# --- Pit-stop plans (from test_pit_stop.py) ---------------------------------------------------


class TestCandidateGeneration:
    def test_one_stop_count(self) -> None:
        cands = generate_candidate_strategies(20, ["SOFT", "MEDIUM"], min_stint=8)
        # 2 ordered compound pairs x pit laps in [8, 12] (5 options) = 10.
        assert len(cands) == 10
        assert all(c.total_laps == 20 and c.n_stops == 1 for c in cands)

    def test_two_stop_included(self) -> None:
        cands = generate_candidate_strategies(
            30, ["SOFT", "MEDIUM", "HARD"], max_stops=2, min_stint=8, pit_lap_step=2
        )
        assert any(c.n_stops == 2 for c in cands)
        assert all(c.total_laps == 30 for c in cands)

    def test_different_compounds_enforced(self) -> None:
        cands = generate_candidate_strategies(20, ["SOFT", "MEDIUM"], min_stint=8)
        assert all(len(set(c.compounds)) >= 2 for c in cands)


class TestRecommend:
    @pytest.fixture
    def engine(self) -> StrategyEngine:
        return StrategyEngine(
            RaceModel(
                total_laps=20,
                base_lap_time=90.0,
                compound_offsets={"SOFT": -0.6, "MEDIUM": 0.0, "HARD": 0.5},
                degradation={"SOFT": 0.12, "MEDIUM": 0.05, "HARD": 0.03},
                pit_loss=20.0,
            )
        )

    def test_deterministic_best(self, engine: StrategyEngine) -> None:
        rec = recommend_strategy(engine, ["SOFT", "MEDIUM", "HARD"], min_stint=8)
        cands = generate_candidate_strategies(20, ["SOFT", "MEDIUM", "HARD"], min_stint=8)
        best_time = min(engine.race_time(c) for c in cands)
        assert rec.expected_race_time == pytest.approx(best_time)
        assert rec.pit_laps  # at least one stop
        assert rec.confidence == 1.0  # no simulator -> deterministic certainty
        assert rec.candidates_evaluated == len(cands)

    def test_with_simulator_has_confidence(self, engine: StrategyEngine) -> None:
        sim = MonteCarloSimulator(engine, MonteCarloConfig(n_sims=800), seed=4)
        rec = recommend_strategy(engine, ["SOFT", "MEDIUM"], simulator=sim, min_stint=8)
        assert 0.0 < rec.confidence <= 1.0
        assert rec.expected_gain_s >= 0.0

    def test_no_candidates_raises(self, engine: StrategyEngine) -> None:
        # min_stint too large to fit two stints -> no candidates.
        with pytest.raises(ValueError):
            recommend_strategy(engine, ["SOFT", "MEDIUM"], min_stint=15)


# --- Strategy Room planning (from test_strategy_room.py) --------------------------------------


class TestUndercut:
    def test_shift_pit_lap_bounds(self) -> None:
        s = _strategy(pit=22)
        moved = shift_pit_lap(s, 0, 25)
        assert moved is not None and moved.pit_laps == [25]
        assert shift_pit_lap(s, 0, 0) is None
        assert shift_pit_lap(s, 0, 50) is None  # would leave an empty last stint

    def test_window_sweep_finds_cheaper_stop(self) -> None:
        # Soft cliffs at 18: stopping at 22 wastes cliff laps; optimum is earlier.
        engine = StrategyEngine(_model(cliff_lap={"SOFT": 18}))
        windows = pit_window_sweep(engine, _strategy(pit=22))
        assert len(windows) == 1
        assert windows[0].best_lap < 22
        assert windows[0].best_saving_s > 0
        assert windows[0].window_start <= windows[0].best_lap <= windows[0].window_end

    def test_undercut_report_signs(self) -> None:
        engine = StrategyEngine(_model())
        user = _strategy(pit=20)
        rival = _strategy(pit=26)  # stays out longer on old softs
        report = undercut_report(engine, user, rival)
        assert report[0]["pit_lap"] == 20
        # Fresh mediums vs 21+-lap-old softs: the undercut should work.
        assert report[0]["works"] is True


class TestWetCandidates:
    def test_wet_mode_allows_single_compound_and_zero_stop(self) -> None:
        candidates = generate_candidate_strategies(
            50, ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"], condition="wet"
        )
        compound_sets = {tuple(c.compounds) for c in candidates}
        assert ("WET",) in compound_sets  # zero-stop on wets is legal
        assert all(set(c.compounds) <= {"INTERMEDIATE", "WET"} for c in candidates)  # never slicks

    def test_dry_mode_still_requires_two_compounds(self) -> None:
        candidates = generate_candidate_strategies(50, ["SOFT", "MEDIUM"], condition="dry")
        assert all(len(set(c.compounds)) >= 2 for c in candidates)


class TestMonteCarloExtensions:
    def test_forced_sc_lap_fixes_the_scenario(self) -> None:
        engine = StrategyEngine(_model())
        cfg = MonteCarloConfig(n_sims=64, forced_sc_lap=21, rain_probability=0.0)
        sim = MonteCarloSimulator(engine, cfg, seed=1)
        scenarios = sim._draw(sim._make_rng())
        assert scenarios.safety_car.all()
        assert (scenarios.sc_start == 21).all()

    def test_multiplier_penalty_replaces_flat_when_set(self) -> None:
        engine = StrategyEngine(_model())
        flat = MonteCarloSimulator(
            engine, MonteCarloConfig(n_sims=256, rain_probability=0.0), seed=7
        ).simulate(_strategy())
        mult = MonteCarloSimulator(
            engine,
            MonteCarloConfig(n_sims=256, rain_probability=0.0, sc_lap_multiplier=1.6),
            seed=7,
        ).simulate(_strategy())
        # (1.6-1)*90 = 54 s/SC-lap vs flat 25 => multiplier mode is costlier.
        assert mult.mean > flat.mean


class TestReviewFixes:
    """Regression tests for the adversarial-review findings (the planning ones)."""

    def test_mc_sc_penalty_truncated_at_race_end(self) -> None:
        engine = StrategyEngine(_model())
        early = MonteCarloSimulator(
            engine,
            MonteCarloConfig(
                n_sims=64,
                lap_noise_s=0.0,
                rain_probability=0.0,
                forced_sc_lap=10,
                sc_lap_multiplier=1.6,
            ),
            seed=3,
        ).simulate(_strategy())
        at_end = MonteCarloSimulator(
            engine,
            MonteCarloConfig(
                n_sims=64,
                lap_noise_s=0.0,
                rain_probability=0.0,
                forced_sc_lap=50,
                sc_lap_multiplier=1.6,
            ),
            seed=3,
        ).simulate(_strategy())
        # SC on the last lap can only neutralise 1 lap, not 4.
        assert at_end.mean < early.mean

    def test_undercut_horizon_capped_at_rival_pit(self) -> None:
        engine = StrategyEngine(_model())
        user = _strategy(pit=20)
        rival = _strategy(pit=21)  # covers immediately
        report = undercut_report(engine, user, rival)
        assert report[0]["laps_evaluated"] == 1

    def test_mc_pit_saving_never_exceeds_pit_loss(self) -> None:
        # Tiny pit loss + legacy flat saving must not create a negative stop.
        model = _model(pit_loss=8.0)
        engine = StrategyEngine(model)
        cfg = MonteCarloConfig(
            n_sims=64,
            lap_noise_s=0.0,
            rain_probability=0.0,
            safety_car_probability=1.0,
            forced_sc_lap=21,
        )
        sim = MonteCarloSimulator(engine, cfg, seed=5).simulate(_strategy(pit=22))
        no_sc = float(engine.per_lap_times(_strategy(pit=22)).sum())
        # Finishing time includes at least the SC penalty; the stop is >= free.
        assert sim.mean >= no_sc
