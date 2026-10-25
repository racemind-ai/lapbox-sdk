"""Unit tests for lapbox.errors: the refusals, and that old ``except ValueError`` code still catches them."""

from __future__ import annotations

import pytest

from lapbox.errors import InvalidPlanError, LapboxError, NotEnoughDataError
from lapbox.strategy import MonteCarloSimulator, RaceModel, StrategyEngine, recommend_strategy

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("error", [NotEnoughDataError, InvalidPlanError])
def test_each_refusal_is_a_lapbox_error_and_a_value_error(error: type[Exception]) -> None:
    assert issubclass(error, LapboxError)
    assert issubclass(error, ValueError)


def test_the_two_kinds_are_told_apart() -> None:
    assert not issubclass(NotEnoughDataError, InvalidPlanError)
    assert not issubclass(InvalidPlanError, NotEnoughDataError)


@pytest.fixture
def engine() -> StrategyEngine:
    return StrategyEngine(
        RaceModel(total_laps=20, base_lap_time=90.0, degradation={"SOFT": 0.1, "MEDIUM": 0.05})
    )


def test_constraints_no_plan_meets_are_an_invalid_plan(engine: StrategyEngine) -> None:
    with pytest.raises(InvalidPlanError, match="No valid candidate strategies"):
        recommend_strategy(engine, ["SOFT", "MEDIUM"], min_stint=15)


def test_comparing_no_plans_is_an_invalid_plan(engine: StrategyEngine) -> None:
    with pytest.raises(InvalidPlanError, match="at least one strategy"):
        MonteCarloSimulator(engine).compare([])
