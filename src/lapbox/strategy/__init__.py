"""Strategy: the race-strategy engine, race time for a tyre plan lap by lap.

Plans are :class:`Strategy` objects (stints of a compound for a number of laps); the race is a
:class:`RaceModel` (pace, compound offsets, degradation, fuel, pit loss, safety-car windows,
track condition lap by lap); :class:`StrategyEngine` prices a plan under a model. Fitting a
model to a real session comes later; the engine has not been backtested against real races.
"""

from lapbox.strategy.engine import (
    CONDITION_MULTIPLIER,
    RAIN_COMPOUNDS,
    RaceModel,
    Stint,
    Strategy,
    StrategyEngine,
    StrategyResult,
    condition_pace_multiplier,
    temp_degradation_factor,
    wrong_compound_penalty,
)

__all__ = [
    # Plans
    "Stint",
    "Strategy",
    # The race and the engine
    "RaceModel",
    "StrategyEngine",
    "StrategyResult",
    # Track condition and temperature
    "RAIN_COMPOUNDS",
    "CONDITION_MULTIPLIER",
    "condition_pace_multiplier",
    "wrong_compound_penalty",
    "temp_degradation_factor",
]
