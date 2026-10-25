"""Strategy: race time for a tyre plan, lap by lap, and the planning built on it.

Plans are :class:`Strategy` objects (stints of a compound for a number of laps); the race is a
:class:`RaceModel` (pace, compound offsets, degradation, fuel, pit loss, safety-car windows,
track condition lap by lap); :class:`StrategyEngine` prices a plan under a model. On top of it:
:class:`MonteCarloSimulator` (a distribution of finishing times, plans compared on the same
random draws), :func:`recommend_strategy` (every legal plan, best first) and
:func:`pit_window_sweep` / :func:`undercut_report`. Fitting a model to a real session comes
later; the engine has not been backtested against real races.
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
from lapbox.strategy.monte_carlo import (
    MonteCarloConfig,
    MonteCarloSimulator,
    SimulationResult,
    StrategyComparison,
)
from lapbox.strategy.pit_stop import (
    PitStopRecommendation,
    generate_candidate_strategies,
    recommend_strategy,
)
from lapbox.strategy.undercut import (
    OUT_LAP_PENALTY_S,
    WINDOW_TOLERANCE_S,
    PitWindow,
    pit_window_sweep,
    shift_pit_lap,
    undercut_report,
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
    # Monte Carlo
    "MonteCarloConfig",
    "MonteCarloSimulator",
    "SimulationResult",
    "StrategyComparison",
    # Planning
    "PitStopRecommendation",
    "generate_candidate_strategies",
    "recommend_strategy",
    # Pit window and undercut
    "PitWindow",
    "pit_window_sweep",
    "shift_pit_lap",
    "undercut_report",
    "OUT_LAP_PENALTY_S",
    "WINDOW_TOLERANCE_S",
]
