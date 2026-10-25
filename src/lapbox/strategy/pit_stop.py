"""Pit-stop planning: every legal tyre plan for a race, and the best of them.

Enumerates viable tyre strategies, ranks them by the deterministic
:class:`~lapbox.strategy.StrategyEngine`, and (optionally) runs the top candidates through the
Monte-Carlo simulator to attach a **confidence**: the probability the recommendation wins once
race variance is considered.

The candidate generator respects two real constraints: a minimum stint length and the "must use
at least two compounds" dry-race rule. The recommendation is only as good as the race model it
is ranked under, and the engine has not been backtested against real races.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import product

from lapbox.errors import InvalidPlanError
from lapbox.strategy.engine import RAIN_COMPOUNDS, Strategy, StrategyEngine
from lapbox.strategy.monte_carlo import MonteCarloSimulator

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PitStopRecommendation:
    """The recommended strategy plus why it was chosen."""

    strategy: Strategy
    pit_laps: list[int]
    compounds: list[str]
    expected_race_time: float
    expected_gain_s: float  # vs the median candidate (higher = better)
    confidence: float  # Monte-Carlo win probability (0..1)
    candidates_evaluated: int


def generate_candidate_strategies(
    total_laps: int,
    compounds: list[str],
    *,
    max_stops: int = 1,
    min_stint: int = 8,
    pit_lap_step: int = 1,
    condition: str = "dry",
) -> list[Strategy]:
    """Enumerate valid strategies for the race.

    Args:
        total_laps: Race distance.
        compounds: Compounds available to choose stints from.
        max_stops: Maximum number of pit stops (1 or 2 supported).
        min_stint: Minimum laps per stint.
        pit_lap_step: Step for the pit-lap grid on 2-stop strategies (keeps the
            candidate count bounded).
        condition: 'dry' | 'damp' | 'wet'. In damp/wet races the field runs
            rain tyres and the two-dry-compound rule lapses (Sporting Regs),
            so same-compound and zero-stop plans become legal.
    """
    strategies: list[Strategy] = []

    if condition in ("damp", "wet"):
        rain = [c for c in RAIN_COMPOUNDS if c in compounds]
        if not rain:
            rain = ["INTERMEDIATE" if condition == "damp" else "WET"]
        # Zero-stop: nurse one set of rain tyres home (legal once inters/wets run).
        for c in rain:
            strategies.append(Strategy.from_pairs([(c, total_laps)]))
        # One-stop: any rain-compound pairing, same compound allowed.
        for c1, c2 in product(rain, rain):
            for pit in range(min_stint, total_laps - min_stint + 1, max(1, pit_lap_step)):
                strategies.append(Strategy.from_pairs([(c1, pit), (c2, total_laps - pit)]))
        return strategies

    # --- One-stop: two stints, different compounds ---------------------------
    for c1, c2 in product(compounds, compounds):
        if c1 == c2:
            continue
        for pit in range(min_stint, total_laps - min_stint + 1):
            strategies.append(Strategy.from_pairs([(c1, pit), (c2, total_laps - pit)]))

    # --- Two-stop: three stints, at least two distinct compounds -------------
    if max_stops >= 2:
        for c1, c2, c3 in product(compounds, repeat=3):
            if len({c1, c2, c3}) < 2:
                continue
            for p1 in range(min_stint, total_laps - 2 * min_stint + 1, pit_lap_step):
                for p2 in range(p1 + min_stint, total_laps - min_stint + 1, pit_lap_step):
                    strategies.append(
                        Strategy.from_pairs([(c1, p1), (c2, p2 - p1), (c3, total_laps - p2)])
                    )
    return strategies


def recommend_strategy(
    engine: StrategyEngine,
    compounds: list[str],
    *,
    simulator: MonteCarloSimulator | None = None,
    max_stops: int = 1,
    min_stint: int = 8,
    top_k: int = 5,
    pit_lap_step: int = 1,
    condition: str = "dry",
) -> PitStopRecommendation:
    """Recommend the best tyre strategy for the race.

    Ranks all candidates by deterministic race time; if a ``simulator`` is given,
    the top ``top_k`` candidates are compared under identical random scenarios and
    the winner (by win probability) is recommended with that probability as its
    confidence.

    Raises:
        InvalidPlanError: If no valid candidate strategies exist (e.g. ``min_stint``
            too large for the race, or fewer than two compounds); a :class:`ValueError`.
    """
    candidates = generate_candidate_strategies(
        engine.model.total_laps,
        compounds,
        max_stops=max_stops,
        min_stint=min_stint,
        pit_lap_step=pit_lap_step,
        condition=condition,
    )
    if not candidates:
        raise InvalidPlanError("No valid candidate strategies for the given constraints.")

    # One deterministic simulation per candidate. Sorting by ``engine.race_time``
    # and then rebuilding the list of times simulated every plan twice, which
    # on a 100-lap, 2-stop search is ~100k extra full-race replays.
    scored = sorted(((engine.race_time(s), s) for s in candidates), key=lambda pair: pair[0])
    ranked = [s for _, s in scored]
    times = [t for t, _ in scored]
    median_time = times[len(times) // 2]

    best = ranked[0]
    confidence = 1.0
    if simulator is not None and len(ranked) > 1:
        comparison = simulator.compare(ranked[:top_k])
        winner = comparison.best
        best = winner["strategy"]
        confidence = winner["win_probability"]

    recommendation = PitStopRecommendation(
        strategy=best,
        pit_laps=best.pit_laps,
        compounds=best.compounds,
        expected_race_time=engine.race_time(best),
        expected_gain_s=float(median_time - engine.race_time(best)),
        confidence=confidence,
        candidates_evaluated=len(candidates),
    )
    logger.info(
        "Strategy recommended",
        extra={
            "pit_laps": recommendation.pit_laps,
            "compounds": recommendation.compounds,
            "confidence": round(recommendation.confidence, 3),
            "candidates": recommendation.candidates_evaluated,
        },
    )
    return recommendation
