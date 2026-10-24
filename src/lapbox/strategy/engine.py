"""The race-strategy engine: race time for a tyre plan, lap by lap.

Each green-flag lap costs::

    lap_time = (base_lap_time + compound_offset[compound]) × condition multiplier
             + wrong-compound penalty
             + degradation[compound] × temperature factor × tyre_age
             + fuel_effect_per_lap × laps still to run
             + cliff penalty past the compound's cliff lap

Behind a safety car or VSC every car laps at a fixed multiple of base pace, and a stop costs
less. Pit losses are added on the in-lap. Everything is deterministic, and ``lap_cost`` is
memoised, so planners and the Monte Carlo layer can call it millions of times.

The parameters are yours to give (:class:`RaceModel`); fitting them to a session comes later.
The engine has **not** been backtested against real race times: in LapBox, the gap it predicted
between teammates on different strategies was further from the real gap than predicting no gap
at all. Read its race times as the consequence of the model you gave it, not as a forecast.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Fallbacks used when a compound is absent from the model's dicts.
_DEFAULT_COMPOUND_OFFSET = 0.0
_DEFAULT_DEGRADATION = 0.06

RAIN_COMPOUNDS = ("INTERMEDIATE", "WET")

# Track-condition pace multiplier on the (base + compound offset) term.
# Anchored to Pirelli crossover guidance: inters become faster than slicks at
# ~110-112% of dry pace, full wets beyond ~118%.
CONDITION_MULTIPLIER = {"dry": 1.00, "damp": 1.12, "wet": 1.18}

# Flat per-lap penalty for running the wrong tyre for the conditions
# (e.g. slicks on a damp track). Compounds not listed are slicks.
_WRONG_COMPOUND_PENALTY = {
    "dry": {"INTERMEDIATE": 5.5, "WET": 9.0},  # rain tyres overheat on a dry track
    "damp": {"_SLICK": 5.5, "WET": 1.0},
    "wet": {"_SLICK": 8.0, "INTERMEDIATE": 2.0},
}


def condition_pace_multiplier(condition: str) -> float:
    """Lap-pace multiplier for a track condition (unknown → dry)."""
    return CONDITION_MULTIPLIER.get(condition, 1.0)


def wrong_compound_penalty(condition: str, compound: str) -> float:
    """Flat s/lap penalty for a compound unsuited to the track condition."""
    table = _WRONG_COMPOUND_PENALTY.get(condition, {})
    key = compound if compound in RAIN_COMPOUNDS else "_SLICK"
    return table.get(key, 0.0)


def temp_degradation_factor(track_temp_c: float, condition: str = "dry") -> float:
    """Degradation scaling with track temperature (~+3%/°C above 30 °C).

    Damp/wet running cools the surface, easing thermal degradation further.
    """
    factor = max(0.7, min(2.0, 1.0 + 0.03 * (track_temp_c - 30.0)))
    if condition in ("damp", "wet"):
        factor *= 0.75
    return factor


@dataclass(frozen=True, slots=True)
class Stint:
    """A single tyre stint: a compound run for a number of laps."""

    compound: str
    laps: int

    def __post_init__(self) -> None:
        if self.laps <= 0:
            raise ValueError("A stint must be at least one lap.")


@dataclass(frozen=True, slots=True)
class Strategy:
    """An ordered sequence of stints describing a full race plan."""

    stints: tuple[Stint, ...]

    def __post_init__(self) -> None:
        if not self.stints:
            raise ValueError("A strategy needs at least one stint.")

    @classmethod
    def from_pairs(cls, pairs: list[tuple[str, int]]) -> Strategy:
        """Build a strategy from ``[(compound, laps), ...]`` pairs."""
        return cls(tuple(Stint(compound=c, laps=n) for c, n in pairs))

    @property
    def total_laps(self) -> int:
        return sum(s.laps for s in self.stints)

    @property
    def n_stops(self) -> int:
        return len(self.stints) - 1

    @property
    def compounds(self) -> list[str]:
        return [s.compound for s in self.stints]

    @property
    def pit_laps(self) -> list[int]:
        """Global lap numbers on which a pit stop occurs (end of each stint)."""
        laps: list[int] = []
        cumulative = 0
        for stint in self.stints[:-1]:
            cumulative += stint.laps
            laps.append(cumulative)
        return laps


@dataclass(frozen=True, slots=True)
class RaceModel:
    """Parameters describing race pace, tyre wear, fuel and pit loss.

    Args:
        total_laps: Race distance in laps.
        base_lap_time: Reference fresh-tyre lap time (s) for the base compound.
        compound_offsets: Per-compound pace offset (s); negative = faster.
        degradation: Per-compound tyre-wear rate (s per lap of tyre age).
        pit_loss: Time lost per pit stop (pit lane + stationary), seconds.
        fuel_effect_per_lap: Extra seconds per lap of fuel still on board.
    """

    total_laps: int
    base_lap_time: float
    compound_offsets: dict[str, float] = field(default_factory=dict)
    degradation: dict[str, float] = field(default_factory=dict)
    pit_loss: float = 22.0
    fuel_effect_per_lap: float = 0.06
    # --- Strategy Room extensions (defaults reproduce legacy behaviour) -----
    # Tyre age (laps) past which a compound falls off a cliff; absent = never.
    cliff_lap: dict[str, int] = field(default_factory=dict)
    cliff_slope_s: float = 0.5  # EXTRA s/lap of deg per lap beyond the cliff
    condition: str = "dry"  # 'dry' | 'damp' | 'wet'
    track_temp_c: float = 30.0
    # Safety-car windows: (start_lap, end_lap) inclusive, kind 'SC' or 'VSC'.
    sc_windows: tuple[tuple[int, int, str], ...] = ()
    sc_lap_multiplier: float = 1.6  # cars lap ~60% slower behind the SC
    vsc_lap_multiplier: float = 1.4
    sc_pit_loss_factor: float = 0.5  # pitting under SC costs ~half
    vsc_pit_loss_factor: float = 0.65
    # Track condition lap by lap (index 0 = lap 1) when it changes during the
    # race, e.g. rain arriving; empty = ``condition`` on every lap.
    condition_by_lap: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.total_laps <= 0:
            raise ValueError("total_laps must be positive.")
        if self.base_lap_time <= 0:
            raise ValueError("base_lap_time must be positive.")
        if self.condition_by_lap and len(self.condition_by_lap) != self.total_laps:
            raise ValueError("condition_by_lap must give one condition per lap.")

    def compound_offset(self, compound: str) -> float:
        return self.compound_offsets.get(compound, _DEFAULT_COMPOUND_OFFSET)

    def degradation_rate(self, compound: str) -> float:
        return self.degradation.get(compound, _DEFAULT_DEGRADATION)

    def condition_at(self, lap: int) -> str:
        """The track condition on a lap (1-based)."""
        if self.condition_by_lap and 1 <= lap <= len(self.condition_by_lap):
            return self.condition_by_lap[lap - 1]
        return self.condition

    def sc_state(self, lap: int) -> str | None:
        """'SC' / 'VSC' if the lap falls inside a neutralisation window."""
        for start, end, kind in self.sc_windows:
            if start <= lap <= end:
                return kind
        return None

    def pit_loss_for_lap(self, lap: int) -> float:
        """Pit-lane loss for a stop on this lap (cheaper under SC/VSC)."""
        state = self.sc_state(lap)
        if state == "SC":
            return self.pit_loss * self.sc_pit_loss_factor
        if state == "VSC":
            return self.pit_loss * self.vsc_pit_loss_factor
        return self.pit_loss


@dataclass(slots=True)
class StrategyResult:
    """Deterministic outcome of simulating one strategy."""

    strategy: Strategy
    total_time: float
    green_lap_time: float  # race time excluding pit losses
    pit_loss_total: float
    n_stops: int
    per_lap_times: np.ndarray

    @property
    def average_lap_time(self) -> float:
        return float(self.green_lap_time / len(self.per_lap_times))


class StrategyEngine:
    """Computes deterministic race times for tyre strategies under a race model."""

    def __init__(self, model: RaceModel) -> None:
        self.model = model
        # Memo for lap_cost. The best-response search prices the same
        # (compound, lap, tyre age) millions of times across candidate plans
        # that share lap states — on a 52-lap race it dominated the solve at
        # ~4M calls. RaceModel is frozen, so a value computed for one is valid
        # for its whole lifetime; the identity check below rebuilds if the
        # engine is ever pointed at a different model.
        self._cost_cache: dict[tuple[str, int, int], float] = {}
        self._cost_cache_model = model

    def per_lap_times(
        self,
        strategy: Strategy,
        *,
        start_lap: int = 1,
        start_tyre_age: int = 0,
    ) -> np.ndarray:
        """Return the deterministic per-lap time array (excludes pit losses).

        Args:
            strategy: The stint plan to simulate.
            start_lap: Global lap number the plan begins on. Anything above 1
                makes this a mid-race *continuation*: fuel load and safety-car
                windows are read at the true race lap rather than from a
                standing start, so a plan evaluated from lap 30 is not treated
                as if the car were still full of fuel.
            start_tyre_age: Laps already on the first stint's tyres. Lets a
                continuation begin on a part-worn set instead of pretending
                every plan starts on fresh rubber.

        Raises:
            ValueError: If the strategy doesn't cover the laps that remain.
        """
        remaining = self.model.total_laps - (start_lap - 1)
        if strategy.total_laps != remaining:
            raise ValueError(
                f"Strategy covers {strategy.total_laps} laps but "
                f"{remaining} remain from lap {start_lap}."
            )
        times = np.empty(remaining, dtype=float)
        lap_index = 0
        for stint_no, stint in enumerate(strategy.stints):
            # Only the FIRST stint of a continuation inherits worn tyres; every
            # later stint in the plan starts on a fresh set as normal.
            age_offset = start_tyre_age if stint_no == 0 else 0
            for stint_lap in range(1, stint.laps + 1):
                times[lap_index] = self.lap_cost(
                    stint.compound, start_lap + lap_index, stint_lap + age_offset
                )
                lap_index += 1
        return times

    def lap_cost(self, compound: str, lap: int, tyre_age: int) -> float:
        """Green-running time for ONE lap in a given tyre/fuel/flag state.

        The same formula :meth:`per_lap_times` applies, exposed for callers that
        need to price a single lap (regret accounting) without simulating a
        whole plan. Memoised: it is a pure function of its arguments for a given
        (frozen) model, and the duel's best-response search calls it millions of
        times with a few thousand distinct argument triples.
        """
        model = self.model
        if model is not self._cost_cache_model:  # pragma: no cover - defensive
            self._cost_cache = {}
            self._cost_cache_model = model
        key = (compound, lap, tyre_age)
        hit = self._cost_cache.get(key)
        if hit is not None:
            return hit
        value = self._lap_cost_uncached(compound, lap, tyre_age)
        self._cost_cache[key] = value
        return value

    def _lap_cost_uncached(self, compound: str, lap: int, tyre_age: int) -> float:
        model = self.model
        condition = model.condition_at(lap)
        pace_mult = condition_pace_multiplier(condition)
        temp_factor = temp_degradation_factor(model.track_temp_c, condition)

        offset = model.compound_offset(compound)
        if condition in ("damp", "wet") and compound in RAIN_COMPOUNDS:
            # Rain tyres are in their element: their large DRY-pace deficit
            # doesn't apply, and the wrong-compound table already ranks
            # inter vs wet per condition. Without this, the condition
            # multiplier amplifies the dry offsets and slicks would
            # (absurdly) simulate faster than rain tyres in the rain.
            offset = 0.0
        deg = model.degradation_rate(compound) * temp_factor
        penalty = wrong_compound_penalty(condition, compound)
        cliff = model.cliff_lap.get(compound)

        state = model.sc_state(lap)
        if state in ("SC", "VSC"):
            # Behind the (virtual) safety car everyone laps at the
            # delta-limited pace — tyre state and fuel stop mattering,
            # so the multiplier applies to the base pace, not to the
            # car's own degraded lap time.
            mult = model.sc_lap_multiplier if state == "SC" else model.vsc_lap_multiplier
            return model.base_lap_time * pace_mult * mult

        lap_time = (
            (model.base_lap_time + offset) * pace_mult
            + penalty
            + deg * tyre_age
            + model.fuel_effect_per_lap * (model.total_laps - lap)
        )
        if cliff is not None and tyre_age > cliff:
            lap_time += model.cliff_slope_s * (tyre_age - cliff)
        return lap_time

    def remaining_time(
        self, strategy: Strategy, *, start_lap: int, start_tyre_age: int = 0
    ) -> float:
        """Time to complete a continuation plan from mid-race.

        Green-running plus the pit losses incurred *within* the continuation,
        with pit laps translated to their true race lap so a stop lands in the
        right safety-car window.
        """
        per_lap = self.per_lap_times(strategy, start_lap=start_lap, start_tyre_age=start_tyre_age)
        green = float(per_lap.sum())
        pit_total = float(
            sum(self.model.pit_loss_for_lap(start_lap - 1 + lap) for lap in strategy.pit_laps)
        )
        return green + pit_total

    def simulate(self, strategy: Strategy) -> StrategyResult:
        """Return a full :class:`StrategyResult` for a strategy."""
        per_lap = self.per_lap_times(strategy)
        green = float(per_lap.sum())
        pit_total = float(sum(self.model.pit_loss_for_lap(lap) for lap in strategy.pit_laps))
        return StrategyResult(
            strategy=strategy,
            total_time=green + pit_total,
            green_lap_time=green,
            pit_loss_total=pit_total,
            n_stops=strategy.n_stops,
            per_lap_times=per_lap,
        )

    def lap_trace(self, strategy: Strategy) -> dict:
        """Full per-lap telemetry of a simulated strategy, chart-ready.

        Returns parallel lists (length ``total_laps``): ``lap`` (1-based),
        ``lap_time`` (green running), ``cum_time`` (including pit losses, added
        on the in-lap), ``compound``, ``tyre_age``, ``fuel_laps_left``,
        ``sc_state`` (``None``/``'SC'``/``'VSC'``) and ``in_cliff``.
        """
        model = self.model
        per_lap = self.per_lap_times(strategy)
        pit_laps = set(strategy.pit_laps)

        laps: list[int] = []
        compounds: list[str] = []
        ages: list[int] = []
        fuel: list[int] = []
        sc_states: list[str | None] = []
        in_cliff: list[bool] = []

        lap_number = 0
        for stint in strategy.stints:
            cliff = model.cliff_lap.get(stint.compound)
            for tyre_age in range(1, stint.laps + 1):
                lap_number += 1
                laps.append(lap_number)
                compounds.append(stint.compound)
                ages.append(tyre_age)
                fuel.append(model.total_laps - lap_number)
                sc_states.append(model.sc_state(lap_number))
                in_cliff.append(cliff is not None and tyre_age > cliff)

        cum = []
        running = 0.0
        for i, lap in enumerate(laps):
            running += float(per_lap[i])
            if lap in pit_laps:
                running += model.pit_loss_for_lap(lap)
            cum.append(running)

        return {
            "lap": laps,
            "lap_time": [float(v) for v in per_lap],
            "cum_time": cum,
            "compound": compounds,
            "tyre_age": ages,
            "fuel_laps_left": fuel,
            "sc_state": sc_states,
            "in_cliff": in_cliff,
        }

    def race_time(self, strategy: Strategy) -> float:
        """Return just the total race time (green running + pit losses)."""
        return self.simulate(strategy).total_time
