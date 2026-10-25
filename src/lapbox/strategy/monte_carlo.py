"""Monte Carlo: many races run with random variation, as a distribution of finishing times.

Wraps the deterministic :class:`~lapbox.strategy.StrategyEngine` with random race variation:

* **Lap-time noise**: per-lap Gaussian variation, summed over the race.
* **Safety cars**: a probabilistic safety-car period that slows the field and makes a pit stop
  taken *within* it cheaper.
* **Weather**: a chance of a wet race that adds a per-lap penalty.

The defaults in :class:`MonteCarloConfig` (0.15 s of lap noise, a 35 % safety-car chance, 4
safety-car laps, a 15 % rain chance and its 3 s per lap) are set by hand, not measured. Pass
your own: LapBox passes each circuit's measured safety-car chance and the laps safety cars start
on (from its race history since 2018, where there is any) and turns random rain off.

Reproducible: all randomness comes from a seeded NumPy generator. Comparing plans uses **common
random numbers** (the same drawn scenarios for every plan), so a win probability reflects the
plans, not the luck of the draws.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from lapbox.errors import InvalidPlanError
from lapbox.strategy.engine import Strategy, StrategyEngine

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MonteCarloConfig:
    """Stochastic parameters for the Monte-Carlo simulation."""

    n_sims: int = 2000
    lap_noise_s: float = 0.15
    safety_car_probability: float = 0.35  # chance of at least one SC per race
    safety_car_laps: int = 4
    safety_car_lap_penalty_s: float = 25.0  # extra time per SC lap (field slows)
    safety_car_pit_saving_s: float = 12.0  # saving if a stop falls under the SC
    rain_probability: float = 0.15
    rain_lap_penalty_s: float = 3.0  # extra per-lap time in a wet race
    # What-if hybrid: force the SC to appear on this lap in EVERY scenario
    # (noise still varies), instead of drawing SC occurrence at random.
    forced_sc_lap: int | None = None
    # When set, SC laps cost (multiplier - 1) x base lap time each — consistent
    # with the deterministic engine — instead of the legacy flat penalty.
    sc_lap_multiplier: float | None = None
    # When set, a stop under the SC costs pit_loss x factor (saving the rest),
    # mirroring the deterministic engine, instead of the legacy flat 12 s.
    sc_pit_cost_factor: float | None = None
    # When set, the lap a safety car starts on is drawn from these weights
    # (index 0 = lap 1) instead of evenly.
    sc_start_weights: tuple[float, ...] | None = None


@dataclass(slots=True)
class _Scenarios:
    """Pre-drawn random scenarios shared across strategies (common random numbers)."""

    noise_sum: np.ndarray
    rain: np.ndarray
    safety_car: np.ndarray
    sc_start: np.ndarray


@dataclass(slots=True)
class SimulationResult:
    """Distribution of finishing times for one strategy."""

    strategy: Strategy
    samples: np.ndarray
    mean: float
    std: float
    p05: float
    p50: float
    p95: float

    @property
    def n_sims(self) -> int:
        return len(self.samples)


@dataclass(slots=True)
class StrategyComparison:
    """Head-to-head Monte-Carlo comparison of several strategies."""

    rows: list[dict] = field(default_factory=list)

    @property
    def best(self) -> dict:
        """The strategy with the highest win probability."""
        return max(self.rows, key=lambda r: r["win_probability"])


class MonteCarloSimulator:
    """Runs seeded Monte-Carlo race simulations over a :class:`StrategyEngine`."""

    def __init__(
        self,
        engine: StrategyEngine,
        config: MonteCarloConfig | None = None,
        *,
        seed: int = 42,
    ) -> None:
        self.engine = engine
        self.config = config or MonteCarloConfig()
        self.seed = seed

    # -- Public API ----------------------------------------------------------

    def simulate(self, strategy: Strategy) -> SimulationResult:
        """Simulate a single strategy and summarise the finishing-time distribution."""
        scenarios = self._draw(self._make_rng())
        samples = self._finishing_times(strategy, scenarios)
        result = SimulationResult(
            strategy=strategy,
            samples=samples,
            mean=float(samples.mean()),
            std=float(samples.std(ddof=0)),
            p05=float(np.percentile(samples, 5)),
            p50=float(np.percentile(samples, 50)),
            p95=float(np.percentile(samples, 95)),
        )
        logger.info(
            "Monte-Carlo simulated",
            extra={"n_sims": result.n_sims, "mean": round(result.mean, 2)},
        )
        return result

    def compare(self, strategies: list[Strategy]) -> StrategyComparison:
        """Compare strategies under identical drawn scenarios (paired comparison).

        Returns win probabilities (fraction of scenarios in which each strategy
        finishes fastest) plus mean finishing time.

        Raises:
            InvalidPlanError: If no strategies are given (a :class:`ValueError`).
        """
        if not strategies:
            raise InvalidPlanError("Provide at least one strategy to compare.")
        scenarios = self._draw(self._make_rng())
        all_samples = np.vstack(
            [self._finishing_times(s, scenarios) for s in strategies]
        )  # shape (n_strategies, n_sims)
        winners = np.argmin(all_samples, axis=0)  # index of fastest per scenario
        counts = np.bincount(winners, minlength=len(strategies))

        rows = [
            {
                "strategy": s,
                "mean_time": float(all_samples[i].mean()),
                "win_probability": float(counts[i] / all_samples.shape[1]),
            }
            for i, s in enumerate(strategies)
        ]
        rows.sort(key=lambda r: r["win_probability"], reverse=True)
        return StrategyComparison(rows=rows)

    # -- Internals -----------------------------------------------------------

    def _make_rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)

    def _draw(self, rng: np.random.Generator) -> _Scenarios:
        cfg = self.config
        n = cfg.n_sims
        total_laps = self.engine.model.total_laps
        # Sum of per-lap Gaussian noise ~ Normal(0, sigma·sqrt(L)).
        noise_sum = rng.normal(0.0, cfg.lap_noise_s * np.sqrt(total_laps), n)
        rain = rng.random(n) < cfg.rain_probability
        if cfg.forced_sc_lap is not None:
            # What-if mode: the SC definitely appears, on the chosen lap.
            safety_car = np.ones(n, dtype=bool)
            sc_start = np.full(n, int(cfg.forced_sc_lap), dtype=np.int64)
        else:
            safety_car = rng.random(n) < cfg.safety_car_probability
            weights = np.asarray(cfg.sc_start_weights or (), dtype=float)[:total_laps]
            if weights.size and weights.sum() > 0:
                laps = np.arange(1, weights.size + 1)
                sc_start = rng.choice(laps, size=n, p=weights / weights.sum())
            else:
                high = max(2, total_laps - cfg.safety_car_laps + 1)
                sc_start = rng.integers(1, high, n)
        return _Scenarios(noise_sum=noise_sum, rain=rain, safety_car=safety_car, sc_start=sc_start)

    def _finishing_times(self, strategy: Strategy, sc: _Scenarios) -> np.ndarray:
        cfg = self.config
        total_laps = self.engine.model.total_laps
        green_base = float(self.engine.per_lap_times(strategy).sum())

        times = np.full(cfg.n_sims, green_base) + sc.noise_sum
        times += sc.rain * (cfg.rain_lap_penalty_s * total_laps)
        if cfg.sc_lap_multiplier is not None:
            sc_penalty = (cfg.sc_lap_multiplier - 1.0) * self.engine.model.base_lap_time
        else:
            sc_penalty = cfg.safety_car_lap_penalty_s
        # An SC drawn near the flag only neutralises the laps that remain.
        sc_laps_eff = np.minimum(cfg.safety_car_laps, np.maximum(0, total_laps - sc.sc_start + 1))
        times += sc.safety_car * (sc_penalty * sc_laps_eff)

        pit_loss = self.engine.model.pit_loss
        if cfg.sc_pit_cost_factor is not None:
            saving = pit_loss * (1.0 - cfg.sc_pit_cost_factor)
        else:
            saving = min(cfg.safety_car_pit_saving_s, pit_loss)  # never a free stop
        pit_total = np.full(cfg.n_sims, pit_loss * strategy.n_stops)
        for pit_lap in strategy.pit_laps:
            under_sc = (
                sc.safety_car
                & (sc.sc_start <= pit_lap)
                & (pit_lap < sc.sc_start + cfg.safety_car_laps)
            )
            pit_total = pit_total - under_sc * saving
        return times + pit_total
