"""The pit window and the undercut, as a strategist asks about them.

* **Where is my pit window?** For each stop in a plan, sweep the pit lap a few
  laps either side and measure the total-race-time cost — the *window* is the
  band of laps within a tolerance of the optimum.
* **Does the undercut work?** Against a rival running a reference strategy,
  compute the classic undercut ledger: fresh-tyre pace advantage per lap
  (minus a cold-tyre out-lap penalty) accumulated over the laps before the
  rival covers the stop, versus the live gap.

Everything derives from the deterministic engine's per-lap arrays, so a full
sweep costs microseconds. The 1 s out-lap penalty and the 1 s window tolerance are set by
hand, not measured.
"""

from __future__ import annotations

from dataclasses import dataclass

from lapbox.strategy.engine import Strategy, StrategyEngine

# Cold fresh tyres give some of the advantage back on the out-lap.
OUT_LAP_PENALTY_S = 1.0
# A pit lap within this cost of the optimum counts as "inside the window".
WINDOW_TOLERANCE_S = 1.0


@dataclass(slots=True)
class PitWindow:
    """Cost curve for moving one stop earlier/later, plus the optimal band."""

    stop_index: int  # 0-based stop number
    planned_lap: int
    best_lap: int
    best_saving_s: float  # race time saved by moving to best_lap (>= 0)
    window_start: int
    window_end: int
    # Parallel arrays: candidate pit lap -> race-time delta vs the plan.
    laps: list[int]
    cost_delta_s: list[float]


def shift_pit_lap(strategy: Strategy, stop_index: int, new_pit_lap: int) -> Strategy | None:
    """Return the strategy with one stop moved to ``new_pit_lap`` (same compounds).

    Returns None when the shift would create a stint shorter than one lap or
    reorder stops.
    """
    pit_laps = strategy.pit_laps
    if stop_index >= len(pit_laps):
        return None
    bounds_low = pit_laps[stop_index - 1] + 1 if stop_index > 0 else 1
    bounds_high = (
        pit_laps[stop_index + 1] - 1 if stop_index + 1 < len(pit_laps) else strategy.total_laps - 1
    )
    if not bounds_low <= new_pit_lap <= bounds_high:
        return None

    new_pits = list(pit_laps)
    new_pits[stop_index] = new_pit_lap
    boundaries = [0, *new_pits, strategy.total_laps]
    pairs = []
    for i, stint in enumerate(strategy.stints):
        laps = boundaries[i + 1] - boundaries[i]
        if laps <= 0:
            return None
        pairs.append((stint.compound, laps))
    return Strategy.from_pairs(pairs)


def pit_window_sweep(
    engine: StrategyEngine,
    strategy: Strategy,
    *,
    span: int = 8,
    min_stint: int = 1,
) -> list[PitWindow]:
    """Sweep each stop ±``span`` laps and report the cost curve + optimal window."""
    base_time = engine.race_time(strategy)
    windows: list[PitWindow] = []

    for stop_index, planned in enumerate(strategy.pit_laps):
        laps: list[int] = []
        deltas: list[float] = []
        for candidate in range(planned - span, planned + span + 1):
            shifted = shift_pit_lap(strategy, stop_index, candidate)
            if shifted is None or any(s.laps < min_stint for s in shifted.stints):
                continue
            laps.append(candidate)
            deltas.append(engine.race_time(shifted) - base_time)
        if not laps:
            continue

        best_i = min(range(len(laps)), key=lambda i: deltas[i])
        in_window = [
            laps[i] for i in range(len(laps)) if deltas[i] <= deltas[best_i] + WINDOW_TOLERANCE_S
        ]
        windows.append(
            PitWindow(
                stop_index=stop_index,
                planned_lap=planned,
                best_lap=laps[best_i],
                best_saving_s=max(0.0, -deltas[best_i]),
                window_start=min(in_window),
                window_end=max(in_window),
                laps=laps,
                cost_delta_s=[round(d, 3) for d in deltas],
            )
        )
    return windows


def undercut_report(
    engine: StrategyEngine,
    user: Strategy,
    rival: Strategy,
    *,
    cover_laps: int = 3,
) -> list[dict]:
    """The undercut ledger for each of the user's stops against a rival.

    For a stop on lap L: the rival is assumed to cover ``cover_laps`` later.
    Advantage per intervening lap = (rival's old-tyre pace) − (user's fresh-tyre
    pace + out-lap penalty on the first). The undercut *works* when the summed
    advantage exceeds zero net of nothing else — the caller compares against a
    live gap; here we report the raw seconds gained.
    """
    user_trace = engine.per_lap_times(user)
    rival_trace = engine.per_lap_times(rival)
    total = engine.model.total_laps
    rival_pits = rival.pit_laps

    reports: list[dict] = []
    for stop_index, pit_lap in enumerate(user.pit_laps):
        # The ledger only makes sense while the rival is still out on the old
        # tyre: stop accumulating once they actually cover (their next stop).
        next_cover = min((p for p in rival_pits if p > pit_lap), default=None)
        horizon = min(
            cover_laps,
            total - pit_lap,
            (next_cover - pit_lap) if next_cover is not None else cover_laps,
        )
        gained = 0.0
        for k in range(1, horizon + 1):
            adv = float(rival_trace[pit_lap + k - 1] - user_trace[pit_lap + k - 1])
            if k == 1:
                adv -= OUT_LAP_PENALTY_S
            gained += adv
        reports.append(
            {
                "stop_index": stop_index,
                "pit_lap": pit_lap,
                "laps_evaluated": max(0, horizon),
                "advantage_s": round(gained, 3),
                "works": gained > 0,
            }
        )
    return reports
