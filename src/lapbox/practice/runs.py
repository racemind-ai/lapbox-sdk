"""Separating practice running into long runs and qualifying simulations.

A practice session is not one thing. In ninety minutes a team will do a low-fuel
qualifying simulation, one or two high-fuel long runs, aero rakes, pit-stop
practice and a lot of laps that are neither. Averaging all of it produces a
number that describes nothing.

The paddock reads practice by splitting it into **runs** — stretches of
consecutive flying laps between visits to the pit lane — and then treating two
kinds very differently:

* a **qualifying simulation** is one or two laps on fresh soft tyres and low
  fuel, and predicts Saturday,
* a **long run** is a stretch of five or more consecutive laps on a heavier
  fuel load, and predicts Sunday.

This module does that split. It is pure and column-tolerant: it takes FastF1's
laps as they come (``session.laps``), laps from :func:`lapbox.data.normalize_laps`,
or snake_case laps (``driver``, ``lap_number``, ``lap_time_seconds``, ...).

**What cannot be known:** fuel loads are never published. Within a single run
the car burns fuel at a roughly constant rate, so the *slope* of a long run can
be corrected for fuel burn to isolate tyre degradation — that number is sound.
The *absolute* pace of one team's long run against another's is not, because
the two may have been carrying very different fuel. Callers are expected to
label it as indicative, which is exactly what a TV analyst does.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# A run needs at least this many consecutive timed laps to say anything about
# race pace. Below it, a "long run" is just a couple of laps with a gap.
MIN_LONG_RUN = 5
# A qualifying simulation is this short...
MAX_QUALI_SIM = 3
# ...and within this fraction of the driver's best lap of the session.
QUALI_SIM_CUTOFF = 1.02

# Fuel burn is ~1.6 kg/lap in modern F1, and ~0.03 s/lap per kg of fuel, so a
# car gets roughly this much faster each lap purely from burning fuel off.
FUEL_EFFECT_S_PER_LAP = 0.048

# Within a run a driver may cruise a lap, catch traffic or practise a start
# without pitting. Those laps are timed and consecutive, so run detection keeps
# them, but they are not race pace: left in, they produce degradation slopes of
# several seconds per lap, which is physically impossible (real degradation is
# ~0.05-0.3 s/lap). Laps slower than this multiple of the run's BEST lap are
# dropped before pace and degradation are measured — the same 107 % convention
# the race-analysis consistency panel uses.
#
# Anchoring to the best lap rather than the median matters: when half a run is
# compromised the median is contaminated too, and a median-relative cutoff
# keeps the very laps it is meant to remove. 107 % of the best lap still leaves
# ample room for genuine degradation across a long run.
RUN_OUTLIER_CUTOFF = 1.07

_LAP_COL = "lap_number"
_TIME_COL = "lap_time_seconds"


@dataclass(slots=True)
class Run:
    """One stretch of consecutive timed laps by a single driver."""

    driver: str
    compound: str | None
    start_lap: int
    end_lap: int
    lap_times: list[float] = field(default_factory=list)
    stint: float | None = None

    @property
    def length(self) -> int:
        return len(self.lap_times)

    @property
    def best(self) -> float:
        return min(self.lap_times)

    @property
    def median(self) -> float:
        return float(np.median(self.lap_times))


@dataclass(slots=True)
class LongRunPace:
    """A driver's race-pace read from one long run."""

    driver: str
    compound: str | None
    laps: int
    median_s: float
    """Median lap time — the headline pace. Fuel load unknown, so indicative."""

    raw_slope_s_per_lap: float
    """Observed trend across the run: tyre degradation MINUS fuel burn."""

    deg_slope_s_per_lap: float
    """Degradation with the fuel-burn effect added back — the tyre's own trend."""

    consistency_s: float
    """Standard deviation of the run's representative lap times."""

    laps_dropped: int = 0
    """Laps excluded as unrepresentative (cruising, traffic, practice starts)."""


def _column(laps: pd.DataFrame, *candidates: str) -> str | None:
    """First present column name out of ``candidates`` (schema-tolerant)."""
    for name in candidates:
        if name in laps.columns:
            return name
    return None


def detect_runs(laps: pd.DataFrame) -> list[Run]:
    """Split a session into runs of consecutive timed laps.

    A run breaks whenever the lap sequence is interrupted — a missing lap time
    (an in-lap, out-lap or aborted attempt), a pit stop, or a compound change.
    That is what separates one visit to the track from the next.

    Args:
        laps: Session laps with driver, lap-number, lap-time and (optionally)
            compound and stint columns: FastF1's own laps (``Driver``,
            ``LapNumber``, ``LapTime``), the same with ``LapTimeSeconds``, or
            snake_case engineered laps.

    Returns:
        Every run found, in driver then lap order. Runs of length 0 are omitted.
    """
    driver_col = _column(laps, "driver", "Driver")
    lap_col = _column(laps, _LAP_COL, "LapNumber")
    time_col = _column(laps, _TIME_COL, "LapTimeSeconds")
    if time_col is None and "LapTime" in laps.columns:
        # FastF1's own laps carry the time as a Timedelta only.
        laps = laps.assign(LapTimeSeconds=pd.to_timedelta(laps["LapTime"]).dt.total_seconds())
        time_col = "LapTimeSeconds"
    if not (driver_col and lap_col and time_col):
        logger.warning("Cannot detect runs — missing driver/lap/time columns")
        return []

    compound_col = _column(laps, "compound", "Compound")
    stint_col = _column(laps, "stint", "Stint")

    runs: list[Run] = []
    for driver, group in laps.groupby(driver_col, sort=True):
        group = group.sort_values(lap_col)
        current: Run | None = None
        previous_lap: int | None = None

        for _, row in group.iterrows():
            time = row[time_col]
            lap_no = row[lap_col]
            if pd.isna(time) or pd.isna(lap_no):
                current = None  # untimed lap ends the run
                previous_lap = None
                continue

            lap_no = int(lap_no)
            compound = (
                str(row[compound_col]) if compound_col and pd.notna(row[compound_col]) else None
            )
            stint = row[stint_col] if stint_col and pd.notna(row[stint_col]) else None

            broken = (
                current is None
                or previous_lap is None
                or lap_no != previous_lap + 1  # a gap means the car pitted
                or compound != current.compound
                or (stint is not None and current.stint is not None and stint != current.stint)
            )
            if broken:
                current = Run(
                    driver=str(driver),
                    compound=compound,
                    start_lap=lap_no,
                    end_lap=lap_no,
                    lap_times=[],
                    stint=float(stint) if stint is not None else None,
                )
                runs.append(current)

            current.lap_times.append(float(time))
            current.end_lap = lap_no
            previous_lap = lap_no

    return [r for r in runs if r.length > 0]


def classify_run(run: Run, session_best: float | None = None) -> str:
    """Label a run ``long_run``, ``quali_sim`` or ``other``.

    A long run needs :data:`MIN_LONG_RUN` laps *at race pace* — within
    :data:`RUN_OUTLIER_CUTOFF` of the run's best, the laps :func:`long_run_pace`
    measures. Counting every timed lap is not enough: a qualifying run with its
    out-lap and cool-down laps is five consecutive timed laps, but only one or
    two of them are pace, and they are qualifying pace.

    A qualifying simulation is short AND fast — brevity alone is not enough,
    since an aborted long run is also short. ``session_best`` is the driver's
    fastest lap of the session; without it, short runs cannot be confirmed as
    representative and fall through to ``other``.
    """
    at_pace = int(representative_mask(np.asarray(run.lap_times, dtype=float)).sum())
    if at_pace >= MIN_LONG_RUN:
        return "long_run"
    if run.length <= MAX_QUALI_SIM and session_best and run.best <= session_best * QUALI_SIM_CUTOFF:
        return "quali_sim"
    return "other"


def representative_mask(times: np.ndarray, cutoff: float = RUN_OUTLIER_CUTOFF) -> np.ndarray:
    """Which laps of a run are genuine running rather than cruising or traffic."""
    if times.size == 0:
        return np.zeros(0, dtype=bool)
    return times <= float(np.min(times)) * cutoff


def long_run_pace(run: Run, *, cutoff: float = RUN_OUTLIER_CUTOFF) -> LongRunPace:
    """Pace, degradation and consistency for one long run.

    Unrepresentative laps are dropped first (see :data:`RUN_OUTLIER_CUTOFF`);
    the surviving laps keep their ORIGINAL positions in the run, so the slope
    stays a true per-lap rate rather than being compressed by the gaps.

    The observed slope of a long run understates tyre degradation, because the
    car is simultaneously getting lighter. Adding the fuel effect back recovers
    the tyre's own trend, which is the number worth comparing between teams —
    unlike absolute pace, it does not depend on how much fuel was on board.
    """
    all_times = np.asarray(run.lap_times, dtype=float)
    positions = np.arange(all_times.size, dtype=float)
    keep = representative_mask(all_times, cutoff)
    times = all_times[keep]
    x = positions[keep]  # original lap positions, so the slope stays per-lap
    if times.size == 0:  # pathological; fall back to the raw run
        times, x = all_times, positions

    raw_slope = float(np.polyfit(x, times, 1)[0]) if times.size >= 2 else 0.0
    return LongRunPace(
        driver=run.driver,
        compound=run.compound,
        laps=int(times.size),
        median_s=float(np.median(times)),
        raw_slope_s_per_lap=raw_slope,
        deg_slope_s_per_lap=raw_slope + FUEL_EFFECT_S_PER_LAP,
        consistency_s=float(np.std(times)),
        laps_dropped=int(all_times.size - times.size),
    )


def session_runs(laps: pd.DataFrame) -> dict[str, list[tuple[Run, str]]]:
    """Every run in the session, classified, grouped by driver."""
    runs = detect_runs(laps)
    bests: dict[str, float] = {}
    for run in runs:
        best = bests.get(run.driver)
        bests[run.driver] = run.best if best is None else min(best, run.best)

    out: dict[str, list[tuple[Run, str]]] = {}
    for run in runs:
        out.setdefault(run.driver, []).append((run, classify_run(run, bests.get(run.driver))))
    return out


def longest_run_length(laps: pd.DataFrame) -> int:
    """The longest run of consecutive timed laps anyone managed.

    Used to decide whether a session can support long-run analysis at all —
    a disrupted or wet session may simply not contain one.
    """
    runs = detect_runs(laps)
    return max((r.length for r in runs), default=0)
