"""Race pace from a session's laps: fuel-corrected pace, clean air, consistency, ideal lap.

Ported from LapBox's Race Analysis page, where the same maths runs in the browser.
Every function takes FastF1's laps as they come (``session.laps``); a lap is
*timed* when it has a lap time, and a *pit lap* when it has a pit-in or pit-out
time. Drivers keep the order they first appear in, which for ``session.laps`` is
FastF1's own driver order, and rankings are stable sorts over it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from lapbox.data import gaps_to_car_ahead
from lapbox.practice import FUEL_EFFECT_S_PER_LAP

# Laps slower than this multiple of a driver's best are compromised (safety car,
# traffic, mistakes) and are not a measure of pace.
REPRESENTATIVE = 1.07

# Aerodynamic wake meaningfully hurts a following car within roughly this gap
# in the current ground-effect era. Wider than DRS range (1s), because the
# performance loss starts well before a driver is close enough to attack.
DIRTY_AIR_S = 2.0

# Lap 1 is decided by grid slot and the start, not by pace or traffic.
_FIRST_RACING_LAP = 2

# Below this many laps on either side of the clean-air split, a median is noise.
_MIN_SAMPLE = 3

Gaps = Mapping[tuple[str, int], float]


# --------------------------------------------------------------------------- #
# Reading FastF1 laps
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class _Lap:
    lap: int
    seconds: float | None
    compound: str | None
    pit: bool
    sectors: tuple[float | None, float | None, float | None]
    gap_ahead: float | None


def _seconds(frame: pd.DataFrame, name: str) -> pd.Series:
    """``{name}Seconds`` when present, else ``{name}`` read as a Timedelta."""
    if f"{name}Seconds" in frame.columns:
        return pd.to_numeric(frame[f"{name}Seconds"], errors="coerce")
    if name in frame.columns:
        column = frame[name]
        if pd.api.types.is_datetime64_any_dtype(column):
            # An all-NaT column is typed datetime by pandas; it holds no durations.
            column = column.astype(object).where(column.notna(), None)
        return pd.to_timedelta(column, errors="coerce").dt.total_seconds()
    return pd.Series(float("nan"), index=frame.index)


def _present(frame: pd.DataFrame, name: str) -> pd.Series:
    """Whether ``{name}Seconds`` (or ``{name}``) holds a value on each lap."""
    for column in (f"{name}Seconds", name):
        if column in frame.columns:
            return frame[column].notna()
    return pd.Series(False, index=frame.index)


def _value(value: object) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


def _drivers(laps: pd.DataFrame, gaps: Gaps | None = None) -> dict[str, list[_Lap]]:
    """Every driver's laps in lap order, keyed in first-appearance order."""
    if laps.empty or "Driver" not in laps.columns or "LapNumber" not in laps.columns:
        return {}
    if gaps is None:
        gaps = gaps_to_car_ahead(laps)

    frame = laps[laps["LapNumber"].notna()].reset_index(drop=True)
    # Each column is read once, in lap order: about 3.5x faster than looking every
    # value up row by row (49 -> 13 ms for a 1,054-lap race).
    order = frame.sort_values("LapNumber", kind="stable").index.to_numpy()
    drivers = frame["Driver"].astype(str).to_numpy()[order]
    lap_numbers = frame["LapNumber"].to_numpy()[order]
    seconds = _seconds(frame, "LapTime").to_numpy()[order]
    s1, s2, s3 = (_seconds(frame, f"Sector{i}Time").to_numpy()[order] for i in (1, 2, 3))
    pit = (_present(frame, "PitInTime") | _present(frame, "PitOutTime")).to_numpy()[order]
    compounds = (
        frame["Compound"].to_numpy(dtype=object)[order]
        if "Compound" in frame.columns
        else [None] * len(order)
    )

    out: dict[str, list[_Lap]] = {}
    for driver, lap_no, secs, sec1, sec2, sec3, is_pit, compound in zip(
        drivers, lap_numbers, seconds, s1, s2, s3, pit, compounds, strict=True
    ):
        lap = int(lap_no)
        out.setdefault(driver, []).append(
            _Lap(
                lap=lap,
                seconds=_value(secs),
                compound=None if compound is None or pd.isna(compound) else str(compound),
                pit=bool(is_pit),
                sectors=(_value(sec1), _value(sec2), _value(sec3)),
                gap_ahead=gaps.get((driver, lap)),
            )
        )
    first_seen = {str(d): i for i, d in enumerate(pd.unique(frame["Driver"].astype(str)))}
    return dict(sorted(out.items(), key=lambda item: first_seen[item[0]]))


def _own_laps(laps: pd.DataFrame, driver: str) -> list[_Lap]:
    """One driver's laps, parsing only their rows rather than the whole session."""
    if "Driver" not in laps.columns:
        return []
    return _drivers(laps[laps["Driver"].astype(str) == driver], gaps={}).get(driver, [])


def _teams(laps: pd.DataFrame) -> dict[str, str | None]:
    if "Team" not in laps.columns or "Driver" not in laps.columns:
        return {}
    teams: dict[str, str | None] = {}
    for driver, team in zip(laps["Driver"].astype(str), laps["Team"], strict=False):
        if driver not in teams or teams[driver] is None:
            teams[driver] = None if team is None or pd.isna(team) else str(team)
    return teams


def _total_laps(laps: pd.DataFrame, total_laps: int | None) -> int:
    if total_laps is not None:
        return int(total_laps)
    if "LapNumber" not in laps.columns or laps["LapNumber"].dropna().empty:
        return 0
    return int(laps["LapNumber"].max())


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def _slope(x: list[float], y: list[float]) -> float:
    """Least-squares slope of y against x, or 0 when it is not defined."""
    n = min(len(x), len(y))
    if n < 2:
        return 0.0
    mx = sum(x[:n]) / n
    my = sum(y[:n]) / n
    num = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    den = sum((x[i] - mx) ** 2 for i in range(n))
    return 0.0 if den == 0 else num / den


# --------------------------------------------------------------------------- #
# Fuel-corrected pace
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class _Corrected:
    lap: int
    raw: float
    corrected: float
    compound: str | None


def _fuel_correct(laps: list[_Lap], total_laps: int) -> list[_Corrected]:
    return [
        _Corrected(
            lap=lap.lap,
            raw=lap.seconds,
            corrected=lap.seconds - max(0, total_laps - lap.lap) * FUEL_EFFECT_S_PER_LAP,
            compound=lap.compound,
        )
        for lap in laps
        if lap.seconds is not None and not lap.pit
    ]


def _representative(laps: list[_Corrected]) -> list[_Corrected]:
    if not laps:
        return []
    best = min(lap.raw for lap in laps)
    return [lap for lap in laps if lap.raw <= best * REPRESENTATIVE]


def fuel_correct(laps: pd.DataFrame, total_laps: int | None = None) -> pd.DataFrame:
    """Every timed, non-pit lap normalised to end-of-race fuel.

    A car gets roughly two seconds a lap faster over a Grand Prix purely from
    burning fuel, so a raw lap-50 time flatters the car. Each lap has the fuel
    still to be burned taken off: ``(total_laps - lap) × FUEL_EFFECT_S_PER_LAP``.
    ``total_laps`` defaults to the highest lap number in ``laps``.

    Returns:
        ``Driver``, ``LapNumber``, ``Compound``, ``raw`` and ``corrected`` (s).
    """
    total = _total_laps(laps, total_laps)
    rows = [
        {
            "Driver": driver,
            "LapNumber": c.lap,
            "Compound": c.compound,
            "raw": c.raw,
            "corrected": c.corrected,
        }
        for driver, driver_laps in _drivers(laps, gaps={}).items()
        for c in _fuel_correct(driver_laps, total)
    ]
    return pd.DataFrame(rows, columns=["Driver", "LapNumber", "Compound", "raw", "corrected"])


def true_pace_ranking(laps: pd.DataFrame, total_laps: int | None = None) -> pd.DataFrame:
    """The field ranked by fuel-corrected pace, with the raw rank kept alongside.

    Each driver's median over representative laps (within 107 % of their best),
    raw and fuel-corrected. Drivers with fewer than three such laps are left out.
    ``rank_change`` is positive when a driver moves UP once fuel is accounted
    for: they ran their quick laps on heavy fuel.

    Returns:
        ``driver``, ``team``, ``laps``, ``raw_median``, ``corrected_median``,
        ``gap`` (to the quickest corrected pace), ``raw_rank``,
        ``corrected_rank`` and ``rank_change``, quickest corrected pace first.
    """
    total = _total_laps(laps, total_laps)
    teams = _teams(laps)
    rows = []
    for driver, driver_laps in _drivers(laps, gaps={}).items():
        rep = _representative(_fuel_correct(driver_laps, total))
        if len(rep) < 3:
            continue
        rows.append(
            {
                "driver": driver,
                "team": teams.get(driver),
                "laps": len(rep),
                "raw_median": _median([c.raw for c in rep]),
                "corrected_median": _median([c.corrected for c in rep]),
            }
        )
    columns = [
        "driver",
        "team",
        "laps",
        "raw_median",
        "corrected_median",
        "gap",
        "raw_rank",
        "corrected_rank",
        "rank_change",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)

    raw_order = [r["driver"] for r in sorted(rows, key=lambda r: r["raw_median"])]
    corrected = sorted(rows, key=lambda r: r["corrected_median"])
    leader = corrected[0]["corrected_median"]
    out = []
    for i, r in enumerate(corrected):
        raw_rank = raw_order.index(r["driver"]) + 1
        out.append(
            {
                **r,
                "gap": r["corrected_median"] - leader,
                "raw_rank": raw_rank,
                "corrected_rank": i + 1,
                "rank_change": raw_rank - (i + 1),
            }
        )
    return pd.DataFrame(out, columns=columns)


# --------------------------------------------------------------------------- #
# Clean air vs traffic
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class CleanAirSplit:
    """One driver's race split by the gap to the car ahead."""

    driver: str
    team: str | None
    clean_laps: int
    traffic_laps: int
    clean_pace: float
    """Fuel-corrected median of the clean-air laps (s)."""
    traffic_pace: float
    delta: float
    """``traffic_pace - clean_pace``: positive means traffic cost time."""
    traffic_share: float
    """Percentage of racing laps spent within the dirty-air threshold."""
    led_laps: int
    """Laps with nobody ahead at all, counted as clean air."""


def _clean_air(
    driver: str,
    team: str | None,
    driver_laps: list[_Lap],
    total_laps: int,
    threshold: float,
) -> CleanAirSplit | None:
    by_lap = {c.lap: c for c in _fuel_correct(driver_laps, total_laps)}
    racing = [
        lap
        for lap in driver_laps
        if lap.lap >= _FIRST_RACING_LAP
        and lap.seconds is not None
        and not lap.pit
        and lap.lap in by_lap
    ]
    if not racing:
        return None

    best = min(lap.seconds for lap in racing)
    usable = [lap for lap in racing if lap.seconds <= best * REPRESENTATIVE]

    clean: list[float] = []
    traffic: list[float] = []
    led = 0
    for lap in usable:
        value = by_lap[lap.lap].corrected
        if lap.gap_ahead is None:
            led += 1
            clean.append(value)  # nobody ahead: the cleanest air there is
        elif lap.gap_ahead <= threshold:
            traffic.append(value)
        else:
            clean.append(value)

    if len(clean) < _MIN_SAMPLE or len(traffic) < _MIN_SAMPLE:
        return None
    clean_pace = _median(clean)
    traffic_pace = _median(traffic)
    return CleanAirSplit(
        driver=driver,
        team=team,
        clean_laps=len(clean),
        traffic_laps=len(traffic),
        clean_pace=clean_pace,
        traffic_pace=traffic_pace,
        delta=traffic_pace - clean_pace,
        traffic_share=len(traffic) / (len(clean) + len(traffic)) * 100,
        led_laps=led,
    )


def clean_air_split(
    laps: pd.DataFrame,
    driver: str,
    total_laps: int | None = None,
    *,
    threshold: float = DIRTY_AIR_S,
    gaps: Gaps | None = None,
) -> CleanAirSplit | None:
    """Split one driver's race into clean-air and traffic laps and compare pace.

    Racing laps (lap 2 on, timed, not pit laps, within 107 % of the driver's
    best) are fuel-corrected, then split by the gap to the car ahead at the
    line: within ``threshold`` seconds is traffic. Leading a lap counts as clean
    air. ``gaps`` defaults to :func:`lapbox.data.gaps_to_car_ahead`, which only
    counts cars on the same lap — a lapped car in front is not traffic.

    Returns ``None`` when either side has fewer than three laps, which is common
    for a driver who led throughout or never got clear.
    """
    driver_laps = _drivers(laps, gaps).get(str(driver), [])
    return _clean_air(
        str(driver),
        _teams(laps).get(str(driver)),
        driver_laps,
        _total_laps(laps, total_laps),
        threshold,
    )


def clean_air_ranking(
    laps: pd.DataFrame,
    total_laps: int | None = None,
    *,
    threshold: float = DIRTY_AIR_S,
    gaps: Gaps | None = None,
) -> pd.DataFrame:
    """Every driver who ran enough of both, biggest traffic penalty first.

    Returns:
        One row per :class:`CleanAirSplit` field.
    """
    total = _total_laps(laps, total_laps)
    teams = _teams(laps)
    splits = [
        s
        for driver, driver_laps in _drivers(laps, gaps).items()
        if (s := _clean_air(driver, teams.get(driver), driver_laps, total, threshold)) is not None
    ]
    splits.sort(key=lambda s: -s.delta)
    columns = list(CleanAirSplit.__dataclass_fields__)
    return pd.DataFrame([{c: getattr(s, c) for c in columns} for s in splits], columns=columns)


# --------------------------------------------------------------------------- #
# Consistency
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class DriverConsistency:
    """Spread of a driver's representative laps."""

    driver: str
    laps: int
    """Representative laps counted: timed, not pit laps, within 107 % of the best."""
    best: float
    mean: float
    std: float
    """Population standard deviation (s)."""
    cv: float
    """Coefficient of variation, % (``std / mean``)."""
    series: tuple[tuple[int, float], ...]
    """``(lap, seconds off the driver's best)`` for every representative lap."""


def _consistency(driver: str, driver_laps: list[_Lap]) -> DriverConsistency | None:
    timed = [
        (lap.lap, lap.seconds) for lap in driver_laps if lap.seconds is not None and not lap.pit
    ]
    if len(timed) < 3:
        return None
    raw_best = min(s for _, s in timed)
    rep = [(n, s) for n, s in timed if s <= raw_best * REPRESENTATIVE]
    if len(rep) < 3:
        return None

    secs = [s for _, s in rep]
    best = min(secs)
    mean = sum(secs) / len(secs)
    std = math.sqrt(sum((s - mean) ** 2 for s in secs) / len(secs))
    return DriverConsistency(
        driver=driver,
        laps=len(rep),
        best=best,
        mean=mean,
        std=std,
        cv=std / mean * 100 if mean else 0.0,
        series=tuple((n, s - best) for n, s in rep),
    )


def driver_consistency(laps: pd.DataFrame, driver: str) -> DriverConsistency | None:
    """Consistency of one driver's representative laps.

    Pit laps and laps over 107 % of the driver's best (safety car, traffic,
    mistakes) are left out. ``None`` with fewer than three laps to measure.
    Only ``driver``'s rows are read, so it can be called once per driver on a
    whole session; :func:`consistency_ranking` does every driver in one pass.
    """
    return _consistency(str(driver), _own_laps(laps, str(driver)))


def consistency_ranking(laps: pd.DataFrame, drivers: list[str] | None = None) -> pd.DataFrame:
    """Consistency for each driver (or only ``drivers``), most consistent first.

    Returns:
        ``driver``, ``laps``, ``best``, ``mean``, ``std`` and ``cv`` — the
        per-lap ``series`` is on :func:`driver_consistency`.
    """
    wanted = None if drivers is None else {str(d) for d in drivers}
    results = [
        c
        for driver, driver_laps in _drivers(laps, gaps={}).items()
        if (wanted is None or driver in wanted)
        and (c := _consistency(driver, driver_laps)) is not None
    ]
    results.sort(key=lambda c: c.cv)
    columns = ["driver", "laps", "best", "mean", "std", "cv"]
    return pd.DataFrame([{k: getattr(c, k) for k in columns} for c in results], columns=columns)


# --------------------------------------------------------------------------- #
# Ideal lap
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class BestSector:
    """A driver's best time in one sector, and what their fastest lap lost there."""

    sector: int
    seconds: float
    lap: int
    """The lap this best sector was set on."""
    loss_on_fastest: float | None
    """Fastest actual lap's sector minus this best (s, >= 0); ``None`` if untimed."""


@dataclass(frozen=True, slots=True)
class IdealLap:
    """The sum of a driver's best sectors against their fastest actual lap."""

    driver: str
    sectors: tuple[BestSector, BestSector, BestSector]
    ideal: float
    """Sum of the three best sectors (s)."""
    fastest: float
    """The driver's fastest clean lap (s)."""
    fastest_lap: int
    gain: float
    """``fastest - ideal`` (s, >= 0): time left on the table."""
    fastest_sectors: tuple[float, float, float] | None
    """Sector times of the fastest lap, when all three were recorded."""
    laps: int
    """Clean laps considered."""


def _sector(lap: _Lap, index: int) -> float | None:
    value = lap.sectors[index]
    return value if value is not None and math.isfinite(value) and value > 0 else None


def ideal_lap(laps: pd.DataFrame, driver: str) -> IdealLap | None:
    """A driver's ideal lap: the sum of their best S1, S2 and S3.

    Only clean laps count (timed, not pit laps). Each sector's best is searched
    independently, so a lap missing one split still contributes the others. It
    is rarely achievable in one lap — the best sectors usually come from laps
    on different fuel loads and tyre states.

    Returns ``None`` when a sector was never timed on a clean lap, and when the
    ideal lap comes out *slower* than the fastest actual lap: that happens only
    when the fastest lap is itself missing a split (FastF1 leaves lap 1's S1
    blank after a standing start), and there is no honest gain to state then.

    Only ``driver``'s rows are read, so it can be called once per driver on a
    whole session.
    """
    clean = [
        lap
        for lap in _own_laps(laps, str(driver))
        if not lap.pit and lap.seconds is not None and lap.seconds > 0
    ]
    if not clean:
        return None

    fastest_lap = clean[0]
    for lap in clean:
        if lap.seconds < fastest_lap.seconds:
            fastest_lap = lap
    fastest = fastest_lap.seconds

    bests: list[BestSector] = []
    for index in (0, 1, 2):
        best: tuple[float, int] | None = None
        for lap in clean:
            value = _sector(lap, index)
            if value is not None and (best is None or value < best[0]):
                best = (value, lap.lap)
        if best is None:
            return None  # a sector never timed -> no ideal lap
        on_fastest = _sector(fastest_lap, index)
        bests.append(
            BestSector(
                sector=index + 1,
                seconds=best[0],
                lap=best[1],
                loss_on_fastest=None if on_fastest is None else max(0.0, on_fastest - best[0]),
            )
        )

    ideal = sum(b.seconds for b in bests)
    fastest_split = [_sector(fastest_lap, i) for i in (0, 1, 2)]
    fastest_sectors = (
        (fastest_split[0], fastest_split[1], fastest_split[2])
        if all(v is not None for v in fastest_split)
        else None
    )
    # The tolerance sits far below timing resolution (1 ms): it only absorbs the
    # disagreement possible between FastF1's lap time and its separately
    # measured sector times. A real inversion is always orders larger.
    if ideal > fastest + 1e-6:
        return None

    return IdealLap(
        driver=str(driver),
        sectors=(bests[0], bests[1], bests[2]),
        ideal=ideal,
        fastest=fastest,
        fastest_lap=fastest_lap.lap,
        gain=max(0.0, fastest - ideal),
        fastest_sectors=fastest_sectors,
        laps=len(clean),
    )


def biggest_loss(result: IdealLap) -> BestSector | None:
    """The sector with the most time to find on the fastest lap, if any."""
    with_loss = [
        s for s in result.sectors if s.loss_on_fastest is not None and s.loss_on_fastest > 0
    ]
    worst: BestSector | None = None
    for s in with_loss:
        if worst is None or s.loss_on_fastest > worst.loss_on_fastest:
            worst = s
    return worst
