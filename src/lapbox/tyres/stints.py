"""Degradation stint by stint: the tyre's own trend, once fuel is out of the way.

Ported from LapBox's Race Analysis page (``truePace.ts``), where the same maths
runs in the browser for the tyre-life panel. Takes FastF1's laps as they come
(``session.laps``); stints are FastF1's ``Stint`` numbers.

Degradation and fuel burn pull lap times in opposite directions, so a stint
whose raw times stay flat is not a stint without degradation — the tyre was
going off at the rate the car was getting lighter. Each lap is fuel-corrected
first (:func:`lapbox.pace.fuel_correct`), so the slope across a stint is the
tyre falling away.
"""

from __future__ import annotations

import pandas as pd

from lapbox.pace.race import (
    _drivers,
    _fuel_correct,
    _median,
    _representative,
    _slope,
    _teams,
    _total_laps,
)

# Fewer clean laps than this and a stint is too short to fit a trend through.
_MIN_STINT_LAPS = 3

_STINT_COLUMNS = ["driver", "team", "stint", "compound", "laps", "median_corrected", "deg_per_lap"]
_SUMMARY_COLUMNS = ["compound", "stints", "laps", "median_deg", "best_deg", "worst_deg"]


def _stints(laps: pd.DataFrame) -> dict[str, list[tuple[int, str | None, int, int]]]:
    """Each driver's ``(stint, compound, first lap, last lap)``, in stint order.

    The compound is the one on the stint's first row, as the race-analysis
    payload records it.
    """
    if not {"Driver", "Stint", "LapNumber"}.issubset(laps.columns):
        return {}
    out: dict[str, list[tuple[int, str | None, int, int]]] = {}
    for driver, group in laps.groupby("Driver", sort=False):
        for stint, rows in group.dropna(subset=["Stint", "LapNumber"]).groupby("Stint"):
            compound = rows["Compound"].iloc[0] if "Compound" in rows.columns else None
            out.setdefault(str(driver), []).append(
                (
                    int(stint),
                    None if compound is None or pd.isna(compound) else str(compound),
                    int(rows["LapNumber"].min()),
                    int(rows["LapNumber"].max()),
                )
            )
    return out


def stint_degradation(laps: pd.DataFrame, total_laps: int | None = None) -> pd.DataFrame:
    """Fuel-corrected pace and degradation for every driver's every stint.

    A stint's laps are its timed, non-pit laps, fuel-corrected to end-of-race
    fuel, keeping those within 107 % of the stint's best raw lap (safety car,
    traffic and mistakes out). A stint with fewer than three such laps is left
    out. ``total_laps`` defaults to the highest lap number in ``laps``.

    Returns:
        One row per stint: ``driver``, ``team``, ``stint``, ``compound``,
        ``laps`` (counted), ``median_corrected`` (s) and ``deg_per_lap``
        (s/lap; the slope of the corrected times against lap number).
    """
    total = _total_laps(laps, total_laps)
    teams = _teams(laps)
    stints = _stints(laps)
    rows = []
    for driver, driver_laps in _drivers(laps, gaps={}).items():
        corrected = _fuel_correct(driver_laps, total)
        for stint, compound, first, last in stints.get(driver, []):
            clean = _representative([lap for lap in corrected if first <= lap.lap <= last])
            if len(clean) < _MIN_STINT_LAPS:
                continue
            rows.append(
                {
                    "driver": driver,
                    "team": teams.get(driver),
                    "stint": stint,
                    "compound": compound,
                    "laps": len(clean),
                    "median_corrected": _median([lap.corrected for lap in clean]),
                    "deg_per_lap": _slope(
                        [lap.lap for lap in clean], [lap.corrected for lap in clean]
                    ),
                }
            )
    return pd.DataFrame(rows, columns=_STINT_COLUMNS)


def compound_summary(stints: pd.DataFrame) -> pd.DataFrame:
    """How each compound behaved across the field, kindest-wearing first.

    Takes :func:`stint_degradation`'s rows. The median is used rather than the
    mean, so one driver's ruined stint does not define a compound; stints with
    no compound recorded are left out.

    Returns:
        ``compound``, ``stints``, ``laps``, ``median_deg``, ``best_deg`` and
        ``worst_deg`` (s/lap), lowest median degradation first.
    """
    by_compound: dict[str, list[tuple[int, float]]] = {}
    for compound, laps, deg in zip(
        stints["compound"], stints["laps"], stints["deg_per_lap"], strict=True
    ):
        if compound is None or pd.isna(compound):
            continue
        by_compound.setdefault(str(compound), []).append((int(laps), float(deg)))
    rows = [
        {
            "compound": compound,
            "stints": len(entries),
            "laps": sum(n for n, _ in entries),
            "median_deg": _median([d for _, d in entries]),
            "best_deg": min(d for _, d in entries),
            "worst_deg": max(d for _, d in entries),
        }
        for compound, entries in by_compound.items()
    ]
    rows.sort(key=lambda row: row["median_deg"])
    return pd.DataFrame(rows, columns=_SUMMARY_COLUMNS)
