"""Session-level lap data: seconds columns, weather, pit stops, stints and gaps.

Pure data-shaping helpers over FastF1's own frames (``session.laps``,
``session.weather_data``). Each takes a plain DataFrame and returns a new one,
with no I/O or global state.
"""

from __future__ import annotations

import pandas as pd

from lapbox.data.timing import timedelta_to_seconds

# Duration columns (FastF1 stores these as timedeltas) and the float-seconds
# companion column added during normalisation.
_LAP_TIME_COLUMNS: dict[str, str] = {
    "LapTime": "LapTimeSeconds",
    "Sector1Time": "Sector1TimeSeconds",
    "Sector2Time": "Sector2TimeSeconds",
    "Sector3Time": "Sector3TimeSeconds",
    "PitInTime": "PitInTimeSeconds",
    "PitOutTime": "PitOutTimeSeconds",
}


def _add_seconds_columns(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Add float-seconds companion columns for any present timedelta columns."""
    for source, target in mapping.items():
        if source in df.columns:
            df[target] = df[source].apply(timedelta_to_seconds)
    return df


def normalize_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the laps frame with float-seconds time columns added."""
    result = laps.copy()
    return _add_seconds_columns(result, _LAP_TIME_COLUMNS)


def normalize_weather(weather: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the weather frame with a numeric ``TimeSeconds`` column."""
    result = weather.copy()
    if "Time" in result.columns:
        result["TimeSeconds"] = result["Time"].apply(timedelta_to_seconds)
    return result


def derive_pit_stops(laps: pd.DataFrame) -> pd.DataFrame:
    """Derive pit-stop events by pairing each in-lap with the next out-lap.

    FastF1 records ``PitInTime`` on the lap a car enters the pits and
    ``PitOutTime`` on the following out-lap. This pairs them per driver to
    compute a stationary-plus-lane pit-stop duration where both are available.

    Returns:
        One row per pit stop with columns ``Driver``, ``LapNumber`` (in-lap),
        ``Stint``, ``Compound``, ``PitInTimeSeconds``, ``PitOutTimeSeconds`` and
        ``PitStopDurationSeconds`` (``NaN`` when the out-lap time is missing).
    """
    required = {"Driver", "LapNumber", "PitInTime"}
    if not required.issubset(laps.columns):
        return pd.DataFrame(
            columns=[
                "Driver",
                "LapNumber",
                "Stint",
                "Compound",
                "PitInTimeSeconds",
                "PitOutTimeSeconds",
                "PitStopDurationSeconds",
            ]
        )

    records: list[dict[str, object]] = []
    has_out = "PitOutTime" in laps.columns
    for driver, group in laps.sort_values("LapNumber").groupby("Driver", sort=False):
        next_out = group["PitOutTime"].shift(-1) if has_out else None
        for pos, (_, row) in enumerate(group.iterrows()):
            if pd.isna(row["PitInTime"]):
                continue
            pit_in = timedelta_to_seconds(row["PitInTime"])
            pit_out = timedelta_to_seconds(next_out.iloc[pos]) if has_out else None
            duration = (
                pit_out - pit_in if pit_in is not None and pit_out is not None else float("nan")
            )
            records.append(
                {
                    "Driver": driver,
                    "LapNumber": row["LapNumber"],
                    "Stint": row.get("Stint"),
                    "Compound": row.get("Compound"),
                    "PitInTimeSeconds": pit_in,
                    "PitOutTimeSeconds": pit_out,
                    "PitStopDurationSeconds": duration,
                }
            )
    return pd.DataFrame.from_records(
        records,
        columns=[
            "Driver",
            "LapNumber",
            "Stint",
            "Compound",
            "PitInTimeSeconds",
            "PitOutTimeSeconds",
            "PitStopDurationSeconds",
        ],
    )


def derive_tyre_stints(laps: pd.DataFrame) -> pd.DataFrame:
    """Aggregate laps into per-driver tyre stints.

    Returns:
        One row per (driver, stint) with the compound, start/end lap, stint
        length and start/end tyre life (in laps) where available.
    """
    required = {"Driver", "Stint", "Compound", "LapNumber"}
    if not required.issubset(laps.columns):
        return pd.DataFrame(
            columns=[
                "Driver",
                "Stint",
                "Compound",
                "StintStartLap",
                "StintEndLap",
                "StintLength",
                "StartTyreLife",
                "EndTyreLife",
            ]
        )

    agg: dict[str, tuple[str, str]] = {
        "StintStartLap": ("LapNumber", "min"),
        "StintEndLap": ("LapNumber", "max"),
        "StintLength": ("LapNumber", "count"),
    }
    if "TyreLife" in laps.columns:
        agg["StartTyreLife"] = ("TyreLife", "min")
        agg["EndTyreLife"] = ("TyreLife", "max")

    stints = (
        laps.groupby(["Driver", "Stint", "Compound"], dropna=False)
        .agg(**agg)
        .reset_index()
        .sort_values(["Driver", "StintStartLap"])
        .reset_index(drop=True)
    )
    return stints


def gaps_to_car_ahead(laps: pd.DataFrame) -> dict[tuple[str, int], float]:
    """Seconds to the car ahead for every ``(driver, lap)``.

    FastF1's ``Time`` is elapsed session time as a driver crosses the line, so
    sorting one lap number by it gives the running order and the differences
    are the real gaps. Comparing WITHIN a lap number matters: everyone in the
    comparison has covered the same distance, so the difference is a time gap
    rather than a lap of daylight.

    The leader of each lap has no car ahead and is simply absent from the
    result. Lapped cars sit on a different lap number, so they are not counted
    as traffic even though they are physically in front — a limitation worth
    stating rather than papering over.
    """
    if "Time" not in laps.columns or "LapNumber" not in laps.columns:
        return {}

    frame = laps[["Driver", "LapNumber", "Time"]].copy()
    frame["elapsed"] = pd.to_timedelta(frame["Time"], errors="coerce").dt.total_seconds()
    frame = frame.dropna(subset=["elapsed", "LapNumber"])

    gaps: dict[tuple[str, int], float] = {}
    for lap_no, group in frame.groupby("LapNumber"):
        ordered = group.sort_values("elapsed")
        previous: float | None = None
        for _, row in ordered.iterrows():
            elapsed = float(row["elapsed"])
            if previous is not None:
                gaps[(str(row["Driver"]), int(lap_no))] = round(elapsed - previous, 3)
            previous = elapsed
    return gaps
