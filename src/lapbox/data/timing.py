"""Time-conversion helpers for Formula 1 lap and sector times.

FastF1 represents durations as :class:`pandas.Timedelta`. Machine-learning
models and JSON APIs, however, work with plain floating-point seconds. These
helpers provide a single, well-tested conversion boundary so the rest of the
codebase never re-implements the (surprisingly error-prone) NaT/None handling.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd


def timedelta_to_seconds(value: timedelta | pd.Timedelta | None) -> float | None:
    """Convert a duration to floating-point seconds.

    Args:
        value: A ``timedelta``/``pandas.Timedelta``, or ``None``/``NaT``.

    Returns:
        Seconds as a float, or ``None`` when the input is missing.
    """
    if value is None or value is pd.NaT or pd.isna(value):
        return None
    return pd.Timedelta(value).total_seconds()


def seconds_to_timedelta(seconds: float | int | None) -> pd.Timedelta | None:
    """Convert floating-point seconds to a :class:`pandas.Timedelta`.

    Args:
        seconds: Number of seconds, or ``None``.

    Returns:
        A ``pandas.Timedelta`` or ``None`` when the input is missing.
    """
    if seconds is None or pd.isna(seconds):
        return None
    return pd.Timedelta(seconds=float(seconds))


def format_laptime(value: timedelta | pd.Timedelta | float | None) -> str:
    """Render a lap/sector time as the canonical ``M:SS.mmm`` F1 string.

    Accepts either a duration or a raw number of seconds. Returns ``"-"`` for
    missing values so it is always safe to drop into a table or UI.

    Examples:
        >>> format_laptime(83.245)
        '1:23.245'
        >>> format_laptime(None)
        '-'
    """
    seconds = value if isinstance(value, (int, float)) else timedelta_to_seconds(value)
    if seconds is None or pd.isna(seconds):
        return "-"
    seconds = float(seconds)
    sign = "-" if seconds < 0 else ""
    seconds = abs(seconds)
    minutes, remainder = divmod(seconds, 60)
    return f"{sign}{int(minutes)}:{remainder:06.3f}"
