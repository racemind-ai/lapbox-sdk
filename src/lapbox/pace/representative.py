"""A circuit's normal lap time, ignoring laps that were not run at it."""

from __future__ import annotations

import pandas as pd

# A lap slower than this share of the session best was not run at
# representative pace: rain, a safety car, traffic, or a damaged car.
BASE_TIME_CLEAN_PCT = 1.07
# Below this many representative laps the filter is measuring noise, so the
# unfiltered median is the safer answer.
MIN_CLEAN_LAPS = 20


def representative_base_time(lap_times: pd.Series) -> float:
    """A circuit's normal pace, ignoring laps that were not run at it.

    Falls back to the plain median when too few laps survive, because a
    handful of laps is a worse estimator than a contaminated many.
    """
    clean = lap_times.dropna()
    if clean.empty:
        return float("nan")
    best = clean.min()
    representative = clean[clean <= best * BASE_TIME_CLEAN_PCT]
    if len(representative) < MIN_CLEAN_LAPS:
        return float(clean.median())
    return float(representative.median())
