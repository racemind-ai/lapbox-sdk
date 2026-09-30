"""Lap-data cleaning pipeline.

Turns FastF1's laps (``session.laps``, or the same after
:func:`lapbox.data.normalize_laps`) into a clean, analysis-ready frame by
removing duplicates, invalid/implausible laps and rows unsuitable for modelling.

Every step is:

* **Config-driven** — thresholds live in :class:`LapCleaningConfig`, never
  hard-coded, so the same pipeline serves lap-time modelling (drop pit laps) and
  degradation studies (keep them) by swapping config.
* **Column-tolerant** — a step is skipped if its input columns are absent, so
  the pipeline is robust to partial data.
* **Observable** — :class:`CleaningResult` reports exactly how many rows each
  step removed, which is logged and handy for data-quality dashboards.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LapCleaningConfig:
    """Thresholds and switches controlling lap cleaning."""

    dedupe_keys: tuple[str, ...] = ("Driver", "LapNumber")
    require_laptime: bool = True
    drop_pit_laps: bool = True
    use_is_accurate: bool = True
    drop_deleted: bool = True
    min_lap_time_s: float = 40.0
    max_lap_time_s: float = 300.0
    # Per-driver robust outlier removal via median absolute deviation. ``None``
    # disables it (keeps cleaning fully deterministic and conservative).
    outlier_mad_threshold: float | None = None


@dataclass(slots=True)
class CleaningResult:
    """Outcome of a cleaning run: the cleaned frame plus a per-step audit."""

    frame: pd.DataFrame
    initial_rows: int
    removed: dict[str, int] = field(default_factory=dict)

    @property
    def final_rows(self) -> int:
        return len(self.frame)

    @property
    def total_removed(self) -> int:
        return self.initial_rows - self.final_rows


class LapCleaningPipeline:
    """Configurable, reusable cleaning pipeline for lap data."""

    def __init__(self, config: LapCleaningConfig | None = None) -> None:
        self.config = config or LapCleaningConfig()

    def run(self, laps: pd.DataFrame) -> pd.DataFrame:
        """Clean ``laps`` and return only the cleaned frame."""
        return self.clean(laps).frame

    def clean(self, laps: pd.DataFrame) -> CleaningResult:
        """Clean ``laps`` and return the frame plus a per-step removal report."""
        df = laps.copy()
        if "LapTimeSeconds" not in df.columns and "LapTime" in df.columns:
            # FastF1's own laps carry the time as a Timedelta only; without this
            # the missing-time and plausibility steps would silently do nothing.
            df["LapTimeSeconds"] = pd.to_timedelta(df["LapTime"]).dt.total_seconds()
        result = CleaningResult(frame=df, initial_rows=len(df))

        df = self._step(result, "duplicates", df, self._drop_duplicates)
        df = self._step(result, "missing_laptime", df, self._drop_missing_laptime)
        df = self._step(result, "pit_laps", df, self._drop_pit_laps)
        df = self._step(result, "inaccurate", df, self._drop_inaccurate)
        df = self._step(result, "deleted", df, self._drop_deleted)
        df = self._step(result, "implausible_time", df, self._drop_implausible_times)
        df = self._step(result, "outliers", df, self._drop_outliers)

        result.frame = df.reset_index(drop=True)
        logger.info(
            "Lap cleaning complete",
            extra={
                "initial_rows": result.initial_rows,
                "final_rows": result.final_rows,
                "removed": result.total_removed,
            },
        )
        return result

    # -- Step runner ---------------------------------------------------------

    @staticmethod
    def _step(result: CleaningResult, name: str, df: pd.DataFrame, fn) -> pd.DataFrame:
        before = len(df)
        cleaned = fn(df)
        result.removed[name] = before - len(cleaned)
        return cleaned

    # -- Individual steps (each returns a possibly-smaller frame) -------------

    def _drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        keys = [k for k in self.config.dedupe_keys if k in df.columns]
        return df.drop_duplicates(subset=keys or None)

    def _drop_missing_laptime(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.config.require_laptime or "LapTimeSeconds" not in df.columns:
            return df
        return df[df["LapTimeSeconds"].notna()]

    def _drop_pit_laps(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.config.drop_pit_laps:
            return df
        mask = pd.Series(True, index=df.index)
        for col in ("PitInTime", "PitOutTime"):
            if col in df.columns:
                mask &= df[col].isna()
        return df[mask]

    def _drop_inaccurate(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.config.use_is_accurate or "IsAccurate" not in df.columns:
            return df
        return df[df["IsAccurate"].astype("boolean").fillna(False)]

    def _drop_deleted(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.config.drop_deleted or "Deleted" not in df.columns:
            return df
        return df[~df["Deleted"].astype("boolean").fillna(False)]

    def _drop_implausible_times(self, df: pd.DataFrame) -> pd.DataFrame:
        if "LapTimeSeconds" not in df.columns:
            return df
        lt = df["LapTimeSeconds"]
        # Keep NaN here (handled by the missing-laptime step per config); only
        # remove values that are present *and* outside the plausible window.
        keep = lt.isna() | ((lt >= self.config.min_lap_time_s) & (lt <= self.config.max_lap_time_s))
        return df[keep]

    def _drop_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        threshold = self.config.outlier_mad_threshold
        if threshold is None or "LapTimeSeconds" not in df.columns or "Driver" not in df.columns:
            return df

        # Robust per-driver outlier detection via the median absolute deviation
        # (MAD), computed vectorised with grouped transforms. 0.6745 scales the
        # MAD to be comparable to a standard deviation for a normal distribution.
        times = df["LapTimeSeconds"]
        grouped = times.groupby(df["Driver"])
        median = grouped.transform("median")
        abs_dev = (times - median).abs()
        mad = abs_dev.groupby(df["Driver"]).transform("median")
        robust_z = 0.6745 * abs_dev / mad.replace(0, np.nan)
        # Keep rows with no defined spread (mad==0/NaN) or within the threshold.
        keep = robust_z.isna() | (robust_z <= threshold)
        return df[keep]
