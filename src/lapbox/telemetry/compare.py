"""Two-lap telemetry comparison.

Compares two drivers' laps by aligning their telemetry onto a common distance
axis, then computing per-channel deltas, a cumulative time gap, and minisector
dominance (who is faster where on track).

All functions are pure and operate on telemetry DataFrames, so they compose with
:mod:`lapbox.telemetry.lap` and are unit-tested without any network.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from lapbox.telemetry.lap import ANALYSIS_CHANNELS, resample_by_distance

logger = logging.getLogger(__name__)


# A lap's distance axis is integrated from speed, so two drivers' traces drift
# against each other and can start from slightly different track points. Search
# no further than this fraction of a lap for the offset between them; the worst
# real case measured across a season was 2.6 %.
_MAX_SHIFT_FRACTION = 0.05


def _best_circular_shift(ref: np.ndarray, other: np.ndarray, max_shift: int) -> int:
    """Return the circular shift of ``other`` that best matches ``ref``.

    A lap is a closed loop sampled over exactly one circuit of it, so rolling
    the trace is the physically correct way to slide one lap against another.
    Matching is on speed, which has sharp, unmistakable corner minima.
    """
    if max_shift < 1 or len(ref) != len(other) or len(ref) < 3:
        return 0
    best_shift, best_err = 0, float("inf")
    for shift in range(-max_shift, max_shift + 1):
        err = float(np.mean((np.roll(other, shift) - ref) ** 2))
        if err < best_err:
            best_err, best_shift = err, shift
    return best_shift


def align_by_distance(
    tel_a: pd.DataFrame,
    tel_b: pd.DataFrame,
    *,
    num_points: int = 1000,
    channels: tuple[str, ...] | None = None,
    suffixes: tuple[str, str] = ("_a", "_b"),
) -> pd.DataFrame:
    """Resample both laps onto one distance grid, matched by track position.

    Sampling both laps at the same *number* is not the same as sampling them at
    the same *place*. ``Distance`` is integrated from speed, so two laps of the
    same circuit disagree on total length (up to 606 m in one measured session)
    and can begin from slightly different points. Sampling naively at equal
    absolute distances then compares a corner on one lap against a straight on
    the other: measured over a season, 14 % of driver pairs produced corner
    speeds that differed by more than 40 km/h, up to 181 km/h — physically
    impossible between two F1 cars.

    So B is put onto A's basis in two steps: its distance axis is rescaled to
    A's lap length, then it is slid against A by the circular offset that best
    matches the two speed traces. A keeps its own basis, which means apex
    distances taken from A's lap stay valid against the result.

    Returns a DataFrame with ``Distance`` plus ``{channel}{suffix}`` columns.
    """
    common_channels = channels
    if common_channels is None:
        shared = [c for c in tel_a.columns if c in tel_b.columns]
        common_channels = tuple(c for c in ANALYSIS_CHANNELS if c in shared)

    res_a = resample_by_distance(tel_a, num_points=num_points, channels=common_channels)
    res_b = resample_by_distance(tel_b, num_points=num_points, channels=common_channels)

    a_lo, a_hi = float(res_a["Distance"].min()), float(res_a["Distance"].max())
    b_lo, b_hi = float(res_b["Distance"].min()), float(res_b["Distance"].max())
    grid = np.linspace(a_lo, a_hi, num_points)

    # Stretch B's lap onto A's length so the same fraction of the lap lines up.
    b_span = b_hi - b_lo
    a_span = a_hi - a_lo
    if b_span > 0 and a_span > 0:
        b_distance = (res_b["Distance"].to_numpy(dtype=float) - b_lo) / b_span * a_span + a_lo
    else:
        b_distance = res_b["Distance"].to_numpy(dtype=float)

    sampled_b = {
        channel: np.interp(grid, b_distance, res_b[channel]) for channel in common_channels
    }

    # Then slide B until its corners sit on top of A's.
    shift = 0
    if "Speed" in common_channels:
        speed_a = np.interp(grid, res_a["Distance"], res_a["Speed"])
        max_shift = int(num_points * _MAX_SHIFT_FRACTION)
        shift = _best_circular_shift(speed_a, sampled_b["Speed"], max_shift)
        if shift:
            step = a_span / max(num_points - 1, 1)
            logger.info(
                "Laps re-aligned before comparison",
                extra={"shift_samples": shift, "shift_metres": round(shift * step, 1)},
            )

    out: dict[str, np.ndarray] = {"Distance": grid}
    for channel in common_channels:
        out[f"{channel}{suffixes[0]}"] = np.interp(grid, res_a["Distance"], res_a[channel])
        out[f"{channel}{suffixes[1]}"] = (
            np.roll(sampled_b[channel], shift) if shift else sampled_b[channel]
        )
    return pd.DataFrame(out)


# Two laps of the same circuit, correctly matched, track each other closely:
# measured over 250 driver pairs, physically-plausible comparisons sat at a
# median residual of 7.6 km/h and 90 % below 12.2. Corrupted laps -- a dropout
# leaves the distance axis locally stretched, which no global shift can undo --
# sat at a median of 34.7. A 15 km/h cut caught 32 of 33 impossible pairs while
# withholding only 8 of 217 sound ones.
MAX_ALIGNMENT_RESIDUAL = 15.0


def alignment_residual(aligned: pd.DataFrame, suffixes: tuple[str, str] = ("_a", "_b")) -> float:
    """Return the RMS speed difference (km/h) between two aligned laps.

    A measure of how well :func:`align_by_distance` managed to put the two laps
    on the same piece of track. Compare against :data:`MAX_ALIGNMENT_RESIDUAL`:
    above it, the laps could not be matched and anything derived by comparing
    them sample-for-sample -- corner speeds, dominance, the time gap -- is
    describing a mismatch rather than a performance difference.

    Returns ``inf`` when there is no speed to compare.
    """
    col_a, col_b = f"Speed{suffixes[0]}", f"Speed{suffixes[1]}"
    if col_a not in aligned.columns or col_b not in aligned.columns or aligned.empty:
        return float("inf")
    va = aligned[col_a].to_numpy(dtype=float)
    vb = aligned[col_b].to_numpy(dtype=float)
    finite = np.isfinite(va) & np.isfinite(vb)
    if not finite.any():
        return float("inf")
    return float(np.sqrt(np.mean((va[finite] - vb[finite]) ** 2)))


def channel_delta(
    aligned: pd.DataFrame, channel: str, suffixes: tuple[str, str] = ("_a", "_b")
) -> pd.Series:
    """Return the per-sample delta ``channel_a - channel_b`` from an aligned frame."""
    col_a, col_b = f"{channel}{suffixes[0]}", f"{channel}{suffixes[1]}"
    if col_a not in aligned.columns or col_b not in aligned.columns:
        raise KeyError(f"Aligned frame is missing {channel!r} columns.")
    return aligned[col_a] - aligned[col_b]


def cumulative_time_delta(
    aligned: pd.DataFrame, suffixes: tuple[str, str] = ("_a", "_b")
) -> pd.Series:
    """Return the cumulative time gap (seconds) between two aligned laps.

    Integrates ``dt = dx / v`` for each driver over the shared distance grid and
    returns ``time_b - time_a`` per sample: positive means driver A is ahead
    (has taken less time to reach that point), matching the ``a - b`` sign
    convention of :func:`channel_delta`.

    Speeds are clamped to a 1 km/h floor so standing starts or data glitches
    cannot divide by zero.
    """
    col_a, col_b = f"Speed{suffixes[0]}", f"Speed{suffixes[1]}"
    if col_a not in aligned.columns or col_b not in aligned.columns:
        raise KeyError("Aligned frame is missing 'Speed' columns.")

    distance = aligned["Distance"].to_numpy(dtype=float)
    dx = np.diff(distance, prepend=distance[0])  # first sample contributes 0 s
    v_a = np.maximum(aligned[col_a].to_numpy(dtype=float), 1.0) / 3.6  # km/h -> m/s
    v_b = np.maximum(aligned[col_b].to_numpy(dtype=float), 1.0) / 3.6
    delta = np.cumsum(dx / v_b - dx / v_a)
    return pd.Series(delta, index=aligned.index, name="time_delta")


def corner_speeds(
    aligned: pd.DataFrame,
    apex_distances: list[float],
    *,
    window: float = 40.0,
    suffixes: tuple[str, str] = ("_a", "_b"),
) -> pd.DataFrame:
    """Return each driver's minimum speed through every corner.

    For each apex, takes the minimum aligned speed within ``window`` metres
    either side, so small apex-position differences between the two laps do not
    skew the comparison. Apexes whose window falls outside the shared grid are
    skipped.

    Returns:
        DataFrame with ``corner`` (1-based), ``distance``, ``speed_a`` and
        ``speed_b`` columns, one row per detected corner.
    """
    col_a, col_b = f"Speed{suffixes[0]}", f"Speed{suffixes[1]}"
    if col_a not in aligned.columns or col_b not in aligned.columns:
        raise KeyError("Aligned frame is missing 'Speed' columns.")

    distance = aligned["Distance"].to_numpy(dtype=float)
    speed_a = aligned[col_a].to_numpy(dtype=float)
    speed_b = aligned[col_b].to_numpy(dtype=float)

    rows = []
    for number, apex in enumerate(apex_distances, start=1):
        mask = (distance >= apex - window) & (distance <= apex + window)
        if not mask.any():
            continue
        rows.append(
            {
                "corner": number,
                "distance": float(apex),
                "speed_a": float(speed_a[mask].min()),
                "speed_b": float(speed_b[mask].min()),
            }
        )
    return pd.DataFrame(rows, columns=["corner", "distance", "speed_a", "speed_b"])


def minisector_dominance(
    tel_a: pd.DataFrame,
    tel_b: pd.DataFrame,
    *,
    driver_a: str,
    driver_b: str,
    num_minisectors: int = 21,
) -> pd.DataFrame:
    """Split the lap into minisectors and flag the faster driver in each.

    Speed is averaged per driver within each equal-distance minisector; the
    driver with the higher average "owns" that minisector.

    Returns:
        DataFrame with ``minisector``, ``start_distance``, ``end_distance``,
        ``mean_speed_a``, ``mean_speed_b`` and ``fastest`` (driver label).
    """
    aligned = align_by_distance(tel_a, tel_b, channels=("Speed",))
    distance = aligned["Distance"].to_numpy()
    lo, hi = distance.min(), distance.max()
    edges = np.linspace(lo, hi, num_minisectors + 1)
    # Bin index 0..num_minisectors-1 for each sample.
    idx = np.clip(np.digitize(distance, edges[1:-1]), 0, num_minisectors - 1)

    frame = aligned.assign(_ms=idx)
    rows = []
    for ms, group in frame.groupby("_ms"):
        mean_a = float(group["Speed_a"].mean())
        mean_b = float(group["Speed_b"].mean())
        rows.append(
            {
                "minisector": int(ms) + 1,
                "start_distance": float(edges[ms]),
                "end_distance": float(edges[ms + 1]),
                "mean_speed_a": mean_a,
                "mean_speed_b": mean_b,
                "fastest": driver_a if mean_a >= mean_b else driver_b,
            }
        )
    return pd.DataFrame(rows)


@dataclass(slots=True)
class ComparisonResult:
    """Structured output of a two-driver telemetry comparison.

    ``residual`` is :func:`alignment_residual` of ``aligned``: how far apart the
    two speed traces still are after alignment, in km/h RMS.
    """

    driver_a: str
    driver_b: str
    aligned: pd.DataFrame
    minisectors: pd.DataFrame
    summary: dict[str, float | str]
    residual: float

    @property
    def matched(self) -> bool:
        """Whether the two laps could be put on the same points of track.

        When ``False``, everything that compares them sample-for-sample -- the
        time delta, speed delta, corner speeds and minisectors -- describes the
        mismatch rather than the drivers, and should not be shown. Each lap's own
        trace is unaffected. ``True`` means the traces line up; it does not by
        itself make every derived number accurate.
        """
        return self.residual <= MAX_ALIGNMENT_RESIDUAL


def compare_drivers(
    tel_a: pd.DataFrame,
    tel_b: pd.DataFrame,
    *,
    driver_a: str,
    driver_b: str,
    num_points: int = 1000,
    num_minisectors: int = 21,
) -> ComparisonResult:
    """Run the full comparison between two laps and summarise the result.

    Check :attr:`ComparisonResult.matched` before using anything that compares
    the two laps sample-for-sample: a telemetry dropout can leave one lap
    impossible to line up with the other.
    """
    aligned = align_by_distance(tel_a, tel_b, num_points=num_points)
    minisectors = minisector_dominance(
        tel_a, tel_b, driver_a=driver_a, driver_b=driver_b, num_minisectors=num_minisectors
    )

    counts = minisectors["fastest"].value_counts()
    summary: dict[str, float | str] = {
        "minisectors_a": int(counts.get(driver_a, 0)),
        "minisectors_b": int(counts.get(driver_b, 0)),
        "dominant_driver": (
            driver_a if counts.get(driver_a, 0) >= counts.get(driver_b, 0) else driver_b
        ),
    }
    if "Speed_a" in aligned.columns:
        summary["max_speed_a"] = float(aligned["Speed_a"].max())
        summary["max_speed_b"] = float(aligned["Speed_b"].max())

    residual = alignment_residual(aligned)
    logger.info(
        "Drivers compared",
        extra={
            "driver_a": driver_a,
            "driver_b": driver_b,
            "dominant": summary["dominant_driver"],
            "residual_kmh": round(residual, 1),
        },
    )
    return ComparisonResult(
        driver_a=driver_a,
        driver_b=driver_b,
        aligned=aligned,
        minisectors=minisectors,
        summary=summary,
        residual=residual,
    )
