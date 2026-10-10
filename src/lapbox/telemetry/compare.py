"""Two-lap telemetry comparison.

Compares two drivers' laps by aligning their telemetry onto a common distance
axis, then computing per-channel deltas, a cumulative time gap, who took less
time through each minisector, and the time each took between the lap's speed peaks.

All functions are pure and operate on telemetry DataFrames, so they compose with
:mod:`lapbox.telemetry.lap` and are unit-tested without any network.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from lapbox.telemetry.lap import ANALYSIS_CHANNELS, _monotonic_distance, resample_by_distance

logger = logging.getLogger(__name__)


# A lap's distance axis is integrated from speed, so two drivers' traces drift
# against each other and can start from slightly different track points. Search
# no further than this fraction of a lap for the offset between them; the worst
# real case measured across a season was 2.6 %.
_MAX_SHIFT_FRACTION = 0.05

# The offset between two laps is searched on its own fine grid and refined to a
# fraction of a sample, so it doesn't depend on the grid the laps are compared on.
# Shifting by whole grid samples instead flipped minisector owners in 102 of 945
# minisectors between 500 and 4,000 points (2023 Bahrain qualifying, 45 pairs);
# with this, 17.
_SHIFT_POINTS = 10000
_COARSE_STEP = 10

# Minisectors are measured on their own alignment, whatever grid the comparison uses.
_MINISECTOR_POINTS = 4000


def _match_error(ref: np.ndarray, other: np.ndarray, shift: int) -> float:
    return float(np.mean((np.roll(other, shift) - ref) ** 2))


def _best_circular_shift(ref: np.ndarray, other: np.ndarray, max_shift: int) -> float:
    """Return the circular shift of ``other`` that best matches ``ref``, in samples.

    A lap is a closed loop sampled over exactly one circuit of it, so rolling
    the trace is the physically correct way to slide one lap against another.
    Matching is on speed, which has sharp, unmistakable corner minima. Every
    ``_COARSE_STEP``-th shift is tried first, then every shift around the best
    one, then a parabola through the match error at the best shift and its two
    neighbours gives the fraction of a sample.
    """
    if max_shift < 1 or len(ref) != len(other) or len(ref) < 3:
        return 0.0
    coarse = range(-max_shift, max_shift + 1, _COARSE_STEP)
    best = min(coarse, key=lambda s: _match_error(ref, other, s))
    around = range(max(-max_shift, best - _COARSE_STEP), min(max_shift, best + _COARSE_STEP) + 1)
    best = min(around, key=lambda s: _match_error(ref, other, s))
    if -max_shift < best < max_shift:
        before, at, after = (_match_error(ref, other, best + d) for d in (-1, 0, 1))
        curvature = before - 2 * at + after
        if curvature > 0:
            return best + 0.5 * (before - after) / curvature
    return float(best)


def _shift_metres(
    tel_a: pd.DataFrame, tel_b: pd.DataFrame, a_lo: float, a_span: float, b_lo: float, b_span: float
) -> float:
    """How far to slide B (already stretched to A's length) to match A's speed trace."""
    fine_a = resample_by_distance(tel_a, num_points=_SHIFT_POINTS, channels=("Speed",))
    fine_b = resample_by_distance(tel_b, num_points=_SHIFT_POINTS, channels=("Speed",))
    fine = np.linspace(a_lo, a_lo + a_span, _SHIFT_POINTS)
    b_distance = (fine_b["Distance"].to_numpy(dtype=float) - b_lo) / b_span * a_span + a_lo
    speed_a = np.interp(fine, fine_a["Distance"], fine_a["Speed"])
    speed_b = np.interp(fine, b_distance, fine_b["Speed"])
    max_shift = int(_SHIFT_POINTS * _MAX_SHIFT_FRACTION)
    return _best_circular_shift(speed_a, speed_b, max_shift) * a_span / (_SHIFT_POINTS - 1)


def _clock(tel: pd.DataFrame) -> tuple[np.ndarray, np.ndarray] | None:
    """A lap's own timing: seconds since the start of the lap, against distance.

    Read from ``TimeSeconds`` when present, else FastF1's ``Time`` (a Timedelta
    from the lap's start). ``None`` when the lap carries neither.
    """
    if "TimeSeconds" in tel.columns:
        seconds = pd.to_numeric(tel["TimeSeconds"], errors="coerce")
    elif "Time" in tel.columns and pd.api.types.is_timedelta64_dtype(tel["Time"]):
        seconds = tel["Time"].dt.total_seconds()
    else:
        return None
    frame = pd.DataFrame({"Distance": tel["Distance"], "seconds": seconds}).dropna()
    if len(frame) < 2:
        return None
    ordered = _monotonic_distance(frame)
    return ordered["Distance"].to_numpy(dtype=float), ordered["seconds"].to_numpy(dtype=float)


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
    matches the two speed traces. The offset is found on a fine grid of its own,
    to a fraction of a metre, so it doesn't depend on ``num_points``. A keeps its
    own basis, which means apex distances taken from A's lap stay valid against
    the result.

    When both laps carry their timing (FastF1's ``Time``, or ``TimeSeconds``),
    ``Time{suffix}`` columns give each driver's own clock at every point: seconds
    since the start of A's lap for A, and B's clock at the matching place for B
    (a place before B's line belongs to the end of B's lap, one lap time
    earlier). :func:`cumulative_time_delta` and :func:`minisector_dominance` read
    them in preference to integrating speed.

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
    stretched = b_span > 0 and a_span > 0
    if stretched:
        b_distance = (res_b["Distance"].to_numpy(dtype=float) - b_lo) / b_span * a_span + a_lo
    else:
        b_distance = res_b["Distance"].to_numpy(dtype=float)

    # Then slide B until its corners sit on top of A's.
    shift = 0.0
    if "Speed" in common_channels and stretched:
        shift = _shift_metres(tel_a, tel_b, a_lo, a_span, b_lo, b_span)
        if shift:
            logger.info(
                "Laps re-aligned before comparison", extra={"shift_metres": round(shift, 1)}
            )
    # B at (x - shift), wrapped around the lap.
    positions = (grid - a_lo - shift) % a_span + a_lo if shift else grid

    out: dict[str, np.ndarray] = {"Distance": grid}
    for channel in common_channels:
        out[f"{channel}{suffixes[0]}"] = np.interp(grid, res_a["Distance"], res_a[channel])
        out[f"{channel}{suffixes[1]}"] = np.interp(positions, b_distance, res_b[channel])

    clocks = _clock(tel_a), _clock(tel_b)
    if stretched and clocks[0] is not None and clocks[1] is not None:
        (dist_a, secs_a), (dist_b, secs_b) = clocks
        lap_b = float(np.interp(b_hi, dist_b, secs_b) - np.interp(b_lo, dist_b, secs_b))
        into_b = (grid - a_lo - shift) / a_span * b_span  # metres into B's lap
        laps = np.floor(into_b / b_span)
        out[f"Time{suffixes[0]}"] = np.interp(grid, dist_a, secs_a)
        out[f"Time{suffixes[1]}"] = (
            np.interp(into_b - laps * b_span + b_lo, dist_b, secs_b) + laps * lap_b
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

    Returns ``time_b - time_a`` per sample: positive means driver A is ahead
    (has taken less time to reach that point), matching the ``a - b`` sign
    convention of :func:`channel_delta`. It starts at 0.

    It integrates ``dt = dx / v`` for each driver over the shared grid (speeds
    clamped to a 1 km/h floor), which gives the shape of each lap. When the laps
    carry their own clocks (``Time{suffix}`` columns, which :func:`align_by_distance`
    adds), each driver's integral is scaled to add up to their measured lap time,
    so the gap ends exactly at the lap-time gap. On 2023 Bahrain, 2025 Monza,
    2025 Silverstone and 2026 Melbourne qualifying (45 pairs each) it was then within
    a median of 29 to 42 ms (at most 143 ms) of the official gap at the sector lines. Integrating
    speed alone ended a median of 92 to 113 ms (up to 360 ms) from the official gap
    and on the wrong side of it in 2 to 9 of 45 pairs; reading the clocks
    point by point was exact at the line but rougher in between (up to 199 ms at
    the sector lines), because a few metres of misplacement in a slow corner is a
    large time.
    """
    t_a, t_b = _timed_samples(aligned, suffixes)
    return pd.Series(np.cumsum(t_b - t_a), index=aligned.index, name="time_delta")


def _sample_times(
    aligned: pd.DataFrame, suffixes: tuple[str, str] = ("_a", "_b")
) -> tuple[np.ndarray, np.ndarray]:
    """Seconds each driver takes over each sample's stretch of the shared grid (``dx / v``)."""
    col_a, col_b = f"Speed{suffixes[0]}", f"Speed{suffixes[1]}"
    if col_a not in aligned.columns or col_b not in aligned.columns:
        raise KeyError("Aligned frame is missing 'Speed' columns.")

    distance = aligned["Distance"].to_numpy(dtype=float)
    dx = np.diff(distance, prepend=distance[0])  # first sample contributes 0 s
    v_a = np.maximum(aligned[col_a].to_numpy(dtype=float), 1.0) / 3.6  # km/h -> m/s
    v_b = np.maximum(aligned[col_b].to_numpy(dtype=float), 1.0) / 3.6
    return dx / v_a, dx / v_b


def _timed_samples(
    aligned: pd.DataFrame, suffixes: tuple[str, str] = ("_a", "_b")
) -> tuple[np.ndarray, np.ndarray]:
    """:func:`_sample_times`, each lap scaled to its own clock's lap time when it has one."""
    t_a, t_b = _sample_times(aligned, suffixes)
    time_a, time_b = f"Time{suffixes[0]}", f"Time{suffixes[1]}"
    if time_a in aligned.columns and time_b in aligned.columns and t_a.sum() > 0 and t_b.sum() > 0:
        lap_a = float(aligned[time_a].iloc[-1] - aligned[time_a].iloc[0])
        lap_b = float(aligned[time_b].iloc[-1] - aligned[time_b].iloc[0])
        t_a, t_b = t_a * (lap_a / t_a.sum()), t_b * (lap_b / t_b.sum())
    return t_a, t_b


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
    """Split the lap into equal-distance minisectors and flag the faster driver in each.

    A minisector goes to the driver who spends less time in it: ``dx / v`` summed
    over its stretch, each lap scaled to its own clock's lap time when the laps
    carry their timing, exactly as :func:`cumulative_time_delta` builds the gap. So
    the minisector times add up to the time gap at the line. Mean speeds are kept
    for reference, but they don't decide: a minisector holding a slow corner and a
    straight can have the higher mean speed and still take longer.

    Many minisectors are close: on 2023 Bahrain qualifying 29 % of 945 were
    decided by less than 20 ms. Read a margin of a few hundredths as too close
    to call.

    Returns:
        DataFrame with ``minisector``, ``start_distance``, ``end_distance``,
        ``mean_speed_a``, ``mean_speed_b``, ``time_a``, ``time_b`` (seconds) and
        ``fastest`` (driver label; ties go to driver A).
    """
    aligned = align_by_distance(tel_a, tel_b, num_points=_MINISECTOR_POINTS, channels=("Speed",))
    distance = aligned["Distance"].to_numpy()
    lo, hi = distance.min(), distance.max()
    edges = np.linspace(lo, hi, num_minisectors + 1)
    # Bin index 0..num_minisectors-1 for each sample.
    idx = np.clip(np.digitize(distance, edges[1:-1]), 0, num_minisectors - 1)
    t_a, t_b = _timed_samples(aligned)
    times = {
        ms: (float(t_a[idx == ms].sum()), float(t_b[idx == ms].sum()))
        for ms in range(num_minisectors)
    }

    frame = aligned.assign(_ms=idx)
    rows = []
    for ms, group in frame.groupby("_ms"):
        time_a, time_b = times[ms]
        rows.append(
            {
                "minisector": int(ms) + 1,
                "start_distance": float(edges[ms]),
                "end_distance": float(edges[ms + 1]),
                "mean_speed_a": float(group["Speed_a"].mean()),
                "mean_speed_b": float(group["Speed_b"].mean()),
                "time_a": time_a,
                "time_b": time_b,
                "fastest": driver_a if time_a <= time_b else driver_b,
            }
        )
    return pd.DataFrame(rows)


def segment_gaps(
    aligned: pd.DataFrame,
    *,
    prominence: float = 20.0,
    suffixes: tuple[str, str] = ("_a", "_b"),
) -> pd.DataFrame:
    """Split the lap at its speed peaks and return each driver's time between them.

    A segment runs from one speed peak to the next: from the end of one straight,
    through the corners after it, to the end of the next. The peaks are those of
    the two laps' mean speed that stand at least ``prominence`` km/h above their
    surroundings, so swapping the drivers keeps the same segments. Times are built
    exactly as :func:`cumulative_time_delta` builds the gap, so the segments'
    ``time_delta`` add up to its value at the line.

    Where in the lap time changes hands depends on where each lap is placed, and a
    few metres of misplacement cost the most where the car is slowest: at an apex
    they move tenths of a second from one side of the corner to the other, at the
    end of a straight almost nothing. Hence segments that end at the peaks. The
    same lap compared with itself, with 5 m of distance error at its slowest
    corner, has the gap curve gaining 0.10 s and losing 0.17 s around that corner;
    no segment moves by more than 49 ms. On six qualifying sessions (246 matched
    pairs) the curve's largest rise and fall, which start and end wherever it
    turns, were a median 113 ms (p90 314 ms) from the same stretch rebuilt from the
    cars' positions; the largest gaining and losing segments were 65 ms away,
    about the accuracy of that check itself. Peaks of 10 to 50 km/h gave the same.

    Returns:
        DataFrame with ``segment`` (1-based), ``start_distance``, ``end_distance``,
        ``time_a``, ``time_b`` (seconds) and ``time_delta`` (``time_b - time_a``:
        positive means driver A gained time in the segment).
    """
    t_a, t_b = _timed_samples(aligned, suffixes)
    distance = aligned["Distance"].to_numpy(dtype=float)
    mean_speed = (
        aligned[f"Speed{suffixes[0]}"].to_numpy(dtype=float)
        + aligned[f"Speed{suffixes[1]}"].to_numpy(dtype=float)
    ) / 2
    peaks, _ = find_peaks(mean_speed, prominence=prominence)
    ends = np.unique(np.concatenate([[0], peaks, [len(distance) - 1]]))
    # Sample i holds the time from distance[i - 1] to distance[i].
    time_a, time_b = np.diff(np.cumsum(t_a)[ends]), np.diff(np.cumsum(t_b)[ends])
    return pd.DataFrame(
        {
            "segment": np.arange(1, len(ends)),
            "start_distance": distance[ends[:-1]],
            "end_distance": distance[ends[1:]],
            "time_a": time_a,
            "time_b": time_b,
            "time_delta": time_b - time_a,
        }
    )


@dataclass(slots=True)
class ComparisonResult:
    """Structured output of a two-driver telemetry comparison.

    ``residual`` is :func:`alignment_residual` of ``aligned``: how far apart the
    two speed traces still are after alignment, in km/h RMS. ``segments`` is
    :func:`segment_gaps` of ``aligned``: each driver's time between the lap's
    speed peaks.
    """

    driver_a: str
    driver_b: str
    aligned: pd.DataFrame
    minisectors: pd.DataFrame
    summary: dict[str, float | str]
    residual: float
    segments: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def matched(self) -> bool:
        """Whether the two laps could be put on the same points of track.

        When ``False``, everything that compares them sample-for-sample -- the
        time delta, speed delta, corner speeds, minisectors and segments -- describes the
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

    ``num_points`` sets the grid of ``aligned``; the minisectors are decided on
    their own finer alignment (see :func:`minisector_dominance`), so a coarse
    display grid doesn't make them noisier. The segments (:func:`segment_gaps`) use
    ``aligned``: on 2023 Bahrain qualifying the largest gaining and losing segment
    were the same at 500 and 4,000 points in 42 and 38 of 45 pairs, their times a
    median 5 ms apart.

    ``summary["more_minisectors"]`` is the driver who won more minisectors: a
    count, not who was quicker. A driver can lose most minisectors and still
    take pole by gaining big in a few (2023 Bahrain qualifying: LEC won 12 of 21,
    VER was 0.292 s faster). For who was quicker, read the lap times or
    :func:`cumulative_time_delta` at the line.
    """
    aligned = align_by_distance(tel_a, tel_b, num_points=num_points)
    minisectors = minisector_dominance(
        tel_a, tel_b, driver_a=driver_a, driver_b=driver_b, num_minisectors=num_minisectors
    )

    counts = minisectors["fastest"].value_counts()
    summary: dict[str, float | str] = {
        "minisectors_a": int(counts.get(driver_a, 0)),
        "minisectors_b": int(counts.get(driver_b, 0)),
        "more_minisectors": (
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
            "more_minisectors": summary["more_minisectors"],
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
        segments=segment_gaps(aligned),
    )
