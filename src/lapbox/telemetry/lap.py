"""Single-lap telemetry analysis.

Pure, side-effect-free analysis over one lap's car telemetry, as FastF1 returns it
from ``lap.get_telemetry()``: FastF1's own column names (``Distance``, ``Speed``,
``Throttle``, ``Brake``, ``nGear``, ``RPM``, ``DRS``, ``X``, ``Y``), extra columns
ignored. Everything here operates on and returns plain ``pandas``/``numpy``
objects, and :mod:`lapbox.telemetry.compare` builds on it.

Capabilities:

* Speed / throttle / brake / RPM / gear traces vs distance (channel access +
  resampling onto a uniform distance grid for fair comparison).
* Braking-zone detection.
* Full-throttle (WOT) section detection.
* Corner (apex) detection from prominent speed minima.
* Per-channel summary statistics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)

# Canonical analysis channels, all numeric and interpolatable over distance.
ANALYSIS_CHANNELS: tuple[str, ...] = ("Speed", "Throttle", "Brake", "RPM", "nGear", "DRS")


@dataclass(frozen=True, slots=True)
class Zone:
    """A contiguous section of a lap between two distance points."""

    start_distance: float
    end_distance: float
    entry_speed: float
    min_speed: float
    max_speed: float

    @property
    def length(self) -> float:
        return self.end_distance - self.start_distance


def _require_distance(tel: pd.DataFrame) -> None:
    if "Distance" not in tel.columns:
        raise ValueError("Telemetry must contain a 'Distance' column (call add_distance()).")


def _monotonic_distance(tel: pd.DataFrame) -> pd.DataFrame:
    """Return telemetry sorted by strictly-increasing distance (dups dropped)."""
    _require_distance(tel)
    ordered = tel.sort_values("Distance")
    return ordered[~ordered["Distance"].duplicated(keep="first")].reset_index(drop=True)


def resample_by_distance(
    tel: pd.DataFrame,
    num_points: int = 1000,
    channels: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Interpolate telemetry channels onto a uniform distance grid.

    Aligning laps to a common distance axis is the prerequisite for fair
    driver-vs-driver comparison and for smooth plotting.

    Args:
        tel: Telemetry with a ``Distance`` column.
        num_points: Number of evenly-spaced samples across the lap.
        channels: Channels to interpolate; defaults to those present among
            :data:`ANALYSIS_CHANNELS`.

    Returns:
        A DataFrame with ``Distance`` plus one column per interpolated channel.
    """
    ordered = _monotonic_distance(tel)
    if len(ordered) < 2:
        raise ValueError("Need at least two distinct distance samples to resample.")

    chosen = channels or tuple(c for c in ANALYSIS_CHANNELS if c in ordered.columns)
    dist = ordered["Distance"].to_numpy(dtype=float)
    grid = np.linspace(dist.min(), dist.max(), num_points)

    out: dict[str, np.ndarray] = {"Distance": grid}
    for channel in chosen:
        out[channel] = np.interp(grid, dist, ordered[channel].to_numpy(dtype=float))
    return pd.DataFrame(out)


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return (start_idx, end_idx) inclusive index ranges where ``mask`` is True."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _zones_from_mask(tel: pd.DataFrame, mask: np.ndarray) -> list[Zone]:
    distance = tel["Distance"].to_numpy(dtype=float)
    speed = (
        tel["Speed"].to_numpy(dtype=float) if "Speed" in tel.columns else np.full(len(tel), np.nan)
    )
    zones: list[Zone] = []
    for start, end in _contiguous_runs(mask):
        seg_speed = speed[start : end + 1]
        zones.append(
            Zone(
                start_distance=float(distance[start]),
                end_distance=float(distance[end]),
                entry_speed=float(seg_speed[0]),
                min_speed=float(np.nanmin(seg_speed)),
                max_speed=float(np.nanmax(seg_speed)),
            )
        )
    return zones


def _zones_to_frame(zones: list[Zone]) -> pd.DataFrame:
    columns = ["start_distance", "end_distance", "length", "entry_speed", "min_speed", "max_speed"]
    if not zones:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(
        [
            {
                "start_distance": z.start_distance,
                "end_distance": z.end_distance,
                "length": z.length,
                "entry_speed": z.entry_speed,
                "min_speed": z.min_speed,
                "max_speed": z.max_speed,
            }
            for z in zones
        ],
        columns=columns,
    )


def detect_braking_zones(tel: pd.DataFrame, *, min_length: float = 0.0) -> pd.DataFrame:
    """Detect braking zones (contiguous sections where the brake is applied).

    Handles both boolean and 0–100 % ``Brake`` channels.

    Args:
        tel: Telemetry with ``Distance``, ``Brake`` (and ideally ``Speed``).
        min_length: Discard zones shorter than this distance (metres).
    """
    _require_distance(tel)
    if "Brake" not in tel.columns:
        return _zones_to_frame([])
    brake = tel["Brake"].to_numpy()
    mask = (brake.astype(float) > 0) if brake.dtype != bool else brake
    zones = [z for z in _zones_from_mask(tel, mask) if z.length >= min_length]
    return _zones_to_frame(zones)


def detect_full_throttle_zones(
    tel: pd.DataFrame, *, threshold: float = 99.0, min_length: float = 0.0
) -> pd.DataFrame:
    """Detect wide-open-throttle sections (``Throttle`` >= ``threshold``)."""
    _require_distance(tel)
    if "Throttle" not in tel.columns:
        return _zones_to_frame([])
    mask = tel["Throttle"].to_numpy(dtype=float) >= threshold
    zones = [z for z in _zones_from_mask(tel, mask) if z.length >= min_length]
    return _zones_to_frame(zones)


def detect_corners(
    tel: pd.DataFrame,
    *,
    prominence: float = 12.0,
    min_separation: float = 60.0,
) -> pd.DataFrame:
    """Detect corner apexes as *prominent* local speed minima.

    A corner is somewhere the car is genuinely slower than on either side of,
    so apexes are found by the **prominence** of each speed minimum — how far
    the speed climbs back up before the lap dips lower elsewhere — rather than
    by an absolute speed ceiling.

    Gating on a fraction of top speed, as this used to, cannot work: at every
    real circuit ``0.6 × vmax`` lands at 173–209 km/h, *below* the 210 km/h
    ``fast`` band of :func:`corner_analysis`, so the two conditions were
    mutually exclusive and high-speed corners (Copse, Blanchimont, 130R) were
    never detected at all. Prominence has no such ceiling: a 300 km/h kink
    between two 320 km/h straights is as detectable as a hairpin.

    Note that the count will not match a circuit's official FIA corner
    numbering, which includes flat-out kinks that produce no speed minimum.
    What matters for driver and car comparison is that the *same* corners are
    found on every lap of a given circuit, which speed-minimum detection gives.

    Args:
        tel: Telemetry with ``Distance`` and ``Speed``.
        prominence: Minimum speed drop (km/h) into a minimum from the higher of
            its two surrounding ridges. Filters out ripples along a straight.
        min_separation: Minimum spacing between apexes (metres); the slower
            minimum wins inside that window.

    Returns:
        DataFrame with ``apex_distance`` and ``apex_speed`` per corner.
    """
    ordered = _monotonic_distance(tel)
    if "Speed" not in ordered.columns or len(ordered) < 3:
        return pd.DataFrame(columns=["apex_distance", "apex_speed"])

    distance = ordered["Distance"].to_numpy(dtype=float)
    speed = ordered["Speed"].to_numpy(dtype=float)

    # Non-finite speed samples would poison the prominence walk; dropping them
    # matches the old behaviour, where a NaN never compared as a minimum.
    finite = np.isfinite(speed) & np.isfinite(distance)
    distance, speed = distance[finite], speed[finite]
    if len(speed) < 3:
        return pd.DataFrame(columns=["apex_distance", "apex_speed"])

    # ``find_peaks`` counts in samples, not metres, and telemetry is sampled in
    # time, so convert via the median distance step.
    steps = np.diff(distance)
    median_step = float(np.median(steps)) if len(steps) else 0.0
    separation = max(1, round(min_separation / median_step)) if median_step > 0 else 1

    # Apexes are peaks of the inverted trace. Inverting also means the distance
    # filter keeps the *tallest* peak — i.e. the slowest minimum — in each
    # window, which is the behaviour we want when two minima share a corner.
    apex_idx, _ = find_peaks(-speed, prominence=prominence, distance=separation)

    return pd.DataFrame({"apex_distance": distance[apex_idx], "apex_speed": speed[apex_idx]})


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average with edge padding (no phase shift, no edge dip)."""
    if window < 3 or len(values) < window:
        return values
    if window % 2 == 0:
        window += 1
    padded = np.pad(values, window // 2, mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def compute_gforces(
    distance: np.ndarray | list[float],
    speed_kmh: np.ndarray | list[float],
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    *,
    smooth_window: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Longitudinal and lateral acceleration (in g) along a lap.

    Derived from speed and the racing-line geometry on a distance grid — no time
    channel is needed:

    * ``a_long = v · dv/ds`` (v in m/s, s in metres): positive = accelerating,
      negative = braking.
    * ``a_lat  = v² · dθ/ds`` where ``θ = atan2(dy, dx)`` is the heading angle.
      Heading is scale-invariant, so this is correct regardless of the X/Y unit
      (FastF1's position units cancel), and ``dθ/ds`` is the path curvature.

    Both are returned in g (÷ 9.81). Inputs must share the same (ideally uniform)
    distance grid; arrays shorter than 3 or mismatched return zeros.

    Position telemetry is noisy and curvature is a *second* derivative of it, so
    speed and heading are lightly smoothed (``smooth_window`` samples) before
    differentiating. Without it a handful of samples spike far beyond what a car
    can physically pull. Set ``smooth_window=0`` to disable.
    """
    dist = np.asarray(distance, dtype=float)
    v = np.asarray(speed_kmh, dtype=float) / 3.6  # km/h -> m/s
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    n = len(dist)
    if n < 3 or not (len(v) == len(xs) == len(ys) == n):
        z = np.zeros(n, dtype=float)
        return z, z.copy()

    g = 9.81
    v = _smooth(v, smooth_window)
    a_long = v * np.gradient(v, dist) / g
    # Heading angle (scale-free in x/y magnitude); its rate over distance is curvature.
    theta = np.unwrap(np.arctan2(np.gradient(ys), np.gradient(xs)))
    a_lat = v**2 * np.gradient(_smooth(theta, smooth_window), dist) / g
    return np.nan_to_num(a_long), np.nan_to_num(a_lat)


def corner_analysis(
    distance: np.ndarray | list[float],
    speed_kmh: np.ndarray | list[float],
    apex_distances: list[float],
    *,
    approach_m: float = 80.0,
    exit_m: float = 80.0,
    apex_window_m: float = 25.0,
) -> list[dict]:
    """Per-corner entry / minimum / exit speed and a speed-class label.

    For each detected apex: entry speed is sampled ``approach_m`` before it, exit
    speed ``exit_m`` after, and the minimum within ``±apex_window_m`` is the
    corner speed. Corners are classed slow/medium/fast by that minimum (rough F1
    bands: <130, 130–210, >210 km/h) so the UI can filter by corner type.
    """
    dist = np.asarray(distance, dtype=float)
    v = np.asarray(speed_kmh, dtype=float)
    if len(dist) < 3 or len(v) != len(dist):
        return []
    out: list[dict] = []
    for i, apex in enumerate(sorted(float(a) for a in apex_distances), start=1):
        entry = float(np.interp(apex - approach_m, dist, v))
        exit_speed = float(np.interp(apex + exit_m, dist, v))
        near = np.abs(dist - apex) <= apex_window_m
        min_speed = float(np.min(v[near])) if near.any() else float(np.interp(apex, dist, v))
        speed_class = "slow" if min_speed < 130 else "medium" if min_speed < 210 else "fast"
        out.append(
            {
                "corner": i,
                "apex_distance": round(apex, 1),
                "entry_speed": round(entry, 1),
                "min_speed": round(min_speed, 1),
                "exit_speed": round(exit_speed, 1),
                "speed_class": speed_class,
            }
        )
    return out


def channel_summary(tel: pd.DataFrame) -> dict[str, float]:
    """Return summary statistics across a lap's telemetry channels."""
    _require_distance(tel)
    summary: dict[str, float] = {"lap_distance": float(tel["Distance"].max())}
    if "Speed" in tel.columns:
        summary["max_speed"] = float(tel["Speed"].max())
        summary["mean_speed"] = float(tel["Speed"].mean())
        summary["min_speed"] = float(tel["Speed"].min())
    if "RPM" in tel.columns:
        summary["max_rpm"] = float(tel["RPM"].max())
    if "Throttle" in tel.columns:
        summary["full_throttle_pct"] = float((tel["Throttle"] >= 99).mean() * 100)
    if "Brake" in tel.columns:
        brake = tel["Brake"]
        active = brake > 0 if brake.dtype != bool else brake
        summary["braking_pct"] = float(active.mean() * 100)
    if "nGear" in tel.columns:
        summary["max_gear"] = float(tel["nGear"].max())
    return summary


@dataclass(slots=True)
class LapAnalysis:
    """Bundled analysis of a single lap's telemetry."""

    driver: str
    summary: dict[str, float]
    braking_zones: pd.DataFrame
    full_throttle_zones: pd.DataFrame
    corners: pd.DataFrame


class TelemetryAnalyzer:
    """Convenience wrapper computing a full :class:`LapAnalysis` for one lap."""

    def __init__(self, *, corner_prominence: float = 12.0) -> None:
        self._corner_prominence = corner_prominence

    def analyze(self, tel: pd.DataFrame, *, driver: str = "") -> LapAnalysis:
        analysis = LapAnalysis(
            driver=driver,
            summary=channel_summary(tel),
            braking_zones=detect_braking_zones(tel),
            full_throttle_zones=detect_full_throttle_zones(tel),
            corners=detect_corners(tel, prominence=self._corner_prominence),
        )
        logger.info(
            "Lap analysed",
            extra={
                "driver": driver,
                "corners": len(analysis.corners),
                "braking_zones": len(analysis.braking_zones),
            },
        )
        return analysis
