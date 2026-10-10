"""Telemetry analysis: one lap on its own, and two laps matched by track position.

Every function takes FastF1 telemetry as it comes from ``lap.get_telemetry()``.
"""

from lapbox.telemetry.compare import (
    MAX_ALIGNMENT_RESIDUAL,
    ComparisonResult,
    align_by_distance,
    alignment_residual,
    channel_delta,
    compare_drivers,
    corner_speeds,
    cumulative_time_delta,
    minisector_dominance,
    segment_gaps,
)
from lapbox.telemetry.lap import (
    ANALYSIS_CHANNELS,
    LapAnalysis,
    TelemetryAnalyzer,
    Zone,
    channel_summary,
    compute_gforces,
    corner_analysis,
    detect_braking_zones,
    detect_corners,
    detect_full_throttle_zones,
    resample_by_distance,
)

__all__ = [
    # One lap
    "ANALYSIS_CHANNELS",
    "Zone",
    "LapAnalysis",
    "TelemetryAnalyzer",
    "resample_by_distance",
    "detect_braking_zones",
    "detect_full_throttle_zones",
    "detect_corners",
    "corner_analysis",
    "compute_gforces",
    "channel_summary",
    # Two laps
    "MAX_ALIGNMENT_RESIDUAL",
    "ComparisonResult",
    "compare_drivers",
    "align_by_distance",
    "alignment_residual",
    "channel_delta",
    "cumulative_time_delta",
    "corner_speeds",
    "minisector_dominance",
    "segment_gaps",
]
