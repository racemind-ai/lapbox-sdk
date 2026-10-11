"""Telemetry analysis: one lap on its own, and two laps matched by track position.

Every function takes FastF1 telemetry as it comes from ``lap.get_telemetry()``.
"""

from lapbox.telemetry.comparison import (
    MAX_ALIGNMENT_RESIDUAL,
    Comparison,
    align,
    alignment_residual,
    channel_delta,
    compare,
    corner_speeds,
    delta_time,
    minisector_dominance,
    segment_gaps,
)
from lapbox.telemetry.lap import (
    ANALYSIS_CHANNELS,
    LapAnalysis,
    TelemetryAnalyzer,
    Zone,
    channel_summary,
    corner_analysis,
    detect_braking_zones,
    detect_corners,
    detect_frozen_stretches,
    detect_full_throttle_zones,
    gforces,
    resample,
)

__all__ = [
    # One lap
    "ANALYSIS_CHANNELS",
    "Zone",
    "LapAnalysis",
    "TelemetryAnalyzer",
    "resample",
    "detect_braking_zones",
    "detect_full_throttle_zones",
    "detect_corners",
    "corner_analysis",
    "gforces",
    "channel_summary",
    "detect_frozen_stretches",
    # Two laps
    "MAX_ALIGNMENT_RESIDUAL",
    "Comparison",
    "compare",
    "align",
    "alignment_residual",
    "channel_delta",
    "delta_time",
    "corner_speeds",
    "minisector_dominance",
    "segment_gaps",
]
