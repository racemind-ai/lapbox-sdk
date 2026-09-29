"""Reading practice: runs, qualifying simulations and long-run pace.

Takes FastF1's laps as they come (``session.laps``).
"""

from lapbox.practice.runs import (
    FUEL_EFFECT_S_PER_LAP,
    MAX_QUALI_SIM,
    MIN_LONG_RUN,
    QUALI_SIM_CUTOFF,
    RUN_OUTLIER_CUTOFF,
    LongRunPace,
    Run,
    classify_run,
    detect_runs,
    long_run_pace,
    longest_run_length,
    representative_mask,
    session_runs,
)

__all__ = [
    "Run",
    "LongRunPace",
    "detect_runs",
    "classify_run",
    "long_run_pace",
    "session_runs",
    "longest_run_length",
    "representative_mask",
    "MIN_LONG_RUN",
    "MAX_QUALI_SIM",
    "QUALI_SIM_CUTOFF",
    "RUN_OUTLIER_CUTOFF",
    "FUEL_EFFECT_S_PER_LAP",
]
