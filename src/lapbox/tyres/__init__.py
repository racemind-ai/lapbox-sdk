"""Tyres: per-compound degradation, and degradation stint by stint.

Takes FastF1's laps as they come (``session.laps``); the per-compound model also
reads LapBox's engineered lap columns.
"""

from lapbox.tyres.degradation import CompoundDegradation, TyreDegradationModel
from lapbox.tyres.stints import compound_summary, stint_degradation

__all__ = [
    # Per-compound regression
    "TyreDegradationModel",
    "CompoundDegradation",
    # Stint by stint
    "stint_degradation",
    "compound_summary",
]
