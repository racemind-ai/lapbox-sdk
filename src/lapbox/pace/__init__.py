"""Race pace: lap cleaning, representative pace, fuel-corrected pace, clean air,
consistency and the ideal lap.

Takes FastF1's laps as they come (``session.laps``).
"""

from lapbox.pace.cleaning import CleaningResult, LapCleaningConfig, LapCleaningPipeline
from lapbox.pace.race import (
    DIRTY_AIR_S,
    REPRESENTATIVE,
    BestSector,
    CleanAirSplit,
    DriverConsistency,
    IdealLap,
    biggest_loss,
    clean_air_ranking,
    clean_air_split,
    consistency_ranking,
    driver_consistency,
    fuel_correct,
    ideal_lap,
    true_pace_ranking,
)
from lapbox.pace.representative import (
    BASE_TIME_CLEAN_PCT,
    MIN_CLEAN_LAPS,
    representative_base_time,
)

__all__ = [
    # Cleaning
    "LapCleaningPipeline",
    "LapCleaningConfig",
    "CleaningResult",
    # A circuit's normal pace
    "representative_base_time",
    "BASE_TIME_CLEAN_PCT",
    "MIN_CLEAN_LAPS",
    # Race pace
    "REPRESENTATIVE",
    "DIRTY_AIR_S",
    "fuel_correct",
    "true_pace_ranking",
    "CleanAirSplit",
    "clean_air_split",
    "clean_air_ranking",
    "DriverConsistency",
    "driver_consistency",
    "consistency_ranking",
    "BestSector",
    "IdealLap",
    "ideal_lap",
    "biggest_loss",
]
