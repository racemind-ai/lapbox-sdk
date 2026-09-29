"""Session data: time conversion, laps, weather, pit stops, stints and gaps.

Every function takes FastF1's own frames (``session.laps``,
``session.weather_data``) and returns new ones; nothing is downloaded here.
"""

from lapbox.data.laps import (
    derive_pit_stops,
    derive_tyre_stints,
    gaps_to_car_ahead,
    normalize_laps,
    normalize_weather,
)
from lapbox.data.timing import format_laptime, seconds_to_timedelta, timedelta_to_seconds

__all__ = [
    # Time
    "timedelta_to_seconds",
    "seconds_to_timedelta",
    "format_laptime",
    # Laps
    "normalize_laps",
    "normalize_weather",
    "derive_pit_stops",
    "derive_tyre_stints",
    "gaps_to_car_ahead",
]
