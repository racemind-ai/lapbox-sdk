"""Plotly figures for lapbox's results: the optional ``plot`` extra.

``pip install "lapbox[plot]"``. Nothing else in lapbox imports Plotly, so the core
install stays without it.
"""

try:
    import plotly  # noqa: F401
except ImportError as exc:
    raise ImportError('lapbox.plot needs Plotly: pip install "lapbox[plot]"') from exc

from lapbox.plot.telemetry import channel_traces, comparison, speed_delta, track_dominance

__all__ = ["channel_traces", "comparison", "speed_delta", "track_dominance"]
