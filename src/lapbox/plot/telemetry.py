"""Plotly figures for one lap, and for two laps matched by track position.

Moved from LapBox's ``telemetry/visualization.py`` with what the library knows
added: a lap's frozen car data is shaded, a pair that could not be matched is
drawn without anything that compares the two laps sample for sample (and says
why), and minisectors decided by less than 20 ms are drawn as too close to call.

Every function returns a :class:`plotly.graph_objects.Figure`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from lapbox.telemetry.comparison import MAX_ALIGNMENT_RESIDUAL, Comparison, delta_time
from lapbox.telemetry.lap import detect_frozen_stretches

# Axis label per channel; the order stacks the subplots.
_CHANNEL_AXES: dict[str, str] = {
    "Speed": "Speed (km/h)",
    "Throttle": "Throttle (%)",
    "Brake": "Brake",
    "RPM": "RPM",
    "nGear": "Gear",
    "DRS": "DRS",
}

# Colour-blind-safe pair for two drivers; grey for what can't be told apart.
_COLOR_A = "#0072B2"
_COLOR_B = "#D55E00"
_COLOR_TIE = "#9A9A9A"
_FROZEN_FILL = "rgba(150, 150, 160, 0.25)"
_LAYOUT = {"template": "plotly_white", "margin": {"t": 70, "l": 60, "r": 20, "b": 40}}


def _present_channels(frame: pd.DataFrame, suffix: str = "") -> list[str]:
    return [c for c in _CHANNEL_AXES if f"{c}{suffix}" in frame.columns]


def _longest_freeze(driver: str, frozen: pd.DataFrame) -> str | None:
    if frozen is None or frozen.empty:
        return None
    worst = frozen.loc[frozen["seconds"].astype(float).idxmax()]
    return (
        f"{driver}'s car data froze for {float(worst['seconds']):.1f} s from "
        f"{float(worst['start_distance']):.0f} to {float(worst['end_distance']):.0f} m of its lap"
    )


def _unmatched_reason(result: Comparison) -> str:
    """Why the two laps can't be compared point by point, in one sentence or two."""
    reason = (
        f"Not matched: the speed traces still differ by {result.residual:.0f} km/h RMS after "
        f"alignment (limit {MAX_ALIGNMENT_RESIDUAL:.0f})."
    )
    frozen = [
        p
        for p in (
            _longest_freeze(result.driver_a, result.frozen_a),
            _longest_freeze(result.driver_b, result.frozen_b),
        )
        if p
    ]
    return reason + (" " + "; ".join(frozen) + "." if frozen else "")


def _refusal(title: str, reason: str) -> go.Figure:
    """An empty figure that says what isn't drawn and why."""
    fig = go.Figure()
    text = reason.replace(". ", ".<br>").replace("; ", ";<br>")  # Plotly doesn't wrap
    fig.add_annotation(
        text=text, showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper", align="center"
    )
    fig.update_layout(title=title, height=240, xaxis={"visible": False}, yaxis={"visible": False})
    fig.update_layout(**_LAYOUT)
    return fig


def channel_traces(
    tel: pd.DataFrame,
    *,
    channels: list[str] | None = None,
    driver: str = "",
    title: str | None = None,
) -> go.Figure:
    """Each telemetry channel against distance for one lap, in stacked rows.

    Stretches where the timing feed's car data froze
    (:func:`~lapbox.telemetry.detect_frozen_stretches`) are shaded grey: the
    readings there are one value repeated, not the car.

    Raises:
        ValueError: Without a ``Distance`` column or any channel to draw.
    """
    if "Distance" not in tel.columns:
        raise ValueError("Telemetry must contain a 'Distance' column.")
    channels = channels or _present_channels(tel)
    if not channels:
        raise ValueError("No plottable telemetry channels present.")

    fig = make_subplots(
        rows=len(channels),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=[_CHANNEL_AXES.get(c, c) for c in channels],
    )
    for row, channel in enumerate(channels, start=1):
        fig.add_trace(
            go.Scatter(
                x=tel["Distance"],
                y=tel[channel],
                mode="lines",
                name=channel,
                line={"color": _COLOR_A, "width": 1.5},
                showlegend=False,
            ),
            row=row,
            col=1,
        )
        fig.update_yaxes(title_text=_CHANNEL_AXES.get(channel, channel), row=row, col=1)
    fig.update_xaxes(title_text="Distance (m)", row=len(channels), col=1)

    frozen = detect_frozen_stretches(tel)
    for stretch in frozen.itertuples(index=False):
        fig.add_vrect(
            x0=stretch.start_distance,
            x1=stretch.end_distance,
            fillcolor=_FROZEN_FILL,
            line_width=0,
            layer="below",
            row="all",
            col=1,
        )
    name = title or (f"Telemetry — {driver}" if driver else "Telemetry")
    if not frozen.empty:
        name += " · grey: car data frozen"
    fig.update_layout(title=name, height=180 * len(channels), **_LAYOUT)
    return fig


def comparison(
    result: Comparison,
    *,
    channels: list[str] | None = None,
    title: str | None = None,
) -> go.Figure:
    """Two laps from :func:`~lapbox.telemetry.compare`: the time gap, then each channel.

    The top row is :func:`~lapbox.telemetry.delta_time` (above zero, driver A is
    ahead at that point of the lap); below it both drivers' channels on the shared
    distance axis. When ``result.matched`` is ``False`` the gap is left out and the
    title says why: the laps could not be put on the same points of track, so the
    gap would describe the mismatch. Each lap's own trace is still drawn.
    """
    aligned = result.aligned
    label_a, label_b = result.driver_a, result.driver_b
    channels = channels or _present_channels(aligned, suffix="_a")
    if not channels:
        raise ValueError("No plottable channels present in the aligned frame.")

    rows = len(channels) + (1 if result.matched else 0)
    titles = (["Gap (s)"] if result.matched else []) + [_CHANNEL_AXES.get(c, c) for c in channels]
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=titles
    )
    distance = aligned["Distance"]
    first = 1
    if result.matched:
        fig.add_trace(
            go.Scatter(
                x=distance,
                y=delta_time(aligned),
                mode="lines",
                fill="tozeroy",
                name=f"Gap (above 0: {label_a} ahead)",
                line={"color": _COLOR_A, "width": 1.5},
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_hline(y=0, line={"color": "gray", "width": 1, "dash": "dash"}, row=1, col=1)
        fig.update_yaxes(title_text=f"Gap (s), + {label_a} ahead", row=1, col=1)
        first = 2
    for row, channel in enumerate(channels, start=first):
        for label, suffix, color in ((label_a, "_a", _COLOR_A), (label_b, "_b", _COLOR_B)):
            fig.add_trace(
                go.Scatter(
                    x=distance,
                    y=aligned[f"{channel}{suffix}"],
                    mode="lines",
                    name=label,
                    legendgroup=label,
                    showlegend=(row == first),  # one legend entry per driver
                    line={"color": color, "width": 1.5},
                ),
                row=row,
                col=1,
            )
        fig.update_yaxes(title_text=_CHANNEL_AXES.get(channel, channel), row=row, col=1)
    fig.update_xaxes(title_text="Distance (m)", row=rows, col=1)
    name = title or f"{label_a} vs {label_b}"
    top = _LAYOUT["margin"]["t"]
    if not result.matched:
        reason = f"{_unmatched_reason(result)} No gap drawn.".replace(". ", ".<br>")
        reason = reason.replace("; ", ";<br>")
        name += f"<br><sup>{reason}</sup>"
        top += 20 * (reason.count("<br>") + 1)  # room for every line of the reason
    height = 180 * len(channels) + (200 if result.matched else 0) + top
    fig.update_layout(**_LAYOUT)
    fig.update_layout(
        # Anchored to the top, so a long reason grows downwards into the margin made for it.
        title={"text": name, "yref": "container", "y": 1, "yanchor": "top", "pad": {"t": 16}},
        height=height,
        margin={**_LAYOUT["margin"], "t": top},
    )
    return fig


def speed_delta(result: Comparison) -> go.Figure:
    """Driver A's speed minus driver B's against distance (above zero, A is faster there).

    A pair that could not be matched gets an empty figure saying why: the
    difference would compare different points of track.
    """
    label_a, label_b = result.driver_a, result.driver_b
    title = f"Speed delta: {label_a} − {label_b}"
    if not result.matched:
        return _refusal(title, _unmatched_reason(result))
    aligned = result.aligned
    if "Speed_a" not in aligned.columns or "Speed_b" not in aligned.columns:
        raise ValueError("The aligned frame needs Speed for both drivers.")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=aligned["Distance"],
            y=aligned["Speed_a"] - aligned["Speed_b"],
            mode="lines",
            fill="tozeroy",
            line={"color": _COLOR_A, "width": 1.5},
            name=f"Δ Speed ({label_a} − {label_b})",
        )
    )
    fig.add_hline(y=0, line={"color": "gray", "width": 1, "dash": "dash"})
    fig.update_layout(
        title=title, xaxis_title="Distance (m)", yaxis_title="Δ Speed (km/h)", height=320
    )
    fig.update_layout(**_LAYOUT)
    return fig


def track_dominance(
    result: Comparison,
    tel_a: pd.DataFrame,
    *,
    tie_margin: float = 0.020,
) -> go.Figure:
    """Driver A's racing line coloured by who took less time through each minisector.

    Minisectors decided by less than ``tie_margin`` seconds are grey, too close to
    call: on 2023 Bahrain qualifying 29 % of 945 minisectors were within 20 ms.
    ``tel_a`` is driver A's lap (``X``, ``Y``, ``Distance``): the comparison's
    shared axis is A's. A pair that could not be matched gets an empty figure
    saying why.

    Raises:
        ValueError: If ``tel_a`` has no ``X``, ``Y`` or ``Distance``.
    """
    for col in ("X", "Y", "Distance"):
        if col not in tel_a.columns:
            raise ValueError(f"Telemetry must contain '{col}' for the track map.")
    label_a, label_b = result.driver_a, result.driver_b
    title = f"Track dominance: {label_a} vs {label_b}"
    if not result.matched:
        return _refusal(title, _unmatched_reason(result))

    ms = result.minisectors
    # Each point's minisector: the last one starting at or before it.
    starts = ms["start_distance"].to_numpy(dtype=float)[1:]
    index = np.searchsorted(starts, tel_a["Distance"].to_numpy(dtype=float), side="right")
    margin = (ms["time_a"] - ms["time_b"]).abs().to_numpy(dtype=float)
    owner = np.where(margin < tie_margin, "", ms["fastest"].to_numpy())[index]

    fig = go.Figure()
    for label, color, name in (
        (label_a, _COLOR_A, label_a),
        (label_b, _COLOR_B, label_b),
        ("", _COLOR_TIE, f"within {tie_margin * 1000:.0f} ms"),
    ):
        mask = owner == label
        fig.add_trace(
            go.Scatter(
                x=tel_a.loc[mask, "X"],
                y=tel_a.loc[mask, "Y"],
                mode="markers",
                marker={"color": color, "size": 4},
                name=name,
            )
        )
    fig.update_layout(
        title=title,
        height=600,
        xaxis={"visible": False},
        yaxis={"visible": False, "scaleanchor": "x", "scaleratio": 1},
    )
    fig.update_layout(**{**_LAYOUT, "margin": {"t": 70, "l": 20, "r": 20, "b": 20}})
    return fig
