"""Unit tests for lapbox.plot, the optional Plotly extra.

The figure tests moved from LapBox's ``tests/unit/test_visualization.py`` with
the figures; figures are checked by structure (traces, shapes, titles), not by
rendering. Plus what the library adds: frozen car data shaded, unmatched pairs
drawn without anything that compares them point by point, near-tie minisectors
grey, and a core install that never imports Plotly.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from lapbox.plot import channel_traces, comparison, speed_delta, track_dominance
from lapbox.telemetry import (
    Comparison,
    compare,
    delta_time,
    detect_frozen_stretches,
    minisector_dominance,
)

pytestmark = pytest.mark.unit


def _with_clock(lap: pd.DataFrame) -> pd.DataFrame:
    distance = lap["Distance"].to_numpy(dtype=float)
    speed_ms = lap["Speed"].to_numpy(dtype=float) / 3.6
    return lap.assign(TimeSeconds=np.r_[0.0, np.cumsum(np.diff(distance) / speed_ms[1:])])


def _frozen(lap: pd.DataFrame, first: int = 100, samples: int = 15) -> pd.DataFrame:
    """The lap with one car reading repeated for ``samples`` samples (about 1.5 s)."""
    out = _with_clock(lap)
    held = ["Speed", "RPM", "nGear", "Throttle"]
    out.loc[first : first + samples, held] = out.loc[first, held].to_numpy()
    return out


def _unmatched(lap_tel_a: pd.DataFrame, lap_tel_b: pd.DataFrame) -> Comparison:
    """A comparison whose laps could not be matched, B's car data frozen."""
    good = compare(lap_tel_a, lap_tel_b, driver_a="VER", driver_b="HAM")
    frozen_b = pd.DataFrame(
        [
            {
                "start_distance": 0.0,
                "end_distance": 140.0,
                "seconds": 3.0,
                "speed": 168.0,
                "sentinel": True,
            }
        ]
    )
    return Comparison(
        driver_a="VER",
        driver_b="HAM",
        aligned=good.aligned,
        minisectors=good.minisectors,
        summary=good.summary,
        residual=40.0,
        frozen_b=frozen_b,
    )


class TestChannelTraces:
    def test_one_trace_per_channel(self, lap_tel_a: pd.DataFrame) -> None:
        channels = ["Speed", "Throttle", "Brake"]
        fig = channel_traces(lap_tel_a, channels=channels, driver="VER")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == len(channels)
        assert "VER" in fig.layout.title.text
        assert not fig.layout.shapes  # nothing frozen, nothing shaded

    def test_requires_distance(self) -> None:
        with pytest.raises(ValueError):
            channel_traces(pd.DataFrame({"Speed": [1, 2]}))

    def test_frozen_car_data_is_shaded(self, lap_tel_a: pd.DataFrame) -> None:
        lap = _frozen(lap_tel_a)
        fig = channel_traces(lap, channels=["Speed", "Throttle"], driver="VER")
        shaded = {(s.x0, s.x1) for s in fig.layout.shapes}
        # Exactly what detect_frozen_stretches finds (the synthetic straights are a constant
        # 300 km/h, so they count too), the injected freeze among them.
        found = detect_frozen_stretches(lap)
        assert shaded == set(zip(found["start_distance"], found["end_distance"], strict=True))
        assert (lap["Distance"].iloc[100], lap["Distance"].iloc[115]) in shaded
        assert "frozen" in fig.layout.title.text


class TestComparison:
    def test_gap_then_two_traces_per_channel(
        self, lap_tel_a: pd.DataFrame, lap_tel_b: pd.DataFrame
    ) -> None:
        result = compare(lap_tel_a, lap_tel_b, driver_a="VER", driver_b="HAM")
        assert result.matched
        fig = comparison(result, channels=["Speed", "Throttle"])
        assert len(fig.data) == 1 + 4  # the gap, then 2 drivers x 2 channels
        np.testing.assert_allclose(fig.data[0].y, delta_time(result.aligned))
        # Exactly one legend entry per driver (shown only on the first channel row).
        assert sorted(t.name for t in fig.data if t.showlegend) == ["HAM", "VER"]

    def test_unmatched_pair_has_no_gap_and_says_why(
        self, lap_tel_a: pd.DataFrame, lap_tel_b: pd.DataFrame
    ) -> None:
        fig = comparison(_unmatched(lap_tel_a, lap_tel_b), channels=["Speed"])
        assert len(fig.data) == 2  # each lap's own trace, no gap
        assert all("Gap" not in (t.name or "") for t in fig.data)
        title = fig.layout.title.text
        assert "Not matched" in title and "40 km/h" in title
        assert "HAM's car data froze for 3.0 s from 0 to 140 m" in title

    def test_comparison_plot_method(self, lap_tel_a: pd.DataFrame, lap_tel_b: pd.DataFrame) -> None:
        result = compare(lap_tel_a, lap_tel_b, driver_a="VER", driver_b="HAM")
        assert result.plot(channels=["Speed"]) == comparison(result, channels=["Speed"])


class TestSpeedDelta:
    def test_single_trace(self, lap_tel_a: pd.DataFrame, lap_tel_b: pd.DataFrame) -> None:
        result = compare(lap_tel_a, lap_tel_b, driver_a="VER", driver_b="HAM")
        fig = speed_delta(result)
        assert len(fig.data) == 1
        assert fig.data[0].fill == "tozeroy"
        np.testing.assert_allclose(
            fig.data[0].y, result.aligned["Speed_a"] - result.aligned["Speed_b"]
        )

    def test_missing_speed_raises(self, lap_tel_a: pd.DataFrame, lap_tel_b: pd.DataFrame) -> None:
        result = compare(lap_tel_a, lap_tel_b, driver_a="VER", driver_b="HAM")
        result.aligned = result.aligned.drop(columns=["Speed_b"])
        with pytest.raises(ValueError):
            speed_delta(result)

    def test_unmatched_pair_is_refused(
        self, lap_tel_a: pd.DataFrame, lap_tel_b: pd.DataFrame
    ) -> None:
        fig = speed_delta(_unmatched(lap_tel_a, lap_tel_b))
        assert len(fig.data) == 0
        assert "Not matched" in fig.layout.annotations[0].text


class TestTrackDominance:
    def test_two_driver_traces(self, lap_tel_a: pd.DataFrame, lap_tel_b: pd.DataFrame) -> None:
        result = compare(lap_tel_a, lap_tel_b, driver_a="VER", driver_b="HAM")
        fig = track_dominance(result, lap_tel_a, tie_margin=0.0)
        assert [t.name for t in fig.data] == ["VER", "HAM", "within 0 ms"]
        assert sum(len(t.x) for t in fig.data) == len(lap_tel_a)  # every point drawn once

    def test_near_ties_are_grey(self, lap_tel_a: pd.DataFrame, lap_tel_b: pd.DataFrame) -> None:
        result = compare(lap_tel_a, lap_tel_b, driver_a="VER", driver_b="HAM")
        ms = minisector_dominance(lap_tel_a, lap_tel_b, driver_a="VER", driver_b="HAM")
        ms.loc[0, "time_b"] = ms.loc[0, "time_a"] + 0.005  # minisector 1 by 5 ms
        result.minisectors = ms
        fig = track_dominance(result, lap_tel_a)
        grey = next(t for t in fig.data if t.name == "within 20 ms")
        first = lap_tel_a["Distance"] < ms.loc[1, "start_distance"]
        assert len(grey.x) >= int(first.sum())

    def test_requires_xy(self, lap_tel_a: pd.DataFrame, lap_tel_b: pd.DataFrame) -> None:
        result = compare(lap_tel_a, lap_tel_b, driver_a="VER", driver_b="HAM")
        with pytest.raises(ValueError):
            track_dominance(result, lap_tel_a.drop(columns=["X"]))

    def test_unmatched_pair_is_refused(
        self, lap_tel_a: pd.DataFrame, lap_tel_b: pd.DataFrame
    ) -> None:
        fig = track_dominance(_unmatched(lap_tel_a, lap_tel_b), lap_tel_a)
        assert len(fig.data) == 0
        assert "froze" in fig.layout.annotations[0].text


class TestOptionalExtra:
    def test_the_core_never_imports_plotly(self) -> None:
        code = (
            "import sys, lapbox, lapbox.data, lapbox.pace, lapbox.practice, lapbox.telemetry, "
            "lapbox.tyres; assert 'plotly' not in sys.modules, 'plotly imported'"
        )
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_a_missing_plotly_says_how_to_install_it(self) -> None:
        code = (
            "import sys; sys.modules['plotly'] = None\n"
            "try:\n    import lapbox.plot\nexcept ImportError as exc:\n    print(exc)\n"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert 'pip install "lapbox[plot]"' in out.stdout
