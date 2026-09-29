# lapbox

[![tests](https://github.com/racemind-ai/lapbox-sdk/actions/workflows/tests.yml/badge.svg)](https://github.com/racemind-ai/lapbox-sdk/actions/workflows/tests.yml)

**FastF1 gives you the data. `lapbox` tells you what it means — and says so when it can't.**

`lapbox` is a Python library of Formula 1 analysis built on
[FastF1](https://docs.fastf1.dev/). It is the analysis engine behind
[LapBox](https://lapbox.in), extracted so anyone can use it on FastF1 data.

> **Status: early development release (`0.1.0.dev0`).** Only `lapbox.telemetry` is in
> the library so far. Race pace, tyre degradation, practice long runs and the strategy
> engine are moving over from LapBox next. Expect the API to change before `0.1.0`.

## Install

Until the first PyPI release, install from the tagged source:

```bash
pip install "lapbox @ https://github.com/racemind-ai/lapbox-sdk/archive/refs/tags/v0.1.0.dev0.tar.gz"
```

Python 3.11+. Depends on FastF1, pandas, NumPy and SciPy.

## Example

Every function takes FastF1 telemetry exactly as `lap.get_telemetry()` returns it.

```python
import fastf1
from lapbox.telemetry import compare_drivers, cumulative_time_delta

session = fastf1.get_session(2025, "Silverstone", "Q")
session.load()
ver = session.laps.pick_drivers("VER").pick_fastest().get_telemetry()
nor = session.laps.pick_drivers("NOR").pick_fastest().get_telemetry()

result = compare_drivers(ver, nor, driver_a="VER", driver_b="NOR")
if result.matched:
    print(result.summary)
    print(f"{cumulative_time_delta(result.aligned).iloc[-1]:+.3f} s")
else:
    print(f"Can't compare: laps differ by {result.residual:.0f} km/h RMS after alignment")
```

```text
{'minisectors_a': 15, 'minisectors_b': 6, 'dominant_driver': 'VER', 'max_speed_a': 324.0, 'max_speed_b': 319.0}
+0.056 s
```

## Why the laps are matched first

FastF1's `Distance` is integrated from speed, so two laps of the same circuit disagree
on their total length and can start from slightly different points. Comparing them at
equal distances lines one car's corner up against the other's straight. Over a season
of LapBox comparisons, 14 % of driver pairs compared that way gave corner speeds more
than 40 km/h apart — up to 181 km/h, which is impossible between two F1 cars.

So `align_by_distance` puts lap B onto lap A's basis first: it stretches B to A's lap
length, then slides it by the offset that best lines up the two speed traces.

Some laps still can't be matched: a telemetry dropout stretches one part of a lap, and
no global shift can undo that. `compare_drivers` reports how far apart the two speed
traces still are after alignment (`result.residual`, km/h RMS), and `result.matched` is
`False` above 15 km/h. That threshold was measured on 250 driver pairs: it caught 32 of
33 physically impossible pairs and withheld 8 of 217 sound ones. When `matched` is
`False`, don't use the time delta, speed delta, corner speeds or minisectors — they
describe the mismatch, not the drivers.

FastF1's own `utils.delta_time` is deprecated and documented as *"not actually very
accurate"*.

## Limits

- **The time delta is an estimate.** It integrates each lap's speed over the shared
  distance grid. In the example above it gives +0.056 s at the line; the official
  lap-time gap is +0.118 s.
- **Corners are found where the car slows down,** so the count won't match the
  circuit's official corner numbering: flat-out kinks produce no speed minimum. What
  you get is the same corners on every lap of a circuit, which is what comparisons need.
- **Corner classes are rough bands** of apex speed: slow < 130 km/h, medium 130–210,
  fast > 210.

## What's in `lapbox.telemetry`

| One lap | |
|---|---|
| `resample_by_distance(tel)` | Channels on a uniform distance grid |
| `detect_corners(tel)` | Apexes found by the size of each speed drop, so fast corners count too |
| `corner_analysis(distance, speed, apexes)` | Entry, minimum and exit speed per corner, classed slow / medium / fast |
| `detect_braking_zones(tel)`, `detect_full_throttle_zones(tel)` | Zone tables |
| `compute_gforces(distance, speed, x, y)` | Longitudinal and lateral g from speed and the racing line |
| `channel_summary(tel)` | Top speed, full-throttle %, braking %, … |
| `TelemetryAnalyzer().analyze(tel)` | All of the above for one lap |

| Two laps | |
|---|---|
| `compare_drivers(tel_a, tel_b, ...)` | Alignment, minisectors, summary, `residual` and `matched` in one call |
| `align_by_distance(tel_a, tel_b)` | Both laps matched by track position |
| `alignment_residual(aligned)`, `MAX_ALIGNMENT_RESIDUAL` | How well they matched, and the cut-off |
| `cumulative_time_delta(aligned)` | Time gap along the lap (positive: A ahead) |
| `channel_delta(aligned, channel)` | Speed / throttle / brake difference |
| `corner_speeds(aligned, apexes)` | Both drivers' minimum speed at the same corners |
| `minisector_dominance(tel_a, tel_b, ...)` | Who is faster in each minisector |

## Development

```bash
pip install -e ".[dev]"
ruff check . && black --check . && pytest
```

The unit tests are synthetic and run offline. CI runs them on Python 3.11 and 3.12.

## License

MIT
