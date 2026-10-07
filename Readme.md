# lapbox

[![tests](https://github.com/racemind-ai/lapbox-sdk/actions/workflows/tests.yml/badge.svg)](https://github.com/racemind-ai/lapbox-sdk/actions/workflows/tests.yml)

**FastF1 gives you the data. `lapbox` tells you what it means — and says so when it can't.**

`lapbox` is a Python library of Formula 1 analysis built on
[FastF1](https://docs.fastf1.dev/). It is the analysis engine behind
[LapBox](https://lapbox.in), extracted so anyone can use it on FastF1 data.

> **Status: early development release (`0.1.0.dev6`).** In the library so far:
> `lapbox.telemetry` (one lap, two laps matched by track position), `lapbox.data` (laps,
> pit stops, stints, gaps), `lapbox.practice` (long runs), `lapbox.pace` (lap cleaning,
> fuel-corrected race pace, clean air vs traffic, consistency, ideal lap) and `lapbox.tyres`
> (degradation per compound and stint by stint). The strategy engine is moving over from
> LapBox next. Expect the API to change before `0.1.0`.

## Install
  
Until the first PyPI release, install from the tagged source:

```bash
pip install "lapbox @ https://github.com/racemind-ai/lapbox-sdk/archive/refs/tags/v0.1.0.dev6.tar.gz"
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
{'minisectors_a': 14, 'minisectors_b': 7, 'more_minisectors': 'VER', 'max_speed_a': 324.0, 'max_speed_b': 319.0}
+0.118 s
```

### Long runs in practice

`session.laps` goes straight in. A practice session is split into runs between pit
visits; a run is a long run when at least five of its laps were at race pace (within
107 % of the run's best), so a qualifying run with its out-lap and cool-down laps is not one.

```python
import fastf1
from lapbox.practice import long_run_pace, session_runs

session = fastf1.get_session(2025, "Monza", "FP2")
session.load(telemetry=False, weather=False, messages=False)

paces = [
    long_run_pace(run)
    for runs in session_runs(session.laps).values()
    for run, kind in runs
    if kind == "long_run"
]
for p in sorted(paces, key=lambda p: p.median_s)[:5]:
    print(f"{p.driver} {p.compound:<6} {p.laps:>2} laps  {p.median_s:.3f} s  deg {p.deg_slope_s_per_lap:+.3f} s/lap")
```

```text
NOR MEDIUM 11 laps  83.616 s  deg -0.004 s/lap
VER MEDIUM 10 laps  83.688 s  deg +0.008 s/lap
PIA MEDIUM  9 laps  83.956 s  deg -0.032 s/lap
LEC SOFT    7 laps  84.025 s  deg -0.175 s/lap
RUS HARD    9 laps  84.034 s  deg -0.154 s/lap
```

### Race pace, with the fuel taken out

A car gets roughly two seconds a lap quicker over a Grand Prix just by burning fuel, so
raw lap times compare fuel loads as much as pace. `true_pace_ranking` normalises every lap
to end-of-race fuel and ranks each driver's median over their representative laps (within
107 % of their best, no pit laps). `clean_air_ranking` splits those laps by the gap to the
car ahead at the line (2 s or less is traffic) and shows what traffic cost.

```python
import fastf1
from lapbox.pace import clean_air_ranking, true_pace_ranking

session = fastf1.get_session(2025, "Monza", "R")
session.load(telemetry=False, weather=False, messages=False)

pace = true_pace_ranking(session.laps)
print(pace[["driver", "corrected_median", "gap", "raw_rank", "corrected_rank"]].head(5).round(3))

traffic = clean_air_ranking(session.laps)
print(traffic[["driver", "clean_pace", "traffic_pace", "delta", "traffic_share"]].head(3).round(3))
```

```text
  driver  corrected_median    gap  raw_rank  corrected_rank
0    VER            81.215  0.000         1               1
1    NOR            81.560  0.345         3               2
2    PIA            81.651  0.436         2               3
3    LEC            81.803  0.588         4               4
4    RUS            81.892  0.677         5               5
  driver  clean_pace  traffic_pace  delta  traffic_share
0    LAW      82.109        82.756  0.647           80.0
1    ALB      81.953        82.496  0.542           52.0
2    PIA      81.643        82.134  0.492           16.0
```

On raw medians PIA was second and NOR third. Once fuel is taken out they swap, and on this
race the corrected top five matches the finishing order. One race proves nothing
about the method.

### Tyre degradation, two ways

`stint_degradation` fits a slope through each stint's fuel-corrected laps, and
`compound_summary` takes the median per compound. `TyreDegradationModel` fits one
regression per compound over every driver, with the fuel load as a second term; clean the
laps first, because a safety-car lap is as slow as a worn tyre.

```python
import fastf1
from lapbox.pace import LapCleaningPipeline
from lapbox.tyres import TyreDegradationModel, compound_summary, stint_degradation

session = fastf1.get_session(2025, "Monza", "R")
session.load(telemetry=False, weather=False, messages=False)

stints = stint_degradation(session.laps)
print(compound_summary(stints).round(3))

model = TyreDegradationModel().fit(LapCleaningPipeline().run(session.laps))
print(model.summary()[["compound", "degradation_s_per_lap", "r2", "laps"]].round(3))
```

```text
  compound  stints  laps  median_deg  best_deg  worst_deg
0     SOFT       3    20      -0.039    -0.149      0.043
1   MEDIUM      15   404      -0.001    -0.091      0.024
2     HARD      16   489       0.005    -0.026      0.061
  compound  degradation_s_per_lap     r2  laps
0     SOFT                 -0.083  0.892    24
1   MEDIUM                 -0.000  0.633   399
2     HARD                  0.025  0.564   493
```

Monza barely wears tyres, and here both methods say the soft got *quicker* with age, on
20–24 laps. Neither is a measurement of the tyre alone: the slopes also carry track
evolution, fuel assumptions and who ran which stint. Read every rate with its laps (and,
for the model, its `r2`).

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

- **Winning more minisectors is not being quicker.** Each minisector goes to the driver who
  spent less time in it, and `summary["more_minisectors"]` counts them — but a driver can
  lose most minisectors and gain the lap in a few. In 2023 Bahrain qualifying LEC won 12 of
  21 and VER took pole by 0.292 s.
- **Many minisectors are too close to call.** 22–45 % of minisectors on four qualifying
  sessions were decided by less than 20 ms. `time_a` and `time_b` are in the table: read a
  few hundredths as a tie.
- **The time delta takes its total from the laps' own clocks.** The shape of each lap comes
  from integrating its speed; with FastF1's telemetry (which carries `Time`) each lap is then
  scaled to its measured lap time, so the gap ends exactly at the official one, as in the
  example above. In between it is as good as the alignment: at the official sector lines it
  was a median of 29–42 ms off (at most 143 ms), and on the wrong side in 2–7 of 90 checks
  where the gap was tiny (four qualifying sessions, 45 pairs each). Without a clock the
  integral stands alone: it ended a median of 92–113 ms from the official gap and on the
  wrong side in 2–9 of 45 pairs.
- **Corners are found where the car slows down,** so the count won't match the
  circuit's official corner numbering: flat-out kinks produce no speed minimum. What
  you get is the same corners on every lap of a circuit, which is what comparisons need.
- **Corner classes are rough bands** of apex speed: slow < 130 km/h, medium 130–210,
  fast > 210.
- **Long-run pace is indicative between teams.** Fuel loads are never published, so two
  cars' absolute long-run times may reflect different fuel, not different pace.
- **The long-run degradation slope** adds back an assumed fuel effect of 0.048 s/lap
  (`FUEL_EFFECT_S_PER_LAP`), the same for every car, not measured. The slope also contains
  track evolution and the driver building up, so it can be negative: the car got
  quicker through the run, not the tyres younger.
- **Gaps to the car ahead only count cars on the same lap.** A lapped car physically in
  front is not counted as traffic.
- **Race-pace fuel correction uses the same assumed 0.048 s/lap** for every car and race.
- **The 2 s dirty-air threshold is a convention, not a measurement.** Pass `threshold=`
  to change it; a per-circuit curve fitted from data is on the roadmap.
- **An ideal lap is refused, not faked.** `ideal_lap` returns `None` when a sector was never
  timed on a clean lap, or when the sum of best sectors comes out slower than the fastest
  lap (FastF1 leaves lap 1's S1 blank after a standing start).
- **Per-compound degradation is weakly determined on low-wear circuits.**
  `TyreDegradationModel` pools every driver on a compound. On 2026 Monza it gave the soft
  −0.06 s/lap with R² 0.04–0.08; adding one intercept per driver, or fitting stint by
  stint, moved the estimates by up to 2–3× on the same laps. Nothing published says which
  is right, so the library reports the fit as it comes out, with `r2` and `laps` beside it.
- **The model's fuel term follows your input.** With LapBox's `fuel_load_est` (kg) the
  coefficient is s/kg; from FastF1's laps the load is taken as the laps still to run, so it
  is s per lap of fuel. The degradation rate is the same either way.
- **Stint degradation uses the same assumed 0.048 s/lap** fuel effect, and a stint needs
  three laps within 107 % of its best; a three-lap stint is a weak trend.

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
| `cumulative_time_delta(aligned)` | Time gap along the lap (positive: A ahead), from the laps' own clocks when they have them |
| `channel_delta(aligned, channel)` | Speed / throttle / brake difference |
| `corner_speeds(aligned, apexes)` | Both drivers' minimum speed at the same corners |
| `minisector_dominance(tel_a, tel_b, ...)` | Who took less time through each minisector (with both times and mean speeds) |

## What's in `lapbox.data`

| | |
|---|---|
| `normalize_laps(laps)` | Adds float-seconds columns (`LapTimeSeconds`, sector and pit times) |
| `normalize_weather(weather)` | Adds `TimeSeconds` |
| `derive_pit_stops(laps)` | One row per stop, in-lap paired with the next out-lap, with the pit-lane time |
| `derive_tyre_stints(laps)` | One row per stint: compound, first and last lap, length, tyre life |
| `gaps_to_car_ahead(laps)` | Seconds to the car ahead at the line, per `(driver, lap)` |
| `timedelta_to_seconds`, `seconds_to_timedelta`, `format_laptime` | Time conversion, `NaT`-safe; `format_laptime(83.245)` → `'1:23.245'` |

## What's in `lapbox.practice`

| | |
|---|---|
| `detect_runs(laps)` → `[Run]` | Runs of consecutive timed laps, broken by pit visits, untimed laps and tyre changes |
| `classify_run(run, session_best)` | `long_run`, `quali_sim` or `other` |
| `session_runs(laps)` | Every driver's runs, classified |
| `long_run_pace(run)` → `LongRunPace` | Median pace, degradation slope (fuel-corrected), consistency, laps dropped |
| `longest_run_length(laps)` | Whether a session contains a long run at all |

## What's in `lapbox.pace`

| | |
|---|---|
| `LapCleaningPipeline(config).clean(laps)` → `CleaningResult` | Drops duplicate, untimed, pit, inaccurate, deleted, implausible (and optionally outlier) laps, **and reports how many each rule removed** |
| `representative_base_time(lap_times)` | A circuit's normal pace: the median of laps within 107 % of the best, falling back to the plain median when fewer than 20 survive |
| `fuel_correct(laps)` | Every timed, non-pit lap normalised to end-of-race fuel |
| `true_pace_ranking(laps)` | Drivers ranked on fuel-corrected median pace, raw rank alongside |
| `clean_air_split(laps, driver)`, `clean_air_ranking(laps)` | Fuel-corrected pace in clean air vs within 2 s of the car ahead |
| `driver_consistency(laps, driver)`, `consistency_ranking(laps)` | Spread (std, CV %) of representative laps, lap by lap |
| `ideal_lap(laps, driver)`, `biggest_loss(ideal)` | Sum of best sectors vs the fastest lap, and where that lap lost most |

These are the same calculations the LapBox Race Analysis page runs in the browser. On
two real races (Monza 2025, Silverstone 2026) the Python and the site's TypeScript agree
on all 7,226 values compared.

## What's in `lapbox.tyres`

| | |
|---|---|
| `TyreDegradationModel().fit(laps)` | One regression per compound: lap time on tyre age plus fuel load. `.compounds` (rate, intercept, fuel term, `r2`, laps), `.degradation_rate(c)`, `.predict_time_loss(c, age)`, `.summary()` |
| `stint_degradation(laps)` | Every driver's every stint: laps counted, fuel-corrected median and slope (s/lap) |
| `compound_summary(stints)` | Per compound: stints, laps, median / best / worst slope, kindest first |

`TyreDegradationModel` is LapBox's model with scikit-learn replaced by a NumPy
least-squares fit: on nine real compound fits it gives identical coefficients and R².
The stint functions are the LapBox tyre-life panel's TypeScript; on three races they agree
on every count, compound and median, and on the slopes to within 6e-17 s/lap.

## Development

```bash
pip install -e ".[dev]"
ruff check . && black --check . && pytest
```

The unit tests are synthetic and run offline. CI runs them on Python 3.11 and 3.12.

## License

MIT
