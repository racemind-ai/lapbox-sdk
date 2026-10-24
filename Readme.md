# lapbox

[![tests](https://github.com/racemind-ai/lapbox-sdk/actions/workflows/tests.yml/badge.svg)](https://github.com/racemind-ai/lapbox-sdk/actions/workflows/tests.yml)

**FastF1 gives you the data. `lapbox` tells you what it means — and says so when it can't.**

`lapbox` is a Python library of Formula 1 analysis built on
[FastF1](https://docs.fastf1.dev/). It is the analysis engine behind
[LapBox](https://lapbox.in), extracted so anyone can use it on FastF1 data.

**Documentation: [lapbox.in/docs](https://lapbox.in/docs/)** — a quickstart, a guide per
module, the full API reference and the measured limits, with every example run on real
sessions.

> **Status: `0.1.0`, the first release.** In the library: `lapbox.telemetry` (one lap, two
> laps matched by track position), `lapbox.data` (laps, pit stops, stints, gaps),
> `lapbox.practice` (long runs), `lapbox.pace` (lap cleaning, fuel-corrected race pace, clean
> air vs traffic, consistency, ideal lap) and `lapbox.tyres` (degradation per compound and
> stint by stint), with charts as an optional extra (`lapbox.plot`). The strategy engine
> is moving over from LapBox for 0.2: `lapbox.strategy` (the engine itself) is on `main`
> and in the `0.2.0.dev0` pre-release; fitting it to a session, planning, regret and the
> duel follow. Until 1.0 the API may still change between minor
> releases; every change is listed in the
> [changelog](https://github.com/racemind-ai/lapbox-sdk/blob/main/CHANGELOG.md).

## Install

```bash
pip install lapbox
```

Python 3.11+. Depends on FastF1, pandas, NumPy and SciPy. For the charts, add the `plot`
extra, which brings Plotly:

```bash
pip install "lapbox[plot]"
```

**Getting the data (October 2026).** lapbox reads what FastF1 downloads. Since mid-October
F1's live-timing archive answers downloads of past sessions with *401 Unauthorized* unless
you sign in with an F1 TV Access subscription
([FastF1 issue #978](https://github.com/theOehrly/Fast-F1/issues/978)). FastF1 3.8.3, the
latest stable release, can't sign in; the pre-release `fastf1==3.9.0a0` can (see its
[authentication docs](https://docs.fastf1.dev/versions/dev/api_reference/accounts_auth.html)).
lapbox's tests pass with it; downloading through an account has not been tested here. A
FastF1 cache filled before mid-October still works; every example here was run that way.

## Example

For the whole library on real sessions, with every output saved, see the notebook
[`examples/lapbox-tour.ipynb`](https://github.com/racemind-ai/lapbox-sdk/blob/main/examples/lapbox-tour.ipynb): two laps compared, a pair the
library refuses, race pace with the fuel taken out, and tyre wear. The short version:

Every function takes FastF1 telemetry exactly as `lap.get_telemetry()` returns it.

```python
import fastf1
from lapbox.telemetry import compare, delta_time

session = fastf1.get_session(2025, "Silverstone", "Q")
session.load()
ver = session.laps.pick_drivers("VER").pick_fastest().get_telemetry()
nor = session.laps.pick_drivers("NOR").pick_fastest().get_telemetry()

result = compare(ver, nor, driver_a="VER", driver_b="NOR")
if result.matched:
    print(result.summary)
    print(f"{delta_time(result.aligned).iloc[-1]:+.3f} s")
    gain = result.segments.loc[result.segments["time_delta"].idxmax()]
    print(f"VER gains most from {gain.start_distance:.0f} to {gain.end_distance:.0f} m: {gain.time_delta:+.3f} s")
else:
    print(f"Can't compare: laps differ by {result.residual:.0f} km/h RMS after alignment")
```

```text
{'minisectors_a': 14, 'minisectors_b': 7, 'more_minisectors': 'VER', 'max_speed_a': 324.0, 'max_speed_b': 319.0}
+0.118 s
VER gains most from 2957 to 4765 m: +0.202 s
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
to end-of-race fuel and ranks each driver's median over their representative laps (from lap
2, no pit laps, within 107 % of their best). `clean_air_ranking` splits those laps by the gap to the
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
`compound_summary` takes the median per compound. `TyreDegradationModel` fits one line
per compound over every driver: lap time, with the same assumed fuel effect taken off,
against tyre age, on laps within 107 % of the driver's best. Clean the laps first; the
107 % filter catches safety-car laps, not every slow one.

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
0     SOFT                 -0.133  0.173    24
1   MEDIUM                 -0.008  0.021   399
2     HARD                  0.014  0.067   491
```

Monza barely wears tyres, and here both methods say the soft got *quicker* with age, on
20–24 laps. Neither is a measurement of the tyre alone: the slopes also carry track
evolution, fuel assumptions and who ran which stint. The low `r2` is the honest part:
tyre age explains little of how the lap times vary here. Read every rate with its laps
(and, for the model, its `r2`).

### Race strategy (pre-release)

`lapbox.strategy` prices a tyre plan lap by lap: pace, compound offset, wear, fuel still on
board, a cliff past a compound's cliff lap, the track condition, and cautions (behind a safety
car every car laps at a fixed multiple of base pace, and a stop costs half). In `0.2.0.dev0`
you give it the race yourself; fitting a `RaceModel` to a real session comes in a later
pre-release. The numbers below come from a made-up race.

```python
from lapbox.strategy import RaceModel, Strategy, StrategyEngine

model = RaceModel(
    total_laps=53,
    base_lap_time=82.0,
    compound_offsets={"SOFT": 0.0, "MEDIUM": 0.3, "HARD": 0.7},
    degradation={"SOFT": 0.10, "MEDIUM": 0.06, "HARD": 0.04},
    pit_loss=22.0,
    sc_windows=((30, 33, "SC"),),  # a safety car on laps 30-33
)
engine = StrategyEngine(model)

for plan in (
    Strategy.from_pairs([("MEDIUM", 25), ("HARD", 28)]),
    Strategy.from_pairs([("MEDIUM", 31), ("HARD", 22)]),  # stops under the safety car
    Strategy.from_pairs([("SOFT", 18), ("MEDIUM", 17), ("HARD", 18)]),
):
    race = engine.simulate(plan)
    stops = ", ".join(map(str, plan.pit_laps))
    print(f"{'-'.join(plan.compounds):<17} stop lap {stops:<7} {race.total_time:7.1f} s "
          f"(pit loss {race.pit_loss_total:4.1f} s)")
```

```text
MEDIUM-HARD       stop lap 25       4701.3 s (pit loss 22.0 s)
MEDIUM-HARD       stop lap 31       4690.1 s (pit loss 11.0 s)
SOFT-MEDIUM-HARD  stop lap 18, 35   4710.7 s (pit loss 44.0 s)
```

Stopping on lap 31, under the safety car, saves half the pit loss and wins this race by 11 s
over stopping on lap 25. That is the model's arithmetic, not a forecast: the engine has not
been backtested against real races (see the limits).

### Charts (optional)

`lapbox.plot` draws the results with Plotly; nothing else in lapbox imports it. The charts
keep the library's caveats: a pair that couldn't be matched is drawn without its time gap
and says why, frozen car data is shaded, and minisectors decided by less than 20 ms are
grey. Continuing the first example:

```python
from lapbox.plot import channel_traces, speed_delta, track_dominance

result.plot().show()                 # the time gap, then both laps' channels
speed_delta(result).show()
track_dominance(result, ver).show()  # VER's racing line, coloured by who was quicker
channel_traces(ver, driver="VER").show()
```

## Why the laps are matched first

FastF1's `Distance` is integrated from speed, so two laps of the same circuit disagree
on their total length and can start from slightly different points. Comparing them at
equal distances lines one car's corner up against the other's straight. Over a season
of LapBox comparisons, 14 % of driver pairs compared that way gave corner speeds more
than 40 km/h apart — up to 181 km/h, which is impossible between two F1 cars.

So `align` puts lap B onto lap A's basis first: it stretches B to A's lap
length, then slides it by the offset that best lines up the two speed traces.

Some laps still can't be matched: a telemetry dropout stretches one part of a lap, and
no global shift can undo that. `compare` reports how far apart the two speed
traces still are after alignment (`result.residual`, km/h RMS), and `result.matched` is
`False` above 15 km/h. That threshold was measured on 250 driver pairs: it caught 32 of
33 physically impossible pairs and withheld 8 of 217 sound ones. When `matched` is
`False`, don't use the time delta, speed delta, corner speeds, minisectors or segments —
they describe the mismatch, not the drivers. `result.frozen_a` and `result.frozen_b` say
whether a lap's car data froze in the timing feed, and where (see Limits): on 2026
qualifying that was behind every refusal measured.

FastF1's own `utils.delta_time` is deprecated and documented as *"not actually very
accurate"*. lapbox's `delta_time` takes its name on purpose: it compares the laps on
the same piece of track and ends exactly at the official lap-time gap.

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
  (four qualifying sessions, 45 pairs each), each time where the official gap was 59 ms or
  less; the median gap at those lines was 181 ms. Without a clock the
  integral stands alone: it ended a median of 92–113 ms from the official gap and on the
  wrong side in 2–9 of 45 pairs. Over 19 qualifying sessions (2023–2026, 500-point grid) the
  sessions' medians were 12–65 ms (2026 Suzuka, with 3 clean pairs, 40 / 75 ms), except 2026
  China: 114 / 77 ms at the two lines on laps with no frozen data, 6–9 m of misplacement. One lap's speed reading stalled for under a second at a time
  while RPM and gear moved on (stalls like it occur in every season without raising the error);
  without it China is still 88 / 39 ms. The cause is not known.
- **Where time changes hands is less certain than how much.** A few metres of misplacement
  between two laps cost the most where the car is slowest: the same lap compared with itself,
  with 5 m of distance error at its slowest corner, shows the gap curve gaining 0.10 s and
  losing 0.17 s around that corner. So read gains and losses from `segment_gaps`, which
  splits the lap at its speed peaks, where misplacement costs almost nothing: in that case no
  segment moved by more than 49 ms. On six qualifying sessions the curve's biggest rise and
  fall were a median 113 ms from a gap rebuilt from the cars' positions, the biggest gaining
  and losing segments 65 ms, about that check's own accuracy. A segment often holds several
  corners.
- **Some 2026 laps have frozen car data.** The timing feed sometimes repeats one car reading —
  speed, RPM, gear and throttle all unchanged, often with a throttle of 104 — for one to 25
  seconds while the car's position moves on. `Distance` is integrated from that speed, so the
  lap is misplaced (the five worst laps came out 380 m short to 85 m long), and it can't be
  matched with another. `detect_frozen_stretches(tel)` finds these stretches. On 2026 China, Suzuka
  and Monza qualifying it was behind all 57 refused driver pairs (14, 16 and 5 of 22 fastest
  laps had one); on 183 laps from 2023, 2025 and three other 2026 sessions it flagged none.
  Rebuilding the speed from the positions was tried twice and isn't reliable: with a smoothed
  position speed the repaired pairs were a median 132 ms off at the sector lines at Monza
  (clean pairs 19 ms), 69 ms at Suzuka and 172 ms at China.
- **Don't read the peak lateral g.** `gforces` takes lateral g from the curvature of the
  racing line, and the position feed bends it: on 1,008 clean qualifying laps (2023–2026) the
  single highest lateral reading passed 6 g on 554 (up to 105 g), usually at the same place on
  every lap of a session. The 95th percentile of `|a_lat|` stayed at 2.3–5.3 g and ranked
  drivers consistently from lap to lap (median Spearman 0.66, against 0.24 for the peak).
  Longitudinal g comes from the car's speed and is physical.
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
  `TyreDegradationModel` pools every driver on a compound, and the slope is small next to
  the noise: over 65 races of 2024–2026, 23 % of whole-race compound rates still come out
  negative. The library reports the fit as it comes out, with `r2` and `laps` beside it;
  clamp it before planning with it.
- **The model's fuel effect is assumed, not fitted:** the same 0.048 s per lap of fuel as
  everywhere else. Fitting it (up to `0.1.0.dev11`) could not tell fuel from tyre age,
  which both move with the lap number. Fitting each race's first half and predicting the
  second half's representative laps, the fixed effect with the 107 % filter cut the median
  error by 19 % (0.79 → 0.64 s) on 65 races and was better in 71 % of them; how lap times
  change *inside* a stint was predicted no better (−2 %, better in half the races).
  `fuel_coef` reports the correction in your fuel's unit: s/kg with LapBox's
  `fuel_load_est` (which needs a lap-number column to know how much a lap burns), s per
  lap of fuel from FastF1's laps, 0 when the load never falls.
- **Rain tyres are mostly left out in a mixed-weather race.** The 107 % filter keeps laps
  against the driver's best of the race, usually on slicks: of eight rain-tyre fits in
  those 65 races, five went and two kept under a third of their laps. Fit rain tyres
  on their own laps.
- **Stint degradation uses the same assumed 0.048 s/lap** fuel effect, and a stint needs
  three laps within 107 % of its best; a three-lap stint is a weak trend.
- **The strategy engine is not a forecast.** It prices the model you give it. In LapBox it
  was checked once against real races: for 145 pairs of teammates who finished on the lead
  lap with different strategies (57 races), the gap it predicted between their two
  strategies was a median 18–19 s off the real one, against 13.5 s for predicting no gap,
  with a correlation of about 0.1. A proper backtest comes before anything here is called
  a prediction.
- **`RaceModel`'s fuel effect defaults to 0.06 s/lap**, while `lapbox.pace` and
  `lapbox.tyres` assume 0.048 (`FUEL_EFFECT_S_PER_LAP`). The two are not reconciled yet; pass
  `fuel_effect_per_lap` if they should agree.

## What's in `lapbox.telemetry`

| One lap | |
|---|---|
| `resample(tel)` | Channels on a uniform distance grid |
| `detect_corners(tel)` | Apexes found by the size of each speed drop, so fast corners count too |
| `corner_analysis(distance, speed, apexes)` | Entry, minimum and exit speed per corner, classed slow / medium / fast |
| `detect_braking_zones(tel)`, `detect_full_throttle_zones(tel)` | Zone tables |
| `gforces(distance, speed, x, y)` | Longitudinal and lateral g from speed and the racing line |
| `channel_summary(tel)` | Top speed, full-throttle %, braking %, … |
| `detect_frozen_stretches(tel)` | Where the timing feed's car data froze (one reading repeated), which misplaces the lap |
| `TelemetryAnalyzer().analyze(tel)` | All of the above for one lap |

| Two laps | |
|---|---|
| `compare(tel_a, tel_b, ...)` | Alignment, minisectors, segments, summary, `residual` and `matched` in one call |
| `align(tel_a, tel_b)` | Both laps matched by track position |
| `alignment_residual(aligned)`, `MAX_ALIGNMENT_RESIDUAL` | How well they matched, and the cut-off |
| `delta_time(aligned)` | Time gap along the lap (positive: A ahead), from the laps' own clocks when they have them |
| `channel_delta(aligned, channel)` | Speed / throttle / brake difference |
| `corner_speeds(aligned, apexes)` | Both drivers' minimum speed at the same corners |
| `minisector_dominance(tel_a, tel_b, ...)` | Who took less time through each minisector (with both times and mean speeds) |
| `segment_gaps(aligned)` | Each driver's time between the lap's speed peaks: where time changes hands, read where misplacement costs least |

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
| `longest_run_length(laps)` | The most laps at race pace in one run: at least `MIN_LONG_RUN` (5) exactly when the session contains a long run |

## What's in `lapbox.pace`

| | |
|---|---|
| `LapCleaningPipeline(config).clean(laps)` → `CleaningResult` | Drops duplicate, untimed, pit, inaccurate, deleted, implausible (and optionally outlier) laps, **and reports how many each rule removed** |
| `representative_base_time(lap_times)` | A circuit's normal pace: the median of laps within 107 % of the best, falling back to the plain median when fewer than 20 survive |
| `fuel_correct(laps)` | Every timed, non-pit lap normalised to end-of-race fuel |
| `true_pace_ranking(laps)` | Drivers ranked on fuel-corrected median pace over representative laps (from lap 2, no pit laps, within 107 % of the best), raw rank alongside |
| `clean_air_split(laps, driver)`, `clean_air_ranking(laps)` | Fuel-corrected pace in clean air vs within 2 s of the car ahead |
| `driver_consistency(laps, driver)`, `consistency_ranking(laps)` | Spread (std, CV %) of representative laps (from lap 2, no pit laps, within 107 % of the best), lap by lap |
| `ideal_lap(laps, driver)`, `biggest_loss(ideal)` | Sum of best sectors vs the fastest lap, and where that lap lost most |

These are the same calculations the LapBox Race Analysis page runs in the browser. On
two real races (Monza 2025, Silverstone 2026) the Python and the site's TypeScript agree
on all 7,226 values compared.

## What's in `lapbox.tyres`

| | |
|---|---|
| `TyreDegradationModel().fit(laps)` | One line per compound: fuel-corrected lap time on tyre age, laps within 107 % of each driver's best. `.compounds` (rate, intercept, fuel correction, `r2`, laps), `.degradation_rate(c)`, `.predict_time_loss(c, age)`, `.summary()` |
| `stint_degradation(laps)` | Every driver's every stint: laps counted, fuel-corrected median and slope (s/lap) |
| `compound_summary(stints)` | Per compound: stints, laps, median / best / worst slope, kindest first |

`TyreDegradationModel` started as LapBox's model, with scikit-learn replaced by a NumPy
least-squares fit (identical coefficients and R² on nine real compound fits). Since
`0.1.0.dev12` it fixes the fuel effect and fits only representative laps (see the limits
above).
The stint functions are the LapBox tyre-life panel's TypeScript; on three races they agree
on every count, compound and median, and on the slopes to within 6e-17 s/lap.

## What's in `lapbox.strategy` (pre-release)

| | |
|---|---|
| `Stint(compound, laps)`, `Strategy(stints)`, `Strategy.from_pairs([(c, n), …])` | A plan: `total_laps`, `n_stops`, `pit_laps`, `compounds` |
| `RaceModel(total_laps, base_lap_time, …)` | The race: compound offsets, degradation, pit loss, fuel effect, cliff laps, track condition (or one per lap), track temperature, safety-car / VSC windows |
| `StrategyEngine(model).simulate(plan)` → `StrategyResult` | Total time, green-flag time, pit losses, stops, per-lap times; `.race_time(plan)` for the total alone |
| `.lap_trace(plan)` | Per lap: time, cumulative time, compound, tyre age, fuel laps left, safety-car state, past the cliff |
| `.per_lap_times(plan, start_lap=, start_tyre_age=)`, `.remaining_time(…)`, `.lap_cost(c, lap, age)` | A plan from mid-race on part-worn tyres; one lap priced on its own |
| `condition_pace_multiplier`, `wrong_compound_penalty`, `temp_degradation_factor` | Damp and wet pace (1.12 / 1.18 × dry), the wrong tyre for the conditions, wear with track temperature (+3 % per °C over 30) |

Moved from LapBox's strategy engine with its names and behaviour unchanged; the website now
imports it from here.

## What's in `lapbox.plot` (the `plot` extra)

| | |
|---|---|
| `channel_traces(tel)` | One lap's channels against distance, frozen car data shaded |
| `comparison(result)`, `result.plot()` | The time gap from `compare`, then both laps' channels; an unmatched pair without the gap, and the reason in the title |
| `speed_delta(result)` | Driver A's speed minus B's; for an unmatched pair, the reason instead |
| `track_dominance(result, tel_a)` | A's racing line coloured by who took less time through each minisector; within `tie_margin` (20 ms) grey |

Moved from LapBox's Plotly figures. Without Plotly, `import lapbox.plot` raises an
`ImportError` that says what to install.

## Development

```bash
pip install -e ".[dev]"
ruff check . && black --check . && pytest
```

The unit tests are synthetic and run offline. CI runs them on Python 3.11 and 3.12.

## License

MIT
