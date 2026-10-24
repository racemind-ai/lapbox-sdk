# Changelog

All notable changes to `lapbox`. Until 1.0 a minor release (0.2, 0.3, …) may change the API;
each change is listed here. The `0.1.0.devN` versions were development tags on GitHub, installed
from their source tarballs (the LapBox website ran on them); `0.1.0` is the first release on PyPI.

## 0.2.0.dev0 (pre-release) — 2026-10-24

- `lapbox.strategy`: the race-strategy engine, moved from LapBox with its names and behaviour
  unchanged — `Stint`, `Strategy`, `RaceModel` (compound offsets, wear, fuel, cliff laps, track
  condition lap by lap, safety-car and VSC windows), `StrategyEngine` (`simulate`, `lap_trace`,
  `race_time`, `per_lap_times`, `remaining_time`, `lap_cost`), `StrategyResult`, and the wet and
  temperature helpers; 14 tests moved with it. You build the `RaceModel` yourself for now, and
  the engine has not been backtested against real races (see the Readme's limits).
- Documentation at https://lapbox.in/docs/ (a quickstart, a guide per module, the full API
  reference, the limits), linked from the Readme and as the package's `Documentation` URL.
- Readme: F1's live-timing archive now needs an F1 TV Access sign-in; FastF1's pre-release
  `3.9.0a0` supports it, and lapbox's 236 tests pass with it.

## 0.1.0 — 2026-10-19

The first release on PyPI. The code is `0.1.0.dev14`'s; what changed is the packaging and two
source comments.

- `pip install lapbox`, and `pip install "lapbox[plot]"` for the charts.
- `MAX_ALIGNMENT_RESIDUAL`: its comment dates the 250-pair measurement behind the 15 km/h cut
  and adds a re-check with today's alignment (pairs without frozen car data at 2023 Bahrain,
  2025 Silverstone and 2026 China qualifying: median 4.2–7.3 km/h, none above 12.8).
- `DIRTY_AIR_S`: its comment now says what it is, a rule of thumb that lapbox has not measured.

What is in it:

- `lapbox.telemetry`: one lap (`resample`, `detect_corners`, `corner_analysis`, braking and
  full-throttle zones, `gforces`, `detect_frozen_stretches`) and two laps matched by track
  position (`compare` → `Comparison` with `residual` and `matched`, `align`, `delta_time`,
  `minisector_dominance`, `segment_gaps`).
- `lapbox.data`: time conversion, laps, weather, pit stops, stints, `gaps_to_car_ahead`.
- `lapbox.practice`: runs, long runs against qualifying simulations, long-run pace.
- `lapbox.pace`: lap cleaning with a per-step audit, a circuit's representative lap time,
  fuel-corrected pace, clean air against traffic, consistency, the ideal lap.
- `lapbox.tyres`: degradation per compound (`TyreDegradationModel`) and stint by stint.
- `lapbox.plot` (the `plot` extra): channel traces, a two-lap comparison, speed delta, track
  dominance, each keeping the library's caveats.

What it can't do yet is in the Readme's Limits section: the lateral-g peak, China 2026's
accuracy, rebuilding frozen car data, low-wear tyre rates, and F1's live-timing archive, which
has refused FastF1's downloads since mid-October 2026.

## Development releases

- **0.1.0.dev14** (2026-10-19): `true_pace_ranking` counts from lap 2, before the 107 % best is
  taken, as consistency and the clean-air split do. On 12 races the 30 driver-races that had
  counted lap 1 changed and nothing else: the corrected median by a median 9.5 ms, up to 1.08 s.
- **0.1.0.dev13** (2026-10-17): `lapbox.plot` and the `plot` extra; `Comparison.plot()`. A pair
  `compare` refuses is drawn without its time gap and says why.
- **0.1.0.dev12** (2026-10-16): `TyreDegradationModel` takes the fuel effect off at
  `FUEL_EFFECT_S_PER_LAP` instead of fitting it, and fits only laps within 107 % of each
  driver's best. Held out on 65 races (first half fitted, second half predicted): error 0.79 →
  0.64 s, better in 71 % of races. `r2` is now of the fuel-corrected laps.
- **0.1.0.dev11** (2026-10-15): `driver_consistency` and `consistency_ranking` count from lap 2.
  Lap 1 had been counted in 30 of 245 driver-races and moved σ by a median 0.18 s, up to 1.9 s.
- **0.1.0.dev10** (2026-10-12): `longest_run_length` counts laps at race pace, as
  `classify_run` does; counting timed laps let 6 of 145 practice sessions pass with no long run.
- **0.1.0.dev9** (2026-10-11): the naming pass, with no aliases. `align_by_distance` → `align`,
  `compare_drivers` → `compare`, `ComparisonResult` → `Comparison`, `cumulative_time_delta` →
  `delta_time`, `resample_by_distance` → `resample`, `compute_gforces` → `gforces`,
  `frozen_stretches` → `detect_frozen_stretches`; `telemetry/compare.py` → `comparison.py`.
- **0.1.0.dev8** (2026-10-11): `frozen_stretches`, where the timing feed's car data froze, and
  `ComparisonResult.frozen_a` / `frozen_b`. Every pair refused on 2026 China, Suzuka and Monza
  qualifying (57) had such a lap.
- **0.1.0.dev7** (2026-10-10): `segment_gaps` and `ComparisonResult.segments`: each driver's time
  between the lap's speed peaks.
- **0.1.0.dev6** (2026-10-07): `align_by_distance` finds the shift to a fraction of a sample;
  the time gap and minisector times come from each lap's own clock, so the gap ends at the
  official one and is a median 29–42 ms off it at the sector lines.
- **0.1.0.dev5** (2026-10-06): minisectors go to the driver with less time in them, on their own
  4,000-point alignment; `summary["dominant_driver"]` → `summary["more_minisectors"]`.
- **0.1.0.dev4** (2026-10-04): `lapbox.tyres`.
- **0.1.0.dev3** (2026-10-03): per-driver race-pace calls about 7× faster; outputs identical.
- **0.1.0.dev2** (2026-09-30): `lapbox.pace`; `classify_run` needs 5 laps at race pace for a long
  run, not 5 timed laps.
- **0.1.0.dev1** (2026-09-29): `lapbox.data` and `lapbox.practice`.
- **0.1.0.dev0** (2026-09-29): `lapbox.telemetry`, with `ComparisonResult.residual` and
  `.matched`.
