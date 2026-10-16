"""Tyre degradation per compound: seconds lost per lap of tyre age.

Moved from LapBox's ``ml/training/tyre_degradation.py``. The model is a
deliberately *interpretable* straight line per compound:

    lap_time − fuel_effect ≈ intercept + degradation · tyre_age

The car gets lighter as fuel burns, so every lap is first corrected by the
fuel still to burn at :data:`lapbox.practice.FUEL_EFFECT_S_PER_LAP` per lap —
the constant :func:`lapbox.pace.fuel_correct` uses. The slope against tyre age
is then the **seconds lost per lap of tyre age** race engineers reason about.
Only laps within :data:`lapbox.pace.REPRESENTATIVE` (107 %) of the driver's
best lap are fitted: safety-car, traffic and mistake laps are not tyre wear.

Until 0.1.0.dev12 the fuel effect was a second fitted coefficient. Tyre age and
fuel both move with the lap number, so the two were weakly separable: on 2026
Monza the soft tyre came out at −0.06 s/lap with R² 0.04–0.08. Measured on 65
Grands Prix of 2024–2026, fitting each race's first half and predicting its
second half's representative laps, the fixed fuel effect and the 107 % filter
cut the median lap-time error by 19.4 % (0.79 → 0.64 s) and were better in
71 % of races. How lap times change *inside* a stint was predicted no better
(−1.7 %, better in half the races), and 33 % of those half-race compound rates
still came out negative (42 % before): on low-wear circuits the slope is small
next to the noise. R² is lower than before because it no longer counts the fuel
burn as explained. Read ``degradation_s_per_lap`` together with ``r2`` and
``laps``, and clamp it before planning with it.

In a mixed-weather race most intermediate and wet laps are slower than 107 % of
the driver's best on slicks and are left out (of eight rain-tyre fits in those
65 races, five went and two kept under a third of their laps). Fit rain tyres
on their own laps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from lapbox.pace.race import REPRESENTATIVE, _present, _seconds
from lapbox.practice import FUEL_EFFECT_S_PER_LAP

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CompoundDegradation:
    """Fitted degradation parameters for a single tyre compound."""

    compound: str
    degradation_s_per_lap: float
    intercept_s: float
    """Lap time on a new tyre with the fuel all burned."""
    fuel_coef: float | None
    """The fuel correction applied, in seconds per unit of fuel: per kg with a
    ``fuel_load_est`` column, per lap still to run when the load is taken from
    ``LapNumber``. ``None`` when no correction was made."""
    r2: float
    """R² of the fuel-corrected lap times against tyre age."""
    laps: int


def _least_squares(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Coefficients, intercept and R² of an ordinary least-squares fit.

    The steps scikit-learn's ``LinearRegression`` takes: centre both sides, take
    the minimum-norm least-squares solution, put the intercept back from the
    means. On nine real compound fits it gave identical coefficients and
    intercepts, and R² within 1e-16.
    """
    x_mean = x.mean(axis=0)
    y_mean = y.mean()
    coef, *_ = np.linalg.lstsq(x - x_mean, y - y_mean, rcond=None)
    intercept = float(y_mean - x_mean @ coef)
    ss_res = float(((y - (x @ coef + intercept)) ** 2).sum())
    ss_tot = float(((y - y_mean) ** 2).sum())
    if ss_tot == 0:
        return coef, intercept, 1.0 if ss_res == 0 else 0.0
    return coef, intercept, 1.0 - ss_res / ss_tot


class TyreDegradationModel:
    """Per-compound tyre-degradation fit.

    Takes either LapBox's engineered laps (``compound``, ``lap_time_seconds``,
    ``tyre_age`` or ``stint_lap``, ``fuel_load_est`` with ``lap_number``) or
    FastF1's laps as they come (``Compound``, ``LapTime``, ``TyreLife``,
    ``LapNumber``). With FastF1's laps, pit laps are dropped and the fuel load is
    taken as the laps still to run, as in a race; clean them first
    (:class:`lapbox.pace.LapCleaningPipeline`). A ``fuel_load_est`` in kilograms
    is turned into laps of fuel by how fast it falls with the lap number; one
    that never changes (practice, qualifying) gets no correction. Fit one session
    at a time: the 107 % filter keeps each driver's laps against their best.

    Args:
        min_laps: Minimum laps required to fit a compound.
        age_column: Column holding tyre age in laps; falls back to ``stint_lap``,
            then FastF1's ``TyreLife``.
        use_fuel_correction: Take the fuel effect off each lap when there is a
            fuel load to read.
    """

    def __init__(
        self,
        *,
        min_laps: int = 5,
        age_column: str = "tyre_age",
        use_fuel_correction: bool = True,
    ) -> None:
        self.min_laps = min_laps
        self.age_column = age_column
        self.use_fuel_correction = use_fuel_correction
        self._compounds: dict[str, CompoundDegradation] = {}

    @property
    def compounds(self) -> dict[str, CompoundDegradation]:
        """Mapping of compound -> fitted :class:`CompoundDegradation`."""
        return self._compounds

    def fit(self, laps: pd.DataFrame) -> TyreDegradationModel:
        """Fit a degradation line per compound present in ``laps``.

        Raises:
            ValueError: If a column is missing, or ``fuel_load_est`` changes from
                lap to lap with no lap number to say how much fuel a lap burns.
        """
        frame = self._frame(laps)
        if frame is None:
            raise ValueError("Laps need 'compound', 'lap_time_seconds' and a tyre-age column.")

        fuel_coef = self._fuel_coef(frame) if self.use_fuel_correction else None
        usable = frame.dropna(subset=["time", "age"] + (["fuel"] if fuel_coef else []))
        best = usable.groupby("driver", dropna=False)["time"].transform("min")
        usable = usable[usable["time"] <= best * REPRESENTATIVE]
        self._compounds = {}

        for compound, group in usable.groupby("compound"):
            if len(group) < self.min_laps or group["age"].nunique() < 2:
                continue

            time = group["time"].to_numpy(dtype=float)
            if fuel_coef:
                time = time - fuel_coef * group["fuel"].to_numpy(dtype=float)
            coef, intercept, r2 = _least_squares(group[["age"]].to_numpy(dtype=float), time)

            self._compounds[str(compound)] = CompoundDegradation(
                compound=str(compound),
                degradation_s_per_lap=float(coef[0]),
                intercept_s=intercept,
                fuel_coef=fuel_coef,
                r2=r2,
                laps=int(len(group)),
            )

        logger.info("Tyre degradation fitted", extra={"compounds": len(self._compounds)})
        return self

    def predict_time_loss(self, compound: str, tyre_age: float) -> float:
        """Expected lap-time loss (s) at ``tyre_age`` vs a fresh tyre.

        Raises:
            KeyError: If the compound was not fitted.
        """
        deg = self._compounds.get(str(compound))
        if deg is None:
            raise KeyError(f"No degradation model for compound {compound!r}.")
        return deg.degradation_s_per_lap * float(tyre_age)

    def degradation_rate(self, compound: str) -> float:
        """Return the fitted degradation rate (s/lap) for a compound."""
        deg = self._compounds.get(str(compound))
        if deg is None:
            raise KeyError(f"No degradation model for compound {compound!r}.")
        return deg.degradation_s_per_lap

    def summary(self) -> pd.DataFrame:
        """Return a per-compound summary table of the fitted parameters."""
        if not self._compounds:
            return pd.DataFrame(columns=["compound", "degradation_s_per_lap", "r2", "laps"])
        rows = [
            {
                "compound": d.compound,
                "degradation_s_per_lap": d.degradation_s_per_lap,
                "intercept_s": d.intercept_s,
                "fuel_coef": d.fuel_coef,
                "r2": d.r2,
                "laps": d.laps,
            }
            for d in self._compounds.values()
        ]
        return pd.DataFrame(rows).sort_values("degradation_s_per_lap").reset_index(drop=True)

    # -- Reading the laps ----------------------------------------------------

    def _frame(self, laps: pd.DataFrame) -> pd.DataFrame | None:
        """``compound``, ``driver``, ``time``, ``age`` (``lap``, ``fuel``) from either
        naming, or None. Without a driver column every lap counts as one driver's."""
        compound = next((c for c in ("compound", "Compound") if c in laps.columns), None)
        driver = next((c for c in ("driver", "Driver") if c in laps.columns), None)
        lap = next((c for c in ("lap_number", "LapNumber") if c in laps.columns), None)
        age = self._resolve_age_column(laps)
        if "lap_time_seconds" in laps.columns:
            time = laps["lap_time_seconds"]
        elif "LapTimeSeconds" in laps.columns or "LapTime" in laps.columns:
            time = _seconds(laps, "LapTime")
        else:
            time = None
        if compound is None or time is None or age is None:
            return None

        frame = pd.DataFrame(
            {
                "compound": laps[compound],
                "driver": laps[driver] if driver else "",
                "time": time,
                "age": laps[age],
            },
            index=laps.index,
        )
        if lap is not None:
            frame["lap"] = laps[lap]
        if "fuel_load_est" in laps.columns:
            frame["fuel"] = laps["fuel_load_est"]
        elif lap == "LapNumber" and laps[lap].notna().any():
            frame["fuel"] = (laps[lap].max() - laps[lap]).clip(lower=0)
        # FastF1's laps still hold the pit laps; LapBox's engineered laps never do.
        if any(c in laps.columns for c in ("PitInTime", "PitOutTime", "PitInTimeSeconds")):
            frame = frame[~(_present(laps, "PitInTime") | _present(laps, "PitOutTime"))]
        return frame

    @staticmethod
    def _fuel_coef(frame: pd.DataFrame) -> float | None:
        """Seconds per unit of ``fuel`` that make one lap's burn ``FUEL_EFFECT_S_PER_LAP``.

        The unit is read off how fast the load falls with the lap number: one per
        lap for laps still to run, the kilograms burned per lap for
        ``fuel_load_est``. A load that does not fall gets 0.0, no load at all None.
        """
        known = frame.dropna(subset=["fuel"]) if "fuel" in frame.columns else frame.iloc[:0]
        if known.empty:
            return None
        if known["fuel"].nunique() < 2:
            return 0.0
        paired = known.dropna(subset=["lap"]) if "lap" in known.columns else known.iloc[:0]
        if paired.empty:
            raise ValueError(
                "'fuel_load_est' changes from lap to lap but there is no lap number to tell "
                "how much a lap burns; add 'lap_number', or pass use_fuel_correction=False."
            )
        if paired["lap"].nunique() < 2:
            return 0.0
        lap, fuel = paired["lap"].astype(float), paired["fuel"].astype(float)
        per_lap = -float(np.polyfit(lap, fuel, 1)[0])
        return FUEL_EFFECT_S_PER_LAP / per_lap if per_lap > 0 else 0.0

    def _resolve_age_column(self, laps: pd.DataFrame) -> str | None:
        for name in (self.age_column, "stint_lap", "TyreLife"):
            if name in laps.columns and laps[name].notna().any():
                return name
        return None
