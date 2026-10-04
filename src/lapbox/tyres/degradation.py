"""Tyre degradation per compound: seconds lost per lap of tyre age.

Moved from LapBox's ``ml/training/tyre_degradation.py``. The model is a
deliberately *interpretable* linear regression per compound:

    lap_time ≈ intercept + degradation · tyre_age + fuel_coef · fuel_load

The fuel covariate absorbs the large, confounding effect of the car getting
lighter as fuel burns, so the ``tyre_age`` coefficient estimates **seconds lost
per lap of tyre age** — the number race engineers reason about.

It pools every driver on a compound. On low-wear circuits that is weakly
determined: measured on 2026 Monza it gave the soft tyre *negative* degradation
(−0.06 s/lap) with R² 0.04–0.08, and adding one intercept per driver, or fitting
stint by stint (:func:`lapbox.tyres.stint_degradation`), moved the estimates by
up to 2–3× on the same laps. Nothing published says which is right. Read
``degradation_s_per_lap`` together with ``r2`` and ``laps``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from lapbox.pace.race import _present, _seconds

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CompoundDegradation:
    """Fitted degradation parameters for a single tyre compound."""

    compound: str
    degradation_s_per_lap: float
    intercept_s: float
    fuel_coef: float | None
    """Seconds per unit of fuel: per kg with a ``fuel_load_est`` column, per lap
    still to run when the load is estimated from ``LapNumber``."""
    r2: float
    laps: int


def _least_squares(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Coefficients, intercept and R² of an ordinary least-squares fit.

    The steps scikit-learn's ``LinearRegression`` takes: centre both sides, take
    the minimum-norm least-squares solution, put the intercept back from the
    means. On nine real compound fits it gave identical coefficients and
    intercepts, and R² within 1e-16. A covariate with no spread (fuel in a
    practice session) gets a coefficient of 0, as there.
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
    """Per-compound tyre-degradation regression.

    Takes either LapBox's engineered laps (``compound``, ``lap_time_seconds``,
    ``tyre_age`` or ``stint_lap``, ``fuel_load_est``) or FastF1's laps as they
    come (``Compound``, ``LapTime``, ``TyreLife``, ``LapNumber``). With FastF1's
    laps, pit laps are dropped and the fuel load is taken as the laps still to
    run; clean them first (:class:`lapbox.pace.LapCleaningPipeline`), because a
    safety-car lap is as slow as a worn tyre.

    Args:
        min_laps: Minimum laps required to fit a compound.
        age_column: Column holding tyre age in laps; falls back to ``stint_lap``,
            then FastF1's ``TyreLife``.
        use_fuel_correction: Include the fuel load as a covariate when there is one.
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
        """Fit a degradation regression per compound present in ``laps``."""
        frame = self._frame(laps)
        if frame is None:
            raise ValueError("Laps need 'compound', 'lap_time_seconds' and a tyre-age column.")

        use_fuel = self.use_fuel_correction and "fuel" in frame.columns
        self._compounds = {}

        for compound, group in frame.groupby("compound"):
            feature_cols = ["age"] + (["fuel"] if use_fuel else [])
            usable = group.dropna(subset=["time", "age"])
            if use_fuel:
                usable = usable.dropna(subset=["fuel"])
            if len(usable) < self.min_laps or usable["age"].nunique() < 2:
                continue

            x = usable[feature_cols].to_numpy(dtype=float)
            y = usable["time"].to_numpy(dtype=float)
            coef, intercept, r2 = _least_squares(x, y)

            self._compounds[str(compound)] = CompoundDegradation(
                compound=str(compound),
                degradation_s_per_lap=float(coef[0]),
                intercept_s=intercept,
                fuel_coef=float(coef[1]) if use_fuel else None,
                r2=r2,
                laps=int(len(usable)),
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
        """``compound``, ``time``, ``age`` (and ``fuel``) from either naming, or None."""
        compound = next((c for c in ("compound", "Compound") if c in laps.columns), None)
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
            {"compound": laps[compound], "time": time, "age": laps[age]}, index=laps.index
        )
        if "fuel_load_est" in laps.columns:
            frame["fuel"] = laps["fuel_load_est"]
        elif "LapNumber" in laps.columns and laps["LapNumber"].notna().any():
            frame["fuel"] = (laps["LapNumber"].max() - laps["LapNumber"]).clip(lower=0)
        # FastF1's laps still hold the pit laps; LapBox's engineered laps never do.
        if any(c in laps.columns for c in ("PitInTime", "PitOutTime", "PitInTimeSeconds")):
            frame = frame[~(_present(laps, "PitInTime") | _present(laps, "PitOutTime"))]
        return frame

    def _resolve_age_column(self, laps: pd.DataFrame) -> str | None:
        for name in (self.age_column, "stint_lap", "TyreLife"):
            if name in laps.columns and laps[name].notna().any():
                return name
        return None
