"""The refusals lapbox raises on purpose.

When lapbox can't give an honest answer it raises one of these instead of returning a number.
Both kinds are also :class:`ValueError`, so code written to catch ``ValueError`` keeps working,
and an application can tell them apart: a web service would answer
:class:`NotEnoughDataError` with 404 (there is nothing to show) and :class:`InvalidPlanError`
with 422 (the request can't be met as asked).
"""

from __future__ import annotations


class LapboxError(Exception):
    """Base class of every refusal lapbox raises on purpose."""


class NotEnoughDataError(LapboxError, ValueError):
    """The data can't support an answer: for example, laps from which no compound can be fitted."""


class InvalidPlanError(LapboxError, ValueError):
    """The request can't be met as asked.

    For example, constraints no tyre plan satisfies (a minimum stint longer than half the race,
    fewer than two compounds in a dry race), or a comparison of no plans at all.
    """


__all__ = ["LapboxError", "NotEnoughDataError", "InvalidPlanError"]
