"""lapbox — Formula 1 analysis on top of FastF1.

FastF1 gives you the data. ``lapbox`` tells you what it means — and says so when
it can't.
"""

from __future__ import annotations

import logging

__version__ = "0.1.0.dev0"

# A library never configures logging: records go wherever the application sends
# them, and nowhere if it sets nothing up.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["__version__"]
