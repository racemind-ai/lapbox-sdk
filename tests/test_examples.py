"""The example notebooks are committed executed: every code cell run, in order, none failed.

They can't run in CI -- F1's live-timing servers refuse cloud machines, so FastF1 can't
download the sessions there -- so this checks the saved notebooks instead, and a stale or
broken one can't be committed unnoticed. Read as plain JSON: no Jupyter dependency.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
NOTEBOOKS = sorted(EXAMPLES.glob("*.ipynb"))


def test_there_is_an_example_notebook() -> None:
    assert NOTEBOOKS


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_every_cell_ran_in_order_without_an_error(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    # One run, top to bottom: execution counts 1, 2, 3, ...
    assert [c["execution_count"] for c in cells] == list(range(1, len(cells) + 1))
    for cell in cells:
        assert cell["outputs"], f"no output saved: {''.join(cell['source'])[:60]!r}"
        assert not [o for o in cell["outputs"] if o["output_type"] == "error"]
        stderr = "".join(
            "".join(o.get("text", "")) for o in cell["outputs"] if o.get("name") == "stderr"
        )
        assert "Traceback" not in stderr
