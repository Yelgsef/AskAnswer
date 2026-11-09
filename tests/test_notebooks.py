"""Committed notebooks carry no outputs: outputs hold competition data (regulations §4.3)."""
from pathlib import Path

import nbformat
import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb"))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_no_outputs(path):
    nb = nbformat.read(path, as_version=4)
    assert not any(c.get("outputs") or c.get("execution_count") for c in nb.cells if c.cell_type == "code"), \
        f"clear it: jupyter nbconvert --clear-output --inplace {path}"
