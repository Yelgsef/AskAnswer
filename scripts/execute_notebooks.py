"""Execute copies of notebooks/*.ipynb with chosen parameters and save them, outputs included, outside the repo
(the outputs hold competition data, so the repo copies stay clear; tests/test_notebooks.py checks that).

    .venv/bin/python scripts/execute_notebooks.py --out ~/projects/datamate_notebooks_outputs 01_eda 03_evaluation
    .venv/bin/python scripts/execute_notebooks.py --out ... 02_inference --set LIMIT=20

The kernel starts in notebooks/, as when a person opens the notebook, so the setup cell finds the checkout. Values given with --set
replace the matching assignments in the cell tagged `parameters` (the line keeps its comment)."""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parents[1]


def set_parameters(nb, values: dict[str, str]) -> None:
    cells = [c for c in nb.cells if "parameters" in c.get("metadata", {}).get("tags", [])]
    if values and not cells:
        raise SystemExit("no cell tagged 'parameters'")
    for name, value in values.items():
        pattern = re.compile(rf"^({re.escape(name)}\s*=\s*)([^#\n]*?)(\s*#.*)?$", re.M)
        source, n = pattern.subn(lambda m: f"{m.group(1)}{value}" + (f"  {m.group(3).strip()}" if m.group(3) else ""), cells[0].source)
        if not n:
            raise SystemExit(f"parameter {name} not found")
        cells[0].source = source


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+", help="notebook names, e.g. 01_eda")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--set", action="append", default=[], help="NAME=python-expression for the parameters cell")
    ap.add_argument("--timeout", type=int, default=6 * 3600, help="seconds per cell")
    a = ap.parse_args()
    values = dict(s.split("=", 1) for s in a.set)
    a.out.mkdir(parents=True, exist_ok=True)
    for name in a.names:
        src = REPO / "notebooks" / f"{name}.ipynb"
        nb = nbformat.read(src, as_version=4)
        set_parameters(nb, values)
        t = time.time()
        client = NotebookClient(nb, timeout=a.timeout, kernel_name="python3", resources={"metadata": {"path": str(REPO / "notebooks")}})
        try:
            client.execute()
            status = "ok"
        except Exception as exc:          # keep the partly executed notebook: the failing cell shows its traceback
            status = f"FAILED: {type(exc).__name__}: {str(exc)[:300]}"
        nbformat.write(nb, a.out / f"{name}.ipynb")
        print(f"{name}: {status} in {time.time() - t:.0f}s -> {a.out / (name + '.ipynb')}", flush=True)


if __name__ == "__main__":
    main()
