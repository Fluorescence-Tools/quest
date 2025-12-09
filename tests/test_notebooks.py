from __future__ import annotations

from pathlib import Path

import nbformat
import pytest


NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"
# Known malformed legacy file kept for reference; excluded from smoke tests.
SKIP_NOTEBOOKS = {"quest_core_headless.ipynb"}


def _iter_notebooks():
    for nb_path in sorted(NOTEBOOK_DIR.glob("*.ipynb")):
        if nb_path.name in SKIP_NOTEBOOKS:
            continue
        yield nb_path


@pytest.mark.parametrize("nb_path", list(_iter_notebooks()))
def test_notebook_is_readable(nb_path: Path) -> None:
    nb = nbformat.read(nb_path, as_version=4)
    assert nb.cells, f"Notebook {nb_path.name} has no cells"
