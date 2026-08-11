"""The notebooks must run, not merely parse.

The previous version of this file asserted that each notebook had cells. Every
notebook in the folder passed that while importing `mfm`, a package that has not
existed for years — so the suite was green and the notebooks were dead.

These execute them. That is slower, and it is the only check worth having: a
notebook is documentation that claims to work.
"""

from __future__ import annotations

from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat", reason="nbformat is not installed")
nbclient = pytest.importorskip("nbclient", reason="nbclient is not installed")
pytest.importorskip("matplotlib", reason="the notebooks plot")

from nbclient.exceptions import CellExecutionError  # noqa: E402

# Deselected by default: executing four notebooks spawns four kernels and runs
# real simulations, which is too much to add to a process that has already
# imported IMP, numba and FastAPI. Run with `pixi run test-notebooks`.
pytestmark = pytest.mark.notebooks

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"

#: Notebooks are sized to finish in seconds, but a Monte-Carlo run plus a numba
#: warm-up is not instant.
TIMEOUT_SECONDS = 900


def _notebooks():
    return sorted(NOTEBOOK_DIR.glob("*.ipynb"))


def test_there_are_notebooks() -> None:
    assert _notebooks(), "the notebooks folder is empty"


@pytest.mark.parametrize("nb_path", _notebooks(), ids=lambda p: p.stem)
def test_notebook_runs(nb_path: Path) -> None:
    """Execute end to end, in the notebooks folder, with a clean kernel."""
    notebook = nbformat.read(nb_path, as_version=4)
    client = nbclient.NotebookClient(
        notebook,
        timeout=TIMEOUT_SECONDS,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_DIR)}},
    )
    try:
        client.execute()
    except CellExecutionError as error:  # pragma: no cover - failure path
        pytest.fail(f"{nb_path.name} failed to execute:\n{error}")


@pytest.mark.parametrize("nb_path", _notebooks(), ids=lambda p: p.stem)
def test_notebook_explains_itself(nb_path: Path) -> None:
    """A notebook that is only code is a script with extra syntax."""
    notebook = nbformat.read(nb_path, as_version=4)
    markdown = [c for c in notebook.cells if c.cell_type == "markdown"]
    assert markdown, f"{nb_path.name} has no prose"
    assert any(cell.source.strip().startswith("#") for cell in markdown), (
        f"{nb_path.name} has no title"
    )
