"""The example scripts must run.

Same reasoning as the notebooks: an example that no longer works is worse than
no example, and the only way to know is to run it. They are deliberately sized
to finish in seconds.

Deselected by default with the notebooks — each spawns a subprocess and runs
real simulations. Run with `pixi run test-examples`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.notebooks

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def _examples():
    return sorted(p for p in EXAMPLES_DIR.glob("*.py") if not p.name.startswith("_"))


def test_there_are_examples() -> None:
    assert _examples(), "the examples folder is empty"


@pytest.mark.parametrize("script", _examples(), ids=lambda p: p.stem)
def test_example_runs(script: Path) -> None:
    """Run it exactly as the docstring says to, from the examples folder."""
    result = subprocess.run(
        [sys.executable, script.name],
        cwd=EXAMPLES_DIR,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, (
        f"{script.name} exited {result.returncode}\n"
        f"stdout:\n{result.stdout[-2000:]}\nstderr:\n{result.stderr[-3000:]}"
    )
    # An example that prints nothing has not demonstrated anything.
    assert result.stdout.strip(), f"{script.name} produced no output"


@pytest.mark.parametrize("script", _examples(), ids=lambda p: p.stem)
def test_example_explains_itself(script: Path) -> None:
    source = script.read_text(encoding="utf8")
    assert source.lstrip().startswith('"""'), f"{script.name} has no module docstring"
    assert "python examples/" in source, (
        f"{script.name} does not say how to run it"
    )
