"""Reach a ChiSurf checkout, so QuEst can borrow the host's UI machinery.

QuEst is a ChiSurf plugin as well as a standalone application. Where ChiSurf is
importable, its **AutoForm** renders a control panel straight from QuEst's own
`manifest.json` RPC declarations — the same JSON a host reads to learn what
QuEst offers — instead of QuEst hand-building Qt widgets or carrying `.ui`
files. That is the deduplication: one declaration, rendered by the host.

Where ChiSurf is absent, QuEst still runs: the in-tree form remains the
fallback, and nothing outside :mod:`quest.gui` imports this module.

Like :mod:`quest.imp_tricks`, this works from a **source checkout** rather than
an install, because that is what is under active development.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = ["DEFAULT_CHISURF_ROOT", "chisurf_root", "enable_chisurf", "autoform_available"]

#: Where the checkout is expected when nothing says otherwise.
DEFAULT_CHISURF_ROOT = Path.home() / "dev" / "chisurf"


def chisurf_root(path: Path | str | None = None) -> Path | None:
    """Return the ChiSurf checkout root, or ``None`` if there is none.

    Resolution order: the *path* argument, ``CHISURF_ROOT``, then
    :data:`DEFAULT_CHISURF_ROOT`.
    """

    candidate = path or os.environ.get("CHISURF_ROOT") or DEFAULT_CHISURF_ROOT
    resolved = Path(candidate).expanduser()
    if (resolved / "chisurf" / "__init__.py").exists():
        return resolved
    return None


def enable_chisurf(path: Path | str | None = None) -> bool:
    """Put a ChiSurf checkout on ``sys.path``. Returns whether one was found.

    Never raises and never overrides an already-imported ChiSurf: a host
    application that loaded QuEst as a plugin is already running its own copy,
    and replacing it mid-flight would be worse than doing nothing.
    """

    if "chisurf" in sys.modules:
        return True
    root = chisurf_root(path)
    if root is None:
        return False
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return True


def autoform_available() -> bool:
    """Whether ChiSurf's AutoForm can be imported (after :func:`enable_chisurf`)."""

    enable_chisurf()
    try:  # pragma: no cover - depends on the checkout
        from chisurf.gui.autoform.auto_form import AutoForm  # noqa: F401
    except Exception:
        return False
    return True
