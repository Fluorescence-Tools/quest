"""Make an imp-tricks checkout importable in the live interpreter.

imp-tricks ships its extensions as ``IMP.<name>`` namespace directories under
``src/``. Adding ``src`` to ``sys.path`` is enough **only** when IMP itself is
absent or is a namespace package. In an environment with a compiled IMP — the
`arm64` conda environment here — ``IMP`` and ``IMP.bff`` are *regular* packages
with their own ``__init__.py``, so they shadow the checkout completely and
``import IMP.bff.av`` fails no matter what is on the path.

The fix is to extend the already-imported package's ``__path__``, which is where
Python looks for submodules. That is what :func:`enable_imp_tricks` does, so
``IMP.bff.av`` and ``IMP.bff.distance_metrics`` from the checkout sit alongside
the compiled ``IMP.bff.AV`` rather than replacing it.

This is a development convenience. Installing imp-tricks properly is the real
answer; see [okf/specs/imp-tricks-dedup.md].
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Iterable

__all__ = ["DEFAULT_IMP_TRICKS_SRC", "imp_tricks_src", "enable_imp_tricks"]

#: Where the checkout is expected when nothing says otherwise.
DEFAULT_IMP_TRICKS_SRC = Path.home() / "dev" / "imp-tricks" / "src"


def imp_tricks_src(path: Path | str | None = None) -> Path | None:
    """Return the imp-tricks ``src`` directory, or ``None`` if there is none.

    Resolution order: the *path* argument, the ``IMP_TRICKS_SRC`` environment
    variable, then :data:`DEFAULT_IMP_TRICKS_SRC`.
    """

    candidate = path or os.environ.get("IMP_TRICKS_SRC") or DEFAULT_IMP_TRICKS_SRC
    resolved = Path(candidate).expanduser()
    if (resolved / "IMP").is_dir():
        return resolved
    return None


def _subpackages(src: Path) -> Iterable[str]:
    for entry in sorted((src / "IMP").iterdir()):
        if entry.is_dir() and not entry.name.startswith((".", "__")):
            yield entry.name


def enable_imp_tricks(path: Path | str | None = None) -> list[str]:
    """Splice an imp-tricks checkout into the importable ``IMP`` package.

    Returns the fully-qualified names that became importable (for example
    ``["IMP.bff", "IMP.cgmol"]``). Returns an empty list when no checkout is
    found — this is a convenience, never a hard requirement, so it does not
    raise.

    Idempotent: re-running adds nothing a second time.
    """

    src = imp_tricks_src(path)
    if src is None:
        return []

    enabled: list[str] = []

    try:
        imp_module = importlib.import_module("IMP")
    except Exception:
        # No IMP at all: the plain path trick is enough, since the checkout's
        # IMP/ directory then becomes the namespace package itself.
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        return [f"IMP.{name}" for name in _subpackages(src)]

    imp_path = getattr(imp_module, "__path__", None)
    if imp_path is None:
        return []

    if str(src / "IMP") not in list(imp_path):
        imp_path.append(str(src / "IMP"))

    for name in _subpackages(src):
        target = src / "IMP" / name
        try:
            submodule = importlib.import_module(f"IMP.{name}")
        except Exception:
            # Not yet imported and not importable on its own: extending IMP's
            # __path__ above is all that is needed for it to be found later.
            enabled.append(f"IMP.{name}")
            continue

        sub_path = getattr(submodule, "__path__", None)
        if sub_path is not None and str(target) not in list(sub_path):
            # A compiled package of the same name is already loaded; extend it
            # so the checkout's submodules resolve underneath it.
            sub_path.append(str(target))
        enabled.append(f"IMP.{name}")

    return enabled
