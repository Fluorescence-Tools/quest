"""Test-session setup: reach the sibling checkouts, in a safe order.

Two hosts sit beside QuEst — imp-tricks for fluorescence machinery, ChiSurf for
presentation — and both are used from source checkouts rather than installs
(okf/workflows/environment.md).

**ChiSurf is imported here, first, on purpose.** It pulls in tttrlib, and
SWIG extensions share one global type table: whichever registers a
``std::vector<double>`` first owns the entry, so importing IMP before tttrlib
has been observed to destabilise the process. Importing ChiSurf before anything
touches IMP puts the ordering beyond doubt for the whole session.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    # This machine has `quest` installed editable from a *different* checkout;
    # the tests must exercise this one.
    sys.path.insert(0, str(REPO_ROOT))

# Numba's threading layer, chosen before anything imports numba.
#
# ChiSurf brings tttrlib, which carries its own OpenMP runtime. With numba's
# *workqueue* layer in the same process, building ChiSurf's Qt forms and then
# running QuEst's parallel kernels aborts the interpreter
# (`Fatal Python error: Aborted`, reproducible). Both `omp` and `tbb` are
# stable; `omp` keeps one OpenMP runtime for everyone.
os.environ.setdefault("NUMBA_THREADING_LAYER", "omp")

# Numba's thread count, pinned for the same reason and at the same moment.
#
# Numba re-reads its configuration on every *fresh* compile, and raises if the
# value it computes then differs from the number of threads already launched:
#
#     RuntimeError: Cannot set NUMBA_NUM_THREADS to a different value once the
#     threads have been launched (currently have 7, trying to set 8)
#
# With no `NUMBA_NUM_THREADS` set, the value it computes is a *default* derived
# from the machine, which something else in the process had already changed by
# the time the first cold compile happened. The failure is therefore invisible
# until the numba cache is cold — a warm cache never recompiles, so a green
# suite says nothing. Reproduce with:
#
#     find quest -name '__pycache__' -exec sh -c 'rm -f "$1"/*.nbi "$1"/*.nbc' _ {} \;
#
# An explicit value removes the recomputation: what is read on reload is what
# was read at launch. `setdefault`, so a caller can still choose.
os.environ.setdefault("NUMBA_NUM_THREADS", str(max(1, (os.cpu_count() or 1))))

#: Where ChiSurf lives unless told otherwise. Exported so subprocesses and
#: `quest.chisurf_host` agree without every caller setting PYTHONPATH.
DEFAULT_CHISURF_ROOT = Path.home() / "dev" / "chisurf"
os.environ.setdefault("CHISURF_ROOT", str(DEFAULT_CHISURF_ROOT))

#: Same for imp-tricks.
DEFAULT_IMP_TRICKS_SRC = Path.home() / "dev" / "imp-tricks" / "src"
os.environ.setdefault("IMP_TRICKS_SRC", str(DEFAULT_IMP_TRICKS_SRC))


def _qt_binding_available() -> bool:
    import importlib.util

    return any(
        importlib.util.find_spec(name) is not None
        for name in ("PyQt5", "PyQt6", "PySide2", "PySide6")
    )


def _import_chisurf_first() -> None:
    """Load ChiSurf ahead of IMP — but only where it is actually used.

    Only the AutoForm tests need ChiSurf, and they need Qt, so an interpreter
    without a Qt binding gains nothing from the import and pays a lot for it:
    ChiSurf drags in tttrlib, mdtraj, pandas and pyarrow, and in the *web*
    environment that extra footprint is enough to make Chromium die mid-test
    ("Page crashed") once the rest of the suite has run. Reproducible: the
    browser suite passes alone, fails after `test_pdb_id_support.py`, and passes
    again with `CHISURF_ROOT=/nonexistent`.

    Where Qt *is* present, importing ChiSurf first is what keeps the process
    stable: it and IMP both wrap C++ through SWIG, which keeps one global type
    table.
    """

    if not _qt_binding_available():
        return

    from quest.chisurf_host import enable_chisurf

    if not enable_chisurf():
        return
    try:
        import chisurf  # noqa: F401
    except Exception:
        # A checkout that cannot be imported is not fatal: QuEst runs standalone.
        pass


#: Loader variables ChiSurf exports that must not reach a child process.
#:
#: Importing ChiSurf sets ``DYLD_LIBRARY_PATH=<mambaforge>/lib`` -- the *base*
#: prefix -- into ``os.environ``. In-process that is inert, because dyld reads it
#: once at exec; but every subprocess inherits it, and an ``arm64`` child then
#: resolves ``libboost_filesystem.dylib`` against base's copy, so ``import IMP``
#: dies with ``Symbol not found: ..._ZN5boost10filesystem...filename_v3E``.
#:
#: That is the whole of the long-standing "`pytest -m notebooks` fails as one
#: selection" issue. It was written up as base-versus-arm64 skew and as an
#: ordering effect between the two test files; it is neither. Measured
#: 2026-07-28: `tests/test_examples.py tests/test_notebooks.py` together give
#: **16 passed**, adding *any* other module gives 16 passed, and adding
#: `tests/test_chisurf_autoform.py` -- the one that calls `autoform_available()`
#: at import -- gives **6 failed, 10 passed**. Unsetting the variable restores 16.
_LEAKED_LOADER_VARS = ("DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH")


def _drop_foreign_loader_paths() -> None:
    """Remove loader paths that point outside this interpreter's prefix.

    Anything a *deliberate* caller set inside ``sys.prefix`` is left alone.
    """

    prefix = str(Path(sys.prefix).resolve())
    for name in _LEAKED_LOADER_VARS:
        value = os.environ.get(name)
        if not value:
            continue
        kept = [part for part in value.split(os.pathsep) if part and part.startswith(prefix)]
        if kept:
            os.environ[name] = os.pathsep.join(kept)
        else:
            os.environ.pop(name, None)


def pytest_collection_finish(session) -> None:  # noqa: ARG001 - pytest hook
    """Scrub the leak once every test module has been imported.

    It has to be here rather than at conftest import time: the variable is set
    when `tests/test_chisurf_autoform.py` calls `autoform_available()` *during
    collection*, which is after this file has already run.
    """

    _drop_foreign_loader_paths()


_import_chisurf_first()
_drop_foreign_loader_paths()
