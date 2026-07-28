"""Reaching the sibling checkouts QuEst is developed against.

Two host-adjacent trees, and neither is installed: `imp-tricks` supplies the
fluorescence machinery under `IMP.bff`, and `chisurf` supplies the form renderer
and is the application QuEst becomes a plugin of. Both are spliced in from
source, because the source tree is what changes
([workflows/environment.md](/workflows/environment.md)).

They lived at the top level as `imp_tricks.py` and `chisurf_host.py`, which made
two bridges of the same kind look unrelated.

**Nothing here is required.** Each returns empty or False when its checkout is
absent, and QuEst runs standalone.
"""

from .chisurf import autoform_available, chisurf_root, enable_chisurf
from .imp_tricks import (
    DEFAULT_IMP_TRICKS_SRC,
    enable_imp_tricks,
    imp_tricks_src,
)

__all__ = [
    "enable_imp_tricks",
    "imp_tricks_src",
    "DEFAULT_IMP_TRICKS_SRC",
    "enable_chisurf",
    "chisurf_root",
    "autoform_available",
]
