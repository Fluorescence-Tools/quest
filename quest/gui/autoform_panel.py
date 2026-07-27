"""Control panels rendered by the host, from QuEst's manifest.

The step that removes the `.ui` files: instead of hand-building Qt forms, ask
ChiSurf's AutoForm to render the `rpc_methods` declarations QuEst already ships
in `quest/manifest.json`. One declaration serves the host, the RPC surface and
the UI.

`build_method_panel("quest.simulate")` returns a bound form whose values come
back as a plain params dict — exactly what
:func:`quest.backend.services.simulate_handler` takes, so the panel and the
service cannot drift apart.

Nothing here is required: where ChiSurf is absent :func:`autoform_supported`
is false and the in-tree form remains.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

__all__ = [
    "MANIFEST_PATH",
    "autoform_supported",
    "method_declaration",
    "declared_methods",
    "build_method_panel",
]

MANIFEST_PATH = Path(__file__).resolve().parents[1] / "manifest.json"


@lru_cache(maxsize=1)
def _manifest() -> dict[str, Any]:
    with MANIFEST_PATH.open("r", encoding="utf8") as handle:
        return json.load(handle)


def declared_methods() -> list[str]:
    """Names of the RPC methods the manifest declares, in declaration order."""
    return [method["name"] for method in _manifest()["rpc_methods"]]


def method_declaration(name: str) -> dict[str, Any]:
    """Return one `rpc_methods` entry.

    Raises
    ------
    KeyError
        If the manifest does not declare *name* — which is a manifest bug, not
        a caller bug, so it fails loudly.
    """
    for method in _manifest()["rpc_methods"]:
        if method["name"] == name:
            return method
    raise KeyError(f"manifest declares no method named {name!r}")


def autoform_supported() -> bool:
    """Whether the host's AutoForm can be reached from here."""
    from ..chisurf_host import autoform_available

    return autoform_available()


def build_method_panel(name: str, parent=None, **kwargs):
    """Render the parameter form for one declared method.

    Parameters
    ----------
    name : str
        An RPC method name, e.g. ``"quest.simulate"``.
    parent : QWidget, optional
    **kwargs
        Forwarded to AutoForm (``values``, ``on_change``, ``title``).

    Returns
    -------
    QWidget
        A bound form; read the entered values with ``form.model.params()``.

    Raises
    ------
    RuntimeError
        If ChiSurf is not reachable. The caller should check
        :func:`autoform_supported` and fall back to the in-tree form.
    """
    if not autoform_supported():
        raise RuntimeError(
            "AutoForm needs a ChiSurf checkout; set CHISURF_ROOT or use the "
            "in-tree form."
        )

    from chisurf.gui.autoform.auto_form import AutoForm

    kwargs.setdefault("title", name)
    return AutoForm.from_rpc_method(method_declaration(name), parent=parent, **kwargs)
