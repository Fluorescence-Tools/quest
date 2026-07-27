"""QuEst's desktop presentation layer.

The Qt widgets live here, but the *description* of the form does not: the view
spec (`quest.view.json`) and the model it binds to (:mod:`quest.gui.form_model`)
are plain data and plain Python. Importing them costs nothing and needs no
display, which is why they are importable from this package without a Qt
binding — `from quest.gui.form_model import ProjectFormModel` works in the web
environment, and its tests run there.

The widgets are resolved lazily on first access, so `import quest.gui` alone
still does not pull in Qt.
"""

from __future__ import annotations

__all__ = [
    "TransientDecayGenerator",
    "MODEL_ARGUMENT_SOURCES",
    "ProjectFormModel",
    "start_gui",
]

#: Attribute → module that defines it. Only the Qt-bearing ones are deferred.
_LAZY = {
    "start_gui": ".app",
    "TransientDecayGenerator": ".dye_widget",
    "MODEL_ARGUMENT_SOURCES": ".dye_widget",
}


def __getattr__(name: str):
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module_name, __name__), name)


def __dir__():
    return sorted(__all__)


from .form_model import ProjectFormModel  # noqa: E402  (Qt-free, import eagerly)
