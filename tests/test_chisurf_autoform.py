"""QuEst's forms are rendered by the host, from QuEst's own manifest.

The dependency arrow for *presentation* points at ChiSurf: `AutoForm` renders a
control panel straight from an `rpc_methods` entry in `quest/manifest.json` —
the same declaration a host reads to learn what QuEst offers. That removes the
second description of every parameter (okf/specs/assessment.md, LAY-06).

Runs in-process. That is safe only because `tests/conftest.py` imports ChiSurf
**before** anything touches IMP: both wrap C++ through SWIG, which keeps one
global type table, and loading IMP first aborted the process once the
numba-parallel kernels ran.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quest.hosts import autoform_available, chisurf_root

pytestmark = pytest.mark.skipif(
    not autoform_available(),
    reason="needs a ChiSurf checkout (CHISURF_ROOT) and a Qt binding",
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((REPO_ROOT / "quest" / "manifest.json").read_text(encoding="utf8"))
METHODS = {method["name"]: method for method in MANIFEST["rpc_methods"]}


@pytest.fixture(scope="module")
def app():
    from qtpy import QtWidgets

    yield QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_the_checkout_is_configured() -> None:
    assert chisurf_root() is not None


@pytest.mark.parametrize("method_name", sorted(METHODS))
def test_every_declared_method_renders(app, method_name: str) -> None:
    """If a method cannot be rendered, its declaration is what is at fault."""
    from chisurf.gui.autoform.auto_form import AutoForm

    form = AutoForm.from_rpc_method(METHODS[method_name], title=method_name)
    form.resize(520, 360)
    try:
        assert not form.grab().isNull()
    finally:
        form.close()


def test_the_form_exposes_the_declared_parameters(app) -> None:
    from chisurf.gui.autoform.auto_form import AutoForm

    form = AutoForm.from_rpc_method(METHODS["quest.simulate"])
    try:
        declared = set(METHODS["quest.simulate"]["params_schema"]["properties"])
        params = set(form.model.params())
        assert params <= declared, params - declared
        # The required parameter must be reachable, not just the optional ones.
        assert "project" in params
    finally:
        form.close()


def test_rendering_needs_no_quest_side_ui_code(app) -> None:
    """The point: the manifest is the whole UI description for these forms."""
    from qtpy import QtWidgets

    from chisurf.gui.autoform.auto_form import AutoForm

    form = AutoForm.from_rpc_method(METHODS["quest.structure.metadata"])
    try:
        assert isinstance(form, QtWidgets.QWidget)
    finally:
        form.close()


class TestQuestPanelHelper:
    """`quest.gui.autoform_panel` is the seam the `.ui` files get replaced through."""

    def test_it_lists_what_the_manifest_declares(self) -> None:
        from quest.gui.autoform_panel import declared_methods

        assert set(declared_methods()) == set(METHODS)

    def test_an_undeclared_method_fails_loudly(self) -> None:
        from quest.gui.autoform_panel import method_declaration

        with pytest.raises(KeyError):
            method_declaration("quest.nope")

    def test_it_builds_a_bound_panel(self, app) -> None:
        from quest.gui.autoform_panel import autoform_supported, build_method_panel

        assert autoform_supported()
        panel = build_method_panel("quest.simulate")
        try:
            # The values come back shaped like the RPC params, so the panel and
            # `simulate_handler` cannot drift apart.
            assert "project" in panel.model.params()
            assert not panel.grab().isNull()
        finally:
            panel.close()

    def test_every_declared_method_has_a_panel(self, app) -> None:
        from quest.gui.autoform_panel import build_method_panel, declared_methods

        for name in declared_methods():
            panel = build_method_panel(name)
            try:
                assert panel is not None
            finally:
                panel.close()
