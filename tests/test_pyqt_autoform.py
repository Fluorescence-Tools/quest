from __future__ import annotations

import pytest
from qtpy import QtWidgets

from quest.gui.dye_widget import TransientDecayGenerator
from quest.gui.autoform_panel import InTreePyQtAutoForm, InTreeCollapsiblePanel
from quest.gui.form_model import ProjectFormModel


def test_intree_pyqt_autoform_creation(qtbot):
    model = ProjectFormModel()
    form = InTreePyQtAutoForm(model)
    qtbot.addWidget(form)

    assert form.model is model
    assert len(form._fields) > 0

    # Verify sync_fields
    form.sync_fields()

    # Find a collapsible panel
    panels = form.findChildren(InTreeCollapsiblePanel)
    assert len(panels) > 0

    panel = panels[0]
    initial_state = panel.is_collapsed
    panel._toggle()
    assert panel.is_collapsed != initial_state


def test_transient_decay_generator(qtbot):
    w = TransientDecayGenerator()
    qtbot.addWidget(w)
    assert w.form is not None
