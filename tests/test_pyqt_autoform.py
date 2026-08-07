from __future__ import annotations

import pytest
from qtpy import QtWidgets

from quest.gui.app import QuEstWindow
from quest.gui.autoform_panel import InTreeCollapsiblePanel, InTreePyQtAutoForm
from quest.gui.dye_widget import TransientDecayGenerator
from quest.gui.form_model import ProjectFormModel
from quest.gui.plots import QuEstPlotCanvas
from quest.gui.quencher_widget import QuencherTableWidget


def test_intree_pyqt_autoform_creation(qtbot):
    model = ProjectFormModel()
    form = InTreePyQtAutoForm(model)
    qtbot.addWidget(form)

    assert form.model is model
    assert len(form._fields) > 0

    # Verify sync_fields & refresh_plots
    form.sync_fields()
    form.refresh_plots()

    # Find collapsible panel
    panels = form.findChildren(InTreeCollapsiblePanel)
    assert len(panels) > 0

    panel = panels[0]
    initial_state = panel.is_collapsed
    panel._toggle()
    assert panel.is_collapsed != initial_state


def test_plot_canvas(qtbot):
    canvas = QuEstPlotCanvas(title="Test Plot", x_label="x", y_label="y", log_y=True)
    qtbot.addWidget(canvas)

    series = [{"x": [0, 1, 2], "y": [1, 10, 100], "name": "Test Series"}]
    canvas.set_data(series)
    canvas.set_log_scale(log_x=False, log_y=True)


def test_quencher_widget(qtbot):
    model = ProjectFormModel()
    tbl = QuencherTableWidget(model)
    qtbot.addWidget(tbl)

    assert tbl.table.rowCount() > 0
    tbl._on_cell_changed(0, 1)  # cell update hook
    tbl._reset_defaults()


def test_transient_decay_generator_workspace(qtbot):
    from pathlib import Path
    pdb_path = Path(__file__).parent / "148l.pdb"

    w = TransientDecayGenerator()
    w.model.pdb = str(pdb_path)
    w.model.attachment_chain = "E"
    w.model.attachment_residue = 118
    w.model.attachment_atom = "CB"
    w.model.acceptor_chain = "E"
    w.model.acceptor_residue = 130
    w.model.acceptor_atom = "CB"
    qtbot.addWidget(w)





    assert w.form is not None
    assert w.splitter is not None
    assert w.plot_vsplitter is not None
    assert w.plot_hsplitter is not None
    assert w.tabs.count() == 3


    # Test simulation run
    results = w.update_all()
    assert results is not None
    assert results.quantum_yield_donor is not None



def test_quest_window(qtbot):
    win = QuEstWindow()
    qtbot.addWidget(win)

    assert win.dg is not None
    assert win.menuBar() is not None
    assert win.statusBar() is not None

