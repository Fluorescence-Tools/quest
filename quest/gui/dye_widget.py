"""QuEst's desktop form.

There is almost nothing here, and that is the point. The form's *layout* is
`quest.view.json`, generated from the parameter catalog; its *state* is a
:class:`~quest.gui.form_model.ProjectFormModel`, which is a QuEst project dict;
its *physics* is `quest.api`, the same service layer the CLI and the web backend
call. This module only does what none of those can: put an AutoForm in a widget,
and connect the three menu actions a host application expects.

What it replaced
----------------
A 1,899-line widget that loaded `dye_diffusion2.ui`, kept the state in ~40 Qt
controls, and converted them to a project dict and back by hand. Both directions
were hand-written, which is precisely where the GUI drifted away from the CLI
before — it once shipped an entirely different quenching chemistry. Here there
is no conversion step to drift: the controls are bound to the project itself.

`TransientDecayGenerator` keeps the name and the three `on*` slots because
ChiSurf's `quenching_estimator` plugin and QuEst's own menu bar call them.
"""

from __future__ import annotations

from pathlib import Path

from qtpy import QtWidgets

from ..i18n import ui
from .form_model import ProjectFormModel

__all__ = ["TransientDecayGenerator"]

# Reach the host the same way everything else in QuEst does, so a plain
# `python -m quest.gui` finds ChiSurf without the caller setting PYTHONPATH.
try:
    from ..hosts import enable_chisurf

    enable_chisurf()
except Exception:  # pragma: no cover - never fatal; QuEst runs standalone
    pass


class TransientDecayGenerator(QtWidgets.QWidget):
    """The QuEst form: an AutoForm over a project.

    Parameters
    ----------
    parent : QWidget, optional
    project : dict, optional
        Initial project. Defaults to :func:`quest.project.template_project`.
    locale : str, optional
        Language of the form. Defaults to :func:`quest.i18n.resolve_locale`.

    Attributes
    ----------
    model : ProjectFormModel
        The state. `model.project` is a QuEst project dict at all times.
    form : QWidget or None
        The rendered AutoForm; None when ChiSurf is unavailable.
    """

    def __init__(
        self, parent=None, project: dict | None = None, locale: str | None = None
    ) -> None:
        super().__init__(parent)
        self.model = ProjectFormModel(project, locale=locale)
        self.form = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        try:
            from chisurf.gui.autoform.auto_form import AutoForm

            self.form = AutoForm(self.model)
            layout.addWidget(self.form)
        except Exception:
            from .autoform_panel import InTreePyQtAutoForm

            self.form = InTreePyQtAutoForm(self.model)
            layout.addWidget(self.form)


    def _t(self, key: str) -> str:
        return ui(key, self.model.locale)

    # -- the project, in and out -------------------------------------------

    def to_project_dict(self) -> dict:
        """The current project — the same JSON the CLI and web UI consume."""
        return self.model.to_project_dict()

    def load_project_dict(self, project: dict) -> None:
        """Replace the state, then redraw the controls to match."""
        self.model.load_project_dict(project)
        self.refresh()

    def refresh(self) -> None:
        """Re-read the model into the controls, the plots and the 3D view."""
        if self.form is None:
            return
        self.form.sync_fields()
        self.form.refresh_plots()

    # -- menu actions ------------------------------------------------------
    #
    # A host's menu bar connects to these; the form's own toolbar calls the
    # model methods directly. Both end in the same place.

    def onLoadPDB(self) -> str | None:
        """Pick a structure. Also the only way to fill the 3D view."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self._t("dialog.open_structure"),
            "",
            f"{self._t('filter.structures')};;{self._t('filter.all')}",
        )
        if not path:
            return None
        self.model.pdb = path
        self.model.status = self._t("status.loaded").format(name=Path(path).name)
        self.refresh()
        return path

    def onSaveProject(self):
        result = self.model.save_project_dialog()
        self.refresh()
        return result

    def onLoadProject(self):
        result = self.model.load_project_dialog()
        self.refresh()
        return result

    def update_all(self):
        """Simulate, then show the result. The toolbar's ▶ button in method form."""
        result = self.model.run_simulation()
        self.refresh()
        return result
