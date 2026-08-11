---
type: Subsystem
title: Qt GUI
description: The desktop form — generated from the parameter catalog, rendered by ChiSurf's AutoForm, bound to a project dict.
resource: quest/gui/
tags: [qt, gui, autoform, view-spec]
status: stable
generated: { by: process:okf-v0.1-import, at: 2026-07-27T00:00:00Z }
stale_after: 2027-01-27
sources:
  - id: quest-gui
    resource: ../../quest/gui
    title: quest/gui/
    author: human:tpeulen
    last_modified: 2026-07-28
---

# What it is

`quest/gui/` is four small files and one generated one:

| file | what it is |
|---|---|
| `form_model.py` | `ProjectFormModel` — the state. **Qt-free.** |
| `quest.view.json` | the layout. **Generated**; never hand-edit. |
| `generate_view_spec.py` | the generator, run by hand like `doc/generate_parameter_docs.py`. |
| `dye_widget.py` | puts an AutoForm in a `QWidget`. |
| `app.py` | the standalone window and its menu bar. |
| `autoform_panel.py` | renders one RPC method's parameters from `manifest.json`. |
| `chimol_widget.py` | the 3D viewer + sequence bar, both **embedded from ChiSurf**. |
| `quencher_widget.py`, `plots.py`, `theme.py`, `guided_tour.py` | the table, the canvases, the palette, the tour. |

Entry points: `quest gui` (CLI subcommand), the `cs.quest` gui-script, and
ChiSurf's `quenching_estimator` plugin, which puts `TransientDecayGenerator` in
a `QMainWindow` and connects its File menu to `onLoadPDB`, `onSaveProject` and
`onLoadProject`.

# The form's state is a project — there is no conversion

`ProjectFormModel` presents a project dict as flat attributes. `model.tau0 = 3.5`
writes `project["tau0"]`; `model.acceptor_chain = "B"` writes
`project["fret"]["dyes"][1]["attachment"]["chain"]`. :data:`FIELD_PATHS` is that
map, and the attribute names are the catalog's own paths, shortened where they
hurt (`av_parameter.` → `av_`, `fret.dyes.1.` → `acceptor_`).

This is the whole reason the migration was worth doing. The old widget kept its
state in ~40 Qt controls and converted to a project on the way out
(`to_project_dict`) and back on the way in (`load_project_dict`) — two
hand-written directions, and **the conversion is where the GUI drifted away from
the CLI**. It once shipped an entirely different quenching chemistry. There is
now no conversion step in which to drift: what the form saves is byte-for-byte
what `quest simulate` runs and what the web UI posts.

Two consequences worth knowing:

- The model imports no Qt, so it is testable — and usable — headless. A
  subprocess guardrail asserts that importing it pulls no Qt binding in. (It
  used to be phrased as "runs in the *base* environment"; base is retired —
  [workflows/environment.md](/workflows/environment.md).)
- Save/load is `save_project` / `load_project`, writing plain project JSON.
  AutoForm's generic `save_state`/`load_state` also work, but the project file
  is the one that means something to another surface.

Two derived members exist so that the form does not need special cases:

- **`structure_label`** is a read-only property returning the loaded file's
  name. Without it the dock's structure display and the file picker were both
  bound to `pdb`, i.e. two controls on one value — which the drift guard
  `test_each_scalar_field_appears_exactly_once` fails on.
- **`auto_attach_to_structure`** runs on `load_project_dict`. The template's
  default site (chain A, residue 1) frequently does not exist in the structure
  just opened, which left a project that validates but cannot run. It only
  overwrites an attachment that is missing or invalid, prefers `CB`/`CA`/`OG`/
  `SG`, and never selects a non-standard residue — a water could otherwise be
  chosen, the same class of error as
  [`lastLabelableResidue`](/subsystems/webui.md) on the web.

# The layout is generated from the parameter catalog

`quest/settings/parameter_catalog.json` already named every input by its project
path and carried its label, type, unit and description; `doc/parameters.md` and
the web UI were generated from it. `generate_view_spec.py` now generates the Qt
form from it too, so a parameter is described **once**:

    python -m quest.gui.generate_view_spec

What the generator derives: one panel per catalog *category*, one field per
parameter, labels with units, descriptions as tooltips. What is authored in the
generator: the arrangement — which panels share a dock, where the 3D view and
the plots sit, what the toolbar does. Those are judgements about a form, not
facts about a parameter.

One spec is generated per language — `quest.view.json`,
`quest.view.de.json`, `quest.view.fr.json` — and `ProjectFormModel(locale=…)`
picks the matching one ([subsystems/i18n.md](/subsystems/i18n.md)).

`tests/test_view_spec.py` regenerates in memory and fails if any committed file
differs, and `tests/test_form_model.py` fails if the catalog documents a
parameter the form does not bind (or the reverse). A new input therefore cannot
reach the docs and the web UI while quietly missing from the desktop.

Single-column panels throughout: the parameter dock is narrow by default and
QuEst's labels are long ("Donor diffusion coefficient (Å²/ns)"), so two columns
clipped the field being typed in.

# QuEst owns no Qt widgets and no plotting code

Everything the form draws is an AutoForm section type:

- the structure picker — `value` of `kind: "file"`;
- the quencher table — an editable `table` over `quencher_rows`, writing back
  through `update_quencher_cell`;
- the 3D view — the `chimol` custom section, bound to `preview_models`, backed
  by `ChiMolStructureWidget`, which embeds ChiSurf's `MolView` *and*
  `SequenceDock` and emits `residueSelected(chain, residue)` when either is
  clicked; donor/acceptor toggle buttons say which attachment the next click
  sets. The widget is a host of ChiSurf's widgets, not a re-implementation —
  the mapping of what to embed is
  [subsystems/chisurf-widgets.md](/subsystems/chisurf-widgets.md);
- the three plots (decay, trajectory, autocorrelation) — `plot` sections,
  rendered through ChiSurf's `chiplot`. QuEst supplies only series data
  (`decay_series`, `trajectory_series`, `autocorrelation_series`).

That closed `LAY-06`, `LAY-10` and `LAY-11` in one change, and deleted:
`dye_diffusion2.ui`, `pdb_widget.ui`, `ui/av_property.ui`, `ui/dye_diffusion.ui`,
`ui/dye_diffusion3.ui`, `widgets.py` (`PDBSelector`), `dye_diffusion.json` and
`ui/rescource_rc.py` (98 kB of generated base64). The window icon survives as a
plain file, `quest/gui/icons/dye-diffusion.ico`.

# The quencher table edits the real chemistry

`LAY-10` was that the Qt table bound to `DonorDecay.quencher`
(`{residue: {atoms, kQ}}`), which has no per-residue contact radius and no
per-residue stickiness — so the desktop could not express what
`amino_acid_interactions` does, and ran `stickiness_mode="global"` while every
other surface ran `"amino_acid"`. The table now edits `amino_acid_interactions`
itself: kQ, contact radius, quenching atoms and slow factor, per residue. There
is no second shape left to disagree.

A blank contact radius means **null** — "fall back to `critical_distance`" — not
zero, which would switch quenching off. Bad input in a cell is dropped rather
than raised: the user is mid-typing, and a traceback out of a cell editor is not
a useful way to say "not a number".

# Running goes through the service layer

The ▶ button calls `ProjectFormModel.run_simulation()`, which calls
`quest.api.simulate_project(project, save_outputs=False)` — the same entry point
the CLI and the web backend use. The form contributes no physics of its own;
that is the rule the old GUI broke.

`save_outputs=False` keeps a GUI run from writing `jobs/<uuid>/` into whatever
directory the app started in. That flag used to *also* suppress the thinned
trajectory and point clouds, so the desktop silently had no plots to draw;
`render_data` is now a separate parameter (see
[simulation-pipeline](/architecture/simulation-pipeline.md)).

**The run is a job, and the window stays alive** (since 2026-08-10). `LAY-07`
closed on 2026-07-28 by building `quest/tasks.py`, but the desktop went on
calling `simulate_project` on the GUI thread — the mechanism existed and the
surface that motivated it did not use it. Now:

| Call | Thread | Does |
|---|---|---|
| `start_simulation()` | caller's | registers a `quest.tasks` job, spawns the worker, returns the `job_id` |
| `wait_simulation(timeout)` | caller's | joins with a timeout, so a GUI can pump events in between |
| `poll_simulation()` | **GUI thread** | applies the outcome and notifies listeners; returns the terminal status once |
| `cancel_simulation()` | caller's | asks; see the honesty note below |
| `run_simulation()` | caller's | the three above, joined — the synchronous entry point, unchanged for headless callers, still raising |

**The worker never notifies.** A listener redraws widgets, and touching a
widget off the GUI thread is undefined; that split is the whole reason this is
several methods instead of one, and `test_the_worker_never_notifies` pins it by
recording the thread every listener fires on.

`TransientDecayGenerator.update_all` pumps the event queue between joins, so
the window keeps painting and Cancel keeps responding, and still returns the
result. Re-entrant clicks are ignored rather than queued — pumping events means
the button can re-enter the handler that started the run.

**What it buys, honestly.** The trajectory kernels are `njit(nogil=True)`
deliberately, so the GUI thread does get the interpreter back during the
dominant cost; the AV grid kernels are `parallel=True` without `nogil` and
still hold it, so a very large volume can still stutter. A smaller freeze, not
none. And **cancellation cannot interrupt a single simulation**: `_simulate_traj`
computes the whole trajectory inside one `njit` call and a jitted kernel cannot
poll a Python flag, so Cancel marks the job `CANCELLING`, the form stops waiting
and the result is *discarded* when the kernel returns. A scan, which checks
between sites, really does stop ([tasks.md](tasks.md)).

One user-visible consequence: clicking ▶ on a broken project no longer raises
out of the Qt slot — the failure is reported in the status line. The exception
is still raised by `run_simulation()`, where a caller can act on it.

# ChiSurf is required for the desktop surface, and only for it

QuEst's Qt form is rendered by the host. Without a ChiSurf checkout there is no
form: `dye_widget.py` shows a panel saying so, because there is no second
in-tree renderer to fall back to and an empty window explains nothing. The host
is found through `quest.hosts.chisurf.enable_chisurf()`, so no caller has to set
`PYTHONPATH`.

This does **not** weaken the optional-Qt rule. Qt lives in `quest/gui/` and
nowhere else; the domain (`quest/core/`), the service layer and the web backend
import no Qt binding — checked by a subprocess guardrail in
`tests/test_service_layer.py`. That is what lets the CLI, the API and the web
backend run in the base environment, where **PyQt5 is not installed**.

Practical consequences:

- The GUI cannot be launched or screenshotted from an interpreter without a Qt
  binding; `arm64` — the one environment — has PyQt5 and imports quest from
  this tree ([workflows/testing.md](/workflows/testing.md)).
- **Do not pass `-p no:pytest-qt`.** It was mandatory only while the retired
  base environment had no binding; in `arm64` it removes the `qtbot` fixture
  and breaks the widget tests.
- `quest/gui/__init__.py` resolves the Qt-bearing names lazily, so
  `from quest.gui.form_model import ProjectFormModel` works with no Qt at all.
- A change that makes a `qtpy` import unconditional breaks the headless
  surfaces. Do not.
