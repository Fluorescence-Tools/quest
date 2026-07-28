"""End-to-end tests that drive the QuEst web UI in a headless browser.

These tests are skipped unless both servers are reachable:

    python -m uvicorn webui.backend.app.main:app --port 8000
    pixi run -e frontend dev

Point them elsewhere with QUEST_E2E_BASE_URL / QUEST_E2E_API_URL.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api", reason="playwright is not installed")

from playwright.sync_api import Page, expect, sync_playwright  # noqa: E402

BASE_URL = os.getenv("QUEST_E2E_BASE_URL", "http://127.0.0.1:3000").rstrip("/")
API_URL = os.getenv("QUEST_E2E_API_URL", "http://127.0.0.1:8000").rstrip("/")
PDB_PATH = Path(__file__).parent / "148l.pdb"


def _server_is_up(url: str) -> bool:
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            return response.status < 500
    except (urllib.error.URLError, OSError):
        return False


pytestmark = [
    # Deselected by default; run with `pixi run test-e2e` or `-m browser`.
    pytest.mark.browser,
    pytest.mark.skipif(
        not (_server_is_up(f"{API_URL}/health") and _server_is_up(BASE_URL)),
        reason="QuEst web UI is not running (start the backend and frontend first)",
    ),
]


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as playwright:
        # `--disable-dev-shm-usage`: Chromium keeps renderer shared memory in
        # /dev/shm, which is small and is already under pressure by the time the
        # rest of the suite has run — the page then dies with a bare
        # "Page crashed". Reproducible: this file passes alone and fails after
        # `test_pdb_id_support.py`, which spins up a FastAPI TestClient first.
        # The flag moves that allocation to a normal temp file.
        instance = playwright.chromium.launch(
            args=["--disable-dev-shm-usage", "--no-sandbox"]
        )
        yield instance
        instance.close()


@pytest.fixture
def page(browser):
    context = browser.new_context(viewport={"width": 1440, "height": 900})
    page = context.new_page()
    errors: list[str] = []

    def _record_console(message) -> None:
        if message.type != "console" and message.type == "error":
            location = message.location or {}
            where = f"{location.get('url', '?')}:{location.get('lineNumber', '?')}"
            errors.append(f"{message.text} [{where}]")

    page.on("pageerror", lambda exc: errors.append(f"{exc} [pageerror] {(getattr(exc, 'stack', '') or '')[:900]}"))
    page.on("console", _record_console)
    page.errors = errors  # type: ignore[attr-defined]
    yield page
    context.close()


def _load(page: Page, path: str) -> Page:
    page.goto(f"{BASE_URL}{path}", wait_until="networkidle")
    return page


def _upload_structure(page: Page) -> None:
    page.set_input_files('input[type="file"]', str(PDB_PATH))
    # Scope to the header: the scan sidebar renders a second `.status` element.
    expect(page.locator(".topbar .status")).to_contain_text("atoms", timeout=60_000)


def _ignorable(message: str) -> bool:
    lowered = message.lower()
    return any(
        token in lowered
        for token in ("favicon", "webgl", "download the react devtools", "hydration")
    )


class TestPagesLoad:
    @pytest.mark.parametrize("path", ["/labeling", "/simulation", "/scan", "/advanced"])
    def test_page_renders_without_console_errors(self, page: Page, path: str) -> None:
        _load(page, path)
        expect(page.locator("header.topbar h1")).to_have_text("Quenching Estimator")
        real_errors = [message for message in page.errors if not _ignorable(message)]
        assert not real_errors, f"console errors on {path}: {real_errors}"

    def test_root_redirects_to_labeling(self, page: Page) -> None:
        _load(page, "/")
        expect(page.locator(".tab-btn.active")).to_contain_text("Labeling")

    def test_no_horizontal_overflow(self, page: Page) -> None:
        _load(page, "/advanced")
        overflow = page.evaluate(
            "() => document.documentElement.scrollWidth - document.documentElement.clientWidth"
        )
        assert overflow <= 1, f"page scrolls horizontally by {overflow}px"


class TestQuenchingEditor:
    def test_shows_pet_active_residues_with_chemistry(self, page: Page) -> None:
        _load(page, "/advanced")
        table = page.locator(".quench-table")
        expect(table).to_be_visible()

        # The PET-active filter is on by default and must list the quenchers.
        for residue in ("TRP", "TYR", "MET", "HIS", "CYS"):
            expect(table.locator("tbody th", has_text=residue)).to_have_count(1)

        trp_row = table.locator("tbody tr", has=page.locator("th", has_text="TRP")).first
        expect(trp_row).to_contain_text("indole ring")

    def test_all_20_residues_available(self, page: Page) -> None:
        _load(page, "/advanced")
        page.get_by_role("button", name="All 20").click()
        expect(page.locator(".quench-table tbody tr")).to_have_count(20)

    def test_editing_kq_updates_the_project(self, page: Page) -> None:
        _load(page, "/advanced")
        trp_row = page.locator(".quench-table tbody tr").filter(has_text="Tryptophan").first
        kq_input = trp_row.get_by_label("TRP quenching rate")
        kq_input.fill("7.25")
        kq_input.blur()
        expect(page.locator(".quench-summary-list")).to_contain_text("7.25")

    def test_quench_atoms_are_editable_and_restorable(self, page: Page) -> None:
        _load(page, "/advanced")
        trp_row = page.locator(".quench-table tbody tr").filter(has_text="Tryptophan").first
        trp_row.locator(".atom-summary").click()

        editor = page.locator(".quench-atom-editor input")
        expect(editor).to_have_value("CD2 CE2 CE3 CZ2 CZ3 CH2 NE1 CG CD1")

        editor.fill("CB")
        editor.blur()
        expect(page.locator(".pill-warn")).to_have_count(1)

        page.get_by_role("button", name="Restore default atoms").click()
        expect(page.locator(".pill-warn")).to_have_count(0)

    def test_clear_and_reset_quenching(self, page: Page) -> None:
        _load(page, "/advanced")
        page.get_by_role("button", name="Clear quenching").click()
        expect(page.locator(".quench-summary-list li")).to_have_count(0)

        page.get_by_role("button", name="Reset to reference").click()
        expect(page.locator(".quench-summary-list li")).to_have_count(6)
        expect(page.locator(".quench-editor-stats")).to_contain_text("6 quenching residue types")


class TestSimulationWorkflow:
    def test_upload_simulate_and_read_results(self, page: Page) -> None:
        _load(page, "/labeling")
        _upload_structure(page)

        # The largest chain must be selected by default, not the 2-residue
        # ligand chain that happens to sort first in this file.
        chain_select = page.locator("label.field").filter(has_text="Chain").locator("select")
        expect(chain_select).to_have_value("E")

        # Pick a site next to the aromatic cluster so quenching is measurable.
        residue_select = page.locator("label.field").filter(has_text="Residue").locator("select")
        residue_select.select_option(value="155", timeout=15_000)

        page.get_by_role("button", name="Run Sim").click()
        expect(page.locator(".topbar .status.ok")).to_contain_text("Simulation complete", timeout=300_000)

        page.get_by_role("link", name="2 - Simulation").click()

        # The decay must be drawn with real axes.
        expect(page.locator("svg.chart").first).to_be_visible()
        expect(page.locator("text=time [ns]").first).to_be_visible()

        # Derived quantities are populated.
        metrics = page.locator(".metrics")
        expect(metrics).to_contain_text("Donor QY")
        assert metrics.inner_text().count("-\n") < 7

        # Quenching is attributed to residue types.
        breakdown = page.locator(".quench-breakdown")
        expect(breakdown).to_be_visible()
        expect(breakdown.locator(".quench-bar-row").first).to_contain_text("%")

        shares = [
            float(value.strip().rstrip("%"))
            for value in breakdown.locator(".quench-bar-value").all_inner_texts()
        ]
        assert shares, "expected at least one quenching residue type"
        assert abs(sum(shares) - 100.0) < 1.0, f"quench shares should sum to 100%, got {shares}"

        # The per-residue table carries the same chemistry, and only residues
        # with a redox-active side chain contribute.
        table = page.locator(".scan-results-table")
        expect(table.locator("thead")).to_contain_text("Quench share")
        quenching_rows = table.locator("tbody tr.is-quenching")
        assert quenching_rows.count() == len(shares)
        for residue in ("ALA", "LEU", "VAL"):
            row = table.locator("tbody tr", has=page.locator("th", has_text=residue)).first
            assert "is-quenching" not in (row.get_attribute("class") or "")

        real_errors = [message for message in page.errors if not _ignorable(message)]
        assert not real_errors, f"console errors during simulation: {real_errors}"


class TestScanWorkflow:
    """The scan is where labeling sites get chosen, so it must name the quencher."""

    def _scan_a_few_sites(self, page: Page, indices=(10, 100, 150)) -> None:
        _load(page, "/scan")
        _upload_structure(page)

        page.locator("label.check-field input").first.uncheck()
        expect(page.locator(".scan-residue-item").first).to_be_visible()
        page.get_by_role("button", name="Select None").click()

        checkboxes = page.locator(".scan-residue-item input")
        for index in indices:
            checkboxes.nth(index).check()

        page.get_by_role("button", name="Calculate Scan").click()
        expect(page.locator(".control-panel .status.ok")).to_contain_text(
            "Successfully scanned", timeout=600_000
        )

    def test_scan_reports_the_dominant_quencher(self, page: Page) -> None:
        self._scan_a_few_sites(page)

        table = page.locator(".scan-results-table")
        expect(table.locator("thead")).to_contain_text("Quenched by")

        rows = table.locator("tbody tr")
        assert rows.count() == 3

        # At least one site must be quenched, and by a PET-active residue.
        quencher_codes = page.locator(".quencher-cell .residue-code").all_inner_texts()
        assert quencher_codes, "no site reported a quenching residue type"
        for code in quencher_codes:
            assert code in {"TRP", "TYR", "MET", "HIS", "CYS", "PRO"}, code

        real_errors = [message for message in page.errors if not _ignorable(message)]
        assert not real_errors, f"console errors during scan: {real_errors}"

    def test_quencher_column_filters(self, page: Page) -> None:
        self._scan_a_few_sites(page)

        rows = page.locator(".scan-results-table tbody tr")
        before = rows.count()
        page.locator(".column-filter-input").last.fill("ZZZ")
        expect(rows).to_have_count(0)

        page.locator(".column-filter-input").last.fill("")
        expect(rows).to_have_count(before)


class TestFretPanel:
    """FRET has to be reachable from the browser, R0 included.

    Before this panel existed the web UI could plot a FRET curve but could not
    switch FRET on or set a Förster radius, so a browser project always fell
    back to the 52 A code default.
    """

    def _enable_fret(self, page: Page) -> None:
        _load(page, "/labeling")
        _upload_structure(page)
        page.get_by_label("Simulate FRET acceptor").check()

    def test_the_orientation_factor_is_editable(self, page: Page) -> None:
        """κ² must exist on **this** surface too.

        It reached the parameter catalog and the Qt form on 2026-07-28 and was
        missing here — the web FRET panel is hand-built React rather than
        generated from the catalog, so a new parameter does not arrive on its
        own. A parameter present on two surfaces and absent from the third is
        the divergence okf/architecture/surfaces.md exists to prevent, and no
        assertion in this file would have noticed: it was found by looking.
        """
        self._enable_fret(page)
        field = page.locator("label.field").filter(has_text="κ²").locator("input")
        expect(field).to_be_visible()
        # 2/3 is the isotropic dynamic average a published R0 already assumes,
        # so the default reproduces every number computed before it existed.
        # Displayed to four figures -- the raw double put 0.6666666666666666 in
        # the box -- while the *stored* value stays exactly 2/3, which is what
        # the rate's neutrality depends on.
        assert abs(float(field.input_value()) - 2 / 3) < 5e-4
        assert len(field.input_value()) <= 8, (
            f"kappa2 renders as {field.input_value()!r}; round it for display"
        )

        field.fill("1.25")
        field.blur()
        page.wait_for_timeout(400)
        assert abs(float(field.input_value()) - 1.25) < 1e-9

        # And the backend template carries it, so the panel round-trips rather
        # than editing a key the domain ignores.
        stored = page.evaluate(
            "async () => (await fetch('/api/template?with_fret=true')).json()"
        )
        assert "kappa2" in stored["project"]["fret"]

    def test_fret_is_off_until_asked_for(self, page: Page) -> None:
        _load(page, "/labeling")
        checkbox = page.get_by_label("Simulate FRET acceptor")
        expect(checkbox).not_to_be_checked()
        # The acceptor controls stay hidden while it is off.
        expect(page.locator("label.field").filter(has_text="Acceptor chain")).to_have_count(0)

    def test_enabling_fret_reveals_the_acceptor_site_and_r0(self, page: Page) -> None:
        self._enable_fret(page)

        for label in ("Acceptor chain", "Acceptor residue", "Acceptor atom"):
            expect(page.locator("label.field").filter(has_text=label)).to_have_count(1)

        r0_field = page.locator("label.field").filter(has_text="R0").locator("input")
        expect(r0_field).to_have_value("52")

    def test_the_acceptor_is_seeded_on_the_loaded_structure(self, page: Page) -> None:
        """Enabling FRET must produce a project that validates, not residue 1 of nothing."""
        self._enable_fret(page)

        chain = page.locator("label.field").filter(has_text="Acceptor chain").locator("select")
        expect(chain).to_have_value("E")
        residue = page.locator("label.field").filter(has_text="Acceptor residue").locator("select")
        assert residue.input_value() not in ("", "0")
        atom = page.locator("label.field").filter(has_text="Acceptor atom").locator("select")
        assert atom.input_value() != ""

    def test_the_acceptor_is_not_seeded_on_a_water(self, page: Page) -> None:
        """148L ends in HOH residues; a dye cannot be tethered to one."""
        self._enable_fret(page)

        residue = page.locator("label.field").filter(has_text="Acceptor residue").locator("select")
        label = residue.locator(f'option[value="{residue.input_value()}"]').inner_text()
        assert "HOH" not in label, f"acceptor seeded on a water: {label}"

        atom = page.locator("label.field").filter(has_text="Acceptor atom").locator("select")
        assert atom.input_value() == "CB", atom.input_value()

    def test_r0_reaches_the_project_in_angstrom(self, page: Page) -> None:
        self._enable_fret(page)

        r0_field = page.locator("label.field").filter(has_text="R0").locator("input")
        r0_field.fill("61")
        r0_field.blur()

        # The JSON-mode input is a styled switch: the real checkbox is hidden,
        # so drive it through its label like a user does.
        page.locator(".toggle-switch .switch").click()
        project = json.loads(page.locator(".json-panel textarea").input_value())

        assert project["fret"]["enabled"] is True
        # Symmetric matrix, donor->acceptor cell carries R0, diagonal empty.
        assert project["fret"]["R0_matrix"] == [[None, 61.0], [61.0, None]]
        assert len(project["fret"]["dyes"]) == 2

        real_errors = [message for message in page.errors if not _ignorable(message)]
        assert not real_errors, f"console errors enabling FRET: {real_errors}"


class TestFretSimulation:
    """The point of the panel: a FRET run the browser can actually produce."""

    def test_fret_run_yields_a_second_decay_and_an_efficiency(self, page: Page) -> None:
        _load(page, "/labeling")
        _upload_structure(page)

        residue_select = page.locator("label.field").filter(has_text="Residue").locator("select")
        residue_select.select_option(value="55", timeout=15_000)
        page.get_by_label("Simulate FRET acceptor").check()

        page.get_by_role("button", name="Run Sim").click()
        expect(page.locator(".topbar .status.ok")).to_contain_text(
            "Simulation complete", timeout=600_000
        )

        page.get_by_role("link", name="2 - Simulation").click()

        # A FRET decay is drawn alongside the donor one...
        expect(page.locator("path.fret-line")).to_have_count(1)
        # ...and the efficiency is a real number, not a dash. A Forster radius
        # in the wrong unit would leave transfer at zero here.
        metrics = page.locator(".metrics")
        expect(metrics).to_contain_text("FRET Eff.")
        efficiency = page.evaluate(
            """() => {
                const cells = [...document.querySelectorAll('.metrics *')].map(e => e.textContent || '');
                const index = cells.findIndex(text => text.trim() === 'FRET Eff.');
                return index >= 0 ? cells[index + 1] : null;
            }"""
        )
        assert efficiency not in (None, "-"), f"no FRET efficiency reported: {efficiency!r}"
        assert 0.0 < float(efficiency) <= 1.0, efficiency

        real_errors = [message for message in page.errors if not _ignorable(message)]
        assert not real_errors, f"console errors during FRET simulation: {real_errors}"


class TestRunPicker:
    def test_job_list_is_capped(self, page: Page) -> None:
        """The picker shows a bounded window, not every run ever made."""
        _load(page, "/labeling")
        options = page.locator(".control-panel select").first.locator("option")
        # One placeholder plus at most the server-side page size.
        assert options.count() <= 51, f"run picker listed {options.count()} entries"

    def test_run_list_response_is_a_summary(self, page: Page) -> None:
        """The key is `runs` since 2026-07-28.

        "job" now means a *running operation*, matching the host; these are
        finished run artefacts. The frontend accepts either shape during the
        transition, so only this test pins the new one.
        """
        _load(page, "/labeling")
        payload = page.evaluate(
            "async () => (await fetch('/api/jobs')).json()",
        )
        assert isinstance(payload, dict)
        assert "runs" in payload and "total" in payload
        assert len(payload["runs"]) <= payload["limit"]
        if payload["runs"]:
            # Summaries only: no embedded project blobs.
            assert set(payload["runs"][0]) == {"job_id", "labeling_site", "date"}


class TestLanguageSwitching:
    """The browser reads the same translations as the desktop form and the docs."""

    def test_the_selector_offers_every_language(self, page) -> None:
        _load(page, "/simulation")
        options = page.get_by_test_id("locale-select").inner_text().split("\n")
        assert {"English", "Deutsch", "Français"} <= set(o.strip() for o in options), options

    def test_switching_translates_the_parameter_labels(self, page) -> None:
        """The labels come from the catalog, so this proves the shared path."""
        _load(page, "/simulation")
        page.get_by_test_id("locale-select").select_option("de")
        page.wait_for_timeout(1500)
        body = page.inner_text("body")
        assert "Ungelöschte Lebensdauer" in body, "tau0 did not translate"
        assert "Simulationsdauer" in body, "t_max did not translate"

        page.get_by_test_id("locale-select").select_option("fr")
        page.wait_for_timeout(1500)
        body = page.inner_text("body")
        assert "Durée de vie non éteinte" in body
        assert "Langue" in body

    def test_english_keeps_its_hand_written_labels(self, page) -> None:
        """They are terser than the catalog's prose and chosen for this layout."""
        _load(page, "/simulation")
        page.get_by_test_id("locale-select").select_option("en")
        page.wait_for_timeout(1200)
        assert "tau0 [ns]" in page.inner_text("body")

    def test_the_language_survives_a_reload(self, page) -> None:
        _load(page, "/simulation")
        page.get_by_test_id("locale-select").select_option("de")
        page.wait_for_timeout(1200)
        page.reload(wait_until="networkidle")
        page.wait_for_timeout(1200)
        assert page.get_by_test_id("locale-select").input_value() == "de"
