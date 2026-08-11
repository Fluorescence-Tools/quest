"use client";

import React from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useAppContext } from "../context/AppContext";
import {
  DEFAULT_FORSTER_RADIUS,
  DEFAULT_KAPPA2,
  defaultAcceptorDye,
  firstUsefulAtom,
  forsterMatrix,
  lastLabelableResidue,
  n,
} from "../lib/projectHelpers";
import { Tour } from "./tour/Tour";
import { getTourLabels, getTourSteps } from "./tour/tourSteps";
import { AutoForm } from "./autoform";
import { AutoFormSpec } from "./autoform/types";


export function AppLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  // Normalize `/` to `/labeling` for layout purposes
  const currentTab = pathname === "/" ? "/labeling" : pathname;

  const {
    locale,
    setLocale,
    locales,
    t,
    project,
    setProject,
    savedProject,
    projectText,
    setProjectText,
    structure,
    result,
    status,
    showJson,
    setShowJson,
    showCitationModal,
    setShowCitationModal,
    scanResults,
    scanAll,
    setScanAll,
    selectedScanResidues,
    setSelectedScanResidues,
    scanProgress,
    scanStatus,
    isScanning,
    jobs,
    jobsTotal,
    dyes,
    selectedDyeName,
    catalog,
    selectedJobId,
    pdbInputRef,
    
    handleLoadJob,
    handleDyeChange,
    uploadPdb,
    loadPdbId,
    runSimulation,
    abortSimulation,
    runScan,
    abortScan,
    downloadResult,
    handleClearState,
    saveDyePreset,
    saveParameter,
    applyJson,
    updateProject,
    allResidues,
  } = useAppContext();

  const chains = structure?.chains || [];
  const selectedChain = chains.find((chain) => chain.id === project.attachment?.chain) || chains[0];
  const residues = selectedChain?.residues || [];
  const selectedResidue = residues.find((residue) => residue.id === project.attachment?.residue) || residues[0];
  const atoms = selectedResidue?.atoms || [];
  const donor = project.fret?.dyes?.[0];

  // FRET acceptor: a second labeling site, resolved against the same structure.
  const fretEnabled = Boolean(project.fret?.enabled);
  const acceptor = project.fret?.dyes?.[1];
  const acceptorChain =
    chains.find((chain) => chain.id === acceptor?.attachment?.chain) || selectedChain;
  const acceptorResidues = acceptorChain?.residues || [];
  const acceptorResidue =
    acceptorResidues.find((residue) => residue.id === acceptor?.attachment?.residue) ||
    acceptorResidues[0];
  const acceptorAtoms = acceptorResidue?.atoms || [];
  // R0 lives in the donor->acceptor cell of the Forster matrix, in Angstrom.
  const r0 = n(project.fret?.R0_matrix?.[0]?.[1], DEFAULT_FORSTER_RADIUS);
  const savedR0 = n(savedProject.fret?.R0_matrix?.[0]?.[1], DEFAULT_FORSTER_RADIUS);
  // kappa2 is *stated*, not modelled: QuEst's dye is a point in a volume with no
  // transition dipole to orient. 2/3 is the isotropic dynamic average that a
  // published Forster radius already assumes, so the default changes no number.
  const kappa2 = n(project.fret?.kappa2, DEFAULT_KAPPA2);
  const savedKappa2 = n(savedProject.fret?.kappa2, DEFAULT_KAPPA2);

  function setFretEnabled(enabled: boolean) {
    setProject((current) => {
      const next = structuredClone(current);
      if (!next.fret) next.fret = { enabled, dyes: [] };
      next.fret.enabled = enabled;
      if (enabled) {
        const dyes = next.fret.dyes || [];
        if (dyes.length < 2) {
          // Seed the acceptor on the loaded structure, so enabling FRET yields
          // a project that validates rather than one pointing at residue 1 of
          // no chain.
          const acceptorDye = defaultAcceptorDye();
          const chain = selectedChain;
          // Seed on the *last labelable* residue, not the last one: structures
          // end in waters and ions, and a dye cannot be tethered to those.
          // Far from the donor is the useful default for a FRET pair.
          const residue = lastLabelableResidue(chain);
          if (chain && residue) {
            acceptorDye.attachment = {
              chain: chain.id,
              residue: residue.id,
              atom: firstUsefulAtom(residue),
            };
          }
          dyes.push(acceptorDye);
        }
        next.fret.dyes = dyes;
        if (!next.fret.R0_matrix?.[0]?.[1]) {
          next.fret.R0_matrix = forsterMatrix(DEFAULT_FORSTER_RADIUS);
        }
      }
      return next;
    });
  }

  function onAcceptorChainChange(chainId: string) {
    setProject((current) => {
      const next = structuredClone(current);
      const dye = next.fret?.dyes?.[1];
      if (!dye) return current;
      const chain = chains.find((candidate) => candidate.id === chainId);
      const residue = lastLabelableResidue(chain) || chain?.residues?.[0];
      dye.attachment = {
        chain: chainId,
        residue: residue?.id ?? 1,
        atom: residue ? firstUsefulAtom(residue) : "CB",
      };
      return next;
    });
  }

  function onAcceptorResidueChange(residueId: number) {
    setProject((current) => {
      const next = structuredClone(current);
      const dye = next.fret?.dyes?.[1];
      if (!dye) return current;
      const residue = acceptorResidues.find((candidate) => candidate.id === residueId);
      dye.attachment = {
        ...(dye.attachment || {}),
        residue: residueId,
        atom: residue ? firstUsefulAtom(residue) : dye.attachment?.atom || "CB",
      };
      return next;
    });
  }

  function setForsterRadius(value: string | number | boolean) {
    const radius = Number(value);
    if (!Number.isFinite(radius)) return;
    setProject((current) => {
      const next = structuredClone(current);
      if (!next.fret) next.fret = { enabled: false, dyes: [] };
      next.fret.R0_matrix = forsterMatrix(radius);
      return next;
    });
  }
  const [pdbId, setPdbId] = React.useState("");
  const [clientReady, setClientReady] = React.useState(false);
  const [theme, setTheme] = React.useState<"light" | "dark">("dark");
  const [tourOpen, setTourOpen] = React.useState(false);

  const startDemoTour = async () => {
    try {
      const res = await fetch("/api/demo-project");
      if (res.ok) {
        const data = await res.json();
        if (data?.project) {
          setProject(data.project);
        }
      }
    } catch (e) {
      console.warn("Could not fetch demo project:", e);
    }
    setTourOpen(true);
  };

  const [autoformSpec, setAutoformSpec] = React.useState<AutoFormSpec | null>(null);
  const router = useRouter();

  React.useEffect(() => {
    fetch(`/api/autoform-spec?locale=${locale}`)
      .then((res) => res.json())
      .then((data) => setAutoformSpec(data))
      .catch((err) => console.error("Failed to load AutoForm spec", err));
  }, [locale]);


  React.useEffect(() => {
    setClientReady(true);
  }, []);

  // Hydrate the theme from storage; default to dark, but never override the
  // OS/hard refresh with a stale guess once an explicit value is saved.
  React.useEffect(() => {
    const saved = window.localStorage.getItem("quest-theme");
    const initial: "light" | "dark" =
      saved === "light" || saved === "dark" ? saved : "dark";
    setTheme(initial);
  }, []);
  React.useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    window.localStorage.setItem("quest-theme", theme);
  }, [theme]);
  const runSimulationDisabled = clientReady && (!structure || status.kind === "busy");
  const downloadDisabled = clientReady && !result && !scanResults;

  function handleToggleJson(checked: boolean) {
    if (!checked) {
      const success = applyJson();
      if (!success) return;
    } else {
      setShowJson(true);
    }
  }

  return (
    <main className="app-shell">
      <input
        ref={pdbInputRef}
        type="file"
        accept=".pdb,.ent,.txt,.cif"
        style={{ display: "none" }}
        onChange={(event) =>
          event.target.files?.[0] && uploadPdb(event.target.files[0])
        }
      />

      <header className="topbar">
        <div className="brand-container">
          <img src="/logo.png" alt="QuEst Logo" className="brand-logo" />
          <div className="brand-text">
            <h1>{t("app.subtitle", "Quenching Estimator")}</h1>
            <p>{t("web.tagline", "Upload a structure, choose a labeling site, run quenching/FRET simulation.")}</p>
          </div>
        </div>
        <div className="topbar-actions">
          {/* Switching language refetches the catalog, so every parameter label
              and help text changes with it — the same translations the desktop
              form and doc/parameters.<code>.md are generated from. */}
          <div className="header-select-group" data-tour="language">
            <span className="header-label">{t("web.language", "Language")}:</span>
            <select
              value={locale}
              onChange={(event) => setLocale(event.target.value)}
              className="header-select"
              aria-label={t("web.language", "Language")}
              data-testid="locale-select"
            >
              {Object.entries(locales).map(([code, name]) => (
                <option key={code} value={code}>
                  {name}
                </option>
              ))}
            </select>
          </div>
          <button
            className="theme-toggle"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            aria-label={t("web.theme_toggle", "Toggle light/dark mode")}
            title={t("web.theme_toggle", "Toggle light/dark mode")}
            data-testid="theme-toggle"
            data-tour="theme"
          >
            {theme === "dark" ? (
              <svg className="btn-icon" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="5"></circle>
                <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"></path>
              </svg>
            ) : (
              <svg className="btn-icon" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
              </svg>
            )}
          </button>
          <button
            className="theme-toggle"
            onClick={startDemoTour}
            aria-label={t("web.guide", "Guide")}
            title={t("web.guide", "Guide")}
            data-testid="guide-btn"
          >
            <svg className="btn-icon" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="3"></circle>
              <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
            </svg>
          </button>
          <div className="header-select-group">
            <span className="header-label">Dye:</span>
            <select
              value={selectedDyeName}
              onChange={(event) => handleDyeChange(event.target.value)}
              className="header-select"
              data-testid="dye-select"
              aria-label={t("web.dye", "Dye")}
            >
              <option value="Custom">Custom...</option>
              {Object.keys(dyes).map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          </div>
          <button
            onClick={saveDyePreset}
            title="Save current parameters as a dye preset"
            style={{ minHeight: "26px", fontSize: "13px", display: "flex", alignItems: "center", gap: "4px" }}
            data-tour="preset"
          >
            💾 Disk Preset
          </button>
          <div style={{ display: "flex", gap: "6px" }}>
            <button
              className="primary"
              onClick={() => runSimulation()}
              disabled={runSimulationDisabled}
              style={{ fontWeight: "bold", display: "flex", alignItems: "center", gap: "4px", flex: 1 }}
              data-tour="run"
            >
              {status.kind === "busy" ? (
                "Running..."
              ) : (
                <>
                  <span style={{ fontSize: "12px" }}>▶</span> Run Sim
                </>
              )}
            </button>
            {status.kind === "busy" && (
              <button
                style={{
                  backgroundColor: "var(--red, #e11d48)",
                  borderColor: "var(--red, #e11d48)",
                  color: "white",
                  fontWeight: "bold",
                  display: "flex",
                  alignItems: "center",
                }}
                onClick={abortSimulation}
              >
                Stop
              </button>
            )}
          </div>
          <button
            onClick={() => setShowCitationModal(true)}
            className="cite-btn"
            title="Show citation details"
            data-tour="cite"
          >
            <svg
              className="btn-icon"
              viewBox="0 0 24 24"
              width="14"
              height="14"
              stroke="currentColor"
              strokeWidth="2"
              fill="none"
              strokeLinecap="round"
              strokeLinejoin="round"
              style={{ marginRight: "6px" }}
            >
              <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"></path>
              <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"></path>
            </svg>
            Cite
          </button>
          <div className={`status ${status.kind}`}>{status.message}</div>
        </div>
      </header>

      <div className="workflow-tab-container">
        <div className="app-tabs" data-tour="tabs">
          <Link
            href="/labeling"
            className={`tab-btn ${currentTab === "/labeling" ? "active" : ""}`}
          >
            {t("web.tab.labeling", "1 - Labeling")}
          </Link>
          <Link
            href="/simulation"
            className={`tab-btn ${currentTab === "/simulation" ? "active" : ""}`}
          >
            {t("web.tab.simulation", "2 - Simulation")}
          </Link>
          <Link
            href="/scan"
            className={`tab-btn ${currentTab === "/scan" ? "active" : ""}`}
          >
            {t("web.tab.scan", "3 - Scan")}
          </Link>
          <Link
            href="/advanced"
            className={`tab-btn ${currentTab === "/advanced" ? "active" : ""}`}
          >
            {t("web.tab.advanced", "4 - Adv. Settings")}
          </Link>
        </div>
        <div className="global-actions">
          <form
            className="pdb-id-form"
            onSubmit={(event) => {
              event.preventDefault();
              void loadPdbId(pdbId);
            }}
            data-tour="pdb"
          >
            <input
              value={pdbId}
              onChange={(event) => setPdbId(event.target.value)}
              placeholder="PDB ID"
              aria-label="PDB ID"
              maxLength={4}
            />
            <button type="submit">Load</button>
          </form>
          <button onClick={() => pdbInputRef.current?.click()} data-tour="upload">Upload Structure (PDB/CIF)</button>
          <button onClick={downloadResult} disabled={downloadDisabled} data-tour="download">
            Download result
          </button>
          <button onClick={handleClearState} className="clear-btn">
            Clear
          </button>
        </div>
      </div>

      <section className="main-grid">
        <aside className="control-panel">
          <div className="sidebar-header-toggle" data-tour="json" style={{ display: "flex", flexWrap: "wrap", gap: "8px", alignItems: "center", justifyContent: "space-between" }}>
            <h2>Configuration</h2>
            <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
              <label className="toggle-switch">
                <span className="toggle-label">JSON</span>
                <div className="switch">
                  <input
                    type="checkbox"
                    checked={showJson}
                    onChange={(event) => handleToggleJson(event.target.checked)}
                  />
                  <span className="slider"></span>
                </div>
              </label>
            </div>
          </div>

          <div
            className="panel-section"
            style={{ borderBottom: "1px solid var(--line)", background: "var(--surface-2)" }}
            data-tour="runs"
          >
            <h2>{t("web.card.runs", "Simulation Runs")}</h2>
            <div className="field" style={{ margin: 0 }}>
              <select
                value={selectedJobId || ""}
                onChange={(e) => handleLoadJob(e.target.value)}
                data-testid="run-select"
                aria-label={t("web.run_select", "Select a previous simulation run")}
              >
                <option value="">-- Active/New Run --</option>
                {jobs.map((job) => (
                  <option key={job.job_id} value={job.job_id}>
                    Site: {job.labeling_site} ({job.date})
                  </option>
                ))}
              </select>
              {jobsTotal > jobs.length && (
                <span className="hint">
                  showing the {jobs.length} most recent of {jobsTotal} runs
                </span>
              )}
            </div>
          </div>
          {!showJson ? (
            <div className="panel-section autoform-wrapper" style={{ padding: "12px" }}>
              {autoformSpec ? (
                <AutoForm
                  spec={autoformSpec}
                  data={project}
                  onChange={(path, val) => {
                    // fret_R0 is a single number stored as the donor->acceptor
                    // cell of a symmetric matrix; setNested cannot index a list.
                    if (path === "fret.R0_matrix.0.1") {
                      setForsterRadius(Number(val));
                      return;
                    }
                    // Enabling FRET must seed the acceptor dye and R0, exactly
                    // like the legacy toggle did; a bare fret.enabled=true
                    // leaves a project with one dye that validates but cannot
                    // transfer.
                    if (path === "fret.enabled") {
                      setFretEnabled(Boolean(val));
                      return;
                    }
                    updateProject(path, val);
                  }}
                  catalog={catalog || undefined}
                />
              ) : (
                <div style={{ padding: "12px", color: "var(--text-muted)" }}>Loading AutoForm spec...</div>
              )}
            </div>
          ) : (
            <section className="json-panel">
              <textarea
                value={projectText}
                onChange={(event) => setProjectText(event.target.value)}
                spellCheck={false}
              />
              <button onClick={applyJson}>Apply JSON</button>
            </section>
          )}
        </aside>

        <section className="workbench">{children}</section>
      </section>

      {showCitationModal && (
        <div className="citation-modal-overlay" onClick={() => setShowCitationModal(false)}>
          <div className="citation-modal" onClick={(e) => e.stopPropagation()}>
            <div className="citation-modal-header">
              <h2>Citation Information</h2>
              <button className="close-btn" onClick={() => setShowCitationModal(false)}>
                &times;
              </button>
            </div>
            <div className="citation-modal-body">
              <p>
                If you use <strong>QuEst</strong> in a scientific publication, please cite:
              </p>
              <blockquote className="citation-text">
                Peulen, T.O., Opanasyuk, O., and Seidel, C.A., 2017. Combining Graphical and Analytical
                Methods with Molecular Simulations To Analyze Time-Resolved FRET Measurements of Labeled
                Macromolecules Accurately. <em>J. Phys. Chem. B</em> 2017, 121, 35, 8211-8241.
              </blockquote>
              <div className="citation-modal-actions">
                <a
                  href="https://pubs.acs.org/doi/10.1021/acs.jpcb.7b03441"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="primary-btn-link"
                >
                  Open Publication URL
                </a>
                <button
                  onClick={() => {
                    navigator.clipboard.writeText(
                      "Peulen, T.O., Opanasyuk, O., and Seidel, C.A., 2017. Combining Graphical and Analytical Methods with Molecular Simulations To Analyze Time-Resolved FRET Measurements of Labeled Macromolecules Accurately. J. Phys. Chem. B 2017, 121, 35, 8211-8241."
                    );
                    alert("Citation copied to clipboard!");
                  }}
                >
                  Copy Citation Text
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      <Tour
        open={tourOpen}
        steps={getTourSteps(t, (path) => {
          if (window.location.pathname !== path) router.push(path);
        })}
        labels={getTourLabels(t)}
        onClose={() => setTourOpen(false)}
      />
    </main>
  );
}
