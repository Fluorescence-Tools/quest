"use client";

import React from "react";
import Link from "next/link";
import { useAppContext } from "../../context/AppContext";
import Field from "../../components/Field";
import QuenchingEditor from "../../components/QuenchingEditor";
import {
  MAX_PARALLEL_TRAJECTORIES,
  parallelTrajectories as normalizeParallelTrajectories,
} from "../../lib/projectHelpers";

export default function AdvancedPage() {
  const {
    autoSimulate,
    catalog,
    project,
    renderSparsity,
    saveParameter,
    savedProject,
    setAutoSimulate,
    setRenderSparsity,
    setSurfaceRendering,
    structure,
    surfaceRendering,
    updateProject,
  } = useAppContext();
  const parallelTrajectories = normalizeParallelTrajectories(project.parallel_trajectories);
  const savedParallelTrajectories = normalizeParallelTrajectories(savedProject.parallel_trajectories);

  return (
    <div className="advanced-settings-page">
      <section className="result-card advanced-settings-card quench-editor-card" data-tour="quench-editor">
        <div className="card-header">
          <h2>Amino-acid specific quenching</h2>
        </div>
        <div className="advanced-settings-body">
          <QuenchingEditor />
        </div>
      </section>

      <section className="result-card advanced-settings-card" data-tour="adv-config">
        <div className="card-header">
          <h2>Simulation Configuration</h2>
        </div>
        <div className="advanced-settings-body">
          <label className="check-field">
            <input
              type="checkbox"
              checked={autoSimulate}
              onChange={(event) => setAutoSimulate(event.target.checked)}
            />
            Auto-update simulation on changes
          </label>

          <label className="check-field">
            <input
              type="checkbox"
              checked={Boolean(project.coarse_grained)}
              onChange={(event) => updateProject("coarse_grained", event.target.checked)}
            />
            Coarse-grained structure (backbone + CB)
          </label>
          <span className="hint">
            CB is moved to the side-chain centre of mass and side-chain atoms are dropped, so every
            quenching centre collapses onto that pseudo-atom and the accessible volume grows. A
            modelling choice, not a speed-up.
          </span>

          <Field
            label="Parallel trajectories"
            value={parallelTrajectories}
            step="1"
            min={-1}
            max={MAX_PARALLEL_TRAJECTORIES}
            onChange={(value) => updateProject("parallel_trajectories", normalizeParallelTrajectories(value))}
            parameterId="parallel_trajectories"
            catalog={catalog}
            onSave={() => saveParameter("parallel_trajectories")}
            isDirty={parallelTrajectories !== savedParallelTrajectories}
          />
        </div>
      </section>

      <section className="result-card advanced-settings-card">
        <div className="card-header">
          <h2>Visualization Configuration</h2>
        </div>
        <div className="advanced-settings-body">
          <label className="check-field">
            <input
              type="checkbox"
              checked={surfaceRendering}
              onChange={(event) => setSurfaceRendering(event.target.checked)}
            />
            Use surface rendering for AV/ACV
          </label>

          <Field
            label="AV/ACV sparsing factor"
            value={renderSparsity}
            step="1"
            min={1}
            max={500}
            onChange={(value) => setRenderSparsity(Math.max(1, Math.floor(value || 1)))}
          />
        </div>
      </section>

      <section className="result-card advanced-settings-card">
        <div className="card-header">
          <h2>Structure Information</h2>
        </div>
        <div className="advanced-settings-body">
          {structure ? (
            <table className="structure-info-table">
              <tbody>
                <tr>
                  <td>Filename</td>
                  <td>{structure.filename}</td>
                </tr>
                <tr>
                  <td>Local Path</td>
                  <td className="monospace-cell">{structure.path}</td>
                </tr>
                <tr>
                  <td>Total Chains</td>
                  <td>{structure.chains.length}</td>
                </tr>
                <tr>
                  <td>Total Atoms</td>
                  <td>{structure.atom_count}</td>
                </tr>
              </tbody>
            </table>
          ) : (
            <p className="advanced-empty-copy">
              No active structure loaded. <Link href="/labeling">Upload one here</Link> to view metadata.
            </p>
          )}
        </div>
      </section>
    </div>
  );
}
