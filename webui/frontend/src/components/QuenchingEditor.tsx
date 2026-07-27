"use client";

import React from "react";
import { useAppContext } from "../context/AppContext";
import {
  AMINO_ACID_RESIDUES,
  DEFAULT_QUENCH_ATOMS,
  PET_ACTIVE_RESIDUES,
  buildAminoAcidInteractions,
  n,
  normalizeAminoAcidInteractions,
  quenchAtoms,
  quenchRadius,
  slowFactor,
} from "../lib/projectHelpers";
import { AminoAcidInteraction } from "../types";

const RESIDUE_LABELS: Record<string, string> = {
  ALA: "Alanine",
  ARG: "Arginine",
  ASN: "Asparagine",
  ASP: "Aspartate",
  CYS: "Cysteine",
  GLN: "Glutamine",
  GLU: "Glutamate",
  GLY: "Glycine",
  HIS: "Histidine",
  ILE: "Isoleucine",
  LEU: "Leucine",
  LYS: "Lysine",
  MET: "Methionine",
  PHE: "Phenylalanine",
  PRO: "Proline",
  SER: "Serine",
  THR: "Threonine",
  TRP: "Tryptophan",
  TYR: "Tyrosine",
  VAL: "Valine",
};

const MOIETY_LABELS: Record<string, string> = {
  TRP: "indole ring",
  TYR: "phenol ring",
  HIS: "imidazole ring",
  MET: "thioether S",
  CYS: "thiol S",
  PRO: "pyrrolidine ring",
  PHE: "phenyl ring",
  ARG: "guanidinium",
  LYS: "ammonium",
  ASP: "carboxylate",
  GLU: "carboxylate",
  ASN: "amide",
  GLN: "amide",
  SER: "hydroxyl",
  THR: "hydroxyl",
};

type Filter = "quenching" | "all";

export default function QuenchingEditor() {
  const { project, savedProject, updateProject, setProject, saveParameter, quenchingDefaults } =
    useAppContext();
  const [filter, setFilter] = React.useState<Filter>("quenching");
  const [expanded, setExpanded] = React.useState<string | null>(null);

  const interactions = normalizeAminoAcidInteractions(project.amino_acid_interactions);
  const savedInteractions = normalizeAminoAcidInteractions(savedProject.amino_acid_interactions);
  const criticalDistance = n(project.critical_distance, 8.5);
  const dyeRadius = n(project.av_parameter?.radius1, 3.5);

  const defaultAtomsFor = (residue: string) =>
    quenchingDefaults?.default_quench_atoms?.[residue] ?? DEFAULT_QUENCH_ATOMS[residue] ?? ["CB"];

  const activeCount = AMINO_ACID_RESIDUES.filter((residue) => interactions[residue].kQ > 0).length;
  const visibleResidues = AMINO_ACID_RESIDUES.filter((residue) =>
    filter === "all" ? true : interactions[residue].kQ > 0 || PET_ACTIVE_RESIDUES.includes(residue)
  );

  function setInteraction(residue: string, patch: Partial<AminoAcidInteraction>) {
    setProject((current) => {
      const next = structuredClone(current);
      const table = normalizeAminoAcidInteractions(next.amino_acid_interactions);
      table[residue] = { ...table[residue], ...patch };
      next.amino_acid_interactions = table;
      if (next.fret?.dyes?.[0]) next.fret.dyes[0].amino_acid_interactions = table;
      return next;
    });
  }

  function applyReferenceChemistry() {
    const table = buildAminoAcidInteractions({
      slowFactorValue: interactions.ALA.slow_factor,
      dyeRadius,
    });
    setProject((current) => {
      const next = structuredClone(current);
      next.amino_acid_interactions = table;
      if (next.fret?.dyes?.[0]) next.fret.dyes[0].amino_acid_interactions = table;
      return next;
    });
  }

  function clearAllQuenching() {
    setProject((current) => {
      const next = structuredClone(current);
      const table = normalizeAminoAcidInteractions(next.amino_acid_interactions);
      for (const residue of AMINO_ACID_RESIDUES) table[residue].kQ = 0;
      next.amino_acid_interactions = table;
      if (next.fret?.dyes?.[0]) next.fret.dyes[0].amino_acid_interactions = table;
      return next;
    });
  }

  const isDirty = (residue: string) =>
    JSON.stringify(interactions[residue]) !== JSON.stringify(savedInteractions[residue]);

  return (
    <div className="quench-editor">
      <div className="quench-editor-intro">
        <p>
          Quenching is resolved per residue type. Each residue quenches at rate <strong>k<sub>Q</sub></strong>{" "}
          while the dye centre is within its <strong>contact radius</strong> of that residue&rsquo;s{" "}
          <strong>quenching centre</strong> &mdash; the centroid of its redox-active atoms, not CB.
          Contributions from overlapping residues add up.
        </p>
        <div className="quench-editor-stats">
          <span className="pill pill-accent">{activeCount} quenching residue types</span>
          <span className="pill">dye radius {dyeRadius.toFixed(1)} Å</span>
          <span className="pill">fallback radius {criticalDistance.toFixed(1)} Å</span>
        </div>
      </div>

      <div className="quench-editor-toolbar">
        <div className="segmented" role="group" aria-label="Filter residues">
          <button
            type="button"
            className={filter === "quenching" ? "active" : ""}
            onClick={() => setFilter("quenching")}
          >
            PET-active
          </button>
          <button
            type="button"
            className={filter === "all" ? "active" : ""}
            onClick={() => setFilter("all")}
          >
            All 20
          </button>
        </div>
        <div className="quench-editor-actions">
          <button type="button" onClick={applyReferenceChemistry} title="Reset to the reference xanthene-dye PET parameters, scaled to this dye's radius">
            Reset to reference
          </button>
          <button type="button" onClick={clearAllQuenching} title="Set every quenching rate to zero">
            Clear quenching
          </button>
          <button type="button" className="primary" onClick={() => saveParameter("amino_acid_interactions")}>
            Save as dye preset
          </button>
        </div>
      </div>

      <div className="quench-table-wrap">
        <table className="quench-table">
          <thead>
            <tr>
              <th scope="col">Residue</th>
              <th scope="col">
                k<sub>Q</sub> <span className="unit">1/ns</span>
              </th>
              <th scope="col">
                Contact radius <span className="unit">Å</span>
              </th>
              <th scope="col">
                Slow factor <span className="unit">0–1</span>
              </th>
              <th scope="col">Quenching centre</th>
            </tr>
          </thead>
          <tbody>
            {visibleResidues.map((residue) => {
              const current = interactions[residue];
              const defaults = defaultAtomsFor(residue);
              const usesDefaultAtoms =
                JSON.stringify(current.quench_atoms) === JSON.stringify(defaults);
              const quenches = current.kQ > 0;
              return (
                <React.Fragment key={residue}>
                  <tr className={`${quenches ? "is-quenching" : ""} ${isDirty(residue) ? "is-dirty" : ""}`}>
                    <th scope="row">
                      <span className="residue-code">{residue}</span>
                      <span className="residue-name">{RESIDUE_LABELS[residue]}</span>
                    </th>
                    <td>
                      <input
                        type="number"
                        step="0.1"
                        min={0}
                        aria-label={`${residue} quenching rate`}
                        value={current.kQ}
                        onChange={(event) =>
                          setInteraction(residue, { kQ: Math.max(0, n(parseFloat(event.target.value), 0)) })
                        }
                      />
                    </td>
                    <td>
                      <input
                        type="number"
                        step="0.1"
                        min={0}
                        aria-label={`${residue} contact radius`}
                        placeholder={criticalDistance.toFixed(1)}
                        value={current.quench_radius ?? ""}
                        onChange={(event) =>
                          setInteraction(residue, { quench_radius: quenchRadius(event.target.value) })
                        }
                      />
                      {current.quench_radius === null && <span className="hint">inherited</span>}
                    </td>
                    <td>
                      <input
                        type="number"
                        step="0.01"
                        min={0}
                        max={1}
                        aria-label={`${residue} slow factor`}
                        value={current.slow_factor}
                        onChange={(event) =>
                          setInteraction(residue, { slow_factor: slowFactor(parseFloat(event.target.value), 1) })
                        }
                      />
                    </td>
                    <td>
                      <button
                        type="button"
                        className="atom-summary"
                        onClick={() => setExpanded(expanded === residue ? null : residue)}
                        aria-expanded={expanded === residue}
                        title="Edit the atoms that define the quenching centre"
                      >
                        <span className="atom-moiety">{MOIETY_LABELS[residue] ?? "side chain"}</span>
                        <span className="atom-count">
                          {current.quench_atoms.length} atom{current.quench_atoms.length === 1 ? "" : "s"}
                        </span>
                        {!usesDefaultAtoms && <span className="pill pill-warn">custom</span>}
                      </button>
                    </td>
                  </tr>
                  {expanded === residue && (
                    <tr className="quench-atom-row">
                      <td colSpan={5}>
                        <label className="quench-atom-editor">
                          <span>
                            Atoms defining the {residue} quenching centre (centroid of those present)
                          </span>
                          <input
                            type="text"
                            value={current.quench_atoms.join(" ")}
                            onChange={(event) =>
                              setInteraction(residue, {
                                quench_atoms: quenchAtoms(event.target.value, defaults),
                              })
                            }
                          />
                        </label>
                        <div className="quench-atom-footer">
                          <span className="hint">Default: {defaults.join(" ")}</span>
                          <button
                            type="button"
                            disabled={usesDefaultAtoms}
                            onClick={() => setInteraction(residue, { quench_atoms: [...defaults] })}
                          >
                            Restore default atoms
                          </button>
                        </div>
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              );
            })}
          </tbody>
        </table>
      </div>

      <label className="field quench-fallback">
        <span>Fallback contact radius [Å]</span>
        <input
          type="number"
          step="0.1"
          min={0}
          value={criticalDistance}
          onChange={(event) => updateProject("critical_distance", n(parseFloat(event.target.value), 8.5))}
        />
        <span className="hint">
          Used by residues with no explicit contact radius. It is measured from the dye centre, so it
          must exceed the dye radius ({dyeRadius.toFixed(1)} Å) plus van der Waals contact.
        </span>
      </label>
    </div>
  );
}
