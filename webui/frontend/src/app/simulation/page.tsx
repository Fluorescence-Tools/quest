"use client";

import React from "react";
import { useAppContext } from "../../context/AppContext";
import {
  AAResidenceTable,
  AutocorrPlot,
  DecayPlot,
  Metrics,
  QuenchingBreakdown,
  TrajectoryPlot,
} from "../../components/Plots";

export default function SimulationPage() {
  const { result } = useAppContext();

  return (
    <div className="simulation-page">
      <div className="simulation-primary">
        <div className="result-row simulation-summary-row">
          <div className="result-card">
            <div className="card-header">
              <h2>Decay Curve</h2>
              <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
                <span style={{ display: "inline-flex", alignItems: "center", gap: "4px", fontSize: "12px" }}>
                  <span style={{ display: "inline-block", width: "12px", height: "4px", backgroundColor: "var(--accent)" }}></span>
                  Donor (sim)
                </span>
                {result?.fret_counts && (
                  <span style={{ display: "inline-flex", alignItems: "center", gap: "4px", fontSize: "12px" }}>
                    <span style={{ display: "inline-block", width: "12px", height: "4px", backgroundColor: "var(--green)" }}></span>
                    FRET (sim)
                  </span>
                )}
              </div>
            </div>
            <DecayPlot result={result} />
          </div>
          <Metrics result={result} />
        </div>

        <div className="result-card">
          <div className="card-header">
            <h2>Quenching by residue type</h2>
            <span className="card-subtitle">share of the time-averaged quenching rate</span>
          </div>
          <QuenchingBreakdown result={result} />
        </div>

        <div className="simulation-chart-grid">
          <div className="result-card">
            <div className="card-header">
              <h2>Dye Distance to Mean Location</h2>
            </div>
            <TrajectoryPlot result={result} />
          </div>

          <div className="result-card">
            <div className="card-header">
              <h2>Trajectory Autocorrelation</h2>
            </div>
            <AutocorrPlot result={result} />
          </div>
        </div>
      </div>

      <div className="result-card aa-residence-card">
        <div className="card-header">
          <h2>AA Residence</h2>
        </div>
        <AAResidenceTable result={result} />
      </div>
    </div>
  );
}
