"use client";

import React, { useState, useMemo } from "react";
import { useAppContext } from "../../context/AppContext";
import { ScanResultItem } from "../../types";
import { syncDonor } from "../../lib/projectHelpers";

export default function ScanPage() {
  const {
    scanResults,
    setProject,
    structure,
    scanAll,
    setScanAll,
    selectedScanResidues,
    setSelectedScanResidues,
    scanProgress,
    scanStatus,
    isScanning,
    allResidues,
    runScan,
    abortScan,
    copyScanResultsToClipboard,
  } = useAppContext();

  const [scanSortConfig, setScanSortConfig] = useState<{
    key: keyof ScanResultItem | null;
    direction: "asc" | "desc";
  }>({ key: "residue", direction: "asc" });

  const [scanFilterResidue, setScanFilterResidue] = useState("");
  const [scanFilterStatus, setScanFilterStatus] = useState("");
  const [scanFilterLifetime, setScanFilterLifetime] = useState("");
  const [scanFilterQY, setScanFilterQY] = useState("");
  const [scanFilterCF, setScanFilterCF] = useState("");
  const [scanFilterAVVol, setScanFilterAVVol] = useState("");
  const [scanFilterContVol, setScanFilterContVol] = useState("");
  const [scanFilterRatio, setScanFilterRatio] = useState("");
  const [scanFilterQuencher, setScanFilterQuencher] = useState("");

  const matchNumericFilter = (val: number | null, filterStr: string): boolean => {
    if (val === null || val === undefined) return false;
    const trimmed = filterStr.trim();
    if (!trimmed) return true;

    // Range support: e.g. "3-4", "3..4", "3:4"
    const rangeMatch = trimmed.match(/^([\d.]+)\s*[-:..]+\s*([\d.]+)$/);
    if (rangeMatch) {
      const minVal = parseFloat(rangeMatch[1]);
      const maxVal = parseFloat(rangeMatch[2]);
      return (!isNaN(minVal) ? val >= minVal : true) && (!isNaN(maxVal) ? val <= maxVal : true);
    }

    // Operators: >=, <=, >, <, =
    if (trimmed.startsWith(">=")) {
      const minVal = parseFloat(trimmed.slice(2).trim());
      return isNaN(minVal) ? true : val >= minVal;
    }
    if (trimmed.startsWith("<=")) {
      const maxVal = parseFloat(trimmed.slice(2).trim());
      return isNaN(maxVal) ? true : val <= maxVal;
    }
    if (trimmed.startsWith(">")) {
      const minVal = parseFloat(trimmed.slice(1).trim());
      return isNaN(minVal) ? true : val > minVal;
    }
    if (trimmed.startsWith("<")) {
      const maxVal = parseFloat(trimmed.slice(1).trim());
      return isNaN(maxVal) ? true : val < maxVal;
    }
    if (trimmed.startsWith("=")) {
      const matchVal = parseFloat(trimmed.slice(1).trim());
      return isNaN(matchVal) ? true : Math.abs(val - matchVal) < 1e-5;
    }

    // Default: treat as minimum threshold (greater than or equal to)
    const defaultVal = parseFloat(trimmed);
    return isNaN(defaultVal) ? true : val >= defaultVal;
  };

  const processedScanResults = useMemo(() => {
    if (!scanResults) return null;

    let filtered = scanResults.map((r) => ({
      ...r,
      lifetime_val: r.lifetime !== null ? Number(r.lifetime) : null,
      qy_val: r.quantum_yield !== null ? Number(r.quantum_yield) : null,
      cf_val: r.contact_fraction !== null ? Number(r.contact_fraction) : null,
      av_vol_val: r.av_volume !== null && r.av_volume !== undefined ? Number(r.av_volume) : null,
      cont_vol_val: r.contact_volume !== null && r.contact_volume !== undefined ? Number(r.contact_volume) : null,
      ratio_val: r.contact_to_free_ratio !== null && r.contact_to_free_ratio !== undefined ? Number(r.contact_to_free_ratio) : null,
    }));

    // 1. Filter
    filtered = filtered.filter((r) => {
      if (scanFilterResidue) {
        const query = scanFilterResidue.toLowerCase();
        const resStr = `${r.name || ""} ${r.chain}:${r.residue} ${r.residue_name || ""}`.toLowerCase();
        if (!resStr.includes(query)) return false;
      }
      if (scanFilterStatus && r.status !== scanFilterStatus) {
        return false;
      }
      if (scanFilterLifetime && !matchNumericFilter(r.lifetime_val, scanFilterLifetime)) {
        return false;
      }
      if (scanFilterQY && !matchNumericFilter(r.qy_val, scanFilterQY)) {
        return false;
      }
      if (scanFilterCF && !matchNumericFilter(r.cf_val, scanFilterCF)) {
        return false;
      }
      if (scanFilterAVVol && !matchNumericFilter(r.av_vol_val, scanFilterAVVol)) {
        return false;
      }
      if (scanFilterContVol && !matchNumericFilter(r.cont_vol_val, scanFilterContVol)) {
        return false;
      }
      if (scanFilterRatio && !matchNumericFilter(r.ratio_val, scanFilterRatio)) {
        return false;
      }
      if (scanFilterQuencher) {
        const query = scanFilterQuencher.toLowerCase();
        const names = (r.quenchers ?? []).map((q) => q.residue_name).join(" ").toLowerCase();
        if (!names.includes(query)) return false;
      }
      return true;
    });

    // 2. Sort
    if (scanSortConfig.key) {
      const key = scanSortConfig.key;
      const direction = scanSortConfig.direction === "asc" ? 1 : -1;
      filtered.sort((a, b) => {
        let valA: any = a[key];
        let valB: any = b[key];

        if (key === "lifetime") { valA = a.lifetime_val; valB = b.lifetime_val; }
        else if (key === "quantum_yield") { valA = a.qy_val; valB = b.qy_val; }
        else if (key === "contact_fraction") { valA = a.cf_val; valB = b.cf_val; }
        else if (key === "av_volume") { valA = a.av_vol_val; valB = b.av_vol_val; }
        else if (key === "contact_volume") { valA = a.cont_vol_val; valB = b.cont_vol_val; }
        else if (key === "contact_to_free_ratio") { valA = a.ratio_val; valB = b.ratio_val; }

        if (valA === null || valA === undefined) return 1;
        if (valB === null || valB === undefined) return -1;

        if (typeof valA === "string" && typeof valB === "string") {
          return valA.localeCompare(valB) * direction;
        }
        return (valA - valB) * direction;
      });
    }

    return filtered;
  }, [
    scanResults,
    scanSortConfig,
    scanFilterResidue,
    scanFilterStatus,
    scanFilterLifetime,
    scanFilterQY,
    scanFilterCF,
    scanFilterAVVol,
    scanFilterContVol,
    scanFilterRatio,
    scanFilterQuencher,
  ]);

  const getSortIcon = (key: keyof ScanResultItem) => {
    if (scanSortConfig.key !== key) return " ↕";
    return scanSortConfig.direction === "asc" ? " ▲" : " ▼";
  };

  const handleSort = (key: keyof ScanResultItem) => {
    let direction: "asc" | "desc" = "asc";
    if (scanSortConfig.key === key && scanSortConfig.direction === "asc") {
      direction = "desc";
    }
    setScanSortConfig({ key, direction });
  };

  const handleRowClick = (item: ScanResultItem) => {
    setProject((current) => {
      const next = structuredClone(current);
      next.attachment = { chain: item.chain, residue: item.residue, atom: item.atom };
      syncDonor(next);
      return next;
    });
  };

  return (
    <div className="results-section" data-tour="scan-page">
      <section className="result-card" data-tour="scan-settings" style={{ marginBottom: "16px" }}>
        <div className="card-header">
          <h2>Scan Settings</h2>
        </div>
        <div className="advanced-settings-body">
          <label className="check-field" style={{ marginBottom: "8px" }}>
            <input
              type="checkbox"
              checked={scanAll}
              onChange={(e) => setScanAll(e.target.checked)}
              disabled={!structure}
            />
            <span>Scan all residues ({allResidues.length})</span>
          </label>

          {!scanAll && structure && (
            <div className="scan-residues-selection">
              <div className="scan-selection-actions">
                <button
                  type="button"
                  onClick={() => {
                    const next: Record<string, boolean> = {};
                    allResidues.forEach((r) => { next[r.key] = true; });
                    setSelectedScanResidues(next);
                  }}
                >
                  Select All
                </button>
                <button
                  type="button"
                  onClick={() => {
                    const next: Record<string, boolean> = {};
                    allResidues.forEach((r) => { next[r.key] = false; });
                    setSelectedScanResidues(next);
                  }}
                >
                  Select None
                </button>
              </div>
              <div className="scan-residues-list">
                {allResidues.map((res) => (
                  <label key={res.key} className="scan-residue-item">
                    <input
                      type="checkbox"
                      checked={Boolean(selectedScanResidues[res.key])}
                      onChange={(e) => {
                        setSelectedScanResidues((curr) => ({
                          ...curr,
                          [res.key]: e.target.checked,
                        }));
                      }}
                    />
                    <span>
                      {res.chain}:{res.residue} ({res.residue_name})
                    </span>
                  </label>
                ))}
              </div>
            </div>
          )}

          <div style={{ marginTop: "12px", display: "flex", gap: "8px" }}>
            <button
              className="primary"
              style={{ flex: 1, fontWeight: "bold" }}
              onClick={runScan}
              disabled={!structure || isScanning}
            >
              {isScanning ? "Scanning..." : "Calculate Scan"}
            </button>
            {isScanning && (
              <button
                style={{
                  backgroundColor: "var(--red, #e11d48)",
                  borderColor: "var(--red, #e11d48)",
                  color: "white",
                  fontWeight: "bold",
                }}
                onClick={abortScan}
              >
                Stop
              </button>
            )}
          </div>
          {scanProgress && (
            <div style={{ marginTop: "10px" }}>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  fontSize: "11px",
                  marginBottom: "4px",
                  color: "var(--muted)",
                }}
              >
                <span>Scan Progress</span>
                <span>
                  {scanProgress.current} / {scanProgress.total} (
                  {Math.round((scanProgress.current / (scanProgress.total || 1)) * 100)}%)
                </span>
              </div>
              <div
                style={{
                  width: "100%",
                  height: "8px",
                  backgroundColor: "var(--line)",
                  borderRadius: "4px",
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    width: `${(scanProgress.current / (scanProgress.total || 1)) * 100}%`,
                    height: "100%",
                    backgroundColor: "var(--accent)",
                    transition: "width 0.2s ease-in-out",
                  }}
                />
              </div>
            </div>
          )}
          <div
            className={`status ${scanStatus.kind}`}
            style={{ marginTop: "10px", width: "100%", maxWidth: "none" }}
          >
            {scanStatus.message}
          </div>
        </div>
      </section>
      <div className="results-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
        <div>
          <h2 style={{ margin: 0 }}>Scan Results</h2>
          <p style={{ margin: "4px 0 0", color: "var(--muted)", fontSize: "12px" }}>
            {scanResults ? `${scanResults.length} residues scanned` : "No scan run yet"}
          </p>
        </div>
        {scanResults && (
          <button onClick={copyScanResultsToClipboard} className="cite-btn">
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
              style={{ marginRight: "6px", display: "inline-block", verticalAlign: "middle" }}
            >
              <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
            </svg>
            Copy to Clipboard
          </button>
        )}
      </div>

      {scanResults ? (
        <div className="scan-table-container">
          <table className="scan-results-table">
            <thead>
              <tr>
                <th className="sortable" onClick={() => handleSort("residue")}>
                  Residue{getSortIcon("residue")}
                </th>
                <th className="sortable" onClick={() => handleSort("status")}>
                  Status{getSortIcon("status")}
                </th>
                <th className="sortable" onClick={() => handleSort("lifetime")}>
                  Lifetime [ns]{getSortIcon("lifetime")}
                </th>
                <th className="sortable" onClick={() => handleSort("quantum_yield")}>
                  QY{getSortIcon("quantum_yield")}
                </th>
                <th className="sortable" onClick={() => handleSort("contact_fraction")}>
                  Coll. Frac.{getSortIcon("contact_fraction")}
                </th>
                <th className="sortable" onClick={() => handleSort("av_volume")}>
                  AV Vol [Å³]{getSortIcon("av_volume")}
                </th>
                <th className="sortable" onClick={() => handleSort("contact_volume")}>
                  Cont. Vol [Å³]{getSortIcon("contact_volume")}
                </th>
                <th className="sortable" onClick={() => handleSort("contact_to_free_ratio")}>
                  Cont/Free Ratio{getSortIcon("contact_to_free_ratio")}
                </th>
                <th className="sortable" onClick={() => handleSort("dominant_quencher")}>
                  Quenched by{getSortIcon("dominant_quencher")}
                </th>
              </tr>
              <tr className="filter-row">
                <th>
                  <input
                    type="text"
                    placeholder="Filter..."
                    value={scanFilterResidue}
                    onChange={(e) => setScanFilterResidue(e.target.value)}
                    className="column-filter-input"
                  />
                </th>
                <th>
                  <select
                    value={scanFilterStatus}
                    onChange={(e) => setScanFilterStatus(e.target.value)}
                    className="column-filter-select"
                  >
                    <option value="">All</option>
                    <option value="success">Success</option>
                    <option value="failed">Failed</option>
                  </select>
                </th>
                <th>
                  <input
                    type="text"
                    placeholder="Filter..."
                    value={scanFilterLifetime}
                    onChange={(e) => setScanFilterLifetime(e.target.value)}
                    className="column-filter-input"
                  />
                </th>
                <th>
                  <input
                    type="text"
                    placeholder="Filter..."
                    value={scanFilterQY}
                    onChange={(e) => setScanFilterQY(e.target.value)}
                    className="column-filter-input"
                  />
                </th>
                <th>
                  <input
                    type="text"
                    placeholder="Filter..."
                    value={scanFilterCF}
                    onChange={(e) => setScanFilterCF(e.target.value)}
                    className="column-filter-input"
                  />
                </th>
                <th>
                  <input
                    type="text"
                    placeholder="Filter..."
                    value={scanFilterAVVol}
                    onChange={(e) => setScanFilterAVVol(e.target.value)}
                    className="column-filter-input"
                  />
                </th>
                <th>
                  <input
                    type="text"
                    placeholder="Filter..."
                    value={scanFilterContVol}
                    onChange={(e) => setScanFilterContVol(e.target.value)}
                    className="column-filter-input"
                  />
                </th>
                <th>
                  <input
                    type="text"
                    placeholder="Filter..."
                    value={scanFilterRatio}
                    onChange={(e) => setScanFilterRatio(e.target.value)}
                    className="column-filter-input"
                  />
                </th>
                <th>
                  <input
                    type="text"
                    placeholder="e.g. TRP"
                    value={scanFilterQuencher}
                    onChange={(e) => setScanFilterQuencher(e.target.value)}
                    className="column-filter-input"
                  />
                </th>
              </tr>
            </thead>
            <tbody>
              {processedScanResults?.map((r, idx) => (
                <tr
                  key={idx}
                  onClick={() => handleRowClick(r)}
                  style={{ cursor: "pointer" }}
                  title="Click to select this residue as active labeling site"
                >
                  <td>
                    <strong>
                      {r.name ? `${r.name} (${r.chain}:${r.residue})` : `${r.chain}:${r.residue}`}
                    </strong>{" "}
                    <span style={{ color: "var(--muted)" }}>({r.residue_name})</span>
                  </td>
                  <td>
                    <span className={`status-badge ${r.status}`}>
                      {r.status === "success" ? "Success" : "No AV"}
                    </span>
                  </td>
                  <td>{r.lifetime !== null ? r.lifetime.toFixed(4) : "-"}</td>
                  <td>{r.quantum_yield !== null ? r.quantum_yield.toFixed(4) : "-"}</td>
                  <td>{r.contact_fraction !== null ? r.contact_fraction.toFixed(4) : "-"}</td>
                  <td>{r.av_volume !== null && r.av_volume !== undefined ? r.av_volume.toFixed(1) : "-"}</td>
                  <td>{r.contact_volume !== null && r.contact_volume !== undefined ? r.contact_volume.toFixed(1) : "-"}</td>
                  <td>{r.contact_to_free_ratio !== null && r.contact_to_free_ratio !== undefined ? r.contact_to_free_ratio.toFixed(4) : "-"}</td>
                  <td>
                    {r.quenchers && r.quenchers.length ? (
                      <span
                        className="quencher-cell"
                        title={r.quenchers
                          .map((q) => `${q.residue_name} ${(((q.quench_rate_fraction ?? 0) * 100)).toFixed(1)}%`)
                          .join(", ")}
                      >
                        <span className="residue-code">{r.dominant_quencher}</span>
                        <span className="muted">
                          {((r.dominant_quencher_fraction ?? 0) * 100).toFixed(0)}%
                        </span>
                        {r.quenchers.length > 1 && (
                          <span className="pill">+{r.quenchers.length - 1}</span>
                        )}
                      </span>
                    ) : (
                      <span className="muted">none</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="empty-results-state">
          <strong>No scan results yet</strong>
          <span>Select residues in the sidebar and click Calculate Scan.</span>
        </div>
      )}
    </div>
  );
}
