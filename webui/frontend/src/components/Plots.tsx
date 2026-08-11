"use client";

import React from "react";
import { SimulatePayload } from "../types";
import { downsamplePoints } from "../lib/projectHelpers";
import { useAppContext } from "../context/AppContext";

type Point = { x: number; y: number };

const PAD = { left: 58, right: 18, top: 16, bottom: 40 };

function niceTicks(min: number, max: number, count = 5): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max) || max <= min) return [min];
  const rawStep = (max - min) / count;
  const magnitude = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const normalized = rawStep / magnitude;
  const step = (normalized >= 5 ? 10 : normalized >= 2 ? 5 : normalized >= 1 ? 2 : 1) * magnitude;
  const ticks: number[] = [];
  for (let value = Math.ceil(min / step) * step; value <= max + step * 1e-9; value += step) {
    ticks.push(Number(value.toFixed(10)));
  }
  return ticks;
}

function formatTick(value: number): string {
  const abs = Math.abs(value);
  if (abs === 0) return "0";
  if (abs >= 1e4 || abs < 1e-3) return value.toExponential(0);
  if (abs >= 100) return value.toFixed(0);
  if (abs >= 1) return value.toFixed(1);
  return value.toFixed(2);
}

/** Shared chart frame: grid, axes, ticks and labels around a plot area. */
function ChartFrame({
  width,
  height,
  xTicks,
  yTicks,
  toX,
  toY,
  xLabel,
  yLabel,
  children,
}: {
  width: number;
  height: number;
  xTicks: Array<{ value: number; label: string }>;
  yTicks: Array<{ value: number; label: string }>;
  toX: (value: number) => number;
  toY: (value: number) => number;
  xLabel: string;
  yLabel: string;
  children: React.ReactNode;
}) {
  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="chart" role="img" preserveAspectRatio="xMidYMid meet">
      <g className="chart-grid">
        {xTicks.map((tick) => (
          <line key={`gx-${tick.value}`} x1={toX(tick.value)} y1={PAD.top} x2={toX(tick.value)} y2={height - PAD.bottom} />
        ))}
        {yTicks.map((tick) => (
          <line key={`gy-${tick.value}`} x1={PAD.left} y1={toY(tick.value)} x2={width - PAD.right} y2={toY(tick.value)} />
        ))}
      </g>
      <g className="chart-axis">
        <line x1={PAD.left} y1={height - PAD.bottom} x2={width - PAD.right} y2={height - PAD.bottom} />
        <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={height - PAD.bottom} />
      </g>
      {children}
      <g className="chart-ticks">
        {xTicks.map((tick) => (
          <text key={`tx-${tick.value}`} x={toX(tick.value)} y={height - PAD.bottom + 16} textAnchor="middle">
            {tick.label}
          </text>
        ))}
        {yTicks.map((tick) => (
          <text key={`ty-${tick.value}`} x={PAD.left - 8} y={toY(tick.value) + 4} textAnchor="end">
            {tick.label}
          </text>
        ))}
      </g>
      <text className="chart-axis-label" x={(PAD.left + width - PAD.right) / 2} y={height - 6} textAnchor="middle">
        {xLabel}
      </text>
      <text
        className="chart-axis-label"
        transform={`rotate(-90 14 ${(PAD.top + height - PAD.bottom) / 2})`}
        x={14}
        y={(PAD.top + height - PAD.bottom) / 2}
        textAnchor="middle"
      >
        {yLabel}
      </text>
    </svg>
  );
}

function EmptyState({ title, hint }: { title: string; hint: string }) {
  return (
    <div className="plot-card">
      <div className="empty-state">
        <strong>{title}</strong>
        <span>{hint}</span>
      </div>
    </div>
  );
}

function linePath(points: Point[], toX: (v: number) => number, toY: (v: number) => number): string {
  return points
    .map((point, index) => `${index === 0 ? "M" : "L"}${toX(point.x).toFixed(1)},${toY(point.y).toFixed(1)}`)
    .join(" ");
}

export function DecayPlot({ result }: { result: SimulatePayload | null }) {
  const width = 760;
  const height = 460;

  if (!result || !result.time_ns?.length) {
    return <EmptyState title="No decay yet" hint="Select a labeling site and run a simulation." />;
  }

  const donor = result.time_ns.map((x, i) => ({ x, y: result.donor_counts[i] ?? 0 }));
  const fret = result.fret_counts
    ? result.time_ns.map((x, i) => ({ x, y: result.fret_counts?.[i] ?? 0 }))
    : [];

  // Counts are Poisson-distributed over decades, so the y axis is log10.
  const allCounts = [...donor, ...fret].map((p) => p.y).filter((y) => y > 0);
  const maxCount = allCounts.length ? Math.max(...allCounts) : 1;
  const minLog = 0;
  const maxLog = Math.max(Math.log10(maxCount), 1);
  const minX = Math.min(...result.time_ns);
  const maxX = Math.max(...result.time_ns);

  const toX = (v: number) => PAD.left + ((v - minX) / Math.max(maxX - minX, 1e-9)) * (width - PAD.left - PAD.right);
  const toY = (v: number) =>
    height - PAD.bottom - ((v - minLog) / Math.max(maxLog - minLog, 1e-9)) * (height - PAD.top - PAD.bottom);
  const toLogY = (count: number) => toY(Math.log10(Math.max(count, 1)));

  const decadeTicks = [];
  for (let decade = 0; decade <= Math.ceil(maxLog); decade++) {
    decadeTicks.push({ value: decade, label: decade === 0 ? "1" : `1e${decade}` });
  }

  return (
    <div className="plot-card">
      <ChartFrame
        width={width}
        height={height}
        xTicks={niceTicks(minX, maxX).map((v) => ({ value: v, label: formatTick(v) }))}
        yTicks={decadeTicks}
        toX={toX}
        toY={toY}
        xLabel="time [ns]"
        yLabel="counts"
      >
        <path d={linePath(donor, toX, toLogY)} className="donor-line" />
        {fret.length ? <path d={linePath(fret, toX, toLogY)} className="fret-line" /> : null}
      </ChartFrame>
    </div>
  );
}

export function TrajectoryPlot({ result }: { result: SimulatePayload | null }) {
  const width = 760;
  const height = 280;
  const series = result?.trajectory_distance_to_mean;

  if (!series?.length) {
    return <EmptyState title="No trajectory data" hint="Run a simulation to plot the dye trajectory." />;
  }

  // The backend ships an explicit time axis; fall back to index * t_step for
  // results produced before that field existed.
  const tStep = result?.project?.t_step ?? 0.02;
  const times = result?.trajectory_time_ns ?? series.map((_, i) => i * tStep);
  const points = downsamplePoints(
    series.map((y, i) => ({ x: times[i] ?? i * tStep, y })),
    800
  );

  const minX = Math.min(...points.map((p) => p.x));
  const maxX = Math.max(...points.map((p) => p.x));
  const minY = Math.min(...points.map((p) => p.y));
  const maxY = Math.max(...points.map((p) => p.y));
  const toX = (v: number) => PAD.left + ((v - minX) / Math.max(maxX - minX, 1e-9)) * (width - PAD.left - PAD.right);
  const toY = (v: number) =>
    height - PAD.bottom - ((v - minY) / Math.max(maxY - minY, 1e-9)) * (height - PAD.top - PAD.bottom);

  return (
    <div className="plot-card">
      <ChartFrame
        width={width}
        height={height}
        xTicks={niceTicks(minX, maxX).map((v) => ({ value: v, label: formatTick(v) }))}
        yTicks={niceTicks(minY, maxY, 4).map((v) => ({ value: v, label: formatTick(v) }))}
        toX={toX}
        toY={toY}
        xLabel="time [ns]"
        yLabel="distance to mean [Å]"
      >
        <path d={linePath(points, toX, toY)} className="trajectory-line" />
      </ChartFrame>
    </div>
  );
}

export function AutocorrPlot({ result }: { result: SimulatePayload | null }) {
  const width = 760;
  const height = 280;
  const series = result?.trajectory_autocorr;

  const tStep = result?.project?.t_step ?? 0.02;
  const lags = result?.trajectory_autocorr_lag_ns ?? series?.map((_, i) => i * tStep) ?? [];
  const points = downsamplePoints(
    (series ?? []).map((y, i) => ({ x: lags[i] ?? i * tStep, y })).filter((p) => p.x > 0),
    800
  );

  if (!points.length) {
    return <EmptyState title="No correlation data" hint="Run a simulation to plot the autocorrelation." />;
  }

  const minX = Math.log10(Math.min(...points.map((p) => p.x)));
  const maxX = Math.log10(Math.max(...points.map((p) => p.x)));
  const minY = Math.min(0, ...points.map((p) => p.y));
  const maxY = Math.max(...points.map((p) => p.y));

  const toX = (logX: number) =>
    PAD.left + ((logX - minX) / Math.max(maxX - minX, 1e-9)) * (width - PAD.left - PAD.right);
  const toY = (v: number) =>
    height - PAD.bottom - ((v - minY) / Math.max(maxY - minY, 1e-9)) * (height - PAD.top - PAD.bottom);

  const xTicks = [];
  for (let decade = Math.floor(minX); decade <= Math.ceil(maxX); decade++) {
    if (decade >= minX && decade <= maxX) {
      const value = Math.pow(10, decade);
      xTicks.push({ value: decade, label: value >= 1 ? value.toFixed(0) : value.toPrecision(1) });
    }
  }

  return (
    <div className="plot-card">
      <ChartFrame
        width={width}
        height={height}
        xTicks={xTicks}
        yTicks={niceTicks(minY, maxY, 4).map((v) => ({ value: v, label: formatTick(v) }))}
        toX={toX}
        toY={toY}
        xLabel="lag time [ns], log scale"
        yLabel="autocorrelation"
      >
        <path
          d={points
            .map((p, i) => `${i === 0 ? "M" : "L"}${toX(Math.log10(p.x)).toFixed(1)},${toY(p.y).toFixed(1)}`)
            .join(" ")}
          className="autocorr-line"
        />
      </ChartFrame>
    </div>
  );
}

/** Which residue types actually quenched, and by how much. */
export function QuenchingBreakdown({ result }: { result: SimulatePayload | null }) {
  const rows = (result?.aa_residence ?? [])
    .filter((row) => (row.mean_quench_rate ?? 0) > 0)
    .sort((a, b) => (b.quench_rate_fraction ?? 0) - (a.quench_rate_fraction ?? 0));

  if (!rows.length) {
    return (
      <EmptyState
        title="No quenching contributions"
        hint="No residue type with a non-zero rate came into contact with the dye."
      />
    );
  }

  const maxFraction = Math.max(...rows.map((row) => row.quench_rate_fraction ?? 0), 1e-9);

  return (
    <div className="quench-breakdown">
      {rows.map((row) => {
        const fraction = row.quench_rate_fraction ?? 0;
        return (
          <div className="quench-bar-row" key={row.residue_name}>
            <span className="residue-code">{row.residue_name}</span>
            <div className="quench-bar-track" title={`${(fraction * 100).toFixed(1)}% of the total quenching rate`}>
              <div className="quench-bar-fill" style={{ width: `${(fraction / maxFraction) * 100}%` }} />
            </div>
            <span className="quench-bar-value">{(fraction * 100).toFixed(1)}%</span>
            <span className="quench-bar-detail">
              k<sub>Q</sub> {(row.kQ ?? 0).toFixed(2)}/ns · r {(row.quench_radius ?? 0).toFixed(1)} Å ·
              contact {((row.fraction_frames_quenched ?? 0) * 100).toFixed(2)}%
            </span>
          </div>
        );
      })}
    </div>
  );
}

export function Metrics({ result }: { result: SimulatePayload | null }) {
  const { t } = useAppContext();
  const metrics: Array<[string, string]> = [
    ["Donor QY", result ? result.quantum_yield_donor.toFixed(4) : "-"],
    ["Contact fraction", result ? `${(result.collisions_fraction * 100).toFixed(2)}%` : "-"],
    ["FRET QY", result?.quantum_yield_fret != null ? result.quantum_yield_fret.toFixed(4) : "-"],
    ["FRET Eff.", result?.fret_efficiency != null ? result.fret_efficiency.toFixed(4) : "-"],
    ["AV volume", result?.av_volume != null ? `${result.av_volume.toFixed(0)} Å³` : "-"],
    ["Contact volume", result?.contact_volume != null ? `${result.contact_volume.toFixed(0)} Å³` : "-"],
    ["Contact/free ratio", result?.contact_to_free_ratio != null ? result.contact_to_free_ratio.toFixed(4) : "-"],
  ];
  return (
    <div className="metrics">
      <div className="metrics-title">{t("web.card.derived", "Derived quantities")}</div>
      {metrics.map(([label, value]) => (
        <div className="metric" key={label}>
          <span>{label}</span>
          <strong>{value}</strong>
        </div>
      ))}
    </div>
  );
}

export function AAResidenceTable({ result }: { result: SimulatePayload | null }) {
  const rows = [...(result?.aa_residence ?? [])].sort(
    (a, b) =>
      (b.quench_rate_fraction ?? 0) - (a.quench_rate_fraction ?? 0) ||
      (b.attributed_total_time_ns ?? b.total_time_ns) - (a.attributed_total_time_ns ?? a.total_time_ns)
  );

  if (!rows.length) {
    return (
      <div className="empty-results-state" style={{ minHeight: "160px" }}>
        <strong>No AA residence data</strong>
        <span>Run a simulation to calculate residue-type residence times.</span>
      </div>
    );
  }

  return (
    <div className="aa-residence-table-scroll">
      <table className="scan-results-table">
        <thead>
          <tr>
            <th scope="col">AA</th>
            <th scope="col">Count</th>
            <th scope="col">
              k<sub>Q</sub> [1/ns]
            </th>
            <th scope="col">r [Å]</th>
            <th scope="col">Quench share</th>
            <th scope="col">Contact frames</th>
            <th scope="col">Slow factor</th>
            <th scope="col">Attributed ns</th>
            <th scope="col">Attributed %</th>
            <th scope="col">Mean event ns</th>
            <th scope="col">Max event ns</th>
            <th scope="col">Mean min Å</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.residue_name} className={(row.mean_quench_rate ?? 0) > 0 ? "is-quenching" : ""}>
              <th scope="row">{row.residue_name}</th>
              <td>{row.center_count}</td>
              <td>{(row.kQ ?? 0).toFixed(2)}</td>
              <td>{(row.quench_radius ?? 0).toFixed(1)}</td>
              <td>{((row.quench_rate_fraction ?? 0) * 100).toFixed(1)}%</td>
              <td>{((row.fraction_frames_quenched ?? 0) * 100).toFixed(2)}%</td>
              <td>{row.effective_slow_factor != null ? row.effective_slow_factor.toExponential(2) : "-"}</td>
              <td>{(row.attributed_total_time_ns ?? row.total_time_ns).toFixed(3)}</td>
              <td>{((row.attributed_fraction_near ?? row.fraction_near) * 100).toFixed(2)}%</td>
              <td>{(row.attributed_mean_event_time_ns ?? row.mean_event_time_ns).toFixed(3)}</td>
              <td>{(row.attributed_max_event_time_ns ?? row.max_event_time_ns).toFixed(3)}</td>
              <td>{row.mean_min_distance_angstrom != null ? row.mean_min_distance_angstrom.toFixed(2) : "-"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
