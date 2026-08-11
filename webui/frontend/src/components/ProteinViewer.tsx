"use client";

import React, { useEffect, useRef, useState } from "react";
import { StructureMetadata, Project, SimulatePayload, NglStage, NglComponent } from "../types";
import { normalizeAminoAcidInteractions, selectionString, slowFactor } from "../lib/projectHelpers";

declare global {
  interface Window {
    NGL?: {
      Stage: new (element: HTMLElement, options?: Record<string, unknown>) => NglStage;
    };
  }
}

interface ProteinViewerProps {
  structure: StructureMetadata | null;
  project: Project;
  result: SimulatePayload | null;
  active: boolean;
  surfaceRendering: boolean;
  renderSparsity: number;
  onResidueSelect?: (chainId: string, residueId: number) => void;
}

let nglLoadPromise: Promise<void> | null = null;

/**
 * Resize the NGL stage without letting a viewer failure escape.
 *
 * `handleResize` reaches into NGL's WebGL viewer, which is absent when WebGL is
 * unavailable (headless browsers, remote sessions, GPU blocklists) or after the
 * stage has been disposed. A resize must never take down the rest of the page.
 */
function safeHandleResize(stage: NglStage | null): void {
  if (!stage) return;
  try {
    stage.handleResize();
  } catch {
    // Viewer is gone or was never initialised; the fallback UI covers it.
  }
}

let webGlSupported: boolean | null = null;

/**
 * Whether this browser can give us a WebGL context.
 *
 * Checked once before creating an NGL stage: WebGL can be missing (headless
 * browsers, remote sessions, GPU blocklists, user preference), and NGL fails
 * asynchronously inside its own render loop rather than at construction, so a
 * try/catch around the constructor would not catch it.
 */
function hasWebGlSupport(): boolean {
  if (webGlSupported !== null) return webGlSupported;
  try {
    const canvas = document.createElement("canvas");
    webGlSupported = Boolean(
      canvas.getContext("webgl2") || canvas.getContext("webgl") || canvas.getContext("experimental-webgl")
    );
  } catch {
    webGlSupported = false;
  }
  return webGlSupported;
}

const TRAJECTORY_COLOR = "#39ff14";
const RESIDUE_LETTERS: Record<string, string> = {
  ALA: "A",
  ARG: "R",
  ASN: "N",
  ASP: "D",
  CYS: "C",
  GLN: "Q",
  GLU: "E",
  GLY: "G",
  HIS: "H",
  ILE: "I",
  LEU: "L",
  LYS: "K",
  MET: "M",
  PHE: "F",
  PRO: "P",
  SER: "S",
  THR: "T",
  TRP: "W",
  TYR: "Y",
  VAL: "V",
};
const STANDARD_RESIDUES = new Set(Object.keys(RESIDUE_LETTERS));

function hexPair(value: number) {
  return Math.round(value).toString(16).padStart(2, "0");
}

function stickinessBlueColor(rawSlowFactor: number) {
  const factor = slowFactor(rawSlowFactor, 1.0);
  const sticky = 1 - factor;
  const light = { r: 147, g: 197, b: 253 };
  const dark = { r: 8, g: 47, b: 111 };
  const r = light.r + (dark.r - light.r) * sticky;
  const g = light.g + (dark.g - light.g) * sticky;
  const b = light.b + (dark.b - light.b) * sticky;
  return `#${hexPair(r)}${hexPair(g)}${hexPair(b)}`;
}

function pointCloudPdb(points: number[][], residueName: string, atomName: string, chainId: string, stride = 1) {
  let pdbText = "";
  let count = 0;
  points.forEach((pt, index) => {
    if (index % stride !== 0) return;
    count++;
    const serial = (count % 100000).toString().padStart(5);
    const residue = ((count - 1) % 9999 + 1).toString().padStart(4);
    const x = pt[0].toFixed(3).padStart(8);
    const y = pt[1].toFixed(3).padStart(8);
    const z = pt[2].toFixed(3).padStart(8);
    pdbText += `ATOM  ${serial} ${atomName.padEnd(4)} ${residueName.padStart(3)} ${chainId}${residue}    ${x}${y}${z}  1.00  0.00           C\n`;
  });
  return pdbText;
}

function ensureNglLoaded() {
  if (typeof window === "undefined") return Promise.reject(new Error("Browser window is unavailable"));
  if (window.NGL) return Promise.resolve();
  if (nglLoadPromise) return nglLoadPromise;

  nglLoadPromise = new Promise<void>((resolve, reject) => {
    const existing = document.querySelector<HTMLScriptElement>('script[src="/vendor/ngl.js"]');
    if (existing && !window.NGL) {
      existing.remove();
    }
    const script = document.createElement("script");
    const timeout = window.setTimeout(() => {
      reject(new Error("NGL library failed to load"));
    }, 10000);

    script.onload = () => {
      window.clearTimeout(timeout);
      if (window.NGL) {
        resolve();
      } else {
        reject(new Error("NGL library loaded but did not initialize"));
      }
    };
    script.onerror = () => {
      window.clearTimeout(timeout);
      reject(new Error("NGL library failed to load"));
    };

    script.src = "/vendor/ngl.js";
    script.async = true;
    document.head.appendChild(script);
  }).catch((error) => {
    nglLoadPromise = null;
    throw error;
  });

  return nglLoadPromise;
}

export default function ProteinViewer({
  structure,
  project,
  result,
  active,
  surfaceRendering,
  renderSparsity,
  onResidueSelect,
}: ProteinViewerProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const fallbackCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const stageRef = useRef<NglStage | null>(null);
  const componentRef = useRef<any | null>(null);
  // Representations rebuilt on parameter changes, tracked so the expensive
  // cartoon/surface pair can stay in place.
  const highlightReprsRef = useRef<any[]>([]);
  const loadGenerationRef = useRef(0);
  const [stage, setStage] = useState<NglStage | null>(null);
  const [component, setComponent] = useState<any | null>(null);
  const [viewerStatus, setViewerStatus] = useState("Upload structure to inspect");
  const [fallbackViewer, setFallbackViewer] = useState(false);

  function disposeCurrentStage() {
    const current = stageRef.current;
    stageRef.current = null;
    componentRef.current = null;
    highlightReprsRef.current = [];
    setStage(null);
    setComponent(null);
    try {
      current?.dispose();
    } catch (error) {
      console.warn("Failed to dispose NGL stage:", error);
    }
  }

  /**
   * Cartoon + molecular surface for the protein itself.
   *
   * Built once per loaded component. NGL constructs surfaces asynchronously on
   * its own task queue, so rebuilding them whenever a parameter changes is both
   * slow and unsafe: `removeAllRepresentations()` frees the structure while a
   * queued surface build still holds a reference to it, which throws from
   * inside NGL. Only the cheap highlight representations are rebuilt.
   */
  const setupBaseRepresentations = (comp: any) => {
    if (!comp || comp !== componentRef.current || !stageRef.current) return false;
    try {
      comp.addRepresentation("cartoon", { sele: "polymer", color: "chainname" });
      // Atomic detail: the quenching spheres sit on specific side-chain atoms,
      // so those atoms have to be visible for the highlight to mean anything.
      comp.addRepresentation("licorice", {
        sele: "polymer and not hydrogen",
        color: "element",
        radiusScale: 0.28,
        opacity: 0.85,
        useWorker: false,
      });
      comp.addRepresentation("surface", {
        sele: "polymer",
        color: "grey",
        opacity: 0.12,
        side: "front",
        useWorker: false,
      });
      return true;
    } catch (error) {
      console.error("Protein viewer representation setup failed:", error);
      disposeCurrentStage();
      setFallbackViewer(true);
      setViewerStatus(`${structure?.filename || "Structure"}: fallback viewer`);
      return false;
    }
  };

  /** Highlight the labeling site, sticky residues and quenchers. */
  const setupHighlightRepresentations = (comp: any, proj: Project) => {
    if (!comp || comp !== componentRef.current || !stageRef.current) return false;

    try {
      // Remove only what this function added last time; the cartoon and the
      // expensive surface stay in place.
      for (const representation of highlightReprsRef.current) {
        try {
          comp.removeRepresentation(representation);
        } catch {
          // Representation already gone.
        }
      }
      highlightReprsRef.current = [];

      const add = (type: string, options: Record<string, unknown>) => {
        const representation = comp.addRepresentation(type, options);
        if (representation) highlightReprsRef.current.push(representation);
      };

      const interactions = normalizeAminoAcidInteractions(proj.amino_acid_interactions);

      const stickyResidues = Object.entries(interactions)
        .map(([residue, params]) => ({
          residue: residue.trim().toUpperCase(),
          slow_factor: slowFactor(params.slow_factor, 1.0),
        }))
        .filter(({ residue, slow_factor }) => Boolean(residue) && slow_factor < 1.0);

      // A sphere per residue only says something when the stickiness actually
      // varies. Most dyes slow the probe uniformly, and drawing all 20 residue
      // types then buries the structure and hides the quenchers.
      const distinctSlowFactors = new Set(stickyResidues.map((item) => item.slow_factor));
      if (stickyResidues.length && distinctSlowFactors.size > 1) {
        stickyResidues.forEach(({ residue, slow_factor }) => {
          const residueSele = /^\d+$/.test(residue) ? residue : `resn ${residue}`;
          add("spacefill", {
            sele: `(${residueSele}) and .CB`,
            color: "uniform",
            colorValue: stickinessBlueColor(slow_factor),
            radiusType: "size",
            radius: 1.6,
            opacity: 0.55,
            useWorker: false,
          });
        });
      }

      // Quenchers are the informative highlight: draw each one on the atoms
      // that define its quenching centre, the same ones the simulation uses.
      const quenchingResidues = Object.entries(interactions)
        .filter(([, params]) => params.kQ > 0)
        .map(([residue]) => residue.trim().toUpperCase())
        .filter(Boolean);

      for (const residue of quenchingResidues) {
        const atoms = interactions[residue]?.quench_atoms ?? ["CB"];
        const atomSele = atoms.map((atom) => `.${atom}`).join(" or ");
        // Drawn on top of the atoms: depthWrite off keeps the underlying
        // side chain visible through the translucent sphere.
        add("spacefill", {
          sele: `(resn ${residue}) and (${atomSele})`,
          color: "magenta",
          radiusType: "size",
          radius: 1.9,
          opacity: 0.55,
          depthWrite: false,
          useWorker: false,
        });
      }

      const sele = selectionString(proj);
      if (sele) {
        add("spacefill", { sele, color: "red", radiusScale: 1.2 });
        add("ball+stick", { sele, color: "element" });
      }
      return true;
    } catch (error) {
      console.error("Protein viewer representation setup failed:", error);
      disposeCurrentStage();
      setFallbackViewer(true);
      setViewerStatus(`${structure?.filename || "Structure"}: fallback viewer`);
      return false;
    }
  };

  const setupRepresentations = (comp: any, proj: Project) => {
    if (!setupBaseRepresentations(comp)) return false;
    return setupHighlightRepresentations(comp, proj);
  };

  // Main structure loader
  useEffect(() => {
    let disposed = false;
    const generation = loadGenerationRef.current + 1;
    loadGenerationRef.current = generation;

    async function loadNgl() {
      if (!containerRef.current) return;

      setFallbackViewer(false);

      if (!hasWebGlSupport()) {
        setFallbackViewer(true);
        setViewerStatus(structure ? `${structure.filename}` : "3D Viewport — WebGL unavailable");
        return;
      }

      await ensureNglLoaded();

      if (disposed || !containerRef.current || !window.NGL) return;

      if (!stageRef.current) {
        containerRef.current.innerHTML = "";
        const stageInstance = new window.NGL.Stage(containerRef.current, {
          backgroundColor: "white",
          quality: "medium",
        });
        stageRef.current = stageInstance;
        setStage(stageInstance);
      }

      const stageInstance = stageRef.current;

      if (componentRef.current) {
        try {
          stageInstance.removeComponent(componentRef.current);
        } catch (error) {}
        componentRef.current = null;
        setComponent(null);
      }

      if (!structure) {
        setViewerStatus("3D Viewport — Ready for PDB structure");
        return;
      }

      setViewerStatus("Loading structure into 3D viewer");

      const fileExt = structure.filename.split(".").pop()?.toLowerCase() || "pdb";
      const blob = new File([structure.pdb_text], structure.filename, { type: "text/plain" });

      const compInstance = await stageInstance.loadFile(blob, {
        ext: fileExt,
        name: structure.filename,
      });

      if (disposed || generation !== loadGenerationRef.current || stageRef.current !== stageInstance) {
        try {
          stageInstance.removeComponent(compInstance);
        } catch (error) {}
        return;
      }

      componentRef.current = compInstance;
      setComponent(compInstance);
      if (!setupRepresentations(compInstance, project)) return;

      // Click-to-select a residue on the 3D structure. NGL fires
      // pickingProxy on stage.signals.clicked with the atom/residue under the
      // cursor. This calls onResidueSelect just like the sequence bar does.
      if (onResidueSelect) {
        stageInstance.signals.clicked.add((picked: any) => {
          if (!picked || !picked.atom) return;
          const atom = picked.atom;
          const chainId = atom.chainname;
          const residueId = atom.resno;
          if (chainId && residueId != null) {
            onResidueSelect(chainId, residueId);
          }
        });
      }

      compInstance.autoView();
      safeHandleResize(stageInstance);
      setViewerStatus(`${structure.filename}: ${structure.atom_count} atoms`);
    }


    loadNgl().catch((error) => {
      if (hasWebGlSupport()) {
        console.error("Protein viewer NGL failed:", error);
      } else {
        // Expected degradation, not a fault: show the 2D fallback quietly.
        console.warn("3D viewer unavailable (no WebGL); using the fallback viewer.");
      }
      setStage(null);
      setComponent(null);
      setFallbackViewer(true);
      setViewerStatus(
        hasWebGlSupport()
          ? `${structure?.filename || "Structure"}: fallback viewer`
          : `${structure?.filename || "Structure"}: WebGL unavailable`
      );
    });

    return () => {
      disposed = true;
      if (generation === loadGenerationRef.current) {
        disposeCurrentStage();
      }
    };
  }, [structure]);

  useEffect(() => {
    if (!fallbackViewer || !structure || !fallbackCanvasRef.current) return;

    const canvas = fallbackCanvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(320, Math.floor(rect.width));
    const height = Math.max(240, Math.floor(rect.height));
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#f8fafb";
    ctx.fillRect(0, 0, width, height);

    const atoms = structure.pdb_text
      .split(/\r?\n/)
      .filter((line) => line.startsWith("ATOM  ") || line.startsWith("HETATM"))
      .map((line) => ({
        atom: line.slice(12, 16).trim(),
        resn: line.slice(17, 20).trim(),
        chain: line.slice(21, 22).trim() || "_",
        resi: Number.parseInt(line.slice(22, 26).trim(), 10),
        x: Number.parseFloat(line.slice(30, 38).trim()),
        y: Number.parseFloat(line.slice(38, 46).trim()),
        z: Number.parseFloat(line.slice(46, 54).trim()),
      }))
      .filter((atom) => Number.isFinite(atom.x) && Number.isFinite(atom.y) && Number.isFinite(atom.z));

    const traceAtoms = atoms.filter((atom) => atom.atom === "CA");
    const points = traceAtoms.length > 0 ? traceAtoms : atoms.filter((_, index) => index % 8 === 0);
    if (points.length === 0) {
      ctx.fillStyle = "#64737a";
      ctx.font = "13px system-ui";
      ctx.fillText("No drawable atom coordinates found", 18, 28);
      return;
    }

    const trajectoryPoints = (result?.trajectory_points ?? [])
      .map((pt) => ({ x: pt[0], y: pt[1], z: pt[2] }))
      .filter((pt) => Number.isFinite(pt.x) && Number.isFinite(pt.y) && Number.isFinite(pt.z));
    const boundsPoints = trajectoryPoints.length > 0 ? [...points, ...trajectoryPoints] : points;
    const minX = Math.min(...boundsPoints.map((atom) => atom.x));
    const maxX = Math.max(...boundsPoints.map((atom) => atom.x));
    const minY = Math.min(...boundsPoints.map((atom) => atom.y));
    const maxY = Math.max(...boundsPoints.map((atom) => atom.y));
    const minZ = Math.min(...points.map((atom) => atom.z));
    const maxZ = Math.max(...points.map((atom) => atom.z));
    const spanX = maxX - minX || 1;
    const spanY = maxY - minY || 1;
    const pad = 28;
    const scale = Math.min((width - pad * 2) / spanX, (height - pad * 2) / spanY);
    const chainColors = ["#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#17becf", "#bcbd22"];
    const chains = Array.from(new Set(points.map((atom) => atom.chain)));
    const interactions = normalizeAminoAcidInteractions(project.amino_acid_interactions);
    const quenchingNames = new Set(
      Object.entries(interactions)
        .filter(([, params]) => params.kQ > 0)
        .map(([name]) => name.toUpperCase())
    );
    const slowingNames = new Set(
      Object.entries(interactions)
        .filter(([, params]) => slowFactor(params.slow_factor, 1.0) < 1.0)
        .map(([name]) => name.toUpperCase())
    );
    const slowFactorsByName = new Map(
      Object.entries(interactions).map(([name, params]) => [name.toUpperCase(), slowFactor(params.slow_factor, 1.0)])
    );

    function projectPoint(atom: { x: number; y: number }) {
      return {
        x: pad + (atom.x - minX) * scale,
        y: height - pad - (atom.y - minY) * scale,
      };
    }

    ctx.lineWidth = 1;
    chains.forEach((chain, index) => {
      const chainPoints = points.filter((atom) => atom.chain === chain);
      ctx.strokeStyle = chainColors[index % chainColors.length];
      ctx.beginPath();
      chainPoints.forEach((atom, pointIndex) => {
        const pt = projectPoint(atom);
        if (pointIndex === 0) ctx.moveTo(pt.x, pt.y);
        else ctx.lineTo(pt.x, pt.y);
      });
      ctx.stroke();
    });

    points.forEach((atom) => {
      const pt = projectPoint(atom);
      const depth = (atom.z - minZ) / (maxZ - minZ || 1);
      const selected =
        atom.chain === project.attachment?.chain &&
        atom.resi === project.attachment?.residue &&
        (!project.attachment?.atom || atom.atom === project.attachment.atom || atom.atom === "CA");
      const quenching = quenchingNames.has(atom.resn.toUpperCase());
      const slowing = slowingNames.has(atom.resn.toUpperCase());
      ctx.fillStyle = selected
        ? "#d62728"
        : slowing
          ? stickinessBlueColor(slowFactorsByName.get(atom.resn.toUpperCase()) ?? 1.0)
          : quenching
            ? "#d33682"
            : chainColors[chains.indexOf(atom.chain) % chainColors.length];
      ctx.globalAlpha = selected || quenching || slowing ? 0.95 : 0.35 + depth * 0.45;
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, selected ? 5 : quenching ? 3.2 : slowing ? 3 : 2.2, 0, Math.PI * 2);
      ctx.fill();
    });
    if (trajectoryPoints.length > 1) {
      ctx.globalAlpha = 0.95;
      ctx.strokeStyle = TRAJECTORY_COLOR;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      trajectoryPoints.forEach((point, pointIndex) => {
        const pt = projectPoint(point);
        if (pointIndex === 0) ctx.moveTo(pt.x, pt.y);
        else ctx.lineTo(pt.x, pt.y);
      });
      ctx.stroke();
      ctx.fillStyle = TRAJECTORY_COLOR;
      const start = projectPoint(trajectoryPoints[0]);
      const end = projectPoint(trajectoryPoints[trajectoryPoints.length - 1]);
      ctx.beginPath();
      ctx.arc(start.x, start.y, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#101820";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(end.x, end.y, 5, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    ctx.fillStyle = "#324148";
    ctx.font = "12px system-ui";
    ctx.fillText(`${structure.filename} - ${points.length} trace atoms`, 12, 20);
  }, [
    fallbackViewer,
    structure,
    project.attachment,
    JSON.stringify(project.amino_acid_interactions),
    JSON.stringify(result?.trajectory_points),
  ]);

  // Handle representation changes when project parameter modifications occur
  useEffect(() => {
    if (!stage || !component || !active || component !== componentRef.current || stage !== stageRef.current) return;
    setupHighlightRepresentations(component, project);
  }, [
    stage,
    component,
    project.attachment?.chain,
    project.attachment?.residue,
    project.attachment?.atom,
    JSON.stringify(project.amino_acid_interactions),
    project.slow_radius,
    project.critical_distance,
    active,
  ]);

  // Handle AV / ACV simulation points rendering
  useEffect(() => {
    if (!stage || !active) return;

    const activeStage = stage;
    let activeToken = true;
    const addedComponents: any[] = [];

    /**
     * Load a point cloud and add its representations.
     *
     * The loads are asynchronous, so by the time one resolves the stage may
     * already have been disposed (navigating away, a new structure, a new
     * result). Touching a disposed stage throws from inside NGL, so every
     * continuation re-checks that this effect and this stage are still current,
     * and rejections are handled rather than left dangling.
     */
    function addPointCloud(
      points: number[][],
      pdbOptions: { resName: string; atomName: string; chain: string; sparsity?: number },
      name: string,
      representations: Array<[string, Record<string, unknown>]>
    ) {
      const pdbText = pointCloudPdb(
        points,
        pdbOptions.resName,
        pdbOptions.atomName,
        pdbOptions.chain,
        pdbOptions.sparsity
      );
      const blob = new Blob([pdbText], { type: "text/plain" });

      activeStage
        .loadFile(blob, { ext: "pdb", name })
        .then((comp: any) => {
          const stillCurrent = activeToken && stageRef.current === activeStage;
          if (!stillCurrent) {
            try {
              activeStage.removeComponent(comp);
            } catch {
              // Stage already gone; nothing to clean up.
            }
            return;
          }
          try {
            for (const [type, options] of representations) {
              comp.addRepresentation(type, options);
            }
            addedComponents.push(comp);
          } catch (error) {
            console.warn(`Could not render ${name}:`, error);
          }
        })
        .catch((error: unknown) => {
          if (activeToken && stageRef.current === activeStage) {
            console.warn(`Could not load ${name}:`, error);
          }
        });
    }

    if (result?.trajectory_points && result.trajectory_points.length > 1 && window.NGL) {
      addPointCloud(
        result.trajectory_points,
        { resName: "GLY", atomName: "CA", chain: "T" },
        "dye trajectory",
        [
          [
            "trace",
            {
              color: TRAJECTORY_COLOR,
              colorValue: TRAJECTORY_COLOR,
              radiusScale: 1.4,
              opacity: 0.95,
              useWorker: false,
            },
          ],
          ["point", { colorValue: TRAJECTORY_COLOR, pointSize: 2.0, opacity: 0.75, useWorker: false }],
        ]
      );
    }

    if (result?.av_points && result.av_points.length > 0 && window.NGL) {
      addPointCloud(
        result.av_points,
        { resName: "AV", atomName: "C", chain: "A", sparsity: renderSparsity },
        "accessible volume",
        [
          surfaceRendering
            ? ["surface", { color: "cyan", opacity: 0.4, useWorker: false }]
            : ["point", { colorValue: "cyan", pointSize: 1.5, opacity: 0.8, useWorker: false }],
        ]
      );
    }

    if (result?.acv_points && result.acv_points.length > 0 && window.NGL) {
      addPointCloud(
        result.acv_points,
        { resName: "ACV", atomName: "C", chain: "C", sparsity: renderSparsity },
        "contact volume",
        [
          surfaceRendering
            ? ["surface", { color: "orange", opacity: 0.5, useWorker: false }]
            : ["point", { colorValue: "orange", pointSize: 1.5, opacity: 0.8, useWorker: false }],
        ]
      );
    }

    return () => {
      activeToken = false;
      addedComponents.forEach((comp) => {
        try {
          activeStage.removeComponent(comp);
        } catch {
          // Stage already disposed.
        }
      });
    };
  }, [stage, result, surfaceRendering, active, renderSparsity]);

  // Handle active resize trigger
  useEffect(() => {
    if (active && stage) {
      const timer = setTimeout(() => safeHandleResize(stage), 50);
      return () => clearTimeout(timer);
    }
  }, [active, stage]);

  // Handle ResizeObserver to follow layout changes
  useEffect(() => {
    if (!containerRef.current || !stage) return;
    const observer = new ResizeObserver(() => safeHandleResize(stage));
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, [stage]);

  const selectedChainId = project.attachment?.chain || "";
  const selectedResidueId = project.attachment?.residue;
  const sequenceChains =
    structure?.chains
      .map((chain) => ({
        ...chain,
        residues: chain.residues.filter((residue) => STANDARD_RESIDUES.has(residue.name.toUpperCase())),
      }))
      .filter((chain) => chain.residues.length > 0) || [];

  return (
    <div className="protein-viewer-panel">
      {sequenceChains.length > 0 && (
        <div className="sequence-viewer" aria-label="Protein sequence">
          {sequenceChains.map((chain) => (
            <div className="sequence-chain" key={chain.id}>
              <div className="sequence-chain-label">Chain {chain.id}</div>
              <div className="sequence-residue-row">
                {chain.residues.map((residue, index) => {
                  const selected = chain.id === selectedChainId && residue.id === selectedResidueId;
                  const letter = RESIDUE_LETTERS[residue.name.toUpperCase()] || "X";
                  const showNumber = selected || index === 0 || residue.id % 10 === 0;
                  return (
                    <button
                      key={`${chain.id}-${residue.id}`}
                      type="button"
                      className={`sequence-residue ${selected ? "selected" : ""}`}
                      aria-pressed={selected}
                      aria-label={`Select ${residue.name} ${residue.id} on chain ${chain.id}`}
                      title={`${residue.name} ${residue.id}, chain ${chain.id}`}
                      data-residue-key={`${chain.id}:${residue.id}`}
                      onClick={() => onResidueSelect?.(chain.id, residue.id)}
                    >
                      {showNumber ? <span className="sequence-residue-number">{residue.id}</span> : null}
                      <span className="sequence-residue-letter">{letter}</span>
                    </button>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      )}
      <div className="protein-viewer">
        <div ref={containerRef} className="ngl-stage" />
        {fallbackViewer && <canvas ref={fallbackCanvasRef} className="fallback-stage" />}
        <div className="viewer-badge">{viewerStatus}</div>
        {structure && (
          // Without this the magenta spheres are unexplained decoration. They
          // are the quenching centres — the same atoms the simulation uses.
          <ul className="viewer-legend" aria-label="Viewer legend">
            <li>
              <span className="legend-swatch legend-quencher" aria-hidden="true" />
              quenching centre
            </li>
            <li>
              <span className="legend-swatch legend-site" aria-hidden="true" />
              labeling site
            </li>
            <li>
              <span className="legend-swatch legend-surface" aria-hidden="true" />
              accessible volume
            </li>
          </ul>
        )}
      </div>
    </div>
  );
}
