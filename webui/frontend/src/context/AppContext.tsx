"use client";

import React, { createContext, useCallback, useContext, useEffect, useRef, useState, useMemo } from "react";
import {
  JobSummary,
  Project,
  QuenchingDefaults,
  StructureMetadata,
  SimulatePayload,
  Status,
  ScanResultItem,
} from "../types";
import { ParameterCatalog } from "../lib/parameterCatalog";
import {
  DEFAULT_PROJECT,
  DEFAULT_DYE_STICKINESS,
  n,
  setNested,
  syncDonor,
  projectJson,
  normalizeAminoAcidInteractions,
  firstUsefulAtom,
  normalizeProjectForComparison,
  withProjectDefaults,
} from "../lib/projectHelpers";

interface AppContextType {
  project: Project;
  setProject: React.Dispatch<React.SetStateAction<Project>>;
  savedProject: Project;
  setSavedProject: React.Dispatch<React.SetStateAction<Project>>;
  projectText: string;
  setProjectText: React.Dispatch<React.SetStateAction<string>>;
  structure: StructureMetadata | null;
  setStructure: React.Dispatch<React.SetStateAction<StructureMetadata | null>>;
  result: SimulatePayload | null;
  setResult: React.Dispatch<React.SetStateAction<SimulatePayload | null>>;
  status: Status;
  setStatus: React.Dispatch<React.SetStateAction<Status>>;
  showJson: boolean;
  setShowJson: React.Dispatch<React.SetStateAction<boolean>>;
  showCitationModal: boolean;
  setShowCitationModal: React.Dispatch<React.SetStateAction<boolean>>;
  surfaceRendering: boolean;
  setSurfaceRendering: React.Dispatch<React.SetStateAction<boolean>>;
  renderSparsity: number;
  setRenderSparsity: React.Dispatch<React.SetStateAction<number>>;
  scanResults: ScanResultItem[] | null;
  setScanResults: React.Dispatch<React.SetStateAction<ScanResultItem[] | null>>;
  scanAll: boolean;
  setScanAll: React.Dispatch<React.SetStateAction<boolean>>;
  selectedScanResidues: Record<string, boolean>;
  setSelectedScanResidues: React.Dispatch<React.SetStateAction<Record<string, boolean>>>;
  scanProgress: { current: number; total: number } | null;
  setScanProgress: React.Dispatch<React.SetStateAction<{ current: number; total: number } | null>>;
  scanStatus: Status;
  setScanStatus: React.Dispatch<React.SetStateAction<Status>>;
  isScanning: boolean;
  setIsScanning: React.Dispatch<React.SetStateAction<boolean>>;
  jobs: JobSummary[];
  setJobs: React.Dispatch<React.SetStateAction<JobSummary[]>>;
  jobsTotal: number;
  dyes: Record<string, any>;
  selectedDyeName: string;
  setSelectedDyeName: React.Dispatch<React.SetStateAction<string>>;
  autoSimulate: boolean;
  setAutoSimulate: React.Dispatch<React.SetStateAction<boolean>>;
  catalog: ParameterCatalog | null;
  quenchingDefaults: QuenchingDefaults | null;
  /** Interface language. Only presentation: the project JSON is identical. */
  locale: string;
  setLocale: React.Dispatch<React.SetStateAction<string>>;
  /** Locale code -> its name in that language, from the backend. */
  locales: Record<string, string>;
  /** UI strings for the current locale; `t()` reads them with a fallback. */
  t: (key: string, fallback?: string) => string;
  selectedJobId: string | null;
  setSelectedJobId: React.Dispatch<React.SetStateAction<string | null>>;
  
  // Ref hooks
  pdbInputRef: React.RefObject<HTMLInputElement | null>;

  // Handlers
  fetchJobs: () => Promise<void>;
  handleLoadJob: (jobId: string) => Promise<void>;
  handleDyeChange: (dyeName: string) => void;
  uploadPdb: (file: File) => Promise<void>;
  loadPdbId: (pdbId: string) => Promise<void>;
  runSimulation: (activeProj?: Project) => Promise<void>;
  abortSimulation: () => void;
  runScan: () => Promise<void>;
  abortScan: () => void;
  copyScanResultsToClipboard: () => void;
  applyJson: () => boolean;
  handleClearState: () => void;
  saveDyePreset: () => void;
  saveParameter: (parameterId?: string) => void;
  downloadParameters: () => void;
  downloadResult: () => void;
  onChainChange: (chainId: string) => void;
  onResidueChange: (id: number) => void;
  onSequenceResidueSelect: (chainId: string, residueId: number) => void;
  updateProject: (path: string, value: string | number | boolean) => void;
  allResidues: Array<{ chain: string; residue: number; residue_name: string; atom: string; key: string }>;
}

const AppContext = createContext<AppContextType | undefined>(undefined);
const API_FALLBACK_ORIGIN = (process.env.NEXT_PUBLIC_QUEST_BACKEND_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

function apiUrl(path: string) {
  return `${API_FALLBACK_ORIGIN}${path}`;
}

async function fetchApi(path: string, init?: RequestInit) {
  try {
    const response = await fetch(path, init);
    if (response.status < 500) return response;
    const text = await response.clone().text();
    if (!text.startsWith("Internal Server Error")) return response;
  } catch (_) {
    // Retry below against the backend directly.
  }
  return fetch(apiUrl(path), init);
}

export function AppProvider({ children }: { children: React.ReactNode }) {
  const [project, setProject] = useState<Project>(DEFAULT_PROJECT);
  const [savedProject, setSavedProject] = useState<Project>(DEFAULT_PROJECT);
  const lastSimulatedProjectRef = useRef<string>("");
  const [projectText, setProjectText] = useState(projectJson(DEFAULT_PROJECT));
  const [structure, setStructure] = useState<StructureMetadata | null>(null);
  const [result, setResult] = useState<SimulatePayload | null>(null);
  const [status, setStatus] = useState<Status>({ kind: "idle", message: "Upload structure to start" });
  const [showJson, setShowJson] = useState<boolean>(false);
  const [showCitationModal, setShowCitationModal] = useState<boolean>(false);
  const [surfaceRendering, setSurfaceRendering] = useState<boolean>(false);
  const [renderSparsity, setRenderSparsity] = useState<number>(50);
  const [scanResults, setScanResults] = useState<ScanResultItem[] | null>(null);
  const [scanAll, setScanAll] = useState<boolean>(true);
  const [selectedScanResidues, setSelectedScanResidues] = useState<Record<string, boolean>>({});
  const [scanProgress, setScanProgress] = useState<{ current: number; total: number } | null>(null);
  const [scanStatus, setScanStatus] = useState<Status>({ kind: "idle", message: "Upload structure to scan" });
  const [isScanning, setIsScanning] = useState<boolean>(false);
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [jobsTotal, setJobsTotal] = useState<number>(0);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [selectedDyeName, setSelectedDyeName] = useState<string>("Alexa488");
  const [autoSimulate, setAutoSimulate] = useState<boolean>(false);
  const [catalog, setCatalog] = useState<ParameterCatalog | null>(null);
  // The language lives here, not in each component, because changing it has to
  // refetch the catalog: parameter labels and help are translated server-side,
  // from the same files the desktop form and the docs read.
  const [locale, setLocale] = useState<string>("en");
  const [locales, setLocales] = useState<Record<string, string>>({});
  const [uiStrings, setUiStrings] = useState<Record<string, string>>({});
  const [quenchingDefaults, setQuenchingDefaults] = useState<QuenchingDefaults | null>(null);
  const [dyes, setDyes] = useState<Record<string, any>>({});

  const scanAbortControllerRef = useRef<AbortController | null>(null);
  const simAbortControllerRef = useRef<AbortController | null>(null);
  const pdbInputRef = useRef<HTMLInputElement | null>(null);

  const abortScan = () => {
    if (scanAbortControllerRef.current) {
      scanAbortControllerRef.current.abort();
    }
  };

  const abortSimulation = () => {
    if (simAbortControllerRef.current) {
      simAbortControllerRef.current.abort();
    }
  };

  async function fetchJobs() {
    try {
      const res = await fetchApi("/api/jobs");
      if (!res.ok) return;
      const data = await res.json();
      // The endpoint returns {jobs, total, ...}; older builds returned a bare
      // array, so accept both.
      setJobs(Array.isArray(data) ? data : data.jobs ?? []);
      if (!Array.isArray(data) && typeof data.total === "number") {
        setJobsTotal(data.total);
      }
    } catch (err) {
      console.error("Error fetching jobs:", err);
    }
  }

  async function restoreStructureFromProject(projectToRestore: Project) {
    if (!projectToRestore.pdb) return;
    setStatus({ kind: "busy", message: "Loading saved structure" });
    try {
      const metadata = await loadStructureFromPath(projectToRestore.pdb);
      setStructure(metadata);
      setProject((current) => {
        const next = structuredClone(current);
        const chain =
          metadata.chains.find((candidate) => candidate.id === next.attachment?.chain) ||
          metadata.chains[0];
        const residue =
          chain?.residues.find((candidate) => candidate.id === next.attachment?.residue) ||
          chain?.residues.find((candidate) => candidate.atoms.includes("CB")) ||
          chain?.residues[0];
        if (chain && residue) {
          next.attachment = {
            chain: chain.id,
            residue: residue.id,
            atom:
              residue.atoms.includes(next.attachment?.atom || "")
                ? next.attachment?.atom
                : firstUsefulAtom(residue),
          };
          syncDonor(next);
        }
        return next;
      });
      setStatus({ kind: "ok", message: `${metadata.filename} loaded: ${metadata.atom_count} atoms` });
    } catch (error) {
      setStructure(null);
      setStatus({
        kind: "error",
        message: error instanceof Error ? error.message : "Failed to load saved structure",
      });
    }
  }

  function projectFromQueryString() {
    const params = new URLSearchParams(window.location.search);
    const pdbPath = params.get("pdb");
    if (!pdbPath) return null;

    const next = structuredClone(DEFAULT_PROJECT);
    next.pdb = pdbPath;
    next.attachment = {
      chain: params.get("chain") || next.attachment?.chain || "A",
      residue: Number(params.get("residue")) || next.attachment?.residue || 1,
      atom: params.get("atom") || next.attachment?.atom || "CB",
    };
    syncDonor(next);
    return withProjectDefaults(next);
  }

  // Load configuration from local storage on mount
  useEffect(() => {
    fetchJobs();
    try {
      const queryProject = projectFromQueryString();
      const stored = localStorage.getItem("quest_project");
      if (queryProject) {
        const normalized = withProjectDefaults(queryProject);
        setProject(normalized);
        setProjectText(projectJson(normalized));
        setSavedProject(normalized);
        void restoreStructureFromProject(normalized);
        window.history.replaceState({}, "", window.location.pathname);
      } else if (stored) {
        const parsed = withProjectDefaults(JSON.parse(stored));
        setProject(parsed);
        setProjectText(projectJson(parsed));
        setSavedProject(parsed);
        void restoreStructureFromProject(parsed);
      } else {
        setSavedProject(DEFAULT_PROJECT);
      }
    } catch (e) {
      console.error("Failed to load project from localStorage", e);
    }

    try {
      const storedSurface = localStorage.getItem("quest_surface_rendering");
      if (storedSurface) {
        setSurfaceRendering(storedSurface === "true");
      }
    } catch (e) {
      console.error("Failed to load surface rendering setting from localStorage", e);
    }

    try {
      const storedSparsity = localStorage.getItem("quest_render_sparsity");
      if (storedSparsity) {
        setRenderSparsity(Number(storedSparsity));
      }
    } catch (e) {
      console.error("Failed to load render sparsity setting from localStorage", e);
    }

    try {
      const storedAuto = localStorage.getItem("quest_auto_simulate");
      if (storedAuto) {
        setAutoSimulate(storedAuto === "true");
      }
    } catch (e) {
      console.error("Failed to load auto simulate setting from localStorage", e);
    }

    // Load dyes repository
    fetchApi("/api/dyes")
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data) setDyes(data);
        else console.warn("Dyes repository unavailable; continuing with custom dye parameters.");
      })
      .catch(() => console.warn("Dyes repository unavailable; continuing with custom dye parameters."));

    // Load the residue chemistry that makes quenching amino-acid specific.
    fetchApi("/api/quenching-defaults")
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data) setQuenchingDefaults(data);
        else console.warn("Quenching defaults unavailable; using built-in residue chemistry.");
      })
      .catch(() => console.warn("Quenching defaults unavailable; using built-in residue chemistry."));
  }, []);

  // Sync settings back to localStorage
  useEffect(() => {
    localStorage.setItem("quest_surface_rendering", String(surfaceRendering));
  }, [surfaceRendering]);

  useEffect(() => {
    localStorage.setItem("quest_render_sparsity", String(renderSparsity));
  }, [renderSparsity]);

  useEffect(() => {
    localStorage.setItem("quest_auto_simulate", String(autoSimulate));
  }, [autoSimulate]);

  // Pick up the language the browser asks for, once, before anything is
  // fetched — but only if we have a translation for it, and never over a
  // choice the user already made here.
  useEffect(() => {
    let stored: string | null = null;
    try {
      stored = localStorage.getItem("quest_locale");
    } catch (e) {
      // Private browsing; fall through to the navigator language.
    }
    fetchApi("/api/i18n")
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (!data) return;
        setLocales(data.available ?? {});
        const preferred =
          stored ??
          (navigator.language || "en").split("-")[0].toLowerCase();
        if (data.available && preferred in data.available) setLocale(preferred);
      })
      .catch(() =>
        console.warn("Locale list unavailable; continuing in English.")
      );
  }, []);

  // Language changed: refetch everything whose text is translated. The catalog
  // is the parameter labels and help; the strings are the rest of the chrome.
  useEffect(() => {
    try {
      localStorage.setItem("quest_locale", locale);
    } catch (e) {
      // Nothing to do: the language still applies to this session.
    }

    fetchApi(`/api/parameter-catalog?locale=${encodeURIComponent(locale)}`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data) setCatalog(data);
        else
          console.warn(
            "Parameter catalog unavailable; continuing without parameter help."
          );
      })
      .catch(() =>
        console.warn(
          "Parameter catalog unavailable; continuing without parameter help."
        )
      );

    fetchApi(`/api/i18n?locale=${encodeURIComponent(locale)}`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data) {
          setUiStrings(data.strings ?? {});
          setLocales(data.available ?? {});
        }
      })
      .catch(() => console.warn("UI strings unavailable; continuing in English."));
  }, [locale]);

  // An unknown key renders its fallback, or the key itself — visible in
  // development, harmless in production, never a crash.
  const t = useCallback(
    (key: string, fallback?: string) => uiStrings[key] ?? fallback ?? key,
    [uiStrings]
  );

  useEffect(() => {
    if (project && JSON.stringify(project) !== JSON.stringify(DEFAULT_PROJECT)) {
      localStorage.setItem("quest_project", JSON.stringify(project));
    }
  }, [project]);

  // Sync project text area whenever project changes
  useEffect(() => {
    setProjectText(projectJson(project));
  }, [project]);

  // Handle selected scan residues setup on structure change
  useEffect(() => {
    if (structure) {
      const initial: Record<string, boolean> = {};
      structure.chains.forEach((chain) => {
        chain.residues.forEach((res) => {
          initial[`${chain.id}_${res.id}`] = true;
        });
      });
      setSelectedScanResidues(initial);
      setScanResults(null);
      setScanProgress(null);
      setScanStatus({ kind: "idle", message: "Structure loaded, ready to scan" });
    } else {
      setSelectedScanResidues({});
      setScanResults(null);
      setScanProgress(null);
      setScanStatus({ kind: "idle", message: "Upload structure to scan" });
    }
  }, [structure]);

  // Sync active dye with project configuration
  useEffect(() => {
    if (!dyes || Object.keys(dyes).length === 0) return;
    let matchedName = "Custom";
    for (const [name, preset] of Object.entries(dyes)) {
      const presetInteractions = normalizeAminoAcidInteractions(preset.amino_acid_interactions);
      const projectInteractions = normalizeAminoAcidInteractions(project.amino_acid_interactions);
      const match = (
        Math.abs((preset.av_parameter?.linker_length ?? 0) - (project.av_parameter?.linker_length ?? 0)) < 1e-4 &&
        Math.abs((preset.av_parameter?.linker_width ?? 0) - (project.av_parameter?.linker_width ?? 0)) < 1e-4 &&
        Math.abs((preset.av_parameter?.radius1 ?? 0) - (project.av_parameter?.radius1 ?? 0)) < 1e-4 &&
        Math.abs((preset.tau0 ?? 0) - (project.tau0 ?? 0)) < 1e-4 &&
        Math.abs((preset.D ?? 0) - (project.fret?.dyes?.[0]?.D ?? 0)) < 1e-4 &&
        Math.abs((preset.slow_radius ?? DEFAULT_DYE_STICKINESS.slow_radius) - (project.slow_radius ?? DEFAULT_DYE_STICKINESS.slow_radius)) < 1e-4 &&
        JSON.stringify(presetInteractions) === JSON.stringify(projectInteractions)
      );
      if (match) {
        matchedName = name;
        break;
      }
    }
    setSelectedDyeName(matchedName);
  }, [project, dyes]);

  function handleDyeChange(dyeName: string) {
    if (dyeName === "Custom") return;
    const preset = dyes[dyeName];
    if (!preset) return;
    setProject((current) => {
      const next = structuredClone(current);
      if (!next.av_parameter) next.av_parameter = {};
      next.av_parameter.linker_length = preset.av_parameter?.linker_length;
      next.av_parameter.linker_width = preset.av_parameter?.linker_width;
      next.av_parameter.radius1 = preset.av_parameter?.radius1;
      next.tau0 = preset.tau0;
      next.slow_radius = n(preset.slow_radius, DEFAULT_DYE_STICKINESS.slow_radius);
      next.amino_acid_interactions = normalizeAminoAcidInteractions(preset.amino_acid_interactions);
      if (next.fret?.dyes?.[0]) {
        next.fret.dyes[0].D = preset.D;
      }
      syncDonor(next);
      setSavedProject(next);
      return next;
    });
  }

  const allResidues = useMemo(() => {
    if (!structure) return [];
    return structure.chains.flatMap((chain) =>
      chain.residues.map((res) => ({
        chain: chain.id,
        residue: res.id,
        residue_name: res.name,
        atom: firstUsefulAtom(res),
        key: `${chain.id}_${res.id}`,
      }))
    );
  }, [structure]);

  function updateProject(path: string, value: string | number | boolean) {
    setProject((current) => setNested(current, path, value));
  }

  function applyStructure(metadata: StructureMetadata) {
    // Default to the largest chain: files often list a short ligand or ion
    // chain first, which is never the one you want to label.
    const chain = [...metadata.chains].sort((a, b) => b.residues.length - a.residues.length)[0];
    const residue = chain?.residues.find((candidate) => candidate.atoms.includes("CB")) || chain?.residues[0];
    const atom = residue ? firstUsefulAtom(residue) : "CB";
    setStructure(metadata);
    setProject((current) => {
      const next = structuredClone(current);
      next.pdb = metadata.path;
      next.attachment = { chain: chain?.id || "", residue: residue?.id || 1, atom };
      syncDonor(next);
      return next;
    });
    setResult(null);
    setStatus({ kind: "ok", message: `${metadata.filename} loaded: ${metadata.atom_count} atoms` });
  }

  async function uploadPdb(file: File) {
    setStatus({ kind: "busy", message: "Parsing uploaded structure" });
    const body = new FormData();
    body.append("pdb_file", file);
    try {
      const response = await fetchApi("/api/structure/metadata", { method: "POST", body });
      const payload = await response.json();
      if (!response.ok) {
        setStatus({ kind: "error", message: payload.detail || "Failed to parse structure" });
        return;
      }
      applyStructure(payload);
    } catch (err) {
      setStatus({ kind: "error", message: String(err) });
    }
  }

  async function loadPdbId(pdbId: string) {
    const cleanId = pdbId.trim();
    if (!cleanId) {
      setStatus({ kind: "error", message: "Enter a PDB ID" });
      return;
    }
    setStatus({ kind: "busy", message: `Downloading PDB ID ${cleanId.toUpperCase()}` });
    try {
      const metadata = await loadStructureFromPath(cleanId);
      applyStructure(metadata);
    } catch (err) {
      setStructure(null);
      setStatus({ kind: "error", message: err instanceof Error ? err.message : String(err) });
    }
  }

  async function handleLoadJob(jobId: string) {
    if (!jobId) {
      setSelectedJobId(null);
      return;
    }
    setSelectedJobId(jobId);
    setStatus({ kind: "busy", message: `Loading job...` });
    try {
      const res = await fetchApi(`/api/jobs/${jobId}`);
      const payload = await res.json();
      if (!res.ok) {
        setStatus({ kind: "error", message: payload.detail || "Failed to load job" });
        return;
      }

      const normalizedProject = withProjectDefaults(payload.project);
      setResult({ ...payload, project: normalizedProject });
      setScanResults(payload.scan_results || null);
      lastSimulatedProjectRef.current = normalizeProjectForComparison(normalizedProject);
      setProject(normalizedProject);
      setSavedProject(normalizedProject);

      if (normalizedProject.pdb) {
        const body = new FormData();
        body.append("pdb_path", normalizedProject.pdb);
        const structureRes = await fetchApi("/api/structure/metadata", { method: "POST", body });
        const structurePayload = await structureRes.json();
        if (structureRes.ok) {
          setStructure(structurePayload);
        } else {
          console.error("Failed to load structure metadata for job PDB:", structurePayload.detail);
        }
      }

      setStatus({ kind: "ok", message: `Job loaded successfully` });
    } catch (err) {
      setStatus({ kind: "error", message: String(err) });
    }
  }

  function getFormData(activeProj?: Project) {
    const p = activeProj || project;
    const body = new FormData();
    body.append("project_json", projectJson(p));
    body.append("pdb_path", p.pdb || "");
    body.append("nbins", String(n(p.n_bins, 256)));
    body.append("tac_start", "0");
    body.append("tac_stop", "50");
    return body;
  }

  async function responsePayload(response: Response) {
    const text = await response.text();
    if (!text) return null;
    try {
      return JSON.parse(text);
    } catch {
      return { detail: text };
    }
  }

  async function runSimulation(activeProj?: Project) {
    const proj = activeProj || project;
    setStatus({ kind: "busy", message: "Running simulation" });

    const controller = new AbortController();
    simAbortControllerRef.current = controller;

    lastSimulatedProjectRef.current = normalizeProjectForComparison(proj);
    try {
      const response = await fetchApi("/api/simulate", {
        method: "POST",
        body: getFormData(proj),
        signal: controller.signal,
      });
      const payload = await responsePayload(response);
      if (!response.ok) {
        setStatus({ kind: "error", message: payload?.detail || "Simulation failed" });
        return;
      }
      if (!payload) {
        setStatus({ kind: "error", message: "Simulation failed: empty response" });
        return;
      }
      const normalizedProject = withProjectDefaults(payload.project);
      setResult({ ...payload, project: normalizedProject });
      lastSimulatedProjectRef.current = normalizeProjectForComparison(normalizedProject);
      setProject(normalizedProject);
      setSavedProject(normalizedProject);
      setSelectedJobId(payload.job_id || null);
      setStatus({ kind: "ok", message: "Simulation complete" });
      fetchJobs();
    } catch (err: any) {
      if (err.name === "AbortError") {
        setStatus({ kind: "error", message: "Simulation aborted by user" });
      } else {
        setStatus({ kind: "error", message: String(err) });
      }
    } finally {
      if (simAbortControllerRef.current === controller) {
        simAbortControllerRef.current = null;
      }
    }
  }

  // Auto simulation triggers
  useEffect(() => {
    if (!autoSimulate || !structure?.pdb_text) return;
    const projectStr = normalizeProjectForComparison(project);
    if (projectStr === lastSimulatedProjectRef.current) return;

    const timer = setTimeout(() => {
      runSimulation(project);
    }, 600);

    return () => clearTimeout(timer);
  }, [project, structure, autoSimulate]);

  async function runScan() {
    if (!structure) return;
    const toScan = allResidues.filter((r) => scanAll || selectedScanResidues[r.key]);
    if (toScan.length === 0) {
      setScanStatus({ kind: "error", message: "No residues selected to scan" });
      return;
    }
    setScanStatus({ kind: "busy", message: `Scanning ${toScan.length} residues...` });
    setScanProgress({ current: 0, total: toScan.length });
    setScanResults([]);
    setIsScanning(true);

    const controller = new AbortController();
    scanAbortControllerRef.current = controller;

    try {
      const body = new FormData();
      body.append("project_json", projectJson(project));
      body.append("pdb_path", project.pdb || "");
      body.append("nbins", String(n(project.n_bins, 256)));
      body.append("tac_start", "0");
      body.append("tac_stop", "50");
      const sitesToScan = toScan.map((r) => ({
        chain: r.chain,
        residue: r.residue,
        residue_name: r.residue_name,
        atom: r.atom,
      }));
      body.append("sites_json", JSON.stringify(sitesToScan));
      
      const response = await fetchApi("/api/scan", {
        method: "POST",
        body,
        signal: controller.signal,
      });

      if (!response.ok) {
        let errMsg = "Scan failed";
        try {
          const errPayload = await response.json();
          errMsg = errPayload.detail || errMsg;
        } catch (_) {}
        setScanStatus({ kind: "error", message: errMsg });
        return;
      }

      if (!response.body) {
        throw new Error("ReadableStream not supported by browser/server response");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";
      const results: any[] = [];

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.trim()) {
            try {
              const chunk = JSON.parse(line);
              if (chunk.current !== undefined && chunk.total !== undefined) {
                setScanProgress({ current: chunk.current, total: chunk.total });
              }
              if (chunk.result) {
                results.push(chunk.result);
                setScanResults([...results]);
              }
            } catch (err) {
              console.error("Error parsing NDJSON line:", err, line);
            }
          }
        }
      }

      if (buffer.trim()) {
        try {
          const chunk = JSON.parse(buffer);
          if (chunk.current !== undefined && chunk.total !== undefined) {
            setScanProgress({ current: chunk.current, total: chunk.total });
          }
          if (chunk.result) {
            results.push(chunk.result);
            setScanResults([...results]);
          }
        } catch (err) {
          console.error("Error parsing final buffer line:", err, buffer);
        }
      }

      setScanStatus({ kind: "ok", message: `Successfully scanned ${toScan.length} residues` });
    } catch (err: any) {
      if (err.name === "AbortError") {
        setScanStatus({ kind: "error", message: "Scan aborted by user" });
      } else {
        setScanStatus({ kind: "error", message: String(err) });
      }
    } finally {
      setIsScanning(false);
      if (scanAbortControllerRef.current === controller) {
        scanAbortControllerRef.current = null;
      }
      setScanProgress(null);
    }
  }

  function copyScanResultsToClipboard() {
    if (!scanResults) return;
    const headers = [
      "Residue",
      "Status",
      "Lifetime (ns)",
      "Quantum Yield",
      "Contact Fraction",
      "AV Volume (Ang3)",
      "Contact Volume (Ang3)",
      "Contact/Free Ratio",
      "Quenched by",
    ];
    const rows = scanResults.map((r) => {
      const resName = r.name ? `${r.name} (${r.chain}:${r.residue})` : `${r.chain}:${r.residue} (${r.residue_name})`;
      const lifetime = r.lifetime !== null ? r.lifetime.toFixed(4) : "-";
      const qy = r.quantum_yield !== null ? r.quantum_yield.toFixed(4) : "-";
      const cf = r.contact_fraction !== null ? r.contact_fraction.toFixed(4) : "-";
      const avVol = r.av_volume !== null && r.av_volume !== undefined ? r.av_volume.toFixed(1) : "-";
      const contVol = r.contact_volume !== null && r.contact_volume !== undefined ? r.contact_volume.toFixed(1) : "-";
      const ratio = r.contact_to_free_ratio !== null && r.contact_to_free_ratio !== undefined ? r.contact_to_free_ratio.toFixed(4) : "-";
      const quenchers =
        r.quenchers?.length
          ? r.quenchers
              .map((q) => `${q.residue_name} ${((q.quench_rate_fraction ?? 0) * 100).toFixed(1)}%`)
              .join("; ")
          : "-";
      return [resName, r.status, lifetime, qy, cf, avVol, contVol, ratio, quenchers].join("\t");
    });
    const text = [headers.join("\t"), ...rows].join("\n");
    navigator.clipboard.writeText(text)
      .then(() => {
        alert("Scan results copied to clipboard!");
      })
      .catch((err) => {
        console.error("Failed to copy scan results:", err);
      });
  }

  async function loadStructureFromPath(pdbPath: string) {
    const body = new FormData();
    body.append("pdb_path", pdbPath);
    const response = await fetchApi("/api/structure/metadata", { method: "POST", body });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Failed to load structure metadata");
    }
    return payload as StructureMetadata;
  }

  function applyJson() {
    try {
      const parsed = withProjectDefaults(JSON.parse(projectText) as Project);
      setProject(parsed);
      setShowJson(false);
      setStatus({ kind: "ok", message: "JSON applied" });
      if (parsed.pdb) {
        setStatus({ kind: "busy", message: "Loading structure from JSON" });
        loadStructureFromPath(parsed.pdb)
          .then((metadata) => {
            setStructure(metadata);
            setStatus({ kind: "ok", message: `${metadata.filename} loaded: ${metadata.atom_count} atoms` });
          })
          .catch((error) => {
            setStructure(null);
            setStatus({ kind: "error", message: error instanceof Error ? error.message : String(error) });
          });
      }
      return true;
    } catch (error) {
      setStatus({ kind: "error", message: error instanceof Error ? error.message : "Invalid JSON" });
      return false;
    }
  }

  function handleClearState() {
    if (!window.confirm("Are you sure you want to clear the current structure and reset all parameters?")) {
      return;
    }
    setStructure(null);
    setProject(DEFAULT_PROJECT);
    setSavedProject(DEFAULT_PROJECT);
    setProjectText(projectJson(DEFAULT_PROJECT));
    setResult(null);
    setScanResults(null);
    setScanProgress(null);
    setSelectedScanResidues({});
    setSelectedJobId(null);
    lastSimulatedProjectRef.current = "";
    localStorage.removeItem("quest_project");

    setStatus({ kind: "idle", message: "Upload structure to start" });
    setScanStatus({ kind: "idle", message: "Upload structure to scan" });

    if (pdbInputRef.current) {
      pdbInputRef.current.value = "";
    }
  }

  function saveDyePreset() {
    const name = prompt("Enter a name for the new dye preset:", selectedDyeName === "Custom" ? "" : selectedDyeName);
    if (!name) return;
    const cleanName = name.trim();
    if (!cleanName) return;

    const preset = {
      name: cleanName,
      tau0: project.tau0 ?? 4.0,
      D: project.fret?.dyes?.[0]?.D ?? 7.0,
      av_parameter: {
        linker_length: project.av_parameter?.linker_length ?? 20.0,
        linker_width: project.av_parameter?.linker_width ?? 0.5,
        radius1: project.av_parameter?.radius1 ?? 3.5,
      },
      slow_radius: n(project.slow_radius, DEFAULT_DYE_STICKINESS.slow_radius),
      amino_acid_interactions: normalizeAminoAcidInteractions(project.amino_acid_interactions),
    };

    fetchApi("/api/dyes", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(preset),
    })
      .then((res) => {
        if (res.ok) return res.json();
        throw new Error("Failed to save dye preset");
      })
      .then((data) => {
        setDyes(data);
        setSavedProject(project);
        alert(`Dye preset "${cleanName}" saved successfully!`);
      })
      .catch((err) => {
        console.error(err);
        alert("Failed to save dye preset.");
      });
  }

  function saveParameter(parameterId?: string) {
    localStorage.setItem("quest_project", JSON.stringify(project));
    setSavedProject(project);

    const isDyeParam = [
      "av_parameter.linker_length",
      "av_parameter.linker_width",
      "av_parameter.radius1",
      "tau0",
      "fret.dyes.0.D",
      "slow_radius",
      "amino_acid_interactions",
    ].includes(parameterId ?? "");

    if (isDyeParam) {
      saveDyePreset();
    } else {
      alert("Parameter settings saved successfully!");
    }
  }

  function downloadParameters() {
    const blob = new Blob([JSON.stringify(project, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${project.output_file || "quest"}-parameters.json`;
    link.click();
    URL.revokeObjectURL(url);
  }

  function downloadResult() {
    if (result && result.job_id) {
      const link = document.createElement("a");
      link.href = `/api/jobs/${result.job_id}/download`;
      link.download = `quest_simulation_${result.job_id}.zip`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      return;
    }
    if (!result && !scanResults) return;
    const payload = {
      ...result,
      scan_results: scanResults,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `${project.output_file || "quest"}-result.json`;
    link.click();
    URL.revokeObjectURL(link.href);
  }

  function onChainChange(chainId: string) {
    if (!structure) return;
    const chains = structure.chains;
    const chain = chains.find((item) => item.id === chainId);
    const residue = chain?.residues[0];
    const atom = residue ? firstUsefulAtom(residue) : "";
    setProject((current) => {
      const next = structuredClone(current);
      next.attachment = { chain: chainId, residue: residue?.id || 1, atom };
      syncDonor(next);
      return next;
    });
  }

  function onResidueChange(id: number) {
    if (!structure) return;
    const chains = structure.chains;
    const selectedChain = chains.find((chain) => chain.id === project.attachment?.chain) || chains[0];
    const residues = selectedChain?.residues || [];
    const residue = residues.find((item) => item.id === id);
    const atom = residue ? firstUsefulAtom(residue) : project.attachment?.atom || "";
    setProject((current) => {
      const next = structuredClone(current);
      next.attachment = { ...next.attachment, residue: id, atom };
      syncDonor(next);
      return next;
    });
  }

  function onSequenceResidueSelect(chainId: string, residueId: number) {
    if (!structure) return;
    const chain = structure.chains.find((item) => item.id === chainId);
    const residue = chain?.residues.find((item) => item.id === residueId);
    const atom = residue ? firstUsefulAtom(residue) : project.attachment?.atom || "";
    setProject((current) => {
      const next = structuredClone(current);
      next.attachment = {
        ...next.attachment,
        chain: chain?.id || chainId,
        residue: residueId,
        atom,
      };
      syncDonor(next);
      return next;
    });
  }

  return (
    <AppContext.Provider
      value={{
        project,
        setProject,
        savedProject,
        setSavedProject,
        projectText,
        setProjectText,
        structure,
        setStructure,
        result,
        setResult,
        status,
        setStatus,
        showJson,
        setShowJson,
        showCitationModal,
        setShowCitationModal,
        surfaceRendering,
        setSurfaceRendering,
        renderSparsity,
        setRenderSparsity,
        scanResults,
        setScanResults,
        scanAll,
        setScanAll,
        selectedScanResidues,
        setSelectedScanResidues,
        scanProgress,
        setScanProgress,
        scanStatus,
        setScanStatus,
        isScanning,
        setIsScanning,
        jobs,
        setJobs,
        jobsTotal,
        dyes,
        selectedDyeName,
        setSelectedDyeName,
        autoSimulate,
        setAutoSimulate,
        catalog,
        quenchingDefaults,
        locale,
        setLocale,
        locales,
        t,
        selectedJobId,
        setSelectedJobId,
        pdbInputRef,
        
        fetchJobs,
        handleLoadJob,
        handleDyeChange,
        uploadPdb,
        loadPdbId,
        runSimulation,
        abortSimulation,
        runScan,
        abortScan,
        copyScanResultsToClipboard,
        applyJson,
        handleClearState,
        saveDyePreset,
        saveParameter,
        downloadParameters,
        downloadResult,
        onChainChange,
        onResidueChange,
        onSequenceResidueSelect,
        updateProject,
        allResidues,
      }}
    >
      {children}
    </AppContext.Provider>
  );
}

export function useAppContext() {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error("useAppContext must be used within an AppProvider");
  }
  return context;
}
