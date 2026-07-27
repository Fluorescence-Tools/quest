export type NglStage = {
  loadFile: (source: string | Blob, options?: Record<string, unknown>) => Promise<NglComponent>;
  autoView: (selection?: string) => void;
  handleResize: () => void;
  dispose: () => void;
  removeComponent: (component: any) => void;
  addComponentFromObject: (object: any) => any;
};

export type NglComponent = {
  addRepresentation: (name: string, options?: Record<string, unknown>) => void;
  autoView: (selection?: string) => void;
  removeAllRepresentations: () => void;
};

/**
 * Per-residue-type interaction parameters.
 *
 * `quench_radius` is measured from the dye centre to the residue's quenching
 * centre; `null` means "inherit the project-wide `critical_distance`".
 * `quench_atoms` names the atoms whose centroid is that quenching centre —
 * the redox-active moiety (indole, phenol, imidazole, thioether sulfur…),
 * not CB.
 */
export type AminoAcidInteraction = {
  slow_factor: number;
  kQ: number;
  quench_radius: number | null;
  quench_atoms: string[];
};

export type AminoAcidInteractions = Record<string, AminoAcidInteraction>;

export type PartialAminoAcidInteractions = Record<string, Partial<AminoAcidInteraction>>;

export type QuenchingDefaults = {
  residues: string[];
  default_interactions: AminoAcidInteractions;
  default_quench_atoms: Record<string, string[]>;
  reference_pet_quenching: Record<string, { kQ: number; contact_distance: number }>;
};

export type Project = {
  pdb?: string;
  attachment?: { chain?: string; residue?: number; atom?: string };
  av_parameter?: { linker_length?: number; linker_width?: number; radius1?: number };
  dg?: number;
  slow_radius?: number;
  t_max?: number;
  t_step?: number;
  n_photons?: number;
  n_bins?: number;
  parallel_trajectories?: number;
  coarse_grained?: boolean;
  tau0?: number;
  amino_acid_interactions?: PartialAminoAcidInteractions;
  critical_distance?: number;
  output_file?: string;
  save_avs?: boolean;
  fret?: {
    enabled?: boolean;
    dyes?: Array<{
      name?: string;
      role?: string;
      D?: number;
      slow_radius?: number;
      amino_acid_interactions?: PartialAminoAcidInteractions;
      attachment?: { chain?: string; residue?: number; atom?: string };
      av_parameter?: { linker_length?: number; linker_width?: number; radius1?: number };
    }>;
    R0_matrix?: Array<Array<number | null>>;
    /**
     * "trajectory" simulates the acceptor's own walk and uses the instantaneous
     * donor-acceptor distance; "averaged" averages over its static accessible
     * volume (the fast-acceptor limit). Absent means "trajectory".
     */
    acceptor_dynamics?: "trajectory" | "averaged";
  };
};

export type StructureResidue = { id: number; name: string; atoms: string[] };
export type StructureChain = { id: string; residues: StructureResidue[] };
export type StructureMetadata = {
  filename: string;
  path: string;
  atom_count: number;
  chains: StructureChain[];
  pdb_text: string;
};

export type SimulatePayload = {
  job_id?: string;
  time_ns: number[];
  donor_counts: number[];
  fret_counts: number[] | null;
  quantum_yield_donor: number;
  quantum_yield_fret: number | null;
  collisions_fraction: number;
  fret_efficiency: number | null;
  project: Project;
  trajectory_distance_to_mean: number[] | null;
  trajectory_time_ns?: number[] | null;
  trajectory_autocorr: number[] | null;
  trajectory_autocorr_lag_ns?: number[] | null;
  trajectory_points?: number[][] | null;
  av_points: number[][] | null;
  acv_points: number[][] | null;
  av_volume?: number | null;
  contact_volume?: number | null;
  contact_to_free_ratio?: number | null;
  aa_residence?: AAResidenceItem[];
};

export type AAResidenceItem = {
  residue_name: string;
  center_count: number;
  effective_slow_factor?: number | null;
  frames_near: number;
  fraction_near: number;
  total_time_ns: number;
  event_count: number;
  mean_event_time_ns: number;
  max_event_time_ns: number;
  mean_min_distance_angstrom?: number | null;
  attributed_frames_near?: number;
  attributed_fraction_near?: number;
  attributed_total_time_ns?: number;
  attributed_event_count?: number;
  attributed_mean_event_time_ns?: number;
  attributed_max_event_time_ns?: number;
  /** Quenching rate this residue type contributes on contact, 1/ns. */
  kQ?: number;
  /** Contact radius used for this residue type, Angstrom. */
  quench_radius?: number;
  /** Time-averaged quenching rate contributed by this residue type, 1/ns. */
  mean_quench_rate?: number;
  /** Share of the total quenching rate, 0..1. */
  quench_rate_fraction?: number;
  /** Fraction of frames in which this residue type was quenching. */
  fraction_frames_quenched?: number;
};

export type Status = { kind: "idle" | "busy" | "ok" | "error"; message: string };

/** One entry in the run picker. The full project/result live behind /api/jobs/{id}. */
export type JobSummary = {
  job_id: string;
  labeling_site: string;
  date: string;
};

export type ScanResultItem = {
  name?: string;
  chain: string;
  residue: number;
  residue_name: string;
  atom: string;
  status: "success" | "failed";
  lifetime: number | null;
  quantum_yield: number | null;
  contact_fraction: number | null;
  av_volume?: number | null;
  contact_volume?: number | null;
  contact_to_free_ratio?: number | null;
  /** Residue type contributing the largest share of the quenching rate. */
  dominant_quencher?: string | null;
  dominant_quencher_fraction?: number | null;
  /** Up to three strongest quenching residue types at this site. */
  quenchers?: Array<{
    residue_name: string;
    quench_rate_fraction?: number | null;
    mean_quench_rate?: number | null;
  }>;
  error?: string;
};
