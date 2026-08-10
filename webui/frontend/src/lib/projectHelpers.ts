import {
  AminoAcidInteraction,
  AminoAcidInteractions,
  PartialAminoAcidInteractions,
  Project,
  StructureResidue,
} from "../types";

export const AMINO_ACID_RESIDUES = [
  "ALA", "ARG", "ASN", "ASP", "CYS",
  "GLN", "GLU", "GLY", "HIS", "ILE",
  "LEU", "LYS", "MET", "PHE", "PRO",
  "SER", "THR", "TRP", "TYR", "VAL",
] as const;

/**
 * Atoms whose centroid defines each residue's quenching centre. Mirrors
 * `DEFAULT_QUENCHER_ATOMS` in `quest/lib/tools/dye_diffusion/dye_diffusion.py`;
 * the backend remains the source of truth and overrides these at runtime via
 * `/api/quenching-defaults`.
 */
export const DEFAULT_QUENCH_ATOMS: Record<string, string[]> = {
  ALA: ["CB"],
  ARG: ["CZ", "NE", "NH1", "NH2"],
  ASN: ["CG", "OD1", "ND2"],
  ASP: ["CG", "OD1", "OD2"],
  CYS: ["SG"],
  GLN: ["CD", "OE1", "NE2"],
  GLU: ["CD", "OE1", "OE2"],
  GLY: ["CA"],
  HIS: ["CG", "ND1", "CD2", "CE1", "NE2"],
  ILE: ["CB"],
  LEU: ["CB"],
  LYS: ["NZ"],
  MET: ["SD"],
  PHE: ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
  PRO: ["N", "CB", "CG", "CD"],
  SER: ["OG"],
  THR: ["OG1"],
  TRP: ["CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2", "NE1", "CG", "CD1"],
  TYR: ["CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
  VAL: ["CB"],
};

/** Residues with a redox-active side chain, i.e. the ones that can quench. */
export const PET_ACTIVE_RESIDUES = ["TRP", "TYR", "MET", "HIS", "CYS", "PRO"];

export const DEFAULT_AMINO_ACID_INTERACTIONS = Object.fromEntries(
  AMINO_ACID_RESIDUES.map((residue) => [
    residue,
    {
      slow_factor: 1.0,
      kQ: 0.0,
      quench_radius: null,
      quench_atoms: [...(DEFAULT_QUENCH_ATOMS[residue] ?? ["CB"])],
    },
  ])
) as AminoAcidInteractions;

export const DEFAULT_DYE_RADIUS = 3.5;

/**
 * Reference PET chemistry for a xanthene dye. `contact_distance` is measured
 * from the dye *surface*, so it transfers between dyes of different size;
 * `buildAminoAcidInteractions` adds the dye radius to get the centre-to-centre
 * quench radius the simulation uses. Mirrors `REFERENCE_PET_QUENCHING` in the
 * Python model.
 */
export const REFERENCE_PET_QUENCHING: Record<string, { kQ: number; contact_distance: number }> = {
  TRP: { kQ: 3.5, contact_distance: 5.0 },
  TYR: { kQ: 2.0, contact_distance: 5.0 },
  MET: { kQ: 1.67, contact_distance: 3.5 },
  HIS: { kQ: 1.0, contact_distance: 4.7 },
  CYS: { kQ: 0.8, contact_distance: 3.5 },
  PRO: { kQ: 2.0, contact_distance: 4.0 },
};

export function buildAminoAcidInteractions({
  kQScale = 1.0,
  slowFactorValue = 1.0,
  dyeRadius = DEFAULT_DYE_RADIUS,
}: { kQScale?: number; slowFactorValue?: number; dyeRadius?: number } = {}): AminoAcidInteractions {
  const table = normalizeAminoAcidInteractions();
  for (const residue of AMINO_ACID_RESIDUES) {
    table[residue].slow_factor = slowFactor(slowFactorValue, 1.0);
    const reference = REFERENCE_PET_QUENCHING[residue];
    if (!reference) continue;
    table[residue].kQ = Math.max(0, reference.kQ * kQScale);
    table[residue].quench_radius = quenchRadius(Math.max(0, dyeRadius) + reference.contact_distance);
  }
  return table;
}

export const DEFAULT_DYE_STICKINESS = {
  slow_radius: 8.5,
  amino_acid_interactions: DEFAULT_AMINO_ACID_INTERACTIONS,
} as const;
export const MAX_PARALLEL_TRAJECTORIES = 8;

/**
 * Forster radius written into a new project, in Angstrom.
 *
 * Every length in a QuEst project is in Angstrom, R0 included. The transfer
 * rate goes as (R0/r)^6, so a value entered in nanometres (~5.5) is ten times
 * too small and switches FRET off without any error.
 */
export const DEFAULT_FORSTER_RADIUS = 52.0;

/**
 * Isotropic dynamic average of the orientation factor.
 *
 * QuEst does not model kappa2 -- its dye is a structureless point diffusing in
 * a volume, with no transition dipole to orient -- so this is the averaging
 * regime the project *states*. A published Forster radius is already quoted at
 * 2/3, so leaving it here reproduces the transfer rate exactly.
 */
export const DEFAULT_KAPPA2 = 2 / 3;

/** A donor->acceptor Forster matrix, the shape the backend reads. */
export function forsterMatrix(r0: number) {
  return [
    [null, r0],
    [r0, null],
  ];
}

/** The acceptor dye entry a project gains when FRET is switched on. */
export function defaultAcceptorDye() {
  return {
    name: "A0",
    role: "acceptor",
    D: 7.0,
    attachment: { chain: "", residue: 1, atom: "CB" },
    av_parameter: { linker_length: 21.5, linker_width: 0.5, radius1: 3.5 },
  };
}

const DEFAULT_INTERACTIONS = buildAminoAcidInteractions({ slowFactorValue: 0.1 });

export const DEFAULT_PROJECT: Project = {
  pdb: "",
  attachment: { chain: "", residue: 1, atom: "CB" },
  av_parameter: { linker_length: 21.5, linker_width: 0.5, radius1: 3.5 },
  dg: 0.5,
  slow_radius: DEFAULT_DYE_STICKINESS.slow_radius,
  t_max: 200.0,
  t_step: 0.02,
  n_photons: 10000,
  n_bins: 256,
  parallel_trajectories: -1,
  coarse_grained: false,
  tau0: 4.2,
  amino_acid_interactions: DEFAULT_INTERACTIONS,
  // Fallback contact radius for residues that do not set their own
  // quench_radius. Measured from the dye centre, so it has to clear the dye
  // radius plus van der Waals contact.
  critical_distance: 8.5,
  output_file: "quest",
  save_avs: false,
  fret: {
    enabled: false,
    dyes: [
      {
        name: "D0",
        role: "donor",
        D: 7.0,
        slow_radius: DEFAULT_DYE_STICKINESS.slow_radius,
        amino_acid_interactions: DEFAULT_INTERACTIONS,
        attachment: { chain: "", residue: 1, atom: "CB" },
        av_parameter: { linker_length: 21.5, linker_width: 0.5, radius1: 3.5 },
      },
    ],
    R0_matrix: forsterMatrix(DEFAULT_FORSTER_RADIUS),
    kappa2: DEFAULT_KAPPA2,
  },
};

export function n(value: unknown, fallback = 0) {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

export function slowFactor(value: unknown, fallback = 1.0) {
  return Math.min(1.0, Math.max(0.0, n(value, fallback)));
}

export function parallelTrajectories(value: unknown, fallback = -1) {
  const parsed = Math.floor(n(value, fallback));
  return parsed === -1 ? -1 : Math.min(MAX_PARALLEL_TRAJECTORIES, Math.max(1, parsed));
}

export function quenchRadius(value: unknown): number | null {
  if (value === null || value === undefined || value === "") return null;
  const parsed = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) return null;
  return parsed;
}

export function quenchAtoms(value: unknown, fallback: string[]): string[] {
  const source = typeof value === "string" ? value.split(/[\s,]+/) : value;
  if (!Array.isArray(source)) return [...fallback];
  const atoms: string[] = [];
  for (const atom of source) {
    const name = String(atom).trim().toUpperCase();
    if (name && !atoms.includes(name)) atoms.push(name);
  }
  return atoms.length ? atoms : [...fallback];
}

export function normalizeAminoAcidInteractions(
  interactions?: PartialAminoAcidInteractions
): AminoAcidInteractions {
  return Object.fromEntries(
    AMINO_ACID_RESIDUES.map((residue) => {
      const item = interactions?.[residue] || {};
      const defaults = DEFAULT_QUENCH_ATOMS[residue] ?? ["CB"];
      const normalized: AminoAcidInteraction = {
        slow_factor: slowFactor(item.slow_factor, 1.0),
        kQ: Math.max(0, n(item.kQ, 0.0)),
        quench_radius: quenchRadius(item.quench_radius),
        quench_atoms: quenchAtoms(item.quench_atoms, defaults),
      };
      return [residue, normalized];
    })
  ) as AminoAcidInteractions;
}

export function setNested(project: Project, path: string, raw: string | number | boolean): Project {
  const clone = structuredClone(project);
  const parts = path.split(".");
  let cursor: Record<string, unknown> = clone as Record<string, unknown>;
  for (const part of parts.slice(0, -1)) {
    const current = cursor[part];
    if (!current || typeof current !== "object") cursor[part] = {};
    cursor = cursor[part] as Record<string, unknown>;
  }
  cursor[parts[parts.length - 1]] = raw;
  syncDonor(clone);
  return clone;
}

export function syncDonor(project: Project) {
  const donor = project.fret?.dyes?.[0];
  if (!donor) return;
  donor.attachment = { ...project.attachment };
  donor.av_parameter = { ...project.av_parameter };
  donor.slow_radius = n(project.slow_radius, DEFAULT_DYE_STICKINESS.slow_radius);
  donor.amino_acid_interactions = normalizeAminoAcidInteractions(project.amino_acid_interactions);
}

export function projectJson(project: Project) {
  return JSON.stringify(project, null, 2);
}

export function withProjectDefaults(project: Project): Project {
  const next = structuredClone(project);
  const donor = next.fret?.dyes?.[0];
  next.slow_radius = n(next.slow_radius ?? donor?.slow_radius, DEFAULT_DYE_STICKINESS.slow_radius);
  next.parallel_trajectories = parallelTrajectories(next.parallel_trajectories);
  next.amino_acid_interactions = normalizeAminoAcidInteractions(next.amino_acid_interactions || donor?.amino_acid_interactions);
  syncDonor(next);
  return next;
}

export function firstUsefulAtom(residue: StructureResidue) {
  return ["CB", "CA", "N", "C"].find((atom) => residue.atoms.includes(atom)) || residue.atoms[0] || "";
}

/**
 * The last residue of a chain a dye can actually be tethered to.
 *
 * Structures end in waters, ions and ligands; those have no CB and are not
 * labeling sites. Used to seed a FRET acceptor far from the donor without
 * landing on a HOH.
 */
export function lastLabelableResidue(chain?: { residues: StructureResidue[] }) {
  const residues = chain?.residues || [];
  for (let index = residues.length - 1; index >= 0; index -= 1) {
    if (residues[index].atoms.includes("CB")) return residues[index];
  }
  return residues.find((residue) => residue.atoms.includes("CA"));
}

export function selectionString(project: Project) {
  const chain = project.attachment?.chain;
  const residue = project.attachment?.residue;
  const atom = project.attachment?.atom;
  if (!chain || !residue || !atom) return "";
  return `:${chain} and ${residue} and .${atom}`;
}

export function downsamplePoints(points: Array<{ x: number; y: number }>, limit = 500) {
  if (points.length <= limit) return points;
  const step = points.length / limit;
  const downsampled = [];
  for (let i = 0; i < limit; i++) {
    downsampled.push(points[Math.floor(i * step)]);
  }
  return downsampled;
}

export function normalizeProjectForComparison(p: Project): string {
  if (!p) return "";
  return JSON.stringify({
    pdb: p.pdb,
    attachment: {
      chain: p.attachment?.chain,
      residue: p.attachment?.residue,
      atom: p.attachment?.atom
    },
    av_parameter: {
      linker_length: p.av_parameter?.linker_length,
      linker_width: p.av_parameter?.linker_width,
      radius1: p.av_parameter?.radius1
    },
    dg: p.dg,
    slow_radius: p.slow_radius,
    t_max: p.t_max,
    t_step: p.t_step,
    n_photons: p.n_photons,
    n_bins: p.n_bins,
    parallel_trajectories: parallelTrajectories(p.parallel_trajectories),
    coarse_grained: Boolean(p.coarse_grained),
    tau0: p.tau0,
    amino_acid_interactions: normalizeAminoAcidInteractions(p.amino_acid_interactions),
    critical_distance: p.critical_distance,
    fret: {
      enabled: p.fret?.enabled,
      dyes: p.fret?.dyes?.map(d => ({
        name: d.name,
        role: d.role,
        D: d.D,
        slow_radius: d.slow_radius,
        amino_acid_interactions: normalizeAminoAcidInteractions(d.amino_acid_interactions),
        attachment: d.attachment,
        av_parameter: d.av_parameter
      }))
    }
  });
}
