"use client";

/**
 * Map AutoForm view-spec `attr` names to project JSON paths.
 *
 * The view spec (generated from the parameter catalog) names each field by the
 * flat attribute it binds on the Qt `ProjectFormModel` (`attachment_chain`,
 * `fret_enabled`, `fret_R0`, …). The project JSON those names abbreviate is
 * nested (`attachment.chain`, `fret.enabled`, `fret.R0_matrix[0][1]`, …). The
 * web AutoForm must translate between the two: it reads the current value with
 * the nested path and writes edits back to the same nested path.
 *
 * This is the web half of `quest/gui/form_model.py::FIELD_PATHS`; keep the two
 * in step. `test_autoform_blueprint.py` fails if the catalog grows a parameter
 * this map does not cover.
 */
export const ATTR_TO_PATH: Record<string, string> = {
  pdb: "pdb",
  attachment_chain: "attachment.chain",
  attachment_residue: "attachment.residue",
  attachment_atom: "attachment.atom",
  av_linker_length: "av_parameter.linker_length",
  av_linker_width: "av_parameter.linker_width",
  av_radius1: "av_parameter.radius1",
  dg: "dg",
  tau0: "tau0",
  donor_D: "fret.dyes.0.D",
  critical_distance: "critical_distance",
  t_max: "t_max",
  t_step: "t_step",
  n_photons: "n_photons",
  n_bins: "n_bins",
  parallel_trajectories: "parallel_trajectories",
  slow_radius: "slow_radius",
  save_avs: "save_avs",
  output_file: "output_file",
  skip_frame: "skip_frame",
  coarse_grained: "coarse_grained",
  random_seed: "random_seed",
  fret_enabled: "fret.enabled",
  fret_R0: "fret.R0_matrix.0.1",
  fret_kappa2: "fret.kappa2",
  acceptor_chain: "fret.dyes.1.attachment.chain",
  acceptor_residue: "fret.dyes.1.attachment.residue",
  acceptor_atom: "fret.dyes.1.attachment.atom",
  acceptor_linker_length: "fret.dyes.1.av_parameter.linker_length",
  acceptor_linker_width: "fret.dyes.1.av_parameter.linker_width",
  acceptor_radius1: "fret.dyes.1.av_parameter.radius1",
  acceptor_D: "fret.dyes.1.D",
  acceptor_slow_radius: "fret.dyes.1.slow_radius",
  acceptor_dynamics: "fret.acceptor_dynamics",
};

/** Resolve a view-spec `attr` to the project path it edits. */
export function attrToPath(attr: string): string {
  return ATTR_TO_PATH[attr] ?? attr;
}
