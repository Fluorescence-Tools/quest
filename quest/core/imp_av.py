from __future__ import annotations

import contextlib
import os
import tempfile
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

try:  # pragma: no cover - development convenience
    # Splice an imp-tricks checkout in before probing IMP, so its pure-Python
    # extensions (IMP.bff.av, IMP.bff.distance_metrics, IMP.cgmol.*) are
    # importable from source without installing anything. imp-tricks is under
    # active development, so the source tree -- not an installed copy -- is the
    # thing that has to work.
    from ..imp_tricks import enable_imp_tricks

    IMP_TRICKS_ENABLED = enable_imp_tricks()
except Exception:  # pragma: no cover - never fatal
    IMP_TRICKS_ENABLED = []

try:  # pragma: no cover - optional heavy dependency
    import IMP
    import IMP.algebra
    import IMP.atom
    import IMP.bff
    import IMP.core
    import IMP.em

    #: True when the *compiled* IMP.bff AV decorator is present. Note this is a
    #: different thing from imp-tricks' pure-Python ``IMP.bff.av`` module; see
    #: okf/specs/imp-tricks-dedup.md.
    HAS_IMP_BFF = hasattr(IMP.bff, "AV")
except Exception:  # pragma: no cover - keep optional dependency lazy
    HAS_IMP_BFF = False


def imp_tricks_av_available() -> bool:
    """Whether imp-tricks' maintained ``compute_av`` can be imported."""
    try:  # pragma: no cover - depends on the checkout
        from IMP.bff.av import compute_av  # noqa: F401
    except Exception:
        return False
    return True


@dataclass
class IMPAccessibleVolumeResult:
    points: np.ndarray
    density: np.ndarray
    grid_size: int
    attachment_point: np.ndarray


def _ensure_pdb_path(structure) -> Tuple[str, contextlib.AbstractContextManager]:
    """Return a context manager yielding a pdb path for IMP."""
    if isinstance(structure, str) and os.path.isfile(structure):
        @contextlib.contextmanager
        def _existing():
            yield structure

        return structure, _existing()

    if hasattr(structure, "filename") and structure.filename and os.path.isfile(
        structure.filename
    ):
        @contextlib.contextmanager
        def _filename():
            yield structure.filename

        return structure.filename, _filename()

    if hasattr(structure, "write"):
        @contextlib.contextmanager
        def _tmp():
            fd, tmp_path = tempfile.mkstemp(suffix=".pdb")
            os.close(fd)
            structure.write(tmp_path)
            try:
                yield tmp_path
            finally:
                with contextlib.suppress(FileNotFoundError):
                    os.remove(tmp_path)

        return "", _tmp()

    raise ValueError("Structure must be a filepath or provide a write() method.")


def _select_attachment_particle(
    hierarchy: "IMP.atom.Hierarchy",
    residue_seq_number: int,
    atom_name: str,
    chain_identifier: str | None,
) -> "IMP.Particle":
    selection = IMP.atom.Selection(hierarchy)
    if chain_identifier:
        selection.set_chain_id(chain_identifier)
    if residue_seq_number is not None:
        selection.set_residue_index(int(residue_seq_number))
    if atom_name:
        selection.set_atom_type(IMP.atom.AtomType(str(atom_name)))
    particles = selection.get_selected_particles()
    if not particles:
        raise ValueError(
            f"Unable to locate attachment site {chain_identifier}:{residue_seq_number}:{atom_name}"
        )
    return particles[0]


def _path_map_header(path_map: "IMP.bff.PathMap"):
    """Return the PathMapHeader, tolerating an incomplete SWIG build.

    IMP 2.24 ships ``PathMap.get_path_map_header`` bound to a C symbol
    (``DensityMap_get_path_map_header``) that is **not in the compiled
    ``_IMP_bff``**, so calling it raises ``AttributeError``. The writable
    accessor is bound to a different symbol and works, and the header is only
    read here. Without this the whole IMP AV backend is unusable — which went
    unnoticed because environments without IMP.bff fall back to LabelLib.
    """
    try:
        return path_map.get_path_map_header()
    except AttributeError:
        return path_map.get_path_map_header_writable()


def _path_map_density_to_grid(path_map: "IMP.bff.PathMap") -> Tuple[np.ndarray, int]:
    header = path_map.get_header()
    nx, ny, nz = header.get_nx(), header.get_ny(), header.get_nz()
    if not (nx == ny == nz):
        raise ValueError(
            "IMP PathMap grid is not cubic; Quest requires equal nx, ny, nz."
        )
    tile_values: Iterable[float] = path_map.get_tile_values(
        IMP.bff.PM_TILE_ACCESSIBLE_DENSITY,
        (
            0.0,
            _path_map_header(path_map).get_max_path_length(),
        ),
    )
    density = np.asarray(tile_values, dtype=np.float32).reshape((nx, ny, nz), order="C")
    density = np.where(density > 0.0, 1, 0).astype(np.uint8)
    return density, nx


def _model_from_atoms(atoms) -> Tuple["IMP.Model", list, np.ndarray]:
    """Build an IMP model from QuEst's atom array.

    The array is the single source of truth (okf/specs/structure-io.md): its
    coordinates, its radii, its atom set. Re-reading the file here — which this
    function used to do — silently discarded every in-memory transformation, so
    a coarse-grained project got the all-atom volume, and `.cif` projects failed
    because that re-read was `read_pdb` only.

    Radii come from the array, **not** from IMP: IMP's defaults are ~19 % larger
    than the Bondi-like values the FPS/LabelLib convention assumes, and letting
    the two backends use different radii is what made them disagree by ~20 % on
    volume.
    """
    model = IMP.Model()
    root = IMP.atom.Hierarchy.setup_particle(IMP.Particle(model))
    coords = np.ascontiguousarray(atoms["coord"], dtype=np.float64)
    radii = np.ascontiguousarray(atoms["radius"], dtype=np.float64)
    # Keep the particles we made: they are what the attachment index refers to.
    # Reaching back into the model by position instead is how this went wrong
    # once already — `Model` has no `get_particles()`, and even if it did, the
    # ordering would be an assumption rather than a fact.
    particles = []
    for index in range(coords.shape[0]):
        particle = IMP.Particle(model)
        xyzr = IMP.core.XYZR.setup_particle(particle)
        xyzr.set_coordinates(IMP.algebra.Vector3D(*coords[index]))
        xyzr.set_radius(float(radii[index]))
        child = IMP.atom.Hierarchy.setup_particle(particle)
        child.set_name(f"atom_{index}")
        root.add_child(child)
        particles.append(particle)
    return model, particles, coords


def _attachment_index(atoms, residue_seq_number, atom_name, chain_identifier) -> int:
    """Row of the atom array carrying the labelling site."""

    def text(values):
        return np.array([
            value.decode().strip() if isinstance(value, bytes) else str(value).strip()
            for value in values
        ])

    mask = np.ones(atoms.shape[0], dtype=bool)
    if chain_identifier:
        mask &= text(atoms["chain"]) == str(chain_identifier).strip()
    if residue_seq_number is not None:
        mask &= atoms["res_id"] == int(residue_seq_number)
    if atom_name:
        mask &= text(atoms["atom_name"]) == str(atom_name).strip()
    found = np.flatnonzero(mask)
    if not found.size:
        raise ValueError(
            f"Unable to locate attachment site "
            f"{chain_identifier}:{residue_seq_number}:{atom_name}"
        )
    return int(found[0])


def build_imp_accessible_volume(
    structure,
    residue_seq_number: int,
    atom_name: str,
    chain_identifier: str | None,
    linker_length: float,
    linker_width: float,
    radii: Tuple[float, float, float],
    allowed_sphere_radius: float,
    simulation_grid_resolution: float,
) -> IMPAccessibleVolumeResult:
    if not HAS_IMP_BFF:
        raise RuntimeError(
            "IMP.bff is required to calculate accessible volumes. "
            "Install the imp-bff package to enable this feature."
        )

    atoms = structure.atoms
    model, particles, coords = _model_from_atoms(atoms)
    index = _attachment_index(atoms, residue_seq_number, atom_name, chain_identifier)
    source_particle = particles[index]

    av_particle = IMP.Particle(model)
    IMP.bff.AV.do_setup_particle(
        model,
        av_particle,
        source_particle,
        linker_length=linker_length,
        linker_width=linker_width,
        radii=radii,
        # Below ~2 A the attachment atom's own neighbours block every starting
        # voxel and the path search returns an empty volume with no error;
        # imp.bff's own reference test uses 2.0 and expects zero points at 1.0.
        allowed_sphere_radius=allowed_sphere_radius,
        contact_volume_thickness=0.0,
        contact_volume_trapped_fraction=-1,
        simulation_grid_resolution=simulation_grid_resolution,
    )
    av = IMP.bff.AV(model, av_particle)
    av.resample()

    path_map = av.get_map()
    density, grid_size = _path_map_density_to_grid(path_map)
    xyz_density = np.asarray(path_map.get_xyz_density(), dtype=np.float64)
    if xyz_density.size:
        points = np.ascontiguousarray(
            xyz_density.reshape((-1, xyz_density.shape[-1]))[:, :3]
        )
    else:
        points = np.zeros((0, 3), dtype=np.float64)

    return IMPAccessibleVolumeResult(
        points=points,
        density=density,
        grid_size=int(grid_size),
        attachment_point=coords[index],
    )
