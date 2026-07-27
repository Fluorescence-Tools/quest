from __future__ import annotations

import contextlib
import os
import tempfile
import threading
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


#: Serialises IMP object *construction*, not IMP computation.
#:
#: IMP builds its decorators through SWIG, which is not safe to do from several
#: threads at once. `quest/core/pdb.py` already locks the parse for this reason
#: ("Wrong number or type of arguments for overloaded function 'new_Atom'");
#: the accessible-volume path hits the identical failure one layer down
#: (`'XYZR_setup_particle'` receiving a particle *name* instead of a particle)
#: as soon as a scan runs sites in a pool. It only began failing when IMP.bff
#: became the default backend -- with LabelLib first, no thread ever entered
#: this code.
#:
#: The lock deliberately does **not** cover `av.resample()`: measured on 148l
#: E36, resample is 636 ms of a 662 ms build (96 %) while all the SWIG
#: construction is 22 ms (3 %). Serialising the compute would cost a threaded
#: scan almost all of its parallelism to guard a phase that is not the one
#: failing. Each thread resamples its own model.
_IMP_BUILD_LOCK = threading.Lock()


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
    #: Cartesian position of the ``density[0, 0, 0]`` voxel, read from the
    #: PathMap's own header. **Not** derivable from ``attachment_point``: IMP
    #: sizes the map itself and the attachment atom does not land on the middle
    #: voxel when the grid has an even number of them. Assuming it did put every
    #: derived grid half a voxel out -- see :func:`_path_map_density_to_grid`.
    origin: np.ndarray


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


def _path_map_density_to_grid(
    path_map: "IMP.bff.PathMap",
) -> Tuple[np.ndarray, int, np.ndarray]:
    """Density grid, edge length and grid origin, in QuEst's ``[x, y, z]`` order.

    **The axis order is not cosmetic.** IMP numbers its voxels with *x* fastest
    (``i = x + nx*y + nx*ny*z``), so reshaping the flat tile values C-order into
    ``(nx, ny, nz)`` -- which makes the *last* axis fastest -- transposes the
    volume, exchanging x and z. That is what this did until 2026-07-27, and it
    survived unnoticed for two reasons: the backend was never the default, and
    an accessible volume is globular enough that the transpose still overlaps
    the truth by **83 %** of its voxels. It looks approximately right and is
    completely wrong -- the contact volume lands on the mirrored side of the
    protein, so quenchers that should be in reach are not, and the donor comes
    out unquenched.

    The origin comes from the map header for the same reason: IMP sizes the grid
    itself, and the attachment atom sits on the middle voxel only when the edge
    length is odd. Reconstructing the corner from the atom is off by one voxel
    whenever it is even (it is 86 on the 148l reference site).
    """
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
    # (nz, ny, nx) C-order, then transposed: x ends up the slowest-varying axis
    # of the array while remaining the fastest-varying index of the flat data.
    density = (
        np.asarray(tile_values, dtype=np.float32)
        .reshape((nz, ny, nx), order="C")
        .transpose(2, 1, 0)
    )
    density = np.ascontiguousarray(np.where(density > 0.0, 1, 0).astype(np.uint8))
    origin = np.array(
        [header.get_xorigin(), header.get_yorigin(), header.get_zorigin()],
        dtype=np.float64,
    )
    return density, nx, origin


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
    index = _attachment_index(atoms, residue_seq_number, atom_name, chain_identifier)

    # Everything that *creates* SWIG objects is serialised; `resample()` is not.
    # See _IMP_BUILD_LOCK -- the split is worth the two `with` blocks because
    # resample is 96 % of the wall clock and construction is 3 %.
    with _IMP_BUILD_LOCK:
        model, particles, coords = _model_from_atoms(atoms)
        source_particle = particles[index]

        av_particle = IMP.Particle(model)
        IMP.bff.AV.do_setup_particle(
            model,
            av_particle,
            source_particle,
            linker_length=linker_length,
            linker_width=linker_width,
            radii=radii,
            # Below ~2 A the attachment atom's own neighbours block every
            # starting voxel and the path search returns an empty volume with no
            # error; imp.bff's own reference test uses 2.0 and expects zero
            # points at 1.0.
            allowed_sphere_radius=allowed_sphere_radius,
            contact_volume_thickness=0.0,
            contact_volume_trapped_fraction=-1,
            simulation_grid_resolution=simulation_grid_resolution,
        )
        av = IMP.bff.AV(model, av_particle)

    # The C++ path search, on this thread's own model. Unlocked on purpose.
    av.resample()

    with _IMP_BUILD_LOCK:
        path_map = av.get_map()
        density, grid_size, origin = _path_map_density_to_grid(path_map)
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
        origin=origin,
    )
