from __future__ import annotations

import contextlib
import os
import tempfile
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

try:  # pragma: no cover - optional heavy dependency
    import IMP
    import IMP.algebra
    import IMP.atom
    import IMP.bff
    import IMP.core
    import IMP.em

    HAS_IMP_BFF = True
except Exception:  # pragma: no cover - keep optional dependency lazy
    HAS_IMP_BFF = False


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
            path_map.get_path_map_header().get_max_path_length(),
        ),
    )
    density = np.asarray(tile_values, dtype=np.float32).reshape((nx, ny, nz), order="C")
    density = np.where(density > 0.0, 1, 0).astype(np.uint8)
    return density, nx


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

    pdb_hint, ctx_mgr = _ensure_pdb_path(structure)
    with ctx_mgr as pdb_path:
        path_to_use = pdb_hint or pdb_path
        if not path_to_use:
            path_to_use = pdb_path
        model = IMP.Model()
        hierarchy = IMP.atom.read_pdb(path_to_use, model)
        attachment_particle = _select_attachment_particle(
            hierarchy, residue_seq_number, atom_name, chain_identifier
        )

        av_particle = IMP.Particle(model)
        build_kwargs = dict(
            linker_length=linker_length,
            linker_width=linker_width,
            radii=radii,
            allowed_sphere_radius=allowed_sphere_radius,
            contact_volume_thickness=0.0,
            contact_volume_trapped_fraction=-1,
            simulation_grid_resolution=simulation_grid_resolution,
        )
        IMP.bff.AV.do_setup_particle(model, av_particle, attachment_particle, **build_kwargs)
        av = IMP.bff.AV(model, av_particle)
        av.resample()

        path_map = av.get_map()
        density, grid_size = _path_map_density_to_grid(path_map)
        xyz_density = path_map.get_xyz_density()
        if xyz_density:
            points = np.asarray(
                [[vec[0], vec[1], vec[2]] for vec in xyz_density], dtype=np.float64
            )
        else:
            points = np.zeros((0, 3), dtype=np.float64)

        attachment_point = np.asarray(av.get_source_coordinates(), dtype=np.float64)

    return IMPAccessibleVolumeResult(
        points=points,
        density=density,
        grid_size=int(grid_size),
        attachment_point=attachment_point,
    )
