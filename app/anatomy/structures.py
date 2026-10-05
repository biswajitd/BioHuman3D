"""
Structure-labelled layer files.

A layer file (``skeleton.vtp``, ``muscles.vtp`` …) holds many named anatomical
structures — "Right humerus", "Left biceps brachii". They are stored in ONE
``.vtp`` per layer (so the asset folder stays one-file-per-layer) with:

* cell data ``StructureId``  (int32) — which structure each triangle belongs to
* field data ``StructureNames`` (string array) — id → English name

At load time the viewport splits the layer back into one actor per structure,
which is what makes per-structure picking, highlighting, isolation, muscle
animation and disease effects possible. Files without these arrays load as a
single unnamed structure, exactly as before.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from app.core.vtk_utils import VTK_AVAILABLE, vtk

try:
    from vtkmodules.util.numpy_support import (numpy_to_vtk, numpy_to_vtkIdTypeArray,
                                               vtk_to_numpy)
except Exception:  # pragma: no cover
    from vtk.util.numpy_support import (numpy_to_vtk, numpy_to_vtkIdTypeArray,  # type: ignore
                                        vtk_to_numpy)

STRUCTURE_ID = "StructureId"
STRUCTURE_NAMES = "StructureNames"
SOURCE = "BioHuman3DSource"          # provenance tag, e.g. "bodyparts3d"

Named = Tuple[str, object]          # (structure name, vtkPolyData)


def _triangulated(poly):
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(poly)
    tri.PassVertsOff()
    tri.PassLinesOff()
    tri.Update()
    return tri.GetOutput()


def merge_structures(items: Sequence[Named]):
    """Combine named meshes into one labelled ``vtkPolyData``."""
    append = vtk.vtkAppendPolyData()
    names = vtk.vtkStringArray()
    names.SetName(STRUCTURE_NAMES)
    used = 0
    for name, poly in items:
        if poly is None or not poly.GetNumberOfCells():
            continue
        piece = vtk.vtkPolyData()
        piece.DeepCopy(_triangulated(poly))
        # Drop arrays that differ between pieces so the append keeps the ids.
        piece.GetPointData().Initialize()
        piece.GetCellData().Initialize()
        ids = numpy_to_vtk(np.full(piece.GetNumberOfCells(), used, dtype=np.int32), deep=True)
        ids.SetName(STRUCTURE_ID)
        piece.GetCellData().AddArray(ids)
        append.AddInputData(piece)
        names.InsertNextValue(str(name))
        used += 1
    if not used:
        return None
    append.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(append.GetOutput())
    out.GetFieldData().AddArray(names)
    return out


def write_structures(items: Sequence[Named], path: Path, *, normals: bool = True,
                     source: str = "") -> int:
    """Write a labelled layer file. Returns the triangle count (0 on failure)."""
    merged = merge_structures(items)
    if merged is None:
        return 0
    if source:
        tag = vtk.vtkStringArray()
        tag.SetName(SOURCE)
        tag.InsertNextValue(source)
        merged.GetFieldData().AddArray(tag)
    data = merged
    if normals:
        nrm = vtk.vtkPolyDataNormals()
        nrm.SetInputData(merged)
        nrm.SetFeatureAngle(60.0)
        nrm.SplittingOff()
        nrm.ConsistencyOn()
        nrm.AutoOrientNormalsOff()
        nrm.Update()
        data = nrm.GetOutput()
        data.GetFieldData().ShallowCopy(merged.GetFieldData())
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(data)
    writer.SetCompressorTypeToZLib()
    writer.SetDataModeToBinary()
    writer.Write()
    return data.GetNumberOfCells()


def structure_names(poly) -> List[str]:
    """Names stored in a labelled layer, or ``[]``."""
    if poly is None:
        return []
    arr = poly.GetFieldData().GetAbstractArray(STRUCTURE_NAMES)
    if arr is None:
        return []
    return [arr.GetValue(i) for i in range(arr.GetNumberOfValues())]


def source_of(poly) -> str:
    """Provenance tag written by the generator/importer ("" when absent)."""
    if poly is None:
        return ""
    arr = poly.GetFieldData().GetAbstractArray(SOURCE)
    return arr.GetValue(0) if arr is not None and arr.GetNumberOfValues() else ""


def is_labelled(poly) -> bool:
    return bool(poly is not None
                and poly.GetCellData().GetArray(STRUCTURE_ID) is not None
                and structure_names(poly))


def split_structures(poly, fallback_name: str = "") -> List[Named]:
    """Split a labelled layer into ``[(name, polydata)]``.

    Uses numpy on the triangle connectivity instead of N ``vtkThreshold`` passes,
    so a 1,500-structure BodyParts3D layer splits in well under a second.
    Unlabelled input returns a single ``(fallback_name, poly)`` entry.
    """
    if not is_labelled(poly):
        return [(fallback_name, poly)]

    names = structure_names(poly)
    tri = _triangulated(poly) if poly.GetNumberOfCells() != poly.GetNumberOfPolys() else poly
    ids = vtk_to_numpy(tri.GetCellData().GetArray(STRUCTURE_ID)).astype(np.int64)
    polys = tri.GetPolys()
    conn = vtk_to_numpy(polys.GetConnectivityArray()).astype(np.int64)
    offsets = vtk_to_numpy(polys.GetOffsetsArray()).astype(np.int64)
    if len(offsets) < 2 or np.any(np.diff(offsets) != 3):
        return [(fallback_name, poly)]
    triangles = conn.reshape(-1, 3)
    points = vtk_to_numpy(tri.GetPoints().GetData())
    normals_arr = tri.GetPointData().GetNormals()
    normals = vtk_to_numpy(normals_arr) if normals_arr is not None else None

    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    bounds = np.flatnonzero(np.diff(sorted_ids)) + 1
    groups = np.split(order, bounds)

    out: List[Named] = []
    for group in groups:
        if not len(group):
            continue
        sid = int(ids[group[0]])
        name = names[sid] if 0 <= sid < len(names) else f"{fallback_name} {sid}"
        faces = triangles[group]
        used, inverse = np.unique(faces.ravel(), return_inverse=True)
        piece = vtk.vtkPolyData()
        pts = vtk.vtkPoints()
        pts.SetData(numpy_to_vtk(np.ascontiguousarray(points[used]), deep=True))
        piece.SetPoints(pts)
        cells = np.hstack([np.full((len(faces), 1), 3, dtype=np.int64),
                           inverse.reshape(-1, 3)]).ravel()
        ca = vtk.vtkCellArray()
        ca.ImportLegacyFormat(numpy_to_vtkIdTypeArray(np.ascontiguousarray(cells), deep=True))
        piece.SetPolys(ca)
        if normals is not None:
            n = numpy_to_vtk(np.ascontiguousarray(normals[used]), deep=True)
            n.SetName("Normals")
            piece.GetPointData().SetNormals(n)
        out.append((name, piece))
    return out


def read_structures(path: Path) -> Optional[List[Named]]:
    """Read a ``.vtp`` and split it. ``None`` when the file cannot be read."""
    if not VTK_AVAILABLE:
        return None
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = reader.GetOutput()
    if poly is None or not poly.GetNumberOfPoints():
        return None
    return split_structures(poly, Path(path).stem)


def display_name(name: str) -> str:
    """'right_biceps_brachii' / 'Right biceps brachii' → 'Right biceps brachii'."""
    text = (name or "").replace("_", " ").strip()
    return text[:1].upper() + text[1:] if text else text
