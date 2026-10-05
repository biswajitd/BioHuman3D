"""
Normalise an imported atlas into the BioHuman3D anatomical frame.

Every atlas ships in its own convention — BodyParts3D is in millimetres with an
arbitrary origin, Blender exports are often Y-up, some datasets face +Y. The
viewer, the camera presets, the muscle kinematics and the MRI simulator all
assume ONE frame:

    metres · Z up · floor at z = 0 · anterior = -Y · patient's right = -X
    · mid-sagittal plane at x = 0

This module infers the dataset's frame from the anatomy itself (it does not
trust file metadata): the long axis of the body is "up", the skull is at the
top, the sternum is anterior to the vertebral column, and structures named
"left …" lie on +X. It returns a 4×4 matrix that is applied to every mesh.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from vtkmodules.util.numpy_support import vtk_to_numpy
except Exception:  # pragma: no cover
    from vtk.util.numpy_support import vtk_to_numpy  # type: ignore

from app.core.vtk_utils import vtk


@dataclass
class FrameReport:
    matrix: np.ndarray
    scale: float
    notes: List[str] = field(default_factory=list)


def _centroid(poly) -> Optional[np.ndarray]:
    if poly is None or not poly.GetNumberOfPoints():
        return None
    return vtk_to_numpy(poly.GetPoints().GetData()).mean(axis=0)


def _find(named: Dict[str, object], *keywords: str) -> List[np.ndarray]:
    out = []
    for name, poly in named.items():
        low = name.lower()
        if all(k in low for k in keywords):
            c = _centroid(poly)
            if c is not None:
                out.append(c)
    return out


def infer_frame(named: Dict[str, object]) -> FrameReport:
    """Infer the transform taking *named* meshes into the standard frame."""
    pts = [vtk_to_numpy(p.GetPoints().GetData()) for p in named.values()
           if p is not None and p.GetNumberOfPoints()]
    if not pts:
        return FrameReport(np.eye(4), 1.0, ["no geometry"])
    allp = np.vstack(pts)
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    extent = hi - lo
    notes: List[str] = []

    # 1. Units: a standing adult is 1.4–2.1 m tall.
    tallest = float(extent.max())
    scale = 1.0
    if tallest > 100.0:
        scale = 0.001
        notes.append("units: millimetres → metres")
    elif tallest > 10.0:
        scale = 0.01
        notes.append("units: centimetres → metres")

    # 2. Up axis = longest extent; sign from where the skull is.
    up_axis = int(np.argmax(extent))
    rot = np.eye(3)
    head = _find(named, "skull") or _find(named, "cranium") or _find(named, "brain") \
        or _find(named, "cerebr")
    centre = (lo + hi) / 2
    up_sign = 1.0
    if head:
        up_sign = 1.0 if head[0][up_axis] >= centre[up_axis] else -1.0
    new_z = np.zeros(3)
    new_z[up_axis] = up_sign

    # 3. Anterior: sternum (or any anterior trunk structure) vs. vertebral column.
    front = (_find(named, "sternum") or _find(named, "rectus abdominis")
             or _find(named, "stern"))
    back = _find(named, "vertebra") or _find(named, "spinal cord") or _find(named, "sacrum")
    candidates = [k for k in range(3) if k != up_axis]
    if front and back:
        delta = np.mean(front, axis=0) - np.mean(back, axis=0)
        delta[up_axis] = 0.0
        ap_axis = int(candidates[int(np.argmax([abs(delta[k]) for k in candidates]))])
        anterior_sign = 1.0 if delta[ap_axis] > 0 else -1.0
    else:
        ap_axis = candidates[int(np.argmin([extent[k] for k in candidates]))]   # body is thinner AP
        anterior_sign = -1.0
        notes.append("anterior direction not identifiable by name — assumed -axis")
    new_y = np.zeros(3)
    new_y[ap_axis] = -anterior_sign          # standard: anterior = -Y
    new_x = np.cross(new_y, new_z)           # right-handed
    rot = np.vstack([new_x, new_y, new_z])   # rows: old coords → new axes

    # 4. Handedness check: "left" structures must land on +X.
    left = _find(named, "left")
    right = _find(named, "right")
    if left and right:
        lx = float(np.mean([rot @ c for c in left], axis=0)[0])
        rx = float(np.mean([rot @ c for c in right], axis=0)[0])
        if lx < rx:
            rot[0] *= -1.0
            notes.append("dataset is mirror-handed — reflected across the sagittal plane")

    # 5. Translate: floor at z = 0, centred left-right and front-back.
    rotated = (allp @ rot.T) * scale
    rlo, rhi = rotated.min(axis=0), rotated.max(axis=0)
    shift = np.array([-(rlo[0] + rhi[0]) / 2, -(rlo[1] + rhi[1]) / 2, -rlo[2]])
    spine = _find(named, "vertebra")
    if spine:            # centre on the vertebral column rather than the bbox
        sp = (np.mean(spine, axis=0) @ rot.T) * scale
        shift[0] = -sp[0]

    m = np.eye(4)
    m[:3, :3] = rot * scale
    m[:3, 3] = shift
    stature = float(rhi[2] - rlo[2])
    notes.append(f"stature after normalisation: {stature:.2f} m")
    return FrameReport(m, scale, notes)


def apply_matrix(poly, matrix: np.ndarray):
    """Return a transformed copy of *poly* (winding fixed for reflections)."""
    t = vtk.vtkTransform()
    t.SetMatrix([float(v) for v in matrix.ravel()])
    f = vtk.vtkTransformPolyDataFilter()
    f.SetInputData(poly)
    f.SetTransform(t)
    f.Update()
    out = f.GetOutput()
    if np.linalg.det(matrix[:3, :3]) < 0:
        rev = vtk.vtkReverseSense()
        rev.SetInputData(out)
        rev.Update()
        out = rev.GetOutput()
    copy = vtk.vtkPolyData()
    copy.DeepCopy(out)
    return copy
