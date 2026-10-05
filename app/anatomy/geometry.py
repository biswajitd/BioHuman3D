"""
Mesh primitives for anatomical shapes, built directly with numpy.

The old placeholder body was made of VTK spheres and cylinders, which is why it
read as "a stack of ellipsoids". Anatomy is mostly *lathed* forms — a profile
swept along a (possibly curved) axis with an elliptical cross-section: long
bones with flared epiphyses, fusiform muscle bellies, tapering vessels. These
helpers produce exactly that, watertight and with consistent winding, so the
renderer's smoothing and normals behave.

All functions return ``vtkPolyData`` in metres, Z-up, anterior = -Y,
patient's right = -X (so an anterior view shows the patient's right on the
viewer's left, as in every atlas).
"""
from __future__ import annotations

import math
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from app.core.vtk_utils import VTK_AVAILABLE, vtk

try:
    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
except Exception:  # pragma: no cover
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray  # type: ignore

Vec = Sequence[float]
Profile = Callable[[float], float]


# ---------------------------------------------------------------------------
# numpy → vtkPolyData
# ---------------------------------------------------------------------------
def polydata_from_arrays(points: np.ndarray, triangles: np.ndarray):
    """Build a triangle ``vtkPolyData`` from ``(N,3)`` points and ``(M,3)`` ids."""
    poly = vtk.vtkPolyData()
    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(np.ascontiguousarray(points, dtype=np.float64), deep=True))
    poly.SetPoints(pts)

    tris = np.ascontiguousarray(triangles, dtype=np.int64)
    cells = np.hstack([np.full((len(tris), 1), 3, dtype=np.int64), tris]).ravel()
    cell_array = vtk.vtkCellArray()
    cell_array.ImportLegacyFormat(numpy_to_vtkIdTypeArray(cells, deep=True))
    poly.SetPolys(cell_array)
    return poly


def _frame(axis: np.ndarray, hint: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Two unit vectors orthogonal to *axis*. ``hint`` fixes the first one's
    direction (e.g. anterior) so elliptical sections are oriented sensibly."""
    axis = axis / (np.linalg.norm(axis) or 1.0)
    if hint is None:
        hint = np.array([0.0, -1.0, 0.0])
        if abs(float(np.dot(hint, axis))) > 0.9:
            hint = np.array([1.0, 0.0, 0.0])
    u = hint - np.dot(hint, axis) * axis
    u /= np.linalg.norm(u) or 1.0
    v = np.cross(axis, u)
    return u, v


# ---------------------------------------------------------------------------
# Lathe: a profile swept along a polyline axis
# ---------------------------------------------------------------------------
def lathe(path: Sequence[Vec], profile: Profile, *, ratio: float = 1.0,
          rings: int = 40, segments: int = 28, hint: Optional[Vec] = None,
          twist: float = 0.0):
    """Sweep a circular/elliptical section along *path*.

    ``profile(t)`` gives the section radius at ``t ∈ [0, 1]`` along the path;
    ``ratio`` is the section's (depth / width) so a muscle can be flatter than
    it is wide. Ends are closed with pole vertices, so the mesh is watertight.
    """
    path = np.asarray(path, dtype=float)
    if len(path) == 2:
        path = np.linspace(path[0], path[1], 3)
    # Resample the polyline uniformly by arc length.
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = arc[-1] or 1.0
    ts = np.linspace(0.0, 1.0, rings)
    centres = np.stack([np.interp(ts * total, arc, path[:, k]) for k in range(3)], axis=1)
    tangents = np.gradient(centres, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-12

    hint_vec = None if hint is None else np.asarray(hint, dtype=float)
    angles = np.linspace(0.0, 2 * math.pi, segments, endpoint=False)
    points: List[np.ndarray] = []
    u_prev = None
    for i, (c, tan) in enumerate(zip(centres, tangents)):
        if u_prev is None:
            u, v = _frame(tan, hint_vec)
        else:   # parallel transport keeps the section from flipping on curves
            u = u_prev - np.dot(u_prev, tan) * tan
            u /= np.linalg.norm(u) or 1.0
            v = np.cross(tan, u)
        u_prev = u
        r = max(1e-4, float(profile(ts[i])))
        rot = twist * ts[i]
        cos_a, sin_a = np.cos(angles + rot), np.sin(angles + rot)
        ring = c + r * (np.outer(cos_a, u) + ratio * np.outer(sin_a, v))
        points.append(ring)

    pts = np.vstack(points)
    start_pole = centres[0] - tangents[0] * profile(0.0) * 0.15
    end_pole = centres[-1] + tangents[-1] * profile(1.0) * 0.15
    pts = np.vstack([pts, start_pole, end_pole])
    sp, ep = len(pts) - 2, len(pts) - 1

    tris = []
    for i in range(rings - 1):
        a0, b0 = i * segments, (i + 1) * segments
        for j in range(segments):
            j1 = (j + 1) % segments
            tris.append((a0 + j, b0 + j, b0 + j1))
            tris.append((a0 + j, b0 + j1, a0 + j1))
    last = (rings - 1) * segments
    for j in range(segments):
        j1 = (j + 1) % segments
        tris.append((sp, j1, j))
        tris.append((ep, last + j, last + j1))
    return polydata_from_arrays(pts, np.asarray(tris))


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------
def fusiform(r_mid: float, r_ends: float = 0.0, power: float = 0.75) -> Profile:
    """Spindle-shaped muscle belly tapering into tendon at both ends."""
    def f(t: float) -> float:
        return r_ends + (r_mid - r_ends) * math.sin(math.pi * t) ** power
    return f


def long_bone(r_shaft: float, r_prox: float, r_dist: float,
              prox_len: float = 0.14, dist_len: float = 0.14) -> Profile:
    """Diaphysis with flared, rounded epiphyses (humerus, femur, tibia …)."""
    return blend(
        (0.0, r_prox * 0.30), (0.025, r_prox * 0.82), (0.055, r_prox),
        (prox_len, r_shaft * 1.12), (0.5, r_shaft),
        (1.0 - dist_len, r_shaft * 1.10), (0.945, r_dist),
        (0.975, r_dist * 0.82), (1.0, r_dist * 0.30),
    )


def taper(r0: float, r1: float, round_ends: bool = True) -> Profile:
    """Linear taper (vessels, nerves, tendons) with rounded caps."""
    def f(t: float) -> float:
        r = r0 + (r1 - r0) * t
        if round_ends:
            edge = min(t, 1.0 - t)
            if edge < 0.04:
                r *= math.sqrt(max(0.0, 1.0 - ((0.04 - edge) / 0.04) ** 2))
        return max(r, 1e-4)
    return f


def ellipsoid_profile(r: float) -> Profile:
    def f(t: float) -> float:
        return max(1e-4, r * math.sqrt(max(0.0, 1.0 - (2 * t - 1) ** 2)))
    return f


def blend(*knots: Tuple[float, float]) -> Profile:
    """Piecewise-smooth profile through ``(t, radius)`` knots (cosine eased)."""
    ks = sorted(knots)

    def f(t: float) -> float:
        if t <= ks[0][0]:
            return ks[0][1]
        for (t0, r0), (t1, r1) in zip(ks, ks[1:]):
            if t <= t1:
                s = (t - t0) / ((t1 - t0) or 1.0)
                s = 0.5 - 0.5 * math.cos(math.pi * s)
                return r0 + (r1 - r0) * s
        return ks[-1][1]
    return f


# ---------------------------------------------------------------------------
# Convenience shapes
# ---------------------------------------------------------------------------
def ellipsoid(center: Vec, radii: Vec, *, axis: Vec = (0, 0, 1), rings: int = 32,
              segments: int = 32, hint: Optional[Vec] = None):
    """Ellipsoid aligned to *axis* with radii ``(along-axis, width, depth)``."""
    c = np.asarray(center, dtype=float)
    a = np.asarray(axis, dtype=float)
    a /= np.linalg.norm(a) or 1.0
    half, width, depth = radii
    return lathe([c - a * half, c + a * half], ellipsoid_profile(width),
                 ratio=depth / (width or 1.0), rings=rings, segments=segments, hint=hint)


def tube(path: Sequence[Vec], r0: float, r1: Optional[float] = None, *,
         rings: Optional[int] = None, segments: int = 16):
    """Vessel / nerve / tendon along a polyline, smoothly resampled."""
    path = smooth_path(path)
    rings = rings or max(12, min(160, int(len(path) * 2)))
    return lathe(path, taper(r0, r0 if r1 is None else r1), rings=rings,
                 segments=segments)


def smooth_path(points: Sequence[Vec], samples: int = 0) -> np.ndarray:
    """Catmull-Rom resampling so hand-placed control points read as curves."""
    p = np.asarray(points, dtype=float)
    if len(p) < 3:
        return p
    samples = samples or max(16, len(p) * 10)
    padded = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]])
    out = []
    n = len(p) - 1
    for k in range(samples):
        s = k / (samples - 1) * n
        i = min(int(s), n - 1)
        t = s - i
        p0, p1, p2, p3 = padded[i], padded[i + 1], padded[i + 2], padded[i + 3]
        t2, t3 = t * t, t * t * t
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    return np.asarray(out)


def arc_points(center: Vec, rx: float, ry: float, start_deg: float, end_deg: float,
               z0: float, z1: float, n: int = 24) -> List[Tuple[float, float, float]]:
    """Points on a horizontal elliptical arc with a linear vertical slope (ribs)."""
    out = []
    for k in range(n):
        t = k / (n - 1)
        ang = math.radians(start_deg + (end_deg - start_deg) * t)
        out.append((center[0] + rx * math.cos(ang), center[1] + ry * math.sin(ang),
                    z0 + (z1 - z0) * t))
    return out


def transform(poly, *, scale: Vec = (1, 1, 1), rotate: Iterable[Tuple[str, float]] = (),
              translate: Vec = (0, 0, 0), about: Vec = (0, 0, 0)):
    """Scale/rotate about *about*, then translate."""
    t = vtk.vtkTransform()
    t.PostMultiply()
    t.Translate(*[-c for c in about])
    t.Scale(*scale)
    for axis, deg in rotate:
        getattr(t, f"Rotate{axis.upper()}")(deg)
    t.Translate(*about)
    t.Translate(*translate)
    f = vtk.vtkTransformPolyDataFilter()
    f.SetInputData(poly)
    f.SetTransform(t)
    f.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(f.GetOutput())
    return out


def mirror_x(poly):
    """Reflect across the mid-sagittal plane, fixing winding."""
    t = vtk.vtkTransform()
    t.Scale(-1, 1, 1)
    f = vtk.vtkTransformPolyDataFilter()
    f.SetInputData(poly)
    f.SetTransform(t)
    rev = vtk.vtkReverseSense()
    rev.SetInputConnection(f.GetOutputPort())
    rev.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(rev.GetOutput())
    return out


def append(polys: Sequence):
    app = vtk.vtkAppendPolyData()
    for p in polys:
        if p is not None and p.GetNumberOfPoints():
            app.AddInputData(p)
    app.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(app.GetOutput())
    return out
