"""
Low-level VTK helpers.

Everything VTK-specific that is *not* widget behaviour lives here: the guarded
import, the reader factory, mesh cleanup, and actor construction. Keeping it in
one module means ``viewport.py`` reads as scene logic, and the whole stack can be
unit-tested with a stub when VTK is unavailable.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

# ---------------------------------------------------------------------------
# Guarded import — the rest of the app must keep working without VTK.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - environment dependent
    import vtk
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

    VTK_AVAILABLE = True
    VTK_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover
    vtk = None  # type: ignore[assignment]
    QVTKRenderWindowInteractor = object  # type: ignore[assignment,misc]
    VTK_AVAILABLE = False
    VTK_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
def hex_to_rgb(value: str) -> Tuple[float, float, float]:
    """``"#ff8800"`` → ``(1.0, 0.533, 0.0)``."""
    value = value.lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    return tuple(int(value[i:i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def deepen(color: Sequence[float], factor: float = 0.6) -> Tuple[float, float, float]:
    """Darken an RGB triple (used for outline / edge colours)."""
    return tuple(max(0.0, min(1.0, c * factor)) for c in color[:3])  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------
#: extension → zero-argument reader constructor
def _reader_table():
    if not VTK_AVAILABLE:
        return {}
    return {
        ".vtp": vtk.vtkXMLPolyDataReader,
        ".vtk": vtk.vtkPolyDataReader,
        ".obj": vtk.vtkOBJReader,
        ".stl": vtk.vtkSTLReader,
        ".ply": vtk.vtkPLYReader,
        ".glb": None,     # handled by read_gltf_nodes
        ".gltf": None,    # handled by read_gltf_nodes
        ".vtu": vtk.vtkXMLUnstructuredGridReader,
    }


def supported_extensions() -> List[str]:
    return sorted(_reader_table().keys())


def read_polydata(path: Path):
    """Read *path* into a ``vtkPolyData``.

    Returns ``(polydata, name)`` or ``(None, error_message)``.
    GLTF/GLB files return the first surface node only — use
    :func:`read_gltf_nodes` when you need per-node separation.
    """
    if not VTK_AVAILABLE:
        return None, "VTK is not installed"

    path = Path(path)
    ext = path.suffix.lower()

    if ext in (".glb", ".gltf"):
        nodes = read_gltf_nodes(path)
        if not nodes:
            return None, f"no surface nodes in {path.name}"
        return nodes[0][1], nodes[0][0]

    factory = _reader_table().get(ext)
    if factory is None:
        return None, f"unsupported format '{ext}'"

    try:
        reader = factory()
        reader.SetFileName(str(path))
        if hasattr(reader, "ReadAllScalarsOn"):
            reader.ReadAllScalarsOn()
        reader.Update()
        data = reader.GetOutput()
        if path.suffix.lower() == ".vtu":
            data = _unstructured_to_surface(data)
        return data, path.stem
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def read_gltf_nodes(path: Path) -> List[Tuple[str, object]]:
    """Read a GLTF/GLB and return ``[(node_name, polydata), ...]``.

    GLB authors usually name nodes after the anatomy (``Femur.L``, ``Heart``),
    which maps directly onto picking and per-structure metadata.
    """
    if not VTK_AVAILABLE:
        return []
    try:
        reader = vtk.vtkGLTFReader()
        reader.SetFileName(str(path))
        reader.Update()                      # vtkGLTFReader is multi-pass
        output = reader.GetOutput()
        if output is None or output.GetNumberOfBlocks() == 0:
            reader.Update()
            output = reader.GetOutput()
    except Exception:
        return []

    nodes: List[Tuple[str, object]] = []
    if output is None:
        return nodes

    output.InitializeTraversal()
    while True:
        block = output.GetCurrentBlock()
        if isinstance(block, vtk.vtkPolyData):
            name = output.GetCurrentMetaData().Get(vtk.vtkCompositeDataSet.NAME())
            nodes.append((str(name or f"node_{len(nodes)}"), block))
        if not output.GoToNextBlock():
            break
    return nodes


def _unstructured_to_surface(grid):
    """Extract the external surface of an unstructured grid."""
    surf = vtk.vtkDataSetSurfaceFilter()
    surf.SetInputData(grid)
    surf.Update()
    return surf.GetOutput()


# ---------------------------------------------------------------------------
# Mesh conditioning
# ---------------------------------------------------------------------------
def condition_polydata(poly, *, decimate_ratio: float = 0.0,
                       smooth_iterations: int = 12, compute_normals: bool = True):
    """Return a render-ready copy: triangulated, smoothed, decimated, shaded.

    Run this **once at load time**; it is the difference between 22 fps and
    60 fps when a layer holds a 900 k-triangle organ model.
    """
    if not VTK_AVAILABLE or poly is None:
        return poly

    if poly.GetNumberOfCells() == 0:
        return poly

    result = poly
    if result.GetNumberOfPolys() and (result.GetNumberOfCells() != result.GetNumberOfPolys()):
        tri = vtk.vtkTriangleFilter()
        tri.SetInputData(result)
        tri.Update()
        result = tri.GetOutput()

    if smooth_iterations and result.GetNumberOfPoints() > 500:
        smooth = vtk.vtkWindowedSincPolyDataFilter()
        smooth.SetInputData(result)
        smooth.SetNumberOfIterations(smooth_iterations)
        smooth.SetPassBand(0.06)
        smooth.BoundarySmoothingOff()
        smooth.FeatureEdgeSmoothingOff()
        smooth.NonManifoldSmoothingOn()
        smooth.NormalizeCoordinatesOn()
        smooth.Update()
        result = smooth.GetOutput()

    if decimate_ratio and 0.0 < decimate_ratio < 1.0:
        dec = vtk.vtkQuadricDecimation()
        dec.SetInputData(result)
        dec.SetTargetReduction(decimate_ratio)
        dec.Update()
        result = dec.GetOutput()

    if compute_normals:
        nrm = vtk.vtkPolyDataNormals()
        nrm.SetInputData(result)
        nrm.SetFeatureAngle(60.0)
        nrm.ConsistencyOn()
        nrm.SplittingOff()
        nrm.Update()
        result = nrm.GetOutput()

    return result


def compute_bounds_center(polydata) -> Tuple[float, float, float]:
    if not VTK_AVAILABLE or polydata is None:
        return (0.0, 0.0, 0.0)
    b = polydata.GetBounds()
    return ((b[0] + b[1]) / 2.0, (b[2] + b[3]) / 2.0, (b[4] + b[5]) / 2.0)


# ---------------------------------------------------------------------------
# Actor factories
# ---------------------------------------------------------------------------
def make_surface_actor(polydata, color: Sequence[float], opacity: float = 1.0,
                       *, roughness: float = 0.45, metallic: float = 0.0,
                       specular: float = 0.35):
    """Create a physically-plausible surface actor (PBR where available)."""
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    mapper.SetScalarVisibility(False)
    mapper.SetResolveCoincidentTopologyToPolygonOffset()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    prop = actor.GetProperty()
    prop.SetColor(*color[:3])
    prop.SetOpacity(float(opacity))
    prop.SetInterpolationToPhong()
    prop.SetSpecular(specular)
    prop.SetSpecularPower(28.0)
    prop.SetDiffuse(0.82)
    prop.SetAmbient(0.14)
    prop.SetAmbientColor(*[c * 0.45 for c in color[:3]])
    prop.SetBackfaceCulling(False)

    if hasattr(prop, "SetMetallic"):
        prop.SetMetallic(metallic)
        prop.SetRoughness(roughness)

    # NOTE: vtkProperty.ForceOpaqueOn()/ForceTranslucentOn() are legacy OpenGL-era
    # flags. In VTK 9.7 they fail-fast (access violation) and they are not needed:
    # the OpenGL2 backend classifies a property as translucent from its opacity
    # alone, and with depth peeling enabled the blend order is resolved correctly.
    return actor


def make_outline_actor(polydata, color: Sequence[float] = (0.20, 0.85, 0.95),
                       line_width: float = 2.4):
    """Neon silhouette used for hover / selection highlight."""
    outline = vtk.vtkOutlineFilter()
    outline.SetInputData(polydata)
    outline.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(outline.GetOutputPort())
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color[:3])
    actor.GetProperty().SetLineWidth(line_width)
    actor.GetProperty().SetOpacity(0.9)
    actor.GetProperty().SetRepresentationToWireframe()
    actor.PickableOff()
    return actor


def make_edge_actor(polydata, color: Sequence[float] = (0.05, 0.90, 1.0),
                    line_width: float = 1.6):
    """Feature-edge overlay — cheap, readable structure highlight."""
    edges = vtk.vtkFeatureEdges()
    edges.SetInputData(polydata)
    edges.BoundaryEdgesOn()
    edges.FeatureEdgesOn()
    edges.SetFeatureAngle(45.0)
    edges.NonManifoldEdgesOff()
    edges.ManifoldEdgesOff()
    edges.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(edges.GetOutputPort())
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color[:3])
    actor.GetProperty().SetLineWidth(line_width)
    actor.PickableOff()
    return actor


# ---------------------------------------------------------------------------
# Placeholder geometry (so the app runs before real anatomy is supplied)
# ---------------------------------------------------------------------------
def make_placeholder_polydata(layer_id: str, color_hint: int = 0):
    """Deterministic primitive used when a layer has no backing file."""
    if not VTK_AVAILABLE:
        return None
    import math

    seed = sum(ord(c) for c in layer_id) or 7
    src = vtk.vtkSuperquadricSource()
    src.SetPhiRoundness(0.6 + (seed % 5) / 10.0)
    src.SetThetaRoundness(0.6 + (seed % 3) / 10.0)
    src.SetThetaResolution(48)
    src.SetPhiResolution(48)
    src.Update()

    transform = vtk.vtkTransform()
    sx = 0.55 + ((seed % 7) / 20.0)
    sy = 0.30 + ((seed % 5) / 20.0)
    sz = 0.16 + ((seed % 4) / 20.0)
    angle = math.radians((seed % 360))
    transform.Scale(sx, sy, sz)
    transform.RotateZ(math.degrees(angle) * 0.05)

    # vtkTransformPolyDataFilter is deprecated as of VTK 9.7.
    factory = getattr(vtk, "vtkTransformFilter", None) or vtk.vtkTransformPolyDataFilter
    tf = factory()
    tf.SetInputData(src.GetOutput())
    tf.SetTransform(transform)
    tf.Update()
    return tf.GetOutput()
