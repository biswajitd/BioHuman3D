"""
Generate placeholder anatomy so BioHuman3D renders something on first run.

Produces one ``.vtp`` per registry layer under ``app/assets/models``, built from
procedural VTK primitives and arranged in a rough anatomical stack. These are
deliberately schematic — they exist to exercise picking, isolation, opacity,
clipping and the tour camera, not to teach anatomy.

Replace them with real data (Z-Anatomy, BodyParts3D, or your own scans) keeping
the same filenames, and nothing else in the app needs to change.

Usage:
    python tools/generate_demo_models.py            # skip existing files
    python tools/generate_demo_models.py --force    # overwrite everything
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    import vtk
except Exception as exc:  # pragma: no cover
    print(f"[error] VTK is required to generate demo models: {exc}")
    raise SystemExit(1)


def _transform(polydata, transform):
    """Apply a ``vtkTransform`` to polydata.

    ``vtkTransformPolyDataFilter`` was deprecated in VTK 9.7 in favour of
    ``vtkTransformFilter``; prefer the new class and fall back for older wheels.
    """
    factory = getattr(vtk, "vtkTransformFilter", None) or vtk.vtkTransformPolyDataFilter
    filter_ = factory()
    filter_.SetInputData(polydata)
    filter_.SetTransform(transform)
    filter_.Update()
    return filter_.GetOutput()

from app.config import Paths                       # noqa: E402
from app.core.model_registry import ANATOMY_LAYERS  # noqa: E402


# ---------------------------------------------------------------------------
# Primitive helpers
#
# Resolution matters more than it looks: these defaults were originally 40-48,
# which produced visibly faceted silhouettes on close-ups. Doubling them costs
# a few hundred KB per layer and is the difference between "blocky test shapes"
# and something that reads as a smooth surface under supersampled export.
# ---------------------------------------------------------------------------
def _sphere(radius: float = 1.0, center=(0, 0, 0), scale=(1, 1, 1), resolution: int = 96):
    source = vtk.vtkSphereSource()
    source.SetRadius(radius)
    source.SetThetaResolution(resolution)
    source.SetPhiResolution(resolution)
    source.SetCenter(*center)
    source.Update()

    transform = vtk.vtkTransform()
    transform.Scale(*scale)
    return _transform(source.GetOutput(), transform)


def _cylinder(radius: float, height: float, center=(0, 0, 0), axis: str = "z",
              resolution: int = 64):
    source = vtk.vtkCylinderSource()
    source.SetRadius(radius)
    source.SetHeight(height)
    source.SetResolution(resolution)
    source.SetCenter(0, 0, 0)
    source.Update()

    transform = vtk.vtkTransform()
    if axis == "x":
        transform.RotateY(90)
    elif axis == "y":
        transform.RotateX(90)
    transform.Translate(*center)
    return _transform(source.GetOutput(), transform)


def _torus(major: float, minor: float, center=(0, 0, 0), resolution: int = 80):
    source = vtk.vtkParametricTorus()
    source.SetRingRadius(major)
    source.SetCrossSectionRadius(minor)

    function = vtk.vtkParametricFunctionSource()
    function.SetParametricFunction(source)
    function.SetUResolution(resolution)
    function.SetVResolution(resolution // 2)
    function.Update()

    transform = vtk.vtkTransform()
    transform.Translate(*center)
    return _transform(function.GetOutput(), transform)


def _append(polydata_list):
    """Merge several polydata objects into one."""
    append = vtk.vtkAppendPolyData()
    for poly in polydata_list:
        if poly is not None:
            append.AddInputData(poly)
    append.Update()

    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(append.GetOutputPort())
    clean.Update()
    return clean.GetOutput()


def _mirror(polydata, offset_x: float):
    """Return a translated copy (used for paired organs and limbs)."""
    transform = vtk.vtkTransform()
    transform.Translate(offset_x, 0, 0)
    return _transform(polydata, transform)


# ---------------------------------------------------------------------------
# Per-layer builders (units are arbitrary "body units", torso ≈ 2.0 tall)
# ---------------------------------------------------------------------------
def build_skin():
    """Ellipsoid torso + head + simple limbs."""
    torso = _sphere(1.0, center=(0, 0, 0), scale=(0.36, 0.20, 0.62))
    head = _sphere(0.15, center=(0, 0, 0.82), scale=(1.0, 1.05, 1.2))
    neck = _cylinder(0.055, 0.12, center=(0, 0, 0.70))
    arm_left = _cylinder(0.045, 0.62, center=(-0.40, 0, 0.16), axis="z")
    arm_right = _cylinder(0.045, 0.62, center=(0.40, 0, 0.16), axis="z")
    leg_left = _cylinder(0.055, 0.70, center=(-0.13, 0, -0.92), axis="z")
    leg_right = _cylinder(0.055, 0.70, center=(0.13, 0, -0.92), axis="z")
    return _append([torso, head, neck, arm_left, arm_right, leg_left, leg_right])


def build_fascia():
    return _sphere(1.0, center=(0, 0, 0), scale=(0.33, 0.18, 0.60))


def build_muscles():
    """Pectorals, abdominals, deltoids, quadriceps and gluteals."""
    parts = [
        _sphere(0.13, center=(-0.14, -0.10, 0.30), scale=(1.0, 0.55, 0.8)),
        _sphere(0.13, center=(0.14, -0.10, 0.30), scale=(1.0, 0.55, 0.8)),
        _sphere(0.10, center=(0, -0.12, 0.02), scale=(1.5, 0.5, 1.6)),
        _sphere(0.10, center=(0, 0.13, 0.22), scale=(1.6, 0.5, 2.0)),
        _cylinder(0.075, 0.30, center=(-0.36, 0, 0.44), axis="z"),
        _cylinder(0.075, 0.30, center=(0.36, 0, 0.44), axis="z"),
        _cylinder(0.085, 0.34, center=(-0.13, 0, -0.80), axis="z"),
        _cylinder(0.085, 0.34, center=(0.13, 0, -0.80), axis="z"),
        _sphere(0.11, center=(-0.13, 0.08, -0.55), scale=(1.0, 0.9, 0.8)),
        _sphere(0.11, center=(0.13, 0.08, -0.55), scale=(1.0, 0.9, 0.8)),
    ]
    return _append(parts)


def build_skeleton():
    """Skull, ribcage, spine, pelvis and limb bones."""
    skull = _sphere(0.135, center=(0, 0, 0.84), scale=(1.0, 1.1, 1.15))
    parts = [skull]

    # Spine: 24 vertebrae
    for index in range(24):
        z = 0.66 - index * 0.052
        radius = 0.030 + 0.010 * math.sin(index / 4.0)
        parts.append(_sphere(radius, center=(0, 0.045, z), scale=(1.3, 1.0, 0.75)))

    # Ribcage: 10 pairs of arcs approximated with torus segments
    for level in range(10):
        z = 0.52 - level * 0.052
        width = 0.26 - abs(level - 5) * 0.012
        parts.append(_torus(width, 0.012, center=(0, 0.01, z)))

    # Pelvis
    parts.append(_torus(0.20, 0.035, center=(0, 0.01, -0.28)))

    # Sternum, clavicles, limb bones
    parts.append(_cylinder(0.020, 0.30, center=(0, -0.16, 0.36)))
    parts.append(_cylinder(0.016, 0.30, center=(-0.15, -0.10, 0.55), axis="x"))
    parts.append(_cylinder(0.016, 0.30, center=(0.15, -0.10, 0.55), axis="x"))
    for side in (-1, 1):
        parts.append(_cylinder(0.018, 0.50, center=(side * 0.38, 0, 0.20)))
        parts.append(_cylinder(0.016, 0.45, center=(side * 0.40, 0, -0.22)))
        parts.append(_cylinder(0.024, 0.80, center=(side * 0.13, 0, -0.95)))
    return _append(parts)


def build_cartilage():
    """Joint linings at shoulder, hip and knee."""
    parts = [
        _sphere(0.045, center=(-0.36, 0, 0.58)),
        _sphere(0.045, center=(0.36, 0, 0.58)),
        _sphere(0.055, center=(-0.13, 0, -0.55)),
        _sphere(0.055, center=(0.13, 0, -0.55)),
        _sphere(0.045, center=(-0.13, 0, -1.30)),
        _sphere(0.045, center=(0.13, 0, -1.30)),
    ]
    return _append(parts)


def build_tendons():
    parts = [
        _cylinder(0.020, 0.16, center=(-0.37, 0, 0.44)),
        _cylinder(0.020, 0.16, center=(0.37, 0, 0.44)),
        _cylinder(0.026, 0.20, center=(-0.13, 0, -0.60)),
        _cylinder(0.026, 0.20, center=(0.13, 0, -0.60)),
        _cylinder(0.022, 0.18, center=(-0.13, 0.04, -1.22)),
        _cylinder(0.022, 0.18, center=(0.13, 0.04, -1.22)),
    ]
    return _append(parts)


def build_arteries():
    """Aorta with the great vessels and the iliac bifurcation."""
    parts = [_cylinder(0.024, 0.80, center=(0, 0.02, 0.22), axis="z")]
    parts.append(_cylinder(0.018, 0.22, center=(-0.10, 0.02, 0.50), axis="x"))
    parts.append(_cylinder(0.018, 0.22, center=(0.10, 0.02, 0.50), axis="x"))
    for side in (-1, 1):
        parts.append(_cylinder(0.014, 0.55, center=(side * 0.10, 0.02, -0.55)))
        parts.append(_cylinder(0.010, 0.50, center=(side * 0.34, 0.02, 0.10)))
    return _append(parts)


def build_veins():
    parts = [
        _cylinder(0.028, 0.78, center=(0.05, 0.04, 0.22), axis="z"),
        _cylinder(0.016, 0.50, center=(-0.34, 0.04, 0.10)),
        _cylinder(0.016, 0.50, center=(0.34, 0.04, 0.10)),
        _cylinder(0.014, 0.50, center=(-0.12, 0.04, -0.60)),
        _cylinder(0.014, 0.50, center=(0.12, 0.04, -0.60)),
    ]
    return _append(parts)


def build_nerves():
    """Spinal cord with the brachial and sciatic plexuses."""
    parts = [_cylinder(0.014, 0.92, center=(0, 0.05, 0.20), axis="z")]
    for side in (-1, 1):
        parts.append(_cylinder(0.009, 0.60, center=(side * 0.20, 0.05, 0.30)))
        parts.append(_cylinder(0.009, 0.70, center=(side * 0.15, 0.05, -0.70)))
    return _append(parts)


def build_lymphatics():
    parts = [
        _cylinder(0.008, 0.60, center=(0.09, 0.06, 0.10), axis="z"),
        _sphere(0.028, center=(0.10, 0.06, 0.30)),
        _sphere(0.024, center=(-0.10, 0.06, -0.10)),
        _sphere(0.022, center=(0.06, 0.06, -0.45)),
    ]
    return _append(parts)


def build_brain():
    cerebrum = _sphere(0.115, center=(0, 0, 0.86), scale=(1.0, 1.06, 1.0))
    cerebellum = _sphere(0.055, center=(0, 0.07, 0.78), scale=(1.2, 0.9, 0.7))
    stem = _cylinder(0.022, 0.14, center=(0, 0.01, 0.73))
    return _append([cerebrum, cerebellum, stem])


def build_lungs():
    left = _sphere(0.115, center=(-0.12, 0, 0.28), scale=(0.85, 0.85, 1.5))
    right = _sphere(0.115, center=(0.12, 0, 0.28), scale=(0.85, 0.85, 1.5))
    trachea = _cylinder(0.016, 0.20, center=(0, 0, 0.56))
    return _append([left, right, trachea])


def build_heart():
    body = _sphere(0.11, center=(0.02, -0.02, 0.34), scale=(1.0, 0.9, 1.25))
    atria = _sphere(0.06, center=(0.02, 0.06, 0.42))
    vessels = _cylinder(0.020, 0.16, center=(0.02, 0.02, 0.50))
    return _append([body, atria, vessels])


def build_liver():
    right = _sphere(0.15, center=(0.06, -0.05, 0.12), scale=(1.4, 0.7, 0.9))
    left = _sphere(0.09, center=(-0.12, -0.04, 0.14), scale=(1.0, 0.7, 0.8))
    return _append([right, left])


def build_kidneys():
    left = _sphere(0.060, center=(-0.13, 0.10, -0.06), scale=(0.8, 0.7, 1.4))
    right = _sphere(0.060, center=(0.13, 0.10, -0.06), scale=(0.8, 0.7, 1.4))
    return _append([left, right])


def build_digestive():
    stomach = _sphere(0.085, center=(-0.06, -0.05, 0.06), scale=(1.1, 0.8, 0.9))
    small = _torus(0.09, 0.020, center=(0.02, -0.05, -0.10))
    colon = _torus(0.13, 0.022, center=(0, -0.04, -0.20))
    oesophagus = _cylinder(0.014, 0.26, center=(-0.02, -0.02, 0.30))
    return _append([stomach, small, colon, oesophagus])


BUILDERS = {
    "skin": build_skin,
    "fascia": build_fascia,
    "muscles": build_muscles,
    "skeleton": build_skeleton,
    "cartilage": build_cartilage,
    "tendons": build_tendons,
    "arteries": build_arteries,
    "veins": build_veins,
    "nerves": build_nerves,
    "lymphatics": build_lymphatics,
    "brain": build_brain,
    "lungs": build_lungs,
    "heart": build_heart,
    "liver": build_liver,
    "kidneys": build_kidneys,
    "digestive": build_digestive,
}


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------
def write_polydata(polydata, path: Path) -> int:
    """Write VTP (compressed XML polydata). Returns the triangle count."""
    triangulate = vtk.vtkTriangleFilter()
    triangulate.SetInputData(polydata)
    triangulate.Update()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(triangulate.GetOutputPort())
    normals.SetFeatureAngle(60.0)
    normals.SplittingOff()
    normals.ConsistencyOn()
    normals.Update()

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputConnection(normals.GetOutputPort())
    writer.SetCompressorTypeToZLib()
    writer.SetDataModeToBinary()
    writer.Write()
    return normals.GetOutput().GetNumberOfCells()


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate placeholder anatomy models.")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing files")
    parser.add_argument("--out", type=Path, default=None,
                        help="output directory (defaults to app/assets/models)")
    args = parser.parse_args()

    paths = Paths().ensure()
    out_dir = args.out or paths.models
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Writing placeholder anatomy to: {out_dir}")
    written = skipped = failed = 0

    for spec in ANATOMY_LAYERS:
        builder = BUILDERS.get(spec.id)
        target = out_dir / f"{spec.id}.vtp"

        if builder is None:
            print(f"  [skip] {spec.id:<12} no builder defined")
            skipped += 1
            continue
        if target.exists() and not args.force:
            print(f"  [have] {target.name}")
            skipped += 1
            continue

        try:
            polydata = builder()
            triangles = write_polydata(polydata, target)
            print(f"  [ok]   {target.name:<22} {triangles:>8,} triangles")
            written += 1
        except Exception as exc:
            print(f"  [FAIL] {spec.id}: {type(exc).__name__}: {exc}")
            failed += 1

    print(f"\nDone - {written} written, {skipped} skipped, {failed} failed.")
    if written:
        print("Launch the app with:  python main.py")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
