"""
Import real anatomical mesh data and convert it into BioHuman3D layers.

The application ships with *schematic placeholder geometry* — enough to exercise
the viewer, the tours and the export pipeline, but nowhere near the realism of a
production atlas. Photoreal anatomy comes from real scan/atlas data, and this
tool is the bridge: point it at a folder of meshes and it sorts every file into
the right body-system layer.

    # see what WOULD happen, changing nothing
    python tools\\import_anatomy.py D:\\anatomy\\BodyParts3D --dry-run

    # convert into the app's model folder (one .vtp per body system)
    python tools\\import_anatomy.py D:\\anatomy\\BodyParts3D

    # verify the pipeline end-to-end without downloading anything
    python tools\\import_anatomy.py --self-test

Classification
--------------
Each mesh is assigned to a layer by matching its *name* (or its filename) against
ordered keyword rules, most specific first, so "aortic valve" resolves to
``heart`` rather than ``arteries``. Datasets that name files with opaque ids
(FMA/TA codes) need a name table — pass ``--map``, or drop a two-column
``*.tsv``/``*.csv`` (id, English name) in the source folder and it is picked up
automatically. BodyParts3D ships exactly such a table.

Attribution
-----------
If you import BodyParts3D you must credit it:
    "BodyParts3D, © The Database Center for Life Science licensed under
     CC Attribution 4.0 International"
License: https://dbarchive.biosciencedbc.jp/en/bodyparts3d/lic.html
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    import vtk
except Exception as exc:  # pragma: no cover
    print(f"[error] VTK is required: {exc}")
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# Classification rules
# ---------------------------------------------------------------------------
#: (layer_id, keywords) — evaluated in order, first match wins. More specific
#: structures are listed before the systems that contain them.
CLASSIFICATION_RULES: Sequence[Tuple[str, Tuple[str, ...]]] = (
    # -- organs with their own layer -------------------------------------
    ("brain", ("cerebr", "cerebell", "brain", "diencephal", "thalamus",
               "hypothalamus", "brainstem", "medulla oblongata", "pons",
               "hippocamp", "meninges", "ventricle of brain")),
    ("heart", ("heart", "cardiac", "myocard", "atrium", "auricle of heart",
               "ventricle of heart", "mitral", "tricuspid", "aortic valve",
               "pulmonary valve", "coronary", "pericard")),
    ("lungs", ("lung", "pulmonary", "bronch", "alveol", "pleura",
               "trachea", "larynx", "pharynx", "diaphragm")),
    ("liver", ("liver", "hepatic", "bile", "gallbladder", "portal vein")),
    ("kidneys", ("kidney", "renal", "nephron", "ureter", "bladder",
                 "urethra", "adrenal")),
    ("digestive", ("stomach", "gastric", "intestin", "colon", "caecum",
                   "cecum", "duodenum", "jejunum", "ileum", "rectum", "anus",
                   "oesophagus", "esophagus", "pancrea", "spleen", "mesenter",
                   "omentum", "peritone")),
    # -- connective / muscular -------------------------------------------
    ("cartilage", ("cartilage", "meniscus", "labrum", "intervertebral disc",
                   "disc of", "synchondro", "epiphyseal")),
    ("tendons", ("tendon", "ligament", "aponeuro", "retinaculum", "bursa",
                 "fascia lata", "patellar")),
    ("muscles", ("muscle", "muscul", "biceps", "triceps", "deltoid",
                 "pectoral", "latissimus", "trapezius", "quadriceps",
                 "gastrocnemius", "gluteus", "diaphragm muscle", "sartorius",
                 "soleus", "masseter", "temporalis")),
    # -- vascular --------------------------------------------------------
    ("arteries", ("artery", "arteri", "aorta", "aortic", "carotid",
                  "subclavian", "brachial", "radial artery", "femoral artery",
                  "popliteal", "coeliac", "celiac", "mesenteric artery",
                  "coronary artery", "circle of willis")),
    ("veins", ("vein", "vena", "venous", "saphenous", "jugular",
               "sinus of dura", "sinus venosus")),
    # -- neural / lymphatic ----------------------------------------------
    ("nerves", ("nerve", "neural", "plexus", "ganglion", "spinal cord",
                "cauda equina", "ganglia", "sympathetic trunk", "vagus",
                "sciatic", "median nerve", "ulnar nerve", "femoral nerve",
                "optic", "trigeminal")),
    ("lymphatics", ("lymph", "thymus", "tonsil", "spleen lymphoid")),
    # -- surface ---------------------------------------------------------
    ("fascia", ("fascia", "subcutis", "hypoderm", "superficial fascia",
                "adipose", "fat pad")),
    ("skin", ("skin", "epidermis", "dermis", "cutis", "integument")),
    # -- skeletal (last: many bone names are unremarkable words) ----------
    ("skeleton", ("bone", "skeleton", "skull", "cranium", "mandible",
                  "maxilla", "zygomatic", "vertebra", "vertebral", "sacrum",
                  "coccyx", "rib", "sternum", "clavicle", "scapula",
                  "humerus", "radius", "ulna", "carpal", "metacarpal",
                  "phalan", "pelvis", "ilium", "ischium", "pubis", "femur",
                  "patella", "tibia", "fibula", "tarsal", "metatarsal",
                  "calcaneus", "hip bone", "spine")),
)

MESH_EXTENSIONS = (".obj", ".stl", ".ply", ".vtp", ".vtk", ".glb", ".gltf")


def classify(identifier: str, name: str = "") -> Optional[str]:
    """Layer id for a mesh, or ``None`` when nothing matches."""
    haystack = f"{identifier} {name}".lower().replace("_", " ").replace("-", " ")
    for layer_id, keywords in CLASSIFICATION_RULES:
        for keyword in keywords:
            if keyword in haystack:
                return layer_id
    return None


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------
def find_meshes(source: Path) -> List[Path]:
    found: List[Path] = []
    for extension in MESH_EXTENSIONS:
        found.extend(source.rglob(f"*{extension}"))
    return sorted(set(found))


def load_name_map(source: Path, explicit: Optional[Path] = None) -> Dict[str, str]:
    """Build ``id -> English name`` from a TSV/CSV, if one is supplied or found.

    BodyParts3D and similar atlases ship a table mapping opaque ids (FMA/TA codes)
    to names; without it every file is unclassifiable.
    """
    candidates: List[Path] = []
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"mapping file not found: {explicit}")
        candidates.append(explicit)
    else:
        for pattern in ("*.tsv", "*.csv", "*.txt"):
            candidates.extend(sorted(source.rglob(pattern))[:40])

    mapping: Dict[str, str] = {}
    for candidate in candidates:
        try:
            with candidate.open("r", encoding="utf-8", errors="replace") as handle:
                sample = handle.read(4096)
                handle.seek(0)
                delimiter = "\t" if sample.count("\t") > sample.count(",") else ","
                rows = list(csv.reader(handle, delimiter=delimiter))
        except Exception:
            continue
        for row in rows:
            if len(row) < 2:
                continue
            key, value = row[0].strip(), row[1].strip()
            if not key or not value or key.lower() in ("id", "fma_id", "name"):
                continue
            # Prefer a later English-looking column when present.
            for cell in row[1:]:
                if cell.strip() and cell.strip().isascii():
                    value = cell.strip()
            mapping.setdefault(key, value)
    return mapping


def name_for(path: Path, mapping: Dict[str, str]) -> str:
    """English name for a mesh: name table first, then the filename."""
    stem = path.stem
    if stem in mapping:
        return mapping[stem]
    # Some exports append a suffix to the id (FMA5018_L.obj).
    for separator in ("_", "-", "."):
        head = stem.split(separator)[0]
        if head in mapping:
            return mapping[head]
    return stem


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------
def read_polydata(path: Path):
    """Read a mesh as vtkPolyData (GLTF returns a merged multi-block)."""
    extension = path.suffix.lower()
    try:
        if extension == ".obj":
            reader = vtk.vtkOBJReader()
        elif extension == ".stl":
            reader = vtk.vtkSTLReader()
        elif extension == ".ply":
            reader = vtk.vtkPLYReader()
        elif extension == ".vtp":
            reader = vtk.vtkXMLPolyDataReader()
        elif extension == ".vtk":
            reader = vtk.vtkPolyDataReader()
        elif extension in (".glb", ".gltf"):
            gltf = vtk.vtkGLTFReader()
            gltf.SetFileName(str(path))
            gltf.Update()
            gltf.Update()                     # required by vtkGLTFReader
            data = gltf.GetOutput()
            append = vtk.vtkAppendPolyData()
            for index in range(data.GetNumberOfBlocks()):
                block = data.GetBlock(index)
                if block is None:
                    continue
                if block.IsA("vtkMultiBlockDataSet"):
                    for sub in range(block.GetNumberOfBlocks()):
                        leaf = block.GetBlock(sub)
                        if leaf is not None and leaf.GetNumberOfPoints():
                            append.AddInputData(leaf)
                elif block.GetNumberOfPoints():
                    append.AddInputData(block)
            append.Update()
            return append.GetOutput()
        else:
            return None

        reader.SetFileName(str(path))
        reader.Update()
        return reader.GetOutput()
    except Exception:
        return None


def merge(polydata_list: Sequence) -> object:
    """Merge meshes into one polydata with consistent normals."""
    append = vtk.vtkAppendPolyData()
    for polydata in polydata_list:
        if polydata is not None and polydata.GetNumberOfPoints():
            append.AddInputData(polydata)
    append.Update()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(append.GetOutput())
    normals.SetFeatureAngle(60.0)
    normals.ConsistencyOn()
    normals.SplittingOff()
    normals.Update()
    return normals.GetOutput()


def write_vtp(polydata, destination: Path) -> bool:
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(destination))
    writer.SetInputData(polydata)
    writer.SetDataModeToBinary()
    writer.SetCompressorTypeToZLib()
    return bool(writer.Write())


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_import(source: Path, out_dir: Path, mapping_file: Optional[Path],
               dry_run: bool, merge_layers: bool) -> int:
    meshes = find_meshes(source)
    if not meshes:
        print(f"[error] no mesh files found under {source}")
        print(f"        looked for: {', '.join(MESH_EXTENSIONS)}")
        return 1

    mapping = load_name_map(source, mapping_file)
    if mapping:
        print(f"[info]  name table: {len(mapping)} id -> name entries")

    buckets: Dict[str, List[Path]] = defaultdict(list)
    unmapped: List[Path] = []
    for mesh in meshes:
        label = name_for(mesh, mapping)
        layer = classify(mesh.stem, label)
        if layer is None:
            unmapped.append(mesh)
        else:
            buckets[layer].append(mesh)

    print()
    print(f"{'layer':<14} {'meshes':>7}  examples")
    print("-" * 74)
    for layer in sorted(buckets):
        examples = ", ".join(m.name for m in buckets[layer][:3])
        print(f"{layer:<14} {len(buckets[layer]):>7}  {examples[:52]}")
    if unmapped:
        print(f"{'UNMAPPED':<14} {len(unmapped):>7}  "
              f"{', '.join(m.name for m in unmapped[:3])[:52]}")
    print("-" * 74)

    if dry_run:
        print("\n[dry-run] nothing written. Re-run without --dry-run to convert.")
        if unmapped:
            print("[hint]    add rules in CLASSIFICATION_RULES, or pass --map "
                  "<id,name.csv> for opaque filenames.")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for layer, files in sorted(buckets.items()):
        target = out_dir / f"{layer}.vtp"
        if not merge_layers:
            # Keep originals alongside; the app loads <layer>.vtp only.
            for index, mesh in enumerate(files):
                shutil.copy2(mesh, out_dir / f"{layer}_{index:03d}{mesh.suffix}")
            continue
        if target.exists():
            target.unlink()
        polydata = merge([read_polydata(f) for f in files])
        if polydata is None or not polydata.GetNumberOfPoints():
            print(f"[warn]  {layer}: nothing readable, skipped")
            continue
        if write_vtp(polydata, target):
            written += 1
            print(f"[ok]    {target.name:<16} "
                  f"{polydata.GetNumberOfPoints():>9,} points  "
                  f"{polydata.GetNumberOfPolys():>9,} polys")

    print(f"\nDone - {written} layer file(s) written to {out_dir}")
    print("Restart BioHuman3D to load them.")
    print("\nIf you imported BodyParts3D, retain this attribution in your product:")
    print('  "BodyParts3D, © The Database Center for Life Science licensed '
          'under CC Attribution 4.0 International"')
    return 0


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def self_test() -> int:
    """Prove the pipeline without downloading a dataset.

    Builds a small mesh set with anatomically meaningful filenames, imports it,
    and checks that each file landed in the intended layer.
    """
    print("Self-test: generating a synthetic labelled dataset…\n")
    workdir = Path(tempfile.mkdtemp(prefix="biohuman_import_"))
    source = workdir / "source"
    source.mkdir(parents=True)

    samples = {
        "left_femur.obj": "skeleton",
        "skull.obj": "skeleton",
        "heart.obj": "heart",
        "aorta.obj": "arteries",
        "great_saphenous_vein.obj": "veins",
        "cerebrum.obj": "brain",
        "sciatic_nerve.obj": "nerves",
        "stomach.obj": "digestive",
        "right_kidney.obj": "kidneys",
        "deltoid_muscle.obj": "muscles",
        "patellar_tendon.obj": "tendons",
        "costal_cartilage.obj": "cartilage",
        "dermis.obj": "skin",
        "thoracic_lymph_node.obj": "lymphatics",
        "mystery_object.obj": "UNMAPPED",
    }
    for index, filename in enumerate(samples):
        sphere = vtk.vtkSphereSource()
        sphere.SetThetaResolution(16)
        sphere.SetPhiResolution(16)
        sphere.SetCenter(index * 0.5, 0.0, 0.0)
        sphere.Update()
        writer = vtk.vtkOBJWriter()
        writer.SetFileName(str(source / filename))
        writer.SetInputConnection(sphere.GetOutputPort())
        writer.Write()

    # A name table with opaque ids, as BodyParts3D ships.
    (source / "isa_BP3D.txt").write_text(
        "FMA5018\tleft femur\nFMA7088\theart\nFMA7714\tcerebrum\n",
        encoding="utf-8")
    for fid in ("FMA5018", "FMA7088", "FMA7714"):
        shutil.copy2(source / "heart.obj", source / f"{fid}.obj")

    out = workdir / "out"
    code = run_import(source, out, None, dry_run=False, merge_layers=True)
    if code != 0:
        print("[FAIL] importer returned an error")
        return 1

    produced = {p.stem for p in out.glob("*.vtp")}
    print(f"\nProduced: {sorted(produced)}")

    failures = []
    for filename, expected in samples.items():
        if expected == "UNMAPPED":
            continue
        if expected not in produced:
            failures.append(f"{filename} -> expected layer '{expected}' missing")

    if failures:
        print("\n[FAIL] classification errors:")
        for item in failures:
            print(f"   - {item}")
        return 1

    print("\n[PASS] every labelled mesh reached its intended layer")
    shutil.rmtree(workdir, ignore_errors=True)
    return 0


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import real anatomical meshes into BioHuman3D layers.")
    parser.add_argument("source", nargs="?", type=Path,
                        help="folder containing .obj/.stl/.glb/.ply meshes")
    parser.add_argument("--out", type=Path, default=ROOT / "app" / "assets" / "models",
                        help="destination folder (default: app/assets/models)")
    parser.add_argument("--map", dest="mapping", type=Path, default=None,
                        help="TSV/CSV mapping opaque ids to English names")
    parser.add_argument("--dry-run", action="store_true",
                        help="report the mapping without writing anything")
    parser.add_argument("--no-merge", dest="merge_layers", action="store_false",
                        help="copy meshes individually instead of merging per layer")
    parser.add_argument("--self-test", action="store_true",
                        help="verify the pipeline with a synthetic dataset")
    args = parser.parse_args()

    if args.self_test:
        return self_test()
    if args.source is None:
        parser.print_help()
        return 2
    if not args.source.exists():
        print(f"[error] source folder not found: {args.source}")
        return 1
    return run_import(args.source, args.out, args.mapping, args.dry_run,
                      args.merge_layers)


if __name__ == "__main__":
    raise SystemExit(main())
