"""
Generate the built-in reference anatomy so BioHuman3D renders on first run.

Writes one structure-labelled ``.vtp`` per registry layer under
``app/assets/models``. The body is the anthropometric procedural model in
``app/anatomy/procedural_body.py``: correct proportions and topography, several
hundred individually named structures (bones, muscles, vessels, nerves, organs),
but schematic surfaces.

For scan-derived, atlas-quality meshes run ``tools/fetch_anatomy.py`` — it
downloads BodyParts3D and replaces these files with the same layer names, so
nothing else in the app needs to change.

Usage:
    python tools/generate_demo_models.py            # skip existing files
    python tools/generate_demo_models.py --force    # overwrite everything
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    import vtk  # noqa: F401
except Exception as exc:  # pragma: no cover
    print(f"[error] VTK is required to generate demo models: {exc}")
    raise SystemExit(1)

from app.anatomy.procedural_body import BUILDERS  # noqa: E402
from app.anatomy.structures import write_structures  # noqa: E402
from app.config import Paths  # noqa: E402
from app.core.model_registry import ANATOMY_LAYERS  # noqa: E402

SOURCE_TAG = "procedural-reference-body"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the reference anatomy models.")
    parser.add_argument("--force", action="store_true", help="overwrite existing files")
    parser.add_argument("--out", type=Path, default=None,
                        help="output directory (defaults to app/assets/models)")
    args = parser.parse_args()

    paths = Paths().ensure()
    out_dir = args.out or paths.models
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Writing reference anatomy to: {out_dir}")
    written = skipped = failed = structures = 0
    started = time.perf_counter()

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
            items = builder()
            triangles = write_structures(items, target, source=SOURCE_TAG)
            structures += len(items)
            print(f"  [ok]   {target.name:<18} {len(items):>4} structures {triangles:>9,} triangles")
            written += 1
        except Exception as exc:
            print(f"  [FAIL] {spec.id}: {type(exc).__name__}: {exc}")
            failed += 1

    print(f"\nDone in {time.perf_counter() - started:.1f}s - {written} written "
          f"({structures} named structures), {skipped} skipped, {failed} failed.")
    if written:
        print("Launch the app with:  python main.py")
        print("For scan-derived anatomy:  python tools\\fetch_anatomy.py")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
