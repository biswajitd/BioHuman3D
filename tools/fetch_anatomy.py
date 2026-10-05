"""
Download BodyParts3D and install it as BioHuman3D's anatomy.

BodyParts3D (Database Center for Life Science, Japan) is a scan-derived atlas
of ~1,500 named anatomical parts built from a full-body MRI of a male
volunteer, licensed CC BY 4.0 — commercial use is permitted with attribution.
It is what turns the schematic reference body into real anatomy.

    python tools\\fetch_anatomy.py                 # download + install
    python tools\\fetch_anatomy.py --zip D:\\isa_BP3D_4.0_obj_99.zip   # offline
    python tools\\fetch_anatomy.py --restore-reference   # back to the schematic body

What it does:
  1. downloads the OBJ archive and the element→name table (cached in
     %LOCALAPPDATA%\\BioHuman3D\\downloads, so a re-run costs nothing);
  2. extracts it;
  3. runs tools/import_anatomy.py, which names every mesh, sorts it into a body
     system layer, converts millimetres → metres, orients it (Z-up, anterior -Y,
     patient's right -X) and writes one structure-labelled .vtp per layer.

If the download is blocked (proxy, firewall), open the page below in a browser,
download the "isa" OBJ zip and the "isa_element_parts.txt" table, and pass them
with --zip / --table.

    https://dbarchive.biosciencedbc.jp/en/bodyparts3d/download.html

Attribution that must accompany any product or video using this data:
    "BodyParts3D, © The Database Center for Life Science licensed under
     CC Attribution 4.0 International"
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from app.config import Paths  # noqa: E402

PAGE = "https://dbarchive.biosciencedbc.jp/en/bodyparts3d/download.html"
BASES = (
    "https://dbarchive.biosciencedbc.jp/data/bodyparts3d/LATEST/",
    "https://dbarchive.biosciencedbc.jp/data/bodyparts3d/4.0/",
)
ARCHIVES = ("isa_BP3D_4.0_obj_99.zip", "isa_BP3D_4.0_obj_95.zip", "partof_BP3D_4.0_obj_99.zip")
TABLES = ("isa_element_parts.txt", "isa_parts_list_e.txt", "partof_element_parts.txt")
ATTRIBUTION = ("BodyParts3D, © The Database Center for Life Science licensed under "
               "CC Attribution 4.0 International")


def _download(url: str, target: Path) -> bool:
    import requests
    try:
        with requests.get(url, stream=True, timeout=60) as response:
            if response.status_code != 200:
                return False
            total = int(response.headers.get("content-length", 0))
            tmp = target.with_suffix(target.suffix + ".part")
            done = 0
            with tmp.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    handle.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r  {target.name}: {done / 1e6:6.1f} / {total / 1e6:.1f} MB "
                              f"({done * 100 // total:3d}%)", end="", flush=True)
            print()
            tmp.replace(target)
            return True
    except Exception as exc:
        print(f"  [warn] {url}: {exc}")
        return False


def _links_from_page() -> List[str]:
    """Fallback: read the official download page and collect file links."""
    import requests
    try:
        html = requests.get(PAGE, timeout=30).text
    except Exception:
        return []
    return [urljoin(PAGE, href) for href in re.findall(r'href="([^"]+\.(?:zip|txt))"', html)]


def fetch(cache: Path) -> tuple:
    """Return ``(archive_path, table_path)`` — downloaded or from cache."""
    cache.mkdir(parents=True, exist_ok=True)
    archive = next((cache / n for n in ARCHIVES if (cache / n).exists()), None)
    table = next((cache / n for n in TABLES if (cache / n).exists()), None)

    page_links: Optional[List[str]] = None
    for wanted, names in (("archive", ARCHIVES), ("table", TABLES)):
        if (archive if wanted == "archive" else table) is not None:
            continue
        found = None
        for name in names:
            for base in BASES:
                print(f"  trying {base}{name}")
                if _download(base + name, cache / name):
                    found = cache / name
                    break
            if found:
                break
        if found is None:
            if page_links is None:
                page_links = _links_from_page()
            for name in names:
                for link in page_links:
                    if link.endswith("/" + name) and _download(link, cache / name):
                        found = cache / name
                        break
                if found:
                    break
        if wanted == "archive":
            archive = found
        else:
            table = found
    return archive, table


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and install BodyParts3D anatomy.")
    parser.add_argument("--zip", type=Path, help="use an already-downloaded OBJ archive")
    parser.add_argument("--table", type=Path, help="use an already-downloaded element table")
    parser.add_argument("--out", type=Path, default=None, help="models folder (default: app/assets/models)")
    parser.add_argument("--dry-run", action="store_true", help="classify only, write nothing")
    parser.add_argument("--restore-reference", action="store_true",
                        help="remove imported atlas layers and rebuild the schematic reference body")
    args = parser.parse_args()

    paths = Paths().ensure()
    out = args.out or paths.models

    if args.restore_reference:
        for f in list(out.glob("*.vtp")) + [out / "ATLAS_SOURCE.txt"]:
            if f.exists():
                f.unlink()
        import subprocess
        return subprocess.call([sys.executable, str(ROOT / "tools" / "generate_demo_models.py"), "--force"])

    cache = paths.user_data / "downloads" / "bodyparts3d"
    archive, table = args.zip, args.table
    if archive is None or table is None:
        print("Downloading BodyParts3D (CC BY 4.0) …")
        got_archive, got_table = fetch(cache)
        archive = archive or got_archive
        table = table or got_table
    if archive is None or not Path(archive).exists():
        print("\n[error] could not download the BodyParts3D archive.")
        print(f"        Download it manually from:\n          {PAGE}")
        print("        then run:  python tools\\fetch_anatomy.py --zip <path-to-zip> --table <element table>")
        return 1

    extract = cache / Path(archive).stem
    if not extract.exists():
        print(f"Extracting {Path(archive).name} …")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(extract)
    if table is not None and Path(table).exists():
        shutil.copy2(table, extract / Path(table).name)
    else:
        print("[warn] no element table — meshes will be classified by file name only")

    sys.path.insert(0, str(ROOT / "tools"))
    from import_anatomy import run_import  # noqa: E402
    code = run_import(extract, out, None, dry_run=args.dry_run, merge_layers=True,
                      normalise=True, source_tag="bodyparts3d")
    if code == 0 and not args.dry_run:
        (out / "ATTRIBUTION.txt").write_text(ATTRIBUTION + "\n", encoding="utf-8")
        print("\nInstalled. Restart BioHuman3D.")
        print(f"Attribution (keep it with any video or product):\n  {ATTRIBUTION}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
