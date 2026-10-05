"""
Imaging volumes in the BioHuman3D anatomical frame.

Every volume — a real MRI loaded from NIfTI/DICOM or one simulated from the
model — is stored as a numpy array ``data[i, j, k]`` whose axes are the model
axes (x = patient's left, y = posterior, z = superior, i.e. DICOM LPS) in
metres, plus an origin and spacing. Real scans are reoriented on load (closest
canonical orientation), so a slice at z = 1.25 m in the 3D view and the MRI
slice at z = 1.25 m are the same anatomical plane.

Slices are returned in radiological display convention:

* axial    — patient's right on the viewer's left, anterior at the top
* coronal  — patient's right on the viewer's left, superior at the top
* sagittal — anterior on the left, superior at the top
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
PLANE_FOR_AXIS = {"z": "Axial", "y": "Coronal", "x": "Sagittal"}


@dataclass
class ImagingVolume:
    data: np.ndarray                         # (nx, ny, nz) float32
    spacing: Tuple[float, float, float]      # metres
    origin: Tuple[float, float, float]       # world position of voxel (0, 0, 0)
    modality: str = "MRI"
    description: str = ""
    simulated: bool = False
    labels: Optional[np.ndarray] = None      # (nx, ny, nz) uint16 structure index
    label_names: List[str] = field(default_factory=list)
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3))   # alignment shift (m)
    source: str = ""

    # ------------------------------------------------------------ geometry
    @property
    def shape(self) -> Tuple[int, int, int]:
        return tuple(self.data.shape)  # type: ignore[return-value]

    def world_origin(self) -> np.ndarray:
        return np.asarray(self.origin, float) + self.offset

    def bounds(self) -> Tuple[float, ...]:
        o = self.world_origin()
        sp = np.asarray(self.spacing, float)
        hi = o + sp * (np.asarray(self.shape) - 1)
        return (o[0], hi[0], o[1], hi[1], o[2], hi[2])

    def index_of(self, axis: str, position: float) -> int:
        k = AXIS_INDEX[axis]
        return int(round((position - self.world_origin()[k]) / self.spacing[k]))

    def contains(self, axis: str, position: float) -> bool:
        idx = self.index_of(axis, position)
        return 0 <= idx < self.shape[AXIS_INDEX[axis]]

    def world_at(self, i: float, j: float, k: float) -> np.ndarray:
        return self.world_origin() + np.asarray(self.spacing) * np.array([i, j, k], float)

    def voxel_at(self, point: Sequence[float]) -> Optional[Tuple[int, int, int]]:
        idx = np.round((np.asarray(point, float) - self.world_origin()) / np.asarray(self.spacing)).astype(int)
        if np.any(idx < 0) or np.any(idx >= np.asarray(self.shape)):
            return None
        return tuple(int(v) for v in idx)  # type: ignore[return-value]

    # -------------------------------------------------------------- slices
    def slice(self, axis: str, position: float, array: Optional[np.ndarray] = None
              ) -> Tuple[Optional[np.ndarray], float]:
        """2D slice in display convention and its pixel aspect (height/width).

        Returns ``(None, 1.0)`` when *position* is outside the volume.
        """
        src = self.data if array is None else array
        idx = self.index_of(axis, position)
        if not 0 <= idx < src.shape[AXIS_INDEX[axis]]:
            return None, 1.0
        sx, sy, sz = self.spacing
        if axis == "z":                      # rows: y ascending (anterior up); cols: x ascending
            return src[:, :, idx].T, sy / sx
        if axis == "y":                      # rows: z descending (superior up)
            return src[:, idx, :].T[::-1, :], sz / sx
        return src[idx, :, :].T[::-1, :], sz / sy

    def display_to_world(self, axis: str, position: float, row: float, col: float) -> np.ndarray:
        """Inverse of :meth:`slice` for one pixel (used for hover labels)."""
        idx = self.index_of(axis, position)
        nx, ny, nz = self.shape
        if axis == "z":
            return self.world_at(col, row, idx)
        if axis == "y":
            return self.world_at(col, idx, nz - 1 - row)
        return self.world_at(idx, col, nz - 1 - row)

    def label_at(self, point: Sequence[float]) -> str:
        if self.labels is None:
            return ""
        vox = self.voxel_at(point)
        if vox is None:
            return ""
        index = int(self.labels[vox])
        return self.label_names[index] if 0 < index < len(self.label_names) else ""

    def default_window(self) -> Tuple[float, float]:
        """(window, level) from robust percentiles of the non-background signal."""
        sample = self.data[::3, ::3, ::3]
        body = sample[sample > np.percentile(sample, 30)]
        if not body.size:
            body = sample.ravel()
        lo, hi = np.percentile(body, 1), np.percentile(body, 99.5)
        return float(max(hi - lo, 1e-6)), float((hi + lo) / 2)


# ---------------------------------------------------------------------------
# Loading real scans
# ---------------------------------------------------------------------------
def _from_affine(data: np.ndarray, affine: np.ndarray, *, lps: bool) -> Tuple[np.ndarray, Tuple, Tuple]:
    """Reorient *data* (voxel → patient mm via *affine*) onto model axes.

    Uses the closest axis permutation/flip (exact for the usual axial, coronal
    and sagittal acquisitions; oblique scans are snapped to the nearest axes).
    """
    m = np.asarray(affine, float)[:3, :3]
    t = np.asarray(affine, float)[:3, 3]
    if not lps:                         # RAS → LPS
        flip = np.diag([-1.0, -1.0, 1.0])
        m, t = flip @ m, flip @ t
    # For each voxel axis, which world axis does it mostly follow?
    order = [int(np.argmax(np.abs(m[:, c]))) for c in range(3)]
    if sorted(order) != [0, 1, 2]:
        order = [0, 1, 2]
    perm = [order.index(w) for w in range(3)]          # world axis w ← voxel axis perm[w]
    data = np.transpose(data, perm)
    m = m[:, perm]
    spacing = np.abs(np.array([m[w, w] for w in range(3)]))
    spacing = np.where(spacing > 0, spacing, np.linalg.norm(m, axis=0))
    origin = t.copy()
    for w in range(3):
        if m[w, w] < 0:                                 # axis runs backwards: flip it
            data = np.flip(data, axis=w)
            origin = origin + m[:, w] * (data.shape[w] - 1)
    return (np.ascontiguousarray(data, dtype=np.float32), tuple(spacing / 1000.0),
            tuple(origin / 1000.0))


def load_nifti(path: Path) -> ImagingVolume:
    path = Path(path)
    try:
        import nibabel as nib
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Reading NIfTI needs nibabel:  pip install nibabel") from exc
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    while data.ndim > 3:
        data = data[..., 0]
    arr, spacing, origin = _from_affine(data.astype(np.float32), img.affine, lps=False)
    return ImagingVolume(arr, spacing, origin, modality="MRI", source=str(path),
                         description=f"{path.name} · {arr.shape[0]}×{arr.shape[1]}×{arr.shape[2]} · "
                                     f"{spacing[0] * 1000:.1f}×{spacing[1] * 1000:.1f}×{spacing[2] * 1000:.1f} mm")


def load_dicom_series(folder: Path) -> ImagingVolume:
    """Read a single DICOM series from a folder (uncompressed transfer syntaxes)."""
    folder = Path(folder)
    try:
        import pydicom
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Reading DICOM needs pydicom:  pip install pydicom") from exc
    slices = []
    for f in sorted(folder.rglob("*")):
        if not f.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(f), force=True)
            if hasattr(ds, "ImagePositionPatient") and hasattr(ds, "pixel_array"):
                slices.append(ds)
        except Exception:
            continue
    if not slices:
        raise RuntimeError(f"No readable DICOM images in {folder}")
    series = slices[0].get("SeriesInstanceUID", "")
    slices = [s for s in slices if s.get("SeriesInstanceUID", "") == series]
    row = np.array(slices[0].ImageOrientationPatient[:3], float)
    col = np.array(slices[0].ImageOrientationPatient[3:], float)
    normal = np.cross(row, col)
    slices.sort(key=lambda s: float(np.dot(np.array(s.ImagePositionPatient, float), normal)))
    stack = []
    for s in slices:
        px = s.pixel_array.astype(np.float32)
        px = px * float(getattr(s, "RescaleSlope", 1) or 1) + float(getattr(s, "RescaleIntercept", 0) or 0)
        stack.append(px.T)                               # (cols, rows)
    data = np.stack(stack, axis=2)
    dr, dc = (float(v) for v in slices[0].PixelSpacing)  # row spacing, column spacing
    p0 = np.array(slices[0].ImagePositionPatient, float)
    ds_ = (float(np.dot(np.array(slices[-1].ImagePositionPatient, float) - p0, normal)) / (len(slices) - 1)
           if len(slices) > 1 else float(getattr(slices[0], "SliceThickness", 1.0)))
    affine = np.eye(4)
    affine[:3, 0] = row * dc
    affine[:3, 1] = col * dr
    affine[:3, 2] = normal * ds_
    affine[:3, 3] = p0
    arr, spacing, origin = _from_affine(data, affine, lps=True)
    desc = str(slices[0].get("SeriesDescription", "") or slices[0].get("Modality", "MRI"))
    return ImagingVolume(arr, spacing, origin, modality=str(slices[0].get("Modality", "MR")),
                         source=str(folder),
                         description=f"{desc} · {len(slices)} slices · {dc:.2f}×{dr:.2f}×{abs(ds_):.2f} mm")


def load_volume(path: Path) -> ImagingVolume:
    path = Path(path)
    if path.is_dir():
        return load_dicom_series(path)
    name = path.name.lower()
    if name.endswith((".nii", ".nii.gz", ".img", ".hdr")):
        return load_nifti(path)
    if name.endswith(".dcm"):
        return load_dicom_series(path.parent)
    raise RuntimeError(f"Unsupported imaging file: {path.name} (use NIfTI .nii/.nii.gz or a DICOM folder)")


# ---------------------------------------------------------------------------
# Alignment to the model
# ---------------------------------------------------------------------------
#: Region → structure keywords whose bounds give the region centre.
REGIONS: Dict[str, Tuple[str, ...]] = {
    "Head": ("cerebral hemisphere", "cerebellum", "cranium", "brain"),
    "Neck": ("C3 vertebra", "C4 vertebra", "C5 vertebra", "larynx", "trachea"),
    "Thorax": ("lung", "heart", "ventricle", "atrium"),
    "Abdomen": ("liver", "kidney", "stomach", "pancreas"),
    "Pelvis": ("hip bone", "sacrum", "urinary bladder"),
    "Knee": ("patella", "meniscus"),
    "Whole body": (),
}


def guess_region(volume: ImagingVolume) -> str:
    b = volume.bounds()
    ex, ey, ez = b[1] - b[0], b[3] - b[2], b[5] - b[4]
    if ez > 1.0:
        return "Whole body"
    if max(ex, ey) <= 0.26 and ez <= 0.30:
        return "Head"
    return "Thorax" if ex > 0.30 else "Head"


def align_to_region(volume: ImagingVolume, viewport, region: str) -> str:
    """Shift *volume* so its centre sits on the model's *region* centre."""
    names: List[str] = []
    keys = REGIONS.get(region, ())
    if keys:
        for state in viewport.layers.values():
            for name in state.structures:
                low = name.lower()
                if any(k.lower() in low for k in keys):
                    names.append(name)
    target = viewport.structure_bounds(names) if names else None
    if target is None:
        boxes = [p.GetBounds() for s in viewport.layers.values() for p in s.polydata]
        if not boxes:
            return "no model loaded"
        target = (min(b[0] for b in boxes), max(b[1] for b in boxes), min(b[2] for b in boxes),
                  max(b[3] for b in boxes), min(b[4] for b in boxes), max(b[5] for b in boxes))
    vb = volume.bounds()
    centre_t = np.array([(target[0] + target[1]) / 2, (target[2] + target[3]) / 2, (target[4] + target[5]) / 2])
    centre_v = np.array([(vb[0] + vb[1]) / 2, (vb[2] + vb[3]) / 2, (vb[4] + vb[5]) / 2])
    volume.offset = volume.offset + (centre_t - centre_v)
    return region
