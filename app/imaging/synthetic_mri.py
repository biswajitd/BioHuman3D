"""
Simulated MRI from the anatomical model.

When no real scan is loaded, BioHuman3D can still pair every cross-section with
an MRI-like image: the named structures are voxelised into a label volume, each
label is mapped to a tissue class, and each tissue gets the relative signal it
shows on T1-, T2- and proton-density-weighted spin-echo imaging at 1.5 T
(fat bright on T1, fluid bright on T2, cortical bone and air dark on all,
flowing arterial blood a flow void …). Partial-volume blur, a smooth coil bias
field and Rician noise are then applied, because real magnitude images have all
three.

This is a *teaching simulation*, always labelled as such in the UI: geometry
comes from the 3D model and contrast from textbook tissue behaviour, so it shows
what a section looks like and where structures lie — it is not patient data and
not a substitute for a scan. Load a real NIfTI/DICOM study for that.

Relative intensities (0–1) follow the standard tissue appearance tables in
Westbrook, *MRI in Practice* (5th ed.) and McRobbie, *MRI from Picture to Proton*
(3rd ed.).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.core.vtk_utils import vtk
from app.imaging.volume import ImagingVolume

try:
    from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
except Exception:  # pragma: no cover
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy  # type: ignore

#: tissue → (T1-weighted, T2-weighted, PD-weighted) relative signal.
TISSUE_SIGNAL: Dict[str, Tuple[float, float, float]] = {
    "air":           (0.00, 0.00, 0.00),
    "lung":          (0.04, 0.05, 0.06),
    "fat":           (0.95, 0.75, 0.85),
    "soft tissue":   (0.70, 0.55, 0.70),   # subcutaneous / visceral fat-connective mix
    "skin":          (0.55, 0.45, 0.65),
    "muscle":        (0.40, 0.22, 0.55),
    "tendon":        (0.05, 0.04, 0.10),
    "fibrocartilage": (0.08, 0.06, 0.15),
    "cartilage":     (0.38, 0.48, 0.65),
    "disc":          (0.38, 0.80, 0.70),   # hydrated nucleus pulposus: bright on T2
    "cortical bone": (0.02, 0.02, 0.04),
    "marrow":        (0.80, 0.55, 0.75),   # adult fatty marrow
    "artery":        (0.04, 0.03, 0.06),   # flow void on spin echo
    "vein":          (0.18, 0.35, 0.30),
    "blood pool":    (0.08, 0.10, 0.15),   # cardiac chambers (flow)
    "myocardium":    (0.42, 0.24, 0.55),
    "nerve":         (0.42, 0.42, 0.55),
    "spinal cord":   (0.50, 0.40, 0.62),
    "grey matter":   (0.52, 0.58, 0.78),
    "white matter":  (0.72, 0.36, 0.66),
    "csf":           (0.10, 1.00, 0.70),
    "liver":         (0.56, 0.24, 0.60),
    "spleen":        (0.40, 0.58, 0.68),
    "kidney":        (0.46, 0.60, 0.70),
    "adrenal":       (0.45, 0.40, 0.60),
    "pancreas":      (0.62, 0.32, 0.62),
    "bowel":         (0.38, 0.52, 0.60),
    "fluid":         (0.10, 0.98, 0.72),   # urine, bile
    "lymph node":    (0.42, 0.50, 0.65),
}

SEQUENCES = ("T1", "T2", "PD")
SEQUENCE_LABELS = {"T1": "T1-weighted", "T2": "T2-weighted", "PD": "Proton-density weighted"}

#: Layers drawn first are overwritten by later ones (priority low → high).
LAYER_PRIORITY = ("skin", "fascia", "muscles", "tendons", "lymphatics", "digestive", "liver",
                  "kidneys", "lungs", "heart", "brain", "cartilage", "skeleton", "veins",
                  "arteries", "nerves")


def tissue_for(layer: str, name: str) -> str:
    low = name.lower()
    if layer == "skin":
        return "skin"
    if layer == "fascia":
        return "fat"
    if layer == "muscles":
        return "muscle"
    if layer == "tendons":
        return "tendon"
    if layer == "cartilage":
        if "disc" in low:
            return "disc"
        if "menisc" in low or "labrum" in low:
            return "fibrocartilage"
        return "cartilage"
    if layer == "skeleton":
        return "bone"                      # split into cortex + marrow by depth
    if layer == "arteries":
        return "artery"
    if layer == "veins":
        return "vein"
    if layer == "nerves":
        return "spinal cord" if "spinal cord" in low or "cauda" in low else "nerve"
    if layer == "lymphatics":
        if "spleen" in low:
            return "spleen"
        if "thymus" in low:
            return "fat"
        if "duct" in low or "cisterna" in low:
            return "fluid"
        return "lymph node"
    if layer == "brain":
        if "brainstem" in low or "corpus callosum" in low:
            return "white matter"
        return "brain"                     # cortex + white matter by depth
    if layer == "lungs":
        if any(k in low for k in ("trachea", "bronch", "larynx")):
            return "air"
        if "diaphragm" in low:
            return "muscle"
        return "lung"
    if layer == "heart":
        if any(k in low for k in ("ventricle", "atrium")):
            return "heart"                 # myocardium + blood pool by depth
        return "artery"
    if layer == "liver":
        if "gallbladder" in low or "bile" in low:
            return "fluid"
        return "liver"
    if layer == "kidneys":
        if "adrenal" in low:
            return "adrenal"
        if "bladder" in low or "ureter" in low:
            return "fluid"
        return "kidney"
    if layer == "digestive":
        if "pancrea" in low:
            return "pancreas"
        if "oesophagus" in low or "esophagus" in low:
            return "muscle"
        return "bowel"
    return "soft tissue"


#: Depth-split tissues: (outer tissue, inner tissue, shell thickness in metres)
SHELLS = {"bone": ("cortical bone", "marrow", 0.0035),
          "brain": ("grey matter", "white matter", 0.0045),
          "heart": ("myocardium", "blood pool", 0.009)}


@dataclass
class _Item:
    layer: str
    name: str
    poly: object


def _mask(poly, origin, spacing, extent) -> Optional[np.ndarray]:
    """Boolean voxel mask of a closed surface within *extent* (x0,x1,y0,y1,z0,z1)."""
    stencil = vtk.vtkPolyDataToImageStencil()
    stencil.SetInputData(poly)
    stencil.SetOutputOrigin(*origin)
    stencil.SetOutputSpacing(*spacing)
    stencil.SetOutputWholeExtent(*extent)
    stencil.SetTolerance(0.0)
    to_image = vtk.vtkImageStencilToImage()
    to_image.SetInputConnection(stencil.GetOutputPort())
    to_image.SetInsideValue(1)
    to_image.SetOutsideValue(0)
    to_image.SetOutputScalarTypeToUnsignedChar()
    to_image.Update()
    img = to_image.GetOutput()
    dims = img.GetDimensions()
    if min(dims) <= 0:
        return None
    arr = vtk_to_numpy(img.GetPointData().GetScalars()).reshape(dims[2], dims[1], dims[0])
    return np.transpose(arr, (2, 1, 0)).astype(bool)


def _erode(mask: np.ndarray, steps: int) -> np.ndarray:
    """Binary erosion with a 6-neighbour structuring element."""
    out = mask.copy()
    for _ in range(max(0, steps)):
        inner = out.copy()
        inner[1:] &= out[:-1]
        inner[:-1] &= out[1:]
        inner[:, 1:] &= out[:, :-1]
        inner[:, :-1] &= out[:, 1:]
        inner[:, :, 1:] &= out[:, :, :-1]
        inner[:, :, :-1] &= out[:, :, 1:]
        inner[0] = inner[-1] = False
        inner[:, 0] = inner[:, -1] = False
        inner[:, :, 0] = inner[:, :, -1] = False
        out = inner
    return out


#: Bones that enclose other structures: only their wall is bone.
HOLLOW_BONES = ("cranium", "skull", "calvaria")


class MRISimulator:
    """Builds the label volume once; renders any sequence from it on demand."""

    def __init__(self, items: Sequence[Tuple[str, str, object]], spacing: float = 0.004,
                 progress: Optional[Callable[[int, str], None]] = None,
                 cancelled: Optional[Callable[[], bool]] = None) -> None:
        self.spacing = float(spacing)
        self._items = [_Item(*it) for it in items if it[2] is not None and it[2].GetNumberOfCells()]
        self._progress = progress or (lambda p, m: None)
        self._cancelled = cancelled or (lambda: False)
        self.labels: Optional[np.ndarray] = None
        self.tissue: Optional[np.ndarray] = None       # tissue index per voxel
        self.tissue_names: List[str] = list(TISSUE_SIGNAL)
        self.label_names: List[str] = [""]
        self.origin = (0.0, 0.0, 0.0)

    # ---------------------------------------------------------------- build
    def build(self) -> bool:
        if not self._items:
            return False
        b = np.array([it.poly.GetBounds() for it in self._items])
        lo = np.array([b[:, 0].min(), b[:, 2].min(), b[:, 4].min()]) - 0.012
        hi = np.array([b[:, 1].max(), b[:, 3].max(), b[:, 5].max()]) + 0.012
        sp = self.spacing
        dims = np.ceil((hi - lo) / sp).astype(int) + 1
        self.origin = tuple(lo)
        labels = np.zeros(dims, dtype=np.uint16)
        tissue = np.full(dims, self.tissue_names.index("air"), dtype=np.uint8)
        body = np.zeros(dims, dtype=bool)

        order = sorted(self._items, key=lambda it: (LAYER_PRIORITY.index(it.layer)
                                                    if it.layer in LAYER_PRIORITY else 0))
        total = len(order)
        tindex = {name: i for i, name in enumerate(self.tissue_names)}
        for count, item in enumerate(order):
            if self._cancelled():
                return False
            if count % 10 == 0:
                self._progress(int(count * 85 / total), f"Voxelising {count + 1}/{total}: {item.name}")
            bb = item.poly.GetBounds()
            i0 = np.clip(np.floor((np.array(bb[0::2]) - lo) / sp).astype(int) - 1, 0, dims - 1)
            i1 = np.clip(np.ceil((np.array(bb[1::2]) - lo) / sp).astype(int) + 1, 0, dims - 1)
            if np.any(i1 <= i0):
                continue
            extent = (i0[0], i1[0], i0[1], i1[1], i0[2], i1[2])
            mask = _mask(item.poly, tuple(lo), (sp, sp, sp), extent)
            if mask is None or not mask.any():
                continue
            region = (slice(i0[0], i1[0] + 1), slice(i0[1], i1[1] + 1), slice(i0[2], i1[2] + 1))
            self.label_names.append(item.name)
            label_id = len(self.label_names) - 1
            kind = tissue_for(item.layer, item.name)
            if item.layer in ("skin", "fascia"):
                body[region] |= mask
                # Skin is a thin shell; its interior is subcutaneous tissue.
                inner = _erode(mask, max(1, int(round(0.0025 / sp))))
                shell = mask & ~inner
                tissue[region][shell] = tindex["skin" if item.layer == "skin" else "fat"]
                tissue[region][inner] = tindex["soft tissue"]
                labels[region][shell] = label_id
                continue
            body[region] |= mask
            if kind == "bone" and any(k in item.name.lower() for k in HOLLOW_BONES):
                # Skull vault: ~7 mm wall (outer table, diploë, inner table);
                # the cavity keeps the brain and fills the rest with CSF.
                wall_inner = _erode(mask, max(1, int(round(0.007 / sp))))
                wall = mask & ~wall_inner
                diploe = _erode(mask, max(1, int(round(0.002 / sp)))) & ~_erode(mask, max(2, int(round(0.005 / sp))))
                labels[region][wall] = label_id
                tissue[region][wall] = tindex["cortical bone"]
                tissue[region][wall & diploe] = tindex["marrow"]
                sub = tissue[region]
                empty = wall_inner & np.isin(sub, [tindex["air"], tindex["soft tissue"], tindex["fat"]])
                sub[empty] = tindex["csf"]
                continue
            labels[region][mask] = label_id
            if kind in SHELLS:
                outer, inner_kind, thick = SHELLS[kind]
                inner = _erode(mask, max(1, int(round(thick / sp))))
                tissue[region][mask & ~inner] = tindex[outer]
                tissue[region][inner] = tindex[inner_kind]
            else:
                tissue[region][mask] = tindex.get(kind, tindex["soft tissue"])

        # Without a skin layer (some atlases), the body is the structures dilated.
        if not any(it.layer == "skin" for it in self._items):
            grown = body.copy()
            for _ in range(max(1, int(round(0.008 / sp)))):
                g = grown.copy()
                g[1:] |= grown[:-1]; g[:-1] |= grown[1:]
                g[:, 1:] |= grown[:, :-1]; g[:, :-1] |= grown[:, 1:]
                g[:, :, 1:] |= grown[:, :, :-1]; g[:, :, :-1] |= grown[:, :, 1:]
                grown = g
            fill = grown & (tissue == tindex["air"])
            tissue[fill] = tindex["soft tissue"]
        self.labels, self.tissue = labels, tissue
        self._progress(88, "Voxelisation complete")
        return True

    # --------------------------------------------------------------- render
    def render(self, sequence: str = "T1", *, noise: float = 0.022, bias: float = 0.10,
               blur_voxels: float = 0.7, seed: int = 11) -> ImagingVolume:
        assert self.tissue is not None and self.labels is not None
        column = SEQUENCES.index(sequence) if sequence in SEQUENCES else 0
        lut = np.array([TISSUE_SIGNAL[t][column] for t in self.tissue_names], dtype=np.float32)
        img = lut[self.tissue]

        # Partial-volume effect: real voxels average neighbouring tissues.
        if blur_voxels > 0:
            img = _gaussian(img, blur_voxels)

        # Receive-coil bias field: smooth, low-frequency multiplicative shading.
        nx, ny, nz = img.shape
        x = np.linspace(-1, 1, nx)[:, None, None]
        y = np.linspace(-1, 1, ny)[None, :, None]
        z = np.linspace(-1, 1, nz)[None, None, :]
        field = 1.0 + bias * (0.6 * np.cos(1.3 * x + 0.4) * np.cos(0.9 * y) + 0.4 * np.sin(1.7 * z + 0.3))
        img = img * field.astype(np.float32)

        # Rician noise: magnitude of a complex signal with Gaussian noise.
        rng = np.random.default_rng(seed)
        n1 = rng.normal(0.0, noise, img.shape).astype(np.float32)
        n2 = rng.normal(0.0, noise, img.shape).astype(np.float32)
        img = np.sqrt((img + n1) ** 2 + n2 ** 2)

        sp = self.spacing
        return ImagingVolume(
            data=np.ascontiguousarray(img, dtype=np.float32), spacing=(sp, sp, sp), origin=self.origin,
            modality=f"Simulated MRI · {SEQUENCE_LABELS.get(sequence, sequence)}",
            description=(f"Simulated from the 3D model · {sp * 1000:.0f} mm isotropic · "
                         f"{len(self.label_names) - 1} labelled structures"),
            simulated=True, labels=self.labels, label_names=list(self.label_names), source="simulation")


def _gaussian(arr: np.ndarray, sigma_vox: float) -> np.ndarray:
    """Separable Gaussian via VTK (fast C++), preserving shape."""
    image = vtk.vtkImageData()
    nx, ny, nz = arr.shape
    image.SetDimensions(nx, ny, nz)
    flat = np.ascontiguousarray(np.transpose(arr, (2, 1, 0))).ravel()
    image.GetPointData().SetScalars(numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_FLOAT))
    smooth = vtk.vtkImageGaussianSmooth()
    smooth.SetInputData(image)
    smooth.SetStandardDeviations(sigma_vox, sigma_vox, sigma_vox)
    smooth.SetRadiusFactors(2.0, 2.0, 2.0)
    smooth.Update()
    out = vtk_to_numpy(smooth.GetOutput().GetPointData().GetScalars()).reshape(nz, ny, nx)
    return np.ascontiguousarray(np.transpose(out, (2, 1, 0)))


def scene_items(viewport) -> List[Tuple[str, str, object]]:
    """Deep copies of every named structure — safe to voxelise off the GUI thread."""
    items = []
    for state in viewport.layers.values():
        for name, poly in zip(state.structures, state.polydata):
            copy = vtk.vtkPolyData()
            copy.DeepCopy(poly)
            items.append((state.id, name or state.label, copy))
    return items
