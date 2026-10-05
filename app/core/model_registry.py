"""
Anatomy dataset registry.

A *layer* is the atomic unit the user manipulates: one file (or one GLTF node
group) plus its display metadata. Grouping many meshes into a single layer is
what makes hide/fade/isolate instantaneous.

To extend the atlas: drop a model into ``app/assets/models`` and append a
``LayerSpec`` row below. ``LayerRegistry.resolve`` silently skips missing files,
so a partially-populated asset folder still produces a working scene.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

# (R, G, B) in 0..1 — tuned for a dark viewport with a cool key light.
SKIN = (0.86, 0.71, 0.62)
FASCIA = (0.92, 0.90, 0.86)
MUSCLE = (0.72, 0.22, 0.24)
TENDON = (0.90, 0.86, 0.74)
BONE = (0.91, 0.88, 0.79)
CARTILAGE = (0.62, 0.76, 0.84)
ARTERY = (0.80, 0.16, 0.20)
VEIN = (0.18, 0.32, 0.72)
NERVE = (0.95, 0.80, 0.20)
LYMPH = (0.36, 0.72, 0.52)
LUNG = (0.83, 0.55, 0.60)
HEART = (0.70, 0.16, 0.22)
LIVER = (0.47, 0.27, 0.24)
KIDNEY = (0.60, 0.34, 0.28)
BRAIN = (0.80, 0.66, 0.70)
GUT = (0.78, 0.62, 0.42)


@dataclass(frozen=True)
class LayerSpec:
    """Declarative description of one manipulable anatomical layer."""

    id: str
    label: str
    group: str                                   # "Integumentary", "Skeletal", ...
    color: Tuple[float, float, float]
    files: Tuple[str, ...] = ()                  # candidate filenames, first match wins
    default_opacity: float = 1.0
    default_visible: bool = True
    explode_vector: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    description: str = ""
    systems: Tuple[str, ...] = ()

    def filename_candidates(self) -> Sequence[str]:
        return self.files or (f"{self.id}.glb", f"{self.id}.gltf", f"{self.id}.vtp", f"{self.id}.obj")


# ---------------------------------------------------------------------------
# The atlas definition
# ---------------------------------------------------------------------------
ANATOMY_LAYERS: Tuple[LayerSpec, ...] = (
    # ── Surface ────────────────────────────────────────────────────────────
    LayerSpec(
        id="skin", label="Skin / Integument", group="Surface", color=SKIN,
        default_opacity=0.42, explode_vector=(0, 0, 0.12),
        description="Keratinised stratified squamous epithelium with dermis and "
                    "hypodermis; primary thermoregulatory and barrier organ.",
        systems=("Integumentary",),
    ),
    LayerSpec(
        id="fascia", label="Superficial Fascia", group="Surface", color=FASCIA,
        default_opacity=0.28,
        description="Loose connective tissue plane separating skin from deep fascia.",
        systems=("Integumentary", "Muscular"),
    ),
    # ── Musculoskeletal ────────────────────────────────────────────────────
    LayerSpec(
        id="muscles", label="Muscular System", group="Musculoskeletal", color=MUSCLE,
        description="Over 600 skeletal muscles producing voluntary movement and posture.",
        systems=("Muscular",),
    ),
    LayerSpec(
        id="tendons", label="Tendons & Ligaments", group="Musculoskeletal", color=TENDON,
        description="Dense regular collagenous tissue transmitting force to bone.",
        systems=("Muscular", "Skeletal"),
    ),
    LayerSpec(
        id="skeleton", label="Skeletal System", group="Musculoskeletal", color=BONE,
        description="206 bones forming the axial and appendicular skeleton; "
                    "marrow, mineral store and lever system.",
        systems=("Skeletal",),
    ),
    LayerSpec(
        id="cartilage", label="Articular Cartilage", group="Musculoskeletal", color=CARTILAGE,
        default_opacity=0.65,
        description="Hyaline cartilage lining synovial joints; near-frictionless load bearing.",
        systems=("Skeletal",),
    ),
    # ── Vascular & neural ──────────────────────────────────────────────────
    LayerSpec(
        id="arteries", label="Arterial System", group="Vascular", color=ARTERY,
        description="High-pressure distribution network arising from the left ventricle.",
        systems=("Cardiovascular",),
    ),
    LayerSpec(
        id="veins", label="Venous System", group="Vascular", color=VEIN,
        description="Low-pressure capacitance network returning blood to the right atrium.",
        systems=("Cardiovascular",),
    ),
    LayerSpec(
        id="nerves", label="Nervous System", group="Neural", color=NERVE,
        description="Central and peripheral nervous tissue carrying afferent and "
                    "efferent signals.",
        systems=("Nervous",),
    ),
    LayerSpec(
        id="lymphatics", label="Lymphatic System", group="Neural", color=LYMPH,
        default_opacity=0.75,
        description="Interstitial fluid drainage, fat absorption and immune surveillance.",
        systems=("Lymphatic",),
    ),
    # ── Viscera ────────────────────────────────────────────────────────────
    LayerSpec(
        id="brain", label="Brain", group="Viscera", color=BRAIN,
        description="Cerebrum, cerebellum and brainstem; seat of cognition and homeostasis.",
        systems=("Nervous",),
    ),
    LayerSpec(
        id="lungs", label="Lungs & Airways", group="Viscera", color=LUNG,
        default_opacity=0.55,
        description="Paired respiratory organs performing gas exchange across ~480M alveoli.",
        systems=("Respiratory",),
    ),
    LayerSpec(
        id="heart", label="Heart", group="Viscera", color=HEART,
        description="Four-chambered muscular pump; ~5 L/min resting cardiac output.",
        systems=("Cardiovascular",),
    ),
    LayerSpec(
        id="liver", label="Liver", group="Viscera", color=LIVER,
        description="Largest gland; detoxification, bile production, protein synthesis.",
        systems=("Digestive",),
    ),
    LayerSpec(
        id="kidneys", label="Kidneys", group="Viscera", color=KIDNEY,
        description="Retroperitoneal organs regulating fluid, electrolyte and acid-base balance.",
        systems=("Urinary",),
    ),
    LayerSpec(
        id="digestive", label="Digestive Tract", group="Viscera", color=GUT,
        description="Oesophagus to colon; mechanical and enzymatic nutrient breakdown.",
        systems=("Digestive",),
    ),
)


@dataclass
class ResolvedLayer:
    """A :class:`LayerSpec` bound to concrete existing files on disk."""

    spec: LayerSpec
    paths: List[Path] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.spec.id


class LayerRegistry:
    """Resolves :data:`ANATOMY_LAYERS` against a models directory."""

    def __init__(self, models_dir: Path, layers: Iterable[LayerSpec] = ANATOMY_LAYERS):
        self.models_dir = Path(models_dir)
        self._specs: Dict[str, LayerSpec] = {s.id: s for s in layers}

    # -- queries -----------------------------------------------------------
    @property
    def specs(self) -> List[LayerSpec]:
        return list(self._specs.values())

    def spec(self, layer_id: str) -> LayerSpec | None:
        return self._specs.get(layer_id)

    def label(self, layer_id: str) -> str:
        spec = self._specs.get(layer_id)
        return spec.label if spec else layer_id.replace("_", " ").title()

    def groups(self) -> Dict[str, List[LayerSpec]]:
        """Layer specs bucketed by their display group, order preserved."""
        out: Dict[str, List[LayerSpec]] = {}
        for spec in self._specs.values():
            out.setdefault(spec.group, []).append(spec)
        return out

    def resolve(self, layer_id: str) -> ResolvedLayer | None:
        """Return *layer_id* only when at least one backing file exists."""
        spec = self._specs.get(layer_id)
        if spec is None:
            return None
        found = [self.models_dir / name
                 for name in spec.filename_candidates()
                 if (self.models_dir / name).exists()]
        return ResolvedLayer(spec=spec, paths=found) if found else None

    def available_ids(self) -> List[str]:
        return [s.id for s in self._specs.values() if self.resolve(s.id) is not None]

    def missing_ids(self) -> List[str]:
        return [s.id for s in self._specs.values() if self.resolve(s.id) is None]
