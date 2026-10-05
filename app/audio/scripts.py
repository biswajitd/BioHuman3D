"""
Authored narration scripts + guided tours.

**One tour per body system.** Every system in the layer registry has its own
script, so exporting a video of the arterial system narrates the arterial
system — not whatever tour happened to be selected in a dropdown.

A *tour* is an ordered list of :class:`TourKeyframe`. Each keyframe carries:

* a camera state (position / focal point / view-up) or a view preset,
* the layer opacities to fade in for that beat,
* narration text spoken by the audio engine,
* travel timing so the camera glides rather than cuts.

``SYSTEM_TOUR_MAP`` links the sidebar's system presets to the matching tour, which
is what keeps video export in step with the system the user is actually viewing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from app.audio.manager import NarrationCue


@dataclass
class TourKeyframe:
    """One narrated beat of a guided tour."""

    title: str
    narration: str
    camera: Optional[Dict[str, Sequence[float]]] = None
    layers: Dict[str, float] = field(default_factory=dict)   # layer_id → opacity
    view: str = ""                                            # preset fallback
    travel_seconds: float = 1.8
    dwell_seconds: float = 0.0                                # 0 = derive from speech
    focus_layer: str = ""                                     # camera re-frame target
    focus_structures: List[str] = field(default_factory=list) # frame these named structures
    focus_bounds: Optional[Sequence[float]] = None            # explicit framing box (motion envelope)
    effect: Optional[object] = None                           # SceneEffect animated in this beat


@dataclass
class Tour:
    """A named sequence of keyframes."""

    id: str
    title: str
    subtitle: str = ""
    system: str = ""                     # sidebar preset this tour belongs to
    keyframes: List[TourKeyframe] = field(default_factory=list)

    def cues(self) -> List[NarrationCue]:
        """Narration cues in tour order (index-aligned with keyframes)."""
        return [
            NarrationCue(index=i, text=kf.narration, title=kf.title,
                         camera=kf.camera, layer_states=kf.layers)
            for i, kf in enumerate(self.keyframes)
        ]

    @property
    def word_count(self) -> int:
        return sum(len(kf.narration.split()) for kf in self.keyframes)

    @property
    def duration_estimate(self) -> float:
        return sum(kf.travel_seconds + max(kf.dwell_seconds, 2.0) for kf in self.keyframes)


# ---------------------------------------------------------------------------
# 1. Integumentary — skin & fascia
# ---------------------------------------------------------------------------
INTEGUMENTARY_OVERVIEW = Tour(
    id="integumentary_overview",
    title="Skin & Fascia",
    subtitle="The barrier and the insulation beneath it",
    system="Integumentary",
    keyframes=[
        TourKeyframe(
            title="The body's largest organ",
            view="anterior",
            layers={"skin": 1.0, "fascia": 0.05, "muscles": 0.05},
            focus_layer="skin",
            narration=(
                "Begin with the largest organ you have. Skin covers roughly two square "
                "metres, and it is not a passive wrapper. The epidermis is avascular and "
                "continuously shed, while the dermis beneath carries the vessels, nerves "
                "and collagen that give skin its strength."
            ),
        ),
        TourKeyframe(
            title="Beneath the dermis",
            view="right",
            layers={"skin": 0.18, "fascia": 1.0, "muscles": 0.30},
            focus_layer="fascia",
            narration=(
                "The skin fades to reveal superficial fascia: loose connective tissue loaded "
                "with adipose. It insulates, stores energy and lets skin glide over the "
                "structures beneath it. That mobility is why a deep burn contracts, but a "
                "superficial one usually does not."
            ),
        ),
        TourKeyframe(
            title="Clinical takeaway",
            view="isometric",
            layers={"skin": 1.0, "fascia": 0.25, "muscles": 0.20},
            focus_layer="skin",
            narration=(
                "One clinical anchor. Skin loss is staged by depth, not by area alone. A "
                "partial thickness burn heals from retained dermal appendages, while a full "
                "thickness burn destroys them and needs grafting. Depth, not size, drives "
                "the decision."
            ),
        ),
    ],
)


# ---------------------------------------------------------------------------
# 2. Skeletal
# ---------------------------------------------------------------------------
SKELETAL_OVERVIEW = Tour(
    id="skeletal_overview",
    title="Skeletal System",
    subtitle="Layers, leverage and the living bone",
    system="Skeletal",
    keyframes=[
        TourKeyframe(
            title="Meet the skeleton",
            view="anterior",
            layers={"skin": 0.06, "fascia": 0.04, "muscles": 0.12,
                    "tendons": 0.10, "skeleton": 1.0},
            focus_layer="skeleton",
            narration=(
                "Welcome. The skin and muscle layers have faded back so the skeleton "
                "stands alone. Two hundred and six bones form this framework, giving the "
                "body its shape, protecting the organs within, and acting as the lever "
                "system every muscle pulls against."
            ),
        ),
        TourKeyframe(
            title="The axial core",
            view="isometric",
            focus_layer="skeleton",
            layers={"skeleton": 1.0, "cartilage": 0.9},
            narration=(
                "The axial skeleton, the skull, vertebral column and rib cage, forms the "
                "central pillar. Thirty three vertebrae stack into a column with four "
                "natural curves, and those curves are what turn a stack of blocks into a "
                "shock absorber."
            ),
        ),
        TourKeyframe(
            title="Appendicular skeleton",
            view="right",
            focus_layer="skeleton",
            layers={"skeleton": 1.0, "muscles": 0.35},
            narration=(
                "The appendicular skeleton hangs off that core. Notice how the limb bones "
                "are denser and thicker where load is highest at the joints, and how the "
                "shoulder sacrifices stability for an extraordinary range of movement."
            ),
        ),
        TourKeyframe(
            title="Cartilage at the joint",
            layers={"skeleton": 0.35, "cartilage": 1.0, "muscles": 0.05},
            focus_layer="cartilage",
            narration=(
                "Where two bones meet, hyaline cartilage lines the surface. It is smooth, "
                "avascular and almost frictionless, and it is the tissue that osteoarthritis "
                "quietly destroys. Cartilage has no blood supply of its own, which is exactly "
                "why it heals so poorly."
            ),
        ),
        TourKeyframe(
            title="Clinical takeaway",
            view="isometric",
            layers={"skeleton": 1.0, "muscles": 0.25, "skin": 0.12},
            focus_layer="skeleton",
            narration=(
                "One clinical point to carry away. Bone is living tissue, constantly rebuilt "
                "by osteoblasts and resorbed by osteoclasts. That balance is why load bearing "
                "strengthens bone and immobility weakens it, and why the same skeleton looks "
                "different in a patient after six months of bed rest."
            ),
        ),
    ],
)


# ---------------------------------------------------------------------------
# 3. Muscular
# ---------------------------------------------------------------------------
MUSCULAR_OVERVIEW = Tour(
    id="muscular_overview",
    title="Muscular System",
    subtitle="From fascia to forceful contraction",
    system="Muscular",
    keyframes=[
        TourKeyframe(
            title="Under the skin",
            layers={"skin": 0.10, "fascia": 0.35, "muscles": 1.0, "skeleton": 0.20},
            view="anterior",
            focus_layer="muscles",
            narration=(
                "Skin faded, fascia partially transparent. What remains is over six hundred "
                "skeletal muscles, the voluntary motors of the body, arranged in antagonistic "
                "pairs across every joint."
            ),
        ),
        TourKeyframe(
            title="Tendon and bone",
            layers={"muscles": 0.85, "tendons": 1.0, "skeleton": 0.55},
            focus_layer="tendons",
            narration=(
                "Muscle does not attach to muscle. Dense regular collagen in the tendon gathers "
                "the force and delivers it to bone, concentrating tension onto a small footprint "
                "of periosteum. That is also why tendon injuries are so slow to heal."
            ),
        ),
        TourKeyframe(
            title="Nerve supply",
            layers={"muscles": 0.7, "nerves": 1.0, "skeleton": 0.25},
            focus_layer="nerves",
            narration=(
                "Follow the nerves. Every skeletal muscle receives a somatic motor supply, and "
                "the pattern of that supply is one of the most reliable maps in medicine. If a "
                "muscle is weak, the question is almost always where along that nerve the "
                "problem lies."
            ),
        ),
        TourKeyframe(
            title="Vessels in muscle",
            layers={"muscles": 0.45, "arteries": 1.0, "veins": 0.6, "skeleton": 0.15},
            focus_layer="arteries",
            narration=(
                "Muscle is metabolically expensive, so its vascular supply is generous. During "
                "exercise, local metabolites drive vasodilatation here while vessels to the gut "
                "and skin constrict. Blood flow can rise more than tenfold."
            ),
        ),
        TourKeyframe(
            title="Clinical takeaway",
            layers={"muscles": 1.0, "skeleton": 0.35, "skin": 0.15},
            view="isometric",
            focus_layer="muscles",
            narration=(
                "Remember this one fact. After a nerve injury, muscle loses its tone within weeks "
                "and fibroses within months, which is why rehabilitation must start long before "
                "sensation returns."
            ),
        ),
    ],
)


# ---------------------------------------------------------------------------
# 4. Articular — joints, cartilage, tendons
# ---------------------------------------------------------------------------
ARTICULAR_OVERVIEW = Tour(
    id="articular_overview",
    title="Joints & Cartilage",
    subtitle="The surfaces that let bone move on bone",
    system="Articular",
    keyframes=[
        TourKeyframe(
            title="Where bones meet",
            view="right",
            layers={"skin": 0.04, "muscles": 0.10, "skeleton": 0.55, "cartilage": 1.0},
            focus_layer="cartilage",
            narration=(
                "Every movement you make happens at a joint. Synovial joints share a common "
                "design: a fibrous capsule, a synovial membrane secreting lubricating fluid, "
                "and articular cartilage capping each bone end so that surfaces glide rather "
                "than grind."
            ),
        ),
        TourKeyframe(
            title="Tendon and load transfer",
            view="isometric",
            layers={"cartilage": 0.35, "tendons": 1.0, "muscles": 0.50, "skeleton": 0.45},
            focus_layer="tendons",
            narration=(
                "Tendons cross the joint to move it, but they also stabilise it. Because a "
                "tendon concentrates muscle force onto a small insertion, the strain there can "
                "exceed anything the muscle belly experiences. That is the mechanical reason "
                "tendinopathy clusters at the insertion, not the middle."
            ),
        ),
        TourKeyframe(
            title="Clinical takeaway",
            view="anterior",
            layers={"skeleton": 0.70, "cartilage": 1.0, "muscles": 0.15},
            focus_layer="cartilage",
            narration=(
                "One anchor for practice. Cartilage has no vessels and almost no cells, so it "
                "cannot mount an inflammatory repair. Osteoarthritis is therefore not simply "
                "wear from age, it is a failure of a tissue that was never built to heal. "
                "Protecting the joint early matters more than treating it late."
            ),
        ),
    ],
)


# ---------------------------------------------------------------------------
# 5. Arterial
# ---------------------------------------------------------------------------
ARTERIAL_OVERVIEW = Tour(
    id="arterial_overview",
    title="Arterial System",
    subtitle="The high pressure delivery network",
    system="Arterial",
    keyframes=[
        TourKeyframe(
            title="The pressure circuit",
            view="anterior",
            layers={"skin": 0.04, "muscles": 0.08, "skeleton": 0.20,
                    "heart": 0.60, "arteries": 1.0},
            focus_layer="arteries",
            narration=(
                "This is the arterial tree. Blood leaves the left ventricle at around one "
                "hundred and twenty millimetres of mercury, and every artery in the body is "
                "built to tolerate that pressure. Thick tunica media, elastic recoil, and a "
                "smooth endothelial lining are the three design features that make it possible."
            ),
        ),
        TourKeyframe(
            title="Aorta to capillary",
            view="isometric",
            layers={"arteries": 1.0, "veins": 0.25, "heart": 1.0},
            focus_layer="arteries",
            narration=(
                "Follow the calibre change. The aorta is wider than a garden hose, and by the "
                "time the vessel reaches a capillary it is narrower than a red blood cell. "
                "Cross sectional area rises enormously along that path, which is why flow slows "
                "to a crawl exactly where exchange has to happen."
            ),
        ),
        TourKeyframe(
            title="Clinical takeaway",
            view="right",
            layers={"arteries": 1.0, "heart": 0.50, "veins": 0.30},
            focus_layer="arteries",
            narration=(
                "One clinical anchor. Atherosclerosis begins where flow is disturbed and shear "
                "stress is abnormal, which is why plaques favour bifurcations rather than "
                "straight segments. That is also why the pulse is examined where an artery "
                "passes over bone, not where it lies in soft tissue."
            ),
        ),
    ],
)


# ---------------------------------------------------------------------------
# 6. Venous
# ---------------------------------------------------------------------------
VENOUS_OVERVIEW = Tour(
    id="venous_overview",
    title="Venous System",
    subtitle="Return flow against gravity",
    system="Venous",
    keyframes=[
        TourKeyframe(
            title="The low pressure return",
            view="anterior",
            layers={"skin": 0.04, "muscles": 0.10, "arteries": 0.30, "veins": 1.0},
            focus_layer="veins",
            narration=(
                "The venous system carries the same volume as the arterial tree, but at a "
                "fraction of the pressure. Vein walls are thinner, their lumens larger, and "
                "they hold roughly two thirds of the body's total blood volume at any moment. "
                "This is a capacitance system, not a delivery one."
            ),
        ),
        TourKeyframe(
            title="Valves and the muscle pump",
            view="right",
            layers={"veins": 1.0, "muscles": 0.55, "skeleton": 0.30},
            focus_layer="veins",
            narration=(
                "Because pressure is low, return depends on help. One way valves segment the "
                "vein, and every contracting muscle squeezes the segments between them. Blood "
                "can only move upward. Stop moving, and that pump stops with you, which is "
                "exactly how blood pools during prolonged immobility."
            ),
        ),
        TourKeyframe(
            title="Clinical takeaway",
            view="isometric",
            layers={"veins": 1.0, "arteries": 0.25, "muscles": 0.35},
            focus_layer="veins",
            narration=(
                "One clinical anchor. Deep vein thrombosis forms where all three of Virchow's "
                "conditions meet: sluggish flow, endothelial injury and hypercoagulability. "
                "That is why the risk rises after surgery and long flights, and why early "
                "mobilisation is not comfort care, it is prophylaxis."
            ),
        ),
    ],
)


# ---------------------------------------------------------------------------
# 7. Nervous
# ---------------------------------------------------------------------------
NERVOUS_OVERVIEW = Tour(
    id="nervous_overview",
    title="Nervous System",
    subtitle="Central command and the peripheral wiring",
    system="Nervous",
    keyframes=[
        TourKeyframe(
            title="Central command",
            view="superior",
            layers={"skin": 0.03, "muscles": 0.05, "skeleton": 0.15,
                    "brain": 1.0, "nerves": 0.35},
            focus_layer="brain",
            narration=(
                "Everything the body senses and every movement it makes passes through here. "
                "The brain holds roughly eighty six billion neurons, and although it is only "
                "two percent of body weight it consumes about twenty percent of your oxygen "
                "and glucose at rest."
            ),
        ),
        TourKeyframe(
            title="The peripheral wiring",
            view="right",
            layers={"brain": 0.45, "nerves": 1.0, "skeleton": 0.25},
            focus_layer="nerves",
            narration=(
                "Now follow the peripheral nerves outward. Each one carries motor fibres "
                "running to muscle and sensory fibres returning from skin and joint. The "
                "fibres are mixed in the same bundle, which is why a single nerve injury so "
                "often produces both weakness and altered sensation together."
            ),
        ),
        TourKeyframe(
            title="Clinical takeaway",
            view="anterior",
            layers={"nerves": 1.0, "brain": 0.60, "muscles": 0.15},
            focus_layer="nerves",
            narration=(
                "One clinical anchor. Sensory loss respects a dermatome, a strip of skin "
                "supplied by one spinal root, while weakness respects a myotome. Mapping the "
                "pattern tells you whether the lesion sits at the root, the plexus or the "
                "peripheral nerve, and that distinction changes the entire management plan."
            ),
        ),
    ],
)


# ---------------------------------------------------------------------------
# 8. Lymphatic
# ---------------------------------------------------------------------------
LYMPHATIC_OVERVIEW = Tour(
    id="lymphatic_overview",
    title="Lymphatic System",
    subtitle="The second circulation and immune surveillance",
    system="Lymphatic",
    keyframes=[
        TourKeyframe(
            title="The second circulation",
            view="anterior",
            layers={"skin": 0.03, "muscles": 0.08, "skeleton": 0.15,
                    "arteries": 0.20, "veins": 0.20, "lymphatics": 1.0},
            focus_layer="lymphatics",
            narration=(
                "Running alongside the blood vessels is a second, quieter network. Lymphatic "
                "capillaries begin blindly in the tissues, collect the fluid that plasma "
                "filtration leaves behind, and return it to the venous circulation. Without "
                "it, the interstitium would flood within hours."
            ),
        ),
        TourKeyframe(
            title="Nodes and surveillance",
            view="right",
            layers={"lymphatics": 1.0, "muscles": 0.25, "skeleton": 0.20},
            focus_layer="lymphatics",
            narration=(
                "Lymph does not simply drain, it is inspected. Along the route sit lymph "
                "nodes, packed with lymphocytes and macrophages that sample whatever passes "
                "through. That is why a node becomes tender and enlarged during infection, "
                "and why the drainage territory of a tumour determines which nodes are staged."
            ),
        ),
        TourKeyframe(
            title="Clinical takeaway",
            view="isometric",
            layers={"lymphatics": 1.0, "veins": 0.30, "muscles": 0.20},
            focus_layer="lymphatics",
            narration=(
                "One clinical anchor. Lymph moves because of muscle contraction and vessel "
                "peristalsis, not a central pump. Remove nodes during surgery and that pathway "
                "is gone, which is why lymphoedema follows axillary dissection, and why the "
                "affected limb needs compression and movement, not rest."
            ),
        ),
    ],
)


# ---------------------------------------------------------------------------
# 9. Respiratory
# ---------------------------------------------------------------------------
RESPIRATORY_OVERVIEW = Tour(
    id="respiratory_overview",
    title="Respiratory System",
    subtitle="The gas exchange surface",
    system="Thoracic",
    keyframes=[
        TourKeyframe(
            title="The exchange surface",
            view="anterior",
            layers={"skin": 0.05, "muscles": 0.10, "skeleton": 0.30,
                    "heart": 0.35, "lungs": 1.0},
            focus_layer="lungs",
            narration=(
                "The chest wall dims to leave the lungs. Together they contain close to half "
                "a billion alveoli, and if their surface were unfolded it would cover most of "
                "a tennis court. That enormous area, packed into a space this small, is the "
                "whole trick of efficient gas exchange."
            ),
        ),
        TourKeyframe(
            title="Mechanics of breathing",
            view="right",
            layers={"lungs": 1.0, "skeleton": 0.65, "muscles": 0.40},
            focus_layer="lungs",
            narration=(
                "Watch the mechanics. The diaphragm descends and the ribs rise, enlarging the "
                "thoracic cavity. Because the pleural space holds the lung against the chest "
                "wall, the lung is dragged open with it. You do not suck air in, you make room "
                "and the pressure gradient does the rest."
            ),
        ),
        TourKeyframe(
            title="Clinical takeaway",
            view="isometric",
            layers={"lungs": 1.0, "heart": 0.45, "skeleton": 0.20},
            focus_layer="lungs",
            narration=(
                "One clinical anchor. Airways are narrowest where they enter the alveolus and "
                "widest in the trachea, so obstruction hits the small airways first. That is "
                "why early asthma and COPD are felt as breathlessness on exertion long before "
                "any change appears on a chest film."
            ),
        ),
    ],
)


# ---------------------------------------------------------------------------
# 10. Cardiovascular — heart & great vessels
# ---------------------------------------------------------------------------
CARDIORESPIRATORY_OVERVIEW = Tour(
    id="cardiorespiratory_overview",
    title="Heart & Great Vessels",
    subtitle="The two-pump circuit",
    system="Cardiovascular",
    keyframes=[
        TourKeyframe(
            title="The thoracic cavity",
            layers={"skin": 0.08, "muscles": 0.12, "skeleton": 0.45,
                    "heart": 1.0, "lungs": 0.35},
            view="anterior",
            focus_layer="heart",
            narration=(
                "The chest wall dims. At the centre of this space sits the heart, a muscular "
                "pump roughly the size of a closed fist, cradled between two lungs that together "
                "hold nearly half a billion alveoli."
            ),
        ),
        TourKeyframe(
            title="Pulmonary circuit",
            layers={"heart": 1.0, "lungs": 1.0, "arteries": 0.35, "skeleton": 0.15},
            focus_layer="lungs",
            narration=(
                "This is the pulmonary circuit, the only place in the body where an artery "
                "carries deoxygenated blood. It is a short, low-pressure loop, which is why the "
                "right ventricle wall is thinner than the left."
            ),
        ),
        TourKeyframe(
            title="Systemic circuit",
            layers={"heart": 1.0, "arteries": 1.0, "veins": 0.7, "lungs": 0.25},
            focus_layer="arteries",
            narration=(
                "From the left ventricle the systemic circuit begins, driving blood at roughly "
                "one hundred and twenty millimetres of mercury into every tissue. That is six "
                "times the pressure the right side generates, and it is the reason the left "
                "ventricle thickens in untreated hypertension."
            ),
        ),
        TourKeyframe(
            title="Clinical takeaway",
            layers={"heart": 1.0, "lungs": 0.5},
            focus_layer="heart",
            view="isometric",
            narration=(
                "One clinical anchor. When the left ventricle fails, pressure backs up into the "
                "pulmonary circulation and fluid leaks into the alveoli. That is why the earliest "
                "symptom of left heart failure is breathlessness, not swollen ankles."
            ),
        ),
    ],
)


# ---------------------------------------------------------------------------
# 11. Digestive
# ---------------------------------------------------------------------------
DIGESTIVE_OVERVIEW = Tour(
    id="digestive_overview",
    title="Digestive System",
    subtitle="Absorption, detoxification and metabolism",
    system="Abdominal",
    keyframes=[
        TourKeyframe(
            title="The abdominal organs",
            view="anterior",
            layers={"skin": 0.05, "muscles": 0.10, "skeleton": 0.25,
                    "digestive": 1.0, "liver": 0.85},
            focus_layer="digestive",
            narration=(
                "The abdominal wall fades to reveal the gut. The digestive tract is a single "
                "continuous tube roughly nine metres long, from oesophagus to anus, and its "
                "job is to reduce food to molecules small enough to cross a membrane. Nothing "
                "is absorbed unless it is first broken down."
            ),
        ),
        TourKeyframe(
            title="The liver's dual supply",
            view="right",
            layers={"liver": 1.0, "digestive": 0.70, "arteries": 0.35,
                    "veins": 0.55, "skeleton": 0.15},
            focus_layer="liver",
            narration=(
                "Now the liver, and notice something unusual: it has two blood supplies. The "
                "hepatic artery brings oxygen, but the portal vein brings everything absorbed "
                "from the gut. Every nutrient and every drug you swallow passes through the "
                "liver before reaching the rest of the body."
            ),
        ),
        TourKeyframe(
            title="Clinical takeaway",
            view="isometric",
            layers={"liver": 1.0, "digestive": 0.55, "veins": 0.40},
            focus_layer="liver",
            narration=(
                "One clinical anchor. That first pass through the liver is why oral drugs are "
                "given in far larger doses than intravenous ones, and why a drug that is "
                "extensively metabolised may never reach useful blood levels by mouth. "
                "First pass metabolism is a dosing problem, not a side effect."
            ),
        ),
    ],
)


# ---------------------------------------------------------------------------
# 12. Urinary
# ---------------------------------------------------------------------------
URINARY_OVERVIEW = Tour(
    id="urinary_overview",
    title="Urinary System",
    subtitle="Filtration, balance and blood pressure",
    system="Abdominal",
    keyframes=[
        TourKeyframe(
            title="The retroperitoneal position",
            view="posterior",
            layers={"skin": 0.05, "muscles": 0.12, "skeleton": 0.45, "kidneys": 1.0},
            focus_layer="kidneys",
            narration=(
                "Viewed from behind, the kidneys sit high in the abdomen, protected partly by "
                "the lower ribs. They lie retroperitoneal, behind the peritoneal cavity, which "
                "is why renal pain is typically felt in the flank and often radiates forward "
                "toward the groin rather than to the centre of the abdomen."
            ),
        ),
        TourKeyframe(
            title="Filters and regulators",
            view="right",
            layers={"kidneys": 1.0, "arteries": 0.55, "veins": 0.45, "skeleton": 0.20},
            focus_layer="kidneys",
            narration=(
                "Each kidney holds about a million nephrons, and together they filter roughly "
                "one hundred and eighty litres of plasma a day while excreting only one or two "
                "litres of urine. The remainder is reabsorbed. They also regulate blood "
                "pressure, red cell production and acid base balance."
            ),
        ),
        TourKeyframe(
            title="Clinical takeaway",
            view="isometric",
            layers={"kidneys": 1.0, "arteries": 0.50, "veins": 0.40},
            focus_layer="kidneys",
            narration=(
                "One clinical anchor. Because the kidney is a filter fed by a high pressure "
                "arterial bed, it is unusually vulnerable to both hypertension and hypoperfusion. "
                "That is why renal function falls first in shock, and why the kidney is one of "
                "the earliest organs damaged by poorly controlled blood pressure."
            ),
        ),
    ],
)


# ---------------------------------------------------------------------------
# Organ-level tours (selected when the user picks a single structure)
#
# System tours cover a whole system; these are for when a clinician points at
# one organ and asks "talk me through this". Text is deliberately distinct from
# the system scripts — no paragraph is shared (asserted by verify_systems.py).
# ---------------------------------------------------------------------------
LIVER_OVERVIEW = Tour(
    id="liver_overview",
    title="Liver",
    subtitle="Metabolism, detoxification and first-pass handling",
    system="Organ",
    keyframes=[
        TourKeyframe(
            title="The largest solid organ",
            view="right",
            layers={"skin": 0.05, "muscles": 0.10, "skeleton": 0.18,
                    "liver": 1.0, "digestive": 0.20},
            focus_layer="liver",
            narration=(
                "This is the liver, the largest solid organ in the body at around one "
                "and a half kilograms. It sits in the right upper quadrant, tucked "
                "beneath the diaphragm and the lower ribs, which is why it is so well "
                "protected and why injury to it bleeds so heavily."
            ),
        ),
        TourKeyframe(
            title="Portal circulation",
            view="isometric",
            layers={"liver": 1.0, "veins": 0.65, "arteries": 0.35, "digestive": 0.45},
            focus_layer="liver",
            narration=(
                "Notice the liver's unusual dual supply. The hepatic artery brings "
                "oxygenated blood, while the portal vein carries everything absorbed "
                "from the gut. Every nutrient, toxin and swallowed drug therefore passes "
                "through here before it reaches the rest of the body."
            ),
        ),
        TourKeyframe(
            title="Clinical takeaway",
            view="anterior",
            layers={"liver": 1.0, "digestive": 0.25, "skeleton": 0.10},
            focus_layer="liver",
            narration=(
                "One clinical anchor for the liver. Because scar tissue replaces working "
                "hepatocytes in cirrhosis, the organ loses both synthetic and detoxifying "
                "capacity. That single process explains the bruising, the jaundice, the "
                "drug toxicity and the bleeding varices that present together."
            ),
        ),
    ],
)


HEART_OVERVIEW = Tour(
    id="heart_overview",
    title="Heart",
    subtitle="Chambers, valves and coronary supply",
    system="Organ",
    keyframes=[
        TourKeyframe(
            title="Four chambers",
            view="anterior",
            layers={"skin": 0.05, "muscles": 0.10, "skeleton": 0.40,
                    "lungs": 0.25, "heart": 1.0},
            focus_layer="heart",
            narration=(
                "Here is the heart with the chest wall removed. Two atria receive, two "
                "ventricles eject. The wall of the left ventricle is roughly three times "
                "thicker than the right, because it must generate the pressure to push "
                "blood through the entire body rather than just the lungs."
            ),
        ),
        TourKeyframe(
            title="Coronary supply",
            view="superior",
            layers={"heart": 1.0, "arteries": 0.9, "lungs": 0.10},
            focus_layer="heart",
            narration=(
                "The heart cannot use the blood inside it. Its muscle is fed by the "
                "coronary arteries, which branch from the aorta immediately above the "
                "aortic valve and fill mainly during diastole. That timing is why a fast "
                "heart rate, which shortens diastole, can starve the muscle it depends on."
            ),
        ),
        TourKeyframe(
            title="Clinical takeaway",
            view="isometric",
            layers={"heart": 1.0, "arteries": 0.75, "veins": 0.35},
            focus_layer="heart",
            narration=(
                "One clinical anchor. Coronary territories are predictable, so the "
                "pattern on an ECG tells you which vessel has occluded before any "
                "imaging is done. Anterior changes point to the left anterior "
                "descending, inferior changes to the right coronary artery."
            ),
        ),
    ],
)


LUNGS_OVERVIEW = Tour(
    id="lungs_overview",
    title="Lungs",
    subtitle="Alveolar surface and the mechanics of breathing",
    system="Organ",
    keyframes=[
        TourKeyframe(
            title="The exchange surface",
            view="anterior",
            layers={"skin": 0.05, "muscles": 0.10, "skeleton": 0.35,
                    "heart": 0.25, "lungs": 1.0},
            focus_layer="lungs",
            narration=(
                "These are the lungs, and their design is almost entirely about surface "
                "area. Half a billion alveoli give a membrane roughly the size of a "
                "tennis court, packed into a space you can cover with two hands. Gas "
                "exchange is simply diffusion across that membrane."
            ),
        ),
        TourKeyframe(
            title="Inside the airway",
            view="right",
            layers={"lungs": 1.0, "heart": 0.20, "skeleton": 0.30},
            focus_layer="lungs",
            narration=(
                "Follow the airway inward: trachea, bronchi, bronchioles, then alveoli. "
                "Total cross sectional area rises at every division, but the individual "
                "tubes get narrower. The narrowest point is the small airway — which is "
                "why that is where obstruction begins."
            ),
        ),
        TourKeyframe(
            title="Clinical takeaway",
            view="isometric",
            layers={"lungs": 1.0, "heart": 0.30, "skeleton": 0.15},
            focus_layer="lungs",
            narration=(
                "One clinical anchor. Because small airways are narrow and lack "
                "cartilage, they collapse on expiration before larger ones do. That is "
                "the mechanism behind the prolonged expiratory phase and the wheeze of "
                "asthma and COPD, and why air trapping shows up long before any X-ray change."
            ),
        ),
    ],
)


KIDNEYS_OVERVIEW = Tour(
    id="kidneys_overview",
    title="Kidneys",
    subtitle="Filtration, homeostasis and blood pressure",
    system="Organ",
    keyframes=[
        TourKeyframe(
            title="Retroperitoneal filters",
            view="posterior",
            layers={"skin": 0.05, "muscles": 0.12, "skeleton": 0.45, "kidneys": 1.0},
            focus_layer="kidneys",
            narration=(
                "Seen from behind, the kidneys sit high in the abdomen, partly shielded "
                "by the eleventh and twelfth ribs. They lie retroperitoneal, behind the "
                "abdominal cavity, which is why renal pain presents in the flank and "
                "frequently radiates forward toward the groin."
            ),
        ),
        TourKeyframe(
            title="One million nephrons",
            view="right",
            layers={"kidneys": 1.0, "arteries": 0.7, "veins": 0.55},
            focus_layer="kidneys",
            narration=(
                "Each kidney contains about a million nephrons. Together they filter "
                "around one hundred and eighty litres of plasma every day while "
                "excreting only one to two litres as urine — the rest is reclaimed. The "
                "kidney is a reabsorption organ first and a filter second."
            ),
        ),
        TourKeyframe(
            title="Clinical takeaway",
            view="isometric",
            layers={"kidneys": 1.0, "arteries": 0.6, "veins": 0.45},
            focus_layer="kidneys",
            narration=(
                "One clinical anchor. The kidney receives a fifth of cardiac output but "
                "has little tolerance for low pressure, so it is among the first organs "
                "to fail in shock. It is also central to blood pressure control, which "
                "is why renal disease and hypertension so often escalate together."
            ),
        ),
    ],
)


BRAIN_OVERVIEW = Tour(
    id="brain_overview",
    title="Brain",
    subtitle="Cortex, deep structures and vascular territory",
    system="Organ",
    keyframes=[
        TourKeyframe(
            title="The cortex",
            view="superior",
            layers={"skin": 0.02, "muscles": 0.04, "skeleton": 0.14, "brain": 1.0},
            focus_layer="brain",
            narration=(
                "The brain is two percent of body weight and consumes around twenty "
                "percent of your oxygen and glucose. The folded cortex gives it an "
                "enormous surface area inside a skull that cannot change size, and the "
                "folds also create the fixed landmark pattern clinicians use to localise lesions."
            ),
        ),
        TourKeyframe(
            title="Deep structures",
            view="right",
            layers={"brain": 1.0, "nerves": 0.30},
            focus_layer="brain",
            narration=(
                "Beneath the cortex sit the deep grey matter and the fluid-filled "
                "ventricles. Because the skull is a closed box, the brain, blood and "
                "cerebrospinal fluid compete for a fixed volume. Any increase in one "
                "must come at the expense of another."
            ),
        ),
        TourKeyframe(
            title="Clinical takeaway",
            view="isometric",
            layers={"brain": 1.0, "nerves": 0.45, "arteries": 0.40},
            focus_layer="brain",
            narration=(
                "One clinical anchor. That fixed volume is why a haematoma raises "
                "intracranial pressure and threatens perfusion even if the bleed itself "
                "is small. It is also why the brain tolerates hypotension so poorly: "
                "blood flow here is tightly controlled and has very little reserve."
            ),
        ),
    ],
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
#: System-level tours.
SYSTEM_TOURS: tuple = (
    INTEGUMENTARY_OVERVIEW,
    SKELETAL_OVERVIEW,
    MUSCULAR_OVERVIEW,
    ARTICULAR_OVERVIEW,
    ARTERIAL_OVERVIEW,
    VENOUS_OVERVIEW,
    NERVOUS_OVERVIEW,
    LYMPHATIC_OVERVIEW,
    RESPIRATORY_OVERVIEW,
    CARDIORESPIRATORY_OVERVIEW,
    DIGESTIVE_OVERVIEW,
    URINARY_OVERVIEW,
)

#: Organ-level tours, offered when the user points at a single structure.
ORGAN_TOURS: tuple = (
    LIVER_OVERVIEW,
    HEART_OVERVIEW,
    LUNGS_OVERVIEW,
    KIDNEYS_OVERVIEW,
    BRAIN_OVERVIEW,
)

BUILTIN_TOURS: Dict[str, Tour] = {
    tour.id: tour for tour in (*SYSTEM_TOURS, *ORGAN_TOURS)
}


@dataclass(frozen=True)
class BodySystem:
    """A selectable body system: its layers and its dedicated narrated tour."""

    label: str
    layers: tuple
    tour_id: str
    hint: str = ""


#: Every layer id in the registry, used by the "Full body" entry.
_ALL_LAYERS = (
    "skin", "fascia", "muscles", "tendons", "skeleton", "cartilage",
    "arteries", "veins", "nerves", "lymphatics", "brain", "lungs",
    "heart", "liver", "kidneys", "digestive",
)

#: Ordered list driving both the sidebar preset menu and the export selector.
#: Each entry with a ``tour_id`` gets its own narration script.
BODY_SYSTEMS: tuple = (
    BodySystem("All systems", _ALL_LAYERS, "", "Everything, at default opacity"),
    BodySystem("Integumentary", ("skin", "fascia"), "integumentary_overview",
               "Skin and superficial fascia"),
    BodySystem("Skeletal", ("skeleton", "cartilage"), "skeletal_overview",
               "Bones, joints and leverage"),
    BodySystem("Muscular", ("muscles", "tendons"), "muscular_overview",
               "Skeletal muscle and tendons"),
    BodySystem("Articular", ("cartilage", "tendons", "skeleton"), "articular_overview",
               "Joint surfaces and load transfer"),
    BodySystem("Arterial", ("arteries", "heart"), "arterial_overview",
               "High-pressure delivery network"),
    BodySystem("Venous", ("veins",), "venous_overview",
               "Low-pressure return flow"),
    BodySystem("Nervous", ("brain", "nerves"), "nervous_overview",
               "Central and peripheral nervous tissue"),
    BodySystem("Lymphatic", ("lymphatics",), "lymphatic_overview",
               "Immune surveillance and fluid return"),
    BodySystem("Thoracic", ("lungs", "skeleton"), "respiratory_overview",
               "Gas exchange surface and mechanics"),
    BodySystem("Cardiovascular", ("heart", "arteries", "veins"),
               "cardiorespiratory_overview", "Cardiac pump and great vessels"),
    BodySystem("Abdominal", ("liver", "digestive", "kidneys"), "digestive_overview",
               "Gut, liver and first-pass metabolism"),
    BodySystem("Urinary", ("kidneys",), "urinary_overview",
               "Filtration, balance, blood pressure"),
)

#: Sidebar system label -> tour id (empty for "All systems").
SYSTEM_TOUR_MAP: Dict[str, str] = {
    system.label: system.tour_id for system in BODY_SYSTEMS if system.tour_id
}


def tour_list() -> List[Tour]:
    """Every built-in tour: system tours first, then organ tours."""
    ordered: List[Tour] = []
    for system in BODY_SYSTEMS:
        tour = BUILTIN_TOURS.get(system.tour_id)
        if tour is not None and tour not in ordered:
            ordered.append(tour)
    for tour in ORGAN_TOURS:
        if tour not in ordered:
            ordered.append(tour)
    return ordered


#: Structure (layer) -> the tour that narrates it. Lets "click the liver, then
#: export" produce a liver video without the user touching a dropdown.
LAYER_TOUR_MAP: Dict[str, str] = {
    "skin": "integumentary_overview",
    "fascia": "integumentary_overview",
    "skeleton": "skeletal_overview",
    "cartilage": "articular_overview",
    "tendons": "articular_overview",
    "muscles": "muscular_overview",
    "arteries": "arterial_overview",
    "veins": "venous_overview",
    "nerves": "nervous_overview",
    "lymphatics": "lymphatic_overview",
    "brain": "brain_overview",
    "lungs": "lungs_overview",
    "heart": "heart_overview",
    "liver": "liver_overview",
    "kidneys": "kidneys_overview",
    "digestive": "digestive_overview",
}


def tour_for_layer(layer_id: str) -> Optional[Tour]:
    """Organ tour for a structure, falling back to its system tour."""
    return get_tour(LAYER_TOUR_MAP.get(layer_id, ""))


def subject_for_layer(layer_id: str) -> Optional[Tour]:
    """Best video subject for a selected structure."""
    return tour_for_layer(layer_id)


def video_subjects() -> List[tuple]:
    """Selectable export subjects as ``(group, label, tour_id)``.

    Groups are ``"Body system"`` and ``"Organ"``. Organ subjects exist because
    asking for "a video on the liver" should not require picking a system and
    hoping the liver is mentioned.
    """
    subjects: List[tuple] = []
    seen: set = set()
    for system in BODY_SYSTEMS:
        tour = BUILTIN_TOURS.get(system.tour_id)
        if tour is None or tour.id in seen:
            continue
        seen.add(tour.id)
        subjects.append(("Body system", f"{system.label} — {tour.title}", tour.id))
    for tour in ORGAN_TOURS:
        if tour.id in seen:
            continue
        seen.add(tour.id)
        subjects.append(("Organ", f"{tour.title}  ·  {tour.subtitle}", tour.id))
    return subjects


#: Tours generated at runtime (muscle actions, disease simulations, translated
#: copies). They behave exactly like built-in tours for playback and export.
DYNAMIC_TOURS: Dict[str, Tour] = {}


def register_tour(tour: Tour) -> Tour:
    """Make a generated tour addressable by id (replaces an older version)."""
    DYNAMIC_TOURS[tour.id] = tour
    return tour


def get_tour(tour_id: str) -> Optional[Tour]:
    """Look up a tour by id. Returns ``None`` for an unknown id."""
    return BUILTIN_TOURS.get(tour_id) or DYNAMIC_TOURS.get(tour_id)


def tour_for_system(label: str) -> Optional[Tour]:
    """The dedicated tour for a sidebar body-system label, if one exists."""
    return get_tour(SYSTEM_TOUR_MAP.get(label, ""))


def system_for_tour(tour_id: str) -> str:
    """Inverse of :func:`tour_for_system` — used to sync the export selector."""
    for label, mapped in SYSTEM_TOUR_MAP.items():
        if mapped == tour_id:
            return label
    return ""


def system_presets() -> Dict[str, List[str]]:
    """Quick layer-visibility presets for the sidebar (label -> layer ids)."""
    return {system.label: list(system.layers) for system in BODY_SYSTEMS}


def system_hint(label: str) -> str:
    for system in BODY_SYSTEMS:
        if system.label == label:
            return system.hint
    return ""
