"""
Narrated muscle-action tours, generated from the muscle knowledge base.

``muscle_tour(viewport, rig, "biceps brachii", "right")`` returns a normal
:class:`~app.audio.scripts.Tour` — so it plays live with voiceover and exports
to MP4 like any authored tour — made of:

1. *Where it is*: the muscle isolated against the skeleton, origin and insertion.
2. *One beat per action*: the joint moves through that motion while the muscle
   glows (agonist), synergists are amber and antagonists blue; the narration
   states the motion and its normal range.
3. *Nerve supply and clinical relevance.*

Every sentence comes from the knowledge base in ``muscles.py`` (Gray's / Moore's),
so the narration and the animation can never disagree.
"""
from __future__ import annotations

from typing import List, Optional

from app.anatomy.muscle_actions import ContractInPlaceEffect, MuscleActionEffect
from app.anatomy.muscles import (MOTIONS, MUSCLE_BY_KEY, Muscle, agonists_for,
                                 antagonists_for, muscle_for_structure)
from app.audio.scripts import Tour, TourKeyframe, register_tour
from app.core.effects import CompositeEffect, IsolateStructuresEffect

#: Layer opacities for muscle beats: skin and viscera out of the way, bones as
#: landmarks, other muscles faint so the moving muscle reads clearly.
MUSCLE_LAYERS = {"skin": 0.0, "fascia": 0.0, "muscles": 0.22, "tendons": 0.6, "skeleton": 1.0,
                 "cartilage": 0.8, "arteries": 0.08, "veins": 0.05, "nerves": 0.12, "lymphatics": 0.0,
                 "brain": 0.05, "lungs": 0.06, "heart": 0.06, "liver": 0.05, "kidneys": 0.05,
                 "digestive": 0.04}

LIMB_BONES = {"upper": ("humerus", "radius", "ulna", "carpal", "metacarpal", "scapula", "clavicle"),
              "lower": ("hip bone", "femur", "tibia", "fibula", "patella", "talus", "calcaneus", "tarsal")}


def _names(viewport, muscle: Muscle, side: str) -> List[str]:
    state = viewport.layers.get("muscles")
    if state is None:
        return []
    out = []
    for name in state.structures:
        low = name.lower()
        hit = muscle_for_structure(name)
        if hit is not None and hit.key == muscle.key and (not side or side in low):
            out.append(name)
    return out


def _limb_frame(viewport, side: str, limb: str) -> List[str]:
    state = viewport.layers.get("skeleton")
    if state is None:
        return []
    return [n for n in state.structures
            if side in n.lower() and any(b in n.lower() for b in LIMB_BONES.get(limb, ()))]


def _envelope(viewport, rig, motion, side: str, names: List[str]):
    """Bounds covering the limb at rest AND at the motion's start/end angles,
    so the camera never loses the hand or foot mid-movement."""
    limb_key = f"{side}_{motion.limb}"
    boxes = []
    for angle in (0.0, motion.start, motion.end):
        rig.pose(limb_key, {motion.dof: angle})
        b = viewport.structure_bounds(names)
        if b is not None:
            boxes.append(b)
    rig.reset()
    if not boxes:
        return None
    lo = [min(b[2 * k] for b in boxes) for k in range(3)]
    hi = [max(b[2 * k + 1] for b in boxes) for k in range(3)]
    pad = 0.04
    return (lo[0] - pad, hi[0] + pad, lo[1] - pad, hi[1] + pad, lo[2] - pad, hi[2] + pad)


def _side_view(side: str) -> str:
    # VIEW_PRESETS "left" puts the camera on -X, i.e. it looks at the patient's right side.
    return "left" if side == "right" else "right"


def _human_list(items: List[str]) -> str:
    items = [i for i in items if i]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:] if text else text


def _attachments(muscle: Muscle) -> str:
    """Origin/insertion sentence; multi-head origins are read as written."""
    if ":" in muscle.origin:
        origin = f"It has more than one origin. {muscle.origin}."
    else:
        origin = f"It arises from the {_lower_first(muscle.origin)}."
    insertion = muscle.insertion if ":" in muscle.insertion else f"the {_lower_first(muscle.insertion)}"
    return f"{origin} It inserts on {insertion}."


def muscle_tour(viewport, rig, muscle_key: str, side: str = "right") -> Optional[Tour]:
    muscle = MUSCLE_BY_KEY.get(muscle_key)
    if muscle is None:
        return None
    names = _names(viewport, muscle, side) or _names(viewport, muscle, "")
    if not names:
        return None
    side_word = side if side in names[0].lower() else ""
    title = f"{muscle.name}" + (f" ({side_word})" if side_word else "")

    keyframes: List[TourKeyframe] = [TourKeyframe(
        title=f"Locating the {muscle.name.lower()}",
        view="anterior" if muscle.region not in ("Back", "Hip") else "posterior",
        layers=dict(MUSCLE_LAYERS),
        focus_structures=names,
        effect=IsolateStructuresEffect(names, ghost=0.18, context_layers=("skeleton", "cartilage")),
        narration=(f"This is the {muscle.name.lower()}. " + _attachments(muscle)),
    )]

    motions = [MOTIONS[m] for m in muscle.actions if m in MOTIONS]
    can_move = rig is not None and getattr(rig, "available", False)
    if motions and can_move:
        for motion in motions[:3]:
            limb_bones = _limb_frame(viewport, side_word or "right", motion.limb)
            others = [m.name.lower() for m in agonists_for(motion.id) if m.key != muscle.key][:3]
            antagonists = [m.name.lower() for m in antagonists_for(motion.id)][:3]
            sentence = (f"{motion.label}. As the {muscle.name.lower()} contracts, shown glowing, "
                        f"the joint moves through {motion.label.lower()}. Normal range is {motion.normal_range}.")
            if others:
                sentence += f" It works with the {_human_list(others)}, shown in amber."
            if antagonists:
                sentence += (f" On the opposite side of the joint the {_human_list(antagonists)}, "
                             f"shown in blue, must lengthen to allow the movement.")
            view = _side_view(side_word or "right") if motion.plane == "sagittal" else "anterior"
            keyframes.append(TourKeyframe(
                title=motion.label,
                view=view,
                layers=dict(MUSCLE_LAYERS),
                focus_structures=limb_bones + names,
                focus_bounds=_envelope(viewport, rig, motion, side_word or "right", limb_bones + names),
                effect=MuscleActionEffect(rig, motion.id, side_word or "right", focus_muscle=muscle.key),
                narration=sentence,
            ))
    else:
        keyframes.append(TourKeyframe(
            title="In action",
            view="anterior",
            layers=dict(MUSCLE_LAYERS),
            focus_structures=names,
            effect=CompositeEffect([IsolateStructuresEffect(names, ghost=0.18,
                                                            context_layers=("skeleton", "cartilage")),
                                    ContractInPlaceEffect(names)]),
            narration=f"{muscle.action_text} Watch the belly shorten and thicken as it contracts.",
        ))

    closing = f"Nerve supply: {muscle.innervation}."
    if muscle.blood_supply:
        closing += f" Blood supply: {muscle.blood_supply}."
    if muscle.clinical:
        closing += f" Clinically, {_lower_first(muscle.clinical)}"
    keyframes.append(TourKeyframe(
        title="Nerve supply and clinical relevance",
        view="isometric",
        layers=dict(MUSCLE_LAYERS, nerves=0.9),
        focus_structures=names,
        effect=IsolateStructuresEffect(names, ghost=0.18, context_layers=("skeleton", "cartilage", "nerves")),
        narration=closing,
    ))

    tour = Tour(id=f"muscle_{muscle.key.replace(' ', '_')}_{side_word or 'both'}",
                title=title, subtitle="Muscle action — origin, insertion, movement",
                system="Muscular", keyframes=keyframes)
    return register_tour(tour)


def motion_tour(viewport, rig, motion_id: str, side: str = "right") -> Optional[Tour]:
    """Tour for a single joint motion: every agonist and antagonist together."""
    motion = MOTIONS.get(motion_id)
    if motion is None or rig is None or not rig.available:
        return None
    agon = agonists_for(motion.id)
    antag = antagonists_for(motion.id)
    bones = _limb_frame(viewport, side, motion.limb)
    narration = (f"{motion.label}, normal range {motion.normal_range}. The prime movers are the "
                 f"{_human_list([m.name.lower() for m in agon])}, glowing as they shorten.")
    if antag:
        narration += f" The antagonists, the {_human_list([m.name.lower() for m in antag])}, lengthen in blue."
    tour = Tour(id=f"motion_{motion.id.replace('.', '_')}_{side}", title=f"{motion.label} ({side})",
                subtitle="Joint motion — agonists and antagonists", system="Muscular",
                keyframes=[TourKeyframe(
                    title=motion.label,
                    view=_side_view(side) if motion.plane == "sagittal" else "anterior",
                    layers=dict(MUSCLE_LAYERS, muscles=0.35),
                    focus_structures=bones,
                    focus_bounds=_envelope(viewport, rig, motion, side, bones),
                    effect=MuscleActionEffect(rig, motion.id, side),
                    narration=narration)])
    return register_tour(tour)
