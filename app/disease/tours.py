"""
Narrated disease-progression tours.

``disease_tour(viewport, scene, "ckd")`` builds a normal tour: an overview of
the healthy organ, then one beat per stage in which the anatomy morphs from the
previous stage to this one while the narration gives the stage's defining
criteria and what changes, and a closing beat with the sources and the
educational disclaimer. It plays live and exports to video like any tour.
"""
from __future__ import annotations

from typing import List, Optional

from app.audio.scripts import Tour, TourKeyframe, register_tour
from app.core.effects import CompositeEffect, IsolateStructuresEffect
from app.disease.catalog import CONDITION_BY_ID, DISCLAIMER, Condition
from app.disease.effects import DiseaseScene, StageTransitionEffect

ALL_LAYERS = ("skin", "fascia", "muscles", "tendons", "skeleton", "cartilage", "arteries", "veins",
              "nerves", "lymphatics", "brain", "lungs", "heart", "liver", "kidneys", "digestive")


def _layers(condition: Condition) -> dict:
    out = {layer: 0.0 for layer in ALL_LAYERS}
    for layer in condition.context_layers:
        out[layer] = 0.25
    for layer in condition.organ_layers:
        out[layer] = condition.organ_opacity
    return out


def involved(scene: DiseaseScene, condition: Condition) -> List[str]:
    """Focus structures plus every structure any stage changes."""
    names = list(scene.matching(condition.focus))
    for stage in condition.stages:
        for effect in stage.effects:
            names += [n for n in scene.matching(effect["match"]) if n not in names]
    return names


def _effect(scene: DiseaseScene, condition: Condition, a: int, b: int, reset: bool = False):
    stage = StageTransitionEffect(scene, condition, a, b, reset_on_end=reset)
    if not condition.isolate:
        return stage
    return CompositeEffect([IsolateStructuresEffect(involved(scene, condition), ghost=0.10,
                                                    context_layers=()), stage])


def _focus(scene: DiseaseScene, condition: Condition) -> List[str]:
    return scene.matching(condition.focus)


def disease_tour(viewport, scene: DiseaseScene, condition_id: str) -> Optional[Tour]:
    condition = CONDITION_BY_ID.get(condition_id)
    if condition is None:
        return None
    focus = _focus(scene, condition)
    layers = _layers(condition)
    keyframes: List[TourKeyframe] = []
    first = condition.stages[0]
    keyframes.append(TourKeyframe(
        title=condition.name,
        view=condition.view,
        layers=layers,
        focus_structures=focus,
        effect=_effect(scene, condition, 0, 0),
        narration=f"{condition.summary} It is staged using {condition.staging}. {first.narration}",
    ))
    for index in range(1, len(condition.stages)):
        stage = condition.stages[index]
        keyframes.append(TourKeyframe(
            title=f"{stage.label}: {stage.title}",
            view=condition.view,
            layers=layers,
            focus_structures=focus,
            effect=_effect(scene, condition, index - 1, index),
            narration=f"{stage.label}. {stage.criteria}. {stage.narration}",
        ))
    keyframes.append(TourKeyframe(
        title="Sources and note",
        view="isometric",
        layers=layers,
        focus_structures=focus,
        effect=_effect(scene, condition, len(condition.stages) - 1, len(condition.stages) - 1, reset=True),
        narration=f"This simulation follows {condition.sources[0]}. {DISCLAIMER}",
    ))
    tour = Tour(id=f"disease_{condition.id}", title=condition.name,
                subtitle=f"Disease progression — {condition.staging}", system=condition.system,
                keyframes=keyframes)
    return register_tour(tour)
