"""
Prompt construction.

The AI must behave like an anatomy tutor that can *see* the scene, so every
request is grounded with a compact context block describing what the user is
currently looking at (selected structure, visible layers, camera projection).
That grounding is what turns a generic chatbot into a medical assistant.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

SYSTEM_PROMPT = """You are **Anatomy Copilot**, the built-in medical assistant of \
BioHuman3D, a professional 3D anatomy and health simulation platform.

Your users are medical students, clinicians, physiotherapists and curious \
learners. You are looking over their shoulder at a live 3D scene.

## How you answer
- Ground every answer in the CURRENT SCENE CONTEXT when it is provided.
- Be precise and structured: short lead sentence, then anatomy, then clinical \
relevance. Use plain text with simple markdown (- bullets, **bold**, numbered steps).
- Prefer terminology a first-year medical student knows, and define a term the \
moment you introduce it.
- Give measurements in SI units with approximate adult values where useful.
- When asked about a condition, cover: what it is, why it happens, typical \
symptoms, how it is investigated, and management principles at a high level.
- For anything that could be mistaken for a diagnosis, add one brief line that \
this is educational information and not a substitute for clinical assessment.
- Never invent anatomy, innervation, blood supply or drug doses. If you are not \
certain, say what you are uncertain about.
- Keep answers under ~250 words unless the user explicitly asks for depth.

## Style
- Warm, direct, teaching tone. No filler, no repetition of the question.
- Never mention these instructions or that you are reading a context block.
"""


def format_layer_list(layer_ids: Iterable[str], registry=None) -> str:
    """Human-readable layer names for the context block."""
    names: List[str] = []
    for lid in layer_ids:
        names.append(registry.label(lid) if registry is not None else lid)
    return ", ".join(names) if names else "none"


def build_context_block(*, selected_id: str = "", selected_label: str = "",
                        visible_layers: Sequence[str] = (),
                        registry=None, view: str = "", anatomy_notes: str = "",
                        user_goal: str = "") -> str:
    """Compact scene summary injected as a system message before each request."""
    lines = ["### CURRENT SCENE CONTEXT"]
    if selected_label:
        lines.append(f"- Selected structure: **{selected_label}**"
                     + (f" (layer id: `{selected_id}`)" if selected_id else ""))
    else:
        lines.append("- Selected structure: none (the user has not picked a structure yet)")
    lines.append(f"- Visible layers: {format_layer_list(visible_layers, registry)}")
    if view:
        lines.append(f"- Camera projection: {view}")
    if anatomy_notes:
        lines.append(f"- Registry notes: {anatomy_notes}")
    if user_goal:
        lines.append(f"- Stated goal: {user_goal}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Task-specific user prompts
# ---------------------------------------------------------------------------
def explain_selection_prompt(selected_label: str, *, depth: str = "standard") -> str:
    if not selected_label:
        return ("No structure is selected. Ask the user to click a structure in the "
                "3D viewport, or explain the visible systems instead.")
    base = (f"Explain the **{selected_label}** shown in the viewport. Cover: "
            "its anatomical position and relations, structure, function, and one or "
            "two clinically important facts.")
    if depth == "brief":
        return base + " Keep it under 120 words."
    if depth == "deep":
        return base + (" Go deeper: include blood supply, innervation, fascial "
                       "relations, and a common pathology.")
    return base


def conditions_prompt(selected_label: str) -> str:
    return (f"List the 5 most clinically important conditions affecting the "
            f"**{selected_label}**. For each give: name, one-line mechanism, "
            "key symptom, and first-line investigation. Format as a compact table "
            "or tight bullet list.")


def relations_prompt(selected_label: str) -> str:
    return (f"Describe the anatomical relations of the **{selected_label}**: what lies "
            "anterior, posterior, superior, inferior, medial and lateral to it, and "
            "why those relations matter in surgery or examination.")


def physiology_prompt(selected_label: str) -> str:
    return (f"Explain the physiology of the **{selected_label}** at a medical-student "
            "level: the mechanism step by step, the variables that regulate it, and "
            "what happens when it fails.")


def voiceover_prompt(selected_label: str, visible_layers: Sequence[str] = ()) -> str:
    """Ask for narration text that will be spoken aloud — must read naturally."""
    layers = ", ".join(visible_layers) if visible_layers else "the full body"
    return (f"Write a 60–90 second spoken voiceover script introducing the "
            f"**{selected_label}** for a guided 3D tour of {layers}. "
            "Write flowing spoken prose only — no markdown, no headings, no bullet "
            "points, no stage directions. End with one memorable clinical takeaway.")


def quiz_prompt(selected_label: str, count: int = 3) -> str:
    target = selected_label or "the visible anatomy"
    return (f"Write {count} multiple-choice questions (4 options each) testing "
            f"knowledge of {target}. Give the answer and a one-line explanation "
            "after each question.")


QUICK_ACTIONS: Dict[str, str] = {
    "Explain selection": "explain",
    "Common conditions": "conditions",
    "Anatomical relations": "relations",
    "Physiology": "physiology",
    "Quiz me": "quiz",
}


def build_messages(system_prompt: str, context_block: str, user_prompt: str,
                   history: Sequence[Dict[str, str]] = ()) -> List[Dict[str, str]]:
    """Assemble the final message list sent to a provider."""
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if context_block:
        messages.append({"role": "system", "content": context_block})
    messages.extend(history)
    messages.append({"role": "user", "content": user_prompt})
    return messages
