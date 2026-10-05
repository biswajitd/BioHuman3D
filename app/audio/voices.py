"""
Voice catalogue — dialect × gender selection for narration.

Windows SAPI5 exposes a flat list of voices, each tagged with a language and
(usually) a gender. That is not enough for a clinical product: a user wants
"Indian English, female", not "the third voice in the list". This module turns
the flat list into a catalogue you can query, and — critically — tells the truth
when nothing matches instead of silently substituting a US voice.

Two providers:

* ``pyttsx3``  offline SAPI5 voices already installed on the machine.
* ``elevenlabs`` cloud voices, fetched live from the API when a key is present
  (labels carry accent and gender, so regional Indian accents are available
  without a language pack).

Nothing here invents a voice id: if a dialect/gender pair is not installed, the
catalogue reports it as missing and explains how to add it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import pyttsx3
    PYTTsx3_AVAILABLE = True
except Exception:  # pragma: no cover
    pyttsx3 = None  # type: ignore[assignment]
    PYTTsx3_AVAILABLE = False


# ---------------------------------------------------------------------------
# Dialects
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Dialect:
    """An English/Hindi dialect the narration can target."""

    code: str
    label: str
    aliases: Tuple[str, ...]      # substrings that identify it in a voice name
    note: str = ""


#: Ordered with Indian English first — the primary requirement.
DIALECTS: Tuple[Dialect, ...] = (
    Dialect("en-IN", "English (India)",
            ("en-in", "english (india)", "india", "heera", "ravi", "neerja",
             "prabhat", "kalpana", "vani", "kumar", "swara"),
            "Indian English · Windows voice pack"),
    Dialect("hi-IN", "Hindi (India)",
            ("hi-in", "hindi", "hemant", "madhur", "swara", "kalpana", "vani"),
            "Hindi narration, Indian pronunciation"),
    Dialect("en-GB", "English (United Kingdom)",
            ("en-gb", "united kingdom", "british", "hazel", "george", "susan",
             "libby", "sonia", "ryan", "oliver", "thomas")),
    Dialect("en-US", "English (United States)",
            ("en-us", "united states", "american", "zira", "david", "mark",
             "jenny", "aria", "guy", "michelle", "eric", "christopher", "amber")),
    Dialect("en-AU", "English (Australia)",
            ("en-au", "australia", "catherine", "william", "natasha", "liam",
             "clara", "james")),
    Dialect("en-CA", "English (Canada)",
            ("en-ca", "canada")),
    Dialect("en-NZ", "English (New Zealand)",
            ("en-nz", "new zealand")),
    Dialect("en-ZA", "English (South Africa)",
            ("en-za", "south africa")),
    Dialect("en-IE", "English (Ireland)",
            ("en-ie", "ireland", "connor", "emily")),
)

DIALECT_BY_CODE: Dict[str, Dialect] = {d.code: d for d in DIALECTS}

#: Sentinel meaning "any dialect".
ANY_DIALECT = "any"

GENDER_ANY = "any"


# ---------------------------------------------------------------------------
# Voice model
# ---------------------------------------------------------------------------
@dataclass
class VoiceOption:
    """One selectable narration voice."""

    id: str
    name: str                       # raw engine name
    dialect: str = ""               # dialect code, "" when unclassified
    gender: str = ""                # "female" | "male" | ""
    provider: str = "pyttsx3"

    @property
    def dialect_label(self) -> str:
        dialect = DIALECT_BY_CODE.get(self.dialect)
        return dialect.label if dialect else "Other language"

    @property
    def display(self) -> str:
        """Compact label: ``Heera — English (India) — Female``."""
        parts = [self._short_name(), self.dialect_label]
        if self.gender:
            parts.append(self.gender.capitalize())
        return " — ".join(parts)

    def __str__(self) -> str:
        return self.display

    def _short_name(self) -> str:
        """Strip the verbose engine prefix from a SAPI voice name."""
        name = self.name
        for prefix in ("Microsoft ", "Desktop - ", " - Desktop"):
            name = name.replace(prefix, "")
        # "Heera - English (India)" -> "Heera"
        for separator in (" - ", " (", "-"):
            if separator in name:
                name = name.split(separator)[0]
                break
        # "David Desktop" -> "David"
        for suffix in (" Desktop", " Online", " (Natural)"):
            name = name.replace(suffix, "")
        return name.strip() or self.name


#: Common Windows voice names → (dialect, gender). Used when the driver does not
#: report a language or gender. Extend as new voices appear.
KNOWN_VOICES: Dict[str, Tuple[str, str]] = {
    # Indian English
    "heera": ("en-IN", "female"),
    "ravi": ("en-IN", "male"),
    "neerja": ("en-IN", "female"),
    "prabhat": ("en-IN", "male"),
    "kalpana": ("en-IN", "female"),
    "vani": ("en-IN", "female"),
    # Hindi
    "hemant": ("hi-IN", "male"),
    "madhur": ("hi-IN", "male"),
    "swara": ("hi-IN", "female"),
    # UK
    "hazel": ("en-GB", "female"),
    "susan": ("en-GB", "female"),
    "libby": ("en-GB", "female"),
    "sonia": ("en-GB", "female"),
    "george": ("en-GB", "male"),
    "ryan": ("en-GB", "male"),
    "oliver": ("en-GB", "male"),
    "thomas": ("en-GB", "male"),
    # US
    "zira": ("en-US", "female"),
    "aria": ("en-US", "female"),
    "jenny": ("en-US", "female"),
    "michelle": ("en-US", "female"),
    "amber": ("en-US", "female"),
    "david": ("en-US", "male"),
    "mark": ("en-US", "male"),
    "guy": ("en-US", "male"),
    "eric": ("en-US", "male"),
    "christopher": ("en-US", "male"),
    # Australia
    "catherine": ("en-AU", "female"),
    "natasha": ("en-AU", "female"),
    "clara": ("en-AU", "female"),
    "william": ("en-AU", "male"),
    "james": ("en-AU", "male"),
    "liam": ("en-AU", "male"),
}


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def _decode_languages(raw) -> List[str]:
    """Normalise the ``languages`` attribute to lowercase strings."""
    out: List[str] = []
    if not raw:
        return out
    items = raw if isinstance(raw, (list, tuple)) else [raw]
    for item in items:
        if isinstance(item, bytes):
            text = item.decode("utf-8", "replace")
        else:
            text = str(item)
        # SAPI tags look like '\x05en-in' — keep the printable tail.
        text = "".join(ch for ch in text if ch.isprintable()).strip().lower()
        # Strip leading attribute bytes such as the '5' in '\x05en-in'.
        if len(text) > 2 and not text[:2].isalpha():
            text = text.lstrip("0123456789 ")
        if text:
            out.append(text)
    return out


def classify_dialect(name: str, languages: Sequence[str]) -> str:
    """Best-effort dialect code for a voice."""
    lowered = (name or "").lower()
    for tag in languages:
        tag = tag.lower()
        for dialect in DIALECTS:
            if tag.startswith(dialect.code.lower()):
                return dialect.code
    for dialect in DIALECTS:
        for alias in dialect.aliases:
            if alias in lowered:
                return dialect.code
    return ""


def classify_gender(name: str, reported: Optional[str]) -> str:
    """Gender from the driver when available, else from the known-name table."""
    if reported:
        text = str(reported).strip().lower()
        if text.startswith("f"):
            return "female"
        if text.startswith("m"):
            return "male"
    lowered = (name or "").lower()
    for key, (_, gender) in KNOWN_VOICES.items():
        if key in lowered:
            return gender
    return ""


def _known_pair(name: str) -> Tuple[str, str]:
    lowered = (name or "").lower()
    for key, pair in KNOWN_VOICES.items():
        if key in lowered:
            return pair
    return ("", "")


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------
def enumerate_pyttsx3_voices() -> List[VoiceOption]:
    """All offline SAPI5 voices, classified. Never raises."""
    if not PYTTsx3_AVAILABLE:
        return []
    engine = None
    try:
        engine = pyttsx3.init()  # type: ignore[union-attr]
        raw_voices = engine.getProperty("voices") or []
    except Exception:
        return []
    finally:
        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass

    voices: List[VoiceOption] = []
    for voice in raw_voices:
        name = getattr(voice, "name", "") or ""
        languages = _decode_languages(getattr(voice, "languages", None))
        dialect = classify_dialect(name, languages)
        gender = classify_gender(name, getattr(voice, "gender", None))
        if not dialect or not gender:
            known_dialect, known_gender = _known_pair(name)
            dialect = dialect or known_dialect
            gender = gender or known_gender
        voices.append(VoiceOption(
            id=getattr(voice, "id", "") or name,
            name=name,
            dialect=dialect,
            gender=gender,
            provider="pyttsx3",
        ))
    return voices


def enumerate_elevenlabs_voices(api_key: str, timeout: float = 12.0) -> List[VoiceOption]:
    """Cloud voices from ElevenLabs, classified by accent/gender labels.

    Only called when the user has configured an API key. Returns an empty list on
    any failure so callers can fall back to offline voices.
    """
    if not api_key:
        return []
    try:
        import requests
        response = requests.get(
            "https://api.elevenlabs.io/v1/voices",
            headers={"xi-api-key": api_key},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return []

    voices: List[VoiceOption] = []
    for item in payload.get("voices", []) or []:
        labels = item.get("labels") or {}
        accent = str(labels.get("accent", "") or "").lower()
        gender = str(labels.get("gender", "") or "").lower()
        name = item.get("name", "") or ""
        dialect = classify_dialect(f"{name} {accent}", [accent])
        voices.append(VoiceOption(
            id=item.get("voice_id", ""),
            name=name,
            dialect=dialect or "en-US",
            gender="female" if gender.startswith("f") else
                   "male" if gender.startswith("m") else "",
            provider="elevenlabs",
        ))
    return voices


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------
def filter_voices(voices: Iterable[VoiceOption], dialect: str = ANY_DIALECT,
                  gender: str = GENDER_ANY) -> List[VoiceOption]:
    """Voices matching a dialect and gender filter."""
    out: List[VoiceOption] = []
    for voice in voices:
        if dialect != ANY_DIALECT and voice.dialect != dialect:
            continue
        if gender != GENDER_ANY and voice.gender != gender:
            continue
        out.append(voice)
    return out


def dialects_available(voices: Iterable[VoiceOption]) -> List[str]:
    """Dialect codes that at least one installed voice covers."""
    return sorted({v.dialect for v in voices if v.dialect})


def resolve_voice(voices: Sequence[VoiceOption], dialect: str = ANY_DIALECT,
                  gender: str = GENDER_ANY) -> Tuple[Optional[VoiceOption], str]:
    """Best voice for a dialect/gender request.

    Returns ``(voice, note)``. ``note`` explains any fallback that was applied,
    so the UI can say *why* it is not using the requested dialect instead of
    quietly speaking in a different accent.
    """
    if not voices:
        return None, "No speech voices are installed."

    exact = filter_voices(voices, dialect, gender)
    if exact:
        return exact[0], ""

    if dialect != ANY_DIALECT:
        same_dialect = filter_voices(voices, dialect, GENDER_ANY)
        if same_dialect:
            chosen = same_dialect[0]
            return chosen, (
                f"No {DIALECT_BY_CODE[dialect].label} {gender} voice is installed — "
                f"using {chosen.display} instead."
                if gender != GENDER_ANY else ""
            )
        # Dialect missing entirely. Prefer to at least honour the requested
        # GENDER — falling back from "Indian English female" to a male US voice
        # would be a jarring, unexplained change.
        if gender != GENDER_ANY:
            gender_match = filter_voices(voices, ANY_DIALECT, gender)
            if gender_match:
                chosen = gender_match[0]
                return chosen, (
                    f"No {DIALECT_BY_CODE[dialect].label} voice is installed. "
                    f"Falling back to {chosen.display} to match the requested "
                    f"{gender} voice. See the voice setup instructions to add "
                    f"Indian English voices."
                )
        english = [v for v in voices if (v.dialect or "").startswith(("en", "hi"))]
        chosen = (english or list(voices))[0]
        return chosen, (
            f"No {DIALECT_BY_CODE[dialect].label} voice is installed. "
            f"Falling back to {chosen.display}. See the voice setup instructions "
            f"to add Indian English voices."
        )

    if gender != GENDER_ANY:
        same_gender = filter_voices(voices, ANY_DIALECT, gender)
        if same_gender:
            chosen = same_gender[0]
            return chosen, f"No {gender} voice in the requested dialect — using {chosen.display}."

    return list(voices)[0], ""


def catalogue_summary(voices: Sequence[VoiceOption]) -> str:
    """One-line description of what is installed."""
    if not voices:
        return "no voices"
    dialects = dialects_available(voices)
    genders = sorted({v.gender for v in voices if v.gender})
    return (f"{len(voices)} voice(s) · {', '.join(dialects) or 'unclassified'}"
            f" · {', '.join(genders) or 'gender not reported'}")


def install_help(dialect_code: str = "en-IN") -> str:
    """Human instructions for adding a missing dialect."""
    dialect = DIALECT_BY_CODE.get(dialect_code)
    label = dialect.label if dialect else dialect_code
    return (
        f"No {label} voice is installed on this PC.\n\n"
        "To add one (Windows 10/11):\n"
        "  1. Settings → Time & Language → Language & region\n"
        "  2. Add a language → search 'English (India)' → Next\n"
        "  3. Tick 'Speech' (and 'Text-to-speech') in the optional features list\n"
        "  4. Install, then restart BioHuman3D\n\n"
        "Windows installs Indian English voices such as Heera (female) and "
        "Ravi (male) once the Speech feature is added.\n\n"
        "Alternative without a language pack: configure an ElevenLabs API key in "
        "Settings. Its multilingual voices can speak Indian English and provide "
        "both male and female options."
    )
