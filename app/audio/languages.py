"""
Narration languages and natural neural voices.

Every language offers a **female and a male** neural voice. The default engine
is Microsoft's neural voices (the same voice models as Azure AI Speech): it
needs no key through the Edge read-aloud service (``edge-tts``), or uses your
own Azure Speech key for production use. Google Cloud Text-to-Speech is
supported as an alternative, with voices resolved live from its catalogue.

Groups:

* **English dialects** — India, UK, US, Australia, Canada, Ireland, South
  Africa, New Zealand, Singapore, Nigeria, Kenya.
* **Indian languages** — Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati,
  Kannada, Malayalam, Odia, Punjabi, Assamese, Urdu, Nepali.
* **International** — Spanish, French, German, Italian, Portuguese, Russian,
  Chinese, Japanese, Korean, Arabic, Turkish, Indonesian, Dutch, Polish …

Voice names below are Microsoft neural voice ids (``hi-IN-SwaraNeural``). If a
service retires a voice, live voice lists take precedence (see
``neural_tts.list_voices``), so the catalogue degrades instead of breaking.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Language:
    code: str                 # BCP-47, e.g. "hi-IN"
    name: str                 # English name
    native: str               # endonym, shown in the UI
    group: str                # "English dialects" | "Indian languages" | "International"
    female: str               # Microsoft neural voice id
    male: str
    rtl: bool = False
    translate_to: str = ""    # translation target (ISO 639-1) — "" for English

    @property
    def is_english(self) -> bool:
        return self.code.startswith("en-")

    @property
    def label(self) -> str:
        return self.name if self.native == self.name else f"{self.name} — {self.native}"

    def voice(self, gender: str) -> str:
        return self.male if gender == "male" else self.female


ENGLISH = "English dialects"
INDIAN = "Indian languages"
INTERNATIONAL = "International"

LANGUAGES: Tuple[Language, ...] = (
    # -- English dialects (Indian English first) --------------------------------
    Language("en-IN", "English (India)", "English (India)", ENGLISH, "en-IN-NeerjaNeural", "en-IN-PrabhatNeural"),
    Language("en-GB", "English (United Kingdom)", "English (UK)", ENGLISH, "en-GB-SoniaNeural", "en-GB-RyanNeural"),
    Language("en-US", "English (United States)", "English (US)", ENGLISH, "en-US-JennyNeural", "en-US-GuyNeural"),
    Language("en-AU", "English (Australia)", "English (Australia)", ENGLISH, "en-AU-NatashaNeural", "en-AU-WilliamNeural"),
    Language("en-CA", "English (Canada)", "English (Canada)", ENGLISH, "en-CA-ClaraNeural", "en-CA-LiamNeural"),
    Language("en-IE", "English (Ireland)", "English (Ireland)", ENGLISH, "en-IE-EmilyNeural", "en-IE-ConnorNeural"),
    Language("en-ZA", "English (South Africa)", "English (South Africa)", ENGLISH, "en-ZA-LeahNeural", "en-ZA-LukeNeural"),
    Language("en-NZ", "English (New Zealand)", "English (New Zealand)", ENGLISH, "en-NZ-MollyNeural", "en-NZ-MitchellNeural"),
    Language("en-SG", "English (Singapore)", "English (Singapore)", ENGLISH, "en-SG-LunaNeural", "en-SG-WayneNeural"),
    Language("en-NG", "English (Nigeria)", "English (Nigeria)", ENGLISH, "en-NG-EzinneNeural", "en-NG-AbeoNeural"),
    Language("en-KE", "English (Kenya)", "English (Kenya)", ENGLISH, "en-KE-AsiliaNeural", "en-KE-ChilembaNeural"),
    # -- Indian languages ---------------------------------------------------------
    Language("hi-IN", "Hindi", "हिन्दी", INDIAN, "hi-IN-SwaraNeural", "hi-IN-MadhurNeural", translate_to="hi"),
    Language("bn-IN", "Bengali", "বাংলা", INDIAN, "bn-IN-TanishaaNeural", "bn-IN-BashkarNeural", translate_to="bn"),
    Language("ta-IN", "Tamil", "தமிழ்", INDIAN, "ta-IN-PallaviNeural", "ta-IN-ValluvarNeural", translate_to="ta"),
    Language("te-IN", "Telugu", "తెలుగు", INDIAN, "te-IN-ShrutiNeural", "te-IN-MohanNeural", translate_to="te"),
    Language("mr-IN", "Marathi", "मराठी", INDIAN, "mr-IN-AarohiNeural", "mr-IN-ManoharNeural", translate_to="mr"),
    Language("gu-IN", "Gujarati", "ગુજરાતી", INDIAN, "gu-IN-DhwaniNeural", "gu-IN-NiranjanNeural", translate_to="gu"),
    Language("kn-IN", "Kannada", "ಕನ್ನಡ", INDIAN, "kn-IN-SapnaNeural", "kn-IN-GaganNeural", translate_to="kn"),
    Language("ml-IN", "Malayalam", "മലയാളം", INDIAN, "ml-IN-SobhanaNeural", "ml-IN-MidhunNeural", translate_to="ml"),
    Language("or-IN", "Odia", "ଓଡ଼ିଆ", INDIAN, "or-IN-SubhasiniNeural", "or-IN-SukantNeural", translate_to="or"),
    Language("pa-IN", "Punjabi", "ਪੰਜਾਬੀ", INDIAN, "pa-IN-VaaniNeural", "pa-IN-OjasNeural", translate_to="pa"),
    Language("as-IN", "Assamese", "অসমীয়া", INDIAN, "as-IN-YashicaNeural", "as-IN-PriyomNeural", translate_to="as"),
    Language("ur-IN", "Urdu (India)", "اردو", INDIAN, "ur-IN-GulNeural", "ur-IN-SalmanNeural", rtl=True, translate_to="ur"),
    Language("ne-NP", "Nepali", "नेपाली", INDIAN, "ne-NP-HemkalaNeural", "ne-NP-SagarNeural", translate_to="ne"),
    # -- International ------------------------------------------------------------
    Language("es-ES", "Spanish (Spain)", "Español (España)", INTERNATIONAL, "es-ES-ElviraNeural", "es-ES-AlvaroNeural", translate_to="es"),
    Language("es-MX", "Spanish (Mexico)", "Español (México)", INTERNATIONAL, "es-MX-DaliaNeural", "es-MX-JorgeNeural", translate_to="es"),
    Language("fr-FR", "French", "Français", INTERNATIONAL, "fr-FR-DeniseNeural", "fr-FR-HenriNeural", translate_to="fr"),
    Language("de-DE", "German", "Deutsch", INTERNATIONAL, "de-DE-KatjaNeural", "de-DE-ConradNeural", translate_to="de"),
    Language("it-IT", "Italian", "Italiano", INTERNATIONAL, "it-IT-ElsaNeural", "it-IT-DiegoNeural", translate_to="it"),
    Language("pt-BR", "Portuguese (Brazil)", "Português (Brasil)", INTERNATIONAL, "pt-BR-FranciscaNeural", "pt-BR-AntonioNeural", translate_to="pt"),
    Language("pt-PT", "Portuguese (Portugal)", "Português (Portugal)", INTERNATIONAL, "pt-PT-RaquelNeural", "pt-PT-DuarteNeural", translate_to="pt-PT"),
    Language("ru-RU", "Russian", "Русский", INTERNATIONAL, "ru-RU-SvetlanaNeural", "ru-RU-DmitryNeural", translate_to="ru"),
    Language("zh-CN", "Chinese (Mandarin)", "中文（普通话）", INTERNATIONAL, "zh-CN-XiaoxiaoNeural", "zh-CN-YunxiNeural", translate_to="zh-CN"),
    Language("ja-JP", "Japanese", "日本語", INTERNATIONAL, "ja-JP-NanamiNeural", "ja-JP-KeitaNeural", translate_to="ja"),
    Language("ko-KR", "Korean", "한국어", INTERNATIONAL, "ko-KR-SunHiNeural", "ko-KR-InJoonNeural", translate_to="ko"),
    Language("ar-SA", "Arabic (Saudi Arabia)", "العربية", INTERNATIONAL, "ar-SA-ZariyahNeural", "ar-SA-HamedNeural", rtl=True, translate_to="ar"),
    Language("ar-EG", "Arabic (Egypt)", "العربية (مصر)", INTERNATIONAL, "ar-EG-SalmaNeural", "ar-EG-ShakirNeural", rtl=True, translate_to="ar"),
    Language("tr-TR", "Turkish", "Türkçe", INTERNATIONAL, "tr-TR-EmelNeural", "tr-TR-AhmetNeural", translate_to="tr"),
    Language("id-ID", "Indonesian", "Bahasa Indonesia", INTERNATIONAL, "id-ID-GadisNeural", "id-ID-ArdiNeural", translate_to="id"),
    Language("ms-MY", "Malay", "Bahasa Melayu", INTERNATIONAL, "ms-MY-YasminNeural", "ms-MY-OsmanNeural", translate_to="ms"),
    Language("nl-NL", "Dutch", "Nederlands", INTERNATIONAL, "nl-NL-ColetteNeural", "nl-NL-MaartenNeural", translate_to="nl"),
    Language("pl-PL", "Polish", "Polski", INTERNATIONAL, "pl-PL-ZofiaNeural", "pl-PL-MarekNeural", translate_to="pl"),
    Language("uk-UA", "Ukrainian", "Українська", INTERNATIONAL, "uk-UA-PolinaNeural", "uk-UA-OstapNeural", translate_to="uk"),
    Language("fa-IR", "Persian", "فارسی", INTERNATIONAL, "fa-IR-DilaraNeural", "fa-IR-FaridNeural", rtl=True, translate_to="fa"),
    Language("th-TH", "Thai", "ไทย", INTERNATIONAL, "th-TH-PremwadeeNeural", "th-TH-NiwatNeural", translate_to="th"),
    Language("vi-VN", "Vietnamese", "Tiếng Việt", INTERNATIONAL, "vi-VN-HoaiMyNeural", "vi-VN-NamMinhNeural", translate_to="vi"),
    Language("sw-KE", "Swahili", "Kiswahili", INTERNATIONAL, "sw-KE-ZuriNeural", "sw-KE-RafikiNeural", translate_to="sw"),
    Language("fil-PH", "Filipino", "Filipino", INTERNATIONAL, "fil-PH-BlessicaNeural", "fil-PH-AngeloNeural", translate_to="fil"),
    Language("bn-BD", "Bengali (Bangladesh)", "বাংলা (বাংলাদেশ)", INTERNATIONAL, "bn-BD-NabanitaNeural", "bn-BD-PradeepNeural", translate_to="bn"),
)

LANGUAGE_BY_CODE: Dict[str, Language] = {lang.code: lang for lang in LANGUAGES}
DEFAULT_LANGUAGE = "en-IN"


def language(code: str) -> Language:
    return LANGUAGE_BY_CODE.get(code) or LANGUAGE_BY_CODE[DEFAULT_LANGUAGE]


def grouped() -> Dict[str, List[Language]]:
    out: Dict[str, List[Language]] = {}
    for lang in LANGUAGES:
        out.setdefault(lang.group, []).append(lang)
    return out


#: Voice engines, in the order offered in the UI.
ENGINES: Tuple[Tuple[str, str], ...] = (
    ("edge", "Neural voices — Microsoft (free, online)"),
    ("azure", "Neural voices — Azure AI Speech (your key)"),
    ("google", "Neural voices — Google Cloud TTS (your key)"),
    ("elevenlabs", "ElevenLabs multilingual (your key)"),
    ("pyttsx3", "System voices (offline)"),
    ("none", "Silent"),
)
NEURAL_ENGINES = ("edge", "azure", "google", "elevenlabs")
