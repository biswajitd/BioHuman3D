"""MRI volumes, disease staging, narration languages and translation memory."""
from __future__ import annotations

import json

import numpy as np
import pytest

from app.audio.languages import ENGINES, LANGUAGES, language
from app.audio.neural_tts import VoiceRequest
from app.disease.catalog import CONDITIONS
from app.disease.effects import blend_stages
from app.i18n.translate import TranslationMemory
from app.imaging.volume import ImagingVolume, _from_affine
from app.video.captions import chunk_at, chunk_text


# ------------------------------------------------------------------ imaging
def _volume():
    data = np.zeros((10, 12, 14), np.float32)
    data[0, :, :] = 1.0      # patient's right edge (x min)
    data[:, 0, :] = 2.0      # anterior edge (y min)
    data[:, :, -1] = 3.0     # superior edge (z max)
    return ImagingVolume(data, (0.01, 0.01, 0.01), (0.0, 0.0, 0.0))


def test_axial_slice_is_in_radiological_convention():
    vol = _volume()
    img, _ = vol.slice("z", 0.05)
    assert img.shape == (12, 10)
    assert img[5, 0] == 1.0, "patient's right on the viewer's left"
    assert img[0, 5] == 2.0, "anterior at the top"


def test_coronal_and_sagittal_have_superior_at_top():
    vol = _volume()
    coronal, _ = vol.slice("y", 0.05)
    sagittal, _ = vol.slice("x", 0.05)
    assert coronal[0, 5] == 3.0 and sagittal[0, 5] == 3.0


def test_ras_scan_is_reoriented_to_model_frame():
    data = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])                     # RAS mm, 2 mm voxels
    arr, spacing, origin = _from_affine(data, affine, lps=False)
    assert spacing == pytest.approx((0.002, 0.002, 0.002))
    # RAS +x (right) becomes model -x: the first model voxel is the most-left one.
    assert arr[0, 0, 0] == data[1, 2, 0]
    assert origin[0] == pytest.approx(-0.002)


# ------------------------------------------------------------------ disease
def test_every_condition_is_staged_and_sourced():
    for condition in CONDITIONS:
        assert len(condition.stages) >= 3
        assert condition.sources and condition.staging
        for stage in condition.stages:
            assert stage.criteria and stage.narration
            for effect in stage.effects:
                assert effect["kind"] in ("tint", "scale", "nodular", "pinch", "lesion", "opacity")
                assert effect["match"]


def test_stage_blend_starts_from_previous_stage_and_ends_at_target():
    ckd = next(c for c in CONDITIONS if c.id == "ckd")
    start = {e["id"]: e for e in blend_stages(ckd, 3, 4, 0.0)}
    end = {e["id"]: e for e in blend_stages(ckd, 3, 4, 1.0)}
    assert start["size"]["factor"] == pytest.approx(0.85)
    assert end["size"]["factor"] == pytest.approx(0.74)
    first = {e["id"]: e for e in blend_stages(ckd, 0, 1, 0.0)}
    assert first["pallor"]["amount"] == pytest.approx(0.0), "new effects grow from nothing"


# ------------------------------------------------------------------ languages
def test_language_catalogue_covers_requested_groups():
    codes = {lang.code for lang in LANGUAGES}
    assert {"en-IN", "en-GB", "en-US", "en-AU"} <= codes
    assert {"hi-IN", "bn-IN", "ta-IN", "te-IN", "mr-IN", "gu-IN", "kn-IN", "ml-IN", "pa-IN", "or-IN"} <= codes
    assert {"es-ES", "fr-FR", "de-DE", "zh-CN", "ja-JP", "ar-SA", "pt-BR", "ru-RU"} <= codes
    for lang in LANGUAGES:
        assert lang.female != lang.male and lang.female.startswith(lang.code)
    assert language("ur-IN").rtl and not language("hi-IN").rtl
    assert ENGINES[0][0] == "edge"


def test_voice_request_mapping():
    req = VoiceRequest(engine="edge", language="hi-IN", gender="male", rate_wpm=175)
    assert req.voice_name() == "hi-IN-MadhurNeural"
    assert req.rate_percent() == 0
    assert VoiceRequest(engine="edge", rate_wpm=210).rate_percent() == 20
    assert req.cache_key("a") != req.cache_key("b")


def test_translation_memory_never_overwrites_reviewed_entries(tmp_path):
    memory = TranslationMemory(tmp_path, "hi-IN")
    memory.put("Elbow flexion", "कोहनी मोड़ना", "llm")
    memory.save()
    data = json.loads((tmp_path / "hi-IN.json").read_text(encoding="utf-8"))
    entry = next(iter(data.values()))
    entry["reviewed"] = True
    entry["text"] = "कोहनी का मुड़ना"
    (tmp_path / "hi-IN.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    memory = TranslationMemory(tmp_path, "hi-IN")
    memory.put("Elbow flexion", "machine text", "google")
    assert memory.get("Elbow flexion") == "कोहनी का मुड़ना"
    assert memory.missing(["Elbow flexion", "Knee flexion"]) == ["Knee flexion"]


# ------------------------------------------------------------------ captions
def test_captions_split_on_sentences_in_any_script():
    hindi = "द्विशिर पेशी कोहनी को मोड़ती है। इसकी सामान्य गति एक सौ पचास डिग्री है।"
    assert len(chunk_text(hindi)) == 2
    chunks = chunk_text("First sentence here. Second one follows. Third.")
    assert chunk_at(chunks, 0.0, 6.0) == chunks[0]
    assert chunk_at(chunks, 5.9, 6.0) == chunks[-1]
