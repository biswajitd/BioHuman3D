"""
Headless UI smoke test.

Builds the full application shell off-screen, drives the main subsystems
(scene load, layer fading, isolation, camera presets, guided tour, AI context
sync), captures a PNG of the rendered window, and reports pass/fail per check.

This is the fastest way to catch wiring regressions — a broken signal connection
or a misnamed attribute surfaces here in ~3 seconds instead of as a crash on
launch.

    python tools\\ui_smoke_test.py                 # run checks + save screenshot
    python tools\\ui_smoke_test.py --no-capture    # skip the screenshot
    python tools\\ui_smoke_test.py --show          # use the real display instead

Exit code 0 = all checks passed.
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

CHECK_RESULTS: list = []


def record(name: str, ok: bool, detail: str = ""):
    CHECK_RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<44} {detail}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Headless smoke test for BioHuman3D.")
    parser.add_argument("--show", action="store_true", help="use the real display")
    parser.add_argument("--no-capture", action="store_true", help="skip the screenshot")
    parser.add_argument("--out", type=Path, default=None, help="screenshot path")
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    if not args.show:
        # Must be set before QApplication is constructed.
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication

    from app.config import AppConfig
    from app.core.vtk_utils import VTK_AVAILABLE
    from app.ui.main_window import MainWindow
    from app.ui.theme import apply_theme

    print("=" * 78)
    print(" BioHuman3D - headless UI smoke test")
    print(f" platform={os.environ.get('QT_QPA_PLATFORM', 'default')}  vtk={VTK_AVAILABLE}")
    print("=" * 78)

    app = QApplication(sys.argv[:1])
    config = AppConfig.load()
    apply_theme(app, config)

    # Exercise the FIRST-RUN path: discard any geometry saved by an earlier run so
    # the app must choose a size itself and prove it fits the real screen. This is
    # the case the user reported (bottom of the window off-screen). Deliberately
    # NOT forcing a size either — the app has to get this right on its own.
    config["ui.geometry"] = ""

    try:
        window = MainWindow(config)
    except Exception:
        traceback.print_exc()
        record("MainWindow construction", False, "exception during __init__")
        return 1
    record("MainWindow construction", True)
    window.show()

    failures = {"count": 0}

    def expect(name: str, ok: bool, detail: str = ""):
        if not record(name, ok, detail):
            failures["count"] += 1

    def run_checks():
        try:
            # -- scene ------------------------------------------------------
            layers = getattr(window.viewport, "layers", {}) or {}
            expect("Scene loaded layers", len(layers) > 0, f"{len(layers)} layers")

            # -- layers panel -------------------------------------------------
            expect("Layer rows populated", len(window.layer_panel._rows) > 0,
                   f"{len(window.layer_panel._rows)} rows")

            # -- opacity + isolation -----------------------------------------
            target = "muscles" if "muscles" in layers else next(iter(layers), "")
            if target:
                window.controller.set_layer_opacity(target, 0.25)
                state = layers.get(target)
                expect("Fade layer to 25%", abs(state.opacity - 0.25) < 0.01,
                       f"{target} opacity={state.opacity:.2f}")

                window.controller.isolate_layer(target)
                others = [s.opacity for lid, s in layers.items() if lid != target]
                expect("Isolate ghosts other layers",
                       all(o <= 0.1 for o in others) if others else True,
                       f"max other opacity={max(others, default=0):.2f}")

                window.controller.show_all()
                expect("Show all restores opacity",
                       all(s.visible for s in layers.values()))

                window.controller.select_structure(target)
                expect("Selection propagates to AI context",
                       window.ai._selected_id == target,  # noqa: SLF001
                       f"selected={window.ai._selected_id}")  # noqa: SLF001
            else:
                expect("Layer manipulation", False, "no layers to manipulate")

            # -- camera --------------------------------------------------------
            for preset in ("anterior", "left", "isometric"):
                window.controller.set_view(preset)
            expect("Camera presets applied", True, "anterior/left/isometric")

            state = window.viewport.camera_state()
            expect("Camera state readable", bool(state.get("position")),
                   f"pos={[round(v, 2) for v in state.get('position', [])]}")

            # -- clipping + explode ---------------------------------------------
            window.viewport.apply_clip_plane("y", 0.5)
            window.viewport.set_explode(0.4)
            window.viewport.set_explode(0.0)
            window.viewport.disable_clip_plane()
            expect("Slice + explode pipeline", True)

            # -- tours ----------------------------------------------------------
            tour = window.controller.available_tours()[0]
            expect("Guided tours available", tour is not None,
                   f"{len(window.controller.available_tours())} tours")

            window.controller.start_tour(tour.id, speak=False)
            expect("Tour engine started", window.controller.tour.is_running,
                   tour.title)

            window.controller.tour.next_keyframe()
            window.controller.tour.next_keyframe()
            index = window.controller.tour.current_index
            expect("Tour advances keyframes", index >= 1, f"index={index}")

            window.ai_panel.set_cues(tour.cues())
            expect("Narration cues rendered", len(window.ai_panel._cue_rows) > 0,
                   f"{len(window.ai_panel._cue_rows)} cues")

            window.ai_panel.set_cues([])
            window.ai_panel.set_cues(tour.cues())
            expect("Cue list rebuilds safely",
                   len(window.ai_panel._cue_rows) == len(tour.cues()))

            window.controller.stop_tour()
            expect("Tour stops cleanly", not window.controller.tour.is_running)

            # -- AI panel --------------------------------------------------------
            window.ai_panel.append_user("Explain the deltoid")
            window.ai_panel.start_assistant("Ollama", "qwen2.5:7b-instruct")
            answer = (
                "### Deltoid\n"
                "The **deltoid** is a thick, multipennate muscle forming the "
                "contour of the shoulder.\n"
                "- **Origin:** lateral third of the clavicle, acromion, spine of scapula\n"
                "- **Insertion:** deltoid tuberosity of the `humerus`\n"
                "- **Nerve:** axillary nerve (C5-C6)\n"
                "\n"
                "Its three parts act independently: the anterior fibres flex and "
                "medially rotate, the lateral fibres abduct, and the posterior "
                "fibres extend and laterally rotate.\n"
                "\n"
                "Clinically, the axillary nerve is at risk in anterior shoulder "
                "dislocation, which paralyses the deltoid and numbs the regimental "
                "badge area."
            )
            for chunk in answer.split(" "):
                window.ai_panel.append_token(chunk + " ")
            window.ai_panel.finish_assistant(0.42)
            expect("Transcript streaming renders",
                   len(window.ai_panel.last_answer()) > 100)
            # Qt normalises <b>/<i> into style runs, so assert on the absence of
            # raw markdown markers plus the presence of list/styling markup.
            html = window.ai_panel._chat_view.toHtml()  # noqa: SLF001
            expect("Markdown re-rendered as HTML",
                   "**" not in html and "<li" in html.lower(),
                   "bullets + bold converted")
            expect("Transcript messages on separate lines",
                   window.ai_panel._chat_view.toPlainText().count("\n") >= 3,
                   f"{window.ai_panel._chat_view.toPlainText().count(chr(10))} line breaks")

            window.ai_panel.append_error("simulated provider failure")
            expect("Error rendering", True)

            window.ai_panel.set_models([
                type("M", (), {"provider_id": "ollama", "id": "qwen2.5:7b-instruct",
                               "label": "qwen2.5:7b-instruct", "kind": "local",
                               "size_gb": 4.7})(),
                type("M", (), {"provider_id": "openai", "id": "gpt-4o",
                               "label": "gpt-4o", "kind": "cloud", "size_gb": 0.0})(),
            ], "ollama", "qwen2.5:7b-instruct")
            expect("Model dropdown populated",
                   window.ai_panel._model_combo.count() >= 3,
                   f"{window.ai_panel._model_combo.count()} entries")

            # -- audio -------------------------------------------------------------
            expect("Audio manager available (may be silent)",
                   window.audio.available_backends() is not None,
                   ", ".join(window.audio.available_backends()))

            # -- right-panel tabs --------------------------------------------------
            window.ai_panel.show_tab("media")
            expect("Media tab reachable",
                   window.ai_panel._stack.currentIndex() == 1,  # noqa: SLF001
                   "Audio & Video tab shown")
            window.ai_panel.show_tab("assistant")
            expect("Assistant tab reachable",
                   window.ai_panel._stack.currentIndex() == 0)  # noqa: SLF001

            # -- video export UI ---------------------------------------------------
            from app.video.exporter import video_support_status
            from app.audio.scripts import (BUILTIN_TOURS, tour_list,
                                           video_subjects, system_presets)
            video_ok, video_detail = video_support_status()
            expect("Video encoder available", video_ok, video_detail)

            # Regression: the export dropdown was fed a hard-coded 3-tour list with
            # the skeleton first, so every export rendered the skeleton and syncing
            # to any other system silently failed.
            offered = window.ai_panel._video_tour.count()  # noqa: SLF001
            expect("Export offers every authored subject",
                   offered >= len(video_subjects()),
                   f"{offered} entries for {len(BUILTIN_TOURS)} tours")

            expect("Export subject starts non-skeletal",
                   window.ai_panel.current_video_tour_id() != "skeletal_overview",
                   window.ai_panel.current_video_tour_id())

            expect("Liver subject offered",
                   window.ai_panel._video_tour.findData("liver_overview") >= 0)  # noqa: SLF001

            # Selecting a structure must retarget the export subject.
            window._sync_subject_to_layer("liver")  # noqa: SLF001
            expect("Liver selection retargets export",
                   window.ai_panel.current_video_tour_id() == "liver_overview",
                   window.ai_panel.current_video_tour_id())

            window._sync_subject_to_layer("kidneys")  # noqa: SLF001
            expect("Kidney selection retargets export",
                   window.ai_panel.current_video_tour_id() == "kidneys_overview",
                   window.ai_panel.current_video_tour_id())

            # Sidebar system list must come from the registry, not a hard-coded tuple.
            preset_count = window.layer_panel._preset.count()  # noqa: SLF001
            expect("Sidebar lists every body system",
                   preset_count == len(system_presets()),
                   f"{preset_count} presets vs {len(system_presets())} systems")

            window._on_system_selected("Urinary")  # noqa: SLF001
            expect("System selection retargets export",
                   window.ai_panel.current_video_tour_id() == "urinary_overview",
                   window.ai_panel.current_video_tour_id())

            window._update_video_estimate()  # noqa: SLF001
            estimate_ok = window.ai_panel._btn_generate.isEnabled()  # noqa: SLF001
            expect("Export button enabled", estimate_ok or not video_ok,
                   window.ai_panel._video_estimate.text())  # noqa: SLF001

            window.ai_panel.set_video_progress(45, "Rendering frames 50/120…")
            window.ai_panel.set_video_result(str(config.paths.user_data / "videos" / "x.mp4"))
            expect("Video player controls enable",
                   window.ai_panel._btn_play_video.isEnabled())  # noqa: SLF001

            from app.video.player import MULTIMEDIA_AVAILABLE
            expect("QtMultimedia playback available", MULTIMEDIA_AVAILABLE,
                   "in-app player ready" if MULTIMEDIA_AVAILABLE else "fallback only")

            # -- window fits the screen --------------------------------------------
            from PyQt6.QtGui import QGuiApplication
            screen = QGuiApplication.primaryScreen()
            if screen is not None:
                avail = screen.availableGeometry()
                geo = window.frameGeometry()
                # Position matters as much as size: a window can be small enough
                # yet sit low, putting the status bar behind the taskbar.
                fits = (geo.width() <= avail.width() + 2
                        and geo.height() <= avail.height() + 2
                        and geo.left() >= avail.left() - 2
                        and geo.top() >= avail.top() - 2
                        and geo.bottom() <= avail.bottom() + 2
                        and geo.right() <= avail.right() + 2)
                expect("Window fits available screen", fits,
                       f"frame {geo.width()}x{geo.height()} at "
                       f"({geo.left()},{geo.top()})-({geo.right()},{geo.bottom()}) "
                       f"vs available {avail.width()}x{avail.height()} "
                       f"({avail.left()},{avail.top()})-({avail.right()},{avail.bottom()})")
                # The real requirement: nothing (status bar included) sits below
                # the bottom of the usable screen area.
                frame_bottom = window.frameGeometry().bottom()
                expect("Status bar on screen",
                       window.statusBar().isVisible() and frame_bottom <= avail.bottom(),
                       f"frame bottom={frame_bottom} available bottom={avail.bottom()}")

            # -- settings dialog ----------------------------------------------------
            from app.ui.main_window import SettingsDialog
            dialog = SettingsDialog(config, window)
            expect("Settings dialog constructs", dialog is not None)
            dialog.reject()

            # -- screenshot ----------------------------------------------------------
            if not args.no_capture:
                target_path = args.out or (config.paths.user_data / "screenshots"
                                           / "smoke_test.png")
                target_path.parent.mkdir(parents=True, exist_ok=True)
                pixmap = window.grab()
                saved = pixmap.save(str(target_path))
                expect("Window captured to PNG", saved, str(target_path))
                window._smoke_screenshot = target_path  # noqa: SLF001

        except Exception:
            traceback.print_exc()
            failures["count"] += 1
        finally:
            QTimer.singleShot(200, app.quit)

    # Let the deferred startup (scene load + model detection) run first.
    QTimer.singleShot(1800, run_checks)
    app.exec()

    print("-" * 78)
    total = len(CHECK_RESULTS)
    passed = sum(1 for _, ok, _ in CHECK_RESULTS if ok)
    print(f" {passed}/{total} checks passed, {failures['count']} failure(s)")
    print("-" * 78)

    window.close()
    return 1 if failures["count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
