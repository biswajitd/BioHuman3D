"""Background construction of simulated MRI and loading of real scans."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from app.imaging.synthetic_mri import MRISimulator
from app.imaging.volume import load_volume


class MRISimulationWorker(QThread):
    """Voxelises deep-copied structures off the GUI thread (no rendering here)."""

    progress = pyqtSignal(int, str)
    completed = pyqtSignal(object, object)        # simulator, first volume (T1)
    failed = pyqtSignal(str)

    def __init__(self, items, spacing: float, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._items = items
        self._spacing = spacing
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:  # noqa: D102
        try:
            sim = MRISimulator(self._items, self._spacing, progress=self.progress.emit,
                               cancelled=lambda: self._cancel)
            if not sim.build():
                if not self._cancel:
                    self.failed.emit("Nothing to voxelise — load anatomy first.")
                return
            self.progress.emit(92, "Rendering T1-weighted contrast…")
            self.completed.emit(sim, sim.render("T1"))
        except Exception as exc:  # pragma: no cover - surfaced to the UI
            self.failed.emit(f"MRI simulation failed: {type(exc).__name__}: {exc}")


class ScanLoadWorker(QThread):
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, path: str, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._path = path

    def run(self) -> None:  # noqa: D102
        try:
            self.completed.emit(load_volume(Path(self._path)))
        except Exception as exc:
            self.failed.emit(str(exc))
