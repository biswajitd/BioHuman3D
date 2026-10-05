"""Background translation of tour narration."""
from __future__ import annotations

from typing import Optional, Sequence

from PyQt6.QtCore import QObject, QThread, pyqtSignal


class TranslationWorker(QThread):
    progress = pyqtSignal(int, str)
    done = pyqtSignal(bool, str)          # ok, message

    def __init__(self, translator, sources: Sequence[str], lang_code: str,
                 parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._translator = translator
        self._sources = list(sources)
        self._lang = lang_code
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:  # noqa: D102
        try:
            self._translator.translate_all(self._sources, self._lang, progress=self.progress.emit,
                                           cancelled=lambda: self._cancel)
            self.done.emit(not self._cancel, "")
        except Exception as exc:
            self.done.emit(False, str(exc))
