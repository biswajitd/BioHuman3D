"""
COM apartment helpers for Windows speech.

``pyttsx3`` drives SAPI5 through ``comtypes``, and COM must be initialised on
**each thread that uses it**. Worker threads that create a speech engine without
calling ``CoInitialize`` can misbehave — the classic symptom is the first
utterance working and the second hanging or fail-fast crashing the process.

Both are no-ops on non-Windows platforms and when ``pythoncom`` is absent, so
callers do not need to branch on the platform.
"""
from __future__ import annotations


def co_initialize() -> bool:
    """Initialise COM on the calling thread. Returns True when it was available."""
    try:
        import pythoncom  # type: ignore
        pythoncom.CoInitialize()
        return True
    except Exception:
        return False


def co_uninitialize() -> None:
    """Release the calling thread's COM apartment."""
    try:
        import pythoncom  # type: ignore
        pythoncom.CoUninitialize()
    except Exception:
        pass
