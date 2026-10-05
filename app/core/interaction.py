"""
Mouse interaction for the anatomy viewport.

Subclasses ``vtkInteractorStyleTrackballCamera`` so we inherit professional-grade
orbit / dolly / pan, then layers on three behaviours a clinical viewer needs:

* **Hover highlight** — throttled cell picking under the cursor.
* **Click to select** — emits the picked structure so the AI panel gets context.
* **Double-click to isolate** — solos the structure and ghosts everything else.

Picking is restricted with ``PickFromList`` to the *currently visible* actors so
that a click through faded skin still hits the muscle underneath.
"""
from __future__ import annotations

from typing import Optional

from app.core.vtk_utils import VTK_AVAILABLE, vtk


# Throttle: pick every Nth mouse-move event (picking is ~0.3–2 ms per call).
HOVER_PICK_INTERVAL = 5


if VTK_AVAILABLE:

    class AnatomyInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
        """Trackball camera + anatomy-aware picking."""

        def __init__(self, viewport) -> None:
            super().__init__()
            self._viewport = viewport
            self._picker = vtk.vtkCellPicker()
            self._picker.SetTolerance(0.004)
            self._picker.SetPickFromList(True)

            self._move_counter = 0
            self._button_down = False
            self._last_hover_id: Optional[str] = None

            self.AddObserver("MouseMoveEvent", self._on_mouse_move)
            self.AddObserver("LeftButtonPressEvent", self._on_left_press)
            self.AddObserver("LeftButtonReleaseEvent", self._on_left_release)
            self.AddObserver("LeftButtonDoubleClickEvent", self._on_double_click)
            self.AddObserver("MouseWheelForwardEvent", self._on_wheel_forward)
            self.AddObserver("MouseWheelBackwardEvent", self._on_wheel_backward)
            self.AddObserver("EndInteractionEvent", self._on_end_interaction)

        # -- picking helpers ----------------------------------------------
        def _refresh_pick_list(self) -> None:
            """Only visible actors are pickable — clicks fall through faded layers."""
            self._picker.InitializePickList()
            for actor in self._viewport.pickable_actors():
                self._picker.AddPickList(actor)

        def _pick_actor(self):
            x, y = self.GetInteractor().GetEventPosition()
            self._picker.Pick(x, y, 0, self._viewport.renderer)
            return self._picker.GetActor()

        # -- events --------------------------------------------------------
        def _on_mouse_move(self, _obj, _event) -> None:
            if not self._button_down:
                self._move_counter += 1
                if self._move_counter % HOVER_PICK_INTERVAL == 0:
                    self._refresh_pick_list()
                    actor = self._pick_actor()
                    layer_id = self._viewport.layer_id_for_actor(actor)
                    key = (layer_id, self._viewport.structure_for_actor(actor))
                    if key != self._last_hover_id:
                        self._last_hover_id = key
                        self._viewport._on_hover_changed(layer_id, actor)
            self.OnMouseMove()          # preserve default orbit / pan

        def _on_left_press(self, _obj, _event) -> None:
            self._button_down = True
            self.OnLeftButtonDown()

        def _on_left_release(self, _obj, _event) -> None:
            self._button_down = False
            if self.GetInteractor().GetShiftKey():
                # Shift-click = additive selection; handled by the viewport.
                pass
            self._refresh_pick_list()
            actor = self._pick_actor()
            self._viewport._on_click_picked(actor)
            self.OnLeftButtonUp()

        def _on_double_click(self, _obj, _event) -> None:
            self._refresh_pick_list()
            actor = self._pick_actor()
            layer_id = self._viewport.layer_id_for_actor(actor)
            if layer_id:
                self._viewport.isolate_layer(layer_id)
            self.OnLeftButtonDoubleClick()

        def _on_wheel_forward(self, _obj, _event) -> None:
            self.OnMouseWheelForward()

        def _on_wheel_backward(self, _obj, _event) -> None:
            self.OnMouseWheelBackward()

        def _on_end_interaction(self, _obj, _event) -> None:
            self._viewport._on_camera_moved(self)

else:  # pragma: no cover - VTK missing

    class AnatomyInteractorStyle:  # type: ignore[no-redef]
        """Import-safe stub so the rest of the app can be imported without VTK."""

        def __init__(self, *_args, **_kwargs) -> None:
            raise RuntimeError("VTK is not available in this environment.")
