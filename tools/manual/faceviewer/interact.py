#!/usr/bin/env python3
""" Handles the viewport area for mouse hover actions and the active frame """
from __future__ import annotations
import logging
import tkinter as tk
import typing as T
from dataclasses import dataclass

import numpy as np

from lib.logger import parse_class_init

if T.TYPE_CHECKING:
    from lib.align import DetectedFace
    from .viewport import Viewport

logger = logging.getLogger(__name__)


class HoverBox():
    """ Handle the current mouse location when over the :class:`Viewport`.

    Highlights the face currently underneath the cursor and handles actions when clicking
    on a face.

    Parameters
    ----------
    viewport: :class:`Viewport`
        The viewport object for the :class:`~tools.manual.faceviewer.frame.FacesViewer` canvas
    """
    def __init__(self, viewport: Viewport) -> None:
        logger.debug(parse_class_init(locals()))
        self._viewport = viewport
        self._canvas = viewport._canvas
        self._grid = viewport._canvas.layout
        self._globals = viewport._canvas._globals
        self._navigation = viewport._canvas._display_frame.navigation
        self._box = self._canvas.create_rectangle(0.,  # type:ignore[call-overload]
                                                  0.,
                                                  float(self._size),
                                                  float(self._size),
                                                  outline="#0000ff",
                                                  width=2,
                                                  state="hidden",
                                                  fill="#0000ff",
                                                  stipple="gray12",
                                                  tags="hover_box")
        self._current_frame_index = None
        self._current_face_index = None
        self._canvas.bind("<Leave>", lambda e: self._clear())
        self._canvas.bind("<Motion>", self.on_hover)
        self._canvas.bind("<ButtonPress-1>", lambda e: self._select_frame())
        logger.debug("Initialized: %s", self.__class__.__name__)

    @property
    def _size(self) -> int:
        """ int: the currently set viewport face size in pixels. """
        return self._viewport.face_size

    def on_hover(self, event: tk.Event | None) -> None:
        """ Highlight the face and set the mouse cursor for the mouse's current location.

        Parameters
        ----------
        event: :class:`tkinter.Event` or ``None``
            The tkinter mouse event. Provides the current location of the mouse cursor. If ``None``
            is passed as the event (for example when this function is being called outside of a
            mouse event) then the location of the cursor will be calculated
        """
        if event is None:
            pnts = np.array((self._canvas.winfo_pointerx(), self._canvas.winfo_pointery()))
            pnts -= np.array((self._canvas.winfo_rootx(), self._canvas.winfo_rooty()))
        else:
            pnts = np.array((event.x, event.y))

        coords = (int(self._canvas.canvasx(pnts[0])), int(self._canvas.canvasy(pnts[1])))
        face = self._viewport.face_from_point(*coords)
        frame_idx, face_idx = face[:2]

        if frame_idx == self._current_frame_index and face_idx == self._current_face_index:
            return

        is_zoomed = self._globals.is_zoomed
        if (-1 in face or (frame_idx == self._globals.frame_index
                           and (not is_zoomed or
                                (is_zoomed and face_idx == self._globals.face_index)))):
            self._clear()
            self._canvas.config(cursor="")
            self._current_frame_index = None
            self._current_face_index = None
            return

        logger.debug("Viewport hover: frame_idx: %s, face_idx: %s", frame_idx, face_idx)

        self._canvas.config(cursor="hand2")
        self._highlight(face[2:])
        self._current_frame_index = frame_idx
        self._current_face_index = face_idx

    def _clear(self) -> None:
        """ Hide the hover box when the mouse is not over a face. """
        if self._canvas.itemcget(self._box, "state") != "hidden":
            self._canvas.itemconfig(self._box, state="hidden")

    def _highlight(self, top_left: np.ndarray) -> None:
        """ Display the hover box around the face that the mouse is currently over.

        Parameters
        ----------
        top_left: :class:`np.ndarray`
            The top left point of the highlight box location
        """
        coords = (*top_left, *[x + self._size for x in top_left])
        self._canvas.coords(self._box, *coords)
        self._canvas.itemconfig(self._box, state="normal")
        self._canvas.tag_raise(self._box)

    def _select_frame(self) -> None:
        """ Select the face and the subsequent frame (in the editor view) when a face is clicked
        on in the :class:`Viewport`. """
        frame_id = self._current_frame_index
        is_zoomed = self._globals.is_zoomed
        logger.debug("Face clicked. Global frame index: %s, Current frame_id: %s, is_zoomed: %s",
                     self._globals.frame_index, frame_id, is_zoomed)
        if frame_id is None or (frame_id == self._globals.frame_index and not is_zoomed):
            return
        face_idx = self._current_face_index if is_zoomed else 0
        self._globals.set_face_index(face_idx)
        transport_id = self._grid.transport_index_from_frame(frame_id)
        logger.trace("frame_index: %s, transport_id: %s, face_idx: %s",
                     frame_id, transport_id, face_idx)
        if transport_id is None:
            return
        self._navigation.stop_playback()
        self._globals.var_transport_index.set(transport_id)
        self._viewport.move_active_to_top()
        self.on_hover(None)


@dataclass
class Asset:
    """ Holds all of the display assets identifiers for the active frame's face viewer objects

    Parameters
    ----------
    images: list[int]
        Indices for a frame's tk image ids displayed in the active frame
    meshes: list[dict[Literal["polygon", "line"], list[int]]]
        Indices for a frame's tk line/polygon object ids displayed in the active frame
    faces: list[:class:`~lib.align.detected_faces.DetectedFace`]
        DetectedFace objects that exist in the current frame
    boxes: list[int]
        Indices for a frame's bounding box object ids displayed in the active frame
    """
    images: list[int]
    """list[int]: Indices for a frame's tk image ids displayed in the active frame"""
    meshes: list[dict[T.Literal["polygon", "line"], list[int]]]
    """list[dict[Literal["polygon", "line"], list[int]]]:  Indices for a frame's tk line/polygon
    object ids displayed in the active frame"""
    faces: list[DetectedFace]
    """list[:class:`~lib.align.detected_faces.DetectedFace`]: DetectedFace objects that exist
    in the current frame"""
    boxes: list[int]
    """list[int]: Indices for a frame's bounding box object ids displayed in the active
    frame"""


class ActiveFrame():
    """ Handles the display of faces and annotations for the currently active frame.

    Parameters
    ----------
    canvas: :class:`tkinter.Canvas`
        The :class:`~tools.manual.faceviewer.frame.FacesViewer` canvas
    tk_edited_variable: :class:`tkinter.BooleanVar`
        The tkinter callback variable indicating that a face has been edited
    """
    def __init__(self, viewport: Viewport, tk_edited_variable: tk.BooleanVar) -> None:
        logger.debug(parse_class_init(locals()))
        self._objects = viewport._objects
        self._viewport = viewport
        self._grid = viewport._grid
        self._tk_faces = viewport._tk_faces
        self._canvas = viewport._canvas
        self._globals = viewport._canvas._globals
        self._navigation = viewport._canvas._display_frame.navigation
        self._last_execution: dict[T.Literal["frame_index", "size"],
                                   int] = {"frame_index": -1, "size": viewport.face_size}
        self._tk_vars: dict[T.Literal["selected_editor", "edited"],
                            tk.StringVar | tk.BooleanVar] = {
            "selected_editor": self._canvas._display_frame.tk_selected_action,
            "edited": tk_edited_variable}
        self._assets: Asset = Asset([], [], [], [])

        self._globals.var_update_active_viewport.trace_add("write",
                                                           lambda *e: self._reload_callback())
        tk_edited_variable.trace_add("write", lambda *e: self._update_on_edit())
        logger.debug("Initialized: %s", self.__class__.__name__)

    @property
    def frame_index(self) -> int:
        """ int: The frame index of the currently displayed frame. """
        return self._globals.frame_index

    @property
    def current_frame(self) -> np.ndarray:
        """ :class:`numpy.ndarray`: A BGR version of the frame currently being displayed. """
        return self._globals.current_frame.image

    @property
    def _size(self) -> int:
        """ int: The size of the thumbnails displayed in the viewport, in pixels. """
        return self._viewport.face_size

    @property
    def _optional_annotations(self) -> dict[T.Literal["mesh", "mask"], bool]:
        """ dict[Literal["mesh", "mask"], bool]: The currently selected optional
        annotations """
        return self._canvas.optional_annotations

    def _reload_callback(self) -> None:
        """ If a frame has changed, triggering the variable, then update the active frame. Return
        having done nothing if the variable is resetting. """
        if self._globals.var_update_active_viewport.get():
            self.reload_annotations()

    def reload_annotations(self) -> None:
        """ Handles the reloading of annotations for the currently active faces.

        Highlights the faces within the viewport of those faces that exist in the currently
        displaying frame. Applies annotations based on the optional annotations and current
        editor selections.
        """
        logger.trace("Reloading annotations")  # type:ignore[attr-defined]
        if self._assets.images:
            self._clear_previous()

        self._set_active_objects()
        self._check_active_in_view()

        if not self._assets.images:
            logger.trace("No active faces. Returning")  # type:ignore[attr-defined]
            self._last_execution["frame_index"] = self.frame_index
            return

        if self._last_execution["frame_index"] != self.frame_index:
            self.move_to_top()
        self._create_new_boxes()

        self._update_face()
        self._canvas.tag_raise("active_highlighter")
        self._globals.var_update_active_viewport.set(False)
        self._last_execution["frame_index"] = self.frame_index

    def _clear_previous(self) -> None:
        """ Reverts the previously selected annotations to their default state. """
        logger.trace("Clearing previous active frame")  # type:ignore[attr-defined]
        self._canvas.itemconfig("active_highlighter", state="hidden")

        for key in T.get_args(T.Literal["polygon", "line"]):
            tag = f"active_mesh_{key}"
            self._canvas.itemconfig(tag, **self._viewport.mesh_kwargs[key], width=1)
            self._canvas.dtag(tag)

        if self._viewport.selected_editor == "mask" and not self._optional_annotations["mask"]:
            for name, tk_face in self._tk_faces.items():
                if name.startswith(f"{self._last_execution['frame_index']}_"):
                    tk_face.update_mask(None)

    def _set_active_objects(self) -> None:
        """ Collect the objects that exist in the currently active frame from the main grid. """
        if self._grid.is_valid:
            rows, cols = np.where(self._objects.visible_grid[0] == self.frame_index)
            logger.trace("Setting active objects: (rows: %s, "  # type:ignore[attr-defined]
                         "columns: %s)", rows, cols)
            self._assets.images = self._objects.images[rows, cols].tolist()
            self._assets.meshes = self._objects.meshes[rows, cols].tolist()
            self._assets.faces = self._objects.visible_faces[rows, cols].tolist()
        else:
            logger.trace("No valid grid. Clearing active objects")  # type:ignore[attr-defined]
            self._assets.images = []
            self._assets.meshes = []
            self._assets.faces = []

    def _check_active_in_view(self) -> None:
        """  If the frame has changed, there are faces in the frame, but they don't appear in the
        viewport, then bring the active faces to the top of the viewport. """
        if (not self._assets.images and
                self._last_execution["frame_index"] != self.frame_index and
                self._grid.frame_has_faces(self.frame_index)):
            y_coord = self._grid.y_coord_from_frame(self.frame_index)
            logger.trace("Active not in view. Moving to: %s", y_coord)  # type:ignore[attr-defined]
            self._canvas.yview_moveto(y_coord / self._canvas.bbox("backdrop")[3])
            self._viewport.update()

    def move_to_top(self) -> None:
        """ Move the currently selected frame's faces to the top of the viewport if they are moving
        off the bottom of the viewer. """
        height = self._canvas.bbox("backdrop")[3]
        bot = int(self._canvas.coords(self._assets.images[-1])[1] + self._size)

        y_top, y_bot = (int(round(pnt * height)) for pnt in self._canvas.yview())

        if y_top < bot < y_bot:  # bottom face is still in fully visible area
            logger.trace("Active faces in frame. Returning")  # type:ignore[attr-defined]
            return

        top = int(self._canvas.coords(self._assets.images[0])[1])
        if y_top == top:
            logger.trace("Top face already on top row. Returning")  # type:ignore[attr-defined]
            return

        if self._canvas.winfo_height() > self._size:
            logger.trace("Viewport taller than single face height. "  # type:ignore[attr-defined]
                         "Moving Active faces to top: %s", top)
            self._canvas.yview_moveto(top / height)
            self._viewport.update()
        elif self._canvas.winfo_height() <= self._size and y_top != top:
            logger.trace("Viewport shorter than single face height. "  # type:ignore[attr-defined]
                         "Moving Active faces to top: %s", top)
            self._canvas.yview_moveto(top / height)
            self._viewport.update()

    def _create_new_boxes(self) -> None:
        """ The highlight boxes (border around selected faces) are the only additional annotations
        that are required for the highlighter. If more faces are displayed in the current frame
        than highlight boxes are available, then new boxes are created to accommodate the
        additional faces. """
        new_boxes_count = max(0, len(self._assets.images) - len(self._assets.boxes))
        if new_boxes_count == 0:
            return
        logger.debug("new_boxes_count: %s", new_boxes_count)
        for _ in range(new_boxes_count):
            box = self._canvas.create_rectangle(0.,  # type:ignore[call-overload]
                                                0.,
                                                float(self._viewport.face_size),
                                                float(self._viewport.face_size),
                                                outline="#00FF00",
                                                width=2,
                                                state="hidden",
                                                tags=["active_highlighter"])
            logger.trace("Created new highlight_box: %s", box)  # type:ignore[attr-defined]
            self._assets.boxes.append(box)

    def _update_on_edit(self) -> None:
        """ Update the active faces on a frame edit. """
        if not self._tk_vars["edited"].get():
            return
        self._set_active_objects()
        self._update_face()
        assert isinstance(self._tk_vars["edited"], tk.BooleanVar)
        self._tk_vars["edited"].set(False)

    def _update_face(self) -> None:
        """ Update the highlighted annotations for faces in the currently selected frame. """
        for face_idx, (image_id, mesh_ids, box_id, det_face), in enumerate(
                zip(self._assets.images,
                    self._assets.meshes,
                    self._assets.boxes,
                    self._assets.faces)):
            if det_face is None:
                continue
            top_left = self._canvas.coords(image_id)
            coords = [*top_left, *[x + self._size for x in top_left]]
            tk_face = self._viewport.get_tk_face(self.frame_index, face_idx, det_face)
            self._canvas.itemconfig(image_id, image=tk_face.photo)
            self._show_box(box_id, coords)
            self._show_mesh(mesh_ids, face_idx, det_face, top_left)
        self._last_execution["size"] = self._viewport.face_size

    def _show_box(self, item_id: int, coordinates: list[float]) -> None:
        """ Display the highlight box around the given coordinates.

        Parameters
        ----------
        item_id: int
            The tkinter canvas object identifier for the highlight box
        coordinates: list[float]
            The (x, y, x1, y1) coordinates of the top left corner of the box
        """
        self._canvas.coords(item_id, *coordinates)
        self._canvas.itemconfig(item_id, state="normal")

    def _show_mesh(self,
                   mesh_ids: dict[T.Literal["polygon", "line"], list[int]],
                   face_index: int,
                   detected_face: DetectedFace,
                   top_left: list[float]) -> None:
        """ Display the mesh annotation for the given face, at the given location.

        Parameters
        ----------
        mesh_ids: dict[Literal["polygon", "line"], list[int]]
            Dictionary containing the `polygon` and `line` tkinter canvas identifiers that make up
            the mesh for the given face
        face_index: int
            The face index within the frame for the given face
        detected_face: :class:`~lib.align.DetectedFace`
            The detected face object that contains the landmarks for generating the mesh
        top_left: list[float]
            The (x, y) top left co-ordinates of the mesh's bounding box
        """
        state = "normal" if (self._tk_vars["selected_editor"].get() != "Mask" or
                             self._optional_annotations["mesh"]) else "hidden"
        kwargs: dict[T.Literal["polygon", "line"], dict[str, T.Any]] = {
            "polygon": {"fill": "", "width": 2, "outline": self._canvas.control_colors["Mesh"]},
            "line": {"fill": self._canvas.control_colors["Mesh"], "width": 2}}

        assert isinstance(self._tk_vars["edited"], tk.BooleanVar)
        edited = (self._tk_vars["edited"].get() and
                  self._tk_vars["selected_editor"].get() not in ("Mask", "View"))
        landmarks = self._viewport.get_landmarks(self.frame_index,
                                                 face_index,
                                                 detected_face,
                                                 top_left,
                                                 edited)
        for key, kwarg in kwargs.items():
            if key not in mesh_ids:
                continue
            for idx, mesh_id in enumerate(mesh_ids[key]):
                self._canvas.coords(mesh_id, *landmarks[key][idx].flatten())
                self._canvas.itemconfig(mesh_id, state=state, **kwarg)
                self._canvas.addtag_withtag(f"active_mesh_{key}", mesh_id)
