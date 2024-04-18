#!/usr/bin/env python3
""" Handles the visible area of the :class:`~tools.manual.faceviewer.frame.FacesViewer` canvas. """
from __future__ import annotations
import logging
import tkinter as tk
import typing as T

import cv2
import numpy as np
from PIL import Image, ImageTk

from lib.align import AlignedFace, LANDMARK_PARTS, LandmarkType
from lib.logger import parse_class_init

from .interact import ActiveFrame, HoverBox

if T.TYPE_CHECKING:
    from lib.align import CenteringType, DetectedFace
    from .frame import FacesViewer

logger = logging.getLogger(__name__)


class Viewport():
    """ Handles the display of faces and annotations in the currently viewable area of the canvas.

    Parameters
    ----------
    canvas: :class:`tkinter.Canvas`
        The :class:`~tools.manual.faceviewer.frame.FacesViewer` canvas
    tk_edited_variable: :class:`tkinter.BooleanVar`
        The variable that indicates that a face has been edited
    """
    def __init__(self, canvas: FacesViewer, tk_edited_variable: tk.BooleanVar) -> None:
        logger.debug(parse_class_init(locals()))
        self._canvas = canvas
        self._grid = canvas.layout
        self._centering: CenteringType = "face"
        self._tk_selected_editor = canvas._display_frame.tk_selected_action
        self._landmarks: dict[str, dict[T.Literal["polygon", "line"], list[np.ndarray]]] = {}
        self._tk_faces: dict[str, TKFace] = {}
        self._objects = VisibleObjects(self)
        self._hoverbox = HoverBox(self)
        self._active_frame = ActiveFrame(self, tk_edited_variable)
        self._tk_selected_editor.trace(
            "w", lambda *e: self._active_frame.reload_annotations())
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def face_size(self) -> int:
        """ int: The pixel size of each thumbnail """
        return self._grid.face_size

    @property
    def mesh_kwargs(self) -> dict[T.Literal["polygon", "line"], dict[str, T.Any]]:
        """ dict[Literal["polygon", "line"], str | int]: Dynamic keyword arguments defining the
        color and state for the objects that make up a single face's mesh annotation based on the
        current user selected options. Values are the keyword arguments for that given type. """
        state = "normal" if self._canvas.optional_annotations["mesh"] else "hidden"
        color = self._canvas.control_colors["Mesh"]
        return {"polygon": {"fill": "", "outline": color, "state": state},
                "line": {"fill": color, "state": state}}

    @property
    def hover_box(self) -> HoverBox:
        """ :class:`HoverBox`: The hover box for the viewport. """
        return self._hoverbox

    @property
    def selected_editor(self) -> str:
        """ str: The currently selected editor. """
        return self._tk_selected_editor.get().lower()

    def toggle_mesh(self, state: T.Literal["hidden", "normal"]) -> None:
        """ Toggles the mesh optional annotations on and off.

        Parameters
        ----------
        state: Literal["hidden", "normal"]
            The state to set the mesh annotations to
        """
        logger.debug("Toggling mesh annotations to: %s", state)
        self._canvas.itemconfig("viewport_mesh", state=state)
        self.update()

    def toggle_mask(self, state: T.Literal["hidden", "normal"], mask_type: str) -> None:
        """ Toggles the mask optional annotation on and off.

        Parameters
        ----------
        state: Literal["hidden", "normal"]
            Whether the mask should be displayed or hidden
        mask_type: str
            The type of mask to overlay onto the face
        """
        logger.debug("Toggling mask annotations to: %s. mask_type: %s", state, mask_type)
        for (frame_idx, face_idx), det_face in zip(
                self._objects.visible_grid[:2].transpose(1, 2, 0).reshape(-1, 2),
                self._objects.visible_faces.flatten()):
            if frame_idx == -1:
                continue

            key = "_".join([str(frame_idx), str(face_idx)])
            mask = None if state == "hidden" else self._obtain_mask(det_face, mask_type)
            self._tk_faces[key].update_mask(mask)
        self.update()

    @classmethod
    def _obtain_mask(cls, detected_face: DetectedFace, mask_type: str) -> np.ndarray | None:
        """ Obtain the mask for the correct "face" centering that is used in the thumbnail display.

        Parameters
        -----------
        detected_face: :class:`lib.align.DetectedFace`
            The Detected Face object to obtain the mask for
        mask_type: str
            The type of mask to obtain

        Returns
        -------
        :class:`numpy.ndarray` or ``None``
            The single channel mask of requested mask type, if it exists, otherwise ``None``
        """
        mask = detected_face.mask.get(mask_type)
        if not mask:
            return None
        if mask.stored_centering != "face":
            face = AlignedFace(detected_face.landmarks_xy)
            mask.set_sub_crop(face.pose.offset[mask.stored_centering],
                              face.pose.offset["face"],
                              centering="face")
        return mask.mask.squeeze()

    def reset(self) -> None:
        """ Reset all the cached objects on a face size change. """
        self._landmarks = {}
        self._tk_faces = {}

    def update(self, refresh_annotations: bool = False) -> None:
        """ Update the viewport.

        Parameters
        ----------
        refresh_annotations: bool, optional
            ``True`` if mesh annotations should be re-calculated otherwise ``False``.
            Default: ``False``

        Obtains the objects that are currently visible. Updates the visible area of the canvas
        and reloads the active frame's annotations. """
        self._objects.update()
        self._update_viewport(refresh_annotations)
        self._active_frame.reload_annotations()

    def _update_viewport(self, refresh_annotations: bool) -> None:
        """ Update the viewport

        Parameters
        ----------
        refresh_annotations: bool
            ``True`` if mesh annotations should be re-calculated otherwise ``False``

        Clear out cached objects that are not currently in view. Populate the cache for any
        faces that are now in view. Populate the correct face image and annotations for each
        object in the viewport based on current location. If optional mesh annotations are
        enabled, then calculates newly displayed meshes. """
        if not self._grid.is_valid:
            return
        self._discard_tk_faces()

        for collection in zip(self._objects.visible_grid.transpose(1, 2, 0),
                              self._objects.images,
                              self._objects.meshes,
                              self._objects.visible_faces):
            for (frame_idx, face_idx, pnt_x, pnt_y), image_id, mesh_ids, face in zip(*collection):
                if frame_idx == self._active_frame.frame_index and not refresh_annotations:
                    logger.trace("Skipping active frame: %s",  # type:ignore[attr-defined]
                                 frame_idx)
                    continue
                if frame_idx == -1:
                    logger.trace("Blanking non-existant face")  # type:ignore[attr-defined]
                    self._canvas.itemconfig(image_id, image="")
                    for area in mesh_ids.values():
                        for mesh_id in area:
                            self._canvas.itemconfig(mesh_id, state="hidden")
                    continue

                tk_face = self.get_tk_face(frame_idx, face_idx, face)
                self._canvas.itemconfig(image_id, image=tk_face.photo)

                if (self._canvas.optional_annotations["mesh"]
                        or frame_idx == self._active_frame.frame_index
                        or refresh_annotations):
                    landmarks = self.get_landmarks(frame_idx, face_idx, face, [pnt_x, pnt_y],
                                                   refresh=True)
                    self._locate_mesh(mesh_ids, landmarks)

    def _discard_tk_faces(self) -> None:
        """ Remove any :class:`TKFace` objects from the cache that are not currently displayed. """
        keys = [f"{pnt_x}_{pnt_y}"
                for pnt_x, pnt_y in self._objects.visible_grid[:2].T.reshape(-1, 2)]
        for key in list(self._tk_faces):
            if key not in keys:
                del self._tk_faces[key]
        logger.trace("keys: %s allocated_faces: %s",  # type:ignore[attr-defined]
                     keys, len(self._tk_faces))

    def get_tk_face(self, frame_index: int, face_index: int, face: DetectedFace) -> TKFace:
        """ Obtain the :class:`TKFace` object for the given face from the cache. If the face does
        not exist in the cache, then it is generated and added prior to returning.

        Parameters
        ----------
        frame_index: int
            The frame index to obtain the face for
        face_index: int
            The face index of the face within the requested frame
        face: :class:`~lib.align.DetectedFace`
            The detected face object, containing the thumbnail jpg

        Returns
        -------
        :class:`TKFace`
            An object for displaying in the faces viewer canvas populated with the aligned mesh
            landmarks and face thumbnail
        """
        is_active = frame_index == self._active_frame.frame_index
        key = "_".join([str(frame_index), str(face_index)])
        if key not in self._tk_faces or is_active:
            logger.trace("creating new tk_face: (key: %s, "  # type:ignore[attr-defined]
                         "is_active: %s)", key, is_active)
            if is_active:
                image = AlignedFace(face.landmarks_xy,
                                    image=self._active_frame.current_frame,
                                    centering=self._centering,
                                    size=self.face_size).face
            else:
                thumb = face.thumbnail
                assert thumb is not None
                image = AlignedFace(face.landmarks_xy,
                                    image=cv2.imdecode(thumb, cv2.IMREAD_UNCHANGED),
                                    centering=self._centering,
                                    size=self.face_size,
                                    is_aligned=True).face
            assert image is not None
            tk_face = self._get_tk_face_object(face, image, is_active)
            self._tk_faces[key] = tk_face
        else:
            logger.trace("tk_face exists: %s", key)  # type:ignore[attr-defined]
            tk_face = self._tk_faces[key]
        return tk_face

    def _get_tk_face_object(self,
                            face: DetectedFace,
                            image: np.ndarray,
                            is_active: bool) -> TKFace:
        """ Obtain an existing unallocated, or a newly created :class:`TKFace` and populate it with
        face information from the requested frame and face index.

        If the face is currently active, then the face is generated from the currently displayed
        frame, otherwise it is generated from the jpg thumbnail.

        Parameters
        ----------
        face: :class:`lib.align.DetectedFace`
            A detected face object to create the :class:`TKFace` from
        image: :class:`numpy.ndarray`
            The jpg thumbnail or the 3 channel image for the face
        is_active: bool
            ``True`` if the face in the currently active frame otherwise ``False``

        Returns
        -------
        :class:`TKFace`
            An object for displaying in the faces viewer canvas populated with the aligned face
            image with a mask applied, if required.
        """
        get_mask = (self._canvas.optional_annotations["mask"] or
                    (is_active and self.selected_editor == "mask"))
        mask = self._obtain_mask(face, self._canvas.selected_mask) if get_mask else None
        tk_face = TKFace(image, size=self.face_size, mask=mask)
        logger.trace("face: %s, tk_face: %s", face, tk_face)  # type:ignore[attr-defined]
        return tk_face

    def get_landmarks(self,
                      frame_index: int,
                      face_index: int,
                      face: DetectedFace,
                      top_left: list[float],
                      refresh: bool = False
                      ) -> dict[T.Literal["polygon", "line"], list[np.ndarray]]:
        """ Obtain the landmark points for each mesh annotation.

        First tries to obtain the aligned landmarks from the cache. If the landmarks do not exist
        in the cache, or a refresh has been requested, then the landmarks are calculated from the
        detected face object.

        Parameters
        ----------
        frame_index: int
            The frame index to obtain the face for
        face_index: int
            The face index of the face within the requested frame
        face: :class:`lib.align.DetectedFace`
            The detected face object to obtain landmarks for
        top_left: list[float]
            The top left (x, y) points of the face's bounding box within the viewport
        refresh: bool, optional
            Whether to force a reload of the face's aligned landmarks, even if they already exist
            within the cache. Default: ``False``

        Returns
        -------
        dict
            The key is the tkinter canvas object type for each part of the mesh annotation
            (`polygon`, `line`). The value is a list containing the (x, y) coordinates of each
            part of the mesh annotation, from the top left corner location.
        """
        key = f"{frame_index}_{face_index}"
        landmarks = self._landmarks.get(key, None)
        if not landmarks or refresh:
            aligned = AlignedFace(face.landmarks_xy,
                                  centering=self._centering,
                                  size=self.face_size)
            landmarks = {"polygon": [], "line": []}
            for start, end, fill in LANDMARK_PARTS[aligned.landmark_type].values():
                points = aligned.landmarks[start:end] + top_left
                shape: T.Literal["polygon", "line"] = "polygon" if fill else "line"
                landmarks[shape].append(points)
            self._landmarks[key] = landmarks
        return landmarks

    def _locate_mesh(self, mesh_ids, landmarks):
        """ Place the mesh annotation canvas objects in the correct location.

        Parameters
        ----------
        mesh_ids: list
            The list of mesh id objects to set coordinates for
        landmarks: dict
            The mesh point groupings and whether each group should be a line or a polygon
        """
        for key, area in landmarks.items():
            if key not in mesh_ids:
                continue
            for coords, mesh_id in zip(area, mesh_ids[key]):
                self._canvas.coords(mesh_id, *coords.flatten())

    def face_from_point(self, point_x: int, point_y: int) -> np.ndarray:
        """ Given an (x, y) point on the :class:`Viewport`, obtain the face information at that
        location.

        Parameters
        ----------
        point_x: int
            The x position on the canvas of the point to retrieve the face for
        point_y: int
            The y position on the canvas of the point to retrieve the face for

        Returns
        -------
        :class:`numpy.ndarray`
            Array of shape (4, ) containing the (`frame index`, `face index`, `x_point of top left
            corner`, `y point of top left corner`) of the face at the given coordinates.

            If the given coordinates are not over a face, then the frame and face indices will be
            -1
        """
        if not self._grid.is_valid or point_x > self._grid.dimensions[0]:
            retval = np.array((-1, -1, -1, -1))
        else:
            x_idx = np.searchsorted(self._objects.visible_grid[2, 0, :], point_x, side="left") - 1
            y_idx = np.searchsorted(self._objects.visible_grid[3, :, 0], point_y, side="left") - 1
            if x_idx < 0 or y_idx < 0:
                retval = np.array((-1, -1, -1, -1))
            else:
                retval = self._objects.visible_grid[:, y_idx, x_idx]
        logger.trace(retval)  # type:ignore[attr-defined]
        return retval

    def move_active_to_top(self) -> None:
        """ Check whether the active frame is going off the bottom of the viewport, if so: move it
        to the top of the viewport. """
        self._active_frame.move_to_top()


class Recycler:
    """ Tkinter can slow down when constantly creating new objects.

    This class delivers recycled objects, if stale objects are available, otherwise creates a new
    object

    Parameters
    ----------
    :class:`~tools.manual.faceviewe.frame.FacesViewer`
        The canvas that holds the faces display
    """
    def __init__(self, canvas: FacesViewer) -> None:
        self._canvas = canvas
        self._assets: dict[T.Literal["image", "line", "polygon"],
                           list[int]] = {"image": [], "line": [], "polygon": []}
        self._mesh_methods: dict[T.Literal["line", "polygon"],
                                 T.Callable] = {"line": canvas.create_line,
                                                "polygon": canvas.create_polygon}

    def recycle_assets(self, asset_ids: list[int]) -> None:
        """ Recycle assets that are no longer required

        Parameters
        ----------
        asset_ids: list[int]
            The IDs of the assets to be recycled
        """
        logger.trace("Recycling %s objects", len(asset_ids))  # type:ignore[attr-defined]
        for asset_id in asset_ids:
            asset_type = self._canvas.type(asset_id)
            assert asset_type in self._assets
            coords = (0, 0, 0, 0) if asset_type == "line" else (0, 0)
            self._canvas.coords(asset_id, *coords)

            if asset_type == "image":
                self._canvas.itemconfig(asset_id, image="")

            self._assets[asset_type].append(asset_id)
        logger.trace("Recycled objects: %s", self._assets)  # type:ignore[attr-defined]

    def get_image(self, coordinates: tuple[float | int, float | int]) -> int:
        """ Obtain a recycled or new image object ID

        Parameters
        ----------
        coordinates: tuple[float | int, float | int]
            The co-ordinates that the image should be displayed at

        Returns
        -------
        int
            The canvas object id for the created image
        """
        if self._assets["image"]:
            retval = self._assets["image"].pop()
            self._canvas.coords(retval, *coordinates)
            logger.trace("Recycled image: %s", retval)  # type:ignore[attr-defined]
        else:
            retval = self._canvas.create_image(*coordinates,
                                               anchor=tk.NW,
                                               tags=["viewport", "viewport_image"])
            logger.trace("Created new image: %s", retval)  # type:ignore[attr-defined]
        return retval

    def get_mesh(self, face: DetectedFace) -> dict[T.Literal["polygon", "line"], list[int]]:
        """ Get the mesh annotation for the landmarks. This is made up of a series of polygons
        or lines, depending on which part of the face is being annotated. Creates a new series of
        objects, or pulls existing objects from the recycled objects pool if they are available.

        Parameters
        ----------
        face: :class:`~lib.align.detected_face.DetectedFace`
            The detected face object to obrain the mesh for

        Returns
        -------
        dict[Literal["polygon", "line"], list[int]]
            The dictionary of line and polygon tkinter canvas object ids for the mesh annotation
        """
        mesh_kwargs = self._canvas.viewport.mesh_kwargs
        mesh_parts = LANDMARK_PARTS[LandmarkType.from_shape(face.landmarks_xy.shape)]
        retval: dict[T.Literal["polygon", "line"], list[int]] = {}
        for _, _, fill in mesh_parts.values():
            asset_type: T.Literal["polygon", "line"] = "polygon" if fill else "line"
            kwargs = mesh_kwargs[asset_type]
            if self._assets[asset_type]:
                asset_id = self._assets[asset_type].pop()
                self._canvas.itemconfig(asset_id, **kwargs)
                logger.trace("Recycled mesh %s: %s",  # type:ignore[attr-defined]
                             asset_type, asset_id)
            else:
                coords = (0, 0) if asset_type == "polygon" else (0, 0, 0, 0)
                tags = ["viewport", "viewport_mesh", f"viewport_{asset_type}"]
                asset_id = self._mesh_methods[asset_type](coords, width=1, tags=tags, **kwargs)
                logger.trace("Created new mesh %s: %s",  # type:ignore[attr-defined]
                             asset_type, asset_id)

            retval.setdefault(asset_type, []).append(asset_id)
        logger.trace("Got mesh: %s", retval)  # type:ignore[attr-defined]
        return retval


class VisibleObjects():
    """ Holds the objects from the :class:`~tools.manual.faceviewer.frame.Grid` that appear in the
    viewable area of the :class:`Viewport`.

    Parameters
    ----------
    viewport: :class:`Viewport`
        The viewport object for the :class:`~tools.manual.faceviewer.frame.FacesViewer` canvas
    """
    def __init__(self, viewport: Viewport) -> None:
        logger.debug(parse_class_init(locals()))
        self._viewport = viewport
        self._canvas = viewport._canvas
        self._grid = viewport._grid
        self._size = viewport.face_size

        self._visible_grid = np.zeros((4, 0, 0))
        self._visible_faces = np.zeros((0, 0))
        self._recycler = Recycler(self._canvas)
        self._images = np.zeros((0, 0), dtype=np.int64)
        self._meshes = np.zeros((0, 0))
        logger.debug("Initialized: %s", self.__class__.__name__)

    @property
    def visible_grid(self) -> np.ndarray:
        """ :class:`numpy.ndarray`: The currently visible section of the
        :class:`~tools.manual.faceviewer.frame.Grid`

        A numpy array of shape (`4`, `rows`, `columns`) corresponding to the viewable area of the
        display grid. 1st dimension contains frame indices, 2nd dimension face indices. The 3rd and
        4th dimension contain the x and y position of the top left corner of the face respectively.

        Any locations that are not populated by a face will have a frame and face index of -1. """
        return self._visible_grid

    @property
    def visible_faces(self) -> np.ndarray:
        """ :class:`numpy.ndarray`: The currently visible :class:`~lib.align.DetectedFace`
        objects.

        A numpy array of shape (`rows`, `columns`) corresponding to the viewable area of the
        display grid and containing the detected faces at their currently viewable position.

        Any locations that are not populated by a face will have ``None`` in it's place. """
        return self._visible_faces

    @property
    def images(self) -> np.ndarray:
        """ :class:`numpy.ndarray`: The viewport's tkinter canvas image objects.

        A numpy array of shape (`rows`, `columns`) corresponding to the viewable area of the
        display grid and containing the tkinter canvas image object for the face at the
        corresponding location. """
        return self._images

    @property
    def meshes(self) -> np.ndarray:
        """ :class:`numpy.ndarray`: The viewport's tkinter canvas mesh annotation objects.

        A numpy array of shape (`rows`, `columns`) corresponding to the viewable area of the
        display grid and containing a dictionary of the corresponding tkinter polygon and line
        objects required to build a face's mesh annotation for the face at the corresponding
        location. """
        return self._meshes

    @property
    def _top_left(self) -> np.ndarray:
        """ :class:`numpy.ndarray`: The canvas (`x`, `y`) position of the face currently in the
        viewable area's top left position. """
        if not np.any(self._images):
            retval = [0.0, 0.0]
        else:
            retval = self._canvas.coords(self._images[0][0])
        return np.array(retval, dtype="int")

    def update(self) -> None:
        """ Load and unload thumbnails in the visible area of the faces viewer. """
        if self._canvas.optional_annotations["mesh"]:  # Display any hidden end of row meshes
            self._canvas.itemconfig("viewport_mesh", state="normal")

        self._visible_grid, self._visible_faces = self._grid.visible_area
        if (np.any(self._images) and np.any(self._visible_grid)
                and self._visible_grid.shape[1:] != self._images.shape):
            self._reset_viewport()

        required_rows = self._visible_grid.shape[1] if self._grid.is_valid else 0
        existing_rows = len(self._images)
        logger.trace("existing_rows: %s. required_rows: %s",  # type:ignore[attr-defined]
                     existing_rows, required_rows)

        if existing_rows > required_rows:
            self._remove_rows(existing_rows, required_rows)
        if existing_rows < required_rows:
            self._add_rows(existing_rows, required_rows)

        self._shift()

    def _reset_viewport(self) -> None:
        """ Reset all objects in the viewport on a column count change. Reset the viewport size
        to the newly specified face size. """
        logger.debug("Resetting Viewport")
        self._size = self._viewport.face_size
        images = self._images.flatten().tolist()
        meshes = [parts for mesh in [mesh.values() for mesh in self._meshes.flatten()]
                  for parts in mesh]
        mesh_ids = [asset for mesh in meshes for asset in mesh]
        self._recycler.recycle_assets(images + mesh_ids)
        self._images = np.zeros((0, 0), np.int64)
        self._meshes = np.zeros((0, 0))

    def _remove_rows(self, existing_rows: int, required_rows: int) -> None:
        """ Remove and recycle rows from the viewport that are not in the view area.

        Parameters
        ----------
        existing_rows: int
            The number of existing rows within the viewport
        required_rows: int
            The number of rows required by the viewport
        """
        logger.debug("Removing rows from viewport: (existing_rows: %s, required_rows: %s)",
                     existing_rows, required_rows)
        images = self._images[required_rows: existing_rows].flatten().tolist()
        meshes = [parts
                  for mesh in [mesh.values()
                               for mesh in self._meshes[required_rows: existing_rows].flatten()]
                  for parts in mesh]
        mesh_ids = [asset for mesh in meshes for asset in mesh]
        self._recycler.recycle_assets(images + mesh_ids)
        self._images = self._images[:required_rows]
        self._meshes = self._meshes[:required_rows]
        logger.trace("self._images: %s, self._meshes: %s",  # type:ignore[attr-defined]
                     self._images.shape, self._meshes.shape)

    def _add_rows(self, existing_rows: int, required_rows: int) -> None:
        """ Add rows to the viewport.

        Parameters
        ----------
        existing_rows: int
            The number of existing rows within the viewport
        required_rows: int
            The number of rows required by the viewport
        """
        logger.debug("Adding rows to viewport: (existing_rows: %s, required_rows: %s)",
                     existing_rows, required_rows)
        columns = self._grid.columns_rows[0]

        base_coords: list[list[float | int]]

        if not np.any(self._images):
            base_coords = [[col * self._size, 0] for col in range(columns)]
        else:
            base_coords = [self._canvas.coords(item_id) for item_id in self._images[0]]
        logger.trace("existing rows: %s, required_rows: %s, "  # type:ignore[attr-defined]
                     "base_coords: %s", existing_rows, required_rows, base_coords)
        images = []
        meshes = []
        for row in range(existing_rows, required_rows):
            y_coord = base_coords[0][1] + (row * self._size)
            images.append([self._recycler.get_image((coords[0], y_coord))
                           for coords in base_coords])
            meshes.append([{} if face is None else self._recycler.get_mesh(face)
                           for face in self._visible_faces[row]])

        a_images = np.array(images)
        a_meshes = np.array(meshes)

        if not np.any(self._images):
            logger.debug("Adding initial viewport objects: (image shapes: %s, mesh shapes: %s)",
                         a_images.shape, a_meshes.shape)
            self._images = a_images
            self._meshes = a_meshes
        else:
            logger.debug("Adding new viewport objects: (image shapes: %s, mesh shapes: %s)",
                         a_images.shape, a_meshes.shape)
            self._images = np.concatenate((self._images, a_images))
            self._meshes = np.concatenate((self._meshes, a_meshes))

        logger.trace("self._images: %s, self._meshes: %s",  # type:ignore[attr-defined]
                     self._images.shape, self._meshes.shape)

    def _shift(self) -> bool:
        """ Shift the viewport in the y direction if required

        Returns
        -------
        bool
            ``True`` if the viewport was shifted otherwise ``False``
        """
        current_y = self._top_left[1]
        required_y = self.visible_grid[3, 0, 0] if self._grid.is_valid else 0
        logger.trace("current_y: %s, required_y: %s",  # type:ignore[attr-defined]
                     current_y, required_y)
        if current_y == required_y:
            logger.trace("No move required")  # type:ignore[attr-defined]
            return False
        shift_amount = required_y - current_y
        logger.trace("Shifting viewport: %s", shift_amount)  # type:ignore[attr-defined]
        self._canvas.move("viewport", 0, shift_amount)
        return True


class TKFace():
    """ An object that holds a single :class:`tkinter.PhotoImage` face, ready for placement in the
    :class:`Viewport`, Handles the placement of and removal of masks for the face as well as
    updates on any edits.

    Parameters
    ----------
    face: :class:`numpy.ndarray`
        The face, sized correctly as a 3 channel BGR image or an encoded jpg to create a
        :class:`tkinter.PhotoImage` from
    size: int, optional
        The pixel size of the face image. Default: `128`
    mask: :class:`numpy.ndarray` or ``None``, optional
        The mask to be applied to the face image. Pass ``None`` if no mask is to be used.
        Default ``None``
    """
    def __init__(self, face: np.ndarray, size: int = 128, mask: np.ndarray | None = None) -> None:
        logger.trace(parse_class_init(locals()))  # type:ignore[attr-defined]
        self._size = size
        if face.ndim == 2 and face.shape[1] == 1:
            self._face = self._image_from_jpg(face)
        else:
            self._face = face[..., 2::-1]
        self._photo = ImageTk.PhotoImage(self._generate_tk_face_data(mask))

        logger.trace("Initialized %s", self.__class__.__name__)  # type:ignore[attr-defined]

    # << PUBLIC PROPERTIES >> #
    @property
    def photo(self) -> tk.PhotoImage:
        """ :class:`tkinter.PhotoImage`: The face in a format that can be placed on the
         :class:`~tools.manual.faceviewer.frame.FacesViewer` canvas. """
        return self._photo

    # << PUBLIC METHODS >> #
    def update(self, face: np.ndarray, mask: np.ndarray) -> None:
        """ Update the :attr:`photo` with the given face and mask.

        Parameters
        ----------
        face: :class:`numpy.ndarray`
            The face, sized correctly as a 3 channel BGR image
        mask: :class:`numpy.ndarray` or ``None``
            The mask to be applied to the face image. Pass ``None`` if no mask is to be used
        """
        self._face = face[..., 2::-1]
        self._photo.paste(self._generate_tk_face_data(mask))

    def update_mask(self, mask: np.ndarray | None) -> None:
        """ Update the mask in the 4th channel of :attr:`photo` to the given mask.

        Parameters
        ----------
        mask: :class:`numpy.ndarray` or ``None``
            The mask to be applied to the face image. Pass ``None`` if no mask is to be used
        """
        self._photo.paste(self._generate_tk_face_data(mask))

    # << PRIVATE METHODS >> #
    def _image_from_jpg(self, face: np.ndarray) -> np.ndarray:
        """ Convert an encoded jpg into 3 channel BGR image.

        Parameters
        ----------
        face: :class:`numpy.ndarray`
            The encoded jpg as a two dimension numpy array

        Returns
        -------
        :class:`numpy.ndarray`
            The decoded jpg as a 3 channel BGR image
        """
        face = cv2.imdecode(face, cv2.IMREAD_UNCHANGED)
        interp = cv2.INTER_CUBIC if face.shape[0] < self._size else cv2.INTER_AREA
        if face.shape[0] != self._size:
            face = cv2.resize(face, (self._size, self._size), interpolation=interp)
        return face[..., 2::-1]

    def _generate_tk_face_data(self, mask: np.ndarray | None) -> tk.PhotoImage:
        """ Create the :class:`tkinter.PhotoImage` from the currant :attr:`_face`.

        Parameters
        ----------
        mask: :class:`numpy.ndarray` or ``None``
            The mask to add to the image. ``None`` if a mask is not being used

        Returns
        -------
        :class:`tkinter.PhotoImage`
            The face formatted for the  :class:`~tools.manual.faceviewer.frame.FacesViewer` canvas.
        """
        mask = np.ones(self._face.shape[:2], dtype="uint8") * 255 if mask is None else mask
        if mask.shape[0] != self._size:
            mask = cv2.resize(mask, self._face.shape[:2], interpolation=cv2.INTER_AREA)
        img = np.concatenate((self._face, mask[..., None]), axis=-1)
        return Image.fromarray(img)
