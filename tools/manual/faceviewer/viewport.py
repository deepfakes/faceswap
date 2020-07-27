#!/usr/bin/env python3
""" Handles the visible area of the :class:`~tools.manual.faceviewer.frame.FacesViewer` canvas. """

import logging
import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Viewport():
    """ Handles the display of faces and annotations in the currently viewable area of the canvas.

    Parameters
    ----------
    canvas: :class:`tkinter.Canvas`
        The :class:`~tools.manual.faceviewer.frame.FacesViewer` canvas
    tk_edited_variable: :class:`tkinter.BooleanVar`
        The variable that indicates that a face has been edited
    """
    def __init__(self, canvas, tk_edited_variable):
        logger.debug("Initializing: %s: (canvas: %s, tk_edited_variable: %s)",
                     self.__class__.__name__, canvas, tk_edited_variable)
        self._canvas = canvas
        self._grid = canvas.grid
        self._tk_selected_editor = canvas._display_frame.tk_selected_action
        self._landmark_mapping = dict(mouth_inner=(60, 68),
                                      mouth_outer=(48, 60),
                                      right_eyebrow=(17, 22),
                                      left_eyebrow=(22, 27),
                                      right_eye=(36, 42),
                                      left_eye=(42, 48),
                                      nose=(27, 36),
                                      jaw=(0, 17),
                                      chin=(8, 11))
        self._landmarks = dict()
        self._tk_faces = dict()
        self._objects = VisibleObjects(self)
        self._hoverbox = HoverBox(self)
        self._active_frame = ActiveFrame(self, tk_edited_variable)
        self._tk_selected_editor.trace(
            "w", lambda *e: self._active_frame.reload_annotations())

    @property
    def face_size(self):
        """ int: The pixel size of each thumbnail """
        return self._grid.face_size

    @property
    def mesh_kwargs(self):
        """ dict: The color and state keyword arguments for the objects that make up a single
        face's mesh annotation based on the current user selected options. Key is the object
        type (`polygon` or `line`), value are the keyword arguments for that type. """
        state = "normal" if self._canvas.optional_annotations["mesh"] else "hidden"
        color = self._canvas.get_muted_color("Mesh")
        kwargs = dict(polygon=dict(fill="", outline=color, state=state),
                      line=dict(fill=color, state=state))
        return kwargs

    @property
    def hover_box(self):
        """ :class:`HoverBox`: The hover box for the viewport. """
        return self._hoverbox

    @property
    def selected_editor(self):
        """ str: The currently selected editor. """
        return self._tk_selected_editor.get().lower()

    def toggle_mesh(self, state):
        """ Toggles the mesh optional annotations on and off.

        Parameters
        ----------
        state: ["hidden", "normal"]
            The state to set the mesh annotations to
        """
        logger.debug("Toggling mesh annotations to: %s", state)
        self._canvas.itemconfig("viewport_mesh", state=state)
        self.update()

    def toggle_mask(self, state, mask_type):
        """ Toggles the mask optional annotation on and off.

        Parameters
        ----------
        state: ["hidden", "normal"]
            Whether the mask should be displayed or hidden
        mask_type: str
            The type of mask to overlay onto the face
        """
        logger.debug("Toggling mask annotations to: %s. mask_type: %s", state, mask_type)
        for (frame_idx, face_idx), det_faces in zip(
                self._objects.visible_grid[:2].transpose(1, 2, 0).reshape(-1, 2),
                self._objects.visible_faces.flatten()):
            if frame_idx == -1:
                continue
            key = "_".join([str(frame_idx), str(face_idx)])
            mask = None if state == "hidden" else det_faces.mask.get(mask_type, None)
            mask = mask if mask is None else mask.mask.squeeze()
            self._tk_faces[key].update_mask(mask)
        self.update()

    def reset(self):
        """ Reset all the cached objects on a face size change. """
        self._landmarks = dict()
        self._tk_faces = dict()

    def update(self):
        """ Update the viewport.

        Obtains the objects that are currently visible. Updates the visible area of the canvas
        and reloads the active frame's annotations. """
        self._objects.update()
        self._update_viewport()
        self._active_frame.reload_annotations()

    def _update_viewport(self):
        """ Update the viewport

        Clear out cached objects that are not currently in view. Populate the cache for any
        faces that are now in view. Populate the correct face image and annotations for each
        object in the viewport based on current location. If optional mesh annotations are
        enabled, then calculates newly displayed meshes. """
        if not self._grid.is_valid:
            return
        self._discard_tk_faces()

        if self._canvas.optional_annotations["mesh"]:  # Display any hidden end of row meshes
            self._canvas.itemconfig("viewport_mesh", state="normal")

        for collection in zip(self._objects.visible_grid.transpose(1, 2, 0),
                              self._objects.images,
                              self._objects.meshes,
                              self._objects.visible_faces):
            for (frame_idx, face_idx, pnt_x, pnt_y), image_id, mesh_ids, face in zip(*collection):
                top_left = np.array((pnt_x, pnt_y))
                if frame_idx == self._active_frame.frame_index:
                    logger.trace("Skipping active frame: %s", frame_idx)
                    continue
                if frame_idx == -1:
                    logger.debug("Blanking non-existant face")
                    self._canvas.itemconfig(image_id, image="")
                    for area in mesh_ids.values():
                        for mesh_id in area:
                            self._canvas.itemconfig(mesh_id, state="hidden")
                    continue

                tk_face = self.get_tk_face(frame_idx, face_idx, face)
                self._canvas.itemconfig(image_id, image=tk_face.photo)
                if (self._canvas.optional_annotations["mesh"]
                        or frame_idx == self._active_frame.frame_index):
                    landmarks = self.get_landmarks(frame_idx, face_idx, face, top_left)
                    self._locate_mesh(mesh_ids, landmarks)

    def _discard_tk_faces(self):
        """ Remove any :class:`TKFace` objects from the cache that are not currently displayed. """
        keys = ["{}_{}".format(pnt_x, pnt_y)
                for pnt_x, pnt_y in self._objects.visible_grid[:2].T.reshape(-1, 2)]
        for key in list(self._tk_faces):
            if key not in keys:
                del self._tk_faces[key]
        logger.trace("keys: %s allocated_faces: %s", keys, len(self._tk_faces))

    def get_tk_face(self, frame_index, face_index, face):
        """ Obtain the :class:`TKFace` object for the given face from the cache. If the face does
        not exist in the cache, then it is generated and added prior to returning.

        Parameters
        ----------
        frame_index: int
            The frame index to obtain the face for
        face_index: int
            The face index of the face within the requested frame
        face: :class:`~lib.faces_detect.DetectedFace`
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
            logger.trace("creating new tk_face: (key: %s, is_active: %s)", key, is_active)
            if is_active:
                face.load_aligned(self._active_frame.current_frame,
                                  size=self.face_size,
                                  force=True)
                image = face.aligned_face
                face.aligned = dict()
            else:
                image = face.thumbnail
            tk_face = self._get_tk_face_object(face, image, is_active)
            self._tk_faces[key] = tk_face
        else:
            logger.trace("tk_face exists: %s", key)
            tk_face = self._tk_faces[key]
        return tk_face

    def _get_tk_face_object(self, face, image, is_active):
        """ Obtain an existing unallocated, or a newly created :class:`TKFace` and populate it with
        face information from the requested frame and face index.

        If the face is currently active, then the face is generated from the currently displayed
        frame, otherwise it is generated from the jpg thumbnail.

        Parameters
        ----------
        face: :class:`lib.faces_detect.DetectedFace`
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
        mask = face.mask.get(self._canvas.selected_mask, None) if get_mask else None
        mask = mask if mask is None else mask.mask.squeeze()
        tk_face = TKFace(image, size=self.face_size, mask=mask)
        logger.trace("face: %s, tk_face: %s", face, tk_face)
        return tk_face

    def get_landmarks(self, frame_index, face_index, face, top_left, refresh=False):
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
        top_left: tuple
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
        key = "{}_{}".format(frame_index, face_index)
        landmarks = self._landmarks.get(key, None)
        if not landmarks or refresh:
            face.load_aligned(None, size=self.face_size, force=True)
            landmarks = dict(polygon=[], line=[])
            for area, val in self._landmark_mapping.items():
                points = face.aligned_landmarks[val[0]:val[1]] + top_left
                shape = "polygon" if area.endswith("eye") or area.startswith("mouth") else "line"
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
            for coords, mesh_id in zip(area, mesh_ids[key]):
                self._canvas.coords(mesh_id, *coords.flatten())

    def face_from_point(self, point_x, point_y):
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
        if point_x > self._grid.dimensions[0]:
            retval = np.array((-1, -1, -1, -1))
        else:
            x_idx = np.searchsorted(self._objects.visible_grid[2, 0, :], point_x, side="left") - 1
            y_idx = np.searchsorted(self._objects.visible_grid[3, :, 0], point_y, side="left") - 1
            if x_idx < 0 or y_idx < 0:
                retval = np.array((-1, -1, -1, -1))
            else:
                retval = self._objects.visible_grid[:, y_idx, x_idx]
        logger.trace(retval)
        return retval

    def move_active_to_top(self):
        """ Check whether the active frame is going off the bottom of the viewport, if so: move it
        to the top of the viewport. """
        self._active_frame.move_to_top()


class VisibleObjects():
    """ Holds the objects from the :class:`~tools.manual.faceviewer.frame.Grid` that appear in the
    viewable area of the :class:`Viewport`.

    Parameters
    ----------
    viewport: :class:`Viewport`
        The viewport object for the :class:`~tools.manual.faceviewer.frame.FacesViewer` canvas
    """
    def __init__(self, viewport):
        self._viewport = viewport
        self._canvas = viewport._canvas
        self._grid = viewport._grid
        self._size = viewport.face_size

        self._visible_grid = None
        self._visible_faces = None
        self._images = []
        self._meshes = []
        self._recycled = dict(images=[], meshes=[])

    @property
    def visible_grid(self):
        """ :class:`numpy.ndarray`: The currently visible section of the
        :class:`~tools.manual.faceviewer.frame.Grid`

        A numpy array of shape (`4`, `rows`, `columns`) corresponding to the viewable area of the
        display grid. 1st dimension contains frame indices, 2nd dimension face indices. The 3rd and
        4th dimension contain the x and y position of the top left corner of the face respectively.

        Any locations that are not populated by a face will have a frame and face index of -1. """
        return self._visible_grid

    @property
    def visible_faces(self):
        """ :class:`numpy.ndarray`: The currently visible :class:`~lib.faces_detect.DetectedFace`
        objects.

        A numpy array of shape (`rows`, `columns`) corresponding to the viewable area of the
        display grid and containing the detected faces at their currently viewable position.

        Any locations that are not populated by a face will have ``None`` in it's place. """
        return self._visible_faces

    @property
    def images(self):
        """ :class:`numpy.ndarray`: The viewport's tkinter canvas image objects.

        A numpy array of shape (`rows`, `columns`) corresponding to the viewable area of the
        display grid and containing the tkinter canvas image object for the face at the
        corresponding location. """
        return self._images

    @property
    def meshes(self):
        """ :class:`numpy.ndarray`: The viewport's tkinter canvas mesh annotation objects.

        A numpy array of shape (`rows`, `columns`) corresponding to the viewable area of the
        display grid and containing a dictionary of the corresponding tkinter polygon and line
        objects required to build a face's mesh annotation for the face at the corresponding
        location. """
        return self._meshes

    @property
    def _top_left(self):
        """ :class:`numpy.ndarray`: The canvas (`x`, `y`) position of the face currently in the
        viewable area's top left position. """
        return np.array(self._canvas.coords(self._images[0][0]), dtype="int")

    def update(self):
        """ Load and unload thumbnails in the visible area of the faces viewer. """
        self._visible_grid, self._visible_faces = self._grid.visible_area
        if (isinstance(self._images, np.ndarray) and
                self._visible_grid.shape[-1] != self._images.shape[-1]):
            self._recycle_objects()

        required_rows = self._visible_grid.shape[1] if self._grid.is_valid else 0
        existing_rows = len(self._images)
        logger.trace("existing_rows: %s. required_rows: %s", existing_rows, required_rows)

        if existing_rows > required_rows:
            for image_id in self._images[required_rows: existing_rows].flatten():
                logger.trace("Hiding image id: %s", image_id)
                self._canvas.itemconfig(image_id, image="")

        if existing_rows < required_rows:
            self._add_rows(existing_rows, required_rows)

        self._shift()

    def _recycle_objects(self):
        """  On a column count change, place all existing objects into the recycle bin so that
        they can be used for the new grid shape and reset the objects size to the new size. """
        self._size = self._viewport.face_size
        images = self._images.flatten().tolist()
        meshes = self._meshes.flatten().tolist()

        for image_id in images:
            self._canvas.itemconfig(image_id, image="")
            self._canvas.coords(image_id, 0, 0)
        for mesh in meshes:
            for key, mesh_ids in mesh.items():
                coords = (0, 0, 0, 0) if key == "line" else (0, 0)
                for mesh_id in mesh_ids:
                    self._canvas.coords(mesh_id, *coords)

        self._recycled["images"].extend(images)
        self._recycled["meshes"].extend(meshes)
        logger.trace("Recycled objects: %s", self._recycled)

        self._images = []
        self._meshes = []

    def _add_rows(self, existing_rows, required_rows):
        """ Add rows to the viewport.

        Parameters
        ----------
        existing_rows: int
            The number of existing rows within the viewport
        required_rows: int
            The number of rows required by the viewport
        """
        columns = self._grid.columns_rows[0]
        if not isinstance(self._images, np.ndarray):
            base_coords = [(col * self._size, 0) for col in range(columns)]
        else:
            base_coords = [self._canvas.coords(item_id) for item_id in self._images[0]]
        logger.debug("existing rows: %s, required_rows: %s, base_coords: %s",
                     existing_rows, required_rows, base_coords)
        images = []
        meshes = []
        for row in range(existing_rows, required_rows):
            y_coord = base_coords[0][1] + (row * self._size)
            images.append(np.array([self._get_image((coords[0], y_coord))
                                    for coords in base_coords]))
            meshes.append(np.array([self._get_mesh() for _ in range(columns)]))
        images = np.array(images)
        meshes = np.array(meshes)

        if not isinstance(self._images, np.ndarray):
            logger.debug("Adding initial viewport objects: (image shapes: %s, mesh shapes: %s)",
                         images.shape, meshes.shape)
            self._images = images
            self._meshes = meshes
        else:
            logger.debug("Adding new viewport objects: (image shapes: %s, mesh shapes: %s)",
                         images.shape, meshes.shape)
            self._images = np.concatenate((self._images, images))
            self._meshes = np.concatenate((self._meshes, meshes))
        logger.debug("self._images: %s, self._meshes: %s", self._images.shape, self._meshes.shape)

    def _get_image(self, coordinates):
        """ Create or recycle a tkinter canvas image object with the given coordinates.

        Parameters
        ----------
        coordinates: tuple
            The (`x`, `y`) coordinates for the top left corner of the image

        Returns
        -------
        int
            The canvas object id for the created image
        """
        if self._recycled["images"]:
            image_id = self._recycled["images"].pop()
            self._canvas.coords(image_id, *coordinates)
            logger.trace("Recycled image: %s", image_id)
        else:
            image_id = self._canvas.create_image(*coordinates,
                                                 anchor=tk.NW,
                                                 tags=["viewport", "viewport_image"])
            logger.trace("Created new image: %s", image_id)
        return image_id

    def _get_mesh(self):
        """ Get the mesh annotation for the landmarks. This is made up of a series of polygons
        or lines, depending on which part of the face is being annotated. Creates a new series of
        objects, or pulls existing objects from the recycled objects pool if they are available.

        Returns
        -------
        dict
            The dictionary of line and polygon tkinter canvas object ids for the mesh annotation
        """
        kwargs = self._viewport.mesh_kwargs
        logger.trace("self.mesh_kwargs: %s", kwargs)
        if self._recycled["meshes"]:
            mesh = self._recycled["meshes"].pop()
            for key, mesh_ids in mesh.items():
                for mesh_id in mesh_ids:
                    self._canvas.itemconfig(mesh_id, **kwargs[key])
            logger.trace("Recycled mesh: %s", mesh)
        else:
            tags = ["viewport", "viewport_mesh"]
            mesh = dict(polygon=[self._canvas.create_polygon(0, 0,
                                                             width=1,
                                                             tags=tags + ["viewport_polygon"],
                                                             **kwargs["polygon"])
                                 for _ in range(4)],
                        line=[self._canvas.create_line(0, 0, 0, 0,
                                                       width=1,
                                                       tags=tags + ["viewport_line"],
                                                       **kwargs["line"])
                              for _ in range(5)])
            logger.trace("Created new mesh: %s", mesh)
        return mesh

    def _shift(self):
        """ Shift the viewport in the y direction if required

        Returns
        -------
        bool
            ``True`` if the viewport was shifted otherwise ``False``
        """
        current_y = self._top_left[1]
        required_y = self._visible_grid[3, 0, 0] if self._grid.is_valid else 0
        logger.trace("current_y: %s, required_y: %s", current_y, required_y)
        if current_y == required_y:
            logger.trace("No move required")
            return False
        shift_amount = required_y - current_y
        logger.trace("Shifting viewport: %s", shift_amount)
        self._canvas.move("viewport", 0, shift_amount)
        return True


class HoverBox():  # pylint:disable=too-few-public-methods
    """ Handle the current mouse location when over the :class:`Viewport`.

    Highlights the face currently underneath the cursor and handles actions when clicking
    on a face.

    Parameters
    ----------
    viewport: :class:`Viewport`
        The viewport object for the :class:`~tools.manual.faceviewer.frame.FacesViewer` canvas
    """
    def __init__(self, viewport):
        logger.debug("Initializing: %s (viewport: %s)", self.__class__.__name__, viewport)
        self._viewport = viewport
        self._canvas = viewport._canvas
        self._grid = viewport._canvas.grid
        self._globals = viewport._canvas._globals
        self._navigation = viewport._canvas._display_frame.navigation
        self._box = self._canvas.create_rectangle(0, 0, self._size, self._size,
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
    def _size(self):
        """ int: the currently set viewport face size in pixels. """
        return self._viewport.face_size

    def on_hover(self, event):
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
            pnts = (event.x, event.y)

        coords = (int(self._canvas.canvasx(pnts[0])), int(self._canvas.canvasy(pnts[1])))
        face = self._viewport.face_from_point(*coords)
        frame_idx, face_idx = face[:2]
        is_zoomed = self._globals.is_zoomed

        if (-1 in face or (frame_idx == self._globals.frame_index
                           and (not is_zoomed or
                                (is_zoomed and face_idx == self._globals.tk_face_index.get())))):
            self._clear()
            self._canvas.config(cursor="")
            self._current_frame_index = None
            self._current_face_index = None
            return

        self._canvas.config(cursor="hand2")
        self._highlight(face[2:])
        self._current_frame_index = frame_idx
        self._current_face_index = face_idx

    def _clear(self):
        """ Hide the hover box when the mouse is not over a face. """
        if self._canvas.itemcget(self._box, "state") != "hidden":
            self._canvas.itemconfig(self._box, state="hidden")

    def _highlight(self, top_left):
        """ Display the hover box around the face that the mouse is currently over.

        Parameters
        ----------
        top_left: tuple
            The top left point of the highlight box location
        """
        coords = (*top_left, *top_left + self._size)
        self._canvas.coords(self._box, *coords)
        self._canvas.itemconfig(self._box, state="normal")
        self._canvas.tag_raise(self._box)

    def _select_frame(self):
        """ Select the face and the subsequent frame (in the editor view) when a face is clicked
        on in the :class:`Viewport`.
        """
        frame_id = self._current_frame_index
        is_zoomed = self._globals.is_zoomed
        if frame_id is None or (frame_id == self._globals.frame_index and not is_zoomed):
            return
        face_idx = self._current_face_index if is_zoomed else 0
        self._globals.tk_face_index.set(face_idx)
        transport_id = self._grid.transport_index_from_frame(frame_id)
        logger.trace("frame_index: %s, transport_id: %s, face_idx: %s",
                     frame_id, transport_id, face_idx)
        if transport_id is None:
            return
        self._navigation.stop_playback()
        self._globals.tk_transport_index.set(transport_id)
        self._viewport.move_active_to_top()
        self.on_hover(None)


class ActiveFrame():
    """ Handles the display of faces and annotations for the currently active frame.

    Parameters
    ----------
    canvas: :class:`tkinter.Canvas`
        The :class:`~tools.manual.faceviewer.frame.FacesViewer` canvas
    tk_edited_variable: :class:`tkinter.BooleanVar`
        The tkinter callback variable indicating that a face has been edited
    """
    def __init__(self, viewport, tk_edited_variable):
        logger.debug("Initializing: %s (viewport: %s, tk_edited_variable: %s)",
                     self.__class__.__name__, viewport, tk_edited_variable)
        self._objects = viewport._objects
        self._viewport = viewport
        self._grid = viewport._grid
        self._tk_faces = viewport._tk_faces
        self._canvas = viewport._canvas
        self._globals = viewport._canvas._globals
        self._navigation = viewport._canvas._display_frame.navigation
        self._last_execution = dict(frame_index=-1, size=viewport.face_size)
        self._tk_vars = dict(selected_editor=self._canvas._display_frame.tk_selected_action,
                             edited=tk_edited_variable)
        self._assets = dict(images=[], meshes=[], faces=[], boxes=[])

        self._globals.tk_update_active_viewport.trace("w", lambda *e: self._reload_callback())
        tk_edited_variable.trace("w", lambda *e: self._update_on_edit())
        logger.debug("Initialized: %s", self.__class__.__name__)

    @property
    def frame_index(self):
        """ int: The frame index of the currently displayed frame. """
        return self._globals.frame_index

    @property
    def current_frame(self):
        """ :class:`numpy.ndarray`: A BGR version of the frame currently being displayed. """
        return self._globals.current_frame["image"]

    @property
    def _size(self):
        """ int: The size of the thumbnails displayed in the viewport, in pixels. """
        return self._viewport.face_size

    @property
    def _optional_annotations(self):
        """ dict: The currently selected optional annotations """
        return self._canvas.optional_annotations

    def _reload_callback(self):
        """ If a frame has changed, triggering the variable, then update the active frame. Return
        having done nothing if the variable is resetting. """
        if self._globals.tk_update_active_viewport.get():
            self.reload_annotations()

    def reload_annotations(self):
        """ Handles the reloading of annotations for the currently active faces.

        Highlights the faces within the viewport of those faces that exist in the currently
        displaying frame. Applies annotations based on the optional annotations and current
        editor selections.
        """
        logger.trace("Reloading annotations")
        if np.any(self._assets["images"]):
            self._clear_previous()

        self._set_active_objects()
        self._check_active_in_view()

        if not np.any(self._assets["images"]):
            logger.trace("No active faces. Returning")
            self._last_execution["frame_index"] = self.frame_index
            return

        if self._last_execution["frame_index"] != self.frame_index:
            self.move_to_top()
        self._create_new_boxes()

        self._update_face()
        self._canvas.tag_raise("active_highlighter")
        self._globals.tk_update_active_viewport.set(False)
        self._last_execution["frame_index"] = self.frame_index

    def _clear_previous(self):
        """ Reverts the previously selected annotations to their default state. """
        logger.trace("Clearing previous active frame")
        self._canvas.itemconfig("active_highlighter", state="hidden")

        for key in ("polygon", "line"):
            tag = "active_mesh_{}".format(key)
            self._canvas.itemconfig(tag, **self._viewport.mesh_kwargs[key])
            self._canvas.dtag(tag)

        if self._viewport.selected_editor == "mask" and not self._optional_annotations["mask"]:
            for key, tk_face in self._tk_faces.items():
                if key.startswith("{}_".format(self._last_execution["frame_index"])):
                    tk_face.update_mask(None)

    def _set_active_objects(self):
        """ Collect the objects that exist in the currently active frame from the main grid. """
        if self._grid.is_valid:
            rows, cols = np.where(self._objects.visible_grid[0] == self.frame_index)
            logger.trace("Setting active objects: (rows: %s, columns: %s)", rows, cols)
            self._assets["images"] = self._objects.images[rows, cols]
            self._assets["meshes"] = self._objects.meshes[rows, cols]
            self._assets["faces"] = self._objects.visible_faces[rows, cols]
        else:
            logger.trace("No valid grid. Clearing active objects")
            self._assets["images"] = []
            self._assets["meshes"] = []
            self._assets["faces"] = []

    def _check_active_in_view(self):
        """  If the frame has changed, there are faces in the frame, but they don't appear in the
        viewport, then bring the active faces to the top of the viewport. """
        if (not np.any(self._assets["images"]) and
                self._last_execution["frame_index"] != self.frame_index and
                self._grid.frame_has_faces(self.frame_index)):
            y_coord = self._grid.y_coord_from_frame(self.frame_index)
            logger.trace("Active not in view. Moving to: %s", y_coord)
            self._canvas.yview_moveto(y_coord / self._canvas.bbox("backdrop")[3])
            self._viewport.update()

    def move_to_top(self):
        """ Move the currently selected frame's faces to the top of the viewport if they are moving
        off the bottom of the viewer. """
        height = self._canvas.bbox("backdrop")[3]
        bot = int(self._canvas.coords(self._assets["images"][-1])[1] + self._size)

        y_top, y_bot = (int(round(pnt * height)) for pnt in self._canvas.yview())

        if y_top < bot < y_bot:  # bottom face is still in fully visible area
            logger.trace("Active faces in frame. Returning")
            return

        top = int(self._canvas.coords(self._assets["images"][0])[1])
        if y_top == top:
            logger.trace("Top face already on top row. Returning")
            return

        if self._canvas.winfo_height() > self._size:
            logger.trace("Viewport taller than single face height. Moving Active faces to top: %s",
                         top)
            self._canvas.yview_moveto(top / height)
            self._viewport.update()
        elif self._canvas.winfo_height() <= self._size and y_top != top:
            logger.trace("Viewport shorter than single face height. Moving Active faces to "
                         "top: %s", top)
            self._canvas.yview_moveto(top / height)
            self._viewport.update()

    def _create_new_boxes(self):
        """ The highlight boxes (border around selected faces) are the only additional annotations
        that are required for the highlighter. If more faces are displayed in the current frame
        than highlight boxes are available, then new boxes are created to accommodate the
        additional faces. """
        new_boxes_count = max(0, len(self._assets["images"]) - len(self._assets["boxes"]))
        if new_boxes_count == 0:
            return
        logger.debug("new_boxes_count: %s", new_boxes_count)
        for _ in range(new_boxes_count):
            box = self._canvas.create_rectangle(0,
                                                0,
                                                self._viewport.face_size, self._viewport.face_size,
                                                outline="#00FF00",
                                                width=2,
                                                state="hidden",
                                                tags=["active_highlighter"])
            logger.trace("Created new highlight_box: %s", box)
            self._assets["boxes"].append(box)

    def _update_on_edit(self):
        """ Update the active faces on a frame edit. """
        if not self._tk_vars["edited"].get():
            return
        self._set_active_objects()
        self._update_face()
        self._tk_vars["edited"].set(False)

    def _update_face(self):
        """ Update the highlighted annotations for faces in the currently selected frame. """
        for face_idx, (image_id, mesh_ids, box_id, det_face), in enumerate(
                zip(self._assets["images"],
                    self._assets["meshes"],
                    self._assets["boxes"],
                    self._assets["faces"])):
            top_left = np.array(self._canvas.coords(image_id))
            coords = (*top_left, *top_left + self._size)
            tk_face = self._viewport.get_tk_face(self.frame_index, face_idx, det_face)
            self._canvas.itemconfig(image_id, image=tk_face.photo)
            self._show_box(box_id, coords)
            self._show_mesh(mesh_ids, face_idx, det_face, top_left)
        self._last_execution["size"] = self._viewport.face_size

    def _show_box(self, item_id, coordinates):
        """ Display the highlight box around the given coordinates.

        Parameters
        ----------
        item_id: int
            The tkinter canvas object identifier for the highlight box
        coordinates: :class:`numpy.ndarray`
            The (x, y, x1, y1) coordinates of the top left corner of the box
        """
        self._canvas.coords(item_id, *coordinates)
        self._canvas.itemconfig(item_id, state="normal")

    def _show_mesh(self, mesh_ids, face_index, detected_face, top_left):
        """ Display the mesh annotation for the given face, at the given location.

        Parameters
        ----------
        mesh_ids: dict
            Dictionary containing the `polygon` and `line` tkinter canvas identifiers that make up
            the mesh for the given face
        face_index: int
            The face index within the frame for the given face
        detected_face: :class:`~lib.faces_detect.DetectedFace`
            The detected face object that contains the landmarks for generating the mesh
        top_left: tuple
            The (x, y) top left co-ordinates of the mesh's bounding box
        """
        state = "normal" if (self._tk_vars["selected_editor"].get() != "Mask" or
                             self._optional_annotations["mesh"]) else "hidden"
        kwargs = dict(polygon=dict(fill="", outline=self._canvas.control_colors["Mesh"]),
                      line=dict(fill=self._canvas.control_colors["Mesh"]))

        edited = (self._tk_vars["edited"].get() and
                  self._tk_vars["selected_editor"].get() not in ("Mask", "View"))
        relocate = self._viewport.face_size != self._last_execution["size"] or (
            state == "normal" and not self._optional_annotations["mesh"])
        if relocate or edited:
            landmarks = self._viewport.get_landmarks(self.frame_index,
                                                     face_index,
                                                     detected_face,
                                                     top_left,
                                                     edited)
        for key, kwarg in kwargs.items():
            for idx, mesh_id in enumerate(mesh_ids[key]):
                if relocate:
                    self._canvas.coords(mesh_id, *landmarks[key][idx].flatten())
                self._canvas.itemconfig(mesh_id, state=state, **kwarg)
                self._canvas.addtag_withtag("active_mesh_{}".format(key), mesh_id)


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
    def __init__(self, face, size=128, mask=None):
        logger.trace("Initializing %s: (face: %s, size: %s, mask: %s)",
                     self.__class__.__name__,
                     face if face is None else face.shape,
                     size,
                     mask if mask is None else mask.shape)
        self._size = size
        if face.ndim == 2 and face.shape[1] == 1:
            self._face = self._image_from_jpg(face)
        else:
            self._face = face[..., 2::-1]
        self._photo = ImageTk.PhotoImage(self._generate_tk_face_data(mask))

        logger.trace("Initialized %s", self.__class__.__name__)

    # << PUBLIC PROPERTIES >> #
    @property
    def photo(self):
        """ :class:`tkinter.PhotoImage`: The face in a format that can be placed on the
         :class:`~tools.manual.faceviewer.frame.FacesViewer` canvas. """
        return self._photo

    # << PUBLIC METHODS >> #
    def update(self, face, mask):
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

    def update_mask(self, mask):
        """ Update the mask in the 4th channel of :attr:`photo` to the given mask.

        Parameters
        ----------
        mask: :class:`numpy.ndarray` or ``None``
            The mask to be applied to the face image. Pass ``None`` if no mask is to be used
        """
        self._photo.paste(self._generate_tk_face_data(mask))

    # << PRIVATE METHODS >> #
    def _image_from_jpg(self, face):
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

    def _generate_tk_face_data(self, mask):
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
