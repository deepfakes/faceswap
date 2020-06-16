#!/usr/bin/env python3
""" Handles the visible area of the canvas. """

import logging
import tkinter as tk

import numpy as np

from .cache import TKFace

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Viewport():
    """ Handles the display of faces and annotations in the currently viewable area of the canvas.

    Parameters
    ----------
    canvas: :class:`tkinter.Canvas`
        The :class:`~tools.manual.FacesViewer` canvas
    """
    def __init__(self, canvas):
        logger.debug("Initializing: %s: (canvas: %s)", self.__class__.__name__, canvas)
        self._canvas = canvas
        self._grid = canvas.grid
        self._visible_grid = None
        self._detected_faces = canvas.grid._detected_faces
        self._unallocated_faces = []
        self._tk_faces = dict()
        self._images = []
        self._meshes = []
        self._hoverbox = HoverBox(self)

    @property
    def face_size(self):
        """ int: The pixel size of each thumbnail """
        return self._canvas._size

    @property
    def _top_left(self):
        """class:`numpy.ndarray`: The (x, y) coordinates of the first visible face """
        return np.array(self._canvas.coords(self._images[0][0]), dtype="int")

    @property
    def visible_grid(self):
        """ :class:`numpy.ndarray`: The visible area of the face viewer canvas. 4 dimensions of
        shape (4, rows, columns). 1st dim contains frame indices, 2nd dim contains face indices.
        The 3rd and 4th dims contain the x and y position of the top left corner of the face
        respectively. """
        return self._visible_grid

    @property
    def hover_box(self):
        """ :class:`HoverBox`: The hover box for the viewport. """
        return self._hoverbox

    def set_visible_images(self):
        """ Load and unload thumbnails on a canvas resize or scroll event.
        """
        # TODO remove testing code
        # from time import time
        # start = time()
        # split = time()
        self._visible_grid = self._grid.visible_area
        required_rows = self._visible_grid.shape[1]
        existing_rows = len(self._images)
        # print("data", time() - split)

        if existing_rows < required_rows:
            # split = time()
            self._add_rows(existing_rows, required_rows)
            # print("add", time() - split)

        if existing_rows and not self._shift():
            return
        # split = time()
        self._get_tk_faces()
        # print("faces", time() - split)
        # print("total", time() - start)

    def _get_tk_faces(self):
        """ Clear out unused faces and populate with visible faces """
        self._recycle_tk_faces()
        for row in range(self._visible_grid.shape[1]):
            for column in range(self._visible_grid.shape[2]):
                frame_idx, face_idx = self._visible_grid[:2, row, column]
                if frame_idx == -1:
                    logger.trace("Blanking non-existant face")
                    self._canvas.itemconfig(self._images[row][column], image="")
                    continue
                key = "_".join([str(frame_idx), str(face_idx)])
                if key not in self._tk_faces:
                    logger.trace("creating new tk_face: %s", key)
                    tk_face = self._get_tk_face_object(frame_idx, face_idx)
                    self._tk_faces[key] = tk_face
                else:
                    logger.trace("tk_face exists: %s", key)
                    tk_face = self._tk_faces[key]
                self._canvas.itemconfig(self._images[row][column], image=tk_face.face)

    def _recycle_tk_faces(self):
        """ Move any tk_faces not used in the current viewport to the unallocated pool. """
        keys = ["{}_{}".format(pnt_x, pnt_y)
                for pnt_x, pnt_y in self._visible_grid[:2].T.reshape(-1, 2)]
        self._unallocated_faces.extend([self._tk_faces.pop(key)
                                        for key in list(self._tk_faces)
                                        if key not in keys])
        logger.trace("keys: %s unallocated_faces: %s, allocated_faces: %s",
                     keys, len(self._unallocated_faces), len(self._tk_faces))

    def _get_tk_face_object(self, frame_index, face_index):
        """ Obtain an existing unallocated, or a newly created
        :class:`tools.manual.faceviewer.cache.TkFace` and populate it with face information from
        the requested frame and face index.

        Parameters
        ----------
        frame_index: int
            The frame index that contains the face to be populated into the object
        face_index: int
            The face index, within the frame, to obtain face information for

        Returns
        -------
        :class:`tools.manual.faceviewer.cache.TkFace`
            An object for displaying in the faces viewer canvas populated with the aligned mesh
            landmarks and face thumbnail
        """
        # TODO Add jpg thumbnail immediately (requires tk_face slight structure change)
        face = self._detected_faces.current_faces[frame_index][face_index]
        face.load_aligned(None, size=self.face_size)
        image = self._detected_faces.get_thumbnail(frame_index, face_index)
        if self._unallocated_faces:  # Recycle existing object
            logger.trace("Recycling object")
            tk_face = self._unallocated_faces.pop()
            tk_face.update_landmarks(face.aligned_landmarks)
        else:  # Get new object
            logger.trace("Getting new object")
            tk_face = TKFace(face.aligned_landmarks,
                             size=self.face_size,
                             face=None,
                             mask=None)
        tk_face.set_thumbnail(image)
        logger.trace("frame_index: %s, face_index: %s, tk_face: %s",
                     frame_index, face_index, tk_face)
        return tk_face

    def _add_rows(self, existing_rows, required_rows):
        """ Add objects to the viewport

        Parameters
        ----------
        number: int
            The number of objects to add to the viewport
        """
        # TODO This will probably add the row in the wrong place, due to shift.
        # Get position and count from current image locations?
        if not isinstance(self._images, np.ndarray):
            base_coords = [(0, col * self.face_size)
                           for col in range(self._grid.columns_rows[0])]
        else:
            base_coords = [self._canvas.coords(item_id) for item_id in self._images[0]]
        logger.trace("existing rows: %s, required_rows: %s, base_coords: %s",
                     existing_rows, required_rows, base_coords)
        images = []
        meshes = []
        dummy_face = TKFace(np.zeros((68, 2)), size=self.face_size)
        for row in range(existing_rows, required_rows):
            y_coord = base_coords[0][0] + (row * self.face_size)
            images.append(np.array([
                self._canvas.create_image(
                    coords[1],
                    y_coord,
                    anchor=tk.NW,
                    tags=["viewport", "viewport_image"])
                for coords in base_coords]))
            meshes.append(np.array([self._create_mesh(np.array((coords[1], y_coord)), dummy_face)
                                    for coords in base_coords]))
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

    def _shift(self):
        """ Shift the viewport in the y direction if required

        Returns
        -------
        bool
            ``True`` if the viewport was shifted otherwise ``False``
        """
        current_y = self._top_left[1]
        required_y = self._visible_grid[3, 0, 0]
        logger.trace("current_y: %s, required_y: %s", current_y, required_y)
        if current_y == required_y:
            logger.trace("No move required")
            return False
        shift_amount = required_y - current_y
        logger.trace("Shifting viewport: %s", shift_amount)
        self._canvas.move("viewport", 0, shift_amount)
        return True

    def _create_mesh(self, coordinates, tk_face):
        """ Creates the mesh annotation for the landmarks. This is made up of a series of polygons
        or lines, depending on which part of the face is being annotated.

        Parameters
        ----------
        coordinates: :class:`numpy.ndarray`
            The top left co-ordinates of the face that corresponds to the given landmarks.
            The mesh annotations will be offset by this amount, to place them in the correct
            place on the canvas
        tk_face: :class:`~manual.facesviewer.cache.TKFace`
            The tk face object containing the face to be used for the image annotation and the
            mesh landmarks

        Returns
        -------
        list
            The canvas object ids for the created mesh annotations
        """
        retval = []
        state = "normal" if self._canvas.optional_annotations["mesh"] else "hidden"
        color = self._canvas.get_muted_color("Mesh")
        tags = ["viewport", "viewport_mesh"]
        kwargs = dict(polygon=dict(fill="", outline=color), line=dict(fill=color))
        logger.trace("color: %s, coordinates: %s, tag: %s, state: %s, kwargs: %s",
                     color, coordinates, tags, state, kwargs)
        for is_poly, landmarks in zip(tk_face.mesh_is_poly, tk_face.mesh_points):
            key = "polygon" if is_poly else "line"
            tags = tags + ["mesh_{}".format(key)]
            obj = getattr(self._canvas, "create_{}".format(key))
            obj_kwargs = kwargs[key]
            coords = (landmarks + coordinates).flatten()
            retval.append(obj(*coords, state=state, width=1, tags=tags, **obj_kwargs))
        return retval

    def face_from_point(self, point_x, point_y):
        """ Get the face information for the face containing the given point.

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
            corner`, `y point of top left corner`) of the face that the mouse is currently over.

            If the mouse is not over a face, then the frame and face indices will be -1
        """
        if point_x > self._grid._dimensions[0]:
            retval = np.array((-1, -1, -1, -1))
        else:
            x_idx = np.searchsorted(self._visible_grid[2, 1, :], point_x, side="left") - 1
            y_idx = np.searchsorted(self._visible_grid[3, :, 1], point_y, side="left") - 1
            retval = self._visible_grid[:, y_idx, x_idx]
        logger.trace(retval)
        return retval


class HoverBox():  # pylint:disable=too-few-public-methods
    """ Handle the current mouse location in the :class:`~tools.manual.FacesViewer`.

    Highlights the face currently underneath the cursor and handles actions when clicking
    on a face.

    Parameters
    ----------
    viewport: :class:`Viewport`
        The viewport object for the :class:`~tools.manual.FacesViewer` canvas
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
        """ int: the currently set viewport face size. """
        return self._viewport.face_size

    def on_hover(self, event):
        """ The mouse cursor display as bound to the mouse's <Motion> event.

        The canvas only displays faces, so if the mouse is over an object change the cursor
        otherwise use default.

        Parameters
        ----------
        event: :class:`tkinter.Event` or `None`
            The tkinter mouse event. Provides the current location of the mouse cursor.
            If `None` is passed as the event (for example when this function is being called
            outside of a mouse event) then the location of the cursor will be calculated
        """
        if event is None:
            # Get the current mouse pointer position if not triggered by an event
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

        self._canvas.config(cursor="hand1")
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
        on in :class:`~tools.manual.FacesViewer`.
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


class ActiveFrame():
    """ Holds the objects and handles faces for the currently selected frame.

    Parameters
    ----------
    canvas: :class:`tkinter.Canvas`
        The :class:`~tools.manual.FacesViewer` canvas
    """
    def __init__(self, viewport):
        logger.debug("Initializing: %s (canvas: %s)", self.__class__.__name__, viewport)
        self._viewport = viewport
        self._canvas = viewport._canvas
        self._det_faces = viewport._detected_faces
        self._globals = viewport._canvas._globals
        self._highlighter = Highlighter(self)
        self._globals.tk_frame_index.trace("w", lambda *e: self.reload_annotations())
        self._det_faces.tk_edited.trace("w", lambda *e: self._update())
        logger.debug("Initialized: %s", self.__class__.__name__)

    @property
    def _size(self):
        """ int: the currently set viewport face size. """
        return self._viewport.face_size

    @property
    # TODO Change
    def face_count(self):
        """ int: The count of faces in the currently selected frame. """
        return len(self.image_ids)

    @property
    # TODO Change
    def image_ids(self):
        """ tuple: The tkinter canvas image ids for the currently selected frame's faces. """
        retval = self._canvas.find_withtag("image_{}".format(self._globals.frame_index))
        logger.trace(retval)
        return retval

    @property
    def mesh_ids(self):
        # TODO Change
        """ tuple: The tkinter canvas mesh ids for the currently selected frame's faces. """
        retval = self._canvas.find_withtag("mesh_{}".format(self._globals.frame_index))
        logger.trace(retval)
        return retval

    def reload_annotations(self):
        """ Refresh the highlighted annotations for faces in the currently selected frame on an
        add/remove face. """
        logger.trace("Reloading annotations")
        self._highlighter.highlight_selected()

    def _update(self):
        """ Update the highlighted annotations for faces in the currently selected frame on an
        update, add or remove. """
        if not self._det_faces.tk_edited.get():
            return
        logger.trace("Faces viewer update triggered")
        if self._add_remove_face():
            logger.debug("Face count changed. Reloading annotations")
            self.reload_annotations()
            return
        self._canvas.update_face.update(*self._det_faces.update.last_updated_face)
        self._highlighter.highlight_selected()
        self._det_faces.tk_edited.set(False)

    def _add_remove_face(self):
        """ Check the number of displayed faces against the number of faces stored in the
        alignments data for the currently selected frame, and add or remove if appropriate. """
        alignment_faces = len(self._det_faces.current_faces[self._globals.frame_index])
        logger.trace("alignment_faces: %s, face_count: %s", alignment_faces, self.face_count)
        if alignment_faces > self.face_count:
            logger.debug("Adding face")
            self._canvas.update_face.add(self._globals.frame_index)
            retval = True
        elif alignment_faces < self._canvas.active_frame.face_count:
            logger.debug("Removing face")
            self._canvas.update_face.remove(*self._det_faces.update.last_updated_face)
            retval = True
        else:
            logger.trace("Face count unchanged")
            retval = False
        logger.trace(retval)
        return retval


class Highlighter():  # pylint:disable=too-few-public-methods
    """ Handle the highlighting of the currently active frame's annotations.

    Parameters
    ----------
    canvas: :class:`ActiveFrame`
        The objects contained in the currently active frame
    """
    def __init__(self, active_frame):
        logger.debug("Initializing: %s: (active_frame: %s)", self.__class__.__name__, active_frame)
        self._size = active_frame._canvas._size
        self._canvas = active_frame._canvas
        self._globals = active_frame._globals
        self._active_frame = active_frame

        self._tk_vars = dict(selected_editor=self._canvas._display_frame.tk_selected_action,
                             selected_mask=self._canvas._display_frame.tk_selected_mask)
        self._objects = dict(image_ids=[], mesh_ids=[], boxes=[])
        self._hidden_boxes_count = 0
        self._prev_objects = dict()
        self._tk_vars["selected_editor"].trace("w", lambda *e: self._highlight_annotations())
        logger.debug("Initialized: %s", self.__class__.__name__,)

    @property
    def _face_count(self):
        """ int: The number of faces in the currently selected frame. """
        return len(self._objects["image_ids"])

    @property
    def _boxes_count(self):
        """ int: The number of highlight boxes (the selected faces border) currently available. """
        return len(self._objects["boxes"])

    def highlight_selected(self):
        """ Highlight the currently selected frame's faces.

        Scrolls the canvas so that selected frame's faces are on the top row.

        Parameters
        ----------
        image_ids: list
            `list` of tkinter canvas object ids for the currently selected frame's displayed
            face objects
        mesh_ids: list
            `list` of tkinter canvas object ids for the currently selected frame's displayed
            mesh annotations
        frame_index: int
            The currently selected frame index
        """
        self._objects["image_ids"] = self._active_frame.image_ids
        self._objects["mesh_ids"] = self._active_frame.mesh_ids
        self._create_new_boxes()
        self._revert_last_frame()
        if self._face_count == 0:
            return
        self._highlight_annotations()

        top = self._canvas.coords(self._objects["boxes"][0])[1] / self._canvas.bbox("all")[3]
        if top != self._canvas.yview()[0]:
            self._canvas.yview_moveto(top)

    # << Add new highlighters >> #
    def _create_new_boxes(self):
        """ The highlight boxes (border around selected faces) are the only additional annotations
        that are required for the highlighter. If more faces are displayed in the current frame
        than highlight boxes are available, then new boxes are created to accommodate the
        additional faces. """
        new_boxes_count = max(0, self._face_count - self._boxes_count)
        logger.trace("new_boxes_count: %s", new_boxes_count)
        if new_boxes_count == 0:
            return
        for _ in range(new_boxes_count):
            box = self._canvas.create_rectangle(0, 0, 1, 1,
                                                outline="#00FF00", width=2, state="hidden")
            logger.trace("Created new highlight_box: %s", box)
            self._objects["boxes"].append(box)
            self._hidden_boxes_count += 1

    # Remove highlight annotations from previously selected frame
    def _revert_last_frame(self):
        """ Remove the highlighted annotations from the previously selected frame. """
        self._revert_last_mask()
        self._hide_unused_boxes()
        self._revert_last_mesh()

    def _revert_last_mask(self):
        """ Revert the highlighted mask from the previously selected frame. """
        mask_view = self._canvas.optional_annotations["mask"]
        mask_edit = self._tk_vars["selected_editor"].get() == "Mask"
        if self._prev_objects.get("mask", None) is None or mask_view == mask_edit:
            self._prev_objects["mask"] = None
            return
        mask_type = self._tk_vars["selected_mask"].get() if mask_view else None
        logger.warning("MASK CODE: self._prev_objects['mask']: %s, mask_type: %s",
                       self._prev_objects["mask"], mask_type)
        self._prev_objects["mask"] = None

    def _hide_unused_boxes(self):
        """ Hide any box (border) highlighters that were in use in the previously selected frame
        but are not required for the current frame. """
        hide_count = self._boxes_count - self._face_count - self._hidden_boxes_count
        hide_count = max(0, hide_count)
        logger.trace("hide_boxes_count: %s", hide_count)
        if hide_count == 0:
            return
        hide_slice = slice(self._face_count, self._face_count + hide_count)
        for box in self._objects["boxes"][hide_slice]:
            logger.trace("Hiding highlight box: %s", box)
            self._canvas.itemconfig(box, state="hidden")
            self._hidden_boxes_count += 1

    def _revert_last_mesh(self):
        """ Revert the previously selected frame's mesh annotations to their standard state. """
        if self._prev_objects.get("mesh", None) is None:
            return
        color = self._canvas.get_muted_color("Mesh")
        kwargs = dict(polygon=dict(fill="", outline=color), line=dict(fill=color))
        state = "normal" if self._canvas.optional_annotations["mesh"] else "hidden"
        for mesh_id in self._prev_objects["mesh"]:
            lookup = self._canvas.type(mesh_id)
            if lookup is None:  # Item deleted
                logger.debug("Skipping deleted mesh annotation: %s", mesh_id)
                continue
            self._canvas.itemconfig(mesh_id, state=state, **kwargs[self._canvas.type(mesh_id)])
        self._prev_objects["mesh"] = None

    # << Highlight faces for the currently selected frame >> #
    def _highlight_annotations(self):
        """ Highlight the currently selected frame's face objects. """
        self._highlight_mask()
        for image_id, box in zip(self._objects["image_ids"],
                                 self._objects["boxes"][:self._face_count]):
            top_left = np.array(self._canvas.coords(image_id))
            self._highlight_box(box, top_left)
        self._highlight_mesh()

    def _highlight_mask(self):
        """ Displays either the full face of the masked face, depending on the currently selected
        editor. """
        mask_edit = self._tk_vars["selected_editor"].get() == "Mask"
        mask_view = self._canvas.optional_annotations["mask"]
        if mask_edit == mask_view:
            return
        mask_type = self._tk_vars["selected_mask"].get() if mask_edit else None
        logger.warning("MASK CODE: self._prev_objects['mask']: %s, mask_type: %s",
                       self._prev_objects["mask"], mask_type)
        self._prev_objects["mask"] = self._globals.frame_index

    def _highlight_box(self, box, top_left):
        """ Locates the highlight box(es) (border(s)) for the currently selected frame correctly
        and displays the correct number of boxes. """
        coords = (*top_left, *top_left + self._size)
        logger.trace("Highlighting box (id: %s, coords: %s)", box, coords)
        self._canvas.coords(box, *coords)
        if self._canvas.itemcget(box, "state") == "hidden":
            self._hidden_boxes_count -= 1
            self._canvas.itemconfig(box, state="normal")

    def _highlight_mesh(self):
        """ Depending on the selected editor, highlights the landmarks mesh for the currently
        selected frame. """
        show_mesh = (self._tk_vars["selected_editor"].get() != "Mask"
                     or self._canvas.optional_annotations["mesh"])
        color = self._canvas.control_colors["Mesh"]
        kwargs = dict(polygon=dict(fill="", outline=color),
                      line=dict(fill=color))
        state = "normal" if show_mesh else "hidden"
        self._prev_objects["mesh"] = []
        for mesh_id in self._objects["mesh_ids"]:
            self._canvas.itemconfig(mesh_id, **kwargs[self._canvas.type(mesh_id)], state=state)
            self._prev_objects["mesh"].append(mesh_id)
