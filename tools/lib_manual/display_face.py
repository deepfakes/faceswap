#!/usr/bin/env python3
""" Face viewer for the manual adjustments tool """
import logging
import platform
import tkinter as tk

import numpy as np

from lib.gui.custom_widgets import RightClickMenu

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# TODO Make it so user can't save until faces are loaded (so alignments dict doesn't change)


class FacesViewerLoader():  # pylint:disable=too-few-public-methods
    """ Loads the faces into the :class:`tools.manual.FacesViewer` as they become available
    in the :class:`tools.lib_manual.media.FacesCache`.

    Faces are loaded into the Face Cache in a background thread. This class checks for the
    availability of loaded faces (every 0.5 seconds until the faces are loaded) and updates
    the display with the latest loaded faces.

    Parameters
    ----------
    canvas: :class:`tkinter.Canvas`
        The :class:`~tools.manual.FacesViewer` canvas
    progress_bar: :class:~lib.gui.custom_widgets.StatusBar`
        The bottom right progress bar
    enable_buttons_callback: python function
        The callback to trigger, once faces have completed loading, to enable the
        Faces Viewer optional annotations buttons
    """
    def __init__(self, canvas, progress_bar, enable_buttons_callback):
        logger.debug("Initializing: %s (canvas: %s, progress_bar: %s, "
                     "enable_buttons_callback: %s)", self.__class__.__name__, canvas, progress_bar,
                     enable_buttons_callback)
        self._canvas = canvas
        self._faces_cache = canvas._faces_cache
        self._alignments = canvas._alignments
        self._frame_count = canvas._frames.frame_count
        self._progress_bar = progress_bar
        self._enable_buttons_callback = enable_buttons_callback
        self._progress_bar.start(mode="determinate")
        face_count = self._alignments.face_count_per_index
        self._faces_cache.load_faces()
        frame_faces_loaded = [False for _ in range(self._frame_count)]
        self._load_faces(0, face_count, frame_faces_loaded)
        logger.debug("Initialized: %s ", self.__class__.__name__,)

    def _load_faces(self, load_index, faces_count, frame_faces_loaded):
        """ Load the currently available faces from :class:`tools.lib_manual.media.FacesCache`
        into the Faces Viewer.

        Obtains the indices of all faces that have currently been loaded, and displays them in the
        Faces Viewer. This process re-runs every 0.5 seconds until all faces have been loaded.

        Parameters
        ----------
        load index: int
            The number of faces that have already been loaded. For tracking progress
        faces_count: list
            The number of faces that appear in each frame. List is of length :attr:`_frame_count`
            with each value being the number of faces that appear for the given index.
        frame_faces_loaded: list
            List of length :attr:`_frame_count` containing `bool` values indicating whether
            annotations have been created for each frame or not.
        """
        self._update_progress(load_index)
        update_indices = self._faces_cache.load_cache[load_index:]
        logger.debug("load_index: %s, update count: %s", load_index, len(update_indices))
        tk_faces = self._faces_cache.tk_faces[update_indices]
        frame_landmarks = self._faces_cache.mesh_landmarks[update_indices]
        for frame_idx, faces, mesh_landmarks in zip(update_indices, tk_faces, frame_landmarks):
            starting_idx = sum(faces_count[:frame_idx])
            for idx, (face, landmarks) in enumerate(zip(faces, mesh_landmarks)):
                coords = self._canvas.coords_from_index(starting_idx + idx)
                self._canvas.new_objects.create(coords, face, landmarks, frame_idx,
                                                is_multi=len(faces) > 1)
                self._reshape_canvas(coords)
            if faces:
                self._place_in_stack(frame_idx, frame_faces_loaded)

        load_index += len(update_indices)
        if load_index == self._frame_count:
            logger.debug("Load complete")
            self._on_load_complete()
        else:
            logger.debug("Refreshing... (load_index: %s, frame_count: %s",
                         load_index, self._frame_count)
            self._canvas.after(500, self._load_faces, load_index, faces_count, frame_faces_loaded)

    def _reshape_canvas(self, coordinates):
        """ Scroll the canvas to the first row, when the first row has been received from the
        Faces Cache (sometimes the first row doesn't load first).

        Update the canvas scroll region to include any newly added objects

        Parameters
        ----------
        coordinates: tuple
            The (x, y) coordinates of the newly created object
        """
        if coordinates[1] == 0:  # Set the top of canvas when first row seen
            logger.debug("Scrolling to top row")
            self._canvas.yview_moveto(0.0)
        if coordinates[0] == 0:  # Resize canvas on new line
            logger.trace("Extending canvas")
            self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _place_in_stack(self, frame_index, frame_faces_loaded):
        """ Place any out of order objects into their correct position in the stack.
        As the images are loaded in parallel, the faces do not created in order on the stack.
        For the viewer to work correctly, out of order items are placed back in the correct place.

        Parameters
        ----------
        frame_index: int
            The index that the currently loading objects belong to
        frame_faces_loaded: list
            List of length :attr:`_frame_count` containing `bool` values indicating whether
            annotations have been created for each frame or not.
        """
        frame_faces_loaded[frame_index] = True
        offset = frame_index + 1
        higher_frames = frame_faces_loaded[offset:]
        if not any(higher_frames):
            return
        below_frame = next(idx for idx, loaded in enumerate(higher_frames) if loaded) + offset
        logger.trace("Placing frame %s in stack below frame %s", frame_index, below_frame)
        self._canvas.tag_lower("frame_id_{}".format(frame_index),
                               "frame_id_{}".format(below_frame))

    def _update_progress(self, load_index):
        """ Update the progress bar prior to loading the latest faces.

        Parameters
        ----------
        load_index: int
            The number of faces that have already been loaded
        """
        position = load_index + 1
        progress = int(round((position / self._frame_count) * 100))
        msg = "Loading Faces: {}/{} - {}%".format(position, self._frame_count, progress)
        logger.debug("Progress update: %s", msg)
        self._progress_bar.progress_update(msg, progress)

    def _on_load_complete(self):
        """ Final actions to perform once the faces have finished loading into the Faces Viewer.

        Updates any faces where edits have been made whilst the faces were loading.
        Enables the optional annotations buttons.
        Updates any color settings that were changed during load.
        Sets the display to the currently selected filter.
        Highlights the active face,
        """
        # TODO Enable saving
        for frame_idx, faces in enumerate(self._alignments.updated_alignments):
            if faces is None:
                continue
            image_ids = self._canvas.find_withtag("image_{}".format(frame_idx))
            existing_count = len(image_ids)
            new_count = len(faces)
            self._on_load_remove_faces(existing_count, new_count, frame_idx)
            for face_idx in range(new_count):
                if face_idx + 1 > existing_count:
                    self._canvas.update_face.add(frame_idx)
                else:
                    self._canvas.update_face.update(frame_idx, face_idx)
        self._alignments.tk_edited.set(False)
        self._enable_buttons_callback()
        self._canvas.update_mesh_color()
        self._progress_bar.stop()
        self._canvas.switch_filter()
        self._faces_cache.set_load_complete()
        self._canvas.active_frame.reload_annotations()

    def _on_load_remove_faces(self, existing_count, new_count, frame_index):
        """ Remove any faces from the viewer for the given frame index of any faces
        that have been deleted whilst face viewer was loading.

        Parameters
        ----------
        existing_count: int
            The number of faces that currently appear in the Faces Viewer for the given frame
        new_count: int
            The number of faces that should appear in the Faces Viewer for the given frame
        frame_index: int
            The frame index to remove faces for
        """
        logger.debug("existing_count: %s. new_count: %s, frame_index: %s",
                     existing_count, new_count, frame_index)
        if existing_count <= new_count:
            return
        for face_idx in range(new_count, existing_count):
            logger.debug("Deleting face at index %s for frame %s", face_idx, frame_index)
            self._canvas.update_face.remove(frame_index, face_idx)


class ObjectCreator():
    """ Creates the objects and annotations that are to be displayed in the
    :class:`tools.manual.FacesViewer.`

    Parameters
    ----------
    canvas: :class:`tkinter.Canvas`
        The :class:`~tools.manual.FacesViewer` canvas
    """
    def __init__(self, canvas):
        logger.debug("Initializing: %s (canvas: %s)", self.__class__.__name__, canvas)
        self._canvas = canvas
        self._object_types = ("image", "mesh")
        self._current_face_id = 0
        logger.debug("Initialized: %s", self.__class__.__name__)

    def create(self, coordinates, tk_face, mesh_landmarks, frame_index, is_multi=False):
        """ Create all of the annotations for a single Face Viewer face.

        Parameters
        ----------
        coordinates: tuple
            The top left (x, y) coordinates for the annotations' position in the Faces Viewer
        tk_face: :class:`tkinter.PhotoImage`
            The face to be used for the image annotation
        mesh_landmarks: dict
            A dictionary containing the keys `landmarks` holding a `list` of :class:`numpy.ndarray`
            objects and `is_poly` containing a `list` of `bool` types corresponding to the
            `landmarks` indicating whether a line or polygon should be created for each mesh
            annotation
        mesh_color: str
            The hex code holding the color that the mesh should be displayed as
        frame_index: int
            The frame index that this object appears in
        is_multi: bool
            ``True`` if there are multiple faces in the given frame, otherwise ``False``.
            Default: ``False``. Used for creating multi-face tags

        Returns
        -------
        image_id: int
            The item id of the newly created face
        mesh_ids: list
            List of item ids for the newly created mesh
        """
        logger.trace("coordinates: %s, tk_face: %s, frame_index: %s, "
                     "is_multi: %s", coordinates, tk_face, frame_index, is_multi)
        tags = {obj: self._get_viewer_tags(obj, frame_index, is_multi)
                for obj in self._object_types}
        image_id = self._canvas.create_image(*coordinates,
                                             image=tk_face,
                                             anchor=tk.NW,
                                             tags=tags["image"])
        mesh_ids = self.create_mesh_annotations(self._canvas.get_muted_color("Mesh"),
                                                mesh_landmarks,
                                                coordinates,
                                                tags["mesh"])
        self._current_face_id += 1
        logger.trace("image_id: %s, mesh_ids: %s", image_id, mesh_ids)
        return image_id, mesh_ids

    def _get_viewer_tags(self, object_type, frame_index, is_multi):
        """ Generates tags for the given object based on the frame index, the object type,
        the current face identifier and whether multiple faces appear in the given frame.

        Parameters
        ----------
        object_type: ["image" or "mesh"]
            The type of object that these tags will be associated with
        frame_index: int
            The frame index that this object appears in
        is_multi: bool
            ``True`` if there are multiple faces in the given frame, otherwise ``False``

        Returns
        -------
        list
            The list of tags for the Faces Viewer object
        """
        logger.trace("object_type: %s, frame_index: %s, is_multi: %s",
                     object_type, frame_index, is_multi)
        tags = ["viewer",
                "viewer_{}".format(object_type),
                "frame_id_{}".format(frame_index),
                "face_id_{}".format(self._current_face_id),
                "{}_{}".format(object_type, frame_index),
                "{}_face_id_{}".format(object_type, self._current_face_id)]
        if is_multi:
            tags.extend(["multi", "multi_{}".format(object_type)])
        else:
            tags.append("not_multi")
        logger.trace("tags: %s", tags)
        return tags

    def create_mesh_annotations(self, color, mesh_landmarks, offset, tag):
        """ Creates the mesh annotation for the landmarks. This is made up of a series
        of polygons or lines, depending on which part of the face is being annotated.

        Parameters
        ----------
        color: str
            The hex code for the color that the mesh should be displayed as
        mesh_landmarks: dict
            A dictionary containing the keys `landmarks` holding a `list` of :class:`numpy.ndarray`
            objects and `is_poly` containing a `list` of `bool` types corresponding to the
            `landmarks` indicating whether a line or polygon should be created for each mesh
            annotation
        offset: :class:`numpy.ndarray`
            The top left co-ordinates of the face that corresponds to the given landmarks.
            The mesh annotations will be offset by this amount, to place them in the correct
            place on the canvas
        tag: list
            The list of tags, as generated in :func:`_get_viewer_tags` that are to applied to these
            mesh annotations

        Returns
        -------
        list
            The canvas object ids for the created mesh annotations
        """
        retval = []
        state = "normal" if self._canvas.optional_annotations["mesh"] else "hidden"
        kwargs = dict(polygon=dict(fill="", outline=color), line=dict(fill=color))
        logger.trace("color: %s, offset: %s, tag: %s, state: %s, kwargs: %s",
                     color, offset, tag, state, kwargs)
        for is_poly, landmarks in zip(mesh_landmarks["is_poly"], mesh_landmarks["landmarks"]):
            key = "polygon" if is_poly else "line"
            tags = tag + ["mesh_{}".format(key)]
            obj = getattr(self._canvas, "create_{}".format(key))
            obj_kwargs = kwargs[key]
            coords = (landmarks + offset).flatten()
            retval.append(obj(*coords, state=state, width=1, tags=tags, **obj_kwargs))
        return retval


class HoverBox():  # pylint:disable=too-few-public-methods
    """ Handle the current mouse location in the :class:`~tools.manual.FacesViewer`.

    Highlights the face currently underneath the cursor and handles actions when clicking
    on a face.

    Parameters
    ----------
    canvas: :class:`tkinter.Canvas`
        The :class:`~tools.manual.FacesViewer` canvas
    """
    def __init__(self, canvas):
        logger.debug("Initializing: %s (canvas: %s)", self.__class__.__name__, canvas)
        self._canvas = canvas
        self._frames = canvas._frames
        self._alignments = canvas._alignments
        self._face_size = canvas._faces_cache.size
        self._box = self._canvas.create_rectangle(0, 0, self._face_size, self._face_size,
                                                  outline="#FFFF00",
                                                  width=2,
                                                  state="hidden")
        self._canvas.bind("<Leave>", lambda e: self._clear())
        self._canvas.bind("<Motion>", self.on_hover)
        self._canvas.bind("<ButtonPress-1>", lambda e: self._select_frame())
        logger.debug("Initialized: %s", self.__class__.__name__)

    def on_hover(self, event):
        """ The mouse cursor display as bound to the mouse#s <Motion> event.
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
            pnts = np.array((self._canvas.winfo_pointerx(), self._canvas.winfo_pointery()))
            pnts -= np.array((self._canvas.winfo_rootx(), self._canvas.winfo_rooty()))
        else:
            pnts = (event.x, event.y)

        coords = (self._canvas.canvasx(pnts[0]), self._canvas.canvasy(pnts[1]))
        item_id = next((idx for idx in self._canvas.find_overlapping(*coords, *coords)
                        if self._canvas.type(idx) == "image"), None)
        if item_id is None or any(pnt < 0 for pnt in pnts):
            self._clear()
            self._canvas.config(cursor="")
            return
        if self._canvas.frame_index_from_object(item_id) == self._frames.tk_position.get():
            self._clear()
            self._canvas.config(cursor="")
            return
        self._canvas.config(cursor="hand1")
        self._highlight(item_id)

    def _clear(self):
        """ Hide the hover box when the mouse is not over a face. """
        if self._canvas.itemcget(self._box, "state") != "hidden":
            self._canvas.itemconfig(self._box, state="hidden")

    def _highlight(self, item_id):
        """ Display the hover box around the face the the mouse is currently over.

        Parameters
        ----------
        item_id: int
            The tkinter canvas object id that the mouse is over.
        """
        top_left = np.array(self._canvas.coords(item_id))
        coords = (*top_left, *top_left + self._face_size)
        self._canvas.coords(self._box, *coords)
        self._canvas.itemconfig(self._box, state="normal")
        self._canvas.tag_raise(self._box)

    def _select_frame(self):
        """ Select the face and the subsequent frame (in the editor view) when a face is clicked
        on in :class:`~tools.manual.FacesViewer`.
        """
        item_id = next((idx for idx in self._canvas.find_withtag("current")), None)
        logger.trace("item_id: %s", item_id)
        if item_id is None:
            return
        frame_id = self._canvas.frame_index_from_object(item_id)
        logger.trace("frame_id: %s", frame_id)
        if frame_id is None or frame_id == self._frames.tk_position.get():
            return
        transport_id = self._transport_index_from_frame_index(frame_id)
        logger.trace("transport_id: %s", transport_id)
        if transport_id is None:
            return
        self._frames.stop_playback()
        self._frames.tk_transport_position.set(transport_id)

    def _transport_index_from_frame_index(self, frame_index):
        """ When a face is clicked on, the transport index for the frames in the editor view needs
        to be retrieved based on the current filter criteria.

        Parameters
        ----------
        frame_index: int
            The absolute index for the frame within the full frames list

        Returns
        int
            The index of the requested frame within the filtered frames view.
        """
        frames_list = self._alignments.get_filtered_frames_list()
        retval = frames_list.index(frame_index) if frame_index in frames_list else None
        logger.trace("frame_index: %s, transport_index: %s", frame_index, retval)
        return retval


class ContextMenu():  # pylint:disable=too-few-public-methods
    """  Enables a right click context menu for the :class:`~tool.manual.FacesViewer`.

    Parameters
    ----------
    canvas: :class:`tkinter.Canvas`
        The :class:`~tools.manual.FacesViewer` canvas
    """
    def __init__(self, canvas):
        logger.debug("Initializing: %s (canvas: %s)", self.__class__.__name__, canvas)
        self._canvas = canvas
        self._faces_cache = canvas._faces_cache
        self._menu = RightClickMenu(["Delete Face"], [self._delete_face])
        self._face_id = None
        self._canvas.bind("<Button-2>" if platform.system() == "Darwin" else "<Button-3>",
                          self._pop_menu)
        logger.debug("Initialized: %s", self.__class__.__name__)

    def _pop_menu(self, event):
        """ Pop up the context menu on a right click mouse event.

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The mouse event that has triggered the pop up menu
        """
        if not self._faces_cache.is_initialized:
            return
        coords = (self._canvas.canvasx(event.x), self._canvas.canvasy(event.y))
        self._face_id = next((idx for idx in self._canvas.find_overlapping(*coords, *coords)
                              if self._canvas.type(idx) == "image"), None)
        if self._face_id is None:
            logger.trace("No valid item under mouse")
            return
        logger.trace("Popping right click menu")
        self._menu.popup(event)

    def _delete_face(self):
        """ Delete the selected face on a right click mouse delete action. """
        self._canvas.update_face.remove_face_from_viewer(self._face_id)
        self._face_id = None


class ActiveFrame():
    """ Holds the objects and handles faces for the currently selected frame.

    Parameters
    ----------
    canvas: :class:`tkinter.Canvas`
        The :class:`~tools.manual.FacesViewer` canvas
    """
    def __init__(self, canvas):
        logger.debug("Initializing: %s (canvas: %s)", self.__class__.__name__, canvas)
        self._canvas = canvas
        self._alignments = canvas._alignments
        self._faces_cache = canvas._faces_cache
        self._size = canvas._faces_cache.size
        self._tk_position = canvas._frames.tk_position
        self._highlighter = Highlighter(self)
        self._tk_position.trace("w", lambda *e: self.reload_annotations())
        self._alignments.tk_edited.trace("w", lambda *e: self._update())
        logger.debug("Initialized: %s", self.__class__.__name__)

    @property
    def face_count(self):
        """ int: The count of faces in the currently selected frame. """
        return len(self.image_ids)

    @property
    def frame_index(self):
        """ int: The currently selected frame's index. """
        return self._tk_position.get()

    @property
    def image_ids(self):
        """ tuple: The tkinter canvas image ids for the currently selected frame's faces. """
        return self._canvas.find_withtag("image_{}".format(self.frame_index))

    @property
    def mesh_ids(self):
        """ tuple: The tkinter canvas mesh ids for the currently selected frame's faces. """
        return self._canvas.find_withtag("mesh_{}".format(self.frame_index))

    def reload_annotations(self):
        """ Refresh the highlighted annotations for faces in the currently selected frame on an
        add/remove face. """
        if not self._faces_cache.is_initialized:
            return
        self._highlighter.highlight_selected()

    def _update(self):
        """ Update the highlighted annotations for faces in the currently selected frame on an
        update, add or remove. """
        if not self._alignments.tk_edited.get() or not self._faces_cache.is_initialized:
            return
        if self._add_remove_face():
            self.reload_annotations()
            return
        self._canvas.update_face.update(self.frame_index, self._alignments.face_index)
        self._highlighter.highlight_selected()
        self._alignments.tk_edited.set(False)

    def _add_remove_face(self):
        """ Check the number of displayed faces against the number of faces stored in the
        alignments data for the currently selected frame, and add or remove if appropriate. """
        alignment_faces = len(self._alignments.current_faces)
        if alignment_faces > self._canvas.active_frame.face_count:
            self._canvas.update_face.add(self._canvas.active_frame.frame_index)
            retval = True
        elif alignment_faces < self._canvas.active_frame.face_count:
            self._canvas.update_face.remove(self._canvas.active_frame.frame_index,
                                            self._alignments.get_removal_index())
            retval = True
        else:
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
        self._size = active_frame._canvas._faces_cache.size
        self._canvas = active_frame._canvas
        self._faces_cache = active_frame._canvas._faces_cache
        self._active_frame = active_frame
        self._tk_vars = dict(selected_editor=self._canvas._display_frame.tk_selected_action,
                             selected_mask=self._canvas._display_frame.tk_selected_mask,
                             optional_annotations=self._canvas._tk_optional_annotations)
        self._objects = dict(image_ids=[], mesh_ids=[], boxes=[])
        self._hidden_boxes_count = 0
        self._prev_objects = dict()
        self._tk_vars["selected_editor"].trace("w", lambda *e: self._highlight_annotations())
        logger.debug("Initialized: %s", self.__class__.__name__,)

    @property
    def _frame_index(self):
        """ int: The currently selected frame's index. """
        return self._active_frame.frame_index

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
        mask_view = self._tk_vars["optional_annotations"]["mask"].get()
        mask_edit = self._tk_vars["selected_editor"].get() == "Mask"
        if self._prev_objects.get("mask", None) is None or mask_view == mask_edit:
            self._prev_objects["mask"] = None
            return
        mask_type = self._tk_vars["selected_mask"].get() if mask_view else None
        self._faces_cache.update_selected(self._prev_objects["mask"], mask_type)
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
        state = "normal" if self._tk_vars["optional_annotations"]["mesh"].get() else "hidden"
        # TODO None type error on face deletion
        for mesh_id in self._prev_objects["mesh"]:
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
        mask_view = self._tk_vars["optional_annotations"]["mask"].get()
        if mask_edit == mask_view:
            return
        mask_type = self._tk_vars["selected_mask"].get() if mask_edit else None
        self._faces_cache.update_selected(self._frame_index, mask_type)
        self._prev_objects["mask"] = self._frame_index

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
                     or self._tk_vars["optional_annotations"]["mesh"].get())
        color = self._canvas.control_colors["Mesh"]
        kwargs = dict(polygon=dict(fill="", outline=color),
                      line=dict(fill=color))
        state = "normal" if show_mesh else "hidden"
        self._prev_objects["mesh"] = []
        for mesh_id in self._objects["mesh_ids"]:
            self._canvas.itemconfig(mesh_id, **kwargs[self._canvas.type(mesh_id)], state=state)
            self._prev_objects["mesh"].append(mesh_id)


class UpdateFace():
    """ Handles all adding, removing and updating of faces in the
        :class:`~tools.manual.FacesViewer` canvas when a user performs an edit.

    Parameters
    ----------
    canvas: :class:`tkinter.Canvas`
        The :class:`~tools.manual.FacesViewer` canvas
    """

    def __init__(self, canvas):
        logger.debug("Initializing: %s (canvas: %s)", self.__class__.__name__, canvas)
        self._canvas = canvas
        self._alignments = canvas._alignments
        self._faces_cache = canvas._faces_cache
        self._frames = canvas._frames
        logger.debug("Initialized: %s", self.__class__.__name__)

    # TODO moving to new frame and adding faces seems to mess up the tk_face of the existing face

    # << ADD FACE METHODS >> #
    def add(self, frame_index):
        """ Add a face to the :class:`~tools.manual.FacesViewer` canvas for the given frame.

        Generates the face image and mesh annotations for a newly added face, creates the relevant
        tags into the correct location in the object stack.

        Parameters
        ----------
        frame_index: int
            The frame index to add the face for
        """
        face_idx = len(self._canvas.find_withtag("image_{}".format(frame_index)))
        logger.debug("Adding face to frame: (frame_index: %s, face_index: %s)",
                     frame_index, face_idx)
        # Add objects to cache
        tk_face, mesh_landmarks = self._canvas.get_tk_face_and_landmarks(frame_index, face_idx)
        self._faces_cache.add(frame_index, tk_face, mesh_landmarks)
        # Create new annotations
        image_id, mesh_ids = self._canvas.new_objects.create((0, 0),
                                                             tk_face,
                                                             mesh_landmarks,
                                                             frame_index)
        # Place in stack
        frame_tag = self._update_multi_tags(frame_index)
        self._place_in_stack(frame_index, frame_tag)
        # Update viewer
        self._canvas.active_filter.add_face(image_id, mesh_ids, mesh_landmarks["landmarks"])

    def _place_in_stack(self, frame_index, frame_tag):
        """ Place newly added faces in the correct location in the object stack.

        Parameters
        ----------
        frame_index: int
            The frame index that the face(s) need to be placed for
        frame_tag: str
            The tag of the canvas objects that need to be placed
        """
        next_frame_idx = self._get_next_frame_index(frame_index)
        if next_frame_idx is not None:
            next_tag = "frame_id_{}".format(next_frame_idx)
            logger.debug("Lowering annotations for frame %s below frame %s", frame_tag, next_tag)
            self._canvas.tag_lower(frame_tag, next_tag)

    def _get_next_frame_index(self, frame_index):
        """ Get the index of the next frame that contains faces.

        Used for calculating the correct location to place the newly created objects in the stack.

        Parameters
        ----------
        frame_index: int
            The frame index for the objects that require placing in the stack

        Returns
        -------
        int or ``None``
            The frame index for the next frame that contains faces. ``None`` is returned if the
            given frame index is already at the end of the stack
        """
        next_frame_idx = next((
            idx for idx, f_count in enumerate(self._alignments.face_count_per_index[frame_index:])
            if f_count > 0), None)
        if next_frame_idx is None:
            return None
        next_frame_idx += frame_index + 1
        logger.debug("Returning next frame with faces: %s for frame index: %s",
                     next_frame_idx, frame_index)
        return next_frame_idx

    # << REMOVE FACE METHODS >> #
    def remove_face_from_viewer(self, item_id):
        """ Remove a face and it's annotations from the :class:`~tools.manual.FacesViewer` canvas
        for the given item identifier. Also removes the alignments data from the alignments file,
        and the cached face data from :class:`~tools.lib_manual.media.FacesCache`.

        This action is specifically called when a face is deleted from the viewer through a right
        click menu action.

        parameters
        ----------
        item_id: int
            The face group item identifier stored in the face's object tags
        """
        frame_idx = self._canvas.frame_index_from_object(item_id)
        face_idx = self._canvas.find_withtag("image_{}".format(frame_idx)).index(item_id)
        logger.debug("item_id: %s, frame_index: %s, face_index: %s", item_id, frame_idx, face_idx)
        self._alignments.delete_face_at_index_by_frame(frame_idx, face_idx)
        self.remove(frame_idx, face_idx)
        if frame_idx == self._frames.tk_position.get():
            self._frames.tk_update.set(True)
            self._canvas.active_frame.reload_annotations()

    def remove(self, frame_index, face_index):
        """ Remove a face and it's annotations from the :class:`~tools.manual.FacesViewer` canvas
        for the given face index at the given frame index. Also removes the cached face data from
        :class:`~tools.lib_manual.media.FacesCache`.

        Parameters
        ----------
        frame_index: int
            The frame index that contains the face and objects that are to be removed
        face_index: int
            The index of the face within the given frame that is to have its objects removed
        """
        logger.debug("Removing face for frame %s at index: %s", frame_index, face_index)
        self._faces_cache.remove(frame_index, face_index)
        image_id = self._canvas.find_withtag("image_{}".format(frame_index))[face_index]
        # Retrieve the position of the face in the current viewer prior to deletion
        display_index = self._canvas.active_filter.image_ids.index(image_id)
        self._canvas.delete(self._canvas.face_id_from_object(image_id))
        self._update_multi_tags(frame_index)
        self._canvas.active_filter.remove_face(frame_index, face_index, display_index)

    # << ADD AND REMOVE METHODS >> #
    def _update_multi_tags(self, frame_index):
        """ Update the tags indicating whether this frame contains multiple faces or not.

        Parameters
        ----------
        frame_index: int
            The frame index that the tags are to be updated for
        """
        image_tag = "image_{}".format(frame_index)
        mesh_tag = "mesh_{}".format(frame_index)
        frame_tag = "frame_id_{}".format(frame_index)
        num_faces = len(self._canvas.find_withtag(image_tag))
        logger.debug("image_tag: %s, frame_tag: %s, faces_count: %s",
                     image_tag, frame_tag, num_faces)
        if num_faces == 0:
            return None
        self._canvas.dtag(frame_tag, "not_multi")
        self._canvas.dtag(frame_tag, "multi")
        self._canvas.dtag(frame_tag, "multi_image")
        self._canvas.dtag(frame_tag, "multi_mesh")
        if num_faces > 1:
            self._canvas.addtag_withtag("multi", frame_tag)
            self._canvas.addtag_withtag("multi_image", image_tag)
            self._canvas.addtag_withtag("multi_mesh", mesh_tag)
        else:
            self._canvas.addtag_withtag("not_multi", frame_tag)
        return frame_tag

    # << UPDATE METHODS >> #
    def update(self, frame_index, face_index):
        """  Update the face and annotations for the given face index at the given frame index.

        This method is called when an editor update is made. It updates the displayed annotations
        as well as the meta information stored in :class:`~tools.lib_manual.media.FacesCache`.

        Parameters
        ----------
        frame_index: int
            The frame index that contains the face and objects that are to be updated
        face_index: int
            The index of the face within the given frame that is to have its objects updated
        """
        # TODO Decide what to update based on current edit mode
        tk_face, mesh_landmarks = self._canvas.get_tk_face_and_landmarks(frame_index, face_index)
        self._faces_cache.update(frame_index, face_index, tk_face, mesh_landmarks)
        image_id = self._canvas.find_withtag("image_{}".format(frame_index))[face_index]
        self._canvas.itemconfig(image_id, image=tk_face)
        coords = self._canvas.coords(image_id)
        mesh_ids = self._canvas.mesh_ids_for_face_id(image_id)
        logger.trace("frame_index: %s, face_index: %s, image_id: %s, coords: %s, mesh_ids: %s, "
                     "tk_face: %s, mesh_landmarks: %s", frame_index, face_index, image_id, coords,
                     mesh_ids, tk_face, mesh_landmarks)
        for points, item_id in zip(mesh_landmarks["landmarks"], mesh_ids):
            self._canvas.coords(item_id, *(points + coords).flatten())
