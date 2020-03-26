#!/usr/bin/env python3
""" Handle the highlighting, filtering and display characteristics of objects in the Face viewer
for the manual adjustments tool """
import logging
import platform

import numpy as np

from lib.gui.custom_widgets import RightClickMenu

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class HoverBox():  # pylint:disable=too-few-public-methods
    """ Handle the current mouse location in the :class:`~tools.manual.FacesViewer`.

    Highlights the face currently underneath the cursor and handles actions when clicking
    on a face.

    Parameters
    ----------
    canvas: :class:`tkinter.Canvas`
        The :class:`~tools.manual.FacesViewer` canvas
    detected_faces: :class:`~tool.manual.faces.DetectedFaces`
        The :class:`~lib.faces_detect.DetectedFace` objects for this video
    """
    def __init__(self, canvas, detected_faces):
        logger.debug("Initializing: %s (canvas: %s)", self.__class__.__name__, canvas)
        self._canvas = canvas
        self._frames = canvas._frames
        self._det_faces = detected_faces
        self._face_size = canvas._faces_cache.size
        self._box = self._canvas.create_rectangle(0, 0, self._face_size, self._face_size,
                                                  outline="#0000ff",
                                                  width=2,
                                                  state="hidden",
                                                  fill="#0000ff",
                                                  stipple="gray12",
                                                  tags="hover_box")
        self._current_frame_index = None
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
            self._current_frame_index = None
            return
        if self._canvas.frame_index_from_object(item_id) == self._frames.tk_position.get():
            self._clear()
            self._canvas.config(cursor="")
            self._current_frame_index = None
            return
        self._canvas.config(cursor="hand1")
        self._highlight(item_id)
        self._current_frame_index = self._canvas.frame_index_from_object(item_id)

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
        frame_id = self._current_frame_index
        if frame_id is None or frame_id == self._frames.tk_position.get():
            return
        transport_id = self._transport_index_from_frame_index(frame_id)
        logger.trace("frame_index: %s, transport_id: %s", frame_id, transport_id)
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
        frames_list = self._det_faces.filter.frames_list
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
        logger.trace("Right click delete received. face_id: %s", self._face_id)
        self._canvas.update_face.remove_face_from_viewer(self._face_id)
        self._face_id = None


class ActiveFrame():
    """ Holds the objects and handles faces for the currently selected frame.

    Parameters
    ----------
    canvas: :class:`tkinter.Canvas`
        The :class:`~tools.manual.FacesViewer` canvas
    detected_faces: :class:`~tool.manual.faces.DetectedFaces`
        The :class:`~lib.faces_detect.DetectedFace` objects for this video
    """
    def __init__(self, canvas, detected_faces):
        logger.debug("Initializing: %s (canvas: %s)", self.__class__.__name__, canvas)
        self._canvas = canvas
        self._det_faces = detected_faces
        self._faces_cache = canvas._faces_cache
        self._size = canvas._faces_cache.size
        self._tk_position = canvas._frames.tk_position
        self._highlighter = Highlighter(self)
        self._tk_position.trace("w", lambda *e: self.reload_annotations())
        self._det_faces.tk_edited.trace("w", lambda *e: self._update())
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
        logger.trace("Reloading annotations")
        self._highlighter.highlight_selected()

    def _update(self):
        """ Update the highlighted annotations for faces in the currently selected frame on an
        update, add or remove. """
        if not self._det_faces.tk_edited.get() or not self._faces_cache.is_initialized:
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
        alignment_faces = len(self._det_faces.current_faces[self.frame_index])
        logger.trace("alignment_faces: %s, face_count: %s", alignment_faces, self.face_count)
        if alignment_faces > self.face_count:
            logger.debug("Adding face")
            self._canvas.update_face.add(self._canvas.active_frame.frame_index)
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
        self._size = active_frame._canvas._faces_cache.size
        self._canvas = active_frame._canvas
        self._faces_cache = active_frame._canvas._faces_cache
        self._active_frame = active_frame
        self._tk_vars = dict(selected_editor=self._canvas._display_frame.tk_selected_action,
                             selected_mask=self._canvas._display_frame.tk_selected_mask)
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
        mask_view = self._canvas.optional_annotations["mask"]
        mask_edit = self._tk_vars["selected_editor"].get() == "Mask"
        if self._prev_objects.get("mask", None) is None or mask_view == mask_edit:
            self._prev_objects["mask"] = None
            return
        mask_type = self._tk_vars["selected_mask"].get() if mask_view else None
        self._faces_cache.mask_loader.update_selected(self._prev_objects["mask"], mask_type)
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
        self._faces_cache.mask_loader.update_selected(self._frame_index, mask_type)
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
                     or self._canvas.optional_annotations["mesh"])
        color = self._canvas.control_colors["Mesh"]
        kwargs = dict(polygon=dict(fill="", outline=color),
                      line=dict(fill=color))
        state = "normal" if show_mesh else "hidden"
        self._prev_objects["mesh"] = []
        for mesh_id in self._objects["mesh_ids"]:
            self._canvas.itemconfig(mesh_id, **kwargs[self._canvas.type(mesh_id)], state=state)
            self._prev_objects["mesh"].append(mesh_id)


class FaceFilter():
    """ Filters the face viewer based on the selection in the Filter pull down menu.

    The layout of faces and annotations in the face viewer are handled by this class.

    Parameters
    ----------
    canvas: :class:`tkinter.Canvas`
        The :class:`~tools.manual.FacesViewer` canvas
    filter_type: ["all_frames", "no_faces", "multiple_faces"]
        The currently selected filter type from the pull down menu.
    """
    def __init__(self, canvas, filter_type):
        logger.debug("Initializing: %s: (canvas: %s, filter_type: %s)",
                     self.__class__.__name__, canvas, filter_type)
        self._canvas = canvas
        self._filter_type = filter_type
        self._tk_position = canvas._frames.tk_position
        self._size = canvas._faces_cache.size
        self._temporary_image_ids = []
        self._tk_position = canvas._frames.tk_position

        self._set_object_display_state()
        self._set_initial_layout()
        if self._filter_type == "no_faces":
            self._tk_position_callback = self._tk_position.trace("w", self._on_frame_change)
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))
        logger.debug("Initialized: %s", self.__class__.__name__)

    @property
    def _view_states(self):
        """ dict: The :attr:`filter_type` mapped to the object display state for tag prefixes
        that meet the criteria for displaying or hiding objects for the required filter type. """
        return dict(all_frames=dict(normal="viewer"),
                    no_faces=dict(hidden="viewer"),
                    multiple_faces=dict(hidden="not_multi", normal="multi"))

    @property
    def _tag_prefix(self):
        """ str: The tag prefix for objects that should be displayed based on the currently
        selected :attr:`filter_type` """
        return self._view_states[self._filter_type].get("normal", None)

    @property
    def image_ids(self):
        """ list: The canvas item ids for the face images that are currently displayed based on
        the selected :attr`_filter_type`. """
        if self._filter_type == "no_faces":
            return self._temporary_image_ids
        return self._canvas.find_withtag("{}_image".format(self._tag_prefix))

    @property
    def _optional_tags(self):
        """ list: The optional annotation tags that should be displayed based on "
        :attr:`_filter_type. """
        options = ["mesh"]
        if self._filter_type == "no_faces":
            return dict()
        return {opt: "{}_{}".format(self._tag_prefix, opt) for opt in options}

    @property
    def filter_type(self):
        """ str: The currently selected filter type (one of "all_frames", "no_faces",
        multiple_faces"). """
        return self._filter_type

    def _set_object_display_state(self):
        """ On a filter change, displays items that should be shown for the requested
        :attr:`filter_type` and optional annotations and hides those annotations that should
        not be displayed. """
        mesh_state = self._canvas.optional_annotations["mesh"]
        for state, tag in self._view_states[self._filter_type].items():
            tag += "_image" if state == "normal" and not mesh_state else ""
            logger.debug("Setting state to '%s' for tag '%s' in filter_type: '%s'",
                         state, tag, self._filter_type)
            self._canvas.itemconfig(tag, state=state)

    def _set_initial_layout(self):
        """ Layout the faces on the canvas for the selected :attr:`filter_type`.

        The position of each face on the canvas can vary significantly between different
        filter views. Unfortunately there is no getting away from the fact that a lot of
        faces may need to be moved, and this may take some time. Some time is attempted
        to be saved by only moving those faces that require moving.
        """
        for idx, image_id in enumerate(self.image_ids):
            old_position = np.array(self._canvas.coords(image_id), dtype="int")
            new_position = self._canvas.coords_from_index(idx)
            offset = new_position - old_position
            if not offset.any():
                continue
            self._canvas.move(self._canvas.face_id_from_object(image_id), *offset)

    def add_face(self, image_id, mesh_ids, mesh_landmarks):
        """ Place a newly added face in the correct location for the current :attr:`filter_type`
        and shift subsequent faces to their new locations.

        Parameters
        ----------
        image_id: int
            The tkinter canvas object id of the image that is being added
        mesh_ids: list
            The tkinter canvas object ids of the mesh annotation that is being added
        mesh_landmarks: list
            The co-ordinates of each of the mesh annotations based from (0, 0)
        """
        if self._filter_type == "no_faces":
            self.image_ids.append(image_id)
        display_idx = self.image_ids.index(image_id)
        # Update layout - Layout must be updated first so space is made for the new face
        self._update_layout(display_idx + 1, is_insert=True)

        coords = self._canvas.coords_from_index(display_idx)
        logger.debug("display_idx: %s, coords: %s", display_idx, coords)
        # Place newly created annotations
        self._canvas.coords(image_id, *coords)
        for mesh_id, landmarks in zip(mesh_ids, mesh_landmarks):
            self._canvas.coords(mesh_id, *(landmarks + coords).flatten())

    def remove_face(self, frame_index, face_index, display_index):
        """ Remove a face at the given location for the current :attr:`filter_type` and update
        subsequent faces to their correct locations.

        Parameters
        ----------
        frame_index: int
            The frame index that the face is to be removed from
        face_index: int
            The index of the face within the frame
        display_index: int
            The absolute index within the faces viewer that the face is to be deleted from
        """
        logger.debug("frame_index: %s, display_index: %s", frame_index, display_index)
        if self._filter_type == "no_faces":
            del self.image_ids[display_index]
        display_index, last_multi = self._check_last_multi(frame_index, face_index, display_index)
        self._update_layout(display_index, is_insert=False, last_multi_face=last_multi)

    def _check_last_multi(self, frame_index, face_index, display_index):
        """ For multi faces viewer, if a deletion has taken us down to 1 face then hide the last
        face and set the display index to the correct position for hiding 2 faces.

        Parameters
        ----------
        frame_index: int
            The frame index that the face is to be removed from
        face_index: int
            The index of the face within the frame
        display_index: int
            The absolute index within the faces viewer that the face is to be deleted from
        """
        if self._filter_type != "multiple_faces":
            retval = (display_index, False)
        elif len(self._canvas.find_withtag("image_{}".format(frame_index))) != 1:
            retval = (display_index, False)
        else:
            self._canvas.itemconfig("frame_id_{}".format(frame_index), state="hidden")
            retval = (display_index - face_index, True)
        logger.debug("display_index: %s, last_multi_face: %s", *retval)
        return retval

    def _update_layout(self, starting_object_index, is_insert=True, last_multi_face=False):
        """ Reposition faces and annotations on the canvas after a face has been added or removed.

        The faces are moved in bulk in 3 groups: The row that the face is being added to or removed
        from is shifted left or right. The first or last column is shifted to the other side of the
        viewer and 1 row up or down. Finally the block of faces that have not been covered by the
        previous shifts are moved left or right one face.

        Parameters
        ----------
        starting_object_index: int
            The starting absolute face index that new locations should be calculated for
        is_insert: bool, optional
            ``True`` if adjusting display for an added face, ``False`` if adjusting for a removed
            face. Default: ``True``
        last_multi_face: bool, optional
            Only used for multi-face view. ``True`` if the current frame is moving from having
            multiple faces in the frame to having a single face in the frame, otherwise ``False``.
            Default: ``False``
        """
        # TODO addtag_enclosed vs addtag_overlapping - The mesh going out of face box problem
        if len(self.image_ids) == starting_object_index:
            logger.debug("Update is last face. Not changing layout")
            return
        # TODO Do the calculation rather than process twice on last multi_face
        for _ in range(2 if last_multi_face else 1):
            top_bulk_delta = np.array((self._size, 0))
            col_delta = np.array((-(self._canvas.column_count - 1) * self._size, self._size))
            if not is_insert:
                top_bulk_delta *= -1
                col_delta *= -1
            self._tag_objects_to_move(starting_object_index, is_insert)
            self._canvas.move("move_top", *top_bulk_delta)
            self._canvas.move("move_col", *col_delta)
            self._canvas.move("move_bulk", *top_bulk_delta)

            self._canvas.dtag("move_top", "move_top")
            self._canvas.dtag("move_col", "move_col")
            self._canvas.dtag("move_bulk", "move_bulk")

        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _tag_objects_to_move(self, start_index, is_insert):
        """ Tag the 3 object groups that require moving.

        Any hidden annotations are set to normal so that the :func:`tkinter.Canvas.find_enclosed`
        function picks them up for moving. They are immediately hidden again.

        Parameters
        ----------
        start_index: int
            The starting absolute face index that new locations should be calculated for
        is_insert: bool
            ``True`` if adjusting display for an added face, ``False`` if adjusting for a removed
            face.
         """
        # Display hidden annotations so they get tagged
        mesh_state = self._canvas.optional_annotations["mesh"]
        hidden_tags = [tag for key, tag in self._optional_tags.items()] if not mesh_state else []
        for tag in hidden_tags:
            self._canvas.itemconfig(tag, state="normal")
        first_face_xy = self._canvas.coords(self.image_ids[start_index])
        new_row_start = not is_insert and first_face_xy[0] == 0
        first_row_y = first_face_xy[1] - self._size if new_row_start else first_face_xy[1]
        last_col_x = (self._canvas.column_count - 1) * self._size
        logger.debug("is_insert: %s, start_index: %s, new_row_start: %s, first_face_xy: %s, "
                     "first_row_y: %s, last_col_x: %s", is_insert, start_index, new_row_start,
                     first_face_xy, first_row_y, last_col_x)
        # Top Row
        if not new_row_start:
            # Skip top row shift if starting index is a new row
            br_top_xy = (last_col_x if is_insert else last_col_x + self._size,
                         first_row_y + self._size)
            logger.debug("first row: (top left: %s, bottom right:%s)", first_face_xy, br_top_xy)
            self._canvas.addtag_enclosed("move_top", *first_face_xy, *br_top_xy)
        # First or last column (depending on delete or insert)
        tl_col_xy = (last_col_x if is_insert else 0,
                     first_row_y if is_insert else first_row_y + self._size)
        br_col_xy = (tl_col_xy[0] + self._size, self._canvas.bbox("all")[3])
        logger.debug("end column: (top left: %s, bottom right:%s)", tl_col_xy, br_col_xy)
        self._canvas.addtag_enclosed("move_col", *tl_col_xy, *br_col_xy)
        # Bulk faces
        tl_bulk_xy = (0 if is_insert else self._size, first_row_y + self._size)
        br_bulk_xy = (last_col_x if is_insert else last_col_x + self._size,
                      self._canvas.bbox("all")[3])
        logger.debug("bulk: (top left: %s, bottom right:%s)", tl_bulk_xy, br_bulk_xy)
        self._canvas.addtag_enclosed("move_bulk", *tl_bulk_xy, *br_bulk_xy)
        # Re-hide hidden annotations
        for tag in hidden_tags:
            self._canvas.itemconfig(tag, state="hidden")

    def toggle_annotation(self, mesh_state):
        """ Toggle additional object annotations on or off.

        Parameters
        ----------
        mesh_state: bool
            ``True`` if optional mesh annotations button is checked otherwise ``False``
        """
        if not self._optional_tags:
            return
        if mesh_state:
            logger.debug("Toggling mesh annotations on")
            self._canvas.itemconfig(self._optional_tags["mesh"], state="normal")
        else:
            for tag in self._optional_tags.values():
                logger.debug("Toggling %s annotations off", tag)
                self._canvas.itemconfig(tag, state="hidden")

    def _on_frame_change(self, *args):  # pylint:disable=unused-argument
        """ Callback to be executed whenever the frame is changed.

        For no-faces filter, if faces have been added to the current frame, hide the new faces
        and remove the image_ids from tracking.

        Parameters
        ----------
        args: tuple
            Unused but required for tkinter trace callback
        """
        if self._filter_type != "no_faces":
            return
        logger.trace("Frame change callback for Faces Viewer")
        for image_id in self.image_ids:
            self._canvas.itemconfig(self._canvas.face_id_from_object(image_id), state="hidden")
        self._temporary_image_ids = []

    def de_initialize(self):
        """ Remove the trace variable on when changing filter. """
        if self._filter_type == "no_faces":
            logger.debug("Removing on_frame_change_var")
            self._tk_position.trace_vdelete("w", self._tk_position_callback)
