#!/usr/bin/env python3
""" Filters for the Faces Viewer for the manual adjustments tool """
import logging

import numpy as np

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


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
        # TODO Do the maths rather than process twice on last multi_face
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
        for image_id in self.image_ids:
            self._canvas.itemconfig(self._canvas.face_id_from_object(image_id), state="hidden")
        self._temporary_image_ids = []

    def de_initialize(self):
        """ Remove the trace variable on when changing filter. """
        if self._filter_type == "no_faces":
            logger.debug("Removing on_frame_change_var")
            self._tk_position.trace_vdelete("w", self._tk_position_callback)
