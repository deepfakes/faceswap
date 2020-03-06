#!/usr/bin/env python3
""" Filters for the Faces Viewer for the manual adjustments tool """
import logging

import numpy as np

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


# TODO Right click erase on mask editor

class FaceFilter():
    """ Base class for different faces view filters.

    All Filter views inherit from this class. Handles the layout of faces and annotations in the
    face viewer.

    Parameters
    ----------
    face_cache: :class:`FaceCache`
        The main Face Viewers face cache, that holds all of the display items and annotations
    toggle_tags: dict or ``None``, optional
        Tags for toggling optional annotations. Set to ``None`` if there are no annotations to
        be toggled for the selected filter. Default: ``None``
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
        logger.debug("Initialized: %s", self.__class__.__name__)

    @property
    def _view_states(self):
        """ dict: The Filter type mapped to tag hidden states. """
        return dict(all_frames=dict(normal="viewer"),
                    no_faces=dict(hidden="viewer"),
                    multiple_faces=dict(hidden="not_multi", normal="multi"))

    @property
    def _tag_prefix(self):
        """ The tag prefix for the current filter """
        return self._view_states[self._filter_type].get("normal", None)

    @property
    def image_ids(self):
        """ list: The canvas item ids for the face images. """
        if self._filter_type == "no_faces":
            return self._temporary_image_ids
        return self._canvas.find_withtag("{}_image".format(self._tag_prefix))

    @property
    def _optional_tags(self):
        """ list: The tags for the optional annotations. """
        options = ["mesh"]
        if self._filter_type == "no_faces":
            return dict()
        return {opt: "{}_{}".format(self._tag_prefix, opt) for opt in options}

    @property
    def filter_type(self):
        """ str: The currently selected filter type """
        return self._filter_type

    def _set_object_display_state(self):
        """ Hide annotations that are not relevant for this filter type and show those
        that are. """
        opt = self._canvas.optional_annotation
        for state, tag in self._view_states[self._filter_type].items():
            # TODO When mask comes in this will need to be more complex
            tag += "_image" if state == "normal" and opt is None else ""
            logger.debug("Setting state to '%s' for tag '%s' in filter_type: '%s'",
                         state, tag, self._filter_type)
            self._canvas.itemconfig(tag, state=state)

    def _set_initial_layout(self):
        """ Layout the faces on the canvas for the current filter type. """
        for idx, image_id in enumerate(self.image_ids):
            old_position = np.array(self._canvas.coords(image_id), dtype="int")
            new_position = self._canvas.coords_from_index(idx)
            offset = new_position - old_position
            if not offset.any():
                continue
            self._canvas.move(self._canvas.face_id_from_object(image_id), *offset)
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    # TODO Deleting faces on multi face filter is broken
    def add_face(self, image_id, mesh_ids, mesh_landmarks):
        """ Place a newly added face in the correct location and move subsequent faces to their new
        location.

        Parameters
        ----------
        frame_index: int
            The frame index that the face is to be added to
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
        """ Remove a face at the given location and update subsequent faces to the
        correct location.
        """
        logger.debug("frame_index: %s, display_index: %s", frame_index, display_index)
        if self._filter_type == "no_faces":
            del self.image_ids[display_index]
        display_index, last_multi = self._check_last_multi(frame_index, face_index, display_index)
        self._update_layout(display_index, is_insert=False, last_multi_face=last_multi)

    def _check_last_multi(self, frame_index, face_index, display_index):
        """ For multi faces viewer, if a deletion has taken us down to 1 face then hide the last
        face and set the display index to the correct position for hiding 2 faces. """
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

    # TODO Adding faces to last frame breaks this
    def _tag_objects_to_move(self, start_index, is_insert):
        """ Tag the 3 object groups that require moving.

        Any hidden annotations are set to normal so that the :func:`tkinter.Canvas.find_enclosed`
        function picks them up for moving. They are immediately hidden again.

        Parameters
        ----------
        starting_index: int
            The starting absolute face index that new locations should be calculated for
        is_insert: bool
            ``True`` if adjusting display for an added face, ``False`` if adjusting for a removed
            face.
         """
        # Display hidden annotations so they get tagged
        hidden_tags = [tag for key, tag in self._optional_tags.items()
                       if key != self._canvas.optional_annotation]
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

    def toggle_annotation(self):
        """ Toggle additional object annotations on or off. """
        if not self._optional_tags:
            return
        display = self._canvas.optional_annotation
        if display is not None:
            self._canvas.itemconfig(self._optional_tags[display], state="normal")
        else:
            for tag in self._optional_tags.values():
                self._canvas.itemconfig(tag, state="hidden")

    @staticmethod
    def _get_toggle_item_ids(objects, display):
        """ Return the item_ids in a single list for items that are to be toggled.

        Parameters
        ----------
        objects: dict
            The objects dictionary for the current face
        display: str or ``None``
            The key for the object annotation that is to be toggled or ``None`` if annotations
            are to be hidden

        Returns
        -------
        list
            The list of canvas object ids that are to be toggled
        """
        if display is not None:
            retval = objects[display] if isinstance(objects[display], list) else [objects[display]]
        else:
            retval = []
            for key, val in objects.items():
                if key == "image_id":
                    continue
                retval.extend(val if isinstance(val, list) else [val])
        logger.trace(retval)
        return retval

    def _on_frame_change(self, *args):  # pylint:disable=unused-argument
        """ Callback to be executed whenever the frame is changed.

        For no-faces filter, if faces have been added to the current frame, hide the new faces
        and remove the image_ids from tracking.
        """
        if self._filter_type != "no_faces":
            return
        for image_id in self.image_ids:
            self._canvas.itemconfig(self._canvas.face_id_from_object(image_id), state="hidden")
        self._temporary_image_ids = []

    def de_initialize(self):
        """ Remove the trace variable on de-initialization """
        if self._filter_type == "no_faces":
            logger.debug("Removing on_frame_change_var")
            self._tk_position.trace_vdelete("w", self._tk_position_callback)
