#!/usr/bin/env python3
""" Editor objects for the manual adjustments tool """
import logging

from functools import partial

import numpy as np

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Editor():
    """ Parent Class for Object Editors.

    Parameters
    ----------
    canvas: :class:`tkinter.Canvas`
        The canvas that holds the image and annotations
    alignments: :class:`AlignmentsData`
        The alignments data for this manual session
    frames: :class:`FrameNavigation`
        The frames navigator for this manual session
    """
    def __init__(self, canvas, alignments, frames):
        logger.debug("Initializing %s: (canvas: '%s', alignments: %s, frames: %s)",
                     self.__class__.__name__, canvas, alignments, frames)
        self._canvas = canvas
        self._alignments = alignments
        self._frames = frames
        self._colors = dict(red="#ff0000",
                            green="#00ff00",
                            blue="#0000ff",
                            cyan="#00ffff",
                            yellow="#ffff00",
                            magenta="#ff00ff")
        self._objects = []
        self.update_annotation()
        logger.debug("Initialized %s", self.__class__.__name__)

    def update_annotation(self):
        """ Update the display annotations for the current objects.

        Override for specific editors.
        """
        logger.trace("Default annotations. Not storing Objects")
        self._clear_annotation()
        self._objects = []

    def _clear_annotation(self):
        """ Removes all currently drawn annotations for the current :class:`Editor`. """
        for faces in self._objects:
            for obj in faces:
                logger.trace("Deleting object: %s (id: %s)", self._canvas.type(obj), obj)
                self._canvas.delete(obj)

    # Mouse Callbacks
    def set_mouse_cursor_tracking(self):
        """ Default mouse cursor tracking removes all bindings from mouse events to just
        display the standard cursor. Override for specific Editor mouse cursor display.

        NB: Only Mouse Button 1 is cleared as we want to keep Buttons 2 + 3 available for other
        purposes outside of Editor functions.
        """
        for event in ("B1-Motion", "ButtonPress-1", "ButtonRelease-1", "Double-Button-1",
                      "Motion"):
            logger.debug("Unbinding mouse event: %s", event)
            self._canvas.unbind("<{}>".format(event))

    def _scale_to_display(self, points):
        """ Scale and offset the given points to the current display scale and offset values.

        Parameters
        ----------
        points: :class:`numpy.ndarray`
            Array of x, y co-ordinates to adjust

        Returns
        -------
        :class:`numpy.ndarray`
            The adjusted x, y co-ordinates for display purposes
        """
        retval = (points * self._frames.current_scale) + self._canvas.offset
        logger.trace("Original points: %s, scaled points: %s", points, retval)
        return retval

    def scale_from_display(self, points):
        """ Scale and offset the given points from the current display to the correct original
        values.

        Parameters
        ----------
        points: :class:`numpy.ndarray`
            Array of x, y co-ordinates to adjust

        Returns
        -------
        :class:`numpy.ndarray`
            The adjusted x, y co-ordinates to the original frame location
        """
        retval = (points - self._canvas.offset) / self._frames.current_scale
        logger.trace("Original points: %s, scaled points: %s", points, retval)
        return retval


class BoundingBox(Editor):
    """ The Bounding Box Editor. """
    def __init__(self, canvas, alignments, frames):
        self._drag_data = dict()
        super().__init__(canvas, alignments, frames)

    @property
    def _coords_layout(self):
        """ tuple: The layout order of tkinter canvas bounding box points """
        return ("left", "top", "right", "bottom")

    @property
    def _corner_order(self):
        """ dict: The position index of bounding box corners """
        return {0: ("top", "left"),
                1: ("bottom", "left"),
                2: ("top", "right"),
                3: ("bottom", "right")}

    @property
    def _anchors(self):
        """ list: List of bounding box anchors for the corners of each face's bounding box. """
        return [[self._canvas.coords(obj) for obj in face[1:]]
                for face in self._objects]

    @property
    def _corner_points(self):
        """ list: List of bounding box tuples for each face's bounding box """
        return [((self._canvas.coords(obj[0])[0], self._canvas.coords(obj[0])[1]),
                 (self._canvas.coords(obj[0])[0], self._canvas.coords(obj[0])[3]),
                 (self._canvas.coords(obj[0])[2], self._canvas.coords(obj[0])[1]),
                 (self._canvas.coords(obj[0])[2], self._canvas.coords(obj[0])[3]))
                for obj in self._objects]

    @property
    def _bounding_boxes(self):
        """ list: List of (`Left`, `Top`, `Right`, `Bottom`) tuples for each displayed face's
        bounding box. """
        return [self._canvas.coords(face[0]) for face in self._objects]

    def _bbox_objects_for_face(self, index):
        """ Return the bounding box object with the anchor objects for the given face index.

        Parameters
        ----------
        index: int
            The face index to return the bounding box objects for

        Returns
        -------
        list
            A list of bounding box object and bounding box anchor objects. Bounding box is in
            position 0, anchors in positions 1 to 4.
        """
        retval = self._objects[index]
        logger.trace("objects: %s, index: %s, selected object: %s", self._objects, index, retval)
        return retval

    def update_annotation(self):
        """ Draw the bounding box around faces and set the object to :attr:`_object`"""
        if self._drag_data:
            logger.trace("Object being edited. Not updating annotation")
            return
        self._clear_annotation()
        color = self._colors["blue"]
        thickness = 1
        faces = []
        for face in self._alignments.current_faces:
            bbox = []
            box = np.array([(face.left, face.top), (face.right, face.bottom)])
            box = self._scale_to_display(box).astype("int32").flatten()
            corners = ((box[0], box[1]), (box[0], box[3]), (box[2], box[1]), (box[2], box[3]))
            bbox.append(self._canvas.create_rectangle(*box, outline=color, width=thickness))
            radius = thickness * 5
            for cnr in corners:
                anc = (cnr[0] - radius, cnr[1] - radius, cnr[0] + radius, cnr[1] + radius)
                bbox.append(self._canvas.create_oval(*anc,
                                                     outline=color,
                                                     fill="gray",
                                                     width=thickness,
                                                     activefill="white"))
            faces.append(bbox)
        logger.trace("Updated annotations: %s", faces)
        self._objects = faces

    # << MOUSE HANDLING >>
    # Mouse cursor display
    def set_mouse_cursor_tracking(self):
        """ Default mouse cursor tracking removes all bindings from mouse events to just
        display the standard cursor. Override for specific Editor mouse cursor display """
        logger.debug("Setting mouse bindings")
        self._canvas.bind("<Motion>", self._update_cursor)
        self._canvas.bind("<ButtonPress-1>", self._drag_start)
        self._canvas.bind("<ButtonRelease-1>", self._drag_stop)
        self._canvas.bind("<B1-Motion>", self._drag)

    def _update_cursor(self, event):
        """ Update the cursors for hovering over bounding boxes or bounding box corner anchors. """
        display_dims = self._frames.current_meta_data["display_dims"]
        if any(bbox[0] <= event.x <= bbox[2] and bbox[1] <= event.y <= bbox[3]
               for face in self._anchors for bbox in face):
            idx = [idx for face in self._anchors for idx, bbox in enumerate(face)
                   if bbox[0] <= event.x <= bbox[2] and bbox[1] <= event.y <= bbox[3]][0]
            self._canvas.config(
                cursor="{}_{}_corner".format(*self._corner_order[idx]))
        elif any(bbox[0] <= event.x <= bbox[2] and bbox[1] <= event.y <= bbox[3]
                 for bbox in self._bounding_boxes):
            self._canvas.config(cursor="fleur")
        elif (self._canvas.offset[0] <= event.x <= display_dims[0] + self._canvas.offset[0] and
              self._canvas.offset[1] <= event.y <= display_dims[1] + self._canvas.offset[1]):
            self._canvas.config(cursor="plus")
        else:
            self._canvas.config(cursor="")

    # Mouse Actions
    def _drag_start(self, event):
        """ Collect information on start of drag """
        click_object = self._get_click_object(event)
        if click_object is None:
            self._drag_data = dict()
            return
        object_type, self._drag_data["index"] = click_object
        if object_type == "anchor":
            indices = [(face_idx, pnt_idx)
                       for face_idx, face in enumerate(self._anchors)
                       for pnt_idx, bbox in enumerate(face)
                       if bbox[0] <= event.x <= bbox[2] and bbox[1] <= event.y <= bbox[3]][0]
            self._drag_data["objects"] = self._bbox_objects_for_face(indices[0])
            self._drag_data["corner"] = self._corner_order[indices[1]]
            self._drag_data["callback"] = self._resize_bounding_box
        elif object_type == "box":
            face_idx = [idx for idx, bbox in enumerate(self._bounding_boxes)
                        if bbox[0] <= event.x <= bbox[2] and bbox[1] <= event.y <= bbox[3]][0]
            self._drag_data["objects"] = self._bbox_objects_for_face(face_idx)
            self._drag_data["current_location"] = (event.x, event.y)
            self._drag_data["callback"] = self._move_bounding_box

    def _get_click_object(self, event):
        """ Return the object type and index that has been clicked on.

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The tkinter mouse event

        Returns
        -------
        tuple
            (`type`, `index`) The type of object being clicked on and the index of the face.
            If no object clicked on then return value is ``None``
        """
        retval = None
        for idx, face in enumerate(self._anchors):
            if any(bbox[0] <= event.x <= bbox[2] and bbox[1] <= event.y <= bbox[3]
                   for bbox in face):
                retval = "anchor", idx
        if retval is not None:
            return retval

        for idx, bbox in enumerate(self._bounding_boxes):
            if bbox[0] <= event.x <= bbox[2] and bbox[1] <= event.y <= bbox[3]:
                retval = "box", idx
        return retval

    def _drag_stop(self, event):  # pylint:disable=unused-argument
        """ Reset the :attr:`_drag_data` dict

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The tkinter mouse event. Unused but required
        """
        self._drag_data = dict()

    def _drag(self, event):
        """ Drag the bounding box and its anchors to current mouse position.

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The tkinter mouse event.
        """
        if not self._drag_data:
            return
        self._drag_data["callback"](event)

    def _resize_bounding_box(self, event):
        """ Resizes a bounding box on an anchor drag event

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The tkinter mouse event.
        """
        radius = 4  # TODO Variable
        rect = self._drag_data["objects"][0]
        box = list(self._canvas.coords(rect))
        # Switch top/bottom and left/right and set partial so indices match and we don't
        # need branching logic for min/max.
        limits = (partial(min, box[2] - 20),
                  partial(min, box[3] - 20),
                  partial(max, box[0] + 20),
                  partial(max, box[1] + 20))
        rect_xy_indices = [self._coords_layout.index(pnt)
                           for pnt in self._drag_data["corner"]]
        box[rect_xy_indices[1]] = limits[rect_xy_indices[1]](event.x)
        box[rect_xy_indices[0]] = limits[rect_xy_indices[0]](event.y)
        self._canvas.coords(rect, *box)
        corners = ((box[0], box[1]), (box[0], box[3]), (box[2], box[1]), (box[2], box[3]))
        for idx, cnr in enumerate(corners):
            anc = (cnr[0] - radius, cnr[1] - radius, cnr[0] + radius, cnr[1] + radius)
            self._canvas.coords(self._drag_data["objects"][idx + 1], *anc)
        self._alignments.set_current_bounding_box(self._drag_data["index"],
                                                  *self._coords_to_bounding_box(box))

    def _move_bounding_box(self, event):
        """ Moves the bounding box on a bounding box drag event """
        shift_x = event.x - self._drag_data["current_location"][0]
        shift_y = event.y - self._drag_data["current_location"][1]
        selected_objects = self._drag_data["objects"]
        for obj in selected_objects:
            self._canvas.move(obj, shift_x, shift_y)
        box = self._canvas.coords(selected_objects[0])
        self._alignments.set_current_bounding_box(self._drag_data["index"],
                                                  *self._coords_to_bounding_box(box))
        self._drag_data["current_location"] = (event.x, event.y)

    def _coords_to_bounding_box(self, coords):
        """ Converts tkinter coordinates to :class:`lib.faces_detect.DetectedFace` bounding
        box format, scaled up and offset for feeding the model.

        Returns
        -------
        tuple
            The (`x`, `width`, `y`, `height`) integer points of the bounding box.

        """
        coords = self.scale_from_display(
            np.array(coords).reshape((2, 2))).flatten().astype("int32")
        return (coords[0], coords[2] - coords[0], coords[1], coords[3] - coords[1])


class ExtractBox(Editor):
    """ The Bounding Box Editor. """

    def update_annotation(self):
        """ Draw the Extract Box around faces and set the object to :attr:`_object`"""
        self._clear_annotation()
        color = self._colors["green"]
        thickness = 1
        faces = []
        # TODO FIX THIS TEST
        #  if not all(face.original_roi for face in self._alignments.current_faces):
        #      return extract_box
        for idx, face in enumerate(self._alignments.current_faces):
            extract_box = []
            logger.trace("Drawing Extract Box: (idx: %s, roi: %s)", idx, face.original_roi)
            box = self._scale_to_display(face.original_roi).flatten()
            top_left = box[:2] - 10
            extract_box.append(self._canvas.create_text(*top_left,
                                                        fill=color,
                                                        font=("Default", 20, "bold"),
                                                        text=str(idx)))
            extract_box.append(self._canvas.create_polygon(*box,
                                                           fill="",
                                                           outline=color,
                                                           width=thickness))
            faces.append(extract_box)
        logger.trace("Updated annotations: %s", faces)
        self._objects = faces


class Landmarks(Editor):
    """ The Landmarks Editor. """

    def update_annotation(self):
        """ Draw the Landmarks and the Face Mesh set the objects to :attr:`_object`"""
        self._clear_annotation()
        landmarks = self._update_landmarks()
        mesh = self._update_mesh()
        self._objects = landmarks + mesh

    def _update_landmarks(self):
        """ Draw the facial landmarks """
        color = self._colors["red"]
        radius = 1
        faces = []
        for face in self._alignments.current_faces:
            landmarks = []
            for landmark in face.landmarks_xy:
                box = self._scale_to_display(landmark).astype("int32")
                bbox = (box[0] - radius, box[1] - radius, box[0] + radius, box[1] + radius)
                landmarks.append(self._canvas.create_oval(*bbox,
                                                          outline=color,
                                                          fill=color,
                                                          width=radius))
            faces.append(landmarks)
        logger.trace("Updated landmark annotations: %s", faces)
        return faces

    def _update_mesh(self):
        """ Draw the facial landmarks """
        color = self._colors["cyan"]
        thickness = 1
        facial_landmarks_idxs = dict(mouth=(48, 68),
                                     right_eyebrow=(17, 22),
                                     left_eyebrow=(22, 27),
                                     right_eye=(36, 42),
                                     left_eye=(42, 48),
                                     nose=(27, 36),
                                     jaw=(0, 17),
                                     chin=(8, 11))
        faces = []
        for face in self._alignments.current_faces:
            mesh = []
            landmarks = face.landmarks_xy
            logger.trace("Drawing Landmarks Mesh: (landmarks: %s, color: %s, thickness: %s)",
                         landmarks, color, thickness)
            for key, val in facial_landmarks_idxs.items():
                pts = self._scale_to_display(landmarks[val[0]:val[1]]).astype("int32").flatten()
                if key in ("right_eye", "left_eye", "mouth"):
                    mesh.append(self._canvas.create_polygon(*pts,
                                                            fill="",
                                                            outline=color,
                                                            width=thickness))
                else:
                    mesh.append(self._canvas.create_line(*pts, fill=color, width=thickness))
            faces.append(mesh)
        logger.trace("Updated mesh annotations: %s", faces)
        return faces
