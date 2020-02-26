#!/usr/bin/env python3
""" Face viewer for the manual adjustments tool """
import logging
import os
import tkinter as tk

from itertools import accumulate
from threading import Event

import cv2
import numpy as np
from PIL import Image, ImageTk

from lib.image import ImagesLoader
from lib.multithreading import MultiThread

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
_LANDMARK_MAPPING = dict(mouth=(48, 68),
                         right_eyebrow=(17, 22),
                         left_eyebrow=(22, 27),
                         right_eye=(36, 42),
                         left_eye=(42, 48),
                         nose=(27, 36),
                         jaw=(0, 17),
                         chin=(8, 11))


def _get_mesh_points(landmarks):
    is_poly = []
    mesh_landmarks = []
    for key, val in _LANDMARK_MAPPING.items():
        is_poly.append(key in ("right_eye", "left_eye", "mouth"))
        mesh_landmarks.append(landmarks[val[0]:val[1]])
    return dict(is_poly=is_poly, landmarks=mesh_landmarks)


class FaceCache():
    """ Holds the face images for display in the bottom GUI Panel """
    def __init__(self, alignments, progress_bar, scaling_factor):
        self._alignments = alignments
        self._pbar = progress_bar
        self._size = int(round(96 * scaling_factor))
        self._selected = SelectedFrame(self._size, self._alignments)

        # Following set in self._load
        self._canvas = None
        self._annotations = dict(tk_faces=[], tk_objects=[], mesh_landmarks=[])
        self._face_count_per_frame = []
        self._filters = dict(displays=dict(), current_display=None)
        self._annotation_colors = dict(mesh=None)

        self._set_tk_trace()
        self._initialized = Event()

    @property
    def size(self):
        """ int: The size of the thumbnail """
        return self._size

    @property
    def is_initialized(self):
        """ bool: ``True`` if the faces have completed the loading cycle otherwise ``False`` """
        return self._initialized.is_set()

    @property
    def _tk_objects(self):
        """ list: list of frame count length containing list of `dict` objects containing each
        face's canvas object ids.

        The index of each item in the list corresponds to the frame index.
        Each item contains a list of dictionaries, one dictionary for every face that appears
        in the frame.
        The dictionary holds all the canvas annotation objects that can appear for each face.
        """
        return self._annotations["tk_objects"]

    @property
    def _mesh_landmarks(self):
        """ list: list of frame count length containing lists of :class:`numpy.ndarray` split
        up into groups for creating mesh annotations for each face in the frame. """
        return self._annotations["mesh_landmarks"]

    @property
    def _tk_faces(self):
        """ list: list of frame count length containing lists of :class:`PIL.ImageTk.PhotoImage`
            corresponding to each face in a frame. """
        return self._annotations["tk_faces"]

    @property
    def _frames(self):
        """ :class:`FrameNavigation`: The Frames for this manual session """
        return self._alignments.frames

    @property
    def _filtered_display(self):
        """:class:`FaceFilter`: The currently selected filtered faces display. """
        return self._filters["displays"][self._filters["current_display"]]

    @property
    def _colors(self):
        """ dict: Colors for the annotations. """
        return dict(border="#00ff00", mesh="#00ffff", mesh_half="#009999")

    def frame_index_from_object(self, item_id):
        """ Retrieve the frame index that an object belongs to from it's tag.

        Parameters
        ----------
        item_id: int
            The tkinter canvas object id

        Returns
        -------
        int
            The frame index that the object belongs to or ``None`` if the tag cannot be found
        """
        tags = [tag.replace("frame_id_", "")
                for tag in self._canvas.itemcget(item_id, "tags").split()
                if tag.startswith("frame_id_")]
        retval = int(tags[0]) if tags else None
        logger.trace("item_id: %s, frame_id: %s", item_id, retval)
        return retval

    def transport_index_from_frame_index(self, frame_index):
        """ Retrieve the index in the filtered frame list for the given frame index. """
        frames_list = self._alignments.get_filtered_frames_list()
        retval = frames_list.index(frame_index) if frame_index in frames_list else None
        logger.trace("frame_index: %s, transport_index: %s", frame_index, retval)
        return retval

    def _set_tk_trace(self):
        """ Set the trace on tkinter variables:
        self._frames.current_position
        """
        self._alignments.tk_edited.trace("w", self._update_current)
        self._frames.tk_position.trace("w", self._on_frame_change)

    def _switch_filter(self, *args):  # pylint: disable=unused-argument
        """ Change the active display """
        if not self._initialized.is_set():
            return

        nav_mode = self._frames.tk_navigation_mode.get().replace(" ", "")
        nav_mode = "AllFrames" if nav_mode == "HasFace(s)" else nav_mode
        current_display = self._filters["current_display"]
        logger.debug("Current Display: '%s', Requested Display: '%s'", current_display, nav_mode)
        if nav_mode == current_display:
            return
        if current_display is not None:
            self._filtered_display.de_initialize()
        self._filters["current_display"] = nav_mode
        self._filtered_display.initialize(self._tk_objects, self._mesh_landmarks)

    def load_faces(self, canvas, enable_buttons_callback):
        """ Launch a background thread to load the faces into cache and assign the canvas to
        :attr:`_canvas` """
        self._canvas = canvas
        self._annotation_colors = dict(mesh=self._canvas.get_muted_color("Mesh"))
        self._selected.initialize(canvas)
        self._set_displays()
        self._frames.tk_navigation_mode.trace("w", self._switch_filter)

        thread = MultiThread(self._load_faces,
                             enable_buttons_callback,
                             thread_count=1,
                             name="{}.load_faces".format(self.__class__.__name__))
        thread.start()

    def _set_displays(self):
        """ Set the different face viewers """
        self._filters["displays"] = {dsp: eval(dsp)(self)  # pylint:disable=eval-used
                                     for dsp in ("AllFrames", "NoFaces", "MultipleFaces")}

    def _load_faces(self, enable_buttons_callback):
        """ Loads the faces into the :attr:`_faces` dict at 96px size formatted for GUI display.

        Updates a GUI progress bar to show loading progress.
        """
        # TODO Make it so user can't save until faces are loaded (so alignments dict doesn't
        # change)
        # TODO make loading faces a user selected action?
        # TODO Vid deets to alignments file
        try:
            self._pbar.start(mode="determinate")
            faces_seen = 0
            loader = ImagesLoader(self._frames.location, count=self._frames.frame_count)

            for frame_idx, (filename, frame) in enumerate(loader.load()):
                frame_name = os.path.basename(filename)
                progress = int(round(((frame_idx + 1) / self._frames.frame_count) * 100))
                self._pbar.progress_update(
                    "Loading Faces: {}/{} - {}%".format(frame_idx + 1,
                                                        self._frames.frame_count,
                                                        progress), progress)
                frame_items = dict(faces=[], lmarks=[], objects=[])
                for face in self._alignments.saved_alignments.get(frame_name, list()):
                    frame_items["faces"].append(self._load_face(frame, face))
                    frame_items["lmarks"].append(_get_mesh_points(face.aligned_landmarks))
                    frame_items["objects"].append(self._load_annotations(frame_items["faces"][-1],
                                                                         frame_items["lmarks"][-1],
                                                                         faces_seen,
                                                                         frame_idx))
                    faces_seen += 1
                self._tk_faces.append(frame_items["faces"])
                self._mesh_landmarks.append(frame_items["lmarks"])
                self._tk_objects.append(frame_items["objects"])
            self._pbar.stop()
        except Exception as err:  # pylint: disable=broad-except
            logger.error("Error loading face. Error: %s", str(err))
            # TODO Remove this
            import sys; import traceback
            exc_info = sys.exc_info(); traceback.print_exception(*exc_info)
        self._face_count_per_frame.extend([len(faces) for faces in self._tk_faces])
        self._canvas.tk_control_colors["Mesh"].trace("w", self._update_mesh_color)
        self._initialized.set()
        self._switch_filter()
        self._set_selected()
        enable_buttons_callback()

    def _load_face(self, frame, face):
        """ Load the resized aligned face. """
        face.load_aligned(frame, size=self._size, force=True)
        aligned_face = face.aligned_face[..., 2::-1]
        if aligned_face.shape[0] != self._size:
            aligned_face = cv2.resize(aligned_face,
                                      (self._size, self._size),
                                      interpolation=cv2.INTER_AREA)
        face.aligned["face"] = None
        return ImageTk.PhotoImage(Image.fromarray(aligned_face))

    def _load_annotations(self, tk_face, mesh_landmarks, faces_seen, frame_idx):
        """ Load the resized aligned face. """
        objects = dict()
        coords = self._canvas.coords_from_index(faces_seen)
        tag = ["frame_id_{}".format(frame_idx)]
        objects["image_id"] = self._canvas.create_image(*coords,
                                                        image=tk_face,
                                                        anchor=tk.NW,
                                                        tags=tag)
        objects["mesh"] = self._canvas.create_mesh_annotations(self._annotation_colors["mesh"],
                                                               mesh_landmarks,
                                                               coords,
                                                               tag)

        if coords[0] == 0:  # Resize canvas on new line
            self._canvas.configure(scrollregion=self._canvas.bbox("all"))
        return objects

    def _on_frame_change(self, *args):  # pylint:disable=unused-argument
        """ Action to perform on a frame change """
        if not self._initialized.is_set():
            return
        self._set_selected()

    def _set_selected(self):
        """ Set the currently selected annotations. """
        position = self._frames.tk_position.get()
        self._selected.set_selected(self._tk_objects[position],
                                    self._tk_faces[position],
                                    self._mesh_landmarks[position],
                                    position)

    def _update_current(self, *args):  # pylint:disable=unused-argument
        """ Update the currently selected face on editor update """
        if not self._alignments.tk_edited.get():
            return
        if self._add_remove_face():
            self._selected.refresh_highlighter()
            return
        self._selected.update()
        self._alignments.tk_edited.set(False)

    def _update_mesh_color(self, *args):  # pylint:disable=unused-argument
        """ Update the mesh color on control panel change """
        if not self._initialized.is_set():
            return
        color = self._canvas.get_muted_color("Mesh")
        if self._annotation_colors["mesh"] == color:
            return
        self._selected.update_highlighter_color("mesh")
        for frame in self._tk_objects:
            for objects in frame:
                self._canvas.update_object_colors(objects["mesh"], color)
        self._annotation_colors["mesh"] = color

    def _add_remove_face(self):
        """ add or remove a face for the current frame """
        alignment_faces = len(self._alignments.current_faces)
        retval = False
        if alignment_faces > self._selected.face_count:
            self._add_face()
            retval = True
        elif alignment_faces < self._selected.face_count:
            self._remove_face()
            retval = True
        return retval

    def _add_face(self):
        """ Insert a face into current frame """
        logger.debug("Adding face")
        tk_face = self._selected.add_face()
        self._face_count_per_frame[self._frames.tk_position.get()] += 1
        self._filtered_display.add_face(tk_face, self._selected.frame_id)

    def _remove_face(self):
        """ Remove a face from the current frame """
        logger.debug("Removing face")
        self._selected.remove_face()
        self._face_count_per_frame[self._frames.tk_position.get()] -= 1
        self._filtered_display.remove_face(self._selected.frame_id)

    def toggle_annotations(self):
        """ Toggle additional annotations on or off. """
        if not self._initialized.is_set():
            return
        self._filtered_display.toggle_annotation()


class SelectedFrame():
    """ Holds the objects and handles faces for the currently selected frame. """
    def __init__(self, image_size, alignments):
        self._alignments = alignments
        self._size = image_size

        self._canvas = None
        self._tk_objects = None
        self._tk_faces = None
        self._mesh_landmarks = None
        self._highlighter = None

        self._frame_id = 0
        self._face_count = 0

    @property
    def _face_index(self):
        """ int: The currently selected face index """
        return self._alignments.face_index

    @property
    def face_count(self):
        """ int: The count of faces in the currently selected frame. """
        return self._face_count

    @property
    def frame_id(self):
        """ int: The current frame index. """
        return self._frame_id

    def initialize(self, canvas):
        """ Set the canvas object to :attr:`_canvas`. """
        self._canvas = canvas
        self._highlighter = Highlighter(self._size, canvas)

    def set_selected(self, tk_objects, tk_faces, mesh_landmarks, frame_id):
        """ Set the currently selected frame's objects """
        self._tk_faces = tk_faces
        self._tk_objects = tk_objects
        self._mesh_landmarks = mesh_landmarks
        self._frame_id = frame_id
        self._face_count = len(self._tk_objects)
        self.refresh_highlighter()

    def refresh_highlighter(self):
        """ Refresh the highlighter on add/remove faces """
        self._highlighter.highlight_selected(self._tk_objects, self._mesh_landmarks)

    def update_highlighter_color(self, annotation_key):
        """ Update the highlighter annotation color on a control panel update """
        getattr(self._highlighter, "update_{}_color".format(annotation_key))()

    def update(self):
        """ Update the currently selected face on editor update """
        tk_face, landmarks = self._get_tk_face_and_landmarks()
        self._update_face(tk_face)
        self._update_mesh(landmarks)
        self.refresh_highlighter()

    def _update_face(self, tk_face):
        """ Update the face photo image and the face object id """
        self._tk_faces[self._face_index] = tk_face
        self._canvas.itemconfig(self._tk_objects[self._face_index]["image_id"], image=tk_face)

    def _update_mesh(self, landmarks):
        """ Update the optional mesh annotation """
        mesh_landmarks = _get_mesh_points(landmarks)
        coords = self._canvas.coords(self._tk_objects[self._face_index]["image_id"])
        for points, item_id in zip(mesh_landmarks["landmarks"],
                                   self._tk_objects[self._face_index]["mesh"]):
            self._canvas.coords(item_id, *(points + coords).flatten())
        self._mesh_landmarks[self._face_index] = mesh_landmarks

    def add_face(self):
        """ Add a face to the currently selected frame, """
        logger.debug("Adding face to frame(frame_id: %s new face_count: %s)",
                     self._frame_id, self._face_count + 1)
        tk_face, landmarks = self._get_tk_face_and_landmarks()
        self._tk_faces.append(tk_face)
        self._mesh_landmarks.append(_get_mesh_points(landmarks))
        self._face_count += 1
        return tk_face

    def remove_face(self):
        """ Remove a face from the currently selected frame. """
        face_idx = self._alignments.get_removal_index()
        logger.trace("Removing face for frame %s at index: %s", self._frame_id, face_idx)
        self._canvas.delete(self._tk_objects[face_idx]["image_id"])
        del self._tk_faces[face_idx]
        del self._mesh_landmarks[face_idx]
        del self._tk_objects[face_idx]
        self._face_count -= 1

    def _get_tk_face_and_landmarks(self):
        """ Obtain the resized photo image face and scaled landmarks """
        detected_face = self._alignments.current_face
        if detected_face.aligned_face is None:
            # When in zoomed in mode the face isn't loaded, so get a copy
            aligned_face = self._alignments.get_aligned_face_at_index(self._alignments.face_index)
        else:
            aligned_face = detected_face.aligned_face

        display_face = cv2.resize(aligned_face[..., 2::-1],
                                  (self._size, self._size),
                                  interpolation=cv2.INTER_AREA)
        tk_face = ImageTk.PhotoImage(Image.fromarray(display_face))
        scale = self._size / detected_face.aligned["size"]
        scaled_landmarks = detected_face.aligned_landmarks * scale
        return tk_face, scaled_landmarks


class Highlighter():
    """ Highlights the currently active frame's faces """
    def __init__(self, image_size, canvas):
        logger.debug("Initializing: %s: (image_size: %s, canvas: %s)",
                     self.__class__.__name__, image_size, canvas)
        self._size = image_size
        self._canvas = canvas

        self._face_count = 0
        self._hidden_highlighters_count = 0
        self._boxes = []
        self._meshes = []
        logger.debug("Initialized: %s", self.__class__.__name__,)

    @property
    def _highlighter_count(self):
        """ int: The number of highlighter objects currently available. """
        return len(self._boxes)

    def highlight_selected(self, objects, mesh_landmarks):
        """ Highlight the currently selected faces """
        self._face_count = len(objects)
        self._create_new_highlighters(mesh_landmarks)
        self._hide_unused_highlighters()
        if self._face_count == 0:
            return

        boxes = self._boxes[:self._face_count]
        meshes = self._meshes[:self._face_count]
        for objs, landmarks, box, mesh in zip(objects, mesh_landmarks, boxes, meshes):
            top_left = np.array(self._canvas.coords(objs["image_id"]))
            un_hide = self._highlight_box(box, top_left)
            self._highlight_mesh(mesh, landmarks["landmarks"], top_left, un_hide)
            self._hidden_highlighters_count -= 1 if un_hide else 0

        top = self._canvas.coords(self._boxes[0])[1] / self._canvas.bbox("all")[3]
        if top != self._canvas.yview()[0]:
            self._canvas.yview_moveto(top)

    # << Add new highlighters >> #
    def _create_new_highlighters(self, landmarks):
        """ Add new highlight annotations if there are more faces in the frame than
        current highlighters. """
        new_highlighter_count = max(0, self._face_count - self._highlighter_count)
        logger.trace("new_highlighter_count: %s", new_highlighter_count)
        if new_highlighter_count == 0:
            return
        for idx in range(new_highlighter_count):
            self._create_highlight_box()
            self._create_highlight_mesh(landmarks[idx])
            self._hidden_highlighters_count += 1

    def _create_highlight_box(self):
        """ Create a new highlight box and append to :attr:`_boxes`. """
        box = self._canvas.create_rectangle(0, 0, 1, 1, outline="#00FF00", width=2, state="hidden")
        logger.trace("Created new highlight_box: %s", box)
        self._boxes.append(box)

    def _create_highlight_mesh(self, landmarks):
        """ Create new highlight mesh annotations and append to :attr:`_meshes`. """
        mesh_color = self._canvas.control_colors["Mesh"]
        kwargs = dict(polygon=dict(fill="", outline=mesh_color),
                      line=dict(fill=mesh_color))
        mesh_ids = []
        for is_poly, points in zip(landmarks["is_poly"], landmarks["landmarks"]):
            key = "polygon" if is_poly else "line"
            obj = getattr(self._canvas, "create_{}".format(key))
            obj_kwargs = kwargs[key]
            mesh_ids.append(obj(*points.flatten(), state="hidden", width=1, **obj_kwargs))
        logger.trace("Created new highlight_mesh: %s", mesh_ids)
        self._meshes.append(mesh_ids)

    # << Hide unused highlighters >> #
    def _hide_unused_highlighters(self):
        """ Hide any highlighters that are not required for the current frame """
        hide_count = self._highlighter_count - self._face_count - self._hidden_highlighters_count
        hide_count = max(0, hide_count)
        logger.trace("hide_highlighter_count: %s", hide_count)
        if hide_count == 0:
            return
        hide_slice = slice(self._face_count, self._face_count + hide_count)
        for box, mesh in zip(self._boxes[hide_slice], self._meshes[hide_slice]):
            logger.trace("Hiding highlight box: %s", box)
            self._canvas.itemconfig(box, state="hidden")
            logger.trace("Hiding highlight mesh: %s", mesh)
            for mesh_id in mesh:
                self._canvas.itemconfig(mesh_id, state="hidden")
            self._hidden_highlighters_count += 1

    # << Highlight current faces >> #

    def _highlight_box(self, box, top_left):
        """ Locate and display the given highlight box """
        un_hide = False
        coords = (*top_left, *top_left + self._size)
        logger.trace("Highlighting box (id: %s, coords: %s)", box, coords)
        self._canvas.coords(box, *coords)
        if self._canvas.itemcget(box, "state") == "hidden":
            un_hide = True
            self._canvas.itemconfig(box, state="normal")
        return un_hide

    def _highlight_mesh(self, mesh_ids, landmarks, top_left, un_hide):
        """ Locate and display the given mesh annotations """
        logger.trace("Highlighting mesh (id: %s, top_left: %s, unhide: %s)",
                     mesh_ids, top_left, un_hide)
        for points, mesh_id in zip(landmarks, mesh_ids):
            self._canvas.coords(mesh_id, *(points + top_left).flatten())
            if un_hide:
                self._canvas.itemconfig(mesh_id, state="normal")

    def update_mesh_color(self):
        """ Update the highlighted mesh color on control panel update. """
        color = self._canvas.control_colors["Mesh"]
        for item_ids in self._meshes:
            self._canvas.update_object_colors(item_ids, color)


class FaceFilter():
    """ Base class for different faces view filters.

    All Filter views inherit from this class. Handles the layout of faces and annotations in the
    face viewer.

    Parameters
    ----------
    face_cache: :class:`FaceCache`
        The main Face Viewers face cache, that holds all of the display items and annotations
    """
    def __init__(self, face_cache):
        logger.debug("Initializing: %s: (face_cache: %s)", self.__class__.__name__, face_cache)
        self._canvas = face_cache._canvas
        self._tk_position = face_cache._frames.tk_position
        self._tk_objects = face_cache._tk_objects
        self._face_count_per_frame = face_cache._face_count_per_frame
        self._mesh_landmarks = face_cache._mesh_landmarks

        # Set and unset during :func:`initialize` and :func:`de-initialize`
        self._display_indices = []
        self._current_position = -1
        self._tk_position_callback = None
        logger.debug("Initialized: %s", self.__class__.__name__)

    @property
    def _face_indices_per_frame(self):
        """ list: The absolute display indices for each face that should be displayed for the
        current filter. The list is of length total frames with each item in the list containing an
        `int` indicating the number of faces that are to be displayed up to and including the
        current frame.

        Override for filter specific list.
        """
        raise NotImplementedError

    def _set_display_indices(self):
        """ Set the the filtered list of frame indices to :attr:`_display_indices` for the
        current filter.

        Override for criteria specific to each filter.
        """
        raise NotImplementedError

    def _on_frame_change(self, *args):  # pylint:disable=unused-argument
        """ Callback to be executed whenever the frame is changed.

        Override for filter specific actions
        """
        raise NotImplementedError

    def initialize(self, tk_objects, mesh_landmarks):
        """ Initialize the viewer for the selected filter type.

        Hides annotations and faces that should not be displayed for the current filter.
        Displays and moves the faces to the correct position on the canvas based on which faces
        should be displayed.

        Parameters
        ----------
        tk_objects: list
            The list of objects that exist for every frame in the source.
        mesh_landmarks: list
            The list of landmarks, split up into groups for creating mesh annotations for every
            frame in the source
        """
        self._set_display_indices()
        display_idx = 0
        for idx, (frame_landmarks, frame_objects) in enumerate(zip(mesh_landmarks, tk_objects)):
            state = "normal" if idx in self._display_indices else "hidden"
            for landmarks, objects in zip(frame_landmarks, frame_objects):
                if state == "hidden":
                    self._hide_displayed_face(objects)
                    continue
                self._display_face(objects,
                                   landmarks["landmarks"],
                                   self._canvas.coords_from_index(display_idx))
                display_idx += 1
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))
        self._tk_position_callback = self._tk_position.trace("w", self._on_frame_change)
        self._current_position = self._display_indices[0] if self._display_indices else -1

    def _hide_displayed_face(self, objects):
        """ Hide faces and annotations that are displayed which should be hidden.

        Parameters
        ----------
        objects: dict
            The object ids that are to be hidden
        """
        for item_ids in objects.values():
            item_ids = item_ids if isinstance(item_ids, list) else [item_ids]
            for item_id in item_ids:
                if self._canvas.itemcget(item_id, "state") != "hidden":
                    self._canvas.itemconfig(item_id, state="hidden")

    def _display_face(self, objects, landmarks, coordinates):
        """ Display faces and annotations that should be shown and locates the objects correctly
        on the canvas.

        Parameters
        ----------
        objects: dict
            The object ids that are to be displayed
        landmarks: :class:`numpy.ndarray`
            The base landmark mesh points corresponding to point (0, 0)
        coordinates: tuple
            The top left corner location of the face to be displayed
        """
        if self._canvas.itemcget(objects["image_id"], "state") == "hidden":
            self._canvas.itemconfig(objects["image_id"], state="normal")
        self._canvas.coords(objects["image_id"], *coordinates)

        annotation = self._canvas.optional_annotation
        for points, item_id in zip(landmarks, objects["mesh"]):
            mesh_coords = (points + coordinates).flatten()
            self._canvas.coords(item_id, *mesh_coords)
            if annotation == "landmarks" and self._canvas.itemcget(item_id, "state") == "hidden":
                self._canvas.itemconfig(item_id, state="normal")

    def add_face(self, tk_face, frame_id):
        """ Display a new face in the correct location and move subsequent faces to their new
        location.

        Parameters
        ----------
        tk_face: :class:`PIL.ImageTk.PhotoImage`
            The new face that is to be added to the canvas
        frame_id: int
            The frame index that the face is to be added to
        """
        insert_index = max(0, self._face_indices_per_frame[frame_id] - 1)
        coords = self._canvas.coords_from_index(insert_index)
        tag = ["frame_id_{}".format(frame_id)]
        new_face = self._canvas.create_image(*coords, image=tk_face, anchor=tk.NW, tags=tag)
        mesh = self._canvas.create_mesh_annotations(self._canvas.get_muted_color("Mesh"),
                                                    self._mesh_landmarks[frame_id][-1],
                                                    coords,
                                                    tag)
        logger.debug("insert_index: %s, coords: %s, tag: %s, new_face: %s, mesh: %s",
                     insert_index, coords, tag, new_face, mesh)
        self._canvas.tag_lower(tag)
        self._tk_objects[frame_id].append(dict(image_id=new_face, mesh=mesh))
        self._update_layout(frame_id + 1, insert_index + 1)

    def remove_face(self, frame_id):
        """ Relocate all faces after the removed face to the new location.

        Parameters
        ----------
        frame_id: int
            The frame index that the face has been removed from
        """
        starting_index = max(0, (self._face_indices_per_frame[frame_id] -
                                 self._face_count_per_frame[frame_id]))
        self._update_layout(frame_id, starting_index)

    def _update_layout(self, starting_frame_id, starting_object_index):
        """ Reposition faces and annotations on the canvas after a face has been added or removed.

        Parameters
        ----------
        starting_frame_id: int
            The starting frame index that new locations should be calculated for
        starting_object_index: int
            The starting absolute face index that new locations should be calculated for
        """
        idx = starting_object_index
        logger.debug("starting_frame_id: %s, startng_object_index: %s",
                     starting_frame_id, starting_object_index)
        for frame_id in self._display_indices[self._display_indices.index(starting_frame_id):]:
            for obj, landmarks in zip(self._tk_objects[frame_id], self._mesh_landmarks[frame_id]):
                coords = self._canvas.coords_from_index(idx)
                self._canvas.coords(obj["image_id"], *coords)
                for points, mesh_id in zip(landmarks["landmarks"], obj["mesh"]):
                    self._canvas.coords(mesh_id, *(points + coords).flatten())
                idx += 1
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def toggle_annotation(self):
        """ Toggle additional object annotations on or off.
        """
        display = self._canvas.optional_annotation
        display = "mesh" if display == "landmarks" else display
        state = "hidden" if display is None else "normal"
        for frame_idx in self._display_indices:
            for face in self._tk_objects[frame_idx]:
                for item_id in self._get_toggle_item_ids(face, display):
                    self._canvas.itemconfig(item_id, state=state)

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

    def de_initialize(self):
        """ Unloads the Face Filter on filter type change. """
        self._display_indices = []
        self._tk_position.trace_vdelete("w", self._tk_position_callback)
        self._tk_position_callback = None
        self._current_position = -1


class AllFrames(FaceFilter):
    """ The Frames that have Faces viewer

    Parameters
    ----------
    face_cache: :class:`FaceCache`
        The main Face Viewers face cache, that holds all of the display items and annotations
    """
    @property
    def _face_indices_per_frame(self):
        """ list: The absolute display indices for each face that should be displayed for the
        current filter. The list is of length total frames with each item in the list containing an
        `int` indicating the number of faces that are to be displayed up to and including the
        current frame.

        For All Frames this is the absolute index for every face in all frames.
        """
        return list(accumulate(self._face_count_per_frame))

    def _set_display_indices(self):
        """ Set the the filtered list frame indices to :attr:`_display_indices` for the
        current filter.

        For All Frames this is every frame index that exists, regardless of face count.
        """
        self._display_indices = range(len(self._face_count_per_frame))

    def _on_frame_change(self, *args):  # pylint:disable=unused-argument
        """ Callback to be executed whenever the frame is changed.

        All Frames has no frame change specific actions.
        """
        return


class NoFaces(FaceFilter):
    """ The Frames with No Faces viewer.

    Extends the base filter to track when faces have been added to a frame, so that the display
    can be updated accordingly.

    Parameters
    ----------
    face_cache: :class:`FaceCache`
        The main Face Viewers face cache, that holds all of the display items and annotations
    """
    def __init__(self, face_cache):
        self._added_objects = []
        super().__init__(face_cache)

    @property
    def _face_indices_per_frame(self):
        """ list: The absolute display indices for each face that should be displayed for the
        current filter. The list is of length total frames with each item in the list containing an
        `int` indicating the number of faces that are to be displayed up to and including the
        current frame.

        For No Faces this always zero for all frames.
        """
        return [0 for _ in range(len(self._face_count_per_frame))]

    def _set_display_indices(self):
        """ Set the the filtered list of frame indices to :attr:`_display_indices` for the
        current filter.

        For No Faces this is every frame index for each frame that has no face.
        """
        self._display_indices = [
            idx for idx, face_count in enumerate(self._face_count_per_frame)
            if face_count == 0]

    def _on_frame_change(self, *args):  # pylint:disable=unused-argument
        """ Callback to be executed whenever the frame is changed.

        If faces have been added to the current frame, hide the new frame and remove
        the frame id from :attr:`_object_indices`.
        """
        if not self._added_objects:
            return
        for item_id in self._added_objects:
            logger.debug("Hiding newly created face: %s", item_id)
            self._canvas.itemconfig(item_id, state="hidden")
        self._added_objects = []
        position = self._current_position
        if position != -1 and self._face_count_per_frame[position] != 0:
            logger.debug("Removing display frame index: %s", position)
            self._display_indices.remove(position)
        self._current_position = self._tk_position.get()

    def add_face(self, tk_face, frame_id):
        """ Display a new face in the correct location and move subsequent faces to their new
        location.

        Additionally adds the added object ids to :attr:`_added_objects` to be hidden on
        a frame change.

        Parameters
        ----------
        tk_face: :class:`PIL.ImageTk.PhotoImage`
            The new face that is to be added to the canvas
        frame_id: int
            The frame index that the face is to be added to

        """
        super().add_face(tk_face, frame_id)
        self._added_objects.extend(self._tk_objects[frame_id][-1].values())


class MultipleFaces(FaceFilter):
    """ The Frames with Multiple Faces viewer.

    Parameters
    ----------
    face_cache: :class:`FaceCache`
        The main Face Viewers face cache, that holds all of the display items and annotations
    """
    @property
    def _face_indices_per_frame(self):
        """ list: The absolute display indices for each face that should be displayed for the
        current filter. The list is of length total frames with each item in the list containing an
        `int` indicating the number of faces that are to be displayed up to and including the
        current frame.

        For Multiple Faces indices are only incremented if 2 or more faces appear in a frame.
        The value for the current frame's faces is incremented the actual number of faces appearing
        regardless of threshold in case of deletions taking the value below the threshold.
        """
        return list(accumulate(0 if num_faces <= 1 and idx != self._current_position else num_faces
                               for idx, num_faces in enumerate(self._face_count_per_frame)))

    def _set_display_indices(self):
        """ Set the the filtered list of frame indices to :attr:`_display_indices` for the
        current filter.

        For Multiple Faces this is every frame index for each frame that contains 2 or more faces.
        """
        self._display_indices = [
            idx for idx, face_count in enumerate(self._face_count_per_frame)
            if face_count > 1]

    def _on_frame_change(self, *args):  # pylint:disable=unused-argument
        """ Callback to be executed whenever the frame is changed.

        If the face count for the current frame is no longer 2 or more, remove the frame
        from :attr:`_object_indices`, hide any existing sole faces and update the layout.
        """
        position = self._current_position
        displayed_count = self._face_count_per_frame[position]
        if position != -1 and displayed_count < 2:
            logger.debug("Removing display frame index: %s", position)
            self._display_indices.remove(position)
            if displayed_count == 1:
                self._hide_leftover_face(position)
            self.remove_face(self._tk_position.get())
        self._current_position = self._tk_position.get()

    def _hide_leftover_face(self, position):
        """ If one face remains in a frame, hide it from view on a frame change. """
        logger.debug("Hiding remaining face for frame index: %s", position)
        for item_ids in self._tk_objects[position][0].values():
            for item_id in item_ids if isinstance(item_ids, list) else [item_ids]:
                self._canvas.itemconfig(item_id, state="hidden")
