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
    def __init__(self, alignments, progress_bar):
        self._alignments = alignments
        self._pbar = progress_bar
        self._size = 96
        self._selected = SelectedFrame(self._size, self._alignments)

        # Following set in self._load
        self._annotations = []
        self._tk_faces = []
        self._displays = None
        self._current_display = None

        self._faces = dict()
        self._canvas = None
        self._columns = None
        self._set_tk_trace()
        self._initialized = Event()

        self._current_frame_id = 0
        self._filter_mode = "All Faces"
        self._face_count_modified = False
        self._landmark_mapping = dict(mouth=(48, 68),
                                      right_eyebrow=(17, 22),
                                      left_eyebrow=(22, 27),
                                      right_eye=(36, 42),
                                      left_eye=(42, 48),
                                      nose=(27, 36),
                                      jaw=(0, 17),
                                      chin=(8, 11))

    @property
    def annotations(self):
        return self._annotations

    @property
    def size(self):
        return self._size

    @property
    def face_count_per_frame(self):
        # TODO Static this and update on face insertion and deletion
        return [len(faces) for faces in self._tk_faces]

    @property
    def faces(self):
        """ dict: The filename as key with list of aligned faces in :class:`ImageTk.PhotoImage`
        format for display in the GUI. """
        return self._faces

    @property
    def _frames(self):
        """ :class:`FrameNavigation`: The Frames for this manual session """
        return self._alignments.frames

    @property
    def _colors(self):
        """ dict: Colors for the annotations. """
        return dict(border="#00ff00", mesh="#00ffff", border_half="#009900", mesh_half="#009999")

    # Utility Functions
    def _coords_from_index(self, index):
        """ Return the top left co-ordinates and the rectangle dimensions that an object should be
        placed on the canvas from it's index. """
        return ((index % self._columns) * self._size, (index // self._columns) * self._size)

    def frame_index_from_object(self, object_id):
        """ Retrieve the frame index that an object belongs to from it's tag.

        Parameters
        ----------
        object_id: int
            The tkinter canvas object id

        Returns
        -------
        int
            The frame index that the object belongs to or ``None`` if the tag cannot be found
        """
        tags = [tag.replace("frame_id_", "")
                for tag in self._canvas.itemcget(object_id, "tags").split()
                if tag.startswith("frame_id_")]
        retval = int(tags[0]) if tags else None
        logger.trace("object_id: %s, frame_id: %s", object_id, retval)
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
        # TODO
        # self._frames.tk_update.trace("w", self._update_current)
        # self._frames.tk_update.trace("w", self._selected.update)

    def _switch_display(self, *args):  # pylint: disable=unused-argument
        """ Change the active display """
        nav_mode = self._frames.tk_navigation_mode.get().replace(" ", "")
        nav_mode = "AllFrames" if nav_mode == "HasFace(s)" else nav_mode
        logger.debug("Current Display: '%s', Requested Display: '%s'",
                     self._current_display, nav_mode)
        if nav_mode == self._current_display:
            return
        if self._current_display is not None:
            self._displays[self._current_display].de_initialize()
        self._current_display = nav_mode
        self._displays[self._current_display].initialize()

    def load_faces(self, canvas, frame_width):
        """ Launch a background thread to load the faces into cache and assign the canvas to
        :attr:`_canvas` """
        self._columns = frame_width // self._size
        self._canvas = canvas
        self._selected.initialize(canvas)
        self._set_displays()
        self._frames.tk_navigation_mode.trace("w", self._switch_display)
        thread = MultiThread(self._load_faces,
                             thread_count=1,
                             name="{}.load_faces".format(self.__class__.__name__))
        thread.start()

    def _set_displays(self):
        """ Set the different displayers """
        self._displays = {dsp: eval(dsp)(self)
                          for dsp in ("AllFrames", "NoFaces", "MultipleFaces")}

    def _load_faces(self):
        """ Loads the faces into the :attr:`_faces` dict at 96px size formatted for GUI display.

        Updates a GUI progress bar to show loading progress.
        """
        # TODO Make it so user can't save until faces are loaded (so alignments dict doesn't
        # change)
        # TODO Test threading the calculation of faces from the frame to try to speed things
        # up a little
        # TODO make loading faces a user selected action?
        # TODO Vid deets to alignments file
        try:
            self._pbar.start(mode="determinate")
            frame_count = self._frames.frame_count
            faces_seen = 0
            loader = ImagesLoader(self._frames.location, count=frame_count)

            for frame_idx, (filename, frame) in enumerate(loader.load()):
                frame_name = os.path.basename(filename)
                progress = int(round(((frame_idx + 1) / frame_count) * 100))
                self._pbar.progress_update("Loading Faces: {}/{} - {}%".format(frame_idx + 1,
                                                                               frame_count,
                                                                               progress), progress)
                frame_faces = []
                frame_landmarks = []
                frame_objects = []
                faces = self._alignments.saved_alignments.get(frame_name, list())

                for face_idx, face in enumerate(faces):
                    tk_face = self._load_face(frame, face)
                    tag = ["frame_id_{}".format(frame_idx)]
                    mesh_landmarks = _get_mesh_points(face.aligned_landmarks)

                    frame_faces.append(tk_face)
                    frame_landmarks.append(mesh_landmarks)
                    frame_objects.append(self._load_annotations(tk_face,
                                                                mesh_landmarks,
                                                                faces_seen,
                                                                tag))
                    faces_seen += 1
                self._tk_faces.append(frame_faces)
                self._annotations.append(dict(mesh_landmarks=frame_landmarks,
                                              objects=frame_objects))
            self._pbar.stop()
        except Exception as err:  # pylint: disable=broad-except
            logger.error("Error loading face. Error: %s", str(err))
            # TODO Remove this
            import sys; import traceback
            exc_info = sys.exc_info(); traceback.print_exception(*exc_info)
        self._switch_display()
        self._initialized.set()
        # TODO Make this handle filtering after load
        position = self._frames.tk_position.get()
        self._selected.set_selected(self._annotations[position],
                                    self._tk_faces[position], position)

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

    def _load_annotations(self, tk_face, mesh_landmarks, faces_seen, tag):
        """ Load the resized aligned face. """
        objects = dict()
        coords = self._coords_from_index(faces_seen)
        objects["image_id"] = self._canvas.create_image(*coords,
                                                        image=tk_face,
                                                        anchor=tk.NW,
                                                        tags=tag)
        # objects["mesh"] = self._create_mesh_annotations(mesh_landmarks, coords, tag)

        if coords[0] == 0:  # Resize canvas on new line
            self._canvas.configure(scrollregion=self._canvas.bbox("all"))
        return objects

    def _create_mesh_annotations(self, mesh_landmarks, offset, tag):
        """ Create the coordinates for the face mesh. """
        retval = []
        kwargs = dict(polygon=dict(fill="", outline=self._colors["mesh"]),
                      line=dict(fill=self._colors["mesh"]))
        for is_poly, landmarks in zip(mesh_landmarks["is_poly"], mesh_landmarks["landmarks"]):
            key = "polygon" if is_poly else "line"
            obj = getattr(self._canvas, "create_{}".format(key))
            obj_kwargs = kwargs[key]
            coords = (landmarks + offset).flatten()
            retval.append(obj(*coords, state="hidden", width=1, tags=tag + ["mesh"], **obj_kwargs))
        return retval

    def _on_frame_change(self, *args):  # pylint:disable=unused-argument
        """ Action to perform on a frame change """
        if not self._initialized.is_set():
            return
        position = self._frames.tk_position.get()
        self._selected.set_selected(self._annotations[position],
                                    self._tk_faces[position], position)

    def _update_current(self, *args):  # pylint:disable=unused-argument
        """ Update the currently selected face on editor update """
        if not self._alignments.tk_edited.get():
            return
        if self._add_remove_face():
            self._selected.refresh_highlighter()
            return
        self._selected.update()
        self._alignments.tk_edited.set(False)

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
        tk_face = self._selected.add_face()
        self._displays[self._current_display].add_face(tk_face, self._selected.frame_id)

    def _remove_face(self):
        """ Remove a face from the current frame """
        self._selected.remove_face()
        self._displays[self._current_display].remove_face(self._selected.frame_id)

    def update_all(self, display):
        """ Update all widgets with the for the given display type.

        Parameters
        ----------
        display: {"landmarks", "mask" or "none"}
            The annotation that should be displayed
        """
        if not self._initialized.is_set():
            return
        display = "mesh" if display == "landmarks" else display
        for key, val in self._faces.items():
            if key == self._current_frame_id:
                continue
            for face in val:
                # TODO Add masks and change from a get to standard dict retrieval
                current_object_ids = face.get(self._current_display, None)
                new_object_ids = face.get(display, None)
                self._clear_annotations(current_object_ids)
                # TODO Remove when mask is in
                if new_object_ids is None:
                    continue
                self._display_annotations(display, new_object_ids)
        self._current_display = display

    def _clear_annotations(self, object_ids):
        """ Clear the annotations of now deselected objects """
        if object_ids is not None:
            for object_id in object_ids:
                self._canvas.itemconfig(object_id, state="hidden")

    def _display_annotations(self, display, object_ids):
        """ Display the newly selected objects. """
        for object_id in object_ids:
            color_attr = "outline" if self._canvas.type(object_id) == "polygon" else "fill"
            kwargs = {color_attr: self._colors["{}_half".format(display)], "state": "normal"}
            self._canvas.itemconfig(object_id, **kwargs)


class SelectedFrame():
    """ Holds the objects and handles faces for the currently selected frame. """
    def __init__(self, image_size, alignments):
        self._alignments = alignments
        self._size = image_size

        self._canvas = None
        self._objects = None
        self._faces = None
        self._landmarks = None
        self._highlighter = None

        self._frame_id = 0
        self._face_count = 0
        self._highlight_boxes = []
        self._highlight_meshes = []

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

    def set_selected(self, annotations, tk_faces, frame_id):
        """ Set the currently selected frame's objects """
        self._faces = tk_faces
        self._objects = annotations["objects"]
        self._landmarks = annotations["mesh_landmarks"]
        self._frame_id = frame_id
        self._face_count = len(self._objects)
        self.refresh_highlighter()

    def refresh_highlighter(self):
        """ Refresh the highlighter on add/remove faces """
        self._highlighter.highlight_selected(self._objects, self._landmarks)

    def update(self):
        """ Update the currently selected face on editor update """
        tk_face, landmarks = self._get_tk_face_and_landmarks()
        self._faces[self._face_index] = tk_face
        self._canvas.itemconfig(self._objects[self._face_index]["image_id"], image=tk_face)
        self._landmarks[self._face_index] = _get_mesh_points(landmarks)

    def add_face(self):
        """ Add a face to the currently selected frame, """
        logger.debug("Adding face to frame(frame_id: %s new face_count: %s)",
                     self._frame_id, self._face_count + 1)
        tk_face, landmarks = self._get_tk_face_and_landmarks()
        self._faces.append(tk_face)
        self._landmarks.append(_get_mesh_points(landmarks))
        self._face_count += 1
        return tk_face

    def remove_face(self):
        """ Remove a face from the currently selected frame. """
        face_idx = self._alignments.get_removal_index()
        logger.trace("Removing face for frame %s at index: %s", self._frame_id, face_idx)
        self._canvas.delete(self._objects[face_idx]["image_id"])
        del self._faces[face_idx]
        del self._landmarks[face_idx]
        del self._objects[face_idx]
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

    def highlight_selected(self, objects, landmarks):
        """ Highlight the currently selected faces """
        self._face_count = len(objects)
        self._create_new_highlighters(landmarks)
        self._hide_unused_highlighters()

        boxes = self._boxes[:self._face_count]
        meshes = self._meshes[:self._face_count]
        for objs, landmarks, box, mesh in zip(objects, landmarks, boxes, meshes):
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
        """ Create new highlight mesh annotations and append to :attr:`_highlight_meshes`. """
        # TODO Global mesh colors
        kwargs = dict(polygon=dict(fill="", outline="#00ffff"),
                      line=dict(fill="#00ffff"))
        mesh_ids = []
        for is_poly, points in zip(landmarks["is_poly"], landmarks["landmarks"]):
            key = "polygon" if is_poly else "line"
            obj = getattr(self._canvas, "create_{}".format(key))
            obj_kwargs = kwargs[key]
            mesh_ids.append(obj(*points.flatten(), state="hidden", width=1, **obj_kwargs))
        logger.trace("Created new highlight_mesh: %s", mesh_ids)
        self._meshes.append(mesh_ids)

    # << Hide unused highligters >> #
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
        logger.trace("Highlighting mesh (id: %s, top_left: %s)", mesh_ids, top_left)
        for points, mesh_id in zip(landmarks, mesh_ids):
            self._canvas.coords(mesh_id, *(points + top_left).flatten())
            if un_hide:
                self._canvas.itemconfig(mesh_id, state="normal")


class FaceView():
    """ Base class for different faces view filters. """
    def __init__(self, face_cache):
        self._face_cache = face_cache
        self._canvas = face_cache._canvas

        self._display_indices = None
        self._annotations = []

        self._columns = face_cache._columns
        self._size = face_cache._size

    @property
    def _face_indices_per_frame(self):
        return list(accumulate(self._face_cache.face_count_per_frame))

    def initialize(self):
        """ Initialize the viewer for the selected filter type """
        self._display_indices = self._get_filtered_frames_list()
        display_idx = 0
        for idx, annotations in enumerate(self._face_cache.annotations):
            state = "normal" if idx in self._display_indices else "hidden"
            for landmarks, objects in zip(annotations["mesh_landmarks"], annotations["objects"]):
                if state == "hidden":
                    self._hide_displayed_face(objects)
                    continue
                self._display_face(objects,
                                   landmarks["landmarks"],
                                   self._coords_from_index(display_idx))
                display_idx += 1
            self._annotations.append(annotations)

    def _get_filtered_frames_list(self):
        """ Overide for filter criteria for specific viewer """
        raise NotImplementedError

    def _hide_displayed_face(self, objects):
        """ Hide faces that are displayed which should be hidden """
        for item_ids in objects.values():
            item_ids = item_ids if isinstance(item_ids, list) else [item_ids]
            for item_id in item_ids:
                if self._canvas.itemcget(item_id, "state") != "hidden":
                    self._canvas.itemconfig(item_id, state="hidden")

    def _display_face(self, objects, landmarks, coords):
        if self._canvas.itemcget(objects["image_id"], "state") == "hidden":
            self._canvas.itemconfig(objects["image_id"], state="normal")
        self._canvas.coords(objects["image_id"], *coords)

        # for points, object_id in zip(landmarks, objects["mesh"]):
        #     mesh_coords = (points + coords).flatten()
        #     self._canvas.coords(object_id, *mesh_coords)

    def _coords_from_index(self, index):
        """ Return the top left co-ordinates and the rectangle dimensions that an object should be
        placed on the canvas from it's index. """
        return ((index % self._columns) * self._size, (index // self._columns) * self._size)

    def add_face(self, tk_face, frame_id):
        """ Display a face in the correct location and update subsequent faces. """
        insert_index = self._face_indices_per_frame[frame_id] - 1
        coords = self._coords_from_index(insert_index)
        tag = ["frame_id_{}".format(frame_id)]
        new_face = self._canvas.create_image(*coords, image=tk_face, anchor=tk.NW, tags=tag)
        self._face_cache.annotations[frame_id]["objects"].append(dict(image_id=new_face))
        self._update_layout(frame_id + 1, insert_index + 1)

    def remove_face(self, frame_id):
        """ Update subsequent face locations for the removed face """
        starting_index = (self._face_indices_per_frame[frame_id] -
                          self._face_cache.face_count_per_frame[frame_id])
        self._update_layout(frame_id, starting_index)

    def _update_layout(self, frame_id, starting_object_index):
        """ Update the layout of faces when a face has been added or removed """
        idx = starting_object_index
        for annotation in self._annotations[frame_id:]:
            for obj in annotation["objects"]:
                coords = self._coords_from_index(idx)
                self._canvas.coords(obj["image_id"], *coords)
                idx += 1

    def de_initialize(self):
        self._display_indices = None
        self._annotations = []


class AllFrames(FaceView):
    """ The Frames that have Faces viewer """
    def _get_filtered_frames_list(self):
        """ Return the full list of objects """
        return range(len(self._face_cache.annotations))


class NoFaces(FaceView):
    """ The Frames with No Faces viewer """
    def _get_filtered_frames_list(self):
        """ Return the full list of frame indices with no faces """
        return [idx for idx, face_count in enumerate(self._face_cache.face_count_per_frame)
                if face_count == 0]


class MultipleFaces(FaceView):
    """ The Frames with Multiple Faces viewer """
    @property
    def _face_indices_per_frame(self):
        return list(accumulate(0 if face_count <= 1 else face_count
                               for face_count in self._face_cache.face_count_per_frame))

    def _get_filtered_frames_list(self):
        """ Return the full list of frame indices with no faces """
        return [idx for idx, face_count in enumerate(self._face_cache.face_count_per_frame)
                if face_count > 1]
