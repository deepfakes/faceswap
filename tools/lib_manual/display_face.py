#!/usr/bin/env python3
""" Face viewer for the manual adjustments tool """
import logging
import os
import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk

from lib.image import ImagesLoader
from lib.multithreading import MultiThread

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class FaceCache():
    """ Holds the face images for display in the bottom GUI Panel """
    def __init__(self, alignments, frames, progress_bar, scaling_factor):
        self._alignments = alignments
        self._frames = frames
        self._pbar = progress_bar
        self._size = int(round(96 * scaling_factor))
        self._selected = SelectedFrame(self._size, self._alignments)

        # Following set in self._load
        self._canvas = None
        self._tk_faces = []
        self._mesh_landmarks = []
        self._filters = dict(displays=dict(), current_display=None)
        self._annotation_colors = dict(mesh=None)

        self._tk_load_complete = tk.BooleanVar()
        self._set_tk_trace()
        self._initialized = False

    @property
    def size(self):
        """ int: The size of the thumbnail """
        return self._size

    @property
    def is_initialized(self):
        """ bool: ``True`` if the faces have completed the loading cycle otherwise ``False`` """
        return self._initialized

    @property
    def _filtered_display(self):
        """:class:`FaceFilter`: The currently selected filtered faces display. """
        return self._filters["displays"][self._filters["current_display"]]

    @property
    def _colors(self):
        """ dict: Colors for the annotations. """
        return dict(border="#00ff00", mesh="#00ffff", mesh_half="#009999")

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
        self._tk_load_complete.set(False)
        self._tk_load_complete.trace("w", self._on_load_complete)

    def _switch_filter(self, *args, init=False):  # pylint: disable=unused-argument
        """ Change the active display """
        if not init and not self._initialized:
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
        self._filtered_display.initialize()

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

    # TODO moving to new frame and adding faces seems to mess up the tk_face of the existing face
    def _load_faces(self, enable_buttons_callback):
        """ Loads the faces into the :attr:`_faces` dict at 96px size formatted for GUI display.

        Updates a GUI progress bar to show loading progress.
        """
        # TODO Make it so user can't save until faces are loaded (so alignments dict doesn't
        # change)
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
                self._tk_faces.append([])
                self._mesh_landmarks.append([])
                faces = self._alignments.saved_alignments.get(frame_name, list())
                is_multi = len(faces) > 1
                for face in faces:
                    self._tk_faces[-1].append(self._load_face(frame, face))
                    mesh_landmarks = self._canvas.get_mesh_points(face.aligned_landmarks)
                    self._mesh_landmarks[-1].append(mesh_landmarks)
                    coords = self._canvas.coords_from_index(faces_seen)
                    self._canvas.create_viewer_annotations(coords,
                                                           self._tk_faces[-1][-1],
                                                           self._mesh_landmarks[-1][-1],
                                                           frame_idx,
                                                           is_multi=is_multi)
                    if coords[0] == 0:  # Resize canvas on new line
                        self._canvas.configure(scrollregion=self._canvas.bbox("all"))
                    faces_seen += 1
            self._pbar.stop()
        except Exception as err:  # pylint: disable=broad-except
            logger.error("Error loading face. Error: %s", str(err))
            # TODO Remove this
            import sys; import traceback
            exc_info = sys.exc_info(); traceback.print_exception(*exc_info)
        self._canvas.tk_control_colors["Mesh"].trace("w", self._update_mesh_color)
        enable_buttons_callback()
        self._tk_load_complete.set(True)

    # TODO On load completion update the faces/frames for any edits that have been made

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

    def _on_load_complete(self, *args):  # pylint:disable=unused-argument
        """ Actions to perform in the main thread once the faces have finished loading.
        """
        # TODO Enable saving
        self._filters["current_display"] = "AllFrames"
        self._filtered_display.initialize()
        for frame_idx, faces in enumerate(self._alignments.updated_alignments):
            if faces is None:
                continue
            image_ids = self._canvas.find_withtag("image_{}".format(frame_idx))
            existing_count = len(image_ids)
            new_count = len(faces)
            self._on_load_remove_faces(existing_count, new_count, frame_idx)

            for face_idx, face in enumerate(faces):
                tk_face, mesh_landmarks = self._get_tk_face_and_landmarks(detected_face=face)
                if face_idx + 1 > existing_count:
                    self._on_load_add_face(frame_idx, tk_face, mesh_landmarks)
                else:
                    self._on_load_update_face(image_ids[face_idx],
                                              frame_idx,
                                              face_idx,
                                              tk_face,
                                              mesh_landmarks)
        self._switch_filter(init=True)
        self._set_selected()
        self._initialized = True

    def _on_load_remove_faces(self, existing_count, new_count, frame_index):
        """ Remove any faces that have been deleted whilst face viewer was loading. """
        if existing_count <= new_count:
            return
        for face_idx in range(new_count, existing_count):
            logger.debug("Deleting face at index %s for frame %s", face_idx, frame_index)
            self.delete_face_at_index_by_frame(frame_index, face_idx)

    def _on_load_add_face(self, frame_index, tk_face, mesh_landmarks):
        """ Add a face that has been been added face viewer was loading. """
        logger.debug("Adding new face for frame %s", frame_index)
        self._tk_faces[frame_index].append(tk_face)
        self._mesh_landmarks[frame_index].append(mesh_landmarks)
        next_frame_idx = self._get_next_frame_idx(frame_index)
        self._filtered_display.add_face(tk_face, frame_index, next_frame_idx)

    def _on_load_update_face(self, image_id, frame_index, face_index, tk_face, mesh_landmarks):
        """ Add a face that has been been added face viewer was loading. """
        logger.debug("Updating face id %s for frame %s", face_index, frame_index)
        self._tk_faces[frame_index][face_index] = tk_face
        self._mesh_landmarks[frame_index][face_index] = mesh_landmarks
        self._canvas.itemconfig(image_id, image=tk_face)
        coords = self._canvas.coords(image_id)
        mesh_tag = self._canvas.find_withtag("mesh_{}".format(frame_index))
        mesh_ids = self._canvas.mesh_ids_for_face(face_index, mesh_tag)
        for points, item_id in zip(mesh_landmarks["landmarks"], mesh_ids):
            self._canvas.coords(item_id, *(points + coords).flatten())

    def _get_tk_face_and_landmarks(self, detected_face=None):
        """ Obtain the resized photo image face and scaled landmarks """
        detected_face = self._alignments.current_face if detected_face is None else detected_face
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
        mesh_landmarks = self._canvas.get_mesh_points(detected_face.aligned_landmarks * scale)
        return tk_face, mesh_landmarks

    def _on_frame_change(self, *args):  # pylint:disable=unused-argument
        """ Action to perform on a frame change """
        if not self._initialized:
            return
        self._set_selected()

    def _set_selected(self):
        """ Set the currently selected annotations. """
        position = self._frames.tk_position.get()
        self._selected.set_selected(self._tk_faces[position],
                                    self._mesh_landmarks[position],
                                    position)

    def _update_mesh_color(self, *args):  # pylint:disable=unused-argument
        """ Update the mesh color on control panel change """
        if not self._initialized:
            return
        color = self._canvas.get_muted_color("Mesh")
        if self._annotation_colors["mesh"] == color:
            return
        highlight_color = self._canvas.control_colors["Mesh"]
        self._canvas.itemconfig("mesh_polygon", outline=color)
        self._canvas.itemconfig("mesh_line", fill=color)
        self._canvas.itemconfig("highlight_mesh_polygon", outline=highlight_color)
        self._canvas.itemconfig("highlight_mesh_line", fill=highlight_color)
        self._annotation_colors["mesh"] = color

    def _update_current(self, *args):  # pylint:disable=unused-argument
        """ Update the currently selected face on editor update """
        if not self._alignments.tk_edited.get() or not self.is_initialized:
            return
        if self._add_remove_face():
            self._selected.reload_annotations()
            return
        tk_face, mesh_landmarks = self._get_tk_face_and_landmarks()
        self._selected.update(tk_face, mesh_landmarks)
        self._alignments.tk_edited.set(False)

    def delete_face_from_viewer(self, item_id):
        """ Delete a face for the given frame and face indices """
        logger.debug("Deleting face: (item_id: %s)", item_id)
        frame_idx = self._canvas.frame_index_from_object(item_id)
        frame_faces = self._canvas.find_withtag("image_{}".format(frame_idx))
        face_idx = frame_faces.index(item_id)
        logger.debug("frame_idx: %s, frame_faces: %s, face_idx: %s",
                     frame_idx, frame_faces, face_idx)
        self._alignments.delete_face_at_index_by_frame(frame_idx, face_idx)
        self.delete_face_at_index_by_frame(frame_idx, face_idx)
        if frame_idx == self._frames.tk_position.get():
            self._frames.tk_update.set(True)
            self._selected.reload_annotations()

    def delete_face_at_index_by_frame(self, frame_index, face_index):
        """ Remove a face from the viewer for a given frame index and face index. """
        del self._tk_faces[frame_index][face_index]
        del self._mesh_landmarks[frame_index][face_index]
        self._filtered_display.delete_face_from_viewer(frame_index, face_index)

    def _add_remove_face(self):
        """ add or remove a face for the current frame """
        alignment_faces = len(self._alignments.current_faces)
        if alignment_faces > self._selected.face_count:
            self._add_face()
            retval = True
        elif alignment_faces < self._selected.face_count:
            self._remove_face()
            retval = True
        else:
            retval = False
        return retval

    def _add_face(self):
        """ Insert a face into current frame """
        logger.debug("Adding face")
        tk_face, mesh_landmarks = self._get_tk_face_and_landmarks()
        self._selected.add_face(tk_face, mesh_landmarks)
        next_frame_idx = self._get_next_frame_idx(self._selected.frame_index)
        self._filtered_display.add_face(tk_face, self._selected.frame_index, next_frame_idx)

    def _get_next_frame_idx(self, frame_index):
        """ Get the index of the next frame that has faces for placing newly added faces
        in the stack. """
        next_frame_idx = next((
            idx for idx, f_count in enumerate(self._alignments.face_count_per_index[frame_index:])
            if f_count > 0), None)
        if next_frame_idx is None:
            return None
        next_frame_idx += frame_index + 1
        logger.debug("Returning next frame with faces: %s for frame index: %s",
                     next_frame_idx, frame_index)
        return next_frame_idx

    def _remove_face(self):
        """ Remove a face from the current frame """
        logger.debug("Removing face")
        face_idx = self._selected.remove_face()
        self._filtered_display.remove_face(self._selected.frame_index, face_idx)

    def toggle_annotations(self):
        """ Toggle additional annotations on or off. """
        if not self._initialized:
            return
        self._filtered_display.toggle_annotation()


class SelectedFrame():
    """ Holds the objects and handles faces for the currently selected frame. """
    def __init__(self, image_size, alignments):
        self._alignments = alignments
        self._size = image_size
        self._canvas = None
        self._tk_faces = None
        self._mesh_landmarks = None
        self._image_ids = None
        self._mesh_ids = None
        self._highlighter = None
        self._frame_index = 0
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
    def frame_index(self):
        """ int: The current frame index. """
        return self._frame_index

    def initialize(self, canvas):
        """ Set the canvas object to :attr:`_canvas`. """
        self._canvas = canvas
        self._highlighter = Highlighter(self._size, canvas)

    def set_selected(self, tk_faces, mesh_landmarks, frame_index):
        """ Set the currently selected frame's objects """
        self._tk_faces = tk_faces
        self._mesh_landmarks = mesh_landmarks
        self._image_ids = list(self._canvas.find_withtag("image_{}".format(frame_index)))
        self._mesh_ids = list(self._canvas.find_withtag("mesh_{}".format(frame_index)))
        self._frame_index = frame_index
        self._face_count = len(self._image_ids)
        self._highlighter.highlight_selected(self._image_ids, self._mesh_landmarks)

    def reload_annotations(self):
        """ Reload the currently selected annotations on an add/remove face. """
        self._image_ids = list(self._canvas.find_withtag("image_{}".format(self._frame_index)))
        self._mesh_ids = list(self._canvas.find_withtag("mesh_{}".format(self._frame_index)))
        self._face_count = len(self._image_ids)
        self._highlighter.highlight_selected(self._image_ids, self._mesh_landmarks)

    def update(self, tk_face, mesh_landmarks):
        """ Update the currently selected face on editor update """
        if self._face_count != 0:
            self._update_face(tk_face)
            self._update_mesh(mesh_landmarks)
        self._highlighter.highlight_selected(self._image_ids, self._mesh_landmarks)

    def _update_face(self, tk_face):
        """ Update the face photo image and the face object id """
        self._tk_faces[self._face_index] = tk_face
        self._canvas.itemconfig(self._image_ids[self._face_index], image=tk_face)

    def _update_mesh(self, mesh_landmarks):
        """ Update the optional mesh annotation """
        coords = self._canvas.coords(self._image_ids[self._face_index])
        mesh_ids = self._canvas.mesh_ids_for_face(self._face_index, self._mesh_ids)
        for points, item_id in zip(mesh_landmarks["landmarks"], mesh_ids):
            self._canvas.coords(item_id, *(points + coords).flatten())
        self._mesh_landmarks[self._face_index] = mesh_landmarks

    def add_face(self, tk_face, mesh_landmarks):
        """ Add a face to the currently selected frame, """
        logger.debug("Adding face to frame(frame_id: %s new face_count: %s)",
                     self._frame_index, self._face_count + 1)
        self._tk_faces.append(tk_face)
        self._mesh_landmarks.append(mesh_landmarks)

    def remove_face(self):
        """ Remove a face from the currently selected frame. """
        face_idx = self._alignments.get_removal_index()
        logger.debug("Removing face for frame %s at index: %s", self._frame_index, face_idx)
        del self._tk_faces[face_idx]
        del self._mesh_landmarks[face_idx]
        return face_idx

# TODO Split Update, Add, Remove to own class


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

    def highlight_selected(self, image_ids, mesh_landmarks):
        """ Highlight the currently selected faces """
        self._face_count = len(image_ids)
        self._create_new_highlighters(mesh_landmarks)
        self._hide_unused_highlighters()
        if self._face_count == 0:
            return

        boxes = self._boxes[:self._face_count]
        meshes = self._meshes[:self._face_count]
        for image_id, landmarks, box, mesh in zip(image_ids, mesh_landmarks, boxes, meshes):
            top_left = np.array(self._canvas.coords(image_id))
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
        for is_poly, pts in zip(landmarks["is_poly"], landmarks["landmarks"]):
            key = "polygon" if is_poly else "line"
            tag = ["highlight_mesh_{}".format(key)]
            obj = getattr(self._canvas, "create_{}".format(key))
            obj_kwargs = kwargs[key]
            mesh_ids.append(obj(*pts.flatten(), state="hidden", width=1, tags=tag, **obj_kwargs))
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
    def __init__(self, face_cache, toggle_tags=None):
        logger.debug("Initializing: %s: (face_cache: %s, toggle_tags: %s)",
                     self.__class__.__name__, face_cache, toggle_tags)
        self._toggle_tags = toggle_tags
        self._canvas = face_cache._canvas
        self._tk_position = face_cache._frames.tk_position
        self._size = face_cache.size
        self._item_ids = dict(image_ids=[], mesh_ids=[])
        self._removed_image_index = -1
        self._frame_faces_change = 0

        self._mesh_landmarks = face_cache._mesh_landmarks

        # Set and unset during :func:`initialize` and :func:`de-initialize`
        self._updated_frames = []
        self._tk_position_callback = None
        logger.debug("Initialized: %s", self.__class__.__name__)

    @property
    def _image_ids(self):
        """ list: The canvas item ids for the face images. """
        return self._item_ids["image_ids"]

    @property
    def _mesh_ids(self):
        """ list: The canvas item ids for the optional mesh annotations. """
        return self._item_ids["mesh_ids"]

    def _set_object_display_state(self):
        """ Hide annotations that are not relevant for this filter type and show those that are

        Override for filter specific hiding criteria.
        """
        raise NotImplementedError

    def _set_display_objects(self):
        """ Set the :attr:`_image_ids` and :attr:`_mesh_ids` for the current filter

        Override for filter specific hiding criteria.
        """
        raise NotImplementedError

    def _on_frame_change(self, *args):  # pylint:disable=unused-argument
        """ Callback to be executed whenever the frame is changed.

        Override for filter specific actions
        """
        raise NotImplementedError

    def initialize(self):
        """ Initialize the viewer for the selected filter type.

        Hides annotations and faces that should not be displayed for the current filter.
        Displays and moves the faces to the correct position on the canvas based on which faces
        should be displayed.
        """
        self._set_object_display_state()
        self._set_display_objects()
        for idx, image_id in enumerate(self._image_ids):
            old_position = np.array(self._canvas.coords(image_id), dtype="int")
            new_position = self._canvas.coords_from_index(idx)
            offset = new_position - old_position
            if not offset.any():
                continue
            self._canvas.move(self._canvas.face_group_from_object(image_id), *offset)
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))
        self._tk_position_callback = self._tk_position.trace("w", self._on_frame_change)
        self._updated_frames = [self._tk_position.get()]

    def add_face(self, tk_face, frame_index, next_frame_index):
        """ Display a new face in the correct location and move subsequent faces to their new
        location.

        Parameters
        ----------
        tk_face: :class:`PIL.ImageTk.PhotoImage`
            The new face that is to be added to the canvas
        frame_index: int
            The frame index that the face is to be added to
        """
        if not self._image_ids:
            display_idx = 0
        else:
            image_ids = self._canvas.find_withtag("image_{}".format(frame_index))
            if image_ids:
                display_idx = self._image_ids.index(image_ids[-1]) + 1
            else:
                display_idx = self._removed_image_index
        # Update layout - Layout must be updated first so space is made for the new face
        self._update_layout(display_idx, is_insert=True)

        coords = self._canvas.coords_from_index(display_idx)
        logger.debug("display_idx: %s, coords: %s", display_idx, coords)
        # Create new annotations
        image_id, mesh_ids = self._canvas.create_viewer_annotations(
            coords,
            tk_face,
            self._mesh_landmarks[frame_index][-1],
            frame_index)
        # Insert annotations into correct position in object tracking
        self._image_ids.insert(display_idx, image_id)
        mesh_idx_offset = display_idx * self._canvas.items_per_mesh
        self._mesh_ids[mesh_idx_offset:mesh_idx_offset] = mesh_ids
        # Update multi tags
        self._update_multi_tags_on_add(frame_index, image_id, mesh_ids)
        # Place faces in correct position in stack
        if next_frame_index is not None:
            self._canvas.tag_lower("frame_id_{}".format(frame_index),
                                   "frame_id_{}".format(next_frame_index))
        self._frame_faces_change += 1

    def _update_multi_tags_on_add(self, frame_index, image_id, mesh_ids):
        """ Update the tags indicating whether this frame contains multiple faces.

        Parameters
        ----------
        frame_index: int
            The frame index that the tags are to be updated for
        image_id: int
            The item id of the newly added face
        mesh_ids: list
            The item ids of the newly added mesh annotations
        """
        lookup_tag = "frame_id_{}".format(frame_index)
        num_faces = len(self._canvas.find_withtag(lookup_tag))
        if num_faces == 2:
            self._canvas.dtag(lookup_tag, tag="not_multi")
        if num_faces == 1:
            self._canvas.addtag_withtag(lookup_tag, "not_multi")
        else:
            self._canvas.addtag_withtag(lookup_tag, "multi")
            self._canvas.addtag_withtag(image_id, "multi_image")
            for mesh_id in mesh_ids:
                self._canvas.addtag_withtag(mesh_id, "multi_mesh")
        return lookup_tag

    def remove_face(self, frame_index, face_index):
        """ Remove a face at the given location and update subsequent faces to the
        correct location.

        """
        display_idx = self._delete_annotations_and_update(frame_index, face_index)
        # Track the last removal index in case of adding faces back in after removing all faces
        # TODO Check this logic
        self._removed_image_index = display_idx
        self._frame_faces_change -= 1

    def _delete_annotations_and_update(self, frame_index, face_index):
        """ Delete the face annotations for the given frame and face indices and update the display
        to reflect the existing faces.

        Parameters
        ----------
        frame_index: int
            The frame index that the face has been removed from
        face_index: int
            The index of the face within the given frame that is to be removed
        """
        image_id = self._canvas.find_withtag("image_{}".format(frame_index))[face_index]
        display_idx = self._image_ids.index(image_id)
        logger.debug("image_id: %s, display_idx: %s", image_id, display_idx)

        self._canvas.delete(self._canvas.face_group_from_object(image_id))
        del self._image_ids[display_idx]
        mesh_idx_offset = display_idx * self._canvas.items_per_mesh
        self._mesh_ids[mesh_idx_offset:mesh_idx_offset + self._canvas.items_per_mesh] = []
        self._update_multi_tags_on_remove(frame_index)
        self._update_layout(display_idx, is_insert=False)
        return display_idx

    def _update_multi_tags_on_remove(self, frame_index):
        """ Update the tags indicating whether this frame contains multiple faces on face removal

        Parameters
        ----------
        frame_index: int
            The frame index that the tags are to be updated for
        """
        lookup_tag = "frame_id_{}".format(frame_index)
        num_faces = len(self._canvas.find_withtag(lookup_tag))
        if num_faces == 1:
            self._canvas.dtag(lookup_tag, tag="multi")
            self._canvas.dtag("image_{}".format(frame_index), tag="multi_image")
            self._canvas.dtag("mesh_{}".format(frame_index), tag="multi_mesh")
            self._canvas.addtag_withtag(lookup_tag, "not_multi")

    def _update_layout(self, starting_object_index, is_insert=True):
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
        hidden_tags = [val for key, val in self._toggle_tags.items()
                       if key != self._canvas.optional_annotation]
        for tag in hidden_tags:
            self._canvas.itemconfig(tag, state="normal")
        new_row_start = start_index == self._canvas.column_count - 1
        first_face_xy = self._canvas.coords(self._image_ids[start_index])
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

    def delete_face_from_viewer(self, frame_index, face_index):
        """ Delete a face when called from the face viewer.

        It is highly likely that this is called when not displaying the frame that the face is
        deleted from, so updates are run immediately rather than on a frame change.

        Parameters
        ----------
        frame_index: int
            The frame index that the face should be deleted for
        face_index: int
            The face index to remove the face for
        """
        self._delete_annotations_and_update(frame_index, face_index)
        self._updated_frames.append(frame_index)

    def toggle_annotation(self):
        """ Toggle additional object annotations on or off. """
        if self._toggle_tags is None:
            return
        display = self._canvas.optional_annotation
        if display is not None:
            self._canvas.itemconfig(self._toggle_tags[display], state="normal")
        else:
            for tag in self._toggle_tags.values():
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

    def de_initialize(self):
        """ Unloads the Face Filter on filter type change. """
        self._tk_position.trace_vdelete("w", self._tk_position_callback)
        self._tk_position_callback = None
        self._updated_frames = []


class AllFrames(FaceFilter):
    """ The Frames that have Faces viewer

    Parameters
    ----------
    face_cache: :class:`FaceCache`
        The main Face Viewers face cache, that holds all of the display items and annotations
    """
    def __init__(self, face_cache):
        super().__init__(face_cache, dict(landmarks="viewer_mesh"))

    def _set_object_display_state(self):
        """ Hide annotations that are not relevant for this filter type and show those that are

        All viewer annotations should be show
        """
        annotation = self._canvas.optional_annotation
        if annotation == "landmarks":
            self._canvas.itemconfig("viewer", state="normal")
        else:
            self._canvas.itemconfig("viewer_image", state="normal")

    def _set_display_objects(self):
        """ Set the :attr:`_image_ids` and :attr:`_mesh_ids` for the current filter

        All image and mesh annotations should be loaded
        """
        self._item_ids["image_ids"] = list(self._canvas.find_withtag("viewer_image"))
        self._item_ids["mesh_ids"] = list(self._canvas.find_withtag("viewer_mesh"))

    def _on_frame_change(self, *args):  # pylint:disable=unused-argument
        """ Callback to be executed whenever the frame is changed.

        All Frames has no frame change specific actions.
        """
        self._updated_frames = [self._tk_position.get()]


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

    def _set_object_display_state(self):
        """ Hide annotations that are not relevant for this filter type and show those that are

        All viewer annotations should be show
        """
        self._canvas.itemconfig("viewer", state="hidden")

    def _set_display_objects(self):
        """ Set the :attr:`_image_ids` and :attr:`_mesh_ids` for the current filter

        Empty lists are set as no annotations will meet criteria
        """
        self._item_ids["image_ids"] = []
        self._item_ids["mesh_ids"] = []

    def _on_frame_change(self, *args):  # pylint:disable=unused-argument
        """ Callback to be executed whenever the frame is changed.

        If faces have been added to the current frame, hide the new faces and clear
        the viewer objects.
        """
        if self._frame_faces_change != 0:
            self._canvas.itemconfig("frame_id_{}".format(self._updated_frames[0]), state="hidden")
            self._item_ids["image_ids"] = []
            self._item_ids["mesh_ids"] = []
        self._frame_faces_change = 0
        self._updated_frames = [self._tk_position.get()]


class MultipleFaces(FaceFilter):
    """ The Frames with Multiple Faces viewer.

    Parameters
    ----------
    face_cache: :class:`FaceCache`
        The main Face Viewers face cache, that holds all of the display items and annotations
    """
    def __init__(self, face_cache):
        super().__init__(face_cache, dict(landmarks="multi_mesh"))

    def _set_object_display_state(self):
        """ Hide annotations that are not relevant for this filter type and show those that are

        All viewer annotations should be show
        """
        self._canvas.itemconfig("not_multi", state="hidden")
        annotation = self._canvas.optional_annotation
        if annotation == "landmarks":
            self._canvas.itemconfig("multi", state="normal")
        else:
            self._canvas.itemconfig("multi_image", state="normal")

    def _set_display_objects(self):
        """ Set the :attr:`_image_ids` and :attr:`_mesh_ids` for the current filter

        All image and mesh annotations should be loaded
        """
        self._item_ids["image_ids"] = list(self._canvas.find_withtag("multi_image"))
        self._item_ids["mesh_ids"] = list(self._canvas.find_withtag("multi_mesh"))

    def _on_frame_change(self, *args):  # pylint:disable=unused-argument
        """ Callback to be executed whenever the frame is changed.

        Multiple faces display can be impacted by either deleting faces in the frame or by deleting
        faces from the frame viewer. Frames which have had faces deleted from the face viewer will
        have been added to :attr:`update_frames`, so an additional check has to be made here when
        changing frames to ensure that the display is updated accordingly.

        If the face count for any of the impacted frames is no longer 2 or more, remove the frame
        from :attr:`_object_indices`, hide any existing sole faces and update the layout.
        """
        logger.trace("frame_faces_change: %s, updated_frames: %s",
                     self._frame_faces_change, self._updated_frames)
        if self._frame_faces_change == 0 and len(self._updated_frames) == 1:
            self._updated_frames = [self._tk_position.get()]
            return
        for frame_index in self._updated_frames:
            image_ids = self._canvas.find_withtag("image_{}".format(frame_index))
            if len(image_ids) < 2:
                self._canvas.itemconfig("frame_id_{}".format(frame_index, state="hidden"))
            if len(image_ids) == 1:
                # Remove the final face from the display and update
                display_idx = self._image_ids.index(image_ids[0])
                del self._image_ids[display_idx]
                mesh_idx_offset = display_idx * self._canvas.items_per_mesh
                self._mesh_ids[mesh_idx_offset:mesh_idx_offset + self._canvas.items_per_mesh] = []
                self._update_layout(display_idx, is_insert=False)
        self._updated_frames = [self._tk_position.get()]
        self._frame_faces_change = 0
