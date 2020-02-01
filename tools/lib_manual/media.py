#!/usr/bin/env python3
""" Media objects for the manual adjustments tool """
import logging
import os
import tkinter as tk
from threading import Event

import cv2
from PIL import Image, ImageTk

from lib.alignments import Alignments
from lib.faces_detect import DetectedFace
from lib.image import ImagesLoader
from lib.multithreading import MultiThread

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class FrameNavigation():
    """Handles the return of the correct frame for the GUI.

    Parameters
    ----------
    frames_location: str
        The path to the input frames
    """
    def __init__(self, frames_location, scaling_factor):
        logger.debug("Initializing %s: (frames_location: '%s', scaling_factor: %s)",
                     self.__class__.__name__, frames_location, scaling_factor)
        self._loader = ImagesLoader(frames_location, fast_count=False, queue_size=1)
        self._meta = dict()
        self._current_idx = 0
        self._current_scale = 1.0
        self._scaling = scaling_factor
        self._tk_vars = self._set_tk_vars()
        self._current_frame = None
        self._current_display_frame = None
        self._display_dims = (960, 540)
        self._set_current_frame(initialize=True)
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def is_video(self):
        """ bool: 'True' if input is a video 'False' if it is a folder. """
        return self._loader.is_video

    @property
    def location(self):
        """ str: The input folder or video location. """
        return self._loader.location

    @property
    def filename_list(self):
        """ list: List of filenames in correct frame order. """
        return self._loader.file_list

    @property
    def frame_count(self):
        """ int: The total number of frames """
        return self._loader.count

    @property
    def current_meta_data(self):
        """ dict: The current cache item for the current location. Keys are `filename`,
        `display_dims`, `scale` and `interpolation`. """
        return self._meta[self.tk_position.get()]

    @property
    def current_scale(self):
        """ float: The scaling factor for the currently displayed frame """
        return self._current_scale

    @property
    def current_frame(self):
        """ :class:`numpy.ndarray`: The currently loaded, full frame. """
        return self._current_frame

    @property
    def current_frame_dims(self):
        """ tuple: The (`height`, `width`) of the source frame that is being displayed """
        return self._current_frame.shape[:2]

    @property
    def current_display_frame(self):
        """ :class:`ImageTk.PhotoImage`: The currently loaded frame, formatted and sized
        for display. """
        return self._current_display_frame

    @property
    def display_dims(self):
        """ tuple: The (`width`, `height`) of the display image with scaling factor applied. """
        retval = [int(round(dim * self._scaling)) for dim in self._display_dims]
        return tuple(retval)

    @property
    def tk_position(self):
        """ :class:`tkinter.IntVar`: The current frame position. """
        return self._tk_vars["position"]

    @property
    def tk_is_playing(self):
        """ :class:`tkinter.BooleanVar`: Whether the stream is currently playing. """
        return self._tk_vars["is_playing"]

    @property
    def tk_update(self):
        """ :class:`tkinter.BooleanVar`: Whether the display needs to be updated. """
        return self._tk_vars["updated"]

    def _set_tk_vars(self):
        """ Set the initial tkinter variables and add traces. """
        logger.debug("Setting tkinter variables")
        position = tk.IntVar()
        position.set(self._current_idx)
        position.trace("w", self._set_current_frame)

        is_playing = tk.BooleanVar()
        is_playing.set(False)

        updated = tk.BooleanVar()
        updated.set(False)

        retval = dict(position=position, is_playing=is_playing, updated=updated)
        logger.debug("Set tkinter variables: %s", retval)
        return retval

    def _set_current_frame(self, *args,  # pylint:disable=unused-argument
                           initialize=False):
        """ Set the currently loaded frame to :attr:`_current_frame`

        Parameters
        ----------
        args: tuple
            Required for event callback. Unused.
        initialize: bool, optional
            ``True`` if initializing for the first frame to be displayed otherwise ``False``.
            Default: ``False``
        """
        position = self.tk_position.get()
        if not initialize and position == self._current_idx:
            return
        filename, frame = self._loader.image_from_index(position)
        self._add_meta_data(position, frame, filename)
        self._current_frame = frame
        display = cv2.resize(self._current_frame,
                             self.current_meta_data["display_dims"],
                             interpolation=self.current_meta_data["interpolation"])[..., 2::-1]
        self._current_display_frame = ImageTk.PhotoImage(Image.fromarray(display))
        self._current_idx = position
        self._current_scale = self.current_meta_data["scale"]
        self.tk_update.set(True)

    def _add_meta_data(self, position, frame, filename):
        """ Adds the metadata for the current frame to :attr:`meta`.

        Parameters
        ----------
        position: int
            The current frame index
        frame: :class:`numpy.ndarray`
            The current frame
        filename: str
            The filename for the current frame

        """
        if position in self._meta:
            return
        scale = min(self.display_dims[0] / frame.shape[1],
                    self.display_dims[1] / frame.shape[0])
        self._meta[position] = dict(
            scale=scale,
            interpolation=cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA,
            display_dims=(int(round(frame.shape[1] * scale)),
                          int(round(frame.shape[0] * scale))),
            filename=filename)

    def increment_frame(self, is_playing=False):
        """ Update :attr:`self.current_frame` to the next frame.

        Parameters
        ----------
        is_playing: bool, optional
            ``True`` if the frame is being incremented because the Play button has been pressed.
            ``False`` if incremented for other reasons. Default: ``False``
        """
        position = self.tk_position.get()
        if position == self.frame_count - 1:
            logger.trace("End of stream. Not incrementing")
            if self.tk_is_playing.get():
                self.tk_is_playing.set(False)
            return
        self.goto_frame(position + 1, stop_playback=not is_playing and self.tk_is_playing.get())

    def decrement_frame(self):
        """ Update :attr:`self.current_frame` to the previous frame """
        position = self.tk_position.get()
        if position == 0:
            logger.trace("Beginning of stream. Not decrementing")
            return
        self.goto_frame(position - 1, stop_playback=True)

    def set_first_frame(self):
        """ Load the first frame """
        self.goto_frame(0, stop_playback=True)

    def set_last_frame(self):
        """ Load the last frame """
        self.goto_frame(self.frame_count - 1, stop_playback=True)

    def goto_frame(self, index, stop_playback=True):
        """ Load the frame given by the specified index.

        Parameters
        ----------
        index: int
            The frame index to navigate to
        stop_playback: bool, optional
            ``True`` to Stop video playback, if a video is playing, otherwise ``False``.
            Default: ``True``
        """
        if stop_playback and self.tk_is_playing.get():
            self.tk_is_playing.set(False)
        self.tk_position.set(index)


class AlignmentsData():
    """ Holds the alignments and annotations.

    Parameters
    ----------
    alignments_path: str
        Full path to the alignments file. If empty string is passed then location is calculated
        from the source folder
    frames: :class:`FrameNavigation`
        The object that holds the cache of frames.
    """
    def __init__(self, alignments_path, frames, extractor):
        logger.debug("Initializing %s: (alignments_path: '%s')",
                     self.__class__.__name__, alignments_path)
        self.frames = frames
        self._mask_names, self._alignments = self._get_alignments(alignments_path)

        self._tk_position = frames.tk_position
        self._face_index = 0
        self._extractor = extractor
        self._extractor.link_alignments(self)
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def available_masks(self):
        """ set: Names of all masks that exist in the alignments file """
        return self._mask_names

    @property
    def _latest_alignments(self):
        """ dict: The filename as key, and either the modified alignments as values (if they exist)
        or the saved alignments """
        return {key: val.get("new", val["saved"]) for key, val in self._alignments.items()}

    @property
    def current_faces(self):
        """ list: list of the current :class:`lib.faces_detect.DetectedFace` objects. Returns
        modified alignments if they are modified, otherwise original saved alignments. """
        # TODO use get and return a default for when the frame don't exist
        return self._latest_alignments[self.frames.current_meta_data["filename"]]

    @property
    def saved_alignments(self):
        """ dict: The filename as key, and the currently saved alignments as values. """
        return {key: val["saved"] for key, val in self._alignments.items()}

    @property
    def face_index(self):
        """int: The index of the current face in the current frame """
        return self._face_index

    @property
    def _face_count_per_index(self):
        """ list: Count of faces for each frame. List is in frame index order.

        The list needs to be calculated on the fly as the number of faces in a frame
        can change based on user actions. """
        alignments = self._latest_alignments
        return [len(alignments[key]) for key in sorted(alignments)]

    @property
    def current_face(self):
        """ :class:`lib.faces_detect.DetectedFace` The currently selected face """
        retval = None if not self.current_faces else self.current_faces[self._face_index]
        return retval

    @property
    def _no_face(self):
        """ list: The indexes of all frames that contain no faces """
        return [idx for idx, count in enumerate(self._face_count_per_index) if count == 0]

    @property
    def _multi_face(self):
        """ list: The indexes of all frames that contain no faces """
        return [idx for idx, count in enumerate(self._face_count_per_index) if count > 1]

    @property
    def _single_face(self):
        """ list: The indexes of all frames that contain no faces """
        return [idx for idx, count in enumerate(self._face_count_per_index) if count == 1]

    def _get_alignments(self, alignments_path):
        """ Get the alignments object.

        Parameters
        ----------
        alignments_path: str
            Full path to the alignments file. If empty string is passed then location is calculated
            from the source folder

        Returns
        -------
        dict
            `frame name`: list of :class:`lib.faces_detect.DetectedFace` for the current frame
        """
        if alignments_path:
            folder, filename = os.path.split(alignments_path, self.frames)
        else:
            filename = "alignments.fsa"
            if self.frames.is_video:
                folder, vid = os.path.split(os.path.splitext(self.frames.location)[0])
                filename = "{}_{}".format(vid, filename)
            else:
                folder = self.frames.location
        alignments = Alignments(folder, filename)
        mask_names = set(alignments.mask_summary)
        faces = dict()
        for framename, items in alignments.data.items():
            faces[framename] = []
            this_frame_faces = []
            for item in items:
                face = DetectedFace()
                face.from_alignment(item)
                face.load_aligned(None, size=96)
                this_frame_faces.append(face)
            faces[framename] = dict(saved=this_frame_faces)
        return mask_names, faces

    def set_next_frame(self, direction, filter_type):
        """ Set the display frame to the next or previous frame based on the given filter.

        Parameters
        ----------
        direction = ["prev", "next"]
            The direction to search for the next face
        filter_type: ["no", "multi", "single"]
            The filter method to use for selecting the next frame
        """
        position = self._tk_position.get()
        search_list = getattr(self, "_{}_face".format(filter_type))
        try:
            if direction == "prev":
                frame_idx = next(idx for idx in reversed(search_list) if idx < position)
            else:
                frame_idx = next(idx for idx in search_list if idx > position)
        except StopIteration:
            # If no remaining frames meet criteria go to the first or last frame
            frame_idx = 0 if direction == "prev" else self.frames.frame_count - 1
        self.frames.goto_frame(frame_idx)

    def _check_for_new_alignments(self):
        """ Checks whether there are already new alignments in :attr:`_alignments`. If not
        then saved alignments are copied to new ready for update """
        filename = self.frames.current_meta_data["filename"]
        if self._alignments[filename].get("new", None) is None:
            self._alignments[filename]["new"] = self._alignments[filename]["saved"].copy()

    def set_current_bounding_box(self, index, pnt_x, width, pnt_y, height):
        """ Update the bounding box for the current alignments.

        Parameters
        ----------
        index: int
            The face index to set this bounding box for
        pnt_x: int
            The left point of the bounding box
        width: int
            The width of the bounding box
        pnt_y: int
            The top point of the bounding box
        height: int
            The height of the bounding box
        """
        self._check_for_new_alignments()
        self._face_index = index
        face = self.current_face
        face.x = pnt_x
        face.w = width
        face.y = pnt_y
        face.h = height
        face.mask = dict()
        face.landmarks_xy = self._extractor.get_landmarks()
        face.load_aligned(self.frames.current_frame, size=96, force=True)
        self.frames.tk_update.set(True)

    def shift_landmark(self, face_index, landmark_index, shift_x, shift_y):
        """ Shift a single landmark point the given face index and landmark index by the given x and
        y values.

        Parameters
        ----------
        face_index: int
            The face index to shift the landmark for
        landmark_index: int
            The landmark index to shift
        shift_x: int
            The amount to shift the landmark by along the x axis
        shift_y: int
            The amount to shift the landmark by along the y axis
        """
        self._check_for_new_alignments()
        self._face_index = face_index
        face = self.current_face
        face.mask = dict()
        face.landmarks_xy[landmark_index] += (shift_x, shift_y)
        face.load_aligned(self.frames.current_frame, size=96, force=True)
        self.frames.tk_update.set(True)

    def shift_landmarks(self, index, shift_x, shift_y):
        """ Shift the landmarks and bounding box for the given face index by the given x and y
        values.

        Parameters
        ----------
        index: int
            The face index to shift the landmarks for
        shift_x: int
            The amount to shift the landmarks by along the x axis
        shift_y: int
            The amount to shift the landmarks by along the y axis

        Notes
        -----
        Whilst the bounding box does not need to be shifted, it is anyway, to ensure that it is
        aligned with the newly adjusted landmarks.
        """
        self._check_for_new_alignments()
        self._face_index = index
        face = self.current_face
        face.x += shift_x
        face.y += shift_y
        face.mask = dict()
        face.landmarks_xy += (shift_x, shift_y)
        face.load_aligned(self.frames.current_frame, size=96, force=True)
        self.frames.tk_update.set(True)

    def add_face(self, pnt_x, width, pnt_y, height):
        """ Add a face to the current frame with the given dimensions.

        Parameters
        ----------
        pnt_x: int
            The left point of the bounding box
        width: int
            The width of the bounding box
        pnt_y: int
            The top point of the bounding box
        height: int
            The height of the bounding box
        """
        # TODO Make sure this works if there are no pre-existing faces (probably not)
        self._check_for_new_alignments()
        self.current_faces.append(DetectedFace(x=pnt_x, w=width, y=pnt_y, h=height))
        self.set_current_bounding_box(len(self.current_faces) - 1, pnt_x, width, pnt_y, height)

    def delete_face_at_index(self, index):
        """ Delete the :class:`DetectedFace` object for the given face index.

        Parameters
        ----------
        index: int
            The face index to remove the face for
        """
        logger.debug("Deleting face at index: %s", index)
        self._check_for_new_alignments()
        del self.current_faces[index]
        self.frames.tk_update.set(True)


class FaceCache():
    """ Holds the face images for display in the bottom GUI Panel """
    # TODO Handle adding of new faces to a frame
    def __init__(self, alignments, progress_bar):
        self._alignments = alignments
        self._pbar = progress_bar
        self._size = 96
        self._faces = dict()
        self._canvas = None
        self._set_tk_trace()
        self._initialized = Event()
        self._highlighted = []
        self._landmark_mapping = dict(mouth=(48, 68),
                                      right_eyebrow=(17, 22),
                                      left_eyebrow=(22, 27),
                                      right_eye=(36, 42),
                                      left_eye=(42, 48),
                                      nose=(27, 36),
                                      jaw=(0, 17),
                                      chin=(8, 11))

    @property
    def faces(self):
        """ dict: The filename as key with list of aligned faces in :class:`ImageTk.PhotoImage`
        format for display in the GUI. """
        return self._faces

    @property
    def frame_count(self):
        """ int: The total number of frames in :attr:`_frames`. """
        return self._frames.frame_count

    @property
    def _frames(self):
        """ :class:`FrameNavigation`: The Frames for this manual session """
        return self._alignments.frames

    def _set_tk_trace(self):
        """ Set the trace on tkinter variables:
        self._frames.current_position
        """
        self._frames.tk_position.trace("w", self._highlight_current)
        self._frames.tk_update.trace("w", self._update_current)

    def load_faces(self, canvas, frame_width):
        """ Launch a background thread to load the faces into cache and assign the canvas to
        :attr:`_canvas` """
        self._canvas = canvas
        thread = MultiThread(self._load_faces,
                             frame_width,
                             thread_count=1,
                             name="{}.load_faces".format(self.__class__.__name__))
        thread.start()

    def _load_faces(self, frame_width):
        """ Loads the faces into the :attr:`_faces` dict at 128px size formatted for GUI display.

        Updates a GUI progress bar to show loading progress.
        """
        # TODO Make it so user can't save until faces are loaded (so alignments dict doesn't
        # change)
        try:
            self._pbar.start(mode="determinate")
            columns = frame_width // self._size
            face_count = 0
            loader = ImagesLoader(self._frames.location, count=self.frame_count)
            for frame_idx, (filename, frame) in enumerate(loader.load()):
                frame_name = os.path.basename(filename)
                progress = int(round(((frame_idx + 1) / self.frame_count) * 100))
                self._pbar.progress_update("Loading Faces: {}%".format(progress), progress)
                for face in self._alignments.saved_alignments.get(frame_name, list()):
                    face.load_aligned(frame, size=self._size, force=True)
                    self._faces.setdefault(frame_idx, []).append(self._place_face(columns,
                                                                                  face_count,
                                                                                  face))
                    face.aligned["face"] = None
                    face_count += 1
            self._pbar.stop()
        except Exception as err:  # pylint: disable=broad-except
            logger.error("Error loading face. Error: %s", str(err))
            # TODO Remove this
            import sys; import traceback
            exc_info = sys.exc_info(); traceback.print_exception(*exc_info)
        self._initialized.set()
        self._highlight_current()

    def _place_face(self, columns, idx, face):
        """ Places the aligned faces on the canvas and create invisible annotations.

        Returns
        -------
        dict: The `image`, `border` and `mesh` objects
        """
        dsp_face = ImageTk.PhotoImage(Image.fromarray(face.aligned_face[..., 2::-1]))

        pos = ((idx % columns) * self._size, (idx // columns) * self._size)
        rect_dims = (pos[0], pos[1], pos[0] + self._size, pos[1] + self._size)
        image_id = self._canvas.create_image(*pos, image=dsp_face, anchor=tk.NW)
        border = self._canvas.create_rectangle(*rect_dims,
                                               outline="#00ff00",
                                               width=2,
                                               state="hidden")
        mesh = self._draw_mesh(face, pos)
        objects = dict(image=dsp_face, image_id=image_id, border=border, mesh=mesh, position=pos)
        if pos[0] == 0:
            self._canvas.configure(scrollregion=self._canvas.bbox("all"))
        return objects

    def _draw_mesh(self, face, position):
        """ Draw an invisible landmarks mesh on the face that can be displayed when required """
        mesh = []
        for key in sorted(self._landmark_mapping):
            val = self._landmark_mapping[key]
            pts = (face.aligned_landmarks[val[0]:val[1]] + position).flatten()
            if key in ("right_eye", "left_eye", "mouth"):
                mesh.append(self._canvas.create_polygon(*pts,
                                                        fill="",
                                                        outline="#00ffff",
                                                        state="hidden",
                                                        width=1))
            else:
                mesh.append(self._canvas.create_line(*pts,
                                                     fill="#00ffff",
                                                     state="hidden",
                                                     width=1))
        return mesh

    def _clear_highlighted(self):
        """ Clear all the highlighted borders and landmarks """
        for item_id in self._highlighted:
            self._canvas.itemconfig(item_id, state="hidden")
        self._highlighted = []

    def _highlight_current(self, *args):  # pylint:disable=unused-argument
        """ Place a border around current face and display landmarks """
        if not self._initialized.is_set():
            return
        position = self._frames.tk_position.get()
        self._clear_highlighted()

        objects = self._faces.get(position, None)
        if objects is None:
            return

        for obj in objects:
            self._canvas.itemconfig(obj["border"], state="normal")
            self._highlighted.append(obj["border"])
            for mesh in obj["mesh"]:
                self._canvas.itemconfig(mesh, state="normal")
                self._highlighted.append(mesh)

        top = self._canvas.coords(self._highlighted[0])[1] / self._canvas.bbox("all")[3]
        if top != self._canvas.yview()[0]:
            self._canvas.yview_moveto(top)

    def _update_current(self, *args):  # pylint:disable=unused-argument
        """ Update the currently selected face on editor update """
        if not self._initialized.is_set():
            return
        face = self._alignments.current_face
        if face is None or face.aligned_face is None:
            return

        position = self._frames.tk_position.get()
        objects = self._faces[position][self._alignments.face_index]
        objects["image"] = ImageTk.PhotoImage(Image.fromarray(face.aligned_face[..., 2::-1]))
        self._canvas.itemconfig(objects["image_id"], image=objects["image"])

        for idx, key in enumerate(sorted(self._landmark_mapping)):
            val = self._landmark_mapping[key]
            pts = (face.aligned_landmarks[val[0]:val[1]] + objects["position"]).flatten()
            self._canvas.coords(objects["mesh"][idx], *pts)
        face.aligned["face"] = None
