#!/usr/bin/env python3
""" Media objects for the manual adjustments tool """
import logging
import os
import tkinter as tk
from threading import Event

import cv2
import numpy as np
from PIL import Image, ImageTk

from lib.aligner import Extract as AlignerExtract
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
        self._loader = None
        self._meta = dict()
        self._needs_update = False
        self._current_idx = 0
        self._scaling = scaling_factor
        self._tk_vars = self._set_tk_vars()
        self._current_frame = None
        self._current_display_frame = None
        self._display_dims = (896, 504)
        self._init_thread = self._background_init_frames(frames_location)
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def is_initialized(self):
        """ bool: ``True`` if the aligner has completed initialization otherwise ``False``. """
        thread_is_alive = self._init_thread.is_alive()
        if thread_is_alive:
            self._init_thread.check_and_raise_error()
        else:
            self._init_thread.join()
            # Setting the initial frame cannot be done in the thread, so set when queried from main
            self._set_current_frame(initialize=True)
        return not thread_is_alive

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
        return self.current_meta_data["scale"]

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
    def needs_update(self):
        """ bool: ``True`` if the position has changed and displayed frame needs to be updated
        otherwise ``False`` """
        return self._needs_update

    @property
    def tk_position(self):
        """ :class:`tkinter.IntVar`: The current frame position. """
        return self._tk_vars["position"]

    @property
    def tk_transport_position(self):
        """ :class:`tkinter.IntVar`: The current index of the display frame's transport slider. """
        return self._tk_vars["transport_position"]

    @property
    def tk_is_playing(self):
        """ :class:`tkinter.BooleanVar`: Whether the stream is currently playing. """
        return self._tk_vars["is_playing"]

    @property
    def tk_update(self):
        """ :class:`tkinter.BooleanVar`: Whether the display needs to be updated. """
        return self._tk_vars["updated"]

    @property
    def tk_navigation_mode(self):
        """ :class:`tkinter.StringVar`: The variable holding the selected frame navigation
        mode. """
        return self._tk_vars["nav_mode"]

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

        nav_mode = tk.StringVar()
        transport_position = tk.IntVar()
        transport_position.set(0)

        retval = dict(position=position,
                      is_playing=is_playing,
                      updated=updated,
                      nav_mode=nav_mode,
                      transport_position=transport_position)
        logger.debug("Set tkinter variables: %s", retval)
        return retval

    def _background_init_frames(self, frames_location):
        """ Launch the images loader in a background thread so we can run other tasks whilst
        waiting for initialization. """
        thread = MultiThread(self._load_images,
                             frames_location,
                             thread_count=1,
                             name="{}.init_frames".format(self.__class__.__name__))
        thread.start()
        return thread

    def _load_images(self, frames_location):
        """ Load the images in a background thread. """
        self._loader = ImagesLoader(frames_location, fast_count=False, queue_size=1)

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
        self._needs_update = True
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

    def clear_update_flag(self):
        """ Trigger to clear the update flag once the canvas has been updated with the latest
        display frame """
        logger.trace("Clearing update flag")
        self._needs_update = False

    def stop_playback(self):
        """ Stop play back if playing """
        if self.tk_is_playing.get():
            logger.trace("Stopping playback")
            self.tk_is_playing.set(False)


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
    def __init__(self, alignments_path, frames, extractor, input_location, is_video):
        logger.debug("Initializing %s: (alignments_path: '%s', frames: %s, extractor: %s, "
                     "input_location: %s, is_video: %s)", self.__class__.__name__, alignments_path,
                     frames, extractor, input_location, is_video)
        self.frames = frames
        self._remove_idx = None
        self._face_size = min(self.frames.display_dims)
        self._mask_names, self._alignments = self._get_alignments(alignments_path,
                                                                  input_location,
                                                                  is_video)

        self._tk_position = frames.tk_position
        self._face_index = 0
        self._face_count_modified = False
        self._extractor = extractor
        self._extractor.link_alignments(self)
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def extractor(self):
        """ :class:`lib.manual.Aligner`: The aligner for calculating landmarks. """
        return self._extractor

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
    def face_count_modified(self):
        """ bool: ``True`` if a face has been deleted or inserted for the current frame
        otherwise ``False``. """
        return self._face_count_modified

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
    def with_face_count(self):
        """ int: The count of frames that contain no faces """
        return sum(1 for faces in self._latest_alignments.values() if len(faces) != 0)

    @property
    def no_face_count(self):
        """ int: The count of frames that contain no faces """
        return sum(1 for faces in self._latest_alignments.values() if len(faces) == 0)

    @property
    def multi_face_count(self):
        """ int: The count of frames that contain multiple faces """
        return sum(1 for faces in self._latest_alignments.values() if len(faces) > 1)

    def reset_face_id(self):
        """ Reset the attribute :attr:`_face_index` to 0 """
        self._face_index = 0

    def get_filtered_frames_list(self):
        """ Return a list of filtered faces based on navigation mode """
        nav_mode = self.frames.tk_navigation_mode.get()
        if nav_mode == "No Faces":
            retval = [idx for idx, count in enumerate(self._face_count_per_index) if count == 0]
        elif nav_mode == "Multiple Faces":
            retval = [idx for idx, count in enumerate(self._face_count_per_index) if count > 1]
        elif nav_mode == "Has Face(s)":
            retval = [idx for idx, count in enumerate(self._face_count_per_index) if count != 0]
        else:
            retval = range(self.frames.frame_count)
        logger.trace("nav_mode: %s, number_frames: %s", nav_mode, len(retval))
        return retval

    def _get_alignments(self, alignments_path, input_location, is_video):
        """ Get the alignments object.

        Parameters
        ----------
        alignments_path: str
            Full path to the alignments file. If empty string is passed then location is calculated
            from the source folder
        is_video: bool
            ``True`` if the input file is a video otherwise ``False``

        Returns
        -------
        dict
            `frame name`: list of :class:`lib.faces_detect.DetectedFace` for the current frame
        """
        if alignments_path:
            folder, filename = os.path.split(alignments_path)
        else:
            filename = "alignments.fsa"
            if is_video:
                folder, vid = os.path.split(os.path.splitext(input_location)[0])
                filename = "{}_{}".format(vid, filename)
            else:
                folder = input_location
        alignments = Alignments(folder, filename)
        mask_names = set(alignments.mask_summary)
        faces = dict()
        for framename, items in alignments.data.items():
            faces[framename] = []
            this_frame_faces = []
            for item in items:
                face = DetectedFace()
                face.from_alignment(item)
                # Size is set so attributes are correct for zooming into a face in the frame viewer
                face.load_aligned(None, size=self._face_size)
                this_frame_faces.append(face)
            faces[framename] = dict(saved=this_frame_faces)
        return mask_names, faces

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

        Notes
        -----
        The aligned face image is loaded so that the faces viewer can pick it up. This image
        is cleared by the faces viewer after collection to save ram.
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
        face.load_aligned(self.frames.current_frame, size=self._face_size, force=True)
        self.frames.tk_update.set(True)

    def shift_landmark(self, face_index, landmark_index, shift_x, shift_y, is_zoomed):
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
        is_zoomed: bool
            ``` True if landmarks are being adjusted on a zoomed image otherwise ``False``

        Notes
        -----
        The aligned face image is loaded so that the faces viewer can pick it up. This image
        is cleared by the faces viewer after collection to save ram.
        """
        self._check_for_new_alignments()
        self._face_index = face_index
        face = self.current_face
        face.mask = dict()  # TODO Something with masks that doesn't involve clearing them
        if is_zoomed:
            landmark = face.aligned_landmarks[landmark_index]
            landmark += (shift_x, shift_y)
            matrix = AlignerExtract.transform_matrix(face.aligned["matrix"],
                                                     face.aligned["size"],
                                                     face.aligned["padding"])
            matrix = cv2.invertAffineTransform(matrix)
            landmark = np.reshape(landmark, (1, 1, 2))
            landmark = cv2.transform(landmark, matrix, landmark.shape).squeeze()
            face.landmarks_xy[landmark_index] = landmark
        else:
            face.landmarks_xy[landmark_index] += (shift_x, shift_y)
        face.load_aligned(self.frames.current_frame, size=self._face_size, force=True)
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

        The aligned face image is loaded so that the faces viewer can pick it up. This image
        is cleared by the faces viewer after collection to save ram.
        """
        self._check_for_new_alignments()
        self._face_index = index
        face = self.current_face
        face.x += shift_x
        face.y += shift_y
        face.mask = dict()
        face.landmarks_xy += (shift_x, shift_y)
        face.load_aligned(self.frames.current_frame, size=self._face_size, force=True)
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
        self._face_count_modified = True
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
        self._remove_idx = index  # Set the remove_idx to this index for Faces window to pick up
        del self.current_faces[index]
        self._face_count_modified = True
        self._face_index = 0
        self.frames.tk_update.set(True)

    def reset_face_count_modified(self):
        """ Reset :attr:`_face_count_modified` to ``False``. """
        self._face_count_modified = False

    def get_removal_index(self):
        """ Return the index for the face set for removal and reset :attr:`_remove_idx` to None.

        Called from the Faces viewer when a face has been removed from the alignments file

        Returns
        -------
        int:
            The index of the currently displayed faces that has been removed
        """
        retval = self._remove_idx
        self._remove_idx = None
        return retval

    def get_aligned_face_at_index(self, index):
        """ Return the aligned face sized for frame viewer.

        Parameters
        ----------
        index: int
            The face index to return the face for

        Returns
        -------
        :class:`numpy.ndarray`
            The aligned face
        """
        face = self.current_faces[index]
        face.load_aligned(self.frames.current_frame, size=self._face_size, force=True)
        retval = face.aligned_face.copy()
        face.aligned["face"] = None
        return retval

    def copy_alignments(self, direction):
        """ Copy the alignments from the previous or next frame that has alignments
        to the current frame.

        Notes
        -----
        The aligned face image is loaded so that the faces viewer can pick it up. This image
        is cleared by the faces viewer after collection to save ram.
        """
        alignments = self._latest_alignments
        if direction == "previous":
            search_list = reversed(sorted(alignments)[:self.frames.tk_position.get()])
        else:
            search_list = sorted(alignments)[self.frames.tk_position.get() + 1:]
        for frame_idx in search_list:
            if len(alignments[frame_idx]) == 0:
                continue
            self.current_faces.extend(alignments[frame_idx])
            break
        for face in self.current_faces:
            face.load_aligned(self.frames.current_frame, size=self._face_size, force=True)
        self._face_index = len(self.current_faces) - 1
        self.frames.tk_update.set(True)


class FaceCache():
    """ Holds the face images for display in the bottom GUI Panel """
    def __init__(self, alignments, progress_bar):
        self._alignments = alignments
        self._pbar = progress_bar
        self._size = 96
        self._faces = dict()
        self._canvas = None
        self._columns = None
        self._set_tk_trace()
        self._initialized = Event()
        self._selected = []
        self._current_display = None
        self._hovered = None
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

    @property
    def _colors(self):
        """ dict: Colors for the annotations. """
        return dict(border="#00ff00", mesh="#00ffff", border_half="#009900", mesh_half="#009999")

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

    def clear_hovered(self):
        """ Hide the hovered box and clear the :attr:`_hovered` attribute """
        if self._hovered is None:
            return
        self._canvas.itemconfig(self._hovered, state="hidden")
        self._hovered = None

    def highlight_hovered(self, frame_id, object_id):
        """ Display the box around the face the mouse is over

        Parameters
        ----------
        object_id: int
            The tkinter canvas object id
        """
        self.clear_hovered()
        tags = [tag.replace("face_id_", "")
                for tag in self._canvas.itemcget(object_id, "tags").split()
                if tag.startswith("face_id_")]
        if not tags:
            return
        self._hovered = self._faces[frame_id][int(tags[0])]["border"]
        self._canvas.itemconfig(self._hovered, state="normal")

    def _set_tk_trace(self):
        """ Set the trace on tkinter variables:
        self._frames.current_position
        """
        self._frames.tk_position.trace("w", self._highlight_selected)
        self._frames.tk_update.trace("w", self._update_current)
        self._frames.tk_navigation_mode.trace("w", self._filter_faces)

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
        """ Loads the faces into the :attr:`_faces` dict at 96px size formatted for GUI display.

        Updates a GUI progress bar to show loading progress.
        """
        # TODO Make it so user can't save until faces are loaded (so alignments dict doesn't
        # change)
        try:
            self._pbar.start(mode="determinate")
            self._columns = frame_width // self._size
            faces_seen = 0
            loader = ImagesLoader(self._frames.location, count=self.frame_count)
            for frame_idx, (filename, frame) in enumerate(loader.load()):
                frame_name = os.path.basename(filename)
                progress = int(round(((frame_idx + 1) / self.frame_count) * 100))
                self._pbar.progress_update("Loading Faces: {}%".format(progress), progress)
                frame_faces = []
                faces = self._alignments.saved_alignments.get(frame_name, list())
                for face_idx, face in enumerate(faces):
                    face.load_aligned(frame, size=self._size, force=True)
                    frame_faces.append(self._place_face(self._create_tag(frame_idx, face_idx),
                                                        faces_seen,
                                                        face))
                    face.aligned["face"] = None
                    faces_seen += 1
                self._faces[frame_idx] = frame_faces
            self._pbar.stop()
        except Exception as err:  # pylint: disable=broad-except
            logger.error("Error loading face. Error: %s", str(err))
            # TODO Remove this
            import sys; import traceback
            exc_info = sys.exc_info(); traceback.print_exception(*exc_info)
        self._initialized.set()
        self._highlight_selected()

    @staticmethod
    def _create_tag(frame_index, face_index):
        """ Create an object tag from the frame and face index """
        return ["frame_id_{}".format(frame_index), "face_id_{}".format(face_index)]

    def _position_from_index(self, index):
        """ Return the position an object should be placed on the canvas from it's index. """
        return ((index % self._columns) * self._size, (index // self._columns) * self._size)

    def _place_face(self, tags, idx, face, landmarks=None):
        """ Places the aligned faces on the canvas and create invisible annotations.

        Returns
        -------
        dict: The `image`, `border` and `mesh` objects
        """
        aligned_face = face.aligned_face[..., 2::-1]
        if aligned_face.shape[0] != self._size:
            aligned_face = cv2.resize(aligned_face,
                                      (self._size, self._size),
                                      interpolation=cv2.INTER_AREA)
        dsp_face = ImageTk.PhotoImage(Image.fromarray(aligned_face))

        pos = self._position_from_index(idx)
        rect_dims = (pos[0], pos[1], pos[0] + self._size, pos[1] + self._size)
        image_id = self._canvas.create_image(*pos, image=dsp_face, anchor=tk.NW, tags=tags)
        border = self._canvas.create_rectangle(*rect_dims,
                                               outline=self._colors["border"],
                                               width=2,
                                               state="hidden",
                                               tags=tags)
        mesh = self._draw_mesh(face, pos, tags)
        # Creating new object, the landmarks are scaled correctly. For existing objects
        # the scaled landmarks are passed in.
        landmarks = face.aligned_landmarks if landmarks is None else landmarks
        objects = dict(image=dsp_face,
                       image_id=image_id,
                       border=border,
                       mesh=mesh,
                       position=pos,
                       index=idx,
                       landmarks=landmarks)
        if pos[0] == 0:
            self._canvas.configure(scrollregion=self._canvas.bbox("all"))
        return objects

    def _draw_mesh(self, face, position, tags):
        """ Draw an invisible landmarks mesh on the face that can be displayed when required """
        mesh = []
        for key in sorted(self._landmark_mapping):
            val = self._landmark_mapping[key]
            pts = (face.aligned_landmarks[val[0]:val[1]] + position).flatten()
            if key in ("right_eye", "left_eye", "mouth"):
                mesh.append(self._canvas.create_polygon(*pts,
                                                        fill="",
                                                        outline=self._colors["mesh"],
                                                        state="hidden",
                                                        width=1,
                                                        tags=tags + ["mesh"]))
            else:
                mesh.append(self._canvas.create_line(*pts,
                                                     fill=self._colors["mesh"],
                                                     state="hidden",
                                                     width=1,
                                                     tags=tags + ["mesh"]))
        return mesh

    def _clear_selected(self):
        """ Clear all the highlighted borders and landmarks """
        for object_id in self._selected:
            if self._current_display in self._canvas.itemcget(object_id, "tags").split():
                self._display_annotations(self._current_display, [object_id])
            else:
                self._canvas.itemconfig(object_id, state="hidden")
        self._selected = []

    def _highlight_selected(self, *args, new_frame=True):  # pylint:disable=unused-argument
        """ Place a border around current face and display landmarks """
        if not self._initialized.is_set():
            return
        position = self._frames.tk_position.get()
        self._clear_selected()
        if new_frame:
            self._filter_faces()
        objects = self._faces.get(position, None)
        if not objects or objects is None:
            return
        for obj in objects:
            self._canvas.itemconfig(obj["border"], state="normal")
            self._selected.append(obj["border"])
            for mesh in obj["mesh"]:
                color_attr = "outline" if self._canvas.type(mesh) == "polygon" else "fill"
                kwargs = {color_attr: self._colors["mesh"], "state": "normal"}
                self._canvas.itemconfig(mesh, **kwargs)
                self._selected.append(mesh)

        top = self._canvas.coords(self._selected[0])[1] / self._canvas.bbox("all")[3]
        if top != self._canvas.yview()[0]:
            self._canvas.yview_moveto(top)

    def _update_current(self, *args):  # pylint:disable=unused-argument
        """ Update the currently selected face on editor update """
        if not self._initialized.is_set():
            return

        position = self._frames.tk_position.get()
        if position != self._current_frame_id:
            self._current_frame_id = position
            self._alignments.reset_face_id()

        self._add_remove_face()
        face = self._alignments.current_face
        if face is None:
            return
        if face.aligned_face is None:
            # When in zoomed in mode the face isn't loaded, so get a copy
            aligned_face = self._alignments.get_aligned_face_at_index(self._alignments.face_index)
        else:
            aligned_face = face.aligned_face

        objects = self._faces[position][self._alignments.face_index]
        display_face = cv2.resize(aligned_face[..., 2::-1],
                                  (self._size, self._size),
                                  interpolation=cv2.INTER_AREA)
        objects["image"] = ImageTk.PhotoImage(Image.fromarray(display_face))
        scale = self._size / face.aligned["size"]
        landmarks = face.aligned_landmarks * scale
        objects["landmarks"] = landmarks
        self._canvas.itemconfig(objects["image_id"], image=objects["image"])

        for idx, key in enumerate(sorted(self._landmark_mapping)):
            val = self._landmark_mapping[key]
            pts = (landmarks[val[0]:val[1]] + objects["position"]).flatten()
            self._canvas.coords(objects["mesh"][idx], *pts)
        face.aligned["face"] = None

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

    def _add_remove_face(self):
        """ Add or remove a face into the viewer """
        current_faces = self._faces.get(self._current_frame_id, [])
        alignment_faces = len(self._alignments.current_faces)
        display_faces = len(current_faces)
        if alignment_faces > display_faces:
            tags = self._create_tag(self._current_frame_id, alignment_faces - 1)
            starting_idx = self._insert_new_face(current_faces, tags)
            increment = True
        elif alignment_faces < display_faces:
            starting_idx = self._delete_face(current_faces)
            increment = False
        else:
            return
        self._update_following_faces(starting_idx, increment)
        self._face_count_modified = True
        self._highlight_selected(new_frame=False)

    def _insert_new_face(self, current_faces, tags):
        """ Insert a new face into the faces viewer. """
        insert_index = current_faces[-1]["index"] + 1 if current_faces else 0
        scale = self._size / self._alignments.current_face.aligned["size"]
        landmarks = self._alignments.current_face.aligned_landmarks * scale
        current_faces.append(self._place_face(tags,
                                              insert_index,
                                              self._alignments.current_face,
                                              landmarks=landmarks))
        return insert_index + 1

    def _delete_face(self, current_faces):
        """ Remove a face from the faces viewer. """
        face_idx = self._alignments.get_removal_index()
        rem_face = current_faces[face_idx]
        removed_index = rem_face["index"]
        indices = [rem_face["image_id"], rem_face["border"]] + rem_face["mesh"]
        for idx in indices:
            self._canvas.delete(idx)
        del current_faces[face_idx]
        return removed_index

    def _update_following_faces(self, starting_idx, increment=True):
        """ Update the face positions for all faces following an insert or delete. """
        update_faces = [obj for k, v in self._faces.items() for obj in v
                        if k > self._current_frame_id or (k == self._current_frame_id
                                                          and obj["index"] >= starting_idx)]
        for objects in update_faces:
            idx = objects["index"] + 1 if increment else objects["index"] - 1
            objects["index"] = idx
            pos = self._position_from_index(idx)
            rect_dims = (pos[0], pos[1], pos[0] + self._size, pos[1] + self._size)
            objects["position"] = pos
            self._canvas.coords(objects["image_id"], *pos)
            self._canvas.coords(objects["border"], *rect_dims)
            for idx, key in enumerate(sorted(self._landmark_mapping)):
                val = self._landmark_mapping[key]
                pts = (objects["landmarks"][val[0]:val[1]] + objects["position"]).flatten()
                self._canvas.coords(objects["mesh"][idx], *pts)

    def _filter_faces(self, *args):  # pylint:disable=unused-argument
        """ Filter the viewed faces based on currently selected navigation mode. """
        if not self._initialized.is_set():
            return
        nav_mode = self._frames.tk_navigation_mode.get()
        nav_mode = "All Frames" if nav_mode == "Has Face(s)" else nav_mode
        if ((nav_mode == self._filter_mode and not self._face_count_modified) or
                (self._face_count_modified and nav_mode == "All Frames")):
            logger.trace("Not filtering faces: (nav_mode: %s, self._filter_mode: %s, "
                         "self._face_count_modified: %s)", nav_mode, self._filter_mode,
                         self._face_count_modified)
            self._face_count_modified = False
            return
        frames_list = self._alignments.get_filtered_frames_list()
        dsp_idx = 0
        for frame_idx, faces in self._faces.items():
            show_face = frame_idx in frames_list
            state = "normal" if show_face else "hidden"
            for objects in faces:
                self._canvas.itemconfig(objects["image_id"], state=state)
                if show_face:
                    pos = self._position_from_index(dsp_idx)
                    rect_dims = (pos[0], pos[1], pos[0] + self._size, pos[1] + self._size)
                    self._canvas.coords(objects["image_id"], *pos)
                    self._canvas.coords(objects["border"], *rect_dims)

                for idx, key in enumerate(sorted(self._landmark_mapping)):
                    if self._current_display == "mesh":
                        self._canvas.itemconfig(objects["mesh"][idx], state=state)
                    if show_face:
                        val = self._landmark_mapping[key]
                        pts = (objects["landmarks"][val[0]:val[1]] + pos).flatten()
                        self._canvas.coords(objects["mesh"][idx], *pts)
                if show_face:
                    dsp_idx += 1

        self._filter_mode = nav_mode
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))
        self._canvas.yview_moveto(0.0)
        self._face_count_modified = False
