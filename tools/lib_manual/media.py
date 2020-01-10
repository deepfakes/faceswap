#!/usr/bin/env python3
""" Media objects for the manual adjustments tool """
import logging
import os
import tkinter as tk

import cv2
from PIL import Image, ImageTk

from lib.alignments import Alignments
from lib.faces_detect import DetectedFace
from lib.image import ImagesLoader

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Annotations():
    """ The landmark and bounding box annotations.

    Parameters
    ----------
    alignments: :class:`AlignmentsData`
        The alignments cache object for the manual tool

    """
    def __init__(self, alignments, frames, canvas):
        self._alignments = alignments
        self._frames = frames
        self._canvas = canvas
        self._colors = dict(red="#ff0000",
                            green="#00ff00",
                            blue="#0000ff",
                            cyan="#00ffff",
                            yellow="#ffff00",
                            magenta="#ff00ff")
        # NB: Order of building is important, as we want interactive objects on top.
        self._mesh = self._update_mesh()
        self._extract_box = self._update_extract_box()
        self._landmarks = self._update_landmarks()
        self._bounding_box = self._update_bounding_box()

    @property
    def _scaling(self):
        """ float: The scaling factor for the currently displayed frame """
        return self._frames.current_scale

    @property
    def bounding_box_layout(self):
        """ tuple: The layout order of tkinter canvas bounding box points """
        return ("left", "top", "right", "bottom")

    @property
    def bounding_box_corner_order(self):
        """ dict: The position index of bounding box corners """
        return {0: ("top", "left"),
                1: ("bottom", "left"),
                2: ("top", "right"),
                3: ("bottom", "right")}

    @property
    def bounding_boxes(self):
        """ list: List of (`Left`, `Top`, `Right`, `Bottom`) tuples for each displayed face's
        bounding box. """
        return [self._canvas.coords(face[0]) for face in self._bounding_box]

    @property
    def bounding_box_anchors(self):
        """ list: List of bounding box anchors for the corners of each face's bounding box. """
        return [[self._canvas.coords(obj) for obj in face[1:]]
                for face in self._bounding_box]

    @property
    def bounding_box_points(self):
        """ list: List of bounding box tuples for each face's bounding box """
        return [((self._canvas.coords(obj[0])[0], self._canvas.coords(obj[0])[1]),
                 (self._canvas.coords(obj[0])[0], self._canvas.coords(obj[0])[3]),
                 (self._canvas.coords(obj[0])[2], self._canvas.coords(obj[0])[1]),
                 (self._canvas.coords(obj[0])[2], self._canvas.coords(obj[0])[3]))
                for obj in self._bounding_box]

    def update(self, skip_bounding_box=False):
        """ Update the annotations for the currently displayed frame.

        Parameters
        ----------
        skip_bounding_box: bool, optional
            ``True`` if the annotations for the bounding box should not be updated
            otherwise ``False``. Default: ``False``
        """
        self._clear_annotations(skip_bounding_box)
        self._mesh = self._update_mesh()
        self._extract_box = self._update_extract_box()
        self._landmarks = self._update_landmarks()
        if not skip_bounding_box:
            self._bounding_box = self._update_bounding_box()

    def _clear_annotations(self, skip_bounding_box):
        """ Removes all currently drawn annotations.

        Parameters
        ----------
        skip_bounding_box: bool
            ``True`` if the annotations for the bounding box should not be updated
            otherwise ``False``
        """
        for title in ("mesh", "extract_box", "landmarks", "bounding_box"):
            if title == "bounding_box" and skip_bounding_box:
                continue
            for face in getattr(self, "_{}".format(title)):
                for instance in face:
                    self._canvas.delete(instance)

    def bbox_objects_for_face(self, index):
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
        return self._bounding_box[index]

    def _update_bounding_box(self):
        """ Draw the bounding box around faces """
        color = self._colors["blue"]
        thickness = 1
        faces = []
        for face in self._alignments.current_faces:
            bbox = []
            box = (face.left * self._scaling,
                   face.top * self._scaling,
                   face.right * self._scaling,
                   face.bottom * self._scaling)
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
        return faces

    def _update_extract_box(self):
        """ Draw the extracted face box """
        color = self._colors["green"]
        thickness = 1
        faces = []
        # TODO FIX THIS TEST
        #  if not all(face.original_roi for face in self._alignments.current_faces):
        #      return extract_box
        for idx, face in enumerate(self._alignments.current_faces):
            extract_box = []
            logger.trace("Drawing Extract Box: (idx: %s, roi: %s)", idx, face.original_roi)
            box = face.original_roi.flatten() * self._scaling
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
        return faces

    def _update_landmarks(self):
        """ Draw the facial landmarks """
        color = self._colors["red"]
        radius = 1
        faces = []
        for face in self._alignments.current_faces:
            landmarks = []
            for landmark in face.landmarks_xy:
                box = (landmark * self._scaling).astype("int32")
                bbox = (box[0] - radius, box[1] - radius, box[0] + radius, box[1] + radius)
                landmarks.append(self._canvas.create_oval(*bbox,
                                                          outline=color,
                                                          fill=color,
                                                          width=radius))
            faces.append(landmarks)
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
                pts = (landmarks[val[0]:val[1]] * self._scaling).astype("int32").flatten()
                if key in ("right_eye", "left_eye", "mouth"):
                    mesh.append(self._canvas.create_polygon(*pts,
                                                            fill="",
                                                            outline=color,
                                                            width=thickness))
                else:
                    mesh.append(self._canvas.create_line(*pts, fill=color, width=thickness))
            faces.append(mesh)
        return faces


class FrameNavigation():
    """Handles the return of the correct frame for the GUI.

    Parameters
    ----------
    frames_location: str
        The path to the input frames
    """
    def __init__(self, frames_location):
        logger.debug("Initializing %s: (frames_location: '%s')",
                     self.__class__.__name__, frames_location)
        self._loader = ImagesLoader(frames_location, fast_count=False, queue_size=1)
        self._delay = int(round(1000 / self._loader.fps))
        self._meta = dict()
        self._current_idx = 0
        self._current_scale = 1.0
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
    def current_display_frame(self):
        """ :class:`ImageTk.PhotoImage`: The currently loaded frame, formatted and sized
        for display. """
        return self._current_display_frame

    @property
    def delay(self):
        """ int: The number of milliseconds between updates to playback the video at the
        correct fps """
        return self._delay

    @property
    def display_dims(self):
        """ tuple: The (`width`, `height`) of the display image. """
        return self._display_dims

    @property
    def tk_playback_speed(self):
        """ :class:`tkinter.IntVar`: Multiplier to playback speed. """
        return self._tk_vars["speed"]

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

        speed = tk.StringVar()

        is_playing = tk.BooleanVar()
        is_playing.set(False)

        updated = tk.BooleanVar()
        updated.set(False)

        retval = dict(position=position, is_playing=is_playing, speed=speed, updated=updated)
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
        scale = min(self._display_dims[0] / frame.shape[1],
                    self._display_dims[1] / frame.shape[0])
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
        if not is_playing and self.tk_is_playing.get():
            self.tk_is_playing.set(False)
        self.tk_position.set(position + 1)

    def decrement_frame(self):
        """ Update :attr:`self.current_frame` to the previous frame """
        position = self.tk_position.get()
        if self.tk_is_playing.get():
            # Stop playback
            self.tk_is_playing.set(False)
        if position == 0:
            logger.trace("Beginning of stream. Not decrementing")
            return
        self.tk_position.set(position - 1)

    def set_first_frame(self):
        """ Load the first frame """
        if self.tk_is_playing.get():
            self.tk_is_playing.set(False)
        self.tk_position.set(0)

    def set_last_frame(self):
        """ Load the last frame """
        if self.tk_is_playing.get():
            self.tk_is_playing.set(False)
        self.tk_position.set(self.frame_count - 1)


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
        self._alignments = self._get_alignments(alignments_path)
        self._tk_position = frames.tk_position
        self._face_index = 0
        self._extractor = extractor
        self._extractor.link_alignments(self)
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def current_faces(self):
        """ list: list of the current :class:`lib.faces_detect.DetectedFace` objects. Returns
        modified alignments if they are modified, otherwise original saved alignments. """
        alignments = self._alignments[self.frames.current_meta_data["filename"]]
        # TODO use get and return a default for when alignments don't exist
        retval = alignments.get("new", alignments["saved"])
        return retval

    @property
    def current_face(self):
        """ :class:`lib.faces_detect.DetectedFace` The currently selected face """
        return self.current_faces[self._face_index]

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
        self._face_index = index
        filename = self.frames.current_meta_data["filename"]
        if self._alignments[filename].get("new", None) is None:
            # Copy over saved alignments to new alignments
            self._alignments[filename]["new"] = self._alignments[filename]["saved"].copy()
        face = self.current_face
        face.x = pnt_x
        face.w = width
        face.y = pnt_y
        face.h = height
        face.mask = dict()
        face.landmarks_xy = self._extractor.get_landmarks()
        face.load_aligned(None, size=128, force=True)
        self.frames.tk_update.set(True)

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
        faces = dict()
        for framename, items in alignments.data.items():
            faces[framename] = []
            this_frame_faces = []
            for item in items:
                face = DetectedFace()
                face.from_alignment(item)
                face.load_aligned(None, size=128)
                this_frame_faces.append(face)
            faces[framename] = dict(saved=this_frame_faces)
        return faces
