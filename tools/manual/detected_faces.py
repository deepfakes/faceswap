#!/usr/bin/env python3
""" Alignments handling for the manual adjustments tool """
import logging
import os
import tkinter as tk
from copy import deepcopy

import cv2
import numpy as np

from lib.aligner import Extract as AlignerExtract
from lib.alignments import Alignments
from lib.faces_detect import DetectedFace

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class DetectedFaces():
    """ Handles the manipulation of :class:`~lib.faces_detect.DetectedFace` objects stored
    in the alignments file.

    Parameters
    ----------
    tk_globals: :class:`TkGlobals`
        The tkinter variables that apply to the whole of the GUI
    alignments_path: str
        The full path to the alignments file
    input_location: str
        The location of the input folder of frames or video file
    extractor: :class:`~tools.manual.manual.Aligner`
        The pipeline for passing faces through the aligner and retrieving results
    is_video: bool
        ``True`` if the :attr:`input_location` is a video file otherwise ``False``
    """
    def __init__(self, tk_globals, alignments_path, input_location, extractor, is_video):
        logger.debug("Initializing: %s: (tk_globals: %s. alignments_path: %s, input_location: %s "
                     "extractor: %s, is_video: %s)", self.__class__.__name__, tk_globals,
                     alignments_path, input_location, extractor, is_video)
        self._globals = tk_globals
        self._saved_faces = []
        self._updated_faces = []
        self._extract_size = None  # Updated to correct size in in :func:`_load`
        self._is_video = is_video

        self._alignments = self._get_alignments(alignments_path, input_location)
        self._extractor = extractor
        self._tk_vars = self._set_tk_vars()
        self._io = DiskIO(self)
        self._update = FaceUpdate(self)
        self._filter = Filter(self)
        logger.debug("Initialized: %s", self.__class__.__name__)

    # <<<< PUBLIC PROPERTIES >>>> #
    # << SUBCLASSES >> #
    @property
    def extractor(self):
        """ :class:`~tools.manual.manual.Aligner`: The pipeline for passing faces through the
        aligner and retrieving results. """
        return self._extractor

    @property
    def filter(self):
        """ :class:`Filter`: Handles returning of faces and stats based on the current user set
        navigation mode filter. """
        return self._filter

    @property
    def update(self):
        """ :class:`FacFaceUpdate`: Handles the adding, removing and updating of
        :class:`~lib.faces_detect.DetectedFace` stored within the alignments file. """
        return self._update

    # << TKINTER VARIABLES >> #
    @property
    def tk_unsaved(self):
        """ :class:`tkinter.BooleanVar`: The variable indicating whether the alignments have been
        updated since the last save. """
        return self._tk_vars["unsaved"]

    @property
    def tk_edited(self):
        """ :class:`tkinter.BooleanVar`: The variable indicating whether an edit has occurred
        meaning a GUI redraw needs to be triggered. """
        return self._tk_vars["edited"]

    # << STATISTICS >> #
    @property
    def available_masks(self):
        """ dict: The mask type names stored in the alignments; type as key with the number
        of faces which possess the mask type as value. """
        return self._alignments.mask_summary

    @property
    def current_faces(self):
        """ list: The most up to date full list of :class:`~lib.faces_detect.DetectedFace`
        objects. """
        return [saved_faces if updated_faces is None else updated_faces
                for saved_faces, updated_faces in zip(self._saved_faces, self._updated_faces)]

    @property
    def video_meta_data(self):
        """ dict: The frame meta data stored in the alignments file. If data does not exist in the
        alignments file then ``None`` is returned for each Key """
        return self._alignments.video_meta_data

    @property
    def face_count_per_index(self):
        """ list: Count of faces for each frame. List is in frame index order.

        The list needs to be calculated on the fly as the number of faces in a frame
        can change based on user actions. """
        return [len(faces) for faces in self.current_faces]

    # <<<< PUBLIC METHODS >>>> #
    def is_frame_updated(self, frame_index):
        """ bool: ``True`` if the given frame index has updated faces within it otherwise
        ``False`` """
        return self._updated_faces[frame_index] is not None

    def load_faces(self, frames):
        """ Load the faces as :class:`~lib.faces_detect.DetectedFace` from the alignments file.

        Set the extract size to be the zoomed face face size. This is the largest a face will be
        extracted at, so every other use can be scaled down from this value
        Load the faces.

        Parameters
        ----------
        frames: :class:`~tools.manual.media.FrameNavigation`
            The frames navigation object for the Manual Tool
        """
        self._extract_size = min(frames.display_dims)
        self._io.load()

    def save(self):
        """ Save the alignments file with the latest edits. """
        self._io._save()  # pylint:disable=protected-access

    def enable_save(self):
        """ Enable saving of alignments file. Triggered when the
        :class:`tools.manual.manual.FacesViewer` has finished loading.
        """
        self._io._enable_save()  # pylint:disable=protected-access

    def save_video_meta_data(self, pts_time, keyframes):
        """ Save video meta data to the alignments file.

        Parameters
        ----------
        pts_time: list
            A list of presentation timestamps (`float`) in frame index order for every frame in
            the input video
        keyframes: list
            A list of frame indices corresponding to the key frames in the input video.
        """
        if self._is_video:
            self._alignments.save_video_meta_data(pts_time, keyframes)

    def get_face_at_index(self, frame_index, face_index, image, size,
                          with_landmarks=False, with_mask=False):
        """ Return an aligned face for the given frame and face index sized at the given size.

        Optionally also return aligned landmarks and mask objects for the requested face.l

        Parameters
        ----------
        frame_index: int
            The frame that the required face exists in
        face_index: int
            The face index within the frame to retrieve the face for
        image: :class:`numpy.ndarray`
            The original frame that contains the face to be extracted
        size: int
            The required pixel size of the aligned face. NB The default size is set for a zoomed
            display frame image. Rather than resize the underlying Detected Face object, the
            returned results are adjusted for the requested size
        with_landmarks: bool, optional
            Set to `True` if the aligned landmarks should be returned with the aligned face.
            Default: ``False``
        with_mask: bool
            Set to `True` if the Detected Face's Mask object should be returned with the aligned
            face. Default: ``True``

        Returns
        -------
        :class:`numpy.ndarray` or tuple
            If :attr:`with_landmarks` and :attr:`with_mask` are both set to ``False`` then just the
            aligned face will be returned.
            If any of these attributes are set to ``True`` then a tuple will be returned with the
            aligned face in position 0 and the requested additional data populated in the following
            order (`aligned face`, `aligned landmarks`, `mask objects`)
        """
        logger.trace("frame_index: %s, face_index: %s, image: %s, size: %s, with_landmarks: %s, "
                     "with_mask: %s", frame_index, face_index, image.shape, size, with_landmarks,
                     with_mask)
        face = self.current_faces[frame_index][face_index]
        resize = self._extract_size != size
        logger.trace("Requires resize: %s", resize)
        face.load_aligned(image, size=self._extract_size, force=True)

        retval = cv2.resize(face.aligned_face,
                            (size, size)) if resize else face.aligned_face.copy()
        retval = [retval] if with_landmarks or with_mask else retval
        face.aligned["face"] = None
        if with_landmarks:
            retval.append(face.aligned_landmarks * (size / self._extract_size)
                          if resize else face.aligned_landmarks)
        if with_mask:
            retval.append(face.mask)
        logger.trace("returning: %s", [item.shape if isinstance(item, np.ndarray) else item
                                       for item in retval])
        return retval

    # <<<< PRIVATE METHODS >>> #
    # << INIT >> #
    @staticmethod
    def _set_tk_vars():
        """ Set the required tkinter variables.

        The alignments specific `unsaved` and `edited` are set here.
        The global variables are added into the dictionary with `None` as value, so the
        objects exist. Their actual variables are populated during :func:`load_faces`.

        Returns
        -------
        dict
            The internal variable name as key with the tkinter variable as value
        """
        retval = dict()
        for name in ("unsaved", "edited"):
            var = tk.BooleanVar()
            var.set(False)
            retval[name] = var
        logger.debug(retval)
        return retval

    def _get_alignments(self, alignments_path, input_location):
        """ Get the :class:`~lib.alignments.Alignments` object for the given location.

        Parameters
        ----------
        alignments_path: str
            Full path to the alignments file. If empty string is passed then location is calculated
            from the source folder
        input_location: str
            The location of the input folder of frames or video file

        Returns
        -------
        :class:`~lib.alignments.Alignments`
            The alignments object for the given input location
        """
        if alignments_path:
            folder, filename = os.path.split(alignments_path)
        else:
            filename = "alignments.fsa"
            if self._is_video:
                folder, vid = os.path.split(os.path.splitext(input_location)[0])
                filename = "{}_{}".format(vid, filename)
            else:
                folder = input_location
        return Alignments(folder, filename)


class DiskIO():  # pylint:disable=too-few-public-methods
    """ Handles the loading of :class:`~lib.faces_detect.DetectedFaces` from the alignments file
    into :class:`DetectedFaces` and the saving of this data (in the opposite direction) to an
    alignments file.

    Parameters
    ----------
    detected_faces: :class:`DetectedFaces`
        The parent :class:`DetectedFaces` object
    """
    def __init__(self, detected_faces):
        self._alignments = detected_faces._alignments
        self._saved_faces = detected_faces._saved_faces
        self._updated_faces = detected_faces._updated_faces
        self._tk_unsaved = detected_faces.tk_unsaved
        self._sorted_frame_names = sorted(self._alignments.data)
        self._save_enabled = False

    def load(self):
        """ Load the faces from the alignments file, convert to
        :class:`~lib.faces_detect.DetectedFace`. objects and add to :attr:`_saved_faces`. """
        for key in sorted(self._alignments.data):
            this_frame_faces = []
            for item in self._alignments.data[key]["faces"]:
                face = DetectedFace()
                face.from_alignment(item)
                this_frame_faces.append(face)
            self._saved_faces.append(this_frame_faces)
        self._updated_faces.extend([None for _ in range(len(self._saved_faces))])

    def _save(self):
        """ Convert updated :class:`~lib.faces_detect.DetectedFace` objects to alignments format
        and save the alignments file. """
        if not self._tk_unsaved.get():
            logger.debug("Alignments not updated. Returning")
            return
        if not self._save_enabled:
            tk.messagebox.showinfo(title="Save Alignments...",
                                   message="Please wait for faces to completely load before "
                                           "saving the alignments file.")
            return
        to_save = [(idx, faces) for idx, faces in enumerate(self._updated_faces)
                   if faces is not None]
        logger.verbose("Saving alignments for %s updated frames", len(to_save))

        save_count = 0
        for idx, faces in to_save:
            frame = self._sorted_frame_names[idx]
            self._alignments.data[frame]["faces"] = [face.to_alignment() for face in faces]
            self._saved_faces[idx] = faces
            self._updated_faces[idx] = None
            save_count += 1
        self._alignments.backup()
        self._alignments.save()
        self._tk_unsaved.set(False)

    def _enable_save(self):
        """ Enable saving of alignments file. Triggered when the
        :class:`tools.manual.manual.FacesViewer` has completed loading. """
        self._save_enabled = True


class Filter():
    """ Returns stats and Faces for filtered frames based on the user selected navigation mode
    filter.

    Parameters
    ----------
    detected_faces: :class:`DetectedFaces`
        The parent :class:`DetectedFaces` object
    """
    def __init__(self, detected_faces):
        self._globals = detected_faces._globals
        self._det_faces = detected_faces

    @property
    def count(self):
        """ int: The number of frames that meet the filter criteria returned by
        :attr:`_globals.filter_mode`. """
        face_count_per_index = self._det_faces.face_count_per_index
        if self._globals.filter_mode == "No Faces":
            retval = sum(1 for fcount in face_count_per_index if fcount == 0)
        elif self._globals.filter_mode == "Has Face(s)":
            retval = sum(1 for fcount in face_count_per_index if fcount != 0)
        elif self._globals.filter_mode == "Multiple Faces":
            retval = sum(1 for fcount in face_count_per_index if fcount > 1)
        else:
            retval = len(face_count_per_index)
        logger.trace("filter mode: %s, frame count: %s", self._globals.filter_mode, retval)
        return retval

    @property
    def frames_list(self):
        """ list: The list of frame indices that meet the filter criteria returned by
        :attr:`_globals.filter_mode`. """
        face_count_per_index = self._det_faces.face_count_per_index
        if self._globals.filter_mode == "No Faces":
            retval = [idx for idx, count in enumerate(face_count_per_index) if count == 0]
        elif self._globals.filter_mode == "Multiple Faces":
            retval = [idx for idx, count in enumerate(face_count_per_index) if count > 1]
        elif self._globals.filter_mode == "Has Face(s)":
            retval = [idx for idx, count in enumerate(face_count_per_index) if count != 0]
        else:
            retval = range(len(face_count_per_index))
        logger.trace("filter mode: %s, number_frames: %s", self._globals.filter_mode, len(retval))
        return retval


class FaceUpdate():
    """ Perform updates on :class:`~lib.faces_detect.DetectedFace` objects stored in
    :class:`DetectedFaces`.

    Parameters
    ----------
    detected_faces: :class:`DetectedFaces`
        The parent :class:`DetectedFaces` object
    """
    def __init__(self, detected_faces):
        self._det_faces = detected_faces
        self._globals = detected_faces._globals
        self._saved_faces = detected_faces._saved_faces
        self._updated_faces = detected_faces._updated_faces
        self._tk_unsaved = detected_faces.tk_unsaved
        self._extractor = detected_faces.extractor
        self._last_updated_face = None

    @property
    def _tk_edited(self):
        """ :class:`tkinter.BooleanVar`: The variable indicating whether an edit has occurred
        meaning a GUI redraw needs to be triggered.

        Notes
        -----
        The variable is still a ``None`` when this class is initialized, so referenced explicitly.
        """
        return self._det_faces.tk_edited

    @property
    def _zoomed_size(self):
        """ int: The size of the face when the editor is in zoomed in mode
        """
        return self._det_faces._zoomed_size  # pylint:disable=protected-access

    @property
    def last_updated_face(self):
        """ tuple: (`frame index`, `face index`) of the last face to be updated.

        This attribute is populated after any update, just before :attr:_`tk_edited`
        is triggered, so that any process picking up the update knows which face to update.
        """
        return self._last_updated_face

    def _current_faces_at_index(self, frame_index):
        """ Checks whether there are already new alignments in :attr:`_alignments`. If not
        then saved alignments are copied to :attr:`_updated_faces` ready for update.

        Parameters
        ----------
        frame_index: int
            The frame index to check whether there are updated alignments available
        """
        if self._updated_faces[frame_index] is None:
            retval = [deepcopy(face) for face in self._saved_faces[frame_index]]
            self._updated_faces[frame_index] = retval
            if not self._tk_unsaved.get():
                self._tk_unsaved.set(True)
        else:
            retval = self._updated_faces[frame_index]
        return retval

    def add(self, frame_index, pnt_x, width, pnt_y, height):
        """ Add a :class:`DetectedFace` object to the current frame with the given dimensions.

        Parameters
        ----------
        frame_index: int
            The frame that the face is being set for
        pnt_x: int
            The left point of the bounding box
        width: int
            The width of the bounding box
        pnt_y: int
            The top point of the bounding box
        height: int
            The height of the bounding box
        """
        faces = self._current_faces_at_index(frame_index)
        faces.append(DetectedFace())
        face_index = len(faces) - 1
        self.bounding_box(frame_index, face_index, pnt_x, width, pnt_y, height, aligner="cv2-dnn")

    def delete(self, frame_index, face_index):
        """ Delete the :class:`DetectedFace` object for the given frame and face indices.

        Parameters
        ----------
        frame_index: int
            The frame that the face is being set for
        face_index: int
            The face index within the frame
        """
        logger.debug("Deleting face at frame index: %s face index: %s", frame_index, face_index)
        faces = self._current_faces_at_index(frame_index)
        del faces[face_index]
        self._last_updated_face = (frame_index, face_index)
        self._tk_edited.set(True)
        self._globals.tk_update.set(True)

    def bounding_box(self, frame_index, face_index, pnt_x, width, pnt_y, height, aligner="FAN"):
        """ Update the bounding box for the :class:`DetectedFace` object at the given frame and
        face indices, with the given dimensions.

        Parameters
        ----------
        frame_index: int
            The frame that the face is being set for
        face_index: int
            The face index within the frame
        pnt_x: int
            The left point of the bounding box
        width: int
            The width of the bounding box
        pnt_y: int
            The top point of the bounding box
        height: int
            The height of the bounding box
        aligner: ["cv2-dnn", "FAN], optional
            The aligner to use to generate the landmarks. Default: "FAN"
        """
        logger.trace("frame_index: %s, face_index %s, pnt_x %s, width %s, pnt_y %s, height %s, "
                     "aligner: %s", frame_index, face_index, pnt_x, width, pnt_y, height, aligner)
        face = self._current_faces_at_index(frame_index)[face_index]
        face.x = pnt_x
        face.w = width
        face.y = pnt_y
        face.h = height
        face.landmarks_xy = self._extractor.get_landmarks(frame_index, face_index, aligner)
        self._last_updated_face = (frame_index, face_index)
        self._tk_edited.set(True)
        # TODO Link this in to edited
        self._globals.tk_update.set(True)

    def landmark(self, frame_index, face_index, landmark_index, shift_x, shift_y, is_zoomed):
        """ Shift a single landmark point for the :class:`DetectedFace` object at the given frame
        and face indices by the given x and y values.

        Parameters
        ----------
        frame_index: int
            The frame that the face is being set for
        face_index: int
            The face index within the frame
        landmark_index: int
            The landmark index to shift
        shift_x: int
            The amount to shift the landmark by along the x axis
        shift_y: int
            The amount to shift the landmark by along the y axis
        is_zoomed: bool
            ``` True if landmarks are being adjusted on a zoomed image otherwise ``False``
        """
        face = self._current_faces_at_index(frame_index)[face_index]
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
        self._last_updated_face = (frame_index, face_index)
        self._tk_edited.set(True)
        self._globals.tk_update.set(True)

    def landmarks(self, frame_index, face_index, shift_x, shift_y):
        """ Shift all of the landmarks and bounding box for the :class:`DetectedFace` object at
        the given frame and face indices by the given x and y values.

        Parameters
        ----------
        frame_index: int
            The frame that the face is being set for
        face_index: int
            The face index within the frame
        shift_x: int
            The amount to shift the landmarks by along the x axis
        shift_y: int
            The amount to shift the landmarks by along the y axis

        Notes
        -----
        Whilst the bounding box does not need to be shifted, it is anyway, to ensure that it is
        aligned with the newly adjusted landmarks.
        """
        face = self._current_faces_at_index(frame_index)[face_index]
        face.x += shift_x
        face.y += shift_y
        face.landmarks_xy += (shift_x, shift_y)
        self._last_updated_face = (frame_index, face_index)
        self._tk_edited.set(True)
        self._globals.tk_update.set(True)

    def mask(self, frame_index, face_index, mask, mask_type):
        """ Update the mask on an edit for the :class:`DetectedFace` object at the given frame and
        face indices, for the given mask and mask type.

        Parameters
        ----------
        frame_index: int
            The frame that the face is being set for
        face_index: int
            The face index within the frame
        mask: class:`numpy.ndarray`:
            The mask to replace
        mask_type: str
            The name of the mask that is to be replaced
        """
        face = self._current_faces_at_index(frame_index)[face_index]
        face.mask[mask_type].replace_mask(mask)
        self._last_updated_face = (frame_index, face_index)
        self._tk_edited.set(True)
        self._globals.tk_update.set(True)

    def copy(self, frame_index, direction):
        """ Copy the alignments from the previous or next frame that has alignments
        to the current frame.

        Parameters
        ----------
        frame_index: int
            The frame that the needs to have alignments copied to it
        direction: ["prev", "next"]
            Whether to copy alignments from the previous frame with alignments, or the next
            frame with alignments
        """
        logger.debug("frame: %s, direction: %s", frame_index, direction)
        faces = self._current_faces_at_index(frame_index)
        frames_with_faces = [idx for idx, faces in enumerate(self._det_faces.current_faces)
                             if len(faces) > 0]
        if direction == "prev":
            idx = next((idx for idx in reversed(frames_with_faces)
                        if idx < frame_index), None)
        else:
            idx = next((idx for idx in frames_with_faces
                        if idx > frame_index), None)
        if idx is None:
            # No previous/next frame available
            return
        logger.debug("Copying alignments from frame %s to frame: %s", idx, frame_index)
        faces.extend(deepcopy(self._current_faces_at_index(idx)))
        face_index = len(self._det_faces.current_faces[frame_index]) - 1
        self._last_updated_face = (frame_index, face_index)
        self._tk_edited.set(True)
        self._globals.tk_update.set(True)

    def revert_to_saved(self, frame_index):
        """ Revert the frame's alignments to their saved version for the given frame index.

        Parameters
        ----------
        frame_index: int
            The frame that should have their faces reverted to their saved version

        """
        if self._updated_faces[frame_index] is None:
            logger.debug("Alignments not amended. Returning")
            return
        logger.debug("Reverting alignments for frame_index %s", frame_index)
        self._updated_faces[frame_index] = None
        self._last_updated_face = (frame_index, -1)
        self._tk_edited.set(True)
        self._globals.tk_update.set(True)
