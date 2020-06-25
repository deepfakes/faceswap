#!/usr/bin/env python3
""" Alignments handling for Faceswap's Manual Adjustments tool. Handles the conversion of
alignments data to :class:`~lib.faces_detect.DetectedFace` objects, and the update of these faces
when edits are made in the GUI. """

import logging
import os
import tkinter as tk
from copy import deepcopy
from time import sleep
from threading import Lock

import cv2
import imageio
import numpy as np
from tqdm import tqdm

from lib.aligner import Extract as AlignerExtract
from lib.alignments import Alignments
from lib.faces_detect import DetectedFace
from lib.image import SingleFrameLoader
from lib.multithreading import MultiThread

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class DetectedFaces():
    """ Handles the manipulation of :class:`~lib.faces_detect.DetectedFace` objects stored
    in the alignments file. Acts as a parent class for the IO operations (saving and loading from
    and alignments file), the face update operations (when changes are made to alignments in the
    GUI) and the face filters (when a user changes the filter navigation mode.)

    Parameters
    ----------
    tk_globals: :class:`~tools.manual.manual.TkGlobals`
        The tkinter variables that apply to the whole of the GUI
    alignments_path: str
        The full path to the alignments file
    input_location: str
        The location of the input folder of frames or video file
    extractor: :class:`~tools.manual.manual.Aligner`
        The pipeline for passing faces through the aligner and retrieving results
    """
    def __init__(self, tk_globals, alignments_path, input_location, extractor):
        logger.debug("Initializing %s: (tk_globals: %s. alignments_path: %s, input_location: %s "
                     "extractor: %s)", self.__class__.__name__, tk_globals, alignments_path,
                     input_location, extractor)
        self._globals = tk_globals
        self._frame_faces = []
        self._updated_frame_indices = set()
        self._extract_size = min(self._globals.frame_display_dims)

        self._alignments = self._get_alignments(alignments_path, input_location)
        self._extractor = extractor
        self._tk_vars = self._set_tk_vars()
        self._children = dict(io=_DiskIO(self), update=FaceUpdate(self), filter=Filter(self))
        logger.debug("Initialized %s", self.__class__.__name__)

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
        return self._children["filter"]

    @property
    def update(self):
        """ :class:`FaceUpdate`: Handles the adding, removing and updating of
        :class:`~lib.faces_detect.DetectedFace` stored within the alignments file. """
        return self._children["update"]

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

    @property
    def tk_face_count_changed(self):
        """ :class:`tkinter.BooleanVar`: The variable indicating whether a face has been added or
        removed meaning the :class:`FaceViewer` grid redraw needs to be triggered. """
        return self._tk_vars["face_count_changed"]

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
        return self._frame_faces

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
        return frame_index in self._updated_frame_indices

    def load_faces(self):
        """ Load the faces as :class:`~lib.faces_detect.DetectedFace` objects from the alignments
        file. """
        self._children["io"]._load()

    def save(self):
        """ Save the alignments file with the latest edits. """
        self._children["io"]._save()

    def revert_to_saved(self, frame_index):
        """ Revert the frame's alignments to their saved version for the given frame index.

        Parameters
        ----------
        frame_index: int
            The frame that should have their faces reverted to their saved version
        """
        self._children["io"].revert_to_saved(frame_index)

    def save_video_meta_data(self, pts_time, keyframes):
        """ Save video meta data to the alignments file. This is executed if the video meta data
        does not already exist in the alignments file, so the video does not need to be scanned
        on every use of the Manual Tool.

        Parameters
        ----------
        pts_time: list
            A list of presentation timestamps (`float`) in frame index order for every frame in
            the input video
        keyframes: list
            A list of frame indices corresponding to the key frames in the input video.
        """
        if self._globals.is_video:
            self._alignments.save_video_meta_data(pts_time, keyframes)

    def get_thumbnail(self, frame_index, face_index):
        """ Obtain the compressed jpg thumbnail for the given face in the given frame.

        Parameters
        ----------
        frame_index: int
            The frame index that contains the face to return the thumbnail for
        face_index: int
            The face index within the given frame to return the thumbnail for

        Returns
        :class:`numpy.ndarray`
            The encoded jpg thumbnail image
        """
        return self._alignments.thumbnails.get_thumbnail_by_index(frame_index, face_index)

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
            face. Default: ``False``

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
        for name in ("unsaved", "edited", "face_count_changed"):
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
        logger.debug("alignments_path: %s, input_location: %s", alignments_path, input_location)
        if alignments_path:
            folder, filename = os.path.split(alignments_path)
        else:
            filename = "alignments.fsa"
            if self._globals.is_video:
                folder, vid = os.path.split(os.path.splitext(input_location)[0])
                filename = "{}_{}".format(vid, filename)
            else:
                folder = input_location
        retval = Alignments(folder, filename)
        logger.debug("folder: %s, filename: %s, alignments: %s", folder, filename, retval)
        return retval


class _DiskIO():  # pylint:disable=too-few-public-methods
    """ Handles the loading of :class:`~lib.faces_detect.DetectedFaces` from the alignments file
    into :class:`DetectedFaces` and the saving of this data (in the opposite direction) to an
    alignments file.

    Parameters
    ----------
    detected_faces: :class:`DetectedFaces`
        The parent :class:`DetectedFaces` object
    """
    def __init__(self, detected_faces):
        logger.debug("Initializing %s: (detected_faces: %s)",
                     self.__class__.__name__, detected_faces)
        self._alignments = detected_faces._alignments
        self._frame_faces = detected_faces._frame_faces
        self._updated_frame_indices = detected_faces._updated_frame_indices
        self._tk_unsaved = detected_faces.tk_unsaved
        self._tk_edited = detected_faces.tk_edited
        self._globals = detected_faces._globals
        self._sorted_frame_names = sorted(self._alignments.data)
        logger.debug("Initialized %s", self.__class__.__name__)

    def _load(self):
        """ Load the faces from the alignments file, convert to
        :class:`~lib.faces_detect.DetectedFace`. objects and add to :attr:`_frame_faces`. """
        for key in sorted(self._alignments.data):
            this_frame_faces = []
            for item in self._alignments.data[key]["faces"]:
                face = DetectedFace()
                face.from_alignment(item)
                this_frame_faces.append(face)
            self._frame_faces.append(this_frame_faces)

    def _save(self):
        """ Convert updated :class:`~lib.faces_detect.DetectedFace` objects to alignments format
        and save the alignments file. """
        if not self._tk_unsaved.get():
            logger.debug("Alignments not updated. Returning")
            return
        to_save = zip(list(self._updated_frame_indices),
                      np.array(self._frame_faces)[np.array(self._updated_frame_indices)])
        logger.verbose("Saving alignments for %s updated frames", len(to_save))

        for idx, faces in to_save:
            frame = self._sorted_frame_names[idx]
            self._alignments.data[frame]["faces"] = [face.to_alignment() for face in faces]

        self._alignments.backup()
        self._alignments.save()
        self._updated_frame_indices.clear()
        self._tk_unsaved.set(False)

    def revert_to_saved(self, frame_index):
        """ Revert the frame's alignments to their saved version for the given frame index.

        Parameters
        ----------
        frame_index: int
            The frame that should have their faces reverted to their saved version
        """
        if frame_index not in self._updated_frame_indices:
            logger.debug("Alignments not amended. Returning")
            return
        logger.debug("Reverting alignments for frame_index %s", frame_index)
        faces = self._alignments.data[self._sorted_frame_names[frame_index]]["faces"]
        # TODO Add or removed frames
        for detected_face, face in zip(self._frame_faces[frame_index], faces):
            detected_face.from_alignment(face)
        self._updated_frame_indices.remove(frame_index)
        self._tk_edited.set(True)
        self._globals.tk_update.set(True)


class Filter():
    """ Returns stats and frames for filtered frames based on the user selected navigation mode
    filter.

    Parameters
    ----------
    detected_faces: :class:`DetectedFaces`
        The parent :class:`DetectedFaces` object
    """
    def __init__(self, detected_faces):
        logger.debug("Initializing %s: (detected_faces: %s)",
                     self.__class__.__name__, detected_faces)
        self._globals = detected_faces._globals
        self._det_faces = detected_faces
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def count(self):
        """ int: The number of frames that meet the filter criteria returned by
        :attr:`~tools.manual.manual.TkGlobals.filter_mode`. """
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
    def raw_indices(self):
        """ dict: The frame and face indices that meet the current filter criteria for each
        displayed face. """
        frame_indices = []
        face_indices = []
        if self._globals.filter_mode != "No Faces":
            for frame_idx, face_count in enumerate(self._det_faces.face_count_per_index):
                if face_count <= 1 and self._globals.filter_mode == "Multiple Faces":
                    continue
                for face_idx in range(face_count):
                    frame_indices.append(frame_idx)
                    face_indices.append(face_idx)
        logger.trace("frame_indices: %s, face_indices: %s", frame_indices, face_indices)
        retval = dict(frame=frame_indices, face=face_indices)
        return retval

    @property
    def frames_list(self):
        """ list: The list of frame indices that meet the filter criteria returned by
        :attr:`~tools.manual.manual.TkGlobals.filter_mode`. """
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
    :class:`DetectedFaces` when changes are made within the GUI.

    Parameters
    ----------
    detected_faces: :class:`DetectedFaces`
        The parent :class:`DetectedFaces` object
    """
    def __init__(self, detected_faces):
        logger.debug("Initializing %s: (detected_faces: %s)",
                     self.__class__.__name__, detected_faces)
        self._det_faces = detected_faces
        self._globals = detected_faces._globals
        self._frame_faces = detected_faces._frame_faces
        self._updated_frame_indices = detected_faces._updated_frame_indices
        self._tk_unsaved = detected_faces.tk_unsaved
        self._extractor = detected_faces.extractor
        logger.debug("Initialized %s", self.__class__.__name__)

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
    def _tk_face_count_changed(self):
        """ :class:`tkinter.BooleanVar`: The variable indicating whether an edit has occurred
        meaning a GUI redraw needs to be triggered.

        Notes
        -----
        The variable is still a ``None`` when this class is initialized, so referenced explicitly.
        """
        return self._det_faces.tk_face_count_changed

    @property
    def _zoomed_size(self):
        """ int: The size of the face when the editor is in zoomed in mode
        """
        return self._det_faces._zoomed_size  # pylint:disable=protected-access

    def _faces_at_frame_index(self, frame_index):
        """ Checks whether there are already new alignments in :attr:`_alignments`. If not
        then saved alignments are copied to :attr:`_updated_faces` ready for update.

        Parameters
        ----------
        frame_index: int
            The frame index to check whether there are updated alignments available
        """
        self._updated_frame_indices.add(frame_index)
        retval = self._frame_faces[frame_index]
        return retval

    def add(self, frame_index, pnt_x, width, pnt_y, height):
        """ Add a :class:`~lib.faces_detect.DetectedFace` object to the current frame with the
        given dimensions.

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
        faces = self._faces_at_frame_index(frame_index)
        faces.append(DetectedFace())
        face_index = len(faces) - 1
        self.bounding_box(frame_index, face_index, pnt_x, width, pnt_y, height, aligner="cv2-dnn")
        self._tk_face_count_changed.set(True)

    def delete(self, frame_index, face_index):
        """ Delete the :class:`~lib.faces_detect.DetectedFace` object for the given frame and face
        indices.

        Parameters
        ----------
        frame_index: int
            The frame that the face is being set for
        face_index: int
            The face index within the frame
        """
        logger.debug("Deleting face at frame index: %s face index: %s", frame_index, face_index)
        faces = self._faces_at_frame_index(frame_index)
        del faces[face_index]
        self._tk_face_count_changed.set(True)
        self._globals.tk_update.set(True)

    def bounding_box(self, frame_index, face_index, pnt_x, width, pnt_y, height, aligner="FAN"):
        """ Update the bounding box for the :class:`~lib.faces_detect.DetectedFace` object at the
        given frame and face indices, with the given dimensions and update the 68 point landmarks
        from the :class:`~tools.manual.manual.Aligner` for the updated bounding box.

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
        aligner: ["cv2-dnn", "FAN"], optional
            The aligner to use to generate the landmarks. Default: "FAN"
        """
        logger.trace("frame_index: %s, face_index %s, pnt_x %s, width %s, pnt_y %s, height %s, "
                     "aligner: %s", frame_index, face_index, pnt_x, width, pnt_y, height, aligner)
        face = self._faces_at_frame_index(frame_index)[face_index]
        face.x = pnt_x
        face.w = width
        face.y = pnt_y
        face.h = height
        face.landmarks_xy = self._extractor.get_landmarks(frame_index, face_index, aligner)
        self._tk_edited.set(True)
        # TODO Link this in to edited
        self._globals.tk_update.set(True)

    def landmark(self, frame_index, face_index, landmark_index, shift_x, shift_y, is_zoomed):
        """ Shift a single landmark point for the :class:`~lib.faces_detect.DetectedFace` object
        at the given frame and face indices by the given x and y values.

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
            ``True`` if landmarks are being adjusted on a zoomed image otherwise ``False``
        """
        face = self._faces_at_frame_index(frame_index)[face_index]
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
        face.mask = self._extractor.get_masks(frame_index, face_index)
        self._tk_edited.set(True)
        self._globals.tk_update.set(True)

    def landmarks(self, frame_index, face_index, shift_x, shift_y):
        """ Shift all of the landmarks and bounding box for the
        :class:`~lib.faces_detect.DetectedFace` object at the given frame and face indices by the
        given x and y values and update the masks.

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
        face = self._faces_at_frame_index(frame_index)[face_index]
        face.x += shift_x
        face.y += shift_y
        face.landmarks_xy += (shift_x, shift_y)
        face.mask = self._extractor.get_masks(frame_index, face_index)
        self._tk_edited.set(True)
        self._globals.tk_update.set(True)

    def landmarks_rotate(self, frame_index, face_index, angle, center):
        """ Rotate the landmarks on an Extract Box rotate for the
        :class:`~lib.faces_detect.DetectedFace` object at the given frame and face indices for the
        given angle from the given center point.

        Parameters
        ----------
        frame_index: int
            The frame that the face is being set for
        face_index: int
            The face index within the frame
        angle: :class:`numpy.ndarray`
            The angle, in radians to rotate the points by
        center: :class:`numpy.ndarray`
            The center point of the Landmark's Extract Box
        """
        face = self._faces_at_frame_index(frame_index)[face_index]
        rot_mat = cv2.getRotationMatrix2D(tuple(center), angle, 1.)
        face.landmarks_xy = cv2.transform(np.expand_dims(face.landmarks_xy, axis=0),
                                          rot_mat).squeeze()
        face.mask = self._extractor.get_masks(frame_index, face_index)
        self._tk_edited.set(True)
        self._globals.tk_update.set(True)

    def landmarks_scale(self, frame_index, face_index, scale, center):
        """ Scale the landmarks on an Extract Box resize for the
        :class:`~lib.faces_detect.DetectedFace` object at the given frame and face indices from the
        given center point.

        Parameters
        ----------
        frame_index: int
            The frame that the face is being set for
        face_index: int
            The face index within the frame
        scale: float
            The amount to scale the landmarks by
        center: :class:`numpy.ndarray`
            The center point of the Landmark's Extract Box
        """
        face = self._faces_at_frame_index(frame_index)[face_index]
        face.landmarks_xy = ((face.landmarks_xy - center) * scale) + center
        face.mask = self._extractor.get_masks(frame_index, face_index)
        self._tk_edited.set(True)
        self._globals.tk_update.set(True)

    def mask(self, frame_index, face_index, mask, mask_type):
        """ Update the mask on an edit for the :class:`~lib.faces_detect.DetectedFace` object at
        the given frame and face indices, for the given mask and mask type.

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
        face = self._faces_at_frame_index(frame_index)[face_index]
        face.mask[mask_type].replace_mask(mask)
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
        faces = self._faces_at_frame_index(frame_index)
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
        faces.extend(deepcopy(self._faces_at_frame_index(idx)))
        self._tk_edited.set(True)
        self._globals.tk_update.set(True)


class ThumbsCreator():
    """ Background loader to generate thumbnails for the alignments file. Generates low resolution
    thumbnails in parallel threads for faster processing.

    Parameters
    ----------
    detected_faces: :class:`~tool.manual.faces.DetectedFaces`
        The :class:`~lib.faces_detect.DetectedFace` objects for this video
    input_location: str
        The location of the input folder of frames or video file
    """
    def __init__(self, detected_faces, input_location):
        logger.debug("Initializing %s: (detected_faces: %s, input_location: %s)",
                     self.__class__.__name__, detected_faces, input_location)
        self._size = 96
        self._jpeg_quality = 75
        self._pbar = None
        self._lock = Lock()
        self._location = input_location
        self._key_frames = detected_faces.video_meta_data.get("keyframes", None)
        self._pts_times = detected_faces.video_meta_data.get("pts_time", None)
        self._alignments = detected_faces._alignments
        self._frame_faces = detected_faces._frame_faces

        self._is_video = self._key_frames is not None and self._pts_times is not None
        self._num_threads = os.cpu_count() - 2
        if self._is_video:
            self._num_threads = min(self._num_threads, len(self._key_frames))
        else:
            self._num_threads = max(self._num_threads, 32)
        self._threads = []
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def has_thumbs(self):
        """ bool: ``True`` if the underlying alignments file holds thumbnail images
        otherwise ``False``. """
        return self._alignments.thumbnails.has_thumbails

    def generate_cache(self):
        """ Extract the face thumbnails from a video or folder of images into the
        alignments file. """
        self._pbar = tqdm(desc="Caching Thumbails", leave=False, total=len(self._frame_faces))
        if self._is_video:
            self._launch_video()
        else:
            self._launch_folder()
        while True:
            self._check_and_raise_error()
            if all(not thread.is_alive() for thread in self._threads):
                break
            sleep(1)
        self._join_threads()
        self._pbar.close()
        self._alignments.save()

    # << PRIVATE METHODS >> #
    def _check_and_raise_error(self):
        """ Monitor the loading threads for errors and raise if any occur. """
        for thread in self._threads:
            thread.check_and_raise_error()

    def _join_threads(self):
        """ Join the loading threads """
        logger.debug("Joining face viewer loading threads")
        for thread in self._threads:
            thread.join()

    def _launch_video(self):
        """ Launch multiple :class:`lib.multithreading.MultiThread` objects to load faces from
        a video file.

        Splits the video into segments and passes each of these segments to separate background
        threads for some speed up.
        """
        key_frame_split = len(self._key_frames) // self._num_threads
        for idx in range(self._num_threads):
            is_final = idx == self._num_threads - 1
            start_idx = idx * key_frame_split
            keyframe_idx = len(self._key_frames) - 1 if is_final else start_idx + key_frame_split
            end_idx = self._key_frames[keyframe_idx]
            start_pts = self._pts_times[self._key_frames[start_idx]]
            end_pts = False if idx + 1 == self._num_threads else self._pts_times[end_idx]
            starting_index = self._pts_times.index(start_pts)
            if end_pts:
                segment_count = len(self._pts_times[self._key_frames[start_idx]:end_idx])
            else:
                segment_count = len(self._pts_times[self._key_frames[start_idx]:])
            logger.debug("thread index: %s, start_idx: %s, end_idx: %s, start_pts: %s, "
                         "end_pts: %s, starting_index: %s, segment_count: %s", idx, start_idx,
                         end_idx, start_pts, end_pts, starting_index, segment_count)
            thread = MultiThread(self._load_from_video,
                                 start_pts,
                                 end_pts,
                                 starting_index,
                                 segment_count)
            thread.start()
            self._threads.append(thread)

    def _launch_folder(self):
        """ Launch :class:`lib.multithreading.MultiThread` to retrieve faces from a
        folder of images.

        Goes through the file list one at a time, passing each file to a separate background
        thread for some speed up.
        """
        reader = SingleFrameLoader(self._location)
        num_threads = min(reader.count, self._num_threads)
        frame_split = reader.count // self._num_threads
        logger.debug("total images: %s, num_threads: %s, frames_per_thread: %s",
                     reader.count, num_threads, frame_split)
        for idx in range(num_threads):
            is_final = idx == num_threads - 1
            start_idx = idx * frame_split
            end_idx = reader.count if is_final else start_idx + frame_split
            thread = MultiThread(self._load_from_folder, reader, start_idx, end_idx)
            thread.start()
            self._threads.append(thread)

    def _load_from_video(self, pts_start, pts_end, start_index, segment_count):
        """ Loads faces from video for the given segment of the source video.

        Each segment of the video is extracted from in a different background thread.

        Parameters
        ----------
        pts_start: float
            The start time to cut the segment out of the video
        pts_end: float
            The end time to cut the segment out of the video
        start_index: int
            The frame index that this segment starts from. Used for calculating the actual frame
            index of each frame extracted
        segment_count: int
            The number of frames that appear in this segment. Used for ending early in case more
            frames come out of the segment than should appear (sometimes more frames are picked up
            at the end of the segment, so these are discarded)
        """
        logger.debug("pts_start: %s, pts_end: %s, start_index: %s, segment_count: %s",
                     pts_start, pts_end, start_index, segment_count)
        reader = self._get_reader(pts_start, pts_end)
        idx = 0
        vidname = os.path.splitext(os.path.basename(self._location))[0]
        for idx, frame in enumerate(reader):
            frame_idx = idx + start_index
            filename = "{}_{:06d}.png".format(vidname, frame_idx + 1)
            self._set_thumbail(filename, frame[..., ::-1], frame_idx)
            if idx == segment_count - 1:
                # Sometimes extra frames are picked up at the end of a segment, so stop
                # processing when segment frame count has been hit.
                break
        reader.close()
        logger.debug("Segment complete: (starting_frame_index: %s, processed_count: %s)",
                     start_index, idx)

    def _get_reader(self, pts_start, pts_end):
        """ Get an imageio iterator for this thread's segment.

        Parameters
        ----------
        pts_start: float
            The start time to cut the segment out of the video
        pts_end: float
            The end time to cut the segment out of the video

        Returns
        -------
        :class:`imageio.Reader`
            A reader iterator for the requested segment of video
        """
        input_params = ["-ss", str(pts_start)]
        if pts_end:
            input_params.extend(["-to", str(pts_end)])
        logger.debug("pts_start: %s, pts_end: %s, input_params: %s",
                     pts_start, pts_end, input_params)
        return imageio.get_reader(self._location, "ffmpeg", input_params=input_params)

    def _load_from_folder(self, reader, start_index, end_index):
        """ Loads faces from the given range of frame indices from a folder of images.

        Each frame range is extracted in a different background thread.

        Parameters
        ----------
        reader: :class:`lib.image.SingleFrameLoader`
            The reader that is used to retrieve the requested frame
        start_index: int
            The starting frame index for the images to extract faces from
        end_index: int
            The end frame index for the images to extract faces from
        """
        logger.debug("reader: %s, start_index: %s, end_index: %s",
                     reader, start_index, end_index)
        for frame_index in range(start_index, end_index):
            filename, frame = reader.image_from_index(frame_index)
            self._set_thumbail(filename, frame, frame_index)
        logger.debug("Segment complete: (start_index: %s, processed_count: %s)",
                     start_index, end_index - start_index)

    def _set_thumbail(self, filename, frame, frame_index):
        """ Extracts the faces from the frame and adds to alignments file

        Parameters
        ----------
        filename: str
            The filename of the frame within the alignments file
        frame: :class:`numpy.ndarray`
            The frame that contains the faces
        frame_index: int
            The frame index of this frame in the :attr:`_frame_faces`
        """
        for face_idx, face in enumerate(self._frame_faces[frame_index]):
            face.load_aligned(frame, size=self._size, force=True)
            jpg = cv2.imencode(".jpg",
                               face.aligned_face,
                               [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])[1]
            self._alignments.thumbnails.add_thumbnail(filename, face_idx, jpg)
            face.aligned["face"] = None
        with self._lock:
            self._pbar.update(1)
