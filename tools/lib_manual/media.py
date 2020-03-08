#!/usr/bin/env python3
""" Media objects for the manual adjustments tool """
import logging
import bisect
import os
import tkinter as tk
from copy import deepcopy

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
    def __init__(self, frames_location, scaling_factor, video_meta_data):
        logger.debug("Initializing %s: (frames_location: '%s', scaling_factor: %s, "
                     "video_meta_data: %s)", self.__class__.__name__, frames_location,
                     scaling_factor, video_meta_data)
        self._video_meta_data = video_meta_data
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
    def video_meta_data(self):
        """ dict: The pts_time and key frames for the loader. """
        return self._loader.video_meta_data

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
                             self._video_meta_data,
                             thread_count=1,
                             name="{}.init_frames".format(self.__class__.__name__))
        thread.start()
        return thread

    def _load_images(self, frames_location, video_meta_data):
        """ Load the images in a background thread. """
        self._loader = ImagesLoader(frames_location,
                                    queue_size=1,
                                    single_frame_reader=True,
                                    video_meta_data=video_meta_data)

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
    """
    def __init__(self, alignments_path, extractor, input_location, is_video):
        logger.debug("Initializing %s: (alignments_path: '%s', extractor: %s, input_location: %s, "
                     "is_video: %s)", self.__class__.__name__, alignments_path, extractor,
                     input_location, is_video)
        self._frames = None
        self._remove_idx = None
        self._is_video = is_video

        self._alignments_file = None
        self._mask_names = None
        self._alignments = None
        self._get_alignments_file(alignments_path, input_location)

        self._face_size = None  # Set in load_faces
        self._alignments = dict()  # Populated in load_faces

        self._tk_unsaved = tk.BooleanVar()
        self._tk_unsaved.set(False)
        self._tk_edited = tk.BooleanVar()
        self._tk_edited.set(False)

        self._face_index = 0
        self._face_count_modified = False
        self._extractor = extractor
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
    def video_meta_data(self):
        """ dict: The frame meta data stored in the alignments file. """
        retval = dict(pts_time=None, keyframes=None)
        if not self._is_video:
            return retval
        pts_time = []
        keyframes = []
        for idx, key in enumerate(sorted(self._alignments_file.data)):
            if "video_meta" not in self._alignments_file.data[key]:
                return retval
            meta = self._alignments_file.data[key]["video_meta"]
            pts_time.append(meta["pts_time"])
            if meta["keyframe"]:
                keyframes.append(idx)
        retval = dict(pts_time=pts_time, keyframes=keyframes)
        return retval

    @property
    def latest_alignments(self):
        """ dict: The filename as key, and either the modified alignments as values (if they exist)
        or the saved alignments """
        return {key: val.get("new", val["saved"]) for key, val in self._alignments.items()}

    @property
    def current_faces(self):
        """ list: list of the current :class:`lib.faces_detect.DetectedFace` objects. Returns
        modified alignments if they are modified, otherwise original saved alignments. """
        # TODO use get and return a default for when the frame don't exist
        return self.latest_alignments[self._frames.current_meta_data["filename"]]

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
    def face_count_per_index(self):
        """ list: Count of faces for each frame. List is in frame index order.

        The list needs to be calculated on the fly as the number of faces in a frame
        can change based on user actions. """
        alignments = self.latest_alignments
        return [len(alignments[key]) for key in sorted(alignments)]

    @property
    def current_face(self):
        """ :class:`lib.faces_detect.DetectedFace` The currently selected face """
        retval = None if not self.current_faces else self.current_faces[self._face_index]
        return retval

    @property
    def _frames_with_faces(self):
        """ list: A list of frame numbers that contain faces. """
        alignments = self.latest_alignments
        return [key for key in sorted(alignments) if len(alignments[key]) != 0]

    @property
    def with_face_count(self):
        """ int: The count of frames that contain no faces """
        return sum(1 for faces in self.latest_alignments.values() if len(faces) != 0)

    @property
    def no_face_count(self):
        """ int: The count of frames that contain no faces """
        return sum(1 for faces in self.latest_alignments.values() if len(faces) == 0)

    @property
    def multi_face_count(self):
        """ int: The count of frames that contain multiple faces """
        return sum(1 for faces in self.latest_alignments.values() if len(faces) > 1)

    @property
    def tk_unsaved(self):
        """ :class:`tkinter.BooleanVar`: The variable indicating whether the alignments have been
        updated since the last save. """
        return self._tk_unsaved

    @property
    def tk_edited(self):
        """ :class:`tkinter.BooleanVar`: The variable indicating whether the alignments have been
        edited since last Face Display update. """
        return self._tk_edited

    @property
    def current_frame_updated(self):
        """ bool: ``True`` if the current frame has been updated otherwise ``False`` """
        return "new" in self._alignments[self._frames.current_meta_data["filename"]]

    @property
    def updated_alignments(self):
        """ dict: The full frame list, with `None` as the value if alignments not updated. """
        return [self._alignments[frame].get("new", None) for frame in sorted(self._alignments)]

    def link_frames(self, frames):
        """ Add the :class:`FrameNavigation` object as a property of the AlignmentsData.

        Parameters
        ----------
        frames: :class:`~tools.lib_manual.media.FrameNavigation`
            The Frame Navigation object for the manual tool
        """
        self._frames = frames

    def save_video_meta_data(self):
        """ Save the calculated video meta data to the alignments file. """
        if not self._is_video:
            return
        logger.info("Saving video meta information to Alignments file")
        for idx, key in enumerate(sorted(self._alignments_file.data)):
            meta = dict(pts_time=self._frames.video_meta_data["pts_time"][idx],
                        keyframe=idx in self._frames.video_meta_data["keyframes"])
            self._alignments_file.data[key]["video_meta"] = meta
        self._alignments_file.save()

    def reset_face_id(self):
        """ Reset the attribute :attr:`_face_index` to 0 """
        self._face_index = 0

    def get_filtered_frames_list(self):
        """ Return a list of filtered faces based on navigation mode """
        nav_mode = self._frames.tk_navigation_mode.get()
        if nav_mode == "No Faces":
            retval = [idx for idx, count in enumerate(self.face_count_per_index) if count == 0]
        elif nav_mode == "Multiple Faces":
            retval = [idx for idx, count in enumerate(self.face_count_per_index) if count > 1]
        elif nav_mode == "Has Face(s)":
            retval = [idx for idx, count in enumerate(self.face_count_per_index) if count != 0]
        else:
            retval = range(self._frames.frame_count)
        logger.trace("nav_mode: %s, number_frames: %s", nav_mode, len(retval))
        return retval

    def _get_alignments_file(self, alignments_path, input_location):
        """ Get the alignments file.

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
            folder, filename = os.path.split(alignments_path)
        else:
            filename = "alignments.fsa"
            if self._is_video:
                folder, vid = os.path.split(os.path.splitext(input_location)[0])
                filename = "{}_{}".format(vid, filename)
            else:
                folder = input_location
        alignments = Alignments(folder, filename)
        mask_names = set(alignments.mask_summary)
        self._alignments_file = alignments
        self._mask_names = mask_names

    def load_faces(self):
        """ Load the faces at correct size. """
        self._face_size = min(self._frames.display_dims)
        for framename, val in self._alignments_file.data.items():
            this_frame_faces = []
            for item in val["faces"]:
                face = DetectedFace()
                face.from_alignment(item)
                # Size is set so attributes are correct for zooming into a face in the frame viewer
                face.load_aligned(None, size=self._face_size)
                this_frame_faces.append(face)
            self._alignments[framename] = dict(saved=this_frame_faces)

    def save(self):
        """ Save the alignments file """
        if not self._tk_unsaved.get():
            logger.debug("Alignments not updated. Returning")
            return
        to_save = {key: val["new"] for key, val in self._alignments.items() if "new" in val}
        logger.info("Saving alignments for frames: '%s'", list(to_save.keys()))

        for frame, faces in to_save.items():
            self._alignments_file.data[frame]["faces"] = [face.to_alignment() for face in faces]
            self._alignments[frame]["saved"] = faces
            del self._alignments[frame]["new"]

        self._alignments_file.backup()
        self._alignments_file.save()
        self._tk_unsaved.set(False)

    def _check_for_new_alignments(self, filename=None):
        """ Checks whether there are already new alignments in :attr:`_alignments`. If not
        then saved alignments are copied to new ready for update.

        Parameters
        filename: str, optional
            The filename of the frame to check for new alignments. If ``None`` then the current
            frame is checked. Default: ``None``
        """
        filename = self._frames.current_meta_data["filename"] if filename is None else filename
        if self._alignments[filename].get("new", None) is None:
            new_faces = [deepcopy(face) for face in self._alignments[filename]["saved"]]
            self._alignments[filename]["new"] = new_faces
            if not self._tk_unsaved.get():
                self._tk_unsaved.set(True)

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
        face.load_aligned(self._frames.current_frame, size=self._face_size, force=True)
        self._tk_edited.set(True)
        # TODO Link this in to edited
        self._frames.tk_update.set(True)

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
        face.load_aligned(self._frames.current_frame, size=self._face_size, force=True)
        self._tk_edited.set(True)
        self._frames.tk_update.set(True)

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
        face.load_aligned(self._frames.current_frame, size=self._face_size, force=True)
        self._tk_edited.set(True)
        self._frames.tk_update.set(True)

    def update_mask(self, mask, mask_type, index):
        """ Update the mask on an edit """
        self._check_for_new_alignments()
        self._face_index = index
        self.current_face.mask[mask_type].replace_mask(mask)
        self._tk_edited.set(True)
        self._frames.tk_update.set(True)

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
        self._tk_edited.set(True)
        self._frames.tk_update.set(True)

    def delete_face_at_index_by_frame(self, frame_index, face_index):
        """ Delete the :class:`DetectedFace` object for the given face index and frame index.

        Is called from the Faces Viewer, so no frame specific settings should be modified or
        updates executed.

        Parameters
        ----------
        frame_index: int
            The frame index that the face should be deleted for
        face_index: int
            The face index to remove the face for
        """
        logger.debug("Deleting face at index: %s for frame: %s", face_index, frame_index)

        filename = sorted(self._alignments)[frame_index]
        self._check_for_new_alignments(filename=filename)
        del self.latest_alignments[filename][face_index]
        self._face_count_modified = True

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

    def get_aligned_face_at_index(self,
                                  index,
                                  frame_index=None,
                                  size=None,
                                  with_landmarks=False,
                                  with_mask=False):
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
        size = self._face_size if size is None else size
        if frame_index is None:
            face = self.current_faces[index]
        else:
            frame_name = sorted(self._alignments)[frame_index]
            face = self.latest_alignments[frame_name][index]
        face.load_aligned(self._frames.current_frame, size=size, force=True)
        retval = face.aligned_face.copy()
        retval = [retval] if with_landmarks or with_mask else retval
        face.aligned["face"] = None
        if with_landmarks:
            retval.append(face.aligned_landmarks)
        if with_mask:
            retval.append(face.mask)
        return retval

    def copy_alignments(self, direction):
        """ Copy the alignments from the previous or next frame that has alignments
        to the current frame.

        Notes
        -----
        The aligned face image is loaded so that the faces viewer can pick it up. This image
        is cleared by the faces viewer after collection to save ram.
        """
        self._check_for_new_alignments()
        frames_with_faces = self._frames_with_faces
        frame_name = self._frames.current_meta_data["filename"]

        if direction == "previous":
            idx = bisect.bisect_left(frames_with_faces, frame_name) - 1
            if idx < 0:
                return
        else:
            idx = bisect.bisect(frames_with_faces, frame_name)
            if idx == len(frames_with_faces):
                return
        frame_idx = frames_with_faces[idx]
        logger.debug("Copying frame: %s", frame_idx)
        self.current_faces.extend(deepcopy(self.latest_alignments[frame_idx]))
        for face in self.current_faces:
            face.load_aligned(self._frames.current_frame, size=self._face_size, force=True)
        self._face_index = len(self.current_faces) - 1
        self._tk_edited.set(True)
        self._frames.tk_update.set(True)

    def revert_to_saved(self):
        """ Revert the current frame's alignments to their saved version """
        frame_name = self._frames.current_meta_data["filename"]
        if "new" not in self._alignments[frame_name]:
            logger.info("Alignments not amended. Returning")
            return
        logger.debug("Reverting alignments for '%s'", frame_name)
        del self._alignments[frame_name]["new"]
        self._tk_edited.set(True)
        self._frames.tk_update.set(True)


class FaceCache():
    """ Holds the face images for display in the bottom GUI Panel """
    def __init__(self, alignments, frames, scaling_factor, progress_bar, root):
        logger.debug("Initializing %s: (alignments: %s, frames: %s, scaling_factor: %s)",
                     self.__class__.__name__, alignments, frames, scaling_factor)
        self._alignments = alignments
        self._frames = frames
        self._face_size = int(round(96 * scaling_factor))
        self._root = root
        self._progress_bar = progress_bar
        self._tk_load_complete = self._set_load_complete_var()
        self._alpha = np.ones((self._face_size, self._face_size), dtype="uint8") * 255
        self._landmark_mapping = dict(mouth=(48, 68),
                                      right_eyebrow=(17, 22),
                                      left_eyebrow=(22, 27),
                                      right_eye=(36, 42),
                                      left_eye=(42, 48),
                                      nose=(27, 36),
                                      jaw=(0, 17),
                                      chin=(8, 11))
        self._current_mask_type = None
        self._tk_faces = []
        self._mesh_landmarks = []
        self._background_load_faces()
        logger.debug("Initialized %s", self.__class__.__name__)

    @staticmethod
    def _set_load_complete_var():
        """ Set the load completion variable. """
        var = tk.BooleanVar()
        var.set(False)
        return var

    @property
    def is_initialized(self):
        """ bool: ``True`` if the faces have completed the loading cycle otherwise ``False`` """
        return self._tk_load_complete.get()

    @property
    def tk_faces(self):
        """ list: Item for each frame containing a list of :class:`tkinter.PhotoImage` objects
        for each face. """
        return self._tk_faces

    @property
    def mesh_landmarks(self):
        """ list: Item for each frame containing a dictionary for each face. """
        return self._mesh_landmarks

    @property
    def size(self):
        """ int: The size of each individual face in pixels. """
        return self._face_size

    def _background_load_faces(self):
        """ Launch a background thread to load the faces into cache and assign the canvas to
        :attr:`_canvas` """
        thread = MultiThread(self._load_faces,
                             thread_count=1,
                             name="{}.load_faces".format(self.__class__.__name__))
        thread.start()

    def _load_faces(self):
        """ Loads the faces into the :attr:`_faces` dict at 96px size formatted for GUI display.
        """
        loader = ImagesLoader(self._frames.location, count=self._frames.frame_count)
        for filename, frame in loader.load():
            tk_faces = []
            mesh_landmarks = []
            faces = self._alignments.saved_alignments.get(os.path.basename(filename), list())
            for face in faces:
                tk_faces.append(self._load_face(frame, face))
                mesh_landmarks.append(self.get_mesh_points(face.aligned_landmarks))
            self._tk_faces.append(tk_faces)
            self._mesh_landmarks.append(mesh_landmarks)

    def _load_face(self, frame, face):
        """ Load the resized aligned face. """
        face.load_aligned(frame, size=self._face_size, force=True)
        bytes_string = self.generate_tk_face_data(face.aligned_face)
        face.aligned["face"] = None
        # TODO Document that this is set to a photo image from the canvas.
        # Lists are thread safe (for our purposes) PhotoImage is not.
        return bytes_string

    def generate_tk_face_data(self, image, mask=None):
        """ Generate a new :tkinter:`PhotoImage` object with an empty mask in the 4th channel. """
        mask = self._alpha if mask is None else mask
        if mask.shape[0] != self._face_size:
            mask = cv2.resize(mask,
                              (self._face_size, self._face_size),
                              interpolation=cv2.INTER_AREA)

        face = np.concatenate((image[..., :3], mask[..., None]), axis=-1)
        return cv2.imencode(".png", face, [cv2.IMWRITE_PNG_COMPRESSION, 0])[1].tostring()

    def set_load_complete(self):
        """ TODO """
        self._tk_load_complete.set(True)

    def get_mesh_points(self, landmarks):
        """ Obtain the mesh annotation points for a given set of landmarks. """
        is_poly = []
        mesh_landmarks = []
        for key, val in self._landmark_mapping.items():
            is_poly.append(key in ("right_eye", "left_eye", "mouth"))
            mesh_landmarks.append(landmarks[val[0]:val[1]])
        return dict(is_poly=is_poly, landmarks=mesh_landmarks)

    def add(self, frame_index, tk_face, mesh_landmarks):
        """ Add new objects to the faces cache. """
        logger.debug("Adding objects: (frame_index: %s, tk_face: %s, mesh_landmarks: %s)",
                     frame_index, tk_face, mesh_landmarks)
        self._tk_faces[frame_index].append(tk_face)
        self._mesh_landmarks[frame_index].append(mesh_landmarks)

    def remove(self, frame_index, face_index):
        """ Remove objects from the faces cache. """
        logger.debug("Removing objects: (frame_index: %s, face_index: %s)",
                     frame_index, face_index)
        del self._tk_faces[frame_index][face_index]
        del self._mesh_landmarks[frame_index][face_index]

    def update(self, frame_index, face_index, tk_face, mesh_landmarks):
        """ Update existing objects in the faces cache. """
        logger.trace("Updating objects: (frame_index: %s, face_index: %s, tk_face: %s, "
                     "mesh_landmarks: %s)", frame_index, face_index, tk_face, mesh_landmarks)
        self._tk_faces[frame_index][face_index] = tk_face
        self._mesh_landmarks[frame_index][face_index] = mesh_landmarks

    def update_tk_face_for_masks(self, mask_type, is_enabled):
        """ Load the selected masks """
        mask_type = None if not is_enabled or mask_type == "" else mask_type.lower()
        if not self.is_initialized or mask_type == self._current_mask_type:
            return
        self._start_progress()
        if mask_type is None:
            self._unload_masks()
        else:
            self._load_masks(mask_type)
        self._stop_progress()
        self._current_mask_type = mask_type

    def _unload_masks(self):
        """ Remove mask from stored face. """
        for frame_idx, tk_faces in enumerate(self._tk_faces):
            self._update_progress(None, frame_idx)
            for tk_face in tk_faces:
                self._update_mask_to_photoimage(tk_face, self._alpha)

    def _load_masks(self, mask_type):
        """ Load the selected masks """
        latest_alignments = self._alignments.latest_alignments
        for frame_idx, (key, tk_faces) in enumerate(zip(sorted(latest_alignments),
                                                        self._tk_faces)):
            self._update_progress(mask_type, frame_idx)
            for face, tk_face in zip(latest_alignments[key], tk_faces):
                self._update_mask_to_photoimage(tk_face, self._get_mask(mask_type, face))

    def _get_mask(self, mask_type, detected_face):
        """ Obtain the mask from the alignments file. """
        if mask_type is None:
            return self._alpha
        mask_class = detected_face.mask.get(mask_type, None)
        mask = self._alpha if mask_class is None else mask_class.mask.squeeze()
        if mask.shape[0] != self._face_size:
            mask = cv2.resize(mask,
                              (self._face_size, self._face_size),
                              interpolation=cv2.INTER_AREA)
        return mask

    @staticmethod
    def _update_mask_to_photoimage(tk_face, mask):
        """ Adjust the mask of the current tk_face """
        img = cv2.imdecode(np.fromstring(tk_face.cget("data"), dtype="uint8"),
                           cv2.IMREAD_UNCHANGED)
        img[..., -1] = mask
        tk_face.put(cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 0])[1].tostring())

    def _start_progress(self):
        """ Start the progress bar. """
        self._progress_bar.start(mode="determinate")
        self._root.config(cursor="watch")

    def _update_progress(self, mask_type, frame_index):
        """ Update the progress bar. """
        position = frame_index + 1
        progress = int(round((position / self._frames.frame_count) * 100))
        msg = "Removing Mask: " if mask_type is None else "Loading {} Mask: ".format(mask_type)
        msg += "{}/{} - {}%".format(position, self._frames.frame_count, progress)
        self._progress_bar.progress_update(msg, progress)
        self._root.update_idletasks()

    def _stop_progress(self):
        """ Stop the progress bar. """
        self._progress_bar.stop()
        self._root.config(cursor="")
