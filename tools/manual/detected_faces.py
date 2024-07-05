#!/usr/bin/env python3
""" Alignments handling for Faceswap's Manual Adjustments tool. Handles the conversion of
alignments data to :class:`~lib.align.DetectedFace` objects, and the update of these faces
when edits are made in the GUI. """
from __future__ import annotations
import logging
import os
import sys
import tkinter as tk
import typing as T
from copy import deepcopy
from queue import Queue, Empty

import cv2
import numpy as np

from lib.align import Alignments, AlignedFace, DetectedFace
from lib.gui.custom_widgets import PopupProgress
from lib.gui.utils import FileHandler
from lib.image import ImagesLoader, ImagesSaver, encode_image, generate_thumbnail
from lib.multithreading import MultiThread
from lib.utils import get_folder

if T.TYPE_CHECKING:
    from . import manual
    from lib.align.alignments import AlignmentFileDict, PNGHeaderDict

logger = logging.getLogger(__name__)


class DetectedFaces():
    """ Handles the manipulation of :class:`~lib.align.DetectedFace` objects stored
    in the alignments file. Acts as a parent class for the IO operations (saving and loading from
    an alignments file), the face update operations (when changes are made to alignments in the
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
    def __init__(self,
                 tk_globals: manual.TkGlobals,
                 alignments_path: str,
                 input_location: str,
                 extractor: manual.Aligner) -> None:
        logger.debug("Initializing %s: (tk_globals: %s. alignments_path: %s, input_location: %s "
                     "extractor: %s)", self.__class__.__name__, tk_globals, alignments_path,
                     input_location, extractor)
        self._globals = tk_globals
        self._frame_faces: list[list[DetectedFace]] = []
        self._updated_frame_indices: set[int] = set()

        self._alignments: Alignments = self._get_alignments(alignments_path, input_location)
        self._alignments.update_legacy_has_source(os.path.basename(input_location))

        self._extractor = extractor
        self._tk_vars = self._set_tk_vars()

        self._io = _DiskIO(self, input_location)
        self._update = FaceUpdate(self)
        self._filter = Filter(self)
        logger.debug("Initialized %s", self.__class__.__name__)

    # <<<< PUBLIC PROPERTIES >>>> #
    @property
    def extractor(self) -> manual.Aligner:
        """ :class:`~tools.manual.manual.Aligner`: The pipeline for passing faces through the
        aligner and retrieving results. """
        return self._extractor

    @property
    def filter(self) -> Filter:
        """ :class:`Filter`: Handles returning of faces and stats based on the current user set
        navigation mode filter. """
        return self._filter

    @property
    def update(self) -> FaceUpdate:
        """ :class:`FaceUpdate`: Handles the adding, removing and updating of
        :class:`~lib.align.DetectedFace` stored within the alignments file. """
        return self._update

    # << TKINTER VARIABLES >> #
    @property
    def tk_unsaved(self) -> tk.BooleanVar:
        """ :class:`tkinter.BooleanVar`: The variable indicating whether the alignments have been
        updated since the last save. """
        return self._tk_vars["unsaved"]

    @property
    def tk_edited(self) -> tk.BooleanVar:
        """ :class:`tkinter.BooleanVar`: The variable indicating whether an edit has occurred
        meaning a GUI redraw needs to be triggered. """
        return self._tk_vars["edited"]

    @property
    def tk_face_count_changed(self) -> tk.BooleanVar:
        """ :class:`tkinter.BooleanVar`: The variable indicating whether a face has been added or
        removed meaning the :class:`FaceViewer` grid redraw needs to be triggered. """
        return self._tk_vars["face_count_changed"]

    # << STATISTICS >> #
    @property
    def frame_list(self) -> list[str]:
        """ list[str]: The list of all frame names that appear in the alignments file """
        return list(self._alignments.data)

    @property
    def available_masks(self) -> dict[str, int]:
        """ dict[str, int]: The mask type names stored in the alignments; type as key with the
        number of faces which possess the mask type as value. """
        return self._alignments.mask_summary

    @property
    def current_faces(self) -> list[list[DetectedFace]]:
        """ list[list[:class:`~lib.align.DetectedFace`]]: The most up to date full list of detected
        face objects. """
        return self._frame_faces

    @property
    def video_meta_data(self) -> dict[str, list[int] | list[float] | None]:
        """ dict[str, list[int] | list[float] | None]: The frame meta data stored in the alignments
        file. If data does not exist in the alignments file then ``None`` is returned for each
        Key """
        return self._alignments.video_meta_data

    @property
    def face_count_per_index(self) -> list[int]:
        """ list[int]: Count of faces for each frame. List is in frame index order.

        The list needs to be calculated on the fly as the number of faces in a frame
        can change based on user actions. """
        return [len(faces) for faces in self._frame_faces]

    # <<<< PUBLIC METHODS >>>> #
    def is_frame_updated(self, frame_index: int) -> bool:
        """ Check whether the given frame index has been updated

        Parameters
        ----------
        frame_index: int
            The frame index to check

        Returns
        -------
        bool:
            ``True`` if the given frame index has updated faces within it otherwise ``False``
        """
        return frame_index in self._updated_frame_indices

    def load_faces(self) -> None:
        """ Load the faces as :class:`~lib.align.DetectedFace` objects from the alignments
        file. """
        self._io.load()

    def save(self) -> None:
        """ Save the alignments file with the latest edits. """
        self._io.save()

    def revert_to_saved(self, frame_index):
        """ Revert the frame's alignments to their saved version for the given frame index.

        Parameters
        ----------
        frame_index: int
            The frame that should have their faces reverted to their saved version
        """
        self._io.revert_to_saved(frame_index)

    def extract(self) -> None:
        """ Extract the faces in the current video to a user supplied folder. """
        self._io.extract()

    def save_video_meta_data(self, pts_time: list[float], keyframes: list[int]) -> None:
        """ Save video meta data to the alignments file. This is executed if the video meta data
        does not already exist in the alignments file, so the video does not need to be scanned
        on every use of the Manual Tool.

        Parameters
        ----------
        pts_time: list[float]
            A list of presentation timestamps in frame index order for every frame in the input
            video
        keyframes: list[int]
            A list of frame indices corresponding to the key frames in the input video.
        """
        if self._globals.is_video:
            self._alignments.save_video_meta_data(pts_time, keyframes)

    # <<<< PRIVATE METHODS >>> #
    # << INIT >> #
    @staticmethod
    def _set_tk_vars() -> dict[T.Literal["unsaved", "edited", "face_count_changed"],
                               tk.BooleanVar]:
        """ Set the required tkinter variables.

        The alignments specific `unsaved` and `edited` are set here.
        The global variables are added into the dictionary with `None` as value, so the
        objects exist. Their actual variables are populated during :func:`load_faces`.

        Returns
        -------
        dict
            The internal variable name as key with the tkinter variable as value
        """
        retval = {}
        for name in T.get_args(T.Literal["unsaved", "edited", "face_count_changed"]):
            var = tk.BooleanVar()
            var.set(False)
            retval[name] = var
        logger.debug(retval)
        return retval

    def _get_alignments(self, alignments_path: str, input_location: str) -> Alignments:
        """ Get the :class:`~lib.align.Alignments` object for the given location.

        Parameters
        ----------
        alignments_path: str
            Full path to the alignments file. If empty string is passed then location is calculated
            from the source folder
        input_location: str
            The location of the input folder of frames or video file

        Returns
        -------
        :class:`~lib.align.Alignments`
            The alignments object for the given input location
        """
        logger.debug("alignments_path: %s, input_location: %s", alignments_path, input_location)
        if alignments_path:
            folder, filename = os.path.split(alignments_path)
        else:
            filename = "alignments.fsa"
            if self._globals.is_video:
                folder, vid = os.path.split(os.path.splitext(input_location)[0])
                filename = f"{vid}_{filename}"
            else:
                folder = input_location
        retval = Alignments(folder, filename)
        if retval.version == 1.0:
            logger.error("The Manual Tool is not compatible with legacy Alignments files.")
            logger.info("You can update legacy Alignments files by using the Extract job in the "
                        "Alignments tool to re-extract the faces in full-head format.")
            sys.exit(0)
        logger.debug("folder: %s, filename: %s, alignments: %s", folder, filename, retval)
        return retval


class _DiskIO():
    """ Handles the loading of :class:`~lib.align.DetectedFaces` from the alignments file
    into :class:`DetectedFaces` and the saving of this data (in the opposite direction) to an
    alignments file.

    Parameters
    ----------
    detected_faces: :class:`DetectedFaces`
        The parent :class:`DetectedFaces` object
    input_location: str
        The location of the input folder of frames or video file
    """
    def __init__(self, detected_faces: DetectedFaces, input_location: str) -> None:
        logger.debug("Initializing %s: (detected_faces: %s, input_location: %s)",
                     self.__class__.__name__, detected_faces, input_location)
        self._input_location = input_location
        self._alignments = detected_faces._alignments
        self._frame_faces = detected_faces._frame_faces
        self._updated_frame_indices = detected_faces._updated_frame_indices
        self._tk_unsaved = detected_faces.tk_unsaved
        self._tk_edited = detected_faces.tk_edited
        self._tk_face_count_changed = detected_faces.tk_face_count_changed
        self._globals = detected_faces._globals

        # Must be populated after loading faces as video_meta_data may have increased frame count
        self._sorted_frame_names: list[str] = []
        logger.debug("Initialized %s", self.__class__.__name__)

    def load(self) -> None:
        """ Load the faces from the alignments file, convert to
        :class:`~lib.align.DetectedFace`. objects and add to :attr:`_frame_faces`. """
        for key in sorted(self._alignments.data):
            this_frame_faces: list[DetectedFace] = []
            for item in self._alignments.data[key]["faces"]:
                face = DetectedFace()
                face.from_alignment(item, with_thumb=True)
                face.load_aligned(None)
                _ = face.aligned.average_distance  # cache the distances
                this_frame_faces.append(face)
            self._frame_faces.append(this_frame_faces)
        self._sorted_frame_names = sorted(self._alignments.data)

    def save(self) -> None:
        """ Convert updated :class:`~lib.align.DetectedFace` objects to alignments format
        and save the alignments file. """
        if not self._tk_unsaved.get():
            logger.debug("Alignments not updated. Returning")
            return
        frames = list(self._updated_frame_indices)
        logger.verbose("Saving alignments for %s updated frames",  # type:ignore[attr-defined]
                       len(frames))

        for idx, faces in zip(frames,
                              np.array(self._frame_faces, dtype="object")[np.array(frames)]):
            frame = self._sorted_frame_names[idx]
            self._alignments.data[frame]["faces"] = [face.to_alignment() for face in faces]

        self._alignments.backup()
        self._alignments.save()
        self._updated_frame_indices.clear()
        self._tk_unsaved.set(False)

    def revert_to_saved(self, frame_index: int) -> None:
        """ Revert the frame's alignments to their saved version for the given frame index.

        Parameters
        ----------
        frame_index: int
            The frame that should have their faces reverted to their saved version
        """
        if frame_index not in self._updated_frame_indices:
            logger.debug("Alignments not amended. Returning")
            return
        logger.verbose("Reverting alignments for frame_index %s",  # type:ignore[attr-defined]
                       frame_index)
        alignments = self._alignments.data[self._sorted_frame_names[frame_index]]["faces"]
        faces = self._frame_faces[frame_index]

        reset_grid = self._add_remove_faces(alignments, faces)

        for detected_face, face in zip(faces, alignments):
            detected_face.from_alignment(face, with_thumb=True)
            detected_face.load_aligned(None, force=True)
            _ = detected_face.aligned.average_distance  # cache the distances

        self._updated_frame_indices.remove(frame_index)
        if not self._updated_frame_indices:
            self._tk_unsaved.set(False)

        if reset_grid:
            self._tk_face_count_changed.set(True)
        else:
            self._tk_edited.set(True)
        self._globals.var_full_update.set(True)

    @classmethod
    def _add_remove_faces(cls,
                          alignments: list[AlignmentFileDict],
                          faces: list[DetectedFace]) -> bool:
        """ On a revert, ensure that the alignments and detected face object counts for each frame
        are in sync.

        Parameters
        ----------
        alignments: list[:class:`~lib.align.alignments.AlignmentFileDict`]
            Alignments stored for a frame

        faces: list[:class:`~lib.align.DetectedFace`]
            List of detected faces for a frame

        Returns
        -------
        bool
            ``True`` if a face was added or removed otherwise ``False``
        """
        num_alignments = len(alignments)
        num_faces = len(faces)
        if num_alignments == num_faces:
            retval = False
        elif num_alignments > num_faces:
            faces.extend([DetectedFace() for _ in range(num_faces, num_alignments)])
            retval = True
        else:
            del faces[num_alignments:]
            retval = True
        return retval

    def extract(self) -> None:
        """ Extract the current faces to a folder.

        To stop the GUI becoming completely unresponsive (particularly in Windows) the extract is
        done in a background thread, with the process count passed back in a queue to the main
        thread to update the progress bar.
        """
        dirname = FileHandler("dir", None,
                              initial_folder=os.path.dirname(self._input_location),
                              title="Select output folder...").return_file
        if not dirname:
            return
        logger.debug(dirname)

        queue: Queue = Queue()
        pbar = PopupProgress("Extracting Faces...", self._alignments.frames_count + 1)
        thread = MultiThread(self._background_extract, dirname, queue)
        thread.start()
        self._monitor_extract(thread, queue, pbar)

    def _monitor_extract(self,
                         thread: MultiThread,
                         queue: Queue,
                         progress_bar: PopupProgress) -> None:
        """ Monitor the extraction thread, and update the progress bar.

        On completion, save alignments and clear progress bar.

        Parameters
        ----------
        thread: :class:`~lib.multithreading.MultiThread`
            The thread that is performing the extraction task
        queue: :class:`queue.Queue`
            The queue that the worker thread is putting it's incremental counts to
        progress_bar: :class:`~lib.gui.custom_widget.PopupProgress`
            The popped up progress bar
        """
        thread.check_and_raise_error()
        if not thread.is_alive():
            thread.join()
            progress_bar.stop()
            return

        while True:
            try:
                progress_bar.step(queue.get(False, 0))
            except Empty:
                break
        progress_bar.after(100, self._monitor_extract, thread, queue, progress_bar)

    def _background_extract(self, output_folder: str, progress_queue: Queue) -> None:
        """ Perform the background extraction in a thread so GUI doesn't become unresponsive.

        Parameters
        ----------
        output_folder: str
            The location to save the output faces to
        progress_queue: :class:`queue.Queue`
            The queue to place incremental counts to for updating the GUI's progress bar
        """
        saver = ImagesSaver(get_folder(output_folder), as_bytes=True)
        loader = ImagesLoader(self._input_location, count=self._alignments.frames_count)
        for frame_idx, (filename, image) in enumerate(loader.load()):
            logger.trace("Outputting frame: %s: %s",  # type:ignore[attr-defined]
                         frame_idx, filename)
            src_filename = os.path.basename(filename)
            progress_queue.put(1)

            for face_idx, face in enumerate(self._frame_faces[frame_idx]):
                output = f"{os.path.splitext(src_filename)[0]}_{face_idx}.png"
                aligned = AlignedFace(face.landmarks_xy,
                                      image=image,
                                      centering="head",
                                      size=512)  # TODO user selectable size
                meta: PNGHeaderDict = {"alignments": face.to_png_meta(),
                                       "source": {"alignments_version": self._alignments.version,
                                                  "original_filename": output,
                                                  "face_index": face_idx,
                                                  "source_filename": src_filename,
                                                  "source_is_video": self._globals.is_video,
                                                  "source_frame_dims": image.shape[:2]}}

                assert aligned.face is not None
                b_image = encode_image(aligned.face, ".png", metadata=meta)
                saver.save(output, b_image)
        saver.close()


class Filter():
    """ Returns stats and frames for filtered frames based on the user selected navigation mode
    filter.

    Parameters
    ----------
    detected_faces: :class:`DetectedFaces`
        The parent :class:`DetectedFaces` object
    """
    def __init__(self, detected_faces: DetectedFaces) -> None:
        logger.debug("Initializing %s: (detected_faces: %s)",
                     self.__class__.__name__, detected_faces)
        self._globals = detected_faces._globals
        self._detected_faces = detected_faces
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def frame_meets_criteria(self) -> bool:
        """ bool: ``True`` if the current frame meets the selected filter criteria otherwise
        ``False`` """
        filter_mode = self._globals.var_filter_mode.get()
        frame_faces = self._detected_faces.current_faces[self._globals.frame_index]
        distance = self._filter_distance

        retval = (
            filter_mode == "All Frames" or
            (filter_mode == "No Faces" and not frame_faces) or
            (filter_mode == "Has Face(s)" and len(frame_faces) > 0) or
            (filter_mode == "Multiple Faces" and len(frame_faces) > 1) or
            (filter_mode == "Misaligned Faces" and any(face.aligned.average_distance > distance
                                                       for face in frame_faces)))
        assert isinstance(retval, bool)
        logger.trace("filter_mode: %s, frame meets criteria: %s",  # type:ignore[attr-defined]
                     filter_mode, retval)
        return retval

    @property
    def _filter_distance(self) -> float:
        """ float: The currently selected distance when Misaligned Faces filter is selected. """
        try:
            retval = self._globals.var_filter_distance.get()
        except tk.TclError:
            # Suppress error when distance box is empty
            retval = 0
        return retval / 100.

    @property
    def count(self) -> int:
        """ int: The number of frames that meet the filter criteria returned by
        :attr:`~tools.manual.manual.TkGlobals.var_filter_mode.get()`. """
        face_count_per_index = self._detected_faces.face_count_per_index
        if self._globals.var_filter_mode.get() == "No Faces":
            retval = sum(1 for fcount in face_count_per_index if fcount == 0)
        elif self._globals.var_filter_mode.get() == "Has Face(s)":
            retval = sum(1 for fcount in face_count_per_index if fcount != 0)
        elif self._globals.var_filter_mode.get() == "Multiple Faces":
            retval = sum(1 for fcount in face_count_per_index if fcount > 1)
        elif self._globals.var_filter_mode.get() == "Misaligned Faces":
            distance = self._filter_distance
            retval = sum(1 for frame in self._detected_faces.current_faces
                         if any(face.aligned.average_distance > distance for face in frame))
        else:
            retval = len(face_count_per_index)
        logger.trace("filter mode: %s, frame count: %s",  # type:ignore[attr-defined]
                     self._globals.var_filter_mode.get(), retval)
        return retval

    @property
    def raw_indices(self) -> dict[T.Literal["frame", "face"], list[int]]:
        """ dict[str, int]: The frame and face indices that meet the current filter criteria for
        each displayed face. """
        frame_indices: list[int] = []
        face_indices: list[int] = []
        face_counts = self._detected_faces.face_count_per_index  # Copy to avoid recalculations

        for frame_idx in self.frames_list:
            for face_idx in range(face_counts[frame_idx]):
                frame_indices.append(frame_idx)
                face_indices.append(face_idx)

        retval: dict[T.Literal["frame", "face"], list[int]] = {"frame": frame_indices,
                                                               "face": face_indices}
        logger.trace("frame_indices: %s, face_indices: %s",  # type:ignore[attr-defined]
                     frame_indices, face_indices)
        return retval

    @property
    def frames_list(self) -> list[int]:
        """ list[int]: The list of frame indices that meet the filter criteria returned by
        :attr:`~tools.manual.manual.TkGlobals.var_filter_mode.get()`. """
        face_count_per_index = self._detected_faces.face_count_per_index
        if self._globals.var_filter_mode.get() == "No Faces":
            retval = [idx for idx, count in enumerate(face_count_per_index) if count == 0]
        elif self._globals.var_filter_mode.get() == "Multiple Faces":
            retval = [idx for idx, count in enumerate(face_count_per_index) if count > 1]
        elif self._globals.var_filter_mode.get() == "Has Face(s)":
            retval = [idx for idx, count in enumerate(face_count_per_index) if count != 0]
        elif self._globals.var_filter_mode.get() == "Misaligned Faces":
            distance = self._filter_distance
            retval = [idx for idx, frame in enumerate(self._detected_faces.current_faces)
                      if any(face.aligned.average_distance > distance for face in frame)]
        else:
            retval = list(range(len(face_count_per_index)))
        logger.trace("filter mode: %s, number_frames: %s",  # type:ignore[attr-defined]
                     self._globals.var_filter_mode.get(), len(retval))
        return retval


class FaceUpdate():
    """ Perform updates on :class:`~lib.align.DetectedFace` objects stored in
    :class:`DetectedFaces` when changes are made within the GUI.

    Parameters
    ----------
    detected_faces: :class:`DetectedFaces`
        The parent :class:`DetectedFaces` object
    """
    def __init__(self, detected_faces: DetectedFaces) -> None:
        logger.debug("Initializing %s: (detected_faces: %s)",
                     self.__class__.__name__, detected_faces)
        self._detected_faces = detected_faces
        self._globals = detected_faces._globals
        self._frame_faces = detected_faces._frame_faces
        self._updated_frame_indices = detected_faces._updated_frame_indices
        self._tk_unsaved = detected_faces.tk_unsaved
        self._extractor = detected_faces.extractor
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def _tk_edited(self) -> tk.BooleanVar:
        """ :class:`tkinter.BooleanVar`: The variable indicating whether an edit has occurred
        meaning a GUI redraw needs to be triggered.

        Notes
        -----
        The variable is still a ``None`` when this class is initialized, so referenced explicitly.
        """
        return self._detected_faces.tk_edited

    @property
    def _tk_face_count_changed(self) -> tk.BooleanVar:
        """ :class:`tkinter.BooleanVar`: The variable indicating whether an edit has occurred
        meaning a GUI redraw needs to be triggered.

        Notes
        -----
        The variable is still a ``None`` when this class is initialized, so referenced explicitly.
        """
        return self._detected_faces.tk_face_count_changed

    def _faces_at_frame_index(self, frame_index: int) -> list[DetectedFace]:
        """ Checks whether the frame has already been added to :attr:`_updated_frame_indices` and
        adds it. Triggers the unsaved variable if this is the first edited frame. Returns the
        detected face objects for the given frame.

        Parameters
        ----------
        frame_index: int
            The frame index to check whether there are updated alignments available

        Returns
        -------
        list
            The :class:`~lib.align.DetectedFace` objects for the requested frame
        """
        if not self._updated_frame_indices and not self._tk_unsaved.get():
            self._tk_unsaved.set(True)
        self._updated_frame_indices.add(frame_index)
        retval = self._frame_faces[frame_index]
        return retval

    def add(self, frame_index: int, pnt_x: int, width: int, pnt_y: int, height: int) -> None:
        """ Add a :class:`~lib.align.DetectedFace` object to the current frame with the
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
        face = DetectedFace()
        faces = self._faces_at_frame_index(frame_index)
        faces.append(face)
        face_index = len(faces) - 1

        self.bounding_box(frame_index, face_index, pnt_x, width, pnt_y, height, aligner="cv2-dnn")
        face.load_aligned(None)
        self._tk_face_count_changed.set(True)

    def delete(self, frame_index: int, face_index: int) -> None:
        """ Delete the :class:`~lib.align.DetectedFace` object for the given frame and face
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
        self._globals.var_full_update.set(True)

    def bounding_box(self,
                     frame_index: int,
                     face_index: int,
                     pnt_x: int,
                     width: int,
                     pnt_y: int,
                     height: int,
                     aligner: manual.TypeManualExtractor = "FAN") -> None:
        """ Update the bounding box for the :class:`~lib.align.DetectedFace` object at the
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
        logger.trace("frame_index: %s, face_index %s, pnt_x %s, "  # type:ignore[attr-defined]
                     "width %s, pnt_y %s, height %s, aligner: %s",
                     frame_index, face_index, pnt_x, width, pnt_y, height, aligner)
        face = self._faces_at_frame_index(frame_index)[face_index]
        face.left = pnt_x
        face.width = width
        face.top = pnt_y
        face.height = height
        face.add_landmarks_xy(self._extractor.get_landmarks(frame_index, face_index, aligner))
        self._globals.var_full_update.set(True)

    def landmark(self,
                 frame_index: int, face_index: int,
                 landmark_index: int,
                 shift_x: int,
                 shift_y: int,
                 is_zoomed: bool) -> None:
        """ Shift a single landmark point for the :class:`~lib.align.DetectedFace` object
        at the given frame and face indices by the given x and y values.

        Parameters
        ----------
        frame_index: int
            The frame that the face is being set for
        face_index: int
            The face index within the frame
        landmark_index: int or list
            The landmark index to shift. If a list is provided, this should be a list of landmark
            indices to be shifted
        shift_x: int
            The amount to shift the landmark by along the x axis
        shift_y: int
            The amount to shift the landmark by along the y axis
        is_zoomed: bool
            ``True`` if landmarks are being adjusted on a zoomed image otherwise ``False``
        """
        face = self._faces_at_frame_index(frame_index)[face_index]
        if is_zoomed:
            aligned = AlignedFace(face.landmarks_xy,
                                  centering="face",
                                  size=min(self._globals.frame_display_dims))
            landmark = aligned.landmarks[landmark_index]
            landmark += (shift_x, shift_y)
            matrix = aligned.adjusted_matrix
            matrix = cv2.invertAffineTransform(matrix)
            if landmark.ndim == 1:
                landmark = np.reshape(landmark, (1, 1, 2))
                landmark = cv2.transform(landmark, matrix, landmark.shape).squeeze()
                face.landmarks_xy[landmark_index] = landmark
            else:
                for lmk, idx in zip(landmark, landmark_index):  # type:ignore[call-overload]
                    lmk = np.reshape(lmk, (1, 1, 2))
                    lmk = cv2.transform(lmk, matrix, lmk.shape).squeeze()
                    face.landmarks_xy[idx] = lmk
        else:
            face.landmarks_xy[landmark_index] += (shift_x, shift_y)
        self._globals.var_full_update.set(True)

    def landmarks(self, frame_index: int, face_index: int, shift_x: int, shift_y: int) -> None:
        """ Shift all of the landmarks and bounding box for the
        :class:`~lib.align.DetectedFace` object at the given frame and face indices by the
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
        assert face.left is not None and face.top is not None
        face.left += shift_x
        face.top += shift_y
        face.add_landmarks_xy(face.landmarks_xy + (shift_x, shift_y))
        self._globals.var_full_update.set(True)

    def landmarks_rotate(self,
                         frame_index: int,
                         face_index: int,
                         angle: float,
                         center: np.ndarray) -> None:
        """ Rotate the landmarks on an Extract Box rotate for the
        :class:`~lib.align.DetectedFace` object at the given frame and face indices for the
        given angle from the given center point.

        Parameters
        ----------
        frame_index: int
            The frame that the face is being set for
        face_index: int
            The face index within the frame
        angle: float
            The angle, in radians to rotate the points by
        center: :class:`numpy.ndarray`
            The center point of the Landmark's Extract Box
        """
        face = self._faces_at_frame_index(frame_index)[face_index]
        rot_mat = cv2.getRotationMatrix2D(tuple(center.astype("float32")), angle, 1.)
        face.add_landmarks_xy(cv2.transform(np.expand_dims(face.landmarks_xy, axis=0),
                                            rot_mat).squeeze())
        self._globals.var_full_update.set(True)

    def landmarks_scale(self,
                        frame_index: int,
                        face_index: int,
                        scale: np.ndarray,
                        center: np.ndarray) -> None:
        """ Scale the landmarks on an Extract Box resize for the
        :class:`~lib.align.DetectedFace` object at the given frame and face indices from the
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
        face.add_landmarks_xy(((face.landmarks_xy - center) * scale) + center)
        self._globals.var_full_update.set(True)

    def mask(self, frame_index: int, face_index: int, mask: np.ndarray, mask_type: str) -> None:
        """ Update the mask on an edit for the :class:`~lib.align.DetectedFace` object at
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
        self._globals.var_full_update.set(True)

    def copy(self, frame_index: int, direction: T.Literal["prev", "next"]) -> None:
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
        frames_with_faces = [idx for idx, faces in enumerate(self._detected_faces.current_faces)
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

        # aligned_face cannot be deep copied, so remove and recreate
        to_copy = self._faces_at_frame_index(idx)
        for face in to_copy:
            face._aligned = None  # pylint:disable=protected-access
        copied = deepcopy(to_copy)

        for old_face, new_face in zip(to_copy, copied):
            old_face.load_aligned(None)
            new_face.load_aligned(None)

        faces.extend(copied)
        self._tk_face_count_changed.set(True)
        self._globals.var_full_update.set(True)

    def post_edit_trigger(self, frame_index: int, face_index: int) -> None:
        """ Update the jpg thumbnail, the viewport thumbnail, the landmark masks and the aligned
        face on a face edit.

        Parameters
        ----------
        frame_index: int
            The frame that the face is being set for
        face_index: int
            The face index within the frame
        """
        face = self._frame_faces[frame_index][face_index]
        face.load_aligned(None, force=True)  # Update average distance
        face.mask = self._extractor.get_masks(frame_index, face_index)
        face.clear_all_identities()

        aligned = AlignedFace(face.landmarks_xy,
                              image=self._globals.current_frame.image,
                              centering="head",
                              size=96)
        assert aligned.face is not None
        face.thumbnail = generate_thumbnail(aligned.face, size=96)
        if self._globals.var_filter_mode.get() == "Misaligned Faces":
            self._detected_faces.tk_face_count_changed.set(True)
        self._tk_edited.set(True)
