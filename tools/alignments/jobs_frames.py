#!/usr/bin/env python3
""" Tools for manipulating the alignments using Frames as a source """
from __future__ import annotations
import logging
import os
import sys
import typing as T

from datetime import datetime

import cv2
import numpy as np
from tqdm import tqdm

from lib.align import DetectedFace, EXTRACT_RATIOS, LANDMARK_PARTS, LandmarkType
from lib.align.alignments import _VERSION, PNGHeaderDict
from lib.image import encode_image, generate_thumbnail, ImagesSaver
from plugins.extract import ExtractMedia, Extractor
from .media import ExtractedFaces, Frames

if T.TYPE_CHECKING:
    from argparse import Namespace
    from .media import AlignmentData

logger = logging.getLogger(__name__)


class Draw():
    """ Draws annotations onto original frames and saves into a sub-folder next to the original
    frames.

    Parameters
    ---------
    alignments: :class:`tools.alignments.media.AlignmentsData`
        The loaded alignments corresponding to the frames to be annotated
    arguments: :class:`argparse.Namespace`
        The command line arguments that have called this job
    """
    def __init__(self, alignments: AlignmentData, arguments: Namespace) -> None:
        logger.debug("Initializing %s: (arguments: %s)", self.__class__.__name__, arguments)
        self._alignments = alignments
        self._frames = Frames(arguments.frames_dir)
        self._output_folder = self._set_output()
        logger.debug("Initialized %s", self.__class__.__name__)

    def _set_output(self) -> str:
        """ Set the output folder path.

        If annotating a folder of frames, output will be placed in a sub folder within the frames
        folder. If annotating a video, output will be a folder next to the original video.

        Returns
        -------
        str
            Full path to the output folder

        """
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"drawn_landmarks_{now}"
        if self._frames.is_video:
            dest_folder = os.path.dirname(self._frames.folder)
        else:
            dest_folder = self._frames.folder
        output_folder = os.path.join(dest_folder, folder_name)
        logger.debug("Creating folder: '%s'", output_folder)
        os.makedirs(output_folder)
        return output_folder

    def process(self) -> None:
        """ Runs the process to draw face annotations onto original source frames. """
        logger.info("[DRAW LANDMARKS]")  # Tidy up cli output
        frames_drawn = 0
        for frame in tqdm(self._frames.file_list_sorted, desc="Drawing landmarks", leave=False):
            frame_name = frame["frame_fullname"]

            if not self._alignments.frame_exists(frame_name):
                logger.verbose("Skipping '%s' - Alignments not found", frame_name)  # type:ignore
                continue

            self._annotate_image(frame_name)
            frames_drawn += 1
        logger.info("%s Frame(s) output", frames_drawn)

    def _annotate_image(self, frame_name: str) -> None:
        """ Annotate the frame with each face that appears in the alignments file.

        Parameters
        ----------
        frame_name: str
            The full path to the original frame
        """
        logger.trace("Annotating frame: '%s'", frame_name)  # type:ignore
        image = self._frames.load_image(frame_name)

        for idx, alignment in enumerate(self._alignments.get_faces_in_frame(frame_name)):
            face = DetectedFace()
            face.from_alignment(alignment, image=image)
            # Bounding Box
            assert face.left is not None
            assert face.top is not None
            cv2.rectangle(image, (face.left, face.top), (face.right, face.bottom), (255, 0, 0), 1)
            self._annotate_landmarks(image, np.rint(face.landmarks_xy).astype("int32"))
            self._annotate_extract_boxes(image, face, idx)
            self._annotate_pose(image, face)  # Pose (head is still loaded)

        self._frames.save_image(self._output_folder, frame_name, image)

    def _annotate_landmarks(self, image: np.ndarray, landmarks: np.ndarray) -> None:
        """ Annotate the extract boxes onto the frame.

        Parameters
        ----------
        image: :class:`numpy.ndarray`
            The frame that extract boxes are to be annotated on to
        landmarks: :class:`numpy.ndarray`
            The facial landmarks that are to be annotated onto the frame
        """
        # Mesh
        for start, end, fill in LANDMARK_PARTS[LandmarkType.from_shape(landmarks.shape)].values():
            cv2.polylines(image, [landmarks[start:end]], fill, (255, 255, 0), 1)
        # Landmarks
        for (pos_x, pos_y) in landmarks:
            cv2.circle(image, (pos_x, pos_y), 1, (0, 255, 255), -1)

    @classmethod
    def _annotate_extract_boxes(cls, image: np.ndarray, face: DetectedFace, index: int) -> None:
        """ Annotate the mesh and landmarks boxes onto the frame.

        Parameters
        ----------
        image: :class:`numpy.ndarray`
            The frame that mesh and landmarks are to be annotated on to
        face: :class:`lib.align.DetectedFace`
            The aligned face
        index: int
            The face index for the given face
        """
        for area in T.get_args(T.Literal["face", "head"]):
            face.load_aligned(image, centering=area, force=True)
            color = (0, 255, 0) if area == "face" else (0, 0, 255)
            top_left = face.aligned.original_roi[0]
            top_left = (top_left[0], top_left[1] - 10)
            cv2.putText(image, str(index), top_left, cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 1)
            cv2.polylines(image, [face.aligned.original_roi], True, color, 1)

    @classmethod
    def _annotate_pose(cls, image: np.ndarray, face: DetectedFace) -> None:
        """ Annotate the pose onto the frame.

        Parameters
        ----------
        image: :class:`numpy.ndarray`
            The frame that pose is to be annotated on to
        face: :class:`lib.align.DetectedFace`
            The aligned face loaded for head centering
        """
        center = np.array((face.aligned.size / 2,
                           face.aligned.size / 2)).astype("int32").reshape(1, 2)
        center = np.rint(face.aligned.transform_points(center, invert=True)).astype("int32")
        points = face.aligned.pose.xyz_2d * face.aligned.size
        points = np.rint(face.aligned.transform_points(points, invert=True)).astype("int32")
        cv2.line(image, tuple(center), tuple(points[1]), (0, 255, 0), 2)
        cv2.line(image, tuple(center), tuple(points[0]), (255, 0, 0), 2)
        cv2.line(image, tuple(center), tuple(points[2]), (0, 0, 255), 2)


class Extract():
    """ Re-extract faces from source frames based on Alignment data

    Parameters
    ----------
    alignments: :class:`tools.lib_alignments.media.AlignmentData`
        The alignments data loaded from an alignments file for this rename job
    arguments: :class:`argparse.Namespace`
        The :mod:`argparse` arguments as passed in from :mod:`tools.py`
    """
    def __init__(self, alignments: AlignmentData, arguments: Namespace) -> None:
        logger.debug("Initializing %s: (arguments: %s)", self.__class__.__name__, arguments)
        self._arguments = arguments
        self._alignments = alignments
        self._is_legacy = self._alignments.version == 1.0  # pylint:disable=protected-access
        self._mask_pipeline: Extractor | None = None
        self._faces_dir = arguments.faces_dir
        self._min_size = self._get_min_size(arguments.size, arguments.min_size)

        self._frames = Frames(arguments.frames_dir, self._get_count())
        self._extracted_faces = ExtractedFaces(self._frames,
                                               self._alignments,
                                               size=arguments.size)
        self._saver: ImagesSaver | None = None
        logger.debug("Initialized %s", self.__class__.__name__)

    @classmethod
    def _get_min_size(cls, extract_size: int, min_size: int) -> int:
        """ Obtain the minimum size that a face has been resized from to be included as a valid
        extract.

        Parameters
        ----------
        extract_size: int
            The requested size of the extracted images
        min_size: int
            The percentage amount that has been supplied for valid faces (as a percentage of
            extract size)

        Returns
        -------
        int
            The minimum size, in pixels, that a face is resized from to be considered valid
        """
        retval = 0 if min_size == 0 else max(4, int(extract_size * (min_size / 100.)))
        logger.debug("Extract size: %s, min percentage size: %s, min_size: %s",
                     extract_size, min_size, retval)
        return retval

    def _get_count(self) -> int | None:
        """ If the alignments file has been run through the manual tool, then it will hold video
        meta information, meaning that the count of frames in the alignment file can be relied
        on to be accurate.

        Returns
        -------
        int or ``None``
        For video input which contain video meta-data in the alignments file then the count of
        frames is returned. In all other cases ``None`` is returned
        """
        meta = self._alignments.video_meta_data
        has_meta = all(val is not None for val in meta.values())
        if has_meta:
            retval: int | None = len(T.cast(dict[str, list[int] | list[float]], meta["pts_time"]))
        else:
            retval = None
        logger.debug("Frame count from alignments file: (has_meta: %s, %s", has_meta, retval)
        return retval

    def process(self) -> None:
        """ Run the re-extraction from Alignments file process"""
        logger.info("[EXTRACT FACES]")  # Tidy up cli output
        self._check_folder()
        if self._is_legacy:
            self._legacy_check()
        self._saver = ImagesSaver(self._faces_dir, as_bytes=True)

        if self._min_size > 0:
            logger.info("Only selecting faces that have been resized from a minimum resolution "
                        "of %spx", self._min_size)

        self._export_faces()

    def _check_folder(self) -> None:
        """ Check that the faces folder doesn't pre-exist and create. """
        err = None
        if not self._faces_dir:
            err = "ERROR: Output faces folder not provided."
        elif not os.path.isdir(self._faces_dir):
            logger.debug("Creating folder: '%s'", self._faces_dir)
            os.makedirs(self._faces_dir)
        elif os.listdir(self._faces_dir):
            err = f"ERROR: Output faces folder should be empty: '{self._faces_dir}'"
        if err:
            logger.error(err)
            sys.exit(0)
        logger.verbose("Creating output folder at '%s'", self._faces_dir)  # type:ignore

    def _legacy_check(self) -> None:
        """ Check whether the alignments file was created with the legacy extraction method.

        If so, force user to re-extract all faces if any options have been specified, otherwise
        raise the appropriate warnings and set the legacy options.
        """
        if self._min_size > 0 or self._arguments.extract_every_n != 1:
            logger.warning("This alignments file was generated with the legacy extraction method.")
            logger.warning("You should run this extraction job, but with 'min_size' set to 0 and "
                           "'extract-every-n' set to 1 to update the alignments file.")
            logger.warning("You can then re-run this extraction job with your chosen options.")
            sys.exit(0)

        maskers = ["components", "extended"]
        nn_masks = [mask for mask in list(self._alignments.mask_summary) if mask not in maskers]
        logtype = logger.warning if nn_masks else logger.info
        logtype("This alignments file was created with the legacy extraction method and will be "
                "updated.")
        logtype("Faces will be extracted using the new method and landmarks based masks will be "
                "regenerated.")
        if nn_masks:
            logtype("However, the NN based masks '%s' will be cropped to the legacy extraction "
                    "method, so you may want to run the mask tool to regenerate these "
                    "masks.", "', '".join(nn_masks))
        self._mask_pipeline = Extractor(None, None, maskers, multiprocess=True)
        self._mask_pipeline.launch()
        # Update alignments versioning
        self._alignments._io._version = _VERSION  # pylint:disable=protected-access

    def _export_faces(self) -> None:
        """ Export the faces to the output folder. """
        extracted_faces = 0
        skip_list = self._set_skip_list()
        count = self._frames.count if skip_list is None else self._frames.count - len(skip_list)

        for filename, image in tqdm(self._frames.stream(skip_list=skip_list),
                                    total=count, desc="Saving extracted faces",
                                    leave=False):
            frame_name = os.path.basename(filename)
            if not self._alignments.frame_exists(frame_name):
                logger.verbose("Skipping '%s' - Alignments not found", frame_name)  # type:ignore
                continue
            extracted_faces += self._output_faces(frame_name, image)
        if self._is_legacy and extracted_faces != 0 and self._min_size == 0:
            self._alignments.save()
        logger.info("%s face(s) extracted", extracted_faces)

    def _set_skip_list(self) -> list[int] | None:
        """ Set the indices for frames that should be skipped based on the `extract_every_n`
        command line option.

        Returns
        -------
        list or ``None``
            A list of indices to be skipped if extract_every_n is not `1` otherwise
            returns ``None``
        """
        skip_num = self._arguments.extract_every_n
        if skip_num == 1:
            logger.debug("Not skipping any frames")
            return None
        skip_list = []
        for idx, item in enumerate(T.cast(list[dict[str, str]], self._frames.file_list_sorted)):
            if idx % skip_num != 0:
                logger.trace("Adding image '%s' to skip list due to "  # type:ignore
                             "extract_every_n = %s", item["frame_fullname"], skip_num)
                skip_list.append(idx)
        logger.debug("Adding skip list: %s", skip_list)
        return skip_list

    def _output_faces(self, filename: str, image: np.ndarray) -> int:
        """ For each frame save out the faces

        Parameters
        ----------
        filename: str
            The filename (without the full path) of the current frame
        image: :class:`numpy.ndarray`
            The full frame that faces are to be extracted from

        Returns
        -------
        int
            The total number of faces that have been extracted
        """
        logger.trace("Outputting frame: %s", filename)  # type:ignore
        face_count = 0
        frame_name = os.path.splitext(filename)[0]
        faces = self._select_valid_faces(filename, image)
        assert self._saver is not None
        if not faces:
            return face_count
        if self._is_legacy:
            faces = self._process_legacy(filename, image, faces)

        for idx, face in enumerate(faces):
            output = f"{frame_name}_{idx}.png"
            meta: PNGHeaderDict = {
                "alignments": face.to_png_meta(),
                "source": {"alignments_version": self._alignments.version,
                           "original_filename": output,
                           "face_index": idx,
                           "source_filename": filename,
                           "source_is_video": self._frames.is_video,
                           "source_frame_dims": T.cast(tuple[int, int], image.shape[:2])}}
            assert face.aligned.face is not None
            self._saver.save(output, encode_image(face.aligned.face, ".png", metadata=meta))
            if self._min_size == 0 and self._is_legacy:
                face.thumbnail = generate_thumbnail(face.aligned.face, size=96, quality=60)
                self._alignments.data[filename]["faces"][idx] = face.to_alignment()
            face_count += 1
        self._saver.close()
        return face_count

    def _select_valid_faces(self, frame: str, image: np.ndarray) -> list[DetectedFace]:
        """ Return the aligned faces from a frame that meet the selection criteria,

        Parameters
        ----------
        frame: str
            The filename (without the full path) of the current frame
        image: :class:`numpy.ndarray`
            The full frame that faces are to be extracted from

        Returns
        -------
        list:
            List of valid :class:`lib,align.DetectedFace` objects
        """
        faces = self._extracted_faces.get_faces_in_frame(frame, image=image)
        if self._min_size == 0:
            valid_faces = faces
        else:
            sizes = self._extracted_faces.get_roi_size_for_frame(frame)
            valid_faces = [faces[idx] for idx, size in enumerate(sizes)
                           if size >= self._min_size]
        logger.trace("frame: '%s', total_faces: %s, valid_faces: %s",  # type:ignore
                     frame, len(faces), len(valid_faces))
        return valid_faces

    def _process_legacy(self,
                        filename: str,
                        image: np.ndarray,
                        detected_faces: list[DetectedFace]) -> list[DetectedFace]:
        """ Process legacy face extractions to new extraction method.

        Updates stored masks to new extract size

        Parameters
        ----------
        filename: str
            The current frame filename
        image: :class:`numpy.ndarray`
            The current image the contains the faces
        detected_faces: list
            list of :class:`lib.align.DetectedFace` objects for the current frame

        Returns
        -------
        list
            The updated list of :class:`lib.align.DetectedFace` objects for the current frame
        """
        # Update landmarks based masks for face centering
        assert self._mask_pipeline is not None
        mask_item = ExtractMedia(filename, image, detected_faces=detected_faces)
        self._mask_pipeline.input_queue.put(mask_item)
        faces = next(self._mask_pipeline.detected_faces()).detected_faces

        # Pad and shift Neural Network based masks to face centering
        for face in faces:
            self._pad_legacy_masks(face)
        return faces

    @classmethod
    def _pad_legacy_masks(cls, detected_face: DetectedFace) -> None:
        """ Recenter legacy Neural Network based masks from legacy centering to face centering
        and pad accordingly.

        Update the masks back into the detected face objects.

        Parameters
        ----------
        detected_face: :class:`lib.align.DetectedFace`
            The detected face to update the masks for
        """
        offset = detected_face.aligned.pose.offset["face"]
        for name, mask in detected_face.mask.items():  # Re-center mask and pad to face size
            if name in ("components", "extended"):
                continue
            old_mask = mask.mask.astype("float32") / 255.0
            size = old_mask.shape[0]
            new_size = int(size + (size * EXTRACT_RATIOS["face"]) / 2)

            shift = np.rint(offset * (size - (size * EXTRACT_RATIOS["face"]))).astype("int32")
            pos = np.array([(new_size // 2 - size // 2) - shift[1],
                            (new_size // 2) + (size // 2) - shift[1],
                            (new_size // 2 - size // 2) - shift[0],
                            (new_size // 2) + (size // 2) - shift[0]])
            bounds = np.array([max(0, pos[0]), min(new_size, pos[1]),
                               max(0, pos[2]), min(new_size, pos[3])])

            slice_in = [slice(0 - (pos[0] - bounds[0]), size - (pos[1] - bounds[1])),
                        slice(0 - (pos[2] - bounds[2]), size - (pos[3] - bounds[3]))]
            slice_out = [slice(bounds[0], bounds[1]), slice(bounds[2], bounds[3])]

            new_mask = np.zeros((new_size, new_size, 1), dtype="float32")
            new_mask[slice_out[0], slice_out[1], :] = old_mask[slice_in[0], slice_in[1], :]

            mask.replace_mask(new_mask)
            # Get the affine matrix from recently generated components mask
            # pylint:disable=protected-access
            mask._affine_matrix = detected_face.mask["components"].affine_matrix
