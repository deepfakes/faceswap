#!/usr/bin/env python3
""" Tools for manipulating the alignments serialized file """
from __future__ import annotations
import logging
import os
import sys
import typing as T

from datetime import datetime

import numpy as np
from scipy import signal
from sklearn import decomposition
from tqdm import tqdm

from lib.logger import parse_class_init
from lib.serializer import get_serializer
from lib.utils import FaceswapError

from .media import Faces, Frames
from .jobs_faces import FaceToFile

if T.TYPE_CHECKING:
    from collections.abc import Generator
    from argparse import Namespace
    from lib.align.alignments import AlignmentFileDict, PNGHeaderDict
    from .media import AlignmentData

logger = logging.getLogger(__name__)


class Check:
    """ Frames and faces checking tasks.

    Parameters
    ---------
    alignments: :class:`tools.alignments.media.AlignmentsData`
        The loaded alignments corresponding to the frames to be annotated
    arguments: :class:`argparse.Namespace`
        The command line arguments that have called this job
    """
    def __init__(self, alignments: AlignmentData, arguments: Namespace) -> None:
        logger.debug(parse_class_init(locals()))
        self._alignments = alignments
        self._job = arguments.job
        self._type: T.Literal["faces", "frames"] | None = None
        self._is_video = False  # Set when getting items
        self._output = arguments.output
        self._source_dir = self._get_source_dir(arguments)
        self._validate()
        self._items = self._get_items()

        self.output_message = ""
        logger.debug("Initialized %s", self.__class__.__name__)

    def _get_source_dir(self, arguments: Namespace) -> str:
        """ Set the correct source folder

        Parameters
        ----------
        arguments: :class:`argparse.Namespace`
            The command line arguments for the Alignments tool

        Returns
        -------
        str
            Full path to the source folder
        """
        if (hasattr(arguments, "faces_dir") and arguments.faces_dir and
                hasattr(arguments, "frames_dir") and arguments.frames_dir):
            logger.error("Only select a source frames (-fr) or source faces (-fc) folder")
            sys.exit(1)
        elif hasattr(arguments, "faces_dir") and arguments.faces_dir:
            self._type = "faces"
            source_dir = arguments.faces_dir
        elif hasattr(arguments, "frames_dir") and arguments.frames_dir:
            self._type = "frames"
            source_dir = arguments.frames_dir
        else:
            logger.error("No source folder (-fr or -fc) was provided")
            sys.exit(1)
        logger.debug("type: '%s', source_dir: '%s'", self._type, source_dir)
        return source_dir

    def _get_items(self) -> list[dict[str, str]] | list[tuple[str, PNGHeaderDict]]:
        """ Set the correct items to process

        Returns
        -------
        list
            Sorted list of dictionaries for either faces or frames. If faces the dictionaries
            have the current filename as key, with the header source data as value. If frames
            the dictionaries will contain the keys 'frame_fullname', 'frame_name', 'extension'.
        """
        assert self._type is not None
        items: Frames | Faces = globals()[self._type.title()](self._source_dir)
        self._is_video = items.is_video
        return T.cast(list[dict[str, str]] | list[tuple[str, "PNGHeaderDict"]],
                      items.file_list_sorted)

    def process(self) -> None:
        """ Process the frames check against the alignments file """
        assert self._type is not None
        logger.info("[CHECK %s]", self._type.upper())
        items_output = self._compile_output()

        if self._type == "faces":
            filelist = T.cast(list[tuple[str, "PNGHeaderDict"]], self._items)
            check_update = FaceToFile(self._alignments, [val[1] for val in filelist])
            if check_update():
                self._alignments.save()

        self._output_results(items_output)

    def _validate(self) -> None:
        """ Check that the selected type is valid for selected task and job """
        if self._job == "missing-frames" and self._output == "move":
            logger.warning("Missing_frames was selected with move output, but there will "
                           "be nothing to move. Defaulting to output: console")
            self._output = "console"
        if self._type == "faces" and self._job != "multi-faces":
            logger.error("The selected folder is not valid. Faces folder (-fc) is only "
                         "supported for 'multi-faces'")
            sys.exit(1)

    def _compile_output(self) -> list[str] | list[tuple[str, int]]:
        """ Compile list of frames that meet criteria

        Returns
        -------
        list
            List of filenames or filenames and face indices for the selected criteria
        """
        action = self._job.replace("-", "_")
        processor = getattr(self, f"_get_{action}")
        logger.debug("Processor: %s", processor)
        return [item for item in processor()]  # pylint:disable=unnecessary-comprehension

    def _get_no_faces(self) -> Generator[str, None, None]:
        """ yield each frame that has no face match in alignments file

        Yields
        ------
        str
            The frame name of any frames which have no faces
        """
        self.output_message = "Frames with no faces"
        for frame in tqdm(T.cast(list[dict[str, str]], self._items),
                          desc=self.output_message,
                          leave=False):
            logger.trace(frame)  # type:ignore
            frame_name = frame["frame_fullname"]
            if not self._alignments.frame_has_faces(frame_name):
                logger.debug("Returning: '%s'", frame_name)
                yield frame_name

    def _get_multi_faces(self) -> (Generator[str, None, None] |
                                   Generator[tuple[str, int], None, None]):
        """ yield each frame or face that has multiple faces matched in alignments file

        Yields
        ------
        str or tuple
            The frame name of any frames which have multiple faces and potentially the face id
        """
        process_type = getattr(self, f"_get_multi_faces_{self._type}")
        for item in process_type():
            yield item

    def _get_multi_faces_frames(self) -> Generator[str, None, None]:
        """ Return Frames that contain multiple faces

        Yields
        ------
        str
            The frame name of any frames which have multiple faces
        """
        self.output_message = "Frames with multiple faces"
        for item in tqdm(T.cast(list[dict[str, str]], self._items),
                         desc=self.output_message,
                         leave=False):
            filename = item["frame_fullname"]
            if not self._alignments.frame_has_multiple_faces(filename):
                continue
            logger.trace("Returning: '%s'", filename)  # type:ignore
            yield filename

    def _get_multi_faces_faces(self) -> Generator[tuple[str, int], None, None]:
        """ Return Faces when there are multiple faces in a frame

        Yields
        ------
        tuple
            The frame name and the face id of any frames which have multiple faces
        """
        self.output_message = "Multiple faces in frame"
        for item in tqdm(T.cast(list[tuple[str, "PNGHeaderDict"]], self._items),
                         desc=self.output_message,
                         leave=False):
            src = item[1]["source"]
            if not self._alignments.frame_has_multiple_faces(src["source_filename"]):
                continue
            retval = (item[0], src["face_index"])
            logger.trace("Returning: '%s'", retval)  # type:ignore
            yield retval

    def _get_missing_alignments(self) -> Generator[str, None, None]:
        """ yield each frame that does not exist in alignments file

        Yields
        ------
        str
            The frame name of any frames missing alignments
        """
        self.output_message = "Frames missing from alignments file"
        exclude_filetypes = set(["yaml", "yml", "p", "json", "txt"])
        for frame in tqdm(T.cast(dict[str, str], self._items),
                          desc=self.output_message,
                          leave=False):
            frame_name = frame["frame_fullname"]
            if (frame["frame_extension"] not in exclude_filetypes
                    and not self._alignments.frame_exists(frame_name)):
                logger.debug("Returning: '%s'", frame_name)
                yield frame_name

    def _get_missing_frames(self) -> Generator[str, None, None]:
        """ yield each frame in alignments that does not have a matching file

        Yields
        ------
        str
            The frame name of any frames in alignments with no matching file
        """
        self.output_message = "Missing frames that are in alignments file"
        frames = set(item["frame_fullname"] for item in T.cast(list[dict[str, str]], self._items))
        for frame in tqdm(self._alignments.data.keys(), desc=self.output_message, leave=False):
            if frame not in frames:
                logger.debug("Returning: '%s'", frame)
                yield frame

    def _output_results(self, items_output: list[str] | list[tuple[str, int]]) -> None:
        """ Output the results in the requested format

        Parameters
        ----------
        items_output
            The list of frame names, and potentially face ids, of any items which met the
            selection criteria
        """
        logger.trace("items_output: %s", items_output)  # type:ignore
        if self._output == "move" and self._is_video and self._type == "frames":
            logger.warning("Move was selected with an input video. This is not possible so "
                           "falling back to console output")
            self._output = "console"
        if not items_output:
            logger.info("No %s were found meeting the criteria", self._type)
            return
        if self._output == "move":
            self._move_file(items_output)
            return
        if self._job == "multi-faces" and self._type == "faces":
            # Strip the index for printed/file output
            final_output = [item[0] for item in items_output]
        else:
            final_output = T.cast(list[str], items_output)
        output_message = "-----------------------------------------------\r\n"
        output_message += f" {self.output_message} ({len(final_output)})\r\n"
        output_message += "-----------------------------------------------\r\n"
        output_message += "\r\n".join(final_output)
        if self._output == "console":
            for line in output_message.splitlines():
                logger.info(line)
        if self._output == "file":
            self.output_file(output_message, len(final_output))

    def _get_output_folder(self) -> str:
        """ Return output folder. Needs to be in the root if input is a video and processing
        frames

        Returns
        -------
        str
            Full path to the output folder
        """
        if self._is_video and self._type == "frames":
            return os.path.dirname(self._source_dir)
        return self._source_dir

    def _get_filename_prefix(self) -> str:
        """ Video name needs to be prefixed to filename if input is a video and processing frames

        Returns
        -------
        str
            The common filename prefix to use
        """
        if self._is_video and self._type == "frames":
            return f"{os.path.basename(self._source_dir)}_"
        return ""

    def output_file(self, output_message: str, items_discovered: int) -> None:
        """ Save the output to a text file in the frames directory

        Parameters
        ----------
        output_message: str
            The message to write out to file
        items_discovered: int
            The number of items which matched the criteria
        """
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst_dir = self._get_output_folder()
        filename = (f"{self._get_filename_prefix()}{self.output_message.replace(' ', '_').lower()}"
                    f"_{now}.txt")
        output_file = os.path.join(dst_dir, filename)
        logger.info("Saving %s result(s) to '%s'", items_discovered, output_file)
        with open(output_file, "w", encoding="utf8") as f_output:
            f_output.write(output_message)

    def _move_file(self, items_output: list[str] | list[tuple[str, int]]) -> None:
        """ Move the identified frames to a new sub folder

        Parameters
        ----------
        items_output: list
            List of items to move
        """
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = (f"{self._get_filename_prefix()}"
                       f"{self.output_message.replace(' ','_').lower()}_{now}")
        dst_dir = self._get_output_folder()
        output_folder = os.path.join(dst_dir, folder_name)
        logger.debug("Creating folder: '%s'", output_folder)
        os.makedirs(output_folder)
        move = getattr(self, f"_move_{self._type}")
        logger.debug("Move function: %s", move)
        move(output_folder, items_output)

    def _move_frames(self, output_folder: str, items_output: list[str]) -> None:
        """ Move frames into single sub folder

        Parameters
        ----------
        output_folder: str
            The folder to move the output to
        items_output: list
            List of items to move
        """
        logger.info("Moving %s frame(s) to '%s'", len(items_output), output_folder)
        for frame in items_output:
            src = os.path.join(self._source_dir, frame)
            dst = os.path.join(output_folder, frame)
            logger.debug("Moving: '%s' to '%s'", src, dst)
            os.rename(src, dst)

    def _move_faces(self, output_folder: str, items_output: list[tuple[str, int]]) -> None:
        """ Make additional sub folders for each face that appears Enables easier manual sorting

        Parameters
        ----------
        output_folder: str
            The folder to move the output to
        items_output: list
            List of items and face indices to move
        """
        logger.info("Moving %s faces(s) to '%s'", len(items_output), output_folder)
        for frame, idx in items_output:
            src = os.path.join(self._source_dir, frame)
            dst_folder = os.path.join(output_folder, str(idx)) if idx != -1 else output_folder
            if not os.path.isdir(dst_folder):
                logger.debug("Creating folder: '%s'", dst_folder)
                os.makedirs(dst_folder)
            dst = os.path.join(dst_folder, frame)
            logger.debug("Moving: '%s' to '%s'", src, dst)
            os.rename(src, dst)


class Export:
    """ Export alignments from a Faceswap .fsa file to a json formatted file.

        Parameters
    ----------
    alignments: :class:`tools.lib_alignments.media.AlignmentData`
        The alignments data loaded from an alignments file for this rename job
    arguments: :class:`argparse.Namespace`
        The :mod:`argparse` arguments as passed in from :mod:`tools.py`. Unused
    """
    def __init__(self,
                 alignments: AlignmentData,
                 arguments: Namespace) -> None:  # pylint:disable=unused-argument
        logger.debug(parse_class_init(locals()))
        self._alignments = alignments
        self._serializer = get_serializer("json")
        self._output_file = self._get_output_file()
        logger.debug("Initialized %s", self.__class__.__name__)

    def _get_output_file(self) -> str:
        """ Obtain the name of an output file. If a file of the request name exists, then append a
        digit to the end until a unique filename is found

        Returns
        -------
        str
            Full path to an output json file
        """
        in_file = self._alignments.file
        base_filename = f"{os.path.splitext(in_file)[0]}_export"
        out_file = f"{base_filename}.json"
        idx = 1
        while True:
            if not os.path.exists(out_file):
                break
            logger.debug("Output file exists: '%s'", out_file)
            out_file = f"{base_filename}_{idx}.json"
            idx += 1
        logger.debug("Setting output file to '%s'", out_file)
        return out_file

    @classmethod
    def _format_face(cls, face: AlignmentFileDict) -> dict[str, list[int] | list[list[float]]]:
        """ Format the relevant keys from an alignment file's face into the correct format for
        export/import

        Parameters
        ----------
        face: :class:`~lib.align.alignments.AlignmentFileDict`
            The alignment dictionary for a face to process

        Returns
        -------
        dict[str, list[int] | list[list[float]]]
            The face formatted for exporting to a json file
        """
        lms = face["landmarks_xy"]
        assert isinstance(lms, np.ndarray)
        retval = {"detected": [int(round(face["x"], 0)),
                               int(round(face["y"], 0)),
                               int(round(face["x"] + face["w"], 0)),
                               int(round(face["y"] + face["h"], 0))],
                  "landmarks_2d": lms.tolist()}
        return retval

    def process(self) -> None:
        """ Parse the imported alignments file and output relevant information to a json file """
        logger.info("[EXPORTING ALIGNMENTS]")  # Tidy up cli output
        formatted = {key: [self._format_face(face) for face in val["faces"]]
                     for key, val in self._alignments.data.items()}
        logger.info("Saving export alignments to '%s'...", self._output_file)
        self._serializer.save(self._output_file, formatted)


class Sort:
    """ Sort alignments' index by the order they appear in an image in left to right order.

    Parameters
    ----------
    alignments: :class:`tools.lib_alignments.media.AlignmentData`
        The alignments data loaded from an alignments file for this rename job
    arguments: :class:`argparse.Namespace`
        The :mod:`argparse` arguments as passed in from :mod:`tools.py`. Unused
    """
    def __init__(self,
                 alignments: AlignmentData,
                 arguments: Namespace) -> None:  # pylint:disable=unused-argument
        logger.debug(parse_class_init(locals()))
        self._alignments = alignments
        logger.debug("Initialized %s", self.__class__.__name__)

    def process(self) -> None:
        """ Execute the sort process """
        logger.info("[SORT INDEXES]")  # Tidy up cli output
        reindexed = self.reindex_faces()
        if reindexed:
            self._alignments.save()
            logger.warning("If you have a face-set corresponding to the alignment file you "
                           "processed then you should run the 'Extract' job to regenerate it.")

    def reindex_faces(self) -> int:
        """ Re-Index the faces """
        reindexed = 0
        for alignment in tqdm(self._alignments.yield_faces(),
                              desc="Sort alignment indexes",
                              total=self._alignments.frames_count,
                              leave=False):
            frame, alignments, count, key = alignment
            if count <= 1:
                logger.trace("0 or 1 face in frame. Not sorting: '%s'", frame)  # type:ignore
                continue
            sorted_alignments = sorted(alignments, key=lambda x: (x["x"]))
            if sorted_alignments == alignments:
                logger.trace("Alignments already in correct order. Not "  # type:ignore
                             "sorting: '%s'", frame)
                continue
            logger.trace("Sorting alignments for frame: '%s'", frame)  # type:ignore
            self._alignments.data[key]["faces"] = sorted_alignments
            reindexed += 1
        logger.info("%s Frames had their faces reindexed", reindexed)
        return reindexed


class Spatial:
    """ Apply spatial temporal filtering to landmarks

    Parameters
    ----------
    alignments: :class:`tools.lib_alignments.media.AlignmentData`
        The alignments data loaded from an alignments file for this rename job
    arguments: :class:`argparse.Namespace`
        The :mod:`argparse` arguments as passed in from :mod:`tools.py`

    Reference
    ---------
    https://www.kaggle.com/selfishgene/animating-and-smoothing-3d-facial-keypoints/notebook
    """
    def __init__(self, alignments: AlignmentData, arguments: Namespace) -> None:
        logger.debug(parse_class_init(locals()))
        self.arguments = arguments
        self._alignments = alignments
        self._mappings: dict[int, str] = {}
        self._normalized: dict[str, np.ndarray] = {}
        self._shapes_model: decomposition.PCA | None = None
        logger.debug("Initialized %s", self.__class__.__name__)

    def process(self) -> None:
        """ Perform spatial filtering """
        logger.info("[SPATIO-TEMPORAL FILTERING]")  # Tidy up cli output
        logger.info("NB: The process only processes the alignments for the first "
                    "face it finds for any given frame. For best results only run this when "
                    "there is only a single face in the alignments file and all false positives "
                    "have been removed")

        self._normalize()
        self._shape_model()
        landmarks = self._spatially_filter()
        landmarks = self._temporally_smooth(landmarks)
        self._update_alignments(landmarks)
        self._alignments.save()
        logger.warning("If you have a face-set corresponding to the alignment file you "
                       "processed then you should run the 'Extract' job to regenerate it.")

    # Define shape normalization utility functions
    @staticmethod
    def _normalize_shapes(shapes_im_coords: np.ndarray
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Normalize a 2D or 3D shape

        Parameters
        ----------
        shaped_im_coords: :class:`numpy.ndarray`
            The facial landmarks

        Returns
        -------
        shapes_normalized: :class:`numpy.ndarray`
            The normalized shapes
        scale_factors: :class:`numpy.ndarray`
            The scale factors
        mean_coords: :class:`numpy.ndarray`
            The mean coordinates
        """
        logger.debug("Normalize shapes")
        (num_pts, num_dims, _) = shapes_im_coords.shape

        # Calculate mean coordinates and subtract from shapes
        mean_coords = shapes_im_coords.mean(axis=0)
        shapes_centered = np.zeros(shapes_im_coords.shape)
        shapes_centered = shapes_im_coords - np.tile(mean_coords, [num_pts, 1, 1])

        # Calculate scale factors and divide shapes
        scale_factors = np.sqrt((shapes_centered**2).sum(axis=1)).mean(axis=0)
        shapes_normalized = np.zeros(shapes_centered.shape)
        shapes_normalized = shapes_centered / np.tile(scale_factors, [num_pts, num_dims, 1])

        logger.debug("Normalized shapes: (shapes_normalized: %s, scale_factors: %s, mean_coords: "
                     "%s", shapes_normalized, scale_factors, mean_coords)
        return shapes_normalized, scale_factors, mean_coords

    @staticmethod
    def _normalized_to_original(shapes_normalized: np.ndarray,
                                scale_factors: np.ndarray,
                                mean_coords: np.ndarray) -> np.ndarray:
        """ Transform a normalized shape back to original image coordinates

        Parameters
        ----------
        shapes_normalized: :class:`numpy.ndarray`
            The normalized shapes
        scale_factors: :class:`numpy.ndarray`
            The scale factors
        mean_coords: :class:`numpy.ndarray`
            The mean coordinates

        Returns
        -------
        :class:`numpy.ndarray`
            The normalized shape transformed back to original coordinates
        """
        logger.debug("Normalize to original")
        (num_pts, num_dims, _) = shapes_normalized.shape

        # move back to the correct scale
        shapes_centered = shapes_normalized * np.tile(scale_factors, [num_pts, num_dims, 1])
        # move back to the correct location
        shapes_im_coords = shapes_centered + np.tile(mean_coords, [num_pts, 1, 1])

        logger.debug("Normalized to original: %s", shapes_im_coords)
        return shapes_im_coords

    def _normalize(self) -> None:
        """ Compile all original and normalized alignments """
        logger.debug("Normalize")
        count = sum(1 for val in self._alignments.data.values() if val["faces"])

        sample_lm = next((val["faces"][0]["landmarks_xy"]
                          for val in self._alignments.data.values() if val["faces"]), 68)
        assert isinstance(sample_lm, np.ndarray)
        lm_count = sample_lm.shape[0]
        if lm_count != 68:
            raise FaceswapError("Spatial smoothing only supports 68 point facial landmarks")

        landmarks_all = np.zeros((lm_count, 2, int(count)))

        end = 0
        for key in tqdm(sorted(self._alignments.data.keys()), desc="Compiling", leave=False):
            val = self._alignments.data[key]["faces"]
            if not val:
                continue
            # We should only be normalizing a single face, so just take
            # the first landmarks found
            landmarks = np.array(val[0]["landmarks_xy"]).reshape((lm_count, 2, 1))
            start = end
            end = start + landmarks.shape[2]
            # Store in one big array
            landmarks_all[:, :, start:end] = landmarks
            # Make sure we keep track of the mapping to the original frame
            self._mappings[start] = key

        # Normalize shapes
        normalized_shape = self._normalize_shapes(landmarks_all)
        self._normalized["landmarks"] = normalized_shape[0]
        self._normalized["scale_factors"] = normalized_shape[1]
        self._normalized["mean_coords"] = normalized_shape[2]
        logger.debug("Normalized: %s", self._normalized)

    def _shape_model(self) -> None:
        """ build 2D shape model """
        logger.debug("Shape model")
        landmarks_norm = self._normalized["landmarks"]
        num_components = 20
        normalized_shapes_tbl = np.reshape(landmarks_norm, [68*2, landmarks_norm.shape[2]]).T
        self._shapes_model = decomposition.PCA(n_components=num_components,
                                               whiten=True,
                                               random_state=1).fit(normalized_shapes_tbl)
        explained = self._shapes_model.explained_variance_ratio_.sum()
        logger.info("Total explained percent by PCA model with %s components is %s%%",
                    num_components, round(100 * explained, 1))
        logger.debug("Shaped model")

    def _spatially_filter(self) -> np.ndarray:
        """ interpret the shapes using our shape model (project and reconstruct)

        Returns
        -------
        :class:`numpy.ndarray`
            The filtered landmarks in original coordinate space
        """
        logger.debug("Spatially Filter")
        assert self._shapes_model is not None
        landmarks_norm = self._normalized["landmarks"]
        # Convert to matrix form
        landmarks_norm_table = np.reshape(landmarks_norm, [68 * 2, landmarks_norm.shape[2]]).T
        # Project onto shapes model and reconstruct
        landmarks_norm_table_rec = self._shapes_model.inverse_transform(
            self._shapes_model.transform(landmarks_norm_table))
        # Convert back to shapes (numKeypoint, num_dims, numFrames)
        landmarks_norm_rec = np.reshape(landmarks_norm_table_rec.T,
                                        [68, 2, landmarks_norm.shape[2]])
        # Transform back to image co-ordinates
        retval = self._normalized_to_original(landmarks_norm_rec,
                                              self._normalized["scale_factors"],
                                              self._normalized["mean_coords"])

        logger.debug("Spatially Filtered: %s", retval)
        return retval

    @staticmethod
    def _temporally_smooth(landmarks: np.ndarray) -> np.ndarray:
        """ apply temporal filtering on the 2D points

        Parameters
        ----------
        landmarks: :class:`numpy.ndarray`
            68 point landmarks to be temporally smoothed

        Returns
        -------
        :class: `numpy.ndarray`
            The temporally smoothed landmarks
        """
        logger.debug("Temporally Smooth")
        filter_half_length = 2
        temporal_filter = np.ones((1, 1, 2 * filter_half_length + 1))
        temporal_filter = temporal_filter / temporal_filter.sum()

        start_tileblock = np.tile(landmarks[:, :, 0][:, :, np.newaxis], [1, 1, filter_half_length])
        end_tileblock = np.tile(landmarks[:, :, -1][:, :, np.newaxis], [1, 1, filter_half_length])
        landmarks_padded = np.dstack((start_tileblock, landmarks, end_tileblock))

        retval = signal.convolve(landmarks_padded, temporal_filter, mode='valid', method='fft')
        logger.debug("Temporally Smoothed: %s", retval)
        return retval

    def _update_alignments(self, landmarks: np.ndarray) -> None:
        """ Update smoothed landmarks back to alignments

        Parameters
        ----------
        landmarks: :class:`numpy.ndarray`
            The smoothed landmarks
        """
        logger.debug("Update alignments")
        for idx, frame in tqdm(self._mappings.items(), desc="Updating", leave=False):
            logger.trace("Updating: (frame: %s)", frame)  # type:ignore
            landmarks_update = landmarks[:, :, idx]
            landmarks_xy = landmarks_update.reshape(68, 2).tolist()
            self._alignments.data[frame]["faces"][0]["landmarks_xy"] = landmarks_xy
            logger.trace("Updated: (frame: '%s', landmarks: %s)",  # type:ignore
                         frame, landmarks_xy)
        logger.debug("Updated alignments")
