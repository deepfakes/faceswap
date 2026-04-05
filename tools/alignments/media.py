#!/usr/bin/env python3
"""Media items (Alignments, Faces, Frames) for alignments tool"""
from __future__ import annotations
import logging
from operator import itemgetter
import os
import sys
import typing as T

import cv2
from tqdm import tqdm

from lib.align import Alignments, DetectedFace
from lib.align.objects import PNGHeader
from lib.image import (generate_thumbnail, ImagesLoader, png_write_meta, read_image,
                       read_image_meta_batch)
from lib.utils import get_module_objects, IMAGE_EXTENSIONS
from lib.video import count_frames, VIDEO_EXTENSIONS

if T.TYPE_CHECKING:
    from collections.abc import Generator
    import numpy as np
    from lib.align.objects import FileAlignments

logger = logging.getLogger(__name__)


class AlignmentData(Alignments):
    """Class to hold the alignment data

    Parameters
    ----------
    alignments_file
        Full path to an alignments file
    """
    def __init__(self, alignments_file: str) -> None:
        logger.debug("Initializing %s: (alignments file: '%s')",
                     self.__class__.__name__, alignments_file)
        logger.info("[ALIGNMENT DATA]")  # Tidy up cli output
        folder, filename = self.check_file_exists(alignments_file)
        super().__init__(folder, filename=filename)
        logger.verbose("%s items loaded", self.frames_count)  # type:ignore[attr-defined]
        logger.debug("Initialized %s", self.__class__.__name__)

    @staticmethod
    def check_file_exists(alignments_file: str) -> tuple[str, str]:
        """ Check if the alignments file exists, and returns a tuple of the folder and filename.

        Parameters
        ----------
        alignments_file
            Full path to an alignments file

        Returns
        -------
        folder
            The full path to the folder containing the alignments file
        filename
            The filename of the alignments file
        """
        folder, filename = os.path.split(alignments_file)
        if not os.path.isfile(alignments_file):
            logger.error("ERROR: alignments file not found at: '%s'", alignments_file)
            sys.exit(0)
        if folder:
            logger.verbose(  # type:ignore[attr-defined]
                "Alignments file exists at '%s'", alignments_file)
        return folder, filename

    def save(self) -> None:
        """Backup copy of old alignments and save new alignments """
        self.backup()
        super().save()


class MediaLoader():
    """Class to load images.

    Parameters
    ----------
    folder
        The folder of images or video file to load images from
    count
        If the total frame count is known it can be passed in here which will skip
        analyzing a video file. If the count is not passed in, it will be calculated.
        Default: ``None``
    """
    def __init__(self, folder: str, count: int | None = None):
        logger.debug("Initializing %s: (folder: '%s')", self.__class__.__name__, folder)
        logger.info("[%s DATA]", self.__class__.__name__.upper())
        self._count = count
        self.folder = folder
        self._vid_reader = self.check_input_folder()
        self.file_list_sorted = self.sorted_items()
        self.items = self.load_items()
        logger.verbose("%s items loaded", self.count)  # type:ignore[attr-defined]
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def is_video(self) -> bool:
        """Whether source is a video or not"""
        return self._vid_reader is not None

    @property
    def count(self) -> int:
        """Number of faces or frames"""
        if self._count is not None:
            return self._count
        if self.is_video:
            self._count = int(count_frames(self.folder))
        else:
            self._count = len(self.file_list_sorted)
        return self._count

    def check_input_folder(self) -> cv2.VideoCapture | None:
        """Ensure that the frames or faces folder exists and is valid. If frames folder contains a
        video file return cv2 reader object

        Returns
        -------
        Object for reading a video stream
        """
        err = None
        load_type = self.__class__.__name__
        if not self.folder:
            err = f"ERROR: A {load_type} folder must be specified"
        elif not os.path.exists(self.folder):
            err = f"ERROR: The {load_type} location {self.folder} could not be found"
        if err:
            logger.error(err)
            sys.exit(0)

        if (load_type == "Frames" and
                os.path.isfile(self.folder) and
                os.path.splitext(self.folder)[1].lower() in VIDEO_EXTENSIONS):
            logger.verbose("Video exists at: '%s'", self.folder)  # type:ignore[attr-defined]
            retval = cv2.VideoCapture(self.folder)
        else:
            logger.verbose("Folder exists at '%s'", self.folder)  # type:ignore[attr-defined]
            retval = None
        return retval

    @staticmethod
    def valid_extension(filename) -> bool:
        """Check whether passed in file has a valid extension"""
        extension = os.path.splitext(filename)[1]
        retval = extension.lower() in IMAGE_EXTENSIONS
        logger.trace("Filename has valid extension: '%s': %s",  # type:ignore[attr-defined]
                     filename, retval)
        return retval

    def sorted_items(self) -> list[dict[str, str]] | list[tuple[str, PNGHeader]]:
        """Override for specific folder processing"""
        raise NotImplementedError()

    def process_folder(self) -> (Generator[dict[str, str], None, None] |
                                 Generator[tuple[str, PNGHeader], None, None]):
        """Override for specific folder processing"""
        raise NotImplementedError()

    def load_items(self) -> dict[str, list[int]] | dict[str, tuple[str, str]]:
        """Override for specific item loading"""
        raise NotImplementedError()

    def load_image(self, filename: str) -> np.ndarray:
        """Load an image

        Parameters
        ----------
        filename
            The filename of the image to load

        Returns
        -------
        The loaded image
        """
        if self.is_video:
            image = self.load_video_frame(filename)
        else:
            src = os.path.join(self.folder, filename)
            logger.trace("Loading image: '%s'", src)  # type:ignore[attr-defined]
            image = read_image(src, raise_error=True)
        return image

    def load_video_frame(self, filename: str) -> np.ndarray:
        """Load a requested frame from video

        Parameters
        ----------
        filename
            The frame name to load

        Returns
        -------
        The loaded image
        """
        assert self._vid_reader is not None
        frame = os.path.splitext(filename)[0]
        logger.trace("Loading video frame: '%s'", frame)  # type:ignore[attr-defined]
        frame_no = int(frame[frame.rfind("_") + 1:]) - 1
        self._vid_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

        _, image = self._vid_reader.read()
        return image

    def stream(self, skip_list: list[int] | None = None
               ) -> Generator[tuple[str, np.ndarray], None, None]:
        """Load the images in :attr:`folder` in the order they are received from
        :class:`lib.image.ImagesLoader` in a background thread.

        Parameters
        ----------
        skip_list
            A list of frame indices that should not be loaded. Pass ``None`` if all images should
            be loaded. Default: ``None``

        Yields
        ------
        filename
            The filename of the image that is being returned
        image
            The image that has been loaded from disk
        """
        loader = ImagesLoader(self.folder, queue_size=32, count=self._count)
        if skip_list is not None:
            loader.add_skip_list(skip_list)
        for filename_image in loader.load():
            yield filename_image[0], filename_image[1]

    @staticmethod
    def save_image(output_folder: str,
                   filename: str,
                   image: np.ndarray,
                   metadata: PNGHeader | None = None) -> None:
        """Save an image

        Parameters
        ----------
        filename
            The filename of the image to save
        image
            The image to save
        metadata
            Any faceswap metadata that should be saved
        """
        output_file = os.path.join(output_folder, filename)
        output_file = os.path.splitext(output_file)[0] + ".png"
        logger.trace("Saving image: '%s'", output_file)  # type:ignore[attr-defined]
        if metadata:
            encoded = cv2.imencode(".png", image)[1]
            encoded_image = png_write_meta(encoded.tobytes(), metadata)
            with open(output_file, "wb") as out_file:
                out_file.write(encoded_image)
        else:
            cv2.imwrite(output_file, image)


class Faces(MediaLoader):
    """Object to load Extracted Faces from a folder.

    Parameters
    ----------
    folder
        The folder to load faces from
    alignments
        The alignments object that contains the faces. This can be used for 2 purposes:
        - To update legacy hash based faces for <v2.1 alignments to png header based version.
        - When the remove-faces job is being run, when the process will only load faces that exist
        in the alignments file. Default: ``None``
    """
    def __init__(self, folder: str, alignments: Alignments | None = None) -> None:
        self._alignments = alignments
        super().__init__(folder)

    def _handle_duplicate(self,
                          fullpath: str,
                          header_dict: PNGHeader,
                          seen: dict[str, list[int]]) -> bool:
        """Check whether the given face has already been seen for the source frame and face index
        from an existing face. Can happen when filenames have changed due to sorting etc. and users
        have done multiple extractions/copies and placed all of the faces in the same folder

        Parameters
        ----------
        fullpath
            The full path to the face image that is being checked
        header_dict
            The PNG header dictionary for the given face
        seen
            Dictionary of original source filename and face indices that have already been seen and
            will be updated with the face processing now

        Returns
        -------
        ``True`` if the face was a duplicate and has been removed, otherwise ``False``
        """
        src_filename = header_dict.source.source_filename
        face_index = header_dict.source.face_index

        if src_filename in seen and face_index in seen[src_filename]:
            dupe_dir = os.path.join(self.folder, "_duplicates")
            os.makedirs(dupe_dir, exist_ok=True)
            filename = os.path.basename(fullpath)
            logger.trace("Moving duplicate: %s", filename)  # type:ignore
            os.rename(fullpath, os.path.join(dupe_dir, filename))
            return True

        seen.setdefault(src_filename, []).append(face_index)
        return False

    def process_folder(self) -> Generator[tuple[str, PNGHeader], None, None]:
        """Iterate through the faces folder pulling out various information for each face.

        Yields
        ------
        A dictionary for each face found containing the keys returned from
        :class:`lib.image.read_image_meta_batch`
        """
        logger.info("Loading file list from %s", self.folder)
        filter_count = 0
        dupe_count = 0
        seen: dict[str, list[int]] = {}

        if self._alignments is not None and self._alignments.version < 2.1:  # Legacy updating
            filelist = [os.path.join(self.folder, face)
                        for face in os.listdir(self.folder)
                        if self.valid_extension(face)]
        else:
            filelist = [os.path.join(self.folder, face)
                        for face in os.listdir(self.folder)
                        if os.path.splitext(face)[-1] == ".png"]

        for fullpath, metadata in tqdm(read_image_meta_batch(filelist),
                                       total=len(filelist),
                                       desc="Reading Face Data"):

            if "itxt" not in metadata or "source" not in metadata["itxt"]:
                logger.warning("Non-Faceswap extracted face found. Image skipped: '%s'",
                               fullpath)
                continue
            sub_dict = T.cast(PNGHeader, PNGHeader.from_dict(metadata["itxt"]))

            if self._handle_duplicate(fullpath, sub_dict, seen):
                dupe_count += 1
                continue

            if (self._alignments is not None and  # filter existing
                    not self._alignments.frame_exists(sub_dict.source.source_filename)):
                filter_count += 1
                continue

            retval = (os.path.basename(fullpath), sub_dict)
            yield retval

        if self._alignments is not None:
            logger.debug("Faces filtered out that did not exist in alignments file: %s",
                         filter_count)

        if dupe_count > 0:
            logger.warning("%s Duplicate face images were found. These files have been moved to "
                           "'%s' from where they can be safely deleted",
                           dupe_count, os.path.join(self.folder, "_duplicates"))

    def load_items(self) -> dict[str, list[int]]:
        """Load the face names into dictionary.

        Returns
        -------
        The source filename as key with list of face indices for the frame as value
        """
        faces: dict[str, list[int]] = {}
        for face in T.cast(list[tuple[str, PNGHeader]], self.file_list_sorted):
            src = face[1].source
            faces.setdefault(src.source_filename, []).append(src.face_index)
        logger.trace(faces)  # type:ignore[attr-defined]
        return faces

    def sorted_items(self) -> list[tuple[str, PNGHeader]]:
        """Return the items sorted by the saved file name.

        Returns
        --------
        List of `dict` objects for each face found, sorted by the face's current filename
        """
        items = sorted(self.process_folder(), key=itemgetter(0))
        logger.trace(items)  # type:ignore[attr-defined]
        return items


class Frames(MediaLoader):
    """Object to hold the frames that are to be checked against """

    def process_folder(self) -> Generator[dict[str, str], None, None]:
        """Iterate through the frames folder pulling the base filename

        Yields
        ------
        The full frame name, the filename and the file extension of the frame
        """
        iterator = self.process_video if self.is_video else self.process_frames
        yield from iterator()

    def process_frames(self) -> Generator[dict[str, str], None, None]:
        """Process exported Frames

        Yields
        ------
        The full frame name, the filename and the file extension of the frame
        """
        logger.info("Loading file list from %s", self.folder)
        for frame in os.listdir(self.folder):
            if not self.valid_extension(frame):
                continue
            filename = os.path.splitext(frame)[0]
            file_extension = os.path.splitext(frame)[1]

            retval = {"frame_fullname": frame,
                      "frame_name": filename,
                      "frame_extension": file_extension}
            logger.trace(retval)  # type:ignore[attr-defined]
            yield retval

    def process_video(self) -> Generator[dict[str, str], None, None]:
        """Dummy in frames for video

        Yields
        ------
        The full frame name, the filename and the file extension of the frame
        """
        logger.info("Loading video frames from %s", self.folder)
        vid_name, ext = os.path.splitext(os.path.basename(self.folder))
        for i in range(self.count):
            idx = i + 1
            # Keep filename format for outputted face
            filename = f"{vid_name}_{idx:06d}"
            retval = {"frame_fullname": f"{filename}{ext}",
                      "frame_name": filename,
                      "frame_extension": ext}
            logger.trace(retval)  # type:ignore[attr-defined]
            yield retval

    def load_items(self) -> dict[str, tuple[str, str]]:
        """Load the frame info into dictionary

        Returns
        -------
        Fullname as key, tuple of frame name and extension as value
        """
        frames: dict[str, tuple[str, str]] = {}
        for frame in T.cast(list[dict[str, str]], self.file_list_sorted):
            frames[frame["frame_fullname"]] = (frame["frame_name"],
                                               frame["frame_extension"])
        logger.trace(frames)  # type:ignore[attr-defined]
        return frames

    def sorted_items(self) -> list[dict[str, str]]:
        """Return the items sorted by filename

        Returns
        -------
        The sorted list of frame information
        """
        items = sorted(self.process_folder(), key=lambda x: (x["frame_name"]))
        logger.trace(items)  # type:ignore[attr-defined]
        return items


class ExtractedFaces():
    """Holds the extracted faces and matrix for alignments

    Parameters
    ----------
    frames
        The frames object to extract faces from
    alignments
        The alignment data corresponding to the frames
    size
        The extract face size. Default: 512
    """
    def __init__(self, frames: Frames, alignments: AlignmentData, size: int = 512) -> None:
        logger.trace("Initializing %s: size: %s",  # type:ignore[attr-defined]
                     self.__class__.__name__, size)
        self.size = size
        self.padding = int(size * 0.1875)
        self.alignments = alignments
        self.frames = frames
        self.current_frame: str | None = None
        self.faces: list[DetectedFace] = []
        logger.trace("Initialized %s", self.__class__.__name__)  # type:ignore[attr-defined]

    def get_faces(self, frame: str, image: np.ndarray | None = None) -> None:
        """Obtain faces and transformed landmarks for each face in a given frame with its
        alignments

        Parameters
        ----------
        frame
            The frame name to obtain faces for
        image
            The image to extract the face from, if we already have it, otherwise ``None`` to
            load the image. Default: ``None``
        """
        logger.trace("Getting faces for frame: '%s'", frame)  # type:ignore[attr-defined]
        self.current_frame = None
        alignments = self.alignments.get_faces_in_frame(frame)
        logger.trace(  # type:ignore[attr-defined]
            "Alignments for frame: (frame: '%s', alignments: %s)", frame, alignments)
        if not alignments:
            self.faces = []
            return
        image = self.frames.load_image(frame) if image is None else image
        self.faces = [self.extract_one_face(alignment, image) for alignment in alignments]
        self.current_frame = frame

    def extract_one_face(self,
                         alignment: FileAlignments,
                         image: np.ndarray) -> DetectedFace:
        """Extract one face from image

        Parameters
        ----------
        alignment
            The alignment for a single face
        image
            The image to extract the face from

        Returns
        -------
        The detected face object for the given alignment with the aligned face loaded
        """
        logger.trace(  # type:ignore[attr-defined]
            "Extracting one face: (frame: '%s', alignment: %s)", self.current_frame, alignment)
        face = DetectedFace()
        face.from_alignment(alignment, image=image)
        face.load_aligned(image, size=self.size, centering="head")
        face.thumbnail = generate_thumbnail(face.aligned.face, size=80, quality=60)
        return face

    def get_faces_in_frame(self,
                           frame: str,
                           update: bool = False,
                           image: np.ndarray | None = None) -> list[DetectedFace]:
        """Return the faces for the selected frame

        Parameters
        ----------
        frame
            The frame name to get the faces for
        update
            ``True`` if the faces should be refreshed regardless of current frame. ``False`` to not
            force a refresh. Default ``False``
        image
            Image to load faces from if it exists, otherwise ``None`` to load the image.
            Default: ``None``

        Returns
        -------
        List of :class:`~lib.align.DetectedFace` objects for the frame, with the aligned face
        loaded
        """
        logger.trace("frame: '%s', update: %s", frame, update)  # type:ignore[attr-defined]
        if self.current_frame != frame or update:
            self.get_faces(frame, image=image)
        return self.faces

    def get_roi_size_for_frame(self, frame: str) -> list[int]:
        """Return the size of the original extract box for the selected frame.

        Parameters
        ----------
        frame
            The frame to obtain the original sized bounding boxes for

        Returns
        -------
        List of original pixel sizes of faces held within the frame
        """
        logger.trace("frame: '%s'", frame)  # type:ignore[attr-defined]
        if self.current_frame != frame:
            self.get_faces(frame)
        sizes = []
        for face in self.faces:
            roi = face.aligned.original_roi.squeeze()
            top_left, top_right = roi[0], roi[3]
            len_x = top_right[0] - top_left[0]
            len_y = top_right[1] - top_left[1]
            if top_left[1] == top_right[1]:
                length = len_y
            else:
                length = int(((len_x ** 2) + (len_y ** 2)) ** 0.5)
            sizes.append(length)
        logger.trace("sizes: '%s'", sizes)  # type:ignore[attr-defined]
        return sizes


__all__ = get_module_objects(__name__)
