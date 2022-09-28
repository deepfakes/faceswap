#!/usr/bin/env python3
""" Media items (Alignments, Faces, Frames)
    for alignments tool """

import logging
from operator import itemgetter
import os
import sys
from typing import cast, Generator, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

import cv2
from tqdm import tqdm

# TODO imageio single frame seek seems slow. Look into this
# import imageio

from lib.align import Alignments, DetectedFace, update_legacy_png_header
from lib.image import (count_frames, generate_thumbnail, ImagesLoader,
                       png_write_meta, read_image, read_image_meta_batch)
from lib.utils import _image_extensions, _video_extensions, FaceswapError

if TYPE_CHECKING:
    import numpy as np
    from lib.align.alignments import AlignmentFileDict, PNGHeaderDict

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class AlignmentData(Alignments):
    """ Class to hold the alignment data

    Paramaters
    ----------
    alignments_file: str
        Full path to an alignments file
    """
    def __init__(self, alignments_file: str) -> None:
        logger.debug("Initializing %s: (alignments file: '%s')",
                     self.__class__.__name__, alignments_file)
        logger.info("[ALIGNMENT DATA]")  # Tidy up cli output
        folder, filename = self.check_file_exists(alignments_file)
        super().__init__(folder, filename=filename)
        logger.verbose("%s items loaded", self.frames_count)  # type: ignore
        logger.debug("Initialized %s", self.__class__.__name__)

    @staticmethod
    def check_file_exists(alignments_file: str) -> Tuple[str, str]:
        """ Check the alignments file exists

        Paramaters
        ----------
        alignments_file: str
            Full path to an alignments file

        Returns
        -------
        folder: str
            The full path to the folder containing the alignments file
        filename: str
            The filename of the alignments file
        """
        folder, filename = os.path.split(alignments_file)
        if not os.path.isfile(alignments_file):
            logger.error("ERROR: alignments file not found at: '%s'", alignments_file)
            sys.exit(0)
        if folder:
            logger.verbose("Alignments file exists at '%s'", alignments_file)  # type: ignore
        return folder, filename

    def save(self) -> None:
        """ Backup copy of old alignments and save new alignments """
        self.backup()
        super().save()


class MediaLoader():
    """ Class to load images.

    Parameters
    ----------
    folder: str
        The folder of images or video file to load images from
    count: int or ``None``, optional
        If the total frame count is known it can be passed in here which will skip
        analyzing a video file. If the count is not passed in, it will be calculated.
    """
    def __init__(self, folder: str, count: Optional[int] = None):
        logger.debug("Initializing %s: (folder: '%s')", self.__class__.__name__, folder)
        logger.info("[%s DATA]", self.__class__.__name__.upper())
        self._count = count
        self.folder = folder
        self._vid_reader = self.check_input_folder()
        self.file_list_sorted = self.sorted_items()
        self.items = self.load_items()
        logger.verbose("%s items loaded", self.count)  # type: ignore
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def is_video(self) -> bool:
        """ bool: Return whether source is a video or not """
        return self._vid_reader is not None

    @property
    def count(self) -> int:
        """ int: Number of faces or frames """
        if self._count is not None:
            return self._count
        if self.is_video:
            self._count = int(count_frames(self.folder))
        else:
            self._count = len(self.file_list_sorted)
        return self._count

    def check_input_folder(self) -> Optional[cv2.VideoCapture]:
        """ makes sure that the frames or faces folder exists
            If frames folder contains a video file return imageio reader object

        Returns
        -------
        :class:`cv2.VideoCapture`
            Object for reading a video stream
        """
        err = None
        loadtype = self.__class__.__name__
        if not self.folder:
            err = f"ERROR: A {loadtype} folder must be specified"
        elif not os.path.exists(self.folder):
            err = f"ERROR: The {loadtype} location {self.folder} could not be found"
        if err:
            logger.error(err)
            sys.exit(0)

        if (loadtype == "Frames" and
                os.path.isfile(self.folder) and
                os.path.splitext(self.folder)[1].lower() in _video_extensions):
            logger.verbose("Video exists at: '%s'", self.folder)  # type: ignore
            retval = cv2.VideoCapture(self.folder)  # pylint: disable=no-member
            # TODO ImageIO single frame seek seems slow. Look into this
            # retval = imageio.get_reader(self.folder, "ffmpeg")
        else:
            logger.verbose("Folder exists at '%s'", self.folder)  # type: ignore
            retval = None
        return retval

    @staticmethod
    def valid_extension(filename) -> bool:
        """ bool: Check whether passed in file has a valid extension """
        extension = os.path.splitext(filename)[1]
        retval = extension.lower() in _image_extensions
        logger.trace("Filename has valid extension: '%s': %s", filename, retval)  # type: ignore
        return retval

    def sorted_items(self) -> Union[List[Dict[str, str]],
                                    List[Tuple[str, "PNGHeaderDict"]]]:
        """ Override for specific folder processing """
        raise NotImplementedError()

    def process_folder(self) -> Union[Generator[Dict[str, str], None, None],
                                      Generator[Tuple[str, "PNGHeaderDict"], None, None]]:
        """ Override for specific folder processing """
        raise NotImplementedError()

    def load_items(self) -> Union[Dict[str, List[int]],
                                  Dict[str, Tuple[str, str]]]:
        """ Override for specific item loading """
        raise NotImplementedError()

    def load_image(self, filename: str) -> "np.ndarray":
        """ Load an image

        Parameters
        ----------
        filename: str
            The filename of the image to load

        Returns
        -------
        :class:`numpy.ndarray`
            The loaded image
        """
        if self.is_video:
            image = self.load_video_frame(filename)
        else:
            src = os.path.join(self.folder, filename)
            logger.trace("Loading image: '%s'", src)  # type: ignore
            image = read_image(src, raise_error=True)
        return image

    def load_video_frame(self, filename: str) -> "np.ndarray":
        """ Load a requested frame from video

        Parameters
        ----------
        filename: str
            The frame name to load

        Returns
        -------
        :class:`numpy.ndarray`
            The loaded image
        """
        assert self._vid_reader is not None
        frame = os.path.splitext(filename)[0]
        logger.trace("Loading video frame: '%s'", frame)  # type: ignore
        frame_no = int(frame[frame.rfind("_") + 1:]) - 1
        self._vid_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_no)  # pylint: disable=no-member
        _, image = self._vid_reader.read()
        # TODO imageio single frame seek seems slow. Look into this
        # self._vid_reader.set_image_index(frame_no)
        # image = self._vid_reader.get_next_data()[:, :, ::-1]
        return image

    def stream(self, skip_list: Optional[List[int]] = None
               ) -> Generator[Tuple[str, "np.ndarray"], None, None]:
        """ Load the images in :attr:`folder` in the order they are received from
        :class:`lib.image.ImagesLoader` in a background thread.

        Parameters
        ----------
        skip_list: list, optional
            A list of frame indices that should not be loaded. Pass ``None`` if all images should
            be loaded. Default: ``None``

        Yields
        ------
        str
            The filename of the image that is being returned
        numpy.ndarray
            The image that has been loaded from disk
        """
        loader = ImagesLoader(self.folder, queue_size=32, count=self._count)
        if skip_list is not None:
            loader.add_skip_list(skip_list)
        for filename, image in loader.load():
            yield filename, image

    @staticmethod
    def save_image(output_folder: str,
                   filename: str,
                   image: "np.ndarray",
                   metadata: Optional["PNGHeaderDict"] = None) -> None:
        """ Save an image """
        output_file = os.path.join(output_folder, filename)
        output_file = os.path.splitext(output_file)[0] + ".png"
        logger.trace("Saving image: '%s'", output_file)  # type: ignore
        if metadata:
            encoded_image = cv2.imencode(".png", image)[1]
            encoded_image = png_write_meta(encoded_image.tobytes(), metadata)
            with open(output_file, "wb") as out_file:
                out_file.write(encoded_image)
        else:
            cv2.imwrite(output_file, image)  # pylint: disable=no-member


class Faces(MediaLoader):
    """ Object to load Extracted Faces from a folder.

    Parameters
    ----------
    folder: str
        The folder to load faces from
    alignments: :class:`lib.align.Alignments`, optional
        The alignments object that contains the faces. Used to update legacy hash based faces
        for <v2.1 alignments to png header based version. Pass in ``None`` to not update legacy
        faces (raises error instead). Default: ``None``
    """
    def __init__(self, folder: str, alignments: Optional[Alignments] = None) -> None:
        self._alignments = alignments
        super().__init__(folder)

    def process_folder(self) -> Generator[Tuple[str, "PNGHeaderDict"], None, None]:
        """ Iterate through the faces folder pulling out various information for each face.

        Yields
        ------
        dict
            A dictionary for each face found containing the keys returned from
            :class:`lib.image.read_image_meta_batch`
        """
        logger.info("Loading file list from %s", self.folder)

        if self._alignments is not None:  # Legacy updating
            filelist = [os.path.join(self.folder, face)
                        for face in os.listdir(self.folder)
                        if self.valid_extension(face)]
        else:
            filelist = [os.path.join(self.folder, face)
                        for face in os.listdir(self.folder)
                        if os.path.splitext(face)[-1] == ".png"]

        log_once = False
        for fullpath, metadata in tqdm(read_image_meta_batch(filelist),
                                       total=len(filelist),
                                       desc="Reading Face Data"):

            if "itxt" not in metadata or "source" not in metadata["itxt"]:
                if self._alignments is None:  # Can't update legacy
                    raise FaceswapError(
                        f"The folder '{self.folder}' contains images that do not include Faceswap "
                        "metadata.\nAll images in the provided folder should contain faces "
                        "generated from Faceswap's extraction process.\nPlease double check the "
                        "source and try again.")

                if not log_once:
                    logger.warning("Legacy faces discovered. These faces will be updated")
                    log_once = True
                data = update_legacy_png_header(fullpath, self._alignments)
                if not data:
                    raise FaceswapError(
                        f"Some of the faces being passed in from '{self.folder}' could not be "
                        f"matched to the alignments file '{self._alignments.file}'\nPlease double "
                        "check your sources and try again.")
                sub_dict = data
            else:
                sub_dict = cast("PNGHeaderDict", metadata["itxt"])

            retval = (os.path.basename(fullpath), sub_dict)
            yield retval

    def load_items(self) -> Dict[str, List[int]]:
        """ Load the face names into dictionary.

        Returns
        -------
        dict
            The source filename as key with list of face indices for the frame as value
        """
        faces: Dict[str, List[int]] = {}
        for face in cast(List[Tuple[str, "PNGHeaderDict"]], self.file_list_sorted):
            src = face[1]["source"]
            faces.setdefault(src["source_filename"], []).append(src["face_index"])
        logger.trace(faces)  # type: ignore
        return faces

    def sorted_items(self) -> List[Tuple[str, "PNGHeaderDict"]]:
        """ Return the items sorted by the saved file name.

        Returns
        --------
        list
            List of `dict` objects for each face found, sorted by the face's current filename
        """
        items = sorted(self.process_folder(), key=itemgetter(0))
        logger.trace(items)  # type: ignore
        return items


class Frames(MediaLoader):
    """ Object to hold the frames that are to be checked against """

    def process_folder(self) -> Generator[Dict[str, str], None, None]:
        """ Iterate through the frames folder pulling the base filename

        Yields
        ------
        dict
            The full framename, the filename and the file extension of the frame
        """
        iterator = self.process_video if self.is_video else self.process_frames
        for item in iterator():
            yield item

    def process_frames(self) -> Generator[Dict[str, str], None, None]:
        """ Process exported Frames

        Yields
        ------
        dict
            The full framename, the filename and the file extension of the frame
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
            logger.trace(retval)  # type: ignore
            yield retval

    def process_video(self) -> Generator[Dict[str, str], None, None]:
        """Dummy in frames for video

        Yields
        ------
        dict
            The full framename, the filename and the file extension of the frame
        """
        logger.info("Loading video frames from %s", self.folder)
        vidname = os.path.splitext(os.path.basename(self.folder))[0]
        for i in range(self.count):
            idx = i + 1
            # Keep filename format for outputted face
            filename = f"{vidname}_{idx:06d}"
            retval = {"frame_fullname": f"{filename}.png",
                      "frame_name": filename,
                      "frame_extension": ".png"}
            logger.trace(retval)  # type: ignore
            yield retval

    def load_items(self) -> Dict[str, Tuple[str, str]]:
        """ Load the frame info into dictionary

        Returns
        -------
        dict
            Fullname as key, tuple of frame name and extension as value
        """
        frames: Dict[str, Tuple[str, str]] = {}
        for frame in cast(List[Dict[str, str]], self.file_list_sorted):
            frames[frame["frame_fullname"]] = (frame["frame_name"],
                                               frame["frame_extension"])
        logger.trace(frames)  # type: ignore
        return frames

    def sorted_items(self) -> List[Dict[str, str]]:
        """ Return the items sorted by filename

        Returns
        -------
        list
            The sorted list of frame information
        """
        items = sorted(self.process_folder(), key=lambda x: (x["frame_name"]))
        logger.trace(items)  # type: ignore
        return items


class ExtractedFaces():
    """ Holds the extracted faces and matrix for alignments

    Parameters
    ----------
    frames: :class:`Frames`
        The frames object to extract faces from
    alignments: :class:`AlignmentData`
        The alignment data corresponding to the frames
    size: int, optional
        The extract face size. Default: 512
    """
    def __init__(self, frames: Frames, alignments: AlignmentData, size: int = 512) -> None:
        logger.trace("Initializing %s: size: %s",  # type: ignore
                     self.__class__.__name__, size)
        self.size = size
        self.padding = int(size * 0.1875)
        self.alignments = alignments
        self.frames = frames
        self.current_frame: Optional[str] = None
        self.faces: List[DetectedFace] = []
        logger.trace("Initialized %s", self.__class__.__name__)  # type: ignore

    def get_faces(self, frame: str, image: Optional["np.ndarray"] = None) -> None:
        """ Obtain faces and transformed landmarks for each face in a given frame with its
        alignments

        Parameters
        ----------
        frame: str
            The frame name to obtain faces for
        image: :class:`numpy.ndarray`, optional
            The image to extract the face from, if we already have it, otherwise ``None`` to
            load the image. Default: ``None``
        """
        logger.trace("Getting faces for frame: '%s'", frame)  # type: ignore
        self.current_frame = None
        alignments = self.alignments.get_faces_in_frame(frame)
        logger.trace("Alignments for frame: (frame: '%s', alignments: %s)",  # type: ignore
                     frame, alignments)
        if not alignments:
            self.faces = []
            return
        image = self.frames.load_image(frame) if image is None else image
        self.faces = [self.extract_one_face(alignment, image) for alignment in alignments]
        self.current_frame = frame

    def extract_one_face(self,
                         alignment: "AlignmentFileDict",
                         image: "np.ndarray") -> DetectedFace:
        """ Extract one face from image

        Parameters
        ----------
        alignment: dict
            The alignment for a single face
        image: :class:`numpy.ndarray`
            The image to extract the face from

        Returns
        -------
        :class:`~lib.align.DetectedFace`
            The detected face object for the given alignment with the aligned face loaded
        """
        logger.trace("Extracting one face: (frame: '%s', alignment: %s)",  # type: ignore
                     self.current_frame, alignment)
        face = DetectedFace()
        face.from_alignment(alignment, image=image)
        face.load_aligned(image, size=self.size, centering="head")
        face.thumbnail = generate_thumbnail(face.aligned.face, size=80, quality=60)
        return face

    def get_faces_in_frame(self,
                           frame: str,
                           update: bool = False,
                           image: Optional["np.ndarray"] = None) -> List[DetectedFace]:
        """ Return the faces for the selected frame

        Parameters
        ----------
        frame: str
            The frame name to get the faces for
        update: bool, optional
            ``True`` if the faces should be refreshed regardless of current frame. ``False`` to not
            force a refresh. Default ``False``
        image: :class:`numpy.ndarray`, optional
            Image to load faces from if it exists, otherwise ``None`` to load the image.
            Default: ``None``

        Returns
        -------
        list
            List of :class:`~lib.align.DetectedFace` objects for the frame, with the aligned face
            loaded
        """
        logger.trace("frame: '%s', update: %s", frame, update)  # type: ignore
        if self.current_frame != frame or update:
            self.get_faces(frame, image=image)
        return self.faces

    def get_roi_size_for_frame(self, frame: str) -> List[int]:
        """ Return the size of the original extract box for the selected frame.

        Parameters
        ----------
        frame: str
            The frame to obtain the original sized bounding boxes for

        Returns
        -------
        list
            List of original pixel sizes of faces held within the frame
        """
        logger.trace("frame: '%s'", frame)  # type: ignore
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
        logger.trace("sizes: '%s'", sizes)  # type: ignore
        return sizes
