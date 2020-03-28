#!/usr/bin/env python3
""" Media items (Alignments, Faces, Frames)
    for alignments tool """

import logging
import os
import sys

import cv2
import numpy as np
from tqdm import tqdm

# TODO imageio single frame seek seems slow. Look into this
# import imageio

from lib.aligner import Extract as AlignerExtract
from lib.alignments import Alignments, get_serializer
from lib.faces_detect import DetectedFace
from lib.image import (count_frames, encode_image_with_hash, ImagesLoader, read_image,
                       read_image_hash_batch)
from lib.utils import _image_extensions, _video_extensions

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class AlignmentData(Alignments):
    """ Class to hold the alignment data """

    def __init__(self, alignments_file):
        logger.debug("Initializing %s: (alignments file: '%s')",
                     self.__class__.__name__, alignments_file)
        logger.info("[ALIGNMENT DATA]")  # Tidy up cli output
        folder, filename = self.check_file_exists(alignments_file)
        if filename.lower() == "dfl":
            self._serializer = get_serializer("compressed")
            self._file = "{}.{}".format(filename.lower(), self._serializer.file_extension)
            return
        super().__init__(folder, filename=filename)
        logger.verbose("%s items loaded", self.frames_count)
        logger.debug("Initialized %s", self.__class__.__name__)

    @staticmethod
    def check_file_exists(alignments_file):
        """ Check the alignments file exists"""
        folder, filename = os.path.split(alignments_file)
        if filename.lower() == "dfl":
            folder = None
            filename = "dfl"
            logger.info("Using extracted DFL faces for alignments")
        elif not os.path.isfile(alignments_file):
            logger.error("ERROR: alignments file not found at: '%s'", alignments_file)
            sys.exit(0)
        if folder:
            logger.verbose("Alignments file exists at '%s'", alignments_file)
        return folder, filename

    def save(self):
        """ Backup copy of old alignments and save new alignments """
        self.backup()
        super().save()

    def reload(self):
        """ Read the alignments data from the correct format """
        logger.debug("Re-loading alignments")
        self._data = self._load()
        logger.debug("Re-loaded alignments")

    def add_face_hashes(self, frame_name, hashes):
        """ Recalculate face hashes """
        logger.trace("Adding face hash: (frame: '%s', hashes: %s)", frame_name, hashes)
        faces = self.get_faces_in_frame(frame_name)
        count_match = len(faces) - len(hashes)
        if count_match != 0:
            msg = "more" if count_match > 0 else "fewer"
            logger.warning("There are %s %s face(s) in the alignments file than exist in the "
                           "faces folder. Check your sources for frame '%s'.",
                           abs(count_match), msg, frame_name)
        for idx, i_hash in hashes.items():
            faces[idx]["hash"] = i_hash

    def data_from_dfl(self, alignments, faces_folder):
        """ Set :attr:`data` from alignments extracted from a Deep Face Lab face set.

        Parameters
        ----------
        alignments: dict
            The extracted alignments from a Deep Face Lab face set
        faces_folder: str
            The folder that the faces are in, where the newly generated alignments file will
            be saved
        """
        self._data = alignments
        self.set_filename(self._get_location(faces_folder, "alignments"))

    def set_filename(self, filename):
        """ Set the :attr:`_file` to the given filename.

        Parameters
        ----------
        filename: str
            The full path and filename to se the alignments file name to
        """
        self._file = filename


class MediaLoader():
    """ Class to load filenames from folder """
    def __init__(self, folder):
        logger.debug("Initializing %s: (folder: '%s')", self.__class__.__name__, folder)
        logger.info("[%s DATA]", self.__class__.__name__.upper())
        self._count = None
        self.folder = folder
        self.vid_reader = self.check_input_folder()
        self.file_list_sorted = self.sorted_items()
        self.items = self.load_items()
        logger.verbose("%s items loaded", self.count)
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def is_video(self):
        """ Return whether source is a video or not """
        return self.vid_reader is not None

    @property
    def count(self):
        """ Number of faces or frames """
        if self._count is not None:
            return self._count
        if self.is_video:
            self._count = int(count_frames(self.folder))
        else:
            self._count = len(self.file_list_sorted)
        return self._count

    def check_input_folder(self):
        """ makes sure that the frames or faces folder exists
            If frames folder contains a video file return imageio reader object """
        err = None
        loadtype = self.__class__.__name__
        if not self.folder:
            err = "ERROR: A {} folder must be specified".format(loadtype)
        elif not os.path.exists(self.folder):
            err = ("ERROR: The {} location {} could not be "
                   "found".format(loadtype, self.folder))
        if err:
            logger.error(err)
            sys.exit(0)

        if (loadtype == "Frames" and
                os.path.isfile(self.folder) and
                os.path.splitext(self.folder)[1].lower() in _video_extensions):
            logger.verbose("Video exists at: '%s'", self.folder)
            retval = cv2.VideoCapture(self.folder)  # pylint: disable=no-member
            # TODO ImageIO single frame seek seems slow. Look into this
            # retval = imageio.get_reader(self.folder, "ffmpeg")
        else:
            logger.verbose("Folder exists at '%s'", self.folder)
            retval = None
        return retval

    @staticmethod
    def valid_extension(filename):
        """ Check whether passed in file has a valid extension """
        extension = os.path.splitext(filename)[1]
        retval = extension.lower() in _image_extensions
        logger.trace("Filename has valid extension: '%s': %s", filename, retval)
        return retval

    @staticmethod
    def sorted_items():
        """ Override for specific folder processing """
        return list()

    @staticmethod
    def process_folder():
        """ Override for specific folder processing """
        return list()

    @staticmethod
    def load_items():
        """ Override for specific item loading """
        return dict()

    def load_image(self, filename):
        """ Load an image """
        if self.is_video:
            image = self.load_video_frame(filename)
        else:
            src = os.path.join(self.folder, filename)
            logger.trace("Loading image: '%s'", src)
            image = read_image(src, raise_error=True)
        return image

    def load_video_frame(self, filename):
        """ Load a requested frame from video """
        frame = os.path.splitext(filename)[0]
        logger.trace("Loading video frame: '%s'", frame)
        frame_no = int(frame[frame.rfind("_") + 1:]) - 1
        self.vid_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_no)  # pylint: disable=no-member
        _, image = self.vid_reader.read()
        # TODO imageio single frame seek seems slow. Look into this
        # self.vid_reader.set_image_index(frame_no)
        # image = self.vid_reader.get_next_data()[:, :, ::-1]
        return image

    def stream(self, skip_list=None):
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
        loader = ImagesLoader(self.folder, queue_size=32)
        if skip_list is not None:
            loader.add_skip_list(skip_list)
        for filename, image in loader.load():
            yield filename, image

    @staticmethod
    def save_image(output_folder, filename, image):
        """ Save an image """
        output_file = os.path.join(output_folder, filename)
        output_file = os.path.splitext(output_file)[0]+'.png'
        logger.trace("Saving image: '%s'", output_file)
        cv2.imwrite(output_file, image)  # pylint: disable=no-member


class Faces(MediaLoader):
    """ Object to hold the faces that are to be swapped out """

    def process_folder(self):
        """ Iterate through the faces folder pulling out various information """
        logger.info("Loading file list from %s", self.folder)

        filelist = [os.path.join(self.folder, face)
                    for face in os.listdir(self.folder)
                    if self.valid_extension(face)]
        for fullpath, face_hash in tqdm(read_image_hash_batch(filelist),
                                        total=len(filelist),
                                        desc="Reading Face Hashes"):
            filename = os.path.basename(fullpath)
            face_name, extension = os.path.splitext(filename)
            retval = {"face_fullname": filename,
                      "face_name": face_name,
                      "face_extension": extension,
                      "face_hash": face_hash}
            logger.trace(retval)
            yield retval

    def load_items(self):
        """ Load the face names into dictionary """
        faces = dict()
        for face in self.file_list_sorted:
            faces.setdefault(face["face_hash"], list()).append((face["face_name"],
                                                                face["face_extension"]))
        logger.trace(faces)
        return faces

    def sorted_items(self):
        """ Return the items sorted by face name """
        items = sorted([item for item in self.process_folder()],
                       key=lambda x: (x["face_name"]))
        logger.trace(items)
        return items


class Frames(MediaLoader):
    """ Object to hold the frames that are to be checked against """

    def process_folder(self):
        """ Iterate through the frames folder pulling the base filename """
        iterator = self.process_video if self.is_video else self.process_frames
        for item in iterator():
            yield item

    def process_frames(self):
        """ Process exported Frames """
        logger.info("Loading file list from %s", self.folder)
        for frame in os.listdir(self.folder):
            if not self.valid_extension(frame):
                continue
            filename = os.path.splitext(frame)[0]
            file_extension = os.path.splitext(frame)[1]

            retval = {"frame_fullname": frame,
                      "frame_name": filename,
                      "frame_extension": file_extension}
            logger.trace(retval)
            yield retval

    def process_video(self):
        """Dummy in frames for video """
        logger.info("Loading video frames from %s", self.folder)
        vidname = os.path.splitext(os.path.basename(self.folder))[0]
        for i in range(self.count):
            idx = i + 1
            # Keep filename format for outputted face
            filename = "{}_{:06d}".format(vidname, idx)
            retval = {"frame_fullname": "{}.png".format(filename),
                      "frame_name": filename,
                      "frame_extension": ".png"}
            logger.trace(retval)
            yield retval

    def load_items(self):
        """ Load the frame info into dictionary """
        frames = dict()
        for frame in self.file_list_sorted:
            frames[frame["frame_fullname"]] = (frame["frame_name"],
                                               frame["frame_extension"])
        logger.trace(frames)
        return frames

    def sorted_items(self):
        """ Return the items sorted by filename """
        items = sorted([item for item in self.process_folder()],
                       key=lambda x: (x["frame_name"]))
        logger.trace(items)
        return items


class ExtractedFaces():
    """ Holds the extracted faces and matrix for
        alignments """
    def __init__(self, frames, alignments, size=256, align_eyes=False):
        logger.trace("Initializing %s: size: %s", self.__class__.__name__, size)
        self.size = size
        self.padding = int(size * 0.1875)
        self.align_eyes_bool = align_eyes
        self.alignments = alignments
        self.frames = frames
        self.current_frame = None
        self.faces = list()
        logger.trace("Initialized %s", self.__class__.__name__)

    def get_faces(self, frame, image=None):
        """ Return faces and transformed landmarks
            for each face in a given frame with it's alignments"""
        logger.trace("Getting faces for frame: '%s'", frame)
        self.current_frame = None
        alignments = self.alignments.get_faces_in_frame(frame)
        logger.trace("Alignments for frame: (frame: '%s', alignments: %s)", frame, alignments)
        if not alignments:
            self.faces = list()
            return
        image = self.frames.load_image(frame) if image is None else image
        self.faces = [self.extract_one_face(alignment, image) for alignment in alignments]
        self.current_frame = frame

    def extract_one_face(self, alignment, image):
        """ Extract one face from image """
        logger.trace("Extracting one face: (frame: '%s', alignment: %s)",
                     self.current_frame, alignment)
        face = DetectedFace()
        face.from_alignment(alignment, image=image)
        face.load_aligned(image, size=self.size)
        face = self.align_eyes(face, image) if self.align_eyes_bool else face
        return face

    def get_faces_in_frame(self, frame, update=False, image=None):
        """ Return the faces for the selected frame """
        logger.trace("frame: '%s', update: %s", frame, update)
        if self.current_frame != frame or update:
            self.get_faces(frame, image=image)
        return self.faces

    def get_roi_size_for_frame(self, frame):
        """ Return the size of the original extract box for
            the selected frame """
        logger.trace("frame: '%s'", frame)
        if self.current_frame != frame:
            self.get_faces(frame)
        sizes = list()
        for face in self.faces:
            roi = face.original_roi.squeeze()
            top_left, top_right = roi[0], roi[3]
            len_x = top_right[0] - top_left[0]
            len_y = top_right[1] - top_left[1]
            if top_left[1] == top_right[1]:
                length = len_y
            else:
                length = int(((len_x ** 2) + (len_y ** 2)) ** 0.5)
            sizes.append(length)
        logger.trace("sizes: '%s'", sizes)
        return sizes

    @staticmethod
    def save_face_with_hash(filename, extension, face):
        """ Save a face and return it's hash """
        f_hash, img = encode_image_with_hash(face, extension)
        logger.trace("Saving face: '%s'", filename)
        with open(filename, "wb") as out_file:
            out_file.write(img)
        return f_hash

    @staticmethod
    def align_eyes(face, image):
        """ Re-extract a face with the pupils forced to be absolutely horizontally aligned """
        umeyama_landmarks = face.aligned_landmarks
        left_eye_center = umeyama_landmarks[42:48].mean(axis=0)
        right_eye_center = umeyama_landmarks[36:42].mean(axis=0)
        d_y = right_eye_center[1] - left_eye_center[1]
        d_x = right_eye_center[0] - left_eye_center[0]
        theta = np.pi - np.arctan2(d_y, d_x)
        rot_cos = np.cos(theta)
        rot_sin = np.sin(theta)
        rotation_matrix = np.array([[rot_cos, -rot_sin, 0.],
                                    [rot_sin, rot_cos, 0.],
                                    [0., 0., 1.]])

        mat_umeyama = np.concatenate((face.aligned["matrix"], np.array([[0., 0., 1.]])), axis=0)
        corrected_mat = np.dot(rotation_matrix, mat_umeyama)
        face.aligned["matrix"] = corrected_mat[:2]
        face.aligned["face"] = AlignerExtract().transform(image,
                                                          face.aligned["matrix"],
                                                          face.aligned["size"],
                                                          int(face.aligned["size"] * 0.375) // 2)
        logger.trace("Adjusted matrix: %s", face.aligned["matrix"])
        return face
