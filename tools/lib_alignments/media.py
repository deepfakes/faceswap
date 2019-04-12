#!/usr/bin/env python3
""" Media items (Alignments, Faces, Frames)
    for alignments tool """

import logging
import os
from tqdm import tqdm

import cv2

from lib.alignments import Alignments
from lib.faces_detect import DetectedFace
from lib.utils import _image_extensions, _video_extensions, hash_image_file, hash_encode_image

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class AlignmentData(Alignments):
    """ Class to hold the alignment data """

    def __init__(self, alignments_file, destination_format):
        logger.debug("Initializing %s: (alignments file: '%s', destination_format: '%s')",
                     self.__class__.__name__, alignments_file, destination_format)
        logger.info("[ALIGNMENT DATA]")  # Tidy up cli output
        folder, filename = self.check_file_exists(alignments_file)
        if filename.lower() == "dfl":
            self.set_dfl(destination_format)
            return
        super().__init__(folder, filename=filename)
        self.set_destination_format(destination_format)
        logger.verbose("%s items loaded", self.frames_count)
        logger.debug("Initialized %s", self.__class__.__name__)

    @staticmethod
    def check_file_exists(alignments_file):
        """ Check the alignments file exists"""
        folder, filename = os.path.split(alignments_file)
        if filename.lower() == "dfl":
            folder = None
            filename = "dfl"
            logger.info("Using extracted pngs for alignments")
        elif not os.path.isfile(alignments_file):
            logger.error("ERROR: alignments file not found at: '%s'", alignments_file)
            exit(0)
        if folder:
            logger.verbose("Alignments file exists at '%s'", alignments_file)
        return folder, filename

    def set_dfl(self, destination_format):
        """ Set the alignments for dfl alignments """
        logger.debug("Alignments are DFL format")
        self.file = "dfl"
        self.set_destination_format(destination_format)

    def set_destination_format(self, destination_format):
        """ Standardize the destination format to the correct extension """
        extensions = {".json": "json",
                      ".p": "pickle",
                      ".yml": "yaml",
                      ".yaml": "yaml"}
        dst_fmt = None
        file_ext = os.path.splitext(self.file)[1].lower()
        logger.debug("File extension: '%s'", file_ext)

        if destination_format is not None:
            dst_fmt = destination_format
        elif self.file == "dfl":
            dst_fmt = "json"
        elif file_ext in extensions.keys():
            dst_fmt = extensions[file_ext]
        else:
            logger.error("'%s' is not a supported serializer. Exiting", file_ext)
            exit(0)

        logger.verbose("Destination format set to '%s'", dst_fmt)

        self.serializer = self.get_serializer("", dst_fmt)
        filename = os.path.splitext(self.file)[0]
        self.file = "{}.{}".format(filename, self.serializer.ext)
        logger.debug("Destination file: '%s'", self.file)

    def save(self):
        """ Backup copy of old alignments and save new alignments """
        self.backup()
        super().save()


class MediaLoader():
    """ Class to load filenames from folder """
    def __init__(self, folder):
        logger.debug("Initializing %s: (folder: '%s')", self.__class__.__name__, folder)
        logger.info("[%s DATA]", self.__class__.__name__.upper())
        self.folder = folder
        self.vid_cap = self.check_input_folder()
        self.file_list_sorted = self.sorted_items()
        self.items = self.load_items()
        logger.verbose("%s items loaded", self.count)
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def count(self):
        """ Number of faces or frames """
        if self.vid_cap:
            retval = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))  # pylint: disable=no-member
        else:
            retval = len(self.file_list_sorted)
        return retval

    def check_input_folder(self):
        """ makes sure that the frames or faces folder exists
            If frames folder contains a video file return video capture object """
        err = None
        loadtype = self.__class__.__name__
        if not self.folder:
            err = "ERROR: A {} folder must be specified".format(loadtype)
        elif not os.path.exists(self.folder):
            err = ("ERROR: The {} location {} could not be "
                   "found".format(loadtype, self.folder))
        if err:
            logger.error(err)
            exit(0)

        if (loadtype == "Frames" and
                os.path.isfile(self.folder) and
                os.path.splitext(self.folder)[1] in _video_extensions):
            logger.verbose("Video exists at : '%s'", self.folder)
            retval = cv2.VideoCapture(self.folder)  # pylint: disable=no-member
        else:
            logger.verbose("Folder exists at '%s'", self.folder)
            retval = None
        return retval

    @staticmethod
    def valid_extension(filename):
        """ Check whether passed in file has a valid extension """
        extension = os.path.splitext(filename)[1]
        retval = extension in _image_extensions
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
        if self.vid_cap:
            image = self.load_video_frame(filename)
        else:
            src = os.path.join(self.folder, filename)
            logger.trace("Loading image: '%s'", src)
            image = cv2.imread(src)  # pylint: disable=no-member
        return image

    def load_video_frame(self, filename):
        """ Load a requested frame from video """
        frame = os.path.splitext(filename)[0]
        logger.trace("Loading video frame: '%s'", frame)
        frame_no = int(frame[frame.rfind("_") + 1:]) - 1
        self.vid_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)  # pylint: disable=no-member
        _, image = self.vid_cap.read()
        return image

    @staticmethod
    def save_image(output_folder, filename, image):
        """ Save an image """
        output_file = os.path.join(output_folder, filename)
        logger.trace("Saving image: '%s'", output_file)
        cv2.imwrite(output_file, image)  # pylint: disable=no-member


class Faces(MediaLoader):
    """ Object to hold the faces that are to be swapped out """

    def process_folder(self):
        """ Iterate through the faces dir pulling out various information """
        logger.info("Loading file list from %s", self.folder)
        for face in tqdm(os.listdir(self.folder), desc="Reading Face Hashes"):
            if not self.valid_extension(face):
                continue
            filename = os.path.splitext(face)[0]
            file_extension = os.path.splitext(face)[1]
            face_hash = hash_image_file(os.path.join(self.folder, face))
            retval = {"face_fullname": face,
                      "face_name": filename,
                      "face_extension": file_extension,
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
        """ Iterate through the frames dir pulling the base filename """
        iterator = self.process_video if self.vid_cap else self.process_frames
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
        logger.trace("Initializing %s: (size: %s, align_eyes: %s)",
                     self.__class__.__name__, size, align_eyes)
        self.size = size
        self.padding = int(size * 0.1875)
        self.align_eyes = align_eyes
        self.alignments = alignments
        self.frames = frames

        self.current_frame = None
        self.faces = list()
        logger.trace("Initialized %s", self.__class__.__name__)

    def get_faces(self, frame):
        """ Return faces and transformed landmarks
            for each face in a given frame with it's alignments"""
        logger.trace("Getting faces for frame: '%s'", frame)
        self.current_frame = None
        alignments = self.alignments.get_faces_in_frame(frame)
        logger.trace("Alignments for frame: (frame: '%s', alignments: %s)", frame, alignments)
        if not alignments:
            self.faces = list()
            return
        image = self.frames.load_image(frame)
        self.faces = [self.extract_one_face(alignment, image.copy())
                      for alignment in alignments]
        self.current_frame = frame

    def extract_one_face(self, alignment, image):
        """ Extract one face from image """
        logger.trace("Extracting one face: (frame: '%s', alignment: %s)",
                     self.current_frame, alignment)
        face = DetectedFace()
        face.from_alignment(alignment, image=image)
        face.load_aligned(image, size=self.size, align_eyes=self.align_eyes)
        return face

    def get_faces_in_frame(self, frame, update=False):
        """ Return the faces for the selected frame """
        logger.trace("frame: '%s', update: %s", frame, update)
        if self.current_frame != frame or update:
            self.get_faces(frame)
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
        f_hash, img = hash_encode_image(face, extension)
        logger.trace("Saving face: '%s'", filename)
        with open(filename, "wb") as out_file:
            out_file.write(img)
        return f_hash
