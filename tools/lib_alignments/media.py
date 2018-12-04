#!/usr/bin/env python3
""" Media items (Alignments, Faces, Frames)
    for alignments tool """

import logging
import os

import cv2

from lib.alignments import Alignments
from lib.faces_detect import DetectedFace
from lib.utils import _image_extensions

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class AlignmentData(Alignments):
    """ Class to hold the alignment data """

    def __init__(self, alignments_file, destination_format):
        logger.info("[ALIGNMENT DATA]")  # Tidy up cli output
        folder, filename = self.check_file_exists(alignments_file)
        if filename == "dfl":
            self.set_dfl(destination_format)
            return
        super().__init__(folder, filename=filename)
        self.set_destination_format(destination_format)
        logger.verbose("%s items loaded", self.frames_count)

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
        self.file = "dfl"
        self.set_destination_format(destination_format)

    def set_destination_format(self, destination_format):
        """ Standardise the destination format to the correct extension """
        extensions = {".json": "json",
                      ".p": "pickle",
                      ".yml": "yaml",
                      ".yaml": "yaml"}
        dst_fmt = None
        file_ext = os.path.splitext(self.file)[1].lower()

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

    def save(self):
        """ Backup copy of old alignments and save new alignments """
        self.backup()
        super().save()


class MediaLoader():
    """ Class to load filenames from folder """
    def __init__(self, folder):
        logger.info("[%s DATA]", self.__class__.__name__.upper())
        self.folder = folder
        self.check_folder_exists()
        self.file_list_sorted = self.sorted_items()
        self.items = self.load_items()
        self.count = len(self.file_list_sorted)
        logger.verbose("%s items loaded", self.count)

    def check_folder_exists(self):
        """ makes sure that the faces folder exists """
        err = None
        loadtype = self.__class__.__name__
        if not self.folder:
            err = "ERROR: A {} folder must be specified".format(loadtype)
        elif not os.path.isdir(self.folder):
            err = ("ERROR: The {} folder {} could not be "
                   "found".format(loadtype, self.folder))
        if err:
            logger.error(err)
            exit(0)

        logger.verbose("Folder exists at '%s'", self.folder)

    @staticmethod
    def valid_extension(filename):
        """ Check whether passed in file has a valid extension """
        extension = os.path.splitext(filename)[1]
        return bool(extension in _image_extensions)

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
        src = os.path.join(self.folder, filename)
        image = cv2.imread(src)  # pylint: disable=no-member
        return image

    @staticmethod
    def save_image(output_folder, filename, image):
        """ Save an image """
        output_file = os.path.join(output_folder, filename)
        cv2.imwrite(output_file, image)  # pylint: disable=no-member


class Faces(MediaLoader):
    """ Object to hold the faces that are to be swapped out """
    def __init__(self, folder, dfl=False):
        self.dfl = dfl
        super().__init__(folder)

    def process_folder(self):
        """ Iterate through the faces dir pulling out various information """
        logger.info("Loading file list from %s", self.folder)
        for face in os.listdir(self.folder):
            if not self.valid_extension(face):
                continue
            filename = os.path.splitext(face)[0]
            file_extension = os.path.splitext(face)[1]
            index = 0
            original_file = ""
            if not self.dfl:
                index = int(filename[filename.rindex("_") + 1:])
                original_file = "{}".format(filename[:filename.rindex("_")])
            yield {"face_fullname": face,
                   "face_name": filename,
                   "face_extension": file_extension,
                   "frame_name": original_file,
                   "face_index": index}

    def load_items(self):
        """ Load the face names into dictionary """
        faces = dict()
        for face in self.file_list_sorted:
            faces.setdefault(face["frame_name"],
                             list()).append(face["face_index"])
        return faces

    def sorted_items(self):
        """ Return the items sorted by filename then index """
        return sorted([item for item in self.process_folder()],
                      key=lambda x: (x["frame_name"], x["face_index"]))


class Frames(MediaLoader):
    """ Object to hold the frames that are to be checked against """

    def process_folder(self):
        """ Iterate through the frames dir pulling the base filename """
        logger.info("Loading file list from %s", self.folder)
        for frame in os.listdir(self.folder):
            if not self.valid_extension(frame):
                continue
            filename = os.path.splitext(frame)[0]
            file_extension = os.path.splitext(frame)[1]

            yield {"frame_fullname": frame,
                   "frame_name": filename,
                   "frame_extension": file_extension}

    def load_items(self):
        """ Load the frame info into dictionary """
        frames = dict()
        for frame in self.file_list_sorted:
            frames[frame["frame_fullname"]] = (frame["frame_name"],
                                               frame["frame_extension"])
        return frames

    def sorted_items(self):
        """ Return the items sorted by filename """
        return sorted([item for item in self.process_folder()],
                      key=lambda x: (x["frame_name"]))


class ExtractedFaces():
    """ Holds the extracted faces and matrix for
        alignments """
    def __init__(self, frames, alignments, size=256,
                 padding=48, align_eyes=False):
        self.size = size
        self.padding = padding
        self.align_eyes = align_eyes
        self.alignments = alignments
        self.frames = frames

        self.current_frame = None
        self.faces = list()

    def get_faces(self, frame):
        """ Return faces and transformed landmarks
            for each face in a given frame with it's alignments"""
        self.current_frame = None
        alignments = self.alignments.get_faces_in_frame(frame)
        if not alignments:
            self.faces = list()
            return
        image = self.frames.load_image(frame)
        self.faces = [self.extract_one_face(alignment, image.copy())
                      for alignment in alignments]
        self.current_frame = frame

    def extract_one_face(self, alignment, image):
        """ Extract one face from image """
        face = DetectedFace()
        face.from_alignment(alignment, image=image)
        face.load_aligned(image,
                          size=self.size,
                          padding=self.padding,
                          align_eyes=self.align_eyes)
        return face

    def get_faces_in_frame(self, frame, update=False):
        """ Return the faces for the selected frame """
        if self.current_frame != frame or update:
            self.get_faces(frame)
        return self.faces

    def get_roi_size_for_frame(self, frame):
        """ Return the size of the original extract box for
            the selected frame """
        if self.current_frame != frame:
            self.get_faces(frame)
        sizes = list()
        for face in self.faces:
            top_left, top_right = face.original_roi[0], face.original_roi[3]
            len_x = top_right[0] - top_left[0]
            len_y = top_right[1] - top_left[1]
            if top_left[1] == top_right[1]:
                length = len_y
            else:
                length = int(((len_x ** 2) + (len_y ** 2)) ** 0.5)
            sizes.append(length)
        return sizes
