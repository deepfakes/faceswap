#!/usr/bin/env python3
""" Media items (Alignments, Faces, Frames)
    for alignments tool """

import os
from datetime import datetime

from cv2 import imread, imwrite

from lib import Serializer
from lib.utils import _image_extensions


class AlignmentData():
    """ Class to hold the alignment data """

    def __init__(self, alignments_file, destination_format, verbose):
        print("\n[ALIGNMENT DATA]")  # Tidy up cli output
        self.file = alignments_file
        self.verbose = verbose

        self.check_file_exists()
        self.src_format = self.get_source_format()
        self.dst_format = self.get_destination_format(destination_format)

        if self.src_format == "dfl":
            self.set_destination_serializer()
            return

        self.serializer = Serializer.get_serializer_from_ext(
            self.src_format)
        self.alignments = self.load()
        self.count = len(self.alignments)
        self.count_per_frame = {key: len(value)
                                for key, value in self.alignments.items()}

        self.set_destination_serializer()
        if self.verbose:
            print("{} items loaded".format(self.count))

    def check_file_exists(self):
        """ Check the alignments file exists"""
        if os.path.split(self.file.lower())[1] == "dfl":
            self.file = "dfl"
        if self.file.lower() == "dfl":
            print("Using extracted pngs for alignments")
            return
        if not os.path.isfile(self.file):
            print("ERROR: alignments file not "
                  "found at: {}".format(self.file))
            exit(0)
        if self.verbose:
            print("Alignments file exists at {}".format(self.file))
        return

    def get_source_format(self):
        """ Get the source alignments format """
        if self.file.lower() == "dfl":
            return "dfl"
        return os.path.splitext(self.file)[1].lower()

    def get_destination_format(self, destination_format):
        """ Standardise the destination format to the correct extension """
        extensions = {".json": "json",
                      ".p": "pickle",
                      ".yml": "yaml",
                      ".yaml": "yaml"}
        dst_fmt = None

        if destination_format is not None:
            dst_fmt = destination_format
        elif self.src_format == "dfl":
            dst_fmt = "json"
        elif self.src_format in extensions.keys():
            dst_fmt = extensions[self.src_format]
        else:
            print("{} is not a supported serializer. "
                  "Exiting".format(self.src_format))
            exit(0)

        if self.verbose:
            print("Destination format set to {}".format(dst_fmt))

        return dst_fmt

    def set_destination_serializer(self):
        """ set the destination serializer """
        self.serializer = Serializer.get_serializer(self.dst_format)

    def load(self):
        """ Read the alignments data from the correct format """
        print("Loading alignments from {}".format(self.file))
        with open(self.file, self.serializer.roptions) as align:
            alignments = self.serializer.unmarshal(align.read())
        return alignments

    def save_alignments(self):
        """ Backup copy of old alignments and save new alignments """
        dst = os.path.splitext(self.file)[0]
        dst += ".{}".format(self.serializer.ext)
        self.backup_alignments()

        print("Saving alignments to {}".format(dst))
        with open(dst, self.serializer.woptions) as align:
            align.write(self.serializer.marshal(self.alignments))

    def backup_alignments(self):
        """ Backup copy of old alignments """
        if not os.path.isfile(self.file):
            return
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        src = self.file
        dst = src.split(".")
        dst[0] += "_" + now + "."
        dst = dst[0] + dst[1]
        print("Backing up original alignments to {}".format(dst))
        os.rename(src, dst)

    def get_alignments_one_image(self):
        """ Return the face alignments for one image """
        for image, alignments in self.alignments.items():
            image_stripped = image[:image.rindex(".")]
            number_alignments = len(alignments)
            yield image_stripped, alignments, number_alignments

    @staticmethod
    def get_one_alignment_index_reverse(image_alignments, number_alignments):
        """ Return the correct original index for
            alignment in reverse order """
        for idx, _ in enumerate(reversed(image_alignments)):
            original_idx = number_alignments - 1 - idx
            yield original_idx

    def has_alignments(self, filename, alignments):
        """ Check whether this frame has alignments """
        if not alignments:
            if self.verbose:
                print("Skipping {} - Alignments not found".format(filename))
            return False
        return True


class MediaLoader():
    """ Class to load filenames from folder """
    def __init__(self, folder, verbose):
        print("\n[{} DATA]".format(self.__class__.__name__.upper()))
        self.verbose = verbose
        self.folder = folder
        self.check_folder_exists()
        self.file_list_sorted = self.sorted_items()
        self.items = self.load_items()
        self.count = len(self.file_list_sorted)
        if self.verbose:
            print("{} items loaded".format(self.count))

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
            print(err)
            exit(0)

        if self.verbose:
            print("Folder exists at {}".format(self.folder))

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
        image = imread(src)
        return image

    @staticmethod
    def save_image(output_folder, filename, image):
        """ Save an image """
        output_file = os.path.join(output_folder, filename)
        imwrite(output_file, image)


class Faces(MediaLoader):
    """ Object to hold the faces that are to be swapped out """

    def process_folder(self):
        """ Iterate through the faces dir pulling out various information """
        print("Loading file list from {}".format(self.folder))
        for face in os.listdir(self.folder):
            if not self.valid_extension(face):
                continue
            filename = os.path.splitext(face)[0]
            file_extension = os.path.splitext(face)[1]
            index = int(filename[filename.rindex("_") + 1:])
            original_file = "{}".format(filename[:filename.rindex("_")])
            yield (filename, file_extension, original_file, index)

    def load_items(self):
        """ Load the face names into dictionary """
        faces = dict()
        for face in self.file_list_sorted:
            original_file, index = face[2:4]
            if faces.get(original_file, "") == "":
                faces[original_file] = [index]
            else:
                faces[original_file].append(index)
        return faces

    def sorted_items(self):
        """ Return the items sorted by filename then index """
        return sorted([item for item in self.process_folder()],
                      key=lambda x: (x[2], x[3]))


class Frames(MediaLoader):
    """ Object to hold the frames that are to be checked against """

    def process_folder(self):
        """ Iterate through the frames dir pulling the base filename """
        print("Loading file list from {}".format(self.folder))
        for frame in os.listdir(self.folder):
            if not self.valid_extension(frame):
                continue
            filename = os.path.basename(frame)
            yield filename

    def load_items(self):
        """ Load the frame info into dictionary """
        frames = dict()
        for frame in self.file_list_sorted:
            frames[frame] = (frame[:frame.rfind(".")],
                             frame[frame.rfind("."):])
        return frames

    def sorted_items(self):
        """ Return the items sorted by filename """
        return sorted([item for item in self.process_folder()])


class DetectedFace():
    """ Detected face and landmark information """
    def __init__(self, image, r, x, w, y, h, landmarksXY):
        self.image = image
        self.r = r
        self.x = x
        self.w = w
        self.y = y
        self.h = h
        self.landmarksXY = landmarksXY

    def landmarks_as_xy(self):
        """ Landmarks as XY """
        return self.landmarksXY