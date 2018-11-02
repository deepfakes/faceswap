#!/usr/bin/env python3
""" Alignments file functions for reading, writing and manipulating
    a serialized alignments file """

import os

import cv2

from lib import Serializer
from lib.utils import rotate_landmarks


class Alignments():
    """ Holds processes pertaining to the alignments file.

        folder:     folder alignments file is stored in
        filename:   Filename of alignments file excluding extension. If a
                    valid extension is provided, then it will be used to
                    decide the serializer, and the serializer argument will
                    be ignored.
        serializer: If provided, this will be the format data is saved in
                    if data is to be saved. Can be 'json', 'pickle' or 'yaml'
        is_extract: Used to indicate whether this class is being called by the
                    extract process, so a missing alignments file is ok.
    """
    def __init__(self, folder, filename="alignments", serializer="json",
                 verbose=False):
        self.verbose = verbose
        self.serializer = self.get_serializer(filename, serializer)
        self.file = self.get_location(folder, filename)

        self.data = self.load()

    # << PROPERTIES >> #

    @property
    def frames_count(self):
        """ Return current frames count """
        return len(self.data)

    @property
    def faces_count(self):
        """ Return current faces count """
        return sum(len(faces) for faces in self.data.values())

    @property
    def have_alignments_file(self):
        """ Return whether an alignments file exists """
        return os.path.exists(self.file)

    # << INIT FUNCTIONS >> #

    def get_serializer(self, filename, serializer):
        """ Set the serializer to be used for loading and
            saving alignments """
        extension = os.path.splitext(filename)[1]
        if extension in ("json", "p", "yaml", "yml"):
            retval = Serializer.get_serializer_from_ext(extension)
        elif serializer not in ("json", "pickle", "yaml"):
            raise ValueError("Error: {} is not a valid serializer. Use "
                             "'json', 'pickle' or 'yaml'")
        else:
            retval = Serializer.get_serializer(serializer)
        if self.verbose:
            print("Using {} serializer for alignments".format(retval.ext))
        return retval

    def get_location(self, folder, filename):
        """ Return the path to alignments file """
        location = os.path.join(
            str(folder),
            "{}.{}".format(filename, self.serializer.ext))
        if self.verbose:
            print("Alignments filepath: {}".format(location))
        return location

    # << FILE SYSTEM >> #

    def load(self):
        """ Load the alignments data if it exists or create empty dict """
        if not self.have_alignments_file:
            raise ValueError("Error: Alignments file not found at "
                             "{}".format(self.file))

        try:
            print("Reading alignments from: {}".format(self.file))
            with open(self.file, self.serializer.roptions) as align:
                data = self.serializer.unmarshal(align.read())
        except IOError as err:
            print("Error: {} not read: {}".format(self.file, err.strerror))
            exit(1)
        return data

    def save(self):
        """ Write the serialized alignments file """
        try:
            print("Writing alignments to: {}".format(self.file))
            with open(self.file, self.serializer.woptions) as align:
                align.write(self.serializer.marshal(self.data))
        except IOError as err:
            print("Error: {} not written: {}".format(self.file, err.strerror))

    # << UTILITIES >> #

    def frame_exists(self, frame):
        """ return path of images that have faces """
        return frame in self.data.keys()

    def get_alignments_for_frame(self, frame):
        """ Return the alignments for the selected frame """
        return self.data.get(frame, list())

    # << LEGACY ROTATION FUNCTIONS >> #

    # The old rotation method would rotate the image to find a face, then
    # store the rotated landmarks along with a rotation value to tell the
    # convert process that it had to rotate the frame to find the landmarks.
    # This is problematic for numerous reasons. The process now rotates the
    # landmarks to correctly correspond with the original frame. The below are
    # functions to convert legacy alignments to the currently supported
    # infrastructure.
    # This can eventually be removed

    def get_legacy_frames(self):
        """ Return a list of frames with legacy rotations
            Looks for an 'r' value in the alignments file that
            is not zero """
        keys = list()
        for key, val in self.data.items():
            if any(alignment.get("r", None) for alignment in val):
                keys.append(key)
        return keys

    def rotate_existing_landmarks(self, frame, dimensions):
        """ Backwards compatability fix. Rotates the landmarks to
            their correct position and deletes r

            NB: The original frame dimensions must be passed in otherwise
            the transformation cannot be performed """
        for face in self.get_alignments_for_frame(frame):
            angle = face.get("r", 0)
            if not angle:
                return
            r_mat = self.get_original_rotation_matrix(dimensions, angle)
            rotate_landmarks(face, r_mat)
            del face["r"]

    @staticmethod
    def get_original_rotation_matrix(dimensions, angle):
        """ Calculate original rotation matrix and invert """
        height, width = dimensions
        center = (width/2, height/2)
        r_mat = cv2.getRotationMatrix2D(  # pylint: disable=no-member
            center, -1.0*angle, 1.)

        abs_cos = abs(r_mat[0, 0])
        abs_sin = abs(r_mat[0, 1])
        rotated_width = int(height*abs_sin + width*abs_cos)
        rotated_height = int(height*abs_cos + width*abs_sin)
        r_mat[0, 2] += rotated_width/2 - center[0]
        r_mat[1, 2] += rotated_height/2 - center[1]

        return r_mat
