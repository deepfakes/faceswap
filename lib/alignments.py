#!/usr/bin/env python3
""" Alignments file functions for reading, writing and manipulating
    a serialized alignments file """

import os
from datetime import datetime

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
        serializer: If provided, this will be the format that the data is
                    saved in (if data is to be saved). Can be 'json', 'pickle'
                    or 'yaml'
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
            saving alignments

            If a filename with a valid extension is passed in
            this will be used as the serializer, otherwise the
            specified serializer will be used """
        extension = os.path.splitext(filename)[1]
        if extension in (".json", ".p", ".yaml", ".yml"):
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
        extension = os.path.splitext(filename)[1]
        if extension in (".json", ".p", ".yaml", ".yml"):
            location = os.path.join(str(folder), filename)
        else:
            location = os.path.join(str(folder),
                                    "{}.{}".format(filename,
                                                   self.serializer.ext))
        if self.verbose:
            print("Alignments filepath: {}".format(location))
        return location

    # << I/O >> #

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

    def reload(self):
        """ Read the alignments data from the correct format """
        self.data = self.load()

    def save(self):
        """ Write the serialized alignments file """
        try:
            print("Writing alignments to: {}".format(self.file))
            with open(self.file, self.serializer.woptions) as align:
                align.write(self.serializer.marshal(self.data))
        except IOError as err:
            print("Error: {} not written: {}".format(self.file, err.strerror))

    def backup(self):
        """ Backup copy of old alignments """
        if not os.path.isfile(self.file):
            return
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        src = self.file
        split = os.path.splitext(src)
        dst = split[0] + "_" + now + split[1]
        print("Backing up original alignments to {}".format(dst))
        os.rename(src, dst)

    # << VALIDATION >> #

    def frame_exists(self, frame):
        """ return path of images that have faces """
        return frame in self.data.keys()

    def frame_has_faces(self, frame):
        """ Return true if frame exists and has faces """
        return bool(self.data.get(frame, list()))

    def frame_has_multiple_faces(self, frame):
        """ Return true if frame exists and has faces """
        if not frame:
            return False
        return bool(len(self.data.get(frame, list())) > 1)

    # << DATA >> #

    def get_faces_in_frame(self, frame):
        """ Return the alignments for the selected frame """
        return self.data.get(frame, list())

    def get_full_frame_name(self, frame):
        """ Return a frame with extension for when the extension is
            not known """
        return next(key for key in self.data.keys()
                    if key.startswith(frame))

    def count_faces_in_frame(self, frame):
        """ Return number of alignments within frame """
        return len(self.data.get(frame, list()))

    # << MANIPULATION >> #

    def delete_face_at_index(self, frame, idx):
        """ Delete the face alignment for given frame at given index """
        idx = int(idx)
        if idx + 1 > self.count_faces_in_frame(frame):
            return False
        del self.data[frame][idx]
        return True

    def add_face(self, frame, alignment):
        """ Add a new face for a frame and return it's index """
        self.data[frame].append(alignment)
        return self.count_faces_in_frame(frame) - 1

    def update_face(self, frame, idx, alignment):
        """ Replace a face for given frame and index """
        self.data[frame][idx] = alignment

    # << GENERATORS >> #

    def yield_faces(self):
        """ Yield face alignments for one image """
        for frame_fullname, alignments in self.data.items():
            frame_name = os.path.splitext(frame_fullname)[0]
            yield frame_name, alignments, len(alignments), frame_fullname

    @staticmethod
    def yield_original_index_reverse(image_alignments, number_alignments):
        """ Return the correct original index for
            alignment in reverse order """
        for idx, _ in enumerate(reversed(image_alignments)):
            original_idx = number_alignments - 1 - idx
            yield original_idx

    # << LEGACY FUNCTIONS >> #

    # < Original Frame Dimensions > #
    # For dfaker and convert-adjust the original dimensions of a frame are
    # required to calculate the transposed landmarks. As transposed landmarks
    # will change on face size, we store original frame dimensions
    # These were not previously required, so this adds the dimensions
    # to the landmarks file

    def get_legacy_no_dims(self):
        """ Return a list of frames that do not contain the original frame
            height and width attributes """
        keys = list()
        for key, val in self.data.items():
            for alignment in val:
                if "frame_dims" not in alignment.keys():
                    keys.append(key)
                    break
        return keys

    def add_dimensions(self, frame_name, dimensions):
        """ Backward compatability fix. Add frame dimensions
            to alignments """
        for face in self.get_faces_in_frame(frame_name):
            face["frame_dims"] = dimensions

    # < Rotation > #
    # The old rotation method would rotate the image to find a face, then
    # store the rotated landmarks along with a rotation value to tell the
    # convert process that it had to rotate the frame to find the landmarks.
    # This is problematic for numerous reasons. The process now rotates the
    # landmarks to correctly correspond with the original frame. The below are
    # functions to convert legacy alignments to the currently supported
    # infrastructure.
    # This can eventually be removed

    def get_legacy_rotation(self):
        """ Return a list of frames with legacy rotations
            Looks for an 'r' value in the alignments file that
            is not zero """
        keys = list()
        for key, val in self.data.items():
            if any(alignment.get("r", None) for alignment in val):
                keys.append(key)
        return keys

    def rotate_existing_landmarks(self, frame_name):
        """ Backwards compatability fix. Rotates the landmarks to
            their correct position and deletes r

            NB: The original frame dimensions must be passed in otherwise
            the transformation cannot be performed """
        for face in self.get_faces_in_frame(frame_name):
            angle = face.get("r", 0)
            if not angle:
                return
            dims = face["frame_dims"]
            r_mat = self.get_original_rotation_matrix(dims, angle)
            rotate_landmarks(face, r_mat)
            del face["r"]

    @staticmethod
    def get_original_rotation_matrix(dimensions, angle):
        """ Calculate original rotation matrix and invert """
        height, width = dimensions
        center = (width/2, height/2)
        r_mat = cv2.getRotationMatrix2D(  # pylint: disable=no-member
            center, -1.0 * angle, 1.)

        abs_cos = abs(r_mat[0, 0])
        abs_sin = abs(r_mat[0, 1])
        rotated_width = int(height*abs_sin + width*abs_cos)
        rotated_height = int(height*abs_cos + width*abs_sin)
        r_mat[0, 2] += rotated_width/2 - center[0]
        r_mat[1, 2] += rotated_height/2 - center[1]

        return r_mat
