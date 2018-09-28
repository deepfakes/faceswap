#!/usr/bin/env python3
""" Media items (Alignments, Faces, Frames)
    for alignments tool """

import os
from datetime import datetime

import cv2
import numpy as np

from lib import Serializer
from lib.utils import _image_extensions, rotate_landmarks
from plugins.PluginLoader import PluginLoader


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

    def reload(self):
        """ Read the alignments data from the correct format """
        print("Reloading alignments from {}".format(self.file))
        with open(self.file, self.serializer.roptions) as align:
            self.alignments = self.serializer.unmarshal(align.read())

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
        for frame_fullname, alignments in self.alignments.items():
            frame_name = frame_fullname[:frame_fullname.rindex(".")]
            number_alignments = len(alignments)
            yield frame_name, alignments, number_alignments, frame_fullname

    @staticmethod
    def get_one_alignment_index_reverse(image_alignments, number_alignments):
        """ Return the correct original index for
            alignment in reverse order """
        for idx, _ in enumerate(reversed(image_alignments)):
            original_idx = number_alignments - 1 - idx
            yield original_idx

    def get_alignments_for_frame(self, frame):
        """ Return the alignments for the selected frame """
        return self.alignments.get(frame, list())

    def frame_in_alignments(self, frame):
        """ Return true if frame exists in alignments file """
        return bool(self.alignments.get(frame, -1) != -1)

    def frame_has_faces(self, frame):
        """ Return true if frame exists and has faces """
        return bool(self.alignments.get(frame, list()))

    def frame_has_multiple_faces(self, frame):
        """ Return true if frame exists and has faces """
        if not frame:
            return False
        return bool(len(self.alignments.get(frame, list())) > 1)

    def get_full_frame_name(self, frame):
        """ Return a frame with extension for when the extension is
            not known """
        return next(key for key in self.alignments.keys()
                    if key.startswith(frame))

    def count_alignments_in_frame(self, frame):
        """ Return number of alignments within frame """
        return len(self.alignments.get(frame, list()))

    def delete_alignment_at_index(self, frame, idx):
        """ Delete the face alignment for given frame at given index """
        idx = int(idx)
        if idx + 1 > self.count_alignments_in_frame(frame):
            return False
        del self.alignments[frame][idx]
        return True

    def add_alignment(self, frame, alignment):
        """ Add a new alignment for a frame and return it's index """
        self.alignments[frame].append(alignment)
        return self.count_alignments_in_frame(frame) - 1

    def update_alignment(self, frame, idx, alignment):
        """ Replace an alignment for given frame and index """
        self.alignments[frame][idx] = alignment

    def get_rotated(self):
        """ Return list of keys for alignments containing
            rotated frames """
        keys = list()
        for key, val in self.alignments.items():
            if any(alignment.get("r", None) for alignment in val):
                keys.append(key)
        return keys

    def rotate_existing_landmarks(self, frame, dimensions):
        """ Backwards compatability fix. Rotates the landmarks to
            their correct position and sets r to 0 """
        for alignment in self.alignments.get(frame, list()):
            angle = alignment.get("r", 0)
            if not angle:
                return
            rotation_matrix = self.get_original_rotation_matrix(dimensions,
                                                                angle)
            face = DetectedFace()
            face.alignment_to_face(None, alignment)
            face = rotate_landmarks(face, rotation_matrix)
            alignment = face.face_to_alignment(alignment)
            del alignment["r"]

    @staticmethod
    def get_original_rotation_matrix(dimensions, angle):
        """ Calculate original rotation matrix and invert """
        height, width = dimensions
        center = (width/2, height/2)
        rotation_matrix = cv2.getRotationMatrix2D(center, -1.0*angle, 1.)

        abs_cos = abs(rotation_matrix[0, 0])
        abs_sin = abs(rotation_matrix[0, 1])
        rotated_width = int(height*abs_sin + width*abs_cos)
        rotated_height = int(height*abs_cos + width*abs_sin)
        rotation_matrix[0, 2] += rotated_width/2 - center[0]
        rotation_matrix[1, 2] += rotated_height/2 - center[1]

        return rotation_matrix


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
        image = cv2.imread(src)
        return image

    @staticmethod
    def save_image(output_folder, filename, image):
        """ Save an image """
        output_file = os.path.join(output_folder, filename)
        cv2.imwrite(output_file, image)


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
        print("Loading file list from {}".format(self.folder))
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


class DetectedFace():
    """ Detected face and landmark information """
    def __init__(self):
        self.image = None
        self.x = None
        self.w = None
        self.y = None
        self.h = None
        self.landmarksXY = None

    def alignment_to_face(self, image, alignment):
        """ Convert a face alignment to detected face object """
        self.image = image
        self.x = alignment["x"]
        self.w = alignment["w"]
        self.y = alignment["y"]
        self.h = alignment["h"]
        self.landmarksXY = alignment["landmarksXY"]

    def face_to_alignment(self, alignment):
        """ Convert a face alignment to detected face object """
        alignment["x"] = self.x
        alignment["w"] = self.w
        alignment["y"] = self.y
        alignment["h"] = self.h
        alignment["landmarksXY"] = self.landmarksXY
        return alignment

    def landmarks_as_xy(self):
        """ Landmarks as XY """
        return self.landmarksXY


class ExtractedFaces():
    """ Holds the extracted faces and matrix for
        alignments """
    def __init__(self, frames, alignments, size=256,
                 padding=48, align_eyes=False):
        self.size = size
        self.padding = padding
        self.align_eyes = align_eyes
        self.extractor = PluginLoader.get_extractor("Align")()
        self.alignments = alignments
        self.frames = frames

        self.current_frame = None
        self.faces = list()
        self.matrices = list()

    def get_faces(self, frame):
        """ Return faces and transformed face matrices
            for each face in a given frame with it's alignments"""
        self.current_frame = None
        self.faces = list()
        self.matrices = list()
        alignments = self.alignments.get_alignments_for_frame(frame)
        if not alignments:
            return
        image = self.frames.load_image(frame)
        for alignment in alignments:
            face, matrix = self.extract_one_face(alignment, image.copy())
            self.faces.append(face)
            self.matrices.append(matrix)
        self.current_frame = frame

    def extract_one_face(self, alignment, image):
        """ Extract one face from image """
        face = DetectedFace()
        face.alignment_to_face(image, alignment)
        return self.extractor.extract(image, face, self.size, self.align_eyes)

    def original_roi(self, matrix):
        """ Return the original ROI of an extracted face """
        points = np.array([[0, 0], [0, self.size - 1],
                           [self.size - 1, self.size - 1],
                           [self.size - 1, 0]], np.int32)
        points = points.reshape((-1, 1, 2))

        mat = matrix * (self.size - 2 * self.padding)
        mat[:, 2] += self.padding
        mat = cv2.invertAffineTransform(mat)
        return [cv2.transform(points, mat)]

    def get_faces_for_frame(self, frame, update=False):
        """ Return the faces for the selected frame """
        if self.current_frame != frame or update:
            self.get_faces(frame)
        return self.faces

    def get_roi_for_frame(self, frame, update=False):
        """ Return the original rois for the selected frame """
        if self.current_frame != frame or update:
            self.get_faces(frame)
        return [self.original_roi(matrix) for matrix in self.matrices]

    def get_roi_size_for_frame(self, frame):
        """ Return the size of the original extract box for
            the selected frame """
        if self.current_frame != frame:
            self.get_faces(frame)
        sizes = list()
        for matrix in self.matrices:
            original_roi = self.original_roi(matrix)[0].squeeze()
            top_left, top_right  = original_roi[0], original_roi[3]
            len_x = top_right[0] - top_left[0]
            len_y = top_right[1] - top_left[1]
            if top_left[1] == top_right[1]:
                length = len_y
            else:
                length = int(((len_x ** 2) + (len_y ** 2)) ** 0.5)
            sizes.append(length)
        return sizes


    def get_aligned_landmarks_for_frame(self, frame, landmarks_xy,
                                        update=False):
        """ Return the original rois for the selected frame """
        if self.current_frame != frame or update:
            self.get_faces(frame)
        aligned_landmarks = list()
        if not self.matrices:
            return aligned_landmarks
        for idx, landmarks in enumerate(landmarks_xy):
            matrix = self.matrices[idx]
            aligned_landmarks.append(
                self.extractor.transform_points(landmarks,
                                                matrix,
                                                self.size,
                                                self.padding))
        return aligned_landmarks
