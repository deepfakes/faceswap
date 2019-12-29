#!/usr/bin/env python3
""" Alignments file functions for reading, writing and manipulating
    a serialized alignments file """

import logging
import os
from datetime import datetime

import numpy as np

from lib.serializer import get_serializer, get_serializer_from_filename
from lib.utils import FaceswapError

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Alignments():
    """ Holds processes pertaining to the alignments file.

        folder:     folder alignments file is stored in
        filename:   Filename of alignments file. If a
                    valid extension is provided, then it will be used to
                    decide the serializer otherwise compressed pickle is used.
    """
    # pylint: disable=too-many-public-methods
    def __init__(self, folder, filename="alignments"):
        logger.debug("Initializing %s: (folder: '%s', filename: '%s')",
                     self.__class__.__name__, folder, filename)
        self.serializer = get_serializer("compressed")
        self.file = self.get_location(folder, filename)

        self.data = self.load()
        self.update_legacy()
        self._hashes_to_frame = dict()
        logger.debug("Initialized %s", self.__class__.__name__)

    # << PROPERTIES >> #

    @property
    def frames_count(self):
        """ Return current frames count """
        retval = len(self.data)
        logger.trace(retval)
        return retval

    @property
    def faces_count(self):
        """ Return current faces count """
        retval = sum(len(faces) for faces in self.data.values())
        logger.trace(retval)
        return retval

    @property
    def have_alignments_file(self):
        """ Return whether an alignments file exists """
        retval = os.path.exists(self.file)
        logger.trace(retval)
        return retval

    @property
    def hashes_to_frame(self):
        """ Return :attr:`_hashes_to_frame`. Generate it if it does not exist.
            The dict is of each face_hash with their parent frame name(s) and their index
            in the frame
        """
        if not self._hashes_to_frame:
            logger.debug("Generating hashes to frame")
            for frame_name, faces in self.data.items():
                for idx, face in enumerate(faces):
                    self._hashes_to_frame.setdefault(face["hash"], dict())[frame_name] = idx
        return self._hashes_to_frame

    # << INIT FUNCTIONS >> #

    def get_location(self, folder, filename):
        """ Return the path to alignments file """
        logger.debug("Getting location: (folder: '%s', filename: '%s')", folder, filename)
        noext_name, extension = os.path.splitext(filename)
        if extension in (".json", ".p", ".pickle", ".yaml", ".yml"):
            # Reformat legacy alignments file
            filename = self.update_file_format(folder, filename)
            logger.debug("Updated legacy alignments. New filename: '%s'", filename)
        if extension[1:] == self.serializer.file_extension:
            logger.debug("Valid Alignments filename provided: '%s'", filename)
        else:
            filename = "{}.{}".format(noext_name, self.serializer.file_extension)
            logger.debug("File extension set from serializer: '%s'",
                         self.serializer.file_extension)
        location = os.path.join(str(folder), filename)
        if not os.path.exists(location):
            # Test for old format alignments files and reformat if they exist
            # This will be executed if an alignments file has not been explicitly provided
            # therefore it will not have been picked up in the extension test
            self.test_for_legacy(location)
        logger.verbose("Alignments filepath: '%s'", location)
        return location

    # << I/O >> #

    def load(self):
        """ Load the alignments data
            Override for custom loading logic """
        logger.debug("Loading alignments")
        if not self.have_alignments_file:
            raise FaceswapError("Error: Alignments file not found at "
                                "{}".format(self.file))

        logger.info("Reading alignments from: '%s'", self.file)
        data = self.serializer.load(self.file)
        logger.debug("Loaded alignments")
        return data

    def reload(self):
        """ Read the alignments data from the correct format """
        logger.debug("Re-loading alignments")
        self.data = self.load()
        logger.debug("Re-loaded alignments")

    def save(self):
        """ Write the serialized alignments file """
        logger.debug("Saving alignments")
        logger.info("Writing alignments to: '%s'", self.file)
        self.serializer.save(self.file, self.data)
        logger.debug("Saved alignments")

    def backup(self):
        """ Backup copy of old alignments """
        logger.debug("Backing up alignments")
        if not os.path.isfile(self.file):
            logger.debug("No alignments to back up")
            return
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        src = self.file
        split = os.path.splitext(src)
        dst = split[0] + "_" + now + split[1]
        logger.info("Backing up original alignments to '%s'", dst)
        os.rename(src, dst)
        logger.debug("Backed up alignments")

    # << VALIDATION >> #

    def frame_exists(self, frame):
        """ return path of images that have faces """
        retval = frame in self.data.keys()
        logger.trace("'%s': %s", frame, retval)
        return retval

    def frame_has_faces(self, frame):
        """ Return true if frame exists and has faces """
        retval = bool(self.data.get(frame, list()))
        logger.trace("'%s': %s", frame, retval)
        return retval

    def frame_has_multiple_faces(self, frame):
        """ Return true if frame exists and has faces """
        if not frame:
            retval = False
        else:
            retval = bool(len(self.data.get(frame, list())) > 1)
        logger.trace("'%s': %s", frame, retval)
        return retval

    def mask_is_valid(self, mask_type):
        """ Ensure the given ``mask_type`` is valid for this alignments file.

        Every face in the alignments file must have the given mask type to successfully
        pass the test.

        Parameters
        ----------
        mask_type: str
            The mask type to check against the current alignments

        Returns
        -------
        bool:
            ``True`` if all faces in the current alignments possess the given ``mask_type``
            otherwise ``False``
        """
        retval = any([(face.get("mask", None) is not None and
                       face["mask"].get(mask_type, None) is not None)
                      for faces in self.data.values()
                      for face in faces])
        logger.debug(retval)
        return retval

    @property
    def mask_summary(self):
        """ Dict: The mask types and the number of faces which have each type that exist with in
        the loaded alignments """
        masks = dict()
        for faces in self.data.values():
            for face in faces:
                if face.get("mask", None) is None:
                    masks["none"] = masks.get("none", 0) + 1
                for key in face.get("mask", dict):
                    masks[key] = masks.get(key, 0) + 1
        return masks

    # << DATA >> #

    def get_faces_in_frame(self, frame):
        """ Return the alignments for the selected frame """
        logger.trace("Getting faces for frame: '%s'", frame)
        return self.data.get(frame, list())

    def get_full_frame_name(self, frame):
        """ Return a frame with extension for when the extension is
            not known """
        retval = next(key for key in self.data.keys()
                      if key.startswith(frame))
        logger.trace("Requested: '%s', Returning: '%s'", frame, retval)
        return retval

    def count_faces_in_frame(self, frame):
        """ Return number of alignments within frame """
        retval = len(self.data.get(frame, list()))
        logger.trace(retval)
        return retval

    # << MANIPULATION >> #

    def delete_face_at_index(self, frame, idx):
        """ Delete the face alignment for given frame at given index """
        logger.debug("Deleting face %s for frame '%s'", idx, frame)
        idx = int(idx)
        if idx + 1 > self.count_faces_in_frame(frame):
            logger.debug("No face to delete: (frame: '%s', idx %s)", frame, idx)
            return False
        del self.data[frame][idx]
        logger.debug("Deleted face: (frame: '%s', idx %s)", frame, idx)
        return True

    def add_face(self, frame, alignment):
        """ Add a new face for a frame and return it's index """
        logger.debug("Adding face to frame: '%s'", frame)
        if frame not in self.data:
            self.data[frame] = []
        self.data[frame].append(alignment)
        retval = self.count_faces_in_frame(frame) - 1
        logger.debug("Returning new face index: %s", retval)
        return retval

    def update_face(self, frame, idx, alignment):
        """ Replace a face for given frame and index """
        logger.debug("Updating face %s for frame '%s'", idx, frame)
        self.data[frame][idx] = alignment

    def filter_hashes(self, hashlist, filter_out=False):
        """ Filter in or out faces that match the hash list

            filter_out=True: Remove faces that match in the hash list
            filter_out=False: Remove faces that are not in the hash list
        """
        hashset = set(hashlist)
        for filename, frame in self.data.items():
            for idx, face in reversed(list(enumerate(frame))):
                if ((filter_out and face.get("hash", None) in hashset) or
                        (not filter_out and face.get("hash", None) not in hashset)):
                    logger.verbose("Filtering out face: (filename: %s, index: %s)", filename, idx)
                    del frame[idx]
                else:
                    logger.trace("Not filtering out face: (filename: %s, index: %s)",
                                 filename, idx)

    # << GENERATORS >> #

    def yield_faces(self):
        """ Yield face alignments for one image """
        for frame_fullname, alignments in self.data.items():
            frame_name = os.path.splitext(frame_fullname)[0]
            face_count = len(alignments)
            logger.trace("Yielding: (frame: '%s', faces: %s, frame_fullname: '%s')",
                         frame_name, face_count, frame_fullname)
            yield frame_name, alignments, face_count, frame_fullname

    @staticmethod
    def yield_original_index_reverse(image_alignments, number_alignments):
        """ Return the correct original index for
            alignment in reverse order """
        for idx, _ in enumerate(reversed(image_alignments)):
            original_idx = number_alignments - 1 - idx
            logger.trace("Yielding: face index %s", original_idx)
            yield original_idx

    # << LEGACY FUNCTIONS >> #

    def update_legacy(self):
        """ Update legacy alignments """
        updated = False
        if self.has_legacy_landmarksxy():
            logger.info("Updating legacy landmarksXY to landmarks_xy")
            self.update_legacy_landmarksxy()
            updated = True
        if self.has_legacy_landmarks_list():
            logger.info("Updating legacy landmarks from list to numpy array")
            self.update_legacy_landmarks_list()
            updated = True
        if updated:
            self.save()

    # <File Format> #
    # Serializer is now a compressed pickle .fsa format. This used to be any number of serializers
    def test_for_legacy(self, location):
        """ For alignments filenames passed in with out an extension, test for legacy formats """
        logger.debug("Checking for legacy alignments file formats: '%s'", location)
        filename = os.path.splitext(location)[0]
        for ext in (".json", ".p", ".pickle", ".yaml"):
            legacy_filename = "{}{}".format(filename, ext)
            if os.path.exists(legacy_filename):
                logger.debug("Legacy alignments file exists: '%s'", legacy_filename)
                _ = self.update_file_format(*os.path.split(legacy_filename))
                break
            logger.debug("Legacy alignments file does not exist: '%s'", legacy_filename)

    def update_file_format(self, folder, filename):
        """ Convert old style alignments format to new style format """
        logger.info("Reformatting legacy alignments file...")
        old_location = os.path.join(str(folder), filename)
        new_location = "{}.{}".format(os.path.splitext(old_location)[0],
                                      self.serializer.file_extension)
        if os.path.exists(old_location):
            if os.path.exists(new_location):
                logger.info("Using existing updated alignments file found at '%s'. If you do not "
                            "wish to use this existing file then you should delete or rename it.",
                            new_location)
            else:
                logger.info("Old location: '%s', New location: '%s'", old_location, new_location)
                load_serializer = get_serializer_from_filename(old_location)
                data = load_serializer.load(old_location)
                self.serializer.save(new_location, data)
        return os.path.basename(new_location)

    # <landmarks> #
    # Landmarks renamed from landmarksXY to landmarks_xy for PEP compliance
    def has_legacy_landmarksxy(self):
        """ check for legacy landmarksXY keys """
        logger.debug("checking legacy landmarksXY")
        retval = (any(key == "landmarksXY"
                      for alignments in self.data.values()
                      for alignment in alignments
                      for key in alignment))
        logger.debug("legacy landmarksXY: %s", retval)
        return retval

    def update_legacy_landmarksxy(self):
        """ Update landmarksXY to landmarks_xy and save alignments """
        update_count = 0
        for alignments in self.data.values():
            for alignment in alignments:
                alignment["landmarks_xy"] = alignment.pop("landmarksXY")
                update_count += 1
        logger.debug("Updated landmarks_xy: %s", update_count)

    # Landmarks stored as list instead of numpy array
    def has_legacy_landmarks_list(self):
        """ check for legacy landmarks stored as list """
        logger.debug("checking legacy landmarks as list")
        retval = not all(isinstance(face["landmarks_xy"], np.ndarray)
                         for faces in self.data.values()
                         for face in faces)
        return retval

    def update_legacy_landmarks_list(self):
        """ Update landmarksXY to landmarks_xy and save alignments """
        update_count = 0
        for alignments in self.data.values():
            for alignment in alignments:
                test = alignment["landmarks_xy"]
                if not isinstance(test, np.ndarray):
                    alignment["landmarks_xy"] = np.array(test, dtype="float32")
                    update_count += 1
        logger.debug("Updated landmarks_xy: %s", update_count)
