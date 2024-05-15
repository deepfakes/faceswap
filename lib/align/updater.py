#!/usr/bin/env python3
""" Handles updating of an alignments file from an older version to the current version. """
from __future__ import annotations

import logging
import os
import typing as T

import numpy as np

from lib.logger import parse_class_init
from lib.utils import VIDEO_EXTENSIONS

logger = logging.getLogger(__name__)

if T.TYPE_CHECKING:
    from .alignments import Alignments, AlignmentFileDict


class _Updater():
    """ Base class for inheriting to test for and update of an alignments file property

    Parameters
    ----------
    alignments: :class:`~Alignments`
        The alignments object that is being tested and updated
    """
    def __init__(self, alignments: Alignments) -> None:
        logger.debug(parse_class_init(locals()))
        self._alignments = alignments
        self._needs_update = self._test()
        if self._needs_update:
            self._update()
        logger.debug("Initialized: %s", self.__class__.__name__)

    @property
    def is_updated(self) -> bool:
        """ bool. ``True`` if this updater has been run otherwise ``False`` """
        return self._needs_update

    def _test(self) -> bool:
        """ Calls the child's :func:`test` method and logs output

        Returns
        -------
        bool
            ``True`` if the test condition is met otherwise ``False``
        """
        logger.debug("checking %s", self.__class__.__name__)
        retval = self.test()
        logger.debug("legacy %s: %s", self.__class__.__name__, retval)
        return retval

    def test(self) -> bool:
        """ Override to set the condition to test for.

        Returns
        -------
        bool
            ``True`` if the test condition is met otherwise ``False``
        """
        raise NotImplementedError()

    def _update(self) -> int:
        """ Calls the child's :func:`update` method, logs output and sets the
        :attr:`is_updated` flag

        Returns
        -------
        int
            The number of items that were updated
        """
        retval = self.update()
        logger.debug("Updated %s: %s", self.__class__.__name__, retval)
        return retval

    def update(self) -> int:
        """ Override to set the action to perform on the alignments object if the test has
        passed

        Returns
        -------
        int
            The number of items that were updated
        """
        raise NotImplementedError()


class VideoExtension(_Updater):
    """ Alignments files from video files used to have a dummy '.png' extension for each of the
    keys. This has been changed to be file extension of the original input video (for better)
    identification of alignments files generated from video files

    Parameters
    ----------
    alignments: :class:`~Alignments`
        The alignments object that is being tested and updated
    video_filename: str
        The video filename that holds these alignments
    """
    def __init__(self, alignments: Alignments, video_filename: str) -> None:
        self._video_name, self._extension = os.path.splitext(video_filename)
        super().__init__(alignments)

    def test(self) -> bool:
        """ Requires update if the extension of the key in the alignment file is not the same
        as for the input video file

        Returns
        -------
        bool
            ``True`` if the key extensions need updating otherwise ``False``
        """
        # Note: Don't check on alignments file version. It's possible that the file gets updated to
        # a newer version before this check is run
        if self._extension.lower() not in VIDEO_EXTENSIONS:
            return False

        exts = set(os.path.splitext(k)[-1] for k in self._alignments.data)
        if len(exts) != 1:
            logger.debug("Alignments file has multiple key extensions. Skipping")
            return False

        if self._extension in exts:
            logger.debug("Alignments file contains correct key extensions. Skipping")
            return False

        logger.debug("Needs update for video extension (version: %s, extension: %s)",
                     self._alignments.version, self._extension)
        return True

    def update(self) -> int:
        """ Update alignments files that have been extracted from videos to have the key end in the
        video file extension rather than ',png' (the old way)

        Parameters
        ----------
        video_filename: str
            The filename of the video file that created these alignments
        """
        updated = 0
        for key in list(self._alignments.data):
            fname = os.path.splitext(key)[0]
            if fname.rsplit("_", maxsplit=1)[0] != self._video_name:
                continue  # Key is from a different source

            val = self._alignments.data[key]
            new_key = f"{fname}{self._extension}"

            del self._alignments.data[key]
            self._alignments.data[new_key] = val

            updated += 1

        logger.debug("Updated alignment keys for video extension: %s", updated)
        return updated


class FileStructure(_Updater):
    """ Alignments were structured: {frame_name: <list of faces>}. We need to be able to store
    information at the frame level, so new structure is:  {frame_name: {faces: <list of faces>}}
    """
    def test(self) -> bool:
        """ Test whether the alignments file is laid out in the old structure of
        `{frame_name: [faces]}`

        Returns
        -------
        bool
            ``True`` if the file has legacy structure otherwise ``False``
        """
        return any(isinstance(val, list) for val in self._alignments.data.values())

    def update(self) -> int:
        """ Update legacy alignments files from the format `{frame_name: [faces}` to the
        format `{frame_name: {faces: [faces]}`.

        Returns
        -------
        int
            The number of items that were updated
        """
        updated = 0
        for key, val in self._alignments.data.items():
            if not isinstance(val, list):
                continue
            self._alignments.data[key] = {"faces": val}
            updated += 1
        return updated


class LandmarkRename(_Updater):
    """ Landmarks renamed from landmarksXY to landmarks_xy for PEP compliance """
    def test(self) -> bool:
        """ check for legacy landmarksXY keys.

        Returns
        -------
        bool
            ``True`` if the alignments file contains legacy `landmarksXY` keys otherwise ``False``
        """
        return (any(key == "landmarksXY"
                    for val in self._alignments.data.values()
                    for alignment in val["faces"]
                    for key in alignment))

    def update(self) -> int:
        """ Update legacy `landmarksXY` keys to PEP compliant `landmarks_xy` keys.

        Returns
        -------
        int
            The number of landmarks keys that were changed
        """
        update_count = 0
        for val in self._alignments.data.values():
            for alignment in val["faces"]:
                if "landmarksXY" in alignment:
                    alignment["landmarks_xy"] = alignment.pop("landmarksXY")  # type:ignore
                    update_count += 1
        return update_count


class ListToNumpy(_Updater):
    """ Landmarks stored as list instead of numpy array """
    def test(self) -> bool:
        """ check for legacy landmarks stored as `list` rather than :class:`numpy.ndarray`.

        Returns
        -------
        bool
            ``True`` if not all landmarks are :class:`numpy.ndarray` otherwise ``False``
        """
        return not all(isinstance(face["landmarks_xy"], np.ndarray)
                       for val in self._alignments.data.values()
                       for face in val["faces"])

    def update(self) -> int:
        """ Update landmarks stored as `list` to :class:`numpy.ndarray`.

        Returns
        -------
        int
            The number of landmarks keys that were changed
        """
        update_count = 0
        for val in self._alignments.data.values():
            for alignment in val["faces"]:
                test = alignment["landmarks_xy"]
                if not isinstance(test, np.ndarray):
                    alignment["landmarks_xy"] = np.array(test, dtype="float32")
                    update_count += 1
        return update_count


class MaskCentering(_Updater):
    """ Masks not containing the stored_centering parameters. Prior to this implementation all
    masks were stored with face centering """

    def test(self) -> bool:
        """ Mask centering was introduced in alignments version 2.2

        Returns
        -------
        bool
            ``True`` mask centering requires updating otherwise ``False``
        """
        return self._alignments.version < 2.2

    def update(self) -> int:
        """ Add the mask key to the alignment file and update the centering of existing masks

        Returns
        -------
        int
            The number of masks that were updated
        """
        update_count = 0
        for val in self._alignments.data.values():
            for alignment in val["faces"]:
                if "mask" not in alignment:
                    alignment["mask"] = {}
                for mask in alignment["mask"].values():
                    mask["stored_centering"] = "face"
                    update_count += 1
        return update_count


class IdentityAndVideoMeta(_Updater):
    """ Prior to version 2.3 the identity key did not exist and the video_meta key was not
    compulsory. These should now both always appear, but do not need to be populated. """

    def test(self) -> bool:
        """ Identity Key was introduced in alignments version 2.3

        Returns
        -------
        bool
            ``True`` identity key needs inserting otherwise ``False``
        """
        return self._alignments.version < 2.3

    # Identity information was not previously stored in the alignments file.
    def update(self) -> int:
        """ Add the video_meta and identity keys to the alignment file and leave empty

        Returns
        -------
        int
            The number of keys inserted
        """
        update_count = 0
        for val in self._alignments.data.values():
            this_update = 0
            if "video_meta" not in val:
                val["video_meta"] = {}
                this_update = 1
            for alignment in val["faces"]:
                if "identity" not in alignment:
                    alignment["identity"] = {}
                    this_update = 1
                update_count += this_update
        return update_count


class Legacy():
    """ Legacy alignments properties that are no longer used, but are still required for backwards
    compatibility/upgrading reasons.

    Parameters
    ----------
    alignments: :class:`~Alignments`
        The alignments object that requires these legacy properties
    """
    def __init__(self, alignments: Alignments) -> None:
        self._alignments = alignments
        self._hashes_to_frame: dict[str, dict[str, int]] = {}
        self._hashes_to_alignment: dict[str, AlignmentFileDict] = {}

    @property
    def hashes_to_frame(self) -> dict[str, dict[str, int]]:
        """ dict: The SHA1 hash of the face mapped to the frame(s) and face index within the frame
        that the hash corresponds to. The structure of the dictionary is:

        {**SHA1_hash** (`str`): {**filename** (`str`): **face_index** (`int`)}}.

        Notes
        -----
        This method is deprecated and exists purely for updating legacy hash based alignments
        to new png header storage in :class:`lib.align.update_legacy_png_header`.

        The first time this property is referenced, the dictionary will be created and cached.
        Subsequent references will be made to this cached dictionary.
        """
        if not self._hashes_to_frame:
            logger.debug("Generating hashes to frame")
            for frame_name, val in self._alignments.data.items():
                for idx, face in enumerate(val["faces"]):
                    self._hashes_to_frame.setdefault(
                        face["hash"], {})[frame_name] = idx  # type:ignore
        return self._hashes_to_frame

    @property
    def hashes_to_alignment(self) -> dict[str, AlignmentFileDict]:
        """ dict: The SHA1 hash of the face mapped to the alignment for the face that the hash
        corresponds to. The structure of the dictionary is:

        Notes
        -----
        This method is deprecated and exists purely for updating legacy hash based alignments
        to new png header storage in :class:`lib.align.update_legacy_png_header`.

        The first time this property is referenced, the dictionary will be created and cached.
        Subsequent references will be made to this cached dictionary.
        """
        if not self._hashes_to_alignment:
            logger.debug("Generating hashes to alignment")
            self._hashes_to_alignment = {face["hash"]: face  # type:ignore
                                         for val in self._alignments.data.values()
                                         for face in val["faces"]}
        return self._hashes_to_alignment
