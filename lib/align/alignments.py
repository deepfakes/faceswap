#!/usr/bin/env python3
"""Alignments file functions for reading, writing and manipulating the data stored in a
serialized alignments file. """
from __future__ import annotations
import logging
import os
import sys
import typing as T
from datetime import datetime


from lib.serializer import get_serializer
from lib.utils import FaceswapError, get_module_objects

from .objects import AlignmentsEntry, FileAlignments

from .thumbnails import Thumbnails
from .updater import (FileStructure, IdentityAndVideoMeta, LandmarkRename, NumpyToList,
                      MaskCentering, VideoExtension)

if T.TYPE_CHECKING:
    from collections.abc import Generator

logger = logging.getLogger(__name__)
_VERSION = 2.4
# VERSION TRACKING
# 1.0 - Never really existed. Basically any alignments file prior to version 2.0
# 2.0 - Implementation of full head extract. Any alignments version below this will have used
#       legacy extract
# 2.1 - Alignments data to extracted face PNG header. SHA1 hashes of faces no longer calculated
#       or stored in alignments file
# 2.2 - Add support for differently centered masks (i.e. not all masks stored as face centering)
# 2.3 - Add 'identity' key to alignments file. May or may not be populated, to contain vggface2
#       embeddings. Make 'video_meta' key a standard key. Can be unpopulated
# 2.4 - Update video file alignment keys to end in the video extension rather than '.png'


class Alignments():  # pylint:disable=too-many-public-methods
    """The alignments file is a custom serialized ``.fsa`` file that holds information for each
    frame for a video or series of images.

    Specifically, it holds a list of faces that appear in each frame. Each face contains
    information detailing their detected bounding box location within the frame, the 68 point
    facial landmarks and any masks that have been extracted.

    Additionally it can also hold video meta information (timestamp and whether a frame is a
    key frame.)

    Parameters
    ----------
    folder
        The folder that contains the alignments ``.fsa`` file
    filename
        The filename of the ``.fsa`` alignments file. If not provided then the given folder will be
        checked for a default alignments file filename. Default: "alignments"
    """
    def __init__(self, folder: str, filename: str = "alignments") -> None:
        logger.debug("Initializing %s: (folder: '%s', filename: '%s')",
                     self.__class__.__name__, folder, filename)
        self._io = _IO(self, folder, filename)
        self._data = self._load()
        self._thumbnails = Thumbnails(self)
        logger.debug("Initialized %s", self.__class__.__name__)

    # << PROPERTIES >> #

    @property
    def frames_count(self) -> int:
        """The number of frames that appear in the alignments :attr:`data`."""
        retval = len(self._data)
        logger.trace(retval)  # type:ignore[attr-defined]
        return retval

    @property
    def faces_count(self) -> int:
        """The total number of faces that appear in the alignments :attr:`data`"""
        retval = sum(len(val.faces) for val in self._data.values())
        logger.trace(retval)  # type:ignore[attr-defined]
        return retval

    @property
    def file(self) -> str:
        """The full path to the currently loaded alignments file."""
        return self._io.file

    @property
    def data(self) -> dict[str, AlignmentsEntry]:
        """The loaded alignments :attr:`file` in dictionary form."""
        return self._data

    @property
    def have_alignments_file(self) -> bool:
        """``True`` if an alignments file exists at location :attr:`file` otherwise ``False``."""
        return self._io.have_alignments_file

    @property
    def mask_summary(self) -> dict[str, int]:
        """The mask type names stored in the alignments :attr:`data` as key with the number of
        faces which possess the mask type as value."""
        masks: dict[str, int] = {}
        for val in self._data.values():
            for face in val.faces:
                if not face.mask:
                    masks["none"] = masks.get("none", 0) + 1
                for key in face.mask:
                    masks[key] = masks.get(key, 0) + 1
        return masks

    @property
    def video_meta_data(self) -> dict[T.Literal["pts_time", "keyframes"], list[int]] | None:
        """The frame meta data stored in the alignments file. If data does not exist in the
        alignments file then ``None`` is returned"""
        retval: dict[T.Literal["pts_time", "keyframes"], list[int]] = {}
        pts_time: list[int] = []
        keyframes: list[int] = []
        for idx, key in enumerate(sorted(self.data)):
            if not self.data[key].video_meta:
                return None
            meta = self.data[key].video_meta
            if not isinstance(meta["pts_time"], int):
                # pts_time is now stored as ints so let it regenerate
                return None
            pts_time.append(meta["pts_time"])
            if meta["keyframe"]:
                keyframes.append(idx)
        retval = {"pts_time": pts_time, "keyframes": keyframes}
        return retval

    @property
    def thumbnails(self) -> Thumbnails:
        """The low resolution thumbnail images that exist within the alignments file"""
        return self._thumbnails

    @property
    def version(self) -> float:
        """float: The alignments file version number. """
        return self._io.version

    def _load(self) -> dict[str, AlignmentsEntry]:
        """Load the alignments data from the serialized alignments :attr:`file`.

        Populates :attr:`_version` with the alignment file's loaded version as well as returning
        the serialized data.

        Returns
        -------
        The loaded alignments data
        """
        return self._io.load()

    def save(self) -> None:
        """Write the contents of :attr:`data` and :attr:`_meta` to a serialized ``.fsa`` file at
        the location :attr:`file`."""
        return self._io.save()

    def backup(self) -> None:
        """Create a backup copy of the alignments :attr:`file`.

        Creates a copy of the serialized alignments :attr:`file` appending a
        timestamp onto the end of the file name and storing in the same folder as
        the original :attr:`file`.
        """
        return self._io.backup()

    def save_video_meta_data(self, pts_time: list[int], keyframes: list[int]) -> None:
        """Save video meta data to the alignments file.

        If the alignments file does not have an entry for every frame (e.g. if Extract Every N
        was used) then the frame is added to the alignments file with no faces, so that they video
        meta data can be stored.

        Parameters
        ----------
        pts_time
            A list of presentation timestamps (`int`) in frame index order for every frame in
            the input video
        keyframes
            A list of frame indices corresponding to the key frames in the input video
        """
        sample_filename = next(fname for fname in self.data)
        basename = sample_filename[:sample_filename.rfind("_")]
        ext = os.path.splitext(sample_filename)[-1]
        logger.debug("sample filename: '%s', base filename: '%s' extension: '%s'",
                     sample_filename, basename, ext)
        logger.info("Saving video meta information to Alignments file")

        for idx, pts in enumerate(pts_time):
            meta:  dict[T.Literal["pts_time", "keyframe"], int] = {"pts_time": pts,
                                                                   "keyframe": idx in keyframes}
            key = f"{basename}_{idx + 1:06d}{ext}"
            if key not in self.data:
                self.data[key] = AlignmentsEntry(video_meta=meta)
            else:
                self.data[key].video_meta = meta

        logger.debug("Alignments count: %s, timestamp count: %s", len(self.data), len(pts_time))
        if len(self.data) != len(pts_time):
            raise FaceswapError(
                "There is a mismatch between the number of frames found in the video file "
                f"({len(pts_time)}) and the number of frames found in the alignments file "
                f"({len(self.data)}).\nThis can be caused by a number of issues:"
                "\n  - The video has a Variable Frame Rate and FFMPEG is having a hard time "
                "calculating the correct number of frames."
                "\n  - You are working with a Merged Alignments file. This is not supported for "
                "your current use case."
                "\nYou should either extract the video to individual frames, re-encode the "
                "video at a constant frame rate and re-run extraction or work with a dedicated "
                "alignments file for your requested video.")
        self._io.save()

    # << VALIDATION >> #
    def frame_exists(self, frame_name: str) -> bool:
        """Check whether a given frame_name exists within the alignments :attr:`data`.

        Parameters
        ----------
        frame_name
            The frame name to check. This should be the base name of the frame, not the full path

        Returns
        -------
        ``True`` if the given frame_name exists within the alignments :attr:`data` otherwise
        ``False``
        """
        retval = frame_name in self._data.keys()
        logger.trace("'%s': %s", frame_name, retval)  # type:ignore[attr-defined]
        return retval

    def frame_has_faces(self, frame_name: str) -> bool:
        """Check whether a given frame_name exists within the alignments :attr:`data` and contains
        at least 1 face.

        Parameters
        ----------
        frame_name
            The frame name to check. This should be the base name of the frame, not the full path

        Returns
        -------
        ``True`` if the given frame_name exists within the alignments :attr:`data` and has at least
        1 face associated with it, otherwise ``False``
        """
        frame_data = self._data.get(frame_name, AlignmentsEntry())
        retval = bool(frame_data.faces)
        logger.trace("'%s': %s", frame_name, retval)  # type:ignore[attr-defined]
        return retval

    def frame_has_multiple_faces(self, frame_name: str) -> bool:
        """Check whether a given frame_name exists within the alignments :attr:`data` and contains
        more than 1 face.

        Parameters
        ----------
        frame_name
            The frame_name name to check. This should be the base name of the frame, not the full
            path

        Returns
        -------
        ``True`` if the given frame_name exists within the alignments :attr:`data` and has more
        than 1 face associated with it, otherwise ``False``
        """
        if not frame_name:
            retval = False
        else:
            frame_data = self._data.get(frame_name, AlignmentsEntry)
            retval = bool(len(frame_data.faces) > 1)
        logger.trace("'%s': %s", frame_name, retval)  # type:ignore[attr-defined]
        return retval

    def mask_is_valid(self, mask_type: str) -> bool:
        """Ensure the given ``mask_type`` is valid for the alignments :attr:`data`.

        Every face in the alignments :attr:`data` must have the given mask type to successfully
        pass the test.

        Parameters
        ----------
        mask_type
            The mask type to check against the current alignments :attr:`data`

        Returns
        -------
        ``True`` if all faces in the current alignments possess the given ``mask_type`` otherwise
        ``False``
        """
        retval = all(face.mask.get(mask_type) is not None
                     for val in self._data.values()
                     for face in val.faces)
        logger.debug(retval)
        return retval

    # << DATA >> #
    def get_faces_in_frame(self, frame_name: str) -> list[FileAlignments]:
        """Obtain the faces from :attr:`data` associated with a given frame_name.

        Parameters
        ----------
        frame_name
            The frame name to return faces for. This should be the base name of the frame, not the
            full path

        Returns
        -------
        The list of face dictionaries that appear within the requested frame_name
        """
        logger.trace("Getting faces for frame_name: '%s'", frame_name)  # type:ignore[attr-defined]
        frame_data = self._data.get(frame_name, AlignmentsEntry())
        return frame_data.faces

    def count_faces_in_frame(self, frame_name: str) -> int:
        """Return number of faces that appear within :attr:`data` for the given frame_name.

        Parameters
        ----------
        frame_name
            The frame name to return the count for. This should be the base name of the frame, not
            the full path

        Returns
        -------
        The number of faces that appear in the given frame_name
        """
        frame_data = self._data.get(frame_name, AlignmentsEntry())
        retval = len(frame_data.faces)
        logger.trace(retval)  # type:ignore[attr-defined]
        return retval

    # << MANIPULATION >> #
    def delete_face_at_index(self, frame_name: str, face_index: int) -> bool:
        """Delete the face for the given frame_name at the given face index from :attr:`data`.

        Parameters
        ----------
        frame_name
            The frame name to remove the face from. This should be the base name of the frame, not
            the full path
        face_index
            The index number of the face within the given frame_name to remove

        Returns
        -------
        ``True`` if a face was successfully deleted otherwise ``False``
        """
        logger.debug("Deleting face %s for frame_name '%s'", face_index, frame_name)
        face_index = int(face_index)
        if face_index + 1 > self.count_faces_in_frame(frame_name):
            logger.debug("No face to delete: (frame_name: '%s', face_index %s)",
                         frame_name, face_index)
            return False
        del self._data[frame_name].faces[face_index]
        logger.debug("Deleted face: (frame_name: '%s', face_index %s)", frame_name, face_index)
        return True

    def add_face(self, frame_name: str, face: FileAlignments) -> int:
        """Add a new face for the given frame_name in :attr:`data` and return it's index.

        Parameters
        ----------
        frame_name
            The frame name to add the face to. This should be the base name of the frame, not the
            full path
        face
            The face information to add to the given frame_name, correctly formatted for storing in
            :attr:`data`

        Returns
        -------
        The index of the newly added face within :attr:`data` for the given frame_name
        """
        logger.debug("Adding face to frame_name: '%s'", frame_name)
        if frame_name not in self._data:
            self._data[frame_name] = AlignmentsEntry()
        self._data[frame_name].faces.append(face)
        retval = self.count_faces_in_frame(frame_name) - 1
        logger.debug("Returning new face index: %s", retval)
        return retval

    def update_face(self, frame_name: str, face_index: int, face: FileAlignments) -> None:
        """Update the face for the given frame_name at the given face index in :attr:`data`.

        Parameters
        ----------
        frame_name
            The frame name to update the face for. This should be the base name of the frame, not
            the full path
        face_index
            The index number of the face within the given frame_name to update
        face
            The face information to update to the given frame_name at the given face_index,
            correctly formatted for storing in :attr:`data`
        """
        logger.debug("Updating face %s for frame_name '%s'", face_index, frame_name)
        self._data[frame_name].faces[face_index] = face

    def filter_faces(self, filter_dict: dict[str, list[int]], filter_out: bool = False) -> None:
        """Remove faces from :attr:`data` based on a given filter list.

        Parameters
        ----------
        filter_dict
            Dictionary of source filenames as key with a list of face indices to filter as value.
        filter_out
            ``True`` if faces should be removed from :attr:`data` when there is a corresponding
            match in the given filter_dict. ``False`` if faces should be kept in :attr:`data` when
            there is a corresponding match in the given filter_dict, but removed if there is no
            match. Default: ``False``
        """
        logger.debug("filter_dict: %s, filter_out: %s", filter_dict, filter_out)
        for source_frame, frame_data in self._data.items():
            face_indices = filter_dict.get(source_frame, [])
            if filter_out:
                filter_list = face_indices
            else:
                filter_list = [idx for idx in range(len(frame_data.faces))
                               if idx not in face_indices]
            logger.trace("frame: '%s', filter_list: %s",  # type:ignore[attr-defined]
                         source_frame, filter_list)

            for face_idx in reversed(sorted(filter_list)):
                logger.verbose(  # type:ignore[attr-defined]
                    "Filtering out face: (filename: %s, index: %s)", source_frame, face_idx)
                del frame_data.faces[face_idx]

    def update_from_dict(self, data: dict[str, AlignmentsEntry]) -> None:
        """Replace all alignments with the contents of the given dictionary

        Parameters
        ----------
        data
            The alignments, in correctly formatted dictionary form, to be populated into this
            :class:`Alignments`
        """
        logger.debug("Populating alignments with %s entries", len(data))
        self._data = data

    # << GENERATORS >> #
    def yield_faces(self) -> Generator[tuple[str, list[FileAlignments], int, str], None, None]:
        """Generator to obtain all faces with meta information from :attr:`data`. The results
        are yielded by frame.

        Notes
        -----
        The yielded order is non-deterministic.

        Yields
        ------
        frame_name
            The frame name that the face belongs to. This is the base name of the frame, as it
            appears in :attr:`data`, not the full path
        faces
            The list of face `dict` objects that exist for this frame
        face_count
            The number of faces that exist within :attr:`data` for this frame
        frame_fullname
            The full path (folder and filename) for the yielded frame
        """
        for frame_fullname, val in self._data.items():
            frame_name = os.path.splitext(frame_fullname)[0]
            face_count = len(val.faces)
            logger.trace(  # type:ignore[attr-defined]
                "Yielding: (frame: '%s', faces: %s, frame_fullname: '%s')",
                frame_name, face_count, frame_fullname)
            yield frame_name, val.faces, face_count, frame_fullname

    def update_legacy_has_source(self, filename: str) -> None:
        """Update legacy alignments files when we have the source filename available.

        Updates here can only be performed when we have the source filename

        Parameters
        ----------
        filename
            The filename/folder of the original source images/video for the current alignments
        """
        updates = [updater.is_updated
                   for updater in (VideoExtension(self._data, self.version, filename), )]
        if any(updates):
            self._io.update_version()
            self.save()


class _IO():
    """Class to handle the saving/loading of an alignments file.

    Parameters
    ----------
    alignments
        The parent alignments class that these IO operations belong to
    folder
        The folder that contains the alignments ``.fsa`` file
    filename
        The filename of the ``.fsa`` alignments file.
    """
    def __init__(self, alignments: Alignments, folder: str, filename: str) -> None:
        logger.debug("Initializing %s: (alignments: %s)", self.__class__.__name__, alignments)
        self._alignments = alignments
        self._serializer = get_serializer("compressed")
        self._file = self._get_location(folder, filename)
        self._version: float = _VERSION

    @property
    def file(self) -> str:
        """The full path to the currently loaded alignments file."""
        return self._file

    @property
    def version(self) -> float:
        """The alignments file version number."""
        return self._version

    @property
    def have_alignments_file(self) -> bool:
        """``True`` if an alignments file exists at location :attr:`file` otherwise ``False``."""
        retval = os.path.exists(self._file)
        logger.trace(retval)  # type:ignore[attr-defined]
        return retval

    def _get_location(self, folder: str, filename: str) -> str:
        """Obtains the location of an alignments file.

        Parameters
        ----------
        folder
            The folder that the alignments file is located in
        filename
            The filename of the alignments file

        Returns
        -------
        The full path to the alignments file
        """
        logger.debug("Getting location: (folder: '%s', filename: '%s')", folder, filename)
        assert self._serializer is not None
        no_ext_name, extension = os.path.splitext(filename)
        if extension[1:] == self._serializer.file_extension:
            logger.debug("Valid Alignments filename provided: '%s'", filename)
        else:
            filename = f"{no_ext_name}.{self._serializer.file_extension}"
            logger.debug("File extension set from serializer: '%s'",
                         self._serializer.file_extension)
        location = os.path.join(str(folder), filename)

        logger.verbose("Alignments filepath: '%s'", location)  # type:ignore[attr-defined]
        return location

    def update_version(self) -> None:
        """Update the version of the alignments file to the latest version"""
        self._version = _VERSION
        logger.info("Updating alignments file to version %s", self._version)

    def _update_legacy(self, alignments_dict: dict[str, T.Any]) -> bool:
        """Check whether the alignments are legacy, and if so update them to current alignments
        format.

        Parameters
        ----------
        alignments_dict
            The serialized alignments data loaded from disk
        version
            The alignments file version that has been loaded

        Returns
        -------
        ``True`` if the alignments were updated otherwise ``False``
        """
        updates = [updater.is_updated for updater in (
            FileStructure(alignments_dict, self._version),
            LandmarkRename(alignments_dict, self._version),
            NumpyToList(alignments_dict, self._version),
            MaskCentering(alignments_dict, self._version),
            IdentityAndVideoMeta(alignments_dict, self._version))]
        if any(updates):
            self.update_version()
        return any(updates)

    def load(self) -> dict[str, AlignmentsEntry]:
        """Load the alignments data from the serialized alignments :attr:`file`.

        Populates :attr:`_version` with the alignment file's loaded version as well as returning
        the serialized data.

        Returns
        -------
        The loaded alignments data
        """
        logger.debug("Loading alignments")
        if not self.have_alignments_file:
            raise FaceswapError(f"Alignments file not found at {self._file}")

        logger.info("Reading alignments from: '%s'", self._file)
        data = self._serializer.load(self._file)
        meta = data.get("__meta__", {"version": 1.0})
        self._version = meta["version"]
        if self._version < 2.0:
            logger.error("This alignments file was generated with a very old legacy extraction "
                         "method.")
            logger.error("Updating these very old files is no longer supported.")
            logger.error("To update to a more recent, supported format, you should run the "
                         "alignments tool's 'extract' job with this file in Faceswap v2.3: "
                         "https://github.com/deepfakes/faceswap/releases/tag/v2.3.0")
            sys.exit(1)

        alignments = data["__data__"]
        if self._update_legacy(alignments):
            logger.info("Writing alignments to: '%s'", self._file)
            self._serializer.save(self._file, {"__meta__": {"version": self._version},
                                               "__data__": alignments})
        retval: dict[str, AlignmentsEntry]
        retval = {k: AlignmentsEntry.from_dict(v) for k, v in alignments.items()}
        logger.debug("Loaded alignments")
        return retval

    def save(self) -> None:
        """Write the contents of :attr:`data` and :attr:`_meta` to a serialized ``.fsa`` file at
        the location :attr:`file`."""
        logger.debug("Saving alignments")
        logger.info("Writing alignments to: '%s'", self._file)
        data = {"__meta__": {"version": self._version},
                "__data__": {k: v.to_dict() for k, v in self._alignments.data.items()}}
        self._serializer.save(self._file, data)
        logger.debug("Saved alignments")

    def backup(self) -> None:
        """Create a backup copy of the alignments :attr:`file`.

        Creates a copy of the serialized alignments :attr:`file` appending a
        timestamp onto the end of the file name and storing in the same folder as
        the original :attr:`file`.
        """
        logger.debug("Backing up alignments")
        if not os.path.isfile(self._file):
            logger.debug("No alignments to back up")
            return
        now = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
        src = self._file
        split = os.path.splitext(src)
        dst = f"{split[0]}_bk_{now}{split[1]}"
        idx = 1
        while True:
            if not os.path.exists(dst):
                break
            logger.debug("Backup file %s exists. Incrementing", dst)
            dst = f"{split[0]}_{now}({idx}){split[1]}"
            idx += 1

        logger.info("Backing up original alignments to '%s'", dst)
        os.rename(src, dst)
        logger.debug("Backed up alignments")


__all__ = get_module_objects(__name__)
