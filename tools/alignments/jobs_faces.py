#!/usr/bin/env python3
""" Tools for manipulating the alignments using extracted Faces as a source """
import os
import logging
from argparse import Namespace
from operator import itemgetter
from typing import cast, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

import numpy as np
from tqdm import tqdm

from lib.align import DetectedFace
from lib.image import update_existing_metadata  # TODO remove
from scripts.fsmedia import Alignments

from .media import Faces

if TYPE_CHECKING:
    from .media import AlignmentData
    from lib.align.alignments import (AlignmentDict, AlignmentFileDict,
                                      PNGHeaderDict, PNGHeaderSourceDict)

logger = logging.getLogger(__name__)


class FromFaces():  # pylint:disable=too-few-public-methods
    """ Scan a folder of Faceswap Extracted Faces and re-create the associated alignments file(s)

    Parameters
    ----------
    alignments: NoneType
        Parameter included for standard job naming convention, but not used for this process.
    arguments: :class:`argparse.Namespace`
        The :mod:`argparse` arguments as passed in from :mod:`tools.py`
    """
    def __init__(self, alignments: None, arguments: Namespace) -> None:
        logger.debug("Initializing %s: (alignments: %s, arguments: %s)",
                     self.__class__.__name__, alignments, arguments)
        self._faces_dir = arguments.faces_dir
        self._faces = Faces(arguments.faces_dir, with_alignments=True)
        logger.debug("Initialized %s", self.__class__.__name__)

    def process(self) -> None:
        """ Run the job to read faces from a folder to create alignments file(s). """
        logger.info("[CREATE ALIGNMENTS FROM FACES]")  # Tidy up cli output

        all_versions: Dict[str, List[float]] = {}
        d_align: Dict[str, Dict[str, List[Tuple[int, "AlignmentFileDict", str, dict]]]] = {}
        filelist = cast(List[Tuple[str, "PNGHeaderDict"]], self._faces.file_list_sorted)
        for filename, meta in tqdm(filelist,
                                   desc="Generating Alignments",
                                   total=len(filelist),
                                   leave=False):

            align_fname = self._get_alignments_filename(meta["source"])
            source_name, f_idx, alignment = self._extract_alignment(meta)
            full_info = (f_idx, alignment, filename, meta["source"])

            d_align.setdefault(align_fname, {}).setdefault(source_name, []).append(full_info)
            all_versions.setdefault(align_fname, []).append(meta["source"]["alignments_version"])

        versions = {k: min(v) for k, v in all_versions.items()}
        alignments = self._sort_alignments(d_align)
        self._save_alignments(alignments, versions)

    @classmethod
    def _get_alignments_filename(cls, source_data: dict) -> str:
        """ Obtain the name of the alignments file from the source information contained within the
        PNG metadata.

        Parameters
        ----------
        source_data: dict
            The source information contained within a Faceswap extracted PNG

        Returns
        -------
        str:
            If the face was generated from a video file, the filename will be
            `'<video_name>_alignments.fsa'`. If it was extracted from an image file it will be
            `'alignments.fsa'`
        """
        is_video = source_data["source_is_video"]
        src_name = source_data["source_filename"]
        prefix = f"{src_name.rpartition('_')[0]}_" if is_video else ""
        retval = f"{prefix}alignments.fsa"
        logger.trace("Extracted alignments file filename: '%s'", retval)  # type:ignore
        return retval

    def _extract_alignment(self, metadata: dict) -> Tuple[str, int, "AlignmentFileDict"]:
        """ Extract alignment data from a PNG image's itxt header.

        Formats the landmarks into a numpy array and adds in mask centering information if it is
        from an older extract.

        Parameters
        ----------
        metadata: dict
            An extracted faces PNG Header data

        Returns
        -------
        tuple
            The alignment's source frame name in position 0. The index of the face within the
            alignment file in position 1. The alignment data correctly formatted for writing to an
            alignments file in positin 2
        """
        alignment = metadata["alignments"]
        alignment["landmarks_xy"] = np.array(alignment["landmarks_xy"], dtype="float32")

        src = metadata["source"]
        frame_name = src["source_filename"]
        face_index = int(src["face_index"])

        logger.trace("Extracted alignment for frame: '%s', face index: %s",  # type:ignore
                     frame_name, face_index)
        return frame_name, face_index, alignment

    def _sort_alignments(self,
                         alignments: Dict[str, Dict[str, List[Tuple[int,
                                                                    "AlignmentFileDict",
                                                                    str,
                                                                    dict]]]]
                         ) -> Dict[str, Dict[str, "AlignmentDict"]]:
        """ Sort the faces into face index order as they appeared in the original alignments file.

        If the face index stored in the png header does not match it's position in the alignments
        file (i.e. A face has been removed from a frame) then update the header of the
        corresponding png to the correct index as exists in the newly created alignments file.

        Parameters
        ----------
        alignments: dict
            The unsorted alignments file(s) as generated from the face PNG headers, including the
            face index of the face within it's respective frame, the original face filename and
            the orignal face header source information

        Returns
        -------
        dict
            The alignments file dictionaries sorted into the correct face order, ready for saving
        """
        logger.info("Sorting and checking faces...")
        aln_sorted: Dict[str, Dict[str, "AlignmentDict"]] = {}
        for fname, frames in alignments.items():
            this_file: Dict[str, "AlignmentDict"] = {}
            for frame in tqdm(sorted(frames), desc=f"Sorting {fname}", leave=False):
                this_file[frame] = dict(video_meta={}, faces=[])
                for real_idx, (f_id, almt, f_path, f_src) in enumerate(sorted(frames[frame],
                                                                              key=itemgetter(0))):
                    if real_idx != f_id:
                        self._update_png_header(f_path, real_idx, almt, f_src)
                    this_file[frame]["faces"].append(almt)
            aln_sorted[fname] = this_file
        return aln_sorted

    @classmethod
    def _update_png_header(cls,
                           face_path: str,
                           new_index: int,
                           alignment: "AlignmentFileDict",
                           source_info: dict) -> None:
        """ Update the PNG header for faces where the stored index does not correspond with the
        alignments file. This can occur when frames with multiple faces have had some faces deleted
        from the faces folder.

        Updates the original filename and index in the png header.

        Parameters
        ----------
        face_path: str
            Full path to the saved face image that requires updating
        new_index: int
            The new index as it appears in the newly generated alignments file
        alignment: dict
            The alignment information to store in the png header
        source_info: dict
            The face source information as extracted from the original face png file
        """
        face = DetectedFace()
        face.from_alignment(alignment)
        new_filename = f"{os.path.splitext(source_info['source_filename'])[0]}_{new_index}.png"

        logger.trace("Updating png header for '%s': (face index from %s to %s, "  # type:ignore
                     "original filename from '%s' to '%s'", face_path, source_info["face_index"],
                     new_index, source_info["original_filename"], new_filename)

        source_info["face_index"] = new_index
        source_info["original_filename"] = new_filename
        meta = dict(alignments=face.to_png_meta(), source=source_info)
        update_existing_metadata(face_path, meta)

    def _save_alignments(self,
                         all_alignments: Dict[str, Dict[str, "AlignmentDict"]],
                         versions: Dict[str, float]) -> None:
        """ Save the newely generated alignments file(s).

        If an alignments file already exists in the source faces folder, back it up rather than
        overwriting

        Parameters
        ----------
        all_alignments: dict
            The alignment(s) dictionaries found in the faces folder. Alignment filename as key,
            corresponding alignments as value.
        versions: dict
            The minimum version number that exists in a face set for each alignments file to be
            generated
        """
        for fname, alignments in all_alignments.items():
            version = versions[fname]
            alignments_path = os.path.join(self._faces_dir, fname)
            dummy_args = Namespace(alignments_path=alignments_path)
            aln = Alignments(dummy_args, is_extract=True)
            aln._data = alignments  # pylint:disable=protected-access
            aln._io._version = version  # pylint:disable=protected-access
            aln._io.update_legacy()  # pylint:disable=protected-access
            aln.backup()
            aln.save()


class Rename():  # pylint:disable=too-few-public-methods
    """ Rename faces in a folder to match their filename as stored in an alignments file.

    Parameters
    ----------
    alignments: :class:`tools.lib_alignments.media.AlignmentData`
        The alignments data loaded from an alignments file for this rename job
    arguments: :class:`argparse.Namespace`
        The :mod:`argparse` arguments as passed in from :mod:`tools.py`
    faces: :class:`tools.lib_alignments.media.Faces`, Optional
        An optional faces object, if the rename task is being called by another job.
        Default: ``None``
    """
    def __init__(self,
                 alignments: "AlignmentData",
                 arguments: Optional[Namespace],
                 faces: Optional[Faces] = None) -> None:
        logger.debug("Initializing %s: (arguments: %s, faces: %s)",
                     self.__class__.__name__, arguments, faces)
        self._alignments = alignments

        kwargs: Dict[str, Union[bool, "AlignmentData"]] = dict(with_alignments=False)
        if alignments.version < 2.1:
            # Update headers of faces generated with hash based alignments
            kwargs["alignments"] = alignments
        if faces:
            self._faces = faces
        else:
            assert arguments is not None
            self._faces = Faces(arguments.faces_dir, **kwargs)  # type:ignore  # needs TypedDict :/
        logger.debug("Initialized %s", self.__class__.__name__)

    def process(self) -> None:
        """ Process the face renaming """
        logger.info("[RENAME FACES]")  # Tidy up cli output
        filelist = cast(List[Tuple[str, "PNGHeaderSourceDict"]], self._faces.file_list_sorted)
        rename_mappings = sorted([(face[0], face[1]["original_filename"])
                                  for face in filelist
                                  if face[0] != face[1]["original_filename"]],
                                 key=lambda x: x[1])
        rename_count = self._rename_faces(rename_mappings)
        logger.info("%s faces renamed", rename_count)

    def _rename_faces(self, filename_mappings: List[Tuple[str, str]]) -> int:
        """ Rename faces back to their original name as exists in the alignments file.

        If the source and destination filename are the same then skip that file.

        Parameters
        ----------
        filename_mappings: list
            List of tuples of (`source filename`, `destination filename`) ordered by destination
            filename

        Returns
        -------
        int
            The number of faces that have been renamed
        """
        if not filename_mappings:
            return 0

        rename_count = 0
        conflicts = []
        for src, dst in tqdm(filename_mappings, desc="Renaming Faces", leave=False):
            old = os.path.join(self._faces.folder, src)
            new = os.path.join(self._faces.folder, dst)

            if os.path.exists(new):
                # Interim add .tmp extension to files that will cause a rename conflict, to
                # process afterwards
                logger.debug("interim renaming file to avoid conflict: (src: '%s', dst: '%s')",
                             src, dst)
                new = new + ".tmp"
                conflicts.append(new)

            logger.verbose("Renaming '%s' to '%s'", old, new)  # type:ignore
            os.rename(old, new)
            rename_count += 1
        if conflicts:
            for old in tqdm(conflicts, desc="Renaming Faces", leave=False):
                new = old[:-4]  # Remove .tmp extension
                if os.path.exists(new):
                    # This should only be running on faces. If there is still a conflict
                    # then the user has done something stupid, so we will delete the file and
                    # replace. They can always re-extract :/
                    os.remove(new)
                logger.verbose("Renaming '%s' to '%s'", old, new)  # type:ignore
                os.rename(old, new)
        return rename_count


class RemoveFaces():  # pylint:disable=too-few-public-methods
    """ Remove items from alignments file.

    Parameters
    ---------
    alignments: :class:`tools.alignments.media.AlignmentsData`
        The loaded alignments containing faces to be removed
    arguments: :class:`argparse.Namespace`
        The command line arguments that have called this job
    """
    def __init__(self, alignments: "AlignmentData", arguments: Namespace) -> None:
        logger.debug("Initializing %s: (arguments: %s)", self.__class__.__name__, arguments)
        self._alignments = alignments

        kwargs: Dict[str, Union[bool, "AlignmentData"]] = dict(with_alignments=False)
        if alignments.version < 2.1:
            # Update headers of faces generated with hash based alignments
            kwargs["alignments"] = alignments
        self._items = Faces(arguments.faces_dir, **kwargs)  # type:ignore  # needs TypedDict :/
        logger.debug("Initialized %s", self.__class__.__name__)

    def process(self) -> None:
        """ Run the job to remove faces from an alignments file that do not exist within a faces
        folder. """
        logger.info("[REMOVE FACES FROM ALIGNMENTS]")  # Tidy up cli output

        if not self._items.items:
            logger.error("No matching faces found in your faces folder. This would remove all "
                         "faces from your alignments file. Process aborted.")
            return

        items = cast(Dict[str, List[int]], self._items.items)
        pre_face_count = self._alignments.faces_count
        self._alignments.filter_faces(items, filter_out=False)
        del_count = pre_face_count - self._alignments.faces_count
        if del_count == 0:
            logger.info("No changes made to alignments file. Exiting")
            return

        logger.info("%s alignment(s) were removed from alignments file", del_count)

        self._update_png_headers()
        self._alignments.save()

        rename = Rename(self._alignments, None, self._items)
        rename.process()

    def _update_png_headers(self) -> None:
        """ Update the EXIF iTXt field of any face PNGs that have had their face index changed.

        Notes
        -----
        This could be quicker if parellizing in threads, however, Windows (at least) does not seem
        to like this and has a tendency to throw permission errors, so this remains single threaded
        for now.
        """
        filelist = cast(List[Tuple[str, "PNGHeaderSourceDict"]], self._items.file_list_sorted)
        items = cast(Dict[str, List[int]], self._items.items)
        to_update = [  # Items whose face index has changed
            x for x in filelist
            if x[1]["face_index"] != items[x[1]["source_filename"]].index(x[1]["face_index"])]

        for item in tqdm(to_update, desc="Updating PNG Headers", leave=False):
            filename, file_info = item
            frame = file_info["source_filename"]
            face_index = file_info["face_index"]
            new_index = items[frame].index(face_index)

            fullpath = os.path.join(self._items.folder, filename)
            logger.debug("Updating png header for '%s': face index from %s to %s",
                         fullpath, face_index, new_index)

            # Update file_list_sorted for rename task
            orig_filename = f"{os.path.splitext(frame)[0]}_{new_index}.png"
            file_info["face_index"] = new_index
            file_info["original_filename"] = orig_filename

            face = DetectedFace()
            face.from_alignment(self._alignments.get_faces_in_frame(frame)[new_index])
            meta = dict(alignments=face.to_png_meta(),
                        source=dict(alignments_version=file_info["alignments_version"],
                                    original_filename=orig_filename,
                                    face_index=new_index,
                                    source_filename=frame,
                                    source_is_video=file_info["source_is_video"],
                                    source_frame_dims=file_info.get("source_frame_dims")))
            update_existing_metadata(fullpath, meta)

        logger.info("%s Extracted face(s) had their header information updated", len(to_update))
