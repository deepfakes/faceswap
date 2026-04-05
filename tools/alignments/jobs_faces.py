#!/usr/bin/env python3
"""Tools for manipulating the alignments using extracted Faces as a source """
from __future__ import annotations
import logging
import os
import typing as T

from argparse import Namespace
from operator import itemgetter

from tqdm import tqdm

from lib.align import DetectedFace
from lib.align.objects import AlignmentsEntry, PNGHeader, PNGSource
from lib.image import update_existing_metadata  # TODO remove
from lib.utils import get_module_objects
from scripts.fs_media import Alignments

from .media import Faces

if T.TYPE_CHECKING:
    from .media import AlignmentData
    from lib.align.objects import FileAlignments, PNGAlignments

logger = logging.getLogger(__name__)


class FromFaces():
    """Scan a folder of Faceswap Extracted Faces and re-create the associated alignments file(s)

    Parameters
    ----------
    alignments
        Parameter included for standard job naming convention, but not used for this process.
    arguments
        The :mod:`argparse` arguments as passed in from :mod:`tools.py`
    """
    def __init__(self, alignments: None, arguments: Namespace) -> None:
        logger.debug("Initializing %s: (alignments: %s, arguments: %s)",
                     self.__class__.__name__, alignments, arguments)
        self._faces_dir = arguments.faces_dir
        self._faces = Faces(arguments.faces_dir)
        logger.debug("Initialized %s", self.__class__.__name__)

    def process(self) -> None:
        """Run the job to read faces from a folder to create alignments file(s)."""
        logger.info("[CREATE ALIGNMENTS FROM FACES]")  # Tidy up cli output

        all_versions: dict[str, list[float]] = {}
        d_align: dict[str, dict[str, list[tuple[int, FileAlignments, str, PNGSource]]]] = {}
        filelist = T.cast(list[tuple[str, "PNGHeader"]], self._faces.file_list_sorted)
        for filename, meta in tqdm(filelist,
                                   desc="Generating Alignments",
                                   total=len(filelist),
                                   leave=False):

            align_fname = self._get_alignments_filename(meta.source)
            source_name, f_idx, alignment = self._extract_alignment(meta)
            full_info = (f_idx, alignment, filename, meta.source)

            d_align.setdefault(align_fname, {}).setdefault(source_name, []).append(full_info)
            all_versions.setdefault(align_fname, []).append(meta.source.alignments_version)

        versions = {k: min(v) for k, v in all_versions.items()}
        alignments = self._sort_alignments(d_align)
        self._save_alignments(alignments, versions)

    @classmethod
    def _get_alignments_filename(cls, source_data: PNGSource) -> str:
        """Obtain the name of the alignments file from the source information contained within the
        PNG metadata.

        Parameters
        ----------
        source_data
            The source information contained within a Faceswap extracted PNG

        Returns
        -------
        If the face was generated from a video file, the filename will be
        `'<video_name>_alignments.fsa'`. If it was extracted from an image file it will be
            `'alignments.fsa'`
        """
        is_video = source_data.source_is_video
        src_name = source_data.source_filename
        prefix = f"{src_name.rpartition('_')[0]}_" if is_video else ""
        retval = f"{prefix}alignments.fsa"
        logger.trace("Extracted alignments file filename: '%s'", retval)  # type:ignore
        return retval

    def _extract_alignment(self, metadata: PNGHeader) -> tuple[str, int, FileAlignments]:
        """Extract alignment data from a PNG image's itxt header.

        Formats the landmarks into a numpy array and adds in mask centering information if it is
        from an older extract.

        Parameters
        ----------
        metadata
            An extracted faces PNG Header data

        Returns
        -------
        The alignment's source frame name in position 0. The index of the face within the alignment
        file in position 1. The alignment data correctly formatted for writing to an alignments
        file in position 2
        """
        alignment = T.cast("FileAlignments", metadata.alignments)

        src = metadata.source
        frame_name = src.source_filename
        face_index = src.face_index

        logger.trace("Extracted alignment for frame: '%s', face index: %s",  # type:ignore
                     frame_name, face_index)
        return frame_name, face_index, alignment

    def _sort_alignments(self,
                         alignments: dict[str, dict[str, list[tuple[int,
                                                                    FileAlignments,
                                                                    str,
                                                                    PNGSource]]]]
                         ) -> dict[str, dict[str, AlignmentsEntry]]:
        """Sort the faces into face index order as they appeared in the original alignments file.

        If the face index stored in the png header does not match it's position in the alignments
        file (i.e. A face has been removed from a frame) then update the header of the
        corresponding png to the correct index as exists in the newly created alignments file.

        Parameters
        ----------
        alignments
            The unsorted alignments file(s) as generated from the face PNG headers, including the
            face index of the face within it's respective frame, the original face filename and
            the original face header source information

        Returns
        -------
        The alignments file dictionaries sorted into the correct face order, ready for saving
        """
        logger.info("Sorting and checking faces...")
        aln_sorted: dict[str, dict[str, AlignmentsEntry]] = {}
        for fname, frames in alignments.items():
            this_file: dict[str, AlignmentsEntry] = {}
            for frame in tqdm(sorted(frames), desc=f"Sorting {fname}", leave=False):
                this_file[frame] = AlignmentsEntry(faces=[], video_meta={})
                for real_idx, (f_id, aln, f_path, f_src) in enumerate(sorted(frames[frame],
                                                                             key=itemgetter(0))):
                    if real_idx != f_id:
                        full_path = os.path.join(self._faces_dir, f_path)
                        self._update_png_header(full_path, real_idx, aln, f_src)
                    this_file[frame].faces.append(aln)
            aln_sorted[fname] = this_file
        return aln_sorted

    @classmethod
    def _update_png_header(cls,
                           face_path: str,
                           new_index: int,
                           alignment: FileAlignments,
                           source_info: PNGSource) -> None:
        """Update the PNG header for faces where the stored index does not correspond with the
        alignments file. This can occur when frames with multiple faces have had some faces deleted
        from the faces folder.

        Updates the original filename and index in the png header.

        Parameters
        ----------
        face_path
            Full path to the saved face image that requires updating
        new_index
            The new index as it appears in the newly generated alignments file
        alignment
            The alignment information to store in the png header
        source_info
            The face source information as extracted from the original face png file
        """
        face = DetectedFace()
        face.from_alignment(alignment)
        new_filename = f"{os.path.splitext(source_info.source_filename)[0]}_{new_index}.png"

        logger.trace("Updating png header for '%s': (face index from %s to %s, "  # type:ignore
                     "original filename from '%s' to '%s'", face_path, source_info.face_index,
                     new_index, source_info.original_filename, new_filename)

        source_info.face_index = new_index
        source_info.original_filename = new_filename
        meta = PNGHeader(alignments=face.to_png_meta(), source=source_info)
        update_existing_metadata(face_path, meta)

    def _save_alignments(self,
                         all_alignments: dict[str, dict[str, AlignmentsEntry]],
                         versions: dict[str, float]) -> None:
        """Save the newly generated alignments file(s).

        If an alignments file already exists in the source faces folder, back it up rather than
        overwriting

        Parameters
        ----------
        all_alignments
            The alignment(s) dictionaries found in the faces folder. Alignment filename as key,
            corresponding alignments as value.
        versions
            The minimum version number that exists in a face set for each alignments file to be
            generated
        """
        for fname, alignments in all_alignments.items():
            version = versions[fname]
            alignments_path = os.path.join(self._faces_dir, fname)
            aln = Alignments(alignments_path, "", is_extract=True)
            aln.update_from_dict(alignments)
            aln._io._version = version  # pylint:disable=protected-access
            aln.backup()
            aln.save()


class Rename():
    """Rename faces in a folder to match their filename as stored in an alignments file.

    Parameters
    ----------
    alignments
        The alignments data loaded from an alignments file for this rename job
    arguments
        The :mod:`argparse` arguments as passed in from :mod:`tools.py`
    faces
        An optional faces object, if the rename task is being called by another job.
        Default: ``None``
    """
    def __init__(self,
                 alignments: AlignmentData,
                 arguments: Namespace | None,
                 faces: Faces | None = None) -> None:
        logger.debug("Initializing %s: (arguments: %s, faces: %s)",
                     self.__class__.__name__, arguments, faces)
        self._alignments = alignments

        kwargs = {}
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
        """Process the face renaming """
        logger.info("[RENAME FACES]")  # Tidy up cli output
        filelist = T.cast(list[tuple[str, "PNGHeader"]], self._faces.file_list_sorted)
        rename_mappings = sorted([(face[0], face[1].source.original_filename)
                                  for face in filelist
                                  if face[0] != face[1].source.original_filename],
                                 key=lambda x: x[1])
        rename_count = self._rename_faces(rename_mappings)
        logger.info("%s faces renamed", rename_count)

        copy_back = FaceToFile(self._alignments, [val[1] for val in filelist])
        if copy_back():
            self._alignments.save()

    def _rename_faces(self, filename_mappings: list[tuple[str, str]]) -> int:
        """Rename faces back to their original name as exists in the alignments file.

        If the source and destination filename are the same then skip that file.

        Parameters
        ----------
        filename_mappings
            List of tuples of (`source filename`, `destination filename`) ordered by destination
            filename

        Returns
        -------
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


class RemoveFaces():
    """Remove items from alignments file.

    Parameters
    ---------
    alignments
        The loaded alignments containing faces to be removed
    arguments
        The command line arguments that have called this job
    """
    def __init__(self, alignments: AlignmentData, arguments: Namespace) -> None:
        logger.debug("Initializing %s: (arguments: %s)", self.__class__.__name__, arguments)
        self._alignments = alignments

        self._items = Faces(arguments.faces_dir, alignments=alignments)
        logger.debug("Initialized %s", self.__class__.__name__)

    def process(self) -> None:
        """Run the job to remove faces from an alignments file that do not exist within a faces
        folder."""
        logger.info("[REMOVE FACES FROM ALIGNMENTS]")  # Tidy up cli output

        if not self._items.items:
            logger.error("No matching faces found in your faces folder. This would remove all "
                         "faces from your alignments file. Process aborted.")
            return

        items = T.cast(dict[str, list[int]], self._items.items)
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
        """Update the EXIF iTXt field of any face PNGs that have had their face index changed.

        Notes
        -----
        This could be quicker if parallelizing in threads, however, Windows (at least) does not
        seem to like this and has a tendency to throw permission errors, so this remains single
        threaded for now.
        """
        items = T.cast(dict[str, list[int]], self._items.items)
        src = [(x[0], x[1].source)
               for x in T.cast(list[tuple[str, "PNGHeader"]], self._items.file_list_sorted)]
        to_update = [  # Items whose face index has changed
            x for x in src
            if x[1].face_index != items[x[1].source_filename].index(x[1].face_index)]

        for item in tqdm(to_update, desc="Updating PNG Headers", leave=False):
            filename, file_info = item
            frame = file_info.source_filename
            face_index = file_info.face_index
            new_index = items[frame].index(face_index)

            fullpath = os.path.join(self._items.folder, filename)
            logger.debug("Updating png header for '%s': face index from %s to %s",
                         fullpath, face_index, new_index)

            # Update file_list_sorted for rename task
            orig_filename = f"{os.path.splitext(frame)[0]}_{new_index}.png"
            file_info.face_index = new_index
            file_info.original_filename = orig_filename

            face = DetectedFace()
            face.from_alignment(self._alignments.get_faces_in_frame(frame)[new_index])
            meta = PNGHeader(alignments=face.to_png_meta(),
                             source=PNGSource(alignments_version=file_info.alignments_version,
                                              original_filename=orig_filename,
                                              face_index=new_index,
                                              source_filename=frame,
                                              source_is_video=file_info.source_is_video,
                                              source_frame_dims=file_info.source_frame_dims))
            update_existing_metadata(fullpath, meta)

        logger.info("%s Extracted face(s) had their header information updated", len(to_update))


class FaceToFile():
    """Updates any optional/missing keys in the alignments file with any data that has been
    populated in a PNGHeader. Includes masks and identity fields.

    Parameters
    ---------
    alignments
        The loaded alignments containing faces to be removed
    face_data
        List of :class:`PNGHeader` objects
    """
    def __init__(self, alignments: AlignmentData, face_data: list[PNGHeader]) -> None:
        logger.debug("Initializing %s: alignments: %s, face_data: %s",
                     self.__class__.__name__, alignments, len(face_data))
        self._alignments = alignments
        self._face_alignments = face_data
        self._updatable_keys: list[T.Literal["identity", "mask"]] = ["identity", "mask"]
        self._counts: dict[str, int] = {}
        logger.debug("Initialized %s", self.__class__.__name__)

    def _check_and_update(self,
                          alignment: PNGAlignments,
                          face: FileAlignments) -> None:
        """Check whether the key requires updating and update it.

        Parameters
        ----------
        alignment
            The alignment dictionary from the PNG Header
        face
            The alignment dictionary for the face from the alignments file
        """
        for key in self._updatable_keys:
            if key == "mask":
                exist_masks = face.mask
                for mask_name, mask_data in alignment.mask.items():
                    if mask_name in exist_masks:
                        continue
                    exist_masks[mask_name] = mask_data
                    count_key = f"mask_{mask_name}"
                    self._counts[count_key] = self._counts.get(count_key, 0) + 1
                continue

            if not getattr(face, key) and getattr(alignment, key):
                setattr(face, key, getattr(alignment, key))
                self._counts[key] = self._counts.get(key, 0) + 1

    def __call__(self) -> bool:
        """Parse through the face data updating any entries in the alignments file.

        Returns
        -------
        ``True`` if any alignment information was updated otherwise ``False``
        """
        for meta in tqdm(self._face_alignments,
                         desc="Updating Alignments File from PNG Header",
                         leave=False):
            src = meta.source
            alignment = meta.alignments
            if not any(hasattr(alignment, key) for key in self._updatable_keys):
                continue

            faces = self._alignments.get_faces_in_frame(src.source_filename)
            if len(faces) < src.face_index + 1:  # list index out of range
                logger.debug("Skipped face '%s'. Index does not exist in alignments file",
                             src.original_filename)
                continue

            face = faces[src.face_index]
            self._check_and_update(alignment, face)

        retval = False
        if self._counts:
            retval = True
            logger.info("Updated alignments file from PNG Data: %s", self._counts)
        return retval


__all__ = get_module_objects(__name__)
