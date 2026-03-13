"""Loads images with metadata from disk for the sort tool"""
from __future__ import annotations

import logging
import sys
import typing as T
from collections.abc import Generator

import numpy as np
from tqdm import tqdm

from lib.image import FacesLoader, ImagesLoader, read_image_meta_batch, update_existing_metadata
from lib.utils import get_module_objects

if T.TYPE_CHECKING:
    from lib.align.alignments import PNGHeaderAlignmentsDict, PNGHeaderSourceDict

logger = logging.getLogger(__name__)


ImgMetaType: T.TypeAlias = Generator[tuple[str,
                                           np.ndarray | None,
                                           T.Union["PNGHeaderAlignmentsDict", None]], None, None]


class InfoLoader():
    """Loads aligned faces and/or face metadata

    Parameters
    ----------
    input_dir
        Full path to containing folder of faces to be supported
    loader_type
        Dictates the type of iterator that will be used. "face" just loads the image with the
        filename, "meta" just loads the image alignment data with the filename. "all" loads
        the image and the alignment data with the filename
    """
    def __init__(self,
                 input_dir: str,
                 info_type: T.Literal["face", "meta", "all"]) -> None:
        logger.debug("Initializing: %s (input_dir: %s, info_type: %s)",
                     self.__class__.__name__, input_dir, info_type)
        self._info_type = info_type
        self._iterator = None
        self._description = "Reading image statistics..."
        self._loader = ImagesLoader(input_dir) if info_type == "face" else FacesLoader(input_dir)
        self.cached_source_data: dict[str, PNGHeaderSourceDict] = {}
        """The source data read from the PNG header for each processed face"""
        if self._loader.count == 0:
            logger.error("No images to process in location: '%s'", input_dir)
            sys.exit(1)

        logger.debug("Initialized: %s", self.__class__.__name__)

    @property
    def filelist_count(self) -> int:
        """The number of files to be processed """
        return len(self._loader.file_list)

    def _get_iterator(self) -> ImgMetaType:
        """Obtain the iterator for the selected :attr:`info_type`.

        Returns
        -------
        The correct generator for the given info_type
        """
        if self._info_type == "all":
            return self._full_data_reader()
        if self._info_type == "meta":
            return self._metadata_reader()
        return self._image_data_reader()

    def __call__(self) -> ImgMetaType:
        """Return the selected iterator

        The resulting generator:

        Yields
        ------
        filename
            The filename that has been read
        image
            The aligned face image loaded from disk for 'face' and 'all' info_types
            otherwise ``None``
        alignments
            The alignments dict for 'all' and 'meta' infor_types otherwise ``None``
        """
        iterator = self._get_iterator()
        return iterator

    def _get_alignments(self,
                        filename: str,
                        metadata: dict[str, T.Any]) -> PNGHeaderAlignmentsDict | None:
        """Obtain the alignments from a PNG Header.

        The other image metadata is cached locally in case a sort method needs to write back to the
        PNG header

        Parameters
        ----------
        filename
            Full path to the image PNG file
        metadata
            The header data from a PNG file

        Returns
        -------
        The alignments dictionary from the PNG header, if it exists, otherwise ``None``
        """
        if not metadata or not metadata.get("alignments") or not metadata.get("source"):
            return None
        self.cached_source_data[filename] = metadata["source"]
        return metadata["alignments"]

    def _metadata_reader(self) -> ImgMetaType:
        """Load metadata from saved aligned faces

        Yields
        ------
        filename
            The filename that has been read
        image
            This will always be ``None`` with the metadata reader
        alignments
            The alignment data for the given face or ``None`` if no alignments found
        """
        for filename, metadata in tqdm(read_image_meta_batch(self._loader.file_list),
                                       total=self._loader.count,
                                       desc=self._description,
                                       leave=False):
            alignments = self._get_alignments(filename, metadata.get("itxt", {}))
            yield filename, None, alignments

    def _full_data_reader(self) -> ImgMetaType:
        """Load the image and metadata from a folder of aligned faces

        Yields
        ------
        filename
            The filename that has been read
        image
            The aligned face image loaded from disk
        alignments
            The alignment data for the given face or ``None`` if no alignments found
        """
        for filename, image, metadata in tqdm(self._loader.load(),
                                              desc=self._description,
                                              total=self._loader.count,
                                              leave=False):
            alignments = self._get_alignments(filename, metadata)
            yield filename, image, alignments

    def _image_data_reader(self) -> ImgMetaType:
        """Just loads the images with their filenames

        Yields
        ------
        filename
            The filename that has been read
        image
            The aligned face image loaded from disk
        alignments
            Alignments will always be ``None`` with the image data reader
        """
        for filename, image in tqdm(self._loader.load(),
                                    desc=self._description,
                                    total=self._loader.count,
                                    leave=False):
            yield filename, image, None

    def update_png_header(self, filename: str, alignments: PNGHeaderAlignmentsDict) -> None:
        """Update the PNG header of the given file with the given alignments.

        NB: Header information can only be updated if the face is already on at least alignment
        version 2.2. If below this version, then the header is not updated


        Parameters
        ----------
        filename
            Full path to the PNG file to update
        alignments: dict
            The alignments to update into the PNG header
        """
        vers = self.cached_source_data[filename]["alignments_version"]
        if vers < 2.2:
            return

        self.cached_source_data[filename]["alignments_version"] = 2.3 if vers == 2.2 else vers
        header = {"alignments": alignments, "source": self.cached_source_data[filename]}
        update_existing_metadata(filename, header)


__all__ = get_module_objects(__name__)
