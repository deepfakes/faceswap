#!/usr/bin/env python3
""" Tool to generate masks and previews of masks for existing alignments file """
from __future__ import annotations
import logging
import os
import sys

from argparse import Namespace
from multiprocessing import Process

from lib.align import Alignments

from lib.utils import handle_deprecated_cliopts, VIDEO_EXTENSIONS
from plugins.extract import ExtractMedia

from .loader import Loader
from .mask_import import Import
from .mask_generate import MaskGenerator
from .mask_output import Output


logger = logging.getLogger(__name__)


class Mask:
    """ This tool is part of the Faceswap Tools suite and should be called from
    ``python tools.py mask`` command.

    Faceswap Masks tool. Generate masks from existing alignments files, and output masks
    for preview.

    Wrapper for the mask process to run in either batch mode or single use mode

    Parameters
    ----------
    arguments: :class:`argparse.Namespace`
        The :mod:`argparse` arguments as passed in from :mod:`tools.py`
    """
    def __init__(self, arguments: Namespace) -> None:
        logger.debug("Initializing %s: (arguments: %s", self.__class__.__name__, arguments)
        if arguments.batch_mode and arguments.processing == "import":
            logger.error("Batch mode is not supported for 'import' processing")
            sys.exit(0)

        self._args = arguments
        self._input_locations = self._get_input_locations()

    def _get_input_locations(self) -> list[str]:
        """ Obtain the full path to input locations. Will be a list of locations if batch mode is
        selected, or containing a single location if batch mode is not selected.

        Returns
        -------
        list:
            The list of input location paths
        """
        if not self._args.batch_mode:
            return [self._args.input]

        if not os.path.isdir(self._args.input):
            logger.error("Batch mode is selected but input '%s' is not a folder", self._args.input)
            sys.exit(1)

        retval = [os.path.join(self._args.input, fname)
                  for fname in os.listdir(self._args.input)
                  if os.path.isdir(os.path.join(self._args.input, fname))
                  or os.path.splitext(fname)[-1].lower() in VIDEO_EXTENSIONS]
        logger.info("Batch mode selected. Processing locations: %s", retval)
        return retval

    def _get_output_location(self, input_location: str) -> str:
        """ Obtain the path to an output folder for faces for a given input location.

        A sub-folder within the user supplied output location will be returned based on
        the input filename

        Parameters
        ----------
        input_location: str
            The full path to an input video or folder of images
        """
        retval = os.path.join(self._args.output,
                              os.path.splitext(os.path.basename(input_location))[0])
        logger.debug("Returning output: '%s' for input: '%s'", retval, input_location)
        return retval

    @staticmethod
    def _run_mask_process(arguments: Namespace) -> None:
        """ The mask process to be run in a spawned process.

        In some instances, batch-mode memory leaks. Launching each job in a separate process
        prevents this leak.

        Parameters
        ----------
        arguments: :class:`argparse.Namespace`
            The :mod:`argparse` arguments to be used for the given job
        """
        logger.debug("Starting process: (arguments: %s)", arguments)
        mask = _Mask(arguments)
        mask.process()
        logger.debug("Finished process: (arguments: %s)", arguments)

    def process(self) -> None:
        """ The entry point for triggering the Extraction Process.

        Should only be called from  :class:`lib.cli.launcher.ScriptExecutor`
        """
        for idx, location in enumerate(self._input_locations):
            if self._args.batch_mode:
                logger.info("Processing job %s of %s: %s",
                            idx + 1, len(self._input_locations), location)
                arguments = Namespace(**self._args.__dict__)
                arguments.input = location
                # Due to differences in how alignments are handled for frames/faces, only default
                # locations allowed
                arguments.alignments = None
                if self._args.output:
                    arguments.output = self._get_output_location(location)
            else:
                arguments = self._args

            if len(self._input_locations) > 1:
                proc = Process(target=self._run_mask_process, args=(arguments, ))
                proc.start()
                proc.join()
            else:
                self._run_mask_process(arguments)


class _Mask:
    """ This tool is part of the Faceswap Tools suite and should be called from
    ``python tools.py mask`` command.

    Faceswap Masks tool. Generate masks from existing alignments files, and output masks
    for preview.

    Parameters
    ----------
    arguments: :class:`argparse.Namespace`
        The :mod:`argparse` arguments as passed in from :mod:`tools.py`
    """
    def __init__(self, arguments: Namespace) -> None:
        logger.debug("Initializing %s: (arguments: %s)", self.__class__.__name__, arguments)
        arguments = handle_deprecated_cliopts(arguments)
        self._update_type = arguments.processing
        self._input_is_faces = arguments.input_type == "faces"
        self._check_input(arguments.input)

        self._loader = Loader(arguments.input, self._input_is_faces)
        self._alignments = self._get_alignments(arguments.alignments, arguments.input)

        if self._loader.is_video and self._alignments is not None:
            self._alignments.update_legacy_has_source(os.path.basename(self._loader.location))

        self._loader.add_alignments(self._alignments)

        self._output = Output(arguments, self._alignments, self._loader.file_list)

        self._import = None
        if self._update_type == "import":
            self._import = Import(arguments.mask_path,
                                  arguments.centering,
                                  arguments.storage_size,
                                  self._input_is_faces,
                                  self._loader,
                                  self._alignments,
                                  arguments.input,
                                  arguments.masker)

        self._mask_gen: MaskGenerator | None = None
        if self._update_type in ("all", "missing"):
            self._mask_gen = MaskGenerator(arguments.masker,
                                           self._update_type == "all",
                                           self._input_is_faces,
                                           self._loader,
                                           self._alignments,
                                           arguments.input,
                                           arguments.exclude_gpus)

        logger.debug("Initialized %s", self.__class__.__name__)

    def _check_input(self, mask_input: str) -> None:
        """ Check the input is valid. If it isn't exit with a logged error

        Parameters
        ----------
        mask_input: str
            Path to the input folder/video
        """
        if not os.path.exists(mask_input):
            logger.error("Location cannot be found: '%s'", mask_input)
            sys.exit(0)
        if os.path.isfile(mask_input) and self._input_is_faces:
            logger.error("Input type 'faces' was selected but input is not a folder: '%s'",
                         mask_input)
            sys.exit(0)
        logger.debug("input '%s' is valid", mask_input)

    def _get_alignments(self, alignments: str | None, input_location: str) -> Alignments | None:
        """ Obtain the alignments from either the given alignments location or the default
        location.

        Parameters
        ----------
        alignments: str | None
            Full path to the alignemnts file if provided or ``None`` if not
        input_location: str
            Full path to the source files to be used by the mask tool

        Returns
        -------
        ``None`` or :class:`~lib.align.alignments.Alignments`:
            If output is requested, returns a :class:`~lib.align.alignments.Alignments` otherwise
            returns ``None``
        """
        if alignments:
            logger.debug("Alignments location provided: %s", alignments)
            return Alignments(os.path.dirname(alignments),
                              filename=os.path.basename(alignments))
        if self._input_is_faces and self._update_type == "output":
            logger.debug("No alignments file provided for faces. Using PNG Header for output")
            return None
        if self._input_is_faces:
            logger.warning("Faces input selected without an alignments file. Masks wil only "
                           "be updated in the faces' PNG Header")
            return None

        folder = input_location
        if self._loader.is_video:
            logger.debug("Alignments from Video File: '%s'", folder)
            folder, filename = os.path.split(folder)
            filename = f"{os.path.splitext(filename)[0]}_alignments.fsa"
        else:
            logger.debug("Alignments from Input Folder: '%s'", folder)
            filename = "alignments"

        retval = Alignments(folder, filename=filename)
        return retval

    def _save_output(self, media: ExtractMedia) -> None:
        """ Output masks to disk

        Parameters
        ----------
        media: :class:`~plugins.extract.extract_media.ExtractMedia`
            The extract media holding the faces to output
        """
        filename = os.path.basename(media.frame_metadata["source_filename"]
                                    if self._input_is_faces else media.filename)
        dims = media.frame_metadata["source_frame_dims"] if self._input_is_faces else None
        for idx, face in enumerate(media.detected_faces):
            face_idx = media.frame_metadata["face_index"] if self._input_is_faces else idx
            face.image = media.image
            self._output.save(filename, face_idx, face, frame_dims=dims)

    def _generate_masks(self) -> None:
        """ Generate masks from a mask plugin """
        assert self._mask_gen is not None

        logger.info("Generating masks")

        for media in self._mask_gen.process():
            if self._output.should_save:
                self._save_output(media)

    def _import_masks(self) -> None:
        """ Import masks that have been generated outside of faceswap """
        assert self._import is not None
        logger.info("Importing masks")

        for media in self._loader.load():
            self._import.import_mask(media)
            if self._output.should_save:
                self._save_output(media)

        if self._alignments is not None and self._import.update_count > 0:
            self._alignments.backup()
            self._alignments.save()

        if self._import.skip_count > 0:
            logger.warning("No masks were found for %s item(s), so these have not been imported",
                           self._import.skip_count)

        logger.info("Imported masks for %s faces of %s",
                    self._import.update_count, self._import.update_count + self._import.skip_count)

    def _output_masks(self) -> None:
        """ Output masks to selected output folder """
        for media in self._loader.load():
            self._save_output(media)

    def process(self) -> None:
        """ The entry point for the Mask tool from :file:`lib.tools.cli`. Runs the Mask process """
        logger.debug("Starting masker process")

        if self._update_type in ("all", "missing"):
            self._generate_masks()

        if self._update_type == "import":
            self._import_masks()

        if self._update_type == "output":
            self._output_masks()

        self._output.close()
        logger.debug("Completed masker process")
