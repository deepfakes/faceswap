#!/usr/bin/env python3
""" Tool to generate masks and previews of masks for existing alignments file """
from __future__ import annotations
import logging
import os
import sys
import typing as T

from argparse import Namespace
from multiprocessing import Process

import numpy as np
from tqdm import tqdm

from lib.align import Alignments, DetectedFace, update_legacy_png_header
from lib.image import FacesLoader, ImagesLoader, ImagesSaver, encode_image

from lib.multithreading import MultiThread
from lib.utils import _video_extensions
from plugins.extract.pipeline import Extractor, ExtractMedia

from .output import Output

if T.TYPE_CHECKING:
    from lib.align.alignments import AlignmentFileDict, PNGHeaderDict
    from lib.queue_manager import EventQueue

logger = logging.getLogger(__name__)


class Mask:  # pylint:disable=too-few-public-methods
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
                  or os.path.splitext(fname)[-1].lower() in _video_extensions]
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


class _Mask:  # pylint:disable=too-few-public-methods
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
        self._update_type = arguments.processing
        self._input_is_faces = arguments.input_type == "faces"
        self._mask_type = arguments.masker

        loader = FacesLoader if self._input_is_faces else ImagesLoader
        self._loader = loader(arguments.input)
        self._alignments = self._get_alignments(arguments)

        self._output = Output(arguments, self._alignments, self._loader.file_list)

        self._counts = {"face": 0, "skip": 0, "update": 0}

        self._check_input(arguments.input)
        self._faces_saver: ImagesSaver | None = None

        self._extractor = self._get_extractor(arguments.exclude_gpus)
        self._set_correct_mask_type()
        self._extractor_input_thread = self._feed_extractor()

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

    def _get_alignments(self, arguments: Namespace) -> Alignments | None:
        """ Obtain the alignments from either the given alignments location or the default
        location.

        Parameters
        ----------
        arguments: :class:`argparse.Namespace`
            The :mod:`argparse` arguments as passed in from :mod:`tools.py`

        Returns
        -------
        ``None`` or :class:`lib.align.alignments.Alignments`:
            If output is requested, returns a :class:`lib.image.ImagesSaver` otherwise
            returns ``None``
        """
        if arguments.alignments:
            logger.debug("Alignments location provided: %s", arguments.alignments)
            return Alignments(os.path.dirname(arguments.alignments),
                              filename=os.path.basename(arguments.alignments))
        if self._input_is_faces and arguments.processing == "output":
            logger.debug("No alignments file provided for faces. Using PNG Header for output")
            return None
        if self._input_is_faces:
            logger.warning("Faces input selected without an alignments file. Masks wil only "
                           "be updated in the faces' PNG Header")
            return None

        folder = arguments.input
        if self._loader.is_video:
            logger.debug("Alignments from Video File: '%s'", folder)
            folder, filename = os.path.split(folder)
            filename = f"{os.path.splitext(filename)[0]}_alignments.fsa"
        else:
            logger.debug("Alignments from Input Folder: '%s'", folder)
            filename = "alignments"

        return Alignments(folder, filename=filename)

    def _get_extractor(self, exclude_gpus: list[int]) -> Extractor | None:
        """ Obtain a Mask extractor plugin and launch it
        Parameters
        ----------
        exclude_gpus: list or ``None``
            A list of indices correlating to connected GPUs that Tensorflow should not use. Pass
            ``None`` to not exclude any GPUs.
        Returns
        -------
        :class:`plugins.extract.pipeline.Extractor`:
            The launched Extractor
        """
        if self._update_type == "output":
            logger.debug("Update type `output` selected. Not launching extractor")
            return None
        logger.debug("masker: %s", self._mask_type)
        extractor = Extractor(None, None, self._mask_type, exclude_gpus=exclude_gpus)
        extractor.launch()
        logger.debug(extractor)
        return extractor

    def _set_correct_mask_type(self):
        """ Some masks have multiple variants that they can be saved as depending on config options
        so update the :attr:`_mask_type` accordingly
        """
        if self._extractor is None or self._mask_type != "bisenet-fp":
            return

        # Hacky look up into masker to get the type of mask
        mask_plugin = self._extractor._mask[0]  # pylint:disable=protected-access
        assert mask_plugin is not None
        mtype = "head" if mask_plugin.config.get("include_hair", False) else "face"
        new_type = f"{self._mask_type}_{mtype}"
        logger.debug("Updating '%s' to '%s'", self._mask_type, new_type)
        self._mask_type = new_type

    def _feed_extractor(self) -> MultiThread:
        """ Feed the input queue to the Extractor from a faces folder or from source frames in a
        background thread

        Returns
        -------
        :class:`lib.multithreading.Multithread`:
            The thread that is feeding the extractor.
        """
        masker_input = getattr(self, f"_input_{'faces' if self._input_is_faces else 'frames'}")
        logger.debug("masker_input: %s", masker_input)

        if self._update_type == "output":
            args: tuple = tuple()
        else:
            assert self._extractor is not None
            args = (self._extractor.input_queue, )
        input_thread = MultiThread(masker_input, *args, thread_count=1)
        input_thread.start()
        logger.debug(input_thread)
        return input_thread

    def _process_face(self,
                      filename: str,
                      image: np.ndarray,
                      metadata: PNGHeaderDict) -> ExtractMedia | None:
        """ Process a single face when masking from face images

        filename: str
            the filename currently being processed
        image: :class:`numpy.ndarray`
            The current face being processed
        metadata: dict
            The source frame metadata from the PNG header

        Returns
        -------
        :class:`plugins.pipeline.ExtractMedia` or ``None``
            If the update type is 'output' then nothing is returned otherwise the extract media for
            the face is returned
        """
        frame_name = metadata["source"]["source_filename"]
        face_index = metadata["source"]["face_index"]

        if self._alignments is None:  # mask from PNG header
            lookup_index = 0
            alignments = [T.cast("AlignmentFileDict", metadata["alignments"])]
        else:  # mask from Alignments file
            lookup_index = face_index
            alignments = self._alignments.get_faces_in_frame(frame_name)
            if not alignments or face_index > len(alignments) - 1:
                self._counts["skip"] += 1
                logger.warning("Skipping Face not found in alignments file: '%s'", filename)
                return None

        alignment = alignments[lookup_index]
        self._counts["face"] += 1

        if self._check_for_missing(frame_name, face_index, alignment):
            return None

        detected_face = self._get_detected_face(alignment)
        if self._update_type == "output":
            detected_face.image = image
            self._output.save(frame_name,
                              face_index,
                              detected_face,
                              frame_dims=metadata["source"]["source_frame_dims"])
            return None

        media = ExtractMedia(filename, image, detected_faces=[detected_face], is_aligned=True)
        media.add_frame_metadata(metadata["source"])
        self._counts["update"] += 1
        return media

    def _input_faces(self, *args: tuple | tuple[EventQueue]) -> None:
        """ Input pre-aligned faces to the Extractor plugin inside a thread

        Parameters
        ----------
        args: tuple
            The arguments that are to be loaded inside this thread. Contains the queue that the
            faces should be put to
        """
        log_once = False
        logger.debug("args: %s", args)
        if self._update_type != "output":
            queue = T.cast("EventQueue", args[0])
        for filename, image, metadata in tqdm(self._loader.load(), total=self._loader.count):
            if not metadata:  # Legacy faces. Update the headers
                if self._alignments is None:
                    logger.error("Legacy faces have been discovered, but no alignments file "
                                 "provided. You must provide an alignments file for this face set")
                    break

                if not log_once:
                    logger.warning("Legacy faces discovered. These faces will be updated")
                    log_once = True

                metadata = update_legacy_png_header(filename, self._alignments)
                if not metadata:  # Face not found
                    self._counts["skip"] += 1
                    logger.warning("Legacy face not found in alignments file. This face has not "
                                   "been updated: '%s'", filename)
                    continue

            if "source_frame_dims" not in metadata.get("source", {}):
                logger.error("The faces need to be re-extracted as at least some of them do not "
                             "contain information required to correctly generate masks.")
                logger.error("You can re-extract the face-set by using the Alignments Tool's "
                             "Extract job.")
                break
            media = self._process_face(filename, image, metadata)
            if media is not None:
                queue.put(media)

        if self._update_type != "output":
            queue.put("EOF")

    def _input_frames(self, *args: tuple | tuple[EventQueue]) -> None:
        """ Input frames to the Extractor plugin inside a thread

        Parameters
        ----------
        args: tuple
            The arguments that are to be loaded inside this thread. Contains the queue that the
            faces should be put to
        """
        assert self._alignments is not None
        logger.debug("args: %s", args)
        if self._update_type != "output":
            queue = T.cast("EventQueue", args[0])
        for filename, image in tqdm(self._loader.load(), total=self._loader.count):
            frame = os.path.basename(filename)
            if not self._alignments.frame_exists(frame):
                self._counts["skip"] += 1
                logger.warning("Skipping frame not in alignments file: '%s'", frame)
                continue
            if not self._alignments.frame_has_faces(frame):
                logger.debug("Skipping frame with no faces: '%s'", frame)
                continue

            faces_in_frame = self._alignments.get_faces_in_frame(frame)
            self._counts["face"] += len(faces_in_frame)

            # To keep face indexes correct/cover off where only one face in an image is missing a
            # mask where there are multiple faces we process all faces again for any frames which
            # have missing masks.
            if all(self._check_for_missing(frame, idx, alignment)
                   for idx, alignment in enumerate(faces_in_frame)):
                continue

            detected_faces = [self._get_detected_face(alignment) for alignment in faces_in_frame]
            if self._update_type == "output":
                for idx, detected_face in enumerate(detected_faces):
                    detected_face.image = image
                    self._output.save(frame, idx, detected_face)
            else:
                self._counts["update"] += len(detected_faces)
                queue.put(ExtractMedia(filename, image, detected_faces=detected_faces))
        if self._update_type != "output":
            queue.put("EOF")

    def _check_for_missing(self, frame: str, idx: int, alignment: AlignmentFileDict) -> bool:
        """ Check if the alignment is missing the requested mask_type

        Parameters
        ----------
        frame: str
            The frame name in the alignments file
        idx: int
            The index of the face for this frame in the alignments file
        alignment: dict
            The alignment for a face

        Returns
        -------
        bool:
            ``True`` if the update_type is "missing" and the mask does not exist in the alignments
            file otherwise ``False``
        """
        retval = (self._update_type == "missing" and
                  alignment.get("mask", None) is not None and
                  alignment["mask"].get(self._mask_type, None) is not None)
        if retval:
            logger.debug("Mask pre-exists for face: '%s' - %s", frame, idx)
        return retval

    @classmethod
    def _get_detected_face(cls, alignment: AlignmentFileDict) -> DetectedFace:
        """ Convert an alignment dict item to a detected_face object

        Parameters
        ----------
        alignment: dict
            The alignment dict for a face

        Returns
        -------
        :class:`~lib.align.detected_face.DetectedFace`:
            The corresponding detected_face object for the alignment
        """
        detected_face = DetectedFace()
        detected_face.from_alignment(alignment)
        return detected_face

    def process(self) -> None:
        """ The entry point for the Mask tool from :file:`lib.tools.cli`. Runs the Mask process """
        logger.debug("Starting masker process")
        updater = getattr(self, f"_update_{'faces' if self._input_is_faces else 'frames'}")
        if self._update_type != "output":
            assert self._extractor is not None
            if self._input_is_faces:
                self._faces_saver = ImagesSaver(self._loader.location, as_bytes=True)
            for extractor_output in self._extractor.detected_faces():
                self._extractor_input_thread.check_and_raise_error()
                updater(extractor_output)

            if self._counts["update"] != 0 and self._alignments is not None:
                self._alignments.backup()
                self._alignments.save()

            if self._input_is_faces:
                assert self._faces_saver is not None
                self._faces_saver.close()

        self._extractor_input_thread.join()
        self._output.close()

        if self._counts["skip"] != 0:
            logger.warning("%s face(s) skipped due to not existing in the alignments file",
                           self._counts["skip"])
        if self._update_type != "output":
            if self._counts["update"] == 0:
                logger.warning("No masks were updated of the %s faces seen", self._counts["face"])
            else:
                logger.info("Updated masks for %s faces of %s",
                            self._counts["update"], self._counts["face"])
        logger.debug("Completed masker process")

    def _update_faces(self, extractor_output: ExtractMedia) -> None:
        """ Update alignments for the mask if the input type is a faces folder

        If an output location has been indicated, then puts the mask preview to the save queue

        Parameters
        ----------
        extractor_output: :class:`plugins.extract.pipeline.ExtractMedia`
            The output from the :class:`plugins.extract.pipeline.Extractor` object
        """
        assert self._faces_saver is not None
        for face in extractor_output.detected_faces:
            frame_name = extractor_output.frame_metadata["source_filename"]
            face_index = extractor_output.frame_metadata["face_index"]
            logger.trace("Saving face: (frame: %s, face index: %s)",  # type: ignore
                         frame_name, face_index)

            if self._alignments is not None:
                self._alignments.update_face(frame_name, face_index, face.to_alignment())

            metadata: PNGHeaderDict = {"alignments": face.to_png_meta(),
                                       "source": extractor_output.frame_metadata}
            self._faces_saver.save(extractor_output.filename,
                                   encode_image(extractor_output.image, ".png", metadata=metadata))

            if self._output.should_save is not None:
                face.image = extractor_output.image
                self._output.save(frame_name, face_index, face)

    def _update_frames(self, extractor_output: ExtractMedia) -> None:
        """ Update alignments for the mask if the input type is a frames folder or video

        If an output location has been indicated, then puts the mask preview to the save queue

        Parameters
        ----------
        extractor_output: :class:`plugins.extract.pipeline.ExtractMedia`
            The output from the :class:`plugins.extract.pipeline.Extractor` object
        """
        assert self._alignments is not None
        frame = os.path.basename(extractor_output.filename)
        for idx, face in enumerate(extractor_output.detected_faces):
            self._alignments.update_face(frame, idx, face.to_alignment())
            if self._output.should_save:
                face.image = extractor_output.image
                self._output.save(frame, idx, face)
