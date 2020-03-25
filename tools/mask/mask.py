#!/usr/bin/env python3
""" Tool to generate masks and previews of masks for existing alignments file """
import logging
import os
import sys

import cv2
import numpy as np
from tqdm import tqdm

from lib.alignments import Alignments
from lib.faces_detect import DetectedFace
from lib.image import FacesLoader, ImagesLoader, ImagesSaver

from lib.multithreading import MultiThread
from lib.utils import get_folder
from plugins.extract.pipeline import Extractor, ExtractMedia


logger = logging.getLogger(__name__)  # pylint:disable=invalid-name


class Mask():  # pylint:disable=too-few-public-methods
    """ This tool is part of the Faceswap Tools suite and should be called from
    ``python tools.py mask`` command.

    Faceswap Masks tool. Generate masks from existing alignments files, and output masks
    for preview.

    Parameters
    ----------
    arguments: :class:`argparse.Namespace`
        The :mod:`argparse` arguments as passed in from :mod:`tools.py`
    """
    def __init__(self, arguments):
        logger.debug("Initializing %s: (arguments: %s", self.__class__.__name__, arguments)
        self._update_type = arguments.processing
        self._input_is_faces = arguments.input_type == "faces"
        self._mask_type = arguments.masker
        self._output = dict(opts=dict(blur_kernel=arguments.blur_kernel,
                                      threshold=arguments.threshold),
                            type=arguments.output_type,
                            full_frame=arguments.full_frame,
                            suffix=self._get_output_suffix(arguments))
        self._counts = dict(face=0, skip=0, update=0)

        self._check_input(arguments.input)
        self._saver = self._set_saver(arguments)
        loader = FacesLoader if self._input_is_faces else ImagesLoader
        self._loader = loader(arguments.input)
        self._alignments = Alignments(os.path.dirname(arguments.alignments),
                                      filename=os.path.basename(arguments.alignments))

        self._extractor = self._get_extractor()
        self._extractor_input_thread = self._feed_extractor()

        logger.debug("Initialized %s", self.__class__.__name__)

    def _check_input(self, mask_input):
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

    def _set_saver(self, arguments):
        """ set the saver in a background thread

        Parameters
        ----------
        arguments: :class:`argparse.Namespace`
            The :mod:`argparse` arguments as passed in from :mod:`tools.py`

        Returns
        -------
        ``None`` or :class:`lib.image.ImagesSaver`:
            If output is requested, returns a :class:`lib.image.ImagesSaver` otherwise
            returns ``None``
        """
        if not hasattr(arguments, "output") or arguments.output is None or not arguments.output:
            if self._update_type == "output":
                logger.error("Processing set as 'output' but no output folder provided.")
                sys.exit(0)
            logger.debug("No output provided. Not creating saver")
            return None
        output_dir = str(get_folder(arguments.output, make_folder=True))
        logger.info("Saving preview masks to: '%s'", output_dir)
        saver = ImagesSaver(output_dir)
        logger.debug(saver)
        return saver

    def _get_extractor(self):
        """ Obtain a Mask extractor plugin and launch it

        Returns
        -------
        :class:`plugins.extract.pipeline.Extractor`:
            The launched Extractor
        """
        if self._update_type == "output":
            logger.debug("Update type `output` selected. Not launching extractor")
            return None
        logger.debug("masker: %s", self._mask_type)
        extractor = Extractor(None, None, self._mask_type,
                              image_is_aligned=self._input_is_faces)
        extractor.launch()
        logger.debug(extractor)
        return extractor

    def _feed_extractor(self):
        """ Feed the input queue to the Extractor from a faces folder or from source frames in a
        background thread

        Returns
        -------
        :class:`lib.multithreading.Multithread`:
            The thread that is feeding the extractor.
        """
        masker_input = getattr(self,
                               "_input_{}".format("faces" if self._input_is_faces else "frames"))
        logger.debug("masker_input: %s", masker_input)

        args = tuple() if self._update_type == "output" else (self._extractor.input_queue, )
        input_thread = MultiThread(masker_input, *args, thread_count=1)
        input_thread.start()
        logger.debug(input_thread)
        return input_thread

    def _input_faces(self, *args):
        """ Input pre-aligned faces to the Extractor plugin inside a thread

        Parameters
        ----------
        args: tuple
            The arguments that are to be loaded inside this thread. Contains the queue that the
            faces should be put to
        """
        logger.debug("args: %s", args)
        if self._update_type != "output":
            queue = args[0]
        for filename, image, hsh in tqdm(self._loader.load(), total=self._loader.count):
            if hsh not in self._alignments.hashes_to_frame:
                self._counts["skip"] += 1
                logger.warning("Skipping face not in alignments file: '%s'", filename)
                continue

            frames = self._alignments.hashes_to_frame[hsh]
            if len(frames) > 1:
                # Filter the output by filename in case of multiple frames with the same face
                logger.debug("Filtering multiple hashes to current filename: (filename: '%s', "
                             "frames: %s", filename, frames)
                lookup = os.path.splitext(os.path.basename(filename))[0]
                frames = {k: v
                          for k, v in frames.items()
                          if lookup.startswith(os.path.splitext(k)[0])}
                logger.debug("Filtered: (filename: '%s', frame: '%s')", filename, frames)

            for frame, idx in frames.items():
                self._counts["face"] += 1
                alignment = self._alignments.get_faces_in_frame(frame)[idx]
                if self._check_for_missing(frame, idx, alignment):
                    continue
                detected_face = self._get_detected_face(alignment)
                if self._update_type == "output":
                    detected_face.image = image
                    self._save(frame, idx, detected_face)
                else:
                    queue.put(ExtractMedia(filename, image, detected_faces=[detected_face]))
                    self._counts["update"] += 1
        if self._update_type != "output":
            queue.put("EOF")

    def _input_frames(self, *args):
        """ Input frames to the Extractor plugin inside a thread

        Parameters
        ----------
        args: tuple
            The arguments that are to be loaded inside this thread. Contains the queue that the
            faces should be put to
        """
        logger.debug("args: %s", args)
        if self._update_type != "output":
            queue = args[0]
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
                    self._save(frame, idx, detected_face)
            else:
                self._counts["update"] += len(detected_faces)
                queue.put(ExtractMedia(filename, image, detected_faces=detected_faces))
        if self._update_type != "output":
            queue.put("EOF")

    def _check_for_missing(self, frame, idx, alignment):
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

    def _get_output_suffix(self, arguments):
        """ The filename suffix, based on selected output options.

        Parameters
        ----------
        arguments: :class:`argparse.Namespace`
            The command line arguments for the mask tool

        Returns
        -------
        str:
            The suffix to be appended to the output filename
        """
        sfx = "{}_mask_preview_".format(self._mask_type)
        sfx += "face_" if not arguments.full_frame or self._input_is_faces else "frame_"
        sfx += "{}.png".format(arguments.output_type)
        return sfx

    @staticmethod
    def _get_detected_face(alignment):
        """ Convert an alignment dict item to a detected_face object

        Parameters
        ----------
        alignment: dict
            The alignment dict for a face

        Returns
        -------
        :class:`lib.FacesDetect.detected_face`:
            The corresponding detected_face object for the alignment
        """
        detected_face = DetectedFace()
        detected_face.from_alignment(alignment)
        return detected_face

    def process(self):
        """ The entry point for the Mask tool from :file:`lib.tools.cli`. Runs the Mask process """
        logger.debug("Starting masker process")
        updater = getattr(self, "_update_{}".format("faces" if self._input_is_faces else "frames"))
        if self._update_type != "output":
            for extractor_output in self._extractor.detected_faces():
                self._extractor_input_thread.check_and_raise_error()
                updater(extractor_output)
            self._extractor_input_thread.join()
            if self._counts["update"] != 0:
                self._alignments.backup()
                self._alignments.save()
        else:
            self._extractor_input_thread.join()
            self._saver.close()

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

    def _update_faces(self, extractor_output):
        """ Update alignments for the mask if the input type is a faces folder

        If an output location has been indicated, then puts the mask preview to the save queue

        Parameters
        ----------
        extractor_output: dict
            The output from the :class:`plugins.extract.pipeline.Extractor` object
        """
        for face in extractor_output.detected_faces:
            for frame, idx in self._alignments.hashes_to_frame[face.hash].items():
                self._alignments.update_face(frame, idx, face.to_alignment())
                if self._saver is not None:
                    face.image = extractor_output.image
                    self._save(frame, idx, face)

    def _update_frames(self, extractor_output):
        """ Update alignments for the mask if the input type is a frames folder or video

        If an output location has been indicated, then puts the mask preview to the save queue

        Parameters
        ----------
        extractor_output: dict
            The output from the :class:`plugins.extract.pipeline.Extractor` object
        """
        frame = os.path.basename(extractor_output.filename)
        for idx, face in enumerate(extractor_output.detected_faces):
            self._alignments.update_face(frame, idx, face.to_alignment())
            if self._saver is not None:
                self._save(frame, idx, face)

    def _save(self, frame, idx, detected_face):
        """ Build the mask preview image and save

        Parameters
        ----------
        frame: str
            The frame name in the alignments file
        idx: int
            The index of the face for this frame in the alignments file
        detected_face: `lib.FacesDetect.detected_face`
            A detected_face object for a face
        """
        filename = os.path.join(self._saver.location, "{}_{}_{}".format(
            os.path.splitext(frame)[0],
            idx,
            self._output["suffix"]))

        if detected_face.mask is None or detected_face.mask.get(self._mask_type, None) is None:
            logger.warning("Mask type '%s' does not exist for frame '%s' index %s. Skipping",
                           self._mask_type, frame, idx)
            return
        image = self._create_image(detected_face)
        logger.trace("filename: '%s', image_shape: %s", filename, image.shape)
        self._saver.save(filename, image)

    def _create_image(self, detected_face):
        """ Create a mask preview image for saving out to disk

        Parameters
        ----------
        detected_face: `lib.FacesDetect.detected_face`
            A detected_face object for a face

        Returns
        numpy.ndarray:
            A preview image depending on the output type in one of the following forms:
              - Containing 3 sub images: The original face, the masked face and the mask
              - The mask only
              - The masked face
        """
        mask = detected_face.mask[self._mask_type]
        mask.set_blur_and_threshold(**self._output["opts"])
        if not self._output["full_frame"] or self._input_is_faces:
            if self._input_is_faces:
                face = detected_face.image
            else:
                detected_face.load_aligned(detected_face.image)
                face = detected_face.aligned_face
            mask = cv2.resize(detected_face.mask[self._mask_type].mask,
                              (face.shape[1], face.shape[0]),
                              interpolation=cv2.INTER_CUBIC)[..., None]
        else:
            face = detected_face.image
            mask = mask.get_full_frame_mask(face.shape[1], face.shape[0])
            mask = np.expand_dims(mask, -1)

        height, width = face.shape[:2]
        if self._output["type"] == "combined":
            masked = (face.astype("float32") * mask.astype("float32") / 255.).astype("uint8")
            mask = np.tile(mask, 3)
            for img in (face, masked, mask):
                cv2.rectangle(img, (0, 0), (width - 1, height - 1), (255, 255, 255), 1)
                out_image = np.concatenate((face, masked, mask), axis=1)
        elif self._output["type"] == "mask":
            out_image = mask
        elif self._output["type"] == "masked":
            out_image = np.concatenate([face, mask], axis=-1)
        return out_image
