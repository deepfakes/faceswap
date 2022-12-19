#!/usr/bin/env python3
""" Tool to generate masks and previews of masks for existing alignments file """
import logging
import os
import sys
from typing import cast, List, Optional, Tuple, TYPE_CHECKING, Union

import cv2
import numpy as np
from tqdm import tqdm

from lib.align import Alignments, AlignedFace, DetectedFace, update_legacy_png_header
from lib.image import FacesLoader, ImagesLoader, ImagesSaver, encode_image

from lib.multithreading import MultiThread
from lib.utils import get_folder
from plugins.extract.pipeline import Extractor, ExtractMedia

if TYPE_CHECKING:
    from argparse import Namespace
    from lib.align.aligned_face import CenteringType
    from lib.align.alignments import AlignmentFileDict, PNGHeaderDict
    from lib.queue_manager import EventQueue

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
    def __init__(self, arguments: "Namespace") -> None:
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
        self._faces_saver: Optional[ImagesSaver] = None

        self._alignments = Alignments(os.path.dirname(arguments.alignments),
                                      filename=os.path.basename(arguments.alignments))

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

    def _set_saver(self, arguments: "Namespace") -> Optional[ImagesSaver]:
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
        output_dir = get_folder(arguments.output, make_folder=True)
        logger.info("Saving preview masks to: '%s'", output_dir)
        saver = ImagesSaver(output_dir)
        logger.debug(saver)
        return saver

    def _get_extractor(self, exclude_gpus: List[int]) -> Optional[Extractor]:
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
                      metadata: "PNGHeaderDict") -> Optional["ExtractMedia"]:
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
        alignments = self._alignments.get_faces_in_frame(frame_name)
        if not alignments or face_index > len(alignments) - 1:
            self._counts["skip"] += 1
            logger.warning("Skipping Face not found in alignments file: '%s'", filename)
            return None
        alignment = alignments[face_index]
        self._counts["face"] += 1

        if self._check_for_missing(frame_name, face_index, alignment):
            return None

        detected_face = self._get_detected_face(alignment)
        if self._update_type == "output":
            detected_face.image = image
            self._save(frame_name, face_index, detected_face)
            return None

        media = ExtractMedia(filename, image, detected_faces=[detected_face], is_aligned=True)
        media.add_frame_metadata(metadata["source"])
        self._counts["update"] += 1
        return media

    def _input_faces(self, *args: Union[tuple, Tuple["EventQueue"]]) -> None:
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
            queue = cast("EventQueue", args[0])
        for filename, image, metadata in tqdm(self._loader.load(), total=self._loader.count):
            if not metadata:  # Legacy faces. Update the headers
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

    def _input_frames(self, *args: Union[tuple, Tuple["EventQueue"]]) -> None:
        """ Input frames to the Extractor plugin inside a thread

        Parameters
        ----------
        args: tuple
            The arguments that are to be loaded inside this thread. Contains the queue that the
            faces should be put to
        """
        logger.debug("args: %s", args)
        if self._update_type != "output":
            queue = cast("EventQueue", args[0])
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

    def _check_for_missing(self, frame: str, idx: int, alignment: "AlignmentFileDict") -> bool:
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

    def _get_output_suffix(self, arguments: "Namespace") -> str:
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
        sfx = "mask_preview_"
        sfx += "face_" if not arguments.full_frame or self._input_is_faces else "frame_"
        sfx += f"{arguments.output_type}.png"
        return sfx

    @staticmethod
    def _get_detected_face(alignment: "AlignmentFileDict") -> DetectedFace:
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
            if self._counts["update"] != 0:
                self._alignments.backup()
                self._alignments.save()
            if self._input_is_faces:
                assert self._faces_saver is not None
                self._faces_saver.close()

        self._extractor_input_thread.join()
        if self._saver is not None:
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

            self._alignments.update_face(frame_name, face_index, face.to_alignment())
            metadata: "PNGHeaderDict" = dict(alignments=face.to_png_meta(),
                                             source=extractor_output.frame_metadata)
            self._faces_saver.save(extractor_output.filename,
                                   encode_image(extractor_output.image, ".png", metadata=metadata))

            if self._saver is not None:
                face.image = extractor_output.image
                self._save(frame_name, face_index, face)

    def _update_frames(self, extractor_output: ExtractMedia) -> None:
        """ Update alignments for the mask if the input type is a frames folder or video

        If an output location has been indicated, then puts the mask preview to the save queue

        Parameters
        ----------
        extractor_output: :class:`plugins.extract.pipeline.ExtractMedia`
            The output from the :class:`plugins.extract.pipeline.Extractor` object
        """
        frame = os.path.basename(extractor_output.filename)
        for idx, face in enumerate(extractor_output.detected_faces):
            self._alignments.update_face(frame, idx, face.to_alignment())
            if self._saver is not None:
                face.image = extractor_output.image
                self._save(frame, idx, face)

    def _save(self, frame: str, idx: int, detected_face: DetectedFace) -> None:
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
        assert self._saver is not None
        if self._mask_type == "bisenet-fp":
            mask_types = [f"{self._mask_type}_{area}" for area in ("face", "head")]
        else:
            mask_types = [self._mask_type]

        if detected_face.mask is None or not any(mask in detected_face.mask
                                                 for mask in mask_types):
            logger.warning("Mask type '%s' does not exist for frame '%s' index %s. Skipping",
                           self._mask_type, frame, idx)
            return

        for mask_type in mask_types:
            if mask_type not in detected_face.mask:
                # If extracting bisenet mask, then skip versions which don't exist
                continue
            filename = os.path.join(
                self._saver.location,
                f"{os.path.splitext(frame)[0]}_{idx}_{mask_type}_{self._output['suffix']}")
            image = self._create_image(detected_face, mask_type)
            logger.trace("filename: '%s', image_shape: %s", filename, image.shape)  # type: ignore
            self._saver.save(filename, image)

    def _create_image(self, detected_face: DetectedFace, mask_type: str) -> np.ndarray:
        """ Create a mask preview image for saving out to disk

        Parameters
        ----------
        detected_face: `lib.FacesDetect.detected_face`
            A detected_face object for a face
        mask_type: str
            The stored mask type name to create the image for

        Returns
        -------
        :class:`numpy.ndarray`:
            A preview image depending on the output type in one of the following forms:
              - Containing 3 sub images: The original face, the masked face and the mask
              - The mask only
              - The masked face
        """
        mask = detected_face.mask[mask_type]
        assert detected_face.image is not None
        mask.set_blur_and_threshold(**self._output["opts"])
        if not self._output["full_frame"] or self._input_is_faces:
            if self._input_is_faces:
                face = AlignedFace(detected_face.landmarks_xy,
                                   image=detected_face.image,
                                   centering=mask.stored_centering,
                                   size=detected_face.image.shape[0],
                                   is_aligned=True).face
            else:
                centering: "CenteringType" = ("legacy" if self._alignments.version == 1.0
                                              else mask.stored_centering)
                detected_face.load_aligned(detected_face.image, centering=centering, force=True)
                face = detected_face.aligned.face
            assert face is not None
            imask = cv2.resize(detected_face.mask[mask_type].mask,
                               (face.shape[1], face.shape[0]),
                               interpolation=cv2.INTER_CUBIC)[..., None]
        else:
            face = np.array(detected_face.image)  # cv2 fails if this comes as imageio.core.Array
            imask = mask.get_full_frame_mask(face.shape[1], face.shape[0])
            imask = np.expand_dims(imask, -1)

        height, width = face.shape[:2]
        if self._output["type"] == "combined":
            masked = (face.astype("float32") * imask.astype("float32") / 255.).astype("uint8")
            imask = np.tile(imask, 3)
            for img in (face, masked, imask):
                cv2.rectangle(img, (0, 0), (width - 1, height - 1), (255, 255, 255), 1)
                out_image = np.concatenate((face, masked, imask), axis=1)
        elif self._output["type"] == "mask":
            out_image = imask
        elif self._output["type"] == "masked":
            out_image = np.concatenate([face, imask], axis=-1)
        return out_image
