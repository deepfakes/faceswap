#!/usr/bin python3
""" Main entry point to the extract process of FaceSwap """
from __future__ import annotations

import logging
import os
import sys
import typing as T
from time import sleep

from dataclasses import asdict, dataclass

import cv2
import numpy as np
import torch
from tqdm import tqdm

from lib.align.aligned_utils import (batch_adjust_matrices, batch_align, batch_resize,
                                     batch_transform, get_adjusted_center, get_centered_size)
from lib.align.alignments import AlignmentsFace, PNGHeader, PNGSource
from lib.align.constants import EXTRACT_RATIOS, LandmarkType, MEAN_FACE
from lib.align.detected_face import DetectedFace
from lib.align.pose import get_camera_matrix, get_xyz_2d, Batch3D
from lib.infer import Detect, Align, Identity, Mask, File, Profiler
from lib.infer.identity import FilterLoader
from lib.infer.objects import FrameFaces, frame_faces_to_alignment
from lib.image import encode_image, ImagesLoader, ImagesSaver
from lib.logger import parse_class_init
from lib.multithreading import FSThread
from lib.utils import (get_folder, get_module_objects, handle_deprecated_cli_opts,
                       IMAGE_EXTENSIONS, VIDEO_EXTENSIONS)

from .fs_media import Alignments, finalize

if T.TYPE_CHECKING:
    import numpy.typing as npt
    from argparse import Namespace
    from lib.align.alignments import AlignmentDict, AlignmentFileDict, PNGAlignments
    from lib.infer.runner import ExtractRunner
    from lib.multithreading import ErrorState

logger = logging.getLogger(__name__)


@dataclass
class BatchInfo:
    """ Holds information about each input batch being processed through extract

    Parameters
    ----------
    loader
        The images loader for the batch
    alignments
        The alignments for the input
    """
    loader: Loader
    """The images loader for the batch"""
    alignments: Alignments
    """The alignments for the input"""


class Extract:
    """ The Faceswap Face Extraction Process.

    The extraction process is responsible for detecting faces in a series of images/video, aligning
    them and optionally collecting further data about each face leveraging various user selected
    plugins

    Parameters
    ----------
    arguments
        The arguments to be passed to the extraction process as generated from Faceswap's command
        line arguments
    """
    def __init__(self, arguments: Namespace) -> None:
        logger.debug(parse_class_init(locals()))
        args = handle_deprecated_cli_opts(arguments,
                                          additional={"K": ("to skip saving faces", True, None)})
        args = self._validate_compatible_args(args)
        input_locations = self._get_input_locations(args.input_dir, args.batch_mode)
        self._validate_batch_mode(args.batch_mode, input_locations, args)
        self._configure_torch(args.compile)
        self._face_filter = FilterLoader(args.ref_threshold, args.filter, args.nfilter)
        self._pipeline = self._load_pipeline(args)

        file_input = args.detector == "file" or args.aligner == "file"
        save_alignments = self._should_save_alignments(args)
        self._batches = [BatchInfo(ld := Loader(self._pipeline,
                                                input_location,
                                                file_input,
                                                args.extract_every_n,
                                                args.skip_existing,
                                                args.skip_faces,
                                                idx == len(input_locations) - 1),
                                   Alignments(args.alignments_path,
                                              ld.location,
                                              is_extract=True,
                                              skip_existing_frames=args.skip_existing,
                                              skip_existing_faces=arguments.skip_faces,
                                              plugin_is_file=file_input,
                                              save_alignments=save_alignments,
                                              input_is_video=ld.is_video))
                         for idx, input_location in enumerate(input_locations)]

        self._output = Output(self._pipeline,
                              args.output_dir,
                              args.size,
                              args.min_scale,
                              self._batches,
                              args.save_interval,
                              args.debug_landmarks)

    @classmethod
    def _get_input_locations(cls, input_location: str, batch_mode: bool) -> list[str]:
        """ Obtain the full path to input locations. Will be a list of locations if batch mode is
        selected, or a list containing a single location if batch mode is not selected.

        Parameters
        ----------
        input_location
            The full path to the input location. Either a video file, a folder of images or a
            folder containing either/or videos and sub-folders of images (if batch mode is
            selected)
        batch_mode
            ``True`` if extract is running in batch mode

        Returns
        -------
        The list of input location paths
        """
        if not batch_mode:
            return [input_location]

        if os.path.isfile(input_location):
            logger.warning("Batch mode selected but input is not a folder. Switching to normal "
                           "mode")
            return [input_location]

        retval = [os.path.join(input_location, fname)
                  for fname in os.listdir(input_location)
                  if (os.path.isdir(os.path.join(input_location, fname))  # folder images
                      and any(os.path.splitext(iname)[-1].lower() in IMAGE_EXTENSIONS
                              for iname in os.listdir(os.path.join(input_location, fname))))
                  or os.path.splitext(fname)[-1].lower() in VIDEO_EXTENSIONS]  # video

        retval = list(sorted(retval))
        logger.debug("[Extract] Input locations: %s", retval)
        return retval

    @classmethod
    def _validate_compatible_args(cls, args: Namespace) -> Namespace:
        """Some cli arguments are not compatible with each other. If conflicting arguments have
        been selected, log a warning and make necessary changes

        Parameters
        ----------
        args
            The command line arguments to be checked and updated for conflicts

        Returns
        -------
        The updated command line arguments
        """
        # Can't run a detector if importing landmarks
        if args.aligner == "file" and args.detector != "file":
            logger.warning("Detecting faces is not compatible with importing landmarks from a "
                           "file. Setting Detector to 'file'")
            args.detector = "file"
        # Impossible to skip existing when not running detection
        if args.skip_existing and args.detector == "file":
            logger.warning("Skipping existing frames is not compatible with importing from a file "
                           "for detection. Disabling 'skip_existing'")
            args.skip_existing = False
        # Impossible to get missing faces when we do not have a detector or aligner
        if args.skip_faces and (args.detector == "file" or args.aligner == "file"):
            logger.warning("Skipping existing faces is not compatible with importing from a file. "
                           "Disabling 'skip_existing_faces'")
            args.skip_faces = False
        # Face filtering needs a recognition plugin
        if (args.filter or args.nfilter) and not args.identity:
            logger.warning("Face-filtering is enabled, but an identity plugin has not been "
                           "selected. Selecting 'T-Face' plugin")
            args.identity = ["t-face"]
        # We can only use 1 identity for face filtering, so we select the first given
        if (args.filter or args.nfilter) and len(args.identity) > 1:
            logger.warning("Face-filtering is enabled, but multiple identity plugins have been "
                           "selected. Using '%s' for filtering", args.identity[0])
        return args

    def _validate_batch_mode(self, batch_mode: bool,
                             input_locations: list[str],
                             args: Namespace) -> None:
        """ Validate the command line arguments.

        If batch-mode selected and there is only one object to extract from, then batch mode is
        disabled

        If processing in batch mode, some of the given arguments may not make sense, in which case
        a warning is shown and those options are reset.

        Parameters
        ----------
        batch_mode
            ``True`` if extract is running in batch mode
        input_locations
            The discovered input locations within the input folder
        args
            The passed in command line arguments that may require amending
        """
        if not batch_mode:
            return

        if not input_locations:
            logger.error("Batch mode selected, but no valid files found in input location: '%s'. "
                         "Exiting.", args.input_dir)
            sys.exit(1)

        if args.alignments_path:
            logger.warning("Custom alignments path not supported for batch mode. "
                           "Reverting to default.")
            args.alignments_path = None

    @classmethod
    def _configure_torch(cls, compile_models: bool) -> None:
        """Set various Torch switches for inference optimization

        Parameters
        ----------
        compile_models
            ``True`` if model compilation has been requested
        """
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if not compile_models:
            return
        # pylint:disable=protected-access
        torch._dynamo.config.cache_size_limit = 512

    def _should_save_alignments(self, arguments: Namespace) -> bool:
        """ Decide whether alignments should be saved from the given command line arguments and
        output suitable information

        Parameters
        ---------
        arguments
            The arguments generated from Faceswap's command line arguments

        Returns
        -------
        ``True`` if alignments should be saved
        """
        if arguments.detector == arguments.aligner == "file" and (
                arguments.masker is None and arguments.identity is None):
            logger.debug("[Extract] Extracting directly from file. Not saving alignments")
            return False
        if arguments.detector == arguments.aligner == "file" and arguments.extract_every_n > 1:
            logger.warning("Alignments loaded from file, EEN > 1 and additional plugins selected.")
            logger.warning("The extracted faces will contain the additional plugin data, but an "
                           "updated Alignments File will not be saved.")
            return False
        if arguments.detector == arguments.aligner == "file":
            logger.info("Alignments file will be updated with data from additional plugins")
        return True

    def _load_pipeline(self, arguments: Namespace) -> ExtractRunner:  # noqa[C901]
        """ Create the extraction pipeline and run profiling, if selected

        Parameters
        ---------
        arguments
            The arguments generated from Faceswap's command line arguments

        Returns
        -------
        The final runner, with input interfaces, from the pipeline
        """
        retval = None
        conf_file = arguments.config_file
        profile = arguments.benchmark
        try:
            if arguments.detector != "file":
                retval = Detect(arguments.detector,
                                rotation=arguments.rotate_images,
                                min_size=arguments.min_size,
                                max_size=arguments.max_size,
                                compile_model=arguments.compile,
                                config_file=conf_file)(retval, profile=profile)
            if arguments.aligner != "file":
                retval = Align(arguments.aligner,
                               re_feeds=arguments.re_feed,
                               re_align=arguments.re_align,
                               normalization=arguments.normalization,
                               filters=arguments.align_filters,
                               compile_model=arguments.compile,
                               config_file=conf_file)(retval, profile=profile)
            if arguments.masker is not None:
                for masker in arguments.masker:
                    retval = Mask(masker,
                                  compile_model=arguments.compile,
                                  config_file=conf_file)(retval, profile=profile)
            if arguments.identity:
                for idx, identity in enumerate(arguments.identity):
                    retval = Identity(identity,
                                      self._face_filter.threshold,
                                      compile_model=arguments.compile,
                                      config_file=conf_file)(retval, profile=profile)
                    if self._face_filter.enabled and idx == 0:
                        # Add the first selected identity plugin
                        self._face_filter.add_identity_plugin(retval)

            if retval is not None and profile:
                Profiler(retval)()

            retval = File()() if retval is None else retval

        except Exception:
            logger.debug("[Extract] Error during pipeline initialization")
            if retval is not None:
                retval.stop()
            raise
        logger.debug("[Extract] Pipeline output: %s", retval)
        return retval

    def process(self) -> None:
        """ Run the extraction process """
        try:
            if self._face_filter.enabled:
                self._face_filter.get_embeddings(self._pipeline)
            self._output.start()
            for batch in self._batches:
                batch.loader.start(batch.alignments.data)
                batch.loader.join()
                if batch.loader.error_state.has_error:
                    batch.loader.error_state.re_raise()
            self._output.join()
        except Exception:
            self._output.join()
            self._pipeline.stop()
            raise


class Loader:  # pylint:disable=too-many-instance-attributes
    """ Loads images/video frames from disks and puts to queue for feeding the extraction pipeline

    Parameters
    ----------
    pipeline
        The final plugin in the extraction pipeline
    input_path
        Full path to a folder of images or a video file
    input_is_file
        ``True`` if the input plugin to the pipeline is an alignments file (fsa or json) so
        detected faces should be loaded from the file and passed into the pipeline
    extract_every
        The number of frames to extract from the source. 1 will extract every frame, 5 every 5th
        frame etc
    skip_existing_frames
        ``True`` if existing extracted frames should be skipped
    skip_existing_faces
        ``True`` if frames with existing face detections should be skipped
    is_final
        ``True`` if this loader is for the final batch being processed
    """
    def __init__(self,
                 pipeline: ExtractRunner,
                 input_path: str,
                 input_is_file: bool,
                 extract_every: int,
                 skip_existing_frames: bool,
                 skip_existing_faces: bool,
                 is_final: bool) -> None:
        logger.debug(parse_class_init(locals()))
        self.location = input_path
        """Full path to the input location for the loader"""
        self.existing_count = 0
        """The number of frames that pre-exist within the alignments file that will be skipped
        because skip_existing/skip_existing_faces has been selected"""

        self._input_is_file = input_is_file
        self._pipeline = pipeline
        self._is_final = is_final
        self._extract_every = extract_every
        self._skip_frames = skip_existing_frames
        self._skip_faces = skip_existing_faces

        self._images = ImagesLoader(input_path)
        self._thread = FSThread(self._load, name="ExtractLoader")
        self._alignments: dict[str, AlignmentDict] = {}
        self._missing_count = 0
        self._seen: set[str] = set()
        self._ready = False

    @property
    def count(self) -> int:
        """The number of frames to be processed"""
        # Wait until skip list has been processed before allowing another thread to call the count
        while True:
            if self._ready:
                break
            sleep(0.25)
            continue
        return self._images.process_count

    @property
    def is_video(self) -> bool:
        """``True`` if the input location is a video file, ``False`` for folder of images"""
        return self._images.is_video

    @property
    def error_state(self) -> ErrorState:
        """The global FSThread error state object"""
        return self._thread.error_state

    def _set_skip_list(self) -> None:
        """ Add the skip list to the image loader

        Checks against `extract_every_n` and the existence of alignments data (can exist if
        `skip_existing` or `skip_existing_faces` has been provided) and compiles a list of frame
        indices that should not be processed, providing these to :class:`lib.image.ImagesLoader`.
        """
        existing = list(self._alignments)
        if self._extract_every == 1 and not existing:
            logger.debug("[Extract.Loader] No frames to be skipped")
            self._ready = True
            return

        skip_een = set(i for i in range(self._images.count) if i % self._extract_every != 0)

        file_names = ([os.path.basename(f) for f in self._images.file_list]
                      if self._skip_frames or self._skip_faces else [])
        skip_frames = set(i for i, f in enumerate(file_names)
                          if f in existing) if self._skip_frames else set()
        skip_faces = (
            set(i for i, f in enumerate(file_names)
                if self._alignments.get(f, {}).get("faces"))  # type:ignore[call-overload]
            if self._skip_faces else set()
        )
        skip_exist = skip_frames.union(skip_faces)

        if self._extract_every > 1:
            logger.info("Skipping %s frames of %s for extract every %s",
                        len(skip_een), self._images.count, self._extract_every)
        if skip_exist:
            self.existing_count = len(skip_exist.difference(skip_een))
            logger.info("Skipping %s frames of %s for skip existing frames/faces",
                        self.existing_count, self._images.count - len(skip_een))

        skip = list(skip_exist.union(skip_een))
        logger.debug("[Extract.Loader] Total skip count: %s", len(skip))
        self._images.add_skip_list(skip)
        self._ready = True

    def _get_detected_faces(self, file_path: str) -> list[DetectedFace] | None:
        """When importing data, obtain the existing detected face objects for passing through the
        pipeline

        Parameters
        ----------
        file_path
            The full path to the image being loaded

        Returns
        -------
        list[DetectedFace] | None
            The imported detected face objects or ``None`` if data is not being imported
        """
        if not self._input_is_file:
            return None
        fname = os.path.basename(file_path)
        self._seen.add(fname)
        if fname not in self._alignments:
            self._missing_count += 1
            logger.verbose(  # type:ignore[attr-defined]
                "Adding frame with no detections as does not exist in import file: '%s'", fname)
            return []
        retval = [DetectedFace().from_alignment(a)
                  for a in self._alignments[fname].get("faces", [])]
        logger.trace(  # type:ignore[attr-defined]
            "[Extract.Loader] importing %s faces for file '%s'", len(retval), fname)
        return retval

    def _finalize(self) -> None:
        """Actions to run when the loader is exhausted"""
        if self._is_final:
            self._pipeline.stop()
        if self._missing_count > 0:
            logger.warning("%s images did not exist in the import file. Run in verbose mode to "
                           "see which files have been added with no detected faces.",
                           self._missing_count)
        processed_files = set(self._images.processed_file_list)
        if self._input_is_file and len(self._seen) != len(processed_files):
            logger.warning("%s images exist in the import file but do not exist on disk. Run in "
                           "verbose mode to see which files are missing.",
                           len(processed_files) - len(self._seen))
            logger.verbose(  # type:ignore[attr-defined]
                "Files in import file that do not exist on disk: %s",
                list(sorted(processed_files.difference(self._seen))))

    def _load(self) -> None:
        """ Load images from disk and pass to a queue for the extraction pipeline """
        logger.debug("[Extract.Loader] start")
        for filename, image in self._images.load():
            faces = self._get_detected_faces(filename)
            self._pipeline.put(filename, image, source=self.location, detected_faces=faces)
        if self.error_state.has_error:
            logger.debug("[Extract.Loader] Thread error OUT detected in worker thread")
            return
        self._finalize()
        logger.debug("[Extract.Loader] end")

    def start(self, alignments: dict[str, AlignmentDict]) -> None:
        """ Set the skip list and start loading images from disk

        Parameters
        ----------
        alignments
            Dictionary of existing alignments data for use when importing or skipping existing data
        """
        self._alignments = alignments
        self._set_skip_list()
        logger.debug("[Extract.Loader] start thread")
        if isinstance(self._pipeline.handler, File):
            self._pipeline.register_external_error_state(self._thread.error_state)
        self._thread.start()

    def join(self) -> None:
        """ Join the image loading thread and monitor for keyboard interrupts"""
        logger.debug("[Extract.Loader] join thread")
        while self._thread.is_alive():
            try:
                self._thread.join(timeout=0.2)
            except KeyboardInterrupt:
                logger.debug("Terminate signal received. Stopping...")
                raise
        logger.debug("[Extract.Loader] joined thread")


class DebugLandmarks():
    """Draw debug landmarks on face output.

    Parameters
    ----------
    size
        The size of the extracted face image
    """
    def __init__(self, size: int) -> None:
        logger.debug(parse_class_init(locals()))
        self._size = size
        self._face_size = get_centered_size("head", "face", size)
        self._legacy_size = get_centered_size("head", "legacy", size)
        self._camera_matrix = get_camera_matrix()
        self._mean_face = MEAN_FACE[LandmarkType.LM_2D_51][None]
        self._face_expansion = 1.0 - EXTRACT_RATIOS["face"]
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._font_scale = size / 512
        self._font_pad = size // 64

    def _border_text(self,
                     image: np.ndarray,
                     text: str,
                     color: tuple[int, int, int],
                     position: tuple[int, int]) -> None:
        """Create text on an image with a black border

        Parameters
        ----------
        image
            The image to put bordered text on to
        text
            The text to place the image
        color
            The color of the text
        position
            The (x, y) co-ordinates to place the text
        """
        thickness = 2
        for idx in range(2):
            text_color = (0, 0, 0) if idx == 0 else color
            cv2.putText(image,
                        text,
                        position,
                        self._font,
                        self._font_scale,
                        text_color,
                        thickness,
                        lineType=cv2.LINE_AA)
            thickness //= 2

    def _annotate_face_box(self,
                           face: npt.NDArray[np.uint8],
                           offset_head: npt.NDArray[np.float32],
                           offset_face: npt.NDArray[np.float32],
                           face_size) -> None:
        """Annotate the face extract box and print the original size in pixels

        Parameters
        ----------
        face
            The face image to annotate
        offset_head
            The (X, Y) offset for the head centered extract
        offset_face
            The (X, Y) offset for the face centered extract
        face_size
            The size of the face box in the original frame
        """
        color = (0, 255, 0)
        center = get_adjusted_center(self._size, offset_head, offset_face, "head", 0)
        padding = self._face_size // 2
        roi = np.array([center - padding, center + padding]).tolist()
        cv2.rectangle(face, roi[0], roi[1], color, 1)
        # Size in top right corner
        text_img = face.copy()
        text = f"{face_size}px"
        text_size = cv2.getTextSize(text, self._font, self._font_scale, 1)[0]
        pos_x = roi[1][0] - (text_size[0] + self._font_pad)
        pos_y = roi[0][1] + text_size[1] + self._font_pad
        self._border_text(text_img, text, color, (pos_x, pos_y))
        cv2.addWeighted(text_img, 0.75, face, 0.25, 0, face)

    def _print_stats(self,
                     face: npt.NDArray[np.uint8],
                     distance: float,
                     pitch: float,
                     roll: float,
                     yaw: float) -> None:
        """Print various metrics on the output face images

        Parameters
        ----------
        face
            The face image to annotate
        distance
            The distance of the face from a 'mean' face
        pitch
            The pitch of the face in degrees
        roll
            The roll of the face in degrees
        yaw
            The yaw of the face in degrees
        """
        text_image = face.copy()
        texts = [f"pitch: {pitch:.2f}",
                 f"yaw: {yaw:.2f}",
                 f"roll: {roll: .2f}",
                 f"distance: {distance:.2f}"]
        colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 255, 255)]
        text_sizes = [cv2.getTextSize(text, self._font, self._font_scale, 1)[0] for text in texts]
        final_y = self._size - text_sizes[-1][1]
        pos_y = [(size[1] + self._font_pad) * (idx + 1)
                 for idx, size in enumerate(text_sizes)][:-1] + [final_y]
        pos_x = self._font_pad
        for idx, text in enumerate(texts):
            self._border_text(text_image, text, colors[idx], (pos_x, pos_y[idx]))
        # Apply text to face
        cv2.addWeighted(text_image, 0.75, face, 0.25, 0, face)

    def __call__(self,  # pylint:disable=too-many-locals
                 faces: npt.NDArray[np.uint8],
                 matrices: npt.NDArray[np.float32],
                 media: FrameFaces) -> None:
        """Draw debug annotations on extracted face images

        Parameters
        ----------
        faces
            The aligned face images that are to be saved to disk
        matrices
            The adjustment matrices for transforming from frame space to face space
        media
            The corresponding FrameFaces media object for the faces
        """
        if not np.any(faces):
            return

        landmarks = batch_transform(
            matrices, T.cast("npt.NDArray[np.float32]", media.landmarks)).astype("int32")
        aligned = media.aligned
        norm_mats = aligned.matrices[:, :2, 0]
        sizes = np.rint(1.0 / (self._face_expansion *
                               np.hypot(norm_mats[:, 0], norm_mats[:, 1]))).astype(np.int32)
        dists = np.abs(aligned.landmarks_normalized[:, 17:] -
                       self._mean_face).mean(axis=(1, 2))
        pry = (Batch3D.pitch(aligned.rotation),
               Batch3D.roll(aligned.rotation),
               Batch3D.yaw(aligned.rotation))
        for idx, (face, lms) in enumerate(zip(faces, landmarks)):
            # Landmarks
            for (pos_x, pos_y) in lms:
                cv2.circle(face, (pos_x, pos_y), 1, (0, 255, 255), -1)
            # Pose
            center = (self._size // 2, self._size // 2)
            xyz = get_xyz_2d(aligned.rotation[idx],
                             aligned.translation[idx],
                             self._camera_matrix) - aligned.offsets_head[idx]
            points = (xyz * self._size).astype("int32")
            cv2.line(face, center, tuple(points[1]), (0, 255, 0), 1)
            cv2.line(face, center, tuple(points[0]), (255, 0, 0), 1)
            cv2.line(face, center, tuple(points[2]), (0, 0, 255), 1)
            # Face centering
            self._annotate_face_box(face,
                                    aligned.offsets_head[idx],
                                    aligned.offsets_face[idx],
                                    int(sizes[idx]))
            # Legacy centering
            center_a = get_adjusted_center(self._size,
                                           aligned.offsets_head[idx],
                                           aligned.offsets_legacy[idx],
                                           "head",
                                           0)
            padding = self._legacy_size // 2
            roi = np.array([center_a - padding, center_a + padding]).tolist()
            cv2.rectangle(face, roi[0], roi[1], (0, 0, 255), 1)
            # Pitch/roll/yaw/distance
            self._print_stats(face,
                              float(dists[idx]),
                              float(pry[0][idx]),
                              float(pry[1][idx]),
                              float(pry[2][idx]))


class Output:  # pylint:disable=too-many-instance-attributes
    """ Handles output processing and saving of extracted faces

    Parameters
    ----------
    pipeline
        The output runner from the extraction pipeline
    output_folder
        The full path to the output folder to save extracted faces. ``None`` to not save faces
    size
        The size to save extracted faces at
    min_scale
        The minimum percentage of the output size that should be accepted for outputting a face
        to disk
    batches
        The information about each batch that is to be processed
    save_interval
        How often to save the alignments file
    debug_landmarks
        ``True`` to annotate the output images with debug data
    """
    def __init__(self,
                 pipeline: ExtractRunner,
                 output_folder: str | None,
                 size: int,
                 min_scale: int,
                 batches: list[BatchInfo],
                 save_interval: int,
                 debug_landmarks: bool) -> None:
        logger.debug(parse_class_init(locals()))
        self._pipeline = pipeline
        self._size = size
        self._batches = batches
        self._save_interval = save_interval
        self._min_size = self._get_min_size(size, min_scale)
        self._saver: None | ImagesSaver = None
        self._outputs = self._get_outputs(output_folder)
        self._thread = FSThread(self._process, name="ExtractOutput")
        self._debug = DebugLandmarks(size) if debug_landmarks else None
        self._counts = {"verify": False, "faces": 0, "scale_skip": 0}
        self._align = {"padding": round((size * EXTRACT_RATIOS["head"]) / 2),
                       "padding_thumbnail": round((96 * EXTRACT_RATIOS["head"]) / 2),
                       "empty_faces": np.empty((0, size, size, 3), dtype=np.uint8)}

    @classmethod
    def _get_min_size(cls, extract_size: int, min_scale: int) -> int:
        """ Obtain the minimum size that a face has been resized from to be included as a valid
        extract.

        Parameters
        ----------
        extract_size
            The requested size of the extracted images
        min_scale
            The percentage amount that has been supplied for valid faces (as a percentage of
            extract size)

        Returns
        -------
        The minimum size, in pixels, that a face is resized from to be considered valid
        """
        retval = 0 if min_scale == 0 else max(4, int(extract_size * (min_scale / 100.)))
        logger.debug("[Extract.Output] Extract size: %s, min percentage size: %s, min_size: %s",
                     extract_size, min_scale, retval)
        return retval

    def _get_outputs(self, output_folder: str | None) -> list[str | None]:
        """ Obtain the locations to save the output for each batch input location

        Parameters
        ----------
        output_folder
            The full path to the output folder to save extracted faces. ``None`` to not save faces

        Returns
        -------
        The output locations for each input batch. ``None`` if faces are not to be saved
        """
        num_batches = len(self._batches)
        retval: list[str | None]
        if not output_folder:
            logger.debug("[Extract.Output] No save location selected")
            return [None for _ in range(num_batches)]
        out_folder = get_folder(output_folder)
        if num_batches == 1:
            logger.debug("[Extract.Output] Single save location: '%s'", out_folder)
            return [out_folder]
        retval = [os.path.join(out_folder,
                               os.path.splitext(os.path.basename(b.loader.location))[0])
                  for b in self._batches]
        logger.debug("[Extract.Output] Save locations: %s", retval)
        return retval

    def _should_output(self, matrices: npt.NDArray[np.float32]) -> npt.NDArray[np.bool_]:
        """Test which of the faces should be saved based on the given minimum scale option

        Parameters
        ----------
        matrices
            The normalized aligned matrices to check for original face size

        Returns
        -------
        Mask array containing ``True`` for each face that should be output.
        """
        if self._min_size <= 0:
            return np.fromiter((True for _ in range(len(matrices))), dtype=bool)
        linear = matrices[:, :2, 0]
        sizes = 1.0 / ((1.0 - EXTRACT_RATIOS["face"]) * np.hypot(linear[:, 0], linear[:, 1]))
        return sizes >= self._min_size

    def _save_faces(self,  # pylint:disable=too-many-locals
                    faces: npt.NDArray[np.uint8],
                    matrices: npt.NDArray[np.float32],
                    basename: str,
                    meta: list[PNGAlignments],
                    frame_size: tuple[int, int],
                    alignments_version: float,
                    is_video: bool) -> None:
        """Encode the aligned faces with PNG Header information and save to disk

        Parameters
        ----------
        faces
            The correctly sized and aligned faces to save for a frame
        matrices
            The normalized aligned affine matrices for the faces
        basename
            The base filename (without full path) of the original frame
        meta
            The meta data to add to each of the PNG headers
        frame_size
            The (height, width) of the original frame
        alignments_version
            The current alignments file version
        is_video
            ``True`` if the input is a video otherwise ``False``
        """
        if self._saver is None:
            return
        split_name = os.path.splitext(basename)[0]
        for idx, (face, data, save) in enumerate(zip(faces, meta, self._should_output(matrices))):
            if not save:
                self._counts["scale_skip"] += 1
                continue
            img_name = f"{split_name}_{idx}.png"
            header = PNGHeader(alignments=data,
                               source=PNGSource(alignments_version=alignments_version,
                                                original_filename=img_name,
                                                face_index=idx,
                                                source_filename=basename,
                                                source_is_video=is_video,
                                                source_frame_dims=frame_size))
            img = encode_image(face, ".png", metadata=asdict(header))
            self._saver.save(img_name, img)

    def _get_faces_and_thumbs(self, media: FrameFaces
                              ) -> tuple[npt.NDArray[np.uint8], list[npt.NDArray[np.uint8]]]:
        """Obtain the aligned faces and jpeg thumbnails from the given media object

        Parameters
        ----------
        media
            The FrameFaces object output from the extraction pipeline

        Returns
        -------
        faces
            The (N, size, size, 3) aligned face images from the media object
        thumbnails
            The (N, 96, 96, 3) jpeg thumbnails for the media object
        """
        if not media:
            return (T.cast("npt.NDArray[np.uint8]", self._align["empty_faces"]), [])
        image_ids = np.fromiter((0 for _ in range(len(media))), dtype=np.int32)
        if self._saver is None:
            faces = np.empty((0, self._size, self._size, 3), dtype=np.uint8)
            mats = batch_adjust_matrices(media.aligned.matrices_head,
                                         96,
                                         T.cast(int, self._align["padding_thumbnail"]))
            thumbs = batch_align([media.image], image_ids, mats, 96)
        else:
            mats = batch_adjust_matrices(media.aligned.matrices_head,
                                         self._size,
                                         T.cast(int, self._align["padding"]))
            faces = batch_align([media.image],
                                image_ids,
                                mats,
                                self._size,
                                fast_upscale=False)
            thumbs = batch_resize(faces, 96)
            if self._debug is not None:
                self._debug(faces, mats, media)

        thumbnails = [cv2.imencode(".jpg", t, [cv2.IMWRITE_JPEG_QUALITY, 60])[1]
                      for t in thumbs]
        return faces, thumbnails

    def _process_faces(self, media: FrameFaces, alignments: Alignments, is_video: bool) -> None:
        """ Process the detected face objects into aligned faces, generate the thumbnails and run
        any post process actions

        Parameters
        ----------
        media
            The FrameFaces object output from the extraction pipeline
        alignments
            The alignments object that is to contain these faces
        is_video
            ``True`` if the input is a video otherwise ``False``
        """
        basename = os.path.basename(media.filename)
        faces, thumbnails = self._get_faces_and_thumbs(media)
        media.remove_image()  # Spare the RAM
        meta = frame_faces_to_alignment(media)
        self._save_faces(faces,
                         media.aligned.matrices,
                         basename,
                         meta,
                         media.image_size,
                         alignments.version,
                         is_video)
        alignments_faces = T.cast(list["AlignmentFileDict"],
                                  [asdict(AlignmentsFace(**aln.__dict__, thumb=thumb.tolist()))
                                   for aln, thumb in zip(meta, thumbnails)])
        alignments.data[basename] = {"faces": alignments_faces, "video_meta": {}}
        faces_count = len(media)
        if faces_count == 0:
            logger.verbose("No faces were detected in image: %s", basename)  # type: ignore
        if not self._counts["verify"] and faces_count > 1:
            self._counts["verify"] = True
        self._counts["faces"] += faces_count

    def _set_saver(self, output: str | None) -> None:
        """Close the currently active saver and set the next :attr:`_saver` for the given output

        Parameters
        ----------
        output
            The full path to the next output location
        """
        if self._saver is not None:
            self._saver.close()
        if output is None:
            self._saver = None
        else:
            self._saver = ImagesSaver(get_folder(output), as_bytes=True)
        logger.debug("[Extract.Output] Set image saver to location: %s",
                     repr(self._saver if self._saver is None else self._saver.location))

    def _finalize_batch(self, batch: BatchInfo, batch_index: int) -> None:
        """ Actions to perform when an input batch has finished processing.

        Parameters
        ----------
        batch
            The information about the batch that has finished processing
        batch_index
            The index of the batch in :attr:`_self._batches`
        """
        logger.debug("[Extract.Output] Finalizing batch: %s", batch)
        if batch.alignments.save_alignments:
            if not self._save_interval:
                batch.alignments.backup()
            batch.alignments.save()
        count = batch.loader.count - batch.loader.existing_count
        if self._counts["scale_skip"] > 0:
            logger.info("%s faces not output as they are below the minimum size of %spx. These "
                        "still exist in the alignments file.",
                        self._counts["scale_skip"], self._min_size)
        finalize(count, T.cast(int, self._counts["faces"]), T.cast(bool, self._counts["verify"]))
        self._counts["verify"] = False
        output = None if batch_index == len(self._outputs) - 1 else self._outputs[batch_index + 1]
        self._set_saver(output)
        self._counts["faces"] = 0
        self._counts["scale_skip"] = 0
        del batch.alignments

    def _process(self) -> None:  # noqa[C901]
        """ Process the output from the extraction pipeline within a thread """
        logger.debug("[Extract.Output] start")
        total_batches = len(self._batches)
        self._set_saver(self._outputs[0])
        if self._saver is not None and self._min_size > 0:
            logger.info("Only outputting faces that have been resized from a minimum resolution "
                        "of %spx", self._min_size)

        for batch_idx, batch in enumerate(self._batches):
            msg = f" job {batch_idx + 1} of {total_batches}" if total_batches > 1 else ""
            logger.info("Processing%s: '%s'", msg, batch.loader.location)
            if self._saver is not None:
                logger.info("Faces output: '%s'", self._saver.location)
            has_started = False
            save_interval = 0 if not batch.alignments.save_alignments else self._save_interval
            with tqdm(desc="Extracting faces",
                      total=batch.loader.count,
                      leave=True,
                      smoothing=0) as prog_bar:
                if batch_idx > 0:  # Update for batch picked up at end of previous batch
                    prog_bar.update(1)

                for idx, media in enumerate(self._pipeline):
                    if not has_started:
                        prog_bar.reset()  # Delay before first output, reset timer for better it/s
                        has_started = True

                    if media.source != batch.loader.location:
                        self._finalize_batch(batch, batch_idx)
                        next_batch = self._batches[batch_idx + 1]
                        self._process_faces(media, next_batch.alignments,
                                            next_batch.loader.is_video)
                        break

                    self._process_faces(media, batch.alignments, batch.loader.is_video)
                    if save_interval and (idx + 1) % save_interval == 0:
                        batch.alignments.save()
                    if prog_bar.n + 1 > prog_bar.total:
                        # Don't switch to unknown mode when frame count is under
                        prog_bar.total += 1
                    prog_bar.update(1)

        if self._thread.error_state.has_error:
            logger.debug("[Extract.Output] Thread error detected in worker thread")
            return
        self._finalize_batch(self._batches[-1], len(self._batches) - 1)
        logger.debug("[Extract.Output] end")

    def start(self) -> None:
        """ Start the output thread """
        logger.debug("[Extract.Output] start thread")
        self._thread.start()

    def join(self) -> None:
        """ Join the output thread """
        logger.debug("[Extract.Output] join thread")
        self._thread.join()
        logger.debug("[Extract.Output] joined thread")


__all__ = get_module_objects(__name__)
