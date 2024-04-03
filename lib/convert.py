#!/usr/bin/env python3
""" Converter for Faceswap """
from __future__ import annotations
import logging
import typing as T
from dataclasses import dataclass

import cv2
import numpy as np

from plugins.plugin_loader import PluginLoader

if T.TYPE_CHECKING:
    from argparse import Namespace
    from collections.abc import Callable
    from lib.align.aligned_face import AlignedFace, CenteringType
    from lib.align.detected_face import DetectedFace
    from lib.config import FaceswapConfig
    from lib.queue_manager import EventQueue
    from scripts.convert import ConvertItem
    from plugins.convert.color._base import Adjustment as ColorAdjust
    from plugins.convert.color.seamless_clone import Color as SeamlessAdjust
    from plugins.convert.mask.mask_blend import Mask as MaskAdjust
    from plugins.convert.scaling._base import Adjustment as ScalingAdjust

logger = logging.getLogger(__name__)


@dataclass
class Adjustments:
    """ Dataclass to hold the optional processing plugins

    Parameters
    ----------
    color: :class:`~plugins.color._base.Adjustment`, Optional
        The selected color processing plugin. Default: `None`
    mask: :class:`~plugins.mask_blend.Mask`, Optional
        The selected mask processing plugin. Default: `None`
    seamless: :class:`~plugins.color.seamless_clone.Color`, Optional
        The selected mask processing plugin. Default: `None`
    sharpening: :class:`~plugins.scaling._base.Adjustment`, Optional
        The selected mask processing plugin. Default: `None`
    """
    color: ColorAdjust | None = None
    mask: MaskAdjust | None = None
    seamless: SeamlessAdjust | None = None
    sharpening: ScalingAdjust | None = None


class Converter():
    """ The converter is responsible for swapping the original face(s) in a frame with the output
    of a trained Faceswap model.

    Parameters
    ----------
    output_size: int
        The size of the face, in pixels, that is output from the Faceswap model
    coverage_ratio: float
        The ratio of the training image that was used for training the Faceswap model
    centering: str
        The extracted face centering that the model was trained on (`"face"` or "`legacy`")
    draw_transparent: bool
        Whether the final output should be drawn onto a transparent layer rather than the original
        frame. Only available with certain writer plugins.
    pre_encode: python function
        Some writer plugins support the pre-encoding of images prior to saving out. As patching is
        done in multiple threads, but writing is done in a single thread, it can speed up the
        process to do any pre-encoding as part of the converter process.
    arguments: :class:`argparse.Namespace`
        The arguments that were passed to the convert process as generated from Faceswap's command
        line arguments
    configfile: str, optional
        Optional location of custom configuration ``ini`` file. If ``None`` then use the default
        config location. Default: ``None``
    """
    def __init__(self,
                 output_size: int,
                 coverage_ratio: float,
                 centering: CenteringType,
                 draw_transparent: bool,
                 pre_encode: Callable | None,
                 arguments: Namespace,
                 configfile: str | None = None) -> None:
        logger.debug("Initializing %s: (output_size: %s,  coverage_ratio: %s, centering: %s, "
                     "draw_transparent: %s, pre_encode: %s, arguments: %s, configfile: %s)",
                     self.__class__.__name__, output_size, coverage_ratio, centering,
                     draw_transparent, pre_encode, arguments, configfile)
        self._output_size = output_size
        self._coverage_ratio = coverage_ratio
        self._centering = centering
        self._draw_transparent = draw_transparent
        self._writer_pre_encode = pre_encode
        self._args = arguments
        self._configfile = configfile

        self._scale = arguments.output_scale / 100
        self._face_scale = 1.0 - arguments.face_scale / 100.
        self._adjustments = Adjustments()
        self._full_frame_output: bool = arguments.writer != "patch"

        self._load_plugins()
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def cli_arguments(self) -> Namespace:
        """:class:`argparse.Namespace`: The command line arguments passed to the convert
        process """
        return self._args

    def reinitialize(self, config: FaceswapConfig) -> None:
        """ Reinitialize this :class:`Converter`.

        Called as part of the :mod:`~tools.preview` tool. Resets all adjustments then loads the
        plugins as specified in the given config.

        Parameters
        ----------
        config: :class:`lib.config.FaceswapConfig`
            Pre-loaded :class:`lib.config.FaceswapConfig`. used over any configuration on disk.
        """
        logger.debug("Reinitializing converter")
        self._face_scale = 1.0 - self._args.face_scale / 100.
        self._adjustments = Adjustments()
        self._load_plugins(config=config, disable_logging=True)
        logger.debug("Reinitialized converter")

    def _load_plugins(self,
                      config: FaceswapConfig | None = None,
                      disable_logging: bool = False) -> None:
        """ Load the requested adjustment plugins.

        Loads the :mod:`plugins.converter` plugins that have been requested for this conversion
        session.

        Parameters
        ----------
        config: :class:`lib.config.FaceswapConfig`, optional
            Optional pre-loaded :class:`lib.config.FaceswapConfig`. If passed, then this will be
            used over any configuration on disk. If ``None`` then it is ignored. Default: ``None``
        disable_logging: bool, optional
            Plugin loader outputs logging info every time a plugin is loaded. Set to ``True`` to
            suppress these messages otherwise ``False``. Default: ``False``
        """
        logger.debug("Loading plugins. config: %s", config)
        self._adjustments.mask = PluginLoader.get_converter("mask",
                                                            "mask_blend",
                                                            disable_logging=disable_logging)(
                                                                self._args.mask_type,
                                                                self._output_size,
                                                                self._coverage_ratio,
                                                                configfile=self._configfile,
                                                                config=config)

        if self._args.color_adjustment != "none" and self._args.color_adjustment is not None:
            self._adjustments.color = PluginLoader.get_converter("color",
                                                                 self._args.color_adjustment,
                                                                 disable_logging=disable_logging)(
                                                                    configfile=self._configfile,
                                                                    config=config)

        sharpening = PluginLoader.get_converter("scaling",
                                                "sharpen",
                                                disable_logging=disable_logging)(
                                                    configfile=self._configfile,
                                                    config=config)
        if sharpening.config.get("method") is not None:
            self._adjustments.sharpening = sharpening
        logger.debug("Loaded plugins: %s", self._adjustments)

    def process(self, in_queue: EventQueue, out_queue: EventQueue):
        """ Main convert process.

        Takes items from the in queue, runs the relevant adjustments, patches faces to final frame
        and outputs patched frame to the out queue.

        Parameters
        ----------
        in_queue: :class:`~lib.queue_manager.EventQueue`
            The output from :class:`scripts.convert.Predictor`. Contains detected faces from the
            Faceswap model as well as the frame to be patched.
        out_queue: :class:`~lib.queue_manager.EventQueue`
            The queue to place patched frames into for writing by one of Faceswap's
            :mod:`plugins.convert.writer` plugins.
        """
        logger.debug("Starting convert process. (in_queue: %s, out_queue: %s)",
                     in_queue, out_queue)
        logged = False
        while True:
            inbound: T.Literal["EOF"] | ConvertItem | list[ConvertItem] = in_queue.get()
            if inbound == "EOF":
                logger.debug("EOF Received")
                logger.debug("Patch queue finished")
                # Signal EOF to other processes in pool
                logger.debug("Putting EOF back to in_queue")
                in_queue.put(inbound)
                break

            items = inbound if isinstance(inbound, list) else [inbound]
            for item in items:
                logger.trace("Patch queue got: '%s'",  # type: ignore[attr-defined]
                             item.inbound.filename)
                try:
                    image = self._patch_image(item)
                except Exception as err:  # pylint:disable=broad-except
                    # Log error and output original frame
                    logger.error("Failed to convert image: '%s'. Reason: %s",
                                 item.inbound.filename, str(err))
                    image = item.inbound.image

                    lvl = logger.trace if logged else logger.warning  # type: ignore[attr-defined]
                    lvl("Convert error traceback:", exc_info=True)
                    logged = True
                    # UNCOMMENT THIS CODE BLOCK TO PRINT TRACEBACK ERRORS
                    # import sys; import traceback
                    # exc_info = sys.exc_info(); traceback.print_exception(*exc_info)
                logger.trace("Out queue put: %s",  # type: ignore[attr-defined]
                             item.inbound.filename)
                out_queue.put((item.inbound.filename, image))
        logger.debug("Completed convert process")

    def _get_warp_matrix(self, matrix: np.ndarray, size: int) -> np.ndarray:
        """ Obtain the final scaled warp transformation matrix based on face scaling from the
        original transformation matrix

        Parameters
        ----------
        matrix: :class:`numpy.ndarray`
            The transformation for patching the swapped face back onto the output frame
        size: int
            The size of the face patch, in pixels

        Returns
        -------
        :class:`numpy.ndarray`
            The final transformation matrix with any scaling applied
        """
        if self._face_scale == 1.0:
            mat = matrix
        else:
            mat = matrix * self._face_scale
            patch_center = (size / 2, size / 2)
            mat[..., 2] += (1 - self._face_scale) * np.array(patch_center)

        return mat

    def _patch_image(self, predicted: ConvertItem) -> np.ndarray | list[bytes]:
        """ Patch a swapped face onto a frame.

        Run selected adjustments and swap the faces in a frame.

        Parameters
        ----------
        predicted: :class:`~scripts.convert.ConvertItem`
            The output from :class:`scripts.convert.Predictor`.

        Returns
        -------
        :class: `numpy.ndarray` or pre-encoded image output
            The final frame ready for writing by a :mod:`plugins.convert.writer` plugin.
            Frame is either an array, or the pre-encoded output from the writer's pre-encode
            function (if it has one)

        """
        logger.trace("Patching image: '%s'",  # type: ignore[attr-defined]
                     predicted.inbound.filename)
        frame_size = (predicted.inbound.image.shape[1], predicted.inbound.image.shape[0])
        new_image, background = self._get_new_image(predicted, frame_size)

        if self._full_frame_output:
            patched_face = self._post_warp_adjustments(background, new_image)
            patched_face = self._scale_image(patched_face)
            patched_face *= 255.0
            patched_face = np.rint(patched_face,
                                   out=np.empty(patched_face.shape, dtype="uint8"),
                                   casting='unsafe')
        else:
            patched_face = new_image

        if self._writer_pre_encode is None:
            retval: np.ndarray | list[bytes] = patched_face
        else:
            kwargs: dict[str, T.Any] = {}
            if self.cli_arguments.writer == "patch":
                kwargs["canvas_size"] = (background.shape[1], background.shape[0])
                kwargs["matrices"] = np.array([self._get_warp_matrix(face.adjusted_matrix,
                                                                     patched_face.shape[1])
                                               for face in predicted.reference_faces],
                                              dtype="float32")
            retval = self._writer_pre_encode(patched_face, **kwargs)
        logger.trace("Patched image: '%s'",  # type: ignore[attr-defined]
                     predicted.inbound.filename)
        return retval

    def _warp_to_frame(self,
                       reference: AlignedFace,
                       face: np.ndarray,
                       frame: np.ndarray,
                       multiple_faces: bool) -> None:
        """ Perform affine transformation to place a face patch onto the given frame.

        Affine is done in place on the `frame` array, so this function does not return a value

        Parameters
        ----------
        reference: :class:`lib.align.AlignedFace`
            The object holding the original aligned face
        face: :class:`numpy.ndarray`
            The swapped face patch
        frame: :class:`numpy.ndarray`
            The frame to affine the face onto
        multiple_faces: bool
            Controls the border mode to use. Uses BORDER_CONSTANT if there is only 1 face in
            the image, otherwise uses the inferior BORDER_TRANSPARENT
        """
        # Warp face with the mask
        mat = self._get_warp_matrix(reference.adjusted_matrix, face.shape[0])
        border = cv2.BORDER_TRANSPARENT if multiple_faces else cv2.BORDER_CONSTANT
        cv2.warpAffine(face,
                       mat,
                       (frame.shape[1], frame.shape[0]),
                       frame,
                       flags=cv2.WARP_INVERSE_MAP | reference.interpolators[1],
                       borderMode=border)

    def _get_new_image(self,
                       predicted: ConvertItem,
                       frame_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """ Get the new face from the predictor and apply pre-warp manipulations.

        Applies any requested adjustments to the raw output of the Faceswap model
        before transforming the image into the target frame.

        Parameters
        ----------
        predicted: :class:`~scripts.convert.ConvertItem`
            The output from :class:`scripts.convert.Predictor`.
        frame_size: tuple
            The (`width`, `height`) of the final frame in pixels

        Returns
        -------
        placeholder:  :class: `numpy.ndarray`
            The original frame with the swapped faces patched onto it
        background:  :class: `numpy.ndarray`
            The original frame
        """
        logger.trace("Getting: (filename: '%s', faces: %s)",  # type: ignore[attr-defined]
                     predicted.inbound.filename, len(predicted.swapped_faces))

        placeholder = np.zeros((frame_size[1], frame_size[0], 4), dtype="float32")
        if self._full_frame_output:
            background = predicted.inbound.image / np.array(255.0, dtype="float32")
            placeholder[:, :, :3] = background
        else:
            faces = []  # Collect the faces into final array
            background = placeholder  # Used for obtaining original frame dimensions

        for new_face, detected_face, reference_face in zip(predicted.swapped_faces,
                                                           predicted.inbound.detected_faces,
                                                           predicted.reference_faces):
            predicted_mask = new_face[:, :, -1] if new_face.shape[2] == 4 else None
            new_face = new_face[:, :, :3]
            new_face = self._pre_warp_adjustments(new_face,
                                                  detected_face,
                                                  reference_face,
                                                  predicted_mask)

            if self._full_frame_output:
                self._warp_to_frame(reference_face,
                                    new_face, placeholder,
                                    len(predicted.swapped_faces) > 1)
            else:
                faces.append(new_face)

        if not self._full_frame_output:
            placeholder = np.array(faces, dtype="float32")

        logger.trace("Got filename: '%s'. (placeholders: %s)",  # type: ignore[attr-defined]
                     predicted.inbound.filename, placeholder.shape)

        return placeholder, background

    def _pre_warp_adjustments(self,
                              new_face: np.ndarray,
                              detected_face: DetectedFace,
                              reference_face: AlignedFace,
                              predicted_mask: np.ndarray | None) -> np.ndarray:
        """ Run any requested adjustments that can be performed on the raw output from the Faceswap
        model.

        Any adjustments that can be performed before warping the face into the final frame are
        performed here.

        Parameters
        ----------
        new_face: :class:`numpy.ndarray`
            The swapped face received from the faceswap model.
        detected_face: :class:`~lib.align.DetectedFace`
            The detected_face object as defined in :class:`scripts.convert.Predictor`
        reference_face: :class:`~lib.align.AlignedFace`
            The aligned face object sized to the model output of the original face for reference
        predicted_mask: :class:`numpy.ndarray` or ``None``
            The predicted mask output from the Faceswap model. ``None`` if the model
            did not learn a mask

        Returns
        -------
        :class:`numpy.ndarray`
            The face output from the Faceswap Model with any requested pre-warp adjustments
            performed.
        """
        logger.trace("new_face shape: %s, predicted_mask shape: %s",  # type: ignore[attr-defined]
                     new_face.shape, predicted_mask.shape if predicted_mask is not None else None)
        old_face = T.cast(np.ndarray, reference_face.face)[..., :3] / 255.0
        new_face, raw_mask = self._get_image_mask(new_face,
                                                  detected_face,
                                                  predicted_mask,
                                                  reference_face)
        if self._adjustments.color is not None:
            new_face = self._adjustments.color.run(old_face, new_face, raw_mask)
        if self._adjustments.seamless is not None:
            new_face = self._adjustments.seamless.run(old_face, new_face, raw_mask)
        logger.trace("returning: new_face shape %s", new_face.shape)  # type: ignore[attr-defined]
        return new_face

    def _get_image_mask(self,
                        new_face: np.ndarray,
                        detected_face: DetectedFace,
                        predicted_mask: np.ndarray | None,
                        reference_face: AlignedFace) -> tuple[np.ndarray, np.ndarray]:
        """ Return any selected image mask

        Places the requested mask into the new face's Alpha channel.

        Parameters
        ----------
        new_face: :class:`numpy.ndarray`
            The swapped face received from the faceswap model.
        detected_face: :class:`~lib.DetectedFace`
            The detected_face object as defined in :class:`scripts.convert.Predictor`
        predicted_mask: :class:`numpy.ndarray` or ``None``
            The predicted mask output from the Faceswap model. ``None`` if the model
            did not learn a mask
        reference_face: :class:`~lib.align.AlignedFace`
            The aligned face object sized to the model output of the original face for reference

        Returns
        -------
        :class:`numpy.ndarray`
            The swapped face with the requested mask added to the Alpha channel
        :class:`numpy.ndarray`
            The raw mask with no erosion or blurring applied
        """
        logger.trace("Getting mask. Image shape: %s", new_face.shape)  # type: ignore[attr-defined]
        if self._args.mask_type not in ("none", "predicted"):
            mask_centering = detected_face.mask[self._args.mask_type].stored_centering
        else:
            mask_centering = "face"  # Unused but requires a valid value
        assert self._adjustments.mask is not None
        mask, raw_mask = self._adjustments.mask.run(detected_face,
                                                    reference_face.pose.offset[mask_centering],
                                                    reference_face.pose.offset[self._centering],
                                                    self._centering,
                                                    predicted_mask=predicted_mask)
        logger.trace("Adding mask to alpha channel")  # type: ignore[attr-defined]
        new_face = np.concatenate((new_face, mask), -1)
        logger.trace("Got mask. Image shape: %s", new_face.shape)  # type: ignore[attr-defined]
        return new_face, raw_mask

    def _post_warp_adjustments(self, background: np.ndarray, new_image: np.ndarray) -> np.ndarray:
        """ Perform any requested adjustments to the swapped faces after they have been transformed
        into the final frame.

        Parameters
        ----------
        background: :class:`numpy.ndarray`
            The original frame
        new_image: :class:`numpy.ndarray`
            A blank frame of original frame size with the faces warped onto it

        Returns
        -------
        :class:`numpy.ndarray`
            The final merged and swapped frame with any requested post-warp adjustments applied
        """
        if self._adjustments.sharpening is not None:
            new_image = self._adjustments.sharpening.run(new_image)

        if self._draw_transparent:
            frame = new_image
        else:
            foreground, mask = np.split(new_image,  # pylint:disable=unbalanced-tuple-unpacking
                                        (3, ),
                                        axis=-1)
            foreground *= mask
            background *= (1.0 - mask)
            background += foreground
            frame = background
        np.clip(frame, 0.0, 1.0, out=frame)
        return frame

    def _scale_image(self, frame: np.ndarray) -> np.ndarray:
        """ Scale the final image if requested.

        If output scale has been requested in command line arguments, scale the output
        otherwise return the final frame.

        Parameters
        ----------
        frame: :class:`numpy.ndarray`
            The final frame with faces swapped

        Returns
        -------
        :class:`numpy.ndarray`
            The final frame scaled by the requested scaling factor
        """
        if self._scale == 1:
            return frame
        logger.trace("source frame: %s", frame.shape)  # type: ignore[attr-defined]
        interp = cv2.INTER_CUBIC if self._scale > 1 else cv2.INTER_AREA
        dims = (round((frame.shape[1] / 2 * self._scale) * 2),
                round((frame.shape[0] / 2 * self._scale) * 2))
        frame = cv2.resize(frame, dims, interpolation=interp)
        logger.trace("resized frame: %s", frame.shape)  # type: ignore[attr-defined]
        np.clip(frame, 0.0, 1.0, out=frame)
        return frame
