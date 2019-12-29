#!/usr/bin/env python3
""" Converter for Faceswap """

import logging

import cv2
import numpy as np

from plugins.plugin_loader import PluginLoader

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Converter():
    """ The converter is responsible for swapping the original face(s) in a frame with the output
    of a trained Faceswap model.

    Parameters
    ----------
    output_size: int
        The size of the face, in pixels, that is output from the Faceswap model
    coverage_ratio: float
        The ratio of the training image that was used for training the Faceswap model
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
    def __init__(self, output_size, coverage_ratio, draw_transparent, pre_encode,
                 arguments, configfile=None):
        logger.debug("Initializing %s: (output_size: %s,  coverage_ratio: %s, draw_transparent: "
                     "%s, pre_encode: %s, arguments: %s, configfile: %s)", self.__class__.__name__,
                     output_size, coverage_ratio, draw_transparent, pre_encode, arguments,
                     configfile)
        self._output_size = output_size
        self._coverage_ratio = coverage_ratio
        self._draw_transparent = draw_transparent
        self._writer_pre_encode = pre_encode
        self._args = arguments
        self._configfile = configfile

        self._scale = arguments.output_scale / 100
        self._adjustments = dict(box=None, mask=None, color=None, seamless=None, scaling=None)

        self._load_plugins()
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def cli_arguments(self):
        """:class:`argparse.Namespace`: The command line arguments passed to the convert
        process """
        return self._args

    def reinitialize(self, config):
        """ Reinitialize this :class:`Converter`.

        Called as part of the :mod:`~tools.preview` tool. Resets all adjustments then loads the
        plugins as specified in the given config.

        Parameters
        ----------
        config: :class:`lib.config.FaceswapConfig`
            Pre-loaded :class:`lib.config.FaceswapConfig`. used over any configuration on disk.
        """
        logger.debug("Reinitializing converter")
        self._adjustments = dict(box=None, mask=None, color=None, seamless=None, scaling=None)
        self._load_plugins(config=config, disable_logging=True)
        logger.debug("Reinitialized converter")

    def _load_plugins(self, config=None, disable_logging=False):
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
        self._adjustments["box"] = PluginLoader.get_converter(
            "mask",
            "box_blend",
            disable_logging=disable_logging)(self._output_size,
                                             configfile=self._configfile,
                                             config=config)

        self._adjustments["mask"] = PluginLoader.get_converter(
            "mask",
            "mask_blend",
            disable_logging=disable_logging)(self._args.mask_type,
                                             self._output_size,
                                             self._coverage_ratio,
                                             configfile=self._configfile,
                                             config=config)

        if self._args.color_adjustment != "none" and self._args.color_adjustment is not None:
            self._adjustments["color"] = PluginLoader.get_converter(
                "color",
                self._args.color_adjustment,
                disable_logging=disable_logging)(configfile=self._configfile, config=config)

        if self._args.scaling != "none" and self._args.scaling is not None:
            self._adjustments["scaling"] = PluginLoader.get_converter(
                "scaling",
                self._args.scaling,
                disable_logging=disable_logging)(configfile=self._configfile, config=config)
        logger.debug("Loaded plugins: %s", self._adjustments)

    def process(self, in_queue, out_queue):
        """ Main convert process.

        Takes items from the in queue, runs the relevant adjustments, patches faces to final frame
        and outputs patched frame to the out queue.

        Parameters
        ----------
        in_queue: :class:`queue.Queue`
            The output from :class:`scripts.convert.Predictor`. Contains detected faces from the
            Faceswap model as well as the frame to be patched.
        out_queue: :class:`queue.Queue`
            The queue to place patched frames into for writing by one of Faceswap's
            :mod:`plugins.convert.writer` plugins.
        """
        logger.debug("Starting convert process. (in_queue: %s, out_queue: %s)",
                     in_queue, out_queue)
        while True:
            items = in_queue.get()
            if items == "EOF":
                logger.debug("EOF Received")
                logger.debug("Patch queue finished")
                # Signal EOF to other processes in pool
                logger.debug("Putting EOF back to in_queue")
                in_queue.put(items)
                break

            if isinstance(items, dict):
                items = [items]
            for item in items:
                logger.trace("Patch queue got: '%s'", item["filename"])
                try:
                    image = self._patch_image(item)
                except Exception as err:  # pylint: disable=broad-except
                    # Log error and output original frame
                    logger.error("Failed to convert image: '%s'. Reason: %s",
                                 item["filename"], str(err))
                    image = item["image"]
                    # UNCOMMENT THIS CODE BLOCK TO PRINT TRACEBACK ERRORS
                    # import sys ; import traceback
                    # exc_info = sys.exc_info() ; traceback.print_exception(*exc_info)
                logger.trace("Out queue put: %s", item["filename"])
                out_queue.put((item["filename"], image))
        logger.debug("Completed convert process")

    def _patch_image(self, predicted):
        """ Patch a swapped face onto a frame.

        Run selected adjustments and swap the faces in a frame.

        Parameters
        ----------
        predicted: dict
            The output from :class:`scripts.convert.Predictor`.

        Returns
        -------
        :class: `numpy.ndarray` or pre-encoded image output
            The final frame ready for writing by a :mod:`plugins.convert.writer` plugin.
            Frame is either an array, or the pre-encoded output from the writer's pre-encode
            function (if it has one)

        """
        logger.trace("Patching image: '%s'", predicted["filename"])
        frame_size = (predicted["image"].shape[1], predicted["image"].shape[0])
        new_image, background = self._get_new_image(predicted, frame_size)
        patched_face = self._post_warp_adjustments(background, new_image)
        patched_face = self._scale_image(patched_face)
        patched_face *= 255.0
        patched_face = np.rint(patched_face,
                               out=np.empty(patched_face.shape, dtype="uint8"),
                               casting='unsafe')
        if self._writer_pre_encode is not None:
            patched_face = self._writer_pre_encode(patched_face)
        logger.trace("Patched image: '%s'", predicted["filename"])
        return patched_face

    def _get_new_image(self, predicted, frame_size):
        """ Get the new face from the predictor and apply pre-warp manipulations.

        Applies any requested adjustments to the raw output of the Faceswap model
        before transforming the image into the target frame.

        Parameters
        ----------
        predicted: dict
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
        logger.trace("Getting: (filename: '%s', faces: %s)",
                     predicted["filename"], len(predicted["swapped_faces"]))

        placeholder = np.zeros((frame_size[1], frame_size[0], 4), dtype="float32")
        background = predicted["image"] / np.array(255.0, dtype="float32")
        placeholder[:, :, :3] = background

        for new_face, detected_face in zip(predicted["swapped_faces"],
                                           predicted["detected_faces"]):
            predicted_mask = new_face[:, :, -1] if new_face.shape[2] == 4 else None
            new_face = new_face[:, :, :3]
            interpolator = detected_face.reference_interpolators[1]

            new_face = self._pre_warp_adjustments(new_face, detected_face, predicted_mask)

            # Warp face with the mask
            cv2.warpAffine(new_face,
                           detected_face.reference_matrix,
                           frame_size,
                           placeholder,
                           flags=cv2.WARP_INVERSE_MAP | interpolator,
                           borderMode=cv2.BORDER_TRANSPARENT)

        np.clip(placeholder, 0.0, 1.0, out=placeholder)
        logger.trace("Got filename: '%s'. (placeholders: %s)",
                     predicted["filename"], placeholder.shape)

        return placeholder, background

    def _pre_warp_adjustments(self, new_face, detected_face, predicted_mask):
        """ Run any requested adjustments that can be performed on the raw output from the Faceswap
        model.

        Any adjustments that can be performed before warping the face into the final frame are
        performed here.

        Parameters
        ----------
        new_face: :class:`numpy.ndarray`
            The swapped face received from the faceswap model.
        detected_face: :class:`~lib.faces_detect.DetectedFace`
            The detected_face object as defined in :class:`scripts.convert.Predictor`
        predicted_mask: :class:`numpy.ndarray` or ``None``
            The predicted mask output from the Faceswap model. ``None`` if the model
            did not learn a mask

        Returns
        -------
        :class:`numpy.ndarray`
            The face output from the Faceswap Model with any requested pre-warp adjustments
            performed.
        """
        logger.trace("new_face shape: %s, predicted_mask shape: %s", new_face.shape,
                     predicted_mask.shape if predicted_mask is not None else None)
        old_face = detected_face.reference_face[..., :3] / 255.0
        new_face = self._adjustments["box"].run(new_face)
        new_face, raw_mask = self._get_image_mask(new_face, detected_face, predicted_mask)
        if self._adjustments["color"] is not None:
            new_face = self._adjustments["color"].run(old_face, new_face, raw_mask)
        if self._adjustments["seamless"] is not None:
            new_face = self._adjustments["seamless"].run(old_face, new_face, raw_mask)
        logger.trace("returning: new_face shape %s", new_face.shape)
        return new_face

    def _get_image_mask(self, new_face, detected_face, predicted_mask):
        """ Return any selected image mask and intersect with any box mask.

        Places the requested mask into the new face's Alpha channel, intersecting with any box
        mask that has already been applied.

        Parameters
        ----------
        new_face: :class:`numpy.ndarray`
            The swapped face received from the faceswap model, with any box mask applied
        detected_face: :class:`~lib.faces_detect.DetectedFace`
            The detected_face object as defined in :class:`scripts.convert.Predictor`
        predicted_mask: :class:`numpy.ndarray` or ``None``
            The predicted mask output from the Faceswap model. ``None`` if the model
            did not learn a mask

        Returns
        :class:`numpy.ndarray`
            The swapped face with the requested mask added to the Alpha channel
        """
        logger.trace("Getting mask. Image shape: %s", new_face.shape)
        mask, raw_mask = self._adjustments["mask"].run(detected_face, predicted_mask)
        if new_face.shape[2] == 4:
            logger.trace("Combining mask with alpha channel box mask")
            new_face[:, :, -1] = np.minimum(new_face[:, :, -1], mask.squeeze())
        else:
            logger.trace("Adding mask to alpha channel")
            new_face = np.concatenate((new_face, mask), -1)
        logger.trace("Got mask. Image shape: %s", new_face.shape)
        return new_face, raw_mask

    def _post_warp_adjustments(self, background, new_image):
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
        if self._adjustments["scaling"] is not None:
            new_image = self._adjustments["scaling"].run(new_image)

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

    def _scale_image(self, frame):
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
        logger.trace("source frame: %s", frame.shape)
        interp = cv2.INTER_CUBIC if self._scale > 1 else cv2.INTER_AREA
        dims = (round((frame.shape[1] / 2 * self._scale) * 2),
                round((frame.shape[0] / 2 * self._scale) * 2))
        frame = cv2.resize(frame, dims, interpolation=interp)
        logger.trace("resized frame: %s", frame.shape)
        np.clip(frame, 0.0, 1.0, out=frame)
        return frame
