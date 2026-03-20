#! /usr/env/bin/python3
"""Handles individual plugins within a plugin runner """
from __future__ import annotations

import abc
import logging
import typing as T

import numpy as np
from torch.cuda import OutOfMemoryError

from lib.align.aligned_utils import (batch_adjust_matrices, batch_align, batch_resize,
                                     batch_sub_crop)
from lib.align.constants import EXTRACT_RATIOS, LandmarkType
from lib.logger import parse_class_init
from lib.utils import FaceswapError, get_module_objects
from plugins.plugin_loader import PluginLoader
from plugins.extract.base import ExtractPlugin
from plugins.extract.extract_config import load_config
from .plugin_utils import compile_models, get_torch_modules, warmup_plugin

from .runner import ExtractRunner


if T.TYPE_CHECKING:
    import numpy.typing as npt
    from lib.align.constants import CenteringType
    from plugins.extract.base import FacePlugin
    from .objects import ExtractBatch

logger = logging.getLogger(__name__)


OOM_MESSAGE = (
    "You do not have enough GPU memory available to run detection at the selected batch size. You"
    "can try a number of things:"
    "\n1) Close any other application that is using your GPU (web browsers are particularly bad "
    "for this)."
    "\n2) Try again. Sometimes this can be a transient issue when you are close to VRAM capacity."
    "\n3) Lower the batch size (the amount of images fed into the model) by editing the plugin "
    "settings (GUI: Settings > Configure extract settings, CLI: Edit the file "
    "faceswap/config/extract.ini)."
    "\n4) Use lighter weight plugins."
    "\n5) Enable fewer plugins."
)


class ExtractHandler(abc.ABC):
    """Handles the execution of a plugin's pre_process, process and post_process actions

    Parameters
    ----------
    plugin
        The name of the plugin that this handler is to use
    compile_model
        ``True`` to compile any PyTorch models
    config_file
        Full path to a custom config file to load. ``None`` for default config
    """
    processors: tuple[T.Literal["pre_process", "process", "post_process"],
                      ...] = ("pre_process", "process", "post_process")
    """The processors which should have thread's launched for this handler"""

    def __init__(self,
                 plugin: str,
                 compile_model: bool = False,
                 config_file: str | None = None) -> None:
        self.plugin_type: T.Literal["detect",
                                    "align",
                                    "mask",
                                    "identity",
                                    "file"] = self._get_plugin_type()
        """The type of plugin that this handler manages"""
        self._config_file = config_file
        self.do_compile = compile_model
        """``True`` if any managed Torch modules are to be compiled"""
        self.plugin_name = plugin
        """The name of the plugin that is being handled"""
        load_config(config_file)
        self.plugin = PluginLoader.get_extractor(self.plugin_type, plugin)
        """The extraction plugin that this handler manages"""
        self._overridden: dict[T.Literal["pre_process", "process", "post_process"], bool] = {
            method: self._is_overridden(method) for method in self.processors}
        self._runner: ExtractRunner | None = None

    def __repr__(self) -> str:
        """Pretty print for logging"""
        params = {"plugin": repr(self.plugin_name),
                  "compile_model": self.do_compile,
                  "config_file": repr(self._config_file)}
        return f"{self.__class__.__name__}({', '.join(f'{k}={v}' for k, v in params.items())})"

    @property
    def batch_size(self) -> int:
        """The batch size of the plugin"""
        return self.plugin.batch_size

    @property
    def runner(self) -> ExtractRunner:
        """The runner that runs this handler"""
        assert self._runner is not None, "The handler must be called prior to accessing its runner"
        return self._runner

    @classmethod
    def _get_plugin_type(cls) -> T.Literal["detect", "align", "mask", "identity"]:
        """Obtain the type of extraction plugin that this runner is responsible for

        Returns
        -------
        The type of plugin that this runner is using
        """
        plugin_type = T.cast(T.Literal["detect", "align", "mask", "identity"],
                             cls.__name__.lower().replace("handler", ""))
        assert plugin_type in ("detect", "align", "mask", "identity")
        return plugin_type

    def _is_overridden(self, method_name: T.Literal["pre_process", "process", "post_process"]
                       ) -> bool:
        """Test if a plugin method's method has been overridden

        Parameters
        ----------
        method_name
            The name of the method that is to be checked

        Returns
        -------
        ``True`` if the plugin has overridden the given method
        """
        plugin_class = type(self.plugin)
        retval = (
            method_name in plugin_class.__dict__
            and plugin_class.__dict__[method_name] is not ExtractPlugin.__dict__.get(method_name)
            )
        logger.debug("[%s] Overridden method '%s': %s", self.plugin_name, method_name, retval)
        return retval

    def init_model(self) -> None:
        """Load the model, compile it, if requested, and send a warmup batch through. Called either
        from the main thread, if compiling, or from the inference thread if not."""
        logger.debug("[%s.load] Loading model", self.plugin_name)
        self.plugin.model = self.plugin.load_model()

        torch_modules = get_torch_modules(self.plugin)
        if not torch_modules or not self.do_compile:
            logger.debug("[%s.load] Plugin does not need compiling", self.plugin.name)
            warmup_plugin(self.plugin, self.plugin.batch_size)
            return
        logger.debug("[%s.load] Compiling plugin", self.plugin.name)
        compile_models(self.plugin, torch_modules)

    def _predict(self, feed: np.ndarray) -> np.ndarray:
        """Obtain a prediction from the plugin

        Parameters
        ----------
        feed
            The batch to feed the model

        Returns
        -------
        The prediction from the model

        Raises
        ------
        FaceswapError
            If an OOM occurs
        """
        feed_size = feed.shape[0]
        is_padded = self.do_compile and feed_size < self.plugin.batch_size
        batch_feed = feed
        if is_padded:  # Prevent model re-compile on undersized batch
            batch_feed = np.empty((self.plugin.batch_size, *feed.shape[1:]), dtype=feed.dtype)
            logger.debug("[%s.process] Padding undersized batch of shape %s to %s",
                         self.plugin.name, feed.shape, batch_feed.shape)
            batch_feed[:feed_size] = feed
        try:
            retval = self.plugin.process(batch_feed)
        except OutOfMemoryError as err:
            raise FaceswapError(OOM_MESSAGE) from err
        if is_padded and retval.dtype == "object":
            out = np.empty(retval.shape, dtype="object")
            out[:] = [x[:feed_size] for x in retval]
            retval = out
        elif is_padded:
            retval = retval[:feed_size]
        return retval

    def _format_images(self, images: npt.NDArray[np.uint8]) -> np.ndarray:
        """Format the incoming UINT8 0-255 images to the format specified by the plugin

        Parameters
        ----------
        images
            The batch of UINT8 images to format

        Returns
        -------
        The batch of images formatted and scaled for the plugin
        """
        retval = images if self.plugin.dtype == np.uint8 else images.astype(self.plugin.dtype)
        if self.plugin.scale == (0, 255):
            return retval
        low, high = self.plugin.scale
        im_range = high - low
        retval /= (255. / im_range)
        retval += low
        return retval

    def output_info(self) -> None:
        """Called after the final item is put to the out queue. Override for plugin runner
        specific output"""
        return

    @abc.abstractmethod
    def pre_process(self, batch: ExtractBatch) -> None:
        """ Override to perform plugin type specific behavior for pre-processing on the given batch
        object, ready for inference.

        Parameters
        ----------
        batch
            The incoming ExtractBatch to use for pre-processing
        """

    @abc.abstractmethod
    def process(self, batch: ExtractBatch) -> None:
        """Override to plugin type specific processing to get results from the plugin's inference
        for the given batch.

        Parameters
        ----------
        batch
            The incoming ExtractBatch to use for processing
        """

    @abc.abstractmethod
    def post_process(self, batch: ExtractBatch) -> None:
        """Perform post-processing on the given batch object, ready for exit from the plugin.
        Override for plugin type specific behavior

        Parameters
        ----------
        batch
            The incoming ExtractBatch to use for post-processing
        """

    def __call__(self, input_plugin: ExtractHandler | ExtractRunner | None = None,
                 profile: bool = False) -> ExtractRunner:
        """Build and start the plugin handler's runner

        Parameters
        ----------
        input_plugin
            The input plugin handler or it's runner that feeds this handler. ``None`` if data is
            to be fed through the handler runner's `put` method (ie, the first handler in an
            extraction chain). Default: ``None``
        profile
            ``True`` if the runner is to be profiled, indicating that threads will not be started.
            Default: ``False``

        Returns
        -------
        The extract plugin handler's runner for this handler
        """
        logger.debug("[%s] Initializing runner from handler", self.plugin.name)
        runner = ExtractRunner(self)
        input_runner = input_plugin.runner if isinstance(input_plugin,
                                                         ExtractHandler) else input_plugin
        runner(input_runner, profile)
        return runner


class ExtractHandlerFace(ExtractHandler, abc.ABC):
    """Handles an extract plugin. Extended with methods common to plugins that use aligned face
    images as input

    Parameters
    ----------
    plugin
        The name of the plugin that this runner is to use
    compile_model
        ``True`` to compile any PyTorch models
    config_file
        Full path to a custom config file to load. ``None`` for default config
    """
    _logged_warning: dict[str, bool] = {"mask": False, "identity": False}
    """Stores whether a warning has been issued for non-68 point landmarks for this plugin type"""

    def __init__(self,
                 plugin: str,
                 compile_model: bool = False,
                 config_file: str | None = None) -> None:
        super().__init__(plugin, compile_model=compile_model, config_file=config_file)
        self.plugin: FacePlugin

        self._input_size = self.plugin.input_size
        self._centering: CenteringType = self.plugin.centering
        self.storage_name = self.plugin.storage_name
        """The name that the object will be stored with in the alignments file"""

        self._padding = round((self._input_size * EXTRACT_RATIOS[self._centering]) / 2)
        self._aligned_mat_name = ("matrices" if self._centering == "legacy"
                                  else f"matrices_{self._centering}")

        # Aligned handling
        self._head_to_base_ratio = (1 - EXTRACT_RATIOS["head"]) / 2
        self._head_to_centering_ratio = ((1 - EXTRACT_RATIOS["head"]) /
                                         (1 - EXTRACT_RATIOS[self._centering]) / 2)
        self._aligned_offsets_name = f"offsets_{self._centering}"

    def _maybe_log_warning(self, landmark_type: LandmarkType | None) -> None:
        """Log a warning the first time if/when non-68 point landmarks are seen

        Parameters
        ----------
        landmark_type
            The type of landmarks within the batch
        """
        assert landmark_type is not None
        if self._logged_warning[self.plugin_type] or landmark_type in (LandmarkType.LM_2D_68,
                                                                       LandmarkType.LM_2D_98):
            return
        ptype = "Masks" if self.plugin_type == "mask" else "Identities"
        logger.warning("Faces do not contain landmark data. %s are likely to be sub-standard",
                       ptype)
        self._logged_warning[self.plugin_type] = True

    # Pre-processing
    def _get_matrices(self, matrices: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Obtain the (N, 2, 3) matrices for the face plugin's centering type

        Parameters
        ----------
        matrices
            The normalized alignment matrices for aligning faces from the image

        Returns
        -------
        The adjustment matrices for taking the image patch from the image for plugin input
        """
        return batch_adjust_matrices(matrices, self._input_size, self._padding)

    def _get_faces(self,  # pylint:disable=too-many-locals
                   images: list[npt.NDArray[np.uint8]],
                   image_ids: npt.NDArray[np.int32],
                   matrices: npt.NDArray[np.float32],
                   with_alpha: bool = False) -> npt.NDArray[np.uint8]:
        """Obtain the cropped and aligned faces from the batch of images

        Parameters
        ----------
        images
            The full size frames for the batch
        image_ids
            The image ids for each detected face
        matrices
            The adjustment matrices for taking the image patch from the frame for plugin input
        with_alpha
            ``True`` to add a filled alpha channel to the batch of images prior to warping to
            faces. Default: ``False``

        Returns
        -------
        Batch of 3 or 4 channel face patches for feeding the model. If `with_alpha` is selected
        then the final channel is an ROI mask indicating areas that go out of bounds
        """
        if with_alpha:
            images = [np.concatenate([i, np.zeros((*i.shape[:2], 1), dtype=i.dtype) + 255],
                                     axis=-1)
                      for i in images]
        return batch_align(images, image_ids, matrices, self._input_size)

    # Aligned faces as input methods
    def _get_faces_aligned(self,
                           images: list[npt.NDArray[np.uint8]],
                           image_ids: npt.NDArray[np.int32],
                           source_padding: npt.NDArray[np.float32],
                           dest_padding: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
        """Obtain the batch of faces when input images are a batch of extracted faceswap faces

        Parameters
        ----------
        images
            The batch of faceswap extracted faces to obtain the model input images from
        image_ids
            The image ids for each detected face
        source_padding
            The normalized (N, x, y) padding used for the aligned image's centering
        dest_padding
            The normalized (N, x, y) padding used for the plugin's centering

        Returns
        -------
        The sub-crop from the aligned faces for feeding the model
        """
        imgs = np.array([images[idx] for idx in image_ids] if len(images) != len(image_ids)
                        else images)
        assert imgs.dtype != object, "Aligned images must all be the same size"
        if self._centering == "head":
            return batch_resize(imgs, self._input_size)

        src_size = imgs.shape[1]
        out_size = 2 * int(np.rint(src_size * self._head_to_centering_ratio))
        base_size = 2 * int(np.rint(src_size * self._head_to_base_ratio))
        padding_diff = (src_size - out_size) // 2
        delta = dest_padding - source_padding
        offsets = np.rint(delta * base_size + padding_diff).astype(np.int32)
        imgs = batch_sub_crop(imgs, offsets, out_size)
        return batch_resize(imgs, self._input_size)

    def process(self, batch: ExtractBatch) -> None:
        """Perform inference to get results from the plugin for the given batch. Override for
        plugin type specific processing

        Parameters
        ----------
        batch
            The incoming ExtractBatch to use for processing
        """
        batch.data = self._predict(batch.data)


class FileHandler(ExtractHandler):
    """A pseudo handler that passes through data when the pipeline is driven entirely by an
    alignments file (ie: no plugins are being loaded). This is effectively a No-op which allows
    the data to pass straight from input to output"""
    processors: tuple[T.Literal["pre_process", "process", "post_process"], ...] = tuple()
    """File handler launches no threads"""

    class Plugin:  # pylint:disable=too-few-public-methods
        """Dummy plugin with required properties"""
        name = "file"
        batch_size = 128  # Irrelevant, data is just passed through

    def __init__(self) -> None:  # pylint:disable=super-init-not-called
        # Don't call super as we are not compatible
        logger.debug(parse_class_init(locals()))
        self.do_compile = False
        self.plugin_type = "file"
        self.plugin_name = "file"
        self.plugin = self.Plugin  # type:ignore[assignment]
        self._runner: ExtractRunner | None = None

    def __repr__(self) -> str:
        """Pretty print for logging"""
        return f"{self.__class__.__name__}()"

    def pre_process(self, batch: ExtractBatch) -> None:
        """Not applicable for passthrough plugin."""
        return

    def post_process(self, batch: ExtractBatch) -> None:
        """Not applicable for passthrough plugin."""
        return

    def process(self, batch: ExtractBatch) -> None:
        """Not applicable for passthrough plugin."""
        return

    def __call__(self, input_plugin: ExtractHandler | ExtractRunner | None = None,
                 profile: bool = False) -> ExtractRunner:
        """Build and start the plugin handler's runner. Overridden to ensure that neither an input
        plugin or profile are set

        Parameters
        ----------
        input_plugin
            The input plugin handler or it's runner that feeds this handler. ``None`` if data is
            to be fed through the handler runner's `put` method (ie, the first handler in an
            extraction chain). Must be ``None`` for file handler
        profile
            ``True`` if the runner is to be profiled, indicating that threads will not be started.
            Must be ``False`` for file handler

        Returns
        -------
        The extract plugin handler's runner for this handler
        """
        assert input_plugin is None, "input_plugin must be ``None`` for file handler"
        assert not profile, "profile must be ``False`` for file handler"
        return super().__call__(input_plugin=None, profile=False)


get_module_objects(__name__)
