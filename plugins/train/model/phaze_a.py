#!/usr/bin/env python3
""" Phaze-A Model by TorzDF with thanks to BirbFakes and the myriad of testers. """

# pylint:disable=too-many-lines
from __future__ import annotations
import logging
import typing as T
from dataclasses import dataclass

import numpy as np
import keras
from keras import applications as kapp, layers as kl

from lib.logger import parse_class_init
from lib.model.nn_blocks import (
    Conv2D, Conv2DBlock, Conv2DOutput, ResidualBlock, UpscaleBlock, Upscale2xBlock,
    UpscaleResizeImagesBlock, UpscaleDNYBlock)
from lib.model.normalization import (
    AdaInstanceNormalization, GroupNormalization, InstanceNormalization, RMSNormalization)
from lib.model.networks import ViT, TypeModelsViT
from lib.utils import get_keras_version, FaceswapError
from plugins.train.train_config import Loss as cfg_loss

from ._base import ModelBase, get_all_sub_models
from . import phaze_a_defaults as cfg

if T.TYPE_CHECKING:
    from keras import KerasTensor

logger = logging.getLogger(__name__)


@dataclass
class _EncoderInfo:
    """ Contains model configuration options for various Phaze-A Encoders.

    Parameters
    ----------
    keras_name: str
        The name of the encoder in Keras Applications. Empty string `""` if the encoder does not
        exist in Keras Applications
    default_size: int
        The default input size of the encoder
    keras_min: float, optional
        The lowest version of Keras that the encoder can be used for. Default: `3.0`
    scaling: tuple, optional
        The float scaling that the encoder expects. Default: `(0, 1)`
    min_size: int, optional
        The minimum input size that the encoder will allow. Default: 32
    enforce_for_weights: bool, optional
        ``True`` if the input size for the model must be forced to the default size when loading
        imagenet weights, otherwise ``False``. Default: ``False``
    color_order: str, optional
        The color order that the model expects (`"bgr"` or `"rgb"`). Default: `"rgb"`
    """
    keras_name: str
    default_size: int
    keras_min: tuple[int, int] = (3, 0)
    scaling: tuple[int, int] = (0, 1)
    min_size: int = 32
    enforce_for_weights: bool = False
    color_order: T.Literal["bgr", "rgb"] = "rgb"


_MODEL_MAPPING: dict[str, _EncoderInfo] = {
    "clipv_farl-b-16-16": _EncoderInfo(
        keras_name="FaRL-B-16-16", default_size=224),
    "clipv_farl-b-16-64": _EncoderInfo(
        keras_name="FaRL-B-16-64", default_size=224),
    "clipv_vit-b-16": _EncoderInfo(
        keras_name="ViT-B-16", default_size=224),
    "clipv_vit-b-32": _EncoderInfo(
        keras_name="ViT-B-32", default_size=224),
    "clipv_vit-l-14": _EncoderInfo(
        keras_name="ViT-L-14", default_size=224),
    "clipv_vit-l-14-336px": _EncoderInfo(
        keras_name="ViT-L-14-336px", default_size=336),
    "convnext_tiny": _EncoderInfo(
        keras_name="ConvNeXtTiny", scaling=(0, 255), default_size=224),
    "convnext_small": _EncoderInfo(
        keras_name="ConvNeXtSmall", scaling=(0, 255), default_size=224),
    "convnext_base": _EncoderInfo(
        keras_name="ConvNeXtBase", scaling=(0, 255), default_size=224),
    "convnext_large": _EncoderInfo(
        keras_name="ConvNeXtLarge", scaling=(0, 255), default_size=224),
    "convnext_extra_large": _EncoderInfo(
        keras_name="ConvNeXtXLarge", scaling=(0, 255), default_size=224),
    "densenet121": _EncoderInfo(
        keras_name="DenseNet121", default_size=224),
    "densenet169": _EncoderInfo(
        keras_name="DenseNet169", default_size=224),
    "densenet201": _EncoderInfo(
        keras_name="DenseNet201", default_size=224),
    "efficientnet_b0": _EncoderInfo(
        keras_name="EfficientNetB0", scaling=(0, 255), default_size=224),
    "efficientnet_b1": _EncoderInfo(
        keras_name="EfficientNetB1", scaling=(0, 255), default_size=240),
    "efficientnet_b2": _EncoderInfo(
        keras_name="EfficientNetB2", scaling=(0, 255), default_size=260),
    "efficientnet_b3": _EncoderInfo(
        keras_name="EfficientNetB3", scaling=(0, 255), default_size=300),
    "efficientnet_b4": _EncoderInfo(
        keras_name="EfficientNetB4", scaling=(0, 255), default_size=380),
    "efficientnet_b5": _EncoderInfo(
        keras_name="EfficientNetB5", scaling=(0, 255), default_size=456),
    "efficientnet_b6": _EncoderInfo(
        keras_name="EfficientNetB6", scaling=(0, 255), default_size=528),
    "efficientnet_b7": _EncoderInfo(
        keras_name="EfficientNetB7", scaling=(0, 255), default_size=600),
    "efficientnet_v2_b0": _EncoderInfo(
        keras_name="EfficientNetV2B0", scaling=(-1, 1), default_size=224),
    "efficientnet_v2_b1": _EncoderInfo(
        keras_name="EfficientNetV2B1", scaling=(-1, 1), default_size=240),
    "efficientnet_v2_b2": _EncoderInfo(
        keras_name="EfficientNetV2B2", scaling=(-1, 1), default_size=260),
    "efficientnet_v2_b3": _EncoderInfo(
        keras_name="EfficientNetV2B3", scaling=(-1, 1), default_size=300),
    "efficientnet_v2_s": _EncoderInfo(
        keras_name="EfficientNetV2S", scaling=(-1, 1), default_size=384),
    "efficientnet_v2_m": _EncoderInfo(
        keras_name="EfficientNetV2M", scaling=(-1, 1), default_size=480),
    "efficientnet_v2_l": _EncoderInfo(
        keras_name="EfficientNetV2L", scaling=(-1, 1), default_size=480),
    "inception_resnet_v2": _EncoderInfo(
        keras_name="InceptionResNetV2", scaling=(-1, 1), min_size=75, default_size=299),
    "inception_v3": _EncoderInfo(
        keras_name="InceptionV3", scaling=(-1, 1), min_size=75, default_size=299),
    "mobilenet": _EncoderInfo(
        keras_name="MobileNet", scaling=(-1, 1), default_size=224),
    "mobilenet_v2": _EncoderInfo(
        keras_name="MobileNetV2", scaling=(-1, 1), default_size=224),
    "mobilenet_v3_large": _EncoderInfo(
        keras_name="MobileNetV3Large", scaling=(-1, 1), default_size=224),
    "mobilenet_v3_small": _EncoderInfo(
        keras_name="MobileNetV3Small", scaling=(-1, 1), default_size=224),
    "nasnet_large": _EncoderInfo(
        keras_name="NASNetLarge", scaling=(-1, 1), default_size=331, enforce_for_weights=True),
    "nasnet_mobile": _EncoderInfo(
        keras_name="NASNetMobile", scaling=(-1, 1), default_size=224, enforce_for_weights=True),
    "resnet50": _EncoderInfo(
        keras_name="ResNet50", scaling=(-1, 1), min_size=32, default_size=224),
    "resnet50_v2": _EncoderInfo(
        keras_name="ResNet50V2", scaling=(-1, 1), default_size=224),
    "resnet101": _EncoderInfo(
        keras_name="ResNet101", scaling=(-1, 1), default_size=224),
    "resnet101_v2": _EncoderInfo(
        keras_name="ResNet101V2", scaling=(-1, 1), default_size=224),
    "resnet152": _EncoderInfo(
        keras_name="ResNet152", scaling=(-1, 1), default_size=224),
    "resnet152_v2": _EncoderInfo(
        keras_name="ResNet152V2", scaling=(-1, 1), default_size=224),
    "vgg16": _EncoderInfo(
        keras_name="VGG16", color_order="bgr", scaling=(0, 255), default_size=224),
    "vgg19": _EncoderInfo(
        keras_name="VGG19", color_order="bgr", scaling=(0, 255), default_size=224),
    "xception": _EncoderInfo(
        keras_name="Xception", scaling=(-1, 1), min_size=71, default_size=299),
    "fs_original": _EncoderInfo(
        keras_name="", color_order="bgr", min_size=32, default_size=1024)}


class Model(ModelBase):
    """ Phaze-A Faceswap Model.

    An highly adaptable and configurable model by torzDF

    Parameters
    ----------513
    args: varies
        The default command line arguments passed in from :class:`~scripts.train.Train` or
        :class:`~scripts.train.Convert`
    kwargs: varies
        The default keyword arguments passed in from :class:`~scripts.train.Train` or
        :class:`~scripts.train.Convert`
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if cfg.output_size() % 16 != 0:
            raise FaceswapError("Phaze-A output shape must be a multiple of 16")

        self._validate_encoder_architecture()

        self.input_shape: tuple[int, int, int] = self._get_input_shape()
        self.color_order = _MODEL_MAPPING[cfg.enc_architecture()].color_order

    @property
    def freeze_layers(self) -> list[str]:
        """ list[str] : Valid layers to freeze based on configured options """
        return self._select_real_layers(cfg.freeze_layers())

    @property
    def load_layers(self) -> list[str]:
        """ list[str] : Valid layers to load based on configured options """
        return self._select_real_layers(cfg.load_layers())

    def build(self) -> None:
        """ Build the model and assign to :attr:`model`.

        Override's the default build function for allowing the setting of dropout rate for pre-
        existing models.
        """
        is_summary = hasattr(self._args, "summary") and self._args.summary
        if not self._io.model_exists or self._is_predict or is_summary:
            logger.debug("New model, inference or summary. Falling back to default build: "
                         "(exists: %s, inference: %s, is_summary: %s)",
                         self._io.model_exists, self._is_predict, is_summary)
            super().build()
            return
        model = self.io.load()
        model = self._update_dropouts(model)
        self._model = model
        self._compile_model()
        self._output_summary()

    def _update_dropouts(self, model: keras.models.Model) -> keras.models.Model:
        """ Update the saved model with new dropout rates.

        Keras, annoyingly, does not actually change the dropout of the underlying layer, so we need
        to update the rate, then clone the model into a new model and reload weights.

        Parameters
        ----------
        model: :class:`keras.models.Model`
            The loaded saved Keras Model to update the dropout rates for

        Returns
        -------
        :class:`keras.models.Model`
            The loaded Keras Model with the dropout rates updated
        """
        dropouts = {"fc": cfg.fc_dropout(), "gblock": cfg.fc_gblock_dropout()}
        logger.debug("Config dropouts: %s", dropouts)
        updated = False
        for mod in get_all_sub_models(model):
            if not mod.name.startswith("fc_"):
                continue
            key = "gblock" if "gblock" in mod.name else mod.name.split("_")[0]
            rate = dropouts[key]
            log_once = False
            for layer in mod.layers:
                if not isinstance(layer, kl.Dropout):
                    continue
                if layer.rate != rate:
                    logger.debug("Updating dropout rate for %s from %s to %s",
                                 f"{mod.name} - {layer.name}", layer.rate, rate)
                    if not log_once:
                        logger.info("Updating Dropout Rate for '%s' from %s to %s",
                                    mod.name, layer.rate, rate)
                        log_once = True
                    layer.rate = rate
                    updated = True
        if updated:
            logger.debug("Dropout rate updated. Cloning model")
            new_model = keras.models.clone_model(model)
            new_model.set_weights(model.get_weights())
            del model
            model = new_model
        return model

    def _select_real_layers(self, layers: list[str]) -> list[str]:
        """ Process the selected freeze or load layers configuration options and replace the
        `keras_encoder` option with the actual keras model name for the configured architecture

        Returns
        -------
        list
            The selected layers for weight freezing
        """
        arch = cfg.enc_architecture()
        # EfficientNetV2 is inconsistent with other model's naming conventions
        keras_name = _MODEL_MAPPING[arch].keras_name.replace("EfficientNetV2", "EfficientNetV2-")
        # CLIPv model is always called 'visual' regardless of weights/format loaded
        keras_name = "visual" if arch.startswith("clipv_") else keras_name

        if "keras_encoder" not in cfg.freeze_layers():
            retval = layers
        elif keras_name:
            retval = [layer.replace("keras_encoder", keras_name) for layer in layers]
            logger.debug("Substituting 'keras_encoder' for '%s'", keras_name)
        else:
            retval = [layer for layer in layers if layer != "keras_encoder"]
            logger.debug("Removing 'keras_encoder' for '%s'", keras_name)

        return retval

    def _get_input_shape(self) -> tuple[int, int, int]:
        """ Obtain the input shape for the model.

        Input shape is calculated from the selected Encoder's input size, scaled to the user
        selected Input Scaling, rounded down to the nearest 16 pixels.

        Notes
        -----
        Some models (NasNet) require the input size to be of a certain dimension if loading
        imagenet weights. In these instances resize inputs and raise warning message

        Returns
        -------
        tuple
            The shape tuple for the input size to the Phaze-A model
        """
        arch = cfg.enc_architecture()
        enforce_size = _MODEL_MAPPING[arch].enforce_for_weights
        default_size = _MODEL_MAPPING[arch].default_size
        scaling = cfg.enc_scaling() / 100

        min_size = _MODEL_MAPPING[arch].min_size
        size = int(max(min_size, ((default_size * scaling) // 16) * 16))

        if cfg.enc_load_weights() and enforce_size and scaling != 1.0:
            logger.warning("%s requires input size to be %spx when loading imagenet weights. "
                           "Adjusting input size from %spx to %spx",
                           arch, default_size, size, default_size)
            retval = (default_size, default_size, 3)
        else:
            retval = (size, size, 3)

        logger.debug("Encoder input set to: %s", retval)
        return retval

    def _validate_encoder_architecture(self) -> None:
        """ Validate that the requested architecture is a valid choice for the running system
        configuration.

        If the selection is not valid, an error is logged and system exits.
        """
        arch = cfg.enc_architecture()
        model = _MODEL_MAPPING.get(arch)
        if not model:
            raise FaceswapError(f"'{arch}' is not a valid choice for encoder architecture. Choose "
                                f"one of {list(_MODEL_MAPPING.keys())}.")

        keras_ver = get_keras_version()
        keras_min = model.keras_min
        if keras_ver < keras_min:
            raise FaceswapError(f"{arch}' is not compatible with your version of Keras. The "
                                f"minimum version required is {keras_min} whilst you have version "
                                f"{keras_ver} installed.")

    def build_model(self, inputs: list[KerasTensor]) -> keras.models.Model:
        """ Create the model's structure.

        Parameters
        ----------
        inputs: list[:class:`keras.KerasTensor`]
            A list of input tensors for the model. This will be a list of 2 tensors of
            shape :attr:`input_shape`, the first for side "a", the second for side "b".

        Returns
        -------
        :class:`keras.models.Model`
            The generated model
        """
        # Create sub-Models
        encoders = self._build_encoders(inputs)
        inters = self._build_fully_connected(encoders)
        g_blocks = self._build_g_blocks(inters)
        decoders = self._build_decoders(g_blocks)

        # Create Autoencoder
        outputs = decoders["a"] + decoders["b"]
        autoencoder = keras.models.Model(inputs, outputs, name=self.model_name)
        return autoencoder

    def _build_encoders(self, inputs: list[KerasTensor]) -> dict[str, keras.models.Model]:
        """ Build the encoders for Phaze-A

        Parameters
        ----------
        inputs: list[:class:`keras.KerasTensor`]
            A list of input tensors for the model. This will be a list of 2 tensors of
            shape :attr:`input_shape`, the first for side "a", the second for side "b".

        Returns
        -------
        dict
            side as key ('a' or 'b'), encoder for side as value
        """
        encoder = Encoder(self.input_shape)()
        retval = {"a": encoder(inputs[0]), "b": encoder(inputs[1])}
        logger.debug("Encoders: %s", retval)
        return retval

    def _build_fully_connected(
            self,
            inputs: dict[str, keras.models.Model]) -> dict[str, list[keras.models.Model]]:
        """ Build the fully connected layers for Phaze-A

        Parameters
        ----------
        inputs: dict
            The compiled encoder models that act as inputs to the fully connected layers

        Returns
        -------
        dict
            side as key ('a' or 'b'), fully connected model for side as value
        """
        input_shapes = inputs["a"].shape[1:]

        fc_a = fc_both = None
        if cfg.split_fc():
            fc_a = FullyConnected("a", input_shapes)()
            inter_a = [fc_a(inputs["a"])]
            inter_b = [FullyConnected("b", input_shapes)()(inputs["b"])]
        else:
            fc_both = FullyConnected("both", input_shapes)()
            inter_a = [fc_both(inputs["a"])]
            inter_b = [fc_both(inputs["b"])]

        shared_fc = None if cfg.shared_fc() == "none" else cfg.shared_fc()
        if shared_fc:
            if shared_fc == "full":
                fc_shared = FullyConnected("shared", input_shapes)()
            elif cfg.split_fc():
                assert fc_a is not None
                fc_shared = fc_a
            else:
                assert fc_both is not None
                fc_shared = fc_both
            inter_a = [kl.Concatenate(name="inter_a")([inter_a[0], fc_shared(inputs["a"])])]
            inter_b = [kl.Concatenate(name="inter_b")([inter_b[0], fc_shared(inputs["b"])])]

        if cfg.enable_gblock():
            fc_gblock = FullyConnected("gblock", input_shapes)()
            inter_a.append(fc_gblock(inputs["a"]))
            inter_b.append(fc_gblock(inputs["b"]))

        inter_a = inter_a[0] if len(inter_a) == 1 else inter_a
        inter_b = inter_b[0] if len(inter_b) == 1 else inter_b
        retval = {"a": inter_a, "b": inter_b}
        logger.debug("Fully Connected: %s", retval)
        return retval

    def _build_g_blocks(
                self,
                inputs: dict[str, list[keras.models.Model]]
            ) -> dict[str, list[keras.models.Model] | keras.models.Model]:
        """ Build the g-block layers for Phaze-A.

        If a g-block has not been selected for this model, then the original `inters` models are
        returned for passing straight to the decoder

        Parameters
        ----------
        inputs: dict
            The compiled inter models that act as inputs to the g_blocks

        Returns
        -------
        dict
            side as key ('a' or 'b'), g-block model for side as value. If g-block has been disabled
            then the values will be the fully connected layers
        """
        if not cfg.enable_gblock():
            logger.debug("No G-Block selected, returning Inters: %s", inputs)
            return inputs

        input_shapes = [inter.shape[1:] for inter in inputs["a"]]
        if cfg.split_gblock():
            retval = {"a": GBlock("a", input_shapes)()(inputs["a"]),
                      "b": GBlock("b", input_shapes)()(inputs["b"])}
        else:
            g_block = GBlock("both", input_shapes)()
            retval = {"a": g_block((inputs["a"])), "b": g_block((inputs["b"]))}

        logger.debug("G-Blocks: %s", retval)
        return retval

    def _build_decoders(self,
                        inputs: dict[str, list[keras.models.Model] | keras.models.Model]
                        ) -> dict[str, keras.models.Model]:
        """ Build the encoders for Phaze-A

        Parameters
        ----------
        inputs: dict
            A dict of inputs to the decoder. This will either be g-block output (if g-block is
            enabled) or fully connected layers output (if g-block is disabled).

        Returns
        -------
        dict
            side as key ('a' or 'b'), decoder for side as value
        """
        input_ = inputs["a"]
        # If input is inters, shapes will be a list.
        # There will only ever be 1 input. For inters: either inter out, or concatenate of inters
        # For g-block, this only ever has one output
        input_ = input_[0] if isinstance(input_, list) else input_

        # If learning a mask and upscales have been placed into FC layer, then the mask will also
        # come as an input
        if cfg_loss.learn_mask() and cfg.dec_upscales_in_fc():
            input_ = input_[0]

        input_shape = input_.shape[1:]

        if cfg.split_decoders():
            retval = {"a": Decoder("a", input_shape)()(inputs["a"]),
                      "b": Decoder("b", input_shape)()(inputs["b"])}
        else:
            decoder = Decoder("both", input_shape)()
            retval = {"a": decoder(inputs["a"]), "b": decoder(inputs["b"])}

        logger.debug("Decoders: %s", retval)
        return retval


def _bottleneck(inputs: KerasTensor, bottleneck: str, size: int, normalization: str
                ) -> KerasTensor:
    """ The bottleneck fully connected layer. Can be called from Encoder or FullyConnected layers.

    Parameters
    ----------
    inputs: :class:`keras.KerasTensor`
        The input to the bottleneck layer
    bottleneck: str or ``None``
        The type of layer to use for the bottleneck. ``None`` to not use a bottleneck
    size: int
        The number of nodes for the dense layer (if selected)
    normalization: str
        The normalization method to use prior to the bottleneck layer

    Returns
    -------
    :class:`keras.KerasTensor`
        The output from the bottleneck
    """
    norm = None if normalization == "none" else normalization
    norms = {"layer": kl.LayerNormalization,
             "rms": RMSNormalization,
             "instance": InstanceNormalization}
    bottlenecks = {"average_pooling": kl.GlobalAveragePooling2D(),
                   "dense": kl.Dense(size),
                   "max_pooling": kl.GlobalMaxPooling2D()}
    var_x = inputs
    if norm:
        var_x = norms[norm]()(var_x)
    if bottleneck == "dense" and var_x.ndim > 2:  # Flatten non-1D inputs for dense
        var_x = kl.Flatten()(var_x)
    if bottleneck != "flatten":
        var_x = bottlenecks[bottleneck](var_x)
    if var_x.ndim > 2:
        # Flatten prior to fc layers
        var_x = kl.Flatten()(var_x)
    return var_x


def _get_upscale_layer(method: T.Literal["resize_images", "subpixel", "upscale_dny",
                                         "upscale_fast", "upscale_hybrid", "upsample2d"],
                       filters: int,
                       activation: str | None = None,
                       upsamples: int | None = None,
                       interpolation: str | None = None) -> keras.layers.Layer:
    """ Obtain an instance of the requested upscale method.

    Parameters
    ----------
    method: str
        The user selected upscale method to use. One of `"resize_images"`, `"subpixel"`,
        `"upscale_dny"`, `"upscale_fast"`, `"upscale_hybrid"`, `"upsample2d"`
    filters: int
        The number of filters to use in the upscale layer
    activation: str, optional
        The activation function to use in the upscale layer. ``None`` to use no activation.
        Default: ``None``
    upsamples: int, optional
        Only used for UpSampling2D. If provided, then this is passed to the layer as the ``size``
        parameter. Default: ``None``
    interpolation: str, optional
        Only used for UpSampling2D. If provided, then this is passed to the layer as the
        ``interpolation`` parameter. Default: ``None``

    Returns
    -------
    :class:`keras.layers.Layer`
        The selected configured upscale layer
    """
    if method == "upsample2d":
        kwargs: dict[str, str | int] = {}
        if upsamples:
            kwargs["size"] = upsamples
        if interpolation:
            kwargs["interpolation"] = interpolation
        return kl.UpSampling2D(**kwargs)
    if method == "subpixel":
        return UpscaleBlock(filters, activation=activation)
    if method == "upscale_fast":
        return Upscale2xBlock(filters, activation=activation, fast=True)
    if method == "upscale_hybrid":
        return Upscale2xBlock(filters, activation=activation, fast=False)
    if method == "upscale_dny":
        return UpscaleDNYBlock(filters, activation=activation)
    return UpscaleResizeImagesBlock(filters, activation=activation)


def _get_curve(start_y: int,
               end_y: int,
               num_points: int,
               scale: float,
               mode: T.Literal["full", "cap_max", "cap_min"] = "full") -> list[int]:
    """ Obtain a curve.

    For the given start and end y values, return the y co-ordinates of a curve for the given
    number of points. The points are rounded down to the nearest 8.

    Parameters
    ----------
    start_y: int
        The y co-ordinate for the starting point of the curve
    end_y: int
        The y co-ordinate for the end point of the curve
    num_points: int
        The number of data points to plot on the x-axis
    scale: float
        The scale of the curve (from -.99 to 0.99)
    slope_mode: str, optional
        The method to generate the curve. One of `"full"`, `"cap_max"` or `"cap_min"`. `"full"`
        mode generates a curve from the `"start_y"` to the `"end_y"` values. `"cap_max"` pads the
        earlier points with the `"start_y"` value before filling out the remaining points at a
        fixed divider to the `"end_y"` value. `"cap_min"` starts at the `"start_y" filling points
        at a fixed divider until the `"end_y"` value is reached and pads the remaining points with
        the `"end_y"` value. Default: `"full"`

    Returns
    -------
    list
        List of ints of points for the given curve
     """
    scale = min(.99, max(-.99, scale))
    logger.debug("Obtaining curve: (start_y: %s, end_y: %s, num_points: %s, scale: %s, mode: %s)",
                 start_y, end_y, num_points, scale, mode)
    if mode == "full":
        x_axis = np.linspace(0., 1., num=num_points)
        y_axis = (x_axis - x_axis * scale) / (scale - abs(x_axis) * 2 * scale + 1)
        y_axis = y_axis * (end_y - start_y) + start_y
        retval = [int((y // 8) * 8) for y in y_axis]
    else:
        y_axis = [start_y]
        scale = 1. - abs(scale)
        for _ in range(num_points - 1):
            current_value = max(end_y, int(((y_axis[-1] * scale) // 8) * 8))
            y_axis.append(current_value)
            if current_value == end_y:
                break
        pad = [start_y if mode == "cap_max" else end_y for _ in range(num_points - len(y_axis))]
        retval = pad + y_axis if mode == "cap_max" else y_axis + pad
    logger.debug("Returning curve: %s", retval)
    return retval


def _scale_dim(target_resolution: int, original_dim: int) -> int:
    """ Scale a given `original_dim` so that it is a factor of the target resolution.

    Parameters
    ----------
    target_resolution: int
        The output resolution that is being targetted
    original_dim: int
        The dimension that needs to be checked for compatibility for upscaling to the
        target resolution

    Returns
    -------
    int
        The highest dimension below or equal to `original_dim` that is a factor of the
    target resolution.
    """
    new_dim = target_resolution
    while new_dim > original_dim:
        next_dim = new_dim / 2
        if not next_dim.is_integer():
            break
        new_dim = int(next_dim)
    logger.debug("target_resolution: %s, original_dim: %s, new_dim: %s",
                 target_resolution, original_dim, new_dim)
    return new_dim


class Encoder():
    """ Encoder. Uses one of pre-existing Keras/Faceswap models or custom encoder.

    Parameters
    ----------
    input_shape: tuple
        The shape tuple for the input tensor
    """
    def __init__(self, input_shape: tuple[int, int, int]) -> None:
        logger.debug(parse_class_init(locals()))
        self.input_shape = input_shape
        self._input_shape = input_shape

    @property
    def _model_kwargs(self) -> dict[str, dict[str, float | int | bool]]:
        """ dict: Configuration option for architecture mapped to optional kwargs. """
        return {"mobilenet": {"alpha": cfg.mobilenet_width(),
                              "depth_multiplier": cfg.mobilenet_depth(),
                              "dropout": cfg.mobilenet_dropout()},
                "mobilenet_v2": {"alpha": cfg.mobilenet_width()},
                "mobilenet_v3": {"alpha": cfg.mobilenet_width(),
                                 "minimalist": cfg.mobilenet_minimalistic(),
                                 "include_preprocessing": False}}

    @property
    def _selected_model(self) -> tuple[_EncoderInfo, dict]:
        """ tuple(dict, :class:`_EncoderInfo`): The selected encoder model and it's associated
        keyword arguments """
        arch = cfg.enc_architecture()
        model = _MODEL_MAPPING[arch]
        kwargs = self._model_kwargs.get(arch, {})
        if arch.startswith("efficientnet_v2"):
            kwargs["include_preprocessing"] = False
        return model, kwargs

    def __call__(self) -> keras.models.Model:
        """ Create the Phaze-A Encoder Model.

        Returns
        -------
        :class:`keras.models.Model`
            The selected Encoder Model
        """
        input_ = T.cast("KerasTensor", kl.Input(shape=self._input_shape))
        var_x = input_

        scaling = self._selected_model[0].scaling

        if scaling:
            #  Some models expect different scaling.
            logger.debug("Scaling to %s for '%s'", scaling, cfg.enc_architecture())
            if scaling == (0, 255):
                # models expecting inputs from 0 to 255.
                var_x = var_x * 255.
            if scaling == (-1, 1):
                # models expecting inputs from -1 to 1.
                var_x = var_x * 2.
                var_x = var_x - 1.0

        var_x = self._get_encoder_model()(var_x)

        if cfg.bottleneck_in_encoder():
            var_x = _bottleneck(var_x,
                                cfg.bottleneck_type(),
                                cfg.bottleneck_size(),
                                cfg.bottleneck_norm())

        return keras.models.Model(input_, var_x, name="encoder")

    def _get_encoder_model(self) -> keras.models.Model:
        """ Return the model defined by the selected architecture.

        Returns
        -------
        :class:`keras.Model`
            The selected keras model for the chosen encoder architecture
        """
        model, kwargs = self._selected_model
        if model.keras_name and cfg.enc_architecture().startswith("clipv_"):
            assert model.keras_name in T.get_args(TypeModelsViT)
            kwargs["input_shape"] = self._input_shape
            kwargs["load_weights"] = cfg.enc_load_weights()
            retval = ViT(T.cast(TypeModelsViT, model.keras_name),
                         input_size=self._input_shape[0],
                         load_weights=cfg.enc_load_weights())()
        elif model.keras_name:
            kwargs["input_shape"] = self._input_shape
            kwargs["include_top"] = False
            kwargs["weights"] = "imagenet" if cfg.enc_load_weights() else None
            retval = getattr(kapp, model.keras_name)(**kwargs)
        else:
            retval = _EncoderFaceswap()
        return retval


class _EncoderFaceswap():
    """ A configurable standard Faceswap encoder based off Original model. """
    def __init__(self) -> None:
        logger.debug(parse_class_init(locals()))
        self._type = cfg.enc_architecture()
        self._depth = getattr(cfg, f"{self._type}_depth")()
        self._min_filters = cfg.fs_original_min_filters()
        self._max_filters = cfg.fs_original_max_filters()
        self._is_alt = cfg.fs_original_use_alt()
        self._relu_alpha = 0.2 if self._is_alt else 0.1
        self._kernel_size = 3 if self._is_alt else 5
        self._strides = 1 if self._is_alt else 2

    def __call__(self, inputs: KerasTensor) -> KerasTensor:
        """ Call the original Faceswap Encoder

        Parameters
        ----------
        inputs: :class:`keras.KerasTensor`
            The input tensor to the Faceswap Encoder

        Returns
        -------
        :class:`keras.KerasTensor`
            The output tensor from the Faceswap Encoder
        """
        var_x = inputs
        filters = cfg.fs_original_min_filters()

        if self._is_alt:
            var_x = Conv2DBlock(filters,
                                kernel_size=1,
                                strides=self._strides,
                                relu_alpha=self._relu_alpha)(var_x)

        for i in range(self._depth):
            name = f"fs_{'dny_' if self._is_alt else ''}enc"
            var_x = Conv2DBlock(filters,
                                kernel_size=self._kernel_size,
                                strides=self._strides,
                                relu_alpha=self._relu_alpha,
                                name=f"{name}_convblk_{i}")(var_x)
            filters = min(cfg.fs_original_max_filters(), filters * 2)
            if self._is_alt and i == self._depth - 1:
                var_x = Conv2DBlock(filters,
                                    kernel_size=4,
                                    strides=self._strides,
                                    padding="valid",
                                    relu_alpha=self._relu_alpha,
                                    name=f"{name}_convblk_{i}_1")(var_x)
            elif self._is_alt:
                var_x = Conv2DBlock(filters,
                                    kernel_size=self._kernel_size,
                                    strides=self._strides,
                                    relu_alpha=self._relu_alpha,
                                    name=f"{name}_convblk_{i}_1")(var_x)
                var_x = kl.MaxPooling2D(2, name=f"{name}_pool_{i}")(var_x)
        return var_x


class FullyConnected():
    """ Intermediate Fully Connected layers for Phaze-A Model.

    Parameters
    ----------
    side: ["a", "b", "both", "gblock", "shared"]
        The side of the model that the fully connected layers belong to. Used for naming
    input_shape: tuple
        The input shape for the fully connected layers
    """
    def __init__(self,
                 side: T.Literal["a", "b", "both", "gblock", "shared"],
                 input_shape: tuple) -> None:
        logger.debug(parse_class_init(locals()))
        self._side = side
        self._input_shape = input_shape
        self._final_dims = cfg.fc_dimensions() * (cfg.fc_upsamples() + 1)
        self._prefix = "fc_gblock" if self._side == "gblock" else "fc"

        logger.debug("Initialized: %s (side: %s, min_nodes: %s, max_nodes: %s)",
                     self.__class__.__name__, self._side, self._min_nodes, self._max_nodes)

    @property
    def _min_nodes(self) -> int:
        """ int: The number of nodes for the first Dense. For non g-block layers this will be the
        given minimum filters multiplied by the dimensions squared. For g-block layers, this is the
        given value """
        if self._side == "gblock":
            return cfg.fc_gblock_min_nodes()
        retval = self._scale_filters(cfg.fc_min_filters())
        retval = int(retval * cfg.fc_dimensions() ** 2)
        return retval

    @property
    def _max_nodes(self) -> int:
        """ int: The number of nodes for the final Dense. For non g-block layers this will be the
        given maximum filters multiplied by the dimensions squared. This number will be scaled down
        if the final shape can not be mapped to the requested output size.

        For g-block layers, this is the given config value.
        """
        if self._side == "gblock":
            return cfg.fc_gblock_max_nodes()
        retval = self._scale_filters(cfg.fc_max_filters())
        retval = int(retval * cfg.fc_dimensions() ** 2)
        return retval

    def _scale_filters(self, original_filters: int) -> int:
        """ Scale the filters to be compatible with the model's selected output size.

        Parameters
        ----------
        original_filters: int
            The original user selected number of filters

        Returns
        -------
        int
            The number of filters scaled down for output size
        """
        scaled_dim = _scale_dim(cfg.output_size(), self._final_dims)
        if scaled_dim == self._final_dims:
            logger.debug("filters don't require scaling. Returning: %s", original_filters)
            return original_filters

        flat = self._final_dims ** 2 * original_filters
        modifier = self._final_dims ** 2 * scaled_dim ** 2
        retval = int((flat // modifier) * modifier)
        retval = int(retval / self._final_dims ** 2)
        logger.debug("original_filters: %s, scaled_filters: %s", original_filters, retval)
        return retval

    def _do_upsampling(self, inputs: KerasTensor) -> KerasTensor:
        """ Perform the upsampling at the end of the fully connected layers.

        Parameters
        ----------
        inputs: :class:`keras.KerasTensor`
            The input to the upsample layers

        Returns
        -------
        :class:`keras.KerasTensor`
            The output from the upsample layers
        """
        upsample_filts = self._scale_filters(cfg.fc_upsample_filters())
        upsampler = T.cast(T.Literal["resize_images", "subpixel", "upscale_dny", "upscale_fast",
                                     "upscale_hybrid", "upsample2d"],
                           cfg.fc_upsampler().lower())
        num_upsamples = cfg.fc_upsamples()
        var_x = inputs
        if upsampler == "upsample2d" and num_upsamples > 1:
            upscaler = _get_upscale_layer(upsampler,
                                          upsample_filts,  # Not used but required
                                          upsamples=2 ** num_upsamples,
                                          interpolation="bilinear")
            var_x = upscaler(var_x)
        else:
            for _ in range(num_upsamples):
                upscaler = _get_upscale_layer(upsampler,
                                              upsample_filts,
                                              activation="leakyrelu")
                var_x = upscaler(var_x)
        if upsampler == "upsample2d":
            var_x = kl.LeakyReLU(negative_slope=0.1)(var_x)
        return var_x

    def __call__(self) -> keras.models.Model:
        """ Call the intermediate layer.

        Returns
        -------
        :class:`keras.models.Model`
            The Fully connected model
        """
        input_ = kl.Input(shape=self._input_shape)
        var_x = T.cast("KerasTensor", input_)

        node_curve = _get_curve(self._min_nodes,
                                self._max_nodes,
                                getattr(cfg, f"{self._prefix}_depth")(),
                                getattr(cfg, f"{self._prefix}_filter_slope")())

        if not cfg.bottleneck_in_encoder():
            var_x = _bottleneck(var_x,
                                cfg.bottleneck_type(),
                                cfg.bottleneck_size(),
                                cfg.bottleneck_norm())

        dropout = getattr(cfg, f"{self._prefix}_dropout")()
        for idx, nodes in enumerate(node_curve):
            var_x = kl.Dropout(dropout, name=f"{dropout}_{idx + 1}")(var_x)
            var_x = kl.Dense(nodes)(var_x)

        if self._side != "gblock":
            dim = cfg.fc_dimensions()
            var_x = kl.Reshape((dim, dim, int(self._max_nodes / (dim ** 2))))(var_x)
            var_x = self._do_upsampling(var_x)

            num_upscales = cfg.dec_upscales_in_fc()
            if num_upscales:
                var_x = UpscaleBlocks(self._side,
                                      layer_indicies=(0, num_upscales))(var_x)

        return keras.models.Model(input_, var_x, name=f"fc_{self._side}")


class UpscaleBlocks():
    """ Obtain a block of upscalers.

    This class exists outside of the :class:`Decoder` model, as it is possible to place some of
    the upscalers at the end of the Fully Connected Layers, so the upscale chain needs to be able
    to be calculated by both the Fully Connected Layers and by the Decoder if required.

    For this reason, the Upscale Filter list is created as a class attribute of the
    :class:`UpscaleBlocks` layers for reference by either the Decoder or Fully Connected models

    Parameters
    ----------
    side: ["a", "b", "both", "shared"]
        The side of the model that the Decoder belongs to. Used for naming
    layer_indices: tuple, optional
        The tuple indicies indicating the starting layer index and the ending layer index to
        generate upscales for. Used for when splitting upscales between the Fully Connected Layers
        and the Decoder. ``None`` will generate the full Upscale chain. An end index of -1 will
        generate the layers from the starting index to the final upscale. Default: ``None``
    """
    _filters: list[int] = []

    def __init__(self,
                 side: T.Literal["a", "b", "both", "shared"],
                 layer_indicies: tuple[int, int] | None = None) -> None:
        logger.debug(parse_class_init(locals()))
        self._side = side
        self._is_dny = cfg.dec_upscale_method().lower() == "upscale_dny"
        self._layer_indicies = layer_indicies
        logger.debug("Initialized: %s", self.__class__.__name__,)

    def _reshape_for_output(self, inputs: KerasTensor) -> KerasTensor:
        """ Reshape the input for arbitrary output sizes.

        The number of filters in the input will have been scaled to the model output size allowing
        us to scale the dimensions to the requested output size.

        Parameters
        ----------
        inputs: :class:`keras.KerasTensor`
            The tensor that is to be reshaped

        Returns
        -------
        :class:`keras.KerasTensor`
            The tensor shaped correctly to upscale to output size
        """
        var_x = inputs
        old_dim = inputs.shape[1]
        new_dim = _scale_dim(cfg.output_size(), old_dim)
        if new_dim != old_dim:
            old_shape = inputs.shape[1:]
            new_shape = (new_dim, new_dim, np.prod(old_shape) // new_dim ** 2)
            logger.debug("Reshaping tensor from %s to %s for output size %s",
                         inputs.shape[1:], new_shape, cfg.output_size())
            var_x = kl.Reshape(new_shape)(var_x)
        return var_x

    def _upscale_block(self,
                       inputs: KerasTensor,
                       filters: int,
                       skip_residual: bool = False,
                       is_mask: bool = False) -> KerasTensor:
        """ Upscale block for Phaze-A Decoder.

        Uses requested upscale method, adds requested regularization and activation function.

        Parameters
        ----------
        inputs: :class:`keras.KerasTensor`
            The input tensor for the upscale block
        filters: int
            The number of filters to use for the upscale
        skip_residual: bool, optional
            ``True`` if a residual block should not be placed in the upscale block, otherwise
            ``False``. Default ``False``
        is_mask: bool, optional
            ``True`` if the input is a mask. ``False`` if the input is a face. Default: ``False``

        Returns
        -------
        :class:`keras.KerasTensor`
            The output tensor from the upscale block
        """
        upscaler = _get_upscale_layer(T.cast(T.Literal["resize_images", "subpixel", "upscale_dny",
                                                       "upscale_fast", "upscale_hybrid",
                                                       "upsample2d"],
                                             cfg.dec_upscale_method()),
                                      filters,
                                      activation="leakyrelu",
                                      upsamples=2,
                                      interpolation="bilinear")

        var_x = upscaler(inputs)
        if not is_mask and cfg.dec_gaussian():
            var_x = kl.GaussianNoise(1.0)(var_x)
        if not is_mask and cfg.dec_res_blocks() and not skip_residual:
            var_x = self._normalization(var_x)
            var_x = kl.LeakyReLU(negative_slope=0.2)(var_x)
            for _ in range(cfg.dec_res_blocks()):
                var_x = ResidualBlock(filters)(var_x)
        else:
            var_x = self._normalization(var_x)
            if not self._is_dny:
                var_x = kl.LeakyReLU(negative_slope=0.1)(var_x)
        return var_x

    def _normalization(self, inputs: KerasTensor) -> KerasTensor:
        """ Add a normalization layer if requested.

        Parameters
        ----------
        inputs: :class:`keras.KerasTensor`
            The input tensor to apply normalization to.

        Returns
        --------
        :class:`keras.KerasTensor`
            The tensor with any normalization applied
        """
        dec_norm: str | None = cfg.dec_norm()
        dec_norm = None if dec_norm == "none" else dec_norm
        if not dec_norm:
            return inputs
        norms = {"batch": kl.BatchNormalization,
                 "group": GroupNormalization,
                 "instance": InstanceNormalization,
                 "layer": kl.LayerNormalization,
                 "rms": RMSNormalization}
        return norms[dec_norm]()(inputs)

    def _dny_entry(self, inputs: KerasTensor) -> KerasTensor:
        """ Entry convolutions for using the upscale_dny method.

        Parameters
        ----------
        inputs: :class:`keras.KerasTensor`
            The inputs to the dny entry block

        Returns
        -------
        :class:`keras.KerasTensor`
            The output from the dny entry block
        """
        var_x = Conv2DBlock(cfg.dec_max_filters(),
                            kernel_size=4,
                            strides=1,
                            padding="same",
                            relu_alpha=0.2)(inputs)
        var_x = Conv2DBlock(cfg.dec_max_filters(),
                            kernel_size=3,
                            strides=1,
                            padding="same",
                            relu_alpha=0.2)(var_x)
        return var_x

    def __call__(self, inputs: KerasTensor | list[KerasTensor]) -> KerasTensor | list[KerasTensor]:
        """ Upscale Network.

        Parameters
        inputs: :class:`keras.KerasTensor` | list[:class:`keras.KerasTensor`]
            Input tensor(s) to upscale block. This will be a single tensor if learn mask is not
            selected or if this is the first call to the upscale blocks. If learn mask is selected
            and this is not the first call to upscale blocks, then this will be a list of the face
            and mask tensors.

        Returns
        -------
         :class:`keras.KerasTensor` | list[:class:`keras.KerasTensor`]
            The output of encoder blocks. Either a single tensor (if learn mask is not enabled) or
            list of tensors (if learn mask is enabled)
        """
        start_idx, end_idx = (0, None) if self._layer_indicies is None else self._layer_indicies
        end_idx = None if end_idx == -1 else end_idx

        var_x: KerasTensor
        var_y: KerasTensor
        if cfg_loss.learn_mask() and start_idx == 0:
            # Mask needs to be created
            var_x = inputs
            var_y = inputs
        elif cfg_loss.learn_mask():
            # Mask has already been created and is an input to upscale blocks
            var_x, var_y = inputs
        else:
            # No mask required
            var_x = inputs

        if start_idx == 0:
            var_x = self._reshape_for_output(var_x)

            if cfg_loss.learn_mask():
                var_y = self._reshape_for_output(var_y)

            if self._is_dny:
                var_x = self._dny_entry(var_x)
            if self._is_dny and cfg_loss.learn_mask():
                var_y = self._dny_entry(var_y)

        # De-convolve
        if not self._filters:
            upscales = int(np.log2(cfg.output_size() / var_x.shape[1]))
            self._filters.extend(_get_curve(cfg.dec_max_filters(),
                                            cfg.dec_min_filters(),
                                            upscales,
                                            cfg.dec_filter_slope(),
                                            mode=T.cast(T.Literal["full", "cap_min", "cap_max"],
                                                        cfg.dec_slope_mode())))
            logger.debug("Generated class filters: %s", self._filters)

        filters = self._filters[start_idx: end_idx]

        for idx, filts in enumerate(filters):
            skip_res = idx == len(filters) - 1 and cfg.dec_skip_last_residual()
            var_x = self._upscale_block(var_x, filts, skip_residual=skip_res)
            if cfg_loss.learn_mask():
                var_y = self._upscale_block(var_y, filts, is_mask=True)
        retval = [var_x, var_y] if cfg_loss.learn_mask() else var_x
        return retval


class GBlock():
    """ G-Block model, borrowing from Adain StyleGAN.

    Parameters
    ----------
    side: ["a", "b", "both"]
        The side of the model that the fully connected layers belong to. Used for naming
    input_shapes: list or tuple
        The shape tuples for the input to the G-Block. The first item is the input from each side's
        fully connected model, the second item is the input shape from the combined fully connected
        model.
    """
    def __init__(self, side: T.Literal["a", "b", "both"], input_shapes: list | tuple) -> None:
        logger.debug(parse_class_init(locals()))
        self._side = side
        self._inputs = [kl.Input(shape=shape) for shape in input_shapes]
        self._dense_nodes = 512
        self._dense_recursions = 3
        logger.debug("Initialized: %s", self.__class__.__name__)

    @classmethod
    def _g_block(cls,
                 inputs: KerasTensor,
                 style: KerasTensor,
                 filters: int,
                 recursions: int = 2) -> KerasTensor:
        """ G_block adapted from ADAIN StyleGAN.

        Parameters
        ----------
        inputs: :class:`keras.KerasTensor`
            The input tensor to the G-Block model
        style: :class:`keras.KerasTensor`
            The input combined 'style' tensor to the G-Block model
        filters: int
            The number of filters to use for the G-Block Convolutional layers
        recursions: int, optional
            The number of recursive Convolutions to process. Default: `2`

        Returns
        -------
        :class:`keras.KerasTensor`
            The output tensor from the G-Block model
        """
        var_x = inputs
        for i in range(recursions):
            styles = [kl.Reshape([1, 1, filters])(kl.Dense(filters)(style)) for _ in range(2)]
            noise = kl.Conv2D(filters, 1, padding="same")(kl.GaussianNoise(1.0)(var_x))

            if i == recursions - 1:
                var_x = kl.Conv2D(filters, 3, padding="same")(var_x)

            var_x = AdaInstanceNormalization(dtype="float32")([var_x, *styles])
            var_x = kl.Add()([var_x, noise])
            var_x = kl.LeakyReLU(0.2)(var_x)

        return var_x

    def __call__(self) -> keras.models.Model:
        """ G-Block Network.

        Returns
        -------
        :class:`keras.models.Model`
            The G-Block model
        """
        var_x, style = self._inputs
        for i in range(self._dense_recursions):
            style = kl.Dense(self._dense_nodes, kernel_initializer="he_normal")(style)
            if i != self._dense_recursions - 1:  # Don't add leakyReLu to final output
                style = kl.LeakyReLU(0.1)(style)

        # Scale g_block filters to side dense
        g_filts = var_x.shape[-1]
        var_x = Conv2D(g_filts, 3, strides=1, padding="same")(var_x)
        var_x = kl.GaussianNoise(1.0)(var_x)
        var_x = self._g_block(var_x, style, g_filts)
        return keras.models.Model(self._inputs, var_x, name=f"g_block_{self._side}")


class Decoder():
    """ Decoder Network.

    Parameters
    ----------
    side: ["a", "b", "both"]
        The side of the model that the Decoder belongs to. Used for naming
    input_shape: tuple
        The shape tuple for the input to the decoder.
    """
    def __init__(self,
                 side: T.Literal["a", "b", "both"],
                 input_shape: tuple[int, int, int]) -> None:
        logger.debug(parse_class_init(locals()))
        self._side = side
        self._input_shape = input_shape
        logger.debug("Initialized: %s", self.__class__.__name__,)

    def __call__(self) -> keras.models.Model:
        """ Decoder Network.

        Returns
        -------
        :class:`keras.models.Model`
            The Decoder model
        """
        inputs = T.cast("KerasTensor", kl.Input(shape=self._input_shape))

        num_ups_in_fc = cfg.dec_upscales_in_fc()

        if cfg_loss.learn_mask() and num_ups_in_fc:
            # Mask has already been created in FC and is an output of that model
            inputs = [inputs, kl.Input(shape=self._input_shape)]

        indicies = None if not num_ups_in_fc else (num_ups_in_fc, -1)
        upscales = UpscaleBlocks(self._side, layer_indicies=indicies)(inputs)

        if cfg_loss.learn_mask():
            var_x, var_y = upscales
        else:
            var_x = upscales

        outputs = [Conv2DOutput(3, cfg.dec_output_kernel(), name="face_out")(var_x)]
        if cfg_loss.learn_mask():
            outputs.append(Conv2DOutput(1,
                                        cfg.dec_output_kernel(),
                                        name="mask_out")(var_y))

        return keras.models.Model(inputs, outputs=outputs, name=f"decoder_{self._side}")
