#!/usr/bin/env python3
""" Phaze-A Model by TorzDF with thanks to BirbFakes and the myriad of testers. """

import numpy as np
import tensorflow as tf

from lib.model.nn_blocks import (
    Conv2D, Conv2DBlock, Conv2DOutput, ResidualBlock, UpscaleBlock, Upscale2xBlock,
    UpscaleResizeImagesBlock)
from lib.model.normalization import (
    AdaInstanceNormalization, GroupNormalization, InstanceNormalization, LayerNormalization,
    RMSNormalization)
from lib.utils import get_backend, FaceswapError

from ._base import KerasModel, ModelBase, logger, _get_all_sub_models

if get_backend() == "amd":
    from keras import applications as kapp, backend as K
    from keras.layers import (
        Add, BatchNormalization, Concatenate, Dense, Dropout, Flatten, GaussianNoise,
        GlobalAveragePooling2D, GlobalMaxPooling2D, Input, LeakyReLU, Reshape, UpSampling2D,
        Conv2D as KConv2D)
    from keras.models import clone_model
else:
    # Ignore linting errors from Tensorflow's thoroughly broken import system
    from tensorflow.keras import applications as kapp, backend as K  # pylint:disable=import-error
    from tensorflow.keras.layers import (  # pylint:disable=import-error,no-name-in-module
        Add, BatchNormalization, Concatenate, Dense, Dropout, Flatten, GaussianNoise,
        GlobalAveragePooling2D, GlobalMaxPooling2D, Input, LeakyReLU, Reshape, UpSampling2D,
        Conv2D as KConv2D)
    from tensorflow.keras.models import clone_model  # noqa pylint:disable=import-error,no-name-in-module


_MODEL_MAPPING = dict(
    densenet121=dict(
        keras_name="DenseNet121", default_size=224),
    densenet169=dict(
        keras_name="DenseNet169", default_size=224),
    densenet201=dict(
        keras_name="DenseNet201", default_size=224),
    efficientnet_b0=dict(
        keras_name="EfficientNetB0", no_amd=True, tf_min=2.3, scaling=(0, 255), default_size=224),
    efficientnet_b1=dict(
        keras_name="EfficientNetB1", no_amd=True, tf_min=2.3, scaling=(0, 255), default_size=240),
    efficientnet_b2=dict(
        keras_name="EfficientNetB2", no_amd=True, tf_min=2.3, scaling=(0, 255), default_size=260),
    efficientnet_b3=dict(
        keras_name="EfficientNetB3", no_amd=True, tf_min=2.3, scaling=(0, 255), default_size=300),
    efficientnet_b4=dict(
        keras_name="EfficientNetB4", no_amd=True, tf_min=2.3, scaling=(0, 255), default_size=380),
    efficientnet_b5=dict(
        keras_name="EfficientNetB5", no_amd=True, tf_min=2.3, scaling=(0, 255), default_size=456),
    efficientnet_b6=dict(
        keras_name="EfficientNetB6", no_amd=True, tf_min=2.3, scaling=(0, 255), default_size=528),
    efficientnet_b7=dict(
        keras_name="EfficientNetB7", no_amd=True, tf_min=2.3, scaling=(0, 255), default_size=600),
    inception_resnet_v2=dict(
        keras_name="InceptionResNetV2", scaling=(-1, 1), min_size=75, default_size=299),
    inception_v3=dict(
        keras_name="InceptionV3", scaling=(-1, 1), min_size=75, default_size=299),
    mobilenet=dict(
        keras_name="MobileNet", scaling=(-1, 1), default_size=224),
    mobilenet_v2=dict(
        keras_name="MobileNetV2", scaling=(-1, 1), default_size=224),
    nasnet_large=dict(
        keras_name="NASNetLarge", scaling=(-1, 1), default_size=331, enforce_for_weights=True),
    nasnet_mobile=dict(
        keras_name="NASNetMobile", scaling=(-1, 1), default_size=224, enforce_for_weights=True),
    resnet50=dict(
        keras_name="ResNet50", scaling=(-1, 1), min_size=32, default_size=224),
    resnet50_v2=dict(
        keras_name="ResNet50V2", no_amd=True, scaling=(-1, 1), default_size=224),
    resnet101=dict(
        keras_name="ResNet101", no_amd=True, scaling=(-1, 1), default_size=224),
    resnet101_v2=dict(
        keras_name="ResNet101V2", no_amd=True, scaling=(-1, 1), default_size=224),
    resnet152=dict(
        keras_name="ResNet152", no_amd=True, scaling=(-1, 1), default_size=224),
    resnet152_v2=dict(
        keras_name="ResNet152V2", no_amd=True, scaling=(-1, 1), default_size=224),
    vgg16=dict(
        keras_name="VGG16", color_order="bgr", scaling=(0, 255), default_size=224),
    vgg19=dict(
        keras_name="VGG19", color_order="bgr", scaling=(0, 255), default_size=224),
    xception=dict(
        keras_name="Xception", scaling=(-1, 1), min_size=71, default_size=299),
    fs_original=dict(
        color_order="bgr", min_size=32, default_size=160))


class Model(ModelBase):
    """ Phaze-A Faceswap Model.

    An highly adaptable and configurable model by torzDF

    Parameters
    ----------
    args: varies
        The default command line arguments passed in from :class:`~scripts.train.Train` or
        :class:`~scripts.train.Convert`
    kwargs: varies
        The default keyword arguments passed in from :class:`~scripts.train.Train` or
        :class:`~scripts.train.Convert`
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.config["output_size"] % 64 != 0:
            raise FaceswapError("Phaze-A output shape must be a multiple of 64")

        self._validate_encoder_architecture()
        self.config["freeze_layers"] = self._select_freeze_layers()

        self.input_shape = self._get_input_shape()
        self.color_order = _MODEL_MAPPING[self.config["enc_architecture"]].get("color_order",
                                                                               "rgb")

    def build(self):
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
        with self._settings.strategy_scope():
            model = self._io._load()  # pylint:disable=protected-access
            model = self._update_dropouts(model)
            self._model = model
            self._compile_model()
            self._output_summary()

    def _update_dropouts(self, model):
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
        dropouts = dict(fc=self.config["fc_dropout"],
                        gblock=self.config["fc_gblock_dropout"])
        logger.debug("Config dropouts: %s", dropouts)
        updated = False
        for mod in _get_all_sub_models(model):
            if not mod.name.startswith("fc_"):
                continue
            key = "gblock" if "gblock" in mod.name else mod.name.split("_")[0]
            rate = dropouts[key]
            log_once = False
            for layer in mod.layers:
                if not isinstance(layer, Dropout):
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
            new_model = clone_model(model)
            new_model.set_weights(model.get_weights())
            del model
            model = new_model
        return model

    def _select_freeze_layers(self):
        """ Process the selected frozen layers and replace the `keras_encoder` option with the
        actual keras model name

        Returns
        -------
        list
            The selected layers for weight freezing
        """
        layers = self.config["freeze_layers"]
        keras_name = _MODEL_MAPPING[self.config["enc_architecture"]].get("keras_name")

        if "keras_encoder" not in self.config["freeze_layers"]:
            retval = layers
        elif keras_name:
            retval = [layer.replace("keras_encoder", keras_name.lower()) for layer in layers]
            logger.debug("Substituting 'keras_encoder' for '%s'", self.config["enc_architecture"])
        else:
            retval = [layer for layer in layers if layer != "keras_encoder"]
            logger.debug("Removing 'keras_encoder' for '%s'", self.config["enc_architecture"])
        return retval

    def _get_input_shape(self):
        """ Obtain the input shape for the model.

        Input shape is calculated from the selected Encoder's input size, scaled to the user
        selected Input Scaling, rounded down to the nearest 16 pixels.

        Returns
        -------
        tuple
            The shape tuple for the input size to the Phaze-A model
        """
        size = _MODEL_MAPPING[self.config["enc_architecture"]]["default_size"]
        min_size = _MODEL_MAPPING[self.config["enc_architecture"]].get("min_size", 32)
        scaling = self.config["enc_scaling"] / 100
        size = int(max(min_size, min(size, ((size * scaling) // 16) * 16)))
        retval = (size, size, 3)
        logger.debug("Encoder input set to: %s", retval)
        return retval

    def _validate_encoder_architecture(self):
        """ Validate that the requested architecture is a valid choice for the running system
        configuration.

        If the selection is not valid, an error is logged and system exits.
        """
        arch = self.config["enc_architecture"].lower()
        model = _MODEL_MAPPING.get(arch)
        if not model:
            raise FaceswapError(f"'{arch}' is not a valid choice for encoder architecture. Choose "
                                f"one of {list(_MODEL_MAPPING.keys())}.")

        if get_backend() == "amd" and model.get("no_amd"):
            valid = [k for k, v in _MODEL_MAPPING.items() if not v.get('no_amd')]
            raise FaceswapError(f"'{arch}' is not compatible with the AMD backend. Choose one of "
                                f"{valid}.")

        tf_ver = float(".".join(tf.__version__.split(".")[:2]))  # pylint:disable=no-member
        tf_min = model.get("tf_min", 2.0)
        if get_backend() != "amd" and tf_ver < tf_min:
            raise FaceswapError(f"{arch}' is not compatible with your version of Tensorflow. The "
                                f"minimum version required is {tf_min} whilst you have version "
                                f"{tf_ver} installed.")

    def build_model(self, inputs):
        """ Create the model's structure.

        Parameters
        ----------
        inputs: list
            A list of input tensors for the model. This will be a list of 2 tensors of
            shape :attr:`input_shape`, the first for side "a", the second for side "b".

        Returns
        -------
        :class:`keras.models.Model`
            The output of this function must be a keras model generated from
            :class:`plugins.train.model._base.KerasModel`. See Keras documentation for the correct
            structure, but note that parameter :attr:`name` is a required rather than an optional
            argument in Faceswap. You should assign this to the attribute ``self.name`` that is
            automatically generated from the plugin's filename.
        """
        # Create sub-Models
        encoders = self._build_encoders(inputs)
        inters = self._build_fully_connected(encoders)
        g_blocks = self._build_g_blocks(inters)
        decoders = self._build_decoders(g_blocks)

        # Create Autoencoder
        outputs = [decoders["a"], decoders["b"]]
        autoencoder = KerasModel(inputs, outputs, name=self.model_name)
        return autoencoder

    def _build_encoders(self, inputs):
        """ Build the encoders for Phaze-A

        Parameters
        ----------
        inputs: list
            A list of input tensors for the model. This will be a list of 2 tensors of
            shape :attr:`input_shape`, the first for side "a", the second for side "b".

        Returns
        -------
        dict
            side as key ('a' or 'b'), encoder for side as value
        """
        encoder = Encoder(self.input_shape, self.config)()
        retval = dict(a=encoder(inputs[0]), b=encoder(inputs[1]))
        logger.debug("Encoders: %s", retval)
        return retval

    def _build_fully_connected(self, inputs):
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
        input_shapes = K.int_shape(inputs["a"])[1:]

        if self.config["split_fc"]:
            fc_a = FullyConnected("a", input_shapes, self.config)()
            inter_a = [fc_a(inputs["a"])]
            inter_b = [FullyConnected("b", input_shapes, self.config)()(inputs["b"])]
        else:
            fc_both = FullyConnected("both", input_shapes, self.config)()
            inter_a = [fc_both(inputs["a"])]
            inter_b = [fc_both(inputs["b"])]

        if self.config["shared_fc"]:
            if self.config["shared_fc"] == "full":
                fc_shared = FullyConnected("shared", input_shapes, self.config)()
            elif self.config["split_fc"]:
                fc_shared = fc_a
            else:
                fc_shared = fc_both
            inter_a = [Concatenate(name="inter_a")([inter_a[0], fc_shared(inputs["a"])])]
            inter_b = [Concatenate(name="inter_b")([inter_b[0], fc_shared(inputs["b"])])]

        if self.config["enable_gblock"]:
            fc_gblock = FullyConnected("gblock", input_shapes, self.config)()
            inter_a.append(fc_gblock(inputs["a"]))
            inter_b.append(fc_gblock(inputs["b"]))

        retval = dict(a=inter_a, b=inter_b)
        logger.debug("Fully Connected: %s", retval)
        return retval

    def _build_g_blocks(self, inputs):
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
        if not self.config["enable_gblock"]:
            logger.debug("No G-Block selected, returning Inters: %s", inputs)
            return inputs

        input_shapes = [K.int_shape(inter)[1:] for inter in inputs["a"]]
        if self.config["split_gblock"]:
            retval = dict(a=GBlock("a", input_shapes, self.config)()(inputs["a"]),
                          b=GBlock("b", input_shapes, self.config)()(inputs["b"]))
        else:
            g_block = GBlock("both", input_shapes, self.config)()
            retval = dict(a=g_block((inputs["a"])), b=g_block((inputs["b"])))

        logger.debug("G-Blocks: %s", retval)
        return retval

    def _build_decoders(self, inputs):
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
        input_shape = K.int_shape(input_)[1:]

        if self.config["split_decoders"]:
            retval = dict(a=Decoder("a", input_shape, self.config)()(inputs["a"]),
                          b=Decoder("b", input_shape, self.config)()(inputs["b"]))
        else:
            decoder = Decoder("both", input_shape, self.config)()
            retval = dict(a=decoder(inputs["a"]), b=decoder(inputs["b"]))

        logger.debug("Decoders: %s", retval)
        return retval


def _bottleneck(inputs, bottleneck, size, normalization):
    """ The bottleneck fully connected layer. Can be called from Encoder or FullyConnected layers.

    Parameters
    ----------
    inputs: tensor
        The input to the bottleneck layer
    bottleneck: str
        The type of layer to use for the bottleneck
    size: int
        The number of nodes for the dense layer (if selected)
    normalization: str
        The normalization method to use prior to the bottleneck layer

    Returns
    -------
    tensor
        The output from the bottleneck
    """
    norms = dict(layer=LayerNormalization,
                 rms=RMSNormalization,
                 instance=InstanceNormalization)
    bottlenecks = dict(average_pooling=GlobalAveragePooling2D(),
                       dense=Dense(size),
                       max_pooling=GlobalMaxPooling2D())
    var_x = inputs
    if normalization:
        var_x = norms[normalization]()(var_x)
    if bottleneck == "dense" and len(K.int_shape(var_x)[1:]) > 1:
        # Flatten non-1D inputs for dense bottleneck
        var_x = Flatten()(var_x)
    var_x = bottlenecks[bottleneck](var_x)
    if len(K.int_shape(var_x)[1:]) > 1:
        # Flatten prior to fc layers
        var_x = Flatten()(var_x)
    return var_x


def _get_upscale_layer(method, filters, activation=None):
    """ Obtain an instance of the requested upscale method.

    Parameters
    ----------
    method: str
        The user selected upscale method to use
    filters: int
        The number of filters to use in the upscale layer
    activation: str, optional
        The activation function to use in the upscale layer. ``None`` to use no activation.
        Default: ``None``

    Returns
    -------
    :class:`keras.layers.Layer`
        The selected configured upscale layer
    """
    if method == "upsample2d":
        return UpSampling2D()
    if method == "subpixel":
        return UpscaleBlock(filters, activation=activation)
    if method == "upscale_fast":
        return Upscale2xBlock(filters, activation=activation, fast=True)
    if method == "upscale_hybrid":
        return Upscale2xBlock(filters, activation=activation, fast=False)
    return UpscaleResizeImagesBlock(filters, activation=activation)


def _get_curve(start_y, end_y, num_points, scale):
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

    Returns
    -------
    list
        List of ints of points for the given curve
     """
    scale = min(.99, max(-.99, scale))
    logger.debug("Obtaining curve: (start_y: %s, end_y: %s, num_points: %s, scale: %s)",
                 start_y, end_y, num_points, scale)
    x_axis = np.linspace(0., 1., num=num_points)
    y_axis = (x_axis - x_axis * scale) / (scale - abs(x_axis) * 2 * scale + 1)
    y_axis = y_axis * (end_y - start_y) + start_y
    retval = [int((y // 8) * 8) for y in y_axis]
    logger.debug("Returning curve: %s", retval)
    return retval


def _scale_dim(target_resolution, original_dim):
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


class Encoder():  # pylint:disable=too-few-public-methods
    """ Encoder. Uses one of pre-existing Keras/Faceswap models or custom encoder.

    Parameters
    ----------
    input_shape: tuple
        The shape tuple for the input tensor
    config: dict
        The model configuration options
    """
    def __init__(self, input_shape, config):
        self.input_shape = input_shape
        self._config = config
        self._input_shape = input_shape

    @property
    def _model_kwargs(self):
        """ dict: Configuration option for architecture mapped to optional kwargs. """
        return dict(mobilenet=dict(alpha=self._config["mobilenet_width"],
                                   depth_multiplier=self._config["mobilenet_depth"],
                                   dropout=self._config["mobilenet_dropout"]),
                    mobilenet_v2=dict(alpha=self._config["mobilenet_width"]))

    @property
    def _selected_model(self):
        """ dict: The selected encoder model options dictionary """
        arch = self._config["enc_architecture"]
        model = _MODEL_MAPPING.get(arch)
        model["kwargs"] = self._model_kwargs.get(arch, {})
        return model

    @property
    def _model_input_shape(self):
        """ tuple: The required input shape for the encoder model.

        Notes
        -----
        NasNet does not allow custom input sizes when loading pre-trained weights, so we need to
        resize the input for this model
        """
        default_size = self._selected_model.get("default_size")
        if self._config["enc_load_weights"] and self._selected_model.get("enforce_for_weights"):
            retval = (default_size, default_size, 3)
        else:
            retval = self._input_shape
        return retval

    def __call__(self):
        """ Create the Phaze-A Encoder Model.

        Returns
        -------
        :class:`keras.models.Model`
            The selected Encoder Model
        """
        input_ = Input(shape=self._model_input_shape)
        var_x = input_

        if self._input_shape != self._model_input_shape:
            var_x = self._resize_inputs(var_x)

        scaling = self._selected_model.get("scaling")
        if scaling:
            #  Some models expect different scaling.
            logger.debug("Scaling to %s for '%s'", scaling, self._config["enc_architecture"])
            if scaling == (0, 255):
                # models expecting inputs from 0 to 255.
                var_x = var_x * 255.
            if scaling == (-1, 1):
                # models expecting inputs from -1 to 1.
                var_x = var_x * 2.
                var_x = var_x - 1.0

        var_x = self._get_encoder_model()(var_x)

        if self._config["bottleneck_in_encoder"]:
            var_x = _bottleneck(var_x,
                                self._config["bottleneck_type"],
                                self._config["bottleneck_size"],
                                self._config["bottleneck_norm"])

        return KerasModel(input_, var_x, name="encoder")

    def _resize_inputs(self, inputs):
        """ Some models (specifically NasNet) need a specific input size when loading trained
        weights. This is slightly hacky, but arbitrarily resize the input for these instances.

        Parameters
        ----------
        inputs: tensor
            The input tensor to be resized

        Returns
        -------
        tensor
            The resized input tensor
        """
        input_size = self._input_shape[0]
        new_size = self._model_input_shape[0]
        logger.debug("Resizing input for encoder: '%s' from %s to %s due to trained weights usage",
                     self._config["enc_architecture"], input_size, new_size)
        scale = new_size / input_size
        interp = "bilinear" if scale > 1 else "nearest"
        return K.resize_images(size=scale, interpolation=interp)(inputs)

    def _get_encoder_model(self):
        """ Return the model defined by the selected architecture.

        Parameters
        ----------
        input_shape: tuple
            The input shape for the model

        Returns
        -------
        :class:`keras.Model`
            The selected keras model for the chosen encoder architecture
        """
        if self._selected_model.get("keras_name"):
            kwargs = self._selected_model["kwargs"]
            kwargs["input_shape"] = self._model_input_shape
            kwargs["include_top"] = False
            kwargs["weights"] = "imagenet" if self._config["enc_load_weights"] else None
            retval = getattr(kapp, self._selected_model["keras_name"])(**kwargs)
        else:
            retval = _EncoderFaceswap(self._config)
        return retval


class _EncoderFaceswap():  # pylint:disable=too-few-public-methods
    """ A configurable standard Faceswap encoder based off Original model.

    Parameters
    ----------
    config: dict
        The model configuration options
    """
    def __init__(self, config):
        self._config = config
        self._type = self._config["enc_architecture"]
        self._depth = config[f"{self._type}_depth"]
        self._min_filters = config["fs_original_min_filters"]
        self._max_filters = config["fs_original_max_filters"]

    def __call__(self, inputs):
        """ Call the original Faceswap Encoder

        Parameters
        ----------
        inputs: tensor
            The input tensor to the Faceswap Encoder

        Returns
        -------
        tensor
            The output tensor from the Faceswap Encoder
        """
        var_x = inputs
        filters = self._config["fs_original_min_filters"]
        for i in range(self._depth):
            var_x = Conv2DBlock(filters, activation="leakyrelu", name=f"fs_enc_convblk_{i}")(var_x)
            filters = min(self._config["fs_original_max_filters"], filters * 2)
        return var_x


class FullyConnected():  # pylint:disable=too-few-public-methods
    """ Intermediate Fully Connected layers for Phaze-A Model.

    Parameters
    ----------
    side: ["a", "b", "both", "gblock", "shared"]
        The side of the model that the fully connected layers belong to. Used for naming
    input_shape: tuple
        The input shape for the fully connected layers
    config: dict
        The user configuration dictionary
    """
    def __init__(self, side, input_shape, config):
        logger.debug("Initializing: %s (side: %s, input_shape: %s)",
                     self.__class__.__name__, side, input_shape)
        self._side = side
        self._input_shape = input_shape
        self._config = config
        self._final_dims = self._config["fc_dimensions"] * (self._config["fc_upsamples"] + 1)
        self._prefix = "fc_gblock" if self._side == "gblock" else "fc"

        logger.debug("Initialized: %s (side: %s, min_nodes: %s, max_nodes: %s)",
                     self.__class__.__name__, self._side, self._min_nodes, self._max_nodes)

    @property
    def _min_nodes(self):
        """ int: The number of nodes for the first Dense. For non g-block layers this will be the
        given minimum filters multiplied by the dimensions squared. For g-block layers, this is the
        given value """
        if self._side == "gblock":
            return self._config["fc_gblock_min_nodes"]
        retval = self._scale_filters(self._config["fc_min_filters"])
        retval = int(retval * self._config["fc_dimensions"] ** 2)
        return retval

    @property
    def _max_nodes(self):
        """ int: The number of nodes for the final Dense. For non g-block layers this will be the
        given maximum filters multiplied by the dimensions squared. This number will be scaled down
        if the final shape can not be mapped to the requested output size.

        For g-block layers, this is the given config value.
        """
        if self._side == "gblock":
            return self._config["fc_gblock_max_nodes"]
        retval = self._scale_filters(self._config["fc_max_filters"])
        retval = int(retval * self._config["fc_dimensions"] ** 2)
        return retval

    def _scale_filters(self, original_filters):
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
        scaled_dim = _scale_dim(self._config["output_size"], self._final_dims)
        if scaled_dim == self._final_dims:
            logger.debug("filters don't require scaling. Returning: %s", original_filters)
            return original_filters

        flat = self._final_dims ** 2 * original_filters
        modifier = self._final_dims ** 2 * scaled_dim ** 2
        retval = int((flat // modifier) * modifier)
        retval = int(retval / self._final_dims ** 2)
        logger.debug("original_filters: %s, scaled_filters: %s", original_filters, retval)
        return retval

    def __call__(self):
        """ Call the intermediate layer.

        Returns
        -------
        :class:`keras.models.Model`
            The Fully connected model
        """
        input_ = Input(shape=self._input_shape)
        var_x = input_

        node_curve = _get_curve(self._min_nodes,
                                self._max_nodes,
                                self._config[f"{self._prefix}_depth"],
                                self._config[f"{self._prefix}_filter_slope"])

        if not self._config["bottleneck_in_encoder"]:
            var_x = _bottleneck(var_x,
                                self._config["bottleneck_type"],
                                self._config["bottleneck_size"],
                                self._config["bottleneck_norm"])

        dropout = f"{self._prefix}_dropout"
        for idx, nodes in enumerate(node_curve):
            var_x = Dropout(self._config[dropout], name=f"{dropout}_{idx + 1}")(var_x)
            var_x = Dense(nodes)(var_x)

        if self._side != "gblock":
            dim = self._config["fc_dimensions"]
            upsample_filts = self._scale_filters(self._config["fc_upsample_filters"])

            var_x = Reshape((dim, dim, int(self._max_nodes / (dim ** 2))))(var_x)
            for _ in range(self._config["fc_upsamples"]):
                upscaler = _get_upscale_layer(self._config["fc_upsampler"].lower(),
                                              upsample_filts,
                                              activation="leakyrelu")
                var_x = upscaler(var_x)
                if self._config["fc_upsampler"].lower() == "upsample2d":
                    var_x = LeakyReLU(alpha=0.1)(var_x)

        return KerasModel(input_, var_x, name=f"fc_{self._side}")


class GBlock():  # pylint:disable=too-few-public-methods
    """ G-Block model, borrowing from Adain StyleGAN.

    Parameters
    ----------
    side: ["a", "b", "both"]
        The side of the model that the fully connected layers belong to. Used for naming
    input_shapes: list or tuple
        The shape tuples for the input to the decoder. The first item is the input from each side's
        fully connected model, the second item is the input shape from the combined fully connected
        model.
    config: dict
        The user configuration dictionary
    """
    def __init__(self, side, input_shapes, config):
        logger.debug("Initializing: %s (side: %s, input_shapes: %s)",
                     self.__class__.__name__, side, input_shapes)
        self._side = side
        self._config = config
        self._inputs = [Input(shape=shape) for shape in input_shapes]
        self._dense_nodes = 512
        self._dense_recursions = 3
        logger.debug("Initialized: %s", self.__class__.__name__)

    @classmethod
    def _g_block(cls, inputs, style, filters, recursions=2):
        """ G_block adapted from ADAIN StyleGAN.

        Parameters
        ----------
        inputs: tensor
            The input tensor to the G-Block model
        style: tensor
            The input combined 'style' tensor to the G-Block model
        filters: int
            The number of filters to use for the G-Block Convolutional layers
        recursions: int, optional
            The number of recursive Convolutions to process. Default: `2`

        Returns
        -------
        tensor
            The output tensor from the G-Block model
        """
        var_x = inputs
        for i in range(recursions):
            styles = [Reshape([1, 1, filters])(Dense(filters)(style)) for _ in range(2)]
            noise = KConv2D(filters, 1, padding="same")(GaussianNoise(1.0)(var_x))

            if i == recursions - 1:
                var_x = KConv2D(filters, 3, padding="same")(var_x)

            var_x = AdaInstanceNormalization(dtype="float32")([var_x, *styles])
            var_x = Add()([var_x, noise])
            var_x = LeakyReLU(0.2)(var_x)

        return var_x

    def __call__(self):
        """ G-Block Network.

        Returns
        -------
        :class:`keras.models.Model`
            The G-Block model
        """
        var_x, style = self._inputs
        for i in range(self._dense_recursions):
            style = Dense(self._dense_nodes, kernel_initializer="he_normal")(style)
            if i != self._dense_recursions - 1:  # Don't add leakyReLu to final output
                style = LeakyReLU(0.1)(style)

        # Scale g_block filters to side dense
        g_filts = K.int_shape(var_x)[-1]
        var_x = Conv2D(g_filts, 3, strides=1, padding="same")(var_x)
        var_x = GaussianNoise(1.0)(var_x)
        var_x = self._g_block(var_x, style, g_filts)
        return KerasModel(self._inputs, var_x, name=f"g_block_{self._side}")


class Decoder():  # pylint:disable=too-few-public-methods
    """ Decoder Network.

    Parameters
    ----------
    side: ["a", "b", "both"]
        The side of the model that the fully connected layers belong to. Used for naming
    input_shape: tuple
        The shape tuple for the input to the decoder.
    config: dict
        The user configuration dictionary
    """
    def __init__(self, side, input_shape, config):
        logger.debug("Initializing: %s (side: %s, input_shape: %s)",
                     self.__class__.__name__, side, input_shape)
        self._side = side
        self._input_shape = input_shape
        self._config = config
        logger.debug("Initialized: %s", self.__class__.__name__,)

    def _reshape_for_output(self, inputs):
        """ Reshape the input for arbitrary output sizes.

        The number of filters in the input will have been scaled to the model output size allowing
        us to scale the dimensions to the requested output size.

        Parameters
        ----------
        inputs: tensor
            The tensor that is to be reshaped

        Returns
        -------
        tensor
            The tensor shaped correctly to upscale to output size
        """
        var_x = inputs
        old_dim = K.int_shape(inputs)[1]
        new_dim = _scale_dim(self._config["output_size"], old_dim)
        if new_dim != old_dim:
            old_shape = K.int_shape(inputs)[1:]
            new_shape = (new_dim, new_dim, np.prod(old_shape) // new_dim ** 2)
            logger.debug("Reshaping tensor from %s to %s for output size %s",
                         K.int_shape(inputs)[1:], new_shape, self._config["output_size"])
            var_x = Reshape(new_shape)(var_x)
        return var_x

    def _upscale_block(self, inputs, filters, skip_residual=False, is_mask=False):
        """ Upscale block for Phaze-A Decoder.

        Uses requested upscale method, adds requested regularization and activation function.

        Parameters
        ----------
        inputs: tensor
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
        tensor
            The output tensor from the upscale block
        """
        upscaler = _get_upscale_layer(self._config["dec_upscale_method"].lower(), filters)

        var_x = upscaler(inputs)
        if not is_mask and self._config["dec_gaussian"]:
            var_x = GaussianNoise(1.0)(var_x)
        if not is_mask and self._config["dec_res_blocks"] and not skip_residual:
            var_x = self._normalization(var_x)
            var_x = LeakyReLU(alpha=0.2)(var_x)
            for _ in range(self._config["dec_res_blocks"]):
                var_x = ResidualBlock(filters)(var_x)
        else:
            var_x = self._normalization(var_x)
            var_x = LeakyReLU(alpha=0.1)(var_x)
        return var_x

    def _normalization(self, inputs):
        """ Add a normalization layer if requested.

        Parameters
        ----------
        inputs: tensor
            The input tensor to apply normalization to.

        Returns
        --------
        tensor
            The tensor with any normalization applied
        """
        if not self._config["dec_norm"]:
            return inputs
        norms = dict(batch=BatchNormalization,
                     group=GroupNormalization,
                     instance=InstanceNormalization,
                     layer=LayerNormalization,
                     rms=RMSNormalization)
        return norms[self._config["dec_norm"]]()(inputs)

    def __call__(self):
        """ Decoder Network.

        Returns
        -------
        :class:`keras.models.Model`
            The Decoder model
        """
        inputs = Input(shape=self._input_shape)
        var_x = inputs
        var_x = self._reshape_for_output(var_x)

        if self._config["learn_mask"]:
            var_y = inputs
            var_y = self._reshape_for_output(var_y)

        # De-convolve
        upscales = int(np.log2(self._config["output_size"] / K.int_shape(var_x)[1]))
        filters = _get_curve(self._config["dec_max_filters"],
                             self._config["dec_min_filters"],
                             upscales,
                             self._config["dec_filter_slope"])

        for idx, filts in enumerate(filters):
            skip_res = idx == len(filters) - 1 and self._config["dec_skip_last_residual"]
            var_x = self._upscale_block(var_x, filts, skip_residual=skip_res)
            if self._config["learn_mask"]:
                var_y = self._upscale_block(var_y, filts, is_mask=True)

        outputs = [Conv2DOutput(3, self._config["dec_output_kernel"], name="face_out")(var_x)]
        if self._config["learn_mask"]:
            outputs.append(Conv2DOutput(1,
                                        self._config["dec_output_kernel"],
                                        name="mask_out")(var_y))

        return KerasModel(inputs, outputs=outputs, name=f"decoder_{self._side}")
