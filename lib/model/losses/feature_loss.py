#!/usr/bin/env python3
""" Custom Feature Map Loss Functions for faceswap.py """
from __future__ import annotations
from dataclasses import dataclass, field
import logging
import typing as T

# Ignore linting errors from Tensorflow's thoroughly broken import system
import tensorflow as tf
from tensorflow.keras import applications as kapp  # pylint:disable=import-error
from tensorflow.keras.layers import Dropout, Conv2D, Input, Layer, Resizing  # noqa,pylint:disable=no-name-in-module,import-error
from tensorflow.keras.models import Model  # pylint:disable=no-name-in-module,import-error
import tensorflow.keras.backend as K  # pylint:disable=no-name-in-module,import-error

import numpy as np

from lib.model.networks import AlexNet, SqueezeNet
from lib.utils import GetModel

if T.TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


@dataclass
class NetInfo:
    """ Data class for holding information about Trunk and Linear Layer nets.

    Parameters
    ----------
    model_id: int
        The model ID for the model stored in the deepfakes Model repo
    model_name: str
        The filename of the decompressed model/weights file
    net: callable, Optional
        The net definition to load, if any. Default:``None``
    init_kwargs: dict, optional
        Keyword arguments to initialize any :attr:`net`. Default: empty ``dict``
    needs_init: bool, optional
        True if the net needs initializing otherwise False. Default: ``True``
    """
    model_id: int = 0
    model_name: str = ""
    net: Callable | None = None
    init_kwargs: dict[str, T.Any] = field(default_factory=dict)
    needs_init: bool = True
    outputs: list[Layer] = field(default_factory=list)


class _LPIPSTrunkNet():
    """ Trunk neural network loader for LPIPS Loss function.

    Parameters
    ----------
    net_name: str
        The name of the trunk network to load. One of "alex", "squeeze" or "vgg16"
    eval_mode: bool
        ``True`` for evaluation mode, ``False`` for training mode
    load_weights: bool
        ``True`` if pretrained trunk network weights should be loaded, otherwise ``False``
    """
    def __init__(self, net_name: str, eval_mode: bool, load_weights: bool) -> None:
        logger.debug("Initializing: %s (net_name '%s', eval_mode: %s, load_weights: %s)",
                     self.__class__.__name__, net_name, eval_mode, load_weights)
        self._eval_mode = eval_mode
        self._load_weights = load_weights
        self._net_name = net_name
        self._net = self._nets[net_name]
        logger.debug("Initialized: %s ", self.__class__.__name__)

    @property
    def _nets(self) -> dict[str, NetInfo]:
        """ :class:`NetInfo`: The Information about the requested net."""
        return {
            "alex": NetInfo(model_id=15,
                            model_name="alexnet_imagenet_no_top_v1.h5",
                            net=AlexNet,
                            outputs=[f"features.{idx}" for idx in (0, 3, 6, 8, 10)]),
            "squeeze": NetInfo(model_id=16,
                               model_name="squeezenet_imagenet_no_top_v1.h5",
                               net=SqueezeNet,
                               outputs=[f"features.{idx}" for idx in (0, 4, 7, 9, 10, 11, 12)]),
            "vgg16": NetInfo(model_id=17,
                             model_name="vgg16_imagenet_no_top_v1.h5",
                             net=kapp.vgg16.VGG16,
                             init_kwargs={"include_top": False, "weights": None},
                             outputs=[f"block{i + 1}_conv{2 if i < 2 else 3}" for i in range(5)])}

    @classmethod
    def _normalize_output(cls, inputs: tf.Tensor, epsilon: float = 1e-10) -> tf.Tensor:
        """ Normalize the output tensors from the trunk network.

        Parameters
        ----------
        inputs: :class:`tensorflow.Tensor`
            An output tensor from the trunk model
        epsilon: float, optional
            Epsilon to apply to the normalization operation. Default: `1e-10`
        """
        norm_factor = K.sqrt(K.sum(K.square(inputs), axis=-1, keepdims=True))
        return inputs / (norm_factor + epsilon)

    def _process_weights(self, model: Model) -> Model:
        """ Save and lock weights if requested.

        Parameters
        ----------
        model :class:`keras.models.Model`
            The loaded trunk or linear network

        Returns
        -------
        :class:`keras.models.Model`
            The network with weights loaded/not loaded and layers locked/unlocked
        """
        if self._load_weights:
            weights = GetModel(self._net.model_name, self._net.model_id).model_path
            model.load_weights(weights)

        if self._eval_mode:
            model.trainable = False
            for layer in model.layers:
                layer.trainable = False
        return model

    def __call__(self) -> Model:
        """ Load the Trunk net, add normalization to feature outputs, load weights and set
        trainable state.

        Returns
        -------
        :class:`tensorflow.keras.models.Model`
            The trunk net with normalized feature output layers
        """
        if self._net.net is None:
            raise ValueError("No net loaded")

        model = self._net.net(**self._net.init_kwargs)
        model = model if self._net_name == "vgg16" else model()
        out_layers = [self._normalize_output(model.get_layer(name).output)
                      for name in self._net.outputs]
        model = Model(inputs=model.input, outputs=out_layers)
        model = self._process_weights(model)
        return model


class _LPIPSLinearNet(_LPIPSTrunkNet):
    """ The Linear Network to be applied to the difference between the true and predicted outputs
    of the trunk network.

    Parameters
    ----------
    net_name: str
        The name of the trunk network in use. One of "alex", "squeeze" or "vgg16"
    eval_mode: bool
        ``True`` for evaluation mode, ``False`` for training mode
    load_weights: bool
        ``True`` if pretrained linear network weights should be loaded, otherwise ``False``
    trunk_net: :class:`keras.models.Model`
        The trunk net to place the linear layer on.
    use_dropout: bool
        ``True`` if a dropout layer should be used in the Linear network otherwise ``False``
    """
    def __init__(self,
                 net_name: str,
                 eval_mode: bool,
                 load_weights: bool,
                 trunk_net: Model,
                 use_dropout: bool) -> None:
        logger.debug(
            "Initializing: %s (trunk_net: %s, use_dropout: %s)", self.__class__.__name__,
            trunk_net, use_dropout)
        super().__init__(net_name=net_name, eval_mode=eval_mode, load_weights=load_weights)

        self._trunk = trunk_net
        self._use_dropout = use_dropout

        logger.debug("Initialized: %s", self.__class__.__name__)

    @property
    def _nets(self) -> dict[str, NetInfo]:
        """ :class:`NetInfo`: The Information about the requested net."""
        return {
            "alex": NetInfo(model_id=18,
                            model_name="alexnet_lpips_v1.h5",),
            "squeeze": NetInfo(model_id=19,
                               model_name="squeezenet_lpips_v1.h5"),
            "vgg16": NetInfo(model_id=20,
                             model_name="vgg16_lpips_v1.h5")}

    def _linear_block(self, net_output_layer: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """ Build a linear block for a trunk network output.

        Parameters
        ----------
        net_output_layer: :class:`tensorflow.Tensor`
            An output from the selected trunk network

        Returns
        -------
        :class:`tensorflow.Tensor`
            The input to the linear block
        :class:`tensorflow.Tensor`
            The output from the linear block
        """
        in_shape = K.int_shape(net_output_layer)[1:]
        input_ = Input(in_shape)
        var_x = Dropout(rate=0.5)(input_) if self._use_dropout else input_
        var_x = Conv2D(1, 1, strides=1, padding="valid", use_bias=False)(var_x)
        return input_, var_x

    def __call__(self) -> Model:
        """ Build the linear network for the given trunk network's outputs. Load in trained weights
        and set the model's trainable parameters.

        Returns
        -------
        :class:`tensorflow.keras.models.Model`
            The compiled Linear Net model
        """
        inputs = []
        outputs = []

        for input_ in self._trunk.outputs:
            in_, out = self._linear_block(input_)
            inputs.append(in_)
            outputs.append(out)

        model = Model(inputs=inputs, outputs=outputs)
        model = self._process_weights(model)
        return model


class LPIPSLoss():
    """ LPIPS Loss Function.

    A perceptual loss function that uses linear outputs from pretrained CNNs feature layers.

    Notes
    -----
    Channels Last implementation. All trunks implemented from the original paper.

    References
    ----------
    https://richzhang.github.io/PerceptualSimilarity/

    Parameters
    ----------
    trunk_network: str
        The name of the trunk network to use. One of "alex", "squeeze" or "vgg16"
    trunk_pretrained: bool, optional
        ``True`` Load the imagenet pretrained weights for the trunk network. ``False`` randomly
        initialize the trunk network. Default: ``True``
    trunk_eval_mode: bool, optional
        ``True`` for running inference on the trunk network (standard mode), ``False`` for training
        the trunk network. Default: ``True``
    linear_pretrained: bool, optional
        ``True`` loads the pretrained weights for the linear network layers. ``False`` randomly
        initializes the layers. Default: ``True``
    linear_eval_mode: bool, optional
        ``True`` for running inference on the linear network (standard mode), ``False`` for
        training the linear network. Default: ``True``
    linear_use_dropout: bool, optional
        ``True`` if a dropout layer should be used in the Linear network otherwise ``False``.
        Default: ``True``
    lpips: bool, optional
        ``True`` to use linear network on top of the trunk network. ``False`` to just average the
        output from the trunk network. Default ``True``
    spatial: bool, optional
        ``True`` output the loss in the spatial domain (i.e. as a grayscale tensor of height and
        width of the input image). ``Bool`` reduce the spatial dimensions for loss calculation.
        Default: ``False``
    normalize: bool, optional
        ``True`` if the input Tensor needs to be normalized from the 0. to 1. range to the -1. to
        1. range. Default: ``True``
    ret_per_layer: bool, optional
        ``True`` to return the loss value per feature output layer otherwise ``False``.
        Default: ``False``
    """
    def __init__(self,  # pylint:disable=too-many-arguments
                 trunk_network: str,
                 trunk_pretrained: bool = True,
                 trunk_eval_mode: bool = True,
                 linear_pretrained: bool = True,
                 linear_eval_mode: bool = True,
                 linear_use_dropout: bool = True,
                 lpips: bool = True,
                 spatial: bool = False,
                 normalize: bool = True,
                 ret_per_layer: bool = False) -> None:
        logger.debug(
            "Initializing: %s (trunk_network '%s', trunk_pretrained: %s, trunk_eval_mode: %s, "
            "linear_pretrained: %s, linear_eval_mode: %s, linear_use_dropout: %s, lpips: %s, "
            "spatial: %s, normalize: %s, ret_per_layer: %s)", self.__class__.__name__,
            trunk_network, trunk_pretrained, trunk_eval_mode, linear_pretrained, linear_eval_mode,
            linear_use_dropout, lpips, spatial, normalize, ret_per_layer)

        self._spatial = spatial
        self._use_lpips = lpips
        self._normalize = normalize
        self._ret_per_layer = ret_per_layer
        self._shift = K.constant(np.array([-.030, -.088, -.188],
                                          dtype="float32")[None, None, None, :])
        self._scale = K.constant(np.array([.458, .448, .450],
                                          dtype="float32")[None, None, None, :])

        # Loss needs to be done as fp32. We could cast at output, but better to update the model
        switch_mixed_precision = tf.keras.mixed_precision.global_policy().name == "mixed_float16"
        if switch_mixed_precision:
            logger.debug("Temporarily disabling mixed precision")
            tf.keras.mixed_precision.set_global_policy("float32")

        self._trunk_net = _LPIPSTrunkNet(trunk_network, trunk_eval_mode, trunk_pretrained)()
        self._linear_net = _LPIPSLinearNet(trunk_network,
                                           linear_eval_mode,
                                           linear_pretrained,
                                           self._trunk_net,
                                           linear_use_dropout)()
        if switch_mixed_precision:
            logger.debug("Re-enabling mixed precision")
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
        logger.debug("Initialized: %s", self.__class__.__name__)

    def _process_diffs(self, inputs: list[tf.Tensor]) -> list[tf.Tensor]:
        """ Perform processing on the Trunk Network outputs.

        If :attr:`use_ldip` is enabled, process the diff values through the linear network,
        otherwise return the diff values summed on the channels axis.

        Parameters
        ----------
        inputs: list
            List of the squared difference of the true and predicted outputs from the trunk network

        Returns
        -------
        list
            List of either the linear network outputs (when using lpips) or summed network outputs
        """
        if self._use_lpips:
            return self._linear_net(inputs)
        return [K.sum(x, axis=-1) for x in inputs]

    def _process_output(self, inputs: tf.Tensor, output_dims: tuple) -> tf.Tensor:
        """ Process an individual output based on whether :attr:`is_spatial` has been selected.

        When spatial output is selected, all outputs are sized to the shape of the original True
        input Tensor. When not selected, the mean across the spatial axes (h, w) are returned

        Parameters
        ----------
        inputs: :class:`tensorflow.Tensor`
            An individual diff output tensor from the linear network or summed output
        output_dims: tuple
            The (height, width) of the original true image

        Returns
        -------
        :class:`tensorflow.Tensor`
            Either the original tensor resized to the true image dimensions, or the mean
            value across the height, width axes.
        """
        if self._spatial:
            return Resizing(*output_dims, interpolation="bilinear")(inputs)
        return K.mean(inputs, axis=(1, 2), keepdims=True)

    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """ Perform the LPIPS Loss Function.

        Parameters
        ----------
        y_true: :class:`tensorflow.Tensor`
            The ground truth batch of images
        y_pred: :class:`tensorflow.Tensor`
            The predicted batch of images

        Returns
        -------
        :class:`tensorflow.Tensor`
            The final  loss value
        """
        if self._normalize:
            y_true = (y_true * 2.0) - 1.0
            y_pred = (y_pred * 2.0) - 1.0

        y_true = (y_true - self._shift) / self._scale
        y_pred = (y_pred - self._shift) / self._scale

        net_true = self._trunk_net(y_true)
        net_pred = self._trunk_net(y_pred)

        diffs = [(out_true - out_pred) ** 2
                 for out_true, out_pred in zip(net_true, net_pred)]

        dims = K.int_shape(y_true)[1:3]
        res = [self._process_output(diff, dims) for diff in self._process_diffs(diffs)]

        axis = 0 if self._spatial else None
        val = K.sum(res, axis=axis)

        retval = (val, res) if self._ret_per_layer else val
        return retval / 10.0   # Reduce by factor of 10 'cos this loss is STRONG
