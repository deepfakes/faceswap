#!/usr/bin/env python3
""" Custom Feature Map Loss Functions for faceswap.py """
from dataclasses import dataclass, field
import logging

from typing import Any, Callable, Dict, Optional, List, Tuple

import plaidml
from keras import applications as kapp
from keras.layers import Dropout, Conv2D, Input, Layer
from keras.models import Model
import keras.backend as K

import numpy as np

from lib.model.nets import AlexNet, SqueezeNet
from lib.utils import GetModel

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
    net: Optional[Callable] = None
    init_kwargs: Dict[str, Any] = field(default_factory=dict)
    needs_init: bool = True
    outputs: List[Layer] = field(default_factory=list)


class _TrunkNormLayer(Layer):
    """ Create a layer for normalizing the output of the trunk model.

    Parameters
    ----------
    epsilon: float, optional
        A small number to add to the normalization. Default=`1e-10`
    """
    def __init__(self, epsilon: float = 1e-10, **kwargs):
        super().__init__(*kwargs)
        self._epsilon = epsilon

    def call(self, inputs: plaidml.tile.Value, **kwargs) -> plaidml.tile.Value:
        """ Call the trunk normalization layer.

        Parameters
        ----------
        inputs: :class:`plaidml.tile.Value`
            Input to the trunk output normalization layer

        Returns
        -------
        :class:`plaidml.tile.Value`
            The output from the layer
        """
        norm_factor = K.sqrt(K.sum(K.square(inputs), axis=-1, keepdims=True))
        return inputs / (norm_factor + self._epsilon)


class _LPIPSTrunkNet():  # pylint:disable=too-few-public-methods
    """ Trunk neural network loader for LPIPS Loss function.

    Parameters
    ----------
    net_name: str
        The name of the trunk network to load. One of "alex", "squeeze" or "vgg16"
    """
    def __init__(self, net_name: str) -> None:
        logger.debug("Initializing: %s (net_name '%s')",
                     self.__class__.__name__, net_name)
        self._net = self._nets[net_name]
        logger.debug("Initialized: %s ", self.__class__.__name__)

    @property
    def _nets(self) -> Dict[str, NetInfo]:
        """ :class:`NetInfo`: The Information about the requested net."""
        return dict(
            alex=NetInfo(model_id=15,
                         model_name="alexnet_imagenet_no_top_v1.h5",
                         net=AlexNet,
                         outputs=[f"features.{idx}" for idx in (0, 3, 6, 8, 10)]),
            squeeze=NetInfo(model_id=16,
                            model_name="squeezenet_imagenet_no_top_v1.h5",
                            net=SqueezeNet,
                            outputs=[f"features.{idx}" for idx in (0, 4, 7, 9, 10, 11, 12)]),
            vgg16=NetInfo(model_id=17,
                          model_name="vgg16_imagenet_no_top_v1.h5",
                          net=kapp.vgg16.VGG16,
                          init_kwargs=dict(include_top=False, weights=None),
                          outputs=[f"block{i + 1}_conv{2 if i < 2 else 3}" for i in range(5)]))

    def _process_weights(self, model: Model) -> Model:
        """ Save and lock weights if requested.

        Parameters
        ----------
        model :class:`keras.models.Model`
            The loaded trunk or linear network

        layers: list, optional
            A list of layer names to explicitly load/freeze. If ``None`` then all model
            layers will be processed

        Returns
        -------
        :class:`keras.models.Model`
            The network with weights loaded/not loaded and layers locked/unlocked
        """
        weights = GetModel(self._net.model_name, self._net.model_id).model_path
        model.load_weights(weights)
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
        model = model if self._net.init_kwargs else model()  # Non vgg need init
        out_layers = [_TrunkNormLayer()(model.get_layer(name).output)
                      for name in self._net.outputs]
        model = Model(inputs=model.input, outputs=out_layers)
        model = self._process_weights(model)
        return model


class _LinearLayer(Layer):
    """ Create a layer for normalizing the output of the trunk model.

    Parameters
    ----------
    use_dropout: bool, optional
        Apply a dropout layer prior to the linear layer. Default: ``False``
    """
    def __init__(self, use_dropout: float = False, **kwargs):
        self._use_dropout = use_dropout
        super().__init__(**kwargs)

    def call(self, inputs: plaidml.tile.Value, **kwargs) -> plaidml.tile.Value:
        """ Call the trunk normalization layer.

        Parameters
        ----------
        inputs: :class:`plaidml.tile.Value`
            Input to the trunk output normalization layer

        Returns
        -------
        :class:`plaidml.tile.Value`
            The output from the layer
        """
        input_ = Input(K.int_shape(inputs)[1:])
        var_x = Dropout(rate=0.5)(input_) if self._use_dropout else input_
        var_x = Conv2D(1, 1, strides=1, padding="valid", use_bias=False)(var_x)
        return var_x


class _LPIPSLinearNet(_LPIPSTrunkNet):  # pylint:disable=too-few-public-methods
    """ The Linear Network to be applied to the difference between the true and predicted outputs
    of the trunk network.

    Parameters
    ----------
    net_name: str
        The name of the trunk network in use. One of "alex", "squeeze" or "vgg16"
    trunk_net: :class:`keras.models.Model`
        The trunk net to place the linear layer on.
    use_dropout: bool
        ``True`` if a dropout layer should be used in the Linear network otherwise ``False``
    """
    def __init__(self,
                 net_name: str,
                 trunk_net: Model,
                 use_dropout: bool) -> None:
        logger.debug(
            "Initializing: %s (trunk_net: %s, use_dropout: %s)", self.__class__.__name__,
            trunk_net, use_dropout)
        super().__init__(net_name=net_name)

        self._trunk = trunk_net
        self._use_dropout = use_dropout

        logger.debug("Initialized: %s", self.__class__.__name__)

    @property
    def _nets(self) -> Dict[str, NetInfo]:
        """ :class:`NetInfo`: The Information about the requested net."""
        return dict(
            alex=NetInfo(model_id=18,
                         model_name="alexnet_lpips_v1.h5",),
            squeeze=NetInfo(model_id=19,
                            model_name="squeezenet_lpips_v1.h5"),
            vgg16=NetInfo(model_id=20,
                          model_name="vgg16_lpips_v1.h5"))

    def _linear_block(self, net_output_layer: plaidml.tile.Value) -> Tuple[plaidml.tile.Value,
                                                                           plaidml.tile.Value]:
        """ Build a linear block for a trunk network output.

        Parameters
        ----------
        net_output_layer: :class:`plaidml.tile.Value`
            An output from the selected trunk network

        Returns
        -------
        :class:`plaidml.tile.Value`
            The input to the linear block
        :class:`plaidml.tile.Value`
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
        for layer in self._trunk.outputs:
            inp, out = self._linear_block(layer)
            inputs.append(inp)
            outputs.append(out)

        linear_model = Model(inputs=inputs, outputs=outputs)
        linear_model = self._process_weights(linear_model)

        return linear_model


class LPIPSLoss():  # pylint:disable=too-few-public-methods
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
    linear_use_dropout: bool, optional
        ``True`` if a dropout layer should be used in the Linear network otherwise ``False``.
        Default: ``True``
    lpips: bool, optional
        ``True`` to use linear network on top of the trunk network. ``False`` to just average the
        output from the trunk network. Default ``True``
    normalize: bool, optional
        ``True`` if the input Tensor needs to be normalized from the 0. to 1. range to the -1. to
        1. range. Default: ``True``
    ret_per_layer: bool, optional
        ``True`` to return the loss value per feature output layer otherwise ``False``.
        Default: ``False``
    """
    def __init__(self,
                 trunk_network: str,
                 linear_use_dropout: bool = True,
                 lpips: bool = False,  # TODO This should be True
                 normalize: bool = True,
                 ret_per_layer: bool = False) -> None:
        logger.debug(
            "Initializing: %s (trunk_network '%s', linear_use_dropout: %s, lpips: %s, "
            "normalize: %s, ret_per_layer: %s)", self.__class__.__name__, trunk_network,
            linear_use_dropout, lpips, normalize, ret_per_layer)

        self._use_lpips = lpips
        self._normalize = normalize
        self._ret_per_layer = ret_per_layer
        self._shift = K.constant(np.array([-.030, -.088, -.188],
                                          dtype="float32")[None, None, None, :])
        self._scale = K.constant(np.array([.458, .448, .450],
                                          dtype="float32")[None, None, None, :])

        self._trunk_net = _LPIPSTrunkNet(trunk_network)()
        self._linear_net = _LPIPSLinearNet(trunk_network, self._trunk_net, linear_use_dropout)()

        logger.debug("Initialized: %s", self.__class__.__name__)

    def _process_diffs(self, inputs: List[plaidml.tile.Value]) -> List[plaidml.tile.Value]:
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
            # TODO Fix. Whilst the linear layer compiles and the weights load, PlaidML will
            # error out as the graph is disconnected.
            # The trunk output can be plugged straight into Linear input, but then weights for
            # linear cannot be loaded, and this input would be incorrect (as linear input should
            # be the diff between y_true and y_pred)
            raise NotImplementedError
            return self._linear_net(inputs)  # pylint:disable=unreachable
        return [K.sum(x, axis=-1) for x in inputs]

    def __call__(self,
                 y_true: plaidml.tile.Value,
                 y_pred: plaidml.tile.Value) -> plaidml.tile.Value:
        """ Perform the LPIPS Loss Function.

        Parameters
        ----------
        y_true: :class:`plaidml.tile.Value`
            The ground truth batch of images
        y_pred: :class:`plaidml.tile.Value`
            The predicted batch of images

        Returns
        -------
        :class:`plaidml.tile.Value`
            The final  loss value
        """
        if self._normalize:
            y_true = (y_true * 2.0) - 1.0
            y_pred = (y_pred * 2.0) - 1.0

        y_true = (y_true - self._shift) / self._scale
        y_pred = (y_pred - self._shift) / self._scale

        net_true = self._trunk_net(y_true)
        net_pred = self._trunk_net(y_pred)

        diffs = [K.pow((out_true - out_pred), 2)
                 for out_true, out_pred in zip(net_true, net_pred)]

        res = [K.mean(diff, axis=(1, 2), keepdims=True) for diff in self._process_diffs(diffs)]

        val = K.sum(K.concatenate(res), axis=None)

        retval = (val, res) if self._ret_per_layer else val
        return retval
