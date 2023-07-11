#!/usr/bin/env python3
""" CLIP: https://github.com/openai/CLIP. This implementation only ports the visual transformer
part of the model.
"""
# TODO Fix Resnet. It is correct until final MHA
from __future__ import annotations
import inspect
import logging
import typing as T
import sys

from dataclasses import dataclass

import tensorflow as tf

from lib.model.layers import QuickGELU
from lib.utils import GetModel

keras = tf.keras
layers = tf.keras.layers
K = tf.keras.backend

logger = logging.getLogger(__name__)

TypeModels = T.Literal["RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-B-16",
                       "ViT-B-32", "ViT-L-14", "ViT-L-14-336px", "FaRL-B-16-16", "FaRL-B-16-64"]


@dataclass
class ViTConfig:
    """ Configuration settings for ViT

    Parameters
    ----------
    embed_dim: int
        Dimensionality of the final shared embedding space
    resolution: int
        Spatial resolution of the input images
    layer_conf: tuple[int, int, int, int] | int
        Number of layers in the visual encoder, or a tuple of layer configurations for a custom
        ResNet visual encoder
    width: int
        Width of the visual encoder layers
    patch: int
        Size of the patches to be extracted from the images. Only used for Visual encoder.
    git_id: int, optional
        The id of the model weights file stored in deepfakes_models repo if they exist. Default: 0
    """
    embed_dim: int
    resolution: int
    layer_conf: int | tuple[int, int, int, int]
    width: int
    patch: int
    git_id: int = 0

    def __post_init__(self):
        """ Validate that patch_size is given correctly """
        assert (isinstance(self.layer_conf, (tuple, list)) and self.patch == 0) or (
            isinstance(self.layer_conf, int) and self.patch > 0)


ModelConfig: dict[TypeModels, ViTConfig] = {  # Each model has a different set of parameters
    "RN50": ViTConfig(
        embed_dim=1024, resolution=224, layer_conf=(3, 4, 6, 3), width=64, patch=0, git_id=21),
    "RN101": ViTConfig(
        embed_dim=512, resolution=224, layer_conf=(3, 4, 23, 3), width=64, patch=0, git_id=22),
    "RN50x4": ViTConfig(
        embed_dim=640, resolution=288, layer_conf=(4, 6, 10, 6), width=80, patch=0, git_id=23),
    "RN50x16": ViTConfig(
        embed_dim=768, resolution=384, layer_conf=(6, 8, 18, 8), width=96, patch=0, git_id=24),
    "RN50x64": ViTConfig(
        embed_dim=1024, resolution=448, layer_conf=(3, 15, 36, 10), width=128, patch=0, git_id=25),
    "ViT-B-16": ViTConfig(
        embed_dim=512, resolution=224, layer_conf=12, width=768, patch=16, git_id=26),
    "ViT-B-32": ViTConfig(
        embed_dim=512, resolution=224, layer_conf=12, width=768, patch=32, git_id=27),
    "ViT-L-14": ViTConfig(
        embed_dim=768, resolution=224, layer_conf=24, width=1024, patch=14, git_id=28),
    "ViT-L-14-336px": ViTConfig(
        embed_dim=768, resolution=336, layer_conf=24, width=1024, patch=14, git_id=29),
    "FaRL-B-16-16": ViTConfig(
        embed_dim=512, resolution=224, layer_conf=12, width=768, patch=16, git_id=30),
    "FaRL-B-16-64": ViTConfig(
        embed_dim=512, resolution=224, layer_conf=12, width=768, patch=16, git_id=31)}


# ################## #
# VISUAL TRANSFORMER #
# ################## #

class Transformer():  # pylint:disable=too-few-public-methods
    """ A class representing a Transformer model with attention mechanism and residual connections.

    Parameters
    ----------
    width: int
        The dimension of the input and output vectors.
    num_layers: int
        The number of layers in the Transformer.
    heads: int
        The number of attention heads.
    attn_mask: tf.Tensor, optional
        The attention mask, by default None.
    name: str, optional
        The name of the Transformer model, by default "transformer".

    Methods
    -------
    __call__() -> Model:
        Calls the Transformer layers.
    """
    _layer_names: dict[str, int] = {}
    """ dict[str, int] for tracking unique layer names"""

    def __init__(self,
                 width: int,
                 num_layers: int,
                 heads: int,
                 attn_mask: tf.Tensor = None,
                 name: str = "transformer") -> None:
        logger.debug("Initializing: %s (width: %s, num_layers: %s, heads: %s, attn_mask: %s, "
                     "name: %s)",
                     self.__class__.__name__, width, num_layers, heads, attn_mask, name)
        self._width = width
        self._num_layers = num_layers
        self._heads = heads
        self._attn_mask = attn_mask
        self._name = name
        logger.debug("Initialized: %s ", self.__class__.__name__)

    @classmethod
    def _get_name(cls, name: str) -> str:
        """ Return unique layer name for requested block.

        As blocks can be used multiple times, auto appends an integer to the end of the requested
        name to keep all block names unique

        Parameters
        ----------
        name: str
            The requested name for the layer

        Returns
        -------
        str
            The unique name for this layer
        """
        cls._layer_names[name] = cls._layer_names.setdefault(name, -1) + 1
        name = f"{name}.{cls._layer_names[name]}"
        logger.debug("Generating block name: %s", name)
        return name

    @classmethod
    def _mlp(cls, inputs: tf.Tensor, key_dim: int, name: str) -> tf.Tensor:
        """" Multilayer Perecptron for Block Ateention

        Parameters
        ----------
        inputs: :class:`tensorflow.Tensor`
            The input to the MLP
        key_dim: int
            key dimension per head for MultiHeadAttention
        name: str
            The name to prefix on the layers

        Returns
        -------
        :class:`tensorflow.Tensor`
            The output from the MLP
        """
        name = f"{name}.mlp"
        var_x = layers.Dense(key_dim * 4, name=f"{name}.c_fc")(inputs)
        var_x = QuickGELU(name=f"{name}.gelu")(var_x)
        var_x = layers.Dense(key_dim, name=f"{name}.c_proj")(var_x)
        return var_x

    def residual_attention_block(self,
                                 inputs: tf.Tensor,
                                 key_dim: int,
                                 num_heads: int,
                                 attn_mask: tf.Tensor,
                                 name: str = "ResidualAttentionBlock") -> tf.Tensor:
        """ Call the residual attention block

        Parameters
        ----------
        inputs: :class:`tf.Tensor`
            The input Tensor
        key_dim: int
            key dimension per head for MultiHeadAttention
        num_heads: int
            Number of heads for MultiHeadAttention
        attn_mask: :class:`tensorflow.Tensor`, optional
            Default: ``None``
        name: str, optional
            The name for the layer. Default: "ResidualAttentionBlock"

        Returns
        -------
        :class:`tf.Tensor`
            The return Tensor
        """
        name = self._get_name(name)

        var_x = layers.LayerNormalization(epsilon=1e-05, name=f"{name}.ln_1")(inputs)
        var_x = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim // num_heads,
            name=f"{name}.attn")(var_x, var_x, var_x, attention_mask=attn_mask)
        var_x = layers.Add()([inputs, var_x])
        var_y = var_x
        var_x = layers.LayerNormalization(epsilon=1e-05, name=f"{name}.ln_2")(var_x)
        var_x = layers.Add()([var_y, self._mlp(var_x, key_dim, name)])
        return var_x

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        """ Call the Transformer layers

        Parameters
        ----------
        inputs: :class:`tf.Tensor`
            The input Tensor

        Returns
        -------
        :class:`tf.Tensor`
            The return Tensor
        """
        logger.debug("Calling %s with input: %s", self.__class__.__name__, inputs.shape)
        var_x = inputs
        for _ in range(self._num_layers):
            var_x = self.residual_attention_block(var_x,
                                                  self._width,
                                                  self._heads,
                                                  self._attn_mask,
                                                  name=f"{self._name}.resblocks")
        return var_x


class EmbeddingLayer(tf.keras.layers.Layer):
    """ Parent class for trainable embedding variables

    Parameters
    ----------
    input_shape: tuple[int, ...]
        The shape of the variable
    scale: int
        Amount to scale the random initialization by
    name: str
        The name of the layer
    dtype: str, optional
        The datatype for the layer. Mixed precision can mess up the embeddings. Default: "float32"
    """
    def __init__(self,
                 input_shape: tuple[int, ...],
                 scale: int,
                 name: str,
                 *args,
                 dtype="float32",
                 **kwargs) -> None:
        super().__init__(name=name, dtype=dtype, *args, **kwargs)
        self._input_shape = input_shape
        self._scale = scale
        self._var: tf.Variable

    def build(self, input_shape: tuple[int, ...]) -> None:
        """ Add the weights

        Parameters
        ----------
        input_shape: tuple[int, ...
            The input shape of the incoming tensor
        """
        self._var = tf.Variable(self._scale * tf.random.normal(self._input_shape,
                                                               dtype=self.dtype),
                                trainable=True,
                                dtype=self.dtype)
        super().build(input_shape)

    def get_config(self) -> dict[str, T.Any]:
        """ Get the config dictionary for the layer

        Returns
        -------
        dict[str, Any]
            The config dictionary for the layer
        """
        retval = super().get_config()
        retval["input_shape"] = self._input_shape
        retval["scale"] = self._scale
        return retval


class ClassEmbedding(EmbeddingLayer):
    """ Trainable Class Embedding layer """
    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        """ Get the Class Embedding layer

        Parameters
        ----------
        inputs: :class:`tensorflow.Tensor`
            Input tensor to the embedding layer

        Returns
        -------
        :class:`tensorflow.Tensor`
            The class embedding layer shaped for the input tensor
        """
        return K.tile(self._var[None, None], [K.shape(inputs)[0], 1, 1])


class PositionalEmbedding(EmbeddingLayer):
    """ Trainable Positional Embedding layer """
    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        """ Get the Positional Embedding layer

        Parameters
        ----------
        inputs: :class:`tensorflow.Tensor`
            Input tensor to the embedding layer

        Returns
        -------
        :class:`tensorflow.Tensor`
            The positional embedding layer shaped for the input tensor
        """
        return K.tile(self._var[None], [K.shape(inputs)[0], 1, 1])


class Projection(EmbeddingLayer):
    """ Trainable Projection Embedding Layer """
    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        """ Get the Projection layer

        Parameters
        ----------
        inputs: :class:`tensorflow.Tensor`
            Input tensor to the embedding layer

        Returns
        -------
        :class:`tensorflow.Tensor`
            The Projection layer expanded to the batch dimension and transposed for matmul
        """
        return K.tile(K.transpose(self._var)[None], [K.shape(inputs)[0], 1, 1])


class VisualTransformer():  # pylint:disable=too-few-public-methods
    """ A class representing a Visual Transformer model for image classification tasks.

    Parameters
    ----------
    input_resolution: int
        The input resolution of the images.
    patch_size: int
        The size of the patches to be extracted from the images.
    width: int
        The dimension of the input and output vectors.
    num_layers: int
        The number of layers in the Transformer.
    heads: int
        The number of attention heads.
    output_dim: int
        The dimension of the output vector.
    name: str, optional
        The name of the Visual Transformer model, Default: "VisualTransformer".

    Methods
    -------
    __call__() -> Model:
        Builds and returns the Visual Transformer model.
    """
    def __init__(self,
                 input_resolution: int,
                 patch_size: int,
                 width: int,
                 num_layers: int,
                 heads: int,
                 output_dim: int,
                 name: str = "VisualTransformer") -> None:
        logger.debug("Initializing: %s (input_resolution: %s, patch_size: %s, width: %s, "
                     "layers: %s, heads: %s, output_dim: %s, name: %s)",
                     self.__class__.__name__, input_resolution, patch_size, width, num_layers,
                     heads, output_dim, name)
        self._input_resolution = input_resolution
        self._patch_size = patch_size
        self._width = width
        self._num_layers = num_layers
        self._heads = heads
        self._output_dim = output_dim
        self._name = name
        logger.debug("Initialized: %s", self.__class__.__name__)

    def __call__(self) -> tf.keras.models.Model:
        """ Builds and returns the Visual Transformer model.

        Returns
        -------
        Model
            The Visual Transformer model.
        """
        inputs = layers.Input([self._input_resolution, self._input_resolution, 3])
        var_x: tf.Tensor = layers.Conv2D(self._width,  # shape = [*, grid, grid, width]
                                         self._patch_size,
                                         strides=self._patch_size,
                                         use_bias=False,
                                         name=f"{self._name}.conv1")(inputs)

        var_x = layers.Reshape((-1, self._width))(var_x)  # shape = [*, grid ** 2, width]

        class_embed = ClassEmbedding((self._width, ),
                                     self._width ** -0.5,
                                     name=f"{self._name}.class_embedding")(var_x)
        var_x = layers.Concatenate(axis=1)([class_embed, var_x])

        pos_embed = PositionalEmbedding(((self._input_resolution // self._patch_size) ** 2 + 1,
                                        self._width),
                                        self._width ** -0.5,
                                        name=f"{self._name}.positional_embedding")(var_x)
        var_x = layers.Add()([var_x, pos_embed])
        var_x = layers.LayerNormalization(epsilon=1e-05, name=f"{self._name}.ln_pre")(var_x)
        var_x = Transformer(self._width,
                            self._num_layers,
                            self._heads,
                            name=f"{self._name}.transformer")(var_x)
        var_x = layers.LayerNormalization(epsilon=1e-05,
                                          name=f"{self._name}.ln_post")(var_x[:, 0, :])
        proj = Projection((self._width, self._output_dim),
                          self._width ** -0.5,
                          name=f"{self._name}.proj")(var_x)
        var_x = layers.Dot(axes=-1)([var_x, proj])
        return keras.models.Model(inputs=inputs, outputs=[var_x], name=self._name)


# ################ #
# MODIEFIED RESNET #
# ################ #
class Bottleneck():  # pylint:disable=too-few-public-methods
    """ A ResNet bottleneck block that performs a sequence of convolutions, batch normalization,
    and ReLU activation operations on an input tensor.

    Parameters
    ----------
    inplanes: int
        The number of input channels.
    planes: int
        The number of output channels.
    stride: int, optional
        The stride of the bottleneck block. Default: 1
    name: str, optional
        The name of the bottleneck block. Default: "bottleneck"
    """
    expansion = 4
    """ int: The factor by which the number of input channels is expanded to get the number of
    output channels."""

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 name: str = "bottleneck") -> None:
        logger.debug("Initializing: %s (inplanes: %s, planes: %s, stride: %s, name: %s)",
                     self.__class__.__name__, inplanes, planes, stride, name)
        self._inplanes = inplanes
        self._planes = planes
        self._stride = stride
        self._name = name
        logger.debug("Initialized: %s", self.__class__.__name__)

    def _downsample(self, inputs: tf.Tensor) -> tf.Tensor:
        """ Perform downsample if required

        Parameters
        ----------
        inputs: :class:`tensorflow.Tensor`
            The input the downsample

        Returns
        -------
        :class:`tensorflow.Tensor`
            The original tensor, if downsizing not required, otherwise the downsized tensor
        """
        if self._stride <= 1 and self._inplanes == self._planes * self.expansion:
            return inputs

        name = f"{self._name}.downsample"
        out = layers.AveragePooling2D(self._stride, name=f"{name}.avgpool")(inputs)
        out = layers.Conv2D(self._planes * self.expansion,
                            1,
                            strides=1,
                            use_bias=False,
                            name=f"{name}.0")(out)
        out = layers.BatchNormalization(name=f"{name}.1", epsilon=1e-5)(out)
        return out

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        """ Performs the forward pass for a Bottleneck block.

        All conv layers have stride 1. an avgpool is performed after the second convolution when
        stride > 1

        Parameters
        ----------
        inputs: :class:`tensorflow.Tensor`
            The input tensor to the Bottleneck block.

        Returns
        -------
        :class:`tensorflow.Tensor`
            The result of the forward pass through the Bottleneck block.
        """
        out = layers.Conv2D(self._planes, 1, use_bias=False, name=f"{self._name}.conv1")(inputs)
        out = layers.BatchNormalization(name=f"{self._name}.bn1", epsilon=1e-5)(out)
        out = layers.ReLU()(out)

        out = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(out)
        out = layers.Conv2D(self._planes, 3, use_bias=False, name=f"{self._name}.conv2")(out)
        out = layers.BatchNormalization(name=f"{self._name}.bn2", epsilon=1e-5)(out)
        out = layers.ReLU()(out)

        if self._stride > 1:
            out = layers.AveragePooling2D(self._stride)(out)

        out = layers.Conv2D(self._planes * self.expansion,
                            1,
                            use_bias=False,
                            name=f"{self._name}.conv3")(out)
        out = layers.BatchNormalization(name=f"{self._name}.bn3", epsilon=1e-5)(out)

        identity = self._downsample(inputs)

        out += identity
        out = layers.ReLU()(out)
        return out


class AttentionPool2d():  # pylint:disable=too-few-public-methods
    """ An Attention Pooling layer that applies a multi-head self-attention mechanism over a
    spatial grid of features.

    Parameters
    ----------
    spatial_dim: int
        The dimensionality of the spatial grid of features.
    embed_dim: int
        The dimensionality of the feature embeddings.
    num_heads: int
        The number of attention heads.
    output_dim: int
        The output dimensionality of the attention layer. If None, it defaults to embed_dim.
    name: str
        The name of the layer.
    """
    def __init__(self,
                 spatial_dim: int,
                 embed_dim: int,
                 num_heads: int,
                 output_dim: int | None = None,
                 name="AttentionPool2d"):
        logger.debug("Initializing: %s (spatial_dim: %s, embed_dim: %s, num_heads: %s, "
                     "output_dim: %s, name: %s)",
                     self.__class__.__name__, spatial_dim, embed_dim, num_heads, output_dim, name)

        self._spatial_dim = spatial_dim
        self._embed_dim = embed_dim
        self._num_heads = num_heads
        self._output_dim = output_dim
        self._name = name
        logger.debug("Initialized: %s", self.__class__.__name__)

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        """Performs the attention pooling operation on the input tensor.

        Parameters
        ----------
        inputs: :class:`tensorflow.Tensor`:
                The input tensor of shape [batch_size, height, width, embed_dim].

        Returns
        -------
        :class:`tensorflow.Tensor`:: The result of the attention pooling operation
        """
        var_x: tf.Tensor
        var_x = layers.Reshape((-1, inputs.shape[-1]))(inputs)  # NHWC -> N(HW)C
        var_x = layers.Concatenate(axis=1)([K.mean(var_x, axis=1,  # N(HW)C -> N(HW+1)C
                                                   keepdims=True), var_x])
        pos_embed = PositionalEmbedding((self._spatial_dim ** 2 + 1, self._embed_dim),  # N(HW+1)C
                                        self._embed_dim ** 0.5,
                                        name=f"{self._name}.positional_embedding")(var_x)
        var_x = layers.Add()([var_x, pos_embed])
        # TODO At this point torch + keras match. They mismatch after MHA
        var_x = layers.MultiHeadAttention(num_heads=self._num_heads,
                                          key_dim=self._embed_dim // self._num_heads,
                                          output_shape=self._output_dim or self._embed_dim,
                                          use_bias=True,
                                          name=f"{self._name}.mha")(var_x[:, :1, ...],
                                                                    var_x,
                                                                    var_x)
        # only return the first element in the sequence
        return var_x[:, 0, ...]


class ModifiedResNet():  # pylint:disable=too-few-public-methods
    """ A ResNet class that is similar to torchvision's but contains the following changes:

    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max
      pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions
      with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool

    Parameters
    ----------
        input_resolution: int
            The input resolution of the model. Default is 224.
        width: int
            The width of the model. Default is 64.
        layer_config: list
            A list containing the number of Bottleneck blocks for each layer.
        output_dim: int
            The output dimension of the model.
        heads: int
            The number of heads for the QKV attention.
        name: str
            The name of the model. Default is "ModifiedResNet".
    """
    def __init__(self,
                 input_resolution: int,
                 width: int,
                 layer_config: tuple[int, int, int, int],
                 output_dim: int,
                 heads: int,
                 name="ModifiedResNet"):
        self._input_resolution = input_resolution
        self._width = width
        self._layer_config = layer_config
        self._heads = heads
        self._output_dim = output_dim
        self._name = name

    def _stem(self, inputs: tf.Tensor) -> tf.Tensor:
        """ Applies the stem operation to the input tensor, which consists of 3 convolutional
            layers with BatchNormalization and ReLU activation, followed by an average pooling
            layer.

        Parameters
        ----------
        inputs: :class:`tensorflow.Tensor`
                The input tensor

        Returns
        -------
        :class:`tensorflow.Tensor`
            The output tensor after applying the stem operation.
        """
        var_x = inputs
        for i in range(1, 4):
            width = self._width if i == 3 else self._width // 2
            strides = 2 if i == 1 else 1
            var_x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=f"conv{i}_padding")(var_x)
            var_x = layers.Conv2D(width,
                                  3,
                                  strides=strides,
                                  use_bias=False,
                                  name=f"conv{i}")(var_x)
            var_x = layers.BatchNormalization(name=f"bn{i}", epsilon=1e-5)(var_x)
            var_x = layers.ReLU()(var_x)
        var_x = layers.AveragePooling2D(2, name="avgpool")(var_x)
        return var_x

    def _bottleneck(self,
                    inputs: tf.Tensor,
                    planes: int,
                    blocks: int,
                    stride: int = 1,
                    name: str = "layer") -> tf.Tensor:
        """ A private method that creates a sequential layer of Bottleneck blocks for the
        ModifiedResNet model.

        Parameters
        ----------
        inputs: :class:`tensorflow.Tensor`
                The input tensor
        planes: int
            The number of output channels for the layer.
        blocks: int
            The number of Bottleneck blocks in the layer.
        stride: int
            The stride for the first Bottleneck block in the layer. Default is 1.
        name: str
            The name of the layer. Default is "layer".

        Returns
        -------
        :class:`tensorflow.Tensor`
            Sequential block of bottlenecks
        """
        retval: tf.Tensor
        retval = Bottleneck(planes, planes, stride, name=f"{name}.0")(inputs)
        for i in range(1, blocks):
            retval = Bottleneck(planes * Bottleneck.expansion,
                                planes,
                                name=f"{name}.{i}")(retval)
        return retval

    def __call__(self) -> tf.keras.models.Model:
        """ Implements the forward pass of the ModifiedResNet model.

        Returns
        -------
        :class:`tensorflow.keras.models.Model`
            The modified resnet model.
        """
        inputs = layers.Input((self._input_resolution, self._input_resolution, 3))
        var_x = self._stem(inputs)

        for i in range(4):
            stride = 1 if i == 0 else 2
            var_x = self._bottleneck(var_x,
                                     self._width * (2 ** i),
                                     self._layer_config[i],
                                     stride=stride,
                                     name=f"{self._name}.layer{i + 1}")

        var_x = AttentionPool2d(self._input_resolution // 32,
                                self._width * 32,  # the ResNet feature dimension
                                self._heads,
                                self._output_dim,
                                name=f"{self._name}.attnpool")(var_x)
        return keras.models.Model(inputs, outputs=[var_x], name=self._name)


# ### #
# VIT #
# ### #
class ViT():  # pylint:disable=too-few-public-methods
    """ Visiual Transform from CLIP

    A Convolutional Language-Image Pre-Training (CLIP) model that encodes images and text into a
    shared latent space.

    Reference
    ---------
    https://arxiv.org/abs/2103.00020

    Parameters
    ----------
        name: ["RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-B-32",
               "ViT-B-16", "ViT-L-14", "ViT-L-14-336px", "FaRL-B_16-64"]
            The model configuration to use
        input_size: int, optional
            The required resolution size for the model. ``None`` for default preset size
        load_weights: bool, optional
            ``True`` to load pretrained weights. Default: ``False``
        """
    def __init__(self,
                 name: TypeModels,
                 input_size: int | None = None,
                 load_weights: bool = False) -> None:
        logger.debug("Initializing: %s (name: %s, input_size: %s, load_weights: %s)",
                     self.__class__.__name__, name, input_size, load_weights)
        assert name in ModelConfig, ("Name must be one of %s", list(ModelConfig))

        self._name = name
        self._load_weights = load_weights

        config = ModelConfig[name]
        self._git_id = config.git_id

        res = input_size if input_size is not None else config.resolution
        self._net = self._get_vision_net(config.layer_conf,
                                         config.width,
                                         config.embed_dim,
                                         res,
                                         config.patch)
        logger.debug("Initialized: %s", self.__class__.__name__)

    def _get_vision_net(self,
                        layer_config: int | tuple[int, int, int, int],
                        width: int,
                        embed_dim: int,
                        resolution: int,
                        patch_size: int) -> tf.keras.models.Model:
        """ Obtain the network for the vision layets

        Parameters
        ----------
        layer_config: tuple[int, int, int, int] | int
            Number of layers in the visual encoder, or a tuple of layer configurations for a custom
            ResNet visual encoder.
        width: int
            Width of the visual encoder layers.
        embed_dim: int
            Dimensionality of the final shared embedding space.
        resolution: int
            Spatial resolution of the input images.
        patch_size: int
            Size of the patches to be extracted from the images.

        Returns
        -------
        :class:`tensorflow.keras.models.Model`
            The :class:`ModifiedResNet` or :class:`VisualTransformer` vision model to use
        """
        if isinstance(layer_config, (tuple, list)):
            vision_heads = width * 32 // 64
            return ModifiedResNet(input_resolution=resolution,
                                  width=width,
                                  layer_config=layer_config,
                                  output_dim=embed_dim,
                                  heads=vision_heads,
                                  name="visual")
        vision_heads = width // 64
        return VisualTransformer(input_resolution=resolution,
                                 width=width,
                                 num_layers=layer_config,
                                 output_dim=embed_dim,
                                 heads=vision_heads,
                                 patch_size=patch_size,
                                 name="visual")

    def __call__(self) -> tf.keras.Model:
        """ Get the configured ViT model

        Returns
        -------
        :class:`tensorflow.keras.models.Model`
            The requested Visual Transformer model
        """
        net: tf.keras.models.Model = self._net()
        if self._load_weights and not self._git_id:
            logger.warning("Trained weights are not available for '%s'", self._name)
            return net
        if self._load_weights:
            model_path = GetModel(f"CLIPv_{self._name}_v1.h5", self._git_id).model_path
            logger.info("Loading CLIPv trained weights for '%s'", self._name)
            net.load_weights(model_path, by_name=True, skip_mismatch=True)

        return net


# Update layers into Keras custom objects
for name_, obj in inspect.getmembers(sys.modules[__name__]):
    if (inspect.isclass(obj) and issubclass(obj, tf.keras.layers.Layer)
            and obj.__module__ == __name__):
        keras.utils.get_custom_objects().update({name_: obj})
