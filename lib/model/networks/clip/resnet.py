from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers as klayers


class Bottleneck(klayers.Layer):
    """
    A ResNet bottleneck block that performs a sequence of convolutions, batch normalization, and ReLU activation
    operations on an input tensor.

    Parameters
    ----------
    inplanes: int 
        The number of input channels.
    planes: int 
        The number of output channels.
    stride: int 
        The stride of the bottleneck block.
    name: str 
        The name of the bottleneck block.

    Attributes:
    ---------- 
    expansion: int 
        The factor by which the number of input channels is expanded to get the number of output channels.
    conv1: keras.layers.Conv2D 
        The first 1x1 convolution layer in the bottleneck block.
    bn1: keras.layers.BatchNormalization 
        The first batch normalization layer in the bottleneck block.
    conv2_padding: keras.layers.ZeroPadding2D 
        The zero padding layer applied before the second convolution in the bottleneck block.
    conv2: keras.layers.Conv2D 
        The second 3x3 convolution layer in the bottleneck block.
    bn2: keras.layers.BatchNormalization 
        The second batch normalization layer in the bottleneck block.
    avgpool: keras.layers.AveragePooling2D 
        The average pooling layer that is applied after the second convolution, if the stride is greater than 1.
    conv3: keras.layers.Conv2D 
        The third 1x1 convolution layer in the bottleneck block.
    bn3: keras.layers.BatchNormalization 
        The third batch normalization layer in the bottleneck block.
    relu: keras.layers.ReLU 
        The ReLU activation layer in the bottleneck block.
    downsample: keras.Sequential 
        A downsampling block consisting of an average pooling layer, followed by a 1x1 convolution layer and a batch normalization layer. This block is used to match the dimensions of the input tensor with the output tensor of the bottleneck block when the stride is greater than 1 or the number of input channels is different from the number of output channels.
    stride: int 
        The stride of the bottleneck block.
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, name: str = "bottleneck"):
        """
        Initializes a Bottleneck block.
        """
        super().__init__(name=name)

        with tf.name_scope(name):
            # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
            self.conv1 = klayers.Conv2D(planes, 1, use_bias=False, name="conv1")
            self.bn1 = klayers.BatchNormalization(name="bn1", epsilon=1e-5)

            self.conv2_padding = klayers.ZeroPadding2D(padding=((1, 1), (1, 1)))
            self.conv2 = klayers.Conv2D(planes, 3, use_bias=False, name="conv2")
            self.bn2 = klayers.BatchNormalization(name="bn2", epsilon=1e-5)

            self.avgpool = klayers.AveragePooling2D(stride) if stride > 1 else None

            self.conv3 = klayers.Conv2D(planes * self.expansion, 1, use_bias=False, name="conv3")
            self.bn3 = klayers.BatchNormalization(name="bn3", epsilon=1e-5)

            self.relu = klayers.ReLU()
            self.downsample = None
            self.stride = stride

            self.inplanes = inplanes
            self.planes = planes

            if stride > 1 or inplanes != planes * Bottleneck.expansion:
                # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
                self.downsample = keras.Sequential([
                    klayers.AveragePooling2D(stride, name=name + "/downsample/avgpool"),
                    klayers.Conv2D(planes * self.expansion, 1, strides=1, use_bias=False, name=name + "/downsample/0"),
                    klayers.BatchNormalization(name=name + "/downsample/1", epsilon=1e-5)
                ], name="downsample")

    def get_config(self):
        """
        Returns the configuration dictionary for a Bottleneck block.

        Returns
        -------
            dict: containing the configuration of a Bottleneck block.
        """
        return {
            "inplanes": self.inplanes,
            "planes": self.planes,
            "stride": self.stride,
            "name": self.name
        }

    @classmethod
    def from_config(cls, config):
        """
        Creates a Bottleneck block from its configuration dictionary.

        Parameters
        ----------
            dict: The configuration dictionary for the Bottleneck block.

        Returns
        -------
            klayers.Layer: A Bottleneck block class created from the configuration dictionary.
        """
        return cls(**config)

    def call(self, x: tf.Tensor):
        """
        Performs the forward pass for a Bottleneck block.

        Parameters
        ----------
            x (tf.Tensor): The input tensor to the Bottleneck block.

        Returns
        -------
            tf.Tensor: The result of the forward pass through the Bottleneck block.
        """
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(self.conv2_padding(out))))
        if self.avgpool is not None:
            out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            # x = tf.nn.avg_pool(x, (1, 2, 2, 1), strides=(1, 2, 2, 1), padding='VALID')
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(klayers.Layer):
    """
    An Attention Pooling layer that applies a multi-head self-attention mechanism over a spatial grid of features.

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

    Attributes:
    ---------- 
        spatial_dim: int 
            The dimensionality of the spatial grid of features.
        embed_dim: int 
            The dimensionality of the feature embeddings.
        num_heads: int 
            The number of attention heads.
        output_dim: int 
            The output dimensionality of the attention layer.
        positional_embedding: tf.Variable 
            The positional embedding used in the attention layer.
        _key_dim: int 
            The dimensionality of the attention keys.
        multi_head_attention: klayers.MultiHeadAttention 
            The multi-head attention layer used in the attention pooling.

    """
    def __init__(self, spatial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None,
                 name="AttentionPool2d"):
        """
        Initializes the AttentionPool2d layer.

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
        super().__init__(name=name)

        self.spatial_dim = spatial_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.output_dim = output_dim

        with tf.name_scope(name):
            self.positional_embedding = tf.Variable(
                tf.random.normal((spatial_dim ** 2 + 1, embed_dim)) / embed_dim ** 0.5,
                name="positional_embedding"
            )

        self.num_heads = num_heads
        self._key_dim = embed_dim

        self.multi_head_attention = klayers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            output_shape=output_dim or embed_dim,
            name="mha"
        )

    def get_config(self):
        """
        Returns the configuration dictionary for an AttentionPool2d layer.

        Returns
        -------
            dict: containing the configuration of an AttentionPool2d layer.
        """
        return {
            "spatial_dim": self.spatial_dim,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "output_dim": self.output_dim,
            "name": self.name
        }

    @classmethod
    def from_config(cls, config):
        """
        Creates an AttentionPool2d layer from its configuration dictionary.
        Parameters
        ----------
            dict: The configuration dictionary for the AttentionPool2d layer.

        Returns
        -------
            klayers.Layer: An AttentionPool2d layer created from the configuration dictionary.
        """
        return cls(**config)

    def call(self, x, training=None):
        """Performs the attention pooling operation on the input tensor.

        Parameters
        ----------
            x: tf.Tensor 
                The input tensor of shape [batch_size, height, width, embed_dim].
            bool: Whether the layer is in training mode. Defaults to None.

        Returns
        -------
            tf.Tensor: The result of the attention pooling operation."""
        x_shape = tf.shape(x)
        x = tf.reshape(x, (x_shape[0], x_shape[1] * x_shape[2], x_shape[3]))  # NHWC -> N(HW)C

        x = tf.concat([tf.reduce_mean(x, axis=1, keepdims=True), x], axis=1)  # N(HW+1)C
        x = x + tf.cast(self.positional_embedding[None, :, :], x.dtype)  # N(HW+1)C

        query, key, value = x, x, x
        x = self.multi_head_attention(query, value, key)

        # only return the first element in the sequence
        return x[:, 0, ...]


class ModifiedResNet(keras.Model):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64, name="ModifiedResNet"):
        """
        Initializes a ModifiedResNet model with the given configuration.

        Parameters
        ----------
            layers: list 
                A list containing the number of Bottleneck blocks for each layer.
            output_dim: int 
                The output dimension of the model.
            heads: int 
                The number of heads for the QKV attention.
            input_resolution: int 
                The input resolution of the model. Default is 224.
            width: int 
                The width of the model. Default is 64.
            name: str 
                The name of the model. Default is "ModifiedResNet".

        Attributes:
        ---------- 
            layers_config: list 
                A list containing the number of Bottleneck blocks for each layer.
            output_dim: int 
                The output dimension of the model.
            heads: int 
                The number of heads for the QKV attention.
            input_resolution: int 
                The input resolution of the model.
            width: int 
                The width of the model.
            conv1_padding, conv1, bn1, conv2_padding, conv2, bn2, conv3_padding, conv3, bn3, avgpool, relu:
                Stem layers.
            _inplanes: int 
                A mutable variable used during construction, initialized to the width of the model.
            layer1, layer2, layer3, layer4: 
                Residual layers.
            attnpool: AttentionPool2d 
                The QKV attention pooling layer.
        """
        super().__init__(name=name)
        self.layers_config = layers
        self.output_dim = output_dim
        self.heads = heads
        self.input_resolution = input_resolution
        self.width = width

        # the 3-layer stem
        self.conv1_padding = klayers.ZeroPadding2D(padding=((1, 1), (1, 1)), name="conv1_padding")
        self.conv1 = klayers.Conv2D(width // 2, 3, strides=2, use_bias=False, name="conv1")
        self.bn1 = klayers.BatchNormalization(name="bn1", epsilon=1e-5)
        self.conv2_padding = klayers.ZeroPadding2D(padding=((1, 1), (1, 1)), name="conv2_padding")
        self.conv2 = klayers.Conv2D(width // 2, 3, use_bias=False, name="conv2")
        self.bn2 = klayers.BatchNormalization(name="bn2", epsilon=1e-5)
        self.conv3_padding = klayers.ZeroPadding2D(padding=((1, 1), (1, 1)), name="conv3_padding")
        self.conv3 = klayers.Conv2D(width, 3, use_bias=False, name="conv3")
        self.bn3 = klayers.BatchNormalization(name="bn3", epsilon=1e-5)
        self.avgpool = klayers.AveragePooling2D(2, name="avgpool")
        self.relu = klayers.ReLU()

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0], name=name + "/layer1")
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2, name=name + "/layer2")
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2, name=name + "/layer3")
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2, name=name + "/layer4")

        embed_dim = width * 32  # the ResNet feature dimension
        with tf.name_scope(name):
            self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim, name="attnpool")

    def get_config(self):
        """
        Returns a dictionary containing the configuration of the ModifiedResNet model containing
            the following key-value pairs:
            - "layers": the configuration of layers
            - "output_dim": the output dimension
            - "heads": the number of heads for the QKV attention
            - "input_resolution": the input resolution of the model
            - "width": the width of the model
            - "name": the name of the model
        """
        return {
            "layers": self.layers_config,
            "output_dim": self.output_dim,
            "heads": self.heads,
            "input_resolution": self.input_resolution,
            "width": self.width,
            "name": self.name
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def _make_layer(self, planes, blocks, stride=1, name="layer"):
        """
        A private method that creates a sequential layer of Bottleneck blocks for the ModifiedResNet model.

        Parameters
        ----------
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
            keras.Sequential: A sequential layer of Bottleneck blocks.
        """
        with tf.name_scope(name):
            layers = [Bottleneck(self._inplanes, planes, stride, name=name + "/0")]

            self._inplanes = planes * Bottleneck.expansion
            for i in range(1, blocks):
                layers.append(Bottleneck(self._inplanes, planes, name=name + f"/{i}"))

            return keras.Sequential(layers, name="bla")
        
    def stem(self, x):
        """
        Applies the stem operation to the input tensor, which consists of 3 convolutional
            layers with BatchNormalization and ReLU activation, followed by an average pooling layer.

        Parameters
        ----------
            x: tf.Tensor 
                The input tensor of shape [batch_size, height, width, channels].

        Returns
        -------
            tf.Tensor: The output tensor after applying the stem operation.
        """
        for conv_pad, conv, bn in [
            (self.conv1_padding, self.conv1, self.bn1),
            (self.conv2_padding, self.conv2, self.bn2),
            (self.conv3_padding, self.conv3, self.bn3)
        ]:
            x = self.relu(bn(conv(conv_pad(x))))
        x = self.avgpool(x)
        return x

    def call(self, x):
        """
        Implements the forward pass of the ModifiedResNet model.

        Parameters
        ----------
            x (tf.Tensor): The input tensor of shape [batch_size, height, width, channels].

        Returns
        -------
            tf.Tensor: The output tensor after passing through the ModifiedResNet model.
        """

        # x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x
