from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers as klayers


class Bottleneck(klayers.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, name: str = "bottleneck"):
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
        return {
            "inplanes": self.inplanes,
            "planes": self.planes,
            "stride": self.stride,
            "name": self.name
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, x: tf.Tensor):
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
    def __init__(self, spatial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None,
                 name="AttentionPool2d"):
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
        return {
            "spatial_dim": self.spatial_dim,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "output_dim": self.output_dim,
            "name": self.name
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, x, training=None):
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
        with tf.name_scope(name):
            layers = [Bottleneck(self._inplanes, planes, stride, name=name + "/0")]

            self._inplanes = planes * Bottleneck.expansion
            for i in range(1, blocks):
                layers.append(Bottleneck(self._inplanes, planes, name=name + f"/{i}"))

            return keras.Sequential(layers, name="bla")

    def call(self, x):
        def stem(x):
            for conv_pad, conv, bn in [
                (self.conv1_padding, self.conv1, self.bn1),
                (self.conv2_padding, self.conv2, self.bn2),
                (self.conv3_padding, self.conv3, self.bn3)
            ]:
                x = self.relu(bn(conv(conv_pad(x))))
            x = self.avgpool(x)
            return x

        # x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x
