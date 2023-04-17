import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as klayers
from .transformer import Transformer
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import LayerNormalization, Dense, Input, Concatenate, Reshape
from tensorflow.keras.initializers import RandomNormal
import numpy as np

class VisualTransformer():
    """
    A class representing a Visual Transformer model for image classification tasks.

    Attributes
    ----------
    input_resolution : int
        The input resolution of the images.
    patch_size : int
        The size of the patches to be extracted from the images.
    width : int
        The dimension of the input and output vectors.
    num_layers : int
        The number of layers in the Transformer.
    heads : int
        The number of attention heads.
    output_dim : int
        The dimension of the output vector.
    name : str, optional
        The name of the Visual Transformer model, by default "VisualTransformer".
    conv1 : keras.layers.Conv2D
        The 2D convolution layer.
    transformer : Transformer
        The Transformer model.
    class_embedding : K.constant
        The class embedding vector.
    positional_embedding : K.constant
        The positional embedding matrix.
    ln_pre : keras.layers.LayerNormalization
        The layer normalization applied before the Transformer.
    ln_post : keras.layers.LayerNormalization
        The layer normalization applied after the Transformer.
    proj : K.constant
        The projection matrix.

    Methods
    -------
    get_config() -> Dict[str, Union[int, str]]:
        Returns a dictionary containing the Visual Transformer configuration.
    from_config(cls, config: Dict[str, Union[int, str]]) -> 'VisualTransformer':
        Returns a new VisualTransformer instance from the given configuration dictionary.
    __call__() -> Model:
        Builds and returns the Visual Transformer model.
    """
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, name="VisualTransformer"):
        """
        Initializes a new instance of the VisualTransformer class.

        Parameters
        ----------
        input_resolution : int
            The input resolution of the images.
        patch_size : int
            The size of the patches to be extracted from the images.
        width : int
            The dimension of the input and output vectors.
        layers : int
            The number of layers in the Transformer.
        heads : int
            The number of attention heads.
        output_dim : int
            The dimension of the output vector.
        name : str, optional
            The name of the Visual Transformer model, by default "VisualTransformer".
        """
        self.input_resolution: int = input_resolution
        self.patch_size: int = patch_size
        self.width: int = width
        self.num_layers: int = layers
        self.heads: int = heads
        self.output_dim: int = output_dim
        self.name = name

        self.conv1 = klayers.Conv2D(width, patch_size, strides=patch_size, use_bias=False, name=f"{name}/conv1")

        scale = width ** -0.5

        self.transformer = Transformer(width, layers, heads, name=f"{name}//transformer")()

        self.class_embedding = K.constant(scale * np.random.random((width,)), name=f"{name}/class_embedding")
        self.positional_embedding = K.constant(scale * tf.random.normal(((input_resolution // patch_size) ** 2 + 1, width)), name=f"{name}/positional_embedding")
        self.ln_pre = keras.layers.LayerNormalization(epsilon=1e-05, name=f"{name}/ln_pre")

        self.ln_post = keras.layers.LayerNormalization(epsilon=1e-05, name=f"{name}/ln_post")
        self.proj = K.constant(scale * np.random.random((width, output_dim)), name=f"{name}/proj")

    def get_config(self):
        """
        Returns a dictionary containing the Visual Transformer configuration.

        Returns
        -------
        Dict[str, Union[int, str]]
            The Visual Transformer configuration dictionary.
        """
        return {
            "input_resolution": self.input_resolution,
            "patch_size": self.patch_size,
            "width": self.width,
            "layers": self.num_layers,
            "heads": self.heads,
            "output_dim": self.output_dim,
            "name": self.name
        }

    @classmethod
    def from_config(cls, config):
        """
        Returns a new VisualTransformer instance from the given configuration dictionary.

        Parameters
        ----------
        config : Dict[str, Union[int, str]]
            The configuration dictionary.

        Returns
        -------
        VisualTransformer
            A new VisualTransformer instance with the given configuration.
        """
        return cls(**config)

    def __call__(self):
        """
        Builds and returns the Visual Transformer model.

        Returns
        -------
        Model
            The Visual Transformer model.
        """
        inputs = Input([self.input_resolution, self.input_resolution, 3])
        var_x = self.conv1(inputs)  # shape = [*, grid, grid, width]

        x_shape = var_x.shape
        var_x = Reshape((196, self.width))(var_x)  # shape = [*, grid ** 2, width]

        x_shape = K.shape(var_x)
        class_embedding = K.expand_dims(K.expand_dims(K.cast(self.class_embedding, var_x.dtype),0),0)
        class_embedding_tiled = K.tile(class_embedding, [x_shape[0], 1, 1])
        var_x = Concatenate(axis=1)([class_embedding_tiled, var_x])
        var_x = var_x + K.cast(self.positional_embedding, var_x.dtype)
        var_x = self.ln_pre(var_x)
        var_x = self.transformer(var_x)
        var_x = self.ln_post(var_x[:, 0, :])

        if self.proj is not None:
            if var_x.dtype == tf.float16:
                var_x = K.cast(var_x, tf.float32) #TODO: remove this when tf.matmul supports float16 with float32
                var_x = var_x @ self.proj
                var_x = K.cast(var_x, tf.float16)
            else:
                var_x = var_x @ self.proj
        return Model(inputs=inputs, outputs=[var_x], name=self.name)

