import sys

from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import keras.backend as K
from lib.model.layers import ResidualAttentionBlock

class Transformer():
    """
    A class representing a Transformer model with attention mechanism and residual connections.

    Parameters
    ----------
    width : int
        The dimension of the input and output vectors.
    layers : int
        The number of layers in the Transformer.
    heads : int
        The number of attention heads.
    attn_mask : tf.Tensor, optional
        The attention mask, by default None.
    name : str, optional
        The name of the Transformer model, by default "transformer".

    Attributes
    ----------
    width : int
        The dimension of the input and output vectors.
    num_layers : int
        The number of layers in the Transformer.
    heads : int
        The number of attention heads.
    attn_mask : tf.Tensor, optional
        The attention mask, by default None.
    name : str, optional
        The name of the Transformer model, by default "transformer".
    resblocks : keras.Sequential
        The sequence of residual attention blocks.

    Methods
    -------
    get_config() -> Dict[str, Union[int, str]]:
        Returns a dictionary containing the Transformer configuration.
    from_config(cls, config: Dict[str, Union[int, str]]) -> 'Transformer':
        Returns a new Transformer instance from the given configuration dictionary.
    __call__() -> Model:
        Builds and returns the Transformer model.
    """
    def __init__(self, width: int, layers: int, heads: int, attn_mask: tf.Tensor = None, name="transformer"):
        """
        Initializes a new instance of the Transformer class.
        """
        self.width = width
        self.num_layers = layers
        self.heads = heads
        self.attn_mask = attn_mask
        self.name = name
        self.resblocks = keras.Sequential([
            ResidualAttentionBlock(width, heads, attn_mask, name=f"{name}.resblocks.{i}", idx=i)
            for i in range(layers)
        ], name=name + ".resblocks")

    def get_config(self):
        """
        Returns a dictionary containing the Transformer configuration.

        Returns
        -------
        Dict[str, Union[int, str]]
            The Transformer configuration dictionary.
        """
        return {
            "width": self.width,
            "layers": self.num_layers,
            "heads": self.heads,
            "name": self.name
        }

    @classmethod
    def from_config(cls, config):
        """
        Returns a new Transformer instance from the given configuration dictionary.

        Parameters
        ----------
        config : Dict[str, Union[int, str]]
            The configuration dictionary.

        Returns
        -------
        Transformer
            A new Transformer instance with the given configuration.
        """
        return cls(**config)

    def __call__(self):
        """
        Builds and returns the Transformer model.

        Returns
        -------
        Model
            The Transformer model.
        """
        inputs = Input([197, self.width])
        var_x = self.resblocks(inputs)
        return Model(inputs=inputs, outputs=[var_x], name=self.name)
