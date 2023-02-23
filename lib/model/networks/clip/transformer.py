import sys

from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import keras.backend as K
from lib.model.layers import ResidualAttentionBlock

class Transformer():
    def __init__(self, width: int, layers: int, heads: int, attn_mask: tf.Tensor = None, name="transformer"):
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
        return {
            "width": self.width,
            "layers": self.num_layers,
            "heads": self.heads,
            "name": self.name
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def __call__(self):
        inputs = Input([197, self.width])
        var_x = self.resblocks(inputs)
        return Model(inputs=inputs, outputs=[var_x], name=self.name)
