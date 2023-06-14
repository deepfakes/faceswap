import tensorflow as tf
from tensorflow.keras import layers as klayers


class LayerNorm(klayers.LayerNormalization):
    """Subclass LayerNorm to override epsolon to torch default."""

    def __init__(self, name="LayerNorm"):
        super(LayerNorm, self).__init__(epsilon=1e-05, name=name)

    def call(self, x: tf.Tensor):
        return super().call(x)
