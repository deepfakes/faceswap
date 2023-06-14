# Clip implmented in TF2 from https://github.com/RobertBiehl/CLIP-tf2

from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
from tensorflow import keras
import tensorflow as tf
from .resnet import ModifiedResNet
from .transformer import Transformer
from .visual_transformer import VisualTransformer


class CLIP(keras.Model):
    """
    A Convolutional Language-Image Pre-Training (CLIP) model that encodes images and text into a shared latent space.

    Parameters
    ----------
        embed_dim: int 
            Dimensionality of the final shared embedding space.
        image_resolution: int 
            Spatial resolution of the input images.
        vision_layers: Union[Tuple[int, int, int, int], int] 
            Number of layers in the visual encoder, or a tuple of
                layer configurations for a custom ResNet visual encoder.
        vision_width: int 
            Width of the visual encoder layers.
        vision_patch_size: int 
            Size of the patches to be extracted from the images.
        context_length: int 
            Length of the input text sequences.
        vocab_size :int 
            Size of the vocabulary.
        transformer_width: int 
            Width of the transformer layers.
        transformer_heads: int 
            Number of heads in the transformer attention mechanism.
        transformer_layers: int 
            Number of transformer layers.

    Attributes:
    ---------- 
        image_resolution: int 
            Spatial resolution of the input images.
        context_length: int 
            Length of the input text sequences.
        visual: keras.Model 
            Visual encoder module.
        transformer: Transformer 
            Transformer module for the text encoder.
        vocab_size: int 
            Size of the vocabulary.
        token_embedding: tf.Variable 
            Embedding layer for the text encoder.
        positional_embedding: tf.Variable 
            Positional encoding layer for the text encoder.
        ln_final: keras.layers.LayerNormalization 
            Layer normalization for the final transformer output.
        text_projection: tf.Variable 
            Projection layer for the final text embedding.
        logit_scale: tf.Variable 
            Scaling factor for the final cosine similarity scores.

    """
    def __init__(self,
                 embed_dim: inzt,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        """
        Initializes the CLIP model.

        Parameters
        ----------
            embed_dim: int 
                Dimensionality of the final shared embedding space.
            image_resolution: int 
                Spatial resolution of the input images.
            vision_layers: Union[Tuple[int, int, int, int], int] 
                Number of layers in the visual 
                    encoder, or a tuple of layer configurations for a custom ResNet visual encoder.
            vision_width: int 
                Width of the visual encoder layers.
            vision_patch_size: int 
                Size of the patches to be extracted from the images.
            context_length: int 
                Length of the input text sequences.
            vocab_size: int 
                Size of the vocabulary.
            transformer_width: int 
                Width of the transformer layers.
            transformer_heads: int 
                Number of heads in the transformer attention mechanism.
            transformer_layers: int 
                Number of transformer layers.

        Returns:
        -------
            None
        """
        super().__init__()

        self.image_resolution = image_resolution
        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width,
                name="visual"
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisualTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                name="visual"
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            name="transformer"
        )

        self.vocab_size = vocab_size
        self.token_embedding = tf.Variable(
            tf.zeros((vocab_size, transformer_width)), name="token_embedding")
        self.positional_embedding = tf.Variable(
            tf.zeros((self.context_length, transformer_width)), name="positional_embedding")
        self.ln_final = keras.layers.LayerNormalization(epsilon=1e-05, name="ln_final")

        self.text_projection = tf.Variable(
            tf.zeros((transformer_width, embed_dim)), name="text_projection")
        self.logit_scale = tf.Variable(np.ones([]) * np.log(1 / 0.07),
                                       dtype=tf.float32, name="logit_scale")

    def initialize_parameters(self):
        """
        Initializes the parameters of the CLIP model.

        Returns:
        -------
            None.
        """
        # TODO: convert to tf, for model initialization (not needed for pretrained weights)
        self.token_embedding.assign(tf.random.normal(self.token_embedding.shape, stddev=0.02))
        self.positional_embedding.assign(tf.random.normal(
            self.positional_embedding.shape, stddev=0.01))

        from resnet import ModifiedResNet
        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                self.visual.attnpool.q_proj.weight.assign(tf.random.normal(
                    self.visual.attnpool.q_proj.weight.shape, stddev=std))
                self.visual.attnpool.k_proj.weight.assign(tf.random.normal(
                    self.visual.attnpool.k_proj.weight.shape, stddev=std))
                self.visual.attnpool.v_proj.weight.assign(tf.random.normal(
                    self.visual.attnpool.v_proj.weight.shape, stddev=std))
                self.visual.attnpool.c_proj.weight.assign(tf.random.normal(
                    self.visual.attnpool.c_proj.weight.shape, stddev=std))

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        param.assign(tf.zeros_like(param))

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            block.attn.in_proj.weight.assign(
                tf.random.normal(block.attn.in_proj.weight.shape, stddev=attn_std))
            block.attn.out_proj.weight.assign(
                tf.random.normal(block.attn.out_proj.weight.shape, stddev=proj_std))
            block.mlp.c_fc.weight.assign(
                tf.random.normal(block.mlp.c_fc.weight.shape, stddev=fc_std))
            block.mlp.c_proj.weight.assign(
                tf.random.normal(block.mlp.c_proj.weight.shape, stddev=proj_std))

        if self.text_projection is not None:
            std = self.transformer.width ** -0.5
            self.text_projection.assign(
                tf.random.normal(self.text_projection.shape, stddev=std))

    def build_attention_mask(self):
        """
        Builds an attention mask tensor for the CLIP model.

        Returns:
        -------
            tf.Tensor: of shape [batch_size, context_length, context_length] with boolean values
                indicating which tokens should be attended to.
        """
        n_dest = self.context_length
        n_src = self.context_length
        dtype = tf.bool
        batch_size = 1

        i = tf.range(n_dest)[:, None]
        j = tf.range(n_src)
        mask = i >= j - n_src + n_dest
        mask = tf.cast(mask, dtype)
        mask = tf.reshape(mask, [1, n_dest, n_src])
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
        )
        return tf.tile(mask, mult)

    @property
    def dtype(self):
        """
        Returns the dtype of the weights.

        Returns:
        -------
            tf.Tensor: The dtype of the weights.
        """
        return self.visual.conv1.weight.dtype

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32, name="image")])
    def encode_image(self, image: tf.Tensor):
        """
        Encodes an image tensor using the visual function.

        Parameters
        ----------
            image: tf.Tensor 
                A tensor of shape [batch_size, image_resolution, image_resolution, channels]
                    (Note: Channels could be 1 -monochrome or 3 -RGB color)
                        containing the image to be encoded.

        Returns:
        -------
            tf.Tensor: A tensor of shape[batch_size, embed_size] containing the encoded representation of the image.
        """
        return self.visual(image)

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int32, name="text")])
    def encode_text(self, text: tf.Tensor) -> tf.Tensor:
        """
        Encodes the input text using the pretrained model.

        Parameters
        ----------
            text: (tf.Tensor) 
                of shape [batch_size, n_ctx], containing the tokenized text.

        Returns:
        -------
            tf.Tensor: of shape [batch_size, embed_size], containing the encoded representation
            of the input text.
        """
        var_x = tf.nn.embedding_lookup(self.token_embedding, text)
        var_x = var_x + self.positional_embedding
        var_x = self.transformer(var_x)
        var_x = self.ln_final(var_x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        x_shape = tf.shape(var_x)
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        eot_token = tf.argmax(text, axis=-1)

        # TODO check if dtype is correct
        idx = tf.transpose(
            tf.stack((tf.range(0, x_shape[0], dtype=tf.int64), eot_token), axis=0, name='take_features_idx'))
        var_x = tf.gather_nd(var_x, idx) @ self.text_projection

        return var_x

    @tf.function(input_signature=[(
        tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32, name="image"),
        tf.TensorSpec(shape=(None, None, None), dtype=tf.int32, name="text")
    )])
    def call(self, input: Tuple[tf.Tensor, tf.Tensor]):
        """
        Finds the embeddings for a batch of images and texts and then computes the
            cosine similarity between the embeddings.

        Parameters
        ----------
            input: Tuple[tf.Tensor, tf.Tensor] 
                A tuple of two tensors containing the input
                    images and texts. The image tensor should have shape [batch_size,
                        image_resolution, image_resolution, channels], and the text tensor should have
                            shape [batch_size, sequence_length].
        Returns:
        -------
            Tuple[tf.Tensor, tf.Tensor]: A tuple of two tensors containing the logits for the
                image and text inputs. Both tensors will have shape [batch_size, embed_dim].
        """
        image, text = input
        image_features = self.encode_image(image)

        # TODO: find another way to feed data, but keras requires that all input tensors have to have the same batch size
        text = tf.squeeze(text, axis=0)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / tf.norm(image_features, axis=-1, keepdims=True)
        text_features = text_features / tf.norm(text_features, axis=-1, keepdims=True)

        # cosine similarity as logits
        logit_scale = tf.exp(self.logit_scale)
        logits_per_image = logit_scale * image_features @ tf.transpose(text_features)
        logits_per_text = logit_scale * text_features @ tf.transpose(image_features)

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text
