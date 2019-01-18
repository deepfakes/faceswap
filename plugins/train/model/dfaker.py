#!/usr/bin/env python3
""" DFaker Model
    Based on the dfaker model: https://github.com/dfaker """


from keras.initializers import RandomNormal
from keras.layers import Input
from keras.models import Model as KerasModel

from lib.model.nn_blocks import Conv2D, res_block, upscale

from .original import logger, Model as OriginalModel


class Model(OriginalModel):
    """ Improved Autoeencoder Model """
    def __init__(self, *args, **kwargs):
        logger.debug("Initializing %s: (args: %s, kwargs: %s",
                     self.__class__.__name__, args, kwargs)
        kwargs["input_shape"] = (64, 64, 3)
        kwargs["encoder_dim"] = 1024
        self.kernel_initializer = RandomNormal(0, 0.02)
        super().__init__(*args, **kwargs)
        logger.debug("Initialized %s", self.__class__.__name__)

    def set_training_data(self):
        """ Set the dictionary for training """
        self.training_opts["serializer"] = self.config["alignments_format"]
        self.training_opts["mask_type"] = self.config["mask_type"]
        self.training_opts["full_face"] = True
        self.training_opts["preview_images"] = 10
        super().set_training_data()

    def build_autoencoders(self):
        """ Initialize Dfaker model """
        logger.debug("Initializing model")
        mask_shape = (self.input_shape[0] * 2, self.input_shape[1] * 2, 1)
        for side in ("a", "b"):
            inp = [Input(shape=self.input_shape, name="face"),
                   Input(shape=mask_shape, name="mask")]
            decoder = self.networks["decoder_{}".format(side)].network
            output = decoder(self.networks["encoder"].network(inp[0]))
            autoencoder = KerasModel(inp, output)
            self.add_predictor(side, autoencoder)
        logger.debug("Initialized model")

    def decoder(self):
        """ Decoder Network """
        input_ = Input(shape=(8, 8, 512))
        use_subpixel = self.config["subpixel_upscaling"]

        inp_x = input_
        inp_y = input_

        inp_x = upscale(inp_x, 512, use_subpixel=use_subpixel)
        inp_x = res_block(inp_x, 512, kernel_initializer=self.kernel_initializer)
        inp_x = upscale(inp_x, 256, use_subpixel=use_subpixel)
        inp_x = res_block(inp_x, 256, kernel_initializer=self.kernel_initializer)
        inp_x = upscale(inp_x, 128, use_subpixel=use_subpixel)
        inp_x = res_block(inp_x, 128, kernel_initializer=self.kernel_initializer)
        inp_x = upscale(inp_x, 64, use_subpixel=use_subpixel)
        inp_x = Conv2D(3,
                       kernel_size=5,
                       padding='same',
                       activation='sigmoid')(inp_x)

        inp_y = upscale(inp_y, 512, use_subpixel=use_subpixel)
        inp_y = upscale(inp_y, 256, use_subpixel=use_subpixel)
        inp_y = upscale(inp_y, 128, use_subpixel=use_subpixel)
        inp_y = upscale(inp_y, 64, use_subpixel=use_subpixel)
        inp_y = Conv2D(1,
                       kernel_size=5,
                       padding='same',
                       activation='sigmoid')(inp_y)

        return KerasModel([input_], outputs=[inp_x, inp_y])
