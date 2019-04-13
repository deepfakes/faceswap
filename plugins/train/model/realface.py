#!/usr/bin/env python3
""" RealFaceRC1, codenamed 'Pegasus'
    Based on the original https://www.reddit.com/r/deepfakes/
    code sample + contribs
    Major thanks goes to BryanLyon as it vastly powered by his ideas and insights.
    Without him it would not be possible to come up with the model.
    Additional thanks: Birb - source of inspiration, great Encoder ideas
                       Kvrooman - additional couseling on autoencoders and practical advices
    """

from keras.initializers import RandomNormal
from keras.layers import Conv2D, Dense, Flatten, Input, Reshape
from keras.models import Model as KerasModel

from ._base import ModelBase, logger


LR_REFINE, LR_LOW, LR_LOWER, LR_NORMAL, LR_HIGH = 5e-6, 1e-5, 2.5e-5, 5e-5, 1e-4


class Model(ModelBase):
    """ RealFace(tm) Faceswap Model """
    def __init__(self, *args, **kwargs):
        logger.debug("Initializing %s: (args: %s, kwargs: %s",
                     self.__class__.__name__, args, kwargs)

        self.dowscalers_no = 4  # don't change!

        output_size = self.config['output_size']
        sides = [(output_size // 2**n, n) for n in [4, 5] if (output_size // 2**n) < 10]
        self._downscale_ratio = 2**self.dowscalers_no
        closest = min([x*self._downscale_ratio for x, _ in sides],
                      key=lambda x: abs(x-self.config['input_size']))
        self.dense_width, self.upscalers_no = [(s, n) for s, n in sides
                                               if s * self._downscale_ratio == closest][0]

        self.dense_filters = (int(1024 - (self.dense_width-4)*64) // 16)*16  # don't change!

        # print('self.dense_width', self.dense_width, 'self.upscalers_no', self.upscalers_no,
        #       'self.dense_filters', self.dense_filters)
        self.lowmem = self.config.get("lowmem", False)

        kwargs["input_shape"] = (self.config["input_size"], self.config["input_size"], 3)
        self.kernel_initializer = RandomNormal(0, 0.02)

        super().__init__(*args, **kwargs)
        logger.debug("Initialized %s", self.__class__.__name__)

    def add_networks(self):
        """ Add the original model weights """
        logger.debug("Adding networks")
        self.add_network("decoder", "a", self.decoder_a())
        self.add_network("decoder", "b", self.decoder_b())
        self.add_network("encoder", None, self.encoder())
        logger.debug("Added networks")

    def build_autoencoders(self):
        """ Initialize original model """
        logger.debug("Initializing model")
        inputs = [Input(shape=self.input_shape, name="face")]
        if self.config.get("mask_type", None):
            mask_shape = self.config["output_size"], self.config["output_size"], 1
            inputs.append(Input(shape=mask_shape, name="mask"))

        for side in "a", "b":
            logger.debug("Adding Autoencoder. Side: %s", side)
            decoder = self.networks["decoder_{}".format(side)].network
            output = decoder(self.networks["encoder"].network(inputs[0]))
            autoencoder = KerasModel(inputs, output)
            self.add_predictor(side, autoencoder)
        logger.debug("Initialized model")

    def encoder(self):
        """ RealFace Encoder Network """
        input_ = Input(shape=self.input_shape)
        var_x = input_

        encoder_complexity = self.config['complexity_encoder']

        for idx in range(self.dowscalers_no-1):
            var_x = self.blocks.conv(var_x, encoder_complexity * 2**idx)
            var_x = self.blocks.res_block(var_x, encoder_complexity * 2**idx, use_bias=True)
            var_x = self.blocks.res_block(var_x, encoder_complexity * 2**idx, use_bias=True)

        var_x = self.blocks.conv(var_x, encoder_complexity * 2**(idx + 1))

        return KerasModel(input_, var_x)

    def decoder_b(self):
        """ RealFace Decoder Network """
        input_width = self.config['input_size'] // self._downscale_ratio
        input_ = Input(shape=(input_width, input_width, 1024))

        var_xy = input_

        var_xy = Dense(self.config['dense_nodes'])(Flatten()(var_xy))
        var_xy = Dense(self.dense_width * self.dense_width * self.dense_filters)(var_xy)
        var_xy = Reshape((self.dense_width, self.dense_width, self.dense_filters))(var_xy)
        var_xy = self.blocks.upscale(var_xy, self.dense_filters)

        var_x = var_xy
        var_x = self.blocks.res_block(var_x, self.dense_filters, use_bias=False)

        decoder_b_complexity = self.config['complexity_decoder']
        for idx in range(self.upscalers_no - 2):
            var_x = self.blocks.upscale(var_x, decoder_b_complexity // 2**idx)
            var_x = self.blocks.res_block(var_x, decoder_b_complexity // 2**idx, use_bias=False)
            var_x = self.blocks.res_block(var_x, decoder_b_complexity // 2**idx, use_bias=True)
        var_x = self.blocks.upscale(var_x, decoder_b_complexity // 2**(idx + 1))

        var_x = Conv2D(3, kernel_size=5, padding="same", activation="sigmoid")(var_x)

        outputs = [var_x]

        if self.config.get("mask_type", None) is not None:
            var_y = var_xy
            mask_b_complexity = 384
            for idx in range(self.upscalers_no-2):
                var_y = self.blocks.upscale(var_y, mask_b_complexity // 2**idx)
            var_y = self.blocks.upscale(var_y, mask_b_complexity // 2**(idx + 1))

            var_y = Conv2D(1, kernel_size=5, padding='same', activation='sigmoid')(var_y)

            outputs += [var_y]

        return KerasModel(input_, outputs=outputs)

    def decoder_a(self):
        """ RealFace Decoder (A) Network """
        input_width = self.config['input_size'] // self._downscale_ratio
        input_ = Input(shape=(input_width, input_width, 1024))

        var_xy = input_

        dense_nodes = int(self.config['dense_nodes']/1.5)
        dense_filters = int(self.dense_filters/1.5)

        var_xy = Dense(dense_nodes)(Flatten()(var_xy))
        var_xy = Dense(self.dense_width * self.dense_width * dense_filters)(var_xy)
        var_xy = Reshape((self.dense_width, self.dense_width, dense_filters))(var_xy)

        var_xy = self.blocks.upscale(var_xy, dense_filters)

        var_x = var_xy
        var_x = self.blocks.res_block(var_x, dense_filters, use_bias=False)

        decoder_a_complexity = int(self.config['complexity_decoder'] / 1.5)
        for idx in range(self.upscalers_no-2):
            var_x = self.blocks.upscale(var_x, decoder_a_complexity // 2**idx)
        var_x = self.blocks.upscale(var_x, decoder_a_complexity // 2**(idx + 1))

        var_x = Conv2D(3, kernel_size=5, padding="same", activation="sigmoid")(var_x)

        outputs = [var_x]

        if self.config.get("mask_type", None) is not None:
            var_y = var_xy
            mask_a_complexity = 384
            for idx in range(self.upscalers_no-2):
                var_y = self.blocks.upscale(var_y, mask_a_complexity // 2**idx)
            var_y = self.blocks.upscale(var_y, mask_a_complexity // 2**(idx + 1))

            var_y = Conv2D(1, kernel_size=5, padding='same', activation='sigmoid')(var_y)

            outputs += [var_y]

        return KerasModel(input_, outputs=outputs)
