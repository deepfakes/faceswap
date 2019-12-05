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
from keras.layers import Dense, Flatten, Input, Reshape
from keras.models import Model as KerasModel

from ._base import ModelBase, logger


class Model(ModelBase):
    """ RealFace(tm) Faceswap Model """
    def __init__(self, *args, **kwargs):
        logger.debug("Initializing %s: (args: %s, kwargs: %s",
                     self.__class__.__name__, args, kwargs)

        self.configfile = kwargs.get("configfile", None)
        self.check_input_output()
        self.dense_width, self.upscalers_no = self.get_dense_width_upscalers_numbers()
        kwargs["input_shape"] = (self.config["input_size"], self.config["input_size"], 3)
        self.kernel_initializer = RandomNormal(0, 0.02)

        super().__init__(*args, **kwargs)
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def downscalers_no(self):
        """ Number of downscalers. Don't change! """
        return 4

    @property
    def _downscale_ratio(self):
        """ Downscale Ratio """
        return 2**self.downscalers_no

    @property
    def dense_filters(self):
        """ Dense Filters. Don't change! """
        return (int(1024 - (self.dense_width - 4) * 64) // 16) * 16

    def check_input_output(self):
        """ Confirm valid input and output sized have been provided """
        if not 64 <= self.config["input_size"] <= 128 or self.config["input_size"] % 16 != 0:
            logger.error("Config error: input_size must be between 64 and 128 and be divisible by "
                         "16.")
            exit(1)
        if not 64 <= self.config["output_size"] <= 256 or self.config["output_size"] % 32 != 0:
            logger.error("Config error: output_size must be between 64 and 256 and be divisible "
                         "by 32.")
            exit(1)
        logger.debug("Input and output sizes are valid")

    def get_dense_width_upscalers_numbers(self):
        """ Return the dense width and number of upscalers """
        output_size = self.config["output_size"]
        sides = [(output_size // 2**n, n) for n in [4, 5] if (output_size // 2**n) < 10]
        closest = min([x * self._downscale_ratio for x, _ in sides],
                      key=lambda x: abs(x - self.config["input_size"]))
        dense_width, upscalers_no = [(s, n) for s, n in sides
                                     if s * self._downscale_ratio == closest][0]
        logger.debug("dense_width: %s, upscalers_no: %s", dense_width, upscalers_no)
        return dense_width, upscalers_no

    def add_networks(self):
        """ Add the realface model weights """
        logger.debug("Adding networks")
        self.add_network("decoder", "a", self.decoder_a(), is_output=True)
        self.add_network("decoder", "b", self.decoder_b(), is_output=True)
        self.add_network("encoder", None, self.encoder())
        logger.debug("Added networks")

    def build_autoencoders(self, inputs):
        """ Initialize realface model """
        logger.debug("Initializing model")
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

        encoder_complexity = self.config["complexity_encoder"]

        for idx in range(self.downscalers_no - 1):
            var_x = self.blocks.conv(var_x, encoder_complexity * 2**idx)
            var_x = self.blocks.res_block(var_x, encoder_complexity * 2**idx, use_bias=True)
            var_x = self.blocks.res_block(var_x, encoder_complexity * 2**idx, use_bias=True)

        var_x = self.blocks.conv(var_x, encoder_complexity * 2**(idx + 1))

        return KerasModel(input_, var_x)

    def decoder_b(self):
        """ RealFace Decoder Network """
        input_filters = self.config["complexity_encoder"] * 2**(self.downscalers_no-1)
        input_width = self.config["input_size"] // self._downscale_ratio
        input_ = Input(shape=(input_width, input_width, input_filters))

        var_xy = input_

        var_xy = Dense(self.config["dense_nodes"])(Flatten()(var_xy))
        var_xy = Dense(self.dense_width * self.dense_width * self.dense_filters)(var_xy)
        var_xy = Reshape((self.dense_width, self.dense_width, self.dense_filters))(var_xy)
        var_xy = self.blocks.upscale(var_xy, self.dense_filters)

        var_x = var_xy
        var_x = self.blocks.res_block(var_x, self.dense_filters, use_bias=False)

        decoder_b_complexity = self.config["complexity_decoder"]
        for idx in range(self.upscalers_no - 2):
            var_x = self.blocks.upscale(var_x, decoder_b_complexity // 2**idx)
            var_x = self.blocks.res_block(var_x, decoder_b_complexity // 2**idx, use_bias=False)
            var_x = self.blocks.res_block(var_x, decoder_b_complexity // 2**idx, use_bias=True)
        var_x = self.blocks.upscale(var_x, decoder_b_complexity // 2**(idx + 1))

        var_x = self.blocks.conv2d(var_x, 3,
                                   kernel_size=5,
                                   padding="same",
                                   activation="sigmoid",
                                   name="face_out")

        outputs = [var_x]

        if self.config.get("learn_mask", False):
            var_y = var_xy
            mask_b_complexity = 384
            for idx in range(self.upscalers_no-2):
                var_y = self.blocks.upscale(var_y, mask_b_complexity // 2**idx)
            var_y = self.blocks.upscale(var_y, mask_b_complexity // 2**(idx + 1))

            var_y = self.blocks.conv2d(var_y, 1,
                                       kernel_size=5,
                                       padding="same",
                                       activation="sigmoid",
                                       name="mask_out")

            outputs += [var_y]

        return KerasModel(input_, outputs=outputs)

    def decoder_a(self):
        """ RealFace Decoder (A) Network """
        input_filters = self.config["complexity_encoder"] * 2**(self.downscalers_no-1)
        input_width = self.config["input_size"] // self._downscale_ratio
        input_ = Input(shape=(input_width, input_width, input_filters))

        var_xy = input_

        dense_nodes = int(self.config["dense_nodes"]/1.5)
        dense_filters = int(self.dense_filters/1.5)

        var_xy = Dense(dense_nodes)(Flatten()(var_xy))
        var_xy = Dense(self.dense_width * self.dense_width * dense_filters)(var_xy)
        var_xy = Reshape((self.dense_width, self.dense_width, dense_filters))(var_xy)

        var_xy = self.blocks.upscale(var_xy, dense_filters)

        var_x = var_xy
        var_x = self.blocks.res_block(var_x, dense_filters, use_bias=False)

        decoder_a_complexity = int(self.config["complexity_decoder"] / 1.5)
        for idx in range(self.upscalers_no-2):
            var_x = self.blocks.upscale(var_x, decoder_a_complexity // 2**idx)
        var_x = self.blocks.upscale(var_x, decoder_a_complexity // 2**(idx + 1))

        var_x = self.blocks.conv2d(var_x, 3,
                                   kernel_size=5,
                                   padding="same",
                                   activation="sigmoid",
                                   name="face_out")

        outputs = [var_x]

        if self.config.get("learn_mask", False):
            var_y = var_xy
            mask_a_complexity = 384
            for idx in range(self.upscalers_no-2):
                var_y = self.blocks.upscale(var_y, mask_a_complexity // 2**idx)
            var_y = self.blocks.upscale(var_y, mask_a_complexity // 2**(idx + 1))

            var_y = self.blocks.conv2d(var_y, 1,
                                       kernel_size=5,
                                       padding="same",
                                       activation="sigmoid",
                                       name="mask_out")

            outputs += [var_y]

        return KerasModel(input_, outputs=outputs)
