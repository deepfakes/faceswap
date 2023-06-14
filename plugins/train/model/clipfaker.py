#!/usr/bin/env python3
""" DFaker Model
    Based on the dfaker model: https://github.com/dfaker """
import logging
import sys

from lib.model.nn_blocks import Conv2DOutput, UpscaleBlock, ResidualBlock
from tensorflow import keras
from tensorflow.keras.layers import Dense, Reshape, Input
from lib.utils import get_backend
from .original import Model as OriginalModel, KerasModel
from lib.model.networks.clip.model import _Models, ClipConfig
from lib.model.networks.clip.visual_transformer import VisualTransformer
from lib.utils import GetModel
import numpy as np

if get_backend() == "amd":
    from keras.initializers import RandomNormal  # pylint:disable=no-name-in-module
    from keras.layers import Input, LeakyReLU
else:
    # Ignore linting errors from Tensorflow's thoroughly broken import system
    from tensorflow.keras.initializers import RandomNormal  # noqa pylint:disable=import-error,no-name-in-module
    from tensorflow.keras.layers import Input, LeakyReLU  # noqa pylint:disable=import-error,no-name-in-module

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class Model(OriginalModel):
    """ Clipfaker Model """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._output_size: int = self.config["output_size"]
        if self._output_size not in (128, 256):
            logger.error("Clipfaker output shape should be 128 or 256 px")
            sys.exit(1)
        self.input_shape: tuple [int, int, int] = (224, 224, 3)
        self.encoder_dim: int = 512
        self.kernel_initializer = RandomNormal(0, 0.02)
        clipconfig: ClipConfig = _Models['FaRL-B_16-64']
        self.visualtransformer = VisualTransformer(
                        input_resolution=clipconfig.image_resolution,
                        patch_size=clipconfig.vision_patch_size,
                        width=clipconfig.vision_width,
                        layers=clipconfig.vision_layers,
                        heads=clipconfig.vision_width//64,
                        output_dim=clipconfig.embed_dim,
                        name="visual")()
        # Used to temporarily load FaRL weights
        empty_image = np.zeros((1, 224, 224, 3))
        self.visualtransformer(empty_image)
        # self.visualtransformer.summary()
        
        model_downloader = GetModel("s3fd_keras_v2.h5", 11)
        model_downloader._model_filename = ['FaRL_v1.h5']
        model_downloader._git_model_id = 1
        model_downloader._url_base = "https://github.com/Arkavian/faceswap-models/releases/download"
        model_downloader._get()
        self.visualtransformer.load_weights(model_downloader.model_path, by_name=True)

        # model_downloader._change_base("https://github.com/Arkavian/faceswap-models/releases/download")

        # model_downloader._change_base("Arkavian","faceswap-models")

        # self.visualtransformer.load_weights('/home/nikkelitous/FaRL_Visual.h5', by_name=True, skip_mismatch=True)
        # self.visualtransformer.load_weights('/home/nikkelitous/FaRL_Visual_update.h5', by_name=True)
        self.visualtransformer.trainable = False
        # self.visualtransformer.save_weights('/home/nikkelitous/FaRL_Visual_output.h5')

    def decoder(self, side):
        """ Decoder Network """
        input_ = Input(shape=(8, 8, 512))
        var_x = input_

        if self._output_size == 256:
            var_x = UpscaleBlock(1024, activation=None)(var_x)
            var_x = LeakyReLU(alpha=0.2)(var_x)
            var_x = ResidualBlock(1024, kernel_initializer=self.kernel_initializer)(var_x)
        var_x = UpscaleBlock(512, activation=None)(var_x)
        var_x = LeakyReLU(alpha=0.2)(var_x)
        var_x = ResidualBlock(512, kernel_initializer=self.kernel_initializer)(var_x)
        var_x = UpscaleBlock(256, activation=None)(var_x)
        var_x = LeakyReLU(alpha=0.2)(var_x)
        var_x = ResidualBlock(256, kernel_initializer=self.kernel_initializer)(var_x)
        var_x = UpscaleBlock(128, activation=None)(var_x)
        var_x = LeakyReLU(alpha=0.2)(var_x)
        var_x = ResidualBlock(128, kernel_initializer=self.kernel_initializer)(var_x)
        var_x = UpscaleBlock(64, activation="leakyrelu")(var_x)
        var_x = Conv2DOutput(3, 5, name=f"face_out_{side}")(var_x)
        outputs = [var_x]

        if self.config.get("learn_mask", False):
            var_y = input_
            if self._output_size == 256:
                var_y = UpscaleBlock(1024, activation="leakyrelu")(var_y)
            var_y = UpscaleBlock(512, activation="leakyrelu")(var_y)
            var_y = UpscaleBlock(256, activation="leakyrelu")(var_y)
            var_y = UpscaleBlock(128, activation="leakyrelu")(var_y)
            var_y = UpscaleBlock(64, activation="leakyrelu")(var_y)
            var_y = Conv2DOutput(1, 5, name=f"mask_out_{side}")(var_y)
            outputs.append(var_y)
        return KerasModel([input_], outputs=outputs, name=f"decoder_{side}")

    def encoder(self):
        input_ = Input(shape=(224, 224, 3))
        var_x = input_

        var_x = self.visualtransformer(var_x)
        var_x = Dense(4 * 4 * 1024)(var_x)
        var_x = Reshape((4, 4, 1024))(var_x)
        var_x = UpscaleBlock(512, activation=None)(var_x)
        outputs = [var_x]
        return KerasModel([input_], outputs=outputs, name=f"encoder")
