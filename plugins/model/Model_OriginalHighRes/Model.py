#!/usr/bin/python3

# Based on the original https://www.reddit.com/r/deepfakes/ code sample + contribs
# Based on https://github.com/iperov/OpenDeepFaceSwap for Decoder multiple res block chain
# Based on the https://github.com/shaoanlu/faceswap-GAN repo
# source : https://github.com/shaoanlu/faceswap-GAN/blob/master/FaceSwap_GAN_v2_sz128_train.ipynbtemp/faceswap_GAN_keras.ipynb


import enum
import logging
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from keras.initializers import RandomNormal
from keras.layers import Input, Dense, Flatten, Reshape
from keras.layers import SeparableConv2D, add
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.models import Model as KerasModel
from keras.optimizers import Adam
from keras.utils import multi_gpu_model

from lib.PixelShuffler import PixelShuffler
import lib.Serializer
from lib.utils import backup_file

from . import __version__
from .instance_normalization import InstanceNormalization


if isinstance(__version__, (list, tuple)):
    version_str = ".".join([str(n) for n in __version__[1:]])
else:
    version_str = __version__

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
mswindows = sys.platform=="win32"


class EncoderType(enum.Enum):
    ORIGINAL = "original"
    SHAOANLU = "shaoanlu"


_kern_init = RandomNormal(0, 0.02)


def inst_norm():
    return InstanceNormalization()


ENCODER = EncoderType.ORIGINAL


hdf = {'encoderH5': 'encoder_{version_str}{ENCODER.value}.h5'.format(**vars()),
       'decoder_AH5': 'decoder_A_{version_str}{ENCODER.value}.h5'.format(**vars()),
       'decoder_BH5': 'decoder_B_{version_str}{ENCODER.value}.h5'.format(**vars())}

class Model():

    ENCODER_DIM = 1024 # dense layer size
    IMAGE_SHAPE = 128, 128 # image shape

    assert [n for n in IMAGE_SHAPE if n>=16]

    IMAGE_WIDTH = max(IMAGE_SHAPE)
    IMAGE_WIDTH = (IMAGE_WIDTH//16 + (1 if (IMAGE_WIDTH%16)>=8 else 0))*16
    IMAGE_SHAPE = IMAGE_WIDTH, IMAGE_WIDTH, len('BRG') # good to let ppl know what these are...


    def __init__(self, model_dir, gpus, encoder_type=ENCODER):

        if mswindows:
            from ctypes import cdll
            mydll = cdll.LoadLibrary("user32.dll")
            mydll.SetProcessDPIAware(True)

        self._encoder_type = encoder_type

        self.model_dir = model_dir

        # can't chnage gpu's when the model is initialized no point in making it r/w
        self._gpus = gpus

        Encoder = getattr(self, "Encoder_{}".format(self._encoder_type.value))
        Decoder = getattr(self, "Decoder_{}".format(self._encoder_type.value))

        self.encoder = Encoder()
        self.decoder_A = Decoder()
        self.decoder_B = Decoder()

        self.initModel()


    def initModel(self):
        optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)

        x = Input(shape=self.IMAGE_SHAPE)

        self.autoencoder_A = KerasModel(x, self.decoder_A(self.encoder(x)))
        self.autoencoder_B = KerasModel(x, self.decoder_B(self.encoder(x)))

        if self.gpus > 1:
            self.autoencoder_A = multi_gpu_model( self.autoencoder_A , self.gpus)
            self.autoencoder_B = multi_gpu_model( self.autoencoder_B , self.gpus)


        self.autoencoder_A.compile(optimizer=optimizer, loss='mean_absolute_error')
        self.autoencoder_B.compile(optimizer=optimizer, loss='mean_absolute_error')


    def load(self, swapped):
        model_dir = str(self.model_dir)

        from json import JSONDecodeError
        face_A, face_B = (hdf['decoder_AH5'], hdf['decoder_BH5']) if not swapped else (hdf['decoder_BH5'], hdf['decoder_AH5'])

        state_dir = os.path.join(model_dir, 'state_{version_str}_{ENCODER.value}.json'.format(**globals()))
        ser = lib.Serializer.get_serializer('json')

        try:
            with open(state_dir, 'rb') as fp:
                state = ser.unmarshal(fp.read().decode('utf-8'))
                self._epoch_no = state['epoch_no']
        except IOError as e:
            logger.warning('Error loading training info: %s', str(e.strerror))
            self._epoch_no = 0
        except JSONDecodeError as e:
            logger.warning('Error loading training info: %s', str(e.msg))
            self._epoch_no = 0

        try:
            self.encoder.load_weights(os.path.join(model_dir, hdf['encoderH5']))
            self.decoder_A.load_weights(os.path.join(model_dir, face_A))
            self.decoder_B.load_weights(os.path.join(model_dir, face_B))
            logger.info('loaded model weights')
            return True
        except IOError as e:
            logger.warning('Error loading training info: %s', str(e.strerror))
        except Exception as e:
            logger.warning('Error loading training info: %s', str(e))

        return False

    def converter(self, swap):
        autoencoder = self.autoencoder_B if not swap else self.autoencoder_A
        return autoencoder.predict


    def conv(self, filters, kernel_size=5, strides=2, **kwargs):
        def block(x):
            x = Conv2D(filters, kernel_size=kernel_size, strides=strides, kernel_initializer=_kern_init, padding='same', **kwargs)(x)
            x = LeakyReLU(0.1)(x)
            return x
        return block

    def conv_sep(self, filters, kernel_size=5, strides=2, use_instance_norm=True, **kwargs):
        def block(x):
            x = SeparableConv2D(filters, kernel_size=kernel_size, strides=strides, kernel_initializer=_kern_init, padding='same', **kwargs)(x)
            x = Activation("relu")(x)
            return x
        return block

    def conv_inst_norm(self, filters, kernel_size=3, strides=2, use_instance_norm=True, **kwargs):
        def block(x):
            x = SeparableConv2D(filters, kernel_size=kernel_size, strides=strides, kernel_initializer=_kern_init, padding='same', **kwargs)(x)
            if use_instance_norm:
                x = inst_norm()(x)
            x = Activation("relu")(x)
            return x
        return block

    def upscale(self, filters, **kwargs):
        def block(x):
            x = Conv2D(filters * 4, kernel_size=3, padding='same',
                       kernel_initializer=_kern_init)(x)
            x = LeakyReLU(0.1)(x)
            x = PixelShuffler()(x)
            return x
        return block

    def upscale_inst_norm(self, filters, use_instance_norm=True, **kwargs):
        def block(x):
            x = Conv2D(filters*4, kernel_size=3, use_bias=False,
                       kernel_initializer=_kern_init, padding='same', **kwargs)(x)
            if use_instance_norm:
                x = inst_norm()(x)
            x = LeakyReLU(0.1)(x)
            x = PixelShuffler()(x)
            return x
        return block

    def Encoder_original(self, **kwargs):
        impt = Input(shape=self.IMAGE_SHAPE)

        in_conv_filters = self.IMAGE_SHAPE[0] if self.IMAGE_SHAPE[0] <= 128 else 128 + (self.IMAGE_SHAPE[0]-128)//4

        x = self.conv(in_conv_filters)(impt)
        x = self.conv_sep(256)(x)
        x = self.conv(512)(x)
        x = self.conv_sep(1024)(x)

        dense_shape = self.IMAGE_SHAPE[0] // 16
        x = Dense(self.ENCODER_DIM, kernel_initializer=_kern_init)(Flatten()(x))
        x = Dense(dense_shape * dense_shape * 512, kernel_initializer=_kern_init)(x)
        x = Reshape((dense_shape, dense_shape, 512))(x)
        x = self.upscale(512)(x)

        return KerasModel(impt, x, **kwargs)


    def Encoder_shaoanlu(self, **kwargs):
        impt = Input(shape=self.IMAGE_SHAPE)

        in_conv_filters = self.IMAGE_SHAPE[0] if self.IMAGE_SHAPE[0] <= 128 else 128 + (self.IMAGE_SHAPE[0]-128)//4

        x = Conv2D(in_conv_filters, kernel_size=5, use_bias=False, padding="same")(impt)
        x = self.conv_inst_norm(in_conv_filters+32, use_instance_norm=False)(x)
        x = self.conv_inst_norm(256)(x)
        x = self.conv_inst_norm(512)(x)
        x = self.conv_inst_norm(1024)(x)

        dense_shape = self.IMAGE_SHAPE[0] // 16
        x = Dense(self.ENCODER_DIM, kernel_initializer=_kern_init)(Flatten()(x))
        x = Dense(dense_shape * dense_shape * 768, kernel_initializer=_kern_init)(x)
        x = Reshape((dense_shape, dense_shape, 768))(x)
        x = self.upscale(512)(x)

        return KerasModel(impt, x, **kwargs)


    def Decoder_original(self):
        decoder_shape = self.IMAGE_SHAPE[0]//8
        inpt = Input(shape=(decoder_shape, decoder_shape, 512))

        x = self.upscale(384)(inpt)
        x = self.upscale(256-32)(x)
        x = self.upscale(self.IMAGE_SHAPE[0])(x)

        x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)

        return KerasModel(inpt, x)


    def Decoder_shaoanlu(self):
        decoder_shape = self.IMAGE_SHAPE[0]//8
        inpt = Input(shape=(decoder_shape, decoder_shape, 512))

        x = self.upscale_inst_norm(512)(inpt)
        x = self.upscale_inst_norm(256)(x)
        x = self.upscale_inst_norm(self.IMAGE_SHAPE[0])(x)

        x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)

        return KerasModel(inpt, x)


    def save_weights(self):
        model_dir = str(self.model_dir)

        try:
            for model in hdf.values():
                backup_file(model_dir, model)
        except NameError:
            logger.error('backup functionality not available\n')

        state_dir = os.path.join(model_dir, 'state_{version_str}_{ENCODER.value}.json'.format(**globals()))
        ser = lib.Serializer.get_serializer('json')
        try:
            with open(state_dir, 'wb') as fp:
                state_json = ser.marshal({
                    'epoch_no' : self._epoch_no
                     })
                fp.write(state_json.encode('utf-8'))
        except IOError as e:
            logger.error(e.strerror)

        logger.info('saving model weights')

        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(getattr(self, mdl_name.rstrip('H5')).save_weights, str(self.model_dir / mdl_H5_fn)) for mdl_name, mdl_H5_fn in hdf.items()]
            for future in as_completed(futures):
                future.result()
                print('.', end='', flush=True)

        logger.info('done')


    @property
    def gpus(self):
        return self._gpus

    @property
    def model_name(self):
        try:
            return self._model_name
        except AttributeError:
            import inspect
            self._model_name = os.path.dirname(inspect.getmodule(self).__file__).rsplit("_", 1)[1]
        return self._model_name


    def __str__(self):
        return "<{}: ver={}, dense_dim={}, img_size={}>".format(self.model_name,
                                                              version_str,
                                                              self.ENCODER_DIM,
                                                              "x".join([str(n) for n in self.IMAGE_SHAPE[:2]]))
