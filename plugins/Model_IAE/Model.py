# Improved autoencoder for faceswap.

from keras.models import Model as KerasModel
from keras.layers import Input, Dense, Flatten, Reshape, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam

from .AutoEncoder import AutoEncoder
from lib.PixelShuffler import PixelShuffler

from keras.utils import multi_gpu_model

IMAGE_SHAPE = (64, 64, 3)
ENCODER_DIM = 1024

class Model(AutoEncoder):
    def initModel(self):
        optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)
        x = Input(shape=IMAGE_SHAPE)

        self.autoencoder_A = KerasModel(x, self.decoder(Concatenate()([self.inter_A(self.encoder(x)), self.inter_both(self.encoder(x))])))
        self.autoencoder_B = KerasModel(x, self.decoder(Concatenate()([self.inter_B(self.encoder(x)), self.inter_both(self.encoder(x))])))

        if self.gpus > 1:
            self.autoencoder_A = multi_gpu_model( self.autoencoder_A , self.gpus)
            self.autoencoder_B = multi_gpu_model( self.autoencoder_B , self.gpus)

        self.autoencoder_A.compile(optimizer=optimizer, loss='mean_absolute_error')
        self.autoencoder_B.compile(optimizer=optimizer, loss='mean_absolute_error')

    def converter(self, swap):
        autoencoder = self.autoencoder_B if not swap else self.autoencoder_A
        return lambda img: autoencoder.predict(img)

    def conv(self, filters):
        def block(x):
            x = Conv2D(filters, kernel_size=5, strides=2, padding='same')(x)
            x = LeakyReLU(0.1)(x)
            return x
        return block

    def upscale(self, filters):
        def block(x):
            x = Conv2D(filters * 4, kernel_size=3, padding='same')(x)
            x = LeakyReLU(0.1)(x)
            x = PixelShuffler()(x)
            return x
        return block

    def Encoder(self):
        input_ = Input(shape=IMAGE_SHAPE)
        x = input_
        x = self.conv(128)(x)
        x = self.conv(256)(x)
        x = self.conv(512)(x)
        x = self.conv(1024)(x)
        x = Flatten()(x)
        return KerasModel(input_, x)

    def Intermidiate(self):
        input_ = Input(shape=(None, 4 * 4 * 1024))
        x = input_
        x = Dense(ENCODER_DIM)(x)
        x = Dense(4 * 4 * int(ENCODER_DIM/2))(x)
        x = Reshape((4, 4, int(ENCODER_DIM/2)))(x)
        return KerasModel(input_, x)

    def Decoder(self):
        input_ = Input(shape=(4, 4, ENCODER_DIM))
        x = input_
        x = self.upscale(512)(x)
        x = self.upscale(256)(x)
        x = self.upscale(128)(x)
        x = self.upscale(64)(x)
        x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)
        return KerasModel(input_, x)
