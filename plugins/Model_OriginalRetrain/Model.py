# Based on the original https://www.reddit.com/r/deepfakes/ code sample + contribs

from keras.models import Model as KerasModel
from keras.layers import Input, Dense, Flatten, Reshape, add
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras_contrib.losses import DSSIMObjective

from .AutoEncoder import AutoEncoder
from lib.PixelShuffler import PixelShuffler

IMAGE_SHAPE = (64, 64, 3)
ENCODER_DIM = 1024

conv_init = RandomNormal(0, 0.02)

class Model(AutoEncoder):
    def initModel(self):
        optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)
        x = Input(shape=IMAGE_SHAPE)

        encoder = self.encoder
        encoder.trainable = False
        self.autoencoder_B = KerasModel(x, self.decoder_B(encoder(x)))

        #self.autoencoder_B.compile(optimizer=optimizer, loss='mean_absolute_error')
        self.autoencoder_B.compile(optimizer=optimizer, loss=DSSIMObjective(kernel_size=3))

    def converter(self, swap):
        autoencoder = self.autoencoder_B
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
    
    def res_block(self, input_tensor, f):
        x = input_tensor
        x = Conv2D(f, kernel_size=3, kernel_initializer=conv_init, use_bias=False, padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(f, kernel_size=3, kernel_initializer=conv_init, use_bias=False, padding="same")(x)
        x = add([x, input_tensor])
        x = LeakyReLU(alpha=0.2)(x)
        return x

    def Encoder(self):
        input_ = Input(shape=IMAGE_SHAPE)
        x = input_
        x = self.conv(128)(x)
        x = self.conv(256)(x)
        x = self.conv(512)(x)
        x = self.conv(1024)(x)
        x = Dense(ENCODER_DIM)(Flatten()(x))
        x = Dense(4 * 4 * 1024)(x)
        x = Reshape((4, 4, 1024))(x)
        x = self.upscale(512)(x)
        return KerasModel(input_, x)

    def Decoder(self):
        input_ = Input(shape=(8, 8, 512))
        x = input_
        x = self.upscale(256)(x)
        x = self.upscale(128)(x)
        x = self.upscale(64)(x)
        x = self.upscale(32)(x)
        x = self.upscale(16)(x)
        x = self.res_block(x, 16)
        x = self.res_block(x, 16)
        x = self.res_block(x, 16)
        x = self.res_block(x, 16)
        x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)
        return KerasModel(input_, x)