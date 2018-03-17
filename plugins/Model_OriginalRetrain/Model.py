# Based on the original https://www.reddit.com/r/deepfakes/ code sample + contribs

from keras.models import Model as KerasModel
from keras.layers import Input, Dense, Flatten, Reshape, Conv2DTranspose, Activation
from keras.layers.merge import concatenate, add
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.layers import MaxPool2D
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras import backend as K

from .AutoEncoder import AutoEncoder
from lib.PixelShuffler import PixelShuffler

IMAGE_SHAPE = (64, 64, 3)
ENCODER_DIM = 1024

conv_init = RandomNormal(0, 0.02)

def ground_truth_diff(y_true, y_pred):
    return K.abs(y_pred - y_true)

class Model(AutoEncoder):
    def initModel(self):
        optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)
        x = Input(shape=IMAGE_SHAPE)

        encoder = self.encoder
        encoder.trainable = False
        self.autoencoder_B = KerasModel(x, self.decoder_B(encoder(x)))

        #self.autoencoder_B.compile(optimizer=optimizer, loss='mean_absolute_error')
        self.autoencoder_B.compile(optimizer=optimizer, loss='mse')

        encoder.summary()
        self.autoencoder_B.summary()

        from keras.utils import plot_model
        plot_model(self.encoder, to_file='_model_encoder.png', show_shapes=True, show_layer_names=True)
        plot_model(self.decoder_B, to_file='_model_decoder_B.png', show_shapes=True, show_layer_names=True)

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
        x = model_EED(x)
        x = Conv2D(3, kernel_size=1, padding='same', activation="tanh")(x)
        return KerasModel(input_, x)


#from https://github.com/MarkPrecursor/EEDS-keras/blob/master/EED.py
#Note that original EED is run on YCrCb images
def Res_block(filters):
    _input = Input(shape=(None, None, filters))

    conv = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(_input)
    conv = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='linear')(conv)

    out = add(inputs=[_input, conv])
    out = Activation('relu')(out)

    model = KerasModel(inputs=_input, outputs=out)

    return model


def model_EED(Feature):

    # Feature = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    Feature_out = Res_block(32)(Feature)

    # Upsampling
    Upsampling1 = Conv2D(filters=4, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(Feature_out)
    Upsampling2 = Conv2DTranspose(filters=4, kernel_size=(14, 14), strides=(2, 2),
                                  padding='same', activation='relu')(Upsampling1)
    Upsampling3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(Upsampling2)

    # Mulyi-scale Reconstruction
    Reslayer1 = Res_block(64)(Upsampling3)

    Reslayer2 = Res_block(64)(Reslayer1)

    # ***************//
    Multi_scale1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(Reslayer2)

    Multi_scale2a = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1),
                           padding='same', activation='relu')(Multi_scale1)

    Multi_scale2b = Conv2D(filters=16, kernel_size=(1, 3), strides=(1, 1),
                           padding='same', activation='relu')(Multi_scale1)
    Multi_scale2b = Conv2D(filters=16, kernel_size=(3, 1), strides=(1, 1),
                           padding='same', activation='relu')(Multi_scale2b)

    Multi_scale2c = Conv2D(filters=16, kernel_size=(1, 5), strides=(1, 1),
                           padding='same', activation='relu')(Multi_scale1)
    Multi_scale2c = Conv2D(filters=16, kernel_size=(5, 1), strides=(1, 1),
                           padding='same', activation='relu')(Multi_scale2c)

    Multi_scale2d = Conv2D(filters=16, kernel_size=(1, 7), strides=(1, 1),
                           padding='same', activation='relu')(Multi_scale1)
    Multi_scale2d = Conv2D(filters=16, kernel_size=(7, 1), strides=(1, 1),
                           padding='same', activation='relu')(Multi_scale2d)

    Multi_scale2 = concatenate(inputs=[Multi_scale2a, Multi_scale2b, Multi_scale2c, Multi_scale2d])

    #out = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(Multi_scale2)

    return Multi_scale2
