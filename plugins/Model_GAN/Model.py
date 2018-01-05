# Based on the https://github.com/shaoanlu/faceswap-GAN repo (master/FaceSwap_GAN_v2_train.ipynb)

from keras.models import Model as KerasModel
from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu
from keras.initializers import RandomNormal
from keras.applications import *
import tensorflow as tf
import numpy as np

from lib.PixelShuffler import PixelShuffler
from lib.utils import ensure_file_exists
from .Trainable import Trainable

encoderH5 = 'encoder_GAN.h5'
decoder_AH5 = 'decoder_A_GAN.h5'
decoder_BH5 = 'decoder_B_GAN.h5'
netDAH5 = 'netDA_GAN.h5'
netDBH5 = 'netDB_GAN.h5'

#channel_axis=-1
#channel_first = False

nc_in = 3 # number of input channels of generators
nc_D_inp = 6 # number of input channels of discriminators

# # 5. Define models

def __conv_init(a):
    print("conv_init", a)
    k = RandomNormal(0, 0.02)(a) # for convolution kernel
    k.conv_weight = True    
    return k

conv_init = RandomNormal(0, 0.02)
gamma_init = RandomNormal(1., 0.02) # for batch normalization

#def batchnorm():
#    return BatchNormalization(momentum=0.9, axis=channel_axis, epsilon=1.01e-5, gamma_initializer = gamma_init)

def conv_block(input_tensor, f):
    x = input_tensor
    x = Conv2D(f, kernel_size=3, strides=2, kernel_initializer=conv_init, use_bias=False, padding="same")(x)
    x = Activation("relu")(x)
    return x

def conv_block_d(input_tensor, f, use_instance_norm=True):
    x = input_tensor
    x = Conv2D(f, kernel_size=4, strides=2, kernel_initializer=conv_init, use_bias=False, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    return x

def res_block(input_tensor, f):
    x = input_tensor
    x = Conv2D(f, kernel_size=3, kernel_initializer=conv_init, use_bias=False, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(f, kernel_size=3, kernel_initializer=conv_init, use_bias=False, padding="same")(x)
    x = add([x, input_tensor])
    x = LeakyReLU(alpha=0.2)(x)
    return x

# Legacy
#def upscale_block(input_tensor, f):
#    x = input_tensor
#    x = Conv2DTranspose(f, kernel_size=3, strides=2, use_bias=False, kernel_initializer=conv_init)(x) 
#    x = LeakyReLU(alpha=0.2)(x)
#    return x

def upscale_ps(filters, use_norm=True):
    def block(x):
        x = Conv2D(filters*4, kernel_size=3, use_bias=False, kernel_initializer=RandomNormal(0, 0.02), padding='same' )(x)
        x = LeakyReLU(0.1)(x)
        x = PixelShuffler()(x)
        return x
    return block

def Discriminator(nc_in, input_size=64):
    inp = Input(shape=(input_size, input_size, nc_in))
    #x = GaussianNoise(0.05)(inp)
    x = conv_block_d(inp, 64, False)
    x = conv_block_d(x, 128, False)
    x = conv_block_d(x, 256, False)
    out = Conv2D(1, kernel_size=4, kernel_initializer=conv_init, use_bias=False, padding="same", activation="sigmoid")(x)   
    return KerasModel(inputs=[inp], outputs=out)

def Encoder(nc_in=3, input_size=64):
    inp = Input(shape=(input_size, input_size, nc_in))
    x = Conv2D(64, kernel_size=5, kernel_initializer=conv_init, use_bias=False, padding="same")(inp)
    x = conv_block(x,128)
    x = conv_block(x,256)
    x = conv_block(x,512) 
    x = conv_block(x,1024)
    x = Dense(1024)(Flatten()(x))
    x = Dense(4*4*1024)(x)
    x = Reshape((4, 4, 1024))(x)
    out = upscale_ps(512)(x)
    return KerasModel(inputs=inp, outputs=out)

# Legacy, left for someone to try if interested
#def Decoder(nc_in=512, input_size=8):
#    inp = Input(shape=(input_size, input_size, nc_in))   
#    x = upscale_block(inp, 256)
#    x = Cropping2D(((0,1),(0,1)))(x)
#    x = upscale_block(x, 128)
#    x = res_block(x, 128)
#    x = Cropping2D(((0,1),(0,1)))(x)
#    x = upscale_block(x, 64)
#    x = res_block(x, 64)
#    x = res_block(x, 64)
#    x = Cropping2D(((0,1),(0,1)))(x)
#    x = Conv2D(3, kernel_size=5, kernel_initializer=conv_init, use_bias=False, padding="same")(x)
#    out = Activation("tanh")(x)
#    return KerasModel(inputs=inp, outputs=out)

def Decoder_ps(nc_in=512, input_size=8):
    input_ = Input(shape=(input_size, input_size, nc_in))
    x = input_
    x = upscale_ps(256)(x)
    x = upscale_ps(128)(x)
    x = upscale_ps(64)(x)
    x = res_block(x, 64)
    x = res_block(x, 64)
    #x = Conv2D(4, kernel_size=5, padding='same')(x)   
    alpha = Conv2D(1, kernel_size=5, padding='same', activation="sigmoid")(x)
    rgb = Conv2D(3, kernel_size=5, padding='same', activation="tanh")(x)
    out = concatenate([alpha, rgb])
    return KerasModel(input_, out )    

class Model():
    IMAGE_SHAPE = (64, 64, 3)

    def __init__(self, model_dir):

        self.model_dir = model_dir

        self.encoder = Encoder()
        self.decoder_A = Decoder_ps()
        self.decoder_B = Decoder_ps()

        x = Input(shape=self.IMAGE_SHAPE)

        self.netGA = KerasModel(x, self.decoder_A(self.encoder(x)))
        self.netGB = KerasModel(x, self.decoder_B(self.encoder(x)))

        self.netDA = Discriminator(nc_D_inp)
        self.netDB = Discriminator(nc_D_inp)

    def converter(self, swap):
        converter = None
        if not swap:
            converter = Trainable(self.netGB, self.netDB, self.IMAGE_SHAPE)
        else:
            converter = Trainable(self.netGA, self.netDA, self.IMAGE_SHAPE)

        return lambda img : converter.swap_face(img)

    def load(self, swapped):
        self.encoder.load_weights(ensure_file_exists(self.model_dir, encoderH5))
        self.decoder_A.load_weights(ensure_file_exists(self.model_dir, decoder_AH5))
        self.decoder_B.load_weights(ensure_file_exists(self.model_dir, decoder_BH5))
        self.netDA.load_weights(ensure_file_exists(self.model_dir, netDAH5))
        self.netDB.load_weights(ensure_file_exists(self.model_dir, netDBH5))
        print ("model loaded.")

    def save_weights(self):
        self.encoder.save_weights(ensure_file_exists(self.model_dir, encoderH5))
        self.decoder_A.save_weights(ensure_file_exists(self.model_dir, decoder_AH5))
        self.decoder_B.save_weights(ensure_file_exists(self.model_dir, decoder_BH5))
        self.netDA.save_weights(ensure_file_exists(self.model_dir, netDAH5))
        self.netDB.save_weights(ensure_file_exists(self.model_dir, netDBH5))
