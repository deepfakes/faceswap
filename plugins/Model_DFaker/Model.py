from keras.models import Model as KerasModel
from keras.layers import Input, Dense, Flatten, Reshape, Dropout, Add,Concatenate, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.initializers import RandomNormal
from keras.optimizers import Adam

from .AutoEncoder import AutoEncoder
from lib.PixelShuffler import PixelShuffler

import tensorflow as tf

from keras_contrib.losses import DSSIMObjective
from keras import losses

from keras.utils import multi_gpu_model
import numpy

class penalized_loss(object):

  def __init__(self,mask,lossFunc,maskProp= 1.0):
    self.mask = mask
    self.lossFunc=lossFunc
    self.maskProp = maskProp
    self.maskaskinvProp = 1-maskProp  

  def __call__(self,y_true, y_pred):

    tro, tgo, tbo = tf.split(y_true,3, 3 )
    pro, pgo, pbo = tf.split(y_pred,3, 3 )

    tr = tro
    tg = tgo
    tb = tbo

    pr = pro
    pg = pgo
    pb = pbo
    m  = self.mask 

    m   = m*self.maskProp
    m  += self.maskaskinvProp
    tr *= m
    tg *= m
    tb *= m

    pr *= m
    pg *= m
    pb *= m


    y = tf.concat([tr, tg, tb],3)
    p = tf.concat([pr, pg, pb],3)

    #yo = tf.stack([tro,tgo,tbo],3)
    #po = tf.stack([pro,pgo,pbo],3)

    return self.lossFunc(y,p)

IMAGE_SHAPE = (64,64,3)

ENCODER_DIM = 1024
conv_init = RandomNormal(0, 0.02)
gamma_init = RandomNormal(1., 0.02)

def __conv_init(a):
    print("conv_init", a)
    k = RandomNormal(0, 0.02)(a) # for convolution kernel
    k.conv_weight = True    
    return k

class Model(AutoEncoder):
    def initModel(self):
        optimizer = Adam( lr=5e-5, beta_1=0.5, beta_2=0.999 )

        # print(self.encoder.summary()) 
        # print(self.decoder_A.summary())
        # print(self.decoder_B.summary())

        x1 = Input( shape=IMAGE_SHAPE )
        x2 = Input( shape=IMAGE_SHAPE )
        m1 = Input( shape=(64*2,64*2,1) )
        m2 = Input( shape=(64*2,64*2,1) )

        self.autoencoder_A = KerasModel( [x1,m1], self.decoder_A( self.encoder(x1) ) )
        self.autoencoder_B = KerasModel( [x2,m2], self.decoder_B( self.encoder(x2) ) )

        # print(self.autoencoder_A.summary())
        # print(self.autoencoder_B.summary())

        if self.gpus > 1:
            self.autoencoder_A = multi_gpu_model( self.autoencoder_A ,self.gpus)
            self.autoencoder_B = multi_gpu_model( self.autoencoder_B ,self.gpus)

        # o1,om1  = self.decoder_A( self.encoder(x1))
        # o2,om2  = self.decoder_B( self.encoder(x2))

        DSSIM = DSSIMObjective()
        self.autoencoder_A.compile( optimizer=optimizer, loss=[ penalized_loss(m1, DSSIM),'mse'] )
        self.autoencoder_B.compile( optimizer=optimizer, loss=[ penalized_loss(m2, DSSIM),'mse'] )

    def converter(self, swap):
        zmask = numpy.zeros((1,128, 128,1),float)
        autoencoder = self.autoencoder_B if not swap else self.autoencoder_A 
        return lambda img: autoencoder.predict([img, zmask])

    # def upscale_ps(self, filters, use_norm=True):
    #     def block(x):
    #         x = Conv2D(filters*4, kernel_size=3, use_bias=False, kernel_initializer=RandomNormal(0, 0.02), padding='same' )(x)
    #         x = LeakyReLU(0.1)(x)
    #         x = PixelShuffler()(x)
    #         return x
    #     return block

    def res_block(self, input_tensor, f):
        x = input_tensor
        x = Conv2D(f, kernel_size=3, kernel_initializer=conv_init, use_bias=False, padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(f, kernel_size=3, kernel_initializer=conv_init, use_bias=False, padding="same")(x)
        x = Add()([x, input_tensor])
        x = LeakyReLU(alpha=0.2)(x)
        return x

    def conv( self, filters ):
        def block(x):
            x = Conv2D( filters, kernel_size=5, strides=2, padding='same' )(x)
            x = LeakyReLU(0.1)(x)
            return x
        return block

    def upscale( self, filters ):
        def block(x):
            x = Conv2D( filters*4, kernel_size=3, padding='same' )(x)
            x = LeakyReLU(0.1)(x)
            x = PixelShuffler()(x)
            return x
        return block

    def Encoder(self):
        input_ = Input( shape=IMAGE_SHAPE )
        x = self.conv( 128)(input_)
        x = self.conv( 256)(x)
        x = self.conv( 512)(x)

        x = self.conv(1024)(x)
        x = Dense( ENCODER_DIM )( Flatten()(x) )
        x = Dense(4*4*1024)(x)
        x = Reshape((4,4,1024))(x)
        x = self.upscale(512)(x)
        return KerasModel( input_, [x] )

    def Decoder(self, name):
        input_ = Input( shape=(8,8,512) )
        #skip_in = Input( shape=(8,8,512) )

        x = input_
        x = self.upscale(512)(x)
        x = self.res_block(x, 512)
        x = self.upscale(256)(x)
        x = self.res_block(x, 256)
        x = self.upscale(128)(x)
        x = self.res_block(x, 128)
        x = self.upscale(64)(x)
        x = Conv2D( 3, kernel_size=5, padding='same', activation='sigmoid' )(x)

        y = input_
        y = self.upscale(512)(y)
        y = self.upscale(256)(y)
        y = self.upscale(128)(y)
        y = self.upscale(64)(y)
        y = Conv2D( 1, kernel_size=5, padding='same', activation='sigmoid' )(y)

        return KerasModel( [input_], outputs=[x,y] )
