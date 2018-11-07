# Based on the https://github.com/shaoanlu/faceswap-GAN repo (master/temp/faceswap_GAN_keras.ipynb)

from keras.models import Model
from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu
from keras.initializers import RandomNormal
from keras.applications import *
from keras.optimizers import Adam

from lib.PixelShuffler import PixelShuffler
from .instance_normalization import InstanceNormalization
from lib.utils import backup_file

from keras.utils import multi_gpu_model

hdf = {'netGAH5': 'netGA_GAN.h5',
       'netGBH5': 'netGB_GAN.h5',
       'netDAH5': 'netDA_GAN.h5',
       'netDBH5': 'netDB_GAN.h5'}

def __conv_init(a):
    print("conv_init", a)
    k = RandomNormal(0, 0.02)(a) # for convolution kernel
    k.conv_weight = True
    return k

#def batchnorm():
#    return BatchNormalization(momentum=0.9, axis=channel_axis, epsilon=1.01e-5, gamma_initializer = gamma_init)

def inst_norm():
    return InstanceNormalization()

conv_init = RandomNormal(0, 0.02)
gamma_init = RandomNormal(1., 0.02) # for batch normalization

class GANModel():
    img_size = 64
    channels = 3
    img_shape = (img_size, img_size, channels)
    encoded_dim = 1024
    nc_in = 3 # number of input channels of generators
    nc_D_inp = 6 # number of input channels of discriminators

    def __init__(self, model_dir, gpus):
        self.model_dir = model_dir
        self.gpus = gpus

        optimizer = Adam(1e-4, 0.5)

        # Build and compile the discriminator
        self.netDA, self.netDB = self.build_discriminator()

        # Build and compile the generator
        self.netGA, self.netGB = self.build_generator()

    def converter(self, swap):
        predictor = self.netGB if not swap else self.netGA
        return lambda img: predictor.predict(img)

    def build_generator(self):

        def conv_block(input_tensor, f):
            x = input_tensor
            x = Conv2D(f, kernel_size=3, strides=2, kernel_initializer=conv_init, use_bias=False, padding="same")(x)
            x = Activation("relu")(x)
            return x

        def res_block(input_tensor, f):
            x = input_tensor
            x = Conv2D(f, kernel_size=3, kernel_initializer=conv_init, use_bias=False, padding="same")(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = Conv2D(f, kernel_size=3, kernel_initializer=conv_init, use_bias=False, padding="same")(x)
            x = add([x, input_tensor])
            x = LeakyReLU(alpha=0.2)(x)
            return x

        def upscale_ps(filters, use_instance_norm=True):
            def block(x):
                x = Conv2D(filters*4, kernel_size=3, use_bias=False, kernel_initializer=RandomNormal(0, 0.02), padding='same')(x)
                x = LeakyReLU(0.1)(x)
                x = PixelShuffler()(x)
                return x
            return block

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
            return Model(inputs=inp, outputs=out)

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
            return Model(input_, out )

        encoder = Encoder()
        decoder_A = Decoder_ps()
        decoder_B = Decoder_ps()
        x = Input(shape=self.img_shape)
        netGA = Model(x, decoder_A(encoder(x)))
        netGB = Model(x, decoder_B(encoder(x)))

        self.netGA_sm = netGA
        self.netGB_sm = netGB

        try:
            netGA.load_weights(str(self.model_dir / hdf['netGAH5']))
            netGB.load_weights(str(self.model_dir / hdf['netGBH5']))
            print ("Generator models loaded.")
        except:
            print ("Generator weights files not found.")
            pass

        if self.gpus > 1:
            netGA = multi_gpu_model( self.netGA_sm , self.gpus)
            netGB = multi_gpu_model( self.netGB_sm , self.gpus)

        return netGA, netGB

    def build_discriminator(self):
        def conv_block_d(input_tensor, f, use_instance_norm=True):
            x = input_tensor
            x = Conv2D(f, kernel_size=4, strides=2, kernel_initializer=conv_init, use_bias=False, padding="same")(x)
            x = LeakyReLU(alpha=0.2)(x)
            return x

        def Discriminator(nc_in, input_size=64):
            inp = Input(shape=(input_size, input_size, nc_in))
            #x = GaussianNoise(0.05)(inp)
            x = conv_block_d(inp, 64, False)
            x = conv_block_d(x, 128, False)
            x = conv_block_d(x, 256, False)
            out = Conv2D(1, kernel_size=4, kernel_initializer=conv_init, use_bias=False, padding="same", activation="sigmoid")(x)
            return Model(inputs=[inp], outputs=out)

        netDA = Discriminator(self.nc_D_inp)
        netDB = Discriminator(self.nc_D_inp)
        try:
            netDA.load_weights(str(self.model_dir / hdf['netDAH5']))
            netDB.load_weights(str(self.model_dir / hdf['netDBH5']))
            print ("Discriminator models loaded.")
        except:
            print ("Discriminator weights files not found.")
            pass
        return netDA, netDB

    def load(self, swapped):
        if swapped:
            print("swapping not supported on GAN")
            # TODO load is done in __init__ => look how to swap if possible
        return True

    def save_weights(self):
        model_dir = str(self.model_dir)
        for model in hdf.values():
            backup_file(model_dir, model)
        if self.gpus > 1:
            self.netGA_sm.save_weights(str(self.model_dir / hdf['netGAH5']))
            self.netGB_sm.save_weights(str(self.model_dir / hdf['netGBH5']))
        else:
            self.netGA.save_weights(str(self.model_dir / hdf['netGAH5']))
            self.netGB.save_weights(str(self.model_dir / hdf['netGBH5']))
        self.netDA.save_weights(str(self.model_dir / hdf['netDAH5']))
        self.netDB.save_weights(str(self.model_dir / hdf['netDBH5']))
        print ("Models saved.")
