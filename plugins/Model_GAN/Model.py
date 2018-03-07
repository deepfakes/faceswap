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

netGAH5 = 'netGA_GAN.h5'
netGBH5 = 'netGB_GAN.h5'
netDAH5 = 'netDA_GAN.h5'
netDBH5 = 'netDB_GAN.h5'
netDA2H5 = 'netDA_GAN.h5'
netDB2H5 = 'netDB_GAN.h5'
netD_codeH5 = 'netD_code_GAN.h5'

def __conv_init(a):
    print("conv_init", a)
    k = RandomNormal(0, 0.02)(a) # for convolution kernel
    k.conv_weight = True
    return k

conv_init = RandomNormal(0, 0.02)
gamma_init = RandomNormal(1., 0.02) # for batch normalization

class GANModel():
    img_size = 64
    channels = 3
    img_shape = (img_size, img_size, channels)
    encoded_dim = 1024
    nc_in = 3 # number of input channels of generators
    nc_D_inp = 6 # number of input channels of discriminators

    def __init__(self, model_dir):
        self.model_dir = model_dir

        optimizer = Adam(1e-4, 0.5)

        # Build and compile the discriminator
        self.netDA, self.netDB, self.netDA2, self.netDB2, self.netD_code = self.build_discriminator()

        # Build and compile the generator
        self.netGA, self.netGB = self.build_generator()

    def converter(self, swap):
        predictor = self.netGB if not swap else self.netGA
        return lambda img: predictor.predict(img)

    def build_generator(self):

        def conv_block(input_tensor, f, k=3, strides=2, dilation_rate=1, use_instance_norm=True):
            x = input_tensor
            x = Conv2D(f, kernel_size=k, strides=strides, dilation_rate=dilation_rate, kernel_initializer=conv_init, use_bias=False, padding="same")(x)
            if use_instance_norm:
                x = InstanceNormalization()(x)
            x = LeakyReLU(alpha=0.1)(x)
            return x

        def res_block(input_tensor, f):
            x = input_tensor
            x = Conv2D(f, kernel_size=5, kernel_initializer=conv_init, use_bias=True, padding="same")(x)
            x = InstanceNormalization()(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = Conv2D(f, kernel_size=5, kernel_initializer=conv_init, use_bias=True, padding="same")(x)
            x = InstanceNormalization()(x)
            x = add([x, input_tensor])
            return x

        def upscale_ps(filters, dilation_rate=1, use_instance_norm=True):
            def block(x):
                x = Conv2D(filters*4, kernel_size=3, use_bias=True, kernel_initializer=RandomNormal(0, 0.02), padding='same')(x)
                x = InstanceNormalization()(x)
                x = LeakyReLU(0.1)(x)
                x = PixelShuffler()(x)
                return x
            return block

        def Encoder(nc_in=3, input_size=64):
            def l2_norm(x):
                epsilon = 1e-12
                x_norm = K.sqrt(K.sum(K.square(x)))
                return x / (x_norm + epsilon)
            inp = Input(shape=(input_size, input_size, nc_in))
            x = Conv2D(64, kernel_size=5, kernel_initializer=conv_init, use_bias=False, padding="same")(inp)
            x = LeakyReLU(0.1)(x)
            x = conv_block(x,128)
            x = conv_block(x,256)
            x = conv_block(x,512)
            x = conv_block(x,1024)
            x = Dense(1024)(Flatten()(x))
            x = InstanceNormalization()(x)
            x = LeakyReLU(0.1)(x)
            x = Dense(1024)(x)
            x = Lambda(l2_norm)(x)
            code = x
            x = Dense(4*4*1024)(x)
            x = Reshape((4, 4, 1024))(x)
            out = upscale_ps(512, dilation_rate=1)(x)
            return Model(inputs=inp, outputs=[out, code])

        def Decoder_ps(nc_in=512, input_size=8):
            inp = Input(shape=(input_size, input_size, nc_in))
            code = Input(shape=(1024,))
            x = inp
            x = upscale_ps(256)(x)
            x = upscale_ps(128)(x)
            x = upscale_ps(64)(x)
            x = res_block(x, 64)
            x, _ = mixed_scaled_dense_network(x, 16, 64, 4)
            alpha = Lambda(lambda x: x[:,:,:,0:1])(x)
            alpha = Activation("sigmoid")(alpha)
            bgr = Lambda(lambda x: x[:,:,:,1:])(x)
            bgr = Activation("tanh")(bgr)
            out = concatenate([alpha, bgr])
            return Model([inp, code], [out, code])

        def mixed_scaled_dense_network(input_tensor, num_layers=32, nc_in=3, nc_out=1):
            """
            Inefficient implementation of paper: A mixed-scale dense convolutional neural network for image analysis
            http://www.pnas.org/content/115/2/254
            """
            msd_layers = {}
            x = input_tensor
            msd_layers["input"] = x
            for i in range(num_layers):
                dilation = (i % 10) + 1
                msd_layers["layer{0}".format(i)] = Conv2D(1, kernel_size=3, strides=1, dilation_rate=dilation,
                                                          kernel_initializer=conv_init, padding="same")(x)
                for j in range(i):
                    dilation = ((i + j) %10) + 1
                    conv_3x3 = Conv2D(1, kernel_size=3, strides=1, dilation_rate=dilation,
                                      kernel_initializer=conv_init, use_bias=True, padding="same")(msd_layers["layer{0}".format(j)])
                    msd_layers["layer{0}".format(i)] = add([msd_layers["layer{0}".format(i)], conv_3x3])
                msd_layers["layer{0}".format(i)] = Activation("relu")(msd_layers["layer{0}".format(i)])

            concat_all = x
            for i in range(num_layers):
                concat_all = concatenate([concat_all, msd_layers["layer{0}".format(i)]])
            msd_layers["merge_concat_all"] = concat_all
            out = Conv2D(nc_out, kernel_size=1, kernel_initializer=conv_init, padding="same")(concat_all)
            msd_layers["output"] = out

            return out, msd_layers

        def mixnet_block(input_tensor, nc_concat=12, type="fixed"):
            """
            Inspired by: Mixed Link Networks (https://arxiv.org/abs/1802.01808)
            Inner-loop has different implementation.
            """
            def se_block(input_tensor, compress_rate = 16):
                num_channels = int(input_tensor.shape[-1]) # Tensorflow backend
                bottle_neck = int(num_channels//compress_rate)

                se_branch = GlobalAveragePooling2D()(input_tensor)
                se_branch = Dense(bottle_neck, activation='relu', kernel_initializer='he_normal', use_bias=False)(se_branch)
                se_branch = Dense(num_channels, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se_branch)
                x = input_tensor
                out = multiply([x, se_branch])
                return out

            x = input_tensor
            nc_in = int(input_tensor.shape[-1])

            k2 = nc_concat
            x_out = Conv2D(nc_in//2, kernel_size=1, kernel_initializer=conv_init, use_bias=True, padding="same")(x)
            x_out = InstanceNormalization()(x_out)
            x_out = LeakyReLU(alpha=0.1)(x_out)
            x_out = Conv2D(k2, kernel_size=3, kernel_initializer=conv_init, use_bias=True, padding="same")(x_out)
            x_out = InstanceNormalization()(x_out)
            x_out = LeakyReLU(alpha=0.1)(x_out)

            if type == "fixed":
                k1 = nc_in // 2
                x_in = Conv2D(nc_in//4, kernel_size=1, kernel_initializer=conv_init, use_bias=True, padding="same")(x)
                x_in = InstanceNormalization()(x_in)
                x_in = LeakyReLU(alpha=0.1)(x_in)
                x_in = Conv2D(k1, kernel_size=3, kernel_initializer=conv_init, use_bias=True, padding="same")(x_in)
                #x_in = Conv2D(nc_in, kernel_size=3, kernel_initializer=conv_init, use_bias=True, padding="same")(x_in)
                x_in = InstanceNormalization()(x_in)
                x_in = se_block(x_in)
                x_in_add = Lambda(lambda x: x[0][:,:,:, -k1:] + x[1])([x, x_in])
                x_in_id = Lambda(lambda x: x[:,:,:, :k1])(x)
                x_in = concatenate([x_in_id, x_in_add])
                #x_in = add([x, x_in])
                x_in = LeakyReLU(alpha=0.1)(x_in)
            elif type == "unfixed":
                k1 = nc_concat
                x_in = Conv2D(nc_in//2, kernel_size=1, kernel_initializer=conv_init, use_bias=True, padding="same")(x)
                x_in = InstanceNormalization()(x_in)
                x_in = LeakyReLU(alpha=0.1)(x_in)
                x_in = Conv2D(k1, kernel_size=3, kernel_initializer=conv_init, use_bias=True, padding="same")(x_in)
                x_in = InstanceNormalization()(x_in)
                x_in_add = Lambda(lambda x: x[0][:,:,:, -k1:] + x[1])([x, x_in])
                x_in_id = Lambda(lambda x: x[:,:,:, :k1])(x)
                x_in = concatenate([x_in_id, x_in_add])
                x_in = LeakyReLU(alpha=0.1)(x_in)

            out = concatenate([x_in, x_out])
            return out

        encoder = Encoder()
        decoder_A = Decoder_ps()
        decoder_B = Decoder_ps()
        x = Input(shape=self.img_shape)
        netGA = Model(x, decoder_A(encoder(x)))
        netGB = Model(x, decoder_B(encoder(x)))
        try:
            netGA.load_weights(str(self.model_dir / netGAH5))
            netGB.load_weights(str(self.model_dir / netGBH5))
            print ("Generator models loaded.")
        except:
            print ("Generator weights files not found.")
            pass
        return netGA, netGB

    def build_discriminator(self):
        def conv_block_d(input_tensor, f, use_instance_norm=True):
            x = input_tensor
            x = Conv2D(f, kernel_size=4, strides=2, kernel_initializer=conv_init, use_bias=True, padding="same")(x)
            if use_instance_norm:
                x = InstanceNormalization()(x)
            x = LeakyReLU(alpha=0.2)(x)
            return x

        def Discriminator(nc_in, input_size=64):
            inp = Input(shape=(input_size, input_size, nc_in))
            #x = GaussianNoise(0.05)(inp)
            x = conv_block_d(inp, 64, False)
            x = conv_block_d(x, 128)
            x = conv_block_d(x, 256)
            out = Conv2D(1, kernel_size=4, kernel_initializer=conv_init, use_bias=False, padding="same", activation="sigmoid")(x)
            return Model(inputs=[inp], outputs=out)

        def Discriminator2(nc_in, input_size=64):
            inp = Input(shape=(input_size, input_size, nc_in))
            #x = GaussianNoise(0.05)(inp)
            x = conv_block_d(inp, 64, False)
            x = conv_block_d(x, 128)
            x = conv_block_d(x, 256)
            x = conv_block_d(x, 512)
            out = Conv2D(1, kernel_size=3, kernel_initializer=conv_init, use_bias=False, padding="same", activation="sigmoid")(x)
            return Model(inputs=[inp], outputs=out)

        def Discriminator3(nc_in, input_size=64):
            inp = Input(shape=(input_size, input_size, nc_in))
            x = conv_block_d(inp, 32, False)
            x = mixnet_block(x, 12)
            x = Conv2D(64, kernel_size=1, kernel_initializer=conv_init, use_bias=False, padding="same")(x)
            x = AveragePooling2D()(x)
            x = mixnet_block(x, 24)
            x = Conv2D(128, kernel_size=1, kernel_initializer=conv_init, use_bias=False, padding="same")(x)
            x = AveragePooling2D()(x)
            x = mixnet_block(x, 24)
            x = Conv2D(256, kernel_size=1, kernel_initializer=conv_init, use_bias=False, padding="same")(x)
            x = AveragePooling2D()(x)
            out = Conv2D(1, kernel_size=3, kernel_initializer=conv_init, use_bias=False, padding="same", activation="sigmoid")(x)
            return Model(inputs=[inp], outputs=out)

        def Discriminator_code():
            inp = Input(shape=(1024, ))
            x = Dense(256)(inp)
            x = InstanceNormalization()(x)
            x = Activation('relu')(x)
            x = Dense(128)(x)
            x = InstanceNormalization()(x)
            x = Activation('relu')(x)
            out = Dense(1, activation='sigmoid')(x)
            return Model(inputs=[inp], outputs=out)

        netDA = Discriminator(self.nc_D_inp)
        netDB = Discriminator(self.nc_D_inp)
        netDA2 = Discriminator2(self.nc_D_inp//2)
        netDB2 = Discriminator2(self.nc_D_inp//2)
        netD_code = Discriminator_code()
        try:
            netDA.load_weights(str(self.model_dir / netDAH5))
            netDB.load_weights(str(self.model_dir / netDBH5))
            netDA2.load_weights(str(self.model_dir / netDA2H5))
            netDB2.load_weights(str(self.model_dir / netDB2H5))
            netD_code.load_weights(str(self.model_dir / netD_codeH5))
            print ("Discriminator models loaded.")
        except:
            print ("Discriminator weights files not found.")
            pass
        return netDA, netDB, netDA2, netDB2, netD_code

    def load(self, swapped):
        if swapped:
            print("swapping not supported on GAN")
            # TODO load is done in __init__ => look how to swap if possible
        return True

    def save_weights(self):
        self.netGA.save_weights(str(self.model_dir / netGAH5))
        self.netGB.save_weights(str(self.model_dir / netGBH5))
        self.netDA.save_weights(str(self.model_dir / netDAH5))
        self.netDB.save_weights(str(self.model_dir / netDBH5))
        self.netDA2.save_weights(str(self.model_dir / netDA2H5))
        self.netDB2.save_weights(str(self.model_dir / netDB2H5))
        self.netD_code.save_weights(str(self.model_dir / netD_codeH5))
        print ("Models saved.")
