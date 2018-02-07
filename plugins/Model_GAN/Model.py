# Based on the https://github.com/shaoanlu/faceswap-GAN repo (master/temp/faceswap_GAN_keras.ipynb)

from keras.models import Model
from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu
from keras.initializers import RandomNormal
from keras.applications import *
from keras.optimizers import Adam

from lib.PixelShuffler import PixelShuffler

netGAH5 = 'netGA_GAN.h5'
netGBH5 = 'netGB_GAN.h5'
netDAH5 = 'netDA_GAN.h5'
netDBH5 = 'netDB_GAN.h5'

class GANModel():
    img_size = 64 
    channels = 3
    img_shape = (img_size, img_size, channels)
    encoded_dim = 1024
    
    def __init__(self, model_dir):
        self.model_dir = model_dir

        optimizer = Adam(1e-4, 0.5)

        # Build and compile the discriminator
        self.netDA, self.netDB = self.build_discriminator()

        # For the adversarial_autoencoder model we will only train the generator
        self.netDA.trainable = False
        self.netDB.trainable = False
        
        self.netDA.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        self.netDB.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

        # Build and compile the generator
        self.netGA, self.netGB = self.build_generator()
        self.netGA.compile(loss=['mae', 'mse'], optimizer=optimizer)
        self.netGB.compile(loss=['mae', 'mse'], optimizer=optimizer)

        img = Input(shape=self.img_shape)
        alphaA, reconstructed_imgA = self.netGA(img)
        alphaB, reconstructed_imgB = self.netGB(img)

        def one_minus(x): return 1 - x
        # masked_img = alpha * reconstructed_img + (1 - alpha) * img
        masked_imgA = add([multiply([alphaA, reconstructed_imgA]), multiply([Lambda(one_minus)(alphaA), img])])
        masked_imgB = add([multiply([alphaB, reconstructed_imgB]), multiply([Lambda(one_minus)(alphaB), img])])
        out_discriminatorA = self.netDA(concatenate([masked_imgA, img], axis=-1))
        out_discriminatorB = self.netDB(concatenate([masked_imgB, img], axis=-1))

        # The adversarial_autoencoder model  (stacked generator and discriminator) takes
        # img as input => generates encoded represenation and reconstructed image => determines validity 
        self.adversarial_autoencoderA = Model(img, [reconstructed_imgA, out_discriminatorA])
        self.adversarial_autoencoderB = Model(img, [reconstructed_imgB, out_discriminatorB])
        self.adversarial_autoencoderA.compile(loss=['mae', 'mse'],
                                              loss_weights=[1, 0.5],
                                              optimizer=optimizer)
        self.adversarial_autoencoderB.compile(loss=['mae', 'mse'],
                                              loss_weights=[1, 0.5],
                                              optimizer=optimizer)

    def converter(self, swap):
        predictor = self.netGB if not swap else self.netGA
        return lambda img: predictor.predict(img)

    def build_generator(self):
        def conv_block(input_tensor, f):
            x = input_tensor
            x = Conv2D(f, kernel_size=3, strides=2, kernel_initializer=RandomNormal(0, 0.02), 
                       use_bias=False, padding="same")(x)
            x = LeakyReLU(alpha=0.2)(x)
            return x

        def res_block(input_tensor, f):
            x = input_tensor
            x = Conv2D(f, kernel_size=3, kernel_initializer=RandomNormal(0, 0.02), 
                       use_bias=False, padding="same")(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = Conv2D(f, kernel_size=3, kernel_initializer=RandomNormal(0, 0.02), 
                       use_bias=False, padding="same")(x)
            x = add([x, input_tensor])
            x = LeakyReLU(alpha=0.2)(x)
            return x

        def upscale_ps(filters, use_norm=True):
            def block(x):
                x = Conv2D(filters*4, kernel_size=3, use_bias=False, 
                           kernel_initializer=RandomNormal(0, 0.02), padding='same' )(x)
                x = LeakyReLU(0.1)(x)
                x = PixelShuffler()(x)
                return x
            return block

        def Encoder(img_shape):
            inp = Input(shape=img_shape)
            x = Conv2D(64, kernel_size=5, kernel_initializer=RandomNormal(0, 0.02), 
                       use_bias=False, padding="same")(inp)
            x = conv_block(x,128)
            x = conv_block(x,256)
            x = conv_block(x,512) 
            x = conv_block(x,1024)
            x = Dense(1024)(Flatten()(x))
            x = Dense(4*4*1024)(x)
            x = Reshape((4, 4, 1024))(x)
            out = upscale_ps(512)(x)
            return Model(inputs=inp, outputs=out)

        def Decoder_ps(img_shape):
            nc_in = 512
            input_size = img_shape[0]//8
            inp = Input(shape=(input_size, input_size, nc_in))
            x = inp
            x = upscale_ps(256)(x)
            x = upscale_ps(128)(x)
            x = upscale_ps(64)(x)
            x = res_block(x, 64)
            x = res_block(x, 64)
            alpha = Conv2D(1, kernel_size=5, padding='same', activation="sigmoid")(x)
            rgb = Conv2D(3, kernel_size=5, padding='same', activation="tanh")(x)
            return Model(inp, [alpha, rgb])
        
        encoder = Encoder(self.img_shape)
        decoder_A = Decoder_ps(self.img_shape)
        decoder_B = Decoder_ps(self.img_shape)    
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
        return netGA, netGB, 

    def build_discriminator(self):  
        def conv_block_d(input_tensor, f, use_instance_norm=True):
            x = input_tensor
            x = Conv2D(f, kernel_size=4, strides=2, kernel_initializer=RandomNormal(0, 0.02), 
                       use_bias=False, padding="same")(x)
            x = LeakyReLU(alpha=0.2)(x)
            return x   
        def Discriminator(img_shape):
            inp = Input(shape=(img_shape[0], img_shape[1], img_shape[2]*2))
            x = conv_block_d(inp, 64, False)
            x = conv_block_d(x, 128, False)
            x = conv_block_d(x, 256, False)
            out = Conv2D(1, kernel_size=4, kernel_initializer=RandomNormal(0, 0.02), 
                         use_bias=False, padding="same", activation="sigmoid")(x)   
            return Model(inputs=[inp], outputs=out) 
        
        netDA = Discriminator(self.img_shape)
        netDB = Discriminator(self.img_shape)        
        try:
            netDA.load_weights(str(self.model_dir / netDAH5))
            netDB.load_weights(str(self.model_dir / netDBH5))
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
        self.netGA.save_weights(str(self.model_dir / netGAH5))
        self.netGB.save_weights(str(self.model_dir /  netGBH5))
        self.netDA.save_weights(str(self.model_dir /  netDAH5))
        self.netDB.save_weights(str(self.model_dir /  netDBH5))
        print ("Models saved.")