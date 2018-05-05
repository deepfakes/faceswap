# Based on the original https://www.reddit.com/r/deepfakes/ code sample + contribs
# Based on the https://github.com/shaoanlu/faceswap-GAN repo
# source : https://github.com/shaoanlu/faceswap-GAN/blob/master/FaceSwap_GAN_v2_sz128_train.ipynbtemp/faceswap_GAN_keras.ipynb


import os

from keras.initializers import RandomNormal
from keras.layers import Input, Dense, Flatten, Reshape
from keras.layers import SeparableConv2D, add
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D

from keras.models import Model as KerasModel
from keras.optimizers import Adam
from keras.utils import multi_gpu_model

from lib.PixelShuffler import PixelShuffler

from . import __version__

if isinstance(__version__, (list, tuple)):
    version_str = ".".join([str(n) for n in __version__[1:]])
else: 
    version_str = __version__


try:
    from lib.utils import backup_file
except ImportError:
    pass

hdf = {'encoderH5': f'encoder_{version_str}.h5',
       'decoder_AH5': f'decoder_A_{version_str}.h5',
       'decoder_BH5': f'decoder_B_{version_str}.h5'}


class Model():

    # still playing with dims
    ENCODER_DIM = 2048
        
    IMAGE_SIZE = 128, 128
    IMAGE_DEPTH = len('RGB') # good to let ppl know what these are...
    IMAGE_SHAPE = *IMAGE_SIZE, IMAGE_DEPTH
    
    def __init__(self, model_dir, gpus):
        self.model_dir = model_dir
        
        # can't chnage gpu's when the model is initialized no point in making it r/w
        self._gpus = gpus 
        
        self.encoder = self.Encoder()
        self.decoder_A = self.Decoder()
        self.decoder_B = self.Decoder()
        
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
        
        face_A, face_B = (hdf['decoder_AH5'], hdf['decoder_BH5']) if not swapped else (hdf['decoder_BH5'], hdf['decoder_AH5'])

        try:            
            self.encoder.load_weights(os.path.join(self.model_dir, hdf['encoderH5']))
            self.decoder_A.load_weights(os.path.join(self.model_dir, face_A))
            self.decoder_B.load_weights(os.path.join(self.model_dir, face_B))
            print('loaded model weights')
            return True
        except Exception as e:
            print('Failed loading existing training data.')
            print(e)
            return False        

    def converter(self, swap):
        autoencoder = self.autoencoder_B if not swap else self.autoencoder_A
        return autoencoder.predict
    
    def conv2d(self, filters, kernel_size=5, strides=2):
        def block(x):
            x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same')(x)         
            x = LeakyReLU(0.1)(x)
            return x
        return block    
        
    def sep_conv2d(self, filters):
        def block(x):
            x = SeparableConv2D(filters, kernel_size=5, strides=2, padding='same')(x)        
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
        retina = Input(shape=self.IMAGE_SHAPE)          
        x = self.conv2d(196, kernel_size=5, strides=2)(retina)
        x = self.sep_conv2d(384)(x)                                
        x = self.sep_conv2d(512)(x)        
        x = self.sep_conv2d(768)(x)         
        x = self.sep_conv2d(1024)(x)
        x = Dense(self.ENCODER_DIM)(Flatten()(x))
        x = Dense(4 * 4 * 1024)(x)
        x = Reshape((4, 4, 1024))(x)
        out = self.upscale(512)(x)
        return KerasModel(retina, out)
    

    def Decoder(self):
        inp = Input(shape=(8, 8, 512))
        
        x = self.upscale(256)(inp)
        x = self.upscale(128)(x)
        x = self.upscale(64)(x)        
        
        rb = Conv2D(64, kernel_size=3, padding="same", dilation_rate=2)(x)
        rb = LeakyReLU(alpha=0.2)(rb)
        rb = Conv2D(64, kernel_size=3, padding="same", dilation_rate=2)(rb)
        x = add([rb, x])

        x = self.upscale(32)(x)        
                        
        out = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)
                
        return KerasModel(inp, out)
        
    


    def save_weights(self):
        from threading import Thread
        from time import sleep
        
        model_dir = str(self.model_dir)
        
        try:
            for model in hdf.values():            
                backup_file(model_dir, model)
        except NameError:
            print('backup functionality not available')   
            pass
        
        #thought maybe I/O bound, sometimes saving in parallel is faster
        threads = []
        t = Thread(target=self.encoder.save_weights, args=(os.path.join(model_dir, hdf['encoderH5']),))
        threads.append(t)         
        t = Thread(target=self.decoder_A.save_weights, args=(os.path.join(model_dir, hdf['decoder_AH5']),))
        threads.append(t)
        t = Thread(target=self.decoder_B.save_weights, args=(os.path.join(model_dir, hdf['decoder_BH5']),))
        threads.append(t)
        
        for thread in threads:
            thread.start()            
        
        while any([t.is_alive() for t in threads]):
            sleep(0.1)
            
        print('saved model weights')              
    
        
                
    @property
    def gpus(self):
        return self._gpus
    
    @property
    def model_name(self):
        try:
            return self._model_nomen
        except AttributeError:
            self._model_nomen = self._model_nomen = os.path.split(os.path.dirname(__file__))[1].replace("Model_", "")            
        return self._model_nomen
             
    
    def __str__(self):
        return "<{}: ver={}, nn_dims={}, img_size={}>".format(self.model_name, 
                                                              version_str, 
                                                              self.ENCODER_DIM, 
                                                              "x".join([str(n) for n in self.IMAGE_SHAPE[:2]]))        
        
