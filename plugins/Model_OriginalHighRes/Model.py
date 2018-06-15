# Based on the original https://www.reddit.com/r/deepfakes/ code sample + contribs
# Based on https://github.com/iperov/OpenDeepFaceSwap for Decoder multiple res block chain
# Based on the https://github.com/shaoanlu/faceswap-GAN repo
# source : https://github.com/shaoanlu/faceswap-GAN/blob/master/FaceSwap_GAN_v2_sz128_train.ipynbtemp/faceswap_GAN_keras.ipynb


import enum

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
import os
import sys

from . import __version__
from .instance_normalization import InstanceNormalization


if isinstance(__version__, (list, tuple)):
    version_str = ".".join([str(n) for n in __version__[1:]])
else: 
    version_str = __version__


mswindows = sys.platform=="win32"

try:
    from lib.utils import backup_file
except ImportError:
    pass


conv_init = RandomNormal(0, 0.02)



def inst_norm():
    return InstanceNormalization()

class EncoderType(enum.Enum):
    ORIGINAL = "original"
    SHAOANLU = "shaoanlu"
    
ENCODER = EncoderType.SHAOANLU 

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
        
        face_A, face_B = (hdf['decoder_AH5'], hdf['decoder_BH5']) if not swapped else (hdf['decoder_BH5'], hdf['decoder_AH5'])

        try:            
            self.encoder.load_weights(os.path.join(self.model_dir, hdf['encoderH5']))
            self.decoder_A.load_weights(os.path.join(self.model_dir, face_A))
            self.decoder_B.load_weights(os.path.join(self.model_dir, face_B))
            print('loaded model weights')
            return True
        except Exception as e:
            print('Failed loading existing training data.', e)
            return False        


    def converter(self, swap):
        autoencoder = self.autoencoder_B if not swap else self.autoencoder_A
        return autoencoder.predict
    
    
    def conv(self, filters, kernel_size=5, strides=2, **kwargs):
        def block(x):
            x = Conv2D(filters, kernel_size=kernel_size, strides=strides, kernel_initializer=conv_init, padding='same', **kwargs)(x)         
            x = LeakyReLU(0.1)(x)
            return x
        return block    
        
    def conv_sep3(self, filters, kernel_size=3, strides=2, use_instance_norm=True, **kwargs):
        def block(x):
            x = SeparableConv2D(filters, kernel_size=kernel_size, strides=strides, kernel_initializer=conv_init, padding='same', **kwargs)(x)        
            if use_instance_norm:
                x = inst_norm()(x)
            x = Activation("relu")(x)
            return x    
        return block
    
    def upscale(self, filters, **kwargs):
        def block(x):
            x = Conv2D(filters * 4, kernel_size=3, padding='same')(x)
            x = LeakyReLU(0.1)(x)
            x = PixelShuffler()(x)
            return x
        return block  
    
    def upscale_sep3(self, filters, use_instance_norm=True, **kwargs):
        def block(x):
            x = Conv2D(filters*4, kernel_size=3, use_bias=False, kernel_initializer=RandomNormal(0, 0.02), padding='same', **kwargs)(x)
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
        x = self.conv(256)(x)
        x = self.conv(512)(x)
        x = self.conv(1024)(x)
        
        dense_shape = self.IMAGE_SHAPE[0] // 16         
        x = Dense(self.ENCODER_DIM)(Flatten()(x))
        x = Dense(dense_shape * dense_shape * 768)(x)
        x = Reshape((dense_shape, dense_shape, 768))(x)
        x = self.upscale(512)(x)
        
        return KerasModel(impt, x, **kwargs)    
          
    
    def Encoder_shaoanlu(self, **kwargs):
        impt = Input(shape=self.IMAGE_SHAPE)
                
        in_conv_filters = self.IMAGE_SHAPE[0] if self.IMAGE_SHAPE[0] <= 128 else 128 + (self.IMAGE_SHAPE[0]-128)//4
        
        x = Conv2D(in_conv_filters, kernel_size=5, kernel_initializer=conv_init, use_bias=False, padding="same")(impt)
        x = self.conv_sep3(in_conv_filters+32, use_instance_norm=False)(x)
        x = self.conv_sep3(256)(x)        
        x = self.conv_sep3(512)(x)
        x = self.conv_sep3(1024)(x)        
        
        dense_shape = self.IMAGE_SHAPE[0] // 16         
        x = Dense(self.ENCODER_DIM)(Flatten()(x))
        x = Dense(dense_shape * dense_shape * 768)(x)
        x = Reshape((dense_shape, dense_shape, 768))(x)
        x = self.upscale(512)(x)
        
        return KerasModel(impt, x, **kwargs)    


    def Decoder_original(self):       
        decoder_shape = self.IMAGE_SHAPE[0]//8        
        inpt = Input(shape=(decoder_shape, decoder_shape, 512))
        
        x = self.upscale(512, kernel_initializer=RandomNormal(0, 0.02))(inpt)
        x = self.upscale(256, kernel_initializer=RandomNormal(0, 0.02))(x)
        x = self.upscale(self.IMAGE_SHAPE[0], kernel_initializer=RandomNormal(0, 0.02))(x)
        
        x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)
        
        return KerasModel(inpt, x)
    
    
    def Decoder_shaoanlu(self):       
        decoder_shape = self.IMAGE_SHAPE[0]//8        
        inpt = Input(shape=(decoder_shape, decoder_shape, 512))
        
        x = self.upscale_sep3(512)(inpt)
        x = self.upscale_sep3(256)(x)
        x = self.upscale_sep3(self.IMAGE_SHAPE[0])(x)
        
        x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)
        
        return KerasModel(inpt, x)    


    def save_weights(self):
        from threading import Thread
        from time import sleep
        
        model_dir = str(self.model_dir)
        
        try:
            for model in hdf.values():            
                backup_file(model_dir, model)
        except NameError:
            print('backup functionality not available\n')                                      
        
        # thought maybe I/O bound, sometimes saving in parallel is faster
        threads = []
        t = Thread(target=self.encoder.save_weights, args=(str(self.model_dir / hdf['encoderH5']),))
        threads.append(t)         
        t = Thread(target=self.decoder_A.save_weights, args=(str(self.model_dir / hdf['decoder_AH5']),))
        threads.append(t)
        t = Thread(target=self.decoder_B.save_weights, args=(str(self.model_dir / hdf['decoder_BH5']),))
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
            return self._model_name
        except AttributeError:
            self._model_name = os.path.split(os.path.dirname(__file__))[1].replace("Model_", "")            
        return self._model_name
             
    
    def __str__(self):
        return "<{}: ver={}, nn_dims={}, img_size={}>".format(self.model_name, 
                                                              version_str, 
                                                              self.ENCODER_DIM, 
                                                              "x".join([str(n) for n in self.IMAGE_SHAPE[:2]]))        
        
