# Based on the original https://www.reddit.com/r/deepfakes/ code sample + contribs
# Based on https://github.com/iperov/DeepFaceLab a better 128x Decoder idea
# Based on the https://github.com/shaoanlu/faceswap-GAN repo res_block chain and IN
# source : https://github.com/shaoanlu/faceswap-GAN/blob/master/FaceSwap_GAN_v2_sz128_train.ipynbtemp/faceswap_GAN_keras.ipynb

import enum
from json import JSONDecodeError
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from keras.initializers import RandomNormal
from keras.layers import Input, Dense, Flatten, Reshape
from keras.layers import SeparableConv2D, add
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.models import Model as KerasModel
from keras.optimizers import Adam
from keras.utils import multi_gpu_model

from lib.PixelShuffler import PixelShuffler
import lib.Serializer
from lib.utils import backup_file

from . import __version__
from .instance_normalization import InstanceNormalization


if isinstance(__version__, (list, tuple)):
    version_str = ".".join([str(n) for n in __version__[1:]])
else: 
    version_str = __version__


mswindows = sys.platform=="win32"

_kern_init = RandomNormal(0, 0.02)


class EncoderType(enum.Enum):
    ORIGINAL = "original" # basic encoder for this model type
    STANDARD = "standard" # new, balanced encoder they way I meant it to be; more memory consuming
    HIGHRES = "highres"   # high resolution tensors optimized encoder: 176x and on 
                
        
def inst_norm():
    return InstanceNormalization()


# autoencoder type
ENCODER = EncoderType.ORIGINAL

# might increase overall quality at cost of training speed
USE_DSSIM = False

# might increase upscaling quality at cost of video memory
USE_SUBPIXEL = False


hdf = { 'encoderH5': 'encoder_{version_str}{ENCODER.value}.h5'.format( **vars() ),
        'decoder_AH5': 'decoder_A_{version_str}{ENCODER.value}.h5'.format( **vars() ),
        'decoder_BH5': 'decoder_B_{version_str}{ENCODER.value}.h5'.format( **vars() ) }


class Model():
    
    ENCODER_DIM = 1024 # dense layer size        
    IMAGE_SHAPE = 128, 128 # image shape
    
    ENCODER_COMPLEXITY = 128 # use cauton, sensible ranges 128 - 160; the bigger the more details can be learned
    DECODER_A_COMPLEXITY = 384 # only applicable for STANDARD encoder
    DECODER_B_COMPLEXITY = 512 # only applicable for STANDARD encoder
    
    USE_EXTRA_DOWNSCALING = True # to save video RAM
    
    USE_K_FUNCTION = True
    
    assert [n for n in IMAGE_SHAPE if n>=16]
    
    IMAGE_WIDTH = max(IMAGE_SHAPE)
    IMAGE_WIDTH = (IMAGE_WIDTH//16 + (1 if (IMAGE_WIDTH%16)>=8 else 0))*16
    IMAGE_WIDTH = min(IMAGE_WIDTH, 256)
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
        
        if not self.is_new_training:
            global USE_DSSIM, USE_SUBPIXEL
                        
            self.__class__.ENCODER_DIM = self.current_state['ENCODER_DIM']
            self.__class__.IMAGE_SHAPE = self.current_state['IMAGE_SHAPE']
            self.__class__.USE_EXTRA_DOWNSCALING = self.current_state['USE_EXTRA_DOWNSCALING']
            self.__class__.ENCODER_COMPLEXITY = self.current_state['ENCODER_COMPLEXITY'] 
            
            USE_DSSIM = self.current_state['IMAGE_SHAPE']
            USE_SUBPIXEL = self.current_state['USE_SUBPIXEL']            
        
        Encoder = getattr(self, "Encoder_{}".format(self.encoder_type))        
        Decoder_A = getattr(self, "Decoder_{}_A".format(self.encoder_type))
        Decoder_B = getattr(self, "Decoder_{}_B".format(self.encoder_type))
        
        self.encoder = Encoder()
        self.decoder_A = Decoder_A()
        self.decoder_B = Decoder_B()
        
        self.initModel()        

    
    def initModel(self):
        optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)
                        
        input_A_bgr = Input(shape=self.IMAGE_SHAPE)
        input_B_bgr = Input(shape=self.IMAGE_SHAPE)        
        
        rec_A_bgr = self.decoder_A(self.encoder(input_A_bgr))
        rec_B_bgr = self.decoder_B(self.encoder(input_B_bgr))                   
        
        if USE_DSSIM:
            from .dssim import DSSIMObjective
            loss_func = DSSIMObjective()
            print('Using DSSIM loss ..', flush=True)
        else:
            loss_func = 'mean_absolute_error'
            
        if USE_SUBPIXEL:
            from .subpixel import SubPixelUpscaling
            self.upscale = self.upscale_sub
            print('Using subpixel upscaling ..', flush=True)            
        
        if self.USE_K_FUNCTION:        
            print('Using K.function ..', flush=True)
            
            self.autoencoder = KerasModel([input_B_bgr, input_A_bgr], [rec_B_bgr, rec_A_bgr] )
            
            if self.gpus > 1:
                self.autoencoder = multi_gpu_model( self.autoencoder, self.gpus)            

            self.autoencoder.compile(optimizer=optimizer, loss=[ loss_func, loss_func ] )
            
            import keras.backend as K
      
            self.B_view = K.function([input_B_bgr], [rec_B_bgr])
            self.A_view = K.function([input_A_bgr], [rec_A_bgr])
                                    
        else:        
            self.autoencoder_A = KerasModel(input_A_bgr, rec_A_bgr)
            self.autoencoder_B = KerasModel(input_B_bgr, rec_B_bgr)   
                                         
            if self.gpus > 1:
                self.autoencoder_A = multi_gpu_model( self.autoencoder_A , self.gpus)
                self.autoencoder_B = multi_gpu_model( self.autoencoder_B , self.gpus)         
                    
            self.autoencoder_A.compile(optimizer=optimizer, loss=loss_func)
            self.autoencoder_B.compile(optimizer=optimizer, loss=loss_func)

    def load(self, swapped):                
        model_dir = str(self.model_dir)

        face_A, face_B = (hdf['decoder_AH5'], hdf['decoder_BH5']) if not swapped else (hdf['decoder_BH5'], hdf['decoder_AH5'])                            

        try:            
            self.encoder.load_weights(os.path.join(model_dir, hdf['encoderH5']))
            self.decoder_A.load_weights(os.path.join(model_dir, face_A))
            self.decoder_B.load_weights(os.path.join(model_dir, face_B))
            print('loaded model weights')
            return True
        except IOError as e:
            print('Failed loading training data:', e.strerror)            
        except Exception as e:
            print('Failed loading training data:', str(e))            
      
        return False


    def converter(self, swap):
        if not self.USE_K_FUNCTION:
            autoencoder = self.autoencoder_B if not swap else self.autoencoder_A
            return autoencoder.predict
        else:
            view = self.B_view if not swap else self.A_view
            return lambda x: view( [x] )[0]
    
    @staticmethod
    def conv(filters, kernel_size=5, strides=2, use_instance_norm=False, **kwargs):
        def block(x):
            x = Conv2D(filters, kernel_size=kernel_size, strides=strides, 
                       kernel_initializer=_kern_init, padding='same', **kwargs)(x)
            if use_instance_norm:
                x = inst_norm()(x)                                
            x = LeakyReLU(0.1)(x)            
            return x
        return block   
    
    @staticmethod
    def res_block(input_tensor, f):
        x = input_tensor
        x = Conv2D(f, kernel_size=3, kernel_initializer=_kern_init, use_bias=False, padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(f, kernel_size=3, kernel_initializer=_kern_init, use_bias=False, padding="same")(x)
        x = add([x, input_tensor])
        x = LeakyReLU(alpha=0.2)(x)
        return x        
    
    @staticmethod
    def conv_sep(filters, kernel_size=5, strides=2, use_instance_norm=False, **kwargs):
        def block(x):
            x = SeparableConv2D(filters, kernel_size=kernel_size, strides=strides, 
                       kernel_initializer=_kern_init, padding='same', **kwargs)(x)
            if use_instance_norm:
                x = inst_norm()(x)                                
            x = LeakyReLU(0.1)(x)            
            return x
        return block

    @staticmethod
    def upscale_sub(filters, kernel_size=3, use_instance_norm=False, **kwargs):
        def block(x):
            x = Conv2D(filters * 4, kernel_size=kernel_size, padding='same',
                       kernel_initializer=_kern_init, **kwargs)(x)
            if use_instance_norm:
                x = inst_norm()(x)                       
            x = LeakyReLU(0.1)(x)
            x = SubPixelUpscaling()(x)
            return x
        return block
              
    @staticmethod
    def upscale(filters, kernel_size=3, use_instance_norm=False, **kwargs):
        def block(x):
            x = Conv2D(filters * 4, kernel_size=kernel_size, padding='same',
                       kernel_initializer=_kern_init, **kwargs)(x)
            if use_instance_norm:
                x = inst_norm()(x)                       
            x = LeakyReLU(0.1)(x)
            x = PixelShuffler()(x)
            return x
        return block  
    
    @classmethod
    def Encoder_highres(cls, **kwargs):
        impt = Input(shape=cls.IMAGE_SHAPE)
                
        x = cls.conv(cls.ENCODER_COMPLEXITY)(impt)
        x = cls.conv(cls.ENCODER_COMPLEXITY * 2)(x)
        x = cls.conv(cls.ENCODER_COMPLEXITY * 4)(x)
        if cls.USE_EXTRA_DOWNSCALING: 
            x = cls.conv(cls.ENCODER_COMPLEXITY * 6)(x)        
        x = cls.conv(cls.ENCODER_COMPLEXITY * 8)(x)
        
        dense_shape = cls.IMAGE_SHAPE[0] // 16         
        x = Dense(cls.ENCODER_DIM, kernel_initializer=_kern_init)(Flatten()(x))
        x = Dense(dense_shape * dense_shape * 512, kernel_initializer=_kern_init)(x)
        x = Reshape((dense_shape, dense_shape, 512))(x)
        x = cls.upscale(320)(x)
        
        return KerasModel(impt, x, **kwargs)
    
    def Decoder_highres_A(self):       
        decoder_shape = self.IMAGE_SHAPE[0] // 8        
        inpt = Input(shape=(decoder_shape, decoder_shape, 320))
        
        complexity = 256
        
        x = self.upscale(complexity // 2)(inpt)
        x = self.upscale(complexity // 4)(x)
        x = self.upscale(complexity // 8)(x)
        
        x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)
        
        return KerasModel(inpt, x)        
    
    def Decoder_highres_B(self):               
        decoder_shape = self.IMAGE_SHAPE[0] // 8        
        inpt = Input(shape=(decoder_shape, decoder_shape, 320))
        # 384 192 96
        complexity = 320
        x = self.upscale(complexity)(inpt)
        x = self.upscale(complexity // 2)(x)
        x = self.upscale(complexity // 4)(x)
        
        x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)
        
        return KerasModel(inpt, x)     
             

    @classmethod                              
    def Encoder_standard(cls, **kwargs):
        impt = Input(shape=cls.IMAGE_SHAPE)
                
        x = cls.conv(cls.ENCODER_COMPLEXITY, use_instance_norm=True)(impt)
        x = cls.conv(cls.ENCODER_COMPLEXITY * 2, use_instance_norm=True)(x)
        x = cls.conv(cls.ENCODER_COMPLEXITY * 4)(x)
        if cls.USE_EXTRA_DOWNSCALING:
            x = cls.conv(cls.ENCODER_COMPLEXITY * 6)(x)
        x = cls.conv(cls.ENCODER_COMPLEXITY * 8)(x)
        
        dense_shape = cls.IMAGE_SHAPE[0] // 16         
        x = Dense(cls.ENCODER_DIM, kernel_initializer=_kern_init)(Flatten()(x))
        x = Dense(dense_shape * dense_shape * 512, kernel_initializer=_kern_init)(x)
        x = Reshape((dense_shape, dense_shape, 512))(x)
        
        x = cls.upscale(512)(x)
        
        return KerasModel(impt, x, **kwargs)             
    
    @classmethod
    def Decoder_standard_A(cls):       
        decoder_shape = cls.IMAGE_SHAPE[0]//8        
        inpt = Input(shape=(decoder_shape, decoder_shape, 512))
        
        x = cls.upscale(cls.DECODER_A_COMPLEXITY)(inpt)
        x = cls.upscale(cls.DECODER_A_COMPLEXITY // 2)(x)
        x = cls.upscale(cls.DECODER_A_COMPLEXITY // 4)(x)
        
        x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)
        
        return KerasModel(inpt, x)    
    
    @classmethod
    def Decoder_standard_B(cls):       
        decoder_shape = cls.IMAGE_SHAPE[0] // 8        
        inpt = Input(shape=(decoder_shape, decoder_shape, cls.DECODER_B_COMPLEXITY))
        
        x = cls.upscale(cls.DECODER_B_COMPLEXITY)(inpt)
        x = cls.res_block(x, cls.DECODER_B_COMPLEXITY)                
        x = cls.upscale(cls.DECODER_B_COMPLEXITY // 2)(x)
        x = cls.res_block(x, cls.DECODER_B_COMPLEXITY // 2)        
        x = cls.upscale(cls.DECODER_B_COMPLEXITY // 4)(x)
        x = cls.res_block(x, cls.DECODER_B_COMPLEXITY // 4)
        
        x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)
        
        return KerasModel(inpt, x)        
        
    @classmethod                      
    def Encoder_original(cls, **kwargs):
        impt = Input(shape=cls.IMAGE_SHAPE)        
        
        x = cls.conv(cls.ENCODER_COMPLEXITY)(impt)
        x = cls.conv_sep(cls.ENCODER_COMPLEXITY * 2)(x)
        x = cls.conv(cls.ENCODER_COMPLEXITY * 4)(x)
        x = cls.conv_sep(cls.ENCODER_COMPLEXITY * 8)(x)
        
        dense_shape = cls.IMAGE_SHAPE[0] // 16         
        x = Dense(cls.ENCODER_DIM, kernel_initializer=_kern_init)(Flatten()(x))
        x = Dense(dense_shape * dense_shape * 512, kernel_initializer=_kern_init)(x)
        x = Reshape((dense_shape, dense_shape, 512))(x)
        x = cls.upscale(512)(x)
        
        return KerasModel(impt, x, **kwargs)                

    def Decoder_original_A(self):       
        decoder_shape = self.IMAGE_SHAPE[0] // 8        
        inpt = Input(shape=(decoder_shape, decoder_shape, 512))
        
        x = self.upscale(384)(inpt)
        x = self.upscale(224)(x)
        x = self.upscale(128)(x)
        
        x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)

        return KerasModel(inpt, x)    

    Decoder_original_B = Decoder_original_A
    

    def save_weights(self):        
        model_dir = str(self.model_dir)
        
        from multiprocessing import cpu_count        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        print('\nbacking up the data', end='', flush=True)                            
        
        with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
            futures = [executor.submit(backup_file, model_dir, model) for model in hdf.values()]
            for future in as_completed(futures):
                future.result()
                print('.', end='', flush=True)  
                           
        state_dir = os.path.join(model_dir, 'state_{version_str}.json'.format(**globals()))
        ser = lib.Serializer.get_serializer('json')
        
        try:
            with open(state_dir, 'wb') as fp:                
                state = self.state                
                state_json = ser.marshal(state)
                fp.write(state_json.encode('utf-8'))
                
        except IOError as e:
            print(e.strerror)                   
        
        print('\nsaving model weights', end='', flush=True)                
        
        with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
            futures = [executor.submit(getattr(self, mdl_name.rstrip('H5')).save_weights, str(self.model_dir / mdl_H5_fn)) for mdl_name, mdl_H5_fn in hdf.items()]
            for future in as_completed(futures):
                future.result()
                print('.', end='', flush=True)  

        print('done', flush=True)
        
        
    def _new_state(self):       
        return { 'epoch_no' : 0,
         'USE_DSSIM' : USE_DSSIM,
         'USE_SUBPIXEL' : USE_SUBPIXEL,
         'ENCODER_DIM' :  self.ENCODER_DIM,      
         'IMAGE_SHAPE' : self.IMAGE_SHAPE,
         'USE_EXTRA_DOWNSCALING' : self.USE_EXTRA_DOWNSCALING,
         'ENCODER_COMPLEXITY' : self.ENCODER_COMPLEXITY
         }
        
    def _load_state(self):
        serializer = lib.Serializer.get_serializer('json')
        model_dir = str(self.model_dir)
        state_fn = os.path.join(model_dir, 'state_{}.json'.format(version_str))
                                
        if os.path.exists(os.path.join(model_dir, 'state_{}_original.json'.format( version_str ))):
            os.rename(os.path.join(model_dir, 'state_{}_original.json'.format( version_str )), state_fn)
                
        with open(state_fn, 'rb') as fp:
            state = serializer.unmarshal(fp.read().decode('utf-8'))
            if self.encoder_type in state:
                if not 'USE_EXTRA_DOWNSCALING' in state[self.encoder_type]:
                    state[self.encoder_type]['USE_EXTRA_DOWNSCALING'] = self.USE_EXTRA_DOWNSCALING          
                if not 'ENCODER_COMPLEXITY' in state[self.encoder_type]:
                    state[self.encoder_type]['ENCODER_COMPLEXITY'] = self.ENCODER_COMPLEXITY   
            else:       
                if 'epoch_no' in state:
                    if not EncoderType.ORIGINAL.value in state:
                        state[EncoderType.ORIGINAL.value] = {}                        
                    state[EncoderType.ORIGINAL.value]['epoch_no'] = state['epoch_no']
                if not self.encoder_type in state:
                    state[self.encoder_type] = self._new_state()                                            
        return state   
    
    @property
    def epoch_no(self):
        return self.state[self.encoder_type]['epoch_no']
        
    @epoch_no.setter
    def epoch_no(self, value):
        self.state[self.encoder_type]['epoch_no'] = value    
    
    @property    
    def current_state(self):
        return self.state[self.encoder_type]
    
    @property
    def state(self):        
        try:
            return self._state
        except AttributeError:
            pass
        
        try:
            print('Loading training info ..')
            self._state = self._load_state()
        except IOError as e:
            import errno
            if e.errno==errno.ENOENT:
                print('No training info found.')
            else:
                print('Error loading training info:', e.strerror)
            self._state = { self.encoder_type : self._new_state() }            
        except JSONDecodeError as e:
            print('Error loading training info:', e.msg)
            self._state = { self.encoder_type : self._new_state() }
            
        return self._state
 
    @property
    def gpus(self):
        return self._gpus
    
    @property
    def is_new_training(self):
        return self.epoch_no <= 1
    
    @property
    def encoder_type(self): 
        return self._encoder_type.value
    
    @property
    def model_name(self):
        try:
            return self._model_name
        except AttributeError:
            import inspect
            self._model_name = os.path.dirname(inspect.getmodule(self).__file__).rsplit("_", 1)[1]            
        return self._model_name
             
    
    def __str__(self):
        return "<{}: v={}, enc={}, encoder_dim={}, img_shape={}>".format(self.model_name, 
                                                              version_str, 
                                                              self._encoder_type.name,
                                                              self.ENCODER_DIM, 
                                                              "x".join([str(n) for n in self.IMAGE_SHAPE[:2]]))
    
        
