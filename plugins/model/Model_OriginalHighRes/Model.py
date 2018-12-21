# Based on the original https://www.reddit.com/r/deepfakes/ code sample + contribs
# Based on https://github.com/iperov/DeepFaceLab a better 128x Decoder idea, K.function impl
# Based on the https://github.com/shaoanlu/faceswap-GAN repo res_block chain and IN
# source : https://github.com/shaoanlu/faceswap-GAN/blob/master/FaceSwap_GAN_v2_sz128_train.ipynbtemp/faceswap_GAN_keras.ipynb

from concurrent.futures import ThreadPoolExecutor, as_completed
import enum
from json import JSONDecodeError
import logging
import os
import sys
import configparser
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from keras.initializers import RandomNormal
from keras.layers import Input, Dense, Flatten, Reshape
from keras.layers import SeparableConv2D, add
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.layers.core import SpatialDropout2D
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

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
mswindows = sys.platform=="win32"

_kern_init = RandomNormal(0, 0.02)

# autoencoder type *
# basic encoder for this model type
# new, balanced encoder they way I meant it to be; more memory consuming
# high resolution tensors optimized encoder: 176x and on
EncoderType = enum.Enum('EncoderType', 'ORIGINAL STANDARD HIGHRES')

MODEL_CFG_FN = 'faceswap.ini'


class Model():

    conf_filename = os.path.join(os.path.dirname(sys.modules['__main__'].__file__), MODEL_CFG_FN)
    _conf = configparser.ConfigParser()         

    def __init__(self, model_dir, gpus):                
        if mswindows:  
            from ctypes import cdll    
            mydll = cdll.LoadLibrary("user32.dll")
            mydll.SetProcessDPIAware(True)

        self.load_configuration()    
        
        global hdf
         
        hdf = {'encoderH5': 'encoder_{}{}.h5'.format( version_str, self.encoder_name.lower() ),
               'decoder_AH5': 'decoder_A_{}{}.h5'.format( version_str, self.encoder_name.lower() ),
               'decoder_BH5': 'decoder_B_{}{}.h5'.format( version_str, self.encoder_name.lower() ) }
                
        self.model_dir = model_dir

        # can't chnage gpu's when the model is initialized no point in making it r/w
        self._gpus = gpus 
        
        if not self.is_new_training:                        
            for fname in self.conf:
                fname = fname.upper()                
                setattr(self.__class__, fname, self.current_state[fname])                                
        
        self.encoder, self.decoder_A, self.decoder_B = getattr(self, "build_{}".format(self.encoder_name.lower()))()
        
        self.initModel()        
    
    def initModel(self):
        optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)
                        
        input_A_bgr = Input(shape=self.IMAGE_SHAPE)
        input_B_bgr = Input(shape=self.IMAGE_SHAPE)        
        
        rec_A_bgr = self.decoder_A(self.encoder(input_A_bgr))
        rec_B_bgr = self.decoder_B(self.encoder(input_B_bgr))                   
        
        if self.USE_DSSIM:
            from plugins.Model_OriginalHighRes.dssim import DSSIMObjective
            loss_func = DSSIMObjective()
            print('Using DSSIM loss ..', flush=True)
        else:
            loss_func = 'mean_absolute_error'
            
        if self.USE_SUBPIXEL:
            from plugins.Model_OriginalHighRes.subpixel import SubPixelUpscaling
            self.upscale = self.upscale_sub
            print('Using subpixel upscaling ..', flush=True)            
        
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
            logger.info('loaded model weights')
            return True
        except IOError as e:
            logger.warning('Error loading training info: %s', str(e.strerror))
        except Exception as e:
            logger.warning('Error loading training info: %s', str(e))
            
        self.epoch_no = 1 
      
        return False

    def converter(self, swap):
        autoencoder = self.autoencoder_B if not swap else self.autoencoder_A
        return autoencoder.predict

    @staticmethod
    def inst_norm(): 
        return InstanceNormalization()
    
    @classmethod
    def conv(cls, filters, kernel_size=5, strides=2, use_instance_norm=False, **kwargs):
        def block(x):
            x = Conv2D(filters, kernel_size=kernel_size, strides=strides, 
                       kernel_initializer=_kern_init, padding='same', **kwargs)(x)
            if use_instance_norm:
                x = cls.inst_norm()(x)                                
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
    
    @classmethod
    def conv_sep(cls, filters, kernel_size=5, strides=2, use_instance_norm=False, **kwargs):
        def block(x):
            x = SeparableConv2D(filters, kernel_size=kernel_size, strides=strides, 
                       kernel_initializer=_kern_init, padding='same', **kwargs)(x)
            if use_instance_norm:
                x = cls.inst_norm()(x)                                
            x = LeakyReLU(0.1)(x)            
            return x
        return block

    @classmethod
    def upscale_sub(cls, filters, kernel_size=3, use_instance_norm=False, **kwargs):
        def block(x):
            x = Conv2D(filters * 4, kernel_size=kernel_size, padding='same',
                       kernel_initializer=_kern_init, **kwargs)(x)
            if use_instance_norm:
                x = cls.inst_norm()(x)                       
            x = LeakyReLU(0.1)(x)
            x = SubPixelUpscaling()(x)
            return x
        return block
              
    @classmethod
    def upscale(cls, filters, kernel_size=3, use_instance_norm=False, **kwargs):
        def block(x):
            x = Conv2D(filters * 4, kernel_size=kernel_size, padding='same',
                       kernel_initializer=_kern_init, **kwargs)(x)
            if use_instance_norm:
                x = cls.inst_norm()(x)                       
            x = LeakyReLU(0.1)(x)            
            x = PixelShuffler()(x)
            return x
        return block  
    
    @classmethod
    def build_highres(cls):
        
        def Encoder(input_shape):
            impt = Input(shape=input_shape)
                    
            x = cls.conv(cls.ENCODER_COMPLEXITY)(impt)
            x = cls.conv(cls.ENCODER_COMPLEXITY * 2)(x)
            x = cls.conv(cls.ENCODER_COMPLEXITY * 4)(x)        
            x = cls.conv(cls.ENCODER_COMPLEXITY * 6)(x)        
            x = cls.conv(cls.ENCODER_COMPLEXITY * 8)(x)
            
            dense_shape = cls.IMAGE_SHAPE[0] // 16         
            x = Dense(cls.ENCODER_DIM, kernel_initializer=_kern_init)(Flatten()(x))
            x = Dense(dense_shape * dense_shape * 512, kernel_initializer=_kern_init)(x)
            x = Reshape((dense_shape, dense_shape, 512))(x)
            x = cls.upscale(320)(x)
            
            return KerasModel(impt, x)
    
        def Decoder_A():       
            decoder_shape = cls.IMAGE_SHAPE[0] // 8        
            inpt = Input(shape=(decoder_shape, decoder_shape, 320))
            
            complexity = 256
            
            x = cls.upscale(complexity)(inpt)
            x = cls.upscale(complexity // 2)(x)
            x = cls.upscale(complexity // 4)(x)
            
            x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)
            
            return KerasModel(inpt, x)        
    
        def Decoder_B():               
            decoder_shape = cls.IMAGE_SHAPE[0] // 8        
            inpt = Input(shape=(decoder_shape, decoder_shape, 320))
            # 384 192 96
            complexity = 320
            x = cls.upscale(complexity)(inpt)
            x = cls.upscale(complexity // 2)(x)
            x = cls.upscale(complexity // 4)(x)
            
            x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)
            
            return KerasModel(inpt, x)    
        
        return Encoder(cls.IMAGE_SHAPE), Decoder_A(), Decoder_B()
             

    @classmethod      
    def build_standard(cls):                        
        def Encoder(input_shape):
            impt = Input(shape=input_shape)
                    
            x = cls.conv(cls.ENCODER_COMPLEXITY, use_instance_norm=True)(impt)
            x = cls.conv(cls.ENCODER_COMPLEXITY * 2, use_instance_norm=True)(x)
            x = cls.conv(cls.ENCODER_COMPLEXITY * 4)(x)
            x = cls.conv(cls.ENCODER_COMPLEXITY * 6)(x)
            x = cls.conv(cls.ENCODER_COMPLEXITY * 8)(x)
            
            dense_shape = cls.IMAGE_SHAPE[0] // 16         
            x = Dense(cls.ENCODER_DIM, kernel_initializer=_kern_init)(Flatten()(x))
            x = Dense(dense_shape * dense_shape * 512, kernel_initializer=_kern_init)(x)
            x = Reshape((dense_shape, dense_shape, 512))(x)
            
            x = cls.upscale(512)(x)
            x = SpatialDropout2D(0.5)(x)        
            
            return KerasModel(impt, x)             
    
        def Decoder_A():       
            decoder_shape = cls.IMAGE_SHAPE[0]//8        
            inpt = Input(shape=(decoder_shape, decoder_shape, 512))
                    
            x = cls.upscale(cls.DECODER_A_COMPLEXITY)(inpt)
            x = SpatialDropout2D(0.25)(x)
            x = cls.upscale(cls.DECODER_A_COMPLEXITY // 2)(x)
            x = cls.upscale(cls.DECODER_A_COMPLEXITY // 4)(x)
            
            x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)
            
            return KerasModel(inpt, x)    
    

        def Decoder_B():       
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
        
        return Encoder(cls.IMAGE_SHAPE), Decoder_A(), Decoder_B()
        
    @classmethod        
    def build_original(cls):                  
        def Encoder(input_shape):
            impt = Input(shape=input_shape)        
            
            x = cls.conv(cls.ENCODER_COMPLEXITY)(impt)
            x = cls.conv_sep(cls.ENCODER_COMPLEXITY * 2)(x)
            x = cls.conv(cls.ENCODER_COMPLEXITY * 4)(x)
            x = cls.conv_sep(cls.ENCODER_COMPLEXITY * 8)(x)
            
            dense_shape = cls.IMAGE_SHAPE[0] // 16         
            x = Dense(cls.ENCODER_DIM, kernel_initializer=_kern_init)(Flatten()(x))
            x = Dense(dense_shape * dense_shape * 512, kernel_initializer=_kern_init)(x)
            x = Reshape((dense_shape, dense_shape, 512))(x)
            x = cls.upscale(512)(x)
            
            return KerasModel(impt, x)                

        def Decoder_B():       
            decoder_shape = cls.IMAGE_SHAPE[0] // 8        
            inpt = Input(shape=(decoder_shape, decoder_shape, 512))
            
            x = cls.upscale(cls.DECODER_B_COMPLEXITY - 128)(inpt)
            x = cls.upscale((cls.DECODER_B_COMPLEXITY // 2 ) - 32)(x)
            x = cls.upscale(cls.DECODER_B_COMPLEXITY // 4)(x)
            
            x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)
    
            return KerasModel(inpt, x)    

        Decoder_A = Decoder_B         
        
        return Encoder(cls.IMAGE_SHAPE), Decoder_A(), Decoder_B()   
        

    def save_weights(self):
        model_dir = str(self.model_dir)
        
        state_dir = os.path.join(model_dir, 'state_{version_str}.json'.format(**globals()))
        ser = lib.Serializer.get_serializer('json')            
        try:
            with open(state_dir, 'wb') as fp:                
                state_json = ser.marshal(self._state)
                fp.write(state_json.encode('utf-8'))            
        except IOError as e:
            logger.error(e.strerror)

        logger.info('saving model weights')

        with ThreadPoolExecutor(max_workers=len(hdf)) as executor:
            futures = [executor.submit(backup_file, model_dir, model) for model in hdf.values()]
            for future in as_completed(futures):
                future.result()
                print('.', end='', flush=True)
            futures = [executor.submit(getattr(self, mdl_name.rstrip('H5')).save_weights, str(self.model_dir / mdl_H5_fn)) for mdl_name, mdl_H5_fn in hdf.items()]
            for future in as_completed(futures):
                future.result()
                print('.', end='', flush=True)
        logger.info('done')
        
    @property
    def epoch_no(self):
        return self.state[self.encoder_name]['epoch_no']
        
    @epoch_no.setter
    def epoch_no(self, value):
        self.state[self.encoder_name]['epoch_no'] = value    
    
    @property    
    def current_state(self):
        return self.state[self.encoder_name]
    
    @property
    def state(self):        
        try:
            return self._state
        except AttributeError:
            pass
        
        try:
            logger.info('Loading training info ..')
            self._state = self._load_state()
        except IOError as e:
            import errno
            if e.errno==errno.ENOENT:
                logger.info('No training info found.')
            else:
                logger.error('Error loading training info: %s', e.strerror)
            self._state = { self.encoder_name : self._new_state() }            
        except JSONDecodeError as e:
            logger.error('Error loading training info: %s', e.msg)
            self._state = { self.encoder_name : self._new_state() }
            
        return self._state        
 
    @property
    def gpus(self):
        return self._gpus

    @property
    def is_new_training(self):
        return self.epoch_no <= 1
    
    @property
    def encoder_name(self): 
        return self.ENCODER_TYPE
    
    @property
    def model_name(self):
        try:
            return self._model_name
        except AttributeError:
            import inspect
            self._model_name = os.path.dirname(inspect.getmodule(self).__file__).rsplit("_", 1)[1]            
        return self._model_name     
    
    @classmethod
    def _save_config(cls):        
        with open(cls.conf_filename, 'w') as configfile:
            cls._conf.write(configfile)
            
    @property
    def conf(self):
        self._conf.read(self.conf_filename)        
        if not self.model_name in self._conf:
            self._conf[self.model_name] = {
                'ENCODER_TYPE': 'ORIGINAL',
                'ENCODER_DIM': '1024',
                'IMAGE_SHAPE': '(128, 128)',
                'ENCODER_COMPLEXITY': '128', # conv layers complexity, sensible ranges 128 - 160 *   
                'DECODER_B_COMPLEXITY': '512',  # only applicable for STANDARD and ORIGINAL encoder *
                'DECODER_A_COMPLEXITY': '384',  # only applicable for STANDARD encoder *   
                'USE_DSSIM': 'no', # use DSSIM objective loss; might increase overall quality
                'USE_SUBPIXEL': 'no' # might increase upscaling quality at cost of video memory
             }
            self._save_config()
            # * - requires re-training of the model        
        return self._conf[self.model_name]    
    
    def load_configuration(self):
        """loads configuration from main ini/conf file"""
        cfg = self.conf
        cls = self.__class__
        
        try:
            cls.ENCODER_TYPE = EncoderType[cfg['ENCODER_TYPE']].name
        except KeyError:
            print("Wrong ENCODER_TYPE '{}' in {}; choices are:".format(cfg['ENCODER_TYPE'], os.path.basename(self.conf_filename)), end='')            
            for key in EncoderType:      
                print(" "+key.name, end='')
            print()
            
        cls.ENCODER_DIM = cfg.getint('ENCODER_DIM')
        
        IMAGE_SHAPE = tuple(min(int(n.strip()), 256) for n in cfg['IMAGE_SHAPE'].strip('()').split(','))    
        IMAGE_WIDTH = IMAGE_SHAPE[0]     
        assert IMAGE_WIDTH >= 16
        IMAGE_WIDTH = (IMAGE_WIDTH//16 + (1 if (IMAGE_WIDTH%16)>=8 else 0))*16        
        cls.IMAGE_SHAPE = IMAGE_WIDTH, IMAGE_WIDTH, len('BRG') # good to let ppl know what these are...    
        
        cls.ENCODER_COMPLEXITY = cfg.getint('ENCODER_COMPLEXITY')
        cls.DECODER_B_COMPLEXITY = cfg.getint('DECODER_B_COMPLEXITY')
        cls.DECODER_A_COMPLEXITY = cfg.getint('DECODER_A_COMPLEXITY')
        cls.USE_DSSIM = cfg.getboolean('USE_DSSIM')
        cls.USE_SUBPIXEL = cfg.getboolean('USE_SUBPIXEL')

    def _new_state(self):       
        state = {'epoch_no' : 0,                 
                 }
        state.update({fname.upper() : self.__class__.__dict__[fname.upper()] for fname in self.conf})   
        return state
        
    def _load_state(self):
        serializer = lib.Serializer.get_serializer('json')
        model_dir = str(self.model_dir)
        state_fn = os.path.join(model_dir, 'state_{}.json'.format(version_str))
                                
        if os.path.exists(os.path.join(model_dir, 'state_{}_original.json'.format( version_str ))):
            os.rename(os.path.join(model_dir, 'state_{}_original.json'.format( version_str )), state_fn)
                
        with open(state_fn, 'rb') as fp:
            state = serializer.unmarshal(fp.read().decode('utf-8'))
            if self.encoder_name in state:
                for fname in self.conf:
                    fname = fname.upper()
                    if not fname in state[self.encoder_name]:
                        state[self.encoder_name][fname] = self.__class__.__dict__[fname]
            else:       
                if 'epoch_no' in state:
                    if not EncoderType.ORIGINAL.value in state:
                        state[EncoderType.ORIGINAL.value] = {}                        
                    state[EncoderType.ORIGINAL.value]['epoch_no'] = state['epoch_no']
                if not self.encoder_name in state:
                    state[self.encoder_name] = self._new_state()                                            
        return state        
    
    def __str__(self):
        return "<{}: v={}, enc={}, encoder_dim={}, img_shape={}>".format(self.model_name, 
                                                              version_str, 
                                                              self.encoder_name,
                                                              self.ENCODER_DIM, 
                                                              "x".join([str(n) for n in self.IMAGE_SHAPE[:2]]))       
    
