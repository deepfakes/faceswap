#!/usr/bin/env python3
""" A lightweight variant of DFaker Model
    Based on the dfaker model: https://github.com/dfaker """

import sys
import types

from keras.initializers import RandomNormal
from keras.layers import Dense, Flatten, Input, Reshape, AveragePooling2D, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2DTranspose
from keras.layers.merge import Concatenate
from keras.models import Model as KerasModel

from ._base import logger
from .original import Model as OriginalModel
from lib.utils import FaceswapError
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout

# [P] TODO Move upscale2x_hyb to nnblocks.py (after testing)
# [P] TODO Remove Windows Specific code (yeah, I know you hate me)

# <<< DeLight Model Blocks >>> #
def upscale2x_hyb(self, inp, filters, kernel_size=3, padding='same',
                  sr_ratio=0.5, scale_factor=2, interpolation='bilinear',
                  res_block_follows=False, **kwargs):
    """Hybrid Upscale Layer"""
    name = self.get_name("upscale2x_hyb")
    var_x = inp

    sr_filters = int(filters * sr_ratio)
    upscale_filters = filters - sr_filters

    var_x_sr = self.upscale(var_x, upscale_filters, kernel_size=kernel_size,
                            padding=padding, scale_factor=scale_factor, res_block_follows=res_block_follows, **kwargs)    
    if upscale_filters>0:
        var_x_us = self.conv2d(var_x, upscale_filters,  kernel_size=3, padding=padding,
                        name="{}_conv2d".format(name), **kwargs)
        var_x_us = UpSampling2D(size=(scale_factor, scale_factor), interpolation=interpolation,
                            name="{}_upsampling2D".format(name))(var_x_us) 
        var_x = Concatenate(name="{}_concatenate".format(name))([var_x_sr, var_x_us])                                            
    else:
        var_x = var_x_sr

    return var_x

class Model(OriginalModel):
    """ Improved Autoencoder Model """

    def __init__(self, *args, **kwargs):
        logger.debug("Initializing %s: (args: %s, kwargs: %s",
                     self.__class__.__name__, args, kwargs)

        if sys.platform == 'win32':
            from ctypes import windll
            windll.user32.SetProcessDPIAware(True)

        kwargs["input_shape"] = (128, 128, 3)
        kwargs["encoder_dim"] = -1
        self.kernel_initializer = RandomNormal(0, 0.02)
        self.dense_output = None
        self.detail_level = None
        super().__init__(*args, **kwargs)

        logger.debug("Initialized %s", self.__class__.__name__)

    def _detail_level_setup(self):
        logger.debug('self.config[output_size]: %d', self.config["output_size"])

        self.features = {
            'lowmem': 0,
            'fair':  1,
            'best':  2,
            }[self.config["features"]]
        logger.debug('self.features: %d', self.features)        
        
        self.encoder_filters = 64 if self.features > 0 else 48 
        logger.debug('self.encoder_filters: %d', self.encoder_filters) 
        bonum_fortunam = 128
        self.encoder_dim = {
            0: 512 + bonum_fortunam,
            1: 1024 + bonum_fortunam,
            2: 1536 + bonum_fortunam,
            }[self.features]
        logger.debug('self.encoder_dim: %d', self.encoder_dim)
        
        self.details = {
            'fast': 0,
            'good':  1,
            }[self.config["details"]]
        logger.debug('self.details: %d', self.details)        

        try:
            self.upscale_ratio = {
                128: 2,
                256: 4,
                384: 6
                }[self.config["output_size"]]
        except KeyError:
            logger.error("Config error: output_size must be one of: 128, 256, or 384.")
            raise FaceswapError("Config error: output_size must be one of: 128, 256, or 384.")
        logger.debug('output_size: %r', self.config["output_size"])        
        logger.debug('self.upscale_ratio: %r', self.upscale_ratio)
    
    def build(self):
        self._detail_level_setup()
        self.blocks.upscale2x_hyb = types.MethodType(upscale2x_hyb, self.blocks)
                                
        super().build()

    def add_networks(self):
        """ Add the realface model weights """
        logger.debug("Adding networks")
        self.add_network("decoder", "a", self.decoder_a(), is_output=True)
        self.add_network("decoder", "b", self.decoder_b(), is_output=True)
        self.add_network("encoder", None, self.encoder())
        logger.debug("Added networks")
        
    def compile_predictors(self, **kwargs):
        self.set_networks_trainable()
        super().compile_predictors(**kwargs)        
        
    def set_networks_trainable(self):
        train_encoder = True
        train_decoder_a = True
        train_decoder_b = True        
         
        encoder = self.networks['encoder'].network        
        for layer in encoder.layers:
            layer.trainable = train_encoder            
        
        decoder_a = self.networks['decoder_a'].network            
        for layer in decoder_a.layers:
            layer.trainable = train_decoder_a            
            
        decoder_b = self.networks['decoder_b'].network                        
        for layer in decoder_b.layers:
            layer.trainable = train_decoder_b        

    def encoder(self):
        """ DeLight Encoder Network """        
        input_ = Input(shape=self.input_shape)
        var_x = input_
        
        var_x1 = self.blocks.conv(var_x, self.encoder_filters // 2)
        var_x2 = AveragePooling2D()(var_x)
        var_x2 = LeakyReLU(0.1)(var_x2)
        var_x = Concatenate()([var_x1, var_x2])
        
        var_x1 = self.blocks.conv(var_x, self.encoder_filters)
        var_x2 = AveragePooling2D()(var_x)
        var_x2 = LeakyReLU(0.1)(var_x2)
        var_x = Concatenate()([var_x1, var_x2])
        
        var_x1 = self.blocks.conv(var_x, self.encoder_filters * 2)
        var_x2 = AveragePooling2D()(var_x)
        var_x2 = LeakyReLU(0.1)(var_x2)
        var_x = Concatenate()([var_x1, var_x2])
        
        var_x1 = self.blocks.conv(var_x, self.encoder_filters * 4)
        var_x2 = AveragePooling2D()(var_x)
        var_x2 = LeakyReLU(0.1)(var_x2)
        var_x = Concatenate()([var_x1, var_x2])
        
        var_x1 = self.blocks.conv(var_x, self.encoder_filters * 8)
        var_x2 = AveragePooling2D()(var_x)
        var_x2 = LeakyReLU(0.1)(var_x2)
        var_x = Concatenate()([var_x1, var_x2])        

        var_x = Dense(self.encoder_dim)(Flatten()(var_x))
        var_x = Dropout(0.05)(var_x)
        var_x = Dense(4 * 4 * 1024)(var_x)
        var_x = Dropout(0.05)(var_x)
        var_x = Reshape((4, 4, 1024))(var_x)

        return KerasModel(input_, var_x)

    def decoder_a(self):
        """ AD Model 4 Decoder A(old face) Network """
        input_ = Input(shape=(4, 4, 1024))        
        decoder_a_complexity = 256
        mask_complexity = 128
        
        var_xy = input_                           
        var_xy = UpSampling2D(self.upscale_ratio, interpolation='bilinear')(var_xy)        
        
        var_x = var_xy        
        var_x = self.blocks.upscale2x_hyb(var_x, decoder_a_complexity)
        var_x = self.blocks.upscale2x_hyb(var_x, decoder_a_complexity // 2)
        var_x = self.blocks.upscale2x_hyb(var_x, decoder_a_complexity // 4)
        var_x = self.blocks.upscale2x_hyb(var_x, decoder_a_complexity // 8)
    
        var_x = self.blocks.conv2d(var_x, 3, kernel_size=5, padding="same", activation="sigmoid", name="face_out")

        outputs = [var_x]
        
        if self.config.get("mask_type", False):                                    
            var_y = var_xy  # mask decoder                      
            var_y = self.blocks.upscale2x_hyb(var_y, mask_complexity)
            var_y = self.blocks.upscale2x_hyb(var_y, mask_complexity // 2)
            var_y = self.blocks.upscale2x_hyb(var_y, mask_complexity // 4)
            var_y = self.blocks.upscale2x_hyb(var_y, mask_complexity // 8)                           

            var_y = self.blocks.conv2d(var_y, 1, kernel_size=5, padding="same", activation="sigmoid", name="mask_out")
            
            outputs.append(var_y)

        return KerasModel([input_], outputs=outputs)
    
    def decoder_b(self):
        """ AD Model 4 - Fast Decoder B(new face) Network  """        
        input_ = Input(shape=(4, 4, 1024))        

        decoder_b_complexity = 512
        mask_complexity = 128
        
        var_xy = input_

        var_xy = self.blocks.upscale2x_hyb(var_xy, 512, scale_factor=self.upscale_ratio) #, sr_ratio=0.5 if self.upscale_ratio!=6 else None)  
        
        var_x = var_xy
        
        var_x = self.blocks.res_block(var_x, 512, use_bias=True)
        var_x = self.blocks.res_block(var_x, 512, use_bias=False)
        var_x = self.blocks.res_block(var_x, 512, use_bias=False)
        var_x = self.blocks.upscale2x_hyb(var_x, decoder_b_complexity)
        var_x = self.blocks.res_block(var_x, decoder_b_complexity, use_bias=True)
        var_x = self.blocks.res_block(var_x, decoder_b_complexity, use_bias=False)
        var_x = BatchNormalization()(var_x)
        var_x = self.blocks.upscale2x_hyb(var_x, decoder_b_complexity // 2) 
        var_x = self.blocks.res_block(var_x, decoder_b_complexity // 2, use_bias=True)                                            
        var_x = self.blocks.upscale2x_hyb(var_x, decoder_b_complexity // 4)            
        var_x = self.blocks.res_block(var_x, decoder_b_complexity // 4, use_bias=False)
        var_x = BatchNormalization()(var_x)                 
        var_x = self.blocks.upscale2x_hyb(var_x, decoder_b_complexity // 8)
                         
        var_x = self.blocks.conv2d(var_x, 3, kernel_size=5, padding="same", activation="sigmoid", name="face_out")

        outputs = [var_x]

        if self.config.get("mask_type", False):                                    
            var_y = var_xy  # mask decoder          
            
            var_y = self.blocks.upscale2x_hyb(var_y, mask_complexity)
            var_y = self.blocks.upscale2x_hyb(var_y, mask_complexity // 2)
            var_y = self.blocks.upscale2x_hyb(var_y, mask_complexity // 4)
            var_y = self.blocks.upscale2x_hyb(var_y, mask_complexity // 8)                           

            var_y = self.blocks.conv2d(var_y, 1, kernel_size=5, padding="same", activation="sigmoid", name="mask_out")
            
            outputs.append(var_y)

        return KerasModel([input_], outputs=outputs)   
