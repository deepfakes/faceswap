from models import ModelBase
from models import TrainingDataType
import numpy as np

from nnlib import DSSIMMaskLossClass
from nnlib import conv
from nnlib import upscale
from facelib import FaceType

class Model(ModelBase):

    encoderH5 = 'encoder.h5'
    decoder_srcH5 = 'decoder_src.h5'
    decoder_dstH5 = 'decoder_dst.h5'

    #override
    def get_model_name(self):
        return "H64"
 
    #override
    def onInitialize(self, batch_size=-1, **in_options):
        if self.gpu_total_vram_gb < 2:
            raise Exception ('Sorry, this model works only on 2GB+ GPU')
            
        self.batch_size = batch_size
        if self.batch_size == 0:
            if self.gpu_total_vram_gb == 2:
                self.batch_size = 1
            elif self.gpu_total_vram_gb == 3:
                self.batch_size = 4
            elif self.gpu_total_vram_gb == 4:
                self.batch_size = 8
            elif self.gpu_total_vram_gb == 5:
                self.batch_size = 16
            elif self.gpu_total_vram_gb < 12: 
                self.batch_size = 32
            else:    
                self.batch_size = 64
                
        ae_input_layer = self.keras.layers.Input(shape=(64, 64, 3))
        mask_layer = self.keras.layers.Input(shape=(64, 64, 1)) #same as output
        
        self.encoder = self.Encoder(ae_input_layer, self.created_vram_gb)
        self.decoder_src = self.Decoder(self.created_vram_gb)
        self.decoder_dst = self.Decoder(self.created_vram_gb)
        
        if not self.is_first_run():
            self.encoder.load_weights     (self.get_strpath_storage_for_file(self.encoderH5))
            self.decoder_src.load_weights (self.get_strpath_storage_for_file(self.decoder_srcH5))
            self.decoder_dst.load_weights (self.get_strpath_storage_for_file(self.decoder_dstH5))

        self.autoencoder_src = self.keras.models.Model([ae_input_layer,mask_layer], self.decoder_src(self.encoder(ae_input_layer)))
        self.autoencoder_dst = self.keras.models.Model([ae_input_layer,mask_layer], self.decoder_dst(self.encoder(ae_input_layer)))

        if self.is_training_mode:
            self.autoencoder_src, self.autoencoder_dst = self.to_multi_gpu_model_if_possible ( [self.autoencoder_src, self.autoencoder_dst] )
        
        optimizer = self.keras.optimizers.Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)
        self.autoencoder_src.compile(optimizer=optimizer, loss=[DSSIMMaskLossClass(self.tf)(mask_layer), 'mae'])
        self.autoencoder_dst.compile(optimizer=optimizer, loss=[DSSIMMaskLossClass(self.tf)(mask_layer), 'mae'])
  
        if self.is_training_mode:
            from models import TrainingDataGenerator
            f = TrainingDataGenerator.SampleTypeFlags 
            self.set_training_data_generators ([            
                    TrainingDataGenerator(self, TrainingDataType.SRC,  batch_size=self.batch_size, output_sample_types_flags=[ f.WARPED | f.HALF_FACE | f.MODE_BGR | f.SIZE_64, f.TARGET | f.HALF_FACE | f.MODE_BGR | f.SIZE_64, f.TARGET | f.HALF_FACE | f.MODE_M | f.SIZE_64], random_flip=True ),
                    TrainingDataGenerator(self, TrainingDataType.DST,  batch_size=self.batch_size, output_sample_types_flags=[ f.WARPED | f.HALF_FACE | f.MODE_BGR | f.SIZE_64, f.TARGET | f.HALF_FACE | f.MODE_BGR | f.SIZE_64, f.TARGET | f.HALF_FACE | f.MODE_M | f.SIZE_64] )
                ])
                
    #override
    def onSave(self):        
        self.save_weights_safe( [[self.encoder, self.get_strpath_storage_for_file(self.encoderH5)],
                                [self.decoder_src, self.get_strpath_storage_for_file(self.decoder_srcH5)],
                                [self.decoder_dst, self.get_strpath_storage_for_file(self.decoder_dstH5)]] )
        
    #override
    def onTrainOneEpoch(self, sample):
        warped_src, target_src, target_src_mask = sample[0]
        warped_dst, target_dst, target_dst_mask = sample[1]    
        
        loss_src = self.autoencoder_src.train_on_batch( [warped_src, target_src_mask], [target_src, target_src_mask] )
        loss_dst = self.autoencoder_dst.train_on_batch( [warped_dst, target_dst_mask], [target_dst, target_dst_mask] )

        return ( ('loss_src', loss_src[0]), ('loss_dst', loss_dst[0]) )
        
    #override
    def onGetPreview(self, sample):
        test_A   = sample[0][1][0:4] #first 4 samples
        test_A_m = sample[0][2][0:4]
        test_B   = sample[1][1][0:4]
        test_B_m = sample[1][2][0:4]
        
        AA, mAA = self.autoencoder_src.predict([test_A, test_A_m])                                       
        AB, mAB = self.autoencoder_src.predict([test_B, test_B_m])
        BB, mBB = self.autoencoder_dst.predict([test_B, test_B_m])
        
        mAA = np.repeat ( mAA, (3,), -1)
        mAB = np.repeat ( mAB, (3,), -1)
        mBB = np.repeat ( mBB, (3,), -1)
        
        st = []
        for i in range(0, len(test_A)):
            st.append ( np.concatenate ( (
                test_A[i,:,:,0:3],
                AA[i],
                #mAA[i],
                test_B[i,:,:,0:3], 
                BB[i], 
                #mBB[i],                
                AB[i],
                #mAB[i]
                ), axis=1) )
            
        return [ ('src, dst, src->dst', np.concatenate ( st, axis=0 ) ) ]

    def predictor_func (self, face):
        
        face_64_bgr = face[...,0:3]
        face_64_mask = np.expand_dims(face[...,3],-1)
        
        x, mx = self.autoencoder_src.predict ( [ np.expand_dims(face_64_bgr,0), np.expand_dims(face_64_mask,0) ] )
        x, mx = x[0], mx[0]     
        
        return np.concatenate ( (x,mx), -1 )

    #override
    def get_converter(self, **in_options):
        from models import ConverterMasked
        
        if 'masked_hist_match' not in in_options.keys() or in_options['masked_hist_match'] == None:
            in_options['masked_hist_match'] = True

        if 'erode_mask_modifier' not in in_options.keys():
            in_options['erode_mask_modifier'] = 0
        in_options['erode_mask_modifier'] += 100
            
        if 'blur_mask_modifier' not in in_options.keys():
            in_options['blur_mask_modifier'] = 0
        in_options['blur_mask_modifier'] += 100
        
        return ConverterMasked(self.predictor_func, predictor_input_size=64, output_size=64, face_type=FaceType.HALF, **in_options)
        
    def Encoder(self, input_layer, created_vram_gb):
        x = input_layer
        if created_vram_gb >= 4:
            x = conv(self.keras, x, 128)
            x = conv(self.keras, x, 256)
            x = conv(self.keras, x, 512)
            x = conv(self.keras, x, 1024)
            x = self.keras.layers.Dense(1024)(self.keras.layers.Flatten()(x))
            x = self.keras.layers.Dense(4 * 4 * 1024)(x)
            x = self.keras.layers.Reshape((4, 4, 1024))(x)
            x = upscale(self.keras, x, 512)
        else:
            x = conv(self.keras, x, 128 )
            x = conv(self.keras, x, 256 )
            x = conv(self.keras, x, 512 )
            x = conv(self.keras, x, 768 )
            x = self.keras.layers.Dense(512)(self.keras.layers.Flatten()(x))
            x = self.keras.layers.Dense(4 * 4 * 512)(x)
            x = self.keras.layers.Reshape((4, 4, 512))(x)
            x = upscale(self.keras, x, 256)
            
        return self.keras.models.Model(input_layer, x)

    def Decoder(self, created_vram_gb):
        if created_vram_gb >= 4:    
            input_ = self.keras.layers.Input(shape=(8, 8, 512))
        else:
            input_ = self.keras.layers.Input(shape=(8, 8, 256))
            
        x = input_
        x = upscale(self.keras, x, 256)
        x = upscale(self.keras, x, 128)
        x = upscale(self.keras, x, 64)
        
        y = input_  #mask decoder
        y = upscale(self.keras, y, 256)
        y = upscale(self.keras, y, 128)
        y = upscale(self.keras, y, 64)
        
        x = self.keras.layers.convolutional.Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)
        y = self.keras.layers.convolutional.Conv2D(1, kernel_size=5, padding='same', activation='sigmoid')(y)
        
        
        return self.keras.models.Model(input_, [x,y])
