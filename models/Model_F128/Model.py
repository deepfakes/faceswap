from models import ModelBase
from models import TrainingDataType
import numpy as np
import cv2

from nnlib import conv
from nnlib import upscale
from nnlib import res

class Model(ModelBase):

    encoderH5 = 'encoder.h5'
    decoder_srcH5 = 'decoder_src.h5'
    decoder_dstH5 = 'decoder_dst.h5'

    #override
    def get_model_name(self):
        return "F128"
 
    #override
    def onInitialize(self, batch_size=-1, **in_options):
        if self.gpu_total_vram_gb < 3:
            raise Exception ('Sorry, this model works only on 3GB+ GPU')
            
        self.batch_size = batch_size
        if self.batch_size == 0:          
            if self.gpu_total_vram_gb == 3:
                self.batch_size = 1
            elif self.gpu_total_vram_gb == 4:
                self.batch_size = 4
            elif self.gpu_total_vram_gb <= 6:
                self.batch_size = 8
            elif self.gpu_total_vram_gb < 12: 
                self.batch_size = 16
            else:    
                self.batch_size = 32
                
        ae_input_layer = self.keras.layers.Input(shape=(64, 64, 4))

        self.encoder = self.Encoder(ae_input_layer, self.created_vram_gb)
        self.decoder_src = self.Decoder(self.created_vram_gb)
        self.decoder_dst = self.Decoder(self.created_vram_gb)
        
        if not self.is_first_run():
            self.encoder.load_weights     (self.get_strpath_storage_for_file(self.encoderH5))
            self.decoder_src.load_weights (self.get_strpath_storage_for_file(self.decoder_srcH5))
            self.decoder_dst.load_weights (self.get_strpath_storage_for_file(self.decoder_dstH5))

        self.autoencoder_src = self.keras.models.Model(ae_input_layer, self.decoder_src(self.encoder(ae_input_layer)))
        self.autoencoder_dst = self.keras.models.Model(ae_input_layer, self.decoder_dst(self.encoder(ae_input_layer)))

        if self.is_training_mode:
            self.autoencoder_src, self.autoencoder_dst = self.to_multi_gpu_model_if_possible ( [self.autoencoder_src, self.autoencoder_dst] )
                
        optimizer = self.keras.optimizers.Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)        
        self.autoencoder_src.compile(optimizer=optimizer, loss='mae')
        self.autoencoder_dst.compile(optimizer=optimizer, loss='mae')
  
        if self.is_training_mode:
            from models import FullFaceTrainingDataGenerator
            self.set_training_data_generators ([
                    FullFaceTrainingDataGenerator(self, TrainingDataType.SRC_WITH_NEAREST, batch_size=self.batch_size, warped_size=(64,64), target_size=(128,128), random_flip=True ),
                    FullFaceTrainingDataGenerator(self, TrainingDataType.DST,              batch_size=self.batch_size, warped_size=(64,64), target_size=(128,128) )
                ])
            
    #override
    def onSave(self):        
        self.encoder.save_weights    (self.get_strpath_storage_for_file(self.encoderH5))
        self.decoder_src.save_weights(self.get_strpath_storage_for_file(self.decoder_srcH5))
        self.decoder_dst.save_weights(self.get_strpath_storage_for_file(self.decoder_dstH5))
        
    #override
    def onTrainOneEpoch(self, sample):
        warped_src, target_src = sample[0]
        warped_dst, target_dst = sample[1]
        
        loss_src = self.autoencoder_src.train_on_batch(warped_src, target_src)
        loss_dst = self.autoencoder_dst.train_on_batch(warped_dst, target_dst)       
        return ( ('loss_src', loss_src), ('loss_dst', loss_dst) )


    #override
    def onGetPreview(self, sample):        
        test_A = sample[0][1][0:4] #first 4 samples
        test_B = sample[1][1][0:4]

        
        test_A_64 = np.array( [ cv2.resize ( test, (64,64) ) for test in test_A ] )
        test_B_64 = np.array( [ cv2.resize ( test, (64,64) ) for test in test_B ] )
        
        AA = self.autoencoder_src.predict(test_A_64)                                       
        AB = self.autoencoder_src.predict(test_B_64)
        BB = self.autoencoder_dst.predict(test_B_64)
 
        st = []
        for i in range(0, len(test_A)):
            st.append ( np.concatenate ( (
                test_A[i],
                AA[i],                 
                test_B[i], 
                BB[i],                
                AB[i]
                ), axis=1) )
            
        return [ ('src, dst, src->dst', np.concatenate ( st, axis=0 ) ) ]
    
    def predictor_func (self, face):
        return self.autoencoder_src.predict ( np.expand_dims(face,0) ) [0]
        
    #override
    def get_converter(self, **in_options):
        from models import ConverterMasked
        return ConverterMasked(self.predictor_func, 64, 128, 'full_face', **in_options)
        
    def Encoder(self, input_layer, created_vram_gb):
        x = input_layer
        x = conv(self.keras, x, 128)
        x = conv(self.keras, x, 256)
        x = conv(self.keras, x, 512)
        x = conv(self.keras, x, 1024)

        if created_vram_gb >=4:
            x = self.keras.layers.Dense(1024)(self.keras.layers.Flatten()(x))
            x = self.keras.layers.Dense(4 * 4 * 1024)(x)
            x = self.keras.layers.Reshape((4, 4, 1024))(x)
            x = upscale(self.keras, x, 512)
        else:
            x = self.keras.layers.Dense(512)(self.keras.layers.Flatten()(x))
            x = self.keras.layers.Dense(4 * 4 * 512)(x)
            x = self.keras.layers.Reshape((4, 4, 512))(x)
            x = upscale(self.keras, x, 256)
            
        return self.keras.models.Model(input_layer, x)

    def Decoder(self, created_vram_gb):
        if created_vram_gb >=4:  
            input_ = self.keras.layers.Input(shape=(8, 8, 512))
            x = input_
            x = upscale(self.keras, x, 512)
            x = res(self.keras, x, 512)
            x = upscale(self.keras, x, 256)
            x = res(self.keras, x, 256)
            x = upscale(self.keras, x, 128)
            x = res(self.keras, x, 128)
            x = upscale(self.keras, x, 64)
            x = res(self.keras, x, 64)
        else:
            input_ = self.keras.layers.Input(shape=(8, 8, 256))            
            x = input_
            x = upscale(self.keras, x, 256)
            x = res(self.keras, x, 256)
            x = upscale(self.keras, x, 128)
            x = res(self.keras, x, 128)
            x = upscale(self.keras, x, 64)
            x = res(self.keras, x, 64)
            x = upscale(self.keras, x, 32)
            x = res(self.keras, x, 32)
            
        x = self.keras.layers.convolutional.Conv2D(4, kernel_size=5, padding='same', activation='sigmoid')(x)
        return self.keras.models.Model(input_, x)
