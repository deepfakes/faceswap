from models import ModelBase
from models import TrainingDataType
import numpy as np

from nnlib import conv
from nnlib import upscale
from nnlib import res

class Model(ModelBase):

    encoderH5 = 'encoder.h5'
    decoder_srcH5 = 'decoder_src.h5'
    decoder_dstH5 = 'decoder_dst.h5'

    #override
    def get_model_name(self):
        return "H64"
 
    #override
    def onInitialize(self, batch_size=-1, **in_options):
        if self.gpu_total_vram_gb < 3:
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

        optimizer = self.keras.optimizers.Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)
        self.autoencoder_src.compile(optimizer=optimizer, loss='mae')
        self.autoencoder_dst.compile(optimizer=optimizer, loss='mae')
  
        if self.is_training_mode:
            if len(self.gpu_idxs) > 1:
                self.autoencoder_src = self.keras.utils.multi_gpu_model( self.autoencoder_src, self.gpu_idxs )
                self.autoencoder_dst = self.keras.utils.multi_gpu_model( self.autoencoder_dst, self.gpu_idxs )            
                #make batch_size to divide on GPU count without remainder
                self.batch_size = int( self.batch_size / len(self.gpu_idxs) )
                if self.batch_size == 0:
                    self.batch_size = 1                
                self.batch_size *= len(self.gpu_idxs)

            from models import HalfFaceTrainingDataGenerator
            self.set_training_data_generators ([
                    HalfFaceTrainingDataGenerator(self, TrainingDataType.SRC_YAW_SORTED_AS_DST, batch_size=self.batch_size, warped_size=(64,64), target_size=(64,64) ),
                    HalfFaceTrainingDataGenerator(self, TrainingDataType.DST_YAW_SORTED,        batch_size=self.batch_size, warped_size=(64,64), target_size=(64,64) )
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
        
        AA = self.autoencoder_src.predict(test_A)
                                       
        AB = self.autoencoder_src.predict(test_B)
        BB = self.autoencoder_dst.predict(test_B)

        mAA = AA[:,:,:,3]
        mAA = np.repeat ( np.expand_dims (mAA, axis=3), (4,), axis=3)
        
        mAB = AB[:,:,:,3]
        mAB = np.repeat ( np.expand_dims (mAB, axis=3), (4,), axis=3)
        
        mBB = BB[:,:,:,3]
        mBB = np.repeat ( np.expand_dims (mBB, axis=3), (4,), axis=3)
        
        st = []
        for i in range(0, len(test_A)):
            st.append ( np.concatenate ( [
                test_A[i], 
                AA[i], 
                #AA[i]*mAA[i], 
                test_B[i], 
                BB[i], 
                #BB[i]*mBB[i], 
                AB[i], 
                #AB[i]*mAB[i]], 
                axis=1) )
            
        return [ ('src, dst, src->dst masked', np.concatenate ( st, axis=0 ) ) ]
    
    def predictor_func(self, face):        
        return self.autoencoder_src.predict ( np.expand_dims(face,0) ) [0]
    
    #override
    def get_converter(self, **in_options):
        from models import ConverterMasked
        return ConverterMasked(self.predictor_func, 64, 64, False, **in_options)
        
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
        x = self.keras.layers.convolutional.Conv2D(4, kernel_size=5, padding='same', activation='sigmoid')(x)
        return self.keras.models.Model(input_, x)
