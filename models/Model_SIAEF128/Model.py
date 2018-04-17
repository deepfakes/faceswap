from models import ModelBase
from models import TrainingDataType
import numpy as np
import cv2

from nnlib import PenalizedLossClass
from nnlib import conv
from nnlib import upscale
from nnlib import res

class Model(ModelBase):

    encoderH5 = 'encoder.h5'
    decoderH5 = 'decoder.h5'
    inter_AH5 = 'inter_A.h5'
    inter_BH5 = 'inter_B.h5'
    inter_ABH5 = 'inter_AB.h5'

    #override
    def get_model_name(self):
        return "SIAEF128"
 
    #override
    def onInitialize(self, batch_size=-1, **in_options):
   
        if self.gpu_total_vram_gb < 6:
            raise Exception ('Sorry, this model works only on 6GB+ GPU')
            
        self.batch_size = batch_size
        if self.batch_size == 0:   
            self.batch_size = 2
                
        ae_input_layer = self.keras.layers.Input(shape=(64, 64, 3))
        mask_layer = self.keras.layers.Input(shape=(128, 128, 1)) #same as output

        self.encoder = self.Encoder(ae_input_layer)
        self.decoder = self.Decoder()
        self.inter_A = self.Intermediate ()
        self.inter_B = self.Intermediate ()
        self.inter_AB = self.Intermediate ()
        
        if not self.is_first_run():
            self.encoder.load_weights  (self.get_strpath_storage_for_file(self.encoderH5))
            self.decoder.load_weights  (self.get_strpath_storage_for_file(self.decoderH5))
            self.inter_A.load_weights  (self.get_strpath_storage_for_file(self.inter_AH5))
            self.inter_B.load_weights  (self.get_strpath_storage_for_file(self.inter_BH5))
            self.inter_AB.load_weights (self.get_strpath_storage_for_file(self.inter_ABH5))

        src_inter_A = self.inter_A(self.encoder(ae_input_layer))
        src_inter_AB = self.inter_AB(self.encoder(ae_input_layer))               
        
        src_c = self.keras.layers.Concatenate()( [src_inter_A, src_inter_AB] )   
        self.autoencoder_src = self.keras.models.Model( [ae_input_layer,mask_layer], self.decoder(src_c) )
        
        dst_inter_B = self.inter_B(self.encoder(ae_input_layer))
        dst_inter_AB = self.inter_AB(self.encoder(ae_input_layer))       
        
        dst_c = self.keras.layers.Concatenate()( [dst_inter_B, dst_inter_AB] )
        self.autoencoder_dst = self.keras.models.Model( [ae_input_layer,mask_layer], self.decoder(dst_c) )

        optimizer = self.keras.optimizers.Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)        
        
        self.autoencoder_src.compile(optimizer=optimizer, loss=[PenalizedLossClass(self.tf)(mask_layer,self.keras_contrib.losses.DSSIMObjective()), 'mae'] )
        self.autoencoder_dst.compile(optimizer=optimizer, loss=[PenalizedLossClass(self.tf)(mask_layer,self.keras_contrib.losses.DSSIMObjective()), 'mae'] )
  
        if self.is_training_mode:
            if len(self.gpu_idxs) > 1:
                self.autoencoder_src = self.keras.utils.multi_gpu_model( self.autoencoder_src, self.gpu_idxs )
                self.autoencoder_dst = self.keras.utils.multi_gpu_model( self.autoencoder_dst, self.gpu_idxs )            
                #make batch_size to divide on GPU count without remainder
                self.batch_size = int( self.batch_size / len(self.gpu_idxs) )
                if self.batch_size == 0:
                    self.batch_size = 1                
                self.batch_size *= len(self.gpu_idxs)


            from models import FullFaceTrainingDataGenerator
            self.set_training_data_generators ([
                    FullFaceTrainingDataGenerator(self, TrainingDataType.SRC_YAW_SORTED_AS_DST_WITH_NEAREST, batch_size=self.batch_size, warped_size=(64,64), target_size=(128,128) ),
                    FullFaceTrainingDataGenerator(self, TrainingDataType.DST_YAW_SORTED,                     batch_size=self.batch_size, warped_size=(64,64), target_size=(128,128) )
                ])
            
    #override
    def onSave(self):        
        self.encoder.save_weights  (self.get_strpath_storage_for_file(self.encoderH5))
        self.decoder.save_weights  (self.get_strpath_storage_for_file(self.decoderH5))
        self.inter_A.save_weights  (self.get_strpath_storage_for_file(self.inter_AH5))
        self.inter_B.save_weights  (self.get_strpath_storage_for_file(self.inter_BH5))
        self.inter_AB.save_weights (self.get_strpath_storage_for_file(self.inter_ABH5))
        
    #override
    def onTrainOneEpoch(self, sample):
        warped_src, target_src = sample[0]
        warped_dst, target_dst = sample[1]    
        
        target_src_mask = np.expand_dims (target_src[...,3],-1)
        target_dst_mask = np.expand_dims (target_dst[...,3],-1)
        
        loss_src = self.autoencoder_src.train_on_batch( [warped_src[...,0:3], target_src_mask], [target_src[...,0:3], target_src_mask] )
        loss_dst = self.autoencoder_dst.train_on_batch( [warped_dst[...,0:3], target_dst_mask], [target_dst[...,0:3], target_dst_mask] )
        
        return ( ('loss_src', loss_src[0]), ('loss_dst', loss_dst[0]) )
        

    #override
    def onGetPreview(self, sample):
        test_A = sample[0][1][0:4] #first 4 samples
        test_B = sample[1][1][0:4]
        
        test_A_64 = np.array( [ cv2.resize ( test, (64,64) ) for test in test_A[...,0:3] ] )
        test_A_m = np.expand_dims (test_A[...,3], -1)
        test_B_64 = np.array( [ cv2.resize ( test, (64,64) ) for test in test_B[...,0:3] ] )
        test_B_m = np.expand_dims (test_B[...,3], -1)
        
        AA, mAA = self.autoencoder_src.predict([test_A_64, test_A_m])                                       
        AB, mAB = self.autoencoder_src.predict([test_B_64, test_B_m])
        BB, mBB = self.autoencoder_dst.predict([test_B_64, test_B_m])
        
        mAA = np.repeat ( mAA, (3,), -1)
        mAB = np.repeat ( mAB, (3,), -1)
        mBB = np.repeat ( mBB, (3,), -1)
        
        st = []
        for i in range(0, len(test_A)):
            st.append ( np.concatenate ( (
                test_A[i,:,:,0:3],
                AA[i],
                mAA[i],
                test_B[i,:,:,0:3], 
                BB[i], 
                mBB[i],                
                AB[i],
                mAB[i]
                ), axis=1) )
            
        return [ ('src, dst, src->dst', np.concatenate ( st, axis=0 ) ) ]
    
    def predictor_func (self, face):
        
        face_64_bgr = cv2.resize ( face[...,0:3], (64,64) )
        face_128_mask = np.expand_dims(face[...,3],-1)
        
        x, mx = self.autoencoder_src.predict ( [ np.expand_dims(face_64_bgr,0), np.expand_dims(face_128_mask,0) ] )
        x, mx = x[0], mx[0]
        
        return np.concatenate ( (x,mx), -1 )
        
    #override
    def get_converter(self, **in_options):
        from models import ConverterMasked
        return ConverterMasked(self.predictor_func, 128, 128, 24, **in_options)
        
    def Encoder(self, input_layer):
        x64 = input_layer
        x32 = conv(self.keras, x64, 128)
        x16 = conv(self.keras, x32, 256)
        x8 = conv(self.keras, x16, 512)
        x4 = conv(self.keras, x8, 1024)
        
        return self.keras.models.Model(input_layer, 
                [  
                    self.keras.layers.Flatten()(x32),
                    self.keras.layers.Flatten()(x16),
                    self.keras.layers.Flatten()(x8),
                    self.keras.layers.Flatten()(x4)
                ]
            )

    def Intermediate(self):
        input_x32 = self.keras.layers.Input(shape=(None, 32 * 32 * 128))
        input_x16 = self.keras.layers.Input(shape=(None, 16 * 16 * 256))
        input_x8 = self.keras.layers.Input(shape=(None, 8 * 8 * 512))
        input_x4 = self.keras.layers.Input(shape=(None, 4 * 4 * 1024))
        
        x32 = self.keras.layers.Dense(64)(input_x32)
        x16 = self.keras.layers.Dense(128)(input_x16)
        x8 = self.keras.layers.Dense(256)(input_x8)
        x4 = self.keras.layers.Dense(512)(input_x4)
        
        x = self.keras.layers.Concatenate()( [x32, x16, x8, x4] )
        
        x = self.keras.layers.Dense(4 * 4 * 512)(x)
        x = self.keras.layers.Reshape((4, 4, 512))(x)
            
        return self.keras.models.Model( [input_x32, input_x16,  input_x8, input_x4], [x]  )
        
        
    def Decoder(self):
            
        input  = self.keras.layers.Input(shape=(4, 4, 1024))
        x = input
        x = upscale(self.keras, x, 512)
        x = res(self.keras, x, 512)        
        
        x = upscale(self.keras, x, 256)
        x = res(self.keras, x, 256)        
        
        x = upscale(self.keras, x, 128)
        x = res(self.keras, x, 128)        
        
        x = upscale(self.keras, x, 64)
        x = res(self.keras, x, 64)        
        
        x = upscale(self.keras, x, 32)
        x = res(self.keras, x, 32)

        x = self.keras.layers.convolutional.Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)
        
        y = input  #mask decoder
        y = upscale(self.keras, y, 512)
        y = upscale(self.keras, y, 256)
        y = upscale(self.keras, y, 128)
        y = upscale(self.keras, y, 64)
        y = upscale(self.keras, y, 32)
        y = self.keras.layers.convolutional.Conv2D( 1, kernel_size=5, padding='same', activation='sigmoid' )(y)

        return self.keras.models.Model( [input], [x,y])
  
