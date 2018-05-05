from models import ModelBase
from models import TrainingDataType
import numpy as np
import cv2

from nnlib import DSSIMMaskLossClass
from nnlib import conv
from nnlib import upscale
from nnlib import res

class Model(ModelBase):

    encoderH5 = 'encoder.h5'
    decoderMaskH5 = 'decoderMask.h5'
    decoderCommonAH5 = 'decoderCommonA.h5'
    decoderCommonBH5 = 'decoderCommonB.h5'
    decoderRGBH5 = 'decoderRGB.h5'
    decoderBWH5 = 'decoderBW.h5'
    inter_BH5 = 'inter_B.h5'
    inter_AH5 = 'inter_A.h5'

    #override
    def get_model_name(self):
        return "MIAEF128"
 
    #override
    def onInitialize(self, batch_size=-1, **in_options):
        if self.gpu_total_vram_gb < 5:
            raise Exception ('Sorry, this model works only on 5GB+ GPU')
            
        self.batch_size = batch_size
        if self.batch_size == 0:          
            if self.gpu_total_vram_gb == 5:
                self.batch_size = 4
            elif self.gpu_total_vram_gb == 6:
                self.batch_size = 8
            elif self.gpu_total_vram_gb < 12: 
                self.batch_size = 16
            else: 
                self.batch_size = 32
                
        ae_input_layer = self.keras.layers.Input(shape=(64, 64, 3))
        mask_layer = self.keras.layers.Input(shape=(128, 128, 1)) #same as output

        self.encoder = self.Encoder(ae_input_layer)
        self.decoderMask = self.DecoderMask()
        self.decoderCommonA = self.DecoderCommon()
        self.decoderCommonB = self.DecoderCommon()        
        self.decoderRGB = self.DecoderRGB()
        self.decoderBW = self.DecoderBW()
        self.inter_A = self.Intermediate ()
        self.inter_B = self.Intermediate ()        
        
        if not self.is_first_run():
            self.encoder.load_weights  (self.get_strpath_storage_for_file(self.encoderH5))
            self.decoderMask.load_weights  (self.get_strpath_storage_for_file(self.decoderMaskH5))
            self.decoderCommonA.load_weights  (self.get_strpath_storage_for_file(self.decoderCommonAH5))
            self.decoderCommonB.load_weights  (self.get_strpath_storage_for_file(self.decoderCommonBH5))
            self.decoderRGB.load_weights  (self.get_strpath_storage_for_file(self.decoderRGBH5))
            self.decoderBW.load_weights  (self.get_strpath_storage_for_file(self.decoderBWH5))
            self.inter_A.load_weights (self.get_strpath_storage_for_file(self.inter_AH5))
            self.inter_B.load_weights  (self.get_strpath_storage_for_file(self.inter_BH5))            

        code = self.encoder(ae_input_layer)
        A = self.inter_A(code)
        B = self.inter_B(code)
        
        inter_A_A = self.keras.layers.Concatenate()([A, A])
        inter_B_A = self.keras.layers.Concatenate()([B, A])
 
        x1,m1 = self.decoderCommonA (inter_A_A)
        x2,m2 = self.decoderCommonA (inter_A_A)
        self.autoencoder_src     = self.keras.models.Model([ae_input_layer,mask_layer],
                    [ self.decoderBW  (self.keras.layers.Concatenate()([x1,x2]) ),
                      self.decoderMask(self.keras.layers.Concatenate()([m1,m2]) )
                    ])
                    
        x1,m1 = self.decoderCommonA (inter_A_A)
        x2,m2 = self.decoderCommonB (inter_A_A)
        self.autoencoder_src_RGB = self.keras.models.Model([ae_input_layer,mask_layer], 
                    [ self.decoderRGB  (self.keras.layers.Concatenate()([x1,x2]) ),
                      self.decoderMask (self.keras.layers.Concatenate()([m1,m2]) )
                    ])
        
        x1,m1 = self.decoderCommonA (inter_B_A)
        x2,m2 = self.decoderCommonB (inter_B_A)
        self.autoencoder_dst     = self.keras.models.Model([ae_input_layer,mask_layer], 
                    [ self.decoderRGB  (self.keras.layers.Concatenate()([x1,x2]) ),
                      self.decoderMask (self.keras.layers.Concatenate()([m1,m2]) )
                    ])
        
        if self.is_training_mode:
            self.autoencoder_src, self.autoencoder_dst = self.to_multi_gpu_model_if_possible ( [self.autoencoder_src, self.autoencoder_dst] )
                
        optimizer = self.keras.optimizers.Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)        
        
        self.autoencoder_src.compile(optimizer=optimizer, loss=[DSSIMMaskLossClass(self.tf, self.keras_contrib)(mask_layer), 'mse'] )
        self.autoencoder_dst.compile(optimizer=optimizer, loss=[DSSIMMaskLossClass(self.tf, self.keras_contrib)(mask_layer), 'mse'] )
  
        if self.is_training_mode:
            from models import FullFaceTrainingDataGenerator
            self.set_training_data_generators ([
                    FullFaceTrainingDataGenerator(self, TrainingDataType.SRC_WITH_NEAREST, batch_size=self.batch_size, warped_size=(64,64), target_size=(128,128), random_flip=True, warped_mode='ggga', target_mode='ga' ),
                    FullFaceTrainingDataGenerator(self, TrainingDataType.DST,              batch_size=self.batch_size, warped_size=(64,64), target_size=(128,128) )
                ])
            
    #override
    def onSave(self):        
        self.encoder.save_weights  (self.get_strpath_storage_for_file(self.encoderH5))
        self.decoderMask.save_weights  (self.get_strpath_storage_for_file(self.decoderMaskH5))
        self.decoderCommonA.save_weights  (self.get_strpath_storage_for_file(self.decoderCommonAH5))
        self.decoderCommonB.save_weights  (self.get_strpath_storage_for_file(self.decoderCommonBH5))
        self.decoderRGB.save_weights  (self.get_strpath_storage_for_file(self.decoderRGBH5))
        self.decoderBW.save_weights  (self.get_strpath_storage_for_file(self.decoderBWH5))
        self.inter_A.save_weights (self.get_strpath_storage_for_file(self.inter_AH5))
        self.inter_B.save_weights  (self.get_strpath_storage_for_file(self.inter_BH5))
        
        
    #override
    def onTrainOneEpoch(self, sample):
        warped_src, target_src = sample[0]
        warped_dst, target_dst = sample[1]    
        
        target_src_mask = np.expand_dims (target_src[...,-1],-1)
        target_dst_mask = np.expand_dims (target_dst[...,-1],-1)

        loss_src = self.autoencoder_src.train_on_batch( [ warped_src[...,0:3], target_src_mask], [ np.expand_dims (target_src[...,0],-1), target_src_mask] )
        loss_dst = self.autoencoder_dst.train_on_batch( [ warped_dst[...,0:3], target_dst_mask], [ target_dst[...,0:3], target_dst_mask] )
        
        return ( ('loss_src', loss_src[0]), ('loss_dst', loss_dst[0]) )        

    #override
    def onGetPreview(self, sample):
        test_A = sample[0][1][0:4] #first 4 samples
        test_B = sample[1][1][0:4]
        
        test_A_64 = np.array( [ cv2.resize ( test, (64,64) ) for test in test_A[...,0] ] )
        test_A_64 = np.repeat ( np.expand_dims(test_A_64,-1), (3,), -1 )
        test_A_m = np.expand_dims (test_A[...,-1], -1)
 
        test_B_64 = np.array( [ cv2.resize ( test, (64,64) ) for test in test_B[...,0:3] ] )
        test_B_m = np.expand_dims (test_B[...,-1], -1)
        
        AA, mAA = self.autoencoder_src.predict([test_A_64, test_A_m])                                       
        AB, mAB = self.autoencoder_src_RGB.predict([test_B_64, test_B_m])
        BB, mBB = self.autoencoder_dst.predict([test_B_64, test_B_m])
        
        mAA = np.repeat ( mAA, (3,), -1)
        mAB = np.repeat ( mAB, (3,), -1)
        mBB = np.repeat ( mBB, (3,), -1)
        
        st = []
        for i in range(0, len(test_A)):
            st.append ( np.concatenate ( (
                np.repeat (np.expand_dims (test_A[i,:,:,0],-1), (3,), -1)  ,
                np.repeat (AA[i], (3,), -1),
                #mAA[i],
                test_B[i,:,:,0:3],
                BB[i], 
                #mBB[i],                
                AB[i],
                #mAB[i]
                ), axis=1) )
            
        return [ ('src, dst, src->dst', np.concatenate ( st, axis=0 ) ) ]
    
    def predictor_func (self, face):
        face_64_bgr = cv2.resize ( face[...,0:3], (64,64) )        
        face_128_mask = np.expand_dims(face[...,-1],-1)
        
        x, mx = self.autoencoder_src_RGB.predict ( [ np.expand_dims(face_64_bgr,0), np.expand_dims(face_128_mask,0) ] )
        x, mx = x[0], mx[0]
        
        return np.concatenate ( (x,mx), -1 )
        
    #override
    def get_converter(self, **in_options):
        from models import ConverterMasked
        return ConverterMasked(self.predictor_func, 128, 128, 'full_face', erode_mask=True, blur_mask=True, clip_border_mask_per=0.046875, masked_hist_match=False, **in_options)
  

    def Encoder(self, input_layer,):
        x = input_layer
        x = conv(self.keras, x, 256)
        x = conv(self.keras, x, 512)
        x = conv(self.keras, x, 768)
        x = conv(self.keras, x, 1024)
        x = self.keras.layers.Flatten()(x)
        return self.keras.models.Model(input_layer, x)

    def Intermediate(self):
        input_layer = self.keras.layers.Input(shape=(None, 4 * 4 * 1024))
        x = input_layer
        x = self.keras.layers.Dense(1024)(x)
        x = self.keras.layers.Dense(4 * 4 * 512)(x)
        x = self.keras.layers.Reshape((4, 4, 512))(x)
            
        return self.keras.models.Model(input_layer, x)

    def DecoderCommon(self): 
        input_ = self.keras.layers.Input(shape=(4, 4, 1024))
        x = input_
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
        
        y = input_
        y = upscale(self.keras, y, 256)
        y = upscale(self.keras, y, 128)
        y = upscale(self.keras, y, 64)
        y = upscale(self.keras, y, 32)
        y = upscale(self.keras, y, 16)
        
        return self.keras.models.Model(input_, [x,y])
        
    def DecoderRGB(self): 
        input_ = self.keras.layers.Input(shape=(128, 128, 64))
        x = input_
        x = self.keras.layers.convolutional.Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)
        return self.keras.models.Model(input_, [x])

    def DecoderBW(self): 
        input_ = self.keras.layers.Input(shape=(128, 128, 64))
        x = input_
        x = self.keras.layers.convolutional.Conv2D(1, kernel_size=5, padding='same', activation='sigmoid')(x)
        return self.keras.models.Model(input_, [x])
        
    def DecoderMask(self):
        input_ = self.keras.layers.Input(shape=(128, 128, 32))        
        y = input_
        y = self.keras.layers.convolutional.Conv2D(1, kernel_size=5, padding='same', activation='sigmoid')(y)
        return self.keras.models.Model(input_, [y])
