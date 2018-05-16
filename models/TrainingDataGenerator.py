from facelib import FaceType
from facelib import LandmarksProcessor
import cv2
import numpy as np
from models import TrainingDataGeneratorBase
from utils import image_utils
from utils import random_utils
from enum import IntEnum
'''
generates full face images 
    random_flip 
    output_sample_types is FullFaceTrainingDataGenerator.SampleType
'''   
class TrainingDataGenerator(TrainingDataGeneratorBase):
    class SampleTypeFlags(IntEnum):
        WARPED               = 0x00001,
        TARGET               = 0x00002,
        TARGET_UNTRANSFORMED = 0x00004,
        
        HALF_FACE   = 0x00010,
        FULL_FACE   = 0x00020,
        HEAD_FACE   = 0x00040,
        AVATAR_FACE = 0x00080,
        
        MODE_BGR  = 0x00100,  #BGR
        MODE_G    = 0x00200,  #Grayscale
        MODE_GGG  = 0x00400,  #3xGrayscale 
        MODE_M    = 0x00800,  #mask only    
        MODE_BGRM = 0x01000,  #BGR + mask
        MODE_GM   = 0x02000,  #Grayscale + mask
        MODE_GGGM = 0x04000,  #3xGrayscale + mask            
        
        SIZE_64   = 0x10000,
        SIZE_128  = 0x20000,
        SIZE_256  = 0x40000,
        
    #overrided
    def onInitialize(self, random_flip=False, output_sample_types_flags=[], **kwargs):
        self.random_flip = random_flip        
        self.output_sample_types_flags = output_sample_types_flags
        
    #overrided
    def onProcessSample(self, sample, debug):
        s_bgrm = sample.load_bgrm()

        if debug:
            LandmarksProcessor.draw_landmarks (s_bgrm, sample.landmarks, (0, 1, 0))

        warped_bgrm, target_bgrm, target_bgrm_untransformed = TrainingDataGenerator.warp (s_bgrm, sample.landmarks)

        if self.random_flip and np.random.randint(2) == 1:
            warped_bgrm = warped_bgrm[:,::-1,:]
            target_bgrm = target_bgrm[:,::-1,:]
            target_bgrm_untransformed = target_bgrm[:,::-1,:]

        outputs = []
        
        for t in self.output_sample_types_flags:
            if t & self.SampleTypeFlags.WARPED != 0:
                source = warped_bgrm
            elif t & self.SampleTypeFlags.TARGET != 0:
                source = target_bgrm
            elif t & self.SampleTypeFlags.TARGET_UNTRANSFORMED != 0:
                source = target_bgrm_untransformed
            else:
                raise ValueError ('expected SampleTypeFlags warped/target type')

            if t & self.SampleTypeFlags.SIZE_64 != 0:
                size = 64
            elif t & self.SampleTypeFlags.SIZE_128 != 0:
                size = 128
            elif t & self.SampleTypeFlags.SIZE_256 != 0:
                size = 256
            else:
                raise ValueError ('expected SampleTypeFlags size')
                
            if t & self.SampleTypeFlags.HALF_FACE != 0:
                target_face_type = FaceType.HALF            
            elif t & self.SampleTypeFlags.FULL_FACE != 0:
                target_face_type = FaceType.FULL
            elif t & self.SampleTypeFlags.HEAD_FACE != 0:
                target_face_type = FaceType.HEAD
            elif t & self.SampleTypeFlags.AVATAR_FACE != 0:
                target_face_type = FaceType.AVATAR
            else:
                raise ValueError ('expected SampleTypeFlags face type')
                
            if target_face_type <= sample.face_type:
                source = cv2.warpAffine( source, LandmarksProcessor.get_transform_mat (sample.landmarks, size, target_face_type), (size,size), flags=cv2.INTER_LANCZOS4 )
            else:
                raise Exception ('sample %s type %s does not match model requirement %s. Consider extract necessary type of faces.' % (sample.filename, sample.face_type, target_face_type) )
                
            if t & self.SampleTypeFlags.MODE_BGR != 0:
                source = source[...,0:3]
            elif t & self.SampleTypeFlags.MODE_G != 0:
                source = np.expand_dims(cv2.cvtColor(source[...,0:3], cv2.COLOR_BGR2GRAY),-1)
            elif t & self.SampleTypeFlags.MODE_GGG != 0:
                source = np.repeat ( np.expand_dims(cv2.cvtColor(source[...,0:3], cv2.COLOR_BGR2GRAY),-1), (3,), -1)
            elif t & self.SampleTypeFlags.MODE_BGRM != 0:
                source = source
            elif t & self.SampleTypeFlags.MODE_GM != 0:
                source = np.concatenate( ( np.expand_dims(cv2.cvtColor(source[...,0:3], cv2.COLOR_BGR2GRAY),-1),
                                           np.expand_dims(source[...,-1],-1)
                                         )
                                         , -1
                                       )
            elif t & self.SampleTypeFlags.MODE_GGGM != 0:
                source = np.concatenate( ( np.repeat ( np.expand_dims(cv2.cvtColor(source[...,0:3], cv2.COLOR_BGR2GRAY),-1), (3,), -1),
                                           np.expand_dims(source[...,-1],-1)
                                         )
                                         , -1
                                       )
            elif t & self.SampleTypeFlags.MODE_M != 0:
                source = np.expand_dims(source[...,-1],-1)
            else:
                raise ValueError ('expected SampleTypeFlags mode')
                
            outputs.append ( source )

        if debug:
            result = (s_bgrm,)

            for output in outputs:
                if output.shape[2] < 4:
                    result += (output,)
                elif output.shape[2] == 4:
                    result += ( output[...,0:-1]*np.expand_dims(output[...,-1],-1), )

            return result            
        else:
            return outputs
            
    @staticmethod
    def warp(image_bgrm, s_landmarks):          
        h,w,c = image_bgrm.shape
        if (h != w) or (w != 64 and w != 128 and w != 256 and w != 512 and w != 1024):
            raise ValueError ('TrainingDataGenerator accepts only square power of 2 images.')
            
        rotation = np.random.uniform(-10, 10)
        scale = np.random.uniform(1 - 0.05, 1 + 0.05)
        tx = np.random.uniform(-0.05, 0.05)
        ty = np.random.uniform(-0.05, 0.05)    
     
        #random warp by grid
        cell_size = 32
        cell_count = w // cell_size + 1
        
        grid_points = np.linspace( 0, w, cell_count)
        mapx = np.broadcast_to(grid_points, (cell_count, cell_count)).copy()
        mapy = mapx.T
        
        mapx[1:-1,1:-1] = mapx[1:-1,1:-1] + random_utils.random_normal( size=(cell_count-2, cell_count-2) )*(cell_size*0.24)
        mapy[1:-1,1:-1] = mapy[1:-1,1:-1] + random_utils.random_normal( size=(cell_count-2, cell_count-2) )*(cell_size*0.24)

        half_cell_size = cell_size // 2
        
        mapx = cv2.resize(mapx, (w+cell_size,)*2 )[half_cell_size:-half_cell_size-1,half_cell_size:-half_cell_size-1].astype(np.float32)
        mapy = cv2.resize(mapy, (w+cell_size,)*2 )[half_cell_size:-half_cell_size-1,half_cell_size:-half_cell_size-1].astype(np.float32)
        warped = cv2.remap(image_bgrm, mapx, mapy, cv2.INTER_LANCZOS4 )
        
        #random transform                                   
        random_transform_mat = cv2.getRotationMatrix2D((w // 2, w // 2), rotation, scale)
        random_transform_mat[:, 2] += (tx*w, ty*w)

        target_bgrm = cv2.warpAffine( image_bgrm, random_transform_mat, (w, w), borderMode=cv2.BORDER_CONSTANT, flags=cv2.INTER_LANCZOS4 )        
        warped_bgrm = cv2.warpAffine( warped,  random_transform_mat, (w, w), borderMode=cv2.BORDER_CONSTANT, flags=cv2.INTER_LANCZOS4 )
        
        return warped_bgrm, target_bgrm, image_bgrm
