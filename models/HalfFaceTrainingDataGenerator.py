from facelib import LandmarksProcessor
import cv2
import numpy as np
from models import TrainingDataGeneratorBase

'''
generates half face (warped, target) BGRA images 
where (warped) uniformly randomized in grid

kwargs options to constructor:
    warped_size=(64,64)
    target_size=(64,64)
'''

class HalfFaceTrainingDataGenerator(TrainingDataGeneratorBase):

    #overrided
    def onInitialize(self, warped_size=(64,64), target_size=(64,64),**kwargs ):
        self.warped_size = warped_size
        self.target_size = target_size

    #overrided
    def onProcessSample(self, sample, debug):
        image = sample.load_image()
        
        if debug:
            LandmarksProcessor.draw_landmarks (image, sample.landmarks, (0, 1, 0))
                    
        hull_mask = LandmarksProcessor.get_image_hull_mask (image, sample.landmarks)        
        
        s_image = np.concatenate( (image,hull_mask), axis=2)

        warped_img, target_img = HalfFaceTrainingDataGenerator.warp (s_image, sample.landmarks, self.warped_size, self.target_size)
        

        if debug:
            return (self.target_size[1], image, target_img, warped_img[...,0:3], warped_img[...,0:3]*np.expand_dims(warped_img[...,3],2) )
        else:
            return (warped_img, target_img)
            
    @staticmethod
    def warp(image, landmarks, warped_size, target_size):
        h,w,c = image.shape
        if (h != w) or (w != 64 and w != 128 and w != 256 and w != 512 and w != 1024):
            raise ValueError ('HalfFaceTrainingDataGenerator accepts only square power of 2 images.')

        rotation = np.random.uniform(-10, 10)
        scale = np.random.uniform(1 - 0.05, 1 + 0.05)
        tx = np.random.uniform(-0.05, 0.05)
        ty = np.random.uniform(-0.05, 0.05)
        
        cell_size = 32
        cell_count = w // cell_size + 1
        
        grid_points = np.linspace( 0, w, cell_count)
        mapx = np.broadcast_to(grid_points, (cell_count, cell_count)).copy()
        mapy = mapx.T
        
        mapx[1:-1,1:-1] = mapx[1:-1,1:-1] + np.random.uniform(low=-cell_size*0.2, high=cell_size*0.2, size=(cell_count-2, cell_count-2))
        mapy[1:-1,1:-1] = mapy[1:-1,1:-1] + np.random.uniform(low=-cell_size*0.2, high=cell_size*0.2, size=(cell_count-2, cell_count-2))

        half_cell_size = cell_size // 2
        
        mapx = cv2.resize(mapx, (w+cell_size,)*2 )[half_cell_size:-half_cell_size-1,half_cell_size:-half_cell_size-1].astype('float32')
        mapy = cv2.resize(mapy, (w+cell_size,)*2 )[half_cell_size:-half_cell_size-1,half_cell_size:-half_cell_size-1].astype('float32')
        warped_image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR )
        
        random_transform_mat = cv2.getRotationMatrix2D((w // 2, w // 2), rotation, scale)
        random_transform_mat[:, 2] += (tx*w, ty*w)
        
        warped_image = cv2.warpAffine( warped_image, random_transform_mat, (w,w))
        target_image = cv2.warpAffine( image, random_transform_mat, (w,w))

        warped_image = cv2.warpAffine( warped_image, LandmarksProcessor.get_transform_mat (landmarks, warped_size[0], 'half_face'), warped_size )
        target_image = cv2.warpAffine( target_image, LandmarksProcessor.get_transform_mat (landmarks, target_size[0], 'half_face'), target_size )
        return warped_image, target_image


