from facelib import LandmarksProcessor
import cv2
import numpy as np
from models import TrainingDataGeneratorBase
from utils import image_utils

'''
generates full face (warped, target) BGRA images 
where (warped) matches dst face if sample contains get_random_nearest_target_sample()

kwargs options to constructor:
    warped_size=(64,64)
    target_size=(128,128)
    random_flip 
    warped/target modes = 'bgra', 'ga', 'ggga'

thx to dfaker for original idea
'''

class FullFaceTrainingDataGenerator(TrainingDataGeneratorBase):

    #overrided
    def onInitialize(self, warped_size=(64,64), target_size=(128,128), random_flip=False, warped_mode='bgra', target_mode='bgra', **kwargs):
        self.warped_size = warped_size
        self.target_size = target_size
        self.random_flip = random_flip
        self.warped_mode = warped_mode
        self.target_mode = target_mode
        
    #overrided
    def onProcessSample(self, sample, debug):
        s_image = sample.load_image()
        
        if debug:
            LandmarksProcessor.draw_landmarks (s_image, sample.landmarks, (0, 1, 0))
            
        target_sample = sample.get_random_nearest_target_sample()
        if target_sample is None:
            d_landmarks = None
            d_image = None
        else:
            d_landmarks = target_sample.landmarks
            if debug:                
                d_image = target_sample.load_image()
                LandmarksProcessor.draw_landmarks (d_image, d_landmarks, (0, 1, 0))

        warped, target, debug_image = FullFaceTrainingDataGenerator.warp (s_image, sample.landmarks, d_landmarks, self.warped_size, self.target_size, debug)

        warped = self.bgra_to_mode (warped, self.warped_mode)
        target = self.bgra_to_mode (target, self.target_mode)    

        if self.random_flip and np.random.randint(2) == 1:
            warped = warped[:,::-1,:]
            target = target[:,::-1,:]
            if debug:
                debug_image = debug_image[:,::-1,:]
            
        if debug:
            result = (self.target_size[0], s_image)            
            if d_image is not None:
                result += (d_image,)
                
            result += (debug_image, target[...,0:-1], warped[...,0:-1], warped[...,0:-1]*np.expand_dims(warped[...,-1],-1))
            return result            
        else:
            return ( warped, target )
            
    @staticmethod
    def warp(s_image, s_landmarks, d_landmarks, warped_size, target_size, debug):
        if d_landmarks is None:
            d_landmarks = s_landmarks
            
        s_landmarks_original = s_landmarks
        d_landmarks_original = d_landmarks
            
        h,w,c = s_image.shape
        if (h != w) or (w != 64 and w != 128 and w != 256 and w != 512 and w != 1024):
            raise ValueError ('FullFaceTrainingDataGenerator accepts only square power of 2 images.')
            
        rotation = np.random.uniform(-10, 10)
        scale = np.random.uniform(1 - 0.05, 1 + 0.05)
        tx = np.random.uniform(-0.05, 0.05)
        ty = np.random.uniform(-0.05, 0.05)    
        
        idx_list = np.array( range(0,61) ) #all wo tooth
         
        #remove 'jaw landmarks' which in region of 'not jaw landmarks' and outside
        face_contour = cv2.convexHull( s_landmarks[idx_list[np.argwhere ( idx_list >= 17 )[:,0] ]] )        
        for idx in idx_list[ np.argwhere ( idx_list < 17 )[:,0] ][:]:
            s_l = s_landmarks[idx]
            d_l = d_landmarks[idx]
   
            if not (s_l[0] >= 1 and s_l[0] <= w-2 and s_l[1] >= 1 and s_l[1] <= w-2) or \
               not (d_l[0] >= 1 and d_l[0] <= w-2 and d_l[1] >= 1 and d_l[1] <= w-2) or \
               cv2.pointPolygonTest(face_contour, tuple(s_l[::-1]),False) >= 0 or \
               cv2.pointPolygonTest(face_contour, tuple(d_l[::-1]),False) >= 0:
                idx_list = np.delete (idx_list, np.argwhere (idx_list == idx)[:,0] )
                
        #choose landmarks which not close to each other in random dist
        dist = np.random.ranf()*8.0 + 8.0
        chk_dist = dist*dist   
        
        ar = idx_list
        idx_list = []
        while len(ar) > 0:                                
            idx = ar[0] 
            p = d_landmarks[idx]
            
            idx_list.append (idx)
            idxs_to_remove = []
            for i in range(0, len(ar)):
                p_p2 = d_landmarks[ar[i]] - p                                        
                if np.square(p_p2[0]) + np.square(p_p2[1]) < chk_dist:
                    idxs_to_remove.append (i)
                    
            ar = np.delete ( ar, idxs_to_remove)

        unmodified_d_landmarks = d_landmarks.copy()
 
        #randomizing d_landmarks
        '''
        d_landmarks = d_landmarks.copy()
        for idx in idx_list[:]:
            d_l = d_landmarks[idx]
            
            theta = np.random.ranf()*2*np.pi
            vector_dist = (dist/4.0) + np.random.ranf()*(dist/4.0)
            
            dxy = np.array ( [ np.sin(theta)*vector_dist, np.cos(theta)*vector_dist] )   
            dl = (d_l+dxy).astype(np.int)
            if (dl[0] >= 1 and dl[0] <= w-2 and dl[1] >= 1 and dl[1] <= w-2):
                d_landmarks[idx] = dl


        #remove outside and boundary points
        for idx in idx_list[:]:
            s_l = s_landmarks[idx]
            d_l = d_landmarks[idx]

            if not (s_l[0] >= 1 and s_l[0] <= w-2 and s_l[1] >= 1 and s_l[1] <= w-2) or \
               not (d_l[0] >= 1 and d_l[0] <= w-2 and d_l[1] >= 1 and d_l[1] <= w-2):
                idx_list = np.delete (idx_list, np.argwhere (idx_list == idx)[:,0] )      
        '''
        s_landmarks = s_landmarks[idx_list]
        d_landmarks = d_landmarks[idx_list]
        unmodified_d_landmarks = unmodified_d_landmarks[idx_list]
        
        #4 anchors for warper
        edgeAnchors = np.array ( [ (0,0), (0,h-1), (w-1,h-1), (w-1,0)] )
        
        s_landmarks_anchored = np.concatenate ((edgeAnchors, s_landmarks)) 
        d_landmarks_anchored = np.concatenate ((edgeAnchors, d_landmarks)) 
        unmodified_d_landmarks_anchored = np.concatenate ((edgeAnchors, unmodified_d_landmarks)) 

        if debug:
            debug_image = image_utils.morph_by_points (s_image, s_landmarks_anchored, unmodified_d_landmarks_anchored )
        else:
            debug_image = None
        
        #warped = image_utils.morph_by_points (s_image, s_landmarks_anchored, d_landmarks_anchored)
        warped = image_utils.morph_by_points (s_image, s_landmarks_anchored, unmodified_d_landmarks_anchored)

        #embedding mask as 4 channel
        s_image = np.concatenate( (s_image,
                                    cv2.fillConvexPoly( np.zeros(s_image.shape[0:2]+(1,),dtype=np.float32), cv2.convexHull (s_landmarks_original), (1,) ))
                                   , -1 )
        
        warped = np.concatenate( (warped,
                                    cv2.fillConvexPoly( np.zeros(warped.shape[0:2]+(1,),dtype=np.float32), cv2.convexHull (d_landmarks_original), (1,) ))
                                   , -1 )
                                   
        #
        cell_size = 32
        cell_count = w // cell_size + 1
        
        grid_points = np.linspace( 0, w, cell_count)
        mapx = np.broadcast_to(grid_points, (cell_count, cell_count)).copy()
        mapy = mapx.T
        
        mapx[1:-1,1:-1] = mapx[1:-1,1:-1] + np.random.uniform(low=-cell_size*0.2, high=cell_size*0.2, size=(cell_count-2, cell_count-2))
        mapy[1:-1,1:-1] = mapy[1:-1,1:-1] + np.random.uniform(low=-cell_size*0.2, high=cell_size*0.2, size=(cell_count-2, cell_count-2))

        half_cell_size = cell_size // 2
        
        mapx = cv2.resize(mapx, (w+cell_size,)*2 )[half_cell_size:-half_cell_size-1,half_cell_size:-half_cell_size-1].astype(np.float32)
        mapy = cv2.resize(mapy, (w+cell_size,)*2 )[half_cell_size:-half_cell_size-1,half_cell_size:-half_cell_size-1].astype(np.float32)
        #target_image = cv2.remap(s_image, mapx, mapy, cv2.INTER_LINEAR )
        warped = cv2.remap(warped, mapx, mapy, cv2.INTER_LINEAR )
        #
                                   
        random_transform_mat = cv2.getRotationMatrix2D((w // 2, w // 2), rotation, scale)
        random_transform_mat[:, 2] += (tx*w, ty*w)

        target_image = cv2.warpAffine( s_image, random_transform_mat, (w, w), borderMode=cv2.BORDER_REPLICATE )        
        warped       = cv2.warpAffine( warped,  random_transform_mat, (w, w), borderMode=cv2.BORDER_REPLICATE )
            
        target_image  = cv2.resize( target_image, target_size, cv2.INTER_LINEAR)        
        warped        = cv2.resize( warped      , warped_size, cv2.INTER_LINEAR)
        
        return warped, target_image, debug_image
        
    def bgra_to_mode (self, img, mode):
        if mode == 'ga':
            img = np.concatenate( ( np.expand_dims(cv2.cvtColor(img[...,0:3], cv2.COLOR_BGR2GRAY),-1),
                                    np.expand_dims(img[...,-1],-1)
                                   )
                                   , -1
                                 )
        elif mode == 'ggga':
            img = np.concatenate( ( np.repeat ( np.expand_dims(cv2.cvtColor(img[...,0:3], cv2.COLOR_BGR2GRAY),-1), (3,), -1),
                                    np.expand_dims(img[...,-1],-1)
                                   )
                                   , -1
                                 )
        return img