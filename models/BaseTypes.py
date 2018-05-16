from enum import IntEnum
import cv2
import numpy as np
from random import randint
from facelib import FaceType
from facelib import LandmarksProcessor

class TrainingDataType(IntEnum):
    
    SRC = 0 # raw unsorted
    DST = 1    
    SRC_WITH_NEAREST = 2 # as raw unsorted but samples can return get_random_nearest_target_sample()
    SRC_ONLY_10_NEAREST_TO_DST_ONLY_1 = 3 #currently unused, idea to get only 10 nearest samples to target one face for PHOTO256 model
    DST_ONLY_1 = 4  #currently unused, idea to get only 10 nearest samples to target one face for PHOTO256 model
    SRC_YAW_SORTED = 5 # sorted by yaw
    DST_YAW_SORTED = 6 # sorted by yaw
    SRC_YAW_SORTED_AS_DST = 7 #sorted by yaw but included only yaws which exist in DST_YAW_SORTED also automatic mirrored
    SRC_YAW_SORTED_AS_DST_WITH_NEAREST = 8 #same as SRC_YAW_SORTED_AS_DST but samples can return get_random_nearest_target_sample()
    
    QTY = 9
    
    
class TrainingDataSample(object):

    def __init__(self, filename=None, face_type=None, shape=None, landmarks=None, yaw=None, mirror=None, nearest_target_list=None):
        self.filename = filename
        self.face_type = face_type
        self.shape = shape
        self.landmarks = np.array(landmarks)
        self.yaw = yaw
        self.mirror = mirror
        self.nearest_target_list = nearest_target_list
        
    def copy_and_set(self, filename=None, face_type=None, shape=None, landmarks=None, yaw=None, mirror=None, nearest_target_list=None):
        return TrainingDataSample( 
            filename=filename if filename is not None else self.filename, 
            face_type=face_type if face_type is not None else self.face_type, 
            shape=shape if shape is not None else self.shape, 
            landmarks=landmarks if landmarks is not None else self.landmarks.copy(), 
            yaw=yaw if yaw is not None else self.yaw, 
            mirror=mirror if mirror is not None else self.mirror, 
            nearest_target_list=nearest_target_list if nearest_target_list is not None else self.nearest_target_list)
    
    def load_bgr(self):
        img = cv2.imread (self.filename).astype(np.float32) / 255.0
        if self.mirror:
            img = img[:,::-1].copy()
        return img
        
    def load_bgrm(self):
        img = self.load_bgr()
        img = np.concatenate( (img, LandmarksProcessor.get_image_hull_mask (img, self.landmarks)) , -1 )        
        return img
        
    def get_random_nearest_target_sample(self):
        if self.nearest_target_list is None:
            return None
        return self.nearest_target_list[randint (0, len(self.nearest_target_list)-1)]