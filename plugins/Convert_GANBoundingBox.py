# Based on the https://github.com/shaoanlu/faceswap-GAN repo (master/FaceSwap_GAN_v2_train.ipynb)

import cv2
import numpy as np
import face_recognition

from lib.Converter import Converter

use_smoothed_mask = True
use_smoothed_bbox = True

class Convert(Converter):
    def __init__(self, encoder, **kwargs):
        self.encoder = encoder

    def patch_all( self, original, faces_detected ):
        return process_video(original, list(faces_detected), self.encoder)

global prev_x0, prev_x1, prev_y0, prev_y1
global frames
prev_x0 = prev_x1 = prev_y0 = prev_y1 = 0
frames = 0

def get_smoothed_coord(x0, x1, y0, y1):
    global prev_x0, prev_x1, prev_y0, prev_y1
    x0 = int(0.65*prev_x0 + 0.35*x0)
    x1 = int(0.65*prev_x1 + 0.35*x1)
    y1 = int(0.65*prev_y1 + 0.35*y1)
    y0 = int(0.65*prev_y0 + 0.35*y0)
    return x0, x1, y0, y1    
    
def set_global_coord(x0, x1, y0, y1):
    global prev_x0, prev_x1, prev_y0, prev_y1
    prev_x0 = x0
    prev_x1 = x1
    prev_y1 = y1
    prev_y0 = y0

def process_video(image, faces, encoder):   
    if len(faces) == 0:
        comb_img = np.zeros([image.shape[0], image.shape[1]*2,image.shape[2]])
        comb_img[:, :image.shape[1], :] = image
        comb_img[:, image.shape[1]:, :] = image
        triple_img = np.zeros([image.shape[0], image.shape[1]*3,image.shape[2]])
        triple_img[:, :image.shape[1], :] = image
        triple_img[:, image.shape[1]:image.shape[1]*2, :] = image      
        triple_img[:, image.shape[1]*2:, :] = (image * .15).astype('uint8')
    
    mask_map = np.zeros_like(image)

    global prev_x0, prev_x1, prev_y0, prev_y1
    global frames

    for (idx, face) in faces:
        (x0, y1, x1, y0) = face.locations
        h = x1 - x0
        w = y1 - y0
        
        # smoothing bounding box
        if use_smoothed_bbox:
            if frames != 0:
                x0, x1, y0, y1 = get_smoothed_coord(x0, x1, y0, y1)
                set_global_coord(x0, x1, y0, y1)
            else:
                set_global_coord(x0, x1, y0, y1)
                frames += 1
            
        cv2_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        roi_image = cv2_img[x0+h//15:x1-h//15,y0+w//15:y1-w//15,:]
        roi_size = roi_image.shape  
        
        # smoothing mask
        if use_smoothed_mask:
            mask = np.zeros_like(roi_image)
            mask[h//15:-h//15,w//15:-w//15,:] = 255
            mask = cv2.GaussianBlur(mask,(15,15),10)
            orig_img = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)
        
        (result, result_a) = encoder(roi_image)
        
        mask_map[x0+h//15:x1-h//15, y0+w//15:y1-w//15,:] = np.expand_dims(cv2.resize(result_a, (roi_size[1],roi_size[0])), axis=2)
        mask_map = np.clip(mask_map + .15 * image, 0, 255 )
        
        result = cv2.resize(result, (roi_size[1],roi_size[0]))
        comb_img = np.zeros([image.shape[0], image.shape[1]*2,image.shape[2]])
        comb_img[:, :image.shape[1], :] = image
        comb_img[:, image.shape[1]:, :] = image
        
        if use_smoothed_mask:
            comb_img[x0+h//15:x1-h//15, image.shape[1]+y0+w//15:image.shape[1]+y1-w//15,:] = mask/255*result + (1-mask/255)*orig_img
        else:
            comb_img[x0+h//15:x1-h//15, image.shape[1]+y0+w//15:image.shape[1]+y1-w//15,:] = result
            
        triple_img = np.zeros([image.shape[0], image.shape[1]*3,image.shape[2]])
        triple_img[:, :image.shape[1]*2, :] = comb_img
        triple_img[:, image.shape[1]*2:, :] = mask_map
    
    # ========== Change rthe following line to ==========
    # return comb_img[:, image.shape[1]:, :]  # return only result image
    # return comb_img  # return input and result image combined as one
    return triple_img #return input,result and mask heatmap image combined as one
