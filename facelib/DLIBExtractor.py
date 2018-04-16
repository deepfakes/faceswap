import numpy as np
import os
import cv2

from pathlib import Path

class DLIBExtractor(object):   
    def __init__(self, dlib):
        self.scale_to = 1850 
        #3100 eats ~1.687GB VRAM on 2GB 730 desktop card, but >4Gb on 6GB card, 
        #but 3100 doesnt work on 2GB 850M notebook card, I cant understand this behaviour
        #1850 works on 2GB 850M notebook card, works faster than 3100, produces good result
        self.dlib = dlib

    def __enter__(self):   
        self.dlib_cnn_face_detector = self.dlib.cnn_face_detection_model_v1( str(Path(__file__).parent / "mmod_human_face_detector.dat") )
        self.dlib_cnn_face_detector ( np.zeros ( (self.scale_to, self.scale_to, 3), dtype=np.uint8), 0 ) 
        return self
        
    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        del self.dlib_cnn_face_detector
        return False #pass exception between __enter__ and __exit__ to outter level
        
    def extract_from_bgr (self, input_image):
        input_image = input_image[:,:,::-1].copy()
        (h, w, ch) = input_image.shape

        detected_faces = []    
        input_scale = self.scale_to / (w if w > h else h)
        input_image = cv2.resize (input_image, ( int(w*input_scale), int(h*input_scale) ), interpolation=cv2.INTER_LINEAR)
        detected_faces = self.dlib_cnn_face_detector(input_image, 0)

        result = []        
        for d_rect in detected_faces:
            if type(d_rect) == self.dlib.mmod_rectangle:
                d_rect = d_rect.rect            
            left, top, right, bottom = d_rect.left(), d_rect.top(), d_rect.right(), d_rect.bottom()
            result.append ( (int(left/input_scale), int(top/input_scale), int(right/input_scale), int(bottom/input_scale)) )

        return result
