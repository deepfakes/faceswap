import numpy as np
import os
import cv2

from pathlib import Path

from .mtcnn import *

class MTCExtractor(object):   
    def __init__(self, keras, tf, tf_session):
        self.scale_to = 1920
        self.keras = keras
        self.tf = tf
        self.tf_session = tf_session
        
        self.min_face_size = self.scale_to * 0.042
        self.thresh1 = 0.7
        self.thresh2 = 0.85
        self.thresh3 = 0.6
        self.scale_factor = 0.95
        
        '''
        self.min_face_size = self.scale_to * 0.042
        self.thresh1 = 7
        self.thresh2 = 85
        self.thresh3 = 6
        self.scale_factor = 0.95
        '''

    def __enter__(self):
        with self.tf.variable_scope('pnet2'):
            data = self.tf.placeholder(self.tf.float32, (None,None,None,3), 'input')
            pnet2 = PNet(self.tf, {'data':data})
            pnet2.load(str(Path(__file__).parent/'det1.npy'), self.tf_session)
        with self.tf.variable_scope('rnet2'):
            data = self.tf.placeholder(self.tf.float32, (None,24,24,3), 'input')
            rnet2 = RNet(self.tf, {'data':data})
            rnet2.load(str(Path(__file__).parent/'det2.npy'), self.tf_session)
        with self.tf.variable_scope('onet2'):
            data = self.tf.placeholder(self.tf.float32, (None,48,48,3), 'input')
            onet2 = ONet(self.tf, {'data':data})
            onet2.load(str(Path(__file__).parent/'det3.npy'), self.tf_session)

        self.pnet_fun = self.keras.backend.function([pnet2.layers['data']],[pnet2.layers['conv4-2'], pnet2.layers['prob1']])
        self.rnet_fun = self.keras.backend.function([rnet2.layers['data']],[rnet2.layers['conv5-2'], rnet2.layers['prob1']])
        self.onet_fun = self.keras.backend.function([onet2.layers['data']],[onet2.layers['conv6-2'], onet2.layers['conv6-3'], onet2.layers['prob1']])

        faces, pnts = detect_face ( np.zeros ( (self.scale_to, self.scale_to, 3)), self.min_face_size, self.pnet_fun, self.rnet_fun, self.onet_fun, [ self.thresh1, self.thresh2, self.thresh3 ], self.scale_factor ) 
        return self
        
    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        return False #pass exception between __enter__ and __exit__ to outter level
        
    def extract_from_bgr (self, input_image):
        input_image = input_image[:,:,::-1].copy()
        (h, w, ch) = input_image.shape


        input_scale = self.scale_to / (w if w > h else h)
        input_image = cv2.resize (input_image, ( int(w*input_scale), int(h*input_scale) ), interpolation=cv2.INTER_LINEAR)

        detected_faces, pnts = detect_face ( input_image, self.min_face_size, self.pnet_fun, self.rnet_fun, self.onet_fun, [ self.thresh1, self.thresh2, self.thresh3 ], self.scale_factor )
        detected_faces = [ ( int(face[0]/input_scale), int(face[1]/input_scale), int(face[2]/input_scale), int(face[3]/input_scale)) for face in detected_faces ]
          
        return detected_faces

