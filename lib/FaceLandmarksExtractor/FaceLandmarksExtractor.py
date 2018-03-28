import atexit
import numpy as np
import os
import cv2
import dlib
import keras
from keras import backend as K

dlib_detectors = []
keras_model = None
is_initialized = False

@atexit.register
def onExit():
    global dlib_detectors
    global keras_model
    
    if keras_model is not None:
        del keras_model
        K.clear_session()
        
    for detector in dlib_detectors:
        del detector
        
class TorchBatchNorm2D(keras.engine.topology.Layer):
    def __init__(self, axis=-1, momentum=0.99, epsilon=1e-3, **kwargs):
        super(TorchBatchNorm2D, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon

    def build(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of ' 'input tensor should have a defined dimension ' 'but the layer received an input with shape ' + str(input_shape) + '.')
        shape = (dim,)
        self.gamma = self.add_weight(shape=shape, name='gamma', initializer='ones', regularizer=None, constraint=None)
        self.beta = self.add_weight(shape=shape, name='beta', initializer='zeros', regularizer=None, constraint=None)
        self.moving_mean = self.add_weight(shape=shape, name='moving_mean', initializer='zeros', trainable=False)            
        self.moving_variance = self.add_weight(shape=shape, name='moving_variance', initializer='ones', trainable=False)            
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)

        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]
        
        broadcast_moving_mean = K.reshape(self.moving_mean, broadcast_shape)
        broadcast_moving_variance = K.reshape(self.moving_variance, broadcast_shape)
        broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
        broadcast_beta = K.reshape(self.beta, broadcast_shape)        
        invstd = K.ones (shape=broadcast_shape, dtype='float32') / K.sqrt(broadcast_moving_variance + K.constant(self.epsilon, dtype='float32'))
        
        return (inputs - broadcast_moving_mean) * invstd * broadcast_gamma + broadcast_beta
       
    def get_config(self):
        config = { 'axis': self.axis, 'momentum': self.momentum, 'epsilon': self.epsilon }
        base_config = super(TorchBatchNorm2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def transform(point, center, scale, resolution):
    pt = np.array ( [point[0], point[1], 1.0] )            
    h = 200.0 * scale
    m = np.eye(3)
    m[0,0] = resolution / h
    m[1,1] = resolution / h
    m[0,2] = resolution * ( -center[0] / h + 0.5 )
    m[1,2] = resolution * ( -center[1] / h + 0.5 )
    m = np.linalg.inv(m)
    return np.matmul (m, pt)[0:2]
    
def crop(image, center, scale, resolution=256.0):
    ul = transform([1, 1], center, scale, resolution).astype( np.int )
    br = transform([resolution, resolution], center, scale, resolution).astype( np.int )
    if image.ndim > 2:
        newDim = np.array([br[1] - ul[1], br[0] - ul[0], image.shape[2]], dtype=np.int32)
        newImg = np.zeros(newDim, dtype=np.uint8)
    else:
        newDim = np.array([br[1] - ul[1], br[0] - ul[0]], dtype=np.int)
        newImg = np.zeros(newDim, dtype=np.uint8)
    ht = image.shape[0]
    wd = image.shape[1]
    newX = np.array([max(1, -ul[0] + 1), min(br[0], wd) - ul[0]], dtype=np.int32)
    newY = np.array([max(1, -ul[1] + 1), min(br[1], ht) - ul[1]], dtype=np.int32)
    oldX = np.array([max(1, ul[0] + 1), min(br[0], wd)], dtype=np.int32)
    oldY = np.array([max(1, ul[1] + 1), min(br[1], ht)], dtype=np.int32)
    newImg[newY[0] - 1:newY[1], newX[0] - 1:newX[1] ] = image[oldY[0] - 1:oldY[1], oldX[0] - 1:oldX[1], :]
    newImg = cv2.resize(newImg, dsize=(int(resolution), int(resolution)), interpolation=cv2.INTER_LINEAR)
    return newImg
           
def get_pts_from_predict(a, center, scale):
    b = a.reshape ( (a.shape[0], a.shape[1]*a.shape[2]) )    
    c = b.argmax(1).reshape ( (a.shape[0], 1) ).repeat(2, axis=1).astype(np.float)
    c[:,0] %= a.shape[2]    
    c[:,1] = np.apply_along_axis ( lambda x: np.floor(x / a.shape[2]), 0, c[:,1] )

    for i in range(a.shape[0]):
        pX, pY = int(c[i,0]), int(c[i,1])
        if pX > 0 and pX < 63 and pY > 0 and pY < 63:
            diff = np.array ( [a[i,pY,pX+1]-a[i,pY,pX-1], a[i,pY+1,pX]-a[i,pY-1,pX]] )
            c[i] += np.sign(diff)*0.25
   
    c += 0.5
    return [ transform (c[i], center, scale, a.shape[2]) for i in range(a.shape[0]) ]

def initialize(detector, scale_to=2048):
    global dlib_detectors
    global keras_model
    global is_initialized
    if not is_initialized:
        dlib_cnn_face_detector_path = os.path.join(os.path.dirname(__file__), "mmod_human_face_detector.dat")
        if not os.path.exists(dlib_cnn_face_detector_path):
            raise Exception ("Error: Unable to find %s, reinstall the lib !" % (dlib_cnn_face_detector_path) )
        
        if detector == 'cnn' or detector == "all":
            dlib_cnn_face_detector = dlib.cnn_face_detection_model_v1(dlib_cnn_face_detector_path)            
            #DLIB and TF competiting for VRAM, so dlib must do first allocation to prevent OOM error 
            dlib_cnn_face_detector ( np.zeros ( (scale_to, scale_to, 3), dtype=np.uint8), 0 ) 
            dlib_detectors.append(dlib_cnn_face_detector)
        
        if detector == "hog" or detector == "all":
            dlib_face_detector = dlib.get_frontal_face_detector()
            dlib_face_detector ( np.zeros ( (scale_to, scale_to, 3), dtype=np.uint8), 0 )
            dlib_detectors.append(dlib_face_detector)        
    
        keras_model_path = os.path.join( os.path.dirname(__file__) , "2DFAN-4.h5" )
        if not os.path.exists(keras_model_path):
            print ("Error: Unable to find %s, reinstall the lib !" % (keras_model_path) )
        else:
            print ("Info: initializing keras model...")
            keras_model = keras.models.load_model (keras_model_path, custom_objects={'TorchBatchNorm2D': TorchBatchNorm2D} ) 
            
        is_initialized = True

#scale_to=2048 with dlib upsamples=0 for 3GB VRAM Windows 10 users        
#you should not extract landmarks again from predetected face, because many face data lost, so result will be much different against extract from original big image
def extract(input_image_bgr, detector, verbose, all_faces=True, input_is_predetected_face=False, scale_to=2048):
    initialize(detector, scale_to)
    global dlib_detectors
    global keras_model
    
    (h, w, ch) = input_image_bgr.shape

    detected_faces = []
    
    if input_is_predetected_face:
        input_scale = 1.0
        detected_faces = [ dlib.rectangle(0, 0, w, h) ]
        input_image = input_image_bgr[:,:,::-1].copy()
    else:
        input_scale = scale_to / (w if w > h else h)
        input_image_bgr = cv2.resize (input_image_bgr, ( int(w*input_scale), int(h*input_scale) ), interpolation=cv2.INTER_LINEAR)
        input_image = input_image_bgr[:,:,::-1].copy() #cv2 and numpy inputs differs in rgb-bgr order, this affects chance of dlib face detection
        input_images = [input_image, input_image_bgr]
        for current_detector, current_image in ((current_detector, current_image) for current_detector in dlib_detectors for current_image in input_images):
            detected_faces = current_detector(current_image, 0)
            if len(detected_faces) != 0:
                break

    landmarks = []
    if len(detected_faces) > 0:        
        for i, d_rect in enumerate(detected_faces):
            if i > 0 and not all_faces:
                break
        
            if type(d_rect) == dlib.mmod_rectangle:
                d_rect = d_rect.rect
            
            left, top, right, bottom = d_rect.left(), d_rect.top(), d_rect.right(), d_rect.bottom()
            del d_rect
    
            center = np.array( [ (left + right) / 2.0, (top + bottom) / 2.0] )
            center[1] -= (bottom - top) * 0.12
            scale = (right - left + bottom - top) / 195.0
        
            image = crop(input_image, center, scale).transpose ( (2,0,1) ).astype(np.float32) / 255.0
            image = np.expand_dims(image, 0)
            
            pts_img = get_pts_from_predict ( keras_model.predict (image)[-1][0], center, scale)
            pts_img = [ ( int(pt[0]/input_scale), int(pt[1]/input_scale) ) for pt in pts_img ]             
            landmarks.append ( ((  int(left/input_scale), int(top/input_scale), int(right/input_scale), int(bottom/input_scale) ),pts_img) )
    elif verbose:
        print("Warning: No faces were detected.")
        
    return landmarks
