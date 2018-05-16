import numpy as np
import os
import cv2
from pathlib import Path

from utils import std_utils



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

    
class LandmarksExtractor(object):
    def __init__ (self, keras):
        self.keras = keras
        K = self.keras.backend
        class TorchBatchNorm2D(self.keras.engine.topology.Layer):
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
        self.TorchBatchNorm2D = TorchBatchNorm2D
        
    def __enter__(self):        
        keras_model_path = Path(__file__).parent / "2DFAN-4.h5"
        if not keras_model_path.exists():
            return None

        self.keras_model = self.keras.models.load_model ( str(keras_model_path), custom_objects={'TorchBatchNorm2D': self.TorchBatchNorm2D} ) 
    
        return self
        
    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        del self.keras_model
        return False #pass exception between __enter__ and __exit__ to outter level
        
    def extract_from_bgr (self, input_image, rects):
        input_image = input_image[:,:,::-1].copy()
        (h, w, ch) = input_image.shape
        
        landmarks = []
        for (left, top, right, bottom) in rects:
            
            center = np.array( [ (left + right) / 2.0, (top + bottom) / 2.0] )
            center[1] -= (bottom - top) * 0.12
            scale = (right - left + bottom - top) / 195.0
        
            image = crop(input_image, center, scale).transpose ( (2,0,1) ).astype(np.float32) / 255.0
            image = np.expand_dims(image, 0)
            
            with std_utils.suppress_stdout_stderr():
                predicted = self.keras_model.predict (image)
                
            pts_img = get_pts_from_predict ( predicted[-1][0], center, scale)
            pts_img = [ ( int(pt[0]), int(pt[1]) ) for pt in pts_img ]             
            landmarks.append ( ( (left, top, right, bottom),pts_img ) )
   
        return landmarks
