import copy
'''
You can implement your own Converter, check example ConverterMasked.py
'''
import multiprocessing

class ConverterBase(object):

    #overridable
    def __init__(self, predictor):
        self.predictor = predictor
        
    #overridable
    def convert (self, image, image_face_landmarks, debug):
        #return float32 image        
        #if debug , return tuple (128, images of any size and channels, ...) where 128 is desired size for all images
        return image
        
    #overridable
    def dummy_predict(self):
        #do dummy predict here
        pass

    def copy(self):
        return copy.copy(self)
        
    def copy_and_set_predictor(self, predictor):
        result = self.copy()
        result.predictor = predictor
        return result