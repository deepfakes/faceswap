'''
You can implement your own Converter, check example ConverterMasked.py
'''

class ConverterBase(object):
    #overridable
    def __init__(self, **in_options):
        pass

    #overridable
    def convert (self, image, image_face_landmarks, debug):
        #return float32 image
        
        #if debug , return tuple (128, images of any size and channels, ...) where 128 is desired size for all images
        return image