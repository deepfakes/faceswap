import cv2

from lib.aligner import get_align_mat

class Extract(object):
    def extract(self, image, face, size):
        if face.landmarks == None:
            return cv2.resize(face.image, (size, size))

        alignment = get_align_mat( face )
        return self.transform( image, alignment, size, 48 )
    
    def transform( self, image, mat, size, padding=0 ):
        mat = mat * (size - 2 * padding)
        mat[:,2] += padding
        return cv2.warpAffine( image, mat, ( size, size ) )
