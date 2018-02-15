# Based on the original https://www.reddit.com/r/deepfakes/ code sample + contribs

import cv2

from lib.aligner import get_align_mat

class Extract(object):
    def extract(self, image, face, debug_landmarks, size):
        if face.landmarks == None:
            print("Warning! landmarks not found. Switching to crop!")
            return cv2.resize(face.image, (size, size))

        # Draws landmarks for debug
        if debug_landmarks:
            for (x, y) in face.landmarksAsXY():
                cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

        alignment = get_align_mat( face )
        return self.transform( image, alignment, size, 48 )
    
    def transform( self, image, mat, size, padding=0 ):
        mat = mat * (size - 2 * padding)
        mat[:,2] += padding
        return cv2.warpAffine( image, mat, ( size, size ) )
