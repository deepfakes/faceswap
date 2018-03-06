# Based on the original https://www.reddit.com/r/deepfakes/ code sample

import cv2

class Extract(object):
    def extract(self, image, face, size, *args):
        return cv2.resize(face.image, (size, size))
