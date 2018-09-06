#!/usr/bin python3
""" Face and landmarks detection for faceswap.py """

from lib import face_alignment


def detect_faces(frame, detector, verbose, rotation=0,
                 dlib_buffer=64, mtcnn_kwargs=None):
    """ Detect faces and draw landmarks in an image """
    face_detect = face_alignment.Extract(frame,
                                         detector,
                                         dlib_buffer,
                                         mtcnn_kwargs,
                                         verbose)
    for face in face_detect.landmarks:
        ax_x, ax_y = face[0][0], face[0][1]
        right, bottom = face[0][2], face[0][3]
        landmarks = face[1]

        yield DetectedFace(frame[ax_y: bottom, ax_x: right],
                           rotation,
                           ax_x,
                           right - ax_x,
                           ax_y,
                           bottom - ax_y,
                           landmarksXY=landmarks)


class DetectedFace():
    """ Detected face and landmark information """
    def __init__(self, image=None, r=0, x=None,
                 w=None, y=None, h=None, landmarksXY=None):
        self.image = image
        self.r = r
        self.x = x
        self.w = w
        self.y = y
        self.h = h
        self.landmarksXY = landmarksXY

    def landmarks_as_xy(self):
        """ Landmarks as XY """
        return self.landmarksXY
