from lib import FaceLandmarksExtractor

def detect_faces(frame, model="hog"):
    fd = FaceLandmarksExtractor.extract (frame, True if model == "cnn" else False )
    for face in fd:
        x, y, right, bottom, landmarks = face[0][0], face[0][1], face[0][2], face[0][3], face[1]
        yield DetectedFace(frame[y: bottom, x: right], x, right - x, y, bottom - y, landmarksXY=landmarks)

class DetectedFace(object):
    def __init__(self, image=None, x=None, w=None, y=None, h=None, landmarksXY=None):
        self.image = image
        self.x = x
        self.w = w
        self.y = y
        self.h = h
        self.landmarksXY = landmarksXY

    def landmarksAsXY(self):
        return self.landmarksXY