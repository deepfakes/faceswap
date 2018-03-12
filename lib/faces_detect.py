import dlib
import face_recognition
import face_recognition_models

def detect_faces(frame, model="hog", verbose=False, rotation=0):
    face_locations = face_recognition.face_locations(frame, model=model)
    landmarks = _raw_face_landmarks(frame, face_locations)

    for ((y, right, bottom, x), landmarks) in zip(face_locations, landmarks):
        yield DetectedFace(frame[y: bottom, x: right], rotation, x, right - x, y, bottom - y, [(p.x, p.y) for p in landmarks.parts()])

# Copy/Paste (mostly) from private method in face_recognition
predictor_68_point_model = face_recognition_models.pose_predictor_model_location()
pose_predictor = dlib.shape_predictor(predictor_68_point_model)

def _raw_face_landmarks(face_image, face_locations):
    face_locations = [_css_to_rect(face_location) for face_location in face_locations]
    return [pose_predictor(face_image, face_location) for face_location in face_locations]

def _css_to_rect(css):
    return dlib.rectangle(css[3], css[0], css[1], css[2])
# end of Copy/Paste

class DetectedFace(object):
    def __init__(self, image=None, r=0, x=None, w=None, y=None, h=None, landmarksXY=None, cropped=None):
        self.image = image
        self.r = r
        self.x = x
        self.w = w
        self.y = y
        self.h = h
        self.cropped = cropped
        self.landmarksXY = landmarksXY

    def landmarksAsXY(self):
        return self.landmarksXY
