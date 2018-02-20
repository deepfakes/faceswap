import dlib
import face_recognition
import face_recognition_models
import face_alignment

def detect_faces(frame, model="hog"):
    face_locations = face_recognition.face_locations(frame, model=model)
    #landmarks = _raw_face_landmarks(frame, face_locations)

    if model == "cnn":
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, enable_cuda=True, 
                flip_input=True)
    else:
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, enable_cuda=True, 
                flip_input=True)

    raw_landmarks = fa.get_landmarks(frame, all_faces=True)
    landmarksXY = []

    if raw_landmarks is not None:
        for raw_landmark in raw_landmarks:
            landmarksXY.append([(int(p[0]), int(p[1])) for p in raw_landmark])
        for ((y, right, bottom, x), landmarks) in zip(face_locations, landmarksXY):
            yield DetectedFace(frame[y: bottom, x: right], x, right - x, y, 
                    bottom - y, landmarks=landmarks, landmarksXY=landmarks)

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
    def __init__(self, image=None, x=None, w=None, y=None, h=None, landmarksXY=None):
        self.image = image
        self.x = x
        self.w = w
        self.y = y
        self.h = h
        self.landmarksXY = landmarksXY

    def landmarksAsXY(self):
        return self.landmarksXY