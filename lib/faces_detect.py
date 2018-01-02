import face_recognition
import face_recognition_models
import dlib
from .DetectedFace import DetectedFace

def detect_faces(frame):
    face_locations = face_recognition.face_locations(frame)
    landmarks = _raw_face_landmarks(frame, face_locations)
    landmarks_as_tuples = [[(p.x, p.y) for p in landmark.parts()] for landmark in landmarks]

    for ((top, right, bottom, left), landmarks) in zip(face_locations, landmarks_as_tuples):
        x = left
        y = top
        w = right - left
        h = bottom - top
        yield DetectedFace(frame[y: y + h, x: x + w], x, w, y, h, landmarks)

# Copy/Paste (mostly) from private method in face_recognition
predictor_68_point_model = face_recognition_models.pose_predictor_model_location()
pose_predictor = dlib.shape_predictor(predictor_68_point_model)

def _raw_face_landmarks(face_image, face_locations):
    face_locations = [_css_to_rect(face_location) for face_location in face_locations]
    return [pose_predictor(face_image, face_location) for face_location in face_locations]

def _css_to_rect(css):
    return dlib.rectangle(css[3], css[0], css[1], css[2])
# end of Copy/Paste