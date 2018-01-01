import face_recognition
from .DetectedFace import DetectedFace

def detect_faces(frame):
    face_locations = face_recognition.face_locations(frame)
    #face_encodings = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left) in face_locations:
        x = left
        y = top
        w = right - left
        h = bottom - top
        yield DetectedFace(frame[y: y + h, x: x + w], x, w, y, h, None)
