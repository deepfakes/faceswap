import cv2
from DetectedFace import DetectedFace

# Give right path to the xml file or put it directly in current folder
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def crop_faces(image):
    # Add : cv.EqualizeHist(image, image) ?
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        yield DetectedFace(image[y: y + h, x: x + w], x, w, y, h, None)
