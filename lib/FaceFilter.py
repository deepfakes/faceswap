# import dlib
# import numpy as np
import face_recognition
# import face_recognition_models

def avg(arr):
  return sum(arr)*1.0/len(arr)

class FaceFilter():
    def __init__(self, reference_file_paths, nreference_file_paths, threshold = 0.6):
        images = list(map(face_recognition.load_image_file, reference_file_paths))
        nimages = list(map(face_recognition.load_image_file, nreference_file_paths))
        self.encodings = list(map(lambda im: face_recognition.face_encodings(im)[0], images)) # Note: we take only first face, so the reference file should only contain one face.
        self.nencodings = list(map(lambda im: face_recognition.face_encodings(im)[0], nimages)) # Note: we take only first face, so the reference file should only contain one face.
        self.threshold = threshold
    
    def check(self, detected_face):
        encodings = face_recognition.face_encodings(detected_face.image) # we could use detected landmarks, but I did not manage to do so. TODO The copy/paste below should help
        if encodings is not None and len(encodings) > 0:
            distances = face_recognition.face_distance(self.encodings, encodings[0])
            ndistances = face_recognition.face_distance(self.nencodings, encodings[0])
            ndistance = avg(ndistances)
            distance = avg(distances)
            nmindistance = min(ndistances)
            mindistance = min(distances)
            nmaxdistance = max(ndistances)
            maxdistance = max(distances)
            #distance = max(face_recognition.face_distance(self.encodings, encoding[0]))
            if distance <= self.threshold:
                print("Distance below threshold: %f < %f" % (distance, self.threshold))
            chosen = distance <= self.threshold
            chosen = chosen and (mindistance < nmindistance)
            chosen = chosen and (distance < ndistance)
            # k-nn classifier
            K=min(4, int((len(distances) + len(ndistances))/2))
            N=sum(list(map(lambda x: x[0],
                  list(sorted([(1,d) for d in distances] + [(0,d) for d in ndistances],
                              key=lambda x: x[1]))[:K])))
            ratio = N/K
            chosen = chosen and (ratio > 0.5)
            return chosen
        else:
            print("No face encodings found")
            return False

# # Copy/Paste (mostly) from private method in face_recognition
# face_recognition_model = face_recognition_models.face_recognition_model_location()
# face_encoder = dlib.face_recognition_model_v1(face_recognition_model)

# def convert(detected_face):
#     return np.array(face_encoder.compute_face_descriptor(detected_face.image, detected_face.landmarks, 1))
# # end of Copy/Paste
