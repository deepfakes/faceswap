#!/usr/bin python3
""" Face Filterer for extraction in faceswap.py """

import logging

import face_recognition


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def avg(arr):
    """ Return an average """
    return sum(arr) * 1.0 / len(arr)


class FaceFilter():
    """ Face filter for extraction """
    def __init__(self, reference_file_paths, nreference_file_paths, threshold=0.6):
        logger.debug("Initializing %s: (reference_file_paths: %s, nreference_file_paths: %s, "
                     "threshold: %s)", self.__class__.__name__, reference_file_paths,
                     nreference_file_paths, threshold)
        images = list(map(face_recognition.load_image_file, reference_file_paths))
        nimages = list(map(face_recognition.load_image_file, nreference_file_paths))
        # Note: we take only first face, so the reference file should only contain one face.
        self.encodings = list(map(lambda im: face_recognition.face_encodings(im)[0], images))
        self.nencodings = list(map(lambda im: face_recognition.face_encodings(im)[0], nimages))
        self.threshold = threshold
        logger.trace("encodings: %s", self.encodings)
        logger.trace("nencodings: %s", self.nencodings)
        logger.debug("Initialized %s", self.__class__.__name__)

    def check(self, detected_face):
        """ Check Face
            we could use detected landmarks, but I did not manage to do so.
            TODO The copy/paste below should help """
        logger.trace("Checking face with FaceFilter")
        encodings = face_recognition.face_encodings(detected_face.image)
        if not encodings:
            logger.verbose("No face encodings found")
            return False

        if self.encodings:
            distances = list(face_recognition.face_distance(self.encodings, encodings[0]))
            logger.trace("Distances: %s", distances)
            distance = avg(distances)
            logger.trace("Average Distance: %s", distance)
            mindistance = min(distances)
            logger.trace("Minimum Distance: %s", mindistance)
            if distance > self.threshold:
                logger.verbose("Distance above threshold: %f < %f", distance, self.threshold)
                return False
        if self.nencodings:
            ndistances = list(face_recognition.face_distance(self.nencodings, encodings[0]))
            logger.trace("nDistances: %s", ndistances)
            ndistance = avg(ndistances)
            logger.trace("Average nDistance: %s", ndistance)
            nmindistance = min(ndistances)
            logger.trace("Minimum nDistance: %s", nmindistance)
            if not self.encodings and ndistance < self.threshold:
                logger.verbose("nDistance below threshold: %f < %f", ndistance, self.threshold)
                return False
            if self.encodings:
                if mindistance > nmindistance:
                    logger.verbose("Distance to negative sample is smaller")
                    return False
                if distance > ndistance:
                    logger.verbose("Average distance to negative sample is smaller")
                    return False
                # k-nn classifier
                var_k = min(5, min(len(distances), len(ndistances)) + 1)
                var_n = sum(list(map(lambda x: x[0],
                                     list(sorted([(1, d) for d in distances] +
                                                 [(0, d) for d in ndistances],
                                                 key=lambda x: x[1]))[:var_k])))
                ratio = var_n/var_k
                if ratio < 0.5:
                    logger.verbose("K-nn is %.2f", ratio)
                    return False
        return True


# # Copy/Paste (mostly) from private method in face_recognition
# face_recognition_model = face_recognition_models.face_recognition_model_location()
# face_encoder = dlib.face_recognition_model_v1(face_recognition_model)

# def convert(detected_face):
#     return np.array(face_encoder.compute_face_descriptor(detected_face.image,
#                                                          detected_face.landmarks,
#                                                          1))
# # end of Copy/Paste
