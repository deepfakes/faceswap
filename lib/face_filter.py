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
        logger.debug("Initialized %s", self.__class__.__name__)

    def check(self, detected_face):
        """ Check Face
            we could use detected landmarks, but I did not manage to do so.
            TODO The copy/paste below should help """
        logger.trace("Checking face with FaceFilter")
        encodings = face_recognition.face_encodings(detected_face.image)
        if encodings:
            distances = list(face_recognition.face_distance(self.encodings, encodings[0]))
            distance = avg(distances)
            mindistance = min(distances)
            if distance > self.threshold:
                logger.info("Distance above threshold: %f < %f", distance, self.threshold)
                retval = False
            elif self.nencodings:
                ndistances = list(face_recognition.face_distance(self.nencodings, encodings[0]))
                ndistance = avg(ndistances)
                nmindistance = min(ndistances)
                if mindistance > nmindistance:
                    logger.info("Distance to negative sample is smaller")
                    retval = False
                elif distance > ndistance:
                    logger.info("Average distance to negative sample is smaller")
                    retval = False
                # k-nn classifier
                var_k = min(5, min(len(distances), len(ndistances)) + 1)
                var_n = sum(list(map(lambda x: x[0],
                                     list(sorted([(1, d) for d in distances] +
                                                 [(0, d) for d in ndistances],
                                                 key=lambda x: x[1]))[:var_k])))
                ratio = var_n/var_k
                if ratio < 0.5:
                    logger.info("K-nn is %.2f", ratio)
                    retval = False
            else:
                retval = True
        else:
            retval = False
        if not retval:
            logger.info("No face encodings found")
        logger.trace("Returning: %s", retval)
        return retval

# # Copy/Paste (mostly) from private method in face_recognition
# face_recognition_model = face_recognition_models.face_recognition_model_location()
# face_encoder = dlib.face_recognition_model_v1(face_recognition_model)

# def convert(detected_face):
#     return np.array(face_encoder.compute_face_descriptor(detected_face.image,
#                                                          detected_face.landmarks,
#                                                          1))
# # end of Copy/Paste
