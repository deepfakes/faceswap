import numpy

from lib.umeyama import umeyama
from lib.align_eyes import align_eyes
from numpy.linalg import inv
import cv2

mean_face_x = numpy.array([
0.000213256, 0.0752622, 0.18113, 0.29077, 0.393397, 0.586856, 0.689483, 0.799124,
0.904991, 0.98004, 0.490127, 0.490127, 0.490127, 0.490127, 0.36688, 0.426036,
0.490127, 0.554217, 0.613373, 0.121737, 0.187122, 0.265825, 0.334606, 0.260918,
0.182743, 0.645647, 0.714428, 0.793132, 0.858516, 0.79751, 0.719335, 0.254149,
0.340985, 0.428858, 0.490127, 0.551395, 0.639268, 0.726104, 0.642159, 0.556721,
0.490127, 0.423532, 0.338094, 0.290379, 0.428096, 0.490127, 0.552157, 0.689874,
0.553364, 0.490127, 0.42689 ])

mean_face_y = numpy.array([
0.106454, 0.038915, 0.0187482, 0.0344891, 0.0773906, 0.0773906, 0.0344891,
0.0187482, 0.038915, 0.106454, 0.203352, 0.307009, 0.409805, 0.515625, 0.587326,
0.609345, 0.628106, 0.609345, 0.587326, 0.216423, 0.178758, 0.179852, 0.231733,
0.245099, 0.244077, 0.231733, 0.179852, 0.178758, 0.216423, 0.244077, 0.245099,
0.780233, 0.745405, 0.727388, 0.742578, 0.727388, 0.745405, 0.780233, 0.864805,
0.902192, 0.909281, 0.902192, 0.864805, 0.784792, 0.778746, 0.785343, 0.778746,
0.784792, 0.824182, 0.831803, 0.824182 ])

landmarks_2D = numpy.stack( [ mean_face_x, mean_face_y ], axis=1 )

def get_align_mat(face, size, should_align_eyes):
    mat_umeyama = umeyama(numpy.array(face.landmarksAsXY()[17:]), landmarks_2D, True)[0:2]

    if should_align_eyes is False:
        return mat_umeyama

    mat_umeyama = mat_umeyama * size

    # Convert to matrix
    landmarks = numpy.matrix(face.landmarksAsXY())

    # cv2 expects points to be in the form np.array([ [[x1, y1]], [[x2, y2]], ... ]), we'll expand the dim
    landmarks = numpy.expand_dims(landmarks, axis=1)

    # Align the landmarks using umeyama
    umeyama_landmarks = cv2.transform(landmarks, mat_umeyama, landmarks.shape)

    # Determine a rotation matrix to align eyes horizontally
    mat_align_eyes = align_eyes(umeyama_landmarks, size)

    # Extend the 2x3 transform matrices to 3x3 so we can multiply them
    # and combine them as one
    mat_umeyama = numpy.matrix(mat_umeyama)
    mat_umeyama.resize((3, 3))
    mat_align_eyes = numpy.matrix(mat_align_eyes)
    mat_align_eyes.resize((3, 3))
    mat_umeyama[2] = mat_align_eyes[2] = [0, 0, 1]

    # Combine the umeyama transform with the extra rotation matrix
    transform_mat = mat_align_eyes * mat_umeyama

    # Remove the extra row added, shape needs to be 2x3
    transform_mat = numpy.delete(transform_mat, 2, 0)
    transform_mat = transform_mat / size
    return transform_mat
