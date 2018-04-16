import colorsys
import cv2
import numpy as np
from enum import IntEnum
from mathlib.umeyama import umeyama
from utils import image_utils


mean_face_x = np.array([
0.000213256, 0.0752622, 0.18113, 0.29077, 0.393397, 0.586856, 0.689483, 0.799124,
0.904991, 0.98004, 0.490127, 0.490127, 0.490127, 0.490127, 0.36688, 0.426036,
0.490127, 0.554217, 0.613373, 0.121737, 0.187122, 0.265825, 0.334606, 0.260918,
0.182743, 0.645647, 0.714428, 0.793132, 0.858516, 0.79751, 0.719335, 0.254149,
0.340985, 0.428858, 0.490127, 0.551395, 0.639268, 0.726104, 0.642159, 0.556721,
0.490127, 0.423532, 0.338094, 0.290379, 0.428096, 0.490127, 0.552157, 0.689874,
0.553364, 0.490127, 0.42689 ])

mean_face_y = np.array([
0.106454, 0.038915, 0.0187482, 0.0344891, 0.0773906, 0.0773906, 0.0344891,
0.0187482, 0.038915, 0.106454, 0.203352, 0.307009, 0.409805, 0.515625, 0.587326,
0.609345, 0.628106, 0.609345, 0.587326, 0.216423, 0.178758, 0.179852, 0.231733,
0.245099, 0.244077, 0.231733, 0.179852, 0.178758, 0.216423, 0.244077, 0.245099,
0.780233, 0.745405, 0.727388, 0.742578, 0.727388, 0.745405, 0.780233, 0.864805,
0.902192, 0.909281, 0.902192, 0.864805, 0.784792, 0.778746, 0.785343, 0.778746,
0.784792, 0.824182, 0.831803, 0.824182 ])

landmarks_2D = np.stack( [ mean_face_x, mean_face_y ], axis=1 )

def get_transform_mat (image_landmarks, output_size, is_full_face):
    if output_size != 64 and output_size != 128 and output_size != 256 and output_size != 512:
        raise ValueError ('get_transform_mat() output_size must be power of 2')
        
    padding = (output_size // 64) * 12 if is_full_face else 0

    if not isinstance(image_landmarks, np.ndarray):
        image_landmarks = np.array (image_landmarks) 
    mat = umeyama(image_landmarks[17:], landmarks_2D, True)[0:2]
    mat = mat * (output_size - 2 * padding)
    mat[:,2] += padding
    return mat

def transform_points(points, mat, invert=False):
    if invert:
        mat = cv2.invertAffineTransform (mat)
    points = np.expand_dims(points, axis=1)
    points = cv2.transform(points, mat, points.shape)
    points = np.squeeze(points)
    return points
    
    
def get_image_hull_mask (image, image_landmarks):        
    if len(image_landmarks) != 68:
        raise Exception('get_image_hull_mask work only with 68 landmarks')
        
    hull_mask = np.zeros(image.shape[0:2]+(1,),dtype=np.float32)

    cv2.fillConvexPoly( hull_mask, cv2.convexHull( np.concatenate ( (image_landmarks[0:17], image_landmarks[48:], [image_landmarks[0]], [image_landmarks[8]], [image_landmarks[16]]))    ), (1,) )
    cv2.fillConvexPoly( hull_mask, cv2.convexHull( np.concatenate ( (image_landmarks[27:31], [image_landmarks[33]]) )                                                                    ), (1,) )
    cv2.fillConvexPoly( hull_mask, cv2.convexHull( np.concatenate ( (image_landmarks[17:27], [image_landmarks[0]], [image_landmarks[27]], [image_landmarks[16]], [image_landmarks[33]])) ), (1,) )
    
    return hull_mask
    
def get_image_hull_mask_3D (image, image_landmarks):    
    result = get_image_hull_mask(image, image_landmarks)
    
    return np.repeat ( result, (3,), -1 )

def blur_image_hull_mask (hull_mask):

    maxregion = np.argwhere(hull_mask==1.0)
    miny,minx = maxregion.min(axis=0)[:2]
    maxy,maxx = maxregion.max(axis=0)[:2]
    lenx = maxx - minx;
    leny = maxy - miny;
    masky = int(minx+(lenx//2))
    maskx = int(miny+(leny//2))
    lowest_len = min (lenx, leny)        
    ero = int( lowest_len * 0.085 )
    blur = int( lowest_len * 0.10 )

    hull_mask = cv2.erode(hull_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ero,ero)), iterations = 1 )
    hull_mask = cv2.blur(hull_mask, (blur, blur) )
    hull_mask = np.expand_dims (hull_mask,-1)

    return hull_mask
    
def get_blurred_image_hull_mask(image, image_landmarks):        
    return blur_image_hull_mask ( get_image_hull_mask(image, image_landmarks) )
    
mirror_idxs = [
    [0,16],
    [1,15],
    [2,14],
    [3,13],
    [4,12],
    [5,11],
    [6,10],
    [7,9],
    
    [17,26],
    [18,25],
    [19,24],
    [20,23],
    [21,22],    
    
    [36,45],
    [37,44],
    [38,43],
    [39,42],
    [40,47],
    [41,46],    
    
    [31,35],
    [32,34],
    
    [50,52],
    [49,53],
    [48,54],
    [59,55],
    [58,56],
    [67,65],
    [60,64],
    [61,63] ]
    
def mirror_landmarks (landmarks, val):    
    result = landmarks.copy()
    
    for idx in mirror_idxs:
        result [ idx ] = result [ idx[::-1] ]

    result[:,0] = val - result[:,0] - 1
    return result
    
def draw_landmarks (image, image_landmarks, color):

    for i, (x, y) in enumerate(image_landmarks):
        cv2.circle(image, (x, y), 2, color, -1)
        #text_color = colorsys.hsv_to_rgb ( (i%4) * (0.25), 1.0, 1.0 )
        #cv2.putText(image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.1,text_color,1)
        
def draw_rect_landmarks (image, rect, image_landmarks, face_size, is_full_face):
    image_utils.draw_rect (image, rect, (255,0,0), 2 )
    draw_landmarks(image, image_landmarks, (0,255,0) )
    
    image_to_face_mat = get_transform_mat (image_landmarks, face_size, is_full_face)        
    points = transform_points ( [ (0,0), (0,face_size-1), (face_size-1, face_size-1), (face_size-1,0) ], image_to_face_mat, True)
    image_utils.draw_polygon (image, points, (0,0,255), 2)  
    
def calc_face_pitch(landmarks):
    if not isinstance(landmarks, np.ndarray):
        landmarks = np.array (landmarks)
    t = ( (landmarks[6][1]-landmarks[8][1]) + (landmarks[10][1]-landmarks[8][1]) ) / 2.0   
    b = landmarks[8][1]
    return float(b-t)
def calc_face_yaw(landmarks):
    if not isinstance(landmarks, np.ndarray):
        landmarks = np.array (landmarks)
    l = ( (landmarks[27][0]-landmarks[0][0]) + (landmarks[28][0]-landmarks[1][0]) + (landmarks[29][0]-landmarks[2][0]) ) / 3.0   
    r = ( (landmarks[16][0]-landmarks[27][0]) + (landmarks[15][0]-landmarks[28][0]) + (landmarks[14][0]-landmarks[29][0]) ) / 3.0
    return float(r-l)
  