from models import ConverterBase
from facelib import LandmarksProcessor
import cv2
import numpy as np
from utils import image_utils

'''
predictor_func: 
    input:  [predictor_input_size, predictor_input_size, BGRA]
    output: [predictor_input_size, predictor_input_size, BGRA]
'''

class ConverterMasked(ConverterBase):
    def __init__(self, predictor_func, predictor_input_size, output_size, face_type, mode='seamless', mask_type='intersect', **in_options):
        self.predictor_func = predictor_func
        self.output_size = output_size
        self.face_type = face_type
        self.predictor_input_size = predictor_input_size
        self.mode = mode
        self.mask_type = mask_type
        
    #override
    def convert (self, img, img_face_landmarks, debug):
        
        if debug:        
            debugs = [img.copy()]

        img_size = img.shape[1], img.shape[0]

        img_face_mask = LandmarksProcessor.get_image_hull_mask (img, img_face_landmarks)
        
        face_mat = LandmarksProcessor.get_transform_mat (img_face_landmarks, self.output_size, face_type=self.face_type)
        dst_face      = cv2.warpAffine( img          , face_mat, (self.output_size, self.output_size) )
        dst_face_mask = cv2.warpAffine( img_face_mask, face_mat, (self.output_size, self.output_size) )

        predictor_input      = cv2.resize (dst_face,      (self.predictor_input_size,self.predictor_input_size))
        predictor_input_mask = cv2.resize (dst_face_mask, (self.predictor_input_size,self.predictor_input_size))
        predictor_input_mask = np.expand_dims (predictor_input_mask, -1) 
        
        predict_result = self.predictor_func ( np.concatenate( (predictor_input, predictor_input_mask), -1) )

        prd_face           = np.clip (predict_result[:,:,0:3], 0, 1.0 )
        prd_face_mask_1D_0 = np.clip (predict_result[:,:,3], 0.0, 1.0)
        prd_face_mask_1D_0[ prd_face_mask_1D_0 < 0.001 ] = 0.0
        prd_face_mask_1D = np.expand_dims (prd_face_mask_1D_0, axis=2)
        prd_face_mask    = np.repeat ( prd_face_mask_1D, (3,), axis=2)

        img_prd_face_mask_3D = cv2.warpAffine( prd_face_mask, face_mat, img_size, np.zeros(img.shape, dtype=float), cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC )
        img_prd_face_mask_3D = np.clip (img_prd_face_mask_3D, 0.0, 1.0)
        
        
        if self.mask_type == 'predicted':
            img_mask_hard = img_prd_face_mask_3D
        elif self.mask_type == 'intersect':
            img_mask_hard = img_face_mask*img_prd_face_mask_3D #intersection              
        elif self.mask_type == 'dst':
            img_mask_hard = img_face_mask
            img_mask_hard = np.repeat (img_mask_hard, (3,), -1)

        if debug:
            debugs += [img_mask_hard.copy()]
            
        img_mask_hard_copy = img_mask_hard.copy()
        img_mask_hard_copy[img_mask_hard_copy > 0.1] = 1.0

        maxregion = np.argwhere(img_mask_hard_copy==1.0)        
        out_img = np.copy( img )        
        if maxregion.size != 0:
            miny,minx = maxregion.min(axis=0)[:2]
            maxy,maxx = maxregion.max(axis=0)[:2]
            lenx = maxx - minx
            leny = maxy - miny
            masky = int(minx+(lenx//2))
            maskx = int(miny+(leny//2))
            lowest_len = min (lenx, leny)
            ero = int( lowest_len * 0.085 )
            blur = int( lowest_len * 0.10 )       
            img_mask_blurry = cv2.erode(img_mask_hard, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ero,ero)), iterations = 1 )
            img_mask_blurry = cv2.blur(img_mask_blurry, (blur, blur) )
            
            if self.mode == 'hist-match':
                prd_face = image_utils.color_hist_match(prd_face* prd_face_mask_1D, dst_face* prd_face_mask_1D)
                 
            cv2.warpAffine( prd_face, face_mat, img_size, out_img, cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC, cv2.BORDER_TRANSPARENT )
            
            if debug:
                debugs += [out_img.copy()]
                debugs += [img_mask_blurry.copy()]

            if self.mode == 'seamless':
                out_img =  np.clip( img*(1-img_mask_hard) + (out_img*img_mask_hard) , 0, 1.0 )
                out_img = cv2.seamlessClone( (out_img*255).astype(np.uint8), (img*255).astype(np.uint8), (img_mask_hard_copy*255).astype(np.uint8), (masky,maskx) , cv2.NORMAL_CLONE )
                out_img = out_img.astype(np.float32) / 255.0

            out_img =  np.clip( img*(1-img_mask_blurry) + (out_img*img_mask_blurry) , 0, 1.0 )
           
        if debug:
            debugs += [out_img]
            
        return debugs if debug else out_img     
     