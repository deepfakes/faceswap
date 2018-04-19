import sys
import numpy as np
import cv2
import localization
from scipy.spatial import Delaunay
from PIL import Image, ImageDraw, ImageFont

def hist_match(source, template, mask=None):
    # Code borrowed from:
    # https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x
    masked_source = source
    masked_template = template

    if mask is not None:
        masked_source = source * mask
        masked_template = template * mask

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()
    masked_source = masked_source.ravel()
    masked_template = masked_template.ravel()
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)
    ms_values, mbin_idx, ms_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    mt_values, mt_counts = np.unique(template, return_counts=True)

    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)    

def color_hist_match(src_im, tar_im, mask=None):
    h,w,c = src_im.shape
    matched_R = hist_match(src_im[:,:,0], tar_im[:,:,0], mask)
    matched_G = hist_match(src_im[:,:,1], tar_im[:,:,1], mask)
    matched_B = hist_match(src_im[:,:,2], tar_im[:,:,2], mask)
    
    to_stack = (matched_R, matched_G, matched_B)
    for i in range(3, c):
        to_stack += ( src_im[:,:,i],)
    
    
    matched = np.stack(to_stack, axis=2).astype(src_im.dtype)
    return matched
    

pil_fonts = {}
def get_text_image( shape, text, color=(1,1,1), border=0.2, font=None):
    def get_pil_font (font, size):
        global pil_fonts
        font_str_id = '%s_%d' % (font, size)
        if font_str_id not in pil_fonts.keys():
            pil_fonts[font_str_id] = ImageFont.truetype(font + ".ttf", size=size, encoding="unic")
        pil_font = pil_fonts[font_str_id]
        return pil_font    

    size = shape[1]
    
    try:
        pil_font = get_pil_font( localization.get_default_ttf_font_name() , size)
    except:
        pil_font = ImageFont.load_default()
    
    text_width, text_height = pil_font.getsize(text)

    canvas = Image.new('RGB', shape[0:2], (0,0,0) )

    draw = ImageDraw.Draw(canvas)
    offset = ( 0, 0) #int(min(shape[0],shape[1])*border)
    draw.text(offset, text, font=pil_font, fill=tuple((np.array(color)*255).astype(np.int)) )
    
    result = np.asarray(canvas) / 255
    if shape[2] != 3:        
        result = np.concatenate ( (result, np.ones ( (shape[1],) + (shape[0],) + (shape[2]-3,)) ), axis=2 )

    return result

def draw_text( image, rect, text, color=(1,1,1), border=0.2, font=None):
    h,w,c = image.shape
 
    l,t,r,b = rect
    l = np.clip (l, 0, w-1)
    r = np.clip (r, 0, w-1)
    t = np.clip (t, 0, h-1)
    b = np.clip (b, 0, h-1)
    
    image[t:b, l:r] += get_text_image (  (r-l,b-t,c) , text, color, border, font )
                
def draw_text_lines (image, rect, text_lines, color=(1,1,1), border=0.2, font=None):
    text_lines_len = len(text_lines)
    if text_lines_len == 0:
        return
        
    l,t,r,b = rect
    h = b-t
    h_per_line = h // text_lines_len
    
    for i in range(0, text_lines_len):
        draw_text (image, (l, i*h_per_line, r, (i+1)*h_per_line), text_lines[i], color, border, font)
        
def get_draw_text_lines ( image, rect, text_lines, color=(1,1,1), border=0.2, font=None):
    image = np.zeros ( image.shape, dtype=np.float )
    draw_text_lines ( image, rect, text_lines, color, border, font)
    return image
        
  
def draw_polygon (image, points, color, thickness = 1):
    points_len = len(points)
    for i in range (0, points_len):
        p0 = tuple( points[i] )
        p1 = tuple( points[ (i+1) % points_len] )
        cv2.line (image, p0, p1, color, thickness=thickness)
        
def draw_rect(image, rect, color, thickness=1):
    l,t,r,b = rect
    draw_polygon (image, [ (l,t), (r,t), (r,b), (l,b ) ], color, thickness)

def rectContains(rect, point) :
    return not (point[0] < rect[0] or point[0] >= rect[2] or point[1] < rect[1] or point[1] >= rect[3])

def applyAffineTransform(src, srcTri, dstTri, size) :
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    return cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )    
    
def morphTriangle(dst_img, src_img, st, dt) :                                
    (h,w,c) = dst_img.shape
    sr = np.array( cv2.boundingRect(np.float32(st)) )
    dr = np.array( cv2.boundingRect(np.float32(dt)) )
    sRect = st - sr[0:2]
    dRect = dt - dr[0:2]
    d_mask = np.zeros((dr[3], dr[2], c), dtype = np.float32)
    cv2.fillConvexPoly(d_mask, np.int32(dRect), (1.0,)*c, 8, 0);                                    
    imgRect = src_img[sr[1]:sr[1] + sr[3], sr[0]:sr[0] + sr[2]]                                    
    size = (dr[2], dr[3])                                    
    warpImage1 = applyAffineTransform(imgRect, sRect, dRect, size)                      
    dst_img[dr[1]:dr[1]+dr[3], dr[0]:dr[0]+dr[2]] = dst_img[dr[1]:dr[1]+dr[3], dr[0]:dr[0]+dr[2]]*(1-d_mask) + warpImage1 * d_mask
    
def morph_by_points (image, sp, dp):
    if sp.shape != dp.shape:
        raise ValueError ('morph_by_points() sp.shape != dp.shape')
    (h,w,c) = image.shape    

    result_image = np.zeros(image.shape, dtype = image.dtype)

    for tri in Delaunay(dp).simplices:                                    
        morphTriangle(result_image, image, sp[tri], dp[tri])
        
    return result_image
    
def equalize_and_stack (wh, images, axis=1):
    max_c = max ([ 1 if len(image.shape) == 2 else image.shape[2]  for image in images ] )
            
    for i,image in enumerate(images):
        if len(image.shape) == 2:
            h,w = image.shape
            c = 1
        else:
            h,w,c = image.shape
            
        if c < max_c:
            if c == 1:
                if len(image.shape) == 2:
                    image = np.expand_dims ( image, 2 )                
                image = np.concatenate ( (image,)*max_c, axis=2 )
            else:
                image = np.concatenate ( (image, np.ones((h,w,max_c - c))), axis=2 )

        if h != wh or w != wh:
            image = cv2.resize ( image, (wh, wh) )
            h,w,c = image.shape
                
        images[i] = image
        
    return np.concatenate ( images, axis = 1 )
    
def bgr2hsv (img):    
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
def hsv2bgr (img):    
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    
def bgra2hsva (img):    
    return np.concatenate ( (cv2.cvtColor(img[...,0:3], cv2.COLOR_BGR2HSV ), np.expand_dims (img[...,3], -1)), -1 )

def bgra2hsva_list (imgs):
    return [ bgra2hsva(img) for img in imgs ]
    
def hsva2bgra (img):
    return np.concatenate ( (cv2.cvtColor(img[...,0:3], cv2.COLOR_HSV2BGR ), np.expand_dims (img[...,3], -1)), -1 )

def hsva2bgra_list (imgs):
    return [ hsva2bgra(img) for img in imgs ]