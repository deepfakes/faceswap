import os
import sys
import operator
import numpy as np
import cv2
from tqdm import tqdm
from shutil import copyfile

from pathlib import Path
from utils import Path_utils
from utils.AlignedPNG import AlignedPNG
from facelib import LandmarksProcessor

def estimate_blur(image):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_map = cv2.Laplacian(image, cv2.CV_64F)
    score = np.var(blur_map)
    return score
        
def sort_by_brightness(input_path):
    print ("Sorting by brightness...")
    img_list = [ [x, np.mean ( cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2HSV)[...,2].flatten()  )] for x in tqdm( Path_utils.get_image_paths(input_path), desc="Loading") ]
    print ("Sorting...")
    img_list = sorted(img_list, key=operator.itemgetter(1), reverse=True)    
    return img_list
    
def sort_by_hue(input_path):
    print ("Sorting by hue...")
    img_list = [ [x, np.mean ( cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2HSV)[...,0].flatten()  )] for x in tqdm( Path_utils.get_image_paths(input_path), desc="Loading") ]
    print ("Sorting...")
    img_list = sorted(img_list, key=operator.itemgetter(1), reverse=True)    
    return img_list
    
def sort_by_blur(input_path):
    img_list = []
    print ("Sorting by blur...")        
    for filepath in tqdm( Path_utils.get_image_paths(input_path), desc="Loading"):
        #never mask it by face hull, it worse than whole image blur estimate
        img_list.append ( [filepath, estimate_blur (cv2.imread( filepath ))] )
    
    print ("Sorting...")
    img_list = sorted(img_list, key=operator.itemgetter(1), reverse=True)

    return img_list

def sort_by_face(input_path):

    print ("Sorting by face similarity...")

    img_list = []
    for filepath in tqdm( Path_utils.get_image_paths(input_path), desc="Loading"):
        filepath = Path(filepath)
        
        if filepath.suffix != '.png':
            print ("%s is not a png file required for sort_by_face" % (filepath.name) ) 
            continue
        
        a_png = AlignedPNG.load (str(filepath))
        if a_png is None:
            print ("%s failed to load" % (filepath.name) ) 
            continue
            
        d = a_png.getFaceswapDictData()
        
        if d is None or d['landmarks'] is None:          
            print ("%s - no embedded data found required for sort_by_face" % (filepath.name) )
            continue
        
        img_list.append( [str(filepath), np.array(d['landmarks']) ] )
        

    img_list_len = len(img_list)
    for i in tqdm ( range(0, img_list_len-1), desc="Sorting"):
        min_score = float("inf")
        j_min_score = i+1
        for j in range(i+1,len(img_list)):

            fl1 = img_list[i][1]
            fl2 = img_list[j][1]
            score = np.sum ( np.absolute ( (fl2 - fl1).flatten() ) )

            if score < min_score:
                min_score = score
                j_min_score = j
        img_list[i+1], img_list[j_min_score] = img_list[j_min_score], img_list[i+1]

    return img_list

def sort_by_face_dissim(input_path):

    print ("Sorting by face dissimilarity...")

    img_list = []
    for filepath in tqdm( Path_utils.get_image_paths(input_path), desc="Loading"):
        filepath = Path(filepath)
        
        if filepath.suffix != '.png':
            print ("%s is not a png file required for sort_by_face_dissim" % (filepath.name) ) 
            continue
        
        a_png = AlignedPNG.load (str(filepath))
        if a_png is None:
            print ("%s failed to load" % (filepath.name) ) 
            continue
            
        d = a_png.getFaceswapDictData()
        
        if d is None or d['landmarks'] is None:          
            print ("%s - no embedded data found required for sort_by_face_dissim" % (filepath.name) )
            continue
        
        img_list.append( [str(filepath), np.array(d['landmarks']), 0 ] )
        
    img_list_len = len(img_list)
    for i in tqdm( range(0, img_list_len-1), desc="Sorting"):
        score_total = 0
        for j in range(i+1,len(img_list)):
            if i == j:
                continue
            fl1 = img_list[i][1]
            fl2 = img_list[j][1]
            score_total += np.sum ( np.absolute ( (fl2 - fl1).flatten() ) )

        img_list[i][2] = score_total

    print ("Sorting...")
    img_list = sorted(img_list, key=operator.itemgetter(2), reverse=True)

    return img_list
    
def sort_by_face_yaw(input_path):
    print ("Sorting by face yaw...")
    img_list = []
    for filepath in tqdm( Path_utils.get_image_paths(input_path), desc="Loading"):
        filepath = Path(filepath)
        
        if filepath.suffix != '.png':
            print ("%s is not a png file required for sort_by_face_dissim" % (filepath.name) ) 
            continue
        
        a_png = AlignedPNG.load (str(filepath))
        if a_png is None:
            print ("%s failed to load" % (filepath.name) ) 
            continue
            
        d = a_png.getFaceswapDictData()
        
        if d is None or d['yaw_value'] is None:          
            print ("%s - no embedded data found required for sort_by_face_dissim" % (filepath.name) )
            continue
        
        img_list.append( [str(filepath), np.array(d['yaw_value']) ] )

    print ("Sorting...")
    img_list = sorted(img_list, key=operator.itemgetter(1), reverse=True)
    
    return img_list
    
def sort_by_hist_blur(input_path):

    print ("Sorting by histogram similarity and blur...")

    img_list = []
    for x in tqdm( Path_utils.get_image_paths(input_path), desc="Loading"):
        img = cv2.imread(x)    
        img_list.append ([x, cv2.calcHist([img], [0], None, [256], [0, 256]),
                             cv2.calcHist([img], [1], None, [256], [0, 256]),
                             cv2.calcHist([img], [2], None, [256], [0, 256]),
                             estimate_blur(img)
                         ])

    img_list_len = len(img_list)
    for i in tqdm( range(0, img_list_len-1), desc="Sorting"):
        min_score = float("inf")
        j_min_score = i+1
        for j in range(i+1,len(img_list)):
            score = cv2.compareHist(img_list[i][1], img_list[j][1], cv2.HISTCMP_BHATTACHARYYA) + \
                    cv2.compareHist(img_list[i][2], img_list[j][2], cv2.HISTCMP_BHATTACHARYYA) + \
                    cv2.compareHist(img_list[i][3], img_list[j][3], cv2.HISTCMP_BHATTACHARYYA)
            if score < min_score:
                min_score = score
                j_min_score = j
        img_list[i+1], img_list[j_min_score] = img_list[j_min_score], img_list[i+1]
     
    l = []
    for i in range(0, img_list_len-1):
        score = cv2.compareHist(img_list[i][1], img_list[i+1][1], cv2.HISTCMP_BHATTACHARYYA) + \
                cv2.compareHist(img_list[i][2], img_list[i+1][2], cv2.HISTCMP_BHATTACHARYYA) + \
                cv2.compareHist(img_list[i][3], img_list[i+1][3], cv2.HISTCMP_BHATTACHARYYA)
        l += [score]
    l = np.array(l)
    v = np.mean(l)
    if v*2 < np.max(l):
        v *= 2
    
    new_img_list = []
        
    start_group_i = 0
    odd_counter = 0
    for i in tqdm( range(0, img_list_len), desc="Sorting"):
        end_group_i = -1
        if i < img_list_len-1:
            score = cv2.compareHist(img_list[i][1], img_list[i+1][1], cv2.HISTCMP_BHATTACHARYYA) + \
                    cv2.compareHist(img_list[i][2], img_list[i+1][2], cv2.HISTCMP_BHATTACHARYYA) + \
                    cv2.compareHist(img_list[i][3], img_list[i+1][3], cv2.HISTCMP_BHATTACHARYYA)
                         
            if score >= v:
                end_group_i = i
                
        elif i == img_list_len-1:
            end_group_i = i
    
        if end_group_i >= start_group_i:
            odd_counter += 1
            
            s = sorted(img_list[start_group_i:end_group_i+1] , key=operator.itemgetter(4), reverse=True)         
            if odd_counter % 2 == 0:            
                new_img_list = new_img_list + s
            else:
                new_img_list = s + new_img_list
                
            start_group_i = i + 1

    return new_img_list
    
def sort_by_hist(input_path):

    print ("Sorting by histogram similarity...")

    img_list = []
    for x in tqdm( Path_utils.get_image_paths(input_path), desc="Loading"):
        img = cv2.imread(x)    
        img_list.append ([x, cv2.calcHist([img], [0], None, [256], [0, 256]),
                             cv2.calcHist([img], [1], None, [256], [0, 256]),
                             cv2.calcHist([img], [2], None, [256], [0, 256])
                         ])

    img_list_len = len(img_list)
    for i in tqdm( range(0, img_list_len-1), desc="Sorting"):
        min_score = float("inf")
        j_min_score = i+1
        for j in range(i+1,len(img_list)):
            score = cv2.compareHist(img_list[i][1], img_list[j][1], cv2.HISTCMP_BHATTACHARYYA) + \
                    cv2.compareHist(img_list[i][2], img_list[j][2], cv2.HISTCMP_BHATTACHARYYA) + \
                    cv2.compareHist(img_list[i][3], img_list[j][3], cv2.HISTCMP_BHATTACHARYYA)
            if score < min_score:
                min_score = score
                j_min_score = j
        img_list[i+1], img_list[j_min_score] = img_list[j_min_score], img_list[i+1]

    return img_list

def sort_by_hist_dissim(input_path):

    print ("Sorting by histogram dissimilarity...")

    img_list = []
    for x in tqdm( Path_utils.get_image_paths(input_path), desc="Loading"):
        img = cv2.imread(x)    
        img_list.append ([x, cv2.calcHist([img], [0], None, [256], [0, 256]),
                             cv2.calcHist([img], [1], None, [256], [0, 256]),
                             cv2.calcHist([img], [2], None, [256], [0, 256]), 0
                         ])

    img_list_len = len(img_list)
    for i in tqdm ( range(0, img_list_len), desc="Sorting"):
        score_total = 0
        for j in range( 0, img_list_len):
            if i == j:
                continue
            score_total += cv2.compareHist(img_list[i][1], img_list[j][1], cv2.HISTCMP_BHATTACHARYYA) + \
                           cv2.compareHist(img_list[i][2], img_list[j][2], cv2.HISTCMP_BHATTACHARYYA) + \
                           cv2.compareHist(img_list[i][3], img_list[j][3], cv2.HISTCMP_BHATTACHARYYA)

        img_list[i][4] = score_total


    print ("Sorting...")
    img_list = sorted(img_list, key=operator.itemgetter(4), reverse=True)

    return img_list
            
def final_rename(input_path, img_list):
    for i in tqdm( range(0,len(img_list)), desc="Renaming" , leave=False):
        src = Path (img_list[i][0])        
        dst = input_path / ('%.5d_%s' % (i, src.name ))
        try:
            src.rename (dst)
        except:
            print ('fail to rename %s' % (src.name) )    
            
    for i in tqdm( range(0,len(img_list)) , desc="Renaming" ):
        src = Path (img_list[i][0])
        
        src = input_path / ('%.5d_%s' % (i, src.name))
        dst = input_path / ('%.5d%s' % (i, src.suffix))
        try:
            src.rename (dst)
        except:
            print ('fail to rename %s' % (src.name) )    

def sort_by_origname(input_path):
    print ("Sort by original filename...")
    
    img_list = []
    for filepath in tqdm( Path_utils.get_image_paths(input_path), desc="Loading"):
        filepath = Path(filepath)
        
        if filepath.suffix != '.png':
            print ("%s is not a png file required for sort_by_origname" % (filepath.name) ) 
            continue
        
        a_png = AlignedPNG.load (str(filepath))
        if a_png is None:
            print ("%s failed to load" % (filepath.name) ) 
            continue
            
        d = a_png.getFaceswapDictData()
        
        if d is None or d['source_filename'] is None:          
            print ("%s - no embedded data found required for sort_by_origname" % (filepath.name) )
            continue

        img_list.append( [str(filepath), d['source_filename']] )

    print ("Sorting...")
    img_list = sorted(img_list, key=operator.itemgetter(1))
    return img_list
    
def main (input_path, sort_by_method):
    input_path = Path(input_path)
    sort_by_method = sort_by_method.lower()

    print ("Running sort tool.\r\n")
    
    img_list = []

    if sort_by_method == 'blur':            img_list = sort_by_blur (input_path)
    elif sort_by_method == 'face':          img_list = sort_by_face (input_path)
    elif sort_by_method == 'face-dissim':   img_list = sort_by_face_dissim (input_path)
    elif sort_by_method == 'face-yaw':      img_list = sort_by_face_yaw (input_path)
    elif sort_by_method == 'hist':          img_list = sort_by_hist (input_path)
    elif sort_by_method == 'hist-dissim':   img_list = sort_by_hist_dissim (input_path)
    elif sort_by_method == 'hist-blur':     img_list = sort_by_hist_blur (input_path)
    elif sort_by_method == 'brightness':    img_list = sort_by_brightness (input_path)
    elif sort_by_method == 'hue':           img_list = sort_by_hue (input_path)
    elif sort_by_method == 'origname':      img_list = sort_by_origname (input_path)       
    
    final_rename (input_path, img_list)