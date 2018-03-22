import os
import sys
import FaceLandmarksExtractor
from tqdm import tqdm
import itertools
import operator
import cv2
import numpy as np
from Serializer import JSONSerializer

def find_images(input_dir):
    result = []
    extensions = [".jpg", ".png", ".jpeg"]
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in extensions:
                result.append (os.path.join(root, file))
    return result
    
def calc_landmarks_face_yaw(fl):
        l = ( (fl[27][0]-fl[0][0])  + (fl[28][0]-fl[1][0])  + (fl[29][0]-fl[2][0])  ) / 3.0   
        r = ( (fl[16][0]-fl[27][0]) + (fl[15][0]-fl[28][0]) + (fl[14][0]-fl[29][0]) ) / 3.0
        return r-l

def calc_landmarks_face_pitch(fl):
    t = ( (fl[6][1]-fl[8][1]) + (fl[10][1]-fl[8][1]) ) / 2.0   
    b = fl[8][1]
    return b-t

def main( input_dir, output_path ):
    images = find_images(input_dir)
    
    img_list = []
    for x in tqdm( find_images(input_dir), desc="Loading", file=sys.stdout ):
        d = FaceLandmarksExtractor.extract(cv2.imread(x), 'cnn', True, input_is_predetected_face=True)
        fl = np.array(d[0][1])
        img_list.append( [x, fl, calc_landmarks_face_pitch(fl), calc_landmarks_face_yaw(fl) ] )
    
    lowest_yaw, highest_yaw = float("+inf"), float("-inf")
    for sample in img_list:
        sample_yaw = sample[3]
        lowest_yaw = min (lowest_yaw, sample_yaw)
        highest_yaw = max (highest_yaw, sample_yaw)     
        
    gradations = 180
    diff_rot_per_grad = abs(highest_yaw-lowest_yaw) / gradations

    yaws_sample_list = []
    for i in tqdm( range(0, gradations), desc="Regularizing", file=sys.stdout):
        yaw = lowest_yaw + i*diff_rot_per_grad
        next_yaw = lowest_yaw + (i+1)*diff_rot_per_grad

        yaw_samples = []        
        for sample in img_list:
            sample_yaw = sample[3]
        
            if (i != gradations-1 and sample_yaw >= yaw and sample_yaw < next_yaw) or \
               (i == gradations-1 and sample_yaw >= yaw):
                yaw_samples.append (sample[0])
                
        if len(yaw_samples) > 0:
            yaws_sample_list.append (yaw_samples)
            
    with open(output_path, "w") as fh:
        fh.write( JSONSerializer.marshal(yaws_sample_list) )    

if __name__ == "__main__":
    main (sys.argv[1], sys.argv[2])
