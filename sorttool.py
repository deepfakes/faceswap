import argparse
import os
import sys
import operator
import numpy as np
import cv2
from tqdm import tqdm
import face_recognition

if sys.version_info[0] < 3:
    raise Exception("This program requires at least python3.2")
if sys.version_info[0] == 3 and sys.version_info[1] < 2:
    raise Exception("This program requires at least python3.2")

class SortProcessor(object):

    def __init__(self, parser):
        self.init_parser_arguments(parser)
        
    def process_arguments(self, arguments):
        self.arguments = arguments
        self.process()

    def init_parser_arguments(self, parser):
        parser.add_argument('-i', '--input',
                            dest="input_dir",
                            default="input_dir",
                            help="Input directory of aligned faces.",
                            required=True)
                             
        parser.add_argument('-by', '--by',
                            type=str,
                            choices=("blur", "hist", "face"),
                            dest='method',
                            default="hist",
                            help="Sort by method.")

    def process(self):        
        if self.arguments.method.lower() == 'blur':
            self.process_blur()
        elif self.arguments.method.lower() == 'hist':
            self.process_hist()
        elif self.arguments.method.lower() == 'face':
            self.process_face()
            
    def process_blur(self):
        input_dir = self.arguments.input_dir
        
        print ("Sorting by blur...")         
        img_list = [ [x, self.estimate_blur(cv2.imread(x))] for x in tqdm(self.find_images(input_dir), desc="Loading") ]
        print ("Sorting...")    
        img_list = sorted(img_list, key=operator.itemgetter(1), reverse=True) 
        self.process_final_rename(input_dir, img_list)        
        print ("Done.")
  
    def process_hist(self):
        input_dir = self.arguments.input_dir
        
        print ("Sorting by histogram similarity...")
        
        img_list = [ [x, cv2.calcHist([cv2.imread(x)], [0], None, [256], [0, 256]) ] for x in tqdm( self.find_images(input_dir), desc="Loading") ]

        img_list_len = len(img_list)
        for i in tqdm ( range(0, img_list_len-1), desc="Sorting"):
            min_score = 9999.9
            j_min_score = i+1
            for j in range(i+1,len(img_list)):
                score = cv2.compareHist(img_list[i][1], img_list[j][1], cv2.HISTCMP_BHATTACHARYYA)
                if score < min_score:
                    min_score = score
                    j_min_score = j            
            img_list[i+1], img_list[j_min_score] = img_list[j_min_score], img_list[i+1]
            
        self.process_final_rename (input_dir, img_list)
                
        print ("Done.")
        
    def process_face(self):
        input_dir = self.arguments.input_dir
        
        print ("Sorting by face similarity...")
        
        img_list = [ [x, face_recognition.face_encodings(cv2.imread(x)) ] for x in tqdm( self.find_images(input_dir), desc="Loading") ]

        img_list_len = len(img_list)
        for i in tqdm ( range(0, img_list_len-1), desc="Sorting"):
            min_score = 9999.9
            j_min_score = i+1
            for j in range(i+1,len(img_list)):
            
                f1encs = img_list[i][1]
                f2encs = img_list[j][1]
                if f1encs is not None and f2encs is not None and len(f1encs) > 0 and len(f2encs) > 0:
                    score = face_recognition.face_distance(f1encs[0], f2encs)[0]
                else: 
                    score = 9999.9
                
                if score < min_score:
                    min_score = score
                    j_min_score = j            
            img_list[i+1], img_list[j_min_score] = img_list[j_min_score], img_list[i+1]
            
        self.process_final_rename (input_dir, img_list)
                
        print ("Done.")
        
    def process_final_rename(self, input_dir, img_list):
        for i in tqdm( range(0,len(img_list)), desc="Renaming" , leave=False):
            src = img_list[i][0]
            src_basename = os.path.basename(src)       

            dst = os.path.join (input_dir, '%.5d_%s' % (i, src_basename ) )
            try:
                os.rename (src, dst)
            except:
                print ('fail to rename %s' % (src) )    
                
        for i in tqdm( range(0,len(img_list)) , desc="Renaming" ):
            src = img_list[i][0]
            src_basename = os.path.basename(src)
            
            src = os.path.join (input_dir, '%.5d_%s' % (i, src_basename) )
            dst = os.path.join (input_dir, '%.5d%s' % (i, os.path.splitext(src_basename)[1] ) )
            try:
                os.rename (src, dst)
            except:
                print ('fail to rename %s' % (src) )
                
    def find_images(self, input_dir):
        result = []
        extensions = [".jpg", ".png", ".jpeg"]
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in extensions:
                    result.append (os.path.join(root, file))
        return result

    def estimate_blur(self, image):
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blur_map = cv2.Laplacian(image, cv2.CV_64F)
        score = np.var(blur_map)
        return score
        
    def error(self, message):
        self.print_help(sys.stderr)
        args = {'prog': self.prog, 'message': message}
        self.exit(2, '%(prog)s: error: %(message)s\n' % args)    

def bad_args(args):
    parser.print_help()
    exit(0)

if __name__ == "__main__":
    print ("Images sort tool.\n")
    
    parser = argparse.ArgumentParser()
    parser.set_defaults(func=bad_args)
    
    sort = SortProcessor(parser)    
    sort.process_arguments(parser.parse_args())