#!/usr/bin/env python3
import argparse
import os
import sys
import operator
import numpy as np
import cv2
from tqdm import tqdm
import face_recognition
from shutil import copyfile
import json


if sys.version_info[0] < 3:
    raise Exception("This program requires at least python3.2")
if sys.version_info[0] == 3 and sys.version_info[1] < 2:
    raise Exception("This program requires at least python3.2")


class SortProcessor(object):
    def __init__(self, subparser, command, description='default'):
        self.parse_arguments(description, subparser, command)

    def parse_arguments(self, description, subparser, command):
        parser = subparser.add_parser(
                command,
                help="This command lets you sort images using various methods.",
                description=description,
                epilog="Questions and feedback: \
                        https://github.com/deepfakes/faceswap-playground"
        )

        parser.add_argument('-i', '--input',
                            dest="input_dir",
                            default="input_dir",
                            help="Input directory of aligned faces.",
                            required=True)

        parser.add_argument('-o', '--output',
                            dest="output_dir",
                            default="__default",
                            help="Output directory for sorted aligned faces.")

        parser.add_argument('-g', '--grouping',
                            type=str,
                            choices=("folders", "rename"),
                            dest='grouping',
                            default="rename",
                            help="'folders' sorts input the files them into\
                                  folders.\
                                  'renaming' renames the input files to group\
                                  them.\
                                  Default: rename")

        parser.add_argument('-t', '--ref_threshold',
                            type=float,
                            dest='min_threshold',
                            default=-1.0,
                            help="Float value.\
                                  Minimum threshold to use for comparison with\
                                  'face' and 'hist' methods. The lower the \
                                  value the more discriminating the sorting \
                                  is. \
                                  For face 0.6 should be enough, with 0.5\
                                  being very discriminating. \
                                  For face-cnn 17 should be enough, with 12\
                                  being very discriminating.\
                                  For hist 0.3 should be enough, with 0.2\
                                  being very discriminating.\
                                  Be careful setting a value that's too \
                                  low in a directory with many images, as this\
                                  could result in a lot of directories being \
                                  created.\
                                  Defaults: face 0.6, face-cnn 17, hist 0.3")

        parser.add_argument('-b', '--bins',
                            type=int,
                            dest='num_bins',
                            default=5,
                            help="Integer value.\
                                  Number of folders that will be used to group\
                                  by blur. Folder 0 will be the least blurry,\
                                  while the last folder will be the blurriest.\
                                  If the number of images doesn't divide\
                                  evenly into the number of bins, the\
                                  remaining images get put in the last bin as\
                                  they will be the blurriest by definition.\
                                  Default value: 5")

        parser.add_argument('-k', '--keep',
                            action='store_true',
                            dest='keep_original',
                            default=False,
                            help="Keeps the original files in the input\
                                  directory. Be careful when using this with\
                                  rename grouping and no specified output\
                                  directory as this would keep the original\
                                  and renamed files in the same directory.")

        parser.add_argument('-l', '--log-changes',
                            action='store_true',
                            dest='log_changes',
                            default=False,
                            help="Logs file renaming changes if grouping by\
                                  renaming, or it logs the file\
                                  copying/movement if grouping by folders.\
                                  If no log file is specified with \
                                  '--log-file', then a 'sort_log.json' file \
                                  will be created in the input directory.")

        parser.add_argument('--log-file',
                            dest='log_file',
                            default='__default',
                            help="Specify a log file to use for saving the\
                                  renaming or grouping information.\
                                  Default: sort_log.json")

        parser.add_argument('-by', '--by',
                            type=str,
                            choices=("blur", "face", "face-cnn",
                                     "face-cnn-dissim", "face-dissim", "hist",
                                     "hist-dissim"),
                            dest='method',
                            default="hist",
                            help="Sort by method.\
                                  When grouping by folders face-cnn-dissim and\
                                  face-dissim default to face, \
                                  and hist-dissim defaults to hist.\
                                  Default: hist")
        parser = self.add_optional_arguments(parser)
        parser.set_defaults(func=self.process_arguments)

    def add_optional_arguments(self, parser):
        # Override this for custom arguments
        return parser

    def process_arguments(self, arguments):
        self.arguments = arguments

        # Setting default argument values that cannot be set by argparse

        # Set output dir to the same value as input dir
        # if the user didn't specify it.
        if self.arguments.output_dir.lower() == "__default":
            self.arguments.output_dir = self.arguments.input_dir

        # Assigning default threshold values based on grouping method
        if self.arguments.min_threshold == -1.0 and self.arguments.grouping.lower() == "folders":
            if self.arguments.method.lower() == 'face':
                self.arguments.min_threshold = 0.6
            elif self.arguments.method.lower() == 'face-cnn':
                self.arguments.min_threshold = 17
            elif self.arguments.method.lower() == 'hist':
                self.arguments.min_threshold = 0.3

        # Dissimilarity methods make no sense when grouping by folders
        if self.arguments.grouping.lower() == 'folders':
            if self.arguments.method.lower() == 'face-dissim':
                self.arguments.method = 'face'
            elif self.arguments.method.lower() == 'face-cnn-dissim':
                self.arguments.method = 'face'
            elif self.arguments.method.lower() == 'hist-dissim':
                self.arguments.method = 'hist'

        # Assign default sort_log.json value if user didn't specify one
        if self.arguments.log_file.lower() == '__default':
            self.arguments.log_file = os.path.join(self.arguments.input_dir, 'sort_log.json')

        self.process()

    def process(self):
        if self.arguments.grouping.lower() == 'folders':
            if self.arguments.method.lower() == 'blur':
                self.process_blur_folders()
            elif self.arguments.method.lower() == 'face':
                self.process_face_folders()
            elif self.arguments.method.lower() == 'face-cnn':
                self.process_face_cnn_folders()
            elif self.arguments.method.lower() == 'hist':
                self.process_hist_folders()

        elif self.arguments.grouping.lower() == 'rename':
            if self.arguments.method.lower() == 'blur':
                self.process_blur()
            elif self.arguments.method.lower() == 'face':
                self.process_face()
            elif self.arguments.method.lower() == 'face-dissim':
                self.process_face_dissim()
            elif self.arguments.method.lower() == 'face-cnn':
                self.process_face_cnn()
            elif self.arguments.method.lower() == 'face-cnn-dissim':
                self.process_face_cnn_dissim()
            elif self.arguments.method.lower() == 'hist':
                self.process_hist()
            elif self.arguments.method.lower() == 'hist-dissim':
                self.process_hist_dissim()

    # Methods for grouping by renaming
    def process_blur(self):
        input_dir = self.arguments.input_dir
        output_dir = self.arguments.output_dir
        
        print ("Sorting by blur...")         
        img_list = [ [x, self.estimate_blur(cv2.imread(x))] for x in tqdm(self.find_images(input_dir), desc="Loading") ]
        print ("Sorting...")    
        img_list = sorted(img_list, key=operator.itemgetter(1), reverse=True)
        self.process_final_rename(output_dir, img_list)
        print ("Done.")

    def process_face(self):
        input_dir = self.arguments.input_dir
        output_dir = self.arguments.output_dir
        
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
            
        self.process_final_rename (output_dir, img_list)
                
        print ("Done.")

    def process_face_dissim(self):
        input_dir = self.arguments.input_dir
        output_dir = self.arguments.output_dir

        print ("Sorting by face dissimilarity...")

        img_list = [ [x, face_recognition.face_encodings(cv2.imread(x)), 0 ] for x in tqdm( self.find_images(input_dir), desc="Loading") ]

        img_list_len = len(img_list)
        for i in tqdm ( range(0, img_list_len), desc="Sorting"):
            score_total = 0
            for j in range( 0, img_list_len):
                if i == j:
                    continue
                try:
                    score_total += face_recognition.face_distance([img_list[i][1]], [img_list[j][1]])
                except:
                    pass

            img_list[i][2] = score_total


        print ("Sorting...")
        img_list = sorted(img_list, key=operator.itemgetter(2), reverse=True)
        self.process_final_rename (output_dir, img_list)

        print ("Done.")

    def process_face_cnn(self):
        from lib import FaceLandmarksExtractor

        input_dir = self.arguments.input_dir
        output_dir = self.arguments.output_dir

        print ("Sorting by face-cnn similarity...")

        img_list = []
        for x in tqdm( self.find_images(input_dir), desc="Loading"):
            d = FaceLandmarksExtractor.extract(cv2.imread(x), 'cnn', True)
            img_list.append( [x, np.array(d[0][1]) if len(d) > 0 else np.zeros ( (68,2) ) ] )

        img_list_len = len(img_list)
        for i in tqdm ( range(0, img_list_len-1), desc="Sorting"):
            min_score = 9999999
            j_min_score = i+1
            for j in range(i+1,len(img_list)):

                fl1 = img_list[i][1]
                fl2 = img_list[j][1]
                score = np.sum ( np.absolute ( (fl2 - fl1).flatten() ) )

                if score < min_score:
                    min_score = score
                    j_min_score = j
            img_list[i+1], img_list[j_min_score] = img_list[j_min_score], img_list[i+1]

        self.process_final_rename (output_dir, img_list)

        print ("Done.")

    def process_face_cnn_dissim(self):
        from lib import FaceLandmarksExtractor

        input_dir = self.arguments.input_dir
        output_dir = self.arguments.output_dir

        print ("Sorting by face-cnn dissimilarity...")

        img_list = []
        for x in tqdm( self.find_images(input_dir), desc="Loading"):
            d = FaceLandmarksExtractor.extract(cv2.imread(x), 'cnn', True)
            img_list.append( [x, np.array(d[0][1]) if len(d) > 0 else np.zeros ( (68,2) ), 0 ] )

        img_list_len = len(img_list)
        for i in tqdm ( range(0, img_list_len-1), desc="Sorting"):
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
        self.process_final_rename (output_dir, img_list)

        print ("Done.")

    def process_hist(self):
        input_dir = self.arguments.input_dir
        output_dir = self.arguments.output_dir

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

        self.process_final_rename (output_dir, img_list)

        print ("Done.")

    def process_hist_dissim(self):
        input_dir = self.arguments.input_dir
        output_dir = self.arguments.output_dir

        print ("Sorting by histogram dissimilarity...")

        img_list = [ [x, cv2.calcHist([cv2.imread(x)], [0], None, [256], [0, 256]), 0] for x in tqdm( self.find_images(input_dir), desc="Loading") ]

        img_list_len = len(img_list)
        for i in tqdm ( range(0, img_list_len), desc="Sorting"):
            score_total = 0
            for j in range( 0, img_list_len):
                if i == j:
                    continue
                score_total += cv2.compareHist(img_list[i][1], img_list[j][1], cv2.HISTCMP_BHATTACHARYYA)

            img_list[i][2] = score_total


        print ("Sorting...")
        img_list = sorted(img_list, key=operator.itemgetter(2), reverse=True)
        self.process_final_rename (output_dir, img_list)

        print ("Done.")

    def process_final_rename(self, output_dir, img_list):
        process_file = self.set_process_file_method(self.arguments.log_changes, self.arguments.keep_original)

        if self.arguments.log_changes:
            changes = dict()
        else:
            changes = None

        for i in tqdm(range(0, len(img_list)), desc="Renaming", leave=False):
            src = img_list[i][0]
            src_basename = os.path.basename(src)       

            dst = os.path.join (output_dir, '%.5d_%s' % (i, src_basename ) )
            try:
                process_file (src, dst, changes)
            except:
                print ('fail to rename %s' % (src) )    
                
        for i in tqdm( range(0,len(img_list)) , desc="Renaming" ):
            renaming = self.set_renaming_method(self.arguments.log_changes)
            src, dst = renaming(img_list[i][0], output_dir, i, changes)

            try:
                os.rename (src, dst)
            except:
                print ('fail to rename %s' % (src) )

        if self.arguments.log_changes:
            self.write_to_log(self.arguments.log_file, changes)

    # Methods for grouping by folders
    def process_blur_folders(self):
        input_dir = self.arguments.input_dir
        output_dir = self.arguments.output_dir

        print ("Sorting by blur...")
        img_list = [ [x, self.estimate_blur(cv2.imread(x))] for x in tqdm(self.find_images(input_dir), desc="Loading") ]
        print ("Sorting...")
        img_list = sorted(img_list, key=operator.itemgetter(1), reverse=True)

        # Starting the binning process
        num_bins = self.arguments.num_bins

        # The last bin will get all extra images if it's
        # not possible to distribute them evenly
        num_per_bin = len(img_list) // num_bins
        remainder = len(img_list) % num_bins

        print ("Sorting into bins...")
        bins = [ [] for _ in range(num_bins) ]
        image_index = 0
        for i in range(num_bins):
            for j in range(num_per_bin):
                bins[i].append(img_list[image_index][0])
                image_index += 1

        # If remainder is 0, nothing gets added to the last bin.
        for i in range(1, remainder+1):
            bins[-1].append(img_list[-i][0])

        self.process_final_folders (output_dir, bins)
        print ("Done.")

    def process_face_folders(self):
        input_dir = self.arguments.input_dir
        output_dir = self.arguments.output_dir

        print ("Grouping by face similarity...")

        # Groups are of the form: group_num -> reference face
        reference_groups = dict()

        # Bins array, where index is the group number and value is
        # an array containing the file paths to the images in that group.
        # The first group (0), is always the non-face group.
        bins = [[]]

        # Comparison threshold used to decide how similar
        # faces have to be to be grouped together.
        min_threshold = self.arguments.min_threshold

        img_list = [ [x, face_recognition.face_encodings(cv2.imread(x)) ] for x in tqdm( self.find_images(input_dir), desc="Loading") ]
        img_list_len = len(img_list)
        reference_groups[0] = []

        for i in tqdm(range(img_list_len), desc="Grouping"):
            f1encs = img_list[i][1]

            # Check if current image is a face, if not then
            # add it immediately to the non-face list.
            if f1encs is None or len(f1encs) <= 0:
                bins[0].append(img_list[i][0])

            else:
                current_best = [-1, float("inf")]

                for key, value in reference_groups.items():
                    # Non-faces are not added to reference_groups dict, thus
                    # removing the need to check that f2encs is a face.
                    # The try-catch block is to handle the first face that gets
                    # processed, as the first value is None.
                    try:
                        score = self.get_avg_score_faces(f1encs, value)
                    except TypeError:
                        score = float("inf")
                    except ZeroDivisionError:
                        score = float("inf")
                    if score < current_best[1]:
                        current_best[0], current_best[1] = key, score

                if current_best[1] < min_threshold:
                    reference_groups[current_best[0]].append(f1encs[0])
                    bins[current_best[0]].append(img_list[i][0])
                else:
                    reference_groups[len(reference_groups)] = img_list[i][1]
                    bins.append([img_list[i][0]])

        self.process_final_folders (output_dir, bins)
        print ("Done.")

    def process_face_cnn_folders(self):
        from lib import FaceLandmarksExtractor

        input_dir = self.arguments.input_dir
        output_dir = self.arguments.output_dir

        print ("Grouping by face-cnn similarity...")

        # Groups are of the form: group_num -> reference face
        reference_groups = dict()

        # Bins array, where index is the group number and value is
        # an array containing the file paths to the images in that group.
        # The first group (0), is always the non-face group.
        bins = [[]]

        # Comparison threshold used to decide how similar
        # faces have to be to be grouped together.
        min_threshold = self.arguments.min_threshold

        img_list = []
        for x in tqdm( self.find_images(input_dir), desc="Loading"):
            d = FaceLandmarksExtractor.extract(cv2.imread(x), 'cnn', True)
            img_list.append( [x, np.array(d[0][1]) if len(d) > 0 else np.zeros ( (68,2) ) ] )

        img_list_len = len(img_list)
        reference_groups[0] = []

        for i in tqdm ( range(0, img_list_len-1), desc="Grouping"):
            fl1 = img_list[i][1]

            current_best = [-1, float("inf")]

            for key, references in reference_groups.items():
                try:
                    score = self.get_avg_score_faces(fl1, references)
                except TypeError:
                    score = float("inf")
                except ZeroDivisionError:
                    score = float("inf")
                if score < current_best[1]:
                    current_best[0], current_best[1] = key, score

            if current_best[1] < min_threshold:
                reference_groups[current_best[0]].append(fl1[0])
                bins[current_best[0]].append(img_list[i][0])
            else:
                reference_groups[len(reference_groups)] = [img_list[i][1]]
                bins.append([img_list[i][0]])

        self.process_final_folders (output_dir, bins)
        print ("Done.")

    def process_hist_folders(self):
        input_dir = self.arguments.input_dir
        output_dir = self.arguments.output_dir

        print ("Grouping by histogram...")

        # Groups are of the form: group_num -> reference histogram
        reference_groups = dict()

        # Bins array, where index is the group number and value is
        # an array containing the file paths to the images in that group
        bins = []

        min_threshold = self.arguments.min_threshold

        img_list = [ [x, cv2.calcHist([cv2.imread(x)], [0], None, [256], [0, 256]) ] for x in tqdm( self.find_images(input_dir), desc="Loading") ]
        img_list_len = len(img_list)
        reference_groups[0] = [img_list[0][1]]
        bins.append([img_list[0][0]])

        for i in tqdm(range(1, img_list_len), desc="Grouping"):
            current_best = [-1, 9999.9]
            for key, value in reference_groups.items():
                score = self.get_avg_score_hist(img_list[i][1], value)
                if score < current_best[1]:
                    current_best[0], current_best[1] = key, score

            if current_best[1] < min_threshold:
                reference_groups[current_best[0]].append(img_list[i][1])
                bins[current_best[0]].append(img_list[i][0])
            else:
                reference_groups[len(reference_groups)] = [img_list[i][1]]
                bins.append([img_list[i][0]])

        self.process_final_folders (output_dir, bins)

        print("Done.")

    def process_final_folders(self, output_dir, bins):
        process_file = self.set_process_file_method(self.arguments.log_changes, self.arguments.keep_original)

        if self.arguments.log_changes:
            changes = dict()
        else:
            changes = None

        # First create new directories to avoid checking
        # for directory existence in the moving loop
        print ("Creating group directories.")
        for i in range(len(bins)):
            directory = os.path.join (output_dir, str(i))
            if not os.path.exists (directory):
                os.makedirs (directory)

        print ("Total groups found: {}".format(len(bins)))
        for i in tqdm(range(len(bins)), desc="Moving"):
            for j in range(len(bins[i])):
                src = bins[i][j]
                src_basename = os.path.basename (src)

                dst = os.path.join (output_dir, str(i), src_basename)
                try:
                    process_file (src, dst, changes)
                except FileNotFoundError as e:
                    print (e)
                    print ('Failed to move {0} to {1}'.format(src, dst))
                except:
                    print ('Some other error occurred.')
                    print ('Failed to move {0} to {1}'.format(src, dst))

        if self.arguments.log_changes:
            self.write_to_log(self.arguments.log_file, changes)

    # Various helper methods
    @staticmethod
    def find_images(input_dir):
        result = []
        extensions = [".jpg", ".png", ".jpeg"]
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in extensions:
                    result.append (os.path.join(root, file))
        return result

    @staticmethod
    def estimate_blur(image):
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blur_map = cv2.Laplacian(image, cv2.CV_64F)
        score = np.var(blur_map)
        return score

    @staticmethod
    def set_process_file_method(log_changes, keep_original):
        """
        Assigns the final file processing method based on whether changes are
        being logged and whether the original files are being kept in the
        input directory.
        Relevant cli arguments: -k, -l
        :return: function reference
        """
        if log_changes:
            if keep_original:
                def process_file(src, dst, changes):
                    copyfile(src, dst)
                    changes[src] = dst
                return process_file
            else:
                def process_file(src, dst, changes):
                    os.rename(src, dst)
                    changes[src] = dst
                return process_file
        else:
            if keep_original:
                def process_file(src, dst, changes):
                    copyfile(src, dst)
                return process_file
            else:
                def process_file(src, dst, changes):
                    os.rename(src, dst)
                return process_file

    @staticmethod
    def set_renaming_method(log_changes):
        if log_changes:
            def renaming(src, output_dir, i, changes):
                src_basename = os.path.basename(src)

                __src = os.path.join (output_dir, '%.5d_%s' % (i, src_basename) )
                dst = os.path.join (output_dir, '%.5d%s' % (i, os.path.splitext(src_basename)[1] ) )
                changes[src] = dst
                return __src, dst
            return renaming

        else:
            def renaming(src, output_dir, i, changes):
                src_basename = os.path.basename(src)

                src = os.path.join (output_dir, '%.5d_%s' % (i, src_basename) )
                dst = os.path.join (output_dir, '%.5d%s' % (i, os.path.splitext(src_basename)[1] ) )
                return src, dst
            return renaming

    @staticmethod
    def get_avg_score_hist(img1, references):
        scores = []
        for img2 in references:
            score = cv2.compareHist(img1, img2, cv2.HISTCMP_BHATTACHARYYA)
            scores.append(score)
        return sum(scores)/len(scores)

    @staticmethod
    def get_avg_score_faces(f1encs, references):
        scores = []
        for f2encs in references:
            score = face_recognition.face_distance(f1encs, f2encs)[0]
            scores.append(score)
        return sum(scores)/len(scores)

    @staticmethod
    def get_avg_score_faces_cnn(fl1, references):
        scores = []
        for fl2 in references:
            score = np.sum ( np.absolute ( (fl2 - fl1).flatten() ) )
            scores.append(score)
        return sum(scores)/len(scores)

    @staticmethod
    def write_to_log(log_file, changes):
        with open(log_file, 'w') as lf:
            json.dump(changes, lf, sort_keys=True, indent=4)


def bad_args(args):
    parser.print_help()
    exit(0)


if __name__ == "__main__":
    __warning_string = "Important: face-cnn method will cause an error when "
    __warning_string += "this tool is called directly instead of through the "
    __warning_string += "tools.py command script."
    print (__warning_string)
    print ("Images sort tool.\n")
    
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers()
    sort = SortProcessor(
            subparser, "sort", "Sort images using various methods.")

    parser.set_defaults(func=bad_args)
    arguments = parser.parse_args()
    arguments.func(arguments)
