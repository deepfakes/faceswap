#!/usr/bin python3
import argparse
import os
import sys
import operator
import numpy as np
import cv2
from tqdm import tqdm
from shutil import copyfile
import json
from lib.cli import DirFullPaths, FileFullPaths

# DLIB is a GPU Memory hog, so the following modules should only be imported
# when required
face_recognition = None
FaceLandmarksExtractor = None


def import_face_recognition():
    """ Import the face_recognition module only when it is required """
    global face_recognition
    if face_recognition is None:
        import face_recognition


def import_FaceLandmarksExtractor():
    """ Import the FaceLandmarksExtractor module only when it is required """
    global FaceLandmarksExtractor
    if FaceLandmarksExtractor is None:
        import lib.FaceLandmarksExtractor
        FaceLandmarksExtractor = lib.FaceLandmarksExtractor


class SortProcessor(object):
    def __init__(self, subparser, command, description='default'):
        self.argument_list = self.get_argument_list()
        self.optional_arguments = self.get_optional_arguments()
        self.args = None
        self.changes = None
        self.parse_arguments(description, subparser, command)

    @staticmethod
    def get_argument_list():
        arguments_list = list()
        arguments_list.append({"opts": ('-i', '--input'),
                               "action": DirFullPaths,
                               "dest": "input_dir",
                               "default": "input_dir",
                               "help": "Input directory of aligned faces.",
                               "required": True})

        arguments_list.append({"opts": ('-o', '--output'),
                               "action": DirFullPaths,
                               "dest": "output_dir",
                               "default": "output_dir",
                               "help": "Output directory for sorted aligned "
                                       "faces."})

        arguments_list.append({"opts": ('-f', '--final-process'),
                               "type": str,
                               "choices": ("folders", "rename"),
                               "dest": 'final_process',
                               "default": "rename",
                               "help": "'folders': files are sorted using the "
                                       "-s/--sort-by method, then they are "
                                       "organized into folders using the "
                                       "-g/--group-by grouping method. "
                                       "'rename': files are sorted using the "
                                       "-s/--sort-by then they are renamed. "
                                       "Default: rename"})

        arguments_list.append({"opts": ('-t', '--ref_threshold'),
                               "type": float,
                               "dest": 'min_threshold',
                               "default": -1.0,
                               "help": "Float value. "
                                       "Minimum threshold to use for grouping "
                                       "comparison with 'face' and 'hist' methods. "
                                       "The lower the value the more discriminating "
                                       "the grouping is. "
                                       "Leaving -1.0 will make the program "
                                       "set the default value automatically. "
                                       "For face 0.6 should be enough, with 0.5 "
                                       "being very discriminating. "
                                       "For face-cnn 7.2 should be enough, with 4 "
                                       "being very discriminating. "
                                       "For hist 0.3 should be enough, with 0.2 "
                                       "being very discriminating. "
                                       "Be careful setting a value that's too "
                                       "low in a directory with many images, as "
                                       "this could result in a lot of directories "
                                       " being created. "
                                       "Defaults: face 0.6, face-cnn 7.2, hist 0.3"})

        arguments_list.append({"opts": ('-b', '--bins'),
                               "type": int,
                               "dest": 'num_bins',
                               "default": 5,
                               "help": "Integer value. "
                                       "Number of folders that will be used to "
                                       "group by blur. Folder 0 will be the least "
                                       "blurry, while the last folder will be the "
                                       "blurriest. If the number of images doesn't "
                                       "divide evenly into the number of bins, the "
                                       "remaining images get put in the last bin as "
                                       "they will be the blurriest by definition. "
                                       "Default value: 5"})

        arguments_list.append({"opts": ('-k', '--keep'),
                               "action": 'store_true',
                               "dest": 'keep_original',
                               "default": False,
                               "help": "Keeps the original files in the input "
                                       "directory. Be careful when using this with "
                                       "rename grouping and no specified output "
                                       "directory as this would keep the original "
                                       "and renamed files in the same directory."})

        arguments_list.append({"opts": ('-l', '--log-changes'),
                               "action": 'store_true',
                               "dest": 'log_changes',
                               "default": False,
                               "help": "Logs file renaming changes if grouping by "
                                       "renaming, or it logs the file "
                                       "copying/movement if grouping by folders. "
                                       "If no log file is specified with "
                                       "'--log-file', then a 'sort_log.json' file "
                                       "will be created in the input directory."})

        arguments_list.append({"opts": ('-lf', '--log-file'),
                               "action": FileFullPaths,
                               "filetypes": ("JSON", "*.json"),
                               "dest": 'log_file_path',
                               "default": 'sort_log.json',
                               "help": "Specify a log file to use for saving the "
                                       "renaming or grouping information. "
                                       "Default: sort_log.json"})

        arguments_list.append({"opts": ('-s', '--sort-by'),
                               "type": str,
                               "choices": ("blur", "face", "face-cnn",
                                           "face-cnn-dissim", "face-dissim",
                                           "face-yaw", "hist",
                                           "hist-dissim"),
                               "dest": 'sort_method',
                               "default": "hist",
                               "help": "Sort by method. "
                                       "Choose how images are sorted. "
                                       "Default: hist"})

        arguments_list.append({"opts": ('-g', '--group-by'),
                               "type": str,
                               "choices": ("blur", "face", "face-cnn", "hist"),
                               "dest": 'group_method',
                               "default": "hist",
                               "help": "Group by method. "
                                       "When -fp/--final-processing by folders "
                                       "choose the how the images are grouped after "
                                       "sorting. "
                                       "Default: hist"})
        return arguments_list

    @staticmethod
    def get_optional_arguments():
        """
        Put the arguments in a list so that they are accessible from both
        argparse and gui.
        """
        # Override this for custom arguments
        argument_list = []
        return argument_list

    def parse_arguments(self, description, subparser, command):
        parser = subparser.add_parser(
                command,
                help="This command lets you sort images using various "
                     "methods.",
                description=description,
                epilog="Questions and feedback: \
                        https://github.com/deepfakes/faceswap-playground"
        )

        for option in self.argument_list:
            args = option['opts']
            kwargs = {key: option[key] for key in option.keys() if key != 'opts'}
            parser.add_argument(*args, **kwargs)

        parser = self.add_optional_arguments(parser)
        parser.set_defaults(func=self.process_arguments)

    def add_optional_arguments(self, parser):
        for option in self.optional_arguments:
            args = option['opts']
            kwargs = {key: option[key] for key in option.keys() if key != 'opts'}
            parser.add_argument(*args, **kwargs)
        return parser

    def process_arguments(self, arguments):
        self.args = arguments

        # Setting default argument values that cannot be set by argparse

        # Set output dir to the same value as input dir
        # if the user didn't specify it.
        if self.args.output_dir.lower() == "--default":
            self.args.output_dir = self.args.input_dir

        # Set final_process to group if folders was chosen
        if self.args.final_process.lower() == "folders":
            self.args.final_process = "group"

        # Assign default group_method if not set by user
        #if self.args.group_method == '--default':
        #    self.args.group_method = self.args.sort_method.replace('-dissim', '')

        # Assigning default threshold values based on grouping method
        if self.args.min_threshold == -1.0 and self.args.final_process == "group":
            method = self.args.group_method.lower()
            if method == 'face':
                self.args.min_threshold = 0.6
            elif method == 'face-cnn':
                self.args.min_threshold = 7.2
            elif method == 'hist':
                self.args.min_threshold = 0.3

        # If logging is enabled, prepare container
        if self.args.log_changes:
            self.changes = dict()

        # Assign default sort_log.json value if user didn't specify one
        if self.args.log_file_path.lower() == 'sort_log.json':
            self.args.log_file_path = os.path.join(self.args.input_dir, 'sort_log.json')

        # Prepare sort, group and final process method names
        _sort = "sort_" + self.args.sort_method.lower()
        _group = "group_" + self.args.group_method.lower()
        _final = "final_process_" + self.args.final_process.lower()
        self.args.sort_method = _sort.replace('-', '_')
        self.args.group_method = _group.replace('-', '_')
        self.args.final_process = _final.replace('-', '_')

        self.process()

    def process(self):
        """
        This method dynamically assigns the functions that will be used to run
        the core process of sorting, optionally grouping, renaming/moving into
        folders. After the functions are assigned they are executed.
        """
        sort_method = self.args.sort_method.lower()
        group_method = self.args.group_method.lower()
        final_method = self.args.final_process.lower()

        img_list = getattr(self, sort_method)()
        if "group" in final_method:
            # Check if non-dissim sort method and group method are not the same
            if group_method.replace('group_', '') not in sort_method:
                img_list = self.reload_images(group_method, img_list)
                img_list = getattr(self, group_method)(img_list)
            else:
                img_list = getattr(self, group_method)(img_list)

        getattr(self, final_method)(img_list)

        print("Done.")

    # Methods for sorting
    def sort_blur(self):
        input_dir = self.args.input_dir

        print("Sorting by blur...")
        img_list = [[x, self.estimate_blur(cv2.imread(x))]
                    for x in
                    tqdm(self.find_images(input_dir), desc="Loading", file=sys.stdout)]
        print("Sorting...")

        img_list = sorted(img_list, key=operator.itemgetter(1), reverse=True)

        return img_list

    def sort_face(self):
        import_face_recognition()

        input_dir = self.args.input_dir

        print("Sorting by face similarity...")

        img_list = [[x, face_recognition.face_encodings(cv2.imread(x))]
                    for x in
                    tqdm(self.find_images(input_dir), desc="Loading", file=sys.stdout)]

        img_list_len = len(img_list)
        for i in tqdm(range(0, img_list_len - 1), desc="Sorting", file=sys.stdout):
            min_score = float("inf")
            j_min_score = i + 1
            for j in range(i + 1, len(img_list)):
                f1encs = img_list[i][1]
                f2encs = img_list[j][1]
                if f1encs is not None and f2encs is not None and len(
                        f1encs) > 0 and len(f2encs) > 0:
                    score = face_recognition.face_distance(f1encs[0], f2encs)[0]
                else:
                    score = float("inf")

                if score < min_score:
                    min_score = score
                    j_min_score = j
            img_list[i + 1], img_list[j_min_score] = img_list[j_min_score], img_list[i + 1]

        return img_list

    def sort_face_dissim(self):
        import_face_recognition()

        input_dir = self.args.input_dir

        print("Sorting by face dissimilarity...")

        img_list = [[x, face_recognition.face_encodings(cv2.imread(x)), 0]
                    for x in
                    tqdm(self.find_images(input_dir), desc="Loading", file=sys.stdout)]

        img_list_len = len(img_list)
        for i in tqdm(range(0, img_list_len), desc="Sorting", file=sys.stdout):
            score_total = 0
            for j in range(0, img_list_len):
                if i == j:
                    continue
                try:
                    score_total += face_recognition.face_distance([img_list[i][1]], [img_list[j][1]])
                except:
                    pass

            img_list[i][2] = score_total

        print("Sorting...")
        img_list = sorted(img_list, key=operator.itemgetter(2), reverse=True)
        return img_list

    def sort_face_cnn(self):
        import_FaceLandmarksExtractor()

        input_dir = self.args.input_dir

        print("Sorting by face-cnn similarity...")

        img_list = []
        for x in tqdm(self.find_images(input_dir), desc="Loading", file=sys.stdout):
            d = FaceLandmarksExtractor.extract(cv2.imread(x), 'cnn', True, input_is_predetected_face=True)
            img_list.append([x, np.array(d[0][1]) if len(d) > 0 else np.zeros((68, 2))])

        img_list_len = len(img_list)
        for i in tqdm(range(0, img_list_len - 1), desc="Sorting", file=sys.stdout):
            min_score = float("inf")
            j_min_score = i + 1
            for j in range(i + 1, len(img_list)):
                fl1 = img_list[i][1]
                fl2 = img_list[j][1]
                score = np.sum(np.absolute((fl2 - fl1).flatten()))

                if score < min_score:
                    min_score = score
                    j_min_score = j
            img_list[i + 1], img_list[j_min_score] = img_list[j_min_score], img_list[i + 1]

        return img_list

    def sort_face_cnn_dissim(self):
        import_FaceLandmarksExtractor()

        input_dir = self.args.input_dir

        print("Sorting by face-cnn dissimilarity...")

        img_list = []
        for x in tqdm(self.find_images(input_dir), desc="Loading", file=sys.stdout):
            d = FaceLandmarksExtractor.extract(cv2.imread(x), 'cnn', True, input_is_predetected_face=True)
            img_list.append([x, np.array(d[0][1]) if len(d) > 0 else np.zeros((68, 2)), 0])

        img_list_len = len(img_list)
        for i in tqdm(range(0, img_list_len - 1), desc="Sorting", file=sys.stdout):
            score_total = 0
            for j in range(i + 1, len(img_list)):
                if i == j:
                    continue
                fl1 = img_list[i][1]
                fl2 = img_list[j][1]
                score_total += np.sum(np.absolute((fl2 - fl1).flatten()))

            img_list[i][2] = score_total

        print("Sorting...")
        img_list = sorted(img_list, key=operator.itemgetter(2), reverse=True)

        return img_list

    def sort_face_yaw(self):
        def calc_landmarks_face_pitch(fl):  # unused
            t = ((fl[6][1] - fl[8][1]) + (fl[10][1] - fl[8][1])) / 2.0
            b = fl[8][1]
            return b - t

        def calc_landmarks_face_yaw(fl):
            l = ((fl[27][0] - fl[0][0]) + (fl[28][0] - fl[1][0]) + (fl[29][0] - fl[2][0])) / 3.0
            r = ((fl[16][0] - fl[27][0]) + (fl[15][0] - fl[28][0]) + (fl[14][0] - fl[29][0])) / 3.0
            return r - l

        import_FaceLandmarksExtractor()
        input_dir = self.args.input_dir

        img_list = []
        for x in tqdm(self.find_images(input_dir), desc="Loading", file=sys.stdout):
            d = FaceLandmarksExtractor.extract(cv2.imread(x), 'cnn', True, input_is_predetected_face=True)
            img_list.append([x, calc_landmarks_face_yaw(np.array(d[0][1]))])

        print("Sorting...")
        img_list = sorted(img_list, key=operator.itemgetter(1), reverse=True)

        return img_list

    def sort_hist(self):
        input_dir = self.args.input_dir

        print("Sorting by histogram similarity...")

        img_list = [
            [x, cv2.calcHist([cv2.imread(x)], [0], None, [256], [0, 256])]
            for x in
            tqdm(self.find_images(input_dir), desc="Loading", file=sys.stdout)
        ]

        img_list_len = len(img_list)
        for i in tqdm(range(0, img_list_len - 1), desc="Sorting",
                      file=sys.stdout):
            min_score = float("inf")
            j_min_score = i + 1
            for j in range(i + 1, len(img_list)):
                score = cv2.compareHist(img_list[i][1],
                                        img_list[j][1],
                                        cv2.HISTCMP_BHATTACHARYYA)
                if score < min_score:
                    min_score = score
                    j_min_score = j
            img_list[i + 1], img_list[j_min_score] = img_list[j_min_score], img_list[i + 1]

        return img_list

    def sort_hist_dissim(self):
        input_dir = self.args.input_dir

        print("Sorting by histogram dissimilarity...")

        img_list = [
            [x, cv2.calcHist([cv2.imread(x)], [0], None, [256], [0, 256]), 0]
            for x in
            tqdm(self.find_images(input_dir), desc="Loading", file=sys.stdout)
        ]

        img_list_len = len(img_list)
        for i in tqdm(range(0, img_list_len), desc="Sorting", file=sys.stdout):
            score_total = 0
            for j in range(0, img_list_len):
                if i == j:
                    continue
                score_total += cv2.compareHist(img_list[i][1],
                                               img_list[j][1],
                                               cv2.HISTCMP_BHATTACHARYYA)

            img_list[i][2] = score_total

        print("Sorting...")
        img_list = sorted(img_list, key=operator.itemgetter(2), reverse=True)

        return img_list

    # Methods for grouping
    def group_blur(self, img_list):
        # Starting the binning process
        num_bins = self.args.num_bins

        # The last bin will get all extra images if it's
        # not possible to distribute them evenly
        num_per_bin = len(img_list) // num_bins
        remainder = len(img_list) % num_bins

        print("Grouping by blur...")
        bins = [[] for _ in range(num_bins)]
        image_index = 0
        for i in range(num_bins):
            for j in range(num_per_bin):
                bins[i].append(img_list[image_index][0])
                image_index += 1

        # If remainder is 0, nothing gets added to the last bin.
        for i in range(1, remainder + 1):
            bins[-1].append(img_list[-i][0])

        return bins

    def group_face(self, img_list):
        print("Grouping by face similarity...")

        # Groups are of the form: group_num -> reference face
        reference_groups = dict()

        # Bins array, where index is the group number and value is
        # an array containing the file paths to the images in that group.
        # The first group (0), is always the non-face group.
        bins = [[]]

        # Comparison threshold used to decide how similar
        # faces have to be to be grouped together.
        min_threshold = self.args.min_threshold

        img_list_len = len(img_list)

        for i in tqdm(range(1, img_list_len), desc="Grouping", file=sys.stdout):
            f1encs = img_list[i][1]

            # Check if current image is a face, if not then
            # add it immediately to the non-face list.
            if f1encs is None or len(f1encs) <= 0:
                bins[0].append(img_list[i][0])

            else:
                current_best = [-1, float("inf")]

                for key, references in reference_groups.items():
                    # Non-faces are not added to reference_groups dict, thus
                    # removing the need to check that f2encs is a face.
                    # The try-catch block is to handle the first face that gets
                    # processed, as the first value is None.
                    try:
                        score = self.get_avg_score_faces(f1encs, references)
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

        return bins

    def group_face_cnn(self, img_list):
        print("Grouping by face-cnn similarity...")

        # Groups are of the form: group_num -> reference faces
        reference_groups = dict()

        # Bins array, where index is the group number and value is
        # an array containing the file paths to the images in that group.
        bins = []

        # Comparison threshold used to decide how similar
        # faces have to be to be grouped together.
        # It is multiplied by 1000 here to allow the cli option to use smaller
        # numbers.
        min_threshold = self.args.min_threshold * 1000

        img_list_len = len(img_list)

        for i in tqdm(range(0, img_list_len - 1), desc="Grouping", file=sys.stdout):
            fl1 = img_list[i][1]

            current_best = [-1, float("inf")]

            for key, references in reference_groups.items():
                try:
                    score = self.get_avg_score_faces_cnn(fl1, references)
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

        return bins

    def group_hist(self, img_list):
        print("Grouping by histogram...")

        # Groups are of the form: group_num -> reference histogram
        reference_groups = dict()

        # Bins array, where index is the group number and value is
        # an array containing the file paths to the images in that group
        bins = []

        min_threshold = self.args.min_threshold

        img_list_len = len(img_list)
        reference_groups[0] = [img_list[0][1]]
        bins.append([img_list[0][0]])

        for i in tqdm(range(1, img_list_len), desc="Grouping", file=sys.stdout):
            current_best = [-1, float("inf")]
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

        return bins

    # Final process methods
    def final_process_rename(self, img_list):
        output_dir = self.args.output_dir

        process_file = self.set_process_file_method(self.args.log_changes,
                                                    self.args.keep_original)

        # Make sure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        description = (
            "Copying and Renaming" if self.args.keep_original
            else "Moving and Renaming"
        )

        for i in tqdm(range(0, len(img_list)), desc=description, leave=False, file=sys.stdout):
            src = img_list[i][0]
            src_basename = os.path.basename(src)

            dst = os.path.join(output_dir, '{:05d}_{}'.format(i, src_basename))
            try:
                process_file(src, dst, self.changes)
            except FileNotFoundError as e:
                print(e)
                print('fail to rename {}'.format(src))

        for i in tqdm(range(0, len(img_list)), desc=description, file=sys.stdout):
            renaming = self.set_renaming_method(self.args.log_changes)
            src, dst = renaming(img_list[i][0], output_dir, i, self.changes)

            try:
                os.rename(src, dst)
            except FileNotFoundError as e:
                print(e)
                print('fail to rename {}'.format(src))

        if self.args.log_changes:
            self.write_to_log(self.args.log_file_path, self.changes)

    def final_process_group(self, bins):
        output_dir = self.args.output_dir

        process_file = self.set_process_file_method(self.args.log_changes,
                                                    self.args.keep_original)

        # First create new directories to avoid checking
        # for directory existence in the moving loop
        print("Creating group directories.")
        for i in range(len(bins)):
            directory = os.path.join(output_dir, str(i))
            if not os.path.exists(directory):
                os.makedirs(directory)

        description = (
            "Copying into Groups" if self.args.keep_original
            else "Moving into Groups"
        )

        print("Total groups found: {}".format(len(bins)))
        for i in tqdm(range(len(bins)), desc=description, file=sys.stdout):
            for j in range(len(bins[i])):
                src = bins[i][j]
                src_basename = os.path.basename(src)

                dst = os.path.join(output_dir, str(i), src_basename)
                try:
                    process_file(src, dst, self.changes)
                except FileNotFoundError as e:
                    print(e)
                    print('Failed to move {0} to {1}'.format(src, dst))

        if self.args.log_changes:
            self.write_to_log(self.args.log_file_path, self.changes)

    # Various helper methods
    def reload_images(self, group_method, img_list):
        """
        Reloads the image list by replacing the comparative values with those
        that the chosen grouping method expects.
        :param group_method: str name of the grouping method that will be used.
        :param img_list: image list that has been sorted by one of the sort
        methods.
        :return: img_list but with the comparative values that the chosen
        grouping method expects.
        """
        import_face_recognition()

        input_dir = self.args.input_dir
        print("Preparing to group...")
        if group_method == 'group_blur':
            temp_list = [[x, self.estimate_blur(cv2.imread(x))]
                         for x in
                         tqdm(self.find_images(input_dir), desc="Reloading", file=sys.stdout)]
        elif group_method == 'group_face':
            temp_list = [[x, face_recognition.face_encodings(cv2.imread(x))]
                         for x in
                         tqdm(self.find_images(input_dir), desc="Reloading", file=sys.stdout)]
        elif group_method == 'group_face_cnn':
            import_FaceLandmarksExtractor()
            temp_list = []
            for x in tqdm(self.find_images(input_dir), desc="Reloading", file=sys.stdout):
                d = FaceLandmarksExtractor.extract(cv2.imread(x), 'cnn', True,
                                                   input_is_predetected_face=True)
                temp_list.append([x, np.array(d[0][1]) if len(d) > 0 else np.zeros((68, 2))])
        elif group_method == 'group_hist':
            temp_list = [
                [x, cv2.calcHist([cv2.imread(x)], [0], None, [256], [0, 256])]
                for x in
                tqdm(self.find_images(input_dir), desc="Reloading", file=sys.stdout)
            ]
        else:
            raise ValueError("{} group_method not found.".format(group_method))

        return self.splice_lists(img_list, temp_list)

    @staticmethod
    def splice_lists(sorted_list, new_vals_list):
        """
        This method replaces the value at index 1 in each sub-list in the
        sorted_list with the value that is calculated for the same img_path,
        but found in new_vals_list.

        Format of lists: [[img_path, value], [img_path2, value2], ...]

        :param sorted_list: list that has been sorted by one of the sort
        methods.
        :param new_vals_list: list that has been loaded by a different method
        than the sorted_list.
        :return: list that is sorted in the same way as the input sorted list
        but the values corresponding to each image are from new_vals_list.
        """
        new_list = []
        # Make new list of just image paths to serve as an index
        val_index_list = [i[0] for i in new_vals_list]
        for i in tqdm(range(len(sorted_list)), desc="Splicing", file=sys.stdout):
            current_image = sorted_list[i][0]
            new_val_index = val_index_list.index(current_image)
            new_list.append([current_image, new_vals_list[new_val_index][1]])

        return new_list

    @staticmethod
    def find_images(input_dir):
        result = []
        extensions = [".jpg", ".png", ".jpeg"]
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in extensions:
                    result.append(os.path.join(root, file))
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

                __src = os.path.join(output_dir, '{:05d}_{}'.format(i, src_basename))
                dst = os.path.join(output_dir, '{:05d}{}'.format(i, os.path.splitext(src_basename)[1]))
                changes[src] = dst
                return __src, dst

            return renaming

        else:
            def renaming(src, output_dir, i, changes):
                src_basename = os.path.basename(src)

                src = os.path.join(output_dir, '{:05d}_{}'.format(i, src_basename))
                dst = os.path.join(output_dir, '{:05d}{}'.format(i, os.path.splitext(src_basename)[1]))
                return src, dst

            return renaming

    @staticmethod
    def get_avg_score_hist(img1, references):
        scores = []
        for img2 in references:
            score = cv2.compareHist(img1, img2, cv2.HISTCMP_BHATTACHARYYA)
            scores.append(score)
        return sum(scores) / len(scores)

    @staticmethod
    def get_avg_score_faces(f1encs, references):
        import_face_recognition()
        scores = []
        for f2encs in references:
            score = face_recognition.face_distance(f1encs, f2encs)[0]
            scores.append(score)
        return sum(scores) / len(scores)

    @staticmethod
    def get_avg_score_faces_cnn(fl1, references):
        scores = []
        for fl2 in references:
            score = np.sum(np.absolute((fl2 - fl1).flatten()))
            scores.append(score)
        return sum(scores) / len(scores)

    @staticmethod
    def write_to_log(log_file_path, changes):
        with open(log_file_path, 'w') as lf:
            json.dump(changes, lf, sort_keys=True, indent=4)


def bad_args(args):
    parser.print_help()
    exit(0)


if __name__ == "__main__":
    __warning_string = "Important: face-cnn method will cause an error when "
    __warning_string += "this tool is called directly instead of through the "
    __warning_string += "tools.py command script."
    print(__warning_string)
    print("Images sort tool.\n")

    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers()
    sort = SortProcessor(
            subparser, "sort", "Sort images using various methods.")

    parser.set_defaults(func=bad_args)
    arguments = parser.parse_args()
    arguments.func(arguments)
