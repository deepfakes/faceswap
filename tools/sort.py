#!/usr/bin/env python3
"""
A tool that allows for sorting and grouping images in different ways.
"""
import os
import sys
import operator
from shutil import copyfile

import numpy as np
import cv2
from tqdm import tqdm

# faceswap imports
import face_recognition

from lib.cli import FullHelpArgumentParser
from lib import face_alignment, Serializer

from . import cli


class Sort(object):
    """ Sorts folders of faces based on input criteria """
    def __init__(self, arguments):
        self.args = arguments
        self.changes = None
        self.serializer = None

    def process(self):
        """ Main processing function of the sort tool """

        # Setting default argument values that cannot be set by argparse

        # Set output dir to the same value as input dir
        # if the user didn't specify it.
        if self.args.output_dir.lower() == "_output_dir":
            self.args.output_dir = self.args.input_dir

        # Assigning default threshold values based on grouping method
        if (self.args.final_process == "folders"
                and self.args.min_threshold == -1.0):
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
            if self.args.log_file_path == 'sort_log.json':
                self.args.log_file_path = os.path.join(self.args.input_dir,
                                                       'sort_log.json')

            # Set serializer based on logfile extension
            serializer_ext = os.path.splitext(
                self.args.log_file_path)[-1]
            self.serializer = Serializer.get_serializer_from_ext(
                serializer_ext)

        # Prepare sort, group and final process method names
        _sort = "sort_" + self.args.sort_method.lower()
        _group = "group_" + self.args.group_method.lower()
        _final = "final_process_" + self.args.final_process.lower()
        self.args.sort_method = _sort.replace('-', '_')
        self.args.group_method = _group.replace('-', '_')
        self.args.final_process = _final.replace('-', '_')

        self.sort_process()

    def sort_process(self):
        """
        This method dynamically assigns the functions that will be used to run
        the core process of sorting, optionally grouping, renaming/moving into
        folders. After the functions are assigned they are executed.
        """
        sort_method = self.args.sort_method.lower()
        group_method = self.args.group_method.lower()
        final_method = self.args.final_process.lower()

        img_list = getattr(self, sort_method)()
        if "folders" in final_method:
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
        """ Sort by blur amount """
        input_dir = self.args.input_dir

        print("Sorting by blur...")
        img_list = [[img, self.estimate_blur(cv2.imread(img))]
                    for img in
                    tqdm(self.find_images(input_dir),
                         desc="Loading",
                         file=sys.stdout)]
        print("Sorting...")

        img_list = sorted(img_list, key=operator.itemgetter(1), reverse=True)

        return img_list

    def sort_face(self):
        """ Sort by face similarity """
        input_dir = self.args.input_dir

        print("Sorting by face similarity...")

        img_list = [[img, face_recognition.face_encodings(cv2.imread(img))]
                    for img in
                    tqdm(self.find_images(input_dir),
                         desc="Loading",
                         file=sys.stdout)]

        img_list_len = len(img_list)
        for i in tqdm(range(0, img_list_len - 1),
                      desc="Sorting",
                      file=sys.stdout):
            min_score = float("inf")
            j_min_score = i + 1
            for j in range(i + 1, len(img_list)):
                f1encs = img_list[i][1]
                f2encs = img_list[j][1]
                if f1encs and f2encs:
                    score = face_recognition.face_distance(f1encs[0],
                                                           f2encs)[0]
                else:
                    score = float("inf")

                if score < min_score:
                    min_score = score
                    j_min_score = j
            (img_list[i + 1],
             img_list[j_min_score]) = (img_list[j_min_score],
                                       img_list[i + 1])
        return img_list

    def sort_face_dissim(self):
        """ Sort by face dissimilarity """
        input_dir = self.args.input_dir

        print("Sorting by face dissimilarity...")

        img_list = [[img, face_recognition.face_encodings(cv2.imread(img)), 0]
                    for img in
                    tqdm(self.find_images(input_dir),
                         desc="Loading",
                         file=sys.stdout)]

        img_list_len = len(img_list)
        for i in tqdm(range(0, img_list_len), desc="Sorting", file=sys.stdout):
            score_total = 0
            for j in range(0, img_list_len):
                if i == j:
                    continue
                try:
                    score_total += face_recognition.face_distance(
                        [img_list[i][1]],
                        [img_list[j][1]])
                except:
                    pass

            img_list[i][2] = score_total

        print("Sorting...")
        img_list = sorted(img_list, key=operator.itemgetter(2), reverse=True)
        return img_list

    def sort_face_cnn(self):
        """ Sort by dlib CNN similarity """
        input_dir = self.args.input_dir

        print("Sorting by face-cnn similarity...")

        img_list = []
        for img in tqdm(self.find_images(input_dir),
                        desc="Loading",
                        file=sys.stdout):
            landmarks = face_alignment.Extract(
                input_image_bgr=cv2.imread(img),
                detector='dlib-cnn',
                verbose=True,
                input_is_predetected_face=True).landmarks
            img_list.append([img, np.array(landmarks[0][1])
                             if landmarks
                             else np.zeros((68, 2))])

        img_list_len = len(img_list)
        for i in tqdm(range(0, img_list_len - 1),
                      desc="Sorting",
                      file=sys.stdout):
            min_score = float("inf")
            j_min_score = i + 1
            for j in range(i + 1, len(img_list)):
                fl1 = img_list[i][1]
                fl2 = img_list[j][1]
                score = np.sum(np.absolute((fl2 - fl1).flatten()))

                if score < min_score:
                    min_score = score
                    j_min_score = j
            (img_list[i + 1],
             img_list[j_min_score]) = (img_list[j_min_score],
                                       img_list[i + 1])
        return img_list

    def sort_face_cnn_dissim(self):
        """ Sort by dlib CNN dissimilarity """
        input_dir = self.args.input_dir

        print("Sorting by face-cnn dissimilarity...")

        img_list = []
        for img in tqdm(self.find_images(input_dir),
                        desc="Loading",
                        file=sys.stdout):
            landmarks = face_alignment.Extract(
                input_image_bgr=cv2.imread(img),
                detector='dlib-cnn',
                verbose=True,
                input_is_predetected_face=True).landmarks
            img_list.append([img, np.array(landmarks[0][1])
                             if landmarks
                             else np.zeros((68, 2)), 0])

        img_list_len = len(img_list)
        for i in tqdm(range(0, img_list_len - 1),
                      desc="Sorting",
                      file=sys.stdout):
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
        """ Sort by yaw of face """
        input_dir = self.args.input_dir

        img_list = []
        for img in tqdm(self.find_images(input_dir),
                        desc="Loading",
                        file=sys.stdout):
            landmarks = face_alignment.Extract(
                input_image_bgr=cv2.imread(img),
                detector='dlib-cnn',
                verbose=True,
                input_is_predetected_face=True).landmarks
            img_list.append(
                [img, self.calc_landmarks_face_yaw(np.array(landmarks[0][1]))])

        print("Sorting by face-yaw...")
        img_list = sorted(img_list, key=operator.itemgetter(1), reverse=True)

        return img_list

    def sort_hist(self):
        """ Sort by histogram of face similarity """
        input_dir = self.args.input_dir

        print("Sorting by histogram similarity...")

        img_list = [
            [img, cv2.calcHist([cv2.imread(img)], [0], None, [256], [0, 256])]
            for img in
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
            (img_list[i + 1],
             img_list[j_min_score]) = (img_list[j_min_score],
                                       img_list[i + 1])
        return img_list

    def sort_hist_dissim(self):
        """ Sort by histigram of face dissimilarity """
        input_dir = self.args.input_dir

        print("Sorting by histogram dissimilarity...")

        img_list = [
            [img,
             cv2.calcHist([cv2.imread(img)], [0], None, [256], [0, 256]), 0]
            for img in
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
        """ Group into bins by blur """
        # Starting the binning process
        num_bins = self.args.num_bins

        # The last bin will get all extra images if it's
        # not possible to distribute them evenly
        num_per_bin = len(img_list) // num_bins
        remainder = len(img_list) % num_bins

        print("Grouping by blur...")
        bins = [[] for _ in range(num_bins)]
        idx = 0
        for i in range(num_bins):
            for _ in range(num_per_bin):
                bins[i].append(img_list[idx][0])
                idx += 1

        # If remainder is 0, nothing gets added to the last bin.
        for i in range(1, remainder + 1):
            bins[-1].append(img_list[-i][0])

        return bins

    def group_face(self, img_list):
        """ Group into bins by face similarity """
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

        for i in tqdm(range(1, img_list_len),
                      desc="Grouping",
                      file=sys.stdout):
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
        """ Group into bins by dlib CNN face similarity """
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

        for i in tqdm(range(0, img_list_len - 1),
                      desc="Grouping",
                      file=sys.stdout):
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

    def group_face_yaw(self, img_list):
        """ Group into bins by yaw of face """
        # Starting the binning process
        num_bins = self.args.num_bins

        # The last bin will get all extra images if it's
        # not possible to distribute them evenly
        num_per_bin = len(img_list) // num_bins
        remainder = len(img_list) % num_bins

        print("Grouping by face-yaw...")
        bins = [[] for _ in range(num_bins)]
        idx = 0
        for i in range(num_bins):
            for _ in range(num_per_bin):
                bins[i].append(img_list[idx][0])
                idx += 1

        # If remainder is 0, nothing gets added to the last bin.
        for i in range(1, remainder + 1):
            bins[-1].append(img_list[-i][0])

        return bins

    def group_hist(self, img_list):
        """ Group into bins by histogram """
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

        for i in tqdm(range(1, img_list_len),
                      desc="Grouping",
                      file=sys.stdout):
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
        """ Rename the files """
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

        for i in tqdm(range(0, len(img_list)),
                      desc=description,
                      leave=False,
                      file=sys.stdout):
            src = img_list[i][0]
            src_basename = os.path.basename(src)

            dst = os.path.join(output_dir, '{:05d}_{}'.format(i, src_basename))
            try:
                process_file(src, dst, self.changes)
            except FileNotFoundError as err:
                print(err)
                print('fail to rename {}'.format(src))

        for i in tqdm(range(0, len(img_list)),
                      desc=description,
                      file=sys.stdout):
            renaming = self.set_renaming_method(self.args.log_changes)
            src, dst = renaming(img_list[i][0], output_dir, i, self.changes)

            try:
                os.rename(src, dst)
            except FileNotFoundError as err:
                print(err)
                print('fail to rename {}'.format(src))

        if self.args.log_changes:
            self.write_to_log(self.changes)

    def final_process_folders(self, bins):
        """ Move the files to folders """
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
                except FileNotFoundError as err:
                    print(err)
                    print('Failed to move {0} to {1}'.format(src, dst))

        if self.args.log_changes:
            self.write_to_log(self.changes)

    # Various helper methods
    def write_to_log(self, changes):
        """ Write the changes to log file """
        print("Writing sort log to: {}".format(self.args.log_file_path))
        with open(self.args.log_file_path, 'w') as lfile:
            lfile.write(self.serializer.marshal(changes))

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
        input_dir = self.args.input_dir
        print("Preparing to group...")
        if group_method == 'group_blur':
            temp_list = [[img, self.estimate_blur(cv2.imread(img))]
                         for img in
                         tqdm(self.find_images(input_dir),
                              desc="Reloading",
                              file=sys.stdout)]
        elif group_method == 'group_face':
            temp_list = [
                [img, face_recognition.face_encodings(cv2.imread(img))]
                for img in tqdm(self.find_images(input_dir),
                                desc="Reloading",
                                file=sys.stdout)]
        elif group_method == 'group_face_cnn':
            temp_list = []
            for img in tqdm(self.find_images(input_dir),
                            desc="Reloading",
                            file=sys.stdout):
                landmarks = face_alignment.Extract(
                    input_image_bgr=cv2.imread(img),
                    detector='dlib-cnn',
                    verbose=True,
                    input_is_predetected_face=True).landmarks
                temp_list.append([img, np.array(landmarks[0][1])
                                  if landmarks
                                  else np.zeros((68, 2))])
        elif group_method == 'group_face_yaw':
            temp_list = []
            for img in tqdm(self.find_images(input_dir),
                            desc="Reloading",
                            file=sys.stdout):
                landmarks = face_alignment.Extract(
                    input_image_bgr=cv2.imread(img),
                    detector='dlib-cnn',
                    verbose=True,
                    input_is_predetected_face=True).landmarks
                temp_list.append(
                    [img,
                     self.calc_landmarks_face_yaw(np.array(landmarks[0][1]))])
        elif group_method == 'group_hist':
            temp_list = [
                [img,
                 cv2.calcHist([cv2.imread(img)], [0], None, [256], [0, 256])]
                for img in
                tqdm(self.find_images(input_dir),
                     desc="Reloading",
                     file=sys.stdout)
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
        for i in tqdm(range(len(sorted_list)),
                      desc="Splicing",
                      file=sys.stdout):
            current_image = sorted_list[i][0]
            new_val_index = val_index_list.index(current_image)
            new_list.append([current_image, new_vals_list[new_val_index][1]])

        return new_list

    @staticmethod
    def find_images(input_dir):
        """ Return list of images at specified location """
        result = []
        extensions = [".jpg", ".png", ".jpeg"]
        for root, _, files in os.walk(input_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in extensions:
                    result.append(os.path.join(root, file))
        return result

    @staticmethod
    def estimate_blur(image):
        """ Estimate the amount of blur an image has """
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blur_map = cv2.Laplacian(image, cv2.CV_64F)
        score = np.var(blur_map)
        return score

    @staticmethod
    def calc_landmarks_face_pitch(flm):
        """ UNUSED - Calculate the amount of pitch in a face """
        var_t = ((flm[6][1] - flm[8][1]) + (flm[10][1] - flm[8][1])) / 2.0
        var_b = flm[8][1]
        return var_b - var_t

    @staticmethod
    def calc_landmarks_face_yaw(flm):
        """ Calculate the amount of yaw in a face """
        var_l = ((flm[27][0] - flm[0][0])
                 + (flm[28][0] - flm[1][0])
                 + (flm[29][0] - flm[2][0])) / 3.0
        var_r = ((flm[16][0] - flm[27][0])
                 + (flm[15][0] - flm[28][0])
                 + (flm[14][0] - flm[29][0])) / 3.0
        return var_r - var_l

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
                    """ Process file method if logging changes
                        and keeping original """
                    copyfile(src, dst)
                    changes[src] = dst

            else:
                def process_file(src, dst, changes):
                    """ Process file method if logging changes
                        and not keeping original """
                    os.rename(src, dst)
                    changes[src] = dst

        else:
            if keep_original:
                def process_file(src, dst, changes):
                    """ Process file method if not logging changes
                        and keeping original """
                    copyfile(src, dst)

            else:
                def process_file(src, dst, changes):
                    """ Process file method if not logging changes
                        and not keeping original """
                    os.rename(src, dst)
        return process_file

    @staticmethod
    def set_renaming_method(log_changes):
        """ Set the method for renaming files """
        if log_changes:
            def renaming(src, output_dir, i, changes):
                """ Rename files  method if logging changes """
                src_basename = os.path.basename(src)

                __src = os.path.join(output_dir,
                                     '{:05d}_{}'.format(i, src_basename))
                dst = os.path.join(
                    output_dir,
                    '{:05d}{}'.format(i, os.path.splitext(src_basename)[1]))
                changes[src] = dst
                return __src, dst
        else:
            def renaming(src, output_dir, i, changes):
                """ Rename files method if not logging changes """
                src_basename = os.path.basename(src)

                src = os.path.join(output_dir,
                                   '{:05d}_{}'.format(i, src_basename))
                dst = os.path.join(
                    output_dir,
                    '{:05d}{}'.format(i, os.path.splitext(src_basename)[1]))
                return src, dst
        return renaming

    @staticmethod
    def get_avg_score_hist(img1, references):
        """ Return the average histogram score between a face and
            reference image """
        scores = []
        for img2 in references:
            score = cv2.compareHist(img1, img2, cv2.HISTCMP_BHATTACHARYYA)
            scores.append(score)
        return sum(scores) / len(scores)

    @staticmethod
    def get_avg_score_faces(f1encs, references):
        """ Return the average similarity score between a face and
            reference image """
        scores = []
        for f2encs in references:
            score = face_recognition.face_distance(f1encs, f2encs)[0]
            scores.append(score)
        return sum(scores) / len(scores)

    @staticmethod
    def get_avg_score_faces_cnn(fl1, references):
        """ Return the average dlib CNN similarity score
            between a face and reference image """
        scores = []
        for fl2 in references:
            score = np.sum(np.absolute((fl2 - fl1).flatten()))
            scores.append(score)
        return sum(scores) / len(scores)


def bad_args(args):
    """ Print help on bad arguments """
    PARSER.print_help()
    exit(0)


if __name__ == "__main__":
    __warning_string = "Important: face-cnn method will cause an error when "
    __warning_string += "this tool is called directly instead of through the "
    __warning_string += "tools.py command script."
    print(__warning_string)
    print("Images sort tool.\n")

    PARSER = FullHelpArgumentParser()
    SUBPARSER = PARSER.add_subparsers()
    SORT = cli.SortArgs(
        SUBPARSER, "sort", "Sort images using various methods.")
    PARSER.set_defaults(func=bad_args)
    ARGUMENTS = PARSER.parse_args()
    ARGUMENTS.func(ARGUMENTS)
