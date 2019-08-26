#!/usr/bin/env python3
"""
A tool that allows for sorting and grouping images in different ways.
"""
import logging
import os
import sys
import operator
from shutil import copyfile

import numpy as np
import cv2
from tqdm import tqdm

# faceswap imports
from lib.cli import FullHelpArgumentParser
from lib import Serializer
from lib.faces_detect import DetectedFace
from lib.multithreading import SpawnProcess
from lib.queue_manager import queue_manager, QueueEmpty
from lib.utils import cv2_read_img
from lib.vgg_face2_keras import VGGFace2 as VGGFace
from plugins.plugin_loader import PluginLoader

from . import cli

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Sort():
    """ Sorts folders of faces based on input criteria """
    # pylint: disable=no-member
    def __init__(self, arguments):
        self.args = arguments
        self.changes = None
        self.serializer = None
        self.vgg_face = None

    def process(self):
        """ Main processing function of the sort tool """

        # Setting default argument values that cannot be set by argparse

        # Set output dir to the same value as input dir
        # if the user didn't specify it.
        if self.args.output_dir is None:
            logger.verbose("No output directory provided. Using input dir as output dir.")
            self.args.output_dir = self.args.input_dir

        # Assigning default threshold values based on grouping method
        if (self.args.final_process == "folders"
                and self.args.min_threshold < 0.0):
            method = self.args.group_method.lower()
            if method == 'face-cnn':
                self.args.min_threshold = 7.2
            elif method == 'hist':
                self.args.min_threshold = 0.3

        # Load VGG Face if sorting by face
        if self.args.sort_method.lower() == "face":
            self.vgg_face = VGGFace(backend=self.args.backend, loglevel=self.args.loglevel)

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

    def launch_aligner(self):
        """ Load the aligner plugin to retrieve landmarks """
        out_queue = queue_manager.get_queue("out")
        kwargs = {"in_queue": queue_manager.get_queue("in"),
                  "out_queue": out_queue}

        for plugin in ("fan", "cv2_dnn"):
            aligner = PluginLoader.get_aligner(plugin)(loglevel=self.args.loglevel)
            process = SpawnProcess(aligner.run, **kwargs)
            event = process.event
            process.start()
            # Wait for Aligner to take init
            # The first ever load of the model for FAN has reportedly taken
            # up to 3-4 minutes, hence high timeout.
            event.wait(300)

            if not event.is_set():
                if plugin == "fan":
                    process.join()
                    logger.error("Error initializing FAN. Trying CV2-DNN")
                    continue
                else:
                    raise ValueError("Error inititalizing Aligner")
            if plugin == "cv2_dnn":
                return

            try:
                err = None
                err = out_queue.get(True, 1)
            except QueueEmpty:
                pass
            if not err:
                break
            process.join()
            logger.error("Error initializing FAN. Trying CV2-DNN")

    @staticmethod
    def alignment_dict(image):
        """ Set the image to a dict for alignment """
        height, width = image.shape[:2]
        face = DetectedFace(x=0, w=width, y=0, h=height)
        face = face.to_bounding_box_dict()
        return {"image": image,
                "detected_faces": [face]}

    @staticmethod
    def get_landmarks(filename):
        """ Extract the face from a frame (If not alignments file found) """
        image = cv2_read_img(filename, raise_error=True)
        queue_manager.get_queue("in").put(Sort.alignment_dict(image))
        face = queue_manager.get_queue("out").get()
        landmarks = face["landmarks"][0]
        return landmarks

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

        logger.info("Done.")

    # Methods for sorting
    def sort_blur(self):
        """ Sort by blur amount """
        input_dir = self.args.input_dir

        logger.info("Sorting by blur...")
        img_list = [[img, self.estimate_blur(img)]
                    for img in
                    tqdm(self.find_images(input_dir),
                         desc="Loading",
                         file=sys.stdout)]
        logger.info("Sorting...")

        img_list = sorted(img_list, key=operator.itemgetter(1), reverse=True)

        return img_list

    def sort_face(self):
        """ Sort by face similarity """
        input_dir = self.args.input_dir

        logger.info("Sorting by face similarity...")

        images = np.array(self.find_images(input_dir))
        preds = np.array([self.vgg_face.predict(cv2_read_img(img, raise_error=True))
                          for img in tqdm(images, desc="loading", file=sys.stdout)])
        logger.info("Sorting. Depending on ths size of your dataset, this may take a few "
                    "minutes...")
        indices = self.vgg_face.sorted_similarity(preds, method="ward")
        img_list = images[indices]
        return img_list

    def sort_face_cnn(self):
        """ Sort by CNN similarity """
        self.launch_aligner()
        input_dir = self.args.input_dir

        logger.info("Sorting by face-cnn similarity...")
        img_list = []
        for img in tqdm(self.find_images(input_dir),
                        desc="Loading",
                        file=sys.stdout):
            landmarks = self.get_landmarks(img)
            img_list.append([img, np.array(landmarks)
                             if landmarks
                             else np.zeros((68, 2))])

        queue_manager.terminate_queues()
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
        """ Sort by CNN dissimilarity """
        self.launch_aligner()
        input_dir = self.args.input_dir

        logger.info("Sorting by face-cnn dissimilarity...")

        img_list = []
        for img in tqdm(self.find_images(input_dir),
                        desc="Loading",
                        file=sys.stdout):
            landmarks = self.get_landmarks(img)
            img_list.append([img, np.array(landmarks)
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

        logger.info("Sorting...")
        img_list = sorted(img_list, key=operator.itemgetter(2), reverse=True)

        return img_list

    def sort_face_yaw(self):
        """ Sort by yaw of face """
        self.launch_aligner()
        input_dir = self.args.input_dir

        img_list = []
        for img in tqdm(self.find_images(input_dir),
                        desc="Loading",
                        file=sys.stdout):
            landmarks = self.get_landmarks(img)
            img_list.append(
                [img, self.calc_landmarks_face_yaw(np.array(landmarks))])

        logger.info("Sorting by face-yaw...")
        img_list = sorted(img_list, key=operator.itemgetter(1), reverse=True)

        return img_list

    def sort_hist(self):
        """ Sort by histogram of face similarity """
        input_dir = self.args.input_dir

        logger.info("Sorting by histogram similarity...")

        img_list = [
            [img, cv2.calcHist([cv2_read_img(img, raise_error=True)], [0], None, [256], [0, 256])]
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

        logger.info("Sorting by histogram dissimilarity...")

        img_list = [
            [img,
             cv2.calcHist([cv2_read_img(img, raise_error=True)], [0], None, [256], [0, 256]), 0]
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

        logger.info("Sorting...")
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

        logger.info("Grouping by blur...")
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

    def group_face_cnn(self, img_list):
        """ Group into bins by CNN face similarity """
        logger.info("Grouping by face-cnn similarity...")

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

        logger.info("Grouping by face-yaw...")
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
        logger.info("Grouping by histogram...")

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
            src = img_list[i] if isinstance(img_list[i], str) else img_list[i][0]
            src_basename = os.path.basename(src)

            dst = os.path.join(output_dir, '{:05d}_{}'.format(i, src_basename))
            try:
                process_file(src, dst, self.changes)
            except FileNotFoundError as err:
                logger.error(err)
                logger.error('fail to rename %s', src)

        for i in tqdm(range(0, len(img_list)),
                      desc=description,
                      file=sys.stdout):
            renaming = self.set_renaming_method(self.args.log_changes)
            fname = img_list[i] if isinstance(img_list[i], str) else img_list[i][0]
            src, dst = renaming(fname, output_dir, i, self.changes)

            try:
                os.rename(src, dst)
            except FileNotFoundError as err:
                logger.error(err)
                logger.error('fail to rename %s', format(src))

        if self.args.log_changes:
            self.write_to_log(self.changes)

    def final_process_folders(self, bins):
        """ Move the files to folders """
        output_dir = self.args.output_dir

        process_file = self.set_process_file_method(self.args.log_changes,
                                                    self.args.keep_original)

        # First create new directories to avoid checking
        # for directory existence in the moving loop
        logger.info("Creating group directories.")
        for i in range(len(bins)):
            directory = os.path.join(output_dir, str(i))
            if not os.path.exists(directory):
                os.makedirs(directory)

        description = (
            "Copying into Groups" if self.args.keep_original
            else "Moving into Groups"
        )

        logger.info("Total groups found: %s", len(bins))
        for i in tqdm(range(len(bins)), desc=description, file=sys.stdout):
            for j in range(len(bins[i])):
                src = bins[i][j]
                src_basename = os.path.basename(src)

                dst = os.path.join(output_dir, str(i), src_basename)
                try:
                    process_file(src, dst, self.changes)
                except FileNotFoundError as err:
                    logger.error(err)
                    logger.error("Failed to move '%s' to '%s'", src, dst)

        if self.args.log_changes:
            self.write_to_log(self.changes)

    # Various helper methods
    def write_to_log(self, changes):
        """ Write the changes to log file """
        logger.info("Writing sort log to: '%s'", self.args.log_file_path)
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
        logger.info("Preparing to group...")
        if group_method == 'group_blur':
            temp_list = [[img, self.estimate_blur(cv2_read_img(img, raise_error=True))]
                         for img in
                         tqdm(self.find_images(input_dir),
                              desc="Reloading",
                              file=sys.stdout)]
        elif group_method == 'group_face_cnn':
            self.launch_aligner()
            temp_list = []
            for img in tqdm(self.find_images(input_dir),
                            desc="Reloading",
                            file=sys.stdout):
                landmarks = self.get_landmarks(img)
                temp_list.append([img, np.array(landmarks)
                                  if landmarks
                                  else np.zeros((68, 2))])
        elif group_method == 'group_face_yaw':
            self.launch_aligner()
            temp_list = []
            for img in tqdm(self.find_images(input_dir),
                            desc="Reloading",
                            file=sys.stdout):
                landmarks = self.get_landmarks(img)
                temp_list.append(
                    [img,
                     self.calc_landmarks_face_yaw(np.array(landmarks))])
        elif group_method == 'group_hist':
            temp_list = [
                [img,
                 cv2.calcHist([cv2_read_img(img, raise_error=True)], [0], None, [256], [0, 256])]
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
            current_img = sorted_list[i] if isinstance(sorted_list[i], str) else sorted_list[i][0]
            new_val_index = val_index_list.index(current_img)
            new_list.append([current_img, new_vals_list[new_val_index][1]])

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
    def estimate_blur(image_file):
        """
        Estimate the amount of blur an image has with the variance of the Laplacian.
        Normalize by pixel number to offset the effect of image size on pixel gradients & variance
        """
        image = cv2_read_img(image_file, raise_error=True)
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_map = cv2.Laplacian(image, cv2.CV_32F)
        score = np.var(blur_map) / np.sqrt(image.shape[0] * image.shape[1])
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
                def process_file(src, dst, changes):  # pylint: disable=unused-argument
                    """ Process file method if not logging changes
                        and keeping original """
                    copyfile(src, dst)

            else:
                def process_file(src, dst, changes):  # pylint: disable=unused-argument
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
            def renaming(src, output_dir, i, changes):  # pylint: disable=unused-argument
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
    def get_avg_score_faces_cnn(fl1, references):
        """ Return the average CNN similarity score
            between a face and reference image """
        scores = []
        for fl2 in references:
            score = np.sum(np.absolute((fl2 - fl1).flatten()))
            scores.append(score)
        return sum(scores) / len(scores)


def bad_args(args):  # pylint: disable=unused-argument
    """ Print help on bad arguments """
    PARSER.print_help()
    exit(0)


if __name__ == "__main__":
    __WARNING_STRING = "Important: face-cnn method will cause an error when "
    __WARNING_STRING += "this tool is called directly instead of through the "
    __WARNING_STRING += "tools.py command script."
    print(__WARNING_STRING)
    print("Images sort tool.\n")

    PARSER = FullHelpArgumentParser()
    SUBPARSER = PARSER.add_subparsers()
    SORT = cli.SortArgs(
        SUBPARSER, "sort", "Sort images using various methods.")
    PARSER.set_defaults(func=bad_args)
    ARGUMENTS = PARSER.parse_args()
    ARGUMENTS.func(ARGUMENTS)
