#!/usr/bin/env python3
"""
A tool that allows for sorting and grouping images in different ways.
"""

import os
import sys
import logging
import operator
import argparse
from shutil import copyfile
from itertools import cycle, zip_longest
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
import face_recognition
import cpbd  # pip install cpbd

# faceswap imports
from lib import Serializer
from lib.cli import FullHelpArgumentParser
from scripts.fsmedia import Utils
from scripts.extract import Extract
from .lib_alignments.media import ExtractedFaces, Frames
from . import cli

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Sort():
    """ Sorts folders of faces based on input criteria """
    # pylint: disable=no-member
    def __init__(self, arguments):
        logger.debug("Initializing %s: (args: %s)", self.__class__.__name__, arguments)
        self.args = arguments
        self.changes = None
        self.serializer = None

        # look for alignment file, if none, re-extract faces from images
        parser = argparse.ArgumentParser()
        parser.add_argument('--detector')
        parser.add_argument('--aligner')
        parser.add_argument('--skip_faces')
        parser.add_argument('--skip_existing')
        parser.parse_args(args=['--detector', 'mtcnn'], namespace=self.args)
        parser.parse_args(args=['--aligner', 'fan'], namespace=self.args)
        parser.parse_args(args=['--skip_faces', 'False'], namespace=self.args)
        parser.parse_args(args=['--skip_existing', 'True'], namespace=self.args)
        self.extractor = Extract(self.args)
        self.frames = Frames(self.args.input_dir)

    def process(self):
        """ Main processing function of the sort tool """

        # Setting default argument values that cannot be set by argparse

        # Set output dir to the same value as input dir
        # if the user didn't specify it.
        if self.args.output_dir.lower() == "_output_dir":
            self.args.output_dir = self.args.input_dir

        # If logging is enabled, prepare container
        if self.args.log_changes:
            Utils.set_verbosity(self.args.loglevel)
            self.changes = dict()

            # Assign default sort_log.json value if user didn't specify one
            if self.args.log_file_path == 'sort_log.json':
                self.args.log_file_path = os.path.join(self.args.input_dir,
                                                       'sort_log.json')

            # Set serializer based on logfile extension
            ext = os.path.splitext(self.args.log_file_path)[-1]
            self.serializer = Serializer.get_serializer_from_ext(ext)

        # Prepare sort, group and final process method names
        sort_method = self.args.sort_method.lower()
        group_method = self.args.group_method.lower()

        group_method = 'none'  # TODO implement grouping
        if group_method != 'none':
            sorted_imgs = self.sorting(group_method)
            list_groups = self.group(sorted_imgs)
            self.folders(list_groups)
            if sort_method != 'none':
                for imgs in list_groups:
                    sorted_imgs = self.sorting(sort_method, imgs)
                    self.rename(sorted_imgs)
        else:
            if sort_method != 'none':
                sorted_imgs = self.sorting(sort_method)
                self.rename(sorted_imgs)

        logger.info("Done.")

    # Methods for sorting
    def sorting(self, method, imgs=None):
        """ Sort by methodology """
        logger.info("Sorting by '%s'", method)

        # TODO finalize calc statistics using masked facial area instead of all image
        sorter = {'identity':             [self.__sort_face, False],
                  'hist_gray':            [self.__sort_hist, False],
                  'hist_luma':            [self.__sort_hist, False],
                  'hist_chroma_green':    [self.__sort_hist, False],
                  'hist_chroma_orange':   [self.__sort_hist, False],
                  'landmarks':            [self.__sort_landmarks, True],
                  'landmark_outliers':    [self.__score_angle, True],
                  'luma':                 [self.__score_channel, False],
                  'chroma_green':         [self.__score_channel, False],
                  'chroma_orange':        [self.__score_channel, False],
                  'yaw':                  [self.__score_angle, True],
                  'roll':                 [self.__score_angle, True],
                  'pitch':                [self.__score_angle, True],
                  'blur_quick':           [self.__score_blur, False],
                  'blur_cpbd':            [self.__score_blur, False],
                  'face_area':            [self.__score_area, True],
                  'identity_rarity':      [self.__score_face, False],
                  'hist_rarity_gray':     [self.__score_hist, False],
                  'hist_rarity_luma':     [self.__score_hist, False],
                  'hist_rarity_green':    [self.__score_hist, False],
                  'hist_rarity_orange':   [self.__score_hist, False],
                  'landmark_rarity':      [self.__score_landmarks, True],

                  # legacy parameters
                  'face':                 [self.__sort_face, False],
                  'face_dissim':          [self.__score_face, False],
                  'face_cnn':             [self.__sort_landmarks, True],
                  'face_cnn_dissim':      [self.__score_landmarks, True],
                  'hist':                 [self.__sort_hist, False],  # hist on blue only?
                  'hist_dissim':          [self.__score_hist, False]}  # hist on blue only?

        logger.info("Sorting...")
        if not imgs:
            imgs = self.load_images(sorter[method][1])
        img_list = sorter[method][0](imgs, method)
        if method not in ['landmarks', 'identity', 'hist_gray', 'hist_luma',
                          'hist_chroma_green', 'hist_chroma_orange']:
            img_list = sorted(img_list, key=operator.itemgetter(1), reverse=True)

        return img_list

    def load_images(self, need_landmarks):
        """ Sort by face identity similarity """
        need_landmarks = True if self.args.face_only else need_landmarks
        self.extractor.process()
        images = self.extractor.images
        alignments = self.extractor.alignments
        #img_loader = images.load()  # yield filename, image
        loader = alignments.yield_faces()
        extracts = ExtractedFaces(self.frames, alignments, size=256, align_eyes=False)
        imgs = []
        for frame_name, aligns, face_count, frame_fullname in loader:
            extracts.get_faces(frame_fullname)
            merged = self.merge_lists([frame_fullname], [0.], extracts.faces)
            for merges in merged:
                filename = Path(self.args.input_dir) / merges[0]
                if merges[2]:
                    landmarks = np.array(merges[2].landmarksXY, dtype='float32')
                    face_crop = merges[2].image.astype('float32')
                    imgs.append([filename, merges[1], face_crop, landmarks])
                else:
                    imgs.append([filename, merges[1], np.zeros((64, 64, 3), dtype='float32'), np.zeros((68, 2), dtype='float32')])
        return imgs

    @staticmethod
    def __sort_face(imgs, method):  # still testing all error cases with images of non-uniform size
        """ Sort by face identity similarity """
        ids = face_recognition.face_encodings
        distances = face_recognition.face_distance
        embeddings = [ids(item[2]) for item in tqdm(imgs, desc="Encoding faces", file=sys.stdout)]
        for i, ids in tqdm(enumerate(embeddings[:-1]), desc="Sorting", file=sys.stdout):
            if len(ids) != 0:
                scores = np.stack(np.array(distances(others[0], ids[0])) for others in embeddings[i+1:] if len(others) != 0)
                best = np.argmin(scores)
                imgs[i + 1], imgs[best] = imgs[best], imgs[i + 1]

        return imgs

    def __sort_hist(self, imgs, method):
        """ Sort by histogram similarity """
        cost = cv2.HISTCMP_BHATTACHARYYA
        images, value = self.prep_color(imgs, method)
        p_bar = tqdm(images, desc="Calcing hists", file=sys.stdout)
        hists = [cv2.calcHist([image], [value], None, [256], [0, 256]) for image in p_bar]
        for i, hist in tqdm(enumerate(hists[:-1]), desc="Sorting", file=sys.stdout):
            scores = np.stack(cv2.compareHist(others, hist, cost) for others in hists[i+1:])
            best = np.argmin(scores)
            imgs[i + 1], imgs[best] = imgs[best], imgs[i + 1]

        return imgs

    @staticmethod
    def __sort_landmarks(imgs, method):  # still testing all error cases with imgs of non-uniform size
        """ Sort by landmark similarity """
        for i, img in tqdm(enumerate(imgs[:-1]), desc="Sorting", file=sys.stdout):
            for marks in img[3]:
                mark = np.array(marks['landmarksXY'])
            rest = np.stack(o_marks['landmarksXY'] for items in imgs[i+1:] for o_marks in items[3])
            scores = np.sum(np.absolute(rest - mark))
            best = np.argmin(scores)
            imgs[i + 1], imgs[best] = imgs[best], imgs[i + 1]

        return imgs

    def __score_channel(self, imgs, method):
        """ Score by channel average intensity """
        images, value = self.prep_color(imgs, method)
        if isinstance(images, list):
            scores = [np.mean(img, axis=(0, 1))[value] for img in images]
        else:
            scores = np.mean(images, axis=(1, 2))[:, value]
        for img, score in zip(imgs, scores):
            img[1] = score

        return imgs

    def __score_angle(self, imgs, method):
        """ Score by estimated face pose angle """
        picker = {'pitch': 0, 'yaw':   1, 'roll':  2}
        for img in tqdm(imgs, desc="Scoring", file=sys.stdout):
            for faces, marks in zip(img[2], img[3]):
                if method == 'landmark_outliers':
                    inliers = self.face_pose(marks['landmarksXY'], faces, method)
                    score = len(inliers)
                    img[1] = score
                else:
                    pitch_yaw_roll = self.face_pose(marks['landmarksXY'], faces, method)
                    score = pitch_yaw_roll[picker[method]]
                    img[1] = score

        return imgs

    @staticmethod
    def __score_blur(imgs, method):
        """ Score by estimated blur """
        for img in tqdm(imgs, desc="Scoring", file=sys.stdout):
            image = cv2.cvtColor(img[2], cv2.COLOR_BGR2GRAY) if img[2].ndim == 3 else img[2]
            if method.endswith('quick'):
                pixel_size = np.sqrt(image.shape[0] * image.shape[1])
                score = np.var(cv2.Laplacian(image, cv2.CV_32F)) / pixel_size
            else:
                score = cpbd.compute(image)
            img[1] = score

        return imgs

    @staticmethod
    def __score_area(imgs, method):
        """
        Score by relative size of the face, as measured by the ratio
        of face pixels to total pixels in the image
        """
        for i, img in tqdm(enumerate(imgs), desc="Scoring", file=sys.stdout):
            height, width, _ = img[2].shape
            if height == 0:
                print(i)
            mask = np.zeros((height+1, width+1, 1), dtype='float32')
            hull = cv2.convexHull(img[3].astype('int32'))  # pylint: disable=no-member
            cv2.fillConvexPoly(mask, hull, 1.)  # pylint: disable=no-member
            score = np.count_nonzero(mask) / ((height+1) * (width+1))
            img[1] = score

        return imgs

    @staticmethod
    def __score_face(imgs, method):
        """ Score by face uniqueness """
        ids = face_recognition.face_encodings
        blank = np.zeros((128,), dtype = 'float32')
        p_bar = tqdm(imgs, desc="Encoding Faces", file=sys.stdout)
        embeddings = [ids(item[2].astype('uint8')) for item in p_bar]
        embeddings = np.stack(item[0] if len(item) > 0 else blank for item in embeddings)
        for i, identity in tqdm(enumerate(embeddings), desc="Scoring", file=sys.stdout):
            score = np.linalg.norm(embeddings - identity, axis=1)
            imgs[i][1] = np.sum(score)

        return imgs

    def __score_hist(self, imgs, method):
        """ Score by histogram uniqueness """
        cost = cv2.HISTCMP_BHATTACHARYYA
        images, value = self.prep_color(imgs, method)
        p_bar = tqdm(images, desc="Calcing hists", file=sys.stdout)
        hists = [cv2.calcHist([image], [value], None, [256], [0, 256]) for image in p_bar]
        for i, hist in tqdm(enumerate(hists), desc="Scoring", file=sys.stdout):
            score = np.stack(cv2.compareHist(others, hist, cost) for others in hists)
            imgs[i][1] = np.sum(score)

        return imgs

    @staticmethod
    def __score_landmarks(imgs, method):
        ''' Score by landmark uniqueness '''
        for img in tqdm(imgs, desc="Scoring", file=sys.stdout):
            rest = np.stack(others[3] for others in imgs)
            score = np.sum(np.square(rest - img[3]))
            img[1] = score

        return imgs

    def prep_color(self, imgs, method):
        """ Helper function to construct histogram in proper colorspace """
        picker = {'gray': 0, 'dissim': 0, 'luma': 0, 'green': 1, 'orange': 2}
        value = next(v for (k, v) in picker.items() if method.endswith(k))
        shape_diff = np.sum(np.array(img[2].shape) - np.array(imgs[0][2]).shape for img in imgs)
        all_same_size = False if any(shape_diff) != 0 else True

        if all_same_size:
            if method.endswith('gray'):
                bgr_to_gray = [0.114, 0.587, 0.299]
                path = np.einsum_path('hijk, k -> hij', imgs[:2][2], bgr_to_gray, optimize='optimal')[0]
                images = np.einsum('hijk, k -> hij', imgs[:][2], bgr_to_gray, optimize=path).astype('float32')
            else:
                rgb = np.stack(img[2] for img in imgs)[..., ::-1] / 255.0
                img_array = self.rgb_to_ycocg(rgb, single=False) * 255.0
                if not method.endswith('luma'):
                    img_array = img_array + 127.5
        else:
            if method.endswith('gray'):
                bgr_to_gray = [0.114, 0.587, 0.299]
                images = [np.einsum('ijk, k -> ij', img[2], bgr_to_gray, optimize='greedy').astype('float32') for img in imgs]
            else:
                images = [self.rgb_to_ycocg(img[2][..., ::-1] / 255.0, single=True) * 255.0 for img in imgs]
                if not method.endswith('luma'):
                    images = [img + 127.5 for img in images]

        return images, value

    # Methods for grouping
    # TODO incorporate the UMAP version of grouping
    def group(self, img_list):
        '''
            num_bins = self.args.num_bins
            return bins

        def group_blur(self, img_list):

        def group_yaw(self, img_list):

        def group_identity(self, img_list):

        def group_landmarks(self, img_list):

        def group_histogram(self, img_list):
        '''
        print(self.args.num_bins)

        for img in img_list:
            yield img

    # Final process methods
    def rename(self, img_list):
        """ Rename the files """
        note = "Copying & Renaming" if self.args.keep_original else "Moving & Renaming"
        progress_bar = tqdm(enumerate(img_list), desc=note, leave=False, file=sys.stdout)
        any(self.process_file(img[0], i, self.args.output_dir, img[1]) for i, img in progress_bar)
        if self.args.log_changes:
            self.write_to_log(self.changes)

    def folders(self, bins):
        # TODO simpify like renaming
        """ Move the files to folders """
        o_dir = self.args.output_dir
        process_file = self.set_process_file_method(self.args.log_changes,
                                                    self.args.keep_original)

        # First create new directories to avoid checking
        # for directory existence in the moving loop
        logger.info("Creating group directories.")
        any(os.makedirs(os.path.join(o_dir, str(i)), exist_ok=True) for i in len(bins))

        note = "Copying into Groups" if self.args.keep_original else "Moving into Groups"

        logger.info("Total groups found: %s", len(bins))
        srcs = [[str(i), os.path.basename(x)] for x in group for i, group in enumerate(bins)]
        dsts = [os.path.join(o_dir, item[0], item[1]) for item in srcs]
        progress_bar = tqdm(zip(srcs, dsts), desc=note, file=sys.stdout)

        try:
            any(process_file(src, dst, self.changes) for src, dst in progress_bar)
        except FileNotFoundError as err:
            logger.error(err)
            logger.error("Failed to move '%s' to '%s'", srcs[0], dsts[0])  # TODO actual error file

        if self.args.log_changes:
            self.write_to_log(self.changes)

    # Various helper methods
    def write_to_log(self, changes):
        """ Write the changes to log file """
        logger.info("Writing sort log to: '%s'", self.args.log_file_path)
        with open(self.args.log_file_path, 'w') as lfile:
            lfile.write(self.serializer.marshal(changes))

    def process_file(self, src, i, output_dir, score):
        """ Process file method with logging changes and copying/renaming original """
        try:
            basename = os.path.basename(src)
            i = i
            # dst = os.path.join(output_dir, '{:05d}{}'.format(i, os.path.splitext(basename)[1]))
            dst = os.path.join(output_dir, '{:05f}{}'.format(score, os.path.splitext(basename)[1]))
            if self.args.log_changes:
                self.changes[src] = dst
            if self.args.keep_original:
                copyfile(src, dst)
            else:
                os.rename(src, dst)
        except FileNotFoundError as err:
            logger.error(err)
            logger.error('fail to rename %s', src)

    @staticmethod
    def merge_lists(*iterables, empty_default=None):
        """ Merge arbitrary numbers of lists, padding to the longest length """
        cycles = [cycle(i) for i in iterables]
        for _ in zip_longest(*iterables):
            yield tuple(next(i, empty_default) for i in cycles)

    @staticmethod
    def rgb_to_ycocg(rgb_images, single=False):
        """ RGB to YCoCG color space, efficient conversion and decorrelated channels """
        rgb_to_ycocg = np.array([[.25, .5, .25], [.5, 0., -.5], [-.25, .5, -.25]])
        if single:
            path = 'greedy'
        else:
            path = np.einsum_path('ij,...j', rgb_to_ycocg, rgb_images[:2], optimize='optimal')[0]
        converted = np.einsum('ij,...j', rgb_to_ycocg, rgb_images, optimize=path).astype('float32')

        return converted

    @staticmethod
    def ycocg_to_rgb(ycocg_images, single=False):
        """ YCoCG to RGB color space, efficient conversion and decorrelated channels """
        ycocg_to_rgb = np.array([[1., 1., -1.], [1., 0., 1.], [1., -1., -1.]])
        if single:
            path = 'greedy'
        else:
            path = np.einsum_path('ij,...j', ycocg_to_rgb, ycocg_images[:2], optimize='optimal')[0]
        converted = np.einsum('ij,...j', ycocg_to_rgb, ycocg_images, optimize=path).astype('float32')

        return converted

    @staticmethod
    def face_pose(landmarks, img, method):
        """ Given a set of face landmarks, find the Euler angles of the face pose """

        object_pts = np.float32([[-73.393523, 29.801432, 47.667532],
                                 [-72.775014, 10.949766, 45.909403],
                                 [-70.533638, -7.929818, 44.84258],
                                 [-66.850058, -26.07428, 43.141114],
                                 [-59.790187, -42.56439, 38.635298],
                                 [-48.368973, -56.48108, 30.750622],
                                 [-34.121101, -67.246992, 18.456453],
                                 [-17.875411, -75.056892, 3.609035],
                                 [0.098749, -77.061286, -0.881698],
                                 [17.477031, -74.758448, 5.181201],
                                 [32.648966, -66.929021, 19.176563],
                                 [46.372358, -56.311389, 30.77057],
                                 [57.34348, -42.419126, 37.628629],
                                 [64.388482, -25.45588, 40.886309],
                                 [68.212038, -6.990805, 42.281449],
                                 [70.486405, 11.666193, 44.142567],
                                 [71.375822, 30.365191, 47.140426],
                                 [-61.119406, 49.361602, 14.254422],
                                 [-51.287588, 58.769795, 7.268147],
                                 [-37.8048, 61.996155, 0.442051],
                                 [-24.022754, 61.033399, -6.606501],
                                 [-11.635713, 56.686759, -11.967398],
                                 [12.056636, 57.391033, -12.051204],
                                 [25.106256, 61.902186, -7.315098],
                                 [38.338588, 62.777713, -1.022953],
                                 [51.191007, 59.302347, 5.349435],
                                 [60.053851, 50.190255, 11.615746],
                                 [0.65394, 42.19379, -13.380835],
                                 [0.804809, 30.993721, -21.150853],
                                 [0.992204, 19.944596, -29.284036],
                                 [1.226783, 8.414541, -36.94806],
                                 [-14.772472, -2.598255, -20.132003],
                                 [-7.180239, -4.751589, -23.536684],
                                 [0.55592, -6.5629, -25.944448],
                                 [8.272499, -4.661005, -23.695741],
                                 [15.214351, -2.643046, -20.858157],
                                 [-46.04729, 37.471411, 7.037989],
                                 [-37.674688, 42.73051, 3.021217],
                                 [-27.883856, 42.711517, 1.353629],
                                 [-19.648268, 36.754742, -0.111088],
                                 [-28.272965, 35.134493, -0.147273],
                                 [-38.082418, 34.919043, 1.476612],
                                 [19.265868, 37.032306, -0.665746],
                                 [27.894191, 43.342445, 0.24766],
                                 [37.437529, 43.110822, 1.696435],
                                 [45.170805, 38.086515, 4.894163],
                                 [38.196454, 35.532024, 0.282961],
                                 [28.764989, 35.484289, -1.172675],
                                 [-28.916267, -28.612716, -2.24031],
                                 [-17.533194, -22.172187, -15.934335],
                                 [-6.68459, -19.029051, -22.611355],
                                 [0.381001, -20.721118, -23.748437],
                                 [8.375443, -19.03546, -22.721995],
                                 [18.876618, -22.394109, -15.610679],
                                 [28.794412, -28.079924, -3.217393],
                                 [19.057574, -36.298248, -14.987997],
                                 [8.956375, -39.634575, -22.554245],
                                 [0.381549, -40.395647, -23.591626],
                                 [-7.428895, -39.836405, -22.406106],
                                 [-18.160634, -36.677899, -15.121907],
                                 [-24.37749, -28.677771, -4.785684],
                                 [-6.897633, -25.475976, -20.893742],
                                 [0.340663, -26.014269, -22.220479],
                                 [8.444722, -25.326198, -21.02552],
                                 [24.474473, -28.323008, -5.712776],
                                 [8.449166, -30.596216, -20.671489],
                                 [0.205322, -31.408738, -21.90367],
                                 [-7.198266, -30.844876, -20.328022]])

        def calc_matrix(img, object_pts, image_pts):

            def camera_internals(img, fov_angle=45):
                """ Estimate of default camera internals """
                focal_length_x = (img.shape[1] / 2) / np.tan((fov_angle / 180 * np.pi) / 2)
                focal_length_y = (img.shape[0] / 2) / np.tan((fov_angle / 180 * np.pi) / 2)
                dist = np.zeros((4, 1), dtype='float32')
                cam = np.array([[focal_length_x, 0, img.shape[1]/2],
                                [0, focal_length_y, img.shape[0]/2],
                                [0, 0, 1]], dtype='float32')
                return cam, dist

            cam, dist = camera_internals(img)

            # robustly find pose with default camera intrinics and exclude outliers
            kwargs = {'objectPoints':      object_pts,
                      'imagePoints':       image_pts,
                      'cameraMatrix':      cam,
                      'distCoeffs':        dist,
                      'flags':             cv2.SOLVEPNP_ITERATIVE}
            _, _, _, inliers = cv2.solvePnPRansac(**kwargs)

            # determine camera intinsics using only landmarks that can be modeled accurately
            flat_inliers = [item for sublist in inliers for item in sublist]
            kwargs = {'objectPoints':      [np.ascontiguousarray(object_pts[flat_inliers])],
                      'imagePoints':       [np.ascontiguousarray(image_pts[flat_inliers])],
                      'imageSize':         (img.shape[1], img.shape[0]),
                      'cameraMatrix':      cam,
                      'distCoeffs':        dist,
                      'flags':             cv2.CALIB_USE_INTRINSIC_GUESS}
            _, cam, dist, rot_vects, trans_vects = cv2.calibrateCamera(**kwargs)

            # re-evaluate pose with estimated camera intrinsics
            kwargs = {'objectPoints':      object_pts,
                      'imagePoints':       image_pts,
                      'cameraMatrix':      cam,
                      'distCoeffs':        dist,
                      'rvec':              rot_vects[0],
                      'tvec':              trans_vects[0],
                      'flags':             cv2.SOLVEPNP_ITERATIVE,
                      'useExtrinsicGuess': True}
            _, rot_vect, tran_vect, inliers = cv2.solvePnPRansac(**kwargs)

            return rot_vect, tran_vect, inliers

        def rot_matrix_to_euler(rot_mat, tran_vect):
            """ Calculate Euler Angles ( in degrees ) from the rotation matrix using cv2 """
            proj_matrix = np.hstack((rot_mat, tran_vect))
            e_angles = -cv2.decomposeProjectionMatrix(proj_matrix)[6]
            pitch = 180 - e_angles[0, 0] if e_angles[0, 0] > 0 else -180 - e_angles[0, 0]
            yaw = -e_angles[1, 0]
            roll = e_angles[2, 0]

            return pitch, yaw, roll

        image_pts = np.float32(landmarks).reshape(68, 1, 2)  # pylint: disable=too-many-function-args
        object_pts = object_pts.reshape(68, 1, 3)  # pylint: disable=too-many-function-args
        rot_vect, tran_vect, inliers = calc_matrix(img, object_pts, image_pts)

        if method == 'landmark_outliers':
            return inliers

        rot_mat = cv2.Rodrigues(rot_vect)[0]
        return rot_matrix_to_euler(rot_mat, tran_vect)


if __name__ == "__main__":
    __WARNING_STRING = "Important: face-cnn method will cause an error when "
    __WARNING_STRING += "this tool is called directly instead of through the "
    __WARNING_STRING += "tools.py command script."
    print(__WARNING_STRING)
    print("Images sort tool.\n")

    PARSER = FullHelpArgumentParser()
    SUBPARSER = PARSER.add_subparsers()
    SORT = cli.SortArgs(SUBPARSER, "sort", "Sort images using various methods.")
    try:
        ARGUMENTS = PARSER.parse_args()
    except SystemExit as err:
        if err.code == 2:
            PARSER.print_help()
        exit(0)
    ARGUMENTS.func(ARGUMENTS)
