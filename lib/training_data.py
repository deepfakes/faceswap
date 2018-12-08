#!/usr/bin/env python3
""" Process training data for model training """

import logging
import os
from random import shuffle
import uuid

import cv2
import numpy as np
from scipy.interpolate import griddata

from lib.multithreading import MultiThread
from lib.queue_manager import queue_manager
from lib.umeyama import umeyama

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class TrainingDataGenerator():
    """ Generate training data for models """
    def __init__(self, random_transform_args, coverage,
                 scale=5, zoom=1, training_opts=None):
        logger.debug("Initializing %s: (random_transform_args: %s, coverage: %s, scale: %s, "
                     " zoom: %s, training_opts: %s", self.__class__.__name__,
                     random_transform_args, coverage, scale, zoom, training_opts)
        self.options = training_opts
        self.random_transform_args = random_transform_args
        self.training_opts = dict() if training_opts is None else training_opts
        self.coverage = coverage
        self.scale = scale
        self.zoom = zoom
        self.batchsize = 0
        logger.debug("Initialized %s", self.__class__.__name__)

    def minibatch_ab(self, images, batchsize, side, do_shuffle=True):
        """ Keep a queue filled to 8x Batch Size """
        logger.debug("Queue batches: (image_count: %s, batchsize: %s, side: '%s', do_shuffle: %s",
                     len(images), batchsize, side, do_shuffle)
        self.batchsize = batchsize
        q_name = str(uuid.uuid4())
        q_size = batchsize * 8
        # Don't use a multiprocessing queue because sometimes the MP Manager
        # borks on numpy arrays
        queue_manager.add_queue(q_name, maxsize=q_size, multiprocessing_queue=False)
        load_thread = MultiThread(self.load_batches, images, q_name, side, do_shuffle)
        load_thread.start()
        logger.debug("Batching to queue: (side: '%s', queue: '%s')", side, q_name)
        return self.minibatch(q_name, load_thread)

    def load_batches(self, images, q_name, side, do_shuffle=True):
        """ Load the epoch, warped images and target images to queue """
        logger.debug("Loading batch: (image_count: %s, q_name: '%s', side: '%s'. do_shuffle: %s)",
                     len(images), q_name, side, do_shuffle)
        epoch = 0
        queue = queue_manager.get_queue(q_name)
        self.validate_samples(images)
        while True:
            if do_shuffle:
                shuffle(images)
            for img in images:
                logger.trace("Putting to batch queue: (epoch: %s, q_name: '%s', side: '%s')",
                             epoch, q_name, side)
                queue.put((epoch, self.process_face(img, side)))
#                queue.put((epoch, np.float32(self.process_face(img, side))))
            epoch += 1
        logger.debug("Finished batching: (epoch: %s, q_name: '%s', side: '%s')",
                     epoch, q_name, side)

    def validate_samples(self, data):
        """ Check the total number of images against batchsize and return
            the total number of images """
        length = len(data)
        msg = ("Number of images is lower than batch-size (Note that too few "
               "images may lead to bad training). # images: {}, "
               "batch-size: {}".format(length, self.batchsize))
        assert length >= self.batchsize, msg

    def minibatch(self, q_name, load_thread):
        """ A generator function that yields epoch, batchsize of warped_img
            and batchsize of target_img from the load queue """
        logger.debug("Launching minibatch generator for queue: '%s'", q_name)
        queue = queue_manager.get_queue(q_name)
        while True:
            if load_thread.has_error:
                logger.debug("Thread error detected")
                break
            batch = list()
            for _ in range(self.batchsize):
                epoch, images = queue.get()
                for idx, image in enumerate(images):
                    if len(batch) < idx + 1:
                        batch.append(list())
                    batch[idx].append(image)
            batch = [epoch] + [np.float32(image) for image in batch]
            logger.trace("Yielding batch: (size: %s, queue:  '%s'", len(batch), q_name)
            yield batch
        logger.debug("Finished minibatch generator for queue: '%s'", q_name)
        load_thread.join()

    def process_face(self, filename, side):
        """ Load an image and perform transformation and warping """
        logger.trace("Process face: (filename: '%s', side: '%s'", filename, side)
        landmarks = self.training_opts.get("landmarks", None)
        landmarks = self.get_landmarks(filename, side, landmarks) if landmarks else None
        try:
            # pylint: disable=no-member
            image = self.color_adjust(cv2.imread(filename))
        except TypeError:
            raise Exception("Error while reading image", filename)

        image = self.random_transform(image)
        if not landmarks:
            retval = self.random_warp(image)
        else:
            image = self.add_alpha_channel(image)
            warped_img, target_img, mask_image = self.random_warp_landmarks(
                image,
                landmarks[0],
                landmarks[1])
            if np.random.random() < 0.5:
                warped_img = warped_img[:, ::-1]
                target_img = target_img[:, ::-1]
                mask_image = mask_image[:, ::-1]

            retval = warped_img, target_img, mask_image
        logger.trace("Processed face: (filename: '%s', side: '%s'", filename, side)
        return retval

    @staticmethod
    def get_landmarks(filename, side, landmarks):
        """ Return the landmarks for this face and the closest landmark
            for corresponding set """
        logger.trace("Retrieving landmarks: (filename: '%s', side: '%s'", filename, side)
        lm_key = os.path.splitext(os.path.basename(filename))[0]
        try:
            src_points = landmarks[side][lm_key]
        except KeyError:
            raise Exception("Landmarks not found for {}".format(filename))
        dst_points = landmarks["a"] if side == "b" else landmarks["b"]
        dst_points = list(dst_points.values())
        closest = (np.mean(np.square(src_points - dst_points),
                           axis=(1, 2))).argsort()[:10]
        closest = np.random.choice(closest)
        retval = src_points, dst_points[closest]
        logger.trace("Returning: (src_points: %s, dst_points: %s)", retval[0], retval[1])
        return retval

    @staticmethod
    def color_adjust(img):
        """ Color adjust RGB image """
        logger.trace("Color adjusting image")
        return img / 255.0

    def random_transform(self, image):
        """ Randomly transform an image """
        logger.trace("Randomly transforming image")
        height, width = image.shape[0:2]
        rotation_range = self.random_transform_args["rotation_range"]
        zoom_range = self.random_transform_args["zoom_range"]
        shift_range = self.random_transform_args["shift_range"]
        random_flip = self.random_transform_args["random_flip"]

        rotation = np.random.uniform(-rotation_range, rotation_range)
        scale = np.random.uniform(1 - zoom_range, 1 + zoom_range)
        tnx = np.random.uniform(-shift_range, shift_range) * width
        tny = np.random.uniform(-shift_range, shift_range) * height

        mat = cv2.getRotationMatrix2D(  # pylint: disable=no-member
            (width // 2, height // 2), rotation, scale)
        mat[:, 2] += (tnx, tny)
        result = cv2.warpAffine(  # pylint: disable=no-member
            image, mat, (width, height),
            borderMode=cv2.BORDER_REPLICATE)  # pylint: disable=no-member

        if np.random.random() < random_flip:
            result = result[:, ::-1]
        logger.trace("Randomly transformed image")
        return result

    def random_warp(self, image):
        """ get pair of random warped images from aligned face image """
        logger.trace("Randomly warping image")
        height, width = image.shape[0:2]
        assert height == width and height % 2 == 0

        range_ = np.linspace(height // 2 - self.coverage // 2,
                             height // 2 + self.coverage // 2, self.scale)
        mapx = np.broadcast_to(range_, (self.scale, self.scale))
        mapy = mapx.T

        mapx = mapx + np.random.normal(size=(self.scale, self.scale),
                                       scale=self.scale)
        mapy = mapy + np.random.normal(size=(self.scale, self.scale),
                                       scale=self.scale)

        interp_mapx = cv2.resize(  # pylint: disable=no-member
            mapx, (80 * self.zoom, 80 * self.zoom)
            )[8 * self.zoom:72 * self.zoom,
              8 * self.zoom:72 * self.zoom].astype('float32')
        interp_mapy = cv2.resize(  # pylint: disable=no-member
            mapy, (80 * self.zoom, 80 * self.zoom)
            )[8 * self.zoom:72 * self.zoom,
              8 * self.zoom:72 * self.zoom].astype('float32')

        warped_image = cv2.remap(  # pylint: disable=no-member
            image,
            interp_mapx,
            interp_mapy,
            cv2.INTER_LINEAR)  # pylint: disable=no-member

        src_points = np.stack([mapx.ravel(), mapy.ravel()], axis=-1)
        dst_points = np.mgrid[0:65 * self.zoom:16 * self.zoom,
                              0:65 * self.zoom:16 * self.zoom].T.reshape(-1,
                                                                         2)

        mat = umeyama(src_points, dst_points, True)[0:2]
        target_image = cv2.warpAffine(image,  # pylint: disable=no-member
                                      mat,
                                      (64 * self.zoom, 64 * self.zoom))

        logger.trace("Randomly warped image")
        return warped_image, target_image

    # TODO Unfix variables
    def random_warp_landmarks(self, image, src_points, dst_points):
        """ get warped image, target image and target mask """
        # pylint: disable=no-member
        logger.trace("Randomly warping landmarks")

        edge_anchors = [(0, 0), (0, 255), (255, 255), (255, 0),
                        (127, 0), (127, 255), (255, 127), (0, 127)]
        grid_x, grid_y = np.mgrid[0:255:256j, 0:255:256j]

        source = src_points
        destination = (dst_points.copy().astype("float") +
                       np.random.normal(size=(dst_points.shape),
                                        scale=2))
        destination = destination.astype("uint8")

        face_core = cv2.convexHull(np.concatenate([source[17:],
                                                   destination[17:]],
                                                  axis=0).astype(int))

        source = [(pty, ptx) for ptx, pty in source] + edge_anchors
        destination = [(pty, ptx) for ptx, pty in destination] + edge_anchors

        indicies_to_remove = set()
        for fpl in source, destination:
            for idx, (pty, ptx) in enumerate(fpl):
                if idx > 17:
                    break
                elif cv2.pointPolygonTest(face_core, (pty, ptx), False) >= 0:
                    indicies_to_remove.add(idx)

        for idx in sorted(indicies_to_remove, reverse=True):
            source.pop(idx)
            destination.pop(idx)

        grid_z = griddata(destination,
                          source,
                          (grid_x, grid_y),
                          method="linear")
        map_x = np.append([], [ar[:, 1] for ar in grid_z]).reshape(256, 256)
        map_y = np.append([], [ar[:, 0] for ar in grid_z]).reshape(256, 256)
        map_x_32 = map_x.astype('float32')
        map_y_32 = map_y.astype('float32')

        warped = cv2.remap(image[:, :, :3],
                           map_x_32,
                           map_y_32,
                           cv2.INTER_LINEAR,
                           cv2.BORDER_TRANSPARENT)
        target_mask = image[:, :, 3].reshape((256, 256, 1))
        target_image = image[:, :, :3]

        warped = cv2.resize(warped[128 - 120:128 + 120,
                                   128 - 120:128 + 120, :],
                            (64, 64),
                            cv2.INTER_AREA)
        target_image = cv2.resize(target_image[128 - 120:128 + 120,
                                               128 - 120:128 + 120, :],
                                  (64 * 2, 64 * 2),
                                  cv2.INTER_AREA)
        target_mask = cv2.resize(target_mask[128 - 120:128 + 120,
                                             128 - 120:128 + 120, :],
                                 (64 * 2, 64 * 2),
                                 cv2.INTER_AREA).reshape((64 * 2, 64 * 2, 1))

        logger.trace("Randomly warped image")
        return warped, target_image, target_mask

    @staticmethod
    def add_alpha_channel(image):
        """ Add an alpha channel to the image """
        logger.trace("Add alpha channel to image")
        ch_b, ch_g, ch_r = cv2.split(image)  # pylint: disable=no-member
        ch_a = np.ones(ch_b.shape, dtype=ch_b.dtype) * 50
        image_bgra = cv2.merge(  # pylint: disable=no-member
            (ch_b, ch_g, ch_r, ch_a))
        logger.trace("Added alpha channel to image")
        return image_bgra


def stack_images(images):
    """ Stack images """
    logger.debug("Stack images")

    def get_transpose_axes(num):
        if num % 2 == 0:
            logger.debug("Even number of images to stack")
            y_axes = list(range(1, num - 1, 2))
            x_axes = list(range(0, num - 1, 2))
        else:
            logger.debug("Odd number of images to stack")
            y_axes = list(range(0, num - 1, 2))
            x_axes = list(range(1, num - 1, 2))
        return y_axes, x_axes, [num - 1]

    images_shape = np.array(images.shape)
    new_axes = get_transpose_axes(len(images_shape))
    new_shape = [np.prod(images_shape[x]) for x in new_axes]
    logger.debug("Stacked images")
    return np.transpose(
        images,
        axes=np.concatenate(new_axes)
        ).reshape(new_shape)
