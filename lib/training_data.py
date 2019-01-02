#!/usr/bin/env python3
""" Process training data for model training """

import logging
import uuid

from hashlib import sha1
from random import shuffle

import cv2
import numpy as np
from scipy.interpolate import griddata

from lib.multithreading import MultiThread
from lib.queue_manager import queue_manager
from lib.umeyama import umeyama

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class TrainingDataGenerator():
    """ Generate training data for models """
    def __init__(self, transform_kwargs, training_opts=dict()):
        logger.debug("Initializing %s: (transform_kwargs: %s, training_opts: %s)",
                     self.__class__.__name__, transform_kwargs,
                     {key: val for key, val in training_opts.items() if key != "landmarks"})
        self.batchsize = 0
        self.training_opts = training_opts
        self.processing = ImageManipulation(**transform_kwargs)
        logger.debug("Initialized %s", self.__class__.__name__)

    def minibatch_ab(self, images, batchsize, side, do_shuffle=True):
        """ Keep a queue filled to 8x Batch Size """
        logger.debug("Queue batches: (image_count: %s, batchsize: %s, side: '%s', do_shuffle: %s)",
                     len(images), batchsize, side, do_shuffle)
        self.batchsize = batchsize
        q_name = str(uuid.uuid4())
        q_size = batchsize * 8
        # Don't use a multiprocessing queue because sometimes the MP Manager borks on numpy arrays
        queue_manager.add_queue(q_name, maxsize=q_size, multiprocessing_queue=False)
        load_thread = MultiThread(self.load_batches, images, q_name, side, do_shuffle)
        load_thread.start()
        logger.debug("Batching to queue: (side: '%s', queue: '%s')", side, q_name)
        return self.minibatch(q_name, load_thread)

    def load_batches(self, images, q_name, side, do_shuffle=True):
        """ Load the warped images and target images to queue """
        logger.debug("Loading batch: (image_count: %s, q_name: '%s', side: '%s'. do_shuffle: %s)",
                     len(images), q_name, side, do_shuffle)
        epoch = 0
        queue = queue_manager.get_queue(q_name)
        self.validate_samples(images)
        while True:
            if do_shuffle:
                shuffle(images)
            for img in images:
                logger.trace("Putting to batch queue: (q_name: '%s', side: '%s')", q_name, side)
                queue.put(self.process_face(img, side))
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
                images = queue.get()
                for idx, image in enumerate(images):
                    if len(batch) < idx + 1:
                        batch.append(list())
                    batch[idx].append(image)
            batch = [np.float32(image) for image in batch]
            logger.trace("Yielding batch: (size: %s, queue:  '%s'", len(batch), q_name)
            yield batch
        logger.debug("Finished minibatch generator for queue: '%s'", q_name)
        load_thread.join()

    def process_face(self, filename, side):
        """ Load an image and perform transformation and warping """
        logger.trace("Process face: (filename: '%s', side: '%s'", filename, side)
        try:
            image = cv2.imread(filename)  # pylint: disable=no-member
        except TypeError:
            raise Exception("Error while reading image", filename)

        landmarks = self.training_opts.get("landmarks", None)
        landmarks = self.get_landmarks(filename, image, side, landmarks) if landmarks else None

        image = self.processing.color_adjust(image)
        image = self.processing.random_transform(image)

        mask = None
        if self.training_opts.get("use_mask", False):
            image = self.processing.add_alpha_channel(image)
            mask = image[:, :, 3].reshape((image.shape[0], image.shape[1], 1))

        if not landmarks:
            processed = self.processing.random_warp(image, mask)
        else:
            processed = self.processing.random_warp_landmarks(image,
                                                              mask,
                                                              landmarks[0],
                                                              landmarks[1])

        retval = self.processing.do_random_flip(processed)
        logger.trace("Processed face: (filename: '%s', side: '%s'", filename, side)
        return retval

    @staticmethod
    def get_landmarks(filename, image, side, landmarks):
        """ Return the landmarks for this face and the closest landmark
            for corresponding set """
        logger.trace("Retrieving landmarks: (filename: '%s', side: '%s'", filename, side)
        lm_key = sha1(image).hexdigest()
        try:
            src_points = landmarks[side][lm_key]
        except KeyError:
            raise Exception("Landmarks not found for hash: '{}' file: '{}'".format(lm_key,
                                                                                   filename))
        dst_points = landmarks["a"] if side == "b" else landmarks["b"]
        dst_points = list(dst_points.values())
        closest = (np.mean(np.square(src_points - dst_points),
                           axis=(1, 2))).argsort()[:10]
        closest = np.random.choice(closest)
        retval = src_points, dst_points[closest]
        logger.trace("Returning: (src_points: %s, dst_points: %s)", retval[0], retval[1])
        return retval


class ImageManipulation():
    """ Manipulations to be performed on training images """
    def __init__(self, rotation_range=10, zoom_range=0.05, shift_range=0.05, random_flip=0.4,
                 zoom=1, coverage=160, scale=5):
        """ rotation_range: Used for random transform
            zoom_range: Used for random transform
            shift_range: Used for random transform
            random_flip: Float between 0 and 1. Chance to flip the image
            zoom: Used for random transform and random warp
            coverage: Used for random warp
            scale: Used for random warp
        """
        logger.debug("Initializing %s: (rotation_range: %s, zoom_range: %s, shift_range: %s, "
                     "random_flip: %s, zoom: %s, coverage: %s, scale: %s)",
                     self.__class__.__name__, rotation_range, zoom_range, shift_range, random_flip,
                     zoom, coverage, scale)
        # Transform args
        self.rotation_range = rotation_range
        self.zoom_range = zoom_range
        self.shift_range = shift_range
        self.random_flip = random_flip
        # Transform and Warp args
        self.zoom = zoom
        # Warp args
        self.coverage = coverage
        self.scale = scale
        logger.debug("Initialized %s", self.__class__.__name__)

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

    @staticmethod
    def color_adjust(img):
        """ Color adjust RGB image """
        logger.trace("Color adjusting image")
        return img / 255.0

    def random_transform(self, image):
        """ Randomly transform an image """
        logger.trace("Randomly transforming image")
        height, width = image.shape[0:2]

        rotation = np.random.uniform(-self.rotation_range, self.rotation_range)
        scale = np.random.uniform(1 - self.zoom_range, 1 + self.zoom_range)
        tnx = np.random.uniform(-self.shift_range, self.shift_range) * width
        tny = np.random.uniform(-self.shift_range, self.shift_range) * height

        mat = cv2.getRotationMatrix2D(  # pylint: disable=no-member
            (width // 2, height // 2), rotation, scale)
        mat[:, 2] += (tnx, tny)
        result = cv2.warpAffine(  # pylint: disable=no-member
            image, mat, (width, height),
            borderMode=cv2.BORDER_REPLICATE)  # pylint: disable=no-member

        logger.trace("Randomly transformed image")
        return result

    def random_warp(self, image, mask):
        """ get pair of random warped images from aligned face image """
        logger.trace("Randomly warping image")
        height, width = image.shape[0:2]
        assert height == width and height % 2 == 0

        range_ = np.linspace(height // 2 - self.coverage // 2,
                             height // 2 + self.coverage // 2, self.scale)
        mapx = np.broadcast_to(range_, (self.scale, self.scale))
        mapy = mapx.T

        mapx = mapx + np.random.normal(size=(self.scale, self.scale), scale=self.scale)
        mapy = mapy + np.random.normal(size=(self.scale, self.scale), scale=self.scale)

        interp_mapx = cv2.resize(  # pylint: disable=no-member
            mapx, (80 * self.zoom, 80 * self.zoom))[8 * self.zoom:72 * self.zoom,
                                                    8 * self.zoom:72 * self.zoom].astype('float32')
        interp_mapy = cv2.resize(  # pylint: disable=no-member
            mapy, (80 * self.zoom, 80 * self.zoom))[8 * self.zoom:72 * self.zoom,
                                                    8 * self.zoom:72 * self.zoom].astype('float32')

        warped_image = cv2.remap(  # pylint: disable=no-member
            image, interp_mapx, interp_mapy, cv2.INTER_LINEAR)  # pylint: disable=no-member

        src_points = np.stack([mapx.ravel(), mapy.ravel()], axis=-1)
        dst_points = np.mgrid[0:65 * self.zoom:16 * self.zoom,
                              0:65 * self.zoom:16 * self.zoom].T.reshape(-1, 2)

        mat = umeyama(src_points, dst_points, True)[0:2]
        target_image = cv2.warpAffine(  # pylint: disable=no-member
            image, mat, (64 * self.zoom, 64 * self.zoom))

        if mask is None:
            retval = warped_image, target_image
        else:
            target_mask = cv2.warpAffine(  # pylint: disable=no-member
                mask, mat, (64 * self.zoom, 64 * self.zoom))
            retval = warped_image, target_image, target_mask

        logger.trace("Randomly warped image")
        return retval

    def random_warp_landmarks(self, image, mask, src_points, dst_points):
        """ get warped image, target image and target mask
            From DFAKER plugin """
        logger.trace("Randomly warping landmarks")
        size = image.shape[0]
        p_mx = size - 1
        p_hf = (size // 2) - 1

        edge_anchors = [(0, 0), (0, p_mx), (p_mx, p_mx), (p_mx, 0),
                        (p_hf, 0), (p_hf, p_mx), (p_mx, p_hf), (0, p_hf)]
        grid_x, grid_y = np.mgrid[0:p_mx:complex(size), 0:p_mx:complex(size)]

        source = src_points
        destination = (dst_points.copy().astype("float") +
                       np.random.normal(size=(dst_points.shape), scale=2))
        destination = destination.astype("uint8")

        face_core = cv2.convexHull(np.concatenate(  # pylint: disable=no-member
            [source[17:], destination[17:]], axis=0).astype(int))

        source = [(pty, ptx) for ptx, pty in source] + edge_anchors
        destination = [(pty, ptx) for ptx, pty in destination] + edge_anchors

        indicies_to_remove = set()
        for fpl in source, destination:
            for idx, (pty, ptx) in enumerate(fpl):
                if idx > 17:
                    break
                elif cv2.pointPolygonTest(face_core,  # pylint: disable=no-member
                                          (pty, ptx),
                                          False) >= 0:
                    indicies_to_remove.add(idx)

        for idx in sorted(indicies_to_remove, reverse=True):
            source.pop(idx)
            destination.pop(idx)

        grid_z = griddata(destination,
                          source,
                          (grid_x, grid_y),
                          method="linear")
        map_x = np.append([], [ar[:, 1] for ar in grid_z]).reshape(size, size)
        map_y = np.append([], [ar[:, 0] for ar in grid_z]).reshape(size, size)
        map_x_32 = map_x.astype('float32')
        map_y_32 = map_y.astype('float32')

        warped_image = cv2.remap(image[:, :, :3],  # pylint: disable=no-member
                                 map_x_32,
                                 map_y_32,
                                 cv2.INTER_LINEAR,  # pylint: disable=no-member
                                 cv2.BORDER_TRANSPARENT)  # pylint: disable=no-member
        target_image = image[:, :, :3]

        pad_lt = (64 * self.zoom) - (60 * self.zoom)
        pad_rb = (64 * self.zoom) + (60 * self.zoom)

        warped_image = cv2.resize(  # pylint: disable=no-member
            warped_image[pad_lt:pad_rb, pad_lt:pad_rb, :],
            (64, 64),
            cv2.INTER_AREA)  # pylint: disable=no-member
        target_image = cv2.resize(  # pylint: disable=no-member
            target_image[pad_lt:pad_rb, pad_lt:pad_rb, :],
            (64 * self.zoom, 64 * self.zoom),
            cv2.INTER_AREA)  # pylint: disable=no-member
        if mask is None:
            retval = warped_image, target_image
        else:
            target_mask = cv2.resize(  # pylint: disable=no-member
                mask[pad_lt:pad_rb, pad_lt:pad_rb, :],
                (64 * self.zoom, 64 * self.zoom),
                cv2.INTER_AREA)  # pylint: disable=no-member
            target_mask = target_mask.reshape((64 * self.zoom, 64 * self.zoom, 1))
            retval = warped_image, target_image, target_mask
        logger.trace("Randomly warped image")
        return retval

    def do_random_flip(self, images):
        """ Perform flip on images if random number is within threshold """
        logger.trace("Randomly flipping image")
        if np.random.random() < self.random_flip:
            logger.trace("Flip within threshold. Flipping")
            retval = [image[:, ::-1] for image in images]
        else:
            logger.trace("Flip outside threshold. Not Flipping")
            retval = images
        logger.trace("Randomly flipped image")
        return retval


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
