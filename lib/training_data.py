#!/usr/bin/env python3
""" Process training data for model training """

import logging

from hashlib import sha1
from random import shuffle

import cv2
import numpy as np
from scipy.interpolate import griddata

from lib.model import masks
from lib.multithreading import MultiThread
from lib.queue_manager import queue_manager
from lib.umeyama import umeyama

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

<<<<<<< HEAD
# TODO Add source, dest points to random warp half and ability to
# not have landmarks to random warp full


class TrainingDataGenerator():
    """ Generate training data for models """
    def __init__(self, transform_kwargs, training_opts=dict()):
        logger.debug("Initializing %s: (transform_kwargs: %s, training_opts: %s)",
                     self.__class__.__name__, transform_kwargs,
                     {key: val for key, val in training_opts.items() if key != "landmarks"})
        self.batchsize = 0
        self.training_opts = training_opts
        self.full_face = self.training_opts.get("full_face", False)
        self.mask_function = self.set_mask_function()
        self.processing = ImageManipulation(**transform_kwargs)
=======

class TrainingDataGenerator():
    """ Generate training data for models """
    def __init__(self, model_input_size, model_output_size, training_opts):
        logger.debug("Initializing %s: (model_input_size: %s, model_output_shape: %s, "
                     "training_opts: %s, landmarks: %s)",
                     self.__class__.__name__, model_input_size, model_output_size,
                     {key: val for key, val in training_opts.items() if key != "landmarks"},
                     bool(training_opts.get("landmarks", None)))
        self.batchsize = 0
        self.model_input_size = model_input_size
        self.training_opts = training_opts
        self.mask_function = self.set_mask_function()
        self.landmarks = self.training_opts.get("landmarks", None)

        self.processing = ImageManipulation(model_input_size,
                                            model_output_size,
                                            training_opts.get("coverage_ratio", 0.625))
>>>>>>> 60e0099c4d88a551b33592bf5126ab96bd5dc5ae
        logger.debug("Initialized %s", self.__class__.__name__)

    def set_mask_function(self):
        """ Set the mask function to use if using mask """
        mask_type = self.training_opts.get("mask_type", None)
        if mask_type:
            logger.debug("Mask type: '%s'", mask_type)
            mask_func = getattr(masks, mask_type)
        else:
            mask_func = None
        logger.debug("Mask function: %s", mask_func)
        return mask_func

<<<<<<< HEAD
    def minibatch_ab(self, images, batchsize, side, do_shuffle=True):
        """ Keep a queue filled to 8x Batch Size """
        logger.debug("Queue batches: (image_count: %s, batchsize: %s, side: '%s', do_shuffle: %s)",
                     len(images), batchsize, side, do_shuffle)
        self.batchsize = batchsize
        q_name = "train_{}".format(side)
        q_size = batchsize * 8
        # Don't use a multiprocessing queue because sometimes the MP Manager borks on numpy arrays
        queue_manager.add_queue(q_name, maxsize=q_size, multiprocessing_queue=False)
        load_thread = MultiThread(self.load_batches, images, q_name, side, do_shuffle)
=======
    def minibatch_ab(self, images, batchsize, side, do_shuffle=True, is_timelapse=False):
        """ Keep a queue filled to 8x Batch Size """
        logger.debug("Queue batches: (image_count: %s, batchsize: %s, side: '%s', do_shuffle: %s, "
                     "is_timelapse: %s)", len(images), batchsize, side, do_shuffle, is_timelapse)
        self.batchsize = batchsize
        q_name = "timelapse_{}".format(side) if is_timelapse else "train_{}".format(side)
        q_size = batchsize * 8
        # Don't use a multiprocessing queue because sometimes the MP Manager borks on numpy arrays
        queue_manager.add_queue(q_name, maxsize=q_size, multiprocessing_queue=False)
        load_thread = MultiThread(self.load_batches,
                                  images,
                                  q_name,
                                  side,
                                  is_timelapse,
                                  do_shuffle)
>>>>>>> 60e0099c4d88a551b33592bf5126ab96bd5dc5ae
        load_thread.start()
        logger.debug("Batching to queue: (side: '%s', queue: '%s')", side, q_name)
        return self.minibatch(q_name, load_thread)

<<<<<<< HEAD
    def load_batches(self, images, q_name, side, do_shuffle=True):
        """ Load the warped images and target images to queue """
        logger.debug("Loading batch: (image_count: %s, q_name: '%s', side: '%s'. do_shuffle: %s)",
                     len(images), q_name, side, do_shuffle)
=======
    def load_batches(self, images, q_name, side, is_timelapse, do_shuffle=True):
        """ Load the warped images and target images to queue """
        logger.debug("Loading batch: (image_count: %s, q_name: '%s', side: '%s', "
                     "is_timelapse: %s, do_shuffle: %s)",
                     len(images), q_name, side, is_timelapse, do_shuffle)
>>>>>>> 60e0099c4d88a551b33592bf5126ab96bd5dc5ae
        epoch = 0
        queue = queue_manager.get_queue(q_name)
        self.validate_samples(images)
        while True:
            if do_shuffle:
                shuffle(images)
            for img in images:
                logger.trace("Putting to batch queue: (q_name: '%s', side: '%s')", q_name, side)
<<<<<<< HEAD
                queue.put(self.process_face(img, side))
=======
                queue.put(self.process_face(img, side, is_timelapse))
>>>>>>> 60e0099c4d88a551b33592bf5126ab96bd5dc5ae
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
<<<<<<< HEAD
            logger.trace("Yielding batch: (size: %s, queue:  '%s'", len(batch), q_name)
=======
            logger.trace("Yielding batch: (size: %s, item shapes: %s, queue:  '%s'",
                         len(batch), [item.shape for item in batch], q_name)
>>>>>>> 60e0099c4d88a551b33592bf5126ab96bd5dc5ae
            yield batch
        logger.debug("Finished minibatch generator for queue: '%s'", q_name)
        load_thread.join()

<<<<<<< HEAD
    def process_face(self, filename, side):
        """ Load an image and perform transformation and warping """
        logger.trace("Process face: (filename: '%s', side: '%s')", filename, side)
=======
    def process_face(self, filename, side, is_timelapse):
        """ Load an image and perform transformation and warping """
        logger.trace("Process face: (filename: '%s', side: '%s', is_timelapse: %s)",
                     filename, side, is_timelapse)
>>>>>>> 60e0099c4d88a551b33592bf5126ab96bd5dc5ae
        try:
            image = cv2.imread(filename)  # pylint: disable=no-member
        except TypeError:
            raise Exception("Error while reading image", filename)

<<<<<<< HEAD
        if self.mask_function:
            landmarks = self.training_opts["landmarks"]
            src_pts = self.get_landmarks(filename, image, side, landmarks)
            image = self.mask_function(src_pts, image, coverage=self.processing.coverage)

        image = self.processing.color_adjust(image)
        image = self.processing.random_transform(image)

        if self.full_face:
            dst_pts = self.get_closest_match(filename, side, landmarks, src_pts)
            processed = self.processing.random_warp_full_face(image, src_pts, dst_pts)
        else:
            processed = self.processing.random_warp(image)

        retval = self.processing.do_random_flip(processed)
        logger.trace("Processed face: (filename: '%s', side: '%s', shapes: %s)",
                     filename, side, [img.shape for img in retval])
        return retval

    @staticmethod
    def get_landmarks(filename, image, side, landmarks):
=======
        if self.mask_function or self.training_opts["warp_to_landmarks"]:
            src_pts = self.get_landmarks(filename, image, side)
        if self.mask_function:
            image = self.mask_function(src_pts, image, channels=4)

        image = self.processing.color_adjust(image)

        if not is_timelapse:
            image = self.processing.random_transform(image)
            if not self.training_opts["no_flip"]:
                image = self.processing.do_random_flip(image)
        sample = image.copy()[:, :, :3]

        if self.training_opts["warp_to_landmarks"]:
            dst_pts = self.get_closest_match(filename, side, src_pts)
            processed = self.processing.random_warp_landmarks(image, src_pts, dst_pts)
        else:
            processed = self.processing.random_warp(image)

        processed.insert(0, sample)
        logger.trace("Processed face: (filename: '%s', side: '%s', shapes: %s)",
                     filename, side, [img.shape for img in processed])
        return processed

    def get_landmarks(self, filename, image, side):
>>>>>>> 60e0099c4d88a551b33592bf5126ab96bd5dc5ae
        """ Return the landmarks for this face """
        logger.trace("Retrieving landmarks: (filename: '%s', side: '%s'", filename, side)
        lm_key = sha1(image).hexdigest()
        try:
<<<<<<< HEAD
            src_points = landmarks[side][lm_key]
=======
            src_points = self.landmarks[side][lm_key]
>>>>>>> 60e0099c4d88a551b33592bf5126ab96bd5dc5ae
        except KeyError:
            raise Exception("Landmarks not found for hash: '{}' file: '{}'".format(lm_key,
                                                                                   filename))
        logger.trace("Returning: (src_points: %s)", src_points)
        return src_points

<<<<<<< HEAD
    @staticmethod
    def get_closest_match(filename, side, landmarks, src_points):
        """ Return closest matched landmarks from opposite set """
        logger.trace("Retrieving closest matched landmarks: (filename: '%s', src_points: '%s'",
                     filename, src_points)
        dst_points = landmarks["a"] if side == "b" else landmarks["b"]
=======
    def get_closest_match(self, filename, side, src_points):
        """ Return closest matched landmarks from opposite set """
        logger.trace("Retrieving closest matched landmarks: (filename: '%s', src_points: '%s'",
                     filename, src_points)
        dst_points = self.landmarks["a"] if side == "b" else self.landmarks["b"]
>>>>>>> 60e0099c4d88a551b33592bf5126ab96bd5dc5ae
        dst_points = list(dst_points.values())
        closest = (np.mean(np.square(src_points - dst_points),
                           axis=(1, 2))).argsort()[:10]
        closest = np.random.choice(closest)
        dst_points = dst_points[closest]
        logger.trace("Returning: (dst_points: %s)", dst_points)
        return dst_points


class ImageManipulation():
    """ Manipulations to be performed on training images """
<<<<<<< HEAD
    def __init__(self, rotation_range=10, zoom_range=0.05, shift_range=0.05, random_flip=0.4,
                 zoom=(1, 1), coverage=160, scale=5):
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
=======
    def __init__(self, input_size, output_size, coverage_ratio):
        """ input_size: Size of the face input into the model
            output_size: Size of the face that comes out of the modell
            coverage_ratio: Coverage ratio of full image. Eg: 256 * 0.625 = 160
        """
        logger.debug("Initializing %s: (input_size: %s, output_size: %s, coverage_ratio: %s)",
                     self.__class__.__name__, input_size, output_size, coverage_ratio)
        # Transform args
        self.rotation_range = 10  # Range to randomly rotate the image by
        self.zoom_range = 0.05  # Range to randomly zoom the image by
        self.shift_range = 0.05  # Range to randomly translate the image by
        self.random_flip = 0.5  # Chance to flip the image horizontally
        # Transform and Warp args
        self.input_size = input_size
        self.output_size = output_size
        # Warp args
        self.coverage_ratio = coverage_ratio  # Coverage ratio of full image. Eg: 256 * 0.625 = 160
        self.scale = 5  # Normal random variable scale
>>>>>>> 60e0099c4d88a551b33592bf5126ab96bd5dc5ae
        logger.debug("Initialized %s", self.__class__.__name__)

    @staticmethod
    def color_adjust(img):
        """ Color adjust RGB image """
        logger.trace("Color adjusting image")
<<<<<<< HEAD
        return img / 255.0

    @staticmethod
    def seperate_mask(image):
=======
        return img.astype('float32') / 255.0

    @staticmethod
    def separate_mask(image):
>>>>>>> 60e0099c4d88a551b33592bf5126ab96bd5dc5ae
        """ Return the image and the mask from a 4 channel image """
        mask = None
        if image.shape[2] == 4:
            logger.trace("Image contains mask")
<<<<<<< HEAD
            mask = image[:, :, 3].reshape((image.shape[0], image.shape[1], 1))
            image = image[:, :, :3]
        return image, mask

=======
            mask = np.expand_dims(image[:, :, -1], axis=2)
            image = image[:, :, :3]
        else:
            logger.trace("Image has no mask")
        return image, mask

    def get_coverage(self, image):
        """ Return coverage value for given image """
        coverage = int(image.shape[0] * self.coverage_ratio)
        logger.trace("Coverage: %s", coverage)
        return coverage

>>>>>>> 60e0099c4d88a551b33592bf5126ab96bd5dc5ae
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

<<<<<<< HEAD
    def random_warp(self, image, src_points=None, dst_points=None):
        """ get pair of random warped images from aligned face image """
        logger.trace("Randomly warping image")
        image, mask = self.seperate_mask(image)
        height, width = image.shape[0:2]
        assert height == width and height % 2 == 0

        range_ = np.linspace(height // 2 - self.coverage // 2,
                             height // 2 + self.coverage // 2, self.scale)
        mapx = np.broadcast_to(range_, (self.scale, self.scale))
        mapy = mapx.T

        mapx = mapx + np.random.normal(size=(self.scale, self.scale), scale=self.scale)
        mapy = mapy + np.random.normal(size=(self.scale, self.scale), scale=self.scale)

        interp_mapx = cv2.resize(  # pylint: disable=no-member
            mapx, (80 * self.zoom[0],
                   80 * self.zoom[0]))[8 * self.zoom[0]:72 * self.zoom[0],
                                       8 * self.zoom[0]:72 * self.zoom[0]].astype('float32')
        interp_mapy = cv2.resize(  # pylint: disable=no-member
            mapy, (80 * self.zoom[0],
                   80 * self.zoom[0]))[8 * self.zoom[0]:72 * self.zoom[0],
                                       8 * self.zoom[0]:72 * self.zoom[0]].astype('float32')

        warped_image = cv2.remap(  # pylint: disable=no-member
            image, interp_mapx, interp_mapy, cv2.INTER_LINEAR)  # pylint: disable=no-member
        logger.trace("Warped image shape: %s", warped_image.shape)

        src_points = np.stack([mapx.ravel(), mapy.ravel()], axis=-1)
        dst_points = np.mgrid[0:65 * self.zoom[0]:16 * self.zoom[0],
                              0:65 * self.zoom[0]:16 * self.zoom[0]].T.reshape(-1, 2)

        mat = umeyama(src_points, dst_points, True)[0:2]
        target_image = cv2.warpAffine(  # pylint: disable=no-member
            image, mat, (64 * self.zoom[1], 64 * self.zoom[1]))
        logger.trace("Target image shape: %s", target_image.shape)

        retval = [warped_image, target_image]

        if mask is not None:
            target_mask = cv2.warpAffine(  # pylint: disable=no-member
                mask, mat, (64 * self.zoom[1], 64 * self.zoom[1]))
            target_mask = target_mask.reshape((64 * self.zoom[1], 64 * self.zoom[1], 1))
            logger.trace("Target mask shape: %s", target_mask.shape)

            retval.append(target_mask)

        logger.trace("Randomly warped image")
        return retval

    def random_warp_full_face(self, image, src_points=None, dst_points=None):
        """ get warped image, target image and target mask
            From DFAKER plugin """
        logger.trace("Randomly warping landmarks")
        image, mask = self.seperate_mask(image)
        size = image.shape[0]
=======
    def do_random_flip(self, image):
        """ Perform flip on image if random number is within threshold """
        logger.trace("Randomly flipping image")
        if np.random.random() < self.random_flip:
            logger.trace("Flip within threshold. Flipping")
            retval = image[:, ::-1]
        else:
            logger.trace("Flip outside threshold. Not Flipping")
            retval = image
        logger.trace("Randomly flipped image")
        return retval

    def random_warp(self, image):
        """ get pair of random warped images from aligned face image """
        logger.trace("Randomly warping image")
        height, width = image.shape[0:2]
        coverage = self.get_coverage(image)
        assert height == width and height % 2 == 0

        range_ = np.linspace(height // 2 - coverage // 2,
                             height // 2 + coverage // 2,
                             5, dtype='float32')
        mapx = np.broadcast_to(range_, (5, 5)).copy()
        mapy = mapx.T
        # mapx, mapy = np.float32(np.meshgrid(range_,range_)) # instead of broadcast

        pad = int(1.25 * self.input_size)
        slices = slice(pad // 10, -pad // 10)
        dst_slice = slice(0, (self.output_size + 1), (self.output_size // 4))
        interp = np.empty((2, self.input_size, self.input_size), dtype='float32')
        ####

        for i, map_ in enumerate([mapx, mapy]):
            map_ = map_ + np.random.normal(size=(5, 5), scale=self.scale)
            interp[i] = cv2.resize(map_, (pad, pad))[slices, slices]  # pylint: disable=no-member

        warped_image = cv2.remap(  # pylint: disable=no-member
            image, interp[0], interp[1], cv2.INTER_LINEAR)  # pylint: disable=no-member
        logger.trace("Warped image shape: %s", warped_image.shape)

        src_points = np.stack([mapx.ravel(), mapy.ravel()], axis=-1)
        dst_points = np.mgrid[dst_slice, dst_slice]
        mat = umeyama(src_points, True, dst_points.T.reshape(-1, 2))[0:2]
        target_image = cv2.warpAffine(  # pylint: disable=no-member
            image, mat, (self.output_size, self.output_size))
        logger.trace("Target image shape: %s", target_image.shape)

        warped_image, warped_mask = self.separate_mask(warped_image)
        target_image, target_mask = self.separate_mask(target_image)

        if target_mask is None:
            logger.trace("Randomly warped image")
            return [warped_image, target_image]

        logger.trace("Target mask shape: %s", target_mask.shape)
        logger.trace("Randomly warped image and mask")
        return [warped_image, target_image, target_mask]

    def random_warp_landmarks(self, image, src_points=None, dst_points=None):
        """ get warped image, target image and target mask
            From DFAKER plugin """
        logger.trace("Randomly warping landmarks")
        size = image.shape[0]
        coverage = self.get_coverage(image)

>>>>>>> 60e0099c4d88a551b33592bf5126ab96bd5dc5ae
        p_mx = size - 1
        p_hf = (size // 2) - 1

        edge_anchors = [(0, 0), (0, p_mx), (p_mx, p_mx), (p_mx, 0),
                        (p_hf, 0), (p_hf, p_mx), (p_mx, p_hf), (0, p_hf)]
        grid_x, grid_y = np.mgrid[0:p_mx:complex(size), 0:p_mx:complex(size)]

        source = src_points
<<<<<<< HEAD
        destination = (dst_points.copy().astype("float") +
                       np.random.normal(size=(dst_points.shape), scale=2))
        destination = destination.astype("uint8")
=======
        destination = (dst_points.copy().astype('float32') +
                       np.random.normal(size=dst_points.shape, scale=2.0))
        destination = destination.astype('uint8')
>>>>>>> 60e0099c4d88a551b33592bf5126ab96bd5dc5ae

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

<<<<<<< HEAD
        grid_z = griddata(destination,
                          source,
                          (grid_x, grid_y),
                          method="linear")
=======
        grid_z = griddata(destination, source, (grid_x, grid_y), method="linear")
>>>>>>> 60e0099c4d88a551b33592bf5126ab96bd5dc5ae
        map_x = np.append([], [ar[:, 1] for ar in grid_z]).reshape(size, size)
        map_y = np.append([], [ar[:, 0] for ar in grid_z]).reshape(size, size)
        map_x_32 = map_x.astype('float32')
        map_y_32 = map_y.astype('float32')

        warped_image = cv2.remap(image,  # pylint: disable=no-member
                                 map_x_32,
                                 map_y_32,
                                 cv2.INTER_LINEAR,  # pylint: disable=no-member
                                 cv2.BORDER_TRANSPARENT)  # pylint: disable=no-member
        target_image = image

<<<<<<< HEAD
        pad_lt = size // 32  # 8px on a 256px image
        pad_rb = size - pad_lt

        warped_image = cv2.resize(  # pylint: disable=no-member
            warped_image[pad_lt:pad_rb, pad_lt:pad_rb, :],
            (64 * self.zoom[0], 64 * self.zoom[0]),
            cv2.INTER_AREA)  # pylint: disable=no-member
        logger.trace("Warped image shape: %s", warped_image.shape)
        target_image = cv2.resize(  # pylint: disable=no-member
            target_image[pad_lt:pad_rb, pad_lt:pad_rb, :],
            (64 * self.zoom[1], 64 * self.zoom[1]),
            cv2.INTER_AREA)  # pylint: disable=no-member
        logger.trace("Target image shape: %s", target_image.shape)

        retval = [warped_image, target_image]

        if mask is not None:
            target_mask = cv2.resize(  # pylint: disable=no-member
                mask[pad_lt:pad_rb, pad_lt:pad_rb, :],
                (64 * self.zoom[1], 64 * self.zoom[1]),
                cv2.INTER_AREA)  # pylint: disable=no-member
            target_mask = target_mask.reshape((64 * self.zoom[1], 64 * self.zoom[1], 1))
            logger.trace("Target mask shape: %s", target_mask.shape)

            retval.append(target_mask)

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
=======
        # TODO Make sure this replacement is correct
        slices = slice(size // 2 - coverage // 2, size // 2 + coverage // 2)
#        slices = slice(size // 32, size - size // 32)  # 8px on a 256px image
        warped_image = cv2.resize(  # pylint: disable=no-member
            warped_image[slices, slices, :], (self.input_size, self.input_size),
            cv2.INTER_AREA)  # pylint: disable=no-member
        logger.trace("Warped image shape: %s", warped_image.shape)
        target_image = cv2.resize(  # pylint: disable=no-member
            target_image[slices, slices, :], (self.output_size, self.output_size),
            cv2.INTER_AREA)  # pylint: disable=no-member
        logger.trace("Target image shape: %s", target_image.shape)

        warped_image, warped_mask = self.separate_mask(warped_image)
        target_image, target_mask = self.separate_mask(target_image)

        if target_mask is None:
            logger.trace("Randomly warped image")
            return [warped_image, target_image]

        logger.trace("Target mask shape: %s", target_mask.shape)
        logger.trace("Randomly warped image and mask")
        return [warped_image, target_image, target_mask]
>>>>>>> 60e0099c4d88a551b33592bf5126ab96bd5dc5ae


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
