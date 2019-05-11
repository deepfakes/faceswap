#!/usr/bin/env python3
""" Process training data for model training """

import logging

from hashlib import sha1
from random import shuffle, choice

import cv2
import numpy as np
from scipy.interpolate import griddata

from lib.multithreading import FixedProducerDispatcher
from lib.queue_manager import queue_manager
from lib.umeyama import umeyama

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class TrainingDataGenerator():
    """ Generate training data for models """
    def __init__(self, model_input_size, model_output_size, training_opts):
        logger.debug("Initializing %s: (model_input_size: %s, model_output_shape: %s, "
                     "training_opts: %s, landmarks: %s)",
                     self.__class__.__name__, model_input_size, model_output_size,
                     {key: val for key, val in training_opts.items() if key != "landmarks"},
                     bool(training_opts.get("landmarks", None)))
        self.batch_size = 0
        self.model_input_size = model_input_size
        self.model_output_size = model_output_size
        self.training_opts = training_opts
        self.landmarks = self.training_opts.get("landmarks", None)
        self._nearest_landmarks = None
        self.processing = ImageManipulation(model_input_size,
                                            model_output_size,
                                            training_opts.get("coverage_ratio", 0.625))
        logger.debug("Initialized %s", self.__class__.__name__)

    def minibatch_ab(self, images, batch_size, side, do_shuffle=True, augmenting=True):
        """ Keep a queue filled to 8x Batch Size """
        logger.debug("Queue batches: (image_count: %s, batchsize: %s, side: '%s', do_shuffle: %s, "
                     "augmenting: %s)", len(images), batch_size, side, do_shuffle, augmenting)
        self.batch_size = batch_size
        queue_in, queue_out = self.make_queues(side, augmenting)
        training_size = self.training_opts.get("training_size", 256)
        batch_shape = list(
            ((batch_size, training_size, training_size, 3),                     # sample images
             (batch_size, self.model_input_size, self.model_input_size, 3),     # warped images
             (batch_size, self.model_input_size, self.model_input_size, 1),     # warped masks
             (batch_size, self.model_output_size, self.model_output_size, 3),   # target images
             (batch_size, self.model_output_size, self.model_output_size, 1)))  # target masks

        load_process = FixedProducerDispatcher(
            method=self.load_batches,
            shapes=batch_shape,
            in_queue=queue_in,
            out_queue=queue_out,
            args=(images, side, batch_size, augmenting, do_shuffle))
        load_process.start()
        logger.debug("Batching to queue: (side: '%s', augmenting: %s)", side, augmenting)
        return self.minibatch(side, augmenting, load_process)

    @staticmethod
    def make_queues(side, augmenting):
        """ Create the buffer token queues for Fixed Producer Dispatcher """
        q_name = "train_{}".format(side) if augmenting else "timelapse_{}".format(side)
        q_names = ["{}_{}".format(q_name, direction) for direction in ("in", "out")]
        logger.debug(q_names)
        queues = [queue_manager.get_queue(queue) for queue in q_names]
        return queues

    def load_batches(self, mem_gen, images, side, batch_size, augmenting, do_shuffle):
        """ Load the warped images and target images to queue """
        logger.debug("Loading batch: (image_count: %s, side: '%s', augmenting: %s, "
                     "do_shuffle: %s)", len(images), side, augmenting, do_shuffle)
        self.validate_samples(images)
        # Intialize this for each subprocess
        self._nearest_landmarks = dict()

        def _img_iter(images):
            while True:
                if do_shuffle:
                    shuffle(images)
                yield from images

        img_iter = _img_iter(images)
        epoch = 0
        for memory_wrapper in mem_gen:
            memory = memory_wrapper.get()
            logger.trace("Putting to batch queue: (side: '%s', augmenting: %s)",
                         side, augmenting)
            for i, img_path in enumerate(img_iter):
                imgs = self.process_face(img_path, side, augmenting)
                for j, img in enumerate(imgs):
                    memory[j][i][:] = img
                epoch += 1
                if i == batch_size - 1:
                    break
            memory_wrapper.ready()
        logger.debug("Finished batching: (epoch: %s, side: '%s', augmenting: %s)",
                     epoch, side, augmenting)

    def validate_samples(self, data):
        """ Check the total number of images against batchsize and return
            the total number of images """
        length = len(data)
        msg = ("Number of images is lower than batch-size (Note that too few "
               "images may lead to bad training). # images: {}, "
               "batch-size: {}".format(length, self.batch_size))
        assert length >= self.batch_size, msg

    @staticmethod
    def minibatch(side, augmenting, load_process):
        """ A generator function that yields epoch, batchsize of warped_img
            and batchsize of target_img from the load queue """
        logger.debug("Launching minibatch generator for queue (side: '%s', augmenting: %s)",
                     side, augmenting)
        for batch_wrapper in load_process:
            with batch_wrapper as batch:
                logger.trace("Yielding batch: (size: %s, item shapes: %s, side:  '%s', "
                             "augmenting: %s)",
                             len(batch), [item.shape for item in batch], side, augmenting)
                yield batch
        load_process.stop()
        logger.debug("Finished minibatch generator for queue: (side: '%s', augmenting: %s)",
                     side, augmenting)
        load_process.join()

    def process_face(self, filename, side, augmenting):
        """ Load an image and perform transformation and warping """
        logger.trace("Process face: (filename: '%s', side: '%s', augmenting: %s)",
                     filename, side, augmenting)
        try:
            image = cv2.imread(filename)  # pylint: disable=no-member
        except TypeError:
            raise Exception("Error while reading image", filename)

        image = image.astype("float32") / 255.
        sample = image.copy()[:, :, :3]

        if augmenting:
            image = self.processing.random_transform(image)
            if not self.training_opts["no_flip"]:
                image = self.processing.do_random_flip(image)

        if self.training_opts["warp_to_landmarks"]:
            src_pts = self.get_landmarks(filename, sample, side)
            dst_pts = self.get_closest_match(filename, side, src_pts)
            processed = self.processing.random_warp_landmarks(image, src_pts, dst_pts)
        else:
            processed = self.processing.random_warp(image)

        processed.insert(0, sample)
        logger.trace("Processed face: (filename: '%s', side: '%s', shapes: %s)",
                     filename, side, [img.shape for img in processed])
        return processed

    def get_landmarks(self, filename, image, side):
        """ Return the landmarks for this face """
        logger.trace("Retrieving landmarks: (filename: '%s', side: '%s'", filename, side)
        lm_key = sha1(image).hexdigest()
        try:
            src_points = self.landmarks[side][lm_key]
        except KeyError:
            raise Exception("Landmarks not found for hash: '{}' file: '{}'".format(lm_key,
                                                                                   filename))
        logger.trace("Returning: (src_points: %s)", src_points)
        return src_points

    def get_closest_match(self, filename, side, src_points):
        """ Return closest matched landmarks from opposite set """
        logger.trace("Retrieving closest matched landmarks: (filename: '%s', src_points: '%s'",
                     filename, src_points)
        landmarks = self.landmarks["a"] if side == "b" else self.landmarks["b"]
        closest_hashes = self._nearest_landmarks.get(filename)
        if not closest_hashes:
            dst_points_items = list(landmarks.items())
            dst_points = list(x[1] for x in dst_points_items)
            closest = (np.mean(np.square(src_points - dst_points),
                               axis=(1, 2))).argsort()[:10]
            closest_hashes = tuple(dst_points_items[i][0] for i in closest)
            self._nearest_landmarks[filename] = closest_hashes
        dst_points = landmarks[choice(closest_hashes)]
        logger.trace("Returning: (dst_points: %s)", dst_points)
        return dst_points


class ImageManipulation():
    """ Manipulations to be performed on training images """
    def __init__(self, input_size, output_size, coverage_ratio):
        """ input_size: Size of the face input into the model
            output_size: Size of the face that comes out of the modell
            coverage_ratio: Coverage ratio of full image. Eg: 256 * 0.625 = 160
        """
        logger.debug("Initializing %s: (input_size: %s, output_size: %s, coverage_ratio: %s)",
                     self.__class__.__name__, input_size, output_size, coverage_ratio)
        # Transform args
        self.t_args = {"rotation":  10.,   # Range to randomly rotate the image by
                       "zoom":      0.05,  # Range to randomly zoom the image by
                       "shift":     0.05,  # Range to randomly translate the image by
                       "flip":      0.5}   # Chance to flip the image horizontally
        # Transform and Warp args
        self.input_size = input_size
        self.output_size = output_size
        # Warp args
        self.coverage_ratio = coverage_ratio  # Coverage ratio of full image. Eg: 256 * 0.625 = 160
        self.scale = 5  # Normal random variable scale
        logger.debug("Initialized %s", self.__class__.__name__)

    @staticmethod
    def separate_mask(image):
        """ Return the image and the mask from a 4 channel image """
        logger.trace("Seperating mask from four channel image")
        mask = image[:, :, -1:]
        image = image[:, :, :3]
        return image, mask

    def get_coverage(self, image):
        """ Return coverage value for given image """
        coverage = int(image.shape[0] * self.coverage_ratio)
        logger.trace("Coverage: %s", coverage)
        return coverage

    def random_transform(self, image):
        """ Randomly transform an image """
        logger.trace("Randomly transforming image")
        height, width = image.shape[0:2]
        rotation = np.random.uniform(-self.t_args["rotation"], self.t_args["rotation"])
        scale = np.random.uniform(1 - self.t_args["zoom"], 1 + self.t_args["zoom"])
        tn_x = np.random.uniform(-self.t_args["shift"], self.t_args["shift"]) * width
        tn_y = np.random.uniform(-self.t_args["shift"], self.t_args["shift"]) * height

        mat = cv2.getRotationMatrix2D(  # pylint: disable=no-member
            (width // 2, height // 2), rotation, scale)
        mat[:, 2] += (tn_x, tn_y)
        result = cv2.warpAffine(  # pylint: disable=no-member
            image, mat, (width, height),
            borderMode=cv2.BORDER_REPLICATE)  # pylint: disable=no-member

        logger.trace("Randomly transformed image")
        return result

    def do_random_flip(self, image):
        """ Perform flip on image if random number is within threshold """
        logger.trace("Randomly flipping image")
        if np.random.random() < self.t_args["flip"]:
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
        return [warped_image, warped_mask, target_image, target_mask]

    def random_warp_landmarks(self, image, src_points=None, dst_points=None):
        """ get warped image, target image and target mask
            From DFAKER plugin """
        logger.trace("Randomly warping landmarks")
        size = image.shape[0]
        coverage = self.get_coverage(image)

        p_mx = size - 1
        p_hf = (size // 2) - 1

        edge_anchors = [(0, 0), (0, p_mx), (p_mx, p_mx), (p_mx, 0),
                        (p_hf, 0), (p_hf, p_mx), (p_mx, p_hf), (0, p_hf)]
        grid_x, grid_y = np.mgrid[0:p_mx:complex(size), 0:p_mx:complex(size)]

        source = src_points
        destination = (dst_points.copy().astype('float32') +
                       np.random.normal(size=dst_points.shape, scale=2.0))
        destination = destination.astype('uint8')

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

        grid_z = griddata(destination, source, (grid_x, grid_y), method="linear")
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
        return [warped_image, warped_mask, target_image, target_mask]
