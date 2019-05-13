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

    def batcher_setup(self, images, batch_size, side, purpose, do_shuffle=True, augmenting=True):
        """ Keep a queue filled to 8x Batch Size """
        logger.debug("Queue batches: (image_count: %s, batchsize: %s, side: '%s', do_shuffle: %s, "
                     "augmenting: %s)", len(images), batch_size, side, do_shuffle, augmenting)
        self.batch_size = batch_size
        queue_in, queue_out = self.make_queues(side, purpose)
        training_size = self.training_opts.get("training_size", 256)
        in_height = in_width = self.model_input_size
        out_height = out_width = self.model_output_size
        batch_shapes = [(batch_size, training_size, training_size, 3),  # sample images
                        (batch_size, in_height, in_width, 3),           # warped images
                        (batch_size, in_height, in_width, 1),           # warped masks
                        (batch_size, out_height, out_width, 3),         # target images
                        (batch_size, out_height, out_width, 1)]         # target masks

        load_process = FixedProducerDispatcher(
            method=self.load_batches,
            shapes=batch_shapes,
            in_queue=queue_in,
            out_queue=queue_out,
            args=(images, side, batch_size, augmenting, do_shuffle))
        load_process.start()
        logger.debug("Batching to queue: (side: '%s', augmenting: %s)", side, augmenting)
        return self.minibatch(side, augmenting, load_process)

    @staticmethod
    def make_queues(side, purpose):
        """ Create the buffer token queues for Fixed Producer Dispatcher """
        q_names = ["{}_side_{}_{}".format(purpose, side, direction) for direction in ("in", "out")]
        logger.debug(q_names)
        queues = [queue_manager.get_queue(queue) for queue in q_names]
        return queues

    def load_batches(self, mem_gen, images, side, batch_size, augmenting, do_shuffle):
        """ Load the warped images and target images to queue """
        logger.debug("Loading batch: (image_count: %s, side: '%s', augmenting: %s, "
                     "do_shuffle: %s)", len(images), side, augmenting, do_shuffle)
        self.validate_samples(images["images"])
        # Intialize this for each subprocess
        self._nearest_landmarks = dict()

        def batch_generator(inputs):
        """ doc string """
            if do_shuffle:
                rng_state = numpy.random.get_state()
                np.random.set_state(rng_state)
                np.random.shuffle(inputs[1])
                np.random.set_state(rng_state)
                np.random.shuffle(inputs[2])
            yield from inputs

        batcher = batch_generator(zip(range(batch_size),
                                      images["images"],
                                      images["landmarks"]))
        epoch = 0
        for memory_wrapper in mem_gen:
            memory = memory_wrapper.get()
            logger.trace("Putting to batch queue: (side: '%s', augmenting: %s)", side, augmenting)
            for image_num, image, landmark in batcher:
                imgs = self.process_faces(image, landmark, side, augmenting, image_num)
                for process_output_num, img in enumerate(imgs):
                    memory[process_output_num][image_num][:] = img
                epoch += 1
            memory_wrapper.ready()
        logger.debug("Finished batching: (epoch: %s, side: '%s', augmenting: %s)",
                     epoch, side, augmenting)

    def validate_samples(self, data):
        """ Check the total number of images against batchsize and return
            the total number of images """
        length = data.shape[0]
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

    def process_faces(self, image, src_pts, side, augmenting, img_number):
        """ Load an image and perform transformation and warping """
        logger.trace("Processing face: (image #: '%s', side: '%s', augmenting: %s)",
                     img_number, side, augmenting)
        sample = image.copy()[:, :, :3]
        if augmenting:
            image = self.processing.random_transform(image)
            if not self.training_opts["no_flip"]:
                image = self.processing.do_random_flip(image)
        if self.training_opts["warp_to_landmarks"]:
            dst_pts = self.get_closest_match(filename, side, src_pts)
            processed = self.processing.random_warp_landmarks(image, src_pts, dst_pts)
        else:
            processed = self.processing.random_warp(image)
        processed.insert(0, sample)
        logger.trace("Processed face: (image: '%s', side: '%s', shapes: %s)",
                     image, side, [img.shape for img in processed])
        return processed

    def get_closest_match(self, filename, side, src_points):
        """ Return closest matched landmarks from opposite set """
        logger.trace("Retrieving closest matched landmarks: (filename: '%s', src_points: '%s'",
                     filename, src_points)
        landmarks = self.landmarks["a"] if side == "b" else self.landmarks["b"]
        closest_hashes = self._nearest_landmarks.get(filename)
        if not closest_hashes:
            dst_points_items = list(landmarks.items())
            dst_points = list(x[1] for x in dst_points_items)
            closest = (np.mean(np.square(src_points - dst_points), axis=(1, 2))).argsort()[:10]
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
          # pylint: disable=no-member
        logger.trace("Randomly transforming image")
        height, width = image.shape[0:2]
        rotate = np.random.uniform(-self.t_args["rotation"], self.t_args["rotation"])
        scale = np.random.uniform(1 - self.t_args["zoom"], 1 + self.t_args["zoom"])
        tn_x = np.random.uniform(-self.t_args["shift"], self.t_args["shift"]) * width
        tn_y = np.random.uniform(-self.t_args["shift"], self.t_args["shift"]) * height

        mat = cv2.getRotationMatrix2D((width // 2, height // 2), rotate, scale)
        mat[:, 2] += (tn_x, tn_y)
        result = cv2.warpAffine(image, mat, (width, height),
                                borderMode=cv2.BORDER_REPLICATE)

        logger.trace("Randomly transformed image")
        return result

    def do_random_flip(self, image):
        """ Perform flip on image if random number is within threshold """
        logger.trace("Randomly flipping image")
        do_flip = np.random.random() < self.t_args["flip"]
        retval = image[:, ::-1] if do_flip else image
        logger.trace("Was the image flipped?: %s", str(do_flip))
        return retval

    def random_warp(self, image):
        """ get pair of random warped images from aligned face image """
        # pylint: disable=no-member
        logger.trace("Randomly warping image")
        height, width = image.shape[0:2]
        coverage = self.get_coverage(image) // 2
        assert height == width and height % 2 == 0

        range_ = np.linspace(height // 2 - coverage, height // 2 + coverage, 5, dtype='float32')
        mapx = np.broadcast_to(range_, (5, 5)).copy()
        mapy = mapx.T
        # mapx, mapy = np.float32(np.meshgrid(range_,range_)) # instead of broadcast

        pad = int(1.25 * self.input_size)
        slices = slice(pad // 10, -pad // 10)
        dst_slice = slice(0, (self.output_size + 1), (self.output_size // 4))
        interp = np.empty((2, self.input_size, self.input_size), dtype='float32')

        for i, map_ in enumerate([mapx, mapy]):
            map_ = map_ + np.random.normal(size=(5, 5), scale=self.scale)
            interp[i] = cv2.resize(map_, (pad, pad))[slices, slices]

        warped_image = cv2.remap(image, interp[0], interp[1], cv2.INTER_LINEAR)
        src_points = np.stack([mapx.ravel(), mapy.ravel()], axis=-1)
        dst_points = np.mgrid[dst_slice, dst_slice]
        mat = umeyama(src_points, True, dst_points.T.reshape(-1, 2))[0:2]
        target_image = cv2.warpAffine(image, mat, (self.output_size, self.output_size))

        warped_image, warped_mask = self.separate_mask(warped_image)
        target_image, target_mask = self.separate_mask(target_image)

        logger.trace("Warped image shape: %s", warped_image.shape)
        logger.trace("Warped mask shape: %s", warped_mask.shape)
        logger.trace("Target image shape: %s", target_image.shape)
        logger.trace("Target mask shape: %s", target_mask.shape)
        logger.trace("Randomly warped image and mask")
        return [warped_image, warped_mask, target_image, target_mask]

    def random_warp_landmarks(self, image, source=None, destination=None):
        """ get warped image, target image and target mask From DFAKER plugin """
        # pylint: disable=no-member
        logger.trace("Randomly warping landmarks")
        size = image.shape[0]
        coverage = self.get_coverage(image)
        p_mx = size - 1
        p_hf = (size // 2) - 1

        edge_anchors = [(0, 0), (0, p_mx), (p_mx, p_mx), (p_mx, 0),
                        (p_hf, 0), (p_hf, p_mx), (p_mx, p_hf), (0, p_hf)]
        grid_x, grid_y = np.mgrid[0:p_mx:complex(size), 0:p_mx:complex(size)]

        destination = destination.astype('float32')
        destination = destination + np.random.normal(size=dst_points.shape, scale=2.)
        destination = destination.astype('uint8')
        points = np.concatenate([source[17:], destination[17:]], axis=0).astype('uint32')
        face_core = cv2.convexHull(points)
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

        grid_z = griddata(destination, source, (grid_x, grid_y), method="linear").astype('float32')
        map_x = np.append([], [ar[:, 1] for ar in grid_z]).reshape(size, size)
        map_y = np.append([], [ar[:, 0] for ar in grid_z]).reshape(size, size)
        warped_image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, cv2.BORDER_TRANSPARENT)
        target_image = image

        # TODO Make sure this replacement is correct
        slices = slice(size // 2 - coverage // 2, size // 2 + coverage // 2)
        # slices = slice(size // 32, size - size // 32)  # 8px on a 256px image
        warped_image = cv2.resize(warped_image[slices, slices, :],
                                  (self.input_size, self.input_size),
                                  cv2.INTER_AREA)
        target_image = cv2.resize(target_image[slices, slices, :],
                                  (self.output_size, self.output_size),
                                  cv2.INTER_AREA)

        warped_image, warped_mask = self.separate_mask(warped_image)
        target_image, target_mask = self.separate_mask(target_image)

        logger.trace("Warped image shape: %s", warped_image.shape)
        logger.trace("Warped mask shape: %s", warped_mask.shape)
        logger.trace("Target image shape: %s", target_image.shape)
        logger.trace("Target mask shape: %s", target_mask.shape)
        logger.trace("Randomly warped image and mask")
        return [warped_image, warped_mask, target_image, target_mask]
