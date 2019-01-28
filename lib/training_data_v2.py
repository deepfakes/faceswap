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

# TODO Add source, dest points to random warp half and ability to
# not have landmarks to random warp full


class TrainingDataGenerator():
    """ Generate training data for models """
    def __init__(self, model_input_size, model_output_size, training_opts):
        logger.debug("Initializing %s: (model_input_size: %s, model_output_shape: %s, "
                     "training_opts: %s)",
                     self.__class__.__name__, model_input_size, model_output_size,
                     {key: val for key, val in training_opts.items() if key != "landmarks"})
        self.batchsize = 0
        self.training_opts = training_opts
        self.full_face = self.training_opts.get("full_face", False)
        self.mask_function = self.set_mask_function()
        self.processing = ImageManipulation(model_input_size,
                                            model_output_size,
                                            training_opts.get("coverage_ratio", 0.625))
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
            logger.trace("Yielding batch: (size: %s, item shapes: %s, queue:  '%s'",
                         len(batch), [item.shape for item in batch], q_name)
            yield batch
        logger.debug("Finished minibatch generator for queue: '%s'", q_name)
        load_thread.join()

    def process_face(self, filename, side):
        """ Load an image and perform transformation and warping """
        logger.trace("Process face: (filename: '%s', side: '%s')", filename, side)
        try:
            image = cv2.imread(filename)  # pylint: disable=no-member
        except TypeError:
            raise Exception("Error while reading image", filename)

        if self.mask_function:
            landmarks = self.training_opts["landmarks"]
            src_pts = self.get_landmarks(filename, image, side, landmarks)
            image = self.mask_function(src_pts, image, channels=4)

        image = self.processing.normalize_color(image)
        matrix = self.processing.random_matrix(image)
        image, ground_truth = self.processing.transform_and_resize(image, matrix)

        if self.full_face:
            dst_pts = self.get_closest_match(filename, side, landmarks, src_pts)
            processed = self.processing.random_warp_full_face(image, src_pts, dst_pts)
        else:
            warped = self.processing.random_warp(image)
            warped, warped_mask = self.separate_mask(warped)
            ground_truth, ground_truth_mask = self.separate_mask(ground_truth)
            processed = [warped, warped_mask, ground_truth, ground_truth_mask]
            
        logger.trace("Processed face: (filename: '%s', side: '%s', shapes: %s)",
                     filename, side, [img.shape for img in processed])
        return processed

    @staticmethod
    def get_landmarks(filename, image, side, landmarks):
        """ Return the landmarks for this face """
        logger.trace("Retrieving landmarks: (filename: '%s', side: '%s'", filename, side)
        lm_key = sha1(image).hexdigest()
        try:
            src_points = landmarks[side][lm_key]
        except KeyError:
            raise Exception("Landmarks not found for hash: '{}' file: '{}'".format(lm_key,
                                                                                   filename))
        logger.trace("Returning: (src_points: %s)", src_points)
        return src_points

    @staticmethod
    def get_closest_match(filename, side, landmarks, src_points):
        """ Return closest matched landmarks from opposite set """
        logger.trace("Retrieving closest matched landmarks: (filename: '%s', src_points: '%s'",
                     filename, src_points)
        dst_points = landmarks["a"] if side == "b" else landmarks["b"]
        dst_points = list(dst_points.values())
        closest = (np.mean(np.square(src_points - dst_points),
                           axis=(1, 2))).argsort()[:10]
        closest = np.random.choice(closest)
        dst_points = dst_points[closest]
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
        self.rotation_range = 10  # Range to randomly rotate the image by
        self.zoom_range = 0.05  # Range to randomly zoom the image by
        self.shift_range = 0.05  # Range to randomly shift the image by
        self.random_flip = 0.5  # Chance to flip the image left > right
        self.transforms = numpy.array(self.rotation_range, self.zoom_range, self.shift_range * 256.0, self.shift_range * 256.0, 1.0])
        # Transform and Warp args
        self.zoom_source = input_size // 64
        self.zoom_target = output_size // 64
        # Warp args
        self.coverage_ratio = coverage_ratio  # Coverage ratio of full image. Eg: 256 * 0.625 = 160
        self.scale = 5  # Normal random variable scale
        expanse = numpy.linspace(48.0, 208.0, 64)
        self.mapx, self.mapy = numpy.float32(numpy.meshgrid(expanse,expanse))
        self.two_grids = numpy.float32(numpy.meshgrid(expanse,expanse))
        gridding = numpy.linspace(0.0, 4.0, 64)
        xx, yy = numpy.float32(numpy.meshgrid(gridding,gridding))
        self.grid_x, self.grid_y = cv2.convertMaps(xx, yy, cv2.CV_16SC2)
        
        logger.debug("Initialized %s", self.__class__.__name__)

    @staticmethod
    def normalize_color(image):
        """ Color adjust RGB image """
        logger.trace("Color adjusting image")
        return image.astype(numpy.float32) / 255.0

    @staticmethod
    def separate_mask(image):
        """ Return the image and the mask from a 4 channel image """
        mask = None
        if image.shape[2] == 4:
            logger.trace("Image contains mask")
            mask = image[:, :, -1]
            image = image[:, :, :3]
        return image, mask

    def get_coverage(self, image):
        """ Return coverage value for given image """
        coverage = int(image.shape[0] * self.coverage_ratio)
        logger.trace(coverage)
        return coverage

    def random_matrix(self, image):
        """ """
        # [0]=rotation, [1]=scale, [2]=tnx, [3]=tny, [4]=flip
        logger.trace("Preparing matrix for randomly transformed image")
        
        h, w = image.shape[0:2]
        affine = np.random.uniform(-1.0,1.0,5) * self.transforms
        mat = cv2.getRotationMatrix2D((w // 2, h // 2), affine[0], affine[1] + 1.0) # pylint: disable=no-member
        mat[:,2] += (affine[2]*image.shape[1], affine[3]*image.shape[0])
        if affine[4] >= self.random_flip * 2.0 - 1.0:
            image = image[:,::-1]
        logger.trace("Matrix for randomly transformed image prepared")
        
        return image, mat

    def transform_and_resize(self, image, mat):
        """ """
        h, w = image.shape[0:2]
        warp_input   = cv2.warpAffine(image, mat, (w, h), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC) # pylint: disable=no-member
        logger.trace("Randomly transformed image")
        
        padding = (image.shape[0] * (1 - self.coverage_ratio)) // 2
        cover = (image.shape[0] * self.coverage_ratio) // 2
        crop = slice(padding, -padding)
        ground_truth = cv2.resize(warp_input[crop,crop], (cover,cover), cv2.INTER_CUBIC) # pylint: disable=no-member
        ground_truth = cv2.resize(ground_truth, (self.input_size, self.input_size), cv2.INTER_AREA) # pylint: disable=no-member  # area for 80->64
        logger.trace("Target image shape: %s", ground_truth.shape)
        
        return warp_input, ground_truth
            
    def random_warp(self, warp_input):
        """ """
        large_random_grid = numpy.empty((2, self.input_size, self.input_size), dtype=numpy.float32)
        for i, mapping in enumerate([self.mapx, self.mapy]):
            small_random_grid = numpy.random.normal(size=(5,5), scale=self.scale).astype('float32')
            large_random_grid[i] = cv2.remap(small_random_grid, self.grid_x, self.grid_y, cv2.INTER_LINEAR) + mapping # pylint: disable=no-member
            
        warped_image = cv2.remap(warp_input, large_random_grid[0], large_random_grid[1], cv2.INTER_LINEAR) # pylint: disable=no-member
        logger.trace("Warped image shape: %s", warped_image.shape)
        
        logger.trace("Randomly warped image")
        
        return warped_image

    def random_warp_full_face(self, image, src_points=None, dst_points=None):
        """ get warped image, target image and target mask
            From DFAKER plugin """
        logger.trace("Randomly warping landmarks")
        image, mask = self.separate_mask(image)
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

        warped_image = cv2.remap(image,  # pylint: disable=no-member
                                 map_x_32,
                                 map_y_32,
                                 cv2.INTER_LINEAR,  # pylint: disable=no-member
                                 cv2.BORDER_TRANSPARENT)  # pylint: disable=no-member
        target_image = image

        pad_lt = size // 32  # 8px on a 256px image
        pad_rb = size - pad_lt

        warped_image = cv2.resize(  # pylint: disable=no-member
            warped_image[pad_lt:pad_rb, pad_lt:pad_rb, :],
            (64 * self.zoom_source, 64 * self.zoom_source),
            cv2.INTER_AREA)  # pylint: disable=no-member
        logger.trace("Warped image shape: %s", warped_image.shape)
        target_image = cv2.resize(  # pylint: disable=no-member
            target_image[pad_lt:pad_rb, pad_lt:pad_rb, :],
            (64 * self.zoom_target, 64 * self.zoom_target),
            cv2.INTER_AREA)  # pylint: disable=no-member
        logger.trace("Target image shape: %s", target_image.shape)

        retval = [warped_image, target_image]

        if mask is not None:
            target_mask = cv2.resize(  # pylint: disable=no-member
                mask[pad_lt:pad_rb, pad_lt:pad_rb, :],
                (64 * self.zoom_target, 64 * self.zoom_target),
                cv2.INTER_AREA)  # pylint: disable=no-member
            target_mask = target_mask.reshape((64 * self.zoom_target, 64 * self.zoom_target, 1))
            logger.trace("Target mask shape: %s", target_mask.shape)

            retval.append(target_mask)

        logger.trace("Randomly warped image")
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
