#!/usr/bin/env python3
""" Process training data for model training """

import logging

from hashlib import sha1
from random import shuffle, choice

import numpy as np
import cv2
from scipy.interpolate import griddata

from lib.image import batch_convert_color, read_image_batch
from lib.model import masks
from lib.multithreading import BackgroundGenerator
from lib.utils import FaceswapError

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class TrainingDataGenerator():
    """ Generate training data for models """
    def __init__(self, model_input_size, model_output_shapes, training_opts, config):
        logger.debug("Initializing %s: (model_input_size: %s, model_output_shapes: %s, "
                     "training_opts: %s, landmarks: %s, config: %s)",
                     self.__class__.__name__, model_input_size, model_output_shapes,
                     {key: val for key, val in training_opts.items() if key != "landmarks"},
                     bool(training_opts.get("landmarks", None)), config)
        self.config = config
        self.model_input_size = model_input_size
        self.model_output_shapes = model_output_shapes
        self.training_opts = training_opts
        self.mask_class = self.set_mask_class()
        self.landmarks = self.training_opts.get("landmarks", None)
        self._nearest_landmarks = {}

        # Batchsize and processing class are set when this class is called by a batcher
        # from lib.training_data
        self.batchsize = 0
        self.processing = None
        logger.debug("Initialized %s", self.__class__.__name__)

    def set_mask_class(self):
        """ Set the mask function to use if using mask """
        mask_type = self.training_opts.get("mask_type", None)
        if mask_type:
            logger.debug("Mask type: '%s'", mask_type)
            mask_class = getattr(masks, mask_type)
        else:
            mask_class = None
        logger.debug("Mask class: %s", mask_class)
        return mask_class

    def minibatch_ab(self, images, batchsize, side,
                     do_shuffle=True, is_preview=False, is_timelapse=False):
        """ Keep a queue filled to 8x Batch Size """
        logger.debug("Queue batches: (image_count: %s, batchsize: %s, side: '%s', do_shuffle: %s, "
                     "is_preview, %s, is_timelapse: %s)", len(images), batchsize, side, do_shuffle,
                     is_preview, is_timelapse)
        self.batchsize = batchsize
        self.processing = ImageManipulation(batchsize,
                                            is_preview or is_timelapse,
                                            self.model_input_size,
                                            self.model_output_shapes,
                                            self.training_opts.get("coverage_ratio", 0.625),
                                            self.config)
        args = (images, side, do_shuffle, batchsize)
        batcher = BackgroundGenerator(self.minibatch, thread_count=2, args=args)
        return batcher.iterator()

    def validate_samples(self, data):
        """ Check the total number of images against batchsize and return
            the total number of images """
        length = len(data)
        msg = ("Number of images is lower than batch-size (Note that too few "
               "images may lead to bad training). # images: {}, "
               "batch-size: {}".format(length, self.batchsize))
        try:
            assert length >= self.batchsize, msg
        except AssertionError as err:
            msg += ("\nYou should increase the number of images in your training set or lower "
                    "your batch-size.")
            raise FaceswapError(msg) from err

    def minibatch(self, images, side, do_shuffle, batchsize):
        """ A generator function that yields epoch, batchsize of warped_img
            and batchsize of target_img from the load queue """
        logger.debug("Loading minibatch generator: (image_count: %s, side: '%s', do_shuffle: %s)",
                     len(images), side, do_shuffle)
        self.validate_samples(images)

        def _img_iter(imgs):
            while True:
                if do_shuffle:
                    shuffle(imgs)
                for img in imgs:
                    yield img

        img_iter = _img_iter(images)
        while True:
            img_paths = [next(img_iter) for _ in range(batchsize)]
            batch = self.process_batch(img_paths, side)
            batch = list(zip(*batch))
            batch = [np.array(x, dtype="float32") for x in batch]
            logger.trace("Yielding batch: (size: %s, item shapes: %s, side:  '%s')",
                         len(batch), [item.shape for item in batch], side)
            yield batch

        logger.debug("Finished minibatch generator: (side: '%s')", side)

    def process_batch(self, filenames, side):
        """ Load a batch of images and perform transformation and warping """
        logger.trace("Process batch: (filenames: '%s', side: '%s')", filenames, side)
        batch = read_image_batch(filenames)

        # TODO Remove this test
        # for idx, image in enumerate(batch):
        #     if side == "a" and not self.processing.is_display:
        #         print("Orig:", image.dtype)
        #         cv2.imwrite("/home/matt/fake/test/testing/{}_orig.png".format(idx), image)

        # Initialize processing training size on first image
        if not self.processing.initialized:
            self.processing.initialize(batch.shape[1])

        # Get Landmarks prior to manipulating the image
        if self.mask_class or self.training_opts["warp_to_landmarks"]:
            batch_src_pts = self.get_landmarks(filenames, batch, side)

        # Color augmentation before mask is added
        if self.training_opts["augment_color"]:
            batch = self.processing.color_adjust(batch)

        # Add mask to batch prior to transforms and warps
        if self.mask_class:
            batch = np.array([self.mask_class(src_pts, image, channels=4).mask
                              for src_pts, image in zip(batch_src_pts, batch)])

        # Random Transform and flip
        batch = self.processing.random_transform(batch)
        if not self.training_opts["no_flip"]:
            batch = self.processing.do_random_flip(batch)

        # TODO Remove this test
        # for idx, image in enumerate(batch):
        #     if side == "a" and not self.processing.is_display:
        #         print("warp:", image.dtype, image.shape, image.min(), image.max())
        #         cv2.imwrite("/home/matt/fake/test/testing/{}_tran.png".format(idx), image)

        samples = batch[..., :3]

        # Get Targets
        targets = self.processing.get_targets(batch)

        # TODO Remove this test
        # for idx, (tgt, mask) in enumerate(zip(targets[0][0], targets[1])):
        #     if side == "a" and not self.processing.is_display:
        #         print("tgt:", tgt.dtype, tgt.shape, tgt.min(), tgt.max())
        #         print("mask:", mask.dtype, mask.shape, mask.min(), mask.max())
        #         cv2.imwrite("/home/matt/fake/test/testing/{}_tgt.png".format(idx), tgt)
        #         cv2.imwrite("/home/matt/fake/test/testing/{}_mask.png".format(idx), mask)

        if self.training_opts["warp_to_landmarks"]:
            # TODO
            pass
            # dst_pts = self.get_closest_match(filename, side, src_pts)
            # processed = self.processing.random_warp_landmarks(image, src_pts, dst_pts)
        else:
            batch = self.processing.random_warp(batch[..., :3])

        # TODO Remove this test
        # for idx, image in enumerate(batch):
        #     if side == "a" and not self.processing.is_display:
        #         print("warp:", image.dtype, image.shape, image.min(), image.max())
        #         cv2.imwrite("/home/matt/fake/test/testing/{}_warp.png".format(idx), image)
        # exit(0)
        processed = [samples, batch, targets]

        logger.trace("Processed batch: (filenames: %s, side: '%s', samples: %s, batch: %s, "
                     "targets: %s)",
                     filenames, side, processed[0].shape, processed[1].shape,
                     [[img.shape for img in tgt] if isinstance(tgt, list) else tgt.shape
                      for tgt in processed[2]])
        return processed

    def get_landmarks(self, filenames, batch, side):
        """ Return the landmarks for this batch """
        logger.trace("Retrieving landmarks: (filenames: '%s', side: '%s'", filenames, side)
        src_points = [self.landmarks[side].get(sha1(face).hexdigest(), None) for face in batch]

        # Raise error on missing alignments
        if not all(isinstance(pts, np.ndarray) for pts in src_points):
            indices = [idx for idx, hsh in enumerate(src_points) if hsh is None]
            missing = [filenames[idx] for idx in indices]
            msg = ("Files missing alignments for this batch: {}"
                   "\nAt least one of your images does not have a matching entry in your "
                   "alignments file."
                   "\nIf you are training with a mask or using 'warp to landmarks' then every "
                   "face you intend to train on must exist within the alignments file."
                   "\nThe specific files that caused this failure are listed above."
                   "\nMost likely there will be more than just these files missing from the "
                   "alignments file. You can use the Alignments Tool to help identify missing "
                   "alignments".format(missing))
            raise FaceswapError(msg)

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
            closest = (np.mean(np.square(src_points - dst_points), axis=(1, 2))).argsort()[:10]
            closest_hashes = tuple(dst_points_items[i][0] for i in closest)
            self._nearest_landmarks[filename] = closest_hashes
        dst_points = landmarks[choice(closest_hashes)]
        logger.trace("Returning: (dst_points: %s)", dst_points)
        return dst_points


class ImageManipulation():
    """ Manipulations to be performed on training images """
    def __init__(self, batchsize, is_display, input_size, output_shapes, coverage_ratio, config):
        """ input_size: Size of the face input into the model
            output_shapes: Shapes that come out of the model
            coverage_ratio: Coverage ratio of full image. Eg: 256 * 0.625 = 160
        """
        logger.debug("Initializing %s: (batchsize: %s, is_display: %s, input_size: %s, "
                     "output_shapes: %s, coverage_ratio: %s, config: %s)",
                     self.__class__.__name__, batchsize, is_display, input_size, output_shapes,
                     coverage_ratio, config)
        self.batchsize = batchsize
        self.is_display = is_display
        self.config = config
        # Transform and Warp args
        self.input_size = input_size
        self.output_sizes = [shape[1] for shape in output_shapes if shape[2] == 3]
        logger.debug("Output sizes: %s", self.output_sizes)
        # Warp args
        self.coverage_ratio = coverage_ratio  # Coverage ratio of full image. Eg: 256 * 0.625 = 160
        self.scale = 5  # Normal random variable scale

        # Set on first image load from initialize
        self.initialized = False
        self.training_size = 0
        self.constants = None

        logger.debug("Initialized %s", self.__class__.__name__)

    def initialize(self, training_size):
        """ Initialize the constants once we have the training_size from the
            first batch """
        logger.debug("Initializing constants. training_size: %s", training_size)
        self.training_size = training_size

        coverage = int(self.training_size * self.coverage_ratio)

        # Color Aug
        clahe_base_contrast = training_size // 128
        # Target Images
        tgt_slices = slice(self.training_size // 2 - coverage // 2,
                           self.training_size // 2 + coverage // 2)
        # Warp
        warp_range_ = np.linspace(self.training_size // 2 - coverage // 2,
                                  self.training_size // 2 + coverage // 2, 5, dtype='float32')
        warp_mapx = np.broadcast_to(warp_range_, (self.batchsize, 5, 5)).astype("float32")
        warp_mapy = np.broadcast_to(warp_mapx[0].T, (self.batchsize, 5, 5)).astype("float32")

        warp_pad = int(1.25 * self.input_size)
        warp_slices = slice(warp_pad // 10, -warp_pad // 10)

        self.constants = dict(clahe_base_contrast=clahe_base_contrast,
                              tgt_slices=tgt_slices,
                              warp_mapx=warp_mapx,
                              warp_mapy=warp_mapy,
                              warp_pad=warp_pad,
                              warp_slices=warp_slices)
        self.initialized = True
        logger.debug("Initialized constants: %s", self.constants)

    def color_adjust(self, batch):
        """ Color adjust RGB image """
        if not self.is_display:
            logger.trace("Augmenting color")
            batch = batch_convert_color(batch, "BGR2LAB")
            batch = self.random_clahe(batch)
            batch = self.random_lab(batch)
            batch = batch_convert_color(batch, "LAB2BGR")
        return batch

    def random_clahe(self, batch):
        """ Randomly perform Contrast Limited Adaptive Histogram Equilization on
        a batch of images """
        base_contrast = self.constants["clahe_base_contrast"]

        batch_random = np.random.rand(self.batchsize)
        indices = np.where(batch_random > self.config.get("color_clahe_chance", 50) / 100)[0]

        grid_bases = np.rint(np.random.uniform(0,
                                               self.config.get("color_clahe_max_size", 4),
                                               size=indices.shape[0])).astype("uint8")
        contrast_adjustment = (grid_bases * (base_contrast // 2))
        grid_sizes = contrast_adjustment + base_contrast
        logger.trace("Adjusting Contrast. Grid Sizes: %s", grid_sizes)

        clahes = [cv2.createCLAHE(clipLimit=2.0,  # pylint: disable=no-member
                                  tileGridSize=(grid_size, grid_size))
                  for grid_size in grid_sizes]

        for idx, clahe in zip(indices, clahes):
            batch[idx, :, :, 0] = clahe.apply(batch[idx, :, :, 0])
        return batch

    def random_lab(self, batch):
        """ Perform random color/lightness adjustment in L*a*b* colorspace on a batch of images """
        amount_l = self.config.get("color_lightness", 30) / 100
        amount_ab = self.config.get("color_ab", 8) / 100
        adjust = np.array([amount_l, amount_ab, amount_ab], dtype="float32")
        randoms = (
            (np.random.rand(self.batchsize, 1, 1, 3).astype("float32") * (adjust * 2)) - adjust)
        logger.trace("Random LAB adjustments: %s", randoms)

        for image, rand in zip(batch, randoms):
            for idx in range(rand.shape[-1]):
                adjustment = rand[:, :, idx]
                if adjustment >= 0:
                    image[:, :, idx] = ((255 - image[:, :, idx]) * adjustment) + image[:, :, idx]
                else:
                    image[:, :, idx] = image[:, :, idx] * (1 + adjustment)
        return batch

    def get_coverage(self, image):
        """ Return coverage value for given image """
        coverage = int(image.shape[0] * self.coverage_ratio)
        logger.trace("Coverage: %s", coverage)
        return coverage

    def random_transform(self, batch):
        """ Randomly transform a batch """
        if self.is_display:
            return batch
        logger.trace("Randomly transforming image")
        rotation_range = self.config.get("rotation_range", 10)
        zoom_range = self.config.get("zoom_range", 5) / 100
        shift_range = self.config.get("shift_range", 5) / 100

        rotation = np.random.uniform(-rotation_range,
                                     rotation_range,
                                     size=self.batchsize).astype("float32")
        scale = np.random.uniform(1 - zoom_range,
                                  1 + zoom_range,
                                  size=self.batchsize).astype("float32")
        tform = np.random.uniform(-shift_range,
                                  shift_range,
                                  size=(self.batchsize, 2)).astype("float32") * self.training_size

        mats = np.array(
            [cv2.getRotationMatrix2D((self.training_size // 2, self.training_size // 2),
                                     rot,
                                     scl)
             for rot, scl in zip(rotation, scale)]).astype("float32")
        mats[..., 2] += tform

        batch = np.array([cv2.warpAffine(image,
                                         mat,
                                         (self.training_size, self.training_size),
                                         borderMode=cv2.BORDER_REPLICATE)
                          for image, mat in zip(batch, mats)])

        logger.trace("Randomly transformed image")
        return batch

    def do_random_flip(self, batch):
        """ Perform flip on images in batch if random number is within threshold """
        if not self.is_display:
            logger.trace("Randomly flipping image")
            randoms = np.random.rand(self.batchsize)
            indices = np.where(randoms > self.config.get("random_flip", 50) / 100)[0]
            batch[indices] = batch[indices, :, ::-1]
            logger.trace("Randomly flipped %s images of %s", len(indices), self.batchsize)
        return batch

    def get_targets(self, batch):
        """ Get the target batch """
        logger.trace("Compiling targets")
        slices = self.constants["tgt_slices"]
        target_batch = [np.array([cv2.resize(image[slices, slices, :],
                                             (size, size),
                                             cv2.INTER_AREA)
                                  for image in batch])
                        for size in self.output_sizes]
        logger.trace("Target image shapes: %s",
                     [tgt.shape for tgt_images in target_batch for tgt in tgt_images])

        retval = self.separate_target_mask(target_batch)
        logger.trace("Final shapes: %s", [[img.shape for img in batch]
                                          if isinstance(batch, list)
                                          else batch.shape
                                          for batch in retval])
        return retval

    @staticmethod
    def separate_target_mask(batch):
        """ Return the batch and the batch of final masks from a batch of 4 channel images """
        if all(tgt.shape[-1] == 4 for tgt in batch):
            logger.trace("Batch contains mask")
            sizes = [item.shape[1] for item in batch]
            mask_batch = np.expand_dims(batch[sizes.index(max(sizes))][..., -1], axis=-1)
            batch = [item[..., :3] for item in batch]
            logger.trace("batch shapes: %s, mask_batch shape: %s",
                         [tgt.shape for tgt in batch], mask_batch.shape)
            return batch, mask_batch

        logger.trace("Batch has no mask")
        return batch

    def random_warp(self, batch):
        """ get pair of random warped images from aligned face image """
        logger.trace("Randomly warping batch")
        mapx = self.constants["warp_mapx"]
        mapy = self.constants["warp_mapy"]
        pad = self.constants["warp_pad"]
        slices = self.constants["warp_slices"]

        rands = np.random.normal(size=(self.batchsize, 2, 5, 5),
                                 scale=self.scale).astype("float32")
        batch_maps = np.stack((mapx, mapy), axis=1) + rands
        batch_interp = np.array([[cv2.resize(map_, (pad, pad))[slices, slices] for map_ in maps]
                                 for maps in batch_maps])
        warped_batch = np.array([cv2.remap(image, interp[0], interp[1], cv2.INTER_LINEAR)
                                 for image, interp in zip(batch, batch_interp)])

        logger.trace("Warped image shape: %s", warped_batch.shape)
        return warped_batch

    def random_warp_landmarks(self, image, src_points=None, dst_points=None):
        """ get warped image, target image and target mask
            From DFAKER plugin """
        logger.trace("Randomly warping landmarks")
        size = image.shape[0]
        coverage = self.get_coverage(image) // 2

        p_mx = size - 1
        p_hf = (size // 2) - 1

        edge_anchors = [(0, 0), (0, p_mx), (p_mx, p_mx), (p_mx, 0),
                        (p_hf, 0), (p_hf, p_mx), (p_mx, p_hf), (0, p_hf)]
        grid_x, grid_y = np.mgrid[0:p_mx:complex(size), 0:p_mx:complex(size)]

        source = src_points
        destination = (dst_points.copy().astype('float32') +
                       np.random.normal(size=dst_points.shape, scale=2.0))
        destination = destination.astype('uint8')

        face_core = cv2.convexHull(np.concatenate(
            [source[17:], destination[17:]], axis=0).astype(int))

        source = [(pty, ptx) for ptx, pty in source] + edge_anchors
        destination = [(pty, ptx) for ptx, pty in destination] + edge_anchors

        indicies_to_remove = set()
        for fpl in source, destination:
            for idx, (pty, ptx) in enumerate(fpl):
                if idx > 17:
                    break
                elif cv2.pointPolygonTest(face_core,
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

        warped_image = cv2.remap(image,
                                 map_x_32,
                                 map_y_32,
                                 cv2.INTER_LINEAR,
                                 cv2.BORDER_TRANSPARENT)
        target_image = image

        # TODO Make sure this replacement is correct
        slices = slice(size // 2 - coverage, size // 2 + coverage)
#        slices = slice(size // 32, size - size // 32)  # 8px on a 256px image
        warped_image = cv2.resize(
            warped_image[slices, slices, :], (self.input_size, self.input_size),
            cv2.INTER_AREA)
        logger.trace("Warped image shape: %s", warped_image.shape)
        target_images = [cv2.resize(target_image[slices, slices, :],
                                    (size, size),
                                    cv2.INTER_AREA)
                         for size in self.output_sizes]

        logger.trace("Target image shapea: %s", [img.shape for img in target_images])
        return self.compile_images(warped_image, target_images)


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
