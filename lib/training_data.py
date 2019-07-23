#!/usr/bin/env python3
""" Process training data for model training """

import logging

from hashlib import sha1
from random import random, shuffle, choice

import cv2
import numpy as np
from scipy.interpolate import griddata

from lib.multithreading import FixedProducerDispatcher
from lib.queue_manager import queue_manager
from lib.umeyama import umeyama
from lib.utils import cv2_read_img, FaceswapError

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class TrainingDataGenerator():
    """ Generate training data for models """
    def __init__(self, model_input_size, model_output_size, training_opts, config):
        logger.debug("Initializing %s: (model_input_size: %s, model_output_shape: %s, "
                     "training_opts: %s, landmarks: %s, config: %s)",
                     self.__class__.__name__, model_input_size, model_output_size,
                     {key: val for key, val in training_opts.items() if key != "landmarks"},
                     bool(training_opts.get("landmarks", None)), config)
        self.batch_size = 0
        self.model_input_size = model_input_size
        self.model_output_size = model_output_size
        self.training_opts = training_opts
        self.landmarks = self.training_opts.get("landmarks", None)
        self.fixed_producer_dispatcher = None  # Set by FPD when loading
        self._nearest_landmarks = None
        self.processing = ImageManipulation(model_input_size,
                                            model_output_size,
                                            training_opts.get("coverage_ratio", 0.625),
                                            config)
        logger.debug("Initialized %s", self.__class__.__name__)

<<<<<<< HEAD
    def setup_batcher(self, images, batch_size, side, purpose, do_shuffle=True, augmenting=True):
        """ Keep a queue filled to 8x Batch Size """
        logger.debug("Queue batches: (image_count: %s, batchsize: %s, side: '%s', do_shuffle: %s, "
                     "augmenting: %s)", len(images), batch_size, side, do_shuffle, augmenting)
        self.batch_size = batch_size
        queue_in, queue_out = self.make_queues(side, purpose)
=======
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
        is_display = is_preview or is_timelapse
        queue_in, queue_out = self.make_queues(side, is_preview, is_timelapse)
>>>>>>> staging
        training_size = self.training_opts.get("training_size", 256)
        in_height = in_width = self.model_input_size
        out_height = out_width = self.model_output_size
        batch_shapes = [(batch_size, training_size, training_size, 3),  # sample images
                        (batch_size, in_height, in_width, 3),           # warped images
                        (batch_size, in_height, in_width, 1),           # warped masks
                        (batch_size, out_height, out_width, 3),         # target images
                        (batch_size, out_height, out_width, 1)]         # target masks

        self.fixed_producer_dispatcher = FixedProducerDispatcher(
            method=self.load_batches,
            shapes=batch_shapes,
            in_queue=queue_in,
            out_queue=queue_out,
<<<<<<< HEAD
            args=(images, side, batch_size, augmenting, do_shuffle))
        load_process.start()
        logger.debug("Batching to queue: (side: '%s', augmenting: %s)", side, augmenting)
        return self.minibatch(side, augmenting, load_process)

    @staticmethod
    def make_queues(side, purpose):
        """ Create the buffer token queues for Fixed Producer Dispatcher """
        q_names = ["{}_side_{}_{}".format(purpose, side, direction) for direction in ("in", "out")]
=======
            args=(images, side, is_display, do_shuffle, batchsize))
        self.fixed_producer_dispatcher.start()
        logger.debug("Batching to queue: (side: '%s', is_display: %s)", side, is_display)
        return self.minibatch(side, is_display, self.fixed_producer_dispatcher)

    def join_subprocess(self):
        """ Join the FixedProduceerDispatcher subprocess from outside this module """
        logger.debug("Joining FixedProducerDispatcher")
        if self.fixed_producer_dispatcher is None:
            logger.debug("FixedProducerDispatcher not yet initialized. Exiting")
            return
        self.fixed_producer_dispatcher.join()
        logger.debug("Joined FixedProducerDispatcher")

    @staticmethod
    def make_queues(side, is_preview, is_timelapse):
        """ Create the buffer token queues for Fixed Producer Dispatcher """
        q_name = "_{}".format(side)
        if is_preview:
            q_name = "{}{}".format("preview", q_name)
        elif is_timelapse:
            q_name = "{}{}".format("timelapse", q_name)
        else:
            q_name = "{}{}".format("train", q_name)
        q_names = ["{}_{}".format(q_name, direction) for direction in ("in", "out")]
>>>>>>> staging
        logger.debug(q_names)
        queues = [queue_manager.get_queue(queue) for queue in q_names]
        return queues

<<<<<<< HEAD
    def load_batches(self, mem_gen, images, side, batch_size, augmenting, do_shuffle):
        """ Load the warped images and target images to queue """
        logger.debug("Loading batch: (image_count: %s, side: '%s', augmenting: %s, "
                     "do_shuffle: %s)", len(images), side, augmenting, do_shuffle)
        def batch_gen(images, landmarks, batch_size):
            """ doc string """
=======
    def load_batches(self, mem_gen, images, side, is_display,
                     do_shuffle=True, batchsize=0):
        """ Load the warped images and target images to queue """
        logger.debug("Loading batch: (image_count: %s, side: '%s', is_display: %s, "
                     "do_shuffle: %s)", len(images), side, is_display, do_shuffle)
        self.validate_samples(images)
        # Intialize this for each subprocess
        self._nearest_landmarks = dict()

        def _img_iter(imgs):
>>>>>>> staging
            while True:
                if do_shuffle:
                    rng_state = np.random.get_state()
                    np.random.set_state(rng_state)
                    np.random.shuffle(images)
                    np.random.set_state(rng_state)
                    np.random.shuffle(landmarks)
                gen = zip(images, landmarks)
                for image_num, (image, landmark) in enumerate(gen):
                    yield (image_num + 1) % batch_size, image, landmark

        #self.validate_samples(images["images"])
        # Intialize this for each subprocess
        self._nearest_landmarks = dict()
        img_npy = np.memmap(images["images"], dtype='float32', mode='c',shape=images["data_shape"])
        batcher = batch_gen(img_npy, images["landmarks"], batch_size)
        epoch = 0

        for memory_wrapper in mem_gen:
            memory = memory_wrapper.get()
<<<<<<< HEAD
            logger.trace("Putting to batch queue: (side: '%s', augmenting: %s)", side, augmenting)
            for image_num, image, landmark in batcher:
                imgs = self.process_faces(image, landmark, side, augmenting, epoch)
                for process_output_num, img in enumerate(imgs):
                    memory[process_output_num][image_num - 1][:] = img
                epoch += 1
                if image_num == 0:
                    print(epoch, " batch number created")
                    memory_wrapper.ready()

        logger.debug("Finished batching: (epoch: %s, side: '%s', augmenting: %s)",
                     epoch, side, augmenting)
=======
            logger.trace("Putting to batch queue: (side: '%s', is_display: %s)",
                         side, is_display)
            for i, img_path in enumerate(img_iter):
                imgs = self.process_face(img_path, side, is_display)
                for j, img in enumerate(imgs):
                    memory[j][i][:] = img
                epoch += 1
                if i == batchsize - 1:
                    break
            memory_wrapper.ready()
        logger.debug("Finished batching: (epoch: %s, side: '%s', is_display: %s)",
                     epoch, side, is_display)
>>>>>>> staging

    def validate_samples(self, data):
        """ Check the total number of images against batchsize and return
            the total number of images """
        length = data.shape[0]
        msg = ("Number of images is lower than batch-size (Note that too few "
               "images may lead to bad training). # images: {}, "
<<<<<<< HEAD
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
        if augmenting:
=======
               "batch-size: {}".format(length, self.batchsize))
        try:
            assert length >= self.batchsize, msg
        except AssertionError as err:
            msg += ("\nYou should increase the number of images in your training set or lower "
                    "your batch-size.")
            raise FaceswapError(msg) from err

    @staticmethod
    def minibatch(side, is_display, load_process):
        """ A generator function that yields epoch, batchsize of warped_img
            and batchsize of target_img from the load queue """
        logger.debug("Launching minibatch generator for queue (side: '%s', is_display: %s)",
                     side, is_display)
        for batch_wrapper in load_process:
            with batch_wrapper as batch:
                logger.trace("Yielding batch: (size: %s, item shapes: %s, side:  '%s', "
                             "is_display: %s)",
                             len(batch), [item.shape for item in batch], side, is_display)
                yield batch
        load_process.stop()
        logger.debug("Finished minibatch generator for queue: (side: '%s', is_display: %s)",
                     side, is_display)
        load_process.join()

    def process_face(self, filename, side, is_display):
        """ Load an image and perform transformation and warping """
        logger.trace("Process face: (filename: '%s', side: '%s', is_display: %s)",
                     filename, side, is_display)
        image = cv2_read_img(filename, raise_error=True)
        if self.mask_class or self.training_opts["warp_to_landmarks"]:
            src_pts = self.get_landmarks(filename, image, side)
        if self.mask_class:
            image = self.mask_class(src_pts, image, channels=4).mask

        image = self.processing.color_adjust(image,
                                             self.training_opts["augment_color"],
                                             is_display)

        if not is_display:
>>>>>>> staging
            image = self.processing.random_transform(image)
            if not self.training_opts["no_flip"]:
                image = self.processing.do_random_flip(image)
        if self.training_opts["warp_to_landmarks"]:
            dst_pts = self.get_closest_match(filename, side, src_pts)
            processed = self.processing.random_warp_landmarks(image, src_pts, dst_pts)
        else:
            processed = self.processing.random_warp(image)
        logger.trace("Processed face: (image: '%s', side: '%s', shapes: %s)",
                     image, side, [img.shape for img in processed])
        return processed

<<<<<<< HEAD
=======
    def get_landmarks(self, filename, image, side):
        """ Return the landmarks for this face """
        logger.trace("Retrieving landmarks: (filename: '%s', side: '%s'", filename, side)
        lm_key = sha1(image).hexdigest()
        try:
            src_points = self.landmarks[side][lm_key]
        except KeyError as err:
            msg = ("At least one of your images does not have a matching entry in your alignments "
                   "file."
                   "\nIf you are training with a mask or using 'warp to landmarks' then every "
                   "face you intend to train on must exist within the alignments file."
                   "\nThe specific file that caused the failure was '{}' which has a hash of {}."
                   "\nMost likely there will be more than just this file missing from the "
                   "alignments file. You can use the Alignments Tool to help identify missing "
                   "alignments".format(lm_key, filename))
            raise FaceswapError(msg) from err
        logger.trace("Returning: (src_points: %s)", src_points)
        return src_points

>>>>>>> staging
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
    def __init__(self, input_size, output_size, coverage_ratio, config):
        """ input_size: Size of the face input into the model
            output_size: Size of the face that comes out of the modell
            coverage_ratio: Coverage ratio of full image. Eg: 256 * 0.625 = 160
        """
        logger.debug("Initializing %s: (input_size: %s, output_size: %s, coverage_ratio: %s, "
                     "config: %s)", self.__class__.__name__, input_size, output_size,
                     coverage_ratio, config)
        self.config = config
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

<<<<<<< HEAD
=======
    def color_adjust(self, img, augment_color, is_display):
        """ Color adjust RGB image """
        logger.trace("Color adjusting image")
        if not is_display and augment_color:
            logger.trace("Augmenting color")
            face, _ = self.separate_mask(img)
            face = face.astype("uint8")
            face = self.random_contrast(face)
            face = self.random_lab(face)
            img[:, :, :3] = face
        return img.astype('float32') / 255.0

    def random_contrast(self, image):
        """ Randomly perform Contrast Limited Adaptive Histogram Equilization """
        contrast_random = random()
        if contrast_random > self.config.get("color_clahe_chance", 50) / 100:
            return image

        base_contrast = image.shape[0] // 128
        grid_base = random() * self.config.get("color_clahe_max_size", 4)
        contrast_adjustment = int(grid_base * (base_contrast / 2))
        grid_size = base_contrast + contrast_adjustment
        logger.trace("Adjusting Contrast. Grid Size: %s", grid_size)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(grid_size, grid_size))
        for chan in range(3):
            image[:, :, chan] = clahe.apply(image[:, :, chan])
        return image

    def random_lab(self, image):
        # pylint:disable=no-member
        """ Perform random color/lightness adjustment in L*a*b* colorspace """
        amount_l = self.config.get("color_lightness", 30.) / 100
        amount_ab = self.config.get("color_ab", 8.) / 100

        randoms = [(random() * amount_l * 2) - amount_l,  # L adjust
                   (random() * amount_ab * 2) - amount_ab,  # A adjust
                   (random() * amount_ab * 2) - amount_ab]  # B adjust

        logger.trace("Random LAB adjustments: %s", randoms)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype("float32") / 255.

        for idx, adjustment in enumerate(randoms):
            if adjustment >= 0:
                image[:, :, idx] = ((1 - image[:, :, idx]) * adjustment) + image[:, :, idx]
            else:
                image[:, :, idx] = image[:, :, idx] * (1 + adjustment)
        image = cv2.cvtColor((image * 255.0).astype("uint8"), cv2.COLOR_LAB2BGR)
        return image

>>>>>>> staging
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
        sample = image.copy()[:, :, :3]
        height, width = image.shape[0:2]
        coverage = self.get_coverage(image) // 2
        try:
            assert height == width and height % 2 == 0
        except AssertionError as err:
            msg = ("Training images should be square with an even number of pixels across each "
                   "side. An image was found with width: {}, height: {}."
                   "\nMost likely this is a frame rather than a face within your training set. "
                   "\nMake sure that the only images within your training set are faces generated "
                   "from the Extract process.".format(width, height))
            raise FaceswapError(msg) from err

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
        return [sample, warped_image, warped_mask, target_image, target_mask]

    def random_warp_landmarks(self, image, source=None, destination=None):
        """ get warped image, target image and target mask From DFAKER plugin """
        # pylint: disable=no-member
        logger.trace("Randomly warping landmarks")
        sample = image.copy()[:, :, :3]
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
        return [sample, warped_image, warped_mask, target_image, target_mask]
