#!/usr/bin/env python3
""" Process training data for model training """

import os
from random import shuffle
import uuid

import cv2
import numpy as np

from lib.alignments import Alignments
from lib.multithreading import MultiThread
from lib.queue_manager import queue_manager
from lib.umeyama import umeyama


class TrainingDataGenerator():
    """ Generate training data for models """
    def __init__(self, random_transform_args, coverage,
                 scale=5, zoom=1, training_opts=None):
        self.options = training_opts
        self.random_transform_args = random_transform_args
        self.coverage = coverage
        self.scale = scale
        self.zoom = zoom
        self.batchsize = 0

    def minibatch_ab(self, images, batchsize, do_shuffle=True):
        """ Keep a queue filled to 8x Batch Size """
        self.batchsize = batchsize
        options = self.process_training_options(images)
        q_name = str(uuid.uuid4())
        q_size = batchsize * 8
        queue_manager.add_queue(q_name, maxsize=q_size)
        thread = MultiThread()
        thread.in_thread(self.load_batches,
                         images,
                         q_name,
                         options,
                         do_shuffle)
        return self.minibatch(q_name)

    def process_training_options(self, images):
        """ Process the model specific training data """
        opts = dict()
        if not self.options or not isinstance(self.options, dict):
            return opts
        if self.options.get("use_alignments", False):
            opts["alignments"] = self.get_alignments(images)
        opts["use_mask"] = self.options.get("use_mask", False)
        return opts

    def get_alignments(self, images):
        """ Return the alignments for current image folder """
        image_folder = os.path.dirname(images[0])
        alignments = Alignments(image_folder,
                                filename="alignments",
                                serializer=self.options.get("serializer"))
        alignments.load()
        return alignments

    def load_batches(self, data, q_name, options, do_shuffle=True):
        """ Load the epoch, warped images and target images to queue """
        epoch = 0
        queue = queue_manager.get_queue(q_name)
        self.validate_samples(data)
        while True:
            if do_shuffle:
                shuffle(data)
            for img in data:
                queue.put((epoch, np.float32(self.process_face(img, None))))
            epoch += 1

    def validate_samples(self, data):
        """ Check the total number of images against batchsize and return
            the total number of images """
        length = len(data)
        msg = ("Number of images is lower than batch-size (Note that too few "
               "images may lead to bad training). # images: {}, "
               "batch-size: {}".format(length, self.batchsize))
        assert length >= self.batchsize, msg

    def minibatch(self, q_name):
        """ A generator function that yields epoch, batchsize of warped_img
            and batchsize of target_img from the load queue """
        queue = queue_manager.get_queue(q_name)
        while True:
            batch = list()
            for _ in range(self.batchsize):
                epoch, images = queue.get()
                batch.append(images)
            rtn = np.array(batch)
            yield epoch, rtn[:, 0, :, :, :], rtn[:, 1, :, :, :]

    def process_face(self, filename, landmarks):
        """ Load an image and perform transformation and warping """
        try:
            # pylint: disable=no-member
            image = self.color_adjust(cv2.imread(filename))
        except TypeError:
            raise Exception("Error while reading image", filename)

        image = self.random_transform(image)
        warped_img, target_img = self.random_warp(image)

        return warped_img, target_img

    @staticmethod
    def color_adjust(img):
        """ Color adjust RGB image """
        return img / 255.0

    def random_transform(self, image):
        """ Randomly transform an image """
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
        return result

    def random_warp(self, image):
        """ get pair of random warped images from aligned face image """
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

        return warped_image, target_image


def stack_images(images):
    """ Stack images """
    def get_transpose_axes(num):
        if num % 2 == 0:
            y_axes = list(range(1, num - 1, 2))
            x_axes = list(range(0, num - 1, 2))
        else:
            y_axes = list(range(0, num - 1, 2))
            x_axes = list(range(1, num - 1, 2))
        return y_axes, x_axes, [num - 1]

    images_shape = np.array(images.shape)
    new_axes = get_transpose_axes(len(images_shape))
    new_shape = [np.prod(images_shape[x]) for x in new_axes]
    return np.transpose(
        images,
        axes=np.concatenate(new_axes)
        ).reshape(new_shape)
