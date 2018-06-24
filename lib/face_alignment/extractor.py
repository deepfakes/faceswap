#!/usr/bin python3
""" Facial landmarks extractor for faceswap.py
    Code adapted and modified from:
    https://github.com/1adrianb/face-alignment
"""

import os

import cv2
import numpy as np

from lib import gpu_stats

from .detectors import DLibDetector
from .model import KerasModel


class GPUMem(object):
    """ Sets the scale to factor for dlib images
        and the ratio of vram to use for tensorflow """

    def __init__(self):
        self.gpu_memory = min(gpu_stats.GPUStats(verbose=True).get_free())
        self.tensorflow_ratio = self.set_tensor_gpu_ratio()
        self.scale_to = self.set_scale_to()

    def set_tensor_gpu_ratio(self):
        """ Set the ratio of GPU memory to use
            for tensorflow session

            Ideally at least 2304MB is required, but
            will run with less (with warnings) """

        if self.gpu_memory < 2000:
            ratio = 1024.0 / self.gpu_memory
        elif self.gpu_memory < 3000:
            ratio = 1560.0 / self.gpu_memory
        elif self.gpu_memory < 4000:
            ratio = 2048.0 / self.gpu_memory
        else:
            ratio = 2304.0 / self.gpu_memory
        return ratio

    def set_scale_to(self):
        """ Set the size to scale images down to for specific
            gfx cards.
            DLIB VRAM allocation is linear to pixel count """
        buffer = 256
        free_mem = (self.gpu_memory * (1 - self.tensorflow_ratio)) - buffer
        gradient = 213 / 524288
        constant = 307
        scale_to = int((free_mem - constant) / gradient)
        return scale_to


VRAM = GPUMem()
DLIB_DETECTORS = DLibDetector()
KERAS_MODEL = KerasModel(VRAM.tensorflow_ratio)


class Frame(object):
    """ The current frame for processing """

    def __init__(self, input_image, verbose, input_is_predetected_face):
        self.verbose = verbose
        self.scale_to = VRAM.scale_to

        self.height, self.width = input_image.shape[:2]
        self.input_scale = 1.0
        self.images = self.process_input(input_image,
                                         input_is_predetected_face)

    def process_input(self, input_image, input_is_predetected_face):
        """ Process import image:
            Size down if required
            Duplicate into rgb colour space """
        if not input_is_predetected_face:
            input_image = self.scale_down(input_image)
        return self.compile_color_space(input_image)

    def scale_down(self, image):
        """ Scale down large images based on vram amount """
        pixel_count = self.width * self.height

        if pixel_count > self.scale_to:
            self.input_scale = self.scale_to / pixel_count
            dimensions = (int(self.width * self.input_scale),
                          int(self.height * self.input_scale))
            if self.verbose:
                print("Resizing image from {}x{} "
                      "to {}.".format(str(self.width), str(self.height),
                                      "x".join(str(i) for i in dimensions)))
            image = cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)

        return image

    @staticmethod
    def compile_color_space(image_bgr):
        """ cv2 and numpy inputs differs in rgb-bgr order
        this affects chance of dlib face detection so
        pass both versions """
        image_rgb = image_bgr[:, :, ::-1].copy()
        return (image_rgb, image_bgr)

    @staticmethod
    def transform(point, center, scale, resolution):
        """ Transform Image """
        pnt = np.array([point[0], point[1], 1.0])
        hscl = 200.0 * scale
        eye = np.eye(3)
        eye[0, 0] = resolution / hscl
        eye[1, 1] = resolution / hscl
        eye[0, 2] = resolution * (-center[0] / hscl + 0.5)
        eye[1, 2] = resolution * (-center[1] / hscl + 0.5)
        eye = np.linalg.inv(eye)
        return np.matmul(eye, pnt)[0:2]

    def crop(self, image, center, scale, resolution=256.0):
        """ Crop image around the center point """
        ul = self.transform([1, 1], center, scale, resolution).astype(np.int)
        br = self.transform([resolution, resolution],
                            center,
                            scale,
                            resolution).astype(np.int)
        if image.ndim > 2:
            new_dim = np.array([br[1] - ul[1], br[0] - ul[0], image.shape[2]],
                               dtype=np.int32)
            new_img = np.zeros(new_dim, dtype=np.uint8)
        else:
            new_dim = np.array([br[1] - ul[1], br[0] - ul[0]], dtype=np.int)
            new_img = np.zeros(new_dim, dtype=np.uint8)
        height = image.shape[0]
        width = image.shape[1]
        new_x = np.array([max(1, -ul[0] + 1), min(br[0], width) - ul[0]],
                         dtype=np.int32)
        new_y = np.array([max(1, -ul[1] + 1), min(br[1], height) - ul[1]],
                         dtype=np.int32)
        old_x = np.array([max(1, ul[0] + 1), min(br[0], width)],
                         dtype=np.int32)
        old_y = np.array([max(1, ul[1] + 1), min(br[1], height)],
                         dtype=np.int32)
        new_img[new_y[0] - 1:new_y[1],
                new_x[0] - 1:new_x[1]] = image[old_y[0] - 1:old_y[1],
                                               old_x[0] - 1:old_x[1], :]
        new_img = cv2.resize(new_img,
                             dsize=(int(resolution), int(resolution)),
                             interpolation=cv2.INTER_LINEAR)
        return new_img

    # TODO Move This
    def get_pts_from_predict(self, var_a, center, scale):
        """ Get points from predictor """
        var_b = var_a.reshape((var_a.shape[0],
                               var_a.shape[1] * var_a.shape[2]))
        var_c = var_b.argmax(1).reshape((var_a.shape[0],
                                         1)).repeat(2,
                                                    axis=1).astype(np.float)
        var_c[:, 0] %= var_a.shape[2]
        var_c[:, 1] = np.apply_along_axis(
            lambda x: np.floor(x / var_a.shape[2]),
            0,
            var_c[:, 1])

        for i in range(var_a.shape[0]):
            pt_x, pt_y = int(var_c[i, 0]), int(var_c[i, 1])
            if pt_x > 0 and pt_x < 63 and pt_y > 0 and pt_y < 63:
                diff = np.array([var_a[i, pt_y, pt_x+1]
                                 - var_a[i, pt_y, pt_x-1],
                                 var_a[i, pt_y+1, pt_x]
                                 - var_a[i, pt_y-1, pt_x]])

                var_c[i] += np.sign(diff)*0.25

        var_c += 0.5
        return [self.transform(var_c[i], center, scale, var_a.shape[2])
                for i in range(var_a.shape[0])]


class Extract(object):
    """ Extracts faces from an image, crops and
        calculates landmarks """

    def __init__(self, input_image_bgr, detector, verbose,
                 input_is_predetected_face=False):
        self.verbose = verbose
        self.keras = KERAS_MODEL
        self.dlib = DLIB_DETECTORS

        self.set_verbose()
        self.initialise(detector)

        self.frame = Frame(input_image_bgr, verbose, input_is_predetected_face)

        self.detect_faces(input_is_predetected_face)
        self.landmarks = self.process_landmarks()

    def set_verbose(self):
        """ Set verbosity levels of Keras and Dlib """
        if self.verbose:
            self.keras.verbose = True
            self.dlib.verbose = True

    def initialise(self, detector):
        """ Initialise Keras and Dlib """
        self.keras.load_model()
        self.dlib.add_detectors(detector)

    def detect_faces(self, input_is_predetected_face):
        """ Detect faces """
        # Predetected_face is used for sort tool.
        # Landmarks should not be extracted again from predetected faces,
        # because face data is lost, resulting in a large variance
        # against extract from original image

        if input_is_predetected_face:
            self.dlib.set_predetected(self.frame.width, self.frame.height)
        else:
            self.dlib.detect_faces(self.frame.images)

    def process_landmarks(self):
        """ Process the 68 point facial landmarks """

        landmarks = list()
        if self.dlib.detected_faces:
            for d_rect in self.dlib.detected_faces:

                if self.dlib.is_mmod_rectangle(d_rect):
                    d_rect = d_rect.rect

                left = d_rect.left()
                top = d_rect.top()
                right = d_rect.right()
                bottom = d_rect.bottom()

                del d_rect

                center = np.array([(left + right) / 2.0, (top + bottom) / 2.0])
                center[1] -= (bottom - top) * 0.12
                scale = (right - left + bottom - top) / 195.0

                image = self.frame.crop(
                    self.frame.images[0],
                    center,
                    scale).transpose((2, 0, 1)).astype(np.float32) / 255.0

                image = np.expand_dims(image, 0)

                pts_img = self.frame.get_pts_from_predict(
                    self.keras.model.predict(image)[-1][0],
                    center,
                    scale)

                pts_img = [(int(pt[0] / self.frame.input_scale),
                            int(pt[1] / self.frame.input_scale))
                           for pt in pts_img]

                landmarks.append(((int(left / self.frame.input_scale),
                                   int(top / self.frame.input_scale),
                                   int(right / self.frame.input_scale),
                                   int(bottom / self.frame.input_scale)),
                                  pts_img))
        elif self.verbose:
            print("Warning: No faces were detected.")

        return landmarks


# TODO Remove This
def write_image(image, imname):
    """ TODO Remove this temporary file preview """
    impath = "/home/matt/fake/test/extract"
    img = "test_{}.jpg".format(imname)
    imgfile = os.path.join(impath, img)
    cv2.imwrite(imgfile, image)
