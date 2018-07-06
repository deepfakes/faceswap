#!/usr/bin python3
""" Facial landmarks extractor for faceswap.py
    Code adapted and modified from:
    https://github.com/1adrianb/face-alignment
"""

import cv2
import numpy as np

from .detectors import DLibDetector, MTCNNDetector
from .vram_allocation import GPUMem
from .model import KerasModel

DLIB_DETECTORS = DLibDetector()
MTCNN_DETECTOR = MTCNNDetector()
VRAM = GPUMem()
KERAS_MODEL = KerasModel()


class Frame(object):
    """ The current frame for processing """

    def __init__(self, detector, input_image,
                 verbose, input_is_predetected_face):
        self.verbose = verbose
        self.height, self.width = input_image.shape[:2]

        if not VRAM.scale_to and VRAM.device != -1:
            VRAM.set_scale_to(detector)

        if VRAM.device != -1:
            self.scale_to = VRAM.scale_to
        else:
            self.scale_to = self.height * self.width

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


class Align(object):
    """ Perform transformation to align and get landmarks """
    def __init__(self, frame, detected_faces, keras_model, verbose):
        self.verbose = verbose
        self.frame = frame.images[0]
        self.input_scale = frame.input_scale
        self.detected_faces = detected_faces
        self.keras = keras_model

        self.bounding_box = None
        self.landmarks = self.process_landmarks()

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
        v_ul = self.transform([1, 1], center, scale, resolution).astype(np.int)
        v_br = self.transform([resolution, resolution],
                              center,
                              scale,
                              resolution).astype(np.int)
        if image.ndim > 2:
            new_dim = np.array([v_br[1] - v_ul[1],
                                v_br[0] - v_ul[0],
                                image.shape[2]],
                               dtype=np.int32)
            new_img = np.zeros(new_dim, dtype=np.uint8)
        else:
            new_dim = np.array([v_br[1] - v_ul[1],
                                v_br[0] - v_ul[0]],
                               dtype=np.int)
            new_img = np.zeros(new_dim, dtype=np.uint8)
        height = image.shape[0]
        width = image.shape[1]
        new_x = np.array([max(1, -v_ul[0] + 1), min(v_br[0], width) - v_ul[0]],
                         dtype=np.int32)
        new_y = np.array([max(1, -v_ul[1] + 1),
                          min(v_br[1], height) - v_ul[1]],
                         dtype=np.int32)
        old_x = np.array([max(1, v_ul[0] + 1), min(v_br[0], width)],
                         dtype=np.int32)
        old_y = np.array([max(1, v_ul[1] + 1), min(v_br[1], height)],
                         dtype=np.int32)
        new_img[new_y[0] - 1:new_y[1],
                new_x[0] - 1:new_x[1]] = image[old_y[0] - 1:old_y[1],
                                               old_x[0] - 1:old_x[1], :]
        new_img = cv2.resize(new_img,
                             dsize=(int(resolution), int(resolution)),
                             interpolation=cv2.INTER_LINEAR)
        return new_img

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

    def process_landmarks(self):
        """ Align image and process landmarks """
        landmarks = list()
        if not self.detected_faces:
            if self.verbose:
                print("Warning: No faces were detected.")
            return landmarks

        for d_rect in self.detected_faces:
            self.get_bounding_box(d_rect)
            del d_rect

            center, scale = self.get_center_scale()
            image = self.align_image(center, scale)

            landmarks_xy = self.predict_landmarks(image, center, scale)

            landmarks.append((
                (int(self.bounding_box['left'] / self.input_scale),
                 int(self.bounding_box['top'] / self.input_scale),
                 int(self.bounding_box['right'] / self.input_scale),
                 int(self.bounding_box['bottom'] / self.input_scale)),
                landmarks_xy))

        return landmarks

    def get_bounding_box(self, d_rect):
        """ Return the corner points of the bounding box """
        self.bounding_box = {'left': d_rect.left(),
                             'top': d_rect.top(),
                             'right': d_rect.right(),
                             'bottom': d_rect.bottom()}

    def get_center_scale(self):
        """ Get the center and set scale of bounding box """
        center = np.array([(self.bounding_box['left']
                            + self.bounding_box['right']) / 2.0,
                           (self.bounding_box['top']
                            + self.bounding_box['bottom']) / 2.0])

        center[1] -= (self.bounding_box['bottom']
                      - self.bounding_box['top']) * 0.12

        scale = (self.bounding_box['right']
                 - self.bounding_box['left']
                 + self.bounding_box['bottom']
                 - self.bounding_box['top']) / 195.0

        return center, scale

    def align_image(self, center, scale):
        """ Crop and align image around center """
        image = self.crop(
            self.frame,
            center,
            scale).transpose((2, 0, 1)).astype(np.float32) / 255.0

        return np.expand_dims(image, 0)

    def predict_landmarks(self, image, center, scale):
        """ Predict the 68 point landmarks """
        with self.keras.session.as_default():
            pts_img = self.get_pts_from_predict(
                self.keras.model.predict(image)[-1][0],
                center,
                scale)

        return [(int(pt[0] / self.input_scale),
                 int(pt[1] / self.input_scale))
                for pt in pts_img]


class Extract(object):
    """ Extracts faces from an image, crops and
        calculates landmarks """

    def __init__(self, input_image_bgr, detector, mtcnn_kwargs=None,
                 verbose=False, input_is_predetected_face=False):
        self.initialized = False
        self.verbose = verbose
        self.keras = KERAS_MODEL
        self.detector = None

        self.initialize(detector, mtcnn_kwargs)

        self.frame = Frame(detector=detector,
                           input_image=input_image_bgr,
                           verbose=verbose,
                           input_is_predetected_face=input_is_predetected_face)

        self.detect_faces(input_is_predetected_face)
        self.convert_to_dlib_rectangle()

        self.landmarks = Align(frame=self.frame,
                               detected_faces=self.detector.detected_faces,
                               keras_model=self.keras,
                               verbose=self.verbose).landmarks

    def initialize(self, detector, mtcnn_kwargs):
        """ initialize Keras and Dlib """
        if self.initialized:
            return
        self.initialize_vram(detector)

        self.initialize_keras(detector)
        self.initialize_detector(detector, mtcnn_kwargs)
        self.initialized = True

    def initialize_vram(self, detector):
        """ Initialize vram based on detector """
        VRAM.verbose = self.verbose
        VRAM.detector = detector
        VRAM.output_stats()

    def initialize_keras(self, detector):
        """ Initialize keras. Allocate vram to tensorflow
            based on detector """
        ratio = None
        if detector != "mtcnn" and VRAM.device != -1:
            ratio = VRAM.get_tensor_gpu_ratio()
        placeholder = np.zeros((1, 3, 256, 256))
        self.keras.load_model(verbose=self.verbose,
                              ratio=ratio,
                              dummy=placeholder)

    def initialize_detector(self, detector, mtcnn_kwargs):
        """ Initialize face detector """
        kwargs = {"verbose": self.verbose}
        if detector == "mtcnn":
            self.detector = MTCNN_DETECTOR
            mtcnn_kwargs = self.detector.validate_kwargs(mtcnn_kwargs)
            kwargs["mtcnn_kwargs"] = mtcnn_kwargs
        else:
            self.detector = DLIB_DETECTORS
            kwargs["detector"] = detector

        self.detector.create_detector(**kwargs)

    def detect_faces(self, input_is_predetected_face):
        """ Detect faces """
        # Predetected_face is used for sort tool.
        # Landmarks should not be extracted again from predetected faces,
        # because face data is lost, resulting in a large variance
        # against extract from original image

        if input_is_predetected_face:
            self.detector.set_predetected(self.frame.width, self.frame.height)
        else:
            self.detector.detect_faces(self.frame.images)

    def convert_to_dlib_rectangle(self):
        """ Convert detected faces to dlib_rectangle """
        detected = [d_rect.rect
                    if self.detector.is_mmod_rectangle(d_rect)
                    else d_rect
                    for d_rect in self.detector.detected_faces]

        self.detector.detected_faces = detected
