#!/usr/bin python3
""" Facial landmarks extractor for faceswap.py
    Code adapted and modified from:
    https://github.com/1adrianb/face-alignment
"""

import cv2
import numpy as np

from .detectors import DLibDetector, MTCNNDetector, ManualDetector
from .vram_allocation import GPUMem
from .model import KerasModel

DLIB_DETECTORS = DLibDetector()
MTCNN_DETECTOR = MTCNNDetector()
MANUAL_DETECTOR = ManualDetector()
VRAM = GPUMem()
KERAS_MODEL = KerasModel()


class Frame():
    """ The current frame for processing """

    def __init__(self, detector, input_image,
                 verbose, input_is_predetected_face):
        self.verbose = verbose
        self.height, self.width = input_image.shape[:2]

        self.input_scale = 1.0

        self.image_bgr = input_image
        self.image_rgb = input_image[:, :, ::-1].copy()
        self.image_detect = self.scale_image(input_is_predetected_face,
                                             detector)

    def scale_image(self, input_is_predetected_face, detector):
        """ Scale down large images based on vram amount """
        image = self.image_rgb
        if input_is_predetected_face:
            return image

        if detector == "mtcnn":
            self.scale_mtcnn()
        elif detector == "manual":
            self.input_scale = 1.0
        else:
            self.scale_dlib()

        if self.input_scale == 1.0:
            return image

        if self.input_scale > 1.0:
            interpolation = cv2.INTER_LINEAR
        else:
            interpolation = cv2.INTER_AREA

        dimensions = (int(self.width * self.input_scale),
                      int(self.height * self.input_scale))
        if self.verbose and self.input_scale < 1.0:
            print("Resizing image from {}x{} "
                  "to {}.".format(str(self.width), str(self.height),
                                  "x".join(str(i) for i in dimensions)))
        image = cv2.resize(image,
                           dimensions,
                           interpolation=interpolation).copy()

        return image

    def scale_mtcnn(self):
        """ Set scaling for mtcnn """
        pixel_count = self.width * self.height
        if pixel_count > VRAM.scale_to:
            self.input_scale = (VRAM.scale_to / pixel_count)**0.5

    def scale_dlib(self):
        """ Set scaling for dlib

        DLIB is finickity, and pure pixel count won't help as when an
        initial portrait image goes in, rotating it to landscape sucks
        up VRAM for no discernible reason. This does not happen when the
        initial image is a landscape image.
        To mitigate this we need to make sure that all images fit within
        a square based on the pixel count
        There is also no way to set the acceptable size for a positive
        match, so all images should be scaled to the maximum possible
        to detect all available faces """

        max_length_scale = int(VRAM.scale_to ** 0.5)
        max_length_image = max(self.height, self.width)
        self.input_scale = max_length_scale / max_length_image


class Align():
    """ Perform transformation to align and get landmarks """
    def __init__(self, image, detected_faces, keras_model, verbose):
        self.verbose = verbose
        self.image = image
        self.detected_faces = detected_faces
        self.keras = keras_model

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

        for detected_face in self.detected_faces:

            center, scale = self.get_center_scale(detected_face)
            image = self.align_image(center, scale)

            landmarks_xy = self.predict_landmarks(image, center, scale)

            landmarks.append(((detected_face['left'],
                               detected_face['top'],
                               detected_face['right'],
                               detected_face['bottom']),
                              landmarks_xy))

        return landmarks

    @staticmethod
    def get_center_scale(detected_face):
        """ Get the center and set scale of bounding box """
        center = np.array([(detected_face['left']
                            + detected_face['right']) / 2.0,
                           (detected_face['top']
                            + detected_face['bottom']) / 2.0])

        center[1] -= (detected_face['bottom']
                      - detected_face['top']) * 0.12

        scale = (detected_face['right']
                 - detected_face['left']
                 + detected_face['bottom']
                 - detected_face['top']) / 195.0

        return center, scale

    def align_image(self, center, scale):
        """ Crop and align image around center """
        image = self.crop(
            self.image,
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

        return [(int(pt[0]), int(pt[1])) for pt in pts_img]


class Extract():
    """ Extracts faces from an image, crops and
        calculates landmarks """

    def __init__(self, input_image_bgr, detector, dlib_buffer=64,
                 mtcnn_kwargs=None, verbose=False,
                 input_is_predetected_face=False,
                 initialize_only=False):
        self.verbose = verbose
        self.keras = KERAS_MODEL
        self.detector_name = detector
        self.detector = None
        self.frame = None
        self.bounding_boxes = None
        self.landmarks = None

        self.initialize(mtcnn_kwargs, dlib_buffer)

        if not initialize_only:
            self.execute(input_image_bgr, input_is_predetected_face)

    def initialize(self, mtcnn_kwargs, dlib_buffer):
        """ initialize Keras and Dlib """
        if not VRAM.initialized:
            self.initialize_vram(dlib_buffer)

        if not self.keras.initialized:
            self.initialize_keras()
            # VRAM Scaling factor must be set AFTER Keras has loaded
            VRAM.set_scale_to(self.detector_name)

        if self.detector_name == "mtcnn":
            self.detector = MTCNN_DETECTOR
        elif self.detector_name == "manual":
            self.detector = MANUAL_DETECTOR
        else:
            self.detector = DLIB_DETECTORS

        if not self.detector.initialized:
            self.initialize_detector(mtcnn_kwargs)

    def initialize_vram(self, dlib_buffer):
        """ Initialize vram based on detector """
        VRAM.verbose = self.verbose
        VRAM.detector = self.detector_name
        if dlib_buffer > VRAM.dlib_buffer:
            VRAM.dlib_buffer = dlib_buffer
        VRAM.initialized = True
        VRAM.output_stats()

    def initialize_keras(self):
        """ Initialize keras. Allocate vram to tensorflow
            based on detector """
        ratio = None
        if self.detector_name != "mtcnn" and VRAM.device != -1:
            ratio = VRAM.get_tensor_gpu_ratio()
        placeholder = np.zeros((1, 3, 256, 256))
        self.keras.load_model(verbose=self.verbose,
                              ratio=ratio,
                              dummy=placeholder)

    def initialize_detector(self, mtcnn_kwargs):
        """ Initialize face detector """
        kwargs = {"verbose": self.verbose}
        if self.detector_name == "mtcnn":
            mtcnn_kwargs = self.detector.validate_kwargs(mtcnn_kwargs)
            kwargs["mtcnn_kwargs"] = mtcnn_kwargs
        elif self.detector_name != "manual":
            kwargs["detector"] = self.detector_name
            scale_to = int(VRAM.scale_to ** 0.5)

            if self.verbose:
                print(self.detector.compiled_for_cuda())
                print("Initializing DLib for frame size {}x{}".format(
                    str(scale_to), str(scale_to)))

            placeholder = np.zeros((scale_to, scale_to, 3), dtype=np.uint8)
            kwargs["placeholder"] = placeholder

        self.detector.create_detector(**kwargs)

    def execute(self, input_image_bgr,
                input_is_predetected_face=False, manual_face=None):
        """ Execute extract """
        self.frame = Frame(detector=self.detector_name,
                           input_image=input_image_bgr,
                           verbose=self.verbose,
                           input_is_predetected_face=input_is_predetected_face)

        self.detect_faces(input_is_predetected_face, manual_face)
        self.bounding_boxes = self.get_bounding_boxes()

        self.landmarks = Align(image=self.frame.image_rgb,
                               detected_faces=self.bounding_boxes,
                               keras_model=self.keras,
                               verbose=self.verbose).landmarks

    def detect_faces(self, input_is_predetected_face, manual_face):
        """ Detect faces """
        # Predetected_face is used for sort tool.
        # Landmarks should not be extracted again from predetected faces,
        # because face data is lost, resulting in a large variance
        # against extract from original image

        if input_is_predetected_face:
            self.detector.set_predetected(self.frame.width, self.frame.height)
        elif manual_face:
            self.detector.detect_faces(manual_face)
        else:
            self.detector.detect_faces(self.frame.image_detect)

    def get_bounding_boxes(self):
        """ Return the corner points of the bounding box scaled
            to original image """
        bounding_boxes = list()
        for d_rect in self.detector.detected_faces:
            d_rect = self.convert_to_dlib_rectangle(d_rect)
            bounding_box = {
                'left': int(d_rect.left() / self.frame.input_scale),
                'top': int(d_rect.top() / self.frame.input_scale),
                'right': int(d_rect.right() / self.frame.input_scale),
                'bottom': int(d_rect.bottom() / self.frame.input_scale)}
            bounding_boxes.append(bounding_box)
        return bounding_boxes

    def convert_to_dlib_rectangle(self, d_rect):
        """ Convert detected faces to dlib_rectangle """
        if self.detector.is_mmod_rectangle(d_rect):
            return d_rect.rect
        return d_rect
