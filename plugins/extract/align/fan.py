#!/usr/bin/env python3
""" Facial landmarks extractor for faceswap.py
    Code adapted and modified from:
    https://github.com/1adrianb/face-alignment
"""
import os
import cv2
import numpy as np

from ._base import Aligner, logger


class Align(Aligner):
    """ Perform transformation to align and get landmarks """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vram = 2240
        self.reference_scale = 195.0
        self.model = None
        self.test = None

    def set_model_path(self):
        """ Load the mtcnn models """
        model_path = os.path.join(self.cachepath, "2DFAN-4.pb")
        if not os.path.exists(model_path):
            raise Exception("Error: Unable to find {}, reinstall "
                            "the lib!".format(model_path))
        logger.debug("Loading model: '%s'", model_path)
        return model_path

    def initialize(self, *args, **kwargs):
        """ Initialization tasks to run prior to alignments """
        try:
            super().initialize(*args, **kwargs)
            logger.info("Initializing Face Alignment Network...")
            logger.debug("fan initialize: (args: %s kwargs: %s)", args, kwargs)

            _, _, vram_total = self.get_vram_free()

            if vram_total <= self.vram:
                tf_ratio = 1.0
            else:
                tf_ratio = self.vram / vram_total
            logger.verbose("Reserving %sMB for face alignments", self.vram)

            self.model = FAN(self.model_path, ratio=tf_ratio)

            self.init.set()
            logger.info("Initialized Face Alignment Network.")
        except Exception as err:
            self.error.set()
            raise err

    def align(self, *args, **kwargs):
        """ Perform alignments on detected faces """
        super().align(*args, **kwargs)
        for item in self.get_item():
            if item == "EOF":
                self.finalize(item)
                break
            image = item["image"][:, :, ::-1].copy()

            logger.trace("Aligning faces")
            try:
                item["landmarks"] = self.process_landmarks(image, item["detected_faces"])
                logger.trace("Aligned faces: %s", item["landmarks"])
            except ValueError as err:
                logger.warning("Image '%s' could not be processed. This may be due to corrupted "
                               "data: %s", item["filename"], str(err))
                item["detected_faces"] = list()
                item["landmarks"] = list()
            self.finalize(item)
        logger.debug("Completed Align")

    def process_landmarks(self, image, detected_faces):
        """ Align image and process landmarks """
        logger.trace("Processing landmarks")
        retval = list()
        for detected_face in detected_faces:
            center, scale = self.get_center_scale(detected_face)
            aligned_image = self.align_image(image, center, scale)
            landmarks = self.predict_landmarks(aligned_image, center, scale)
            retval.append(landmarks)
        logger.trace("Processed landmarks: %s", retval)
        return retval

    def get_center_scale(self, detected_face):
        """ Get the center and set scale of bounding box """
        logger.trace("Calculating center and scale")
        center = np.array([(detected_face.left()
                            + detected_face.right()) / 2.0,
                           (detected_face.top()
                            + detected_face.bottom()) / 2.0])

        center[1] -= (detected_face.bottom()
                      - detected_face.top()) * 0.12

        scale = (detected_face.right()
                 - detected_face.left()
                 + detected_face.bottom()
                 - detected_face.top()) / self.reference_scale

        logger.trace("Calculated center and scale: %s, %s", center, scale)
        return center, scale

    def align_image(self, image, center, scale):
        """ Crop and align image around center """
        logger.trace("Aligning image around center")
        image = self.crop(
            image,
            center,
            scale).transpose((2, 0, 1)).astype(np.float32) / 255.0
        logger.trace("Aligned image around center")
        return np.expand_dims(image, 0)

    def predict_landmarks(self, image, center, scale):
        """ Predict the 68 point landmarks """
        logger.trace("Predicting Landmarks")
        prediction = self.model.predict(image)[-1]
        pts_img = self.get_pts_from_predict(prediction, center, scale)
        retval = [(int(pt[0]), int(pt[1])) for pt in pts_img]
        logger.trace("Predicted Landmarks: %s", retval)
        return retval

    @staticmethod
    def transform(point, center, scale, resolution):
        """ Transform Image """
        logger.trace("Transforming Points")
        pnt = np.array([point[0], point[1], 1.0])
        hscl = 200.0 * scale
        eye = np.eye(3)
        eye[0, 0] = resolution / hscl
        eye[1, 1] = resolution / hscl
        eye[0, 2] = resolution * (-center[0] / hscl + 0.5)
        eye[1, 2] = resolution * (-center[1] / hscl + 0.5)
        eye = np.linalg.inv(eye)
        retval = np.matmul(eye, pnt)[0:2]
        logger.trace("Transformed Points: %s", retval)
        return retval

    def crop(self, image, center, scale, resolution=256.0):  # pylint: disable=too-many-locals
        """ Crop image around the center point """
        logger.trace("Cropping image")
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
            self.test = new_dim
            new_img = np.zeros(new_dim, dtype=np.uint8)
        else:
            new_dim = np.array([v_br[1] - v_ul[1],
                                v_br[0] - v_ul[0]],
                               dtype=np.int)
            self.test = new_dim
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
        # pylint: disable=no-member
        new_img = cv2.resize(new_img,
                             dsize=(int(resolution), int(resolution)),
                             interpolation=cv2.INTER_LINEAR)
        logger.trace("Cropped image")
        return new_img

    def get_pts_from_predict(self, var_a, center, scale):
        """ Get points from predictor """
        logger.trace("Obtain points from prediction")
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
        retval = [self.transform(var_c[i], center, scale, var_a.shape[2])
                  for i in range(var_a.shape[0])]
        logger.trace("Obtained points from prediction: %s", retval)

        return retval


class FAN():
    """The FAN Model.
    Converted from pyTorch via ONNX from:
    https://github.com/1adrianb/face-alignment """

    def __init__(self, model_path, ratio=1.0):
        # Must import tensorflow inside the spawned process
        # for Windows machines
        import tensorflow as tf
        self.tf = tf  # pylint: disable=invalid-name

        self.model_path = model_path
        self.graph = self.load_graph()
        self.input = self.graph.get_tensor_by_name("fa/input_1:0")
        self.output = self.graph.get_tensor_by_name("fa/transpose_647:0")
        self.session = self.set_session(ratio)

    def load_graph(self):
        """ Load the tensorflow Model and weights """
        # pylint: disable=not-context-manager
        logger.verbose("Initializing Face Alignment Network model...")

        with self.tf.gfile.GFile(self.model_path, "rb") as gfile:
            graph_def = self.tf.GraphDef()
            graph_def.ParseFromString(gfile.read())
        fa_graph = self.tf.Graph()
        with fa_graph.as_default():
            self.tf.import_graph_def(graph_def, name="fa")
        return fa_graph

    def set_session(self, vram_ratio):
        """ Set the TF Session and initialize """
        # pylint: disable=not-context-manager, no-member
        placeholder = np.zeros((1, 3, 256, 256))
        with self.graph.as_default():
            config = self.tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = vram_ratio
            session = self.tf.Session(config=config)
            with session.as_default():
                if any("gpu" in str(device).lower() for device in session.list_devices()):
                    logger.debug("Using GPU")
                else:
                    logger.warning("Using CPU")
                session.run(self.output, feed_dict={self.input: placeholder})
        return session

    def predict(self, feed_item):
        """ Predict landmarks in session """
        return self.session.run(self.output,
                                feed_dict={self.input: feed_item})
