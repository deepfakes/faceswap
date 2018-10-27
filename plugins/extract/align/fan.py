#!/usr/bin/env python3
""" Facial landmarks extractor for faceswap.py
    Code adapted and modified from:
    https://github.com/1adrianb/face-alignment
"""
import os

import cv2
import numpy as np
import tensorflow as tf

from ._base import Aligner


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
        return model_path

    def initialize(self, *args, **kwargs):
        """ Initialization tasks to run prior to alignments """
        print("Initializing Face Alignment Network...")
        super().initialize(*args, **kwargs)

        card_id, _, vram_total = self.get_vram_free()
        if card_id == -1:
            self.queues["out"].put("No Graphics Card Detected! FAN is not "
                                   "supported on CPU. Use another aligner.")
            self.init.set()
            return

        if vram_total <= self.vram:
            tf_ratio = 1.0
        else:
            tf_ratio = self.vram / vram_total
        if self.verbose:
            print("Reserving {}MB for face alignments".format(self.vram))

        self.model = FAN(self.model_path,
                         verbose=self.verbose, ratio=tf_ratio)

        self.init.set()
        print("Initialized Face Alignment Network.")

    def align(self, *args, **kwargs):
        """ Perform alignments on detected faces """
        super().align(*args, **kwargs)
        try:
            while True:
                item = self.queues["in"].get()
                if item == "EOF":
                    break
                if item.get("exception", False):
                    self.queues["out"].put(item)
                    break
                image = item["image"][:, :, ::-1].copy()
                self.process_landmarks(image, item["detected_faces"])
                self.finalize(item)
            self.finalize("EOF")
        except:
            item["exception"] = True
            self.queues["out"].put(item)
            raise

    def process_landmarks(self, image, detected_faces):
        """ Align image and process landmarks """
        for detected_face in detected_faces:
            process_face = detected_face.to_dlib_rect()
            center, scale = self.get_center_scale(process_face)
            aligned_image = self.align_image(image, center, scale)
            detected_face.landmarksXY = self.predict_landmarks(aligned_image,
                                                               center,
                                                               scale)

    def get_center_scale(self, detected_face):
        """ Get the center and set scale of bounding box """
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

        return center, scale

    def align_image(self, image, center, scale):
        """ Crop and align image around center """
        image = self.crop(
            image,
            center,
            scale).transpose((2, 0, 1)).astype(np.float32) / 255.0

        return np.expand_dims(image, 0)

    def predict_landmarks(self, image, center, scale):
        """ Predict the 68 point landmarks """
        prediction = self.model.predict(image)[-1]
        pts_img = self.get_pts_from_predict(prediction, center, scale)

        return [(int(pt[0]), int(pt[1])) for pt in pts_img]

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


class FAN():
    """The FAN Model.
    Converted from pyTorch via ONNX from:
    https://github.com/1adrianb/face-alignment """

    def __init__(self, model_path, verbose=False, ratio=1.0):
        self.verbose = verbose
        self.model_path = model_path
        self.graph = self.load_graph()
        self.input = self.graph.get_tensor_by_name("fa/0:0")
        self.output = self.graph.get_tensor_by_name("fa/Add_95:0")
        self.session = self.set_session(ratio)

    def load_graph(self):
        """ Load the tensorflow Model and weights """
        if self.verbose:
            print("Initializing Face Alignment Network model...")

        with tf.gfile.GFile(self.model_path, "rb") as gfile:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(gfile.read())
        fa_graph = tf.Graph()
        with fa_graph.as_default():
            tf.import_graph_def(graph_def, name="fa")
        return fa_graph

    def set_session(self, vram_ratio):
        """ Set the TF Session and initialize """
        placeholder = np.zeros((1, 3, 256, 256))
        with self.graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = vram_ratio
            session = tf.Session(config=config)
            with session.as_default():
                session.run(self.output, feed_dict={self.input: placeholder})
        return session

    def predict(self, feed_item):
        """ Predict landmarks in session """
        return self.session.run(self.output,
                                feed_dict={self.input: feed_item})
