#!/usr/bin/env python3
""" S3FD Face detection plugin
https://arxiv.org/abs/1708.05237

Adapted from S3FD Port in FAN:
https://github.com/1adrianb/face-alignment
"""

import os
from scipy.special import logsumexp

import numpy as np

from lib.multithreading import MultiThread
from ._base import Detector, dlib, logger


class Detect(Detector):
    """ S3FD detector for face recognition """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "s3fd"
        self.target = (640, 640)  # Uses approx 4 GB of VRAM
        self.vram = 4096
        self.min_vram = 1024  # Will run at this with warnings
        self.model = None

    def set_model_path(self):
        """ Load the s3fd model """
        model_path = os.path.join(self.cachepath, "s3fd.pb")
        if not os.path.exists(model_path):
            raise Exception("Error: Unable to find {}, reinstall "
                            "the lib!".format(model_path))
        logger.debug("Loading model: '%s'", model_path)
        return model_path

    def initialize(self, *args, **kwargs):
        """ Create the s3fd detector """
        try:
            super().initialize(*args, **kwargs)
            logger.info("Initializing S3FD Detector...")
            card_id, vram_free, vram_total = self.get_vram_free()
            if vram_free <= self.vram:
                tf_ratio = 1.0
            else:
                tf_ratio = self.vram / vram_total

            logger.verbose("Reserving %s%% of total VRAM per s3fd thread",
                           round(tf_ratio * 100, 2))

            confidence = self.config["confidence"] / 100
            self.model = S3fd(self.model_path, self.target, tf_ratio, card_id, confidence)

            if not self.model.is_gpu:
                alloc = 2048
                logger.warning("Using CPU")
            else:
                logger.debug("Using GPU")
                alloc = vram_free
            logger.debug("Allocated for Tensorflow: %sMB", alloc)

            if self.min_vram < alloc < self.vram:
                self.batch_size = 1
                logger.warning("You are running s3fd with %sMB VRAM. The model is optimized for "
                               "%sMB VRAM. Detection should still run but you may get "
                               "warnings/errors", int(alloc), self.vram)
            else:
                self.batch_size = int(alloc / self.vram)
            if self.batch_size < 1:
                raise ValueError("Insufficient VRAM available to continue "
                                 "({}MB)".format(int(alloc)))

            logger.verbose("Processing in %s threads", self.batch_size)

            self.init.set()
            logger.info("Initialized S3FD Detector.")
        except Exception as err:
            self.error.set()
            raise err

    def detect_faces(self, *args, **kwargs):
        """ Detect faces in Multiple Threads """
        super().detect_faces(*args, **kwargs)
        workers = MultiThread(target=self.detect_thread, thread_count=self.batch_size)
        workers.start()
        workers.join()
        sentinel = self.queues["in"].get()
        self.queues["out"].put(sentinel)
        logger.debug("Detecting Faces complete")

    def detect_thread(self):
        """ Detect faces in rgb image """
        logger.debug("Launching Detect")
        while True:
            item = self.get_item()
            if item == "EOF":
                break
            logger.trace("Detecting faces: '%s'", item["filename"])
            detect_image, scale = self.compile_detection_image(item["image"], True, False, False)
            for angle in self.rotation:
                current_image, rotmat = self.rotate_image(detect_image, angle)
                faces = self.model.detect_face(current_image)
                if angle != 0 and faces.any():
                    logger.verbose("found face(s) by rotating image %s degrees", angle)
                if faces.any():
                    break

            detected_faces = self.process_output(faces, rotmat, scale)
            item["detected_faces"] = detected_faces
            self.finalize(item)

        logger.debug("Thread Completed Detect")

    def process_output(self, faces, rotation_matrix, scale):
        """ Compile found faces for output """
        logger.trace("Processing Output: (faces: %s, rotation_matrix: %s)", faces, rotation_matrix)
        faces = [dlib.rectangle(  # pylint: disable=c-extension-no-member
            int(face[0]), int(face[1]), int(face[2]), int(face[3]))
                 for face in faces]
        if isinstance(rotation_matrix, np.ndarray):
            faces = [self.rotate_rect(face, rotation_matrix)
                     for face in faces]
        detected = [dlib.rectangle(  # pylint: disable=c-extension-no-member
            int(face.left() / scale),
            int(face.top() / scale),
            int(face.right() / scale),
            int(face.bottom() / scale))
                    for face in faces]
        logger.trace("Processed Output: %s", detected)
        return detected


class S3fd():
    """ Tensorflow Network """
    def __init__(self, model_path, target_size, vram_ratio, card_id, confidence):
        logger.debug("Initializing: %s: (model_path: '%s', target_size: %s, vram_ratio: %s, "
                     "card_id: %s)",
                     self.__class__.__name__, model_path, target_size, vram_ratio, card_id)
        # Must import tensorflow inside the spawned process for Windows machines
        import tensorflow as tf
        self.is_gpu = False
        self.tf = tf  # pylint: disable=invalid-name
        self.model_path = model_path
        self.confidence = confidence
        self.graph = self.load_graph()
        self.input = self.graph.get_tensor_by_name("s3fd/input_1:0")
        self.output = self.get_outputs()
        self.session = self.set_session(target_size, vram_ratio, card_id)
        logger.debug("Initialized: %s", self.__class__.__name__)

    def load_graph(self):
        """ Load the tensorflow Model and weights """
        # pylint: disable=not-context-manager
        logger.verbose("Initializing S3FD Network model...")
        with self.tf.gfile.GFile(self.model_path, "rb") as gfile:
            graph_def = self.tf.GraphDef()
            graph_def.ParseFromString(gfile.read())
        fa_graph = self.tf.Graph()
        with fa_graph.as_default():
            self.tf.import_graph_def(graph_def, name="s3fd")
        return fa_graph

    def get_outputs(self):
        """ Return the output tensors """
        tensor_names = ["concat_31", "transpose_72", "transpose_75", "transpose_78",
                        "transpose_81", "transpose_84", "transpose_87", "transpose_90",
                        "transpose_93", "transpose_96", "transpose_99", "transpose_102"]
        logger.debug("tensor_names: %s", tensor_names)
        tensors = [self.graph.get_tensor_by_name("s3fd/{}:0".format(t_name))
                   for t_name in tensor_names]
        logger.debug("tensors: %s", tensors)
        return tensors

    def set_session(self, target_size, vram_ratio, card_id):
        """ Set the TF Session and initialize """
        # pylint: disable=not-context-manager, no-member
        placeholder = np.zeros((1, 3, target_size[0], target_size[1]))
        config = self.tf.ConfigProto()
        if card_id != -1:
            config.gpu_options.visible_device_list = str(card_id)
        if vram_ratio != 1.0:
            config.gpu_options.per_process_gpu_memory_fraction = vram_ratio

        with self.graph.as_default():
            session = self.tf.Session(config=config)
            self.is_gpu = any("gpu" in str(device).lower() for device in session.list_devices())
            session.run(self.output, feed_dict={self.input: placeholder})
        return session

    def detect_face(self, feed_item):
        """ Detect faces """
        feed_item = feed_item - np.array([104.0, 117.0, 123.0])
        feed_item = feed_item.transpose(2, 0, 1)
        feed_item = feed_item.reshape((1,) + feed_item.shape).astype('float32')
        bboxlist = self.session.run(self.output, feed_dict={self.input: feed_item})
        bboxlist = self.post_process(bboxlist)

        keep = self.nms(bboxlist, 0.3)
        bboxlist = bboxlist[keep, :]
        bboxlist = [x for x in bboxlist if x[-1] >= self.confidence]

        return np.array(bboxlist)

    def post_process(self, bboxlist):
        """ Perform post processing on output """
        retval = list()
        for i in range(len(bboxlist) // 2):
            bboxlist[i * 2] = self.softmax(bboxlist[i * 2], axis=1)
        for i in range(len(bboxlist) // 2):
            ocls, oreg = bboxlist[i * 2], bboxlist[i * 2 + 1]
            stride = 2 ** (i + 2)    # 4,8,16,32,64,128
            poss = zip(*np.where(ocls[:, 1, :, :] > 0.05))
            for _, hindex, windex in poss:
                axc, ayc = stride / 2 + windex * stride, stride / 2 + hindex * stride
                score = ocls[0, 1, hindex, windex]
                loc = np.ascontiguousarray(oreg[0, :, hindex, windex]).reshape((1, 4))
                priors = np.array([[axc / 1.0, ayc / 1.0, stride * 4 / 1.0, stride * 4 / 1.0]])
                variances = [0.1, 0.2]
                box = self.decode(loc, priors, variances)
                x_1, y_1, x_2, y_2 = box[0] * 1.0
                retval.append([x_1, y_1, x_2, y_2, score])
        retval = np.array(retval)
        if len(retval) == 0:
            retval = np.zeros((1, 5))
        return retval

    @staticmethod
    def softmax(inp, axis):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(inp - logsumexp(inp, axis=axis, keepdims=True))

    @staticmethod
    def decode(loc, priors, variances):
        """Decode locations from predictions using priors to undo
        the encoding we did for offset regression at train time.
        Args:
            loc (tensor): location predictions for loc layers,
                Shape: [num_priors,4]
            priors (tensor): Prior boxes in center-offset form.
                Shape: [num_priors,4].
            variances: (list[float]) Variances of priorboxes
        Return:
            decoded bounding box predictions
        """
        boxes = np.concatenate((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
                                priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])),
                               1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    @staticmethod
    def nms(dets, thresh):
        """ Perform Non-Maximum Suppression """
        keep = list()
        if len(dets) == 0:
            return keep

        x_1, y_1, x_2, y_2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
        areas = (x_2 - x_1 + 1) * (y_2 - y_1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx_1, yy_1 = np.maximum(x_1[i], x_1[order[1:]]), np.maximum(y_1[i], y_1[order[1:]])
            xx_2, yy_2 = np.minimum(x_2[i], x_2[order[1:]]), np.minimum(y_2[i], y_2[order[1:]])

            width, height = np.maximum(0.0, xx_2 - xx_1 + 1), np.maximum(0.0, yy_2 - yy_1 + 1)
            ovr = width * height / (areas[i] + areas[order[1:]] - width * height)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep
