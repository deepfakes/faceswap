#!/usr/bin/env python3
""" MTCNN Face detection plugin """

from __future__ import absolute_import, division, print_function

import os

from six import string_types, iteritems

import cv2
import numpy as np

from lib.multithreading import MultiThread
from ._base import Detector, dlib, logger


# Must import tensorflow inside the spawned process
# for Windows machines
tf = None  # pylint: disable = invalid-name


def import_tensorflow():
    """ Import tensorflow from inside spawned process """
    global tf  # pylint: disable = invalid-name,global-statement
    import tensorflow as tflow
    tf = tflow


class Detect(Detector):
    """ MTCNN detector for face recognition """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = self.validate_kwargs()
        self.name = "mtcnn"
        self.target = 2073600  # Uses approx 1.30 GB of VRAM
        self.vram = 1408

    def validate_kwargs(self):
        """ Validate that config options are correct. If not reset to default """
        valid = True
        threshold = [self.config["threshold_1"],
                     self.config["threshold_2"],
                     self.config["threshold_3"]]
        kwargs = {"minsize": self.config["minsize"],
                  "threshold": threshold,
                  "factor": self.config["scalefactor"]}

        if kwargs["minsize"] < 10:
            valid = False
        elif not all(0.0 < threshold <= 1.0 for threshold in kwargs['threshold']):
            valid = False
        elif not 0.0 < kwargs['factor'] < 1.0:
            valid = False

        if not valid:
            kwargs = {"minsize": 20,                 # minimum size of face
                      "threshold": [0.6, 0.7, 0.7],  # three steps threshold
                      "factor": 0.709}               # scale factor
            logger.warning("Invalid MTCNN options in config. Running with defaults")
        logger.debug("Using mtcnn kwargs: %s", kwargs)
        return kwargs

    def set_model_path(self):
        """ Load the mtcnn models """
        for model in ("det1.npy", "det2.npy", "det3.npy"):
            model_path = os.path.join(self.cachepath, model)
            if not os.path.exists(model_path):
                raise Exception("Error: Unable to find {}, reinstall "
                                "the lib!".format(model_path))
            logger.debug("Loading model: '%s'", model_path)
        return self.cachepath

    def initialize(self, *args, **kwargs):
        """ Create the mtcnn detector """
        try:
            super().initialize(*args, **kwargs)
            logger.info("Initializing MTCNN Detector...")
            is_gpu = False

            # Must import tensorflow inside the spawned process
            # for Windows machines
            import_tensorflow()
            _, vram_free, _ = self.get_vram_free()
            mtcnn_graph = tf.Graph()

            # Windows machines sometimes misreport available vram, and overuse
            # causing OOM. Allow growth fixes that
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True  # pylint: disable=no-member

            with mtcnn_graph.as_default():  # pylint: disable=not-context-manager
                sess = tf.Session(config=config)
                with sess.as_default():  # pylint: disable=not-context-manager
                    pnet, rnet, onet = create_mtcnn(sess, self.model_path)

                if any("gpu" in str(device).lower()
                       for device in sess.list_devices()):
                    logger.debug("Using GPU")
                    is_gpu = True
            mtcnn_graph.finalize()

            if not is_gpu:
                alloc = 2048
                logger.warning("Using CPU")
            else:
                alloc = vram_free
            logger.debug("Allocated for Tensorflow: %sMB", alloc)

            self.batch_size = int(alloc / self.vram)

            if self.batch_size < 1:
                self.error.set()
                raise ValueError("Insufficient VRAM available to continue "
                                 "({}MB)".format(int(alloc)))

            logger.verbose("Processing in %s threads", self.batch_size)

            self.kwargs["pnet"] = pnet
            self.kwargs["rnet"] = rnet
            self.kwargs["onet"] = onet

            self.init.set()
            logger.info("Initialized MTCNN Detector.")
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
            [detect_image, scale] = self.compile_detection_image(item["image"], False, False, True)

            for angle in self.rotation:
                current_image, rotmat = self.rotate_image(detect_image, angle)
                faces, points = detect_face(current_image, **self.kwargs)
                if angle != 0 and faces.any():
                    logger.verbose("found face(s) by rotating image %s degrees", angle)
                if faces.any():
                    break

            detected_faces = self.process_output(faces, points, rotmat, scale)
            item["detected_faces"] = detected_faces
            self.finalize(item)

        logger.debug("Thread Completed Detect")

    def process_output(self, faces, points, rotation_matrix, scale):
        """ Compile found faces for output """
        logger.trace("Processing Output: (faces: %s, points: %s, rotation_matrix: %s)",
                     faces, points, rotation_matrix)
        faces = self.recalculate_bounding_box(faces, points)
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

    @staticmethod
    def recalculate_bounding_box(faces, landmarks):
        """ Recalculate the bounding box for Face Alignment.

            Face Alignment was built to expect a DLIB bounding
            box and calculates center and scale based on that.
            Resize the bounding box around features to present
            a better box to Face Alignment. Helps its chances
            on edge cases and helps remove 'jitter' """
        logger.trace("Recalculating Bounding Boxes: (faces: %s, landmarks: %s)",
                     faces, landmarks)
        retval = list()
        no_faces = len(faces)
        if no_faces == 0:
            return retval
        face_landmarks = np.hsplit(landmarks, no_faces)
        for idx in range(no_faces):
            pts = np.reshape(face_landmarks[idx], (5, 2), order="F")
            nose = pts[2]

            minmax = (np.amin(pts, axis=0), np.amax(pts, axis=0))
            padding = [(minmax[1][0] - minmax[0][0]) / 2,
                       (minmax[1][1] - minmax[0][1]) / 2]

            center = (minmax[1][0] - padding[0], minmax[1][1] - padding[1])
            offset = (center[0] - nose[0], nose[1] - center[1])
            center = (center[0] + offset[0], center[1] + offset[1])

            padding[0] += padding[0]
            padding[1] += padding[1]

            bounding = [center[0] - padding[0], center[1] - padding[1],
                        center[0] + padding[0], center[1] + padding[1]]
            retval.append(bounding)
        logger.trace("Recalculated Bounding Boxes: %s", retval)
        return retval


# MTCNN Detector for face alignment
# Code adapted from: https://github.com/davidsandberg/facenet

# Tensorflow implementation of the face detection / alignment algorithm
# found at
# https://github.com/kpzhang93/MTCNN_face_detection_alignment

# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


def layer(operator):
    """Decorator for composable network layers."""

    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(operator.__name__))
        # Figure out the layer inputs.
        if len(self.terminals) == 0:  # pylint: disable=len-as-condition
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        # Perform the operation and get the output.
        layer_output = operator(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated


class Network():
    """ Tensorflow Network """
    def __init__(self, inputs, trainable=True):
        # The input nodes for this network
        self.inputs = inputs
        # The current list of terminal nodes
        self.terminals = []
        # Mapping from layer names to layers
        self.layers = dict(inputs)
        # If true, the resulting variables are set as trainable
        self.trainable = trainable

        self.setup()

    def setup(self):
        """Construct the network. """
        raise NotImplementedError('Must be implemented by the subclass.')

    @staticmethod
    def load(model_path, session, ignore_missing=False):
        """Load network weights.
        model_path: The path to the numpy-serialized network weights
        session: The current TensorFlow session
        ignore_missing: If true, serialized weights for missing layers are
                        ignored.
        """
        # pylint: disable=no-member
        data_dict = np.load(model_path, encoding='latin1').item()

        for op_name in data_dict:
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in iteritems(data_dict[op_name]):
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                    except ValueError:
                        if not ignore_missing:
                            raise

    def feed(self, *args):
        """Set the input(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        """
        assert len(args) != 0  # pylint: disable=len-as-condition
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, string_types):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def get_output(self):
        """Returns the current network output."""
        return self.terminals[-1]

    def get_unique_name(self, prefix):
        """Returns an index-suffixed unique name for the given prefix.
        This is used for auto-generating layer names based on the type-prefix.
        """
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def make_var(self, name, shape):
        """Creates a new TensorFlow variable."""
        return tf.get_variable(name, shape, trainable=self.trainable)

    @staticmethod
    def validate_padding(padding):
        """Verifies that the padding is one of the supported ones."""
        assert padding in ('SAME', 'VALID')

    @layer
    def conv(self,  # pylint: disable=too-many-arguments
             inp,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             name,
             relu=True,
             padding='SAME',
             group=1,
             biased=True):
        """ Conv Layer """
        # pylint: disable=too-many-locals

        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        c_i = int(inp.get_shape()[-1])
        # Verify that the grouping parameter is valid
        assert c_i % group == 0
        assert c_o % group == 0
        # Convolution for a given input and kernel
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding) # noqa
        with tf.variable_scope(name) as scope:
            kernel = self.make_var('weights',
                                   shape=[k_h, k_w, c_i // group, c_o])
            # This is the common-case. Convolve the input without any
            # further complications.
            output = convolve(inp, kernel)
            # Add the biases
            if biased:
                biases = self.make_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)
            if relu:
                # ReLU non-linearity
                output = tf.nn.relu(output, name=scope.name)
            return output

    @layer
    def prelu(self, inp, name):
        """ Prelu Layer """
        with tf.variable_scope(name):
            i = int(inp.get_shape()[-1])
            alpha = self.make_var('alpha', shape=(i,))
            output = tf.nn.relu(inp) + tf.multiply(alpha, -tf.nn.relu(-inp))
        return output

    @layer
    def max_pool(self, inp, k_h, k_w,  # pylint: disable=too-many-arguments
                 s_h, s_w, name, padding='SAME'):
        """ Max Pool Layer """
        self.validate_padding(padding)
        return tf.nn.max_pool(inp,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def fc(self, inp, num_out, name, relu=True):  # pylint: disable=invalid-name
        """ FC Layer """
        with tf.variable_scope(name):
            input_shape = inp.get_shape()
            if input_shape.ndims == 4:
                # The input is spatial. Vectorize it first.
                dim = 1
                for this_dim in input_shape[1:].as_list():
                    dim *= int(this_dim)
                feed_in = tf.reshape(inp, [-1, dim])
            else:
                feed_in, dim = (inp, input_shape[-1].value)
            weights = self.make_var('weights', shape=[dim, num_out])
            biases = self.make_var('biases', [num_out])
            operator = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = operator(feed_in, weights, biases, name=name)  # pylint: disable=invalid-name
            return fc

    @layer
    def softmax(self, target, axis, name=None):  # pylint: disable=no-self-use
        """ Multi dimensional softmax,
            refer to https://github.com/tensorflow/tensorflow/issues/210
            compute softmax along the dimension of target
            the native softmax only supports batch_size x dimension """

        max_axis = tf.reduce_max(target, axis, keepdims=True)
        target_exp = tf.exp(target-max_axis)
        normalize = tf.reduce_sum(target_exp, axis, keepdims=True)
        softmax = tf.div(target_exp, normalize, name)
        return softmax


class PNet(Network):
    """ Tensorflow PNet """
    def setup(self):
        (self.feed('data')  # pylint: disable=no-value-for-parameter, no-member
         .conv(3, 3, 10, 1, 1, padding='VALID', relu=False, name='conv1')
         .prelu(name='PReLU1')
         .max_pool(2, 2, 2, 2, name='pool1')
         .conv(3, 3, 16, 1, 1, padding='VALID', relu=False, name='conv2')
         .prelu(name='PReLU2')
         .conv(3, 3, 32, 1, 1, padding='VALID', relu=False, name='conv3')
         .prelu(name='PReLU3')
         .conv(1, 1, 2, 1, 1, relu=False, name='conv4-1')
         .softmax(3, name='prob1'))

        (self.feed('PReLU3')  # pylint: disable=no-value-for-parameter
         .conv(1, 1, 4, 1, 1, relu=False, name='conv4-2'))


class RNet(Network):
    """ Tensorflow RNet """
    def setup(self):
        (self.feed('data')  # pylint: disable=no-value-for-parameter, no-member
         .conv(3, 3, 28, 1, 1, padding='VALID', relu=False, name='conv1')
         .prelu(name='prelu1')
         .max_pool(3, 3, 2, 2, name='pool1')
         .conv(3, 3, 48, 1, 1, padding='VALID', relu=False, name='conv2')
         .prelu(name='prelu2')
         .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
         .conv(2, 2, 64, 1, 1, padding='VALID', relu=False, name='conv3')
         .prelu(name='prelu3')
         .fc(128, relu=False, name='conv4')
         .prelu(name='prelu4')
         .fc(2, relu=False, name='conv5-1')
         .softmax(1, name='prob1'))

        (self.feed('prelu4')  # pylint: disable=no-value-for-parameter
         .fc(4, relu=False, name='conv5-2'))


class ONet(Network):
    """ Tensorflow ONet """
    def setup(self):
        (self.feed('data')  # pylint: disable=no-value-for-parameter, no-member
         .conv(3, 3, 32, 1, 1, padding='VALID', relu=False, name='conv1')
         .prelu(name='prelu1')
         .max_pool(3, 3, 2, 2, name='pool1')
         .conv(3, 3, 64, 1, 1, padding='VALID', relu=False, name='conv2')
         .prelu(name='prelu2')
         .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
         .conv(3, 3, 64, 1, 1, padding='VALID', relu=False, name='conv3')
         .prelu(name='prelu3')
         .max_pool(2, 2, 2, 2, name='pool3')
         .conv(2, 2, 128, 1, 1, padding='VALID', relu=False, name='conv4')
         .prelu(name='prelu4')
         .fc(256, relu=False, name='conv5')
         .prelu(name='prelu5')
         .fc(2, relu=False, name='conv6-1')
         .softmax(1, name='prob1'))

        (self.feed('prelu5')  # pylint: disable=no-value-for-parameter
         .fc(4, relu=False, name='conv6-2'))

        (self.feed('prelu5')  # pylint: disable=no-value-for-parameter
         .fc(10, relu=False, name='conv6-3'))


def create_mtcnn(sess, model_path):
    """ Create the network """
    if not model_path:
        model_path, _ = os.path.split(os.path.realpath(__file__))

    with tf.variable_scope('pnet'):
        data = tf.placeholder(tf.float32, (None, None, None, 3), 'input')
        pnet = PNet({'data': data})
        pnet.load(os.path.join(model_path, 'det1.npy'), sess)
    with tf.variable_scope('rnet'):
        data = tf.placeholder(tf.float32, (None, 24, 24, 3), 'input')
        rnet = RNet({'data': data})
        rnet.load(os.path.join(model_path, 'det2.npy'), sess)
    with tf.variable_scope('onet'):
        data = tf.placeholder(tf.float32, (None, 48, 48, 3), 'input')
        onet = ONet({'data': data})
        onet.load(os.path.join(model_path, 'det3.npy'), sess)

    pnet_fun = lambda img: sess.run(('pnet/conv4-2/BiasAdd:0', # noqa
                                     'pnet/prob1:0'),
                                    feed_dict={'pnet/input:0': img})
    rnet_fun = lambda img: sess.run(('rnet/conv5-2/conv5-2:0', # noqa
                                     'rnet/prob1:0'),
                                    feed_dict={'rnet/input:0': img})
    onet_fun = lambda img: sess.run(('onet/conv6-2/conv6-2:0', # noqa
                                     'onet/conv6-3/conv6-3:0',
                                     'onet/prob1:0'),
                                    feed_dict={'onet/input:0': img})
    return pnet_fun, rnet_fun, onet_fun


def detect_face(img, minsize, pnet, rnet,  # pylint: disable=too-many-arguments
                onet, threshold, factor):
    """Detects faces in an image, and returns bounding boxes and points for them.
    img: input image
    minsize: minimum faces' size
    pnet, rnet, onet: caffemodel
    threshold: threshold=[th1, th2, th3], th1-3 are three steps's threshold
    factor: the factor used to create a scaling pyramid of face sizes to
            detect in the image.
    """
    # pylint: disable=too-many-locals,too-many-statements,too-many-branches
    factor_count = 0
    total_boxes = np.empty((0, 9))
    points = np.empty(0)
    height = img.shape[0]
    width = img.shape[1]
    minl = np.amin([height, width])
    var_m = 12.0 / minsize
    minl = minl * var_m
    # create scale pyramid
    scales = []
    while minl >= 12:
        scales += [var_m * np.power(factor, factor_count)]
        minl = minl * factor
        factor_count += 1

    # # # # # # # # # # # # #
    # first stage - fast proposal network (pnet) to obtain face candidates
    # # # # # # # # # # # # #
    for scale in scales:
        height_scale = int(np.ceil(height * scale))
        width_scale = int(np.ceil(width * scale))
        im_data = imresample(img, (height_scale, width_scale))
        im_data = (im_data - 127.5) * 0.0078125
        img_x = np.expand_dims(im_data, 0)
        img_y = np.transpose(img_x, (0, 2, 1, 3))
        out = pnet(img_y)
        out0 = np.transpose(out[0], (0, 2, 1, 3))
        out1 = np.transpose(out[1], (0, 2, 1, 3))

        boxes, _ = generate_bounding_box(out1[0, :, :, 1].copy(),
                                         out0[0, :, :, :].copy(),
                                         scale, threshold[0])

        # inter-scale nms
        pick = nms(boxes.copy(), 0.5, 'Union')
        if boxes.size > 0 and pick.size > 0:
            boxes = boxes[pick, :]
            total_boxes = np.append(total_boxes, boxes, axis=0)

    numbox = total_boxes.shape[0]
    if numbox > 0:
        pick = nms(total_boxes.copy(), 0.7, 'Union')
        total_boxes = total_boxes[pick, :]
        regw = total_boxes[:, 2]-total_boxes[:, 0]
        regh = total_boxes[:, 3]-total_boxes[:, 1]
        qq_1 = total_boxes[:, 0]+total_boxes[:, 5] * regw
        qq_2 = total_boxes[:, 1]+total_boxes[:, 6] * regh
        qq_3 = total_boxes[:, 2]+total_boxes[:, 7] * regw
        qq_4 = total_boxes[:, 3]+total_boxes[:, 8] * regh
        total_boxes = np.transpose(np.vstack([qq_1, qq_2, qq_3, qq_4, total_boxes[:, 4]]))
        total_boxes = rerec(total_boxes.copy())
        total_boxes[:, 0:4] = np.fix(total_boxes[:, 0:4]).astype(np.int32)
        d_y, ed_y, d_x, ed_x, var_y, e_y, var_x, e_x, tmpw, tmph = pad(total_boxes.copy(),
                                                                       width, height)

    numbox = total_boxes.shape[0]

    # # # # # # # # # # # # #
    # second stage - refinement of face candidates with rnet
    # # # # # # # # # # # # #

    if numbox > 0:
        tempimg = np.zeros((24, 24, 3, numbox))
        for k in range(0, numbox):
            tmp = np.zeros((int(tmph[k]), int(tmpw[k]), 3))
            tmp[d_y[k] - 1:ed_y[k], d_x[k] - 1:ed_x[k], :] = img[var_y[k] - 1:e_y[k],
                                                                 var_x[k]-1:e_x[k], :]
            if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:
                tempimg[:, :, :, k] = imresample(tmp, (24, 24))
            else:
                return np.empty()
        tempimg = (tempimg-127.5)*0.0078125
        tempimg1 = np.transpose(tempimg, (3, 1, 0, 2))
        out = rnet(tempimg1)
        out0 = np.transpose(out[0])
        out1 = np.transpose(out[1])
        score = out1[1, :]
        ipass = np.where(score > threshold[1])
        total_boxes = np.hstack([total_boxes[ipass[0], 0:4].copy(),
                                 np.expand_dims(score[ipass].copy(), 1)])
        m_v = out0[:, ipass[0]]
        if total_boxes.shape[0] > 0:
            pick = nms(total_boxes, 0.7, 'Union')
            total_boxes = total_boxes[pick, :]
            total_boxes = bbreg(total_boxes.copy(), np.transpose(m_v[:, pick]))
            total_boxes = rerec(total_boxes.copy())

    numbox = total_boxes.shape[0]

    # # # # # # # # # # # # #
    # third stage - further refinement and facial landmarks positions with onet
    # NB: Facial landmarks code commented out for faceswap
    # # # # # # # # # # # # #

    if numbox > 0:
        # third stage
        total_boxes = np.fix(total_boxes).astype(np.int32)
        d_y, ed_y, d_x, ed_x, var_y, e_y, var_x, e_x, tmpw, tmph = pad(total_boxes.copy(),
                                                                       width, height)
        tempimg = np.zeros((48, 48, 3, numbox))
        for k in range(0, numbox):
            tmp = np.zeros((int(tmph[k]), int(tmpw[k]), 3))
            tmp[d_y[k] - 1:ed_y[k], d_x[k] - 1:ed_x[k], :] = img[var_y[k] - 1:e_y[k],
                                                                 var_x[k] - 1:e_x[k], :]
            if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:
                tempimg[:, :, :, k] = imresample(tmp, (48, 48))
            else:
                return np.empty()
        tempimg = (tempimg-127.5)*0.0078125
        tempimg1 = np.transpose(tempimg, (3, 1, 0, 2))
        out = onet(tempimg1)
        out0 = np.transpose(out[0])
        out1 = np.transpose(out[1])
        out2 = np.transpose(out[2])
        score = out2[1, :]
        points = out1
        ipass = np.where(score > threshold[2])
        points = points[:, ipass[0]]
        total_boxes = np.hstack([total_boxes[ipass[0], 0:4].copy(),
                                 np.expand_dims(score[ipass].copy(), 1)])
        m_v = out0[:, ipass[0]]

        width = total_boxes[:, 2] - total_boxes[:, 0] + 1
        height = total_boxes[:, 3] - total_boxes[:, 1] + 1
        points[0:5, :] = (np.tile(width, (5, 1)) * points[0:5, :] +
                          np.tile(total_boxes[:, 0], (5, 1)) - 1)
        points[5:10, :] = (np.tile(height, (5, 1)) * points[5:10, :] +
                           np.tile(total_boxes[:, 1], (5, 1)) - 1)
        if total_boxes.shape[0] > 0:
            total_boxes = bbreg(total_boxes.copy(), np.transpose(m_v))
            pick = nms(total_boxes.copy(), 0.7, 'Min')
            total_boxes = total_boxes[pick, :]
            points = points[:, pick]

    return total_boxes, points


# function [boundingbox] = bbreg(boundingbox,reg)
def bbreg(boundingbox, reg):
    """Calibrate bounding boxes"""
    if reg.shape[1] == 1:
        reg = np.reshape(reg, (reg.shape[2], reg.shape[3]))

    width = boundingbox[:, 2] - boundingbox[:, 0] + 1
    height = boundingbox[:, 3] - boundingbox[:, 1] + 1
    b_1 = boundingbox[:, 0] + reg[:, 0] * width
    b_2 = boundingbox[:, 1] + reg[:, 1] * height
    b_3 = boundingbox[:, 2] + reg[:, 2] * width
    b_4 = boundingbox[:, 3] + reg[:, 3] * height
    boundingbox[:, 0:4] = np.transpose(np.vstack([b_1, b_2, b_3, b_4]))
    return boundingbox


def generate_bounding_box(imap, reg, scale, threshold):
    """Use heatmap to generate bounding boxes"""
    # pylint: disable=too-many-locals
    stride = 2
    cellsize = 12

    imap = np.transpose(imap)
    d_x1 = np.transpose(reg[:, :, 0])
    d_y1 = np.transpose(reg[:, :, 1])
    d_x2 = np.transpose(reg[:, :, 2])
    d_y2 = np.transpose(reg[:, :, 3])
    dim_y, dim_x = np.where(imap >= threshold)
    if dim_y.shape[0] == 1:
        d_x1 = np.flipud(d_x1)
        d_y1 = np.flipud(d_y1)
        d_x2 = np.flipud(d_x2)
        d_y2 = np.flipud(d_y2)
    score = imap[(dim_y, dim_x)]
    reg = np.transpose(np.vstack([d_x1[(dim_y, dim_x)], d_y1[(dim_y, dim_x)],
                                  d_x2[(dim_y, dim_x)], d_y2[(dim_y, dim_x)]]))
    if reg.size == 0:
        reg = np.empty((0, 3))
    bbox = np.transpose(np.vstack([dim_y, dim_x]))
    q_1 = np.fix((stride * bbox + 1) / scale)
    q_2 = np.fix((stride * bbox + cellsize - 1 + 1) / scale)
    boundingbox = np.hstack([q_1, q_2, np.expand_dims(score, 1), reg])
    return boundingbox, reg


# function pick = nms(boxes,threshold,type)
def nms(boxes, threshold, method):
    """ Non_Max Suppression """
    # pylint: disable=too-many-locals
    if boxes.size == 0:
        return np.empty((0, 3))
    x_1 = boxes[:, 0]
    y_1 = boxes[:, 1]
    x_2 = boxes[:, 2]
    y_2 = boxes[:, 3]
    var_s = boxes[:, 4]
    area = (x_2 - x_1 + 1) * (y_2 - y_1 + 1)
    s_sort = np.argsort(var_s)
    pick = np.zeros_like(var_s, dtype=np.int16)
    counter = 0
    while s_sort.size > 0:
        i = s_sort[-1]
        pick[counter] = i
        counter += 1
        idx = s_sort[0:-1]
        xx_1 = np.maximum(x_1[i], x_1[idx])
        yy_1 = np.maximum(y_1[i], y_1[idx])
        xx_2 = np.minimum(x_2[i], x_2[idx])
        yy_2 = np.minimum(y_2[i], y_2[idx])
        width = np.maximum(0.0, xx_2-xx_1+1)
        height = np.maximum(0.0, yy_2-yy_1+1)
        inter = width * height
        if method == 'Min':
            var_o = inter / np.minimum(area[i], area[idx])
        else:
            var_o = inter / (area[i] + area[idx] - inter)
        s_sort = s_sort[np.where(var_o <= threshold)]
    pick = pick[0:counter]
    return pick


# function [d_y ed_y d_x ed_x y e_y x e_x tmp_width tmp_height] = pad(total_boxes,width,height)
def pad(total_boxes, width, height):
    """Compute the padding coordinates (pad the bounding boxes to square)"""
    tmp_width = (total_boxes[:, 2] - total_boxes[:, 0] + 1).astype(np.int32)
    tmp_height = (total_boxes[:, 3] - total_boxes[:, 1] + 1).astype(np.int32)
    numbox = total_boxes.shape[0]

    d_x = np.ones((numbox), dtype=np.int32)
    d_y = np.ones((numbox), dtype=np.int32)
    ed_x = tmp_width.copy().astype(np.int32)
    ed_y = tmp_height.copy().astype(np.int32)

    dim_x = total_boxes[:, 0].copy().astype(np.int32)
    dim_y = total_boxes[:, 1].copy().astype(np.int32)
    e_x = total_boxes[:, 2].copy().astype(np.int32)
    e_y = total_boxes[:, 3].copy().astype(np.int32)

    tmp = np.where(e_x > width)
    ed_x.flat[tmp] = np.expand_dims(-e_x[tmp] + width + tmp_width[tmp], 1)
    e_x[tmp] = width

    tmp = np.where(e_y > height)
    ed_y.flat[tmp] = np.expand_dims(-e_y[tmp] + height + tmp_height[tmp], 1)
    e_y[tmp] = height

    tmp = np.where(dim_x < 1)
    d_x.flat[tmp] = np.expand_dims(2 - dim_x[tmp], 1)
    dim_x[tmp] = 1

    tmp = np.where(dim_y < 1)
    d_y.flat[tmp] = np.expand_dims(2 - dim_y[tmp], 1)
    dim_y[tmp] = 1

    return d_y, ed_y, d_x, ed_x, dim_y, e_y, dim_x, e_x, tmp_width, tmp_height


# function [bbox_a] = rerec(bbox_a)
def rerec(bbox_a):
    """Convert bbox_a to square."""
    height = bbox_a[:, 3]-bbox_a[:, 1]
    width = bbox_a[:, 2]-bbox_a[:, 0]
    length = np.maximum(width, height)
    bbox_a[:, 0] = bbox_a[:, 0] + width * 0.5 - length * 0.5
    bbox_a[:, 1] = bbox_a[:, 1] + height * 0.5 - length * 0.5
    bbox_a[:, 2:4] = bbox_a[:, 0:2] + np.transpose(np.tile(length, (2, 1)))
    return bbox_a


def imresample(img, size):
    """ Resample image """
    # pylint: disable=no-member
    im_data = cv2.resize(img, (size[1], size[0]),
                         interpolation=cv2.INTER_AREA)  # @UndefinedVariable
    return im_data
