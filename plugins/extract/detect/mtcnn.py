#!/usr/bin/env python3
""" MTCNN Face detection plugin """

from __future__ import absolute_import, division, print_function

import cv2
from keras.layers import Conv2D, Dense, Flatten, Input, MaxPool2D, Permute, PReLU

import numpy as np

from lib.model.session import KSession
from ._base import Detector, logger


class Detect(Detector):
    """ MTCNN detector for face recognition """
    def __init__(self, **kwargs):
        git_model_id = 2
        model_filename = ["mtcnn_det_v2.1.h5", "mtcnn_det_v2.2.h5", "mtcnn_det_v2.3.h5"]
        super().__init__(git_model_id=git_model_id, model_filename=model_filename, **kwargs)
        self.name = "MTCNN"
        self.input_size = 640
        self.vram = 320
        self.vram_warnings = 64  # Will run at this with warnings
        self.vram_per_batch = 32
        self.batchsize = self.config["batch-size"]
        self.kwargs = self.validate_kwargs()
        self.color_format = "RGB"

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
            kwargs = {"minsize": 20,  # minimum size of face
                      "threshold": [0.6, 0.7, 0.7],  # three steps threshold
                      "factor": 0.709}               # scale factor
            logger.warning("Invalid MTCNN options in config. Running with defaults")
        logger.debug("Using mtcnn kwargs: %s", kwargs)
        return kwargs

    def init_model(self):
        """ Initialize S3FD Model"""
        self.model = MTCNN(self.model_path, self.config["allow_growth"], **self.kwargs)

    def process_input(self, batch):
        """ Compile the detection image(s) for prediction """
        batch["feed"] = (batch["image"] - 127.5) / 127.5
        return batch

    def predict(self, batch):
        """ Run model to get predictions """
        prediction, points = self.model.detect_faces(batch["feed"])
        logger.trace("filename: %s, prediction: %s, mtcnn_points: %s",
                     batch["filename"], prediction, points)
        batch["prediction"], batch["mtcnn_points"] = prediction, points
        return batch

    def process_output(self, batch):
        """ Post process the detected faces """
        return batch


# MTCNN Detector
# Code adapted from: https://github.com/xiangrufan/keras-mtcnn
#
# Keras implementation of the face detection / alignment algorithm
# found at
# https://github.com/kpzhang93/MTCNN_face_detection_alignment
#
# MIT License
#
# Copyright (c) 2016 Kaipeng Zhang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


class PNet(KSession):
    """ Keras PNet model for MTCNN """
    def __init__(self, model_path, allow_growth):
        super().__init__("MTCNN-PNet", model_path, allow_growth=allow_growth)
        self.define_model(self.model_definition)
        self.load_model_weights()

    @staticmethod
    def model_definition():
        """ Keras PNetwork for MTCNN """
        input_ = Input(shape=(None, None, 3))
        var_x = Conv2D(10, (3, 3), strides=1, padding='valid', name='conv1')(input_)
        var_x = PReLU(shared_axes=[1, 2], name='PReLU1')(var_x)
        var_x = MaxPool2D(pool_size=2)(var_x)
        var_x = Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(var_x)
        var_x = PReLU(shared_axes=[1, 2], name='PReLU2')(var_x)
        var_x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(var_x)
        var_x = PReLU(shared_axes=[1, 2], name='PReLU3')(var_x)
        classifier = Conv2D(2, (1, 1), activation='softmax', name='conv4-1')(var_x)
        bbox_regress = Conv2D(4, (1, 1), name='conv4-2')(var_x)
        return [input_], [classifier, bbox_regress]


class RNet(KSession):
    """ Keras RNet model for MTCNN """
    def __init__(self, model_path, allow_growth):
        super().__init__("MTCNN-RNet", model_path, allow_growth=allow_growth)
        self.define_model(self.model_definition)
        self.load_model_weights()

    @staticmethod
    def model_definition():
        """ Keras RNetwork for MTCNN """
        input_ = Input(shape=(24, 24, 3))
        var_x = Conv2D(28, (3, 3), strides=1, padding='valid', name='conv1')(input_)
        var_x = PReLU(shared_axes=[1, 2], name='prelu1')(var_x)
        var_x = MaxPool2D(pool_size=3, strides=2, padding='same')(var_x)

        var_x = Conv2D(48, (3, 3), strides=1, padding='valid', name='conv2')(var_x)
        var_x = PReLU(shared_axes=[1, 2], name='prelu2')(var_x)
        var_x = MaxPool2D(pool_size=3, strides=2)(var_x)

        var_x = Conv2D(64, (2, 2), strides=1, padding='valid', name='conv3')(var_x)
        var_x = PReLU(shared_axes=[1, 2], name='prelu3')(var_x)
        var_x = Permute((3, 2, 1))(var_x)
        var_x = Flatten()(var_x)
        var_x = Dense(128, name='conv4')(var_x)
        var_x = PReLU(name='prelu4')(var_x)
        classifier = Dense(2, activation='softmax', name='conv5-1')(var_x)
        bbox_regress = Dense(4, name='conv5-2')(var_x)
        return [input_], [classifier, bbox_regress]


class ONet(KSession):
    """ Keras ONet model for MTCNN """
    def __init__(self, model_path, allow_growth):
        super().__init__("MTCNN-ONet", model_path, allow_growth=allow_growth)
        self.define_model(self.model_definition)
        self.load_model_weights()

    @staticmethod
    def model_definition():
        """ Keras ONetwork for MTCNN """
        input_ = Input(shape=(48, 48, 3))
        var_x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv1')(input_)
        var_x = PReLU(shared_axes=[1, 2], name='prelu1')(var_x)
        var_x = MaxPool2D(pool_size=3, strides=2, padding='same')(var_x)
        var_x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv2')(var_x)
        var_x = PReLU(shared_axes=[1, 2], name='prelu2')(var_x)
        var_x = MaxPool2D(pool_size=3, strides=2)(var_x)
        var_x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv3')(var_x)
        var_x = PReLU(shared_axes=[1, 2], name='prelu3')(var_x)
        var_x = MaxPool2D(pool_size=2)(var_x)
        var_x = Conv2D(128, (2, 2), strides=1, padding='valid', name='conv4')(var_x)
        var_x = PReLU(shared_axes=[1, 2], name='prelu4')(var_x)
        var_x = Permute((3, 2, 1))(var_x)
        var_x = Flatten()(var_x)
        var_x = Dense(256, name='conv5')(var_x)
        var_x = PReLU(name='prelu5')(var_x)

        classifier = Dense(2, activation='softmax', name='conv6-1')(var_x)
        bbox_regress = Dense(4, name='conv6-2')(var_x)
        landmark_regress = Dense(10, name='conv6-3')(var_x)
        return [input_], [classifier, bbox_regress, landmark_regress]


class MTCNN():
    """ MTCNN Detector for face alignment """
    # TODO Batching for rnet and onet

    def __init__(self, model_path, allow_growth, minsize, threshold, factor):
        """
        minsize: minimum faces' size
        threshold: threshold=[th1, th2, th3], th1-3 are three steps threshold
        factor: the factor used to create a scaling pyramid of face sizes to
                detect in the image.
        pnet, rnet, onet: caffemodel
        """
        logger.debug("Initializing: %s: (model_path: '%s', allow_growth: %s, minsize: %s, "
                     "threshold: %s, factor: %s)", self.__class__.__name__, model_path,
                     allow_growth, minsize, threshold, factor)
        self.minsize = minsize
        self.threshold = threshold
        self.factor = factor

        self.pnet = PNet(model_path[0], allow_growth)
        self.rnet = RNet(model_path[1], allow_growth)
        self.onet = ONet(model_path[2], allow_growth)
        self._pnet_scales = None
        logger.debug("Initialized: %s", self.__class__.__name__)

    def detect_faces(self, batch):
        """Detects faces in an image, and returns bounding boxes and points for them.
        batch: input batch
        """
        origin_h, origin_w = batch.shape[1:3]
        rectangles = self.detect_pnet(batch, origin_h, origin_w)
        rectangles = self.detect_rnet(batch, rectangles, origin_h, origin_w)
        rectangles = self.detect_onet(batch, rectangles, origin_h, origin_w)
        ret_boxes = list()
        ret_points = list()
        for rects in rectangles:
            if rects:
                total_boxes = np.array([result[:5] for result in rects])
                points = np.array([result[5:] for result in rects]).T
            else:
                total_boxes = np.empty((0, 9))
                points = np.empty(0)
            ret_boxes.append(total_boxes)
            ret_points.append(points)
        return ret_boxes, ret_points

    def detect_pnet(self, images, height, width):
        # pylint: disable=too-many-locals
        """ first stage - fast proposal network (pnet) to obtain face candidates """
        if self._pnet_scales is None:
            self._pnet_scales = calculate_scales(height, width, self.minsize, self.factor)
        rectangles = [[] for _ in range(images.shape[0])]
        batch_items = images.shape[0]
        for scale in self._pnet_scales:
            rwidth, rheight = int(width * scale), int(height * scale)
            batch = np.empty((batch_items, rheight, rwidth, 3), dtype="float32")
            for idx in range(batch_items):
                batch[idx, ...] = cv2.resize(images[idx, ...], (rwidth, rheight))
            output = self.pnet.predict(batch)
            cls_prob = output[0][..., 1]
            roi = output[1]
            out_h, out_w = cls_prob.shape[1:3]
            out_side = max(out_h, out_w)
            cls_prob = np.swapaxes(cls_prob, 1, 2)
            roi = np.swapaxes(roi, 1, 3)
            for idx in range(batch_items):
                # first index 0 = class score, 1 = one hot repr
                rectangle = detect_face_12net(cls_prob[idx, ...],
                                              roi[idx, ...],
                                              out_side,
                                              1 / scale,
                                              width,
                                              height,
                                              self.threshold[0])
                rectangles[idx].extend(rectangle)
        return [nms(x, 0.7, 'iou') for x in rectangles]

    def detect_rnet(self, images, rectangle_batch, height, width):
        """ second stage - refinement of face candidates with rnet """
        ret = []
        # TODO: batching
        for idx, rectangles in enumerate(rectangle_batch):
            if not rectangles:
                ret.append(list())
                continue
            image = images[idx]
            crop_number = 0
            predict_24_batch = []
            for rect in rectangles:
                crop_img = image[int(rect[1]):int(rect[3]), int(rect[0]):int(rect[2])]
                scale_img = cv2.resize(crop_img, (24, 24))
                predict_24_batch.append(scale_img)
                crop_number += 1
            predict_24_batch = np.array(predict_24_batch)
            output = self.rnet.predict(predict_24_batch, batch_size=128)
            cls_prob = output[0]
            cls_prob = np.array(cls_prob)
            roi_prob = output[1]
            roi_prob = np.array(roi_prob)
            ret.append(filter_face_24net(
                cls_prob, roi_prob, rectangles, width, height, self.threshold[1]
            ))
        return ret

    def detect_onet(self, images, rectangle_batch, height, width):
        """ third stage - further refinement and facial landmarks positions with onet """
        ret = list()
        # TODO: batching
        for idx, rectangles in enumerate(rectangle_batch):
            if not rectangles:
                ret.append(list())
                continue
            image = images[idx]
            crop_number = 0
            predict_batch = []
            for rect in rectangles:
                crop_img = image[int(rect[1]):int(rect[3]), int(rect[0]):int(rect[2])]
                scale_img = cv2.resize(crop_img, (48, 48))
                predict_batch.append(scale_img)
                crop_number += 1
            predict_batch = np.array(predict_batch)
            output = self.onet.predict(predict_batch, batch_size=128)
            cls_prob = output[0]
            roi_prob = output[1]
            pts_prob = output[2]  # index
            ret.append(filter_face_48net(
                cls_prob,
                roi_prob,
                pts_prob,
                rectangles,
                width,
                height,
                self.threshold[2]
            ))
        return ret


def detect_face_12net(cls_prob, roi, out_side, scale, width, height, threshold):
    # pylint: disable=too-many-locals, too-many-arguments
    """ Detect face position and calibrate bounding box on 12net feature map(matrix version)
    Input:
        cls_prob : softmax feature map for face classify
        roi      : feature map for regression
        out_side : feature map's largest size
        scale    : current input image scale in multi-scales
        width    : image's origin width
        height   : image's origin height
        threshold: 0.6 can have 99% recall rate
    """
    in_side = 2*out_side+11
    stride = 0
    if out_side != 1:
        stride = float(in_side-12)/(out_side-1)
    (var_x, var_y) = np.where(cls_prob >= threshold)
    boundingbox = np.array([var_x, var_y]).T
    bb1 = np.fix((stride * (boundingbox) + 0) * scale)
    bb2 = np.fix((stride * (boundingbox) + 11) * scale)
    boundingbox = np.concatenate((bb1, bb2), axis=1)
    dx_1 = roi[0][var_x, var_y]
    dx_2 = roi[1][var_x, var_y]
    dx3 = roi[2][var_x, var_y]
    dx4 = roi[3][var_x, var_y]
    score = np.array([cls_prob[var_x, var_y]]).T
    offset = np.array([dx_1, dx_2, dx3, dx4]).T
    boundingbox = boundingbox + offset*12.0*scale
    rectangles = np.concatenate((boundingbox, score), axis=1)
    rectangles = rect2square(rectangles)
    pick = []
    for rect in rectangles:
        x_1 = int(max(0, rect[0]))
        y_1 = int(max(0, rect[1]))
        x_2 = int(min(width, rect[2]))
        y_2 = int(min(height, rect[3]))
        sc_ = rect[4]
        if x_2 > x_1 and y_2 > y_1:
            pick.append([x_1, y_1, x_2, y_2, sc_])
    return nms(pick, 0.3, "iou")


def filter_face_24net(cls_prob, roi, rectangles, width, height, threshold):
    # pylint: disable=too-many-locals, too-many-arguments
    """ Filter face position and calibrate bounding box on 12net's output
    Input:
        cls_prob  : softmax feature map for face classify
        roi_prob  : feature map for regression
        rectangles: 12net's predict
        width     : image's origin width
        height    : image's origin height
        threshold : 0.6 can have 97% recall rate
    Output:
        rectangles: possible face positions
    """
    prob = cls_prob[:, 1]
    pick = np.where(prob >= threshold)
    rectangles = np.array(rectangles)
    x_1 = rectangles[pick, 0]
    y_1 = rectangles[pick, 1]
    x_2 = rectangles[pick, 2]
    y_2 = rectangles[pick, 3]
    sc_ = np.array([prob[pick]]).T
    dx_1 = roi[pick, 0]
    dx_2 = roi[pick, 1]
    dx3 = roi[pick, 2]
    dx4 = roi[pick, 3]
    r_width = x_2-x_1
    r_height = y_2-y_1
    x_1 = np.array([(x_1 + dx_1 * r_width)[0]]).T
    y_1 = np.array([(y_1 + dx_2 * r_height)[0]]).T
    x_2 = np.array([(x_2 + dx3 * r_width)[0]]).T
    y_2 = np.array([(y_2 + dx4 * r_height)[0]]).T
    rectangles = np.concatenate((x_1, y_1, x_2, y_2, sc_), axis=1)
    rectangles = rect2square(rectangles)
    pick = []
    for rect in rectangles:
        x_1 = int(max(0, rect[0]))
        y_1 = int(max(0, rect[1]))
        x_2 = int(min(width, rect[2]))
        y_2 = int(min(height, rect[3]))
        sc_ = rect[4]
        if x_2 > x_1 and y_2 > y_1:
            pick.append([x_1, y_1, x_2, y_2, sc_])
    return nms(pick, 0.3, 'iou')


def filter_face_48net(cls_prob, roi, pts, rectangles, width, height, threshold):
    # pylint: disable=too-many-locals, too-many-arguments
    """ Filter face position and calibrate bounding box on 12net's output
    Input:
        cls_prob  : cls_prob[1] is face possibility
        roi       : roi offset
        pts       : 5 landmark
        rectangles: 12net's predict, rectangles[i][0:3] is the position, rectangles[i][4] is score
        width     : image's origin width
        height    : image's origin height
        threshold : 0.7 can have 94% recall rate on CelebA-database
    Output:
        rectangles: face positions and landmarks
    """
    prob = cls_prob[:, 1]
    pick = np.where(prob >= threshold)
    rectangles = np.array(rectangles)
    x_1 = rectangles[pick, 0]
    y_1 = rectangles[pick, 1]
    x_2 = rectangles[pick, 2]
    y_2 = rectangles[pick, 3]
    sc_ = np.array([prob[pick]]).T
    dx_1 = roi[pick, 0]
    dx_2 = roi[pick, 1]
    dx3 = roi[pick, 2]
    dx4 = roi[pick, 3]
    r_width = x_2-x_1
    r_height = y_2-y_1
    pts0 = np.array([(r_width * pts[pick, 0] + x_1)[0]]).T
    pts1 = np.array([(r_height * pts[pick, 5] + y_1)[0]]).T
    pts2 = np.array([(r_width * pts[pick, 1] + x_1)[0]]).T
    pts3 = np.array([(r_height * pts[pick, 6] + y_1)[0]]).T
    pts4 = np.array([(r_width * pts[pick, 2] + x_1)[0]]).T
    pts5 = np.array([(r_height * pts[pick, 7] + y_1)[0]]).T
    pts6 = np.array([(r_width * pts[pick, 3] + x_1)[0]]).T
    pts7 = np.array([(r_height * pts[pick, 8] + y_1)[0]]).T
    pts8 = np.array([(r_width * pts[pick, 4] + x_1)[0]]).T
    pts9 = np.array([(r_height * pts[pick, 9] + y_1)[0]]).T
    x_1 = np.array([(x_1 + dx_1 * r_width)[0]]).T
    y_1 = np.array([(y_1 + dx_2 * r_height)[0]]).T
    x_2 = np.array([(x_2 + dx3 * r_width)[0]]).T
    y_2 = np.array([(y_2 + dx4 * r_height)[0]]).T
    rectangles = np.concatenate((x_1, y_1, x_2, y_2, sc_,
                                 pts0, pts1, pts2, pts3, pts4, pts5, pts6, pts7, pts8, pts9),
                                axis=1)
    pick = []
    for rect in rectangles:
        x_1 = int(max(0, rect[0]))
        y_1 = int(max(0, rect[1]))
        x_2 = int(min(width, rect[2]))
        y_2 = int(min(height, rect[3]))
        if x_2 > x_1 and y_2 > y_1:
            pick.append([x_1, y_1, x_2, y_2,
                         rect[4], rect[5], rect[6], rect[7], rect[8], rect[9],
                         rect[10], rect[11], rect[12], rect[13], rect[14]])
    return nms(pick, 0.3, 'iom')


def nms(rectangles, threshold, method):
    # pylint:disable=too-many-locals
    """ apply NMS(non-maximum suppression) on ROIs in same scale(matrix version)
    Input:
        rectangles: rectangles[i][0:3] is the position, rectangles[i][4] is score
    Output:
        rectangles: same as input
    """
    if not rectangles:
        return rectangles
    boxes = np.array(rectangles)
    x_1 = boxes[:, 0]
    y_1 = boxes[:, 1]
    x_2 = boxes[:, 2]
    y_2 = boxes[:, 3]
    var_s = boxes[:, 4]
    area = np.multiply(x_2-x_1+1, y_2-y_1+1)
    s_sort = np.array(var_s.argsort())
    pick = []
    while len(s_sort) > 0:
        # s_sort[-1] have highest prob score, s_sort[0:-1]->others
        xx_1 = np.maximum(x_1[s_sort[-1]], x_1[s_sort[0:-1]])
        yy_1 = np.maximum(y_1[s_sort[-1]], y_1[s_sort[0:-1]])
        xx_2 = np.minimum(x_2[s_sort[-1]], x_2[s_sort[0:-1]])
        yy_2 = np.minimum(y_2[s_sort[-1]], y_2[s_sort[0:-1]])
        width = np.maximum(0.0, xx_2 - xx_1 + 1)
        height = np.maximum(0.0, yy_2 - yy_1 + 1)
        inter = width * height
        if method == 'iom':
            var_o = inter / np.minimum(area[s_sort[-1]], area[s_sort[0:-1]])
        else:
            var_o = inter / (area[s_sort[-1]] + area[s_sort[0:-1]] - inter)
        pick.append(s_sort[-1])
        s_sort = s_sort[np.where(var_o <= threshold)[0]]
    result_rectangle = boxes[pick].tolist()
    return result_rectangle


def calculate_scales(height, width, minsize, factor):
    """ Calculate multi-scale
        Input:
            height: Original image height
            width: Original image width
            minsize: Minimum size for a face to be accepted
            factor: Scaling factor
        Output:
            scales  : Multi-scale
    """
    factor_count = 0
    minl = np.amin([height, width])
    var_m = 12.0 / minsize
    minl = minl * var_m
    # create scale pyramid
    scales = []
    while minl >= 12:
        scales += [var_m * np.power(factor, factor_count)]
        minl = minl * factor
        factor_count += 1
    logger.trace(scales)
    return scales


def rect2square(rectangles):
    """ change rectangles into squares (matrix version)
    Input:
        rectangles: rectangles[i][0:3] is the position, rectangles[i][4] is score
    Output:
        squares: same as input
    """
    width = rectangles[:, 2] - rectangles[:, 0]
    height = rectangles[:, 3] - rectangles[:, 1]
    length = np.maximum(width, height).T
    rectangles[:, 0] = rectangles[:, 0] + width * 0.5 - length * 0.5
    rectangles[:, 1] = rectangles[:, 1] + height * 0.5 - length * 0.5
    rectangles[:, 2:4] = rectangles[:, 0:2] + np.repeat([length], 2, axis=0).T
    return rectangles
