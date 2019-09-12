#!/usr/bin/env python3
""" S3FD Face detection plugin
https://arxiv.org/abs/1708.05237

Adapted from S3FD Port in FAN:
https://github.com/1adrianb/face-alignment
"""

from scipy.special import logsumexp
import numpy as np
from ._base import Detector, logger
import keras
import keras.backend as K
from lib.multithreading import FSThread
from lib.queue_manager import queue_manager
import queue
from os.path import basename


class Detect(Detector):
    """ S3FD detector for face recognition """
    def __init__(self, **kwargs):
        git_model_id = 11
        model_filename = "s3fd_keras_v1.h5"
        super().__init__(
            git_model_id=git_model_id, model_filename=model_filename,
            **kwargs
        )
        self.name = "s3fd_amd"
        self.target = (640, 640)  # Uses approx 4 GB of VRAM
        self.vram = 4096
        self.min_vram = 1024  # Will run at this with warnings
        self.model = None
        self.got_input_eof = False
        self.rotate_queue = None  # set in the detect_faces method
        self.supports_plaidml = True

    def initialize(self, *args, **kwargs):
        """ Create the s3fd detector """
        try:
            super().initialize(*args, **kwargs)
            logger.info("Initializing S3FD-AMD Detector...")
            confidence = self.config["confidence"] / 100
            self.batch_size = self.config["batch-size"]
            self.model = S3fd_amd(self.model_path, self.target, confidence)
            self.init.set()
            logger.info(
                "Initialized S3FD-AMD Detector with batchsize of %i.", self.batch_size
            )
        except Exception as err:
            self.error.set()
            raise err

    def post_processing_thread(self, in_queue, again_queue):
        # If -r is set we move images without found faces and remaining
        # rotations to a queue which is "merged" with the intial input queue.
        # This also means it is possible that we get data after an EOF.
        # This is handled by counting open rotation jobs and propagating
        # a second EOF as soon as we are really done through
        # the preprocsessing thread (detect_faces) and the prediction thread.
        open_rot_jobs = 0
        got_first_eof = False
        while True:
            job = in_queue.get()
            if job == "EOF":
                logger.debug("S3fd-amd post processing got EOF")
                got_first_eof = True
            else:
                predictions, items = job
                bboxes = self.model.finalize_predictions(predictions)
                for bbox, item in zip(bboxes, items):
                    s3fd_opts = item["_s3fd"]
                    detected_faces = self.process_output(bbox, s3fd_opts)
                    did_rotation = s3fd_opts["rotations"].pop(0) != 0
                    if detected_faces:
                        item["detected_faces"] = detected_faces
                        del item["_s3fd"]
                        self.finalize(item)
                        if did_rotation:
                            open_rot_jobs -= 1
                            logger.trace("Found face after rotation.")
                    elif s3fd_opts["rotations"]:  # we have remaining rotations
                        logger.trace("No face detected, remaining rotations: %s", s3fd_opts["rotations"])
                        if not did_rotation:
                            open_rot_jobs += 1
                        logger.trace("Rotate face %s and try again.", item["filename"])
                        again_queue.put(item)
                    else:
                        logger.debug("No face detected for %s.", item["filename"])
                        open_rot_jobs -= 1
                        item["detected_faces"] = []
                        del item["_s3fd"]
                        self.finalize(item)
            if got_first_eof and open_rot_jobs <= 0:
                logger.debug("Sending second EOF")
                again_queue.put("EOF")
                self.finalize("EOF")
                break

    def prediction_thread(self, in_queue, out_queue):
        got_first_eof = False
        while True:
            job = in_queue.get()
            if job == "EOF":
                logger.debug("S3fd-amd prediction processing got EOF")
                if got_first_eof:
                    break
                out_queue.put(job)
                got_first_eof = True
                continue
            batch, items = job
            predictions = self.model.predict(batch)
            out_queue.put((predictions, items))

    def detect_faces(self, *args, **kwargs):
        """ Detect faces in rgb image """
        super().detect_faces(*args, **kwargs)
        self.rotate_queue = queue_manager.get_queue("s3fd_rotate", 8, False)
        prediction_queue = queue_manager.get_queue("s3fd_pred", 8, False)
        post_queue = queue_manager.get_queue("s3fd_post", 8, False)
        worker = FSThread(
            target=self.prediction_thread, args=(prediction_queue, post_queue)
        )
        post_worker = FSThread(
            target=self.post_processing_thread, args=(post_queue, self.rotate_queue)
        )
        worker.start()
        post_worker.start()

        got_first_eof = False
        while True:
            worker.check_and_raise_error()
            post_worker.check_and_raise_error()
            got_eof, in_batch = self.get_batch()
            batch = list()
            for item in in_batch:
                s3fd_opts = item.setdefault("_s3fd", {})
                if "scaled_img" not in s3fd_opts:
                    logger.trace("Resizing %s" % basename(item["filename"]))
                    detect_image, scale, pads = self.compile_detection_image(
                        item["image"], is_square=True, pad_to=self.target
                    )
                    s3fd_opts["scale"] = scale
                    s3fd_opts["pads"] = pads
                    s3fd_opts["rotations"] = list(self.rotation)
                    s3fd_opts["rotmatrix"] = None  # the first "rotation" is always 0
                    img = s3fd_opts["scaled_img"] = detect_image
                else:
                    logger.trace("Rotating %s" % basename(item["filename"]))
                    angle = s3fd_opts["rotations"][0]
                    img, rotmat = self.rotate_image_by_angle(
                        s3fd_opts["scaled_img"], angle, *self.target
                    )
                    s3fd_opts["rotmatrix"] = rotmat
                batch.append((img, item))

            if batch:
                batch_data = np.array([x[0] for x in batch], dtype="float32")
                batch_data = self.model.prepare_batch(batch_data)
                batch_items = [x[1] for x in batch]
                prediction_queue.put((batch_data, batch_items))

            if got_eof:
                logger.debug("S3fd-amd main worker got EOF")
                prediction_queue.put("EOF")
                # Required to prevent hanging when less then BS items are in the
                # again queue and we won't receive new images.
                self.batch_size = 1
                if got_first_eof:
                    break
                got_first_eof = True

        logger.debug("Joining s3fd-amd worker")
        worker.join()
        post_worker.join()
        for qname in ():
            queue_manager.del_queue(qname)
        logger.debug("Detecting Faces complete")

    def process_output(self, faces, opts):
        """ Compile found faces for output """
        logger.trace(
            "Processing Output: (faces: %s, rotation_matrix: %s)",
            faces, opts["rotmatrix"]
        )
        detected = []
        scale = opts["scale"]
        pad_l, pad_t = opts["pads"]
        rot = opts["rotmatrix"]
        for face in faces:
            face = self.to_bounding_box_dict(face[0], face[1], face[2], face[3])
            if isinstance(rot, np.ndarray):
                face = self.rotate_rect(face, rot)
            face = self.to_bounding_box_dict(
                (face["left"] - pad_l) / scale,
                (face["top"] - pad_t) / scale,
                (face["right"] - pad_l) / scale,
                (face["bottom"] - pad_t) / scale
            )
            detected.append(face)
        logger.trace("Processed Output: %s", detected)
        return detected

    def get_item(self):
        """
        Yield one item from the input or rotation
        queue while prioritizing rotation queue to
        prevent deadlocks.
        """
        try:
            item = self.rotate_queue.get(block=self.got_input_eof)
            return item
        except queue.Empty:
            pass
        item = super(Detect, self).get_item()
        if not isinstance(item, dict) and item == "EOF":
            self.got_input_eof = True
        return item


################################################################################
# CUSTOM KERAS LAYERS
# generated by onnx2keras
################################################################################
class O2K_ElementwiseLayer(keras.engine.Layer):
    def __init__(self, **kwargs):
        super(O2K_ElementwiseLayer, self).__init__(**kwargs)

    def call(self, *args):
        raise NotImplementedError()

    def compute_output_shape(self, input_shape):
        # TODO: do this nicer
        ldims = len(input_shape[0])
        rdims = len(input_shape[1])
        if ldims > rdims:
            return input_shape[0]
        if rdims > ldims:
            return input_shape[1]
        lprod = np.prod(list(filter(bool, input_shape[0])))
        rprod = np.prod(list(filter(bool, input_shape[1])))
        return input_shape[0 if lprod > rprod else 1]


class O2K_Add(O2K_ElementwiseLayer):
    def call(self, x, *args):
        return x[0] + x[1]


class O2K_Slice(keras.engine.Layer):
    def __init__(self, starts, ends, axes=None, steps=None, **kwargs):
        self._starts = starts
        self._ends = ends
        self._axes = axes
        self._steps = steps
        super(O2K_Slice, self).__init__(**kwargs)

    def get_config(self):
        config = super(O2K_Slice, self).get_config()
        config.update({
            'starts': self._starts, 'ends': self._ends,
            'axes': self._axes, 'steps': self._steps
        })
        return config

    def get_slices(self, ndims):
        axes = self._axes
        steps = self._steps
        if axes is None:
            axes = tuple(range(ndims))
        if steps is None:
            steps = (1,) * len(axes)
        assert len(axes) == len(steps) == len(self._starts) == len(self._ends)
        return list(zip(axes, self._starts, self._ends, steps))

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        for ax, start, end, steps in self.get_slices(len(input_shape)):
            size = input_shape[ax]
            if ax == 0:
                raise AttributeError("Can not slice batch axis.")
            if size is None:
                if start < 0 or end < 0:
                    raise AttributeError("Negative slices not supported on symbolic axes")
                logger.warning("Slicing symbolic axis might lead to problems.")
                input_shape[ax] = (end - start) // steps
                continue
            if start < 0:
                start = size - start
            if end < 0:
                end = size - end
            input_shape[ax] = (min(size, end) - start) // steps
        return tuple(input_shape)

    def call(self, x, *args):
        ax_map = dict((x[0], slice(*x[1:])) for x in self.get_slices(K.ndim(x)))
        shape = K.int_shape(x)
        slices = [(ax_map[a] if a in ax_map else slice(None)) for a in range(len(shape))]
        x = x[tuple(slices)]
        return x


class O2K_ReduceLayer(keras.engine.Layer):
    def __init__(self, axes=None, keepdims=True, **kwargs):
        self._axes = [axes] if isinstance(axes, int) else axes
        self._keepdims = bool(keepdims)
        super(O2K_ReduceLayer, self).__init__(**kwargs)

    def get_config(self):
        config = super(O2K_ReduceLayer, self).get_config()
        config.update({
            'axes': self._axes,
            'keepdims': self._keepdims
        })
        return config

    def compute_output_shape(self, input_shape):
        if self._axes is None:
            return (1,)*len(input_shape) if self._keepdims else tuple()
        ret = list(input_shape)
        for i in sorted(self._axes, reverse=True):
            if self._keepdims:
                ret[i] = 1
            else:
                ret.pop(i)
        return tuple(ret)

    def call(self, x, *args):
        raise NotImplementedError()


class O2K_Sum(O2K_ReduceLayer):
    def call(self, x, *args):
        return K.sum(x, self._axes, self._keepdims)


class O2K_Sqrt(keras.engine.Layer):
    def call(self, x, *args):
        return K.sqrt(x)


class O2K_Pow(keras.engine.Layer):
    def call(self, x, *args):
        return K.pow(*x)


class O2K_ConstantLayer(keras.engine.Layer):
    def __init__(self, constant_obj, dtype, **kwargs):
        self._dtype = np.dtype(dtype).name
        self._constant = np.array(constant_obj, dtype=self._dtype)
        super(O2K_ConstantLayer, self).__init__(**kwargs)

    def call(self, *args):
        data = K.constant(self._constant, dtype=self._dtype)
        return data

    def compute_output_shape(self, input_shape):
        return self._constant.shape

    def get_config(self):
        config = super(O2K_ConstantLayer, self).get_config()
        config.update({
            'constant_obj': self._constant,
            'dtype': self._dtype
        })
        return config


class O2K_Div(O2K_ElementwiseLayer):
    def call(self, x, *args):
        return x[0] / x[1]


class S3fd_amd():
    """ Keras Network """
    def __init__(self, model_path, target_size, confidence):
        logger.debug("Initializing: %s: (model_path: '%s')",
                     self.__class__.__name__, model_path)
        self.model_path = model_path
        self.confidence = confidence
        self.model = self.load_model()
        logger.debug("Initialized: %s", self.__class__.__name__)

    def load_model(self):
        """ Load the keras Model and weights """
        logger.verbose("Initializing S3FD_amd Network model...")
        layers = {
            'O2K_Add': O2K_Add, 'O2K_Slice': O2K_Slice,
            'O2K_Sum': O2K_Sum, 'O2K_Sqrt': O2K_Sqrt,
            'O2K_Pow': O2K_Pow, 'O2K_ConstantLayer': O2K_ConstantLayer,
            'O2K_Div': O2K_Div
        }
        model = keras.models.load_model(self.model_path, custom_objects=layers)
        model._make_predict_function()  # pylint: disable=protected-access
        return model

    def prepare_batch(self, batch):
        batch = batch - np.array([104.0, 117.0, 123.0])
        batch = batch.transpose(0, 3, 1, 2)
        return batch

    def predict(self, batch):
        bboxlists = self.model.predict(batch)
        return bboxlists

    def finalize_predictions(self, bboxlists):
        """ Detect faces """
        ret = list()
        for i in range(bboxlists[0].shape[0]):
            bboxlist = [x[i:i+1, ...] for x in bboxlists]
            bboxlist = self.post_process(bboxlist)
            keep = self.nms(bboxlist, 0.3)
            bboxlist = bboxlist[keep, :]
            bboxlist = [x for x in bboxlist if x[-1] >= self.confidence]
            ret.append(np.array(bboxlist))
        return ret

    def post_process(self, bboxlist):
        """ Perform post processing on output
            TODO: do this on the batch.
        """
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
