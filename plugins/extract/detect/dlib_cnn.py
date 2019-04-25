#!/usr/bin/env python3
""" DLIB CNN Face detection plugin """

import numpy as np
import face_recognition_models

from ._base import Detector, dlib, logger


class Detect(Detector):
    """ Dlib detector for face recognition """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.target = (1792, 1792)  # Uses approx 1805MB of VRAM
        self.vram = 1600  # Lower as batch size of 2 gives wiggle room
        self.detector = None

    @staticmethod
    def compiled_for_cuda():
        """ Return a message on DLIB Cuda Compilation status """
        cuda = dlib.DLIB_USE_CUDA  # pylint: disable=c-extension-no-member
        msg = "DLib is "
        if not cuda:
            msg += "NOT "
        msg += "compiled to use CUDA"
        logger.verbose(msg)
        return cuda

    def set_model_path(self):
        """ Model path handled by face_recognition_models """
        model_path = face_recognition_models.cnn_face_detector_model_location()
        logger.debug("Loading model: '%s'", model_path)
        return model_path

    def initialize(self, *args, **kwargs):
        """ Calculate batch size """
        try:
            super().initialize(*args, **kwargs)
            logger.verbose("Initializing Dlib-CNN Detector...")
            self.detector = dlib.cnn_face_detection_model_v1(  # pylint: disable=c-extension-no-member
                self.model_path)
            is_cuda = self.compiled_for_cuda()
            if is_cuda:
                logger.debug("Using GPU")
                _, vram_free, _ = self.get_vram_free()
            else:
                logger.verbose("Using CPU")
                vram_free = 2048

            # Batch size of 2 actually uses about 338MB less than a single image??
            # From there batches increase at ~680MB per item in the batch

            self.batch_size = int(((vram_free - self.vram) / 680) + 2)

            if self.batch_size < 1:
                raise ValueError("Insufficient VRAM available to continue "
                                 "({}MB)".format(int(vram_free)))

            logger.verbose("Processing in batches of %s", self.batch_size)

            self.init.set()
            logger.info("Initialized Dlib-CNN Detector...")
        except Exception as err:
            self.error.set()
            raise err

    def detect_faces(self, *args, **kwargs):
        """ Detect faces in rgb image """
        super().detect_faces(*args, **kwargs)
        while True:
            exhausted, batch = self.get_batch()
            if not batch:
                break
            filenames = list()
            images = list()
            for item in batch:
                filenames.append(item["filename"])
                images.append(item["image"])

            [detect_images, scales] = self.compile_detection_images(images)
            batch_detected = self.detect_batch(detect_images)
            processed = self.process_output(batch_detected,
                                            indexes=None,
                                            rotation_matrix=None,
                                            output=None,
                                            scales=scales)
            if not all(faces for faces in processed) and self.rotation != [0]:
                processed = self.process_rotations(detect_images, processed, scales)
            for idx, faces in enumerate(processed):
                filename = filenames[idx]
                for b_idx, item in enumerate(batch):
                    if item["filename"] == filename:
                        output = item
                        del_idx = b_idx
                        break
                output["detected_faces"] = faces
                self.finalize(output)
                del batch[del_idx]
            if exhausted:
                break
        self.queues["out"].put("EOF")
        del self.detector  # Free up VRAM
        logger.debug("Detecting Faces complete")

    def compile_detection_images(self, images):
        """ Compile the detection images into batches """
        logger.trace("Compiling Detection Images: %s", len(images))
        detect_images = list()
        scales = list()
        for image in images:
            detect_image, scale = self.compile_detection_image(image, True, True, True)
            detect_images.append(detect_image)
            scales.append(scale)
        logger.trace("Compiled Detection Images")
        return [detect_images, scales]

    def detect_batch(self, detect_images, disable_message=False):
        """ Pass the batch through detector for consistently sized images
            or each image separately for inconsitently sized images """
        logger.trace("Detecting Batch")
        can_batch = self.check_batch_dims(detect_images)
        if can_batch:
            logger.trace("Valid for batching")
            batch_detected = self.detector(detect_images, 0)
        else:
            if not disable_message:
                logger.verbose("Batch has inconsistently sized images. Processing one "
                               "image at a time")
            batch_detected = dlib.mmod_rectangless(  # pylint: disable=c-extension-no-member
                [self.detector(detect_image, 0) for detect_image in detect_images])
        logger.trace("Detected Batch: %s", [item for item in batch_detected])
        return batch_detected

    @staticmethod
    def check_batch_dims(images):
        """ Check all images are the same size for batching """
        dims = set(frame.shape[:2] for frame in images)
        logger.trace("Batch Dimensions: %s", dims)
        return len(dims) == 1

    def process_output(self, batch_detected,
                       indexes=None, rotation_matrix=None, output=None, scales=None):
        """ Process the output images """
        logger.trace("Processing Output: (batch_detected: %s, indexes: %s, rotation_matrix: %s, "
                     "output: %s, scales: %s",
                     batch_detected, indexes, rotation_matrix, output, scales)
        output = output if output else list()
        for idx, faces in enumerate(batch_detected):
            detected_faces = list()
            scale = scales[idx]

            if isinstance(rotation_matrix, np.ndarray):
                faces = [self.rotate_rect(face.rect, rotation_matrix)
                         for face in faces]

            for face in faces:
                face = self.convert_to_dlib_rectangle(face)
                face = dlib.rectangle(  # pylint: disable=c-extension-no-member
                    int(face.left() / scale),
                    int(face.top() / scale),
                    int(face.right() / scale),
                    int(face.bottom() / scale))
                detected_faces.append(face)
            if indexes:
                target = indexes[idx]
                output[target] = detected_faces
            else:
                output.append(detected_faces)
        logger.trace("Processed Output: %s", output)
        return output

    def process_rotations(self, detect_images, processed, scales):
        """ Rotate frames missing faces until face is found """
        logger.trace("Processing Rotations")
        for angle in self.rotation:
            if all(faces for faces in processed):
                break
            if angle == 0:
                continue
            reprocess, indexes, rotmat = self.compile_reprocess(
                processed,
                detect_images,
                angle)

            batch_detected = self.detect_batch(reprocess, disable_message=True)
            if any(item for item in batch_detected):
                logger.verbose("found face(s) by rotating image %s degrees", angle)
            processed = self.process_output(batch_detected,
                                            indexes=indexes,
                                            rotation_matrix=rotmat,
                                            output=processed,
                                            scales=scales)
        logger.trace("Processed Rotations")
        return processed

    def compile_reprocess(self, processed, detect_images, angle):
        """ Rotate images which did not find a face for reprocessing """
        logger.trace("Compile images for reprocessing")
        indexes = list()
        to_detect = list()
        for idx, faces in enumerate(processed):
            if faces:
                continue
            image = detect_images[idx]
            rot_image, rot_matrix = self.rotate_image_by_angle(image, angle)
            to_detect.append(rot_image)
            indexes.append(idx)
        logger.trace("Compiled images for reprocessing")
        return to_detect, indexes, rot_matrix
