#!/usr/bin/env python3
""" DLIB CNN Face detection plugin """

import numpy as np
import face_recognition_models
from lib.utils import rotate_image_by_angle

from ._base import Detector, dlib


class Detect(Detector):
    """ Dlib detector for face recognition """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.target = (1792, 1792)  # Uses approx 1805MB of VRAM
        self.vram = 1600  # Lower as batch size of 2 gives wiggle room
        self.detector = None

    def compiled_for_cuda(self):
        """ Return a message on DLIB Cuda Compilation status """
        cuda = dlib.DLIB_USE_CUDA  # pylint: disable=c-extension-no-member
        msg = "DLib is "
        if not cuda:
            msg += "NOT "
        msg += "compiled to use CUDA"
        if self.verbose:
            print(msg)
        return cuda

    def set_model_path(self):
        """ Model path handled by face_recognition_models """
        return face_recognition_models.cnn_face_detector_model_location()

    def initialize(self, *args, **kwargs):
        """ Calculate batch size """
        print("Initializing Dlib-CNN Detector...")
        super().initialize(*args, **kwargs)
        self.detector = dlib.cnn_face_detection_model_v1(  # pylint: disable=c-extension-no-member
            self.model_path)
        is_cuda = self.compiled_for_cuda()
        if is_cuda:
            vram_free = self.get_vram_free()
        else:
            vram_free = 2048
            if self.verbose:
                print("Using CPU. Limiting RAM useage to "
                      "{}MB".format(vram_free))

        # Batch size of 2 actually uses about 338MB less than a single image??
        # From there batches increase at ~680MB per item in the batch

        self.batch_size = int(((vram_free - self.vram) / 680) + 2)

        if self.batch_size < 1:
            raise ValueError("Insufficient VRAM available to continue "
                             "({}MB)".format(int(vram_free)))

        if self.verbose:
            print("Processing in batches of {}".format(self.batch_size))

        self.init.set()
        print("Initialized Dlib-CNN Detector...")

    def detect_faces(self, *args, **kwargs):
        """ Detect faces in rgb image """
        super().detect_faces(*args, **kwargs)
        try:
            while True:
                exhausted, batch = self.get_batch()
                if not batch:
                    break
                filenames, images = map(list, zip(*batch))
                detect_images = self.compile_detection_images(images)
                batch_detected = self.detect_batch(detect_images)
                processed = self.process_output(batch_detected,
                                                indexes=None,
                                                rotation_matrix=None,
                                                output=None)
                if not all(faces
                           for faces in processed) and self.rotation != [0]:
                    processed = self.process_rotations(detect_images,
                                                       processed)
                for idx, faces in enumerate(processed):
                    retval = {"filename": filenames[idx],
                              "image": images[idx],
                              "detected_faces": faces}
                    self.finalize(retval)
                if exhausted:
                    break
        except:
            retval = {"exception": True}
            self.queues["out"].put(retval)
            del self.detector  # Free up VRAM
            raise

        self.queues["out"].put("EOF")
        del self.detector  # Free up VRAM

    def compile_detection_images(self, images):
        """ Compile the detection images into batches """
        detect_images = list()
        for image in images:
            self.set_scale(image, is_square=True, scale_up=True)
            detect_images.append(self.set_detect_image(image))
        return detect_images

    def detect_batch(self, detect_images, disable_message=False):
        """ Pass the batch through detector for consistently sized images
            or each image seperately for inconsitently sized images """
        can_batch = self.check_batch_dims(detect_images)
        if can_batch:
            batch_detected = self.detector(detect_images, 0)
        else:
            if self.verbose and not disable_message:
                print("Batch has inconsistently sized images. Processing one "
                      "image at a time")
            batch_detected = dlib.mmod_rectangless(  # pylint: disable=c-extension-no-member
                [self.detector(detect_image, 0) for detect_image in detect_images])
        return batch_detected

    @staticmethod
    def check_batch_dims(images):
        """ Check all images are the same size for batching """
        dims = set(frame.shape[:2] for frame in images)
        return len(dims) == 1

    def process_output(self, batch_detected,
                       indexes=None, rotation_matrix=None, output=None):
        """ Process the output images """
        output = output if output else list()
        for idx, faces in enumerate(batch_detected):
            detected_faces = list()

            if isinstance(rotation_matrix, np.ndarray):
                faces = [self.rotate_rect(face.rect, rotation_matrix)
                         for face in faces]

            for face in faces:
                face = self.convert_to_dlib_rectangle(face)
                face = dlib.rectangle(  # pylint: disable=c-extension-no-member
                    int(face.left() / self.scale),
                    int(face.top() / self.scale),
                    int(face.right() / self.scale),
                    int(face.bottom() / self.scale))
                detected_faces.append(face)
            if indexes:
                target = indexes[idx]
                output[target] = detected_faces
            else:
                output.append(detected_faces)
        return output

    def process_rotations(self, detect_images, processed):
        """ Rotate frames missing faces until face is found """
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
            if self.verbose and any(item for item in batch_detected):
                print("found face(s) by rotating image {} degrees".format(
                    angle))
            processed = self.process_output(batch_detected,
                                            indexes=indexes,
                                            rotation_matrix=rotmat,
                                            output=processed)
        return processed

    @staticmethod
    def compile_reprocess(processed, detect_images, angle):
        """ Rotate images which did not find a face for reprocessing """
        indexes = list()
        to_detect = list()
        for idx, faces in enumerate(processed):
            if faces:
                continue
            image = detect_images[idx]
            rot_image, rot_matrix = rotate_image_by_angle(image, angle)
            to_detect.append(rot_image)
            indexes.append(idx)
        return to_detect, indexes, rot_matrix
