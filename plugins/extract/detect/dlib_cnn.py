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
        self.target = (2048, 2048)  # Uses approx 1805MB of VRAM
        self.vram = 1600  # Lower as batch size of 2 gives wiggle room
        self.detector = None

    def compiled_for_cuda(self):
        """ Return a message on DLIB Cuda Compilation status """
        cuda = dlib.DLIB_USE_CUDA
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
        self.detector = dlib.cnn_face_detection_model_v1(self.model_path)
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
                filenames, images = map(list, zip(*batch))
                detect_images = self.compile_detection_images(images)
                batch_detected = self.detector(detect_images, 0)
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
                    self.queues["out"].put("EOF")
                    break
        except:
            retval = {"exception": True}
            self.queues["out"].put(retval)
            # Free up VRAM
            del self.detector
            raise

        # Free up VRAM
        del self.detector

    def compile_detection_images(self, images):
        """ Compile the detection images into batches """
        detect_images = list()
        for image in images:
            self.set_scale(image, is_square=True, scale_up=True)
            detect_images.append(self.set_detect_image(image))
        return detect_images

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
                face = dlib.rectangle(int(face.left() / self.scale),
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

            batch_detected = self.detector(reprocess, 0)
            if self.verbose and any(item.any() for item in batch_detected):
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
