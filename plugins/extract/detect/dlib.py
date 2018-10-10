#!/usr/bin/env python3
""" DLIB Face detection plugin """

import os
import numpy as np
from queue import Empty
from .base import Detector, dlib
from lib.gpu_stats import GPUStats


class Detect(Detector):
    """ Dlib detector for face recognition """
    def __init__(self):
        super().__init__()
        self.name = "dlib"
        self.detectors = list()
        self.target = (2048, 2048)  # Uses approx 1805MB of VRAM

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

    def set_data_path(self):
        """ Load the face detector data """
        data_path = os.path.join(self.cachepath,
                                 "mmod_human_face_detector.dat")
        if not os.path.exists(data_path):
            raise Exception("Error: Unable to find {}, reinstall "
                            "the lib!".format(data_path))
        return data_path

    def create_detector(self, verbose, detector, buffer):
        """ Add the requested detectors  """
        self.verbose = verbose

        is_cuda = self.compiled_for_cuda()
        if is_cuda:        
            stats = GPUStats()
            free_vram = stats.get_free()
            # Get card with most available vram
            start_vram = max(free_vram)
            card = free_vram.index(max(free_vram))
        if is_cuda and self.verbose:
            print("Using device {} with {}MB free of {}MB".format(
                stats.devices[card],
                int(start_vram),
                int(stats.vram[card])))
        elif not is_cuda and self.verbose:
            print("Using CPU. Limiting RAM useage to 2048MB")

        placeholder = np.zeros((self.target[0],
                                self.target[1], 3), dtype=np.uint8)
        
        self.add_detectors(detector=detector)
        for current_detector in self.detectors:
            current_detector(placeholder, 0)

        if is_cuda and detector != "dlib-hog":
            end_vram = stats.get_free()[card]
            self.vram = int(start_vram - end_vram)
            self.vram = 256 if self.vram == 0 else self.vram
#            NB CNN with GPU is slower using multiprocessing, so
#            limit to a single process. Most of this code is
#            obsolete given this fact, but left here in case anything
#            changes
#            self.processes = int((start_vram - buffer)  / self.vram)
            self.processes = 1
        elif detector == "dlib-hog":
            self.vram = 2048
            self.processes = 99 # Overridden by available cpu Cores
        else:
            self.vram = 2048
            self.processes = 1
        
        self.initialized = True

    def add_detectors(self, detector=None):
        """ Add detectors """
        if detector == "dlib-cnn" or detector == "dlib-all":
            if self.verbose:
                print("Adding DLib - CNN detector")
            self.detectors.append(dlib.cnn_face_detection_model_v1(
                self.data_path))

        if detector == "dlib-hog" or detector == "dlib-all":
            if self.verbose:
                print("Adding DLib - HOG detector")
            self.detectors.append(dlib.get_frontal_face_detector())

    @staticmethod
    def batch_input(input_queue, batch_size):
        is_final = False
        batch = [input_queue.get()]  # block until at least 1
        try:
            while len(batch) < batch_size:
                item = input_queue.get()
                if not item[0]:
                    is_final = True
                    break
                batch.append(item)
        except Empty:
            pass
        return is_final, batch
    
    
    def detect_faces(self, image_queue):
        """ Detect faces in rgb image """
        batch_size = 20
        i = 0
        stats = GPUStats()
        print("start", stats.get_free())
        while True:
            is_final, batch = self.batch_input(image_queue, batch_size)
            filenames, images = map(list, zip(*batch))
            for idx, image in enumerate(images):
                self.set_scale(image, is_square=True, scale_up=True)
                images[idx] = self.set_detect_image(image)
#            print(is_final)
#            print(filenames)
#            print(images)

            detected_faces = None
            for current_detector in self.detectors:
                detected_faces = current_detector(images, 0)
                print(len(detected_faces))
                print("batch", stats.get_free())

#                for rect in detected_faces:
#                    print(rect)
                if detected_faces:
                    break

            for d_rect in detected_faces:
                d_rect = d_rect.rect if self.is_mmod_rectangle(d_rect) else d_rect
                final_faces.append(dlib.rectangle(int(d_rect.left() / self.scale),
                                                int(d_rect.top() / self.scale),
                                                int(d_rect.right() / self.scale),
                                                int(d_rect.bottom() / self.scale)))
            i += len(filenames)
            if is_final:
                break
 #           return final_faces
        print(i)
        exit(0)