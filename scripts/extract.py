#!/usr/bin python3
""" The script to run the extract process of faceswap """

import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from lib.detect_blur import is_blurry
from lib.FaceFilter import FaceFilter as FilterFunc
from lib.multithreading import MultiThread, queue_manager
from lib.utils import camel_case_split
from plugins.extract.align._base import Extract as AlignerExtract
from scripts.fsmedia import Alignments, Faces, Images, Utils

tqdm.monitor_interval = 0  # workaround for TqdmSynchronisationWarning


class Extract():
    """ The extract process. """

    def __init__(self, arguments):
        self.args = arguments

        self.images = Images(self.args)
        self.faces = Faces(self.args)
        self.alignments = Alignments(self.args)
        self.output_dir = self.faces.output_dir
        self.post_process = self.set_postprocess_actions()

        self.export_face = True
        self.save_interval = None
        if hasattr(self.args, "save_interval"):
            self.save_interval = self.args.save_interval

    def set_postprocess_actions(self):
        """ Set the post processing actions """
        postprocess_items = dict()
        # Debug Landmarks
        if (hasattr(self.args, 'debug_landmarks')
                and self.args.debug_landmarks):
            postprocess_items["DebugLandmarks"] = None

        # Blurry Face
        if hasattr(self.args, 'blur_thresh') and self.args.blur_thresh:
            kwargs = {"blur_thresh": self.args.blur_thresh}
            postprocess_items["BlurryFaceFilter"] = {"kwargs": kwargs}

        # Face Filter post processing
        if ((hasattr(self.args, "filter") and self.args.filter is not None) or
                (hasattr(self.args, "nfilter") and
                 self.args.nfilter is not None)):
            face_filter = dict()
            filter_lists = dict()
            if hasattr(self.args, "ref_threshold"):
                face_filter["ref_threshold"] = self.args.ref_threshold
            for filter_type in ('filter', 'nfilter'):
                filter_args = getattr(self.args, filter_type, None)
                filter_args = None if not filter_args else filter_args
                filter_lists[filter_type] = filter_args
            face_filter["filter_lists"] = filter_lists
            postprocess_items["FaceFilter"] = {"kwargs": face_filter}

        return PostProcess(self.args.verbose, postprocess_items)

    def process(self):
        """ Perform the extraction process """
        print('Starting, this may take a while...')
        Utils.set_verbosity(self.args.verbose)
        # queue_manager.debug_monitor(1)
        self.add_queues()
        self.threaded_io("load")
        save_thread = self.threaded_io("save")
        self.run_detection(save_thread)
        self.write_alignments()
        images, faces = Utils.finalize(self.images.images_found,
                                       self.faces.num_faces_detected,
                                       self.faces.verify_output)
        self.images.images_found = images
        self.faces.num_faces_detected = faces

    @staticmethod
    def add_queues():
        """ Add the required processing queues to Queue Manager """
        for task in ("load", "detect", "align", "save"):
            size = 100 if task == "load" else 0
            queue_manager.add_queue(task, maxsize=size)

    def threaded_io(self, task):
        """ Load images in a background thread """
        func = self.load_images if task == "load" else self.save_faces
        io_thread = MultiThread(thread_count=1)
        io_thread.in_thread(func)
        return io_thread

    def load_images(self):
        """ Load the images """
        load_queue = queue_manager.get_queue("load")
        for filename in self.images.input_images:
            image = Utils.cv2_read_write('read', filename)
            load_queue.put((filename, image))
        load_queue.put("EOF")

    def save_faces(self):
        """ Save the generated faces """
        if not self.export_face:
            return

        save_queue = queue_manager.get_queue("save")
        while True:
            item = save_queue.get()
            if item == "EOF":
                break
            filename, output_file, resized_face, idx = item
            out_filename = "{}_{}{}".format(str(output_file),
                                            str(idx),
                                            Path(filename).suffix)
            Utils.cv2_read_write('write', out_filename, resized_face)

    def write_alignments(self):
        """ Save the alignments file """
        self.alignments.write_alignments(self.faces.faces_detected)

    def run_detection(self, save_thread):
        """ Run Face Detection """
        frame_no = 0
        try:
            for faces in tqdm(self.faces.detect_faces(),
                              total=self.images.images_found,
                              file=sys.stdout,
                              desc="Extracting faces"):

                filename = faces["filename"]
                faces["output_file"] = self.output_dir / Path(filename).stem

                self.post_process.do_actions(faces)
                self.process_faces(filename, faces)

                frame_no += 1
                if frame_no == self.save_interval:
                    self.write_alignments()
                    frame_no = 0

        except Exception as err:
            if self.args.verbose:
                msg = "Failed to extract from image"
                if "filename" in locals():
                    msg += ": {}".format(filename)
                msg += ".Reason: {}".format(err)
                print(msg)
            raise

        if self.export_face:
            queue_manager.get_queue("save").put("EOF")
        save_thread.join_threads()

    def process_faces(self, filename, faces):
        """ Perform processing on found faces """
        final_faces = list()
        save_queue = queue_manager.get_queue("save")

        filename = faces["filename"]
        output_file = faces["output_file"]
        resized_faces = faces["resized_faces"]

        for idx, face in enumerate(faces["detected_faces"]):
            if self.export_face:
                save_queue.put((filename,
                                output_file,
                                resized_faces[idx],
                                idx))

            final_faces.append({"x": face.x,
                                "w": face.w,
                                "y": face.y,
                                "h": face.h,
                                "landmarksXY": face.landmarks_as_xy()})
        self.faces.faces_detected[os.path.basename(filename)] = final_faces


class PostProcess():
    """ Optional post processing tasks """
    def __init__(self, verbose=False, postprocess_items=None):
        self.verbose = verbose
        self.actions = self.set_actions(postprocess_items)

    def set_actions(self, postprocess_items):
        """ Compile the actions to be performed into a list """
        actions = list()
        for action, options in postprocess_items.items():
            options = dict() if options is None else options
            args = options.get("args", tuple())
            kwargs = options.get("kwargs", dict())
            args = args if isinstance(args, tuple) else tuple()
            kwargs = kwargs if isinstance(kwargs, dict) else dict()
            kwargs["verbose"] = self.verbose
            task = globals()[action](*args, **kwargs)
            actions.append(task)

        for action in actions:
            action_name = camel_case_split(action.__class__.__name__)
            print("Adding post processing item: "
                  "{}".format(" ".join(action_name)))

        return actions

    def do_actions(self, output_item):
        """ Perform the requested post-processing actions """
        for action in self.actions:
            action.process(output_item)


class PostProcessAction():
    """ Parent class for Post Processing Actions """
    def __init__(self, *args, **kwargs):
        self.verbose = kwargs["verbose"]

    def process(self, output_item):
        """ Override for specific post processing action """
        raise NotImplementedError


class BlurryFaceFilter(PostProcessAction):
    """ Move blurry faces to a different folder """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blur_thresh = kwargs["blur_thresh"]

    def process(self, output_item):
        """ Detect and move blurry face """
        blurry_file = None
        filename = output_item["filename"]
        output_file = output_item["output_file"]
        extractor = AlignerExtract()

        for idx, face in enumerate(output_item["detected_faces"]):
            resized_face = output_item["resized_faces"][idx]
            dims = resized_face.shape[:2]
            size = dims[0]
            t_mat = output_item["t_mats"][idx]

            aligned_landmarks = extractor.transform_points(
                face.landmarksXY(),
                t_mat, size, 48)
            feature_mask = extractor.get_feature_mask(
                aligned_landmarks / size,
                size, 48)
            feature_mask = cv2.blur(feature_mask, (10, 10))
            isolated_face = cv2.multiply(
                feature_mask,
                resized_face.astype(float)).astype(np.uint8)
            blurry, focus_measure = is_blurry(isolated_face, self.blur_thresh)

            if blurry:
                print("{}'s focus measure of {} was below the blur threshold, "
                      "moving to \"blurry\"".format(Path(filename).stem,
                                                    focus_measure))
                blurry_file = os.path.join(os.path.dirname(output_file),
                                           "blurry") / Path(filename).stem
                output_item["output_file"] = blurry_file


class DebugLandmarks(PostProcessAction):
    """ Draw debug landmarks on face """

    def process(self, output_item):
        """ Draw landmarks on image """
        transform_points = AlignerExtract().transform_points
        for idx, face in enumerate(output_item["detected_faces"]):
            dims = output_item["resized_faces"][idx].shape[:2]
            size = dims[0]
            landmarks = transform_points(face.landmarksXY,
                                         output_item["t_mats"][idx],
                                         size,
                                         48)
            for (pos_x, pos_y) in landmarks:
                cv2.circle(output_item["resized_faces"][idx],
                           (pos_x, pos_y), 2, (0, 0, 255), -1)


class FaceFilter(PostProcessAction):
    """ Filter in or out faces based on input image(s) """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        filter_lists = kwargs["filter_lists"]
        ref_threshold = kwargs.get("ref_threshold", 0.6)
        self.filter = self.load_face_filter(filter_lists, ref_threshold)

    def load_face_filter(self, filter_lists, ref_threshold):
        """ Load faces to filter out of images """
        if not any(val for val in filter_lists.values()):
            return None

        filter_files = [self.set_face_filter(key, val)
                        for key, val in filter_lists.items()]

        if any(filters for filters in filter_files):
            facefilter = FilterFunc(filter_files[0],
                                    filter_files[1],
                                    ref_threshold)
        return facefilter

    @staticmethod
    def set_face_filter(f_type, f_args):
        """ Set the required filters """
        if not f_args:
            return list()

        print("{}: {}".format(f_type.title(), f_args))
        filter_files = f_args if isinstance(f_args, list) else [f_args]
        filter_files = list(filter(lambda fnc: Path(fnc).exists(),
                                   filter_files))
        return filter_files

    def process(self, output_item):
        """ Filter in/out wanted/unwanted faces """
        if not self.filter:
            return

        detected_faces = output_item["detected_faces"]
        ret_faces = list()
        for face in detected_faces:
            if not self.filter.check(face):
                if self.verbose:
                    print("Skipping not recognized face!")
                continue
            ret_faces.append(face)
        output_item["detected_faces"] = ret_faces
