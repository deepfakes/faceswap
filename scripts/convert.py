#!/usr/bin python3
""" The script to run the convert process of faceswap """

import re
import os
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

from scripts.fsmedia import Alignments, Images, PostProcess, Utils
from lib.faces_detect import DetectedFace
from lib.multithreading import BackgroundGenerator, SpawnProcess
from lib.queue_manager import queue_manager
from lib.utils import get_folder, get_image_paths

from plugins.plugin_loader import PluginLoader


class Convert():
    """ The convert process. """
    def __init__(self, arguments):
        self.args = arguments
        self.output_dir = get_folder(self.args.output_dir)
        self.extract_faces = False
        self.faces_count = 0

        self.images = Images(self.args)
        self.alignments = Alignments(self.args, False)

        # Legacy rotation conversion
        Rotate(self.alignments, self.args.verbose, self.images.input_images)

        self.post_process = PostProcess(arguments)
        self.verify_output = False

        self.opts = OptionalActions(self.args, self.images.input_images)

    def process(self):
        """ Original & LowMem models go with Adjust or Masked converter

            Note: GAN prediction outputs a mask + an image, while other
            predicts only an image. """
        Utils.set_verbosity(self.args.verbose)

        if not self.alignments.have_alignments_file:
            self.load_extractor()

        model = self.load_model()
        converter = self.load_converter(model)

        batch = BackgroundGenerator(self.prepare_images(), 1)

        for item in batch.iterator():
            self.convert(converter, item)

        if self.extract_faces:
            queue_manager.terminate_queues()

        Utils.finalize(self.images.images_found,
                       self.faces_count,
                       self.verify_output)

    def load_extractor(self):
        """ Set on the fly extraction """
        print("\nNo Alignments file found. Extracting on the fly.\n"
              "NB: This will use the inferior dlib-hog for extraction "
              "and dlib pose predictor for landmarks.\nIt is recommended "
              "to perfom Extract first for superior results\n")
        for task in ("load", "detect", "align"):
            queue_manager.add_queue(task, maxsize=0)

        detector = PluginLoader.get_detector("dlib_hog")(
            verbose=self.args.verbose)
        aligner = PluginLoader.get_aligner("dlib")(verbose=self.args.verbose)

        d_kwargs = {"in_queue": queue_manager.get_queue("load"),
                    "out_queue": queue_manager.get_queue("detect")}
        a_kwargs = {"in_queue": queue_manager.get_queue("detect"),
                    "out_queue": queue_manager.get_queue("align")}

        d_process = SpawnProcess()
        d_event = d_process.event
        a_process = SpawnProcess()
        a_event = a_process.event

        d_process.in_process(detector.detect_faces, **d_kwargs)
        a_process.in_process(aligner.align, **a_kwargs)
        d_event.wait(10)
        if not d_event.is_set():
            raise ValueError("Error inititalizing Detector")
        a_event.wait(10)
        if not a_event.is_set():
            raise ValueError("Error inititalizing Aligner")

        self.extract_faces = True

    def load_model(self):
        """ Load the model requested for conversion """
        model_name = self.args.trainer
        model_dir = get_folder(self.args.model_dir)
        num_gpus = self.args.gpus

        model = PluginLoader.get_model(model_name)(model_dir, num_gpus)

        if not model.load(self.args.swap_model):
            print("Model Not Found! A valid model "
                  "must be provided to continue!")
            exit(1)

        return model

    def load_converter(self, model):
        """ Load the requested converter for conversion """
        args = self.args
        conv = args.converter

        converter = PluginLoader.get_converter(conv)(
            model.converter(False),
            trainer=args.trainer,
            blur_size=args.blur_size,
            seamless_clone=args.seamless_clone,
            sharpen_image=args.sharpen_image,
            mask_type=args.mask_type,
            erosion_kernel_size=args.erosion_kernel_size,
            match_histogram=args.match_histogram,
            smooth_mask=args.smooth_mask,
            avg_color_adjust=args.avg_color_adjust,
            draw_transparent=args.draw_transparent)

        return converter

    def prepare_images(self):
        """ Prepare the images for conversion """
        filename = ""
        for filename in tqdm(self.images.input_images,
                             total=self.images.images_found,
                             file=sys.stdout):

            if (self.args.discard_frames and
                    self.opts.check_skipframe(filename) == "discard"):
                continue

            frame = os.path.basename(filename)
            if self.extract_faces:
                convert_item = self.detect_faces(filename)
            else:
                convert_item = self.alignments_faces(filename, frame)

            if not convert_item:
                continue
            image, detected_faces = convert_item

            faces_count = len(detected_faces)
            if faces_count != 0:
                # Post processing requires a dict with "detected_faces" key
                self.post_process.do_actions(
                    {"detected_faces": detected_faces})
                self.faces_count += faces_count

            if faces_count > 1:
                self.verify_output = True
                if self.args.verbose:
                    print("Warning: found more than one face in "
                          "an image! {}".format(frame))

            yield filename, image, detected_faces

    def detect_faces(self, filename):
        """ Extract the face from a frame (If not alignments file found) """
        image = self.images.load_one_image(filename)
        queue_manager.get_queue("load").put((filename, image))
        item = queue_manager.get_queue("align").get()
        detected_faces = item["detected_faces"]
        return image, detected_faces

    def alignments_faces(self, filename, frame):
        """ Get the face from alignments file """
        if not self.check_alignments(frame):
            return None

        faces = self.alignments.get_alignments_for_frame(frame)
        image = self.images.load_one_image(filename)
        detected_faces = list()

        for rawface in faces:
            face = DetectedFace()
            face.from_alignment(rawface, image=image)
            detected_faces.append(face)
        return image, detected_faces

    def check_alignments(self, frame):
        """ If we have no alignments for this image, skip it """
        have_alignments = self.alignments.frame_exists(frame)
        if not have_alignments:
            tqdm.write("No alignment found for {}, "
                       "skipping".format(frame))
        return have_alignments

    def convert(self, converter, item):
        """ Apply the conversion transferring faces onto frames """
        try:
            filename, image, faces = item
            skip = self.opts.check_skipframe(filename)

            if not skip:
                for idx, face in enumerate(faces):
                    image = self.convert_one_face(converter,
                                                  (filename, image, idx, face))
                filename = str(self.output_dir / Path(filename).name)
                cv2.imwrite(filename, image)
        except Exception as err:
            print("Failed to convert image: {}. "
                  "Reason: {}".format(filename, err))
            raise

    def convert_one_face(self, converter, imagevars):
        """ Perform the conversion on the given frame for a single face """
        filename, image, idx, face = imagevars

        if self.opts.check_skipface(filename, idx):
            return image

        # TODO: This switch between 64 and 128 is a hack for now.
        # We should have a separate cli option for size
        size = 128 if (self.args.trainer.strip().lower()
                       in ('gan128', 'originalhighres')) else 64

        image = converter.patch_image(image,
                                      face,
                                      size)
        return image


class OptionalActions():
    """ Process the optional actions for convert """

    def __init__(self, args, input_images):
        self.args = args
        self.input_images = input_images

        self.faces_to_swap = self.get_aligned_directory()

        self.frame_ranges = self.get_frame_ranges()
        self.imageidxre = re.compile(r"[^(mp4)](\d+)(?!.*\d)")

    # SKIP FACES #
    def get_aligned_directory(self):
        """ Check for the existence of an aligned directory for identifying
            which faces in the target frames should be swapped """
        faces_to_swap = None
        input_aligned_dir = self.args.input_aligned_dir

        if input_aligned_dir is None:
            print("Aligned directory not specified. All faces listed in the "
                  "alignments file will be converted")
        elif not os.path.isdir(input_aligned_dir):
            print("Aligned directory not found. All faces listed in the "
                  "alignments file will be converted")
        else:
            faces_to_swap = [Path(path)
                             for path in get_image_paths(input_aligned_dir)]
            if not faces_to_swap:
                print("Aligned directory is empty, "
                      "no faces will be converted!")
            elif len(faces_to_swap) <= len(self.input_images) / 3:
                print("Aligned directory contains an amount of images much "
                      "less than the input, are you sure this is the right "
                      "directory?")
        return faces_to_swap

    # SKIP FRAME RANGES #
    def get_frame_ranges(self):
        """ split out the frame ranges and parse out 'min' and 'max' values """
        if not self.args.frame_ranges:
            return None

        minmax = {"min": 0,  # never any frames less than 0
                  "max": float("inf")}
        rng = [tuple(map(lambda q: minmax[q] if q in minmax.keys() else int(q),
                         v.split("-")))
               for v in self.args.frame_ranges]
        return rng

    def check_skipframe(self, filename):
        """ Check whether frame is to be skipped """
        if not self.frame_ranges:
            return None
        idx = int(self.imageidxre.findall(filename)[0])
        skipframe = not any(map(lambda b: b[0] <= idx <= b[1],
                                self.frame_ranges))
        if skipframe and self.args.discard_frames:
            skipframe = "discard"
        return skipframe

    def check_skipface(self, filename, face_idx):
        """ Check whether face is to be skipped """
        if self.faces_to_swap is None:
            return False
        face_name = "{}_{}{}".format(Path(filename).stem,
                                     face_idx,
                                     Path(filename).suffix)
        face_file = Path(self.args.input_aligned_dir) / Path(face_name)
        skip_face = face_file not in self.faces_to_swap
        if skip_face:
            print("face {} for frame {} was deleted, skipping".format(
                face_idx, os.path.basename(filename)))
        return skip_face


class Rotate():
    """ Rotate landmarks and bounding boxes on legacy alignments
        and remove the 'r' parameter """
    def __init__(self, alignments, verbose, frames):
        self.verbose = verbose
        self.alignments = alignments
        self.frames = {os.path.basename(frame): frame
                       for frame in frames}
        self.process()

    def process(self):
        """ Run the rotate alignments process """
        rotated = self.alignments.get_legacy_frames()
        if not rotated:
            return
        print("Legacy rotated frames found. Converting...")
        self.rotate_landmarks(rotated)
        self.alignments.save()

    def rotate_landmarks(self, rotated):
        """ Rotate the landmarks """
        for rotate_item in tqdm(rotated,
                                desc="Rotating Landmarks"):
            if rotate_item not in self.frames.keys():
                continue
            filename = self.frames[rotate_item]
            dims = cv2.imread(filename).shape[:2]
            self.alignments.rotate_existing_landmarks(rotate_item, dims)
