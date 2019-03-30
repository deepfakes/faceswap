#!/usr/bin python3
""" The script to run the convert process of faceswap """
# TODO
# Fix dfaker mask for conversion
# blur mask along only along forehead
# vid to vid (sort output order)
# predicted mask


import logging
import re
import os
import sys

import cv2
import numpy as np
from tqdm import tqdm

from scripts.fsmedia import Alignments, Images, PostProcess, Utils
from lib import Serializer
from lib.faces_detect import DetectedFace
from lib.multithreading import MultiThread, PoolProcess
from lib.queue_manager import queue_manager, QueueEmpty
from lib.utils import get_folder, get_image_paths, hash_image_file
from plugins.plugin_loader import PluginLoader

from .extract import Plugins as Extractor

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Convert():
    """ The convert process. """
    def __init__(self, arguments):
        logger.debug("Initializing %s: (args: %s)", self.__class__.__name__, arguments)
        self.args = arguments
        Utils.set_verbosity(self.args.loglevel)

        self.images = Images(self.args)
        self.alignments = Alignments(self.args, False, self.images.is_video)
        # Update Legacy alignments
        Legacy(self.alignments, self.images.input_images, arguments.input_aligned_dir)
        self.opts = OptionalActions(self.args, self.images.input_images, self.alignments)

        self.disk_io = DiskIO(self.alignments, self.images, arguments)
        self.extractor = None
        self.predictor = Predict(self.disk_io.load_queue, arguments)
        self.converter = PluginLoader.get_converter(self.args.converter)(
            get_folder(self.args.output_dir),
            self.predictor.training_size,
            self.predictor.padding,
            self.predictor.crop,
            arguments)

        logger.debug("Initialized %s", self.__class__.__name__)

    def process(self):
        """ Process the conversion """
        logger.debug("Starting Conversion")
        # queue_manager.debug_monitor(2)
        self.convert_images()
        self.disk_io.save_thread.join()
        queue_manager.terminate_queues()

        Utils.finalize(self.images.images_found,
                       self.predictor.faces_count,
                       self.predictor.verify_output)
        logger.debug("Completed Conversion")

    def convert_images(self):
        """ Convert the images """
        logger.debug("Converting images")
        save_queue = queue_manager.get_queue("save")
        patch_queue = queue_manager.get_queue("patch")
        out_queue = queue_manager.get_queue("out")
        pool = PoolProcess(self.converter.process, patch_queue, out_queue)
        pool.start()
        for item in tqdm(self.patch_iterator(pool.procs),
                         desc="Converting",
                         total=self.images.images_found,
                         file=sys.stdout):
            save_queue.put(item)
        pool.join()

        save_queue.put("EOF")
        logger.debug("Converted images")

    def check_thread_error(self):
        """ Check and raise thread errors """
        for thread in (self.predictor.thread, self.disk_io.load_thread, self.disk_io.save_thread):
            thread.check_and_raise_error()

    def patch_iterator(self, processes):
        """ Prepare the images for conversion """
        out_queue = queue_manager.get_queue("out")
        completed = 0

        while True:
            try:
                item = out_queue.get(True, 1)
            except QueueEmpty:
                self.check_thread_error()
                continue
            self.check_thread_error()

            if item == "EOF":
                completed += 1
                logger.debug("Got EOF %s of %s", completed, processes)
                if completed == processes:
                    break
                continue

            logger.trace("Yielding: '%s'", item[0])
            yield item
        logger.debug("iterator exhausted")
        return "EOF"


class DiskIO():
    """ Background threads to:
            Load images from disk and get the detected faces
            Save images back to disk """
    def __init__(self, alignments, images, arguments):
        logger.debug("Initializing %s: (alignments: %s, images: %s, arguments: %s)",
                     self.__class__.__name__, alignments, images, arguments)
        self.alignments = alignments
        self.images = images
        self.args = arguments

        # For frame skipping
        self.imageidxre = re.compile(r"[^(mp4)](\d+)(?!.*\d)")
        self.frame_ranges = self.get_frame_ranges()

        # Extractor for on the fly detection
        self.extractor = None
        if not self.alignments.have_alignments_file:
            self.load_extractor()

        self.load_queue = None
        self.save_queue = None
        self.load_thread = None
        self.save_thread = None
        self.init_threads()
        logger.debug("Initialized %s", self.__class__.__name__)

    # Initalization
    def get_frame_ranges(self):
        """ split out the frame ranges and parse out 'min' and 'max' values """
        if not self.args.frame_ranges:
            logger.debug("No frame range set")
            return None

        minmax = {"min": 0,  # never any frames less than 0
                  "max": float("inf")}
        retval = [tuple(map(lambda q: minmax[q] if q in minmax.keys() else int(q), v.split("-")))
                  for v in self.args.frame_ranges]
        logger.debug("frame ranges: %s", retval)
        return retval

    def load_extractor(self):
        """ Set on the fly extraction """
        logger.debug("Loading extractor")
        logger.warning("No Alignments file found. Extracting on the fly.")
        logger.warning("NB: This will use the inferior dlib-hog for extraction "
                       "and dlib pose predictor for landmarks. It is recommended "
                       "to perfom Extract first for superior results")
        extract_args = {"detector": "dlib-hog",
                        "aligner": "dlib",
                        "loglevel": self.args.loglevel}
        self.extractor = Extractor(None, extract_args)
        self.extractor.launch_detector()
        self.extractor.launch_aligner()
        logger.debug("Loaded extractor")

    def init_threads(self):
        """ Initialize queues and threads """
        logger.debug("Initializing DiskIO Threads")
        for task in ("load", "save"):
            self.add_queue(task)
            self.start_thread(task)
        logger.debug("Initialized DiskIO Threads")

    def add_queue(self, task):
        """ Add the queue to queue_manager and set queue attribute """
        logger.debug("Adding queue for task: '%s'", task)
        q_name = task
        q_size = 0
        if task == "load":
            q_name = "convert_in"
            q_size = 100
        queue_manager.add_queue(q_name, maxsize=q_size)
        setattr(self, "{}_queue".format(task), queue_manager.get_queue(q_name))
        logger.debug("Added queue for task: '%s'", task)

    def start_thread(self, task):
        """ Start the DiskIO thread """
        logger.debug("Starting thread: '%s'", task)
        func = getattr(self, task)
        io_thread = MultiThread(func, thread_count=1)
        io_thread.start()
        setattr(self, "{}_thread".format(task), io_thread)
        logger.debug("Started thread: '%s'", task)

    # Loading tasks
    def load(self):
        """ Load the images with detected_faces"""
        logger.debug("Load Images: Start")
        extract_queue = queue_manager.get_queue("extract_in") if self.extractor else None
        idx = 0
        for filename, image in self.images.load():
            idx += 1
            if self.load_queue.shutdown.is_set():
                logger.debug("Load Queue: Stop signal received. Terminating")
                break
            if (self.args.discard_frames and self.check_skipframe(filename) == "discard"):
                logger.debug("Discarding frame: '%s'", filename)
                continue
            if image is None or not image.any():
                logger.warning("Unable to open image. Skipping: '%s'", filename)
                continue

            detected_faces = self.get_detected_faces(filename, image, extract_queue)
            item = dict(filename=filename, image=image, detected_faces=detected_faces)
            self.load_queue.put(item)

        self.load_queue.put("EOF")
        logger.debug("Load Images: Complete")

    def check_skipframe(self, filename):
        """ Check whether frame is to be skipped """
        if not self.frame_ranges:
            return None
        idx = int(self.imageidxre.findall(filename)[0])
        skipframe = not any(map(lambda b: b[0] <= idx <= b[1], self.frame_ranges))
        if skipframe and self.args.discard_frames:
            skipframe = "discard"
        return skipframe

    def get_detected_faces(self, filename, image, extract_queue):
        """ Return detected faces from alignments or detector """
        logger.trace("Getting faces for: '%s'", filename)
        if not self.extractor:
            detected_faces = self.alignments_faces(os.path.basename(filename), image)
        else:
            detected_faces = self.detect_faces(extract_queue, filename, image)
        logger.trace("Got %s faces for: '%s'", len(detected_faces), filename)
        return detected_faces

    def alignments_faces(self, frame, image):
        """ Get the face from alignments file """
        if not self.check_alignments(frame):
            return list()

        faces = self.alignments.get_faces_in_frame(frame)
        detected_faces = list()

        for rawface in faces:
            face = DetectedFace()
            face.from_alignment(rawface, image=image)
            detected_faces.append(face)
        return detected_faces

    def check_alignments(self, frame):
        """ If we have no alignments for this image, skip it """
        have_alignments = self.alignments.frame_exists(frame)
        if not have_alignments:
            tqdm.write("No alignment found for {}, "
                       "skipping".format(frame))
        return have_alignments

    def detect_faces(self, load_queue, filename, image):
        """ Extract the face from a frame (If alignments file not found) """
        inp = {"filename": filename,
               "image": image}
        load_queue.put(inp)
        faces = next(self.extractor.detect_faces())

        landmarks = faces["landmarks"]
        detected_faces = faces["detected_faces"]
        final_faces = list()

        for idx, face in enumerate(detected_faces):
            detected_face = DetectedFace()
            detected_face.from_dlib_rect(face)
            detected_face.landmarksXY = landmarks[idx]
            final_faces.append(detected_face)
        return final_faces

    # Saving tasks
    def save(self):
        """ Save the converted images """
        logger.debug("Save Images: Start")
        while True:
            if self.save_queue.shutdown.is_set():
                logger.debug("Save Queue: Stop signal received. Terminating")
                break
            item = self.save_queue.get()
            if item == "EOF":
                break
            filename, image = item

            logger.trace("Saving frame: '%s'", filename)
            try:
                cv2.imwrite(filename, image)  # pylint: disable=no-member
            except Exception as err:  # pylint: disable=broad-except
                logger.error("Failed to save image '%s'. Original Error: %s", filename, err)
                continue
        logger.debug("Save Faces: Complete")


class Predict():
    """ Predict faces from incoming queue """
    def __init__(self, in_queue, arguments):
        logger.debug("Initializing %s: (args: %s, in_queue: %s)",
                     self.__class__.__name__, arguments, in_queue)
        self.batchsize = 16
        self.args = arguments
        self.in_queue = in_queue
        self.out_queue = queue_manager.get_queue("patch")
        self.serializer = Serializer.get_serializer("json")
        self.faces_count = 0
        self.verify_output = False
        self.pre_process = PostProcess(arguments)
        self.model = self.load_model()
        self.predictor = self.model.converter(self.swap_model)
        self.queues = dict()

        self.thread = MultiThread(self.predict_faces, thread_count=1)
        self.thread.start()
        logger.debug("Initialized %s: (out_queue: %s)", self.__class__.__name__, self.out_queue)

    @property
    def swap_model(self):
        """ Return swap model from args """
        return self.args.swap_model

    @property
    def training_size(self):
        """ Return the model training size """
        return self.model.state.training_size

    @property
    def coverage(self):
        """ Coverage for this model """
        return int(self.model.training_opts["coverage_ratio"] * self.training_size)

    @property
    def padding(self):
        """ Padding for this model """
        return (self.training_size - self.coverage) // 2

    @property
    def crop(self):
        """ Crop size for this model """
        return slice(self.padding, self.training_size - self.padding)

    @property
    def input_size(self):
        """ Return the model input size as a h,w tuple"""
        return (self.model.input_shape[1], self.model.input_shape[0])

    @property
    def input_mask(self):
        """ Return the input mask, if there is one, else None """
        if not self.model.state.mask_shapes:
            return None
        mask = np.zeros(self.model.state.mask_shapes[0], np.float32)
        retval = np.expand_dims(mask, 0)
        return retval

    def load_model(self):
        """ Load the model requested for conversion """
        logger.debug("Loading Model")
        model_dir = get_folder(self.args.model_dir, make_folder=False)
        if not model_dir:
            logger.error("%s does not exist.", self.args.model_dir)
            exit(1)
        trainer = self.get_trainer(model_dir)
        model = PluginLoader.get_model(trainer)(model_dir, self.args.gpus, predict=True)
        logger.debug("Loaded Model")
        return model

    def get_trainer(self, model_dir):
        """ Return the trainer name if provided, or read from state file """
        if self.args.trainer:
            logger.debug("Trainer name provided: '%s'", self.args.trainer)
            return self.args.trainer

        statefile = [fname for fname in os.listdir(str(model_dir))
                     if fname.endswith("_state.json")]
        if len(statefile) != 1:
            logger.error("There should be 1 state file in your model folder. %s were found. "
                         "Specify a trainer with the '-t', '--trainer' option.")
            exit(1)
        statefile = os.path.join(str(model_dir), statefile[0])

        with open(statefile, "rb") as inp:
            state = self.serializer.unmarshal(inp.read().decode("utf-8"))
            trainer = state.get("name", None)

        if not trainer:
            logger.error("Trainer name could not be read from state file. "
                         "Specify a trainer with the '-t', '--trainer' option.")
            exit(1)
        logger.debug("Trainer from state file: '%s'", trainer)
        return trainer

    def predict_faces(self):
        """ Get detected faces from images """
        faces_seen = 0
        batch = list()
        while True:
            item = self.in_queue.get()
            if item != "EOF":
                logger.trace("Got from queue: '%s'", item["filename"])
                faces_count = len(item["detected_faces"])
                if faces_count != 0:
                    self.pre_process.do_actions(item)
                    self.faces_count += faces_count
                if faces_count > 1:
                    self.verify_output = True
                    logger.verbose("Found more than one face in an image! '%s'",
                                   os.path.basename(item["filename"]))

                self.load_aligned(item)

                faces_seen += faces_count
                batch.append(item)

            if faces_seen < self.batchsize and item != "EOF":
                logger.trace("Continuing. Current batchsize: %s", faces_seen)
                continue

            if batch:
                detected_batch = [detected_face for item in batch
                                  for detected_face in item["detected_faces"]]
                original_faces = self.compile_original_faces(detected_batch)
                feed_faces = self.compile_feed_faces(original_faces)
                predicted = self.predict(feed_faces)
                swapped_faces = predicted[0]
                masks = None if len(predicted) != 2 else predicted[1]

                self.queue_out_frames(batch, original_faces, swapped_faces, masks)

            faces_seen = 0
            batch = list()
            if item == "EOF":
                logger.debug("Load queue complete")
                break
        self.out_queue.put("EOF")

    def load_aligned(self, item):
        """ Load the original aligned face """
        logger.trace("Loading aligned faces: '%s'", item["filename"])
        for detected_face in item["detected_faces"]:
            detected_face.load_aligned(item["image"],
                                       size=self.training_size,
                                       align_eyes=False)
        logger.trace("Loaded aligned faces: '%s'", item["filename"])

    def compile_original_faces(self, detected_faces):
        """ Crop the original faces based on coverage """
        logger.trace("Compiling original faces. Batchsize: %s", len(detected_faces))
        original_faces = np.stack([detected_face.aligned_face[:, :, :3][self.crop, self.crop]
                                   for detected_face in detected_faces])
        logger.trace("Compiled Original faces. Shape: %s", original_faces.shape)
        return original_faces

    def compile_feed_faces(self, original_faces):
        """ Compile the faces for feeding into the predictor """
        logger.trace("Compiling feed face. Batchsize: %s", len(original_faces))
        feed_faces = list()
        for orig_face in original_faces:
            feed_face = cv2.resize(orig_face.copy(),  # pylint: disable=no-member
                                   self.input_size,
                                   interpolation=cv2.INTER_AREA)  # pylint: disable=no-member
            feed_face = np.clip(feed_face / 255.0, 0.0, 1.0)
            feed_faces.append(feed_face)
        feed_faces = np.stack(feed_faces)
        logger.trace("Compiled Feed faces. Shape: %s", feed_faces.shape)
        return feed_faces

    def predict(self, feed_faces):
        """ Perform inference on the feed """
        logger.trace("Predicting: Batchsize: %s", len(feed_faces))
        feed = [feed_faces]
        if self.input_mask is not None:
            feed.append(np.repeat(self.input_mask, feed_faces.shape[0], axis=0))
        logger.trace("Input shape(s): %s", [item.shape for item in feed])
        # TODO Handle mask output
        predicted = self.predictor(feed)
        predicted = predicted if isinstance(predicted, list) else [predicted]
        logger.trace("Output shape(s): %s", [predict.shape for predict in predicted])

        output = list()
        for idx, item in enumerate(predicted):
            logger.trace("Resizing %s", "faces" if idx == 0 else "masks")
            resized = list()
            for image in item:
                resized_image = cv2.resize(  # pylint: disable=no-member
                    image,
                    (self.coverage, self.coverage),
                    interpolation=cv2.INTER_CUBIC)  # pylint: disable=no-member
                resized.append(np.clip(resized_image * 255.0, 0.0, 255.0))
            output.append(np.stack(resized))

        if len(output) == 2:  # Put mask into 3 channel format
            output[1] = np.tile(output[1][..., None], 3)

        # TODO Remove this
#        from uuid import uuid4
#        idu = uuid4()
#        for idx, img in enumerate(output[1]):
#            f_name = "/home/matt/fake/test/conv_ref/{}_{}.png".format(idu, idx)
#            cv2.imwrite(f_name, np.tile(img, 3))

        logger.trace("Predicted: Shape(s): %s", [predicted.shape for predicted in output])
        return output

    def queue_out_frames(self, batch, original_faces, swapped_faces, masks):
        """ Compile the batch back to original frames and put to out_queue """
        logger.trace("Queueing out batch. Batchsize: %s", len(batch))
        pointer = 0
        for item in batch:
            num_faces = len(item["detected_faces"])
            if num_faces == 0:
                item["original_faces"] = np.array(list())
                item["swapped_faces"] = np.array(list())
                item["masks"] = np.array(list())
            else:
                item["original_faces"] = original_faces[pointer:pointer + num_faces]
                item["swapped_faces"] = swapped_faces[pointer:pointer + num_faces]
                item["masks"] = masks[pointer:pointer +
                                      num_faces] if masks is not None else np.array(list())

            logger.trace("Putting to queue. ('%s', detected_faces: %s, original_faces: %s, "
                         "swapped_faces: %s, masks: %s)", item["filename"],
                         len(item["detected_faces"]), item["original_faces"].shape[0],
                         item["swapped_faces"].shape[0], item["masks"].shape[0])
            self.out_queue.put(item)
            pointer += num_faces
        logger.trace("Queued out batch. Batchsize: %s", len(batch))


class OptionalActions():
    """ Process the optional actions for convert """

    def __init__(self, args, input_images, alignments):
        logger.debug("Initializing %s", self.__class__.__name__)
        self.args = args
        self.input_images = input_images
        self.alignments = alignments

        self.remove_skipped_faces()
        logger.debug("Initialized %s", self.__class__.__name__)

    # SKIP FACES #
    def remove_skipped_faces(self):
        """ Remove deleted faces from the loaded alignments """
        logger.debug("Filtering Faces")
        face_hashes = self.get_face_hashes()
        if not face_hashes:
            logger.debug("No face hashes. Not skipping any faces")
            return
        pre_face_count = self.alignments.faces_count
        self.alignments.filter_hashes(face_hashes, filter_out=False)
        logger.info("Faces filtered out: %s", pre_face_count - self.alignments.faces_count)

    def get_face_hashes(self):
        """ Check for the existence of an aligned directory for identifying
            which faces in the target frames should be swapped.
            If it exists, obtain the hashes of the faces in the folder """
        face_hashes = list()
        input_aligned_dir = self.args.input_aligned_dir

        if input_aligned_dir is None:
            logger.verbose("Aligned directory not specified. All faces listed in the "
                           "alignments file will be converted")
        elif not os.path.isdir(input_aligned_dir):
            logger.warning("Aligned directory not found. All faces listed in the "
                           "alignments file will be converted")
        else:
            file_list = [path for path in get_image_paths(input_aligned_dir)]
            logger.info("Getting Face Hashes for selected Aligned Images")
            for face in tqdm(file_list, desc="Hashing Faces"):
                face_hashes.append(hash_image_file(face))
            logger.debug("Face Hashes: %s", (len(face_hashes)))
            if not face_hashes:
                logger.error("Aligned directory is empty, no faces will be converted!")
                exit(1)
            elif len(face_hashes) <= len(self.input_images) / 3:
                logger.warning("Aligned directory contains far fewer images than the input "
                               "directory, are you sure this is the right folder?")
        return face_hashes


class Legacy():
    """ Update legacy alignments:
        - Rotate landmarks and bounding boxes on legacy alignments
          and remove the 'r' parameter
        - Add face hashes to alignments file
        """
    def __init__(self, alignments, frames, faces_dir):
        self.alignments = alignments
        self.frames = {os.path.basename(frame): frame
                       for frame in frames}
        self.process(faces_dir)

    def process(self, faces_dir):
        """ Run the rotate alignments process """
        rotated = self.alignments.get_legacy_rotation()
        hashes = self.alignments.get_legacy_no_hashes()
        if not rotated and not hashes:
            return
        if rotated:
            logger.info("Legacy rotated frames found. Converting...")
            self.rotate_landmarks(rotated)
            self.alignments.save()
        if hashes and faces_dir:
            logger.info("Legacy alignments found. Adding Face Hashes...")
            self.add_hashes(hashes, faces_dir)
            self.alignments.save()

    def rotate_landmarks(self, rotated):
        """ Rotate the landmarks """
        for rotate_item in tqdm(rotated, desc="Rotating Landmarks"):
            frame = self.frames.get(rotate_item, None)
            if frame is None:
                logger.debug("Skipping missing frame: '%s'", rotate_item)
                continue
            self.alignments.rotate_existing_landmarks(rotate_item, frame)

    def add_hashes(self, hashes, faces_dir):
        """ Add Face Hashes to the alignments file """
        all_faces = dict()
        face_files = sorted(face for face in os.listdir(faces_dir) if "_" in face)
        for face in face_files:
            filename, extension = os.path.splitext(face)
            index = filename[filename.rfind("_") + 1:]
            if not index.isdigit():
                continue
            orig_frame = filename[:filename.rfind("_")] + extension
            all_faces.setdefault(orig_frame, dict())[int(index)] = os.path.join(faces_dir, face)

        for frame in tqdm(hashes):
            if frame not in all_faces.keys():
                logger.warning("Skipping missing frame: '%s'", frame)
                continue
            hash_faces = all_faces[frame]
            for index, face_path in hash_faces.items():
                hash_faces[index] = hash_image_file(face_path)
            self.alignments.add_face_hashes(frame, hash_faces)
