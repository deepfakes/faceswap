#!/usr/bin python3
""" The script to run the convert process of faceswap """

import logging
import re
import os
import sys
from threading import Event
from time import sleep

from cv2 import imwrite  # pylint:disable=no-name-in-module
import numpy as np
from tqdm import tqdm

from scripts.fsmedia import Alignments, Images, PostProcess, Utils
from lib.serializer import get_serializer
from lib.convert import Converter
from lib.faces_detect import DetectedFace
from lib.gpu_stats import GPUStats
from lib.image import read_image_hash
from lib.multithreading import MultiThread, total_cpus
from lib.queue_manager import queue_manager
from lib.utils import FaceswapError, get_folder, get_image_paths
from plugins.extract.pipeline import Extractor
from plugins.plugin_loader import PluginLoader

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Convert():
    """ The convert process. """
    def __init__(self, arguments):
        logger.debug("Initializing %s: (args: %s)", self.__class__.__name__, arguments)
        self.args = arguments
        Utils.set_verbosity(self.args.loglevel)

        self.patch_threads = None
        self.images = Images(self.args)
        self.validate()
        self.alignments = Alignments(self.args, False, self.images.is_video)
        self.opts = OptionalActions(self.args, self.images.input_images, self.alignments)

        self.add_queues()
        self.disk_io = DiskIO(self.alignments, self.images, arguments)
        self.predictor = Predict(self.disk_io.load_queue, self.queue_size, arguments)

        configfile = self.args.configfile if hasattr(self.args, "configfile") else None
        self.converter = Converter(get_folder(self.args.output_dir),
                                   self.predictor.output_size,
                                   self.predictor.has_predicted_mask,
                                   self.disk_io.draw_transparent,
                                   self.disk_io.pre_encode,
                                   arguments,
                                   configfile=configfile)

        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def queue_size(self):
        """ Set 16 for singleprocess otherwise 32 """
        if self.args.singleprocess:
            retval = 16
        else:
            retval = 32
        logger.debug(retval)
        return retval

    @property
    def pool_processes(self):
        """ return the maximum number of pooled processes to use """
        if self.args.singleprocess:
            retval = 1
        elif self.args.jobs > 0:
            retval = min(self.args.jobs, total_cpus(), self.images.images_found)
        else:
            retval = min(total_cpus(), self.images.images_found)
        retval = 1 if retval == 0 else retval
        logger.debug(retval)
        return retval

    def validate(self):
        """ Make the output folder if it doesn't exist and check that video flag is
            a valid choice """
        if (self.args.writer == "ffmpeg" and
                not self.images.is_video and
                self.args.reference_video is None):
            raise FaceswapError("Output as video selected, but using frames as input. You must "
                                "provide a reference video ('-ref', '--reference-video').")
        output_dir = get_folder(self.args.output_dir)
        logger.info("Output Directory: %s", output_dir)

    def add_queues(self):
        """ Add the queues for convert """
        logger.debug("Adding queues. Queue size: %s", self.queue_size)
        for qname in ("convert_in", "convert_out", "patch"):
            queue_manager.add_queue(qname, self.queue_size)

    def process(self):
        """ Process the conversion """
        logger.debug("Starting Conversion")
        # queue_manager.debug_monitor(5)
        try:
            self.convert_images()
            self.disk_io.save_thread.join()
            queue_manager.terminate_queues()

            Utils.finalize(self.images.images_found,
                           self.predictor.faces_count,
                           self.predictor.verify_output)
            logger.debug("Completed Conversion")
        except MemoryError as err:
            msg = ("Faceswap ran out of RAM running convert. Conversion is very system RAM "
                   "heavy, so this can happen in certain circumstances when you have a lot of "
                   "cpus but not enough RAM to support them all."
                   "\nYou should lower the number of processes in use by either setting the "
                   "'singleprocess' flag (-sp) or lowering the number of parallel jobs (-j).")
            raise FaceswapError(msg) from err

    def convert_images(self):
        """ Convert the images """
        logger.debug("Converting images")
        save_queue = queue_manager.get_queue("convert_out")
        patch_queue = queue_manager.get_queue("patch")
        self.patch_threads = MultiThread(self.converter.process, patch_queue, save_queue,
                                         thread_count=self.pool_processes, name="patch")

        self.patch_threads.start()
        while True:
            self.check_thread_error()
            if self.disk_io.completion_event.is_set():
                logger.debug("DiskIO completion event set. Joining Pool")
                break
            sleep(1)
        self.patch_threads.join()

        logger.debug("Putting EOF")
        save_queue.put("EOF")
        logger.debug("Converted images")

    def check_thread_error(self):
        """ Check and raise thread errors """
        for thread in (self.predictor.thread,
                       self.disk_io.load_thread,
                       self.disk_io.save_thread,
                       self.patch_threads):
            thread.check_and_raise_error()


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
        self.pre_process = PostProcess(arguments)
        self.completion_event = Event()

        # For frame skipping
        self.imageidxre = re.compile(r"(\d+)(?!.*\d\.)(?=\.\w+$)")
        self.frame_ranges = self.get_frame_ranges()
        self.writer = self.get_writer()

        # Extractor for on the fly detection
        self.extractor = self.load_extractor()

        self.load_queue = None
        self.save_queue = None
        self.load_thread = None
        self.save_thread = None
        self.init_threads()
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def draw_transparent(self):
        """ Draw transparent is an image writer only parameter.
            Return the value here for easy access for predictor """
        return self.writer.config.get("draw_transparent", False)

    @property
    def pre_encode(self):
        """ Return the writer's pre-encoder """
        dummy = np.zeros((20, 20, 3), dtype="uint8")
        test = self.writer.pre_encode(dummy)
        retval = None if test is None else self.writer.pre_encode
        logger.debug("Writer pre_encode function: %s", retval)
        return retval

    @property
    def total_count(self):
        """ Return the total number of frames to be converted """
        if self.frame_ranges and not self.args.keep_unchanged:
            retval = sum([fr[1] - fr[0] + 1 for fr in self.frame_ranges])
        else:
            retval = self.images.images_found
        logger.debug(retval)
        return retval

    # Initalization
    def get_writer(self):
        """ Return the writer plugin """
        args = [self.args.output_dir]
        if self.args.writer in ("ffmpeg", "gif"):
            args.extend([self.total_count, self.frame_ranges])
        if self.args.writer == "ffmpeg":
            if self.images.is_video:
                args.append(self.args.input_dir)
            else:
                args.append(self.args.reference_video)
        logger.debug("Writer args: %s", args)
        configfile = self.args.configfile if hasattr(self.args, "configfile") else None
        return PluginLoader.get_converter("writer", self.args.writer)(*args, configfile=configfile)

    def get_frame_ranges(self):
        """ split out the frame ranges and parse out 'min' and 'max' values """
        if not self.args.frame_ranges:
            logger.debug("No frame range set")
            return None

        minframe, maxframe = None, None
        if self.images.is_video:
            minframe, maxframe = 1, self.images.images_found
        else:
            indices = [int(self.imageidxre.findall(os.path.basename(filename))[0])
                       for filename in self.images.input_images]
            if indices:
                minframe, maxframe = min(indices), max(indices)
        logger.debug("minframe: %s, maxframe: %s", minframe, maxframe)

        if minframe is None or maxframe is None:
            raise FaceswapError("Frame Ranges specified, but could not determine frame numbering "
                                "from filenames")

        retval = list()
        for rng in self.args.frame_ranges:
            if "-" not in rng:
                raise FaceswapError("Frame Ranges not specified in the correct format")
            start, end = rng.split("-")
            retval.append((max(int(start), minframe), min(int(end), maxframe)))
        logger.debug("frame ranges: %s", retval)
        return retval

    def load_extractor(self):
        """ Set on the fly extraction """
        if self.alignments.have_alignments_file:
            return None

        logger.debug("Loading extractor")
        logger.warning("No Alignments file found. Extracting on the fly.")
        logger.warning("NB: This will use the inferior cv2-dnn for extraction "
                       "and  landmarks. It is recommended to perfom Extract first for "
                       "superior results")
        extractor = Extractor(detector="cv2-dnn",
                              aligner="cv2-dnn",
                              masker="none",
                              multiprocess=True,
                              rotate_images=None,
                              min_size=20)
        extractor.launch()
        logger.debug("Loaded extractor")
        return extractor

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
        if task == "load":
            q_name = "convert_in"
        elif task == "save":
            q_name = "convert_out"
        else:
            q_name = task
        setattr(self,
                "{}_queue".format(task),
                queue_manager.get_queue(q_name))
        logger.debug("Added queue for task: '%s'", task)

    def start_thread(self, task):
        """ Start the DiskIO thread """
        logger.debug("Starting thread: '%s'", task)
        args = self.completion_event if task == "save" else None
        func = getattr(self, task)
        io_thread = MultiThread(func, args, thread_count=1)
        io_thread.start()
        setattr(self, "{}_thread".format(task), io_thread)
        logger.debug("Started thread: '%s'", task)

    # Loading tasks
    def load(self, *args):  # pylint: disable=unused-argument
        """ Load the images with detected_faces"""
        logger.debug("Load Images: Start")
        idx = 0
        for filename, image in self.images.load():
            idx += 1
            if self.load_queue.shutdown.is_set():
                logger.debug("Load Queue: Stop signal received. Terminating")
                break
            if image is None or (not image.any() and image.ndim not in (2, 3)):
                # All black frames will return not np.any() so check dims too
                logger.warning("Unable to open image. Skipping: '%s'", filename)
                continue
            if self.check_skipframe(filename):
                if self.args.keep_unchanged:
                    logger.trace("Saving unchanged frame: %s", filename)
                    out_file = os.path.join(self.args.output_dir, os.path.basename(filename))
                    self.save_queue.put((out_file, image))
                else:
                    logger.trace("Discarding frame: '%s'", filename)
                continue

            detected_faces = self.get_detected_faces(filename, image)
            item = dict(filename=filename, image=image, detected_faces=detected_faces)
            self.pre_process.do_actions(item)
            self.load_queue.put(item)

        logger.debug("Putting EOF")
        self.load_queue.put("EOF")
        logger.debug("Load Images: Complete")

    def check_skipframe(self, filename):
        """ Check whether frame is to be skipped """
        if not self.frame_ranges:
            return None
        indices = self.imageidxre.findall(filename)
        if not indices:
            logger.warning("Could not determine frame number. Frame will be converted: '%s'",
                           filename)
            return False
        idx = int(indices[0]) if indices else None
        skipframe = not any(map(lambda b: b[0] <= idx <= b[1], self.frame_ranges))
        logger.trace("idx: %s, skipframe: %s", idx, skipframe)
        return skipframe

    def get_detected_faces(self, filename, image):
        """ Return detected faces from alignments or detector """
        logger.trace("Getting faces for: '%s'", filename)
        if not self.extractor:
            detected_faces = self.alignments_faces(os.path.basename(filename), image)
        else:
            detected_faces = self.detect_faces(filename, image)
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

    def detect_faces(self, filename, image):
        """ Extract the face from a frame (If alignments file not found) """
        inp = {"filename": filename,
               "image": image}
        self.extractor.input_queue.put(inp)
        faces = next(self.extractor.detected_faces())

        final_faces = [face for face in faces["detected_faces"]]
        return final_faces

    # Saving tasks
    def save(self, completion_event):
        """ Save the converted images """
        logger.debug("Save Images: Start")
        write_preview = self.args.redirect_gui and self.writer.is_stream
        preview_image = os.path.join(self.writer.output_folder, ".gui_preview.jpg")
        logger.debug("Write preview for gui: %s", write_preview)
        for idx in tqdm(range(self.total_count), desc="Converting", file=sys.stdout):
            if self.save_queue.shutdown.is_set():
                logger.debug("Save Queue: Stop signal received. Terminating")
                break
            item = self.save_queue.get()
            if item == "EOF":
                logger.debug("EOF Received")
                break
            filename, image = item
            # Write out preview image for the GUI every 10 frames if writing to stream
            if write_preview and idx % 10 == 0 and not os.path.exists(preview_image):
                logger.debug("Writing GUI Preview image: '%s'", preview_image)
                imwrite(preview_image, image)
            self.writer.write(filename, image)
        self.writer.close()
        completion_event.set()
        logger.debug("Save Faces: Complete")


class Predict():
    """ Predict faces from incoming queue """
    def __init__(self, in_queue, queue_size, arguments):
        logger.debug("Initializing %s: (args: %s, queue_size: %s, in_queue: %s)",
                     self.__class__.__name__, arguments, queue_size, in_queue)
        self.batchsize = self.get_batchsize(queue_size)
        self.args = arguments
        self.in_queue = in_queue
        self.out_queue = queue_manager.get_queue("patch")
        self.serializer = get_serializer("json")
        self.faces_count = 0
        self.verify_output = False
        self.model = self.load_model()
        self.output_indices = {"face": self.model.largest_face_index,
                               "mask": self.model.largest_mask_index}
        self.predictor = self.model.converter(self.args.swap_model)
        self.queues = dict()

        self.thread = MultiThread(self.predict_faces, thread_count=1)
        self.thread.start()
        logger.debug("Initialized %s: (out_queue: %s)", self.__class__.__name__, self.out_queue)

    @property
    def coverage_ratio(self):
        """ Return coverage ratio from training options """
        return self.model.training_opts["coverage_ratio"]

    @property
    def input_size(self):
        """ Return the model input size """
        return self.model.input_shape[0]

    @property
    def output_size(self):
        """ Return the model output size """
        return self.model.output_shape[0]

    @property
    def input_mask(self):
        """ Return the input mask """
        mask = np.zeros((1, ) + self.model.state.mask_shapes[0], dtype="float32")
        return mask

    @property
    def has_predicted_mask(self):
        """ Return whether this model has a predicted mask """
        return bool(self.model.state.mask_shapes)

    @staticmethod
    def get_batchsize(queue_size):
        """ Get the batchsize """
        logger.debug("Getting batchsize")
        is_cpu = GPUStats().device_count == 0
        batchsize = 1 if is_cpu else 16
        batchsize = min(queue_size, batchsize)
        logger.debug("Batchsize: %s", batchsize)
        logger.debug("Got batchsize: %s", batchsize)
        return batchsize

    def load_model(self):
        """ Load the model requested for conversion """
        logger.debug("Loading Model")
        model_dir = get_folder(self.args.model_dir, make_folder=False)
        if not model_dir:
            raise FaceswapError("{} does not exist.".format(self.args.model_dir))
        trainer = self.get_trainer(model_dir)
        gpus = 1 if not hasattr(self.args, "gpus") else self.args.gpus
        model = PluginLoader.get_model(trainer)(model_dir, gpus, predict=True)
        logger.debug("Loaded Model")
        return model

    def get_trainer(self, model_dir):
        """ Return the trainer name if provided, or read from state file """
        if hasattr(self.args, "trainer") and self.args.trainer:
            logger.debug("Trainer name provided: '%s'", self.args.trainer)
            return self.args.trainer

        statefile = [fname for fname in os.listdir(str(model_dir))
                     if fname.endswith("_state.json")]
        if len(statefile) != 1:
            raise FaceswapError("There should be 1 state file in your model folder. {} were "
                                "found. Specify a trainer with the '-t', '--trainer' "
                                "option.".format(len(statefile)))
        statefile = os.path.join(str(model_dir), statefile[0])

        state = self.serializer.load(statefile)
        trainer = state.get("name", None)

        if not trainer:
            raise FaceswapError("Trainer name could not be read from state file. "
                                "Specify a trainer with the '-t', '--trainer' option.")
        logger.debug("Trainer from state file: '%s'", trainer)
        return trainer

    def predict_faces(self):
        """ Get detected faces from images """
        faces_seen = 0
        consecutive_no_faces = 0
        batch = list()
        is_plaidml = GPUStats().is_plaidml
        while True:
            item = self.in_queue.get()
            if item != "EOF":
                logger.trace("Got from queue: '%s'", item["filename"])
                faces_count = len(item["detected_faces"])

                # Safety measure. If a large stream of frames appear that do not have faces,
                # these will stack up into RAM. Keep a count of consecutive frames with no faces.
                # If self.batchsize number of frames appear, force the current batch through
                # to clear RAM.
                consecutive_no_faces = consecutive_no_faces + 1 if faces_count == 0 else 0
                self.faces_count += faces_count
                if faces_count > 1:
                    self.verify_output = True
                    logger.verbose("Found more than one face in an image! '%s'",
                                   os.path.basename(item["filename"]))

                self.load_aligned(item)

                faces_seen += faces_count
                batch.append(item)

            if item != "EOF" and (faces_seen < self.batchsize and
                                  consecutive_no_faces < self.batchsize):
                logger.trace("Continuing. Current batchsize: %s, consecutive_no_faces: %s",
                             faces_seen, consecutive_no_faces)
                continue

            if batch:
                logger.trace("Batching to predictor. Frames: %s, Faces: %s",
                             len(batch), faces_seen)
                detected_batch = [detected_face for item in batch
                                  for detected_face in item["detected_faces"]]
                if faces_seen != 0:
                    feed_faces = self.compile_feed_faces(detected_batch)
                    batch_size = None
                    if is_plaidml and feed_faces.shape[0] != self.batchsize:
                        logger.verbose("Fallback to BS=1")
                        batch_size = 1
                    predicted = self.predict(feed_faces, batch_size)
                else:
                    predicted = list()

                self.queue_out_frames(batch, predicted)

            consecutive_no_faces = 0
            faces_seen = 0
            batch = list()
            if item == "EOF":
                logger.debug("EOF Received")
                break
        logger.debug("Putting EOF")
        self.out_queue.put("EOF")
        logger.debug("Load queue complete")

    def load_aligned(self, item):
        """ Load the feed faces and reference output faces """
        logger.trace("Loading aligned faces: '%s'", item["filename"])
        for detected_face in item["detected_faces"]:
            detected_face.load_feed_face(item["image"],
                                         size=self.input_size,
                                         coverage_ratio=self.coverage_ratio,
                                         dtype="float32")
            if self.input_size == self.output_size:
                detected_face.reference = detected_face.feed
            else:
                detected_face.load_reference_face(item["image"],
                                                  size=self.output_size,
                                                  coverage_ratio=self.coverage_ratio,
                                                  dtype="float32")
        logger.trace("Loaded aligned faces: '%s'", item["filename"])

    @staticmethod
    def compile_feed_faces(detected_faces):
        """ Compile the faces for feeding into the predictor """
        logger.trace("Compiling feed face. Batchsize: %s", len(detected_faces))
        feed_faces = np.stack([detected_face.feed_face[..., :3]
                               for detected_face in detected_faces]) / 255.0
        logger.trace("Compiled Feed faces. Shape: %s", feed_faces.shape)
        return feed_faces

    def predict(self, feed_faces, batch_size=None):
        """ Perform inference on the feed """
        logger.trace("Predicting: Batchsize: %s", len(feed_faces))
        feed = [feed_faces]
        if self.has_predicted_mask:
            feed.append(np.repeat(self.input_mask, feed_faces.shape[0], axis=0))
        logger.trace("Input shape(s): %s", [item.shape for item in feed])

        predicted = self.predictor(feed, batch_size=batch_size)
        predicted = predicted if isinstance(predicted, list) else [predicted]
        logger.trace("Output shape(s): %s", [predict.shape for predict in predicted])

        predicted = self.filter_multi_out(predicted)

        # Compile masks into alpha channel or keep raw faces
        predicted = np.concatenate(predicted, axis=-1) if len(predicted) == 2 else predicted[0]
        predicted = predicted.astype("float32")

        logger.trace("Final shape: %s", predicted.shape)
        return predicted

    def filter_multi_out(self, predicted):
        """ Filter the predicted output to the final output """
        if not predicted:
            return predicted
        face = predicted[self.output_indices["face"]]
        mask_idx = self.output_indices["mask"]
        mask = predicted[mask_idx] if mask_idx is not None else None
        predicted = [face, mask] if mask is not None else [face]
        logger.trace("Filtered output shape(s): %s", [predict.shape for predict in predicted])
        return predicted

    def queue_out_frames(self, batch, swapped_faces):
        """ Compile the batch back to original frames and put to out_queue """
        logger.trace("Queueing out batch. Batchsize: %s", len(batch))
        pointer = 0
        for item in batch:
            num_faces = len(item["detected_faces"])
            if num_faces == 0:
                item["swapped_faces"] = np.array(list())
            else:
                item["swapped_faces"] = swapped_faces[pointer:pointer + num_faces]

            logger.trace("Putting to queue. ('%s', detected_faces: %s, swapped_faces: %s)",
                         item["filename"], len(item["detected_faces"]),
                         item["swapped_faces"].shape[0])
            pointer += num_faces
        self.out_queue.put(batch)
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
                face_hashes.append(read_image_hash(face))
            logger.debug("Face Hashes: %s", (len(face_hashes)))
            if not face_hashes:
                raise FaceswapError("Aligned directory is empty, no faces will be converted!")
            if len(face_hashes) <= len(self.input_images) / 3:
                logger.warning("Aligned directory contains far fewer images than the input "
                               "directory, are you sure this is the right folder?")
        return face_hashes
