#!/usr/bin python3
""" The script to run the extract process of faceswap """

import logging
import os
import sys
from pathlib import Path

from tqdm import tqdm

from lib.faces_detect import DetectedFace
from lib.gpu_stats import GPUStats
from lib.multithreading import MultiThread, PoolProcess, SpawnProcess
from lib.queue_manager import queue_manager, QueueEmpty
from lib.utils import get_folder, hash_encode_image
from plugins.plugin_loader import PluginLoader
from scripts.fsmedia import Alignments, Images, PostProcess, Utils

tqdm.monitor_interval = 0  # workaround for TqdmSynchronisationWarning
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Extract():
    """ The extract process. """
    def __init__(self, arguments):
        logger.debug("Initializing %s: (args: %s", self.__class__.__name__, arguments)
        self.args = arguments
        self.output_dir = get_folder(self.args.output_dir)
        logger.info("Output Directory: %s", self.args.output_dir)
        self.images = Images(self.args)
        self.alignments = Alignments(self.args, True, self.images.is_video)
        self.plugins = Plugins(self.args)

        self.post_process = PostProcess(arguments)

        self.verify_output = False
        self.save_interval = None
        if hasattr(self.args, "save_interval"):
            self.save_interval = self.args.save_interval
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def skip_num(self):
        """ Number of frames to skip if extract_every_n is passed """
        return self.args.extract_every_n if hasattr(self.args, "extract_every_n") else 1

    def process(self):
        """ Perform the extraction process """
        logger.info('Starting, this may take a while...')
        Utils.set_verbosity(self.args.loglevel)
#        queue_manager.debug_monitor(1)
        self.threaded_io("load")
        save_thread = self.threaded_io("save")
        self.run_extraction()
        save_thread.join()
        self.alignments.save()
        Utils.finalize(self.images.images_found // self.skip_num,
                       self.alignments.faces_count,
                       self.verify_output)

    def threaded_io(self, task, io_args=None):
        """ Perform I/O task in a background thread """
        logger.debug("Threading task: (Task: '%s')", task)
        io_args = tuple() if io_args is None else (io_args, )
        if task == "load":
            func = self.load_images
        elif task == "save":
            func = self.save_faces
        elif task == "reload":
            func = self.reload_images
        io_thread = MultiThread(func, *io_args, thread_count=1)
        io_thread.start()
        return io_thread

    def load_images(self):
        """ Load the images """
        logger.debug("Load Images: Start")
        load_queue = queue_manager.get_queue("load")
        idx = 0
        for filename, image in self.images.load():
            idx += 1
            if load_queue.shutdown.is_set():
                logger.debug("Load Queue: Stop signal received. Terminating")
                break
            if idx % self.skip_num != 0:
                logger.trace("Skipping image '%s' due to extract_every_n = %s",
                             filename, self.skip_num)
                continue
            if image is None or not image.any():
                logger.warning("Unable to open image. Skipping: '%s'", filename)
                continue
            imagename = os.path.basename(filename)
            if imagename in self.alignments.data.keys():
                logger.trace("Skipping image: '%s'", filename)
                continue
            item = {"filename": filename,
                    "image": image}
            load_queue.put(item)
        load_queue.put("EOF")
        logger.debug("Load Images: Complete")

    def reload_images(self, detected_faces):
        """ Reload the images and pair to detected face """
        logger.debug("Reload Images: Start. Detected Faces Count: %s", len(detected_faces))
        load_queue = queue_manager.get_queue("detect")
        for filename, image in self.images.load():
            if load_queue.shutdown.is_set():
                logger.debug("Reload Queue: Stop signal received. Terminating")
                break
            logger.trace("Reloading image: '%s'", filename)
            detect_item = detected_faces.pop(filename, None)
            if not detect_item:
                logger.warning("Couldn't find faces for: %s", filename)
                continue
            detect_item["image"] = image
            load_queue.put(detect_item)
        load_queue.put("EOF")
        logger.debug("Reload Images: Complete")

    @staticmethod
    def save_faces():
        """ Save the generated faces """
        logger.debug("Save Faces: Start")
        save_queue = queue_manager.get_queue("save")
        while True:
            if save_queue.shutdown.is_set():
                logger.debug("Save Queue: Stop signal received. Terminating")
                break
            item = save_queue.get()
            if item == "EOF":
                break
            filename, face = item

            logger.trace("Saving face: '%s'", filename)
            try:
                with open(filename, "wb") as out_file:
                    out_file.write(face)
            except Exception as err:  # pylint: disable=broad-except
                logger.error("Failed to save image '%s'. Original Error: %s", filename, err)
                continue
        logger.debug("Save Faces: Complete")

    def run_extraction(self):
        """ Run Face Detection """
        save_queue = queue_manager.get_queue("save")
        to_process = self.process_item_count()
        frame_no = 0
        size = self.args.size if hasattr(self.args, "size") else 256
        align_eyes = self.args.align_eyes if hasattr(self.args, "align_eyes") else False

        if self.plugins.is_parallel:
            logger.debug("Using parallel processing")
            self.plugins.launch_aligner()
            self.plugins.launch_detector()
        if not self.plugins.is_parallel:
            logger.debug("Using serial processing")
            self.run_detection(to_process)
            self.plugins.launch_aligner()

        for faces in tqdm(self.plugins.detect_faces(extract_pass="align"),
                          total=to_process,
                          file=sys.stdout,
                          desc="Extracting faces"):

            filename = faces["filename"]

            self.align_face(faces, align_eyes, size, filename)
            self.post_process.do_actions(faces)

            faces_count = len(faces["detected_faces"])
            if faces_count == 0:
                logger.verbose("No faces were detected in image: %s",
                               os.path.basename(filename))

            if not self.verify_output and faces_count > 1:
                self.verify_output = True

            self.output_faces(filename, faces, save_queue)

            frame_no += 1
            if frame_no == self.save_interval:
                self.alignments.save()
                frame_no = 0

        save_queue.put("EOF")

    def process_item_count(self):
        """ Return the number of items to be processedd """
        processed = sum(os.path.basename(frame) in self.alignments.data.keys()
                        for frame in self.images.input_images)
        logger.debug("Items already processed: %s", processed)

        if processed != 0 and self.args.skip_existing:
            logger.info("Skipping previously extracted frames: %s", processed)
        if processed != 0 and self.args.skip_faces:
            logger.info("Skipping frames with detected faces: %s", processed)

        to_process = (self.images.images_found - processed) // self.skip_num
        logger.debug("Items to be Processed: %s", to_process)
        if to_process == 0:
            logger.error("No frames to process. Exiting")
            queue_manager.terminate_queues()
            exit(0)
        return to_process

    def run_detection(self, to_process):
        """ Run detection only """
        self.plugins.launch_detector()
        detected_faces = dict()
        for detected in tqdm(self.plugins.detect_faces(extract_pass="detect"),
                             total=to_process,
                             file=sys.stdout,
                             desc="Detecting faces"):
            exception = detected.get("exception", False)
            if exception:
                break

            del detected["image"]
            filename = detected["filename"]

            detected_faces[filename] = detected

        self.threaded_io("reload", detected_faces)

    def align_face(self, faces, align_eyes, size, filename):
        """ Align the detected face and add the destination file path """
        final_faces = list()
        image = faces["image"]
        landmarks = faces["landmarks"]
        detected_faces = faces["detected_faces"]
        for idx, face in enumerate(detected_faces):
            detected_face = DetectedFace()
            detected_face.from_dlib_rect(face, image)
            detected_face.landmarksXY = landmarks[idx]
            detected_face.load_aligned(image, size=size, align_eyes=align_eyes)
            final_faces.append({"file_location": self.output_dir / Path(filename).stem,
                                "face": detected_face})
        faces["detected_faces"] = final_faces

    def output_faces(self, filename, faces, save_queue):
        """ Output faces to save thread """
        final_faces = list()
        for idx, detected_face in enumerate(faces["detected_faces"]):
            output_file = detected_face["file_location"]
            extension = Path(filename).suffix
            out_filename = "{}_{}{}".format(str(output_file), str(idx), extension)

            face = detected_face["face"]
            resized_face = face.aligned_face

            face.hash, img = hash_encode_image(resized_face, extension)
            save_queue.put((out_filename, img))
            final_faces.append(face.to_alignment())
        self.alignments.data[os.path.basename(filename)] = final_faces


class Plugins():
    """ Detector and Aligner Plugins and queues """
    def __init__(self, arguments, converter_args=None):
        logger.debug("Initializing %s", self.__class__.__name__)
        self.args = arguments
        self.converter_args = converter_args  # Arguments from converter for on the fly extract
        if converter_args is not None:
            self.loglevel = converter_args["loglevel"]
        else:
            self.loglevel = self.args.loglevel

        self.detector = self.load_detector()
        self.aligner = self.load_aligner()
        self.is_parallel = self.set_parallel_processing()

        self.process_detect = None
        self.process_align = None
        self.add_queues()
        logger.debug("Initialized %s", self.__class__.__name__)

    def set_parallel_processing(self):
        """ Set whether to run detect and align together or separately """
        detector_vram = self.detector.vram
        aligner_vram = self.aligner.vram
        gpu_stats = GPUStats()
        if (detector_vram == 0
                or aligner_vram == 0
                or gpu_stats.device_count == 0):
            logger.debug("At least one of aligner or detector have no VRAM requirement. "
                         "Enabling parallel processing.")
            return True

        if hasattr(self.args, "multiprocess") and not self.args.multiprocess:
            logger.info("NB: Parallel processing disabled.You may get faster "
                        "extraction speeds by enabling it with the -mp switch")
            return False

        required_vram = detector_vram + aligner_vram + 320  # 320MB buffer
        stats = gpu_stats.get_card_most_free()
        free_vram = int(stats["free"])
        logger.verbose("%s - %sMB free of %sMB",
                       stats["device"],
                       free_vram,
                       int(stats["total"]))
        if free_vram <= required_vram:
            logger.warning("Not enough free VRAM for parallel processing. "
                           "Switching to serial")
            return False
        return True

    def add_queues(self):
        """ Add the required processing queues to Queue Manager """
        for task in ("load", "detect", "align", "save"):
            size = 0
            if task == "load" or (not self.is_parallel and task == "detect"):
                size = 100
            queue_manager.add_queue(task, maxsize=size)

    def load_detector(self):
        """ Set global arguments and load detector plugin """
        if not self.converter_args:
            detector_name = self.args.detector.replace("-", "_").lower()
        else:
            detector_name = self.converter_args["detector"]
        logger.debug("Loading Detector: '%s'", detector_name)
        # Rotation
        rotation = self.args.rotate_images if hasattr(self.args, "rotate_images") else None
        # Min acceptable face size:
        min_size = self.args.min_size if hasattr(self.args, "min_size") else 0

        detector = PluginLoader.get_detector(detector_name)(
            loglevel=self.loglevel,
            rotation=rotation,
            min_size=min_size)

        return detector

    def load_aligner(self):
        """ Set global arguments and load aligner plugin """
        if not self.converter_args:
            aligner_name = self.args.aligner.replace("-", "_").lower()
        else:
            aligner_name = self.converter_args["aligner"]

        logger.debug("Loading Aligner: '%s'", aligner_name)

        aligner = PluginLoader.get_aligner(aligner_name)(
            loglevel=self.loglevel)

        return aligner

    def launch_aligner(self):
        """ Launch the face aligner """
        logger.debug("Launching Aligner")
        out_queue = queue_manager.get_queue("align")
        kwargs = {"in_queue": queue_manager.get_queue("detect"),
                  "out_queue": out_queue}

        self.process_align = SpawnProcess(self.aligner.run, **kwargs)
        event = self.process_align.event
        error = self.process_align.error
        self.process_align.start()

        # Wait for Aligner to take it's VRAM
        # The first ever load of the model for FAN has reportedly taken
        # up to 3-4 minutes, hence high timeout.
        # TODO investigate why this is and fix if possible
        for mins in reversed(range(5)):
            for seconds in range(60):
                event.wait(seconds)
                if event.is_set():
                    break
                if error.is_set():
                    break
            if event.is_set():
                break
            if mins == 0 or error.is_set():
                raise ValueError("Error initializing Aligner")
            logger.info("Waiting for Aligner... Time out in %s minutes", mins)

        logger.debug("Launched Aligner")

    def launch_detector(self):
        """ Launch the face detector """
        logger.debug("Launching Detector")
        out_queue = queue_manager.get_queue("detect")
        kwargs = {"in_queue": queue_manager.get_queue("load"),
                  "out_queue": out_queue}
        if self.converter_args:
            kwargs["processes"] = 1
        mp_func = PoolProcess if self.detector.parent_is_pool else SpawnProcess
        self.process_detect = mp_func(self.detector.run, **kwargs)

        event = self.process_detect.event if hasattr(self.process_detect, "event") else None
        error = self.process_detect.error if hasattr(self.process_detect, "error") else None
        self.process_detect.start()

        if event is None:
            logger.debug("Launched Detector")
            return

        for mins in reversed(range(5)):
            for seconds in range(60):
                event.wait(seconds)
                if event.is_set():
                    break
                if error and error.is_set():
                    break
            if event.is_set():
                break
            if mins == 0 or (error and error.is_set()):
                raise ValueError("Error initializing Detector")
            logger.info("Waiting for Detector... Time out in %s minutes", mins)

        logger.debug("Launched Detector")

    def detect_faces(self, extract_pass="detect"):
        """ Detect faces from in an image """
        logger.debug("Running Detection. Pass: '%s'", extract_pass)
        if self.is_parallel or extract_pass == "align":
            out_queue = queue_manager.get_queue("align")
        if not self.is_parallel and extract_pass == "detect":
            out_queue = queue_manager.get_queue("detect")

        while True:
            try:
                faces = out_queue.get(True, 1)
                if faces == "EOF":
                    break
                if isinstance(faces, dict) and faces.get("exception"):
                    pid = faces["exception"][0]
                    t_back = faces["exception"][1].getvalue()
                    err = "Error in child process {}. {}".format(pid, t_back)
                    raise Exception(err)
            except QueueEmpty:
                continue

            yield faces
        logger.debug("Detection Complete")
