#!/usr/bin python3
""" The script to run the extract process of faceswap """

import logging
import os
import sys
from pathlib import Path

from tqdm import tqdm

from lib.faces_detect import DetectedFace
from lib.multithreading import MultiThread
from lib.queue_manager import queue_manager
from lib.utils import get_folder, hash_encode_image
from plugins.extract.pipeline import Extractor
from scripts.fsmedia import Alignments, Images, PostProcess, Utils

tqdm.monitor_interval = 0  # workaround for TqdmSynchronisationWarning
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Extract():
    """ The extract process. """
    def __init__(self, arguments):
        logger.debug("Initializing %s: (args: %s", self.__class__.__name__, arguments)
        self.args = arguments
        Utils.set_verbosity(self.args.loglevel)
        self.output_dir = get_folder(self.args.output_dir)
        logger.info("Output Directory: %s", self.args.output_dir)
        self.images = Images(self.args)
        self.alignments = Alignments(self.args, True, self.images.is_video)
        self.post_process = PostProcess(arguments)
        configfile = self.args.configfile if hasattr(self.args, "configfile") else None
        normalization = None if self.args.normalization == "none" else self.args.normalization
        self.extractor = Extractor(self.args.detector,
                                   self.args.aligner,
                                   self.args.loglevel,
                                   configfile=configfile,
                                   multiprocess=not self.args.singleprocess,
                                   rotate_images=self.args.rotate_images,
                                   min_size=self.args.min_size,
                                   normalize_method=normalization)
        self.save_queue = queue_manager.get_queue("extract_save")
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
        # queue_manager.debug_monitor(3)
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
        load_queue = self.extractor.input_queue
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
        load_queue = self.extractor.input_queue
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

    def save_faces(self):
        """ Save the generated faces """
        logger.debug("Save Faces: Start")
        while True:
            if self.save_queue.shutdown.is_set():
                logger.debug("Save Queue: Stop signal received. Terminating")
                break
            item = self.save_queue.get()
            logger.trace(item)
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

    def run_extraction(self):
        """ Run Face Detection """
        to_process = self.process_item_count()
        size = self.args.size if hasattr(self.args, "size") else 256
        align_eyes = self.args.align_eyes if hasattr(self.args, "align_eyes") else False
        exception = False

        for phase in range(self.extractor.passes):
            if exception:
                break
            is_final = self.extractor.final_pass
            detected_faces = dict()
            self.extractor.launch()
            for idx, faces in enumerate(tqdm(self.extractor.detected_faces(),
                                             total=to_process,
                                             file=sys.stdout,
                                             desc="Running pass {} of {}: {}".format(
                                                 phase + 1,
                                                 self.extractor.passes,
                                                 self.extractor.phase.title()))):

                exception = faces.get("exception", False)
                if exception:
                    break
                filename = faces["filename"]

                if self.extractor.final_pass:
                    self.output_processing(faces, align_eyes, size, filename)
                    self.output_faces(filename, faces)
                    if self.save_interval and (idx + 1) % self.save_interval == 0:
                        self.alignments.save()
                else:
                    del faces["image"]
                    detected_faces[filename] = faces

            if is_final:
                logger.debug("Putting EOF to save")
                self.save_queue.put("EOF")
            else:
                logger.debug("Reloading images")
                self.threaded_io("reload", detected_faces)

    def output_processing(self, faces, align_eyes, size, filename):
        """ Prepare faces for output """
        self.align_face(faces, align_eyes, size, filename)
        self.post_process.do_actions(faces)

        faces_count = len(faces["detected_faces"])
        if faces_count == 0:
            logger.verbose("No faces were detected in image: %s",
                           os.path.basename(filename))

        if not self.verify_output and faces_count > 1:
            self.verify_output = True

    def align_face(self, faces, align_eyes, size, filename):
        """ Align the detected face and add the destination file path """
        final_faces = list()
        image = faces["image"]
        landmarks = faces["landmarks"]
        detected_faces = faces["detected_faces"]
        for idx, face in enumerate(detected_faces):
            detected_face = DetectedFace()
            detected_face.from_bounding_box_dict(face, image)
            detected_face.landmarksXY = landmarks[idx]
            detected_face.load_aligned(image, size=size, align_eyes=align_eyes)
            final_faces.append({"file_location": self.output_dir / Path(filename).stem,
                                "face": detected_face})
        faces["detected_faces"] = final_faces

    def output_faces(self, filename, faces):
        """ Output faces to save thread """
        final_faces = list()
        for idx, detected_face in enumerate(faces["detected_faces"]):
            output_file = detected_face["file_location"]
            extension = Path(filename).suffix
            out_filename = "{}_{}{}".format(str(output_file), str(idx), extension)

            face = detected_face["face"]
            resized_face = face.aligned_face

            face.hash, img = hash_encode_image(resized_face, extension)
            self.save_queue.put((out_filename, img))
            final_faces.append(face.to_alignment())
        self.alignments.data[os.path.basename(filename)] = final_faces
