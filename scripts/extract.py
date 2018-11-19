#!/usr/bin python3
""" The script to run the extract process of faceswap """

import os
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

from lib.faces_detect import DetectedFace
from lib.gpu_stats import GPUStats
from lib.multithreading import MultiThread, PoolProcess, SpawnProcess
from lib.queue_manager import queue_manager, QueueEmpty
from lib.utils import get_folder
from plugins.plugin_loader import PluginLoader
from scripts.fsmedia import Alignments, Images, PostProcess, Utils

tqdm.monitor_interval = 0  # workaround for TqdmSynchronisationWarning


class Extract():
    """ The extract process. """

    def __init__(self, arguments):
        self.args = arguments
        self.output_dir = get_folder(self.args.output_dir)
        print("Output Directory: {}".format(self.args.output_dir))
        self.images = Images(self.args)
        self.alignments = Alignments(self.args, True)
        self.plugins = Plugins(self.args)

        self.post_process = PostProcess(arguments)

        self.export_face = True
        self.verify_output = False
        self.save_interval = None
        if hasattr(self.args, "save_interval"):
            self.save_interval = self.args.save_interval

    def process(self):
        """ Perform the extraction process """
        print('Starting, this may take a while...')
        Utils.set_verbosity(self.args.verbose)
#        queue_manager.debug_monitor(1)
        self.threaded_io("load")
        save_thread = self.threaded_io("save")
        self.run_extraction(save_thread)
        self.alignments.save()
        Utils.finalize(self.images.images_found,
                       self.alignments.faces_count,
                       self.verify_output)

    def threaded_io(self, task, io_args=None):
        """ Load images in a background thread """
        io_args = tuple() if io_args is None else (io_args, )
        if task == "load":
            func = self.load_images
        elif task == "save":
            func = self.save_faces
        elif task == "reload":
            func = self.reload_images
        io_thread = MultiThread(thread_count=1)
        io_thread.in_thread(func, *io_args)
        return io_thread

    def load_images(self):
        """ Load the images """
        load_queue = queue_manager.get_queue("load")
        for filename, image in self.images.load():
            imagename = os.path.basename(filename)
            if imagename in self.alignments.data.keys():
                continue
            load_queue.put((filename, image))
        load_queue.put("EOF")

    def reload_images(self, detected_faces):
        """ Reload the images and pair to detected face """
        load_queue = queue_manager.get_queue("detect")
        for filename, image in self.images.load():
            detect_item = detected_faces.pop(filename, None)
            if not detect_item:
                continue
            detect_item["image"] = image
            load_queue.put(detect_item)
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
            # pylint: disable=no-member
            cv2.imwrite(out_filename, resized_face)

    def run_extraction(self, save_thread):
        """ Run Face Detection """
        to_process = self.process_item_count()
        frame_no = 0
        if self.plugins.is_parallel:
            self.plugins.launch_aligner()
            self.plugins.launch_detector()
        if not self.plugins.is_parallel:
            self.run_detection(to_process)
            self.plugins.launch_aligner()

        for faces in tqdm(self.plugins.detect_faces(extract_pass="align"),
                          total=to_process,
                          file=sys.stdout,
                          desc="Extracting faces"):

            exception = faces.get("exception", False)
            if exception:
                exit(1)
            filename = faces["filename"]

            faces["output_file"] = self.output_dir / Path(filename).stem

            self.post_process.do_actions(faces)

            faces_count = len(faces["detected_faces"])
            if self.args.verbose and faces_count == 0:
                print("Warning: No faces were detected in image: "
                      "{}".format(os.path.basename(filename)))

            if not self.verify_output and faces_count > 1:
                self.verify_output = True

            self.process_faces(filename, faces)

            frame_no += 1
            if frame_no == self.save_interval:
                self.alignments.save()
                frame_no = 0

        if self.export_face:
            queue_manager.get_queue("save").put("EOF")
        save_thread.join_threads()

    def process_item_count(self):
        """ Return the number of items to be processedd """
        processed = sum(os.path.basename(frame) in self.alignments.data.keys()
                        for frame in self.images.input_images)

        if processed != 0 and self.args.skip_existing:
            print("Skipping {} previously extracted frames".format(processed))
        if processed != 0 and self.args.skip_faces:
            print("Skipping {} frames with detected faces".format(processed))

        to_process = self.images.images_found - processed
        if to_process == 0:
            print("No frames to process. Exiting")
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

    def process_faces(self, filename, faces):
        """ Perform processing on found faces """
        final_faces = list()
        save_queue = queue_manager.get_queue("save")

        filename = faces["filename"]
        output_file = faces["output_file"]

        for idx, face in enumerate(faces["detected_faces"]):
            if self.export_face:
                save_queue.put((filename,
                                output_file,
                                face.aligned_face,
                                idx))

            final_faces.append(face.to_alignment())
        self.alignments.data[os.path.basename(filename)] = final_faces


class Plugins():
    """ Detector and Aligner Plugins and queues """
    def __init__(self, arguments):
        self.args = arguments
        self.detector = self.load_detector()
        self.aligner = self.load_aligner()
        self.is_parallel = self.set_parallel_processing()

        self.add_queues()

    def set_parallel_processing(self):
        """ Set whether to run detect and align together or seperately """
        detector_vram = self.detector.vram
        aligner_vram = self.aligner.vram
        gpu_stats = GPUStats()
        if (detector_vram == 0
                or aligner_vram == 0
                or gpu_stats.device_count == 0):
            return True

        if hasattr(self.args, "multiprocess") and not self.args.multiprocess:
            print("\nNB: Parallel processing disabled.\nYou may get faster "
                  "extraction speeds by enabling it with the -mp switch\n")
            return False

        required_vram = detector_vram + aligner_vram + 320  # 320MB buffer
        stats = gpu_stats.get_card_most_free()
        free_vram = int(stats["free"])
        if self.args.verbose:
            print("{} - {}MB free of {}MB".format(stats["device"],
                                                  free_vram,
                                                  int(stats["total"])))
        if free_vram <= required_vram:
            if self.args.verbose:
                print("Not enough free VRAM for parallel processing. "
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
        detector_name = self.args.detector.replace("-", "_").lower()

        # Rotation
        rotation = None
        if hasattr(self.args, "rotate_images"):
            rotation = self.args.rotate_images

        detector = PluginLoader.get_detector(detector_name)(
            verbose=self.args.verbose,
            rotation=rotation)

        return detector

    def load_aligner(self):
        """ Set global arguments and load aligner plugin """
        aligner_name = self.args.aligner.replace("-", "_").lower()

        # Align Eyes
        align_eyes = False
        if hasattr(self.args, 'align_eyes'):
            align_eyes = self.args.align_eyes

        # Extracted Face Size
        size = 256
        if hasattr(self.args, 'size'):
            size = self.args.size

        aligner = PluginLoader.get_aligner(aligner_name)(
            verbose=self.args.verbose,
            align_eyes=align_eyes,
            size=size)

        return aligner

    def launch_aligner(self):
        """ Launch the face aligner """
        out_queue = queue_manager.get_queue("align")
        kwargs = {"in_queue": queue_manager.get_queue("detect"),
                  "out_queue": out_queue}

        align_process = SpawnProcess()
        event = align_process.event

        align_process.in_process(self.aligner.align, **kwargs)

        # Wait for Aligner to take it's VRAM
        # The first ever load of the model for FAN has reportedly taken
        # up to 3-4 minutes, hence high timeout.
        # TODO investigate why this is and fix if possible
        event.wait(300)
        if not event.is_set():
            raise ValueError("Error inititalizing Aligner")

        try:
            err = None
            err = out_queue.get(True, 1)
        except QueueEmpty:
            pass

        if err:
            if isinstance(err, str):
                queue_manager.terminate_queues()
                print(err)
                exit(1)
            else:
                queue_manager.get_queue("detect").put(err)

    def launch_detector(self):
        """ Launch the face detector """
        out_queue = queue_manager.get_queue("detect")
        kwargs = {"in_queue": queue_manager.get_queue("load"),
                  "out_queue": out_queue,
                  "detected_face": DetectedFace()}  # Passed in to avoid race condition

        if self.args.detector == "mtcnn":
            mtcnn_kwargs = self.detector.validate_kwargs(
                self.get_mtcnn_kwargs())
            kwargs["mtcnn_kwargs"] = mtcnn_kwargs

        if self.detector.parent_is_pool:
            detect_process = PoolProcess(self.detector.detect_faces)
        else:
            detect_process = SpawnProcess()

        event = None
        if hasattr(detect_process, "event"):
            event = detect_process.event

        detect_process.in_process(self.detector.detect_faces, **kwargs)

        if not event:
            return

        event.wait(60)
        if not event.is_set():
            raise ValueError("Error inititalizing Detector")

    def get_mtcnn_kwargs(self):
        """ Add the mtcnn arguments into a kwargs dictionary """
        mtcnn_threshold = [float(thr.strip())
                           for thr in self.args.mtcnn_threshold]
        return {"minsize": self.args.mtcnn_minsize,
                "threshold": mtcnn_threshold,
                "factor": self.args.mtcnn_scalefactor}

    def detect_faces(self, extract_pass="detect"):
        """ Detect faces from in an image """
        if self.is_parallel or extract_pass == "align":
            out_queue = queue_manager.get_queue("align")
        if not self.is_parallel and extract_pass == "detect":
            out_queue = queue_manager.get_queue("detect")

        while True:
            try:
                faces = out_queue.get(True, 1)
                if faces == "EOF":
                    break
                exception = faces.get("exception", None)
                if exception is not None:
                    queue_manager.terminate_queues()
                    yield faces
                    break
            except QueueEmpty:
                continue

            yield faces
