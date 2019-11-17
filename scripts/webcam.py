#!/usr/bin python3
""" The script to run the convert process of faceswap """

import logging
import re
import os
import sys
from threading import Event
from time import sleep
import time
import cv2

from cv2 import imwrite  # pylint:disable=no-name-in-module
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from tqdm import tqdm

from scripts.fsmedia import Alignments, Images, PostProcess, Utils
from lib.serializer import get_serializer
from lib.convert import Converter
from lib.faces_detect import DetectedFace
from lib.gpu_stats import GPUStats
from lib.image import read_image_hash
from lib.multithreading import MultiThread, total_cpus
from lib.queue_manager import queue_manager, QueueEmpty
from lib.utils import FaceswapError, get_folder, get_image_paths
from plugins.extract.pipeline import Extractor
from plugins.plugin_loader import PluginLoader

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

from scripts.convert import Convert, DiskIO, Predict, OptionalActions

class Webcam():
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
        self.disk_io = CamIO(self.alignments, self.images, arguments)
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
            retval = 32 * 5
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

        load_queue = queue_manager.get_queue("convert_in")
        detectIn_queue = queue_manager.get_queue("extract_detect_in")
        detectOut_queue = queue_manager.get_queue("extract_align_in")
        alignIn_queue = queue_manager.get_queue("extract_align_in")
        alignOut_queue = queue_manager.get_queue("extract_align_out")
        self.patch_threads = MultiThread(self.converter.process, patch_queue, save_queue,
                                         thread_count=self.pool_processes, name="patch")

        self.patch_threads.start()

        # cap = cv2.VideoCapture(0)
        # while True:
        #     startTime = time.time()
        #     cap.grab()
        #     _, frame = cap.retrieve()
        #     cv2.imshow("test", frame)
        #     k = cv2.waitKey(100) & 0xFF
        #     timeElapsed = time.time() - startTime
        #     print(timeElapsed)
        #     if k == ord('q'):
        #         cv2.destroyAllWindows()
        #         break

        #self.disk_io._windowManagerOut.createWindow()
        cv2.namedWindow("outframe")
        while True:
            # self.check_thread_error()
            # if self.disk_io.completion_event.is_set():
            #     logger.debug("DiskIO completion event set. Joining Pool")
            #     print("DiskIO completion event set. Joining Pool")
            #     break
            # if self.patch_threads.completed():
            #     logger.debug("All patch threads completed")
            #     print("All patch threads completed")
            #     break
            # sleep(1)

            #if self.disk_io._windowManagerOut.isWindowCreated:
            startTime = time.time()
            try:
                filename, outframe = save_queue.get(True, 1)
            except QueueEmpty:
                continue

            print("file: %s, alignOut: %d, detectIn: %d, detectOut: %d, alignIn: %d, load: %d, patch: %d, save: %d" %
                  (filename, alignOut_queue.qsize(), detectIn_queue.qsize(), detectOut_queue.qsize(), alignIn_queue.qsize(),
                   load_queue.qsize(), patch_queue.qsize(), save_queue.qsize()))

            # self.disk_io._windowManagerOut.show(frame)
            # self.disk_io._windowManagerOut.processEvents()
            cv2.imshow("outframe", outframe)
            keycode = cv2.waitKey(30) & 0xFF
            timeElapsed = time.time() - startTime
            print(timeElapsed)
            if keycode == ord('q'):
                cv2.destroyAllWindows()
                break

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


class CaptureManager():

    def __init__(self, capture, previewWindowManager = None, shouldMirrorPreview = False):
        self.previewWindowManager = previewWindowManager
        self.shouldMirrorPreview = shouldMirrorPreview

        self._capture = capture
        self._channel = 0
        self._enteredFrame = False
        self._frame = None
        self._imageFilename = None
        self._videoFilename = None
        self._videoEncoding = None
        self._videoWriter = None

        self._startTime = None
        self._framesElapsed = 0
        self._fpsEstimate = None

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, value):
        if self._channel != value:
            self._channel = value
            self._frame = None

    @property
    def frame(self):
        if self._enteredFrame and self._frame is None:
            _, self._frame = self._capture.retrieve()
        return self._frame

    @property
    def isWritingImage(self):
        return self._imageFilename is not None

    @property
    def isWritingVideo(self):
        return self._videoFilename is not None

    def enterFrame(self):
        """Capture the next frame, if any."""

        # check that any previous frame was exited.
        assert not self._enteredFrame, 'previous enterFrame() had no matching exitFrame()'

        if self._capture is not None:
            self._enteredFrame = self._capture.grab()

    def exitFrame(self):
        """Draw to the window. Write to files. Release the frame."""

        # Check whether any grabbed frame is retrievable.
        # The getter may retrieve and cache the frame.
        if self.frame is None:
            self._enteredFrame = False
            return

        # Update the FPS estimate and related variables.
        if self._framesElapsed == 0:
            self._startTime = time.time()
        else:
            timeElapsed = time.time() - self._startTime
            self._fpsEstimate = self._framesElapsed / timeElapsed
        self._framesElapsed += 1

        # Draw to the window, if any.
        if self.previewWindowManager is not None:
            if self.shouldMirrorPreview:
                mirroredFrame = np.fliplr(self._frame).copy()
                self.previewWindowManager.show(mirroredFrame)
            else:
                self.previewWindowManager.show(self._frame)

        # Write to the image file, if any.
        if self.isWritingImage:
            cv2.imwrite(self._imageFilename, self._frame)
            self._imageFilename = None

        # Write to the video file, if any.
        self._writeVideoFrame()

        # Release the frame.
        self._frame = None
        self._enteredFrame = False

    def writeImage(self, filename):
        self._imageFilename = filename

    def startWritingVideo(self, filename, encoding = cv2.VideoWriter_fourcc('I', '4', '2', '0')):
        self._videoFilename = filename
        self._videoEncoding = encoding

    def stopWrtingVideo(self):
        self._videoFilename = None
        self._videoEncoding = None
        self._videoWriter = None

    def _writeVideoFrame(self):

        if not self.isWritingVideo:
            return

        if self._videoWriter is None:
            fps = self._capture.get(cv2.CAP_PROP_FPS)
            if fps == 0.0:
                # The capture's FPS is unknown so use an estimate.
                if self._framesElapsed < 20:
                    # 收集20帧以此来预估实际的fps
                    return
                else:
                    fps = self._fpsEstimate
            size = (int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self._videoWriter = cv2.VideoWriter(self._videoFilename, self._videoEncoding, fps, size)

            self._videoWriter.write(self._frame)


class WindowManager():

    def __init__(self, windowName, keypressCallback = None):
        self.keypressCallback = keypressCallback
        self._windowName = windowName
        self._isWindowCreated = False

    @property
    def isWindowCreated(self):
        return self._isWindowCreated

    def createWindow(self):
        cv2.namedWindow(self._windowName)
        self._isWindowCreated = True

    def show(self, frame):
        cv2.imshow(self._windowName, frame)

    def destroyWindow(self):
        cv2.destroyWindow(self._windowName)
        self._isWindowCreated = False

    def processEvents(self):
        keycode = cv2.waitKey(1)
        if self.keypressCallback is not None and keycode != -1:
            # 过滤非ASCII GTK 编码字符
            keycode &= 0xFF
            self.keypressCallback(keycode)


class CamIO(DiskIO):

    def __init__(self, alignments, images, arguments):
    #def __init__(self):

        # self._windowManagerIn = WindowManager('CamIn', self.onKeypress)
        # self._windowManagerOut = WindowManager('CamOut', self.onKeypress)
        self._windowManager = self._windowManagerOut = None
        self._captureManager = CaptureManager(cv2.VideoCapture(0))
        super().__init__(alignments, images, arguments)

    def onKeypress(self, keycode):
        if keycode == 32: # space
            self._captureManager.writeImage('screenshot.png')
        elif keycode == 9: # tab
            if not self._captureManager.isWritingVideo:
                self._captureManager.startWritingVideo('screencast.avi')
            else:
                self._captureManager.stopWrtingVideo()
        elif keycode == 27: # escape
            self._windowManager.destroyWindow()

    # def run(self):
    #     """逐帧主循环"""
    #     self._windowManager.createWindow()
    #     while self._windowManager.isWindowCreated:
    #         self._captureManager.enterFrame()
    #         frame = self._captureManager.frame
    #
    #         self._captureManager.exitFrame()
    #         self._windowManager.processEvents()

    def load(self, *args):
        """读取摄像头帧，存入extractor_detect_in队列，
        之前extractor.launch()启动的detect线程开始检测并保存faces到extract_detect的输出队列，
        然后循环读取输出队列，存入load即convert_in队列，"""
        #self._windowManager.createWindow()
        idx = 0
        while True: #self._windowManager.isWindowCreated:
            idx += 1
            self._captureManager.enterFrame()
            frame = self._captureManager.frame
            if frame is not None:
                detected_faces = self.get_detected_faces(idx, frame)
                item = dict(filename=idx, image=frame, detected_faces=detected_faces)
                self.pre_process.do_actions(item)
                self.load_queue.put(item)

            self._captureManager.exitFrame()
            keycode = cv2.waitKey(1)
            #self._windowManager.processEvents()

    def save(self, completion_event):
        pass
        # self._windowManagerOut.createWindow()
        # while self._windowManagerOut.isWindowCreated:
        #     item = self.save_queue.get()
        #     filename, frame = item
        #     self._windowManagerOut.show(frame)
        #     self._windowManagerOut.processEvents()

if __name__ == "__main__":
    # CamIO(None, None, None).run()
    cap = cv2.VideoCapture(0)
    while True:
        startTime = time.time()
        cap.grab()
        _, frame = cap.retrieve()
        cv2.imshow("", frame)
        timeElapsed = time.time() - startTime
        print(timeElapsed)
        k = cv2.waitKey(10)
        if k == ord('q'):
            cv2.destroyAllWindows()
            break


