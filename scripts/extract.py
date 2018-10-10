#!/usr/bin python3
""" The script to run the extract process of faceswap """

import os
import sys
from pathlib import Path

from tqdm import tqdm

from lib.multithreading import MultiThread
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

        self.export_face = True
        self.save_interval = None
        if hasattr(self.args, "save_interval"):
            self.save_interval = self.args.save_interval

    def process(self):
        """ Perform the extraction process """
        print('Starting, this may take a while...')
        Utils.set_verbosity(self.args.verbose)

        image_queue = self.threaded_load()
        self.faces.initialize_detector()

        self.run_detection(image_queue)
#        self.extract_single_process(image_queue)

#        self.write_alignments()
#        images, faces = Utils.finalize(self.images.images_found,
#                                       self.faces.num_faces_detected,
#                                       self.faces.verify_output)
#        self.images.images_found = images
#        self.faces.num_faces_detected = faces

    def threaded_load(self):
        """ Load images in a background thread """
        img_thread = MultiThread(thread_count=1)
        img_queue = img_thread.queue
        img_thread.in_thread(target=self.load_images, args=(img_queue, ))
        return img_queue

    def load_images(self, image_queue):
        """ Load the images """
        sentinel = "EOF"
        for filename in self.images.input_images:
            image = Utils.cv2_read_write('read', filename)
            image_queue.put((filename, image))
        image_queue.put(sentinel)

    def write_alignments(self):
        """ Save the alignments file """
        self.alignments.write_alignments(self.faces.faces_detected)

    def run_detection(self, image_queue):
        """ Run Face Detection """
        if self.args.verbose:
            print("Running Face Detection")
        frame_no = 0
        try:
            for item in tqdm(self.faces.detect_faces(image_queue),
                             total=self.images.images_found,
                             file=sys.stdout,
                             desc="Detecting faces"):
                filename, image, faces = item

                self.faces.faces_detected[os.path.basename(filename)] = faces
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

    def process_single_face(self, idx, face, filename, image):
        """ Perform processing on found faces """
        output_file = self.output_dir / Path(
            filename).stem if self.export_face else None

        self.faces.draw_landmarks_on_face(face, image)

        resized_face, t_mat = self.faces.extractor.extract(
            image,
            face,
            256,
            self.faces.align_eyes)

        blurry_file = self.faces.detect_blurry_faces(face,
                                                     t_mat,
                                                     resized_face,
                                                     filename)
        output_file = blurry_file if blurry_file else output_file

        if self.export_face:
            filename = "{}_{}{}".format(str(output_file),
                                        str(idx),
                                        Path(filename).suffix)
            Utils.cv2_read_write('write', filename, resized_face)

        return {"x": face.x,
                "w": face.w,
                "y": face.y,
                "h": face.h,
                "landmarksXY": face.landmarks_as_xy()}
