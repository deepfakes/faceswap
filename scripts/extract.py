#!/usr/bin python3
""" The script to run the extract process of faceswap """

import os
from pathlib import Path

from tqdm import tqdm

from lib.multithreading import pool_process
from scripts.fsmedia import Alignments, Faces, Images, Utils

class Extract(object):
    """ The extract process. """

    def __init__(self, arguments):
        self.args = arguments

        self.images = Images(self.args)
        self.faces = Faces(self.args)
        self.alignments = Alignments(self.args)

        self.output_dir = self.faces.output_dir

        self.export_face = True

    def process(self):
        """ Perform the extraction process """
        print('Starting, this may take a while...')
        Utils.set_verbosity(self.args.verbose)

        if hasattr(self.args, 'processes') and self.args.processes > 1:
            self.extract_multi_process()
        else:
            self.extract_single_process()

        self.alignments.write_alignments(self.faces.faces_detected)

        images, faces = Utils.finalize(self.images.images_found,
                                       self.faces.num_faces_detected,
                                       self.faces.verify_output)
        self.images.images_found = images
        self.faces.num_faces_detected = faces

    def extract_single_process(self):
        """ Run extraction in a single process """
        for filename in tqdm(self.images.input_images):
            filename, faces = self.process_single_image(filename)
            self.faces.faces_detected[os.path.basename(filename)] = faces

    def extract_multi_process(self):
        """ Run the extraction on the correct number of processes """
        for filename, faces in tqdm(pool_process(self.process_single_image,
                                                 self.images.input_images,
                                                 processes=self.args.processes),
                                    total=self.images.images_found):
            self.faces.num_faces_detected += 1
            self.faces.faces_detected[os.path.basename(filename)] = faces

    def process_single_image(self, filename):
        """ Detect faces in an image. Rotate the image the specified amount
            until at least one face is found, or until image rotations are
            depleted.
            Once at least one face has been detected, pass to process_single_face
            to process the individual faces """
        retval = filename, list()
        try:
            image = Utils.cv2_read_write('read', filename)

            for angle in self.images.rotation_angles:
                image = Utils.rotate_image_by_angle(image, angle)
                faces = self.faces.get_faces(image, angle)
                process_faces = [(idx, face) for idx, face in faces]
                if process_faces and angle != 0 and self.args.verbose:
                    print("found face(s) by rotating image {} degrees".format(angle))
                if process_faces:
                    break

            final_faces = [self.process_single_face(idx, face, filename, image)
                           for idx, face in process_faces]

            retval = filename, final_faces
        except Exception as err:
            if self.args.verbose:
                print("Failed to extract from image: {}. Reason: {}".format(filename, err))
        return retval

    def process_single_face(self, idx, face, filename, image):
        """ Perform processing on found faces """
        output_file = self.output_dir / Path(filename).stem if self.export_face else None

        self.faces.draw_landmarks_on_face(face, image)

        resized_face, t_mat = self.faces.extractor.extract(image,
                                                           face,
                                                           256,
                                                           self.faces.align_eyes)

        blurry_file = self.faces.detect_blurry_faces(face, t_mat, resized_face, filename)
        output_file = blurry_file if blurry_file else output_file

        if self.export_face:
            filename = "{}_{}{}".format(str(output_file), str(idx), Path(filename).suffix)
            Utils.cv2_read_write('write', filename, resized_face)

        return {"r": face.r,
                "x": face.x,
                "w": face.w,
                "y": face.y,
                "h": face.h,
                "landmarksXY": face.landmarksAsXY()}
