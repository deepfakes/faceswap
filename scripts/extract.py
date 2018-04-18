#!/usr/bin python3
""" The script to run the extract process of faceswap """

import os

from tqdm import tqdm

from lib.cli import DirectoryArgs, FSProcess
from lib.multithreading import pool_process

class ExtractTrainingData(DirectoryArgs):
    """ Class to parse the command line arguments for extraction.
        Inherits base options from lib.DirectoryArgs """

    @staticmethod
    def get_optional_arguments():
        """ Put the arguments in a list so that they are accessible from both argparse and gui """
        argument_list = []
        argument_list.append({"opts": ("-r", "--rotate-images"),
                              "type": str,
                              "dest": "rotate_images",
                              "default": None,
                              "help": "If a face isn't found, rotate the images to try to "
                                      "find a face. Can find more faces at the cost of extraction "
                                      "speed. Pass in a single number to use increments of that "
                                      "size up to 360, or pass in a list of numbers to enumerate "
                                      "exactly what angles to check"})
        argument_list.append({"opts": ("-bt", "--blur-threshold"),
                              "type": int,
                              "dest": "blur_thresh",
                              "default": None,
                              "help": "Automatically discard images blurrier than the specified "
                                      "threshold. Discarded images are moved into a \"blurry\" "
                                      "sub-folder. Lower values allow more blur"})
        argument_list.append({"opts": ("-j", "--processes"),
                              "type": int,
                              "default": 1,
                              "help": "Number of CPU processes to use. WARNING: ONLY USE THIS "
                                      " IF YOU ARE NOT EXTRACTING ON A GPU. Anything above 1 "
                                      " process on a GPU will run out of memory and will crash"})
        argument_list.append({"opts": ("-s", "--skip-existing"),
                              "action": "store_true",
                              "dest": "skip_existing",
                              "default": False,
                              "help": "Skips frames that have already been extracted"})
        argument_list.append({"opts": ("-dl", "--debug-landmarks"),
                              "action": "store_true",
                              "dest": "debug_landmarks",
                              "default": False,
                              "help": "Draw landmarks on the ouput faces for debug"})
        argument_list.append({"opts": ("-ae", "--align-eyes"),
                              "action": "store_true",
                              "dest": "align_eyes",
                              "default": False,
                              "help": "Perform extra alignment to ensure left/right eyes "
                                      "are  at the same height"})
        return argument_list

    def create_parser(self, subparser, command, description):
        """ Create the extract parser """
        self.optional_arguments = self.get_optional_arguments()
        self.process = Extract

        parser = subparser.add_parser(
            command,
            help="Extract the faces from a pictures.",
            description=description,
            epilog="Questions and feedback: \
            https://github.com/deepfakes/faceswap-playground")
        return parser

class Extract(FSProcess):
    """ The extract process. Inherits from cli.FSProcess, including the additional
        classes: images, faces, alignments

        As most of the extract processes are required for Convert in the case when
        there are no alignments available, most of the processes will be found in
        lib.cli.py """

    def __init__(self, arguments):
        FSProcess.__init__(self, arguments)
        self.extractor = self.load_extractor()
        self.export_face = True

    def process(self):
        """ Perform the extraction process """
        processes = self.args.processes

        if processes != 1:
            self.multi_process(processes)
        else:
            self.single_process()

        self.alignments.write_alignments()

    def multi_process(self, processes):
        """ Run extraction in a multiple processes """
        files = list(self.images.read_directory())
        for filename, faces in tqdm(pool_process(self.extract_face_alignments,
                                                 files,
                                                 processes=processes),
                                    total=len(files)):
            self.faces.num_faces_detected += 1
            self.faces.faces_detected[os.path.basename(filename)] = faces

    def single_process(self):
        """ Run extraction in a single process """
        for filename in tqdm(self.images.read_directory()):
            filename, faces = self.extract_face_alignments(filename)
            self.faces.faces_detected[os.path.basename(filename)] = faces
