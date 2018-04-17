#!/usr/bin python3
""" The script to run the extract process of faceswap """

import os

from tqdm import tqdm

from lib.cli import DirectoryArgs, FSProcess
from lib.multithreading import pool_process

class ExtractArgs(DirectoryArgs):
    """ Class to parse the command line arguments for extraction.
        Inherits base options from lib.DirectoryArgs """

    @staticmethod
    def get_optional_arguments():
        """ Put the arguments in a list so that they are accessible from both argparse and gui """
        argument_list = []
        argument_list.append({"opts": ("-D", "--detector"),
                              "type": str,
                              # case sensitive because this is used to load a plugin.
                              "choices": ("hog", "cnn", "all"),
                              "default": "hog",
                              "help": "Detector to use. 'cnn' detects much more angles but will "
                                      "be much more resource intensive and may fail on large "
                                      "files."})
        argument_list.append({"opts": ("-l", "--ref_threshold"),
                              "type": float,
                              "dest": "ref_threshold",
                              "default": 0.6,
                              "help": "Threshold for positive face recognition"})
        argument_list.append({"opts": ("-n", "--nfilter"),
                              "type": str,
                              "dest": "nfilter",
                              "nargs": "+",
                              "default": "nfilter.jpg",
                              "help": "Reference image for the persons you do not want to "
                                      "process. Should be a front portrait"})
        argument_list.append({"opts": ("-f", "--filter"),
                              "type": str,
                              "dest": "filter",
                              "nargs": "+",
                              "default": "filter.jpg",
                              "help": "Reference image for the person you want to process. "
                                      "Should be a front portrait"})
        argument_list.append({"opts": ("-j", "--processes"),
                              "type": int,
                              "default": 1,
                              "help": "Number of processes to use."})
        argument_list.append({"opts": ("-s", "--skip-existing"),
                              "action": "store_true",
                              "dest": "skip_existing",
                              "default": False,
                              "help": "Skips frames already extracted."})
        argument_list.append({"opts": ("-dl", "--debug-landmarks"),
                              "action": "store_true",
                              "dest": "debug_landmarks",
                              "default": False,
                              "help": "Draw landmarks for debug."})
        argument_list.append({"opts": ("-r", "--rotate-images"),
                              "type": str,
                              "dest": "rotate_images",
                              "default": None,
                              "help": "If a face isn't found, rotate the images to try to "
                                      "find a face. Can find more faces at the cost of extraction "
                                      "speed. Pass in a single number to use increments of that "
                                      "size up to 360, or pass in a list of numbers to enumerate "
                                      "exactly what angles to check."})
        argument_list.append({"opts": ("-ae", "--align-eyes"),
                              "action": "store_true",
                              "dest": "align_eyes",
                              "default": False,
                              "help": "Perform extra alignment to ensure left/right eyes "
                                      "lie at the same height"})
        argument_list.append({"opts": ("-bt", "--blur-threshold"),
                              "type": int,
                              "dest": "blur_thresh",
                              "default": None,
                              "help": "Automatically discard images blurrier than the specified "
                                      "threshold. Discarded images are moved into a \"blurry\" "
                                      "sub-folder. Lower values allow more blur"})
        return argument_list

    def create_parser(self, subparser, command, description):
        """ Create the extract parser """
        self.optional_arguments = self.get_optional_arguments()
        self.parser = subparser.add_parser(
            command,
            help="Extract the faces from a pictures.",
            description=description,
            epilog="Questions and feedback: \
            https://github.com/deepfakes/faceswap-playground")

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

class ExtractTrainingData(object):
    """ TODO: Change this, it shouldn't be a class.
        It's here to keep compatibility during rewrite """
    def __init__(self, subparser, command, description):
        args = ExtractArgs(subparser, command, description).parser.arguments

        self.process = Extract(args)
        self.process.process()
        self.process.finalize()
