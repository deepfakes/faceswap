import argparse
import os
import time

from pathlib import Path
from lib.FaceFilter import FaceFilter
from lib.faces_detect import detect_faces, DetectedFace
from lib.utils import get_image_paths, get_folder
from lib import Serializer

class FullPaths(argparse.Action):
    """Expand user- and relative-paths"""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, os.path.abspath(
            os.path.expanduser(values)))

class DirectoryProcessor(object):
    '''
    Abstract class that processes a directory of images
    and writes output to the specified folder
    '''
    arguments = None
    parser = None

    input_dir = None
    output_dir = None

    images_found = 0
    num_faces_detected = 0
    faces_detected = dict()
    verify_output = False

    def __init__(self, subparser, command, description='default'):
        self.create_parser(subparser, command, description)
        self.parse_arguments(description, subparser, command)


    def process_arguments(self, arguments):
        self.arguments = arguments
        print("Input Directory: {}".format(self.arguments.input_dir))
        print("Output Directory: {}".format(self.arguments.output_dir))
        self.serializer = None
        if self.arguments.serializer is None and self.arguments.alignments_path is not None:
            ext = os.path.splitext(self.arguments.alignments_path)[-1]
            self.serializer = Serializer.get_serializer_fromext(ext)
            print(self.serializer, self.arguments.alignments_path)
        else:
            self.serializer = Serializer.get_serializer(self.arguments.serializer or "json")
        print("Using {} serializer".format(self.serializer.ext))

        print('Starting, this may take a while...')

        self.output_dir = get_folder(self.arguments.output_dir)
        try:
            self.input_dir = get_image_paths(self.arguments.input_dir)
        except:
            print('Input directory not found. Please ensure it exists.')
            exit(1)

        self.filter = self.load_filter()
        self.process()
        self.finalize()

    def read_alignments(self):

        fn = os.path.join(self.arguments.input_dir,"alignments.{}".format(self.serializer.ext))
        if self.arguments.alignments_path is not None:
            fn = self.arguments.alignments_path

        try:
            print("Reading alignments from: {}".format(fn))
            with open(fn, self.serializer.roptions) as f:
                self.faces_detected = self.serializer.unmarshal(f.read())
        except Exception as e:
            print("{} not read!".format(fn))
            print(str(e))
            self.faces_detected = dict()

    def write_alignments(self):

        fn = os.path.join(self.output_dir,"alignments.{}".format(self.serializer.ext))
        if self.arguments.alignments_path is not None:
            fn = self.arguments.alignments_path
        try:
            print("Writing alignments to: {}".format(fn))
            with open(fn, self.serializer.woptions) as fh:
                fh.write(self.serializer.marshal(self.faces_detected))
        except Exception as e:
            print("{} not written!".format(fn))
            print(str(e))
            self.faces_detected = dict()

    def read_directory(self):
        self.images_found = len(self.input_dir)
        return self.input_dir

    def have_face(self, filename):
        return filename in self.faces_detected

    def get_faces_alignments(self, filename):
        faces_count = 0
        faces = self.faces_detected[filename]
        for rawface in faces:
            face = DetectedFace(**rawface)
            if self.filter is not None and not self.filter.check(face):
                print('Skipping not recognized face!')
                continue

            yield faces_count, face
            self.num_faces_detected += 1
            faces_count += 1
        if faces_count > 1 and self.arguments.verbose:
            print('Note: Found more than one face in an image!')
            self.verify_output = True

    def get_faces(self, image):
        faces_count = 0
        faces = detect_faces(image, self.arguments.detector)

        for face in faces:
            if self.filter is not None and not self.filter.check(face):
                print('Skipping not recognized face!')
                continue
            yield faces_count, face

            self.num_faces_detected += 1
            faces_count += 1

        if faces_count > 1 and self.arguments.verbose:
            print('Note: Found more than one face in an image!')
            self.verify_output = True

    def load_filter(self):
        filter_file = self.arguments.filter
        if Path(filter_file).exists():
            print('Loading reference image for filtering')
            return FaceFilter(filter_file)

    # for now, we limit this class responsability to the read of files. images and faces are processed outside this class
    def process(self):
        # implement your image processing!
        raise NotImplementedError()

    def parse_arguments(self, description, subparser, command):
        self.parser.add_argument('-i', '--input-dir',
                            action=FullPaths,
                            dest="input_dir",
                            default="input",
                            help="Input directory. A directory containing the files \
                            you wish to process. Defaults to 'input'")
        self.parser.add_argument('-o', '--output-dir',
                            action=FullPaths,
                            dest="output_dir",
                            default="output",
                            help="Output directory. This is where the converted files will \
                                be stored. Defaults to 'output'")

        self.parser.add_argument('--serializer',
                                type=str.lower,
                                dest="serializer",
                                choices=("yaml", "json", "pickle"),
                                help="serializer for alignments file")

        self.parser.add_argument('--alignments',
                                type=str,
                                dest="alignments_path",
                                help="optional path to alignments file.")

        self.parser.add_argument('-v', '--verbose',
                            action="store_true",
                            dest="verbose",
                            default=False,
                            help="Show verbose output")
        self.parser = self.add_optional_arguments(self.parser)
        self.parser.set_defaults(func=self.process_arguments)

    def create_parser(self, subparser, command, description):
        parser = subparser.add_parser(
            command,
            description=description,
            epilog="Questions and feedback: \
            https://github.com/deepfakes/faceswap-playground"
        )
        return parser

    def add_optional_arguments(self, parser):
        # Override this for custom arguments
        return parser

    def finalize(self):
        print('-------------------------')
        print('Images found:        {}'.format(self.images_found))
        print('Faces detected:      {}'.format(self.num_faces_detected))
        print('-------------------------')

        if self.verify_output:
            print('Note:')
            print('Multiple faces were detected in one or more pictures.')
            print('Double check your results.')
            print('-------------------------')
        print('Done!')
