import argparse
import os
import time

from lib.utils import get_image_paths, get_folder, load_images

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

    verify_output = False
    images_found = 0
    images_processed = 0
    faces_detected = 0

    def __init__(self, subparser, command, description='default'):
        self.create_parser(subparser, command, description)
        self.parse_arguments(description, subparser, command)

    def process_arguments(self, arguments):
        self.arguments = arguments
        print("Input Directory: {}".format(self.arguments.input_dir))
        print("Output Directory: {}".format(self.arguments.output_dir))
        print('Starting, this may take a while...')

        self.output_dir = get_folder(self.arguments.output_dir)
        try:
            self.input_dir = get_image_paths(self.arguments.input_dir)
        except:
            print('Input directory not found. Please ensure it exists.')
            exit(1)

        self.images_found = len(self.input_dir)

        self.process_directory()

    def process_directory(self):
        for filename in self.input_dir:
            if self.arguments.verbose:
                print('Processing: {}'.format(os.path.basename(filename)))

            self.process_image(filename)
            self.images_processed = self.images_processed + 1

        self.finalize()

    # for now, we limit this class responsability to the read of files. images and faces are processed outside this class
    def process_image(self, filename):
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
        print('Images processed:    {}'.format(self.images_processed))
        print('Faces detected:      {}'.format(self.faces_detected))
        print('-------------------------')

        if self.verify_output:
            print('Note:')
            print('Multiple faces were detected in one or more pictures.')
            print('Double check your results.')
            print('-------------------------')
        print('Done!')
