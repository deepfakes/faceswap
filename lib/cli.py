import argparse
import os
import sys
import time

from pathlib import Path
from lib.FaceFilter import FaceFilter
from lib.faces_detect import detect_faces, DetectedFace
from lib.utils import get_image_paths, get_folder, rotate_image
from lib import Serializer

class FullPaths(argparse.Action):
    """Expand user- and relative-paths"""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, os.path.abspath(
            os.path.expanduser(values)))

class FullHelpArgumentParser(argparse.ArgumentParser):
    """
    Identical to the built-in argument parser, but on error
    it prints full help message instead of just usage information
    """
    def error(self, message):
        self.print_help(sys.stderr)
        args = {'prog': self.prog, 'message': message}
        self.exit(2, '%(prog)s: error: %(message)s\n' % args)

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
    rotation_angles = None

    def __init__(self, subparser, command, description='default'):
        self.create_parser(subparser, command, description)
        self.parse_arguments(description, subparser, command)

    def process_arguments(self, arguments):
        self.arguments = arguments
        print("Input Directory: {}".format(self.arguments.input_dir))
        print("Output Directory: {}".format(self.arguments.output_dir))
        print("Filter: {}".format(self.arguments.filter))
        self.serializer = None
        if self.arguments.serializer is None and self.arguments.alignments_path is not None:
            ext = os.path.splitext(self.arguments.alignments_path)[-1]
            self.serializer = Serializer.get_serializer_fromext(ext)
            print(self.serializer, self.arguments.alignments_path)
        else:
            self.serializer = Serializer.get_serializer(self.arguments.serializer or "json")
        print("Using {} serializer".format(self.serializer.ext))

        try:
            if self.arguments.rotate_images is not None and self.arguments.rotate_images != "off":
                if self.arguments.rotate_images == "on":
                    self.rotation_angles = range(90, 360, 90)
                else:
                    rotation_angles = [int(angle) for angle in self.arguments.rotate_images.split(",")]
                    if len(rotation_angles) == 1:
                        rotation_step_size = rotation_angles[0]
                        self.rotation_angles = range(rotation_step_size, 360, rotation_step_size)
                    elif len(rotation_angles) > 1:
                        self.rotation_angles = rotation_angles
        except AttributeError:
            pass

        print('Starting, this may take a while...')

        try:
            if self.arguments.skip_existing:
                self.already_processed = get_image_paths(self.arguments.output_dir)
        except AttributeError:
            pass
    
        self.output_dir = get_folder(self.arguments.output_dir)

        try:
            try:
                if self.arguments.skip_existing:
                    self.input_dir = get_image_paths(self.arguments.input_dir, self.already_processed)
                    print('Excluding %s files' % len(self.already_processed))
                else:
                    self.input_dir = get_image_paths(self.arguments.input_dir)
            except AttributeError:
                self.input_dir = get_image_paths(self.arguments.input_dir)
        except:
            print('Input directory not found. Please ensure it exists.')
            exit(1)

        self.filter = self.load_filter()
        self.process()
        self.finalize()

    def read_alignments(self):

        fn = os.path.join(str(self.arguments.input_dir),"alignments.{}".format(self.serializer.ext))
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

        fn = os.path.join(str(self.arguments.input_dir), "alignments.{}".format(self.serializer.ext))
        if self.arguments.alignments_path is not None:
            fn = self.arguments.alignments_path
        print("Alignments filepath: %s" % fn)
        
        if self.arguments.skip_existing:
            if os.path.exists(fn):
                with open(fn, self.serializer.roptions) as inf:
                    data = self.serializer.unmarshal(inf.read())
                    for k, v in data.items():
                        self.faces_detected[k] = v
            else:
                print('Existing alignments file "%s" not found.' % fn)
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
        return os.path.basename(filename) in self.faces_detected

    def have_alignments(self):
        fn = os.path.join(str(self.arguments.input_dir), "alignments.{}".format(self.serializer.ext))
        return os.path.exists(fn)

    def get_faces_alignments(self, filename, image):
        faces_count = 0
        faces = self.faces_detected[os.path.basename(filename)]
        for rawface in faces:
            face = DetectedFace(**rawface)
            # Rotate the image if necessary
            if face.r != 0: image = rotate_image(image, face.r)
            face.image = image[face.y : face.y + face.h, face.x : face.x + face.w]
            if self.filter is not None and not self.filter.check(face):
                if self.arguments.verbose:
                    print('Skipping not recognized face!')
                continue

            yield faces_count, face
            self.num_faces_detected += 1
            faces_count += 1
        if faces_count > 1 and self.arguments.verbose:
            print('Note: Found more than one face in an image! File: %s' % filename)
            self.verify_output = True

    def get_faces(self, image, rotation=0):
        faces_count = 0
        faces = detect_faces(image, self.arguments.detector, self.arguments.verbose, rotation)
        
        for face in faces:
            if self.filter is not None and not self.filter.check(face):
                if self.arguments.verbose:
                    print('Skipping not recognized face!')
                continue
            yield faces_count, face

            self.num_faces_detected += 1
            faces_count += 1

        if faces_count > 1 and self.arguments.verbose:
            self.verify_output = True

    def load_filter(self):
        nfilter_files = self.arguments.nfilter
        if not isinstance(self.arguments.nfilter, list):
            nfilter_files = [self.arguments.nfilter]
        nfilter_files = list(filter(lambda fn: Path(fn).exists(), nfilter_files))

        filter_files = self.arguments.filter
        if not isinstance(self.arguments.filter, list):
            filter_files = [self.arguments.filter]
        filter_files = list(filter(lambda fn: Path(fn).exists(), filter_files))
        
        if filter_files:
            print('Loading reference images for filtering: %s' % filter_files)
            return FaceFilter(filter_files, nfilter_files, self.arguments.ref_threshold)

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
