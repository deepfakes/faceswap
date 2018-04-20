#!/usr/bin python3
import argparse
import os
import sys
import time

from pathlib import Path
from lib.utils import get_image_paths, get_folder, rotate_image
from lib import Serializer

# DLIB is a GPU Memory hog, so the following modules should only be imported
#  when required
detect_faces = None
DetectedFace = None
FaceFilter = None


def import_faces_detect():
    """ Import the faces_detect module only when it is required """
    global detect_faces
    global DetectedFace
    if detect_faces is None or DetectedFace is None:
        import lib.faces_detect
        detect_faces = lib.faces_detect.detect_faces
        DetectedFace = lib.faces_detect.DetectedFace


def import_FaceFilter():
    """ Import the FaceFilter module only when it is required """
    global FaceFilter
    if FaceFilter is None:
        import lib.FaceFilter
        FaceFilter = lib.FaceFilter.FaceFilter


class FullPaths(argparse.Action):
    """Expand user- and relative-paths"""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, os.path.abspath(
                os.path.expanduser(values)))


class DirFullPaths(FullPaths):
    """ Class that gui uses to determine if you need to open a directory """
    pass


class FileFullPaths(FullPaths):
    """
    Class that gui uses to determine if you need to open a file.

    Filetypes added as an argparse argument must be an iterable, i.e. a
    list of lists, tuple of tuples, list of tuples etc... formatted like so:
        [("File Type", ["*.ext", "*.extension"])]
    A more realistic example:
        [("Video File", ["*.mkv", "mp4", "webm"])]

    If the file extensions are not prepended with '*.', use the
    prep_filetypes() method to format them in the arguments_list.
    """
    def __init__(self, option_strings, dest, nargs=None, filetypes=None,
                 **kwargs):
        super(FileFullPaths, self).__init__(option_strings, dest, **kwargs)
        if nargs is not None:
            raise ValueError("nargs not allowed")
        self.filetypes = filetypes

    @staticmethod
    def prep_filetypes(filetypes):
        all_files = ("All Files", "*.*")
        filetypes_l = list()
        for i in range(len(filetypes)):
            filetypes_l.append(FileFullPaths._process_filetypes(filetypes[i]))
        filetypes_l.append(all_files)
        return tuple(filetypes_l)

    @staticmethod
    def _process_filetypes(filetypes):
        """        """
        if filetypes is None:
            return None

        filetypes_name = filetypes[0]
        filetypes_l = filetypes[1]
        if (type(filetypes_l) == list or type(filetypes_l) == tuple) \
                and all("*." in i for i in filetypes_l):
            return filetypes  # assume filetypes properly formatted

        if type(filetypes_l) != list and type(filetypes_l) != tuple:
            raise ValueError("The filetypes extensions list was "
                             "neither a list nor a tuple: "
                             "{}".format(filetypes_l))

        filetypes_list = list()
        for i in range(len(filetypes_l)):
            filetype = filetypes_l[i].strip("*.")
            filetype = filetype.strip(';')
            filetypes_list.append("*." + filetype)
        return filetypes_name, filetypes_list

    def _get_kwargs(self):
        names = [
            'option_strings',
            'dest',
            'nargs',
            'const',
            'default',
            'type',
            'choices',
            'help',
            'metavar',
            'filetypes'
        ]
        return [(name, getattr(self, name)) for name in names]


class ComboFullPaths(FileFullPaths):
    """
    Class that gui uses to determine if you need to open a file or a
    directory based on which action you are choosing
    """
    def __init__(self,  option_strings, dest, nargs=None, filetypes=None,
                 actions_open_type=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(ComboFullPaths, self).__init__(option_strings, dest,
                                             filetypes=None, **kwargs)

        self.actions_open_type = actions_open_type
        self.filetypes = filetypes

    @staticmethod
    def prep_filetypes(filetypes):
        all_files = ("All Files", "*.*")
        filetypes_d = dict()
        for k, v in filetypes.items():
            filetypes_d[k] = ()
            if v is None:
                filetypes_d[k] = None
                continue
            filetypes_l = list()
            for i in range(len(v)):
                filetypes_l.append(ComboFullPaths._process_filetypes(v[i]))
            filetypes_d[k] = (tuple(filetypes_l), all_files)
        return filetypes_d

    def _get_kwargs(self):
        names = [
            'option_strings',
            'dest',
            'nargs',
            'const',
            'default',
            'type',
            'choices',
            'help',
            'metavar',
            'filetypes',
            'actions_open_type'
        ]
        return [(name, getattr(self, name)) for name in names]


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
    """
    Abstract class that processes a directory of images
    and writes output to the specified folder
    """
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
        self.argument_list = self.get_argument_list()
        self.optional_arguments = self.get_optional_arguments()
        self.create_parser(subparser, command, description)
        self.parse_arguments(description, subparser, command)

    @staticmethod
    def get_argument_list():
        """ Put the arguments in a list so that they are accessible from
        both argparse and gui """
        argument_list = list()
        argument_list.append({"opts": ('-i', '--input-dir'),
                              "action": DirFullPaths,
                              "dest": "input_dir",
                              "default": "input",
                              "help": "Input directory. A directory "
                                      "containing the files \
                                you wish to process. Defaults to 'input'"})
        argument_list.append({"opts": ('-o', '--output-dir'),
                              "action": DirFullPaths,
                              "dest": "output_dir",
                              "default": "output",
                              "help": "Output directory. This is where the "
                                      "converted files will \
                                be stored. Defaults to 'output'"})
        argument_list.append({"opts": ('--serializer',),
                              "type": str.lower,
                              "dest": "serializer",
                              "choices": ("yaml", "json", "pickle"),
                              "help": "serializer for alignments file"})
        argument_list.append({"opts": ('--alignments',),
                              "action": FileFullPaths,
                              "type": str,
                              "dest": "alignments_path",
                              "help": "optional path to alignments file."})
        argument_list.append({"opts": ('-v', '--verbose'),
                              "action": "store_true",
                              "dest": "verbose",
                              "default": False,
                              "help": "Show verbose output"})
        return argument_list

    @staticmethod
    def get_optional_arguments():
        """ Put the arguments in a list so that they are accessible from
        both argparse and gui """
        # Override this for custom arguments
        argument_list = []
        return argument_list

    def process_arguments(self, arguments):
        self.arguments = arguments
        print("Input Directory: {}".format(self.arguments.input_dir))
        print("Output Directory: {}".format(self.arguments.output_dir))
        print("Filter: {}".format(self.arguments.filter))
        self.serializer = None
        if self.arguments.serializer is None and \
                self.arguments.alignments_path is not None:
            ext = os.path.splitext(self.arguments.alignments_path)[-1]
            self.serializer = Serializer.get_serializer_fromext(ext)
            print(self.serializer, self.arguments.alignments_path)
        else:
            self.serializer = Serializer.get_serializer(
                self.arguments.serializer or "json")
        print("Using {} serializer".format(self.serializer.ext))

        try:
            if self.arguments.rotate_images is not None and \
                    self.arguments.rotate_images != "off":
                if self.arguments.rotate_images == "on":
                    self.rotation_angles = range(90, 360, 90)
                else:
                    rotation_angles = [int(angle) for angle in
                                       self.arguments.rotate_images.split(",")]
                    if len(rotation_angles) == 1:
                        rotation_step_size = rotation_angles[0]
                        self.rotation_angles = range(rotation_step_size, 360,
                                                     rotation_step_size)
                    elif len(rotation_angles) > 1:
                        self.rotation_angles = rotation_angles
        except AttributeError:
            pass

        print('Starting, this may take a while...')

        try:
            if self.arguments.skip_existing:
                self.already_processed = get_image_paths(
                    self.arguments.output_dir)
        except AttributeError:
            pass

        self.output_dir = get_folder(self.arguments.output_dir)

        try:
            try:
                if self.arguments.skip_existing:
                    self.input_dir = get_image_paths(self.arguments.input_dir,
                                                     self.already_processed)
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

        fn = os.path.join(str(self.arguments.input_dir),
                          "alignments.{}".format(self.serializer.ext))
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

        fn = os.path.join(str(self.arguments.input_dir),
                          "alignments.{}".format(self.serializer.ext))
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
        fn = os.path.join(str(self.arguments.input_dir),
                          "alignments.{}".format(self.serializer.ext))
        return os.path.exists(fn)

    def get_faces_alignments(self, filename, image):
        import_faces_detect()
        faces_count = 0
        faces = self.faces_detected[os.path.basename(filename)]
        for rawface in faces:
            face = DetectedFace(**rawface)
            # Rotate the image if necessary
            if face.r != 0: image = rotate_image(image, face.r)
            face.image = image[face.y: face.y + face.h,
                         face.x: face.x + face.w]
            if self.filter is not None and not self.filter.check(face):
                if self.arguments.verbose:
                    print('Skipping not recognized face!')
                continue

            yield faces_count, face
            self.num_faces_detected += 1
            faces_count += 1
        if faces_count > 1 and self.arguments.verbose:
            print(
                'Note: Found more than one face in an image! File: %s' %
                filename)
            self.verify_output = True

    def get_faces(self, image, rotation=0):
        import_faces_detect()
        faces_count = 0
        faces = detect_faces(image, self.arguments.detector,
                             self.arguments.verbose, rotation)

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
        nfilter_files = list(
            filter(lambda fn: Path(fn).exists(), nfilter_files))

        filter_files = self.arguments.filter
        if not isinstance(self.arguments.filter, list):
            filter_files = [self.arguments.filter]
        filter_files = list(filter(lambda fn: Path(fn).exists(), filter_files))

        if filter_files:
            import_FaceFilter()
            print('Loading reference images for filtering: %s' % filter_files)
            return FaceFilter(filter_files, nfilter_files,
                              self.arguments.ref_threshold)

    # for now, we limit this class responsability to the read of files.
    # images and faces are processed outside this class
    def process(self):
        # implement your image processing!
        raise NotImplementedError()

    def parse_arguments(self, description, subparser, command):
        for option in self.argument_list:
            args = option['opts']
            kwargs = {key: option[key] for key in option.keys() if
                      key != 'opts'}
            self.parser.add_argument(*args, **kwargs)

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
        for option in self.optional_arguments:
            args = option['opts']
            kwargs = {key: option[key] for key in option.keys() if
                      key != 'opts'}
            parser.add_argument(*args, **kwargs)
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
