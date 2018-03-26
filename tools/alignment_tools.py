''' Tools for manipulating the alignments seralized file '''

#TODO merge alignments
#TODO Identify possible false positives
#TODO Remove whole frames from alignments file

import argparse
import datetime
import os
import sys
import json
import pickle
from tqdm import tqdm

class AlignmentTool(object):
    ''' Main class to process jobs '''

    def __init__(self, subparser, command, description='default'):
        self.set_parser(description, subparser, command)

    def set_parser(self, description, subparser, command):
        ''' Set parent parser and correct subparser '''
        parser = subparser.add_parser(
                command,
                help='This command lets you change an alignments file in '
                      'various ways.',
                description=description,
                epilog='Questions and feedback: \
                        https://github.com/deepfakes/faceswap-playground')
        parser.add_argument('-a', '--alignments_file',
                    type=str,
                    dest='alignments_file',
                    help='Full path to the alignments file to be processed.',
                    required=True)
        subparser = parser.add_subparsers()
        remove = RemoveAlignments(subparser,
                        'remove', 'Remove deleted faces from an alignments file')
        reformat = ReformatAlignments(subparser,
                        'reformat', 'Save a copy of alignments file in a different format')
        frames = CheckFrames(subparser,
                        'frames', 'Check the contents of the alignments file against the '
                        'original frames that the faces were extracted from')

class RemoveAlignments(object):
    ''' Remove items from alignments file '''
    def __init__(self, subparser, command, description='default'):
        self.faces = None
        self.alignment_data = None
        self.count_faces = None
        self.count_items = None
        self.removed = set()

        self.parse_arguments(description, subparser, command)
        
    def parse_arguments(self, description, subparser, command):
        parser = subparser.add_parser(
                command,
                help='This command removes items from an alignments file that do not appear in '
                    'the aligned faces folder.',
                description=description)
        parser.add_argument('-f', '--faces_folder',
                            type=str,
                            dest='faces_dir',
                            help='Input directory of source A extracted faces.',
                            required=True)
        parser.add_argument('-d', '--destination_format',
                            type=str,
                            choices=('json', 'pickle', 'yaml'),
                            dest='destination_format',
                            help='The file format to save the serialized data in. '
                                  'Defaults to same as source.',
                            default=None)  
        parser.set_defaults(func=self.process)

    def process(self, arguments):
        ''' process the arguments and run removal '''
        align_file = arguments.alignments_file
        dst_format = arguments.destination_format
        self.faces = Faces(arguments.faces_dir)
        self.alignment_data = AlignmentData(align_file, dst_format)
        self.count_items = len(self.alignment_data.alignments)
        self.count_faces = len(self.faces.file_list_sorted)
        self.remove_alignment()

    def remove_alignment(self):
        ''' Remove the alignment from the alignments file '''
        del_count = 0
        for item in tqdm(self.alignment_data.get_alignments_one_image(),
                         desc='Processing alignments file',
                         total=self.count_items):
            image_name, alignments, number_alignments = item
            number_faces = len(self.faces.faces.get(image_name,[]))
            if number_alignments == 0 or number_alignments == number_faces:
                continue
            for idx in self.alignment_data.get_one_alignment_index_reverse(alignments, 
                                                                           number_alignments):
                face_indexes = self.faces.faces.get(image_name,[-1]) 
                if idx not in face_indexes:
                    del alignments[idx]
                    self.removed.add(image_name)
                    del_count += 1
        if del_count == 0:
            print('No changes made to alignments file. Exiting')
            return
        print('{} alignments(s) were removed from alignments file\n'.format(del_count))
        self.alignment_data.save_alignments()
        self.rename_faces()

    def rename_faces(self):
        ''' Rename the aligned faces to match their new index in alignments file '''
        current_image = ''
        current_index = 0
        rename_count = 0
        for item in tqdm(self.faces.file_list_sorted,
                         desc='Renaming aligned faces',
                         total=self.count_faces):
            filename, extension, original_file, index = item
            if original_file in self.removed:
                if current_image != original_file:
                    current_index = 0
                current_image = original_file
                if current_index != index:
                    old_file = filename + extension
                    new_file = current_image + '_' + current_index + extension
                    src = os.path.join(self.faces.faces_dir, old_file)
                    dst = os.path.join(self.faces.faces_dir, new_file)
                    os.rename(src, dst)
                    rename_count += 1
                current_index += 1
        if rename_count == 0:
            print('No files were renamed. Exiting')
            return
        print('{} face(s) were renamed to match with alignments file'.format(rename_count))

class ReformatAlignments(object):
    ''' Reformat items from alignments file '''
    def __init__(self, subparser, command, description='default'):
        self.alignment_data = None

        self.parse_arguments(description, subparser, command)
        
    def parse_arguments(self, description, subparser, command):
        parser = subparser.add_parser(
                command,
                help="This command saves the alignments file in the specified format",
                description=description)
        parser.add_argument('-d', '--destination_format',
                            type=str,
                            choices=('json', 'pickle', 'yaml'),
                            dest='destination_format',
                            help='The file format to save the serialized data in.',
                            required=True)  
        parser.set_defaults(func=self.process)

    def process(self, arguments):
        ''' process the arguments and run reformat '''
        align_file = arguments.alignments_file
        dst_format = arguments.destination_format
        self.alignment_data = AlignmentData(align_file, dst_format)
        self.alignment_data.save_alignments()

class CheckFrames(object):
    ''' Check original Frames against alignments file '''
    def __init__(self, subparser, command, description='default'):
        self.arguments = None
        self.alignments = None
        self.frames = None
        self.frames_output = []
        self.frames_discovered = 0
        self.output_message = ''

        self.parse_arguments(description, subparser, command)

    def parse_arguments(self, description, subparser, command):
        parser = subparser.add_parser(
                command,
                help='This command checks the frames directory against the alignments file',
                description=description,
                formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('-f', '--frames_folder',
                            type=str,
                            dest='frames_dir',
                            help='The folder containing the source frames that faces were '
                                'extracted from',
                            required=True)
        parser.add_argument('-t', '--type',
                            type=str,
                            choices=('missing-alignments', 'missing-frames', 'no-faces', 
                                     'multi-faces'),
                            dest='type',
                            help='The type of testing to be performed:\n'
                                'missing-alignments: Identify frames that do not exist in the '
                                'alignments file.\n'
                                'missing-frames: Identify frames in the alignments file that do '
                                'not appear within the frames directory.\n'
                                'no-faces: Identify frames that exist within the alignment file '
                                'but no faces were discovered.\n'
                                'multi-faces: Identify frames where multiple faces exist within '
                                'the alignments file.',
                            required=True)
        parser.add_argument('-o', '--output',
                            type=str,
                            choices=('console', 'file', 'move'),
                            dest='output',
                            help='The output type of the discovered frames:\n'
                                'console: Print the list of frames to the screen. (DEFAULT)\n'
                                'file: Output the list of frames to a text file (stored within '
                                'the source frames directory).\n'
                                'move: Move the discovered frames to a sub-folder within the '
                                'source frames directory.',
                            default='console')
                            
        parser.set_defaults(func=self.process_arguments)

    def process_arguments(self, arguments):
        ''' Process the arguments '''
        self.arguments = arguments
        if self.arguments.type == 'missing-frames' and self.arguments.output == 'move':
            print('WARNING: missing_frames was selected with move output, but there will be '
                'nothing to move. Defaulting to output: console\n')
            self.arguments.output = 'console'
        self.process()

    def process(self):
        ''' Process the frames check against the alignments file '''
        alignments = AlignmentData(self.arguments.alignments_file).alignments
        self.alignments = {key: len(value) for key, value in alignments.items()}
        self.frames = Frames(self.arguments.frames_dir).frames
        self.compile_frames_output()
        self.output_results()

    def compile_frames_output(self):
        ''' Compile list of frames that meet criteria '''
        if self.arguments.type == 'no-faces': 
            processor = self.get_no_faces
            self.output_message = 'Frames with no faces'
        elif self.arguments.type == 'multi-faces': 
            processor = self.get_multiple_faces
            self.output_message = 'Frames with multiple faces'
        elif self.arguments.type == 'missing-alignments':
            processor = self.get_missing_alignments
            self.output_message = 'Frames missing from alignments file'
        elif self.arguments.type == 'missing-frames':
            processor = self.get_missing_frames
            self.output_message = 'Missing frames that are in alignments file'
        self.frames_output = [frame for frame in processor()]

    def get_no_faces(self):
        ''' yield each frame that has no face match in alignments file '''
        for frame in self.frames:
            if self.alignments.get(frame, -1) == 0:
                yield frame

    def get_multiple_faces (self):
        ''' yield each frame that has multiple faces matched in alignments file '''
        for frame in self.frames:
            if self.alignments.get(frame, -1) > 1:
                yield frame

    def get_missing_alignments(self):
        ''' yield each frame that does not exist in alignments file '''
        exclude_filetypes = ['yaml', 'yml', 'p', 'json', 'txt']
        for frame in self.frames:
            extension = frame[frame.rindex('.') + 1:]
            if extension not in exclude_filetypes and self.alignments.get(frame, -1) == -1:
                yield frame

    def get_missing_frames(self):
        ''' yield each frame in alignments that does not have a matching file '''
        for frame in self.alignments.keys():
            if frame not in self.frames:
                yield frame

    def output_results(self):
        ''' Output the results in the requested format '''
        self.frames_discovered = len(self.frames_output)
        if self.frames_discovered == 0:
            print('\nNo frames were found meeting the criteria')
            return
        output_message = '---------------------------------------------------\r\n'
        output_message += ' {} ({})\r\n'.format(self.output_message, self.frames_discovered)
        output_message += '---------------------------------------------------\r\n'
        if self.arguments.output == 'move':
            self.move_file(output_message)
        output_message += '\r\n'.join([frame for frame in self.frames_output])
        if self.arguments.output == 'console':
            print('\n' + output_message)
        if self.arguments.output == 'file':
            self.output_file(output_message)

    def output_file(self, output_message):
        ''' Save the output to a text file in the frames directory '''
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_message.replace(' ','_').lower()
        filename += '_' + now + '.txt'
        output_file = os.path.join(self.arguments.frames_dir, filename)
        print ('Saving {} result(s) to {}'.format(self.frames_discovered, output_file))
        with open(output_file, 'w') as f_output:
            f_output.write(output_message)

    def move_file(self, output_message):
        ''' Move the identified frames to a new subfolder '''
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = self.output_message.replace(' ','_').lower()
        folder_name +=  '_' + now
        output_folder = os.path.join(self.arguments.frames_dir, folder_name)
        os.makedirs(output_folder)
        print ('Moving {} frame(s) to {}'.format(self.frames_discovered, output_folder))
        for frame in self.frames_output:
            src = os.path.join(self.arguments.frames_dir, frame)
            dst = os.path.join(output_folder, frame)
            os.rename(src, dst)

class AlignmentData(object):
    ''' Class to hold the alignment data '''

    def __init__(self, alignments_file, destination_format=None):
        self.extensions = ['json', 'p', 'yaml', 'yml']
        self.alignments_file = alignments_file
        self.check_alignments_file_exists()
        self.alignments_format = self.get_alignments_format()
        self.destination_format = self.get_destination_format(destination_format)
        self.serializer = self.set_serializer()
        self.alignments = self.load_alignments()

        self.set_destination_serializer()

    @staticmethod
    def handle_yaml():
        ''' If YAML is requested but PyYAML is not installed exit gracefully with message, 
            otherwise pass module through '''
        try:
            import yaml
            return yaml
        except ImportError:
            print('You must have PyYAML installed to use YAMLSerializer')
            sys.exit()

    def check_alignments_file_exists(self):
        ''' Check the alignments file exists'''
        if not os.path.isfile(self.alignments_file):
            print('ERROR: alignments file not found at: {}'.format(self.alignments_file))
            sys.exit()

    def get_alignments_format(self):
        ''' Return the extension of the input file to get the format '''
        return os.path.splitext(self.alignments_file)[1][1:]

    def get_destination_format(self, destination_format):
        ''' Standardise the destination format to the correct extension '''
        if destination_format is None :
            return self.alignments_format
        elif destination_format == 'json':
            return 'json'
        elif destination_format == 'pickle':
            return 'p'
        elif destination_format == 'yaml':
            return 'yml'
        else:
            print('{} is not a supported serializer. Exiting'.format(destination_format))
            sys.exit()        

    def set_serializer(self):
        ''' Set the serializer '''
        read_opts = 'r'
        write_opts = 'w'
        serializer_opts = {}
        if self.alignments_format == 'json':
            serializer_opts = {'indent': 2}
            serializer = json
        elif self.alignments_format == 'p':
            read_opts = 'rb'
            write_opts = 'wb'
            serializer = pickle
        elif self.alignments_format in ['yml', 'yaml']:
            serializer_opts = {'default_flow_style': False}
            serializer = self.handle_yaml()
        else:
            print('{} is not a supported serializer. Exiting'.format(self.alignments_format))
            sys.exit()
        return {'serializer': serializer,
                'serializer_opts': serializer_opts,
                'read_opts': read_opts,
                'write_opts': write_opts}

    def set_destination_serializer(self):
        ''' set the destination serializer if it differs from the import serializer '''
        if self.destination_format != self.alignments_format:
            self.alignments_format = self.destination_format
            self.serializer = self.set_serializer()

    def load_alignments(self):
        ''' Read the alignments data from the correct format '''
        serializer = self.serializer['serializer']
        read_opts = self.serializer['read_opts']
        fmt = self.alignments_format
        align_file = self.alignments_file

        print('Loading {} alignments from {}'.format(fmt, align_file))
        with open(align_file, read_opts) as f_alignments:
            if fmt not in ('yaml', 'yml'):
                alignments = serializer.loads(f_alignments.read())
            else:
                alignments = serializer.load(f_alignments.read())
        return alignments

    def backup_alignments(self):
        ''' Backup copy of old alignments '''
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        src = self.alignments_file
        dst = src.split('.')
        dst[0] += '_' + now + '.'
        dst = dst[0] + dst[1]
        print('Backing up original alignments to {}'.format(dst))
        os.rename(src, dst)

    def save_alignments(self):
        ''' Backup copy of old alignments and save new alignments '''
        serializer = self.serializer['serializer']
        write_opts = self.serializer['write_opts']
        serializer_opts = self.serializer['serializer_opts']
        dst = self.alignments_file.split('.')
        dst  = dst[0] + '.' + self.alignments_format

        self.backup_alignments()

        print('Saving alignments to {}\n'.format(dst))
        with open(dst, write_opts) as f_alignments:
            if self.alignments_format not in ('yaml', 'yml'):
                f_alignments.write(serializer.dumps(self.alignments, **serializer_opts))
            else:
                f_alignments.write(serializer.dump(self.alignments, **serializer_opts))

    def get_alignments_one_image(self):
        ''' Return the face alignments for one image '''
        for image, alignments in self.alignments.items():
            image_stripped = image[:image.rindex('.')]
            number_alignments = len(alignments)
            yield image_stripped, alignments, number_alignments

    @staticmethod
    def get_one_alignment_index_reverse(image_alignments, number_alignments):
        ''' Return the correct original index for alignment in reverse order '''
        for idx, alignment in enumerate(reversed(image_alignments)):
            original_idx = number_alignments - 1 - idx
            yield original_idx

class Faces(object):
    ''' Object to hold the faces that are to be swapped out '''
    def __init__(self, faces_dir):
        self.faces_dir = faces_dir
        self.check_folder_exists()
        self.file_list_sorted = sorted([item for item in self.process_faces_dir()])
        self.faces = {}
        self.load_faces()

    def check_folder_exists(self):
        ''' makes sure that the faces folder exists '''
        if not os.path.isdir(self.faces_dir):
            print('ERROR: The folder {} could not be found'.format(self.faces_dir))
            sys.exit()
            
    def process_faces_dir(self):
        ''' Iterate through the faces dir pulling out various information '''
        print('Loading file list from {}'.format(self.faces_dir))
        for face in os.listdir(self.faces_dir):
            face_filename = os.path.splitext(face)[0]
            file_extension = os.path.splitext(face)[1]
            index = int(face_filename[face_filename.rindex('_') + 1:])
            original_file = '{}'.format(face_filename[:face_filename.rindex('_')])
            yield (face_filename, file_extension, original_file, index)
        
    def load_faces(self):
        ''' Load the face names into dictionary '''
        for item in self.file_list_sorted:
            original_file, index = item[2:4]
            if self.faces.get(original_file, '') == '':
                self.faces[original_file] = [index]
            else:
                self.faces[original_file].append(index)

class Frames(object):
    ''' Object to hold the frames that are to be checked against '''
    def __init__(self, frames_dir):
        self.frames_dir = frames_dir
        self.check_folder_exists()
        self.frames = sorted([item for item in self.process_frames_dir()])

    def check_folder_exists(self):
        ''' makes sure that the frames folder exists '''
        if not os.path.isdir(self.frames_dir):
            print('ERROR: The folder {} could not be found'.format(self.frames_dir))
            sys.exit()
            
    def process_frames_dir(self):
        ''' Iterate through the frames dir pulling the base filename '''
        print('Loading file list from {}'.format(self.frames_dir))
        for frame in os.listdir(self.frames_dir):
            frame_filename = os.path.basename(frame)
            yield (frame_filename)
        
def bad_args(args):
    parser.print_help()
    exit(0)

if __name__ == '__main__':
    print ('Faceswap Alignments File Helper Tool.\n')
   
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers()
    alignments = AlignmentTool(subparser,
                         'alignments', 'Perform various edits to an alignments file.')
    parser.set_defaults(func=bad_args)
    arguments = parser.parse_args()
    arguments.func(arguments)
