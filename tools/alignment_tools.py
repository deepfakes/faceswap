''' Tools for manipulating the alignments seralized file '''

#TODO merge alignments
#TODO move/identify frames that are not in alignments file
#TODO Identify frames with multiple/no faces in them

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
                help="This command lets you change an alignments file in\
                      various ways.",
                description=description,
                epilog="Questions and feedback: \
                        https://github.com/deepfakes/faceswap-playground")
        subparser = parser.add_subparsers()
        remove = RemoveAlignments(subparser,
                         'remove', 'Remove deleted faces from an alignments file')
        reformat = ReformatAlignments(subparser,
                         'reformat', 'Save a copy of alignments file in a different format')

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
                help='This command removes items from an alignments file that do not \
                      appear in the  aligned faces folder.',
                description=description)
        parser.add_argument('-a', '--alignments_file',
                            type=str,
                            dest='alignments_file',
                            help='Full path to the alignments file to be processed.',
                            required=True)
        parser.add_argument('-f', '--faces_folder',
                            type=str,
                            dest='faces_dir',
                            help='Input directory of source A extracted faces.',
                            required=True)
        parser.add_argument('-d', '--destination_format',
                            type=str,
                            choices=('json', 'pickle', 'yaml'),
                            dest='destination_format',
                            help='The file format to save the serialized data in. \
                                  Defaults to same as source.',
                            default=None)  
        parser.set_defaults(func=self.process)

    def process(self, arguments):
        ''' process the arguments and run removal '''
        self.faces = Faces(arguments.faces_dir)
        self.alignment_data = AlignmentData(arguments.alignments_file, arguments.destination_format)
        self.count_items = len(self.alignment_data.alignments)
        self.count_faces = len(self.faces.file_list_sorted)
        self.remove_alignment()

    def remove_alignment(self):
        ''' Remove the alignment from the alignments file '''
        del_count = 0
        for image_name, alignments, number_alignments in tqdm(self.alignment_data.get_alignments_one_image(),
                                                              desc='Processing alignments file',
                                                              total=self.count_items):
            number_faces = len(self.faces.faces.get(image_name,[]))
            if number_alignments == 0 or number_alignments == number_faces:
                continue
            for idx in self.alignment_data.get_one_alignment_index_reverse(alignments, number_alignments):
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
        for item in tqdm(self.faces.file_list_sorted, desc='Renaming aligned faces', total=self.count_faces):
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
        parser.add_argument('-a', '--alignments_file',
                            type=str,
                            dest='alignments_file',
                            help='Full path to the alignments file to be processed',
                            required=True)
        parser.add_argument('-d', '--destination_format',
                            type=str,
                            choices=('json', 'pickle', 'yaml'),
                            dest='destination_format',
                            help='The file format to save the serialized data in.',
                            required=True)  
        parser.set_defaults(func=self.process)

    def process(self, arguments):
        ''' process the arguments and run reformat '''
        self.alignment_data = AlignmentData(arguments.alignments_file, arguments.destination_format)
        self.alignment_data.save_alignments()

class AlignmentData(object):
    ''' Class to hold the alignment data '''

    def __init__(self, alignments_file, destination_format):
        self.extensions = ['json', 'p', 'yaml', 'yml']
        self.alignments_file = alignments_file
        self.check_alignments_file_exists()
        self.alignments_format = self.get_alignments_format()
        self.destination_format = self.get_destination_format(destination_format)
        self.serializer = self.set_serializer()

        self.load_alignments()
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
            print('alignments file not found at: {}'.format(self.alignments_file))
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

        print('Loading {} alignments from {}\n'.format(self.alignments_format, self.alignments_file))
        with open(self.alignments_file, read_opts) as alignments:
            if self.alignments_format not in ('yaml', 'yml'):
                self.alignments = serializer.loads(alignments.read())
            else:
                self.alignments = serializer.load(alignments.read())
                
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
        with open(dst, write_opts) as alignments:
            if self.alignments_format not in ('yaml', 'yml'):
                alignments.write(serializer.dumps(self.alignments, **serializer_opts))
            else:
                alignments.write(serializer.dump(self.alignments, **serializer_opts))

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
            print('The folder {} could not be found'.format(self.faces_dir))
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

def bad_args(args):
    parser.print_help()
    exit(0)

if __name__ == '__main__':
    print ('Faceswap Alignments File Helper Tool.\n')
   
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers()
    alignment = AlignmentTool(subparser,
                         'alignments', 'Perform various edits to an alignments file.')
    parser.set_defaults(func=bad_args)
    arguments = parser.parse_args()
    arguments.func(arguments)