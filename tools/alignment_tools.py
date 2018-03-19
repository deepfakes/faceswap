''' Tools for manipulating the alignments seralized file '''

#TODO change format
#TODO merge alignments

import argparse
import datetime
import os
import sys
import json
import pickle
from tqdm import tqdm

class AlignmentData(object):
    ''' Class to hold the alignment data '''

    def __init__(self, alignments_dir):
        self.extensions = ['json', 'p', 'yaml', 'yml']
        self.alignments_file = None
        self.alignments_format = None
        self.alignments = None

        self.load_alignments(alignments_dir)

    def alignments_file_exists(self, alignments_base, extension):
        ''' return 1 if a file is found '''
        alignments_file = '{}.{}'.format(alignments_base, extension)
        if os.path.isfile(alignments_file):
            self.alignments_file = alignments_file
            self.alignments_format = extension
            return 1
        return 0

    def check_alignments_file(self, alignments_base):
        ''' Checks that there is just one alignment file '''
        file_count = 0
        for extension in self.extensions:
            file_count += self.alignments_file_exists(alignments_base, extension)
        if file_count != 1:
            raise ValueError ('1 alignments file should exist, but {} were found'.format(file_count))

    def read_alignments(self):
        ''' Read the alignments data from the correct format '''
        print('Loading {} alignments from {}\n'.format(self.alignments_format, self.alignments_file))
        alignments = open(self.alignments_file, 'r')
        if self.alignments_format == 'json':
            return json.loads(alignments.read())
        elif self.alignments_format == 'p':
            return pickle.loads(alignments.read())
        elif self.alignments_format in ('yaml', 'yml'):
            try:
                import yaml
            except ImportError:
                raise ValueError('You must have PyYAML installed to use YAMLSerializer')
            return yaml.load(alignments.read())
        alignments.close()
    
    def load_alignments(self, alignments_dir):
        ''' load the data from the alignments file '''
        alignments_base = os.path.join(alignments_dir, 'alignments')
        try:
            self.check_alignments_file(alignments_base)
            self.alignments = self.read_alignments()
        except ValueError as v:
            print(v)
            sys.exit()
        except:
            raise

    def backup_alignments(self):
        ''' Backup copy of old alignments '''
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backupfile = '{}_{}{}'.format(os.path.splitext(self.alignments_file)[0], now, os.path.splitext(self.alignments_file)[1])
        print('Backing up old {} to {}'.format(self.alignments_file, backupfile))
        os.rename(self.alignments_file, backupfile)

    def write_alignments(self):
        ''' compile the new alignments '''
        print('Saving alignments to {}\n'.format(self.alignments_file))
        if self.alignments_format == 'json':
            return json.dumps(self.alignments, indent=2)
        elif self.alignments_format == 'p':
            return pickle.dumps(self.alignments)
        elif self.alignments_format in ('yaml', 'yml'):
            try:
                import yaml
            except ImportError:
                raise ValueError('You must have PyYAML installed to use YAMLSerializer')
            return yaml.dump(self.alignments, default_flow_style=False)       

    def save_alignments(self):
        ''' Backup copy of old alignments and save new alignments '''
        self.backup_alignments()
        alignments = open(self.alignments_file, 'w')
        alignments.write(self.write_alignments())
        alignments.close()

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

class RemoveAlignments(object):
    ''' Remove items from alignments file '''
    def __init__(self, arguments):
        self.faces = Faces(arguments.faces_dir)
        self.alignment_data = AlignmentData(arguments.alignments_dir)
        self.count_items = len(self.alignment_data.alignments)
        self.count_faces = len(self.faces.file_list_sorted)
        self.removed = set()

    def remove_alignment(self):
        ''' Remove the alignment from the alignments file '''
        del_count = 0
        for image_name, alignments, number_alignments in tqdm(self.alignment_data.get_alignments_one_image(),
                                                              desc='Processing alignments file',
                                                              total=self.count_items):
            if number_alignments == 0 or number_alignments == len(self.faces.faces.get(image_name,[])):
                continue
            for idx in self.alignment_data.get_one_alignment_index_reverse(alignments, number_alignments):
                if idx not in self.faces.faces.get(image_name,[-1]):
                    print(image_name)
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
                    old_file = os.path.join(self.faces.faces_dir,'{}{}'.format(filename, extension))
                    new_file = os.path.join(self.faces.faces_dir,'{}_{}{}'.format(original_file, current_index, extension))
                    os.rename(old_file, new_file)
                    rename_count += 1
                current_index += 1
        if rename_count == 0:
            print('No files were renamed. Exiting')
            return
        print('{} face(s) were renamed to match with alignments file'.format(rename_count))

def bad_args(args):
    parser.print_help()
    exit(0)

def parser_arguments(parser):
    parser.add_argument('-j', '--job',
                        type=str,
                        choices=('remove', 'merge', 'reformat'),
                        dest='job',
                        help='The job to be performed.',
                        required=True)
    parser.add_argument('-a', '--alignments_folder',
                        type=str,
                        dest='alignments_dir',
                        help='Input directory of source A frames. Alignments file must exist in '
                              'this folder',
                        required=True)
    parser.add_argument('-f', '--faces_folder',
                        type=str,
                        dest='faces_dir',
                        help='Input directory of source A extracted faces.',
                        required=True)

def select_job(arguments):
    ''' Select the job '''
    if arguments.job.lower() == 'remove':
        job = RemoveAlignments(arguments)
        job.remove_alignment()

if __name__ == '__main__':
    print ('Faceswap Alignments File Helper Tool.\n')
    
    parser = argparse.ArgumentParser()
    parser.set_defaults(func=bad_args)
    parser_arguments(parser)
    arguments = parser.parse_args()
    select_job(arguments)

