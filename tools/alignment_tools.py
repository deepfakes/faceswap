''' Tools for manipulating the alignments seralized file '''

#TODO change format
#TODO merge alignments

import argparse
import datetime
import os
import sys
import json
import pickle

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
        print('Loading {} alignments from {}'.format(self.alignments_format, self.alignments_file))
        alignments = open(self.alignments_file, 'r')
        if self.alignments_format == 'json':
            return json.loads(alignments.read())
        elif self.alignments_format == 'p':
            return pickle.loads(alignments.read())
        elif self.alignments_format in ('yaml', 'yml'):
            try:
                import yaml
            except ImportError:
                print("You must have PyYAML installed to use YAMLSerializer")
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
        now = datetime.datetime.now().strftime("%Y-%m-%d--%H_%M_%S")
        backupfile = '{}_{}{}'.format(os.path.splitext(self.alignments_file)[0], now, os.path.splitext(self.alignments_file)[1])
        print('Backing up old {} to {}'.format(self.alignments_file, backupfile))
        os.rename(self.alignments_file, backupfile)

    def write_alignments(self):
        ''' compile the new alignments '''
        print('Saving alignments to {}'.format(self.alignments_file))
        if self.alignments_format == 'json':
            return json.dumps(self.alignments, indent=2)
        elif self.alignments_format == 'p':
            return pickle.dumps(self.alignments)
        elif self.alignments_format in ('yaml', 'yml'):
            try:
                import yaml
            except ImportError:
                print("You must have PyYAML installed to use YAMLSerializer")
            return yaml.dump(self.alignments, default_flow_style=False)       

    def save_alignments(self):
        ''' Backup copy of old alignments and save new alignments '''
        self.backup_alignments()
        alignments = open(self.alignments_file, 'w')
        alignments.write(self.write_alignments())
        alignments.close()

class Faces(object):
    ''' Object to hold the faces that are to be swapped out '''
    def __init__(self, faces_dir):
        self.faces_dir = faces_dir
        self.faces = {}

        self.load_faces()

    def check_folder_exists(self):
        ''' makes sure that the faces folder exists '''
        if not os.path.isdir(self.faces_dir):
            raise ValueError ('The folder {} could not be found'.format(self.faces_dir))

    def load_faces(self):
        ''' Load the face names into list '''
        print('Loading faces from {}'.format(self.faces_dir))
        try:
            self.check_folder_exists()
            for face in os.listdir(self.faces_dir):
                face_filename = os.path.splitext(face)[0]
                index = int(face_filename[face_filename.rindex('_')+1:])
                original_file = '{}{}'.format(face_filename[:face_filename.rindex('_')], os.path.splitext(face)[1])
                if self.faces.get(original_file, '') == '':
                    self.faces[original_file] = [index]
                else:
                    self.faces[original_file].append(index)
        except ValueError as v:
            print(v)
            sys.exit()
        except:
            raise

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

def remove_alignments(arguments):
    ''' Remove items from alignments file '''
    faces = Faces(arguments.faces_dir)
    alignment_data = AlignmentData(arguments.alignments_dir)
    del_count = 0
    print('\n-------------------------------------\n Processing compare...')
    for key, value in alignment_data.alignments.items():
        if len(value) == 0 or len(value) == len(faces.faces.get(key,'')):
            continue
        for idx, val in enumerate(value):
            if idx not in faces.faces.get(key,[-1]):
                del value[idx]
                del_count += 1
                print('removed face index {} for {}'.format(idx, key))
    print('-------------------------------------\n')
    if del_count == 0:
        print('No changes made to alignments file. Exiting')
        return
    print('{} face(s) were removed from alignments file'.format(del_count))
    alignment_data.save_alignments()

def select_job(arguments):
    ''' Select the job '''
    if arguments.job.lower() == 'remove':
        remove_alignments(arguments)

if __name__ == '__main__':
    print ('Faceswap Alignments File Helper Tool.\n')
    
    parser = argparse.ArgumentParser()
    parser.set_defaults(func=bad_args)
    parser_arguments(parser)
    arguments = parser.parse_args()
    select_job(arguments)

