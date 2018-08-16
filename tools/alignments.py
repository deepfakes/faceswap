#!/usr/bin/env python3
""" Tools for manipulating the alignments seralized file """

# TODO merge alignments
# TODO Identify possible false positives
# TODO Remove whole frames from alignments file
# TODO Add help text to tabs in gui
# TODO Re-extract faces
# TODO Draw Alignments
# TODO Change faces_dir/frames_dir to media dir? Do last incase both are needed for another task
# TODO GUI - Analysis time roll past 24 hours
import datetime
import os
import sys
from tqdm import tqdm

from lib import Serializer


class Alignments(object):
    """ Perform tasks relating to alignments file """
    def __init__(self, arguments):
        self.args = arguments

        dest_format = self.get_dest_format()
        self.alignments = AlignmentData(self.args.alignments_file,
                                        dest_format,
                                        self.args.verbose)

    def get_dest_format(self):
        """ Set the destination format for Alignments """
        dest_format = None
        if (hasattr(self.args, 'alignment_format')
                and self.args.alignment_format):
            dest_format = self.args.alignment_format
        return dest_format

    def process(self):
        """ Main processing function of the Align tool """
        if self.args.job in("frames", "faces"):
            job = Check(self.alignments,
                        self.args)
        elif self.args.job == "remove":
            job = RemoveAlignments(self.alignments,
                                   self.args)
        elif self.args.job == "reformat":
            job = self.alignments

        job.process()


class AlignmentData(object):
    """ Class to hold the alignment data """

    def __init__(self, alignments_file, destination_format, verbose):
        print("\n[ALIGNMENT DATA]")  # Tidy up cli output
        self.alignments_file = alignments_file
        self.verbose = verbose

        self.check_alignments_file_exists()
        self.alignments_format = os.path.splitext(
            self.alignments_file)[1].lower()
        self.destination_format = self.get_destination_format(
            destination_format)

        self.serializer = Serializer.get_serializer_from_ext(
            self.alignments_format)
        self.alignments = self.load_alignments()
        self.count = len(self.alignments)
        self.count_per_frame = {key: len(value)
                                for key, value in self.alignments.items()}

        self.set_destination_serializer()
        if self.verbose:
            print("{} items loaded".format(self.count))

    def process(self):
        """ Commmand to run if calling the reformat command """
        print("\n[REFORMAT ALIGNMENTS]")  # Tidy up cli output
        self.save_alignments()

    def check_alignments_file_exists(self):
        """ Check the alignments file exists"""
        if not os.path.isfile(self.alignments_file):
            print("ERROR: alignments file not "
                  "found at: {}".format(self.alignments_file))
            sys.exit()
        if self.verbose:
            print("Alignments file exists at {}".format(self.alignments_file))

    def get_destination_format(self, destination_format):
        """ Standardise the destination format to the correct extension """
        extensions = {".json": "json",
                      ".p": "pickle",
                      ".yml": "yaml",
                      ".yaml": "yaml"}
        dst_fmt = None

        if destination_format is not None:
            dst_fmt = destination_format
        elif self.alignments_format in extensions.keys():
            dst_fmt = extensions[self.alignments_format]
        else:
            print("{} is not a supported serializer. "
                  "Exiting".format(self.alignments_format))
            sys.exit()

        if self.verbose:
            print("Destination format set to {}".format(dst_fmt))

        return dst_fmt

    def set_destination_serializer(self):
        """ set the destination serializer """
        self.serializer = Serializer.get_serializer(self.destination_format)

    def load_alignments(self):
        """ Read the alignments data from the correct format """
        print("Loading alignments from {}".format(self.alignments_file))
        with open(self.alignments_file, self.serializer.roptions) as align:
            alignments = self.serializer.unmarshal(align.read())
        return alignments

    def save_alignments(self):
        """ Backup copy of old alignments and save new alignments """
        dst = os.path.splitext(self.alignments_file)[0]
        dst += ".{}".format(self.serializer.ext)
        self.backup_alignments()

        print("Saving alignments to {}".format(dst))
        with open(dst, self.serializer.woptions) as align:
            align.write(self.serializer.marshal(self.alignments))

    def backup_alignments(self):
        """ Backup copy of old alignments """
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        src = self.alignments_file
        dst = src.split(".")
        dst[0] += "_" + now + "."
        dst = dst[0] + dst[1]
        print("Backing up original alignments to {}".format(dst))
        os.rename(src, dst)

    def get_alignments_one_image(self):
        """ Return the face alignments for one image """
        for image, alignments in self.alignments.items():
            image_stripped = image[:image.rindex(".")]
            number_alignments = len(alignments)
            yield image_stripped, alignments, number_alignments

    @staticmethod
    def get_one_alignment_index_reverse(image_alignments, number_alignments):
        """ Return the correct original index for
            alignment in reverse order """
        for idx, _ in enumerate(reversed(image_alignments)):
            original_idx = number_alignments - 1 - idx
            yield original_idx


class MediaLoader(object):
    """ Class to load filenames from folder """
    def __init__(self, folder, verbose):
        print("\n[{} DATA]".format(self.__class__.__name__.upper()))
        self.verbose = verbose
        self.folder = folder
        self.check_folder_exists()
        self.file_list_sorted = sorted([item
                                        for item in self.process_folder()])
        self.items = self.load_items()
        self.count = len(self.file_list_sorted)
        if self.verbose:
            print("{} items loaded".format(self.count))

    def check_folder_exists(self):
        """ makes sure that the faces folder exists """
        if not self.folder or not os.path.isdir(self.folder):
            print("ERROR: The folder {} could not "
                  "be found".format(self.folder))
            sys.exit()
        if self.verbose:
            print("Folder exists at {}".format(self.folder))

    @staticmethod
    def process_folder():
        """ Override for specific folder processing """
        return list()

    @staticmethod
    def load_items():
        """ Override for specific item loading """
        return dict()


class Faces(MediaLoader):
    """ Object to hold the faces that are to be swapped out """

    def process_folder(self):
        """ Iterate through the faces dir pulling out various information """
        print("Loading file list from {}".format(self.folder))
        for face in os.listdir(self.folder):
            filename = os.path.splitext(face)[0]
            file_extension = os.path.splitext(face)[1]
            index = int(filename[filename.rindex("_") + 1:])
            original_file = "{}".format(filename[:filename.rindex("_")])
            yield (filename, file_extension, original_file, index)

    def load_items(self):
        """ Load the face names into dictionary """
        faces = dict()
        for face in self.file_list_sorted:
            original_file, index = face[2:4]
            if faces.get(original_file, "") == "":
                faces[original_file] = [index]
            else:
                faces[original_file].append(index)
        return faces


class Frames(MediaLoader):
    """ Object to hold the frames that are to be checked against """

    def process_folder(self):
        """ Iterate through the frames dir pulling the base filename """
        print("Loading file list from {}".format(self.folder))
        for frame in os.listdir(self.folder):
            filename = os.path.basename(frame)
            yield filename

    def load_items(self):
        """ Load the face names into dictionary """
        return self.file_list_sorted


class RemoveAlignments(object):
    """ Remove items from alignments file """
    def __init__(self, alignments, arguments):
        self.alignment_data = alignments
        self.verbose = arguments.verbose
        self.faces = Faces(arguments.faces_dir, self.verbose)
        self.removed = set()

    def process(self):
        """ run removal """
        print("\n[REMOVE ALIGNMENTS DATA]")  # Tidy up cli output
        del_count = 0
        for item in tqdm(self.alignment_data.get_alignments_one_image(),
                         desc="Processing alignments file",
                         total=self.alignment_data.count):
            if self.faces_count_matches(item):
                continue
            del_count += self.remove_alignment(item)

        if del_count == 0:
            print("No changes made to alignments file. Exiting")
            return

        print("{} alignments(s) were removed from "
              "alignments file".format(del_count))
        self.alignment_data.save_alignments()
        self.rename_faces()

    def faces_count_matches(self, item):
        """ Check the selected face exits """
        image_name, number_alignments = item[0], item[2]
        number_faces = len(self.faces.items.get(image_name, []))
        return bool(number_alignments == 0
                    or number_alignments == number_faces)

    def remove_alignment(self, item):
        """ Remove the alignment from the alignments file """
        del_count = 0
        image_name, alignments, number_alignments = item
        processor = self.alignment_data.get_one_alignment_index_reverse
        for idx in processor(alignments, number_alignments):
            face_indexes = self.faces.items.get(image_name, [-1])
            if idx not in face_indexes:
                del alignments[idx]
                self.removed.add(image_name)
                if self.verbose:
                    print("Removed alignment data for image:{} "
                          "index: {}".format(image_name, str(idx)))
                del_count += 1
        return del_count

    def rename_faces(self):
        """ Rename the aligned faces to match their "
            new index in alignments file """
        current_image = ""
        current_index = 0
        rename_count = 0
        for item in tqdm(self.faces.file_list_sorted,
                         desc="Renaming aligned faces",
                         total=self.faces.count):
            filename, extension, original_file, index = item
            if original_file not in self.removed:
                continue
            current_index, current_image = self.set_image_index(current_index,
                                                                current_image,
                                                                original_file)
            if current_index != index:
                rename_count += self.rename_file(filename,
                                                 extension,
                                                 current_image,
                                                 current_index)

            current_index += 1
        if rename_count == 0:
            print("No files were renamed. Exiting")
            return
        print("{} face(s) were renamed to match with "
              "alignments file".format(rename_count))

    @staticmethod
    def set_image_index(index, current, original):
        """ Set the current processing image and index """
        idx = 0 if current != original else index
        return idx, original

    def rename_file(self, filename, extension, image, index):
        """ Rename the selected file """
        old_file = filename + extension
        new_file = "{}_{}{}".format(image, str(index), extension)
        src = os.path.join(self.faces.folder, old_file)
        dst = os.path.join(self.faces.folder, new_file)
        os.rename(src, dst)
        if self.verbose:
            print("Renamed {} to {}".format(src, dst))
        return 1


class Check(object):
    """ Frames and faces checking tasks """
    def __init__(self, alignments, arguments):
        self.alignments_data = alignments.count_per_frame
        self.job = arguments.job
        self.type = arguments.type
        self.output = arguments.output
        self.source_dir = self.get_source_dir(arguments)
        self.items = self.get_items(arguments)

        self.items_output = []
        self.items_discovered = 0
        self.output_message = ""

    def get_source_dir(self, arguments):
        """ Set the correct source dir """
        if self.job == "faces":
            return arguments.faces_dir
        return arguments.frames_dir

    def get_items(self, arguments):
        """ Set the correct items to process """
        items = Frames
        if self.job == "faces":
            items = Faces
        return items(self.source_dir, arguments.verbose).file_list_sorted

    def process(self):
        """ Process the frames check against the alignments file """
        print("\n[CHECK {}]".format(self.job.upper()))
        self.validate()
        self.compile_output()
        self.output_results()

    def validate(self):
        """ Check that the selected type is valid for
            selected task and job """
        if (self.job == "frames"
                and self.type == "missing-frames"
                and self.output == "move"):
            print("WARNING: missing_frames was selected with move output, but "
                  "there will be nothing to move. "
                  "Defaulting to output: console")
            self.output = "console"
        elif self.job == "faces" and self.type != "multi-faces":
            print("WARNING: The selected type is not valid. Only "
                  "'multi-faces' is supported for checking faces ")
            sys.exit()

    def compile_output(self):
        """ Compile list of frames that meet criteria """
        action = self.type.replace("-", "_")
        processor = getattr(self, "get_{}".format(action))
        self.items_output = [item for item in processor()]

    def get_no_faces(self):
        """ yield each frame that has no face match in alignments file """
        self.output_message = "Frames with no faces"
        for item in self.items:
            if self.alignments_data.get(item, -1) == 0:
                yield item

    def get_multi_faces(self):
        """ yield each frame that has multiple faces
            matched in alignments file """
        self.output_message = "Frames with multiple faces"
        items = self.items
        if self.job == "faces":
            self.output_message = "Multiple faces in frame"
        for item in items:
            check_item = item
            return_item = item
            if self.job == "faces":
                check_item = str(item[2]) + str(item[1])
                return_item = str(item[0]) + str(item[1])
            if self.alignments_data.get(check_item, -1) > 1:
                yield return_item

    def get_missing_alignments(self):
        """ yield each frame that does not exist in alignments file """
        self.output_message = "Frames missing from alignments file"
        exclude_filetypes = ["yaml", "yml", "p", "json", "txt"]
        for item in self.items:
            extension = item[item.rindex(".") + 1:]
            if (extension not in exclude_filetypes
                    and self.alignments_data.get(item, -1)) == -1:
                yield item

    def get_missing_frames(self):
        """ yield each frame in alignments that does
            not have a matching file """
        self.output_message = "Missing frames that are in alignments file"
        for item in self.alignments_data.keys():
            if item not in self.items:
                yield item

    def output_results(self):
        """ Output the results in the requested format """
        self.items_discovered = len(self.items_output)
        if self.items_discovered == 0:
            print("No {} were found meeting the criteria".format(self.job))
            return
        if self.output == "move":
            self.move_file()
            return
        output_message = "-----------------------------------------------\r\n"
        output_message += " {} ({})\r\n".format(self.output_message,
                                                self.items_discovered)
        output_message += "-----------------------------------------------\r\n"
        output_message += "\r\n".join([frame for frame in self.items_output])
        if self.output == "console":
            print("\n" + output_message)
        if self.output == "file":
            self.output_file(output_message)

    def output_file(self, output_message):
        """ Save the output to a text file in the frames directory """
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_message.replace(" ", "_").lower()
        filename += "_" + now + ".txt"
        output_file = os.path.join(self.source_dir, filename)
        print("Saving {} result(s) to {}".format(self.items_discovered,
                                                 output_file))
        with open(output_file, "w") as f_output:
            f_output.write(output_message)

    def move_file(self):
        """ Move the identified frames to a new subfolder """
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = self.output_message.replace(" ", "_").lower()
        folder_name += "_" + now
        output_folder = os.path.join(self.source_dir, folder_name)
        os.makedirs(output_folder)
        move = getattr(self, "move_{}".format(self.job))
        move(output_folder)

    def move_frames(self, output_folder):
        """ Move frames into single subfolder """
        print("Moving {} frame(s) to {}".format(self.items_discovered,
                                                output_folder))
        for frame in self.items_output:
            src = os.path.join(self.source_dir, frame)
            dst = os.path.join(output_folder, frame)
            os.rename(src, dst)

    def move_faces(self, output_folder):
        """ Make additional subdirs for each face that appears
            Enables easier manual sorting """
        print("Moving {} faces(s) to {}".format(self.items_discovered,
                                                output_folder))
        for frame in self.items_output:
            idx = frame[frame.rfind("_") + 1:frame.rfind(".")]
            src = os.path.join(self.source_dir, frame)
            dst_folder = os.path.join(output_folder, idx)
            if not os.path.isdir(dst_folder):
                os.makedirs(dst_folder)
            dst = os.path.join(dst_folder, frame)
            os.rename(src, dst)
