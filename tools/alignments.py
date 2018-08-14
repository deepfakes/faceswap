#!/usr/bin/env python3
""" Tools for manipulating the alignments seralized file """

# TODO merge alignments
# TODO Identify possible false positives
# TODO Remove whole frames from alignments file
# TODO Format helptext for line breaks for cli and gui
# TODO Add help text to tabs in gui
# TODO Add faces tool
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
        if self.args.job == "frames":
            job = CheckFrames(self.alignments,
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


class Faces(object):
    """ Object to hold the faces that are to be swapped out """
    def __init__(self, faces_dir, verbose):
        print("\n[FACES DATA]")  # Tidy up cli output
        self.faces_dir = faces_dir
        self.verbose = verbose
        self.check_folder_exists()
        self.file_list_sorted = sorted([item
                                        for item in self.process_faces_dir()])
        self.faces = {}
        self.load_faces()
        self.count = len(self.file_list_sorted)

    def check_folder_exists(self):
        """ makes sure that the faces folder exists """
        if not os.path.isdir(self.faces_dir):
            print("ERROR: The folder {} could not "
                  "be found".format(self.faces_dir))
            sys.exit()
        if self.verbose:
            print("Faces folder exists at {}".format(self.faces_dir))

    def process_faces_dir(self):
        """ Iterate through the faces dir pulling out various information """
        print("Loading file list from {}".format(self.faces_dir))
        for face in os.listdir(self.faces_dir):
            filename = os.path.splitext(face)[0]
            file_extension = os.path.splitext(face)[1]
            index = int(filename[filename.rindex("_") + 1:])
            original_file = "{}".format(filename[:filename.rindex("_")])
            yield (filename, file_extension, original_file, index)

    def load_faces(self):
        """ Load the face names into dictionary """
        for item in self.file_list_sorted:
            original_file, index = item[2:4]
            if self.faces.get(original_file, "") == "":
                self.faces[original_file] = [index]
            else:
                self.faces[original_file].append(index)
        if self.verbose:
            print("Faces loaded")


class Frames(object):
    """ Object to hold the frames that are to be checked against """
    def __init__(self, frames_dir, verbose):
        print("\n[FRAMES DATA]")  # Tidy up cli output
        self.verbose = verbose
        self.frames_dir = frames_dir
        self.check_folder_exists()
        self.frames = sorted([item for item in self.process_frames_dir()])
        self.count = len(self.frames)

    def check_folder_exists(self):
        """ makes sure that the frames folder exists """
        if not os.path.isdir(self.frames_dir):
            print("ERROR: The folder {} could not "
                  "be found".format(self.frames_dir))
            sys.exit()
        if self.verbose:
            print("Frames folder exists at {}".format(self.frames_dir))

    def process_frames_dir(self):
        """ Iterate through the frames dir pulling the base filename """
        print("Loading file list from {}".format(self.frames_dir))
        for frame in os.listdir(self.frames_dir):
            filename = os.path.basename(frame)
            yield filename


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
        self.remove_alignment()

    def remove_alignment(self):
        """ Remove the alignment from the alignments file """
        del_count = 0
        for item in tqdm(self.alignment_data.get_alignments_one_image(),
                         desc="Processing alignments file",
                         total=self.alignment_data.count):
            image_name, alignments, number_alignments = item
            number_faces = len(self.faces.faces.get(image_name, []))
            if number_alignments == 0 or number_alignments == number_faces:
                continue
            processor = self.alignment_data.get_one_alignment_index_reverse
            for idx in processor(alignments, number_alignments):
                face_indexes = self.faces.faces.get(image_name, [-1])
                if idx not in face_indexes:
                    del alignments[idx]
                    self.removed.add(image_name)
                    if self.verbose:
                        print("Removed alignment data for image:{} "
                              "index: {}".format(image_name, str(idx)))
                    del_count += 1
        if del_count == 0:
            print("No changes made to alignments file. Exiting")
            return
        print("{} alignments(s) were removed from "
              "alignments file".format(del_count))
        self.alignment_data.save_alignments()
        self.rename_faces()

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
            if original_file in self.removed:
                if current_image != original_file:
                    current_index = 0
                current_image = original_file
                if current_index != index:
                    old_file = filename + extension
                    new_file = "{}_{}{}".format(current_image,
                                                str(current_index),
                                                extension)
                    src = os.path.join(self.faces.faces_dir, old_file)
                    dst = os.path.join(self.faces.faces_dir, new_file)
                    os.rename(src, dst)
                    if self.verbose:
                        print("Renamed {} to {}".format(src, dst))
                    rename_count += 1
                current_index += 1
        if rename_count == 0:
            print("No files were renamed. Exiting")
            return
        print("{} face(s) were renamed to match with "
              "alignments file".format(rename_count))


class CheckFrames(object):
    """ Check original Frames against alignments file """
    def __init__(self, alignments, arguments):
        self.alignments_data = alignments.count_per_frame
        self.type = arguments.type
        self.output = arguments.output
        self.frames_dir = arguments.frames_dir
        self.frames = Frames(self.frames_dir, arguments.verbose).frames

        self.frames_output = []
        self.frames_discovered = 0
        self.output_message = ""

    def process(self):
        """ Process the frames check against the alignments file """
        print("\n[CHECK FRAMES]")
        self.check_output()
        self.compile_frames_output()
        self.output_results()

    def check_output(self):
        """ Check that the selected type makes sense against
            the selected output """
        if self.type == "missing-frames" and self.output == "move":
            print("WARNING: missing_frames was selected with move output, but "
                  "there will be nothing to move. "
                  "Defaulting to output: console")
            self.output = "console"

    def compile_frames_output(self):
        """ Compile list of frames that meet criteria """
        action = self.type.replace("-", "_")
        processor = getattr(self, "get_{}".format(action))
        self.frames_output = [frame for frame in processor()]

    def get_no_faces(self):
        """ yield each frame that has no face match in alignments file """
        self.output_message = "Frames with no faces"
        for frame in self.frames:
            if self.alignments_data.get(frame, -1) == 0:
                yield frame

    def get_multi_faces(self):
        """ yield each frame that has multiple faces
            matched in alignments file """
        self.output_message = "Frames with multiple faces"
        for frame in self.frames:
            if self.alignments_data.get(frame, -1) > 1:
                yield frame

    def get_missing_alignments(self):
        """ yield each frame that does not exist in alignments file """
        self.output_message = "Frames missing from alignments file"
        exclude_filetypes = ["yaml", "yml", "p", "json", "txt"]
        for frame in self.frames:
            extension = frame[frame.rindex(".") + 1:]
            if (extension not in exclude_filetypes
                    and self.alignments_data.get(frame, -1)) == -1:
                yield frame

    def get_missing_frames(self):
        """ yield each frame in alignments that does
            not have a matching file """
        self.output_message = "Missing frames that are in alignments file"
        for frame in self.alignments_data.keys():
            if frame not in self.frames:
                yield frame

    def output_results(self):
        """ Output the results in the requested format """
        self.frames_discovered = len(self.frames_output)
        if self.frames_discovered == 0:
            print("No frames were found meeting the criteria")
            return
        if self.output == "move":
            self.move_file()
            return
        output_message = "-----------------------------------------------\r\n"
        output_message += " {} ({})\r\n".format(self.output_message,
                                                self.frames_discovered)
        output_message += "-----------------------------------------------\r\n"
        output_message += "\r\n".join([frame for frame in self.frames_output])
        if self.output == "console":
            print("\n" + output_message)
        if self.output == "file":
            self.output_file(output_message)

    def output_file(self, output_message):
        """ Save the output to a text file in the frames directory """
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_message.replace(" ", "_").lower()
        filename += "_" + now + ".txt"
        output_file = os.path.join(self.frames_dir, filename)
        print("Saving {} result(s) to {}".format(self.frames_discovered,
                                                 output_file))
        with open(output_file, "w") as f_output:
            f_output.write(output_message)

    def move_file(self):
        """ Move the identified frames to a new subfolder """
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = self.output_message.replace(" ", "_").lower()
        folder_name += "_" + now
        output_folder = os.path.join(self.frames_dir, folder_name)
        os.makedirs(output_folder)
        print("Moving {} frame(s) to {}".format(self.frames_discovered,
                                                output_folder))
        for frame in self.frames_output:
            src = os.path.join(self.frames_dir, frame)
            dst = os.path.join(output_folder, frame)
            os.rename(src, dst)
