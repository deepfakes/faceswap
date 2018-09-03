#!/usr/bin/env python3
""" Tools for manipulating the alignments seralized file """

# TODO merge alignments
# TODO Remove whole frames from alignments file
import os
import pickle
import struct
from datetime import datetime

from cv2 import circle, imread, imwrite, rectangle
from tqdm import tqdm

from lib import Serializer
from lib.utils import _image_extensions
from plugins.PluginLoader import PluginLoader


class Alignments():
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
        if self.args.job == "draw":
            job = Draw(self.alignments, self.args)
        elif self.args.job == "extract":
            job = Extract(self.alignments, self.args)
        elif self.args.job in("missing-alignments", "missing-frames",
                              "multi-faces", "leftover-faces",
                              "no-faces"):
            job = Check(self.alignments, self.args)
        elif self.args.job == "remove":
            job = RemoveAlignments(self.alignments, self.args)
        elif self.args.job == "reformat":
            job = Reformat(self.alignments, self.args)

        job.process()


class AlignmentData():
    """ Class to hold the alignment data """

    def __init__(self, alignments_file, destination_format, verbose):
        print("\n[ALIGNMENT DATA]")  # Tidy up cli output
        self.file = alignments_file
        self.verbose = verbose

        self.check_file_exists()
        self.src_format = self.get_source_format()
        self.dst_format = self.get_destination_format(destination_format)

        if self.src_format == "dfl":
            self.set_destination_serializer()
            return

        self.serializer = Serializer.get_serializer_from_ext(
            self.src_format)
        self.alignments = self.load()
        self.count = len(self.alignments)
        self.count_per_frame = {key: len(value)
                                for key, value in self.alignments.items()}

        self.set_destination_serializer()
        if self.verbose:
            print("{} items loaded".format(self.count))

    def check_file_exists(self):
        """ Check the alignments file exists"""
        if os.path.split(self.file.lower())[1] == "dfl":
            self.file = "dfl"
        if self.file.lower() == "dfl":
            print("Using extracted pngs for alignments")
            return
        if not os.path.isfile(self.file):
            print("ERROR: alignments file not "
                  "found at: {}".format(self.file))
            exit(0)
        if self.verbose:
            print("Alignments file exists at {}".format(self.file))
        return

    def get_source_format(self):
        """ Get the source alignments format """
        if self.file.lower() == "dfl":
            return "dfl"
        return os.path.splitext(self.file)[1].lower()

    def get_destination_format(self, destination_format):
        """ Standardise the destination format to the correct extension """
        extensions = {".json": "json",
                      ".p": "pickle",
                      ".yml": "yaml",
                      ".yaml": "yaml"}
        dst_fmt = None

        if destination_format is not None:
            dst_fmt = destination_format
        elif self.src_format == "dfl":
            dst_fmt = "json"
        elif self.src_format in extensions.keys():
            dst_fmt = extensions[self.src_format]
        else:
            print("{} is not a supported serializer. "
                  "Exiting".format(self.src_format))
            exit(0)

        if self.verbose:
            print("Destination format set to {}".format(dst_fmt))

        return dst_fmt

    def set_destination_serializer(self):
        """ set the destination serializer """
        self.serializer = Serializer.get_serializer(self.dst_format)

    def load(self):
        """ Read the alignments data from the correct format """
        print("Loading alignments from {}".format(self.file))
        with open(self.file, self.serializer.roptions) as align:
            alignments = self.serializer.unmarshal(align.read())
        return alignments

    def save_alignments(self):
        """ Backup copy of old alignments and save new alignments """
        dst = os.path.splitext(self.file)[0]
        dst += ".{}".format(self.serializer.ext)
        self.backup_alignments()

        print("Saving alignments to {}".format(dst))
        with open(dst, self.serializer.woptions) as align:
            align.write(self.serializer.marshal(self.alignments))

    def backup_alignments(self):
        """ Backup copy of old alignments """
        if not os.path.isfile(self.file):
            return
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        src = self.file
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

    def has_alignments(self, filename, alignments):
        """ Check whether this frame has alignments """
        if not alignments:
            if self.verbose:
                print("Skipping {} - Alignments not found".format(filename))
            return False
        return True


class MediaLoader():
    """ Class to load filenames from folder """
    def __init__(self, folder, verbose):
        print("\n[{} DATA]".format(self.__class__.__name__.upper()))
        self.verbose = verbose
        self.folder = folder
        self.check_folder_exists()
        self.file_list_sorted = self.sorted_items()
        self.items = self.load_items()
        self.count = len(self.file_list_sorted)
        if self.verbose:
            print("{} items loaded".format(self.count))

    def check_folder_exists(self):
        """ makes sure that the faces folder exists """
        err = None
        loadtype = self.__class__.__name__
        if not self.folder:
            err = "ERROR: A {} folder must be specified".format(loadtype)
        elif not os.path.isdir(self.folder):
            err = ("ERROR: The {} folder {} could not be "
                   "found".format(loadtype, self.folder))
        if err:
            print(err)
            exit(0)

        if self.verbose:
            print("Folder exists at {}".format(self.folder))

    @staticmethod
    def valid_extension(filename):
        """ Check whether passed in file has a valid extension """
        extension = os.path.splitext(filename)[1]
        return bool(extension in _image_extensions)

    @staticmethod
    def sorted_items():
        """ Override for specific folder processing """
        return list()

    @staticmethod
    def process_folder():
        """ Override for specific folder processing """
        return list()

    @staticmethod
    def load_items():
        """ Override for specific item loading """
        return dict()

    def load_image(self, filename):
        """ Load an image """
        src = os.path.join(self.folder, filename)
        image = imread(src)
        return image

    @staticmethod
    def save_image(output_folder, filename, image):
        """ Save an image """
        output_file = os.path.join(output_folder, filename)
        imwrite(output_file, image)


class Faces(MediaLoader):
    """ Object to hold the faces that are to be swapped out """

    def process_folder(self):
        """ Iterate through the faces dir pulling out various information """
        print("Loading file list from {}".format(self.folder))
        for face in os.listdir(self.folder):
            if not self.valid_extension(face):
                continue
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

    def sorted_items(self):
        """ Return the items sorted by filename then index """
        return sorted([item for item in self.process_folder()],
                      key=lambda x: (x[2], x[3]))


class Frames(MediaLoader):
    """ Object to hold the frames that are to be checked against """

    def process_folder(self):
        """ Iterate through the frames dir pulling the base filename """
        print("Loading file list from {}".format(self.folder))
        for frame in os.listdir(self.folder):
            if not self.valid_extension(frame):
                continue
            filename = os.path.basename(frame)
            yield filename

    def load_items(self):
        """ Load the frame info into dictionary """
        frames = dict()
        for frame in self.file_list_sorted:
            frames[frame] = (frame[:frame.rfind(".")],
                             frame[frame.rfind("."):])
        return frames

    def sorted_items(self):
        """ Return the items sorted by filename """
        return sorted([item for item in self.process_folder()])


class Draw():
    """ Draw Alignments on passed in images """
    def __init__(self, alignments, arguments):
        self.verbose = arguments.verbose
        self.alignments = alignments
        self.frames = Frames(arguments.frames_dir, self.verbose)
        self.output_folder = self.set_output()

    def set_output(self):
        """ Set the output folder path """
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = "drawn_landmarks_{}".format(now)
        output_folder = os.path.join(self.frames.folder, folder_name)
        os.makedirs(output_folder)
        return output_folder

    def process(self):
        """ Run the draw alignments process """
        print("\n[DRAW LANDMARKS]")  # Tidy up cli output
        frames_drawn = 0
        for frame, alignments in tqdm(self.get_frame_alignments(),
                                      desc="Drawing landmarks",
                                      total=self.frames.count):
            if not self.alignments.has_alignments(frame, alignments):
                continue
            self.annotate_image(frame, alignments)
            frames_drawn += 1
        print("{} Frame(s) output".format(frames_drawn))

    def get_frame_alignments(self):
        """ Retrieve each frame and it's corresponding alignments """
        for frame in self.frames.file_list_sorted:
            alignments = self.alignments.alignments.get(frame, None)
            yield frame, alignments

    def annotate_image(self, frame, alignments):
        """ Draw the alignments """
        image = self.frames.load_image(frame)
        for alignment in alignments:
            self.draw_bounding_box(image, alignment)
            self.draw_landmarks(image, alignment["landmarksXY"])
        self.frames.save_image(self.output_folder, frame, image)

    @staticmethod
    def draw_bounding_box(image, alignment):
        """ Draw the bounding box around face """
        top_left = (alignment["x"], alignment["y"])
        bottom_right = (alignment["x"] + alignment["w"],
                        alignment["y"] + alignment["h"])
        rectangle(image, top_left, bottom_right, (0, 0, 255), 1)

    @staticmethod
    def draw_landmarks(image, landmarks):
        """ Draw the facial landmarks """
        for (pos_x, pos_y) in landmarks:
            circle(image, (pos_x, pos_y), 1, (0, 255, 0), -1)


class Reformat():
    """ Reformat Alignment file """
    def __init__(self, alignments, arguments):
        self.verbose = arguments.verbose
        self.alignments = alignments
        self.frames_dir = arguments.frames_dir
        if self.alignments.src_format == "dfl":
            self.frames = Frames(self.frames_dir,
                                 self.verbose).items
            self.faces = Faces(arguments.faces_dir,
                               self.verbose)

    def process(self):
        """ Run reformat """
        print("\n[REFORMAT ALIGNMENTS]")  # Tidy up cli output
        if self.alignments.src_format == "dfl":
            self.alignments.alignments = self.load_dfl()
            self.alignments.file = os.path.join(self.frames_dir,
                                                "alignments.json")
        self.alignments.save_alignments()

    def load_dfl(self):
        """ Load alignments from DeepFaceLab and format for Faceswap """
        alignments = dict()
        frames = {item[0]: item[1]
                  for item in self.frames.values()}
        for face in self.faces.file_list_sorted:
            if not self.validate_dfl(face, frames):
                continue

            fullpath = os.path.join(self.faces.folder, face[0] + face[1])
            dfl = self.get_dfl_alignment(fullpath)

            if not dfl:
                continue

            sourcefile = face[2] + frames[face[2]]
            self.convert_dfl_alignment(dfl, sourcefile, alignments)
        return alignments

    def validate_dfl(self, face, frames):
        """ Validate that current file is good for dfl extraction """
        if face[1] != ".png":
            if self.verbose:
                print("{} is not a png. Skipping".format(face[0] + face[1]))
            return False
        if face[2] not in frames.keys():
            if self.verbose:
                print("{} does not have a matching source frame for {} in "
                      "frames directory. Skipping".format(face[0] + face[1],
                                                          face[2]))
            return False
        return True

    @staticmethod
    def get_dfl_alignment(filename):
        """ Process the alignment of one face """
        with open(filename, "rb") as dfl:
            header = dfl.read(8)
            if header != b"\x89PNG\r\n\x1a\n":
                print("ERROR: No Valid PNG header: {}".format(filename))
                return
            while True:
                chunk_start = dfl.tell()
                chunk_hdr = dfl.read(8)
                if not chunk_hdr:
                    break
                chunk_length, chunk_name = struct.unpack("!I4s", chunk_hdr)
                dfl.seek(chunk_start, os.SEEK_SET)
                if chunk_name == b"fcWp":
                    chunk = dfl.read(chunk_length + 12)
                    return pickle.loads(chunk[8:-4])
                else:
                    dfl.seek(chunk_length+12, os.SEEK_CUR)
            print("ERROR: Couldn't find DFL alignments: {}".format(filename))

    @staticmethod
    def convert_dfl_alignment(dfl_alignments, sourcefile, alignments):
        """ Add DFL Alignments to alignments in Faceswap format """
        if not alignments.get(sourcefile, None):
            alignments[sourcefile] = list()

        left, top, right, bottom = dfl_alignments["source_rect"]
        alignment = {"r": 0,
                     "x": left,
                     "w": right - left,
                     "y": top,
                     "h": bottom - top,
                     "landmarksXY": dfl_alignments["source_landmarks"]}

        alignments[sourcefile].append(alignment)


class Extract():
    """ Re-extract faces from source frames based on
        Alignment data """
    def __init__(self, alignments, arguments):
        self.verbose = arguments.verbose
        self.alignments = alignments
        self.faces_dir = arguments.faces_dir
        self.align_eyes = arguments.align_eyes
        self.frames = Frames(arguments.frames_dir, self.verbose)
        self.extractor = None

    class DetectedFace():
        """ Detected face and landmark information """
        def __init__(self, image, r, x, w, y, h, landmarksXY):
            self.image = image
            self.r = r
            self.x = x
            self.w = w
            self.y = y
            self.h = h
            self.landmarksXY = landmarksXY

        def landmarks_as_xy(self):
            """ Landmarks as XY """
            return self.landmarksXY

    def process(self):
        """ Run extraction """
        print("\n[EXTRACT FACES]")  # Tidy up cli output
        self.check_folder()
        self.extractor = PluginLoader.get_extractor("Align")()
        self.export_faces()

    def check_folder(self):
        """ Check that the faces folder doesn't pre-exist
            and create """
        err = None
        if not self.faces_dir:
            err = "ERROR: Output faces folder not provided."
        elif os.path.isdir(self.faces_dir):
            err = "ERROR: Folder already exists at {}".format(self.faces_dir)
        if err:
            print(err)
            exit(0)
        if self.verbose:
            print("Creating output folder at {}".format(self.faces_dir))
        os.makedirs(self.faces_dir)

    def export_faces(self):
        """ Export the faces """
        extracted_faces = 0
        for frame, frame_info, alignments in tqdm(self.get_frame_alignments(),
                                                  desc="Extracting faces",
                                                  total=self.frames.count):
            if not self.alignments.has_alignments(frame, alignments):
                continue
            extracted_faces += self.output_faces(frame,
                                                 frame_info,
                                                 alignments)
        print("{} face(s) extracted".format(extracted_faces))

    def get_frame_alignments(self):
        """ Return the alignments for each frame """
        for key, value in self.frames.items.items():
            alignments = self.alignments.alignments.get(key, None)
            yield key, value, alignments

    def output_faces(self, frame, frame_info, alignments):
        """ Output the frame's faces to file """
        face_count = 0
        image = self.frames.load_image(frame)
        name, extension = frame_info
        for idx, alignment in enumerate(alignments):
            face = self.DetectedFace(image,
                                     alignment["r"],
                                     alignment["x"],
                                     alignment["w"],
                                     alignment["y"],
                                     alignment["h"],
                                     alignment["landmarksXY"])
            resized_face, _ = self.extractor.extract(image,
                                                     face,
                                                     256,
                                                     self.align_eyes)
            output = "{}_{}{}".format(name, str(idx), extension)
            self.frames.save_image(self.faces_dir, output, resized_face)
            face_count += 1
        return face_count


class RemoveAlignments():
    """ Remove items from alignments file """
    def __init__(self, alignments, arguments):
        self.verbose = arguments.verbose
        self.alignments = alignments
        self.faces = Faces(arguments.faces_dir, self.verbose)
        self.removed = set()

    def process(self):
        """ run removal """
        print("\n[REMOVE ALIGNMENTS DATA]")  # Tidy up cli output
        del_count = 0
        for item in tqdm(self.alignments.get_alignments_one_image(),
                         desc="Processing alignments file",
                         total=self.alignments.count):
            if self.faces_count_matches(item):
                continue
            del_count += self.remove_alignment(item)

        if del_count == 0:
            print("No changes made to alignments file. Exiting")
            return

        print("{} alignments(s) were removed from "
              "alignments file".format(del_count))
        self.alignments.save_alignments()
        self.rename_faces()

    def faces_count_matches(self, item):
        """ Check the selected face exits """
        image_name, number_alignments = item[0], item[2]
        number_faces = len(self.faces.items.get(image_name, None))
        return bool(number_alignments == 0
                    or number_alignments == number_faces)

    def remove_alignment(self, item):
        """ Remove the alignment from the alignments file """
        del_count = 0
        image_name, alignments, number_alignments = item
        processor = self.alignments.get_one_alignment_index_reverse
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


class Check():
    """ Frames and faces checking tasks """
    def __init__(self, alignments, arguments):
        self.alignments_data = alignments.count_per_frame
        self.job = arguments.job
        self.type = None
        self.output = arguments.output
        self.source_dir = self.get_source_dir(arguments)
        self.items = self.get_items(arguments)

        self.items_output = []
        self.items_discovered = 0
        self.output_message = ""

    def get_source_dir(self, arguments):
        """ Set the correct source dir """
        if hasattr(arguments, "faces_dir") and arguments.faces_dir:
            self.type = "faces"
            source_dir = arguments.faces_dir
        elif hasattr(arguments, "frames_dir") and arguments.frames_dir:
            self.type = "frames"
            source_dir = arguments.frames_dir
        else:
            print("No source folder (-fr or -fc) was provided")
            exit(0)
        return source_dir

    def get_items(self, arguments):
        """ Set the correct items to process """
        items = Frames
        if self.type == "faces":
            items = Faces
        return items(self.source_dir, arguments.verbose).file_list_sorted

    def process(self):
        """ Process the frames check against the alignments file """
        print("\n[CHECK {}]".format(self.type.upper()))
        self.validate()
        self.compile_output()
        self.output_results()

    def validate(self):
        """ Check that the selected type is valid for
            selected task and job """
        if self.job == "missing-frames" and self.output == "move":
            print("WARNING: missing_frames was selected with move output, but "
                  "there will be nothing to move. "
                  "Defaulting to output: console")
            self.output = "console"
        elif self.type == "faces" and self.job not in ("multi-faces",
                                                       "leftover-faces"):
            print("WARNING: The selected folder is not valid. "
                  "Only folder set with '-fc' is supported for "
                  "'multi-faces' and 'leftover-faces'")
            exit(0)

    def compile_output(self):
        """ Compile list of frames that meet criteria """
        action = self.job.replace("-", "_")
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
        if self.type == "faces":
            self.output_message = "Multiple faces in frame"
        for item in items:
            check_item = item
            return_item = item
            if self.type == "faces":
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

    def get_leftover_faces(self):
        """yield each face that isn't in the alignments file."""
        self.output_message = "Faces missing from the alignments file"
        for item in self.items:
            frame_id = item[2] + item[1]

            if (frame_id not in self.alignments_data
                    or self.alignments_data[frame_id] <= item[3]):
                yield item[0] + item[1]

    def output_results(self):
        """ Output the results in the requested format """
        self.items_discovered = len(self.items_output)
        if self.items_discovered == 0:
            print("No {} were found meeting the criteria".format(self.type))
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
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_message.replace(" ", "_").lower()
        filename += "_" + now + ".txt"
        output_file = os.path.join(self.source_dir, filename)
        print("Saving {} result(s) to {}".format(self.items_discovered,
                                                 output_file))
        with open(output_file, "w") as f_output:
            f_output.write(output_message)

    def move_file(self):
        """ Move the identified frames to a new subfolder """
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = self.output_message.replace(" ", "_").lower()
        folder_name += "_" + now
        output_folder = os.path.join(self.source_dir, folder_name)
        os.makedirs(output_folder)
        move = getattr(self, "move_{}".format(self.type))
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
