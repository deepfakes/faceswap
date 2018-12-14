#!/usr/bin/env python3
""" Tools for manipulating the alignments seralized file """

import logging
import os
import pickle
import struct
from datetime import datetime

import numpy as np
from scipy import signal
from sklearn import decomposition
from tqdm import tqdm

from . import Annotate, ExtractedFaces, Faces, Frames

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Check():
    """ Frames and faces checking tasks """
    def __init__(self, alignments, arguments):
        self.alignments = alignments
        self.job = arguments.job
        self.type = None
        self.output = arguments.output
        self.source_dir = self.get_source_dir(arguments)
        self.items = self.get_items()

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
            logger.error("No source folder (-fr or -fc) was provided")
            exit(0)
        return source_dir

    def get_items(self):
        """ Set the correct items to process """
        items = globals()[self.type.title()]
        return items(self.source_dir).file_list_sorted

    def process(self):
        """ Process the frames check against the alignments file """
        logger.info("[CHECK %s]", self.type.upper())
        self.validate()
        items_output = self.compile_output()
        self.output_results(items_output)

    def validate(self):
        """ Check that the selected type is valid for
            selected task and job """
        if self.job == "missing-frames" and self.output == "move":
            logger.warning("Missing_frames was selected with move output, but there will "
                           "be nothing to move. Defaulting to output: console")
            self.output = "console"
        elif self.type == "faces" and self.job not in ("multi-faces",
                                                       "leftover-faces"):
            logger.warning("The selected folder is not valid. Only folder set with '-fc' is "
                           "supported for 'multi-faces' and 'leftover-faces'")
            exit(0)

    def compile_output(self):
        """ Compile list of frames that meet criteria """
        action = self.job.replace("-", "_")
        processor = getattr(self, "get_{}".format(action))
        return [item for item in processor()]

    def get_no_faces(self):
        """ yield each frame that has no face match in alignments file """
        self.output_message = "Frames with no faces"
        for frame in tqdm(self.items, total=len(self.items)):
            frame_name = frame["frame_fullname"]
            if not self.alignments.frame_has_faces(frame_name):
                yield frame_name

    def get_multi_faces(self):
        """ yield each frame that has multiple faces
            matched in alignments file """
        if self.type == "faces":
            self.output_message = "Multiple faces in frame"
            frame_key = "frame_name"
            retval_key = "face_fullname"
        elif self.type == "frames":
            self.output_message = "Frames with multiple faces"
            frame_key = "frame_fullname"
            retval_key = "frame_fullname"

        for item in tqdm(self.items, total=len(self.items)):
            frame = item[frame_key]
            if self.type == "faces":
                frame = self.alignments.get_full_frame_name(frame)
            retval = item[retval_key]

            if self.alignments.frame_has_multiple_faces(frame):
                yield retval

    def get_missing_alignments(self):
        """ yield each frame that does not exist in alignments file """
        self.output_message = "Frames missing from alignments file"
        exclude_filetypes = ["yaml", "yml", "p", "json", "txt"]
        for frame in tqdm(self.items, total=len(self.items)):
            frame_name = frame["frame_fullname"]
            if (frame["frame_extension"] not in exclude_filetypes
                    and not self.alignments.frame_exists(frame_name)):
                yield frame_name

    def get_missing_frames(self):
        """ yield each frame in alignments that does
            not have a matching file """
        self.output_message = "Missing frames that are in alignments file"
        frames = [item["frame_fullname"] for item in self.items]
        for frame in tqdm(self.alignments.data.keys(),
                          total=len(self.alignments.frames_count)):
            if frame not in frames:
                yield frame

    def get_leftover_faces(self):
        """yield each face that isn't in the alignments file."""
        self.output_message = "Faces missing from the alignments file"
        for face in tqdm(self.items, total=len(self.items)):
            frame = self.alignments.get_full_frame_name(face["frame_name"])
            alignment_faces = self.alignments.count_faces_in_frame(frame)

            if alignment_faces <= face["face_index"]:
                yield face["face_fullname"]

    def output_results(self, items_output):
        """ Output the results in the requested format """
        if not items_output:
            logger.info("No %s were found meeting the criteria", self.type)
            return
        if self.output == "move":
            self.move_file(items_output)
            return
        output_message = "-----------------------------------------------\r\n"
        output_message += " {} ({})\r\n".format(self.output_message,
                                                len(items_output))
        output_message += "-----------------------------------------------\r\n"
        output_message += "\r\n".join([frame for frame in items_output])
        if self.output == "console":
            print("\n" + output_message)
        if self.output == "file":
            self.output_file(output_message, len(items_output))

    def output_file(self, output_message, items_discovered):
        """ Save the output to a text file in the frames directory """
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_message.replace(" ", "_").lower()
        filename += "_" + now + ".txt"
        output_file = os.path.join(self.source_dir, filename)
        logger.info("Saving %s result(s) to '%s'", items_discovered, output_file)
        with open(output_file, "w") as f_output:
            f_output.write(output_message)

    def move_file(self, items_output):
        """ Move the identified frames to a new subfolder """
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = self.output_message.replace(" ", "_").lower()
        folder_name += "_" + now
        output_folder = os.path.join(self.source_dir, folder_name)
        os.makedirs(output_folder)
        move = getattr(self, "move_{}".format(self.type))
        move(output_folder, items_output)

    def move_frames(self, output_folder, items_output):
        """ Move frames into single subfolder """
        logger.info("Moving %s frame(s) to '%s'", len(items_output), output_folder)
        for frame in items_output:
            src = os.path.join(self.source_dir, frame)
            dst = os.path.join(output_folder, frame)
            os.rename(src, dst)

    def move_faces(self, output_folder, items_output):
        """ Make additional subdirs for each face that appears
            Enables easier manual sorting """
        logger.info("Moving %s faces(s) to '%s'", len(items_output), output_folder)
        for frame in items_output:
            idx = frame[frame.rfind("_") + 1:frame.rfind(".")]
            src = os.path.join(self.source_dir, frame)
            dst_folder = os.path.join(output_folder, idx)
            if not os.path.isdir(dst_folder):
                os.makedirs(dst_folder)
            dst = os.path.join(dst_folder, frame)
            os.rename(src, dst)


class Draw():
    """ Draw Alignments on passed in images """
    def __init__(self, alignments, arguments):
        self.arguments = arguments
        self.alignments = alignments
        self.frames = Frames(arguments.frames_dir)
        self.output_folder = self.set_output()
        self.extracted_faces = None

    def set_output(self):
        """ Set the output folder path """
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = "drawn_landmarks_{}".format(now)
        output_folder = os.path.join(self.frames.folder, folder_name)
        os.makedirs(output_folder)
        return output_folder

    def process(self):
        """ Run the draw alignments process """
        legacy = Legacy(self.alignments, self.arguments,
                        frames=self.frames, child_process=True)
        legacy.process()

        logger.info("[DRAW LANDMARKS]")  # Tidy up cli output
        self.extracted_faces = ExtractedFaces(
            self.frames,
            self.alignments,
            align_eyes=self.arguments.align_eyes)
        frames_drawn = 0
        for frame in tqdm(self.frames.file_list_sorted,
                          desc="Drawing landmarks"):

            frame_name = frame["frame_fullname"]

            if not self.alignments.frame_exists(frame_name):
                logger.verbose("Skipping '%s' - Alignments not found", frame_name)
                continue

            self.annotate_image(frame_name)
            frames_drawn += 1
        logger.info("%s Frame(s) output", frames_drawn)

    def annotate_image(self, frame):
        """ Draw the alignments """
        alignments = self.alignments.get_faces_in_frame(frame)
        image = self.frames.load_image(frame)
        self.extracted_faces.get_faces_in_frame(frame)
        original_roi = [face.original_roi
                        for face in self.extracted_faces.faces]
        annotate = Annotate(image, alignments, original_roi)
        annotate.draw_bounding_box(1, 1)
        annotate.draw_extract_box(2, 1)
        annotate.draw_landmarks(3, 1)
        annotate.draw_landmarks_mesh(4, 1)

        image = annotate.image
        self.frames.save_image(self.output_folder, frame, image)


class Extract():
    """ Re-extract faces from source frames based on
        Alignment data """
    def __init__(self, alignments, arguments):
        self.alignments = alignments
        self.type = arguments.job.replace("extract-", "")
        self.faces_dir = arguments.faces_dir
        self.frames = Frames(arguments.frames_dir)
        self.extracted_faces = ExtractedFaces(self.frames,
                                              self.alignments,
                                              align_eyes=arguments.align_eyes)

    def process(self):
        """ Run extraction """
        logger.info("[EXTRACT FACES]")  # Tidy up cli output
        self.check_folder()
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
            logger.error(err)
            exit(0)
        logger.verbose("Creating output folder at '%s'", self.faces_dir)
        os.makedirs(self.faces_dir)

    def export_faces(self):
        """ Export the faces """
        extracted_faces = 0

        for frame in tqdm(self.frames.file_list_sorted,
                          desc="Saving extracted faces"):

            frame_name = frame["frame_fullname"]

            if not self.alignments.frame_exists(frame_name):
                logger.verbose("Skipping '%s' - Alignments not found", frame_name)
                continue
            extracted_faces += self.output_faces(frame)

        logger.info("%s face(s) extracted", extracted_faces)

    def output_faces(self, frame):
        """ Output the frame's faces to file """
        face_count = 0
        frame_fullname = frame["frame_fullname"]
        frame_name = frame["frame_name"]
        faces = self.select_valid_faces(frame_fullname)

        for idx, face in enumerate(faces):
            output = "{}_{}{}".format(frame_name, str(idx), ".png")
            self.frames.save_image(self.faces_dir, output, face.aligned_face)
            face_count += 1
        return face_count

    def select_valid_faces(self, frame):
        """ Return valid faces for extraction """
        faces = self.extracted_faces.get_faces_in_frame(frame)
        if self.type != "large":
            return faces
        valid_faces = list()
        sizes = self.extracted_faces.get_roi_size_for_frame(frame)
        for idx, size in enumerate(sizes):
            if size >= self.extracted_faces.size:
                valid_faces.append(faces[idx])
        return valid_faces


class Reformat():
    """ Reformat Alignment file """
    def __init__(self, alignments, arguments):
        self.alignments = alignments
        if self.alignments.file == "dfl":
            self.faces = Faces(arguments.faces_dir, dfl=True)

    def process(self):
        """ Run reformat """
        logger.info("[REFORMAT ALIGNMENTS]")  # Tidy up cli output
        if self.alignments.file == "dfl":
            self.alignments.data = self.load_dfl()
            self.alignments.file = self.alignments.get_location(
                self.faces.folder,
                "alignments")
        self.alignments.save()

    def load_dfl(self):
        """ Load alignments from DeepFaceLab and format for Faceswap """
        alignments = dict()
        for face in self.faces.file_list_sorted:
            if face["face_extension"] != ".png":
                logger.verbose("'%s' is not a png. Skipping", face["face_fullname"])
                continue

            fullpath = os.path.join(self.faces.folder, face["face_fullname"])
            dfl = self.get_dfl_alignment(fullpath)

            if not dfl:
                continue

            self.convert_dfl_alignment(dfl, alignments)
        return alignments

    @staticmethod
    def get_dfl_alignment(filename):
        """ Process the alignment of one face """
        with open(filename, "rb") as dfl:
            header = dfl.read(8)
            if header != b"\x89PNG\r\n\x1a\n":
                logger.error("No Valid PNG header: %s", filename)
                return None
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
                dfl.seek(chunk_length+12, os.SEEK_CUR)
            logger.error("Couldn't find DFL alignments: %s", filename)

    @staticmethod
    def convert_dfl_alignment(dfl_alignments, alignments):
        """ Add DFL Alignments to alignments in Faceswap format """
        sourcefile = dfl_alignments["source_filename"]
        if not alignments.get(sourcefile, None):
            alignments[sourcefile] = list()

        left, top, right, bottom = dfl_alignments["source_rect"]
        alignment = {"x": left,
                     "w": right - left,
                     "y": top,
                     "h": bottom - top,
                     "landmarksXY": dfl_alignments["source_landmarks"]}

        alignments[sourcefile].append(alignment)


class RemoveAlignments():
    """ Remove items from alignments file """
    def __init__(self, alignments, arguments):
        self.alignments = alignments
        self.type = arguments.job.replace("remove-", "")
        self.items = self.get_items(arguments)
        self.removed = set()

    def get_items(self, arguments):
        """ Set the correct items to process """
        retval = None
        if self.type == "frames":
            retval = list(Frames(arguments.frames_dir).items.keys())
        elif self.type == "faces":
            retval = Faces(arguments.faces_dir)
        return retval

    def process(self):
        """ run removal """
        logger.info("[REMOVE ALIGNMENTS DATA]")  # Tidy up cli output
        del_count = 0

        iterator = self.alignments.yield_faces
        if self.type == "frames":
            iterator = list(item[3] for item in iterator())

        for item in tqdm(iterator() if self.type == "faces" else iterator,
                         desc="Processing alignments file",
                         total=self.alignments.frames_count):
            task = getattr(self, "remove_{}".format(self.type))
            del_count += task(item)

        if del_count == 0:
            logger.info("No changes made to alignments file. Exiting")
            return

        logger.info("%s alignment(s) were removed from alignments file", del_count)
        self.alignments.save()

        if self.type == "faces":
            self.rename_faces()

    def remove_frames(self, item):
        """ Process to remove frames from an alignments file """
        if item in self.items:
            return 0
        del self.alignments.data[item]
        return 1

    def remove_faces(self, item):
        """ Process to remove faces from an alignments file """
        if self.faces_count_matches(item):
            return 0
        return self.remove_alignment(item)

    def faces_count_matches(self, item):
        """ Check the selected face exits """
        frame_name, number_alignments = item[0], item[2]
        number_faces = len(self.items.items.get(frame_name, list()))
        return bool(number_alignments in(0, number_faces))

    def remove_alignment(self, item):
        """ Remove the alignment from the alignments file """
        del_count = 0
        frame_name, alignments, number_alignments = item[:3]
        processor = self.alignments.yield_original_index_reverse
        for idx in processor(alignments, number_alignments):
            face_indexes = self.items.items.get(frame_name, [-1])
            if idx not in face_indexes:
                del alignments[idx]
                self.removed.add(frame_name)
                logger.verbose("Removed alignment data for image: '%s'"
                               "index: %s", frame_name, str(idx))
                del_count += 1
        return del_count

    def rename_faces(self):
        """ Rename the aligned faces to match their "
            new index in alignments file """
        current_frame = ""
        current_index = 0
        rename_count = 0
        for face in tqdm(self.items.file_list_sorted,
                         desc="Renaming aligned faces",
                         total=self.items.count):

            if face["frame_name"] not in self.removed:
                continue
            current_index, current_frame = self.set_image_index(
                current_index,
                current_frame,
                face["frame_name"])
            if current_index != face["face_index"]:
                rename_count += self.rename_file(face,
                                                 current_frame,
                                                 current_index)

            current_index += 1
        if rename_count == 0:
            logger.info("No files were renamed. Exiting")
            return
        logger.info("%s face(s) were renamed to match with alignments file", rename_count)

    @staticmethod
    def set_image_index(index, current, original):
        """ Set the current processing image and index """
        idx = 0 if current != original else index
        return idx, original

    def rename_file(self, face, frame_name, index):
        """ Rename the selected file """
        old_file = face["face_name"] + face["face_extension"]
        new_file = "{}_{}{}".format(frame_name,
                                    str(index),
                                    face["face_extension"])
        src = os.path.join(self.items.folder, old_file)
        dst = os.path.join(self.items.folder, new_file)
        os.rename(src, dst)
        logger.verbose("Renamed '%s' to '%s'", src, dst)
        return 1


class Legacy():
    """ Update legacy alignments:

        - Add frame dimensions
        - Rotate landmarks and bounding boxes on legacy alignments
        and remove the 'r' parameter """

    def __init__(self, alignments, arguments,
                 frames=None, child_process=False):
        self.alignments = alignments
        self.child_process = child_process
        self.frames = frames
        if not frames:
            self.frames = Frames(arguments.frames_dir)

    def process(self):
        """ Run the rotate alignments process """
        no_dims = self.alignments.get_legacy_no_dims()
        rotated = self.alignments.get_legacy_rotation()
        if self.child_process and not rotated and not no_dims:
            return
        logger.info("[UPDATE LEGACY LANDMARKS]")  # Tidy up cli output

        if no_dims:
            if self.child_process:
                logger.info("Legacy landmarks found. Adding frame dimensions...")
            self.add_dimensions(no_dims)

        if rotated:
            if self.child_process:
                logger.info("Legacy rotated frames found. Rotating landmarks")
            self.rotate_landmarks(rotated)

        self.alignments.save()

    def add_dimensions(self, no_dims):
        """ Add width and height of original frame to alignments """
        for no_dim in tqdm(no_dims, desc="Adding Frame Dimensions"):
            if no_dim not in self.frames.items.keys():
                continue
            dims = self.frames.load_image(no_dim).shape[:2]
            self.alignments.add_dimensions(no_dim, dims)

    def rotate_landmarks(self, rotated):
        """ Rotate the landmarks """
        for rotate_item in tqdm(rotated, desc="Rotating Landmarks"):
            if rotate_item not in self.frames.items.keys():
                continue
            self.alignments.rotate_existing_landmarks(rotate_item)


class Sort():
    """ Sort alignments' index by the order they appear in
        an image """
    def __init__(self, alignments, arguments):
        self.alignments = alignments
        self.axis = arguments.job.replace("sort-", "")
        self.faces = self.get_faces(arguments)

    @staticmethod
    def get_faces(arguments):
        """ If faces argument is specified, load faces_dir
            otherwise return None """
        if not hasattr(arguments, "faces_dir") or not arguments.faces_dir:
            return None
        return Faces(arguments.faces_dir)

    def process(self):
        """ Execute the sort process """
        logger.info("[SORT INDEXES]")  # Tidy up cli output
        self.check_rotated()
        self.reindex_faces()
        self.alignments.save()

    def check_rotated(self):
        """ Legacy rotated alignments will not have the correct x, y
            positions, so generate a warning and exit """
        if any(alignment.get("r", None)
               for val in self.alignments.data.values()
               for alignment in val):
            logger.error("There are rotated frames in the alignments "
                         "file. Position of faces will not be correctly "
                         "calculated for these frames. You should run rotation "
                         "tool to update the file prior to running sort: "
                         "'python tools.py alignments -j rotate -a "
                         "<alignments_file> -fr <frames_folder>'")
            exit(0)

    def reindex_faces(self):
        """ Re-Index the faces """
        reindexed = 0
        for alignment in tqdm(self.alignments.yield_faces(),
                              desc="Sort alignment indexes",
                              total=self.alignments.frames_count):
            frame, alignments, count, key = alignment
            if count <= 1:
                continue
            sorted_alignments = sorted([item for item in alignments],
                                       key=lambda x: (x[self.axis]))
            if sorted_alignments == alignments:
                continue
            map_faces = self.map_face_names(alignments,
                                            sorted_alignments,
                                            frame)
            self.rename_faces(map_faces)
            self.alignments.data[key] = sorted_alignments
            reindexed += 1
        logger.info("%s Frames had their faces reindexed", reindexed)

    def map_face_names(self, alignments, sorted_alignments, frame):
        """ Map the old and new indexes for face renaming """
        map_faces = list()
        if not self.faces:
            return map_faces
        for idx, alignment in enumerate(alignments):
            idx_new = sorted_alignments.index(alignment)
            mapping = [{"old_name": face["face_fullname"],
                        "new_name": "{}_{}{}".format(frame,
                                                     idx_new,
                                                     face["face_extension"])}
                       for face in self.faces.file_list_sorted
                       if face["frame_name"] == frame
                       and face["face_index"] == idx]
            if not mapping:
                logger.warning("No face image found for frame '%s' at index %s", frame, idx)
            map_faces.extend(mapping)
        return map_faces

    def rename_faces(self, map_faces):
        """ Rename faces
            Done in 2 iterations as two files cannot share the same name """
        temp_ext = ".temp_move"
        for action in ("temp", "final"):
            for face in map_faces:
                old = face["old_name"]
                new = face["new_name"]
                if old == new:
                    continue
                old_file = old if action == "temp" else old + temp_ext
                new_file = old + temp_ext if action == "temp" else new
                src = os.path.join(self.faces.folder, old_file)
                dst = os.path.join(self.faces.folder, new_file)
                os.rename(src, dst)
                if action == "final":
                    logger.verbose("Renamed '%s' to '%s'", old, new)


class Spatial():
    """ Apply spatial temporal filtering to landmarks
        Adapted from:
        https://www.kaggle.com/selfishgene/animating-and-smoothing-3d-facial-keypoints/notebook """

    def __init__(self, alignments, arguments):
        self.arguments = arguments
        self.alignments = alignments
        self.mappings = dict()
        self.normalized = dict()
        self.shapes_model = None

    def process(self):
        """ Perform spatial filtering """
        logger.info("[SPATIO-TEMPORAL FILTERING]")  # Tidy up cli output
        logger.info("NB: The process only processes the alignments for the first "
                    "face it finds for any given frame. For best results only run this when "
                    "there is only a single face in the alignments file and all false positives "
                    "have been removed")

        self.normalize()
        self.shape_model()
        landmarks = self.spatially_filter()
        landmarks = self.temporally_smooth(landmarks)
        self.update_alignments(landmarks)
        self.alignments.save()

        logger.info("Done! To re-extract faces run: python tools.py "
                    "alignments -j extract -a %s -fr <path_to_frames_dir> -fc "
                    "<output_folder>", self.arguments.alignments_file)

    # define shape normalization utility functions
    @staticmethod
    def normalize_shapes(shapes_im_coords):
        """ Normalize a 2D or 3D shape """
        (num_pts, num_dims, _) = shapes_im_coords.shape

        # calc mean coords and subtract from shapes
        mean_coords = shapes_im_coords.mean(axis=0)
        shapes_centered = np.zeros(shapes_im_coords.shape)
        shapes_centered = shapes_im_coords - np.tile(mean_coords,
                                                     [num_pts, 1, 1])

        # calc scale factors and divide shapes
        scale_factors = np.sqrt((shapes_centered**2).sum(axis=1)).mean(axis=0)
        shapes_normalized = np.zeros(shapes_centered.shape)
        shapes_normalized = shapes_centered / np.tile(scale_factors,
                                                      [num_pts, num_dims, 1])

        return shapes_normalized, scale_factors, mean_coords

    @staticmethod
    def normalized_to_original(shapes_normalized, scale_factors, mean_coords):
        """ Transform a normalized shape back to original image coordinates """
        (num_pts, num_dims, _) = shapes_normalized.shape

        # move back to the correct scale
        shapes_centered = shapes_normalized * np.tile(scale_factors,
                                                      [num_pts, num_dims, 1])
        # move back to the correct location
        shapes_im_coords = shapes_centered + np.tile(mean_coords,
                                                     [num_pts, 1, 1])

        return shapes_im_coords

    def normalize(self):
        """ Compile all original and normalized alignments """
        count = sum(1 for val in self.alignments.data.values() if val)
        landmarks_all = np.zeros((68, 2, int(count)))

        end = 0
        for key in tqdm(sorted(self.alignments.data.keys()),
                        desc="Compiling"):
            val = self.alignments.data[key]
            if not val:
                continue
            # We should only be normalizing a single face, so just take
            # the first landmarks found
            landmarks = np.array(val[0]["landmarksXY"]).reshape(68, 2, 1)
            start = end
            end = start + landmarks.shape[2]
            # store in one big array
            landmarks_all[:, :, start:end] = landmarks
            # make sure we keep track of the mapping to the original frame
            self.mappings[start] = key

        # normalize shapes
        normalized_shape = self.normalize_shapes(landmarks_all)
        self.normalized["landmarks"] = normalized_shape[0]
        self.normalized["scale_factors"] = normalized_shape[1]
        self.normalized["mean_coords"] = normalized_shape[2]

    def shape_model(self):
        """ build 2D shape model """
        landmarks_norm = self.normalized["landmarks"]
        num_components = 20
        normalized_shapes_tbl = np.reshape(landmarks_norm,
                                           [68*2, landmarks_norm.shape[2]]).T
        self.shapes_model = decomposition.PCA(
            n_components=num_components,
            whiten=True,
            random_state=1).fit(normalized_shapes_tbl)
        explained = self.shapes_model.explained_variance_ratio_.sum()
        logger.info("Total explained percent by PCA model with %s components is %s%%",
                    num_components, round(100 * explained, 1))

    def spatially_filter(self):
        """ interpret the shapes using our shape model
            (project and reconstruct) """
        landmarks_norm = self.normalized["landmarks"]
        # convert to matrix form
        landmarks_norm_table = np.reshape(landmarks_norm,
                                          [68 * 2, landmarks_norm.shape[2]]).T
        # project onto shapes model and reconstruct
        landmarks_norm_table_rec = self.shapes_model.inverse_transform(
            self.shapes_model.transform(landmarks_norm_table))
        # convert back to shapes (numKeypoint, num_dims, numFrames)
        landmarks_norm_rec = np.reshape(landmarks_norm_table_rec.T,
                                        [68, 2, landmarks_norm.shape[2]])
        # transform back to image coords
        return self.normalized_to_original(landmarks_norm_rec,
                                           self.normalized["scale_factors"],
                                           self.normalized["mean_coords"])

    @staticmethod
    def temporally_smooth(landmarks):
        """ apply temporal filtering on the 2D points """
        filter_half_length = 2
        temporal_filter = np.ones((1, 1, 2*filter_half_length+1))
        temporal_filter = temporal_filter / temporal_filter.sum()

        start_tileblock = np.tile(landmarks[:, :, 0][:, :, np.newaxis],
                                  [1, 1, filter_half_length])
        end_tileblock = np.tile(landmarks[:, :, -1][:, :, np.newaxis],
                                [1, 1, filter_half_length])
        landmarks_padded = np.dstack((start_tileblock,
                                      landmarks,
                                      end_tileblock))
        return signal.convolve(landmarks_padded, temporal_filter,
                               mode='valid', method='fft')

    def update_alignments(self, landmarks):
        """ Update smoothed landmarks back to alignments """
        for idx, frame in tqdm(self.mappings.items(), desc="Updating"):
            landmarks_update = landmarks[:, :, idx].astype(int)
            landmarks_xy = landmarks_update.reshape(68, 2).tolist()
            self.alignments.data[frame][0]["landmarksXY"] = landmarks_xy
