#!/usr/bin/env python3
""" Tools for manipulating the alignments serialized file """

import logging
import os
import pickle
import struct
from datetime import datetime
from PIL import Image

import numpy as np
from scipy import signal
from sklearn import decomposition
from tqdm import tqdm

from . import Annotate, ExtractedFaces, Faces, Frames

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Check():
    """ Frames and faces checking tasks """
    def __init__(self, alignments, arguments):
        logger.debug("Initializing %s: (arguments: %s)", self.__class__.__name__, arguments)
        self.alignments = alignments
        self.job = arguments.job
        self.type = None
        self.is_video = False  # Set when getting items
        self.output = arguments.output
        self.source_dir = self.get_source_dir(arguments)
        self.validate()
        self.items = self.get_items()

        self.output_message = ""
        logger.debug("Initialized %s", self.__class__.__name__)

    def get_source_dir(self, arguments):
        """ Set the correct source folder """
        if (hasattr(arguments, "faces_dir") and arguments.faces_dir and
                hasattr(arguments, "frames_dir") and arguments.frames_dir):
            logger.error("Only select a source frames (-fr) or source faces (-fc) folder")
            exit(0)
        elif hasattr(arguments, "faces_dir") and arguments.faces_dir:
            self.type = "faces"
            source_dir = arguments.faces_dir
        elif hasattr(arguments, "frames_dir") and arguments.frames_dir:
            self.type = "frames"
            source_dir = arguments.frames_dir
        else:
            logger.error("No source folder (-fr or -fc) was provided")
            exit(0)
        logger.debug("type: '%s', source_dir: '%s'", self.type, source_dir)
        return source_dir

    def get_items(self):
        """ Set the correct items to process """
        items = globals()[self.type.title()](self.source_dir)
        self.is_video = items.is_video
        return items.file_list_sorted

    def process(self):
        """ Process the frames check against the alignments file """
        logger.info("[CHECK %s]", self.type.upper())
        items_output = self.compile_output()
        self.output_results(items_output)

    def validate(self):
        """ Check that the selected type is valid for
            selected task and job """
        if self.job == "missing-frames" and self.output == "move":
            logger.warning("Missing_frames was selected with move output, but there will "
                           "be nothing to move. Defaulting to output: console")
            self.output = "console"
        if self.type == "faces" and self.job not in ("multi-faces", "leftover-faces"):
            logger.warning("The selected folder is not valid. Faces folder (-fc) is only "
                           "supported for 'multi-faces' and 'leftover-faces'")
            exit(0)

    def compile_output(self):
        """ Compile list of frames that meet criteria """
        action = self.job.replace("-", "_")
        processor = getattr(self, "get_{}".format(action))
        logger.debug("Processor: %s", processor)
        return [item for item in processor()]

    def get_no_faces(self):
        """ yield each frame that has no face match in alignments file """
        self.output_message = "Frames with no faces"
        for frame in tqdm(self.items, desc=self.output_message):
            logger.trace(frame)
            frame_name = frame["frame_fullname"]
            if not self.alignments.frame_has_faces(frame_name):
                logger.debug("Returning: '%s'", frame_name)
                yield frame_name

    def get_multi_faces(self):
        """ yield each frame or face that has multiple faces
            matched in alignments file """
        process_type = getattr(self, "get_multi_faces_{}".format(self.type))
        for item in process_type():
            yield item

    def get_multi_faces_frames(self):
        """ Return Frames that contain multiple faces """
        self.output_message = "Frames with multiple faces"
        for item in tqdm(self.items, desc=self.output_message):
            filename = item["frame_fullname"]
            if not self.alignments.frame_has_multiple_faces(filename):
                continue
            logger.trace("Returning: '%s'", filename)
            yield filename

    def get_multi_faces_faces(self):
        """ Return Faces when there are multiple faces in a frame """
        self.output_message = "Multiple faces in frame"
        seen_hash_dupes = set()
        for item in tqdm(self.items, desc=self.output_message):
            filename = item["face_fullname"]
            f_hash = item["face_hash"]
            frame_idx = [(frame, idx)
                         for frame, idx in self.alignments.hashes_to_frame[f_hash].items()]

            if len(frame_idx) > 1:
                # If the same hash exists in multiple frames, select arbitrary frame
                # and add to seen_hash_dupes so it is not selected again
                logger.trace("Dupe hashes: %s", frame_idx)
                frame_idx = [f_i for f_i in frame_idx if f_i not in seen_hash_dupes][0]
                seen_hash_dupes.add(frame_idx)
                frame_idx = [frame_idx]

            frame_name, idx = frame_idx[0]
            if not self.alignments.frame_has_multiple_faces(frame_name):
                continue
            retval = (filename, idx)
            logger.trace("Returning: '%s'", retval)
            yield retval

    def get_missing_alignments(self):
        """ yield each frame that does not exist in alignments file """
        self.output_message = "Frames missing from alignments file"
        exclude_filetypes = set(["yaml", "yml", "p", "json", "txt"])
        for frame in tqdm(self.items, desc=self.output_message):
            frame_name = frame["frame_fullname"]
            if (frame["frame_extension"] not in exclude_filetypes
                    and not self.alignments.frame_exists(frame_name)):
                logger.debug("Returning: '%s'", frame_name)
                yield frame_name

    def get_missing_frames(self):
        """ yield each frame in alignments that does
            not have a matching file """
        self.output_message = "Missing frames that are in alignments file"
        frames = set(item["frame_fullname"] for item in self.items)
        for frame in tqdm(self.alignments.data.keys(), desc=self.output_message):
            if frame not in frames:
                logger.debug("Returning: '%s'", frame)
                yield frame

    def get_leftover_faces(self):
        """yield each face that isn't in the alignments file."""
        self.output_message = "Faces missing from the alignments file"
        for face in tqdm(self.items, desc=self.output_message):
            f_hash = face["face_hash"]
            if f_hash not in self.alignments.hashes_to_frame:
                logger.debug("Returning: '%s'", face["face_fullname"])
                yield face["face_fullname"], -1

    def output_results(self, items_output):
        """ Output the results in the requested format """
        logger.trace("items_output: %s", items_output)
        if self.output == "move" and self.is_video and self.type == "frames":
            logger.warning("Move was selected with an input video. This is not possible so "
                           "falling back to console output")
            self.output = "console"
        if not items_output:
            logger.info("No %s were found meeting the criteria", self.type)
            return
        if self.output == "move":
            self.move_file(items_output)
            return
        if self.job in ("multi-faces", "leftover-faces") and self.type == "faces":
            # Strip the index for printed/file output
            items_output = [item[0] for item in items_output]
        output_message = "-----------------------------------------------\r\n"
        output_message += " {} ({})\r\n".format(self.output_message,
                                                len(items_output))
        output_message += "-----------------------------------------------\r\n"
        output_message += "\r\n".join([frame for frame in items_output])
        if self.output == "console":
            for line in output_message.splitlines():
                logger.info(line)
        if self.output == "file":
            self.output_file(output_message, len(items_output))

    def get_output_folder(self):
        """ Return output folder. Needs to be in the root if input is a
            video and processing frames """
        if self.is_video and self.type == "frames":
            return os.path.dirname(self.source_dir)
        return self.source_dir

    def get_filename_prefix(self):
        """ Video name needs to be prefixed to filename if input is a
            video and processing frames """
        if self.is_video and self.type == "frames":
            return "{}_".format(os.path.basename(self.source_dir))
        return ""

    def output_file(self, output_message, items_discovered):
        """ Save the output to a text file in the frames directory """
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst_dir = self.get_output_folder()
        filename = "{}{}_{}.txt".format(self.get_filename_prefix(),
                                        self.output_message.replace(" ", "_").lower(),
                                        now)
        output_file = os.path.join(dst_dir, filename)
        logger.info("Saving %s result(s) to '%s'", items_discovered, output_file)
        with open(output_file, "w") as f_output:
            f_output.write(output_message)

    def move_file(self, items_output):
        """ Move the identified frames to a new subfolder """
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = "{}{}_{}".format(self.get_filename_prefix(),
                                       self.output_message.replace(" ", "_").lower(), now)
        dst_dir = self.get_output_folder()
        output_folder = os.path.join(dst_dir, folder_name)
        logger.debug("Creating folder: '%s'", output_folder)
        os.makedirs(output_folder)
        move = getattr(self, "move_{}".format(self.type))
        logger.debug("Move function: %s", move)
        move(output_folder, items_output)

    def move_frames(self, output_folder, items_output):
        """ Move frames into single subfolder """
        logger.info("Moving %s frame(s) to '%s'", len(items_output), output_folder)
        for frame in items_output:
            src = os.path.join(self.source_dir, frame)
            dst = os.path.join(output_folder, frame)
            logger.debug("Moving: '%s' to '%s'", src, dst)
            os.rename(src, dst)

    def move_faces(self, output_folder, items_output):
        """ Make additional subfolders for each face that appears
            Enables easier manual sorting """
        logger.info("Moving %s faces(s) to '%s'", len(items_output), output_folder)
        for frame, idx in items_output:
            src = os.path.join(self.source_dir, frame)
            dst_folder = os.path.join(output_folder, str(idx)) if idx != -1 else output_folder
            if not os.path.isdir(dst_folder):
                logger.debug("Creating folder: '%s'", dst_folder)
                os.makedirs(dst_folder)
            dst = os.path.join(dst_folder, frame)
            logger.debug("Moving: '%s' to '%s'", src, dst)
            os.rename(src, dst)


class Dfl():
    """ Reformat Alignment file """
    def __init__(self, alignments, arguments):
        logger.debug("Initializing %s: (arguments: %s)", self.__class__.__name__, arguments)
        self.alignments = alignments
        if self.alignments.file != "dfl.fsa":
            logger.error("Alignments file must be specified as 'dfl' to reformat dfl alignmnets")
            exit(0)
        logger.debug("Loading DFL faces")
        self.faces = Faces(arguments.faces_dir)
        logger.debug("Initialized %s", self.__class__.__name__)

    def process(self):
        """ Run reformat """
        logger.info("[REFORMAT DFL ALIGNMENTS]")  # Tidy up cli output
        self.alignments.data = self.load_dfl()
        self.alignments.file = self.alignments.get_location(self.faces.folder, "alignments")
        self.alignments.save()

    def load_dfl(self):
        """ Load alignments from DeepFaceLab and format for Faceswap """
        alignments = dict()
        for face in tqdm(self.faces.file_list_sorted, desc="Converting DFL Faces"):
            if face["face_extension"] not in (".png", ".jpg"):
                logger.verbose("'%s' is not a png or jpeg. Skipping", face["face_fullname"])
                continue
            f_hash = face["face_hash"]
            fullpath = os.path.join(self.faces.folder, face["face_fullname"])
            dfl = self.get_dfl_alignment(fullpath)

            if not dfl:
                continue

            self.convert_dfl_alignment(dfl, f_hash, alignments)
        return alignments

    @staticmethod
    def get_dfl_alignment(filename):
        """ Process the alignment of one face """
        ext = os.path.splitext(filename)[1]

        if ext.lower() in (".jpg", ".jpeg"):
            img = Image.open(filename)
            try:
                dfl_alignments = pickle.loads(img.app["APP15"])
                dfl_alignments["source_rect"] = [n.item()  # comes as non-JSONable np.int32
                                                 for n in dfl_alignments["source_rect"]]
                return dfl_alignments
            except pickle.UnpicklingError:
                return None

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
                    retval = pickle.loads(chunk[8:-4])
                    logger.trace("Loaded DFL Alignment: (filename: '%s', alignment: %s",
                                 filename, retval)
                    return retval
                dfl.seek(chunk_length+12, os.SEEK_CUR)
            logger.error("Couldn't find DFL alignments: %s", filename)

    @staticmethod
    def convert_dfl_alignment(dfl_alignments, f_hash, alignments):
        """ Add DFL Alignments to alignments in Faceswap format """
        sourcefile = dfl_alignments["source_filename"]
        left, top, right, bottom = dfl_alignments["source_rect"]
        alignment = {"x": left,
                     "w": right - left,
                     "y": top,
                     "h": bottom - top,
                     "hash": f_hash,
                     "landmarks_xy": np.array(dfl_alignments["source_landmarks"], dtype="float32")}
        logger.trace("Adding alignment: (frame: '%s', alignment: %s", sourcefile, alignment)
        alignments.setdefault(sourcefile, list()).append(alignment)


class Draw():
    """ Draw Alignments on passed in images """
    def __init__(self, alignments, arguments):
        logger.debug("Initializing %s: (arguments: %s)", self.__class__.__name__, arguments)
        self.arguments = arguments
        self.alignments = alignments
        self.frames = Frames(arguments.frames_dir)
        self.output_folder = self.set_output()
        self.extracted_faces = None
        logger.debug("Initialized %s", self.__class__.__name__)

    def set_output(self):
        """ Set the output folder path """
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = "drawn_landmarks_{}".format(now)
        if self.frames.is_video:
            dest_folder = os.path.dirname(self.frames.folder)
        else:
            dest_folder = self.frames.folder
        output_folder = os.path.join(dest_folder, folder_name)
        logger.debug("Creating folder: '%s'", output_folder)
        os.makedirs(output_folder)
        return output_folder

    def process(self):
        """ Run the draw alignments process """
        logger.info("[DRAW LANDMARKS]")  # Tidy up cli output
        self.extracted_faces = ExtractedFaces(self.frames, self.alignments, size=256)
        frames_drawn = 0
        for frame in tqdm(self.frames.file_list_sorted, desc="Drawing landmarks"):
            frame_name = frame["frame_fullname"]

            if not self.alignments.frame_exists(frame_name):
                logger.verbose("Skipping '%s' - Alignments not found", frame_name)
                continue

            self.annotate_image(frame_name)
            frames_drawn += 1
        logger.info("%s Frame(s) output", frames_drawn)

    def annotate_image(self, frame):
        """ Draw the alignments """
        logger.trace("Annotating frame: '%s'", frame)
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
        logger.debug("Initializing %s: (arguments: %s)", self.__class__.__name__, arguments)
        self.arguments = arguments
        self.alignments = alignments
        self.faces_dir = arguments.faces_dir
        self.frames = Frames(arguments.frames_dir)
        self.extracted_faces = ExtractedFaces(self.frames,
                                              self.alignments,
                                              size=arguments.size,
                                              align_eyes=arguments.align_eyes)
        logger.debug("Initialized %s", self.__class__.__name__)

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
        elif not os.path.isdir(self.faces_dir):
            logger.debug("Creating folder: '%s'", self.faces_dir)
            os.makedirs(self.faces_dir)
        elif os.listdir(self.faces_dir):
            err = "ERROR: Output faces folder should be empty: '{}'".format(self.faces_dir)
        if err:
            logger.error(err)
            exit(0)
        logger.verbose("Creating output folder at '%s'", self.faces_dir)

    def export_faces(self):
        """ Export the faces """
        extracted_faces = 0
        skip_num = self.arguments.extract_every_n
        if skip_num != 1:
            logger.info("Skipping every %s frames", skip_num)
        for idx, frame in enumerate(tqdm(self.frames.file_list_sorted,
                                         desc="Saving extracted faces")):
            frame_name = frame["frame_fullname"]
            if idx % skip_num != 0:
                logger.trace("Skipping '%s' due to extract_every_n = %s", frame_name, skip_num)
                continue

            if not self.alignments.frame_exists(frame_name):
                logger.verbose("Skipping '%s' - Alignments not found", frame_name)
                continue

            extracted_faces += self.output_faces(frame)

        if extracted_faces != 0 and not self.arguments.large:
            self.alignments.save()
        logger.info("%s face(s) extracted", extracted_faces)

    def output_faces(self, frame):
        """ Output the frame's faces to file """
        logger.trace("Outputting frame: %s", frame)
        face_count = 0
        frame_fullname = frame["frame_fullname"]
        frame_name = frame["frame_name"]
        extension = os.path.splitext(frame_fullname)[1]
        faces = self.select_valid_faces(frame_fullname)

        for idx, face in enumerate(faces):
            output = "{}_{}{}".format(frame_name, str(idx), extension)
            if self.arguments.large:
                self.frames.save_image(self.faces_dir, output, face.aligned_face)
            else:
                output = os.path.join(self.faces_dir, output)
                f_hash = self.extracted_faces.save_face_with_hash(output,
                                                                  extension,
                                                                  face.aligned_face)
                self.alignments.data[frame_fullname][idx]["hash"] = f_hash
            face_count += 1
        return face_count

    def select_valid_faces(self, frame):
        """ Return valid faces for extraction """
        faces = self.extracted_faces.get_faces_in_frame(frame)
        if not self.arguments.large:
            valid_faces = faces
        else:
            sizes = self.extracted_faces.get_roi_size_for_frame(frame)
            valid_faces = [faces[idx] for idx, size in enumerate(sizes)
                           if size >= self.extracted_faces.size]
        logger.trace("frame: '%s', total_faces: %s, valid_faces: %s",
                     frame, len(faces), len(valid_faces))
        return valid_faces


class Merge():
    """ Merge two alignments files into one """
    def __init__(self, alignments, arguments):
        self.alignments = alignments
        self.faces = self.get_faces(arguments)
        self.final_alignments = alignments[0]
        self.process_alignments = alignments[1:]
        self._hashes_to_frame = None

    @staticmethod
    def get_faces(arguments):
        """ If faces argument is specified, load faces_dir
            otherwise return None """
        if not hasattr(arguments, "faces_dir") or not arguments.faces_dir:
            return None
        return Faces(arguments.faces_dir)

    def process(self):
        """Process the alignments file merge """
        logger.info("[MERGE ALIGNMENTS]")  # Tidy up cli output
        if self.faces is not None:
            self.remove_faces()
        self._hashes_to_frame = self.final_alignments.hashes_to_frame
        skip_count = 0
        merge_count = 0
        total_count = sum([alignments.frames_count for alignments in self.process_alignments])

        with tqdm(desc="Merging Alignments", total=total_count) as pbar:
            for alignments in self.process_alignments:
                for _, src_alignments, _, frame in alignments.yield_faces():
                    for idx, alignment in enumerate(src_alignments):
                        if not alignment.get("hash", None):
                            logger.warning("Alignment '%s':%s has no Hash! Skipping", frame, idx)
                            skip_count += 1
                            continue
                        if self.check_exists(frame, alignment, idx):
                            skip_count += 1
                            continue
                        self.merge_alignment(frame, alignment, idx)
                        merge_count += 1
                    pbar.update(1)
        logger.info("Alignments Merged: %s", merge_count)
        logger.info("Alignments Skipped: %s", skip_count)
        if merge_count != 0:
            self.set_destination_filename()
            self.final_alignments.save()

    def remove_faces(self):
        """ Process to remove faces from an alignments file """
        face_hashes = list(self.faces.items.keys())
        del_faces_count = 0
        del_frames_count = 0
        if not face_hashes:
            logger.error("No face hashes. This would remove all faces from your alignments file.")
            return
        for alignments in tqdm(self.alignments, desc="Filtering out faces"):
            pre_face_count = alignments.faces_count
            pre_frames_count = alignments.frames_count
            alignments.filter_hashes(face_hashes, filter_out=False)
            # Remove frames with no faces
            frames = list(alignments.data.keys())
            for frame in frames:
                if not alignments.frame_has_faces(frame):
                    del alignments.data[frame]
            post_face_count = alignments.faces_count
            post_frames_count = alignments.frames_count
            removed_faces = pre_face_count - post_face_count
            removed_frames = pre_frames_count - post_frames_count
            del_faces_count += removed_faces
            del_frames_count += removed_frames
            logger.verbose("Removed %s faces and %s frames from %s",
                           removed_faces, removed_frames, os.path.basename(alignments.file))
        logger.info("Total removed - faces: %s, frames: %s", del_faces_count, del_frames_count)

    def check_exists(self, frame, alignment, idx):
        """ Check whether this face already exists """
        existing_frame = self._hashes_to_frame.get(alignment["hash"], None)
        if not existing_frame:
            return False
        if frame in existing_frame.keys():
            logger.verbose("Face '%s': %s already exists in destination at position %s. "
                           "Skipping", frame, idx, existing_frame[frame])
        elif frame not in existing_frame.keys():
            logger.verbose("Face '%s': %s exists in destination as: %s. "
                           "Skipping", frame, idx, existing_frame)
        return True

    def merge_alignment(self, frame, alignment, idx):
        """ Merge the source alignment into the destination """
        logger.debug("Merging alignment: (frame: %s, src_idx: %s, hash: %s)",
                     frame, idx, alignment["hash"])
        self._hashes_to_frame.setdefault(alignment["hash"], dict())[frame] = idx
        self.final_alignments.data.setdefault(frame, list()).append(alignment)

    def set_destination_filename(self):
        """ Set the destination filename """
        folder = os.path.split(self.final_alignments.file)[0]
        ext = os.path.splitext(self.final_alignments.file)[1]
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(folder, "alignments_merged_{}{}".format(now, ext))
        logger.debug("Output set to: '%s'", filename)
        self.final_alignments.file = filename


class RemoveAlignments():
    """ Remove items from alignments file """
    def __init__(self, alignments, arguments):
        logger.debug("Initializing %s: (arguments: %s)", self.__class__.__name__, arguments)
        self.alignments = alignments
        self.type = arguments.job.replace("remove-", "")
        self.items = self.get_items(arguments)
        self.removed = set()
        logger.debug("Initialized %s", self.__class__.__name__)

    def get_items(self, arguments):
        """ Set the correct items to process """
        retval = None
        if self.type == "frames":
            retval = Frames(arguments.frames_dir).items
        elif self.type == "faces":
            retval = Faces(arguments.faces_dir)
        return retval

    def process(self):
        """ run removal """
        logger.info("[REMOVE ALIGNMENTS DATA]")  # Tidy up cli output
        del_count = 0
        task = getattr(self, "remove_{}".format(self.type))

        if self.type == "frames":
            logger.debug("Removing Frames")
            for frame in tqdm(list(item[3] for item in self.alignments.yield_faces()),
                              desc="Removing Frames",
                              total=self.alignments.frames_count):
                del_count += task(frame)
        else:
            logger.debug("Removing Faces")
            del_count = task()

        if del_count == 0:
            logger.info("No changes made to alignments file. Exiting")
            return

        logger.info("%s alignment(s) were removed from alignments file", del_count)
        self.alignments.save()

        if self.type == "faces":
            rename = Rename(self.alignments, None, self.items)
            rename.process()

    def remove_frames(self, frame):
        """ Process to remove frames from an alignments file """
        if frame in self.items:
            logger.trace("Not deleting frame: '%s'", frame)
            return 0
        logger.debug("Deleting frame: '%s'", frame)
        del self.alignments.data[frame]
        return 1

    def remove_faces(self):
        """ Process to remove faces from an alignments file """
        face_hashes = self.items.items
        if not face_hashes:
            logger.error("No face hashes. This would remove all faces from your alignments file.")
            return 0
        pre_face_count = self.alignments.faces_count
        self.alignments.filter_hashes(face_hashes, filter_out=False)
        post_face_count = self.alignments.faces_count
        return pre_face_count - post_face_count


class Rename():
    """ Rename faces to match their source frame and position index """
    def __init__(self, alignments, arguments, faces=None):
        logger.debug("Initializing %s: (arguments: %s, faces: %s)",
                     self.__class__.__name__, arguments, faces)
        self.alignments = alignments
        self.faces = faces if faces else Faces(arguments.faces_dir)
        self.seen_multihash = set()
        logger.debug("Initialized %s", self.__class__.__name__)

    def process(self):
        """ Process the face renaming """
        logger.info("[RENAME FACES]")  # Tidy up cli output
        rename_count = 0
        for frame, details, _, frame_fullname in tqdm(self.alignments.yield_faces(),
                                                      desc="Renaming Faces",
                                                      total=self.alignments.frames_count):
            rename_count += self.rename_faces(frame, frame_fullname, details)
        logger.info("%s faces renamed", rename_count)

    def rename_faces(self, frame, frame_fullname, details):
        """ Rename faces
            Done in 2 iterations as two files cannot share the same name """
        logger.trace("Renaming faces for frame: '%s'", frame_fullname)
        temp_ext = ".temp_move"
        frame_faces = [(x["hash"], idx) for idx, x in enumerate(details)]
        rename_count = 0
        rename_files = list()
        for f_hash, idx in frame_faces:
            faces = self.faces.items[f_hash]
            if len(faces) == 1:
                face_name, face_ext = faces[0]
            else:
                face_name, face_ext = self.check_multi_hashes(faces, frame, idx)
            old = face_name + face_ext
            new = "{}_{}{}".format(frame, idx, face_ext)
            if old == new:
                logger.trace("Face does not require renaming: '%s'", old)
                continue
            rename_files.append((old, new))
        for action in ("temp", "final"):
            for files in rename_files:
                old, new = files
                old_file = old if action == "temp" else old + temp_ext
                new_file = old + temp_ext if action == "temp" else new
                src = os.path.join(self.faces.folder, old_file)
                dst = os.path.join(self.faces.folder, new_file)
                logger.trace("Renaming: '%s' to '%s'", old_file, new_file)
                os.rename(src, dst)
                if action == "final":
                    rename_count += 1
                    logger.verbose("Renamed '%s' to '%s'", old, new)
        return rename_count

    def check_multi_hashes(self, faces, frame, idx):
        """ Check filenames for where multiple faces have the
            same hash (e.g. for freeze frames) """
        logger.debug("Multiple hashes: (frame: faces: %s, frame: '%s', idx: %s", faces, frame, idx)
        frame_idx = "{}_{}".format(frame, idx)
        retval = None
        for face_name, extension in faces:
            if (face_name, extension) in self.seen_multihash:
                # Don't return a filename that has already been processed
                logger.debug("Already seen: %s", (face_name, extension))
                continue
            if face_name == frame_idx:
                # If a matching filename already exists return that
                retval = (face_name, extension)
                logger.debug("Matching filename found: %s", retval)
                self.seen_multihash.add(retval)
                break
            if face_name.startswith(frame):
                # If a matching framename already exists return that
                retval = (face_name, extension)
                logger.debug("Matching freamename found: %s", retval)
                self.seen_multihash.add(retval)
                break
        if not retval:
            # If no matches, just pop the first filename
            retval = [face for face in faces if face not in self.seen_multihash][0]
            logger.debug("No matches found. Choosing: %s", retval)
            self.seen_multihash.add(retval)
        logger.debug("Returning: %s", retval)
        return retval


class Sort():
    """ Sort alignments' index by the order they appear in an image """
    def __init__(self, alignments, arguments):
        logger.debug("Initializing %s: (arguments: %s)", self.__class__.__name__, arguments)
        self.alignments = alignments
        self.faces = self.get_faces(arguments)
        logger.debug("Initialized %s", self.__class__.__name__)

    @staticmethod
    def get_faces(arguments):
        """ If faces argument is specified, load faces_dir otherwise return None """
        if not hasattr(arguments, "faces_dir") or not arguments.faces_dir:
            return None
        faces = Faces(arguments.faces_dir)
        return faces

    def process(self):
        """ Execute the sort process """
        logger.info("[SORT INDEXES]")  # Tidy up cli output
        reindexed = self.reindex_faces()
        if reindexed:
            self.alignments.save()
        if self.faces:
            rename = Rename(self.alignments, None, self.faces)
            rename.process()

    def reindex_faces(self):
        """ Re-Index the faces """
        reindexed = 0
        for alignment in tqdm(self.alignments.yield_faces(),
                              desc="Sort alignment indexes", total=self.alignments.frames_count):
            frame, alignments, count, key = alignment
            if count <= 1:
                logger.trace("0 or 1 face in frame. Not sorting: '%s'", frame)
                continue
            sorted_alignments = sorted([item for item in alignments], key=lambda x: (x["x"]))
            if sorted_alignments == alignments:
                logger.trace("Alignments already in correct order. Not sorting: '%s'", frame)
                continue
            logger.trace("Sorting alignments for frame: '%s'", frame)
            self.alignments.data[key] = sorted_alignments
            reindexed += 1
        logger.info("%s Frames had their faces reindexed", reindexed)
        return reindexed


class Spatial():
    """ Apply spatial temporal filtering to landmarks
        Adapted from:
        https://www.kaggle.com/selfishgene/animating-and-smoothing-3d-facial-keypoints/notebook """

    def __init__(self, alignments, arguments):
        logger.debug("Initializing %s: (arguments: %s)", self.__class__.__name__, arguments)
        self.arguments = arguments
        self.alignments = alignments
        self.mappings = dict()
        self.normalized = dict()
        self.shapes_model = None
        logger.debug("Initialized %s", self.__class__.__name__)

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

    # Define shape normalization utility functions
    @staticmethod
    def normalize_shapes(shapes_im_coords):
        """ Normalize a 2D or 3D shape """
        logger.debug("Normalize shapes")
        (num_pts, num_dims, _) = shapes_im_coords.shape

        # Calculate mean coordinates and subtract from shapes
        mean_coords = shapes_im_coords.mean(axis=0)
        shapes_centered = np.zeros(shapes_im_coords.shape)
        shapes_centered = shapes_im_coords - np.tile(mean_coords, [num_pts, 1, 1])

        # Calculate scale factors and divide shapes
        scale_factors = np.sqrt((shapes_centered**2).sum(axis=1)).mean(axis=0)
        shapes_normalized = np.zeros(shapes_centered.shape)
        shapes_normalized = shapes_centered / np.tile(scale_factors, [num_pts, num_dims, 1])

        logger.debug("Normalized shapes: (shapes_normalized: %s, scale_factors: %s, mean_coords: "
                     "%s", shapes_normalized, scale_factors, mean_coords)
        return shapes_normalized, scale_factors, mean_coords

    @staticmethod
    def normalized_to_original(shapes_normalized, scale_factors, mean_coords):
        """ Transform a normalized shape back to original image coordinates """
        logger.debug("Normalize to original")
        (num_pts, num_dims, _) = shapes_normalized.shape

        # move back to the correct scale
        shapes_centered = shapes_normalized * np.tile(scale_factors, [num_pts, num_dims, 1])
        # move back to the correct location
        shapes_im_coords = shapes_centered + np.tile(mean_coords, [num_pts, 1, 1])

        logger.debug("Normalized to original: %s", shapes_im_coords)
        return shapes_im_coords

    def normalize(self):
        """ Compile all original and normalized alignments """
        logger.debug("Normalize")
        count = sum(1 for val in self.alignments.data.values() if val)
        landmarks_all = np.zeros((68, 2, int(count)))

        end = 0
        for key in tqdm(sorted(self.alignments.data.keys()), desc="Compiling"):
            val = self.alignments.data[key]
            if not val:
                continue
            # We should only be normalizing a single face, so just take
            # the first landmarks found
            landmarks = np.array(val[0]["landmarks_xy"]).reshape(68, 2, 1)
            start = end
            end = start + landmarks.shape[2]
            # Store in one big array
            landmarks_all[:, :, start:end] = landmarks
            # Make sure we keep track of the mapping to the original frame
            self.mappings[start] = key

        # Normalize shapes
        normalized_shape = self.normalize_shapes(landmarks_all)
        self.normalized["landmarks"] = normalized_shape[0]
        self.normalized["scale_factors"] = normalized_shape[1]
        self.normalized["mean_coords"] = normalized_shape[2]
        logger.debug("Normalized: %s", self.normalized)

    def shape_model(self):
        """ build 2D shape model """
        logger.debug("Shape model")
        landmarks_norm = self.normalized["landmarks"]
        num_components = 20
        normalized_shapes_tbl = np.reshape(landmarks_norm, [68*2, landmarks_norm.shape[2]]).T
        self.shapes_model = decomposition.PCA(n_components=num_components,
                                              whiten=True,
                                              random_state=1).fit(normalized_shapes_tbl)
        explained = self.shapes_model.explained_variance_ratio_.sum()
        logger.info("Total explained percent by PCA model with %s components is %s%%",
                    num_components, round(100 * explained, 1))
        logger.debug("Shaped model")

    def spatially_filter(self):
        """ interpret the shapes using our shape model
            (project and reconstruct) """
        logger.debug("Spatially Filter")
        landmarks_norm = self.normalized["landmarks"]
        # Convert to matrix form
        landmarks_norm_table = np.reshape(landmarks_norm, [68 * 2, landmarks_norm.shape[2]]).T
        # Project onto shapes model and reconstruct
        landmarks_norm_table_rec = self.shapes_model.inverse_transform(
            self.shapes_model.transform(landmarks_norm_table))
        # Convert back to shapes (numKeypoint, num_dims, numFrames)
        landmarks_norm_rec = np.reshape(landmarks_norm_table_rec.T,
                                        [68, 2, landmarks_norm.shape[2]])
        # Transform back to image coords
        retval = self.normalized_to_original(landmarks_norm_rec,
                                             self.normalized["scale_factors"],
                                             self.normalized["mean_coords"])

        logger.debug("Spatially Filtered: %s", retval)
        return retval

    @staticmethod
    def temporally_smooth(landmarks):
        """ apply temporal filtering on the 2D points """
        logger.debug("Temporally Smooth")
        filter_half_length = 2
        temporal_filter = np.ones((1, 1, 2 * filter_half_length + 1))
        temporal_filter = temporal_filter / temporal_filter.sum()

        start_tileblock = np.tile(landmarks[:, :, 0][:, :, np.newaxis], [1, 1, filter_half_length])
        end_tileblock = np.tile(landmarks[:, :, -1][:, :, np.newaxis], [1, 1, filter_half_length])
        landmarks_padded = np.dstack((start_tileblock, landmarks, end_tileblock))

        retval = signal.convolve(landmarks_padded, temporal_filter, mode='valid', method='fft')
        logger.debug("Temporally Smoothed: %s", retval)
        return retval

    def update_alignments(self, landmarks):
        """ Update smoothed landmarks back to alignments """
        logger.debug("Update alignments")
        for idx, frame in tqdm(self.mappings.items(), desc="Updating"):
            logger.trace("Updating: (frame: %s)", frame)
            landmarks_update = landmarks[:, :, idx]
            landmarks_xy = landmarks_update.reshape(68, 2).tolist()
            self.alignments.data[frame][0]["landmarks_xy"] = landmarks_xy
            logger.trace("Updated: (frame: '%s', landmarks: %s)", frame, landmarks_xy)
        logger.debug("Updated alignments")


class UpdateHashes():
    """ Update hashes in an alignments file """
    def __init__(self, alignments, arguments):
        logger.debug("Initializing %s: (arguments: %s)", self.__class__.__name__, arguments)
        self.alignments = alignments
        self.faces = Faces(arguments.faces_dir).file_list_sorted
        self.face_hashes = dict()
        logger.debug("Initialized %s", self.__class__.__name__)

    def process(self):
        """ Update Face Hashes to the alignments file """
        logger.info("[UPDATE FACE HASHES]")  # Tidy up cli output
        self.get_hashes()
        updated = self.update_hashes()
        if updated == 0:
            logger.info("No hashes were updated. Exiting")
            return
        self.alignments.save()
        logger.info("%s frame(s) had their face hashes updated.", updated)

    def get_hashes(self):
        """ Read the face hashes from the faces """
        logger.info("Getting original filenames, indexes and hashes...")
        for face in self.faces:
            filename = face["face_name"]
            extension = face["face_extension"]
            if "_" not in face["face_name"]:
                logger.warning("Unable to determine index of file. Skipping: '%s'", filename)
                continue
            index = filename[filename.rfind("_") + 1:]
            if not index.isdigit():
                logger.warning("Unable to determine index of file. Skipping: '%s'", filename)
                continue
            orig_frame = filename[:filename.rfind("_")] + extension
            self.face_hashes.setdefault(orig_frame, dict())[int(index)] = face["face_hash"]

    def update_hashes(self):
        """ Update hashes to alignments """
        logger.info("Updating hashes to alignments...")
        updated = 0
        for frame, hashes in self.face_hashes.items():
            if not self.alignments.frame_exists(frame):
                logger.warning("Frame not found in alignments file. Skipping: '%s'", frame)
                continue
            if not self.alignments.frame_has_faces(frame):
                logger.warning("Frame does not have faces. Skipping: '%s'", frame)
                continue
            existing = [face.get("hash", None)
                        for face in self.alignments.get_faces_in_frame(frame)]
            if any(hsh not in existing for hsh in list(hashes.values())):
                self.alignments.add_face_hashes(frame, hashes)
                updated += 1
        return updated
