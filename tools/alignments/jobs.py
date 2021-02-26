#!/usr/bin/env python3
""" Tools for manipulating the alignments serialized file """

import logging
import os
import sys
from datetime import datetime

import cv2
import numpy as np
from scipy import signal
from sklearn import decomposition
from tqdm import tqdm

from lib.align import DetectedFace, _EXTRACT_RATIOS
from lib.align.alignments import _VERSION
from lib.image import encode_image, generate_thumbnail, ImagesSaver, update_existing_metadata
from plugins.extract.pipeline import Extractor, ExtractMedia

from .media import ExtractedFaces, Faces, Frames

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Check():
    """ Frames and faces checking tasks.

    Parameters
    ---------
    alignments: :class:`tools.alignments.media.AlignmentsData`
        The loaded alignments corresponding to the frames to be annotated
    arguments: :class:`argparse.Namespace`
        The command line arguments that have called this job
    """
    def __init__(self, alignments, arguments):
        logger.debug("Initializing %s: (arguments: %s)", self.__class__.__name__, arguments)
        self._alignments = alignments
        self._job = arguments.job
        self._type = None
        self._is_video = False  # Set when getting items
        self._output = arguments.output
        self._source_dir = self._get_source_dir(arguments)
        self._validate()
        self._items = self._get_items()

        self.output_message = ""
        logger.debug("Initialized %s", self.__class__.__name__)

    def _get_source_dir(self, arguments):
        """ Set the correct source folder """
        if (hasattr(arguments, "faces_dir") and arguments.faces_dir and
                hasattr(arguments, "frames_dir") and arguments.frames_dir):
            logger.error("Only select a source frames (-fr) or source faces (-fc) folder")
            sys.exit(1)
        elif hasattr(arguments, "faces_dir") and arguments.faces_dir:
            self._type = "faces"
            source_dir = arguments.faces_dir
        elif hasattr(arguments, "frames_dir") and arguments.frames_dir:
            self._type = "frames"
            source_dir = arguments.frames_dir
        else:
            logger.error("No source folder (-fr or -fc) was provided")
            sys.exit(1)
        logger.debug("type: '%s', source_dir: '%s'", self._type, source_dir)
        return source_dir

    def _get_items(self):
        """ Set the correct items to process """
        items = globals()[self._type.title()](self._source_dir)
        self._is_video = items.is_video
        return items.file_list_sorted

    def process(self):
        """ Process the frames check against the alignments file """
        logger.info("[CHECK %s]", self._type.upper())
        items_output = self._compile_output()
        self._output_results(items_output)

    def _validate(self):
        """ Check that the selected type is valid for
            selected task and job """
        if self._job == "missing-frames" and self._output == "move":
            logger.warning("Missing_frames was selected with move output, but there will "
                           "be nothing to move. Defaulting to output: console")
            self._output = "console"
        if self._type == "faces" and self._job != "multi-faces":
            logger.error("The selected folder is not valid. Faces folder (-fc) is only "
                         "supported for 'multi-faces'")
            sys.exit(1)

    def _compile_output(self):
        """ Compile list of frames that meet criteria """
        action = self._job.replace("-", "_")
        processor = getattr(self, "_get_{}".format(action))
        logger.debug("Processor: %s", processor)
        return [item for item in processor()]  # pylint:disable=unnecessary-comprehension

    def _get_no_faces(self):
        """ yield each frame that has no face match in alignments file """
        self.output_message = "Frames with no faces"
        for frame in tqdm(self._items, desc=self.output_message):
            logger.trace(frame)
            frame_name = frame["frame_fullname"]
            if not self._alignments.frame_has_faces(frame_name):
                logger.debug("Returning: '%s'", frame_name)
                yield frame_name

    def _get_multi_faces(self):
        """ yield each frame or face that has multiple faces
            matched in alignments file """
        process_type = getattr(self, "_get_multi_faces_{}".format(self._type))
        for item in process_type():
            yield item

    def _get_multi_faces_frames(self):
        """ Return Frames that contain multiple faces """
        self.output_message = "Frames with multiple faces"
        for item in tqdm(self._items, desc=self.output_message):
            filename = item["frame_fullname"]
            if not self._alignments.frame_has_multiple_faces(filename):
                continue
            logger.trace("Returning: '%s'", filename)
            yield filename

    def _get_multi_faces_faces(self):
        """ Return Faces when there are multiple faces in a frame """
        self.output_message = "Multiple faces in frame"
        for item in tqdm(self._items, desc=self.output_message):
            if not self._alignments.frame_has_multiple_faces(item["source_filename"]):
                continue
            retval = (item["current_filename"], item["face_index"])
            logger.trace("Returning: '%s'", retval)
            yield retval

    def _get_missing_alignments(self):
        """ yield each frame that does not exist in alignments file """
        self.output_message = "Frames missing from alignments file"
        exclude_filetypes = set(["yaml", "yml", "p", "json", "txt"])
        for frame in tqdm(self._items, desc=self.output_message):
            frame_name = frame["frame_fullname"]
            if (frame["frame_extension"] not in exclude_filetypes
                    and not self._alignments.frame_exists(frame_name)):
                logger.debug("Returning: '%s'", frame_name)
                yield frame_name

    def _get_missing_frames(self):
        """ yield each frame in alignments that does
            not have a matching file """
        self.output_message = "Missing frames that are in alignments file"
        frames = set(item["frame_fullname"] for item in self._items)
        for frame in tqdm(self._alignments.data.keys(), desc=self.output_message):
            if frame not in frames:
                logger.debug("Returning: '%s'", frame)
                yield frame

    def _output_results(self, items_output):
        """ Output the results in the requested format """
        logger.trace("items_output: %s", items_output)
        if self._output == "move" and self._is_video and self._type == "frames":
            logger.warning("Move was selected with an input video. This is not possible so "
                           "falling back to console output")
            self._output = "console"
        if not items_output:
            logger.info("No %s were found meeting the criteria", self._type)
            return
        if self._output == "move":
            self._move_file(items_output)
            return
        if self._job == "multi-faces" and self._type == "faces":
            # Strip the index for printed/file output
            items_output = [item[0] for item in items_output]
        output_message = "-----------------------------------------------\r\n"
        output_message += " {} ({})\r\n".format(self.output_message,
                                                len(items_output))
        output_message += "-----------------------------------------------\r\n"
        output_message += "\r\n".join(items_output)
        if self._output == "console":
            for line in output_message.splitlines():
                logger.info(line)
        if self._output == "file":
            self.output_file(output_message, len(items_output))

    def _get_output_folder(self):
        """ Return output folder. Needs to be in the root if input is a
            video and processing frames """
        if self._is_video and self._type == "frames":
            return os.path.dirname(self._source_dir)
        return self._source_dir

    def _get_filename_prefix(self):
        """ Video name needs to be prefixed to filename if input is a
            video and processing frames """
        if self._is_video and self._type == "frames":
            return "{}_".format(os.path.basename(self._source_dir))
        return ""

    def output_file(self, output_message, items_discovered):
        """ Save the output to a text file in the frames directory """
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst_dir = self._get_output_folder()
        filename = "{}{}_{}.txt".format(self._get_filename_prefix(),
                                        self.output_message.replace(" ", "_").lower(),
                                        now)
        output_file = os.path.join(dst_dir, filename)
        logger.info("Saving %s result(s) to '%s'", items_discovered, output_file)
        with open(output_file, "w") as f_output:
            f_output.write(output_message)

    def _move_file(self, items_output):
        """ Move the identified frames to a new sub folder """
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = "{}{}_{}".format(self._get_filename_prefix(),
                                       self.output_message.replace(" ", "_").lower(), now)
        dst_dir = self._get_output_folder()
        output_folder = os.path.join(dst_dir, folder_name)
        logger.debug("Creating folder: '%s'", output_folder)
        os.makedirs(output_folder)
        move = getattr(self, "_move_{}".format(self._type))
        logger.debug("Move function: %s", move)
        move(output_folder, items_output)

    def _move_frames(self, output_folder, items_output):
        """ Move frames into single sub folder """
        logger.info("Moving %s frame(s) to '%s'", len(items_output), output_folder)
        for frame in items_output:
            src = os.path.join(self._source_dir, frame)
            dst = os.path.join(output_folder, frame)
            logger.debug("Moving: '%s' to '%s'", src, dst)
            os.rename(src, dst)

    def _move_faces(self, output_folder, items_output):
        """ Make additional sub folders for each face that appears
            Enables easier manual sorting """
        logger.info("Moving %s faces(s) to '%s'", len(items_output), output_folder)
        for frame, idx in items_output:
            src = os.path.join(self._source_dir, frame)
            dst_folder = os.path.join(output_folder, str(idx)) if idx != -1 else output_folder
            if not os.path.isdir(dst_folder):
                logger.debug("Creating folder: '%s'", dst_folder)
                os.makedirs(dst_folder)
            dst = os.path.join(dst_folder, frame)
            logger.debug("Moving: '%s' to '%s'", src, dst)
            os.rename(src, dst)


class Draw():  # pylint:disable=too-few-public-methods
    """ Draws annotations onto original frames and saves into a sub-folder next to the original
    frames.

    Parameters
    ---------
    alignments: :class:`tools.alignments.media.AlignmentsData`
        The loaded alignments corresponding to the frames to be annotated
    arguments: :class:`argparse.Namespace`
        The command line arguments that have called this job
    """
    def __init__(self, alignments, arguments):
        logger.debug("Initializing %s: (arguments: %s)", self.__class__.__name__, arguments)
        self._alignments = alignments
        self._frames = Frames(arguments.frames_dir)
        self._output_folder = self._set_output()
        self._mesh_areas = dict(mouth=(48, 68),
                                right_eyebrow=(17, 22),
                                left_eyebrow=(22, 27),
                                right_eye=(36, 42),
                                left_eye=(42, 48),
                                nose=(27, 36),
                                jaw=(0, 17),
                                chin=(8, 11))
        logger.debug("Initialized %s", self.__class__.__name__)

    def _set_output(self):
        """ Set the output folder path.

        If annotating a folder of frames, output will be placed in a sub folder within the frames
        folder. If annotating a video, output will be a folder next to the original video.

        Returns
        -------
        str
            Full path to the output folder

        """
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = "drawn_landmarks_{}".format(now)
        if self._frames.is_video:
            dest_folder = os.path.dirname(self._frames.folder)
        else:
            dest_folder = self._frames.folder
        output_folder = os.path.join(dest_folder, folder_name)
        logger.debug("Creating folder: '%s'", output_folder)
        os.makedirs(output_folder)
        return output_folder

    def process(self):
        """ Runs the process to draw face annotations onto original source frames. """
        logger.info("[DRAW LANDMARKS]")  # Tidy up cli output
        frames_drawn = 0
        for frame in tqdm(self._frames.file_list_sorted, desc="Drawing landmarks"):
            frame_name = frame["frame_fullname"]

            if not self._alignments.frame_exists(frame_name):
                logger.verbose("Skipping '%s' - Alignments not found", frame_name)
                continue

            self._annotate_image(frame_name)
            frames_drawn += 1
        logger.info("%s Frame(s) output", frames_drawn)

    def _annotate_image(self, frame_name):
        """ Annotate the frame with each face that appears in the alignments file.

        Parameters
        ----------
        frame_name: str
            The full path to the original frame
        """
        logger.trace("Annotating frame: '%s'", frame_name)
        image = self._frames.load_image(frame_name)

        for idx, alignment in enumerate(self._alignments.get_faces_in_frame(frame_name)):
            face = DetectedFace()
            face.from_alignment(alignment, image=image)
            # Bounding Box
            cv2.rectangle(image, (face.left, face.top), (face.right, face.bottom), (255, 0, 0), 1)
            self._annotate_landmarks(image, np.rint(face.landmarks_xy).astype("int32"))
            self._annotate_extract_boxes(image, face, idx)
            self._annotate_pose(image, face)  # Pose (head is still loaded)

        self._frames.save_image(self._output_folder, frame_name, image)

    def _annotate_landmarks(self, image, landmarks):
        """ Annotate the extract boxes onto the frame.

        Parameters
        ----------
        image: :class:`numpy.ndarray`
            The frame that extract boxes are to be annotated on to
        landmarks: :class:`numpy.ndarray`
            The 68 point landmarks that are to be annotated onto the frame
        index: int
            The face index for the given face
        """
        # Mesh
        for area, indices in self._mesh_areas.items():
            fill = area in ("right_eye", "left_eye", "mouth")
            cv2.polylines(image, [landmarks[indices[0]:indices[1]]], fill, (255, 255, 0), 1)
        # Landmarks
        for (pos_x, pos_y) in landmarks:
            cv2.circle(image, (pos_x, pos_y), 1, (0, 255, 255), -1)

    @classmethod
    def _annotate_extract_boxes(cls, image, face, index):
        """ Annotate the mesh and landmarks boxes onto the frame.

        Parameters
        ----------
        image: :class:`numpy.ndarray`
            The frame that mesh and landmarks are to be annotated on to
        face: :class:`lib.align.AlignedFace`
            The aligned face
        """
        for area in ("face", "head"):
            face.load_aligned(image, centering=area, force=True)
            color = (0, 255, 0) if area == "face" else (0, 0, 255)
            top_left = face.aligned.original_roi[0]  # pylint:disable=unsubscriptable-object
            top_left = (top_left[0], top_left[1] - 10)
            cv2.putText(image, str(index), top_left, cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 1)
            cv2.polylines(image, [face.aligned.original_roi], True, color, 1)

    @classmethod
    def _annotate_pose(cls, image, face):
        """ Annotate the pose onto the frame.

        Parameters
        ----------
        image: :class:`numpy.ndarray`
            The frame that pose is to be annotated on to
        face: :class:`lib.align.AlignedFace`
            The aligned face loaded for head centering
        """
        center = np.int32((face.aligned.size / 2, face.aligned.size / 2)).reshape(1, 2)
        center = np.rint(face.aligned.transform_points(center, invert=True)).astype("int32")
        points = face.aligned.pose.xyz_2d * face.aligned.size
        points = np.rint(face.aligned.transform_points(points, invert=True)).astype("int32")
        cv2.line(image, tuple(center), tuple(points[1]), (0, 255, 0), 2)
        cv2.line(image, tuple(center), tuple(points[0]), (255, 0, 0), 2)
        cv2.line(image, tuple(center), tuple(points[2]), (0, 0, 255), 2)


class Extract():  # pylint:disable=too-few-public-methods
    """ Re-extract faces from source frames based on Alignment data

    Parameters
    ----------
    alignments: :class:`tools.lib_alignments.media.AlignmentData`
        The alignments data loaded from an alignments file for this rename job
    arguments: :class:`argparse.Namespace`
        The :mod:`argparse` arguments as passed in from :mod:`tools.py`
    """
    def __init__(self, alignments, arguments):
        logger.debug("Initializing %s: (arguments: %s)", self.__class__.__name__, arguments)
        self._arguments = arguments
        self._alignments = alignments
        self._is_legacy = self._alignments.version == 1.0  # pylint:disable=protected-access
        self._mask_pipeline = None
        self._faces_dir = arguments.faces_dir
        self._frames = Frames(arguments.frames_dir)
        self._extracted_faces = ExtractedFaces(self._frames,
                                               self._alignments,
                                               size=arguments.size)
        self._saver = None
        logger.debug("Initialized %s", self.__class__.__name__)

    def process(self):
        """ Run the re-extraction from Alignments file process"""
        logger.info("[EXTRACT FACES]")  # Tidy up cli output
        self._check_folder()
        if self._is_legacy:
            self._legacy_check()
        self._saver = ImagesSaver(self._faces_dir, as_bytes=True)
        self._export_faces()

    def _check_folder(self):
        """ Check that the faces folder doesn't pre-exist and create. """
        err = None
        if not self._faces_dir:
            err = "ERROR: Output faces folder not provided."
        elif not os.path.isdir(self._faces_dir):
            logger.debug("Creating folder: '%s'", self._faces_dir)
            os.makedirs(self._faces_dir)
        elif os.listdir(self._faces_dir):
            err = "ERROR: Output faces folder should be empty: '{}'".format(self._faces_dir)
        if err:
            logger.error(err)
            sys.exit(0)
        logger.verbose("Creating output folder at '%s'", self._faces_dir)

    def _legacy_check(self):
        """ Check whether the alignments file was created with the legacy extraction method.

        If so, force user to re-extract all faces if any options have been specified, otherwise
        raise the appropriate warnings and set the legacy options.
        """
        if self._arguments.large or self._arguments.extract_every_n != 1:
            logger.warning("This alignments file was generated with the legacy extraction method.")
            logger.warning("You should run this extraction job, but with 'large' deselected and "
                           "'extract-every-n' set to 1 to update the alignments file.")
            logger.warning("You can then re-run this extraction job with your chosen options.")
            sys.exit(0)

        maskers = ["components", "extended"]
        nn_masks = [mask for mask in list(self._alignments.mask_summary) if mask not in maskers]
        logtype = logger.warning if nn_masks else logger.info
        logtype("This alignments file was created with the legacy extraction method and will be "
                "updated.")
        logtype("Faces will be extracted using the new method and landmarks based masks will be "
                "regenerated.")
        if nn_masks:
            logtype("However, the NN based masks '%s' will be cropped to the legacy extraction "
                    "method, so you may want to run the mask tool to regenerate these "
                    "masks.", "', '".join(nn_masks))
        self._mask_pipeline = Extractor(None, None, maskers, multiprocess=True)
        self._mask_pipeline.launch()
        # Update alignments versioning
        self._alignments._version = _VERSION  # pylint:disable=protected-access

    def _export_faces(self):
        """ Export the faces to the output folder. """
        extracted_faces = 0
        skip_list = self._set_skip_list()
        count = self._frames.count if skip_list is None else self._frames.count - len(skip_list)
        for filename, image in tqdm(self._frames.stream(skip_list=skip_list),
                                    total=count, desc="Saving extracted faces"):
            frame_name = os.path.basename(filename)
            if not self._alignments.frame_exists(frame_name):
                logger.verbose("Skipping '%s' - Alignments not found", frame_name)
                continue
            extracted_faces += self._output_faces(frame_name, image)
        if self._is_legacy and extracted_faces != 0 and not self._arguments.large:
            self._alignments.save()
        logger.info("%s face(s) extracted", extracted_faces)

    def _set_skip_list(self):
        """ Set the indices for frames that should be skipped based on the `extract_every_n`
        command line option.

        Returns
        -------
        list or ``None``
            A list of indices to be skipped if extract_every_n is not `1` otherwise
            returns ``None``
        """
        skip_num = self._arguments.extract_every_n
        if skip_num == 1:
            logger.debug("Not skipping any frames")
            return None
        skip_list = []
        for idx, item in enumerate(self._frames.file_list_sorted):
            if idx % skip_num != 0:
                logger.trace("Adding image '%s' to skip list due to extract_every_n = %s",
                             item["frame_fullname"], skip_num)
                skip_list.append(idx)
        logger.debug("Adding skip list: %s", skip_list)
        return skip_list

    def _output_faces(self, filename, image):
        """ For each frame save out the faces

        Parameters
        ----------
        filename: str
            The filename (without the full path) of the current frame
        image: :class:`numpy.ndarray`
            The full frame that faces are to be extracted from

        Returns
        -------
        int
            The total number of faces that have been extracted
        """
        logger.trace("Outputting frame: %s", filename)
        face_count = 0
        frame_name = os.path.splitext(filename)[0]
        faces = self._select_valid_faces(filename, image)
        if not faces:
            return face_count
        if self._is_legacy:
            faces = self._process_legacy(filename, image, faces)

        for idx, face in enumerate(faces):
            output = "{}_{}.png".format(frame_name, str(idx))
            meta = dict(alignments=face.to_png_meta(),
                        source=dict(alignments_version=self._alignments.version,
                                    original_filename=output,
                                    face_index=idx,
                                    source_filename=filename,
                                    source_is_video=self._frames.is_video))
            self._saver.save(output, encode_image(face.aligned.face, ".png", metadata=meta))
            if not self._arguments.large and self._is_legacy:
                face.thumbnail = generate_thumbnail(face.aligned.face, size=96, quality=60)
                self._alignments.data[filename]["faces"][idx] = face.to_alignment()
            face_count += 1
        self._saver.close()
        return face_count

    def _select_valid_faces(self, frame, image):
        """ Return the aligned faces from a frame that meet the selection criteria,

        Parameters
        ----------
        frame: str
            The filename (without the full path) of the current frame
        image: :class:`numpy.ndarray`
            The full frame that faces are to be extracted from

        Returns
        -------
        list:
            List of valid :class:`lib,align.DetectedFace` objects
        """
        faces = self._extracted_faces.get_faces_in_frame(frame, image=image)
        if not self._arguments.large:
            valid_faces = faces
        else:
            sizes = self._extracted_faces.get_roi_size_for_frame(frame)
            valid_faces = [faces[idx] for idx, size in enumerate(sizes)
                           if size >= self._extracted_faces.size]
        logger.trace("frame: '%s', total_faces: %s, valid_faces: %s",
                     frame, len(faces), len(valid_faces))
        return valid_faces

    def _process_legacy(self, filename, image, detected_faces):
        """ Process legacy face extractions to new extraction method.

        Updates stored masks to new extract size

        Parameters
        ----------
        filename: str
            The current frame filename
        image: :class:`numpy.ndarray`
            The current image the contains the faces
        detected_faces: list
            list of :class:`lib.align.DetectedFace` objects for the current frame
        """
        # Update landmarks based masks for face centering
        mask_item = ExtractMedia(filename, image, detected_faces=detected_faces)
        self._mask_pipeline.input_queue.put(mask_item)
        faces = next(self._mask_pipeline.detected_faces()).detected_faces

        # Pad and shift Neural Network based masks to face centering
        for face in faces:
            self._pad_legacy_masks(face)
        return faces

    @classmethod
    def _pad_legacy_masks(cls, detected_face):
        """ Recenter legacy Neural Network based masks from legacy centering to face centering
        and pad accordingly.

        Update the masks back into the detected face objects.

        Parameters
        ----------
        detected_face: :class:`lib.align.DetectedFace`
            The detected face to update the masks for
        """
        offset = detected_face.aligned.pose.offset["face"]
        for name, mask in detected_face.mask.items():  # Re-center mask and pad to face size
            if name in ("components", "extended"):
                continue
            old_mask = mask.mask.astype("float32") / 255.0
            size = old_mask.shape[0]
            new_size = int(size + (size * _EXTRACT_RATIOS["face"]) / 2)

            shift = np.rint(offset * (size - (size * _EXTRACT_RATIOS["face"]))).astype("int32")
            pos = np.array([(new_size // 2 - size // 2) - shift[1],
                            (new_size // 2) + (size // 2) - shift[1],
                            (new_size // 2 - size // 2) - shift[0],
                            (new_size // 2) + (size // 2) - shift[0]])
            bounds = np.array([max(0, pos[0]), min(new_size, pos[1]),
                               max(0, pos[2]), min(new_size, pos[3])])

            slice_in = [slice(0 - (pos[0] - bounds[0]), size - (pos[1] - bounds[1])),
                        slice(0 - (pos[2] - bounds[2]), size - (pos[3] - bounds[3]))]
            slice_out = [slice(bounds[0], bounds[1]), slice(bounds[2], bounds[3])]

            new_mask = np.zeros((new_size, new_size, 1), dtype="float32")
            new_mask[slice_out[0], slice_out[1], :] = old_mask[slice_in[0], slice_in[1], :]

            mask.replace_mask(new_mask)
            # Get the affine matrix from recently generated components mask
            # pylint:disable=protected-access
            mask._affine_matrix = detected_face.mask["components"].affine_matrix


class RemoveFaces():  # pylint:disable=too-few-public-methods
    """ Remove items from alignments file.

    Parameters
    ---------
    alignments: :class:`tools.alignments.media.AlignmentsData`
        The loaded alignments containing faces to be removed
    arguments: :class:`argparse.Namespace`
        The command line arguments that have called this job
    """
    def __init__(self, alignments, arguments):
        logger.debug("Initializing %s: (arguments: %s)", self.__class__.__name__, arguments)
        self._alignments = alignments

        kwargs = dict()
        if alignments.version < 2.1:
            # Update headers of faces generated with hash based alignments
            kwargs["alignments"] = alignments
        self._items = Faces(arguments.faces_dir, **kwargs)
        logger.debug("Initialized %s", self.__class__.__name__)

    def process(self):
        """ Run the job to remove faces from an alignments file that do not exist within a faces
        folder. """
        logger.info("[REMOVE FACES FROM ALIGNMENTS]")  # Tidy up cli output

        if not self._items.items:
            logger.error("No matching faces found in your faces folder. This would remove all "
                         "faces from your alignments file. Process aborted.")
            return

        pre_face_count = self._alignments.faces_count
        self._alignments.filter_faces(self._items.items, filter_out=False)
        del_count = pre_face_count - self._alignments.faces_count
        if del_count == 0:
            logger.info("No changes made to alignments file. Exiting")
            return

        logger.info("%s alignment(s) were removed from alignments file", del_count)

        self._update_png_headers()
        self._alignments.save()

        rename = Rename(self._alignments, None, self._items)
        rename.process()

    def _update_png_headers(self):
        """ Update the EXIF iTXt field of any face PNGs that have had their face index changed.

        Notes
        -----
        This could be quicker if parellizing in threads, however, Windows (at least) does not seem
        to like this and has a tendency to throw permission errors, so this remains single threaded
        for now.
        """
        to_update = [  # Items whose face index has changed
            x for x in self._items.file_list_sorted
            if x["face_index"] != self._items.items[x["source_filename"]].index(x["face_index"])]

        for file_info in tqdm(to_update, desc="Updating PNG Headers", leave=False):
            frame = file_info["source_filename"]
            face_index = file_info["face_index"]
            new_index = self._items.items[frame].index(face_index)

            fullpath = os.path.join(self._items.folder, file_info["current_filename"])
            logger.debug("Updating png header for '%s': face index from %s to %s",
                         fullpath, face_index, new_index)

            # Update file_list_sorted for rename task
            orig_filename = "{}_{}.png".format(os.path.splitext(frame)[0], new_index)
            file_info["face_index"] = new_index
            file_info["original_filename"] = orig_filename

            face = DetectedFace()
            face.from_alignment(self._alignments.get_faces_in_frame(frame)[new_index])
            meta = dict(alignments=face.to_png_meta(),
                        source=dict(alignments_version=file_info["alignments_version"],
                                    original_filename=orig_filename,
                                    face_index=new_index,
                                    source_filename=frame,
                                    source_is_video=file_info["source_is_video"]))
            update_existing_metadata(fullpath, meta)

        logger.info("%s Extracted face(s) had their header information updated", len(to_update))


class Rename():  # pylint:disable=too-few-public-methods
    """ Rename faces in a folder to match their filename as stored in an alignments file.

    Parameters
    ----------
    alignments: :class:`tools.lib_alignments.media.AlignmentData`
        The alignments data loaded from an alignments file for this rename job
    arguments: :class:`argparse.Namespace`
        The :mod:`argparse` arguments as passed in from :mod:`tools.py`
    faces: :class:`tools.lib_alignments.media.Faces`, Optional
        An optional faces object, if the rename task is being called by another job.
        Default: ``None``
    """
    def __init__(self, alignments, arguments, faces=None):
        logger.debug("Initializing %s: (arguments: %s, faces: %s)",
                     self.__class__.__name__, arguments, faces)
        self._alignments = alignments

        kwargs = dict()
        if alignments.version < 2.1:
            # Update headers of faces generated with hash based alignments
            kwargs["alignments"] = alignments
        self._faces = faces if faces else Faces(arguments.faces_dir, **kwargs)
        logger.debug("Initialized %s", self.__class__.__name__)

    def process(self):
        """ Process the face renaming """
        logger.info("[RENAME FACES]")  # Tidy up cli output
        rename_mappings = sorted([(face["current_filename"], face["original_filename"])
                                  for face in self._faces.file_list_sorted
                                  if face["current_filename"] != face["original_filename"]],
                                 key=lambda x: x[1])
        rename_count = self._rename_faces(rename_mappings)
        logger.info("%s faces renamed", rename_count)

    def _rename_faces(self, filename_mappings):
        """ Rename faces back to their original name as exists in the alignments file.

        If the source and destination filename are the same then skip that file.

        Parameters
        ----------
        filename_mappings: list
            List of tuples of (`source filename`, `destination filename`) ordered by destination
            filename

        Returns
        -------
        int
            The number of faces that have been renamed
        """
        if not filename_mappings:
            return 0

        rename_count = 0
        conflicts = []
        for src, dst in tqdm(filename_mappings, desc="Renaming Faces"):
            old = os.path.join(self._faces.folder, src)
            new = os.path.join(self._faces.folder, dst)

            if os.path.exists(new):
                # Interim add .tmp extension to files that will cause a rename conflict, to
                # process afterwards
                logger.debug("interim renaming file to avoid conflict: (src: '%s', dst: '%s')",
                             src, dst)
                new = new + ".tmp"
                conflicts.append(new)

            logger.verbose("Renaming '%s' to '%s'", old, new)
            os.rename(old, new)
            rename_count += 1
        if conflicts:
            for old in tqdm(conflicts, desc="Renaming Faces"):
                new = old[:-4]  # Remove .tmp extension
                if os.path.exists(new):
                    # This should only be running on faces. If there is still a conflict
                    # then the user has done something stupid, so we will delete the file and
                    # replace. They can always re-extract :/
                    os.remove(new)
                logger.verbose("Renaming '%s' to '%s'", old, new)
                os.rename(old, new)
        return rename_count


class Sort():
    """ Sort alignments' index by the order they appear in an image in left to right order.

    Parameters
    ----------
    alignments: :class:`tools.lib_alignments.media.AlignmentData`
        The alignments data loaded from an alignments file for this rename job
    arguments: :class:`argparse.Namespace`
        The :mod:`argparse` arguments as passed in from :mod:`tools.py`
    """
    def __init__(self, alignments, arguments):
        logger.debug("Initializing %s: (arguments: %s)", self.__class__.__name__, arguments)
        self._alignments = alignments
        logger.debug("Initialized %s", self.__class__.__name__)

    def process(self):
        """ Execute the sort process """
        logger.info("[SORT INDEXES]")  # Tidy up cli output
        reindexed = self.reindex_faces()
        if reindexed:
            self._alignments.save()
            logger.warning("If you have a face-set corresponding to the alignment file you "
                           "processed then you should run the 'Extract' job to regenerate it.")

    def reindex_faces(self):
        """ Re-Index the faces """
        reindexed = 0
        for alignment in tqdm(self._alignments.yield_faces(),
                              desc="Sort alignment indexes", total=self._alignments.frames_count):
            frame, alignments, count, key = alignment
            if count <= 1:
                logger.trace("0 or 1 face in frame. Not sorting: '%s'", frame)
                continue
            sorted_alignments = sorted(alignments, key=lambda x: (x["x"]))
            if sorted_alignments == alignments:
                logger.trace("Alignments already in correct order. Not sorting: '%s'", frame)
                continue
            logger.trace("Sorting alignments for frame: '%s'", frame)
            self._alignments.data[key]["faces"] = sorted_alignments
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
        self._alignments = alignments
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
        self._alignments.save()
        logger.warning("If you have a face-set corresponding to the alignment file you "
                       "processed then you should run the 'Extract' job to regenerate it.")

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
        count = sum(1 for val in self._alignments.data.values() if val["faces"])
        landmarks_all = np.zeros((68, 2, int(count)))

        end = 0
        for key in tqdm(sorted(self._alignments.data.keys()), desc="Compiling"):
            val = self._alignments.data[key]["faces"]
            if not val:
                continue
            # We should only be normalizing a single face, so just take
            # the first landmarks found
            landmarks = np.array(val[0]["landmarks_xy"]).reshape((68, 2, 1))
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
        # Transform back to image co-ordinates
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
            self._alignments.data[frame]["faces"][0]["landmarks_xy"] = landmarks_xy
            logger.trace("Updated: (frame: '%s', landmarks: %s)", frame, landmarks_xy)
        logger.debug("Updated alignments")
