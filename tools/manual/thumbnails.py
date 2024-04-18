#!/usr/bin/env python3
""" Thumbnail generator for the manual tool """
from __future__ import annotations
import logging
import typing as T
import os

from dataclasses import dataclass
from time import sleep
from threading import Lock

import imageio
import numpy as np

from tqdm import tqdm
from lib.align import AlignedFace
from lib.image import SingleFrameLoader, generate_thumbnail
from lib.multithreading import MultiThread

if T.TYPE_CHECKING:
    from .detected_faces import DetectedFaces

logger = logging.getLogger(__name__)


@dataclass
class ProgressBar:
    """ Thread-safe progress bar for tracking thumbnail generation progress """
    pbar: tqdm | None = None
    lock = Lock()


@dataclass
class VideoMeta:
    """ Holds meta information about a video file

    Parameters
    ----------
    key_frames: list[int]
        List of key frame indices for the video
    pts_times: list[float]
        List of presentation timestams for the video
    """
    key_frames: list[int] | None = None
    pts_times: list[float] | None = None


class ThumbsCreator():
    """ Background loader to generate thumbnails for the alignments file. Generates low resolution
    thumbnails in parallel threads for faster processing.

    Parameters
    ----------
    detected_faces: :class:`~tool.manual.faces.DetectedFaces`
        The :class:`~lib.align.DetectedFace` objects for this video
    input_location: str
        The location of the input folder of frames or video file
    single_process: bool
        ``True`` to generated thumbs in a single process otherwise ``False``
    """
    def __init__(self,
                 detected_faces: DetectedFaces,
                 input_location: str,
                 single_process: bool) -> None:
        logger.debug("Initializing %s: (detected_faces: %s, input_location: %s, "
                     "single_process: %s)", self.__class__.__name__, detected_faces,
                     input_location, single_process)
        self._size = 80
        self._pbar = ProgressBar()
        self._meta = VideoMeta(
            key_frames=T.cast(list[int] | None,
                              detected_faces.video_meta_data.get("keyframes", None)),
            pts_times=T.cast(list[float] | None,
                             detected_faces.video_meta_data.get("pts_time", None)))
        self._location = input_location
        self._alignments = detected_faces._alignments
        self._frame_faces = detected_faces._frame_faces

        self._is_video = self._meta.pts_times is not None and self._meta.key_frames is not None

        cpu_count = os.cpu_count()
        self._num_threads = 1 if cpu_count is None or cpu_count <= 2 else cpu_count - 2

        if self._is_video and single_process:
            self._num_threads = 1
        elif self._is_video and not single_process:
            assert self._meta.key_frames is not None
            self._num_threads = min(self._num_threads, len(self._meta.key_frames))
        else:
            self._num_threads = max(self._num_threads, 32)
        self._threads: list[MultiThread] = []
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def has_thumbs(self) -> bool:
        """ bool: ``True`` if the underlying alignments file holds thumbnail images
        otherwise ``False``. """
        return self._alignments.thumbnails.has_thumbnails

    def generate_cache(self) -> None:
        """ Extract the face thumbnails from a video or folder of images into the
        alignments file. """
        self._pbar.pbar = tqdm(desc="Caching Thumbnails",
                               leave=False,
                               total=len(self._frame_faces))
        if self._is_video:
            self._launch_video()
        else:
            self._launch_folder()
        while True:
            self._check_and_raise_error()
            if all(not thread.is_alive() for thread in self._threads):
                break
            sleep(1)
        self._join_threads()
        self._pbar.pbar.close()
        self._alignments.save()

    # << PRIVATE METHODS >> #
    def _check_and_raise_error(self) -> None:
        """ Monitor the loading threads for errors and raise if any occur. """
        for thread in self._threads:
            thread.check_and_raise_error()

    def _join_threads(self) -> None:
        """ Join the loading threads """
        logger.debug("Joining face viewer loading threads")
        for thread in self._threads:
            thread.join()

    def _launch_video(self) -> None:
        """ Launch multiple :class:`lib.multithreading.MultiThread` objects to load faces from
        a video file.

        Splits the video into segments and passes each of these segments to separate background
        threads for some speed up.
        """
        key_frames = self._meta.key_frames
        pts_times = self._meta.pts_times
        assert key_frames is not None and pts_times is not None
        key_frame_split = len(key_frames) // self._num_threads
        for idx in range(self._num_threads):
            is_final = idx == self._num_threads - 1
            start_idx: int = idx * key_frame_split
            keyframe_idx = len(key_frames) - 1 if is_final else start_idx + key_frame_split
            end_idx = key_frames[keyframe_idx]
            start_pts = pts_times[key_frames[start_idx]]
            end_pts = False if idx + 1 == self._num_threads else pts_times[end_idx]
            starting_index = pts_times.index(start_pts)
            if end_pts:
                segment_count = len(pts_times[key_frames[start_idx]:end_idx])
            else:
                segment_count = len(pts_times[key_frames[start_idx]:])
            logger.debug("thread index: %s, start_idx: %s, end_idx: %s, start_pts: %s, "
                         "end_pts: %s, starting_index: %s, segment_count: %s", idx, start_idx,
                         end_idx, start_pts, end_pts, starting_index, segment_count)
            thread = MultiThread(self._load_from_video,
                                 start_pts,
                                 end_pts,
                                 starting_index,
                                 segment_count)
            thread.start()
            self._threads.append(thread)

    def _launch_folder(self) -> None:
        """ Launch :class:`lib.multithreading.MultiThread` to retrieve faces from a
        folder of images.

        Goes through the file list one at a time, passing each file to a separate background
        thread for some speed up.
        """
        reader = SingleFrameLoader(self._location)
        num_threads = min(reader.count, self._num_threads)
        frame_split = reader.count // self._num_threads
        logger.debug("total images: %s, num_threads: %s, frames_per_thread: %s",
                     reader.count, num_threads, frame_split)
        for idx in range(num_threads):
            is_final = idx == num_threads - 1
            start_idx = idx * frame_split
            end_idx = reader.count if is_final else start_idx + frame_split
            thread = MultiThread(self._load_from_folder, reader, start_idx, end_idx)
            thread.start()
            self._threads.append(thread)

    def _load_from_video(self,
                         pts_start: float,
                         pts_end: float,
                         start_index: int,
                         segment_count: int) -> None:
        """ Loads faces from video for the given segment of the source video.

        Each segment of the video is extracted from in a different background thread.

        Parameters
        ----------
        pts_start: float
            The start time to cut the segment out of the video
        pts_end: float
            The end time to cut the segment out of the video
        start_index: int
            The frame index that this segment starts from. Used for calculating the actual frame
            index of each frame extracted
        segment_count: int
            The number of frames that appear in this segment. Used for ending early in case more
            frames come out of the segment than should appear (sometimes more frames are picked up
            at the end of the segment, so these are discarded)
        """
        logger.debug("pts_start: %s, pts_end: %s, start_index: %s, segment_count: %s",
                     pts_start, pts_end, start_index, segment_count)
        reader = self._get_reader(pts_start, pts_end)
        idx = 0
        sample_filename, ext = os.path.splitext(next(fname for fname in self._alignments.data))
        vidname = sample_filename[:sample_filename.rfind("_")]
        for idx, frame in enumerate(reader):
            frame_idx = idx + start_index
            filename = f"{vidname}_{frame_idx + 1:06d}{ext}"
            self._set_thumbail(filename, frame[..., ::-1], frame_idx)
            if idx == segment_count - 1:
                # Sometimes extra frames are picked up at the end of a segment, so stop
                # processing when segment frame count has been hit.
                break
        reader.close()
        logger.debug("Segment complete: (starting_frame_index: %s, processed_count: %s)",
                     start_index, idx)

    def _get_reader(self, pts_start: float, pts_end: float):
        """ Get an imageio iterator for this thread's segment.

        Parameters
        ----------
        pts_start: float
            The start time to cut the segment out of the video
        pts_end: float
            The end time to cut the segment out of the video

        Returns
        -------
        :class:`imageio.Reader`
            A reader iterator for the requested segment of video
        """
        input_params = ["-ss", str(pts_start)]
        if pts_end:
            input_params.extend(["-to", str(pts_end)])
        logger.debug("pts_start: %s, pts_end: %s, input_params: %s",
                     pts_start, pts_end, input_params)
        return imageio.get_reader(self._location,
                                  "ffmpeg",  # type:ignore[arg-type]
                                  input_params=input_params)

    def _load_from_folder(self,
                          reader: SingleFrameLoader,
                          start_index: int,
                          end_index: int) -> None:
        """ Loads faces from the given range of frame indices from a folder of images.

        Each frame range is extracted in a different background thread.

        Parameters
        ----------
        reader: :class:`lib.image.SingleFrameLoader`
            The reader that is used to retrieve the requested frame
        start_index: int
            The starting frame index for the images to extract faces from
        end_index: int
            The end frame index for the images to extract faces from
        """
        logger.debug("reader: %s, start_index: %s, end_index: %s",
                     reader, start_index, end_index)
        for frame_index in range(start_index, end_index):
            filename, frame = reader.image_from_index(frame_index)
            self._set_thumbail(filename, frame, frame_index)
        logger.debug("Segment complete: (start_index: %s, processed_count: %s)",
                     start_index, end_index - start_index)

    def _set_thumbail(self, filename: str, frame: np.ndarray, frame_index: int) -> None:
        """ Extracts the faces from the frame and adds to alignments file

        Parameters
        ----------
        filename: str
            The filename of the frame within the alignments file
        frame: :class:`numpy.ndarray`
            The frame that contains the faces
        frame_index: int
            The frame index of this frame in the :attr:`_frame_faces`
        """
        for face_idx, face in enumerate(self._frame_faces[frame_index]):
            aligned = AlignedFace(face.landmarks_xy,
                                  image=frame,
                                  centering="head",
                                  size=96)
            face.thumbnail = generate_thumbnail(aligned.face, size=96)
            assert face.thumbnail is not None
            self._alignments.thumbnails.add_thumbnail(filename, face_idx, face.thumbnail)
        with self._pbar.lock:
            assert self._pbar.pbar is not None
            self._pbar.pbar.update(1)
