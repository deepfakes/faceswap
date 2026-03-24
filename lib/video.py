#!/usr/bin python3
"""Utilities for working with videos"""
from __future__ import annotations

import logging
import os
import subprocess
import typing as T

from collections import deque
from fractions import Fraction
from math import ceil

import av
import av.error
import av.filter
import av.logging
import ffmpeg
import numpy as np
from tqdm import tqdm

from lib.logger import parse_class_init
from lib.utils import convert_to_secs, FaceswapError, get_module_objects


if T.TYPE_CHECKING:
    from av.container import InputContainer, OutputContainer
    import numpy.typing as npt

logger = logging.getLogger(__name__)
av.logging.set_level(av.logging.VERBOSE)
logging.getLogger("libav").setLevel(logger.getEffectiveLevel())


VIDEO_EXTENSIONS = [".avi", ".flv", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm", ".wmv",
                    ".ts", ".vob"]
"""List of lowercase valid Video extensions with preceding period"""


def check_for_video(input_location: str) -> bool:
    """Check whether the given input is a video file or a folder

    Parameters
    ----------
    input_location
        Full path to an input file

    Returns
    -------
    bool: 'True' if input is a video 'False' if it is a folder.

    Raises
    ------
    FaceswapError
        If the given location is a file and does not have a valid video extension.
    """
    if not isinstance(input_location, str) or os.path.isdir(input_location):
        retval = False
    elif os.path.splitext(input_location)[1].lower() in VIDEO_EXTENSIONS:
        retval = True
    else:
        raise FaceswapError(f"The input file '{input_location}' is not a valid video")
    logger.debug("Input '%s' is_video: %s", input_location, retval)
    return retval


def validate_video_file(file_path: str) -> str:
    """Validates that a given file exists and is a valid video format

    Parameters
    ----------
    file_path
        The full path to the video file to validate

    Returns
    -------
    The full expanded video file path

    Raises
    ------
    FaceswapError
        If the given video file is not valid
    """
    file_path = os.path.expanduser(os.path.abspath(file_path))
    if not os.path.isfile(file_path):
        raise FaceswapError(f"Video file '{file_path}' does not exist")
    if os.path.splitext(file_path)[-1].lower() not in VIDEO_EXTENSIONS:
        raise FaceswapError(f"File '{file_path}' is not a valid video file")
    return file_path


# TODO look for instances of this and see if we can roll it into VideoInfo
def count_frames(filename, fast=False):
    """ Count the number of frames in a video file

    There is no guaranteed accurate way to get a count of video frames without iterating through
    a video and decoding every frame.

    :func:`count_frames` can return an accurate count (albeit fairly slowly) or a possibly less
    accurate count, depending on the :attr:`fast` parameter. A progress bar is displayed.

    Parameters
    ----------
    filename: str
        Full path to the video to return the frame count from.
    fast: bool, optional
        Whether to count the frames without decoding them. This is significantly faster but
        accuracy is not guaranteed. Default: ``False``.

    Returns
    -------
    int:
        The number of frames in the given video file.

    Example
    -------
    >>> filename = "/path/to/video.mp4"
    >>> frame_count = count_frames(filename)
    """
    logger.debug("filename: %s, fast: %s", filename, fast)
    assert isinstance(filename, str), "Video path must be a string"
    cmd = [str(ffmpeg.FFMPEG_PATH), "-i", filename, "-map", "0:v:0"]
    if fast:
        cmd.extend(["-c", "copy"])
    cmd.extend(["-f", "null", "-"])

    logger.debug("FFMPEG Command: '%s'", " ".join(cmd))
    process = subprocess.Popen(cmd,
                               stderr=subprocess.STDOUT,
                               stdout=subprocess.PIPE,
                               universal_newlines=True, encoding="utf8")
    p_bar = None
    duration = None
    update = 0
    frames = 0
    stdout = process.stdout
    assert stdout is not None
    while True:

        output = stdout.readline().strip()
        if output == "" and process.poll() is not None:
            break

        if output.startswith("Duration:"):
            logger.debug("Duration line: %s", output)
            idx = output.find("Duration:") + len("Duration:")
            duration = int(convert_to_secs(*output[idx:].split(",", 1)[0].strip().split(":")))
            logger.debug("duration: %s", duration)
        if output.startswith("frame="):
            logger.debug("frame line: %s", output)
            if p_bar is None:
                logger.debug("Initializing tqdm")
                p_bar = tqdm(desc="Analyzing Video", leave=False, total=duration, unit="secs")
            time_idx = output.find("time=") + len("time=")
            frame_idx = output.find("frame=") + len("frame=")
            frames = int(output[frame_idx:].strip().split(" ")[0].strip())
            vid_time = int(convert_to_secs(*output[time_idx:].split(" ")[0].strip().split(":")))
            logger.debug("frames: %s, vid_time: %s", frames, vid_time)
            prev_update = update
            update = vid_time
            p_bar.update(update - prev_update)
    if p_bar is not None:
        p_bar.close()
    return_code = process.poll()
    logger.debug("Return code: %s, frames: %s", return_code, frames)
    return frames


class VideoInfo:
    """Collects and stores information about video files

    Parameters
    ----------
    video_file
        Full path to a video file
    fast_count
        Whether to obtain the count of frames quickly, but inaccurately or slowly but accurately.
        If pts and keyframes are provided then the count will be derived from the provided pts
        file. Default: ``True``
    stream_index
        The stream index to select from the video file. Default: 0
    pts
        The Presentation Timestamps if available or ``None`` to retrieve from the video.
        Default: ``None``
    keyframes
        The keyframe frame indices if available or ``None`` to retrieve from the video.
        Default: ``None``
    """
    def __init__(self,
                 video_file: str,
                 fast_count: bool = True,
                 stream_index: int = 0,
                 pts: list[int] | None = None,
                 keyframes: list[int] | None = None) -> None:
        logger.debug(parse_class_init(locals()))
        self._video_file = validate_video_file(video_file)
        self._fast_count = fast_count
        self._stream_index = stream_index
        self._pts = None if pts is None else np.array(pts, dtype=np.int64)
        self._keyframes = None if keyframes is None else np.array(keyframes, dtype=np.int64)
        self._num_keyframes = -1

        self._duration = self._get_duration()
        self._count: int | None = None

    def __repr__(self) -> str:
        """Pretty print for logging"""
        params = {k[1:]: v.tolist() if isinstance(v, np.ndarray) else v
                  for k, v in self.__dict__.items()
                  if k in ("_video_file",
                           "_fast_count",
                           "_stream_index",
                           "_pts",
                           "_keyframes")}
        s_params = ", ".join(f"{k}={repr(v)}" for k, v in params.items())
        return f"{self.__class__.__name__}({s_params})"

    @property
    def duration(self) -> int:
        """The duration of the video file in seconds"""
        return self._duration

    @property
    def count(self) -> int:
        """The number of frames in the video"""
        if self._count is not None:
            return self._count
        if self._pts is not None:
            self._count = len(self._pts)
            return self._count
        if self._fast_count:
            self._count = count_frames(self._video_file, fast=True)
            return self._count
        self._count = len(self.pts)
        return self._count

    @property
    def keyframes_count(self) -> int:
        """The number of keyframes that exist in the video"""
        if self._num_keyframes < 0:
            self._num_keyframes = len(self.keyframes)
        return self._num_keyframes

    @property
    def pts(self) -> npt.NDArray[np.int64]:
        """The Presentation Time Stamp for each frame in the video"""
        if self._pts is None:
            self._get_pts_and_keyframes()
        assert self._pts is not None
        return self._pts

    @property
    def keyframes(self) -> npt.NDArray[np.int64]:
        """The frame index of each key frame in the video"""
        if self._keyframes is None:
            self._get_pts_and_keyframes()
        assert self._keyframes is not None
        return self._keyframes

    def _get_stream(self, container: InputContainer) -> av.VideoStream:
        """Obtain the first video stream from the given container and set threading

        Parameters
        ----------
        container
            The opened video container

        Returns
        -------
        stream
            The first video stream within the container with AUTO threading mode set

        Raises
        ------
        FaceswapError
            If time_base is not stored within the stream
        """
        stream = container.streams.video[self._stream_index]
        stream.thread_type = "AUTO"
        if stream.time_base is None:
            raise FaceswapError(f"Video file '{self._video_file}' cannot be processed. Missing "
                                "duration metadata")
        return stream

    def _get_duration(self) -> int:
        """Obtain the duration of the video, in seconds. First attempt to obtain it from the
        stream. If this does not exist attempt to obtain it from the container. If this also
        does not exist, raise an error

        Parameters
        ----------
        stream
            The stream to attempt to obtain the duration from

        Returns
        -------
        The duration of the stream in seconds

        Raises
        ------
        FaceswapError
            If the duration of the video could not be obtained
        """
        with av.open(self._video_file, "r") as container:
            stream = self._get_stream(container)
            if stream.duration is not None and stream.time_base is not None:
                duration = int(stream.duration * stream.time_base)
                logger.debug("[%s] '%s' duration from stream: %s",
                             self.__class__.__name__, self._video_file, duration)
            elif container.duration is None:
                raise FaceswapError(f"Video file '{self._video_file}' cannot be processed. "
                                    "Missing duration metadata")
            else:
                duration = int(container.duration / 1000000)
                logger.debug("[%s] '%s' duration from container: %s",
                             self.__class__.__name__, self._video_file, duration)
        return duration

    def _get_pts_and_keyframes(self) -> None:
        """Parse the video for Presentation Time Stamps and keyframes and populate to :attr:`_pts`
        and :attr:`_keyframes"""
        logger.debug("[%s] Parsing video for PTS and keyframes: '%s'",
                     self.__class__.__name__, self._video_file)
        pts: list[int] = []
        keyframes: list[int] = []
        with av.open(self._video_file, "r") as container:
            stream = self._get_stream(container)
            assert stream.time_base is not None

            p_bar = tqdm(desc="Analyzing Video", leave=False, total=self.duration, unit="secs")
            i = last_update = offset = 0
            decoder = container.decode(stream)
            while True:
                try:
                    frame = next(decoder)
                except StopIteration:
                    break
                except av.error.InvalidDataError:
                    logger.warning("Invalid data encountered at frame %s in video '%s'",
                                   i, self._video_file)
                    continue
                assert frame.pts is not None
                if i == 0:
                    offset = frame.pts
                pts.append(frame.pts)
                if frame.key_frame:  # pyright:ignore[reportAttributeAccessIssue]
                    keyframes.append(i)
                cur_sec = int((frame.pts - offset) * stream.time_base)
                i += 1
                if cur_sec == last_update:
                    continue
                p_bar.update(cur_sec - last_update)
                last_update = cur_sec
        self._pts = np.array(pts, dtype=np.int64)
        self._keyframes = np.array(keyframes, dtype=np.int64)
        logger.debug("[%s] '%s' frame_pts: %s, keyframes: %s, frame_count: %s",
                     self.__class__.__name__, self._video_file, pts, keyframes, len(pts))


class VideoReader:
    """A wrapper around pyAV that allows obtaining frames by frame index and iterating video files

    Parameters
    ----------
    video_file
        Full path to a video file
    fast_count
        Whether to obtain the count of frames quickly, but inaccurately or slowly but accurately.
        If pts and keyframes are provided then the count will be derived from the provided pts
        file. Default: ``True``
    stream_index
        The stream index to select from the video file. Default: 0
    pts
        The Presentation Timestamps if available or ``None`` to retrieve from the video.
        Default: ``None``
    keyframes
        The keyframe frame indices if available or ``None`` to retrieve from the video.
        Default: ``None``
    """
    def __init__(self,
                 video_file: str,
                 fast_count: bool = True,
                 stream_index: int = 0,
                 pts: list[int] | None = None,
                 keyframes: list[int] | None = None) -> None:
        logger.debug(parse_class_init(locals()))
        self._video_file = validate_video_file(video_file)
        self._stream_index = stream_index
        self._info = VideoInfo(self._video_file,
                               fast_count,
                               self._stream_index,
                               pts,
                               keyframes)

        self._container = av.open(self._video_file, "r")
        self._stream = self._container.streams.video[stream_index]
        self._stream.thread_type = "AUTO"
        self._decoder = self._container.decode(self._stream)

        self._count: int | None = None
        self._current_pts = 0
        self._current_index = 0
        """The index of the next frame to be returned from the frame iterator"""

    @property
    def info(self) -> VideoInfo:
        """The metadata information for the video file"""
        return self._info

    def __iter__(self) -> T.Self:
        """ This is an iterator """
        return self

    def __repr__(self) -> str:
        """ Pretty print for logging """
        pts = self._info._pts
        keyframes = self._info._keyframes
        params = {"video_file": self._video_file,
                  "fast_count": self._info._fast_count,
                  "stream_index": self._stream_index,
                  "pts": pts if pts is None else pts.tolist(),
                  "keyframes": keyframes if keyframes is None else keyframes.tolist()}
        s_params = ", ".join(f"{k}={repr(v)}" for k, v in params.items())
        return f"{self.__class__.__name__}({s_params})"

    def __len__(self) -> int:
        """The number of frames in the video file. Either inaccurate (if fast_count is ``True``)
        or accurate (if fast_count is ``False`` or pts and keyframes were provided)"""
        return self._info.count

    def close(self) -> None:
        """Shut down the AV Container object"""
        logger.debug("[%s] '%s' Closing container", self.__class__.__name__, self._video_file)
        self._container.close()

    def __next__(self) -> av.VideoFrame:
        """Obtain the next video frame object

        Returns
        -------
        The next available video frame object
        """
        frame = None
        while True:
            try:
                frame = next(self._decoder)
                break
            except StopIteration:
                break
            except av.error.InvalidDataError:
                logger.warning("Invalid data encountered at frame %s. Skipping.",
                               self._current_index)
                continue
        if frame is None:
            logger.debug("[%s] Closing Frame Iterator", self.__class__.__name__)
            self.close()
            raise StopIteration
        self._current_index += 1
        return frame

    def _get_previous_keyframe(self, index: int) -> int:
        """Obtain the keyframe that appears directly prior to the given frame index

        Parameters
        ----------
        index
            The target frame that is being navigated to

        Returns
            The keyframe that appears directly prior to the given target frame
        """
        if index in self._info.keyframes:
            logger.trace("[%s] Index is keyframe: %s",  # type:ignore[attr-defined]
                         self.__class__.__name__, index)
            return index
        keyframe_index = np.searchsorted(self._info.keyframes, index, side="left") - 1
        keyframe = int(self._info.keyframes[keyframe_index])
        logger.trace("[%s] Previous keyframe for frame %s: %s",  # type:ignore[attr-defined]
                     self.__class__.__name__, index, keyframe)
        return keyframe

    def _jump_to_keyframe(self, index: int, target_pts: int) -> None:
        """Jump the iterator to the first keyframe prior to the requested frame, or leave it where
        it is if the next requested frame is before the next keyframe. If we are seeking we always
        replace our iterator with a new one due to possible internal pyAV logic getting scrambled

        Parameters
        ----------
        index
            The frame index of the requested frame to retrieve
        target_pts
            The Presentation Timestamp of the requested frame
        """
        if index == self._current_index:
            logger.trace(  # type:ignore[attr-defined]
                "[%s] Requested frame is next queued. Not seeking: %s",
                self.__class__.__name__, index)
            return

        if index < self._current_index:  # Moving backwards
            logger.trace("[%s] Seeking backwards from %s to %s",  # type:ignore[attr-defined]
                         self.__class__.__name__, self._current_index, index)
            self._container.seek(target_pts, backward=True, any_frame=False, stream=self._stream)
            self._decoder = self._container.decode(self._stream)
            self._current_index = self._get_previous_keyframe(index)
            return

        next_key_index = np.searchsorted(self._info.keyframes, self._current_index, side="right")
        next_keyframe = self._info.keyframes[next_key_index]

        if next_keyframe > index:
            logger.trace(  # type:ignore[attr-defined]
                "[%s] Next keyframe is past target. Not seeking: %s",
                self.__class__.__name__, next_keyframe)
            return

        next_keyframe = self._get_previous_keyframe(index)
        logger.trace("[%s] Seeking forwards to %s",  # type:ignore[attr-defined]
                     self.__class__.__name__, next_keyframe)
        self._container.seek(target_pts, backward=True, any_frame=False, stream=self._stream)
        self._decoder = self._container.decode(self._stream)
        self._current_index = next_keyframe

    def get(self, index: int) -> av.VideoFrame:
        """Obtain the video frame at the given frame index

        Parameters
        ----------
        index
            The index number of the frame to retrieve

        Returns
        -------
        The pyAV frame object for the given index
        """
        target_pts = int(self._info.pts[index])
        logger.trace(  # type:ignore[attr-defined]
            "[%s] Requested frame: %s, current frame: %s, target pts: %s",
            self.__class__.__name__, index, self._current_index, target_pts)
        self._jump_to_keyframe(index, target_pts)
        frame = next(self)
        assert frame.pts is not None
        current_pts = frame.pts
        while current_pts < target_pts:
            frame = next(self)
            assert frame.pts is not None
            current_pts = frame.pts
        logger.trace("[%s] Returning frame: %s",  # type:ignore[attr-defined]
                     self.__class__.__name__, frame)
        return frame


class VideoMux:  # pylint:disable=too-many-instance-attributes
    """A basic muxer for muxing converted faceswap frames to a video file using the original video
    as a reference

    Parameters
    ----------
    source_video
        The path to the source video to use as a reference for Audio and FPS
    destination_video
        The full path to save the final video to
    codec
        The codec to use to encode the video
    codec_parameters
        The options to use for the codec
    mux_audio
        ``True`` to mux order from the source video to the output
    """
    def __init__(self,
                 source_video: str,
                 destination_video: str,
                 codec: T.Literal["libx264", "libx265"],
                 codec_parameters: dict[str, str],
                 mux_audio: bool = True) -> None:
        logger.debug(parse_class_init(locals()))
        self._source_video = validate_video_file(source_video)
        self._destination_video = destination_video
        self._codec = codec
        self._codec_parameters = codec_parameters
        self._mux_audio = mux_audio

        self._containers: dict[T.Literal["src", "dst"], InputContainer | OutputContainer] = {
            "src": av.open(self._source_video, "r"),
            "dst": av.open(self._destination_video, "w")
            }

        self._next_audio_packet: av.Packet | None = None
        self._audio_packets, self._fps = self._analyze_source()
        self._video_packets: deque[av.Packet] = deque()
        self._streams = self._set_output_streams()

        self._graph: av.filter.Graph | None = None
        self._initialized = False
        self._frame_index = 0

    def __repr__(self) -> str:
        """ Pretty print for logging """
        opts = ["_source_video", "_destination_video", "_codec", "_codec_parameters", "_mux_audio"]
        params = {k[1:]: v for k, v in self.__dict__.items() if k in opts}
        s_params = ", ".join(f"{k}={repr(v)}" for k, v in params.items())
        return f"{self.__class__.__name__}({s_params})"

    def _analyze_source(self) -> tuple[T.Generator[av.Packet, None, None] | None, Fraction]:
        """Analyze the source to obtain the audio packets and the frame rate

        Returns
        -------
        audio_packets
            A generator containing audio packets from the source video, if audio is to be muxed
            otherwise ``None``
        fps
            The framerate of the original video
        """
        src = T.cast("InputContainer", self._containers["src"])
        fps = src.streams.video[0].average_rate
        assert fps is not None
        logger.debug("[%s] Source fps: %s", self.__class__.__name__, fps)

        if not self._mux_audio:
            logger.debug("[%s] Not muxing audio due to input parameters", self.__class__.__name__)
            return None, fps

        audio = next((s for s in src.streams if s.type == "audio"), None)
        if audio is None:
            logger.warning("No audio stream could be found in the source video '%s'. Audio mux "
                           "will be disabled.", self._source_video)
            self._mux_audio = False
            return None, fps

        packets = (p for p in src.demux(audio) if p.dts is not None)
        logger.debug("[%s] Muxing audio from source: %s", self.__class__.__name__, packets)
        self._next_audio_packet = next(packets)
        logger.debug("[%s] Queued first audio packet: %s",
                     self.__class__.__name__, self._next_audio_packet)
        return packets, fps

    def _set_output_streams(self) -> dict[T.Literal["audio", "video"],
                                          av.AudioStream | av.VideoStream]:
        """Set the output audio and video streams

        Returns
        -------
        The output streams. Audio stream is only included if muxing audio is selected and supported
        """
        retval:  dict[T.Literal["audio", "video"], av.AudioStream | av.VideoStream] = {}
        dst = T.cast("OutputContainer", self._containers["dst"])
        video = dst.add_stream(self._codec, rate=self._fps, options=self._codec_parameters)
        assert isinstance(video, av.VideoStream)
        video.thread_type = "AUTO"
        video.pix_fmt = "yuv420p"
        retval["video"] = video

        if self._mux_audio:
            src = self._containers["src"]
            src_audio = next(s for s in src.streams if s.type == "audio")
            audio = dst.add_stream_from_template(src_audio)
            assert isinstance(audio, av.AudioStream)
            retval["audio"] = audio
        logger.debug("[%s] Added output streams: %s", self.__class__.__name__, retval)
        return retval

    def _add_rescale_filter(self,
                            input_dimensions: tuple[int, int],
                            output_dimensions: tuple[int, int],
                            pixel_format: str) -> None:
        """Add a rescale filter if the input dimensions are not divisible by 16

        Parameters
        ----------
        input_dimensions
            The (W, H) size of the input frames to the video
        output_dimensions
            The (W, H) size of the output video
        pixel_format
            The pixel format of the output video
        """
        if input_dimensions == output_dimensions:
            return
        self._graph = av.filter.Graph()
        str_dims = f"{output_dimensions[0]}:{output_dimensions[1]}"
        filters = [self._graph.add_buffer(width=input_dimensions[0],
                                          height=input_dimensions[1],
                                          format=av.VideoFormat(pixel_format),
                                          time_base=Fraction(1, self._fps)),
                   self._graph.add("scale", f"{str_dims}:force_original_aspect_ratio=1"),
                   self._graph.add("pad", f"{str_dims}:(ow-iw)/2:(oh-ih)/2"),
                   self._graph.add("buffersink")]
        for i in range(len(filters) - 1):
            filters[i].link_to(filters[i + 1])
        self._graph.configure()
        logger.debug("[%s] Created scale filter: %s", self.__class__.__name__, self._graph)

    def _initialize_video(self, image: npt.NDArray[np.uint8]) -> None:
        """Initialize the video dimensions based on the first frame seen. We scale dimensions to be
        divisible by 16 due to macro-blocking.

        Parameters
        ----------
        image
            The first frame passed into the muxer
        """
        vid = T.cast(av.VideoStream, self._streams["video"])
        input_dimensions = (image.shape[1], image.shape[0])
        output_dimensions = (int(ceil(input_dimensions[0] / 16) * 16),
                             int(ceil(input_dimensions[1] / 16) * 16))
        vid.width = output_dimensions[0]
        vid.height = output_dimensions[1]
        logger.debug("[%s] Set video dimensions for first frame input: %s output: %s (%s)",
                     self.__class__.__name__, input_dimensions, output_dimensions, vid)
        self._add_rescale_filter(input_dimensions, output_dimensions, T.cast(str, vid.pix_fmt))

        logger.debug("[%s] Initialized video stream", self.__class__.__name__)
        self._initialized = True

    def _encode_frame(self, image: npt.NDArray[np.uint8]) -> None:
        """Encode the frame into packets and add the packets to the list of encoded packets to be
        muxed

        Parameters
        ----------
        image
            The image to be encoded
        """
        vid = T.cast(av.VideoStream, self._streams["video"])
        frame = av.VideoFrame.from_ndarray(image, format="bgr24")
        frame.pts = self._frame_index
        frame.time_base = Fraction(1, self._fps)

        if self._graph is not None:
            # Need to convert to output format before running through filter graph
            self._graph.push(frame.reformat(format=vid.pix_fmt))
            frame = T.cast(av.VideoFrame, self._graph.pull())

        logger.trace("[%s] Encoded frame of shape %s to: %s",  # type:ignore[attr-defined]
                     self.__class__.__name__, image.shape, frame)

        packets = vid.encode(frame)
        self._video_packets.extend(packets)
        logger.trace("[%s] Added video packets: %s",  # type:ignore[attr-defined]
                     self.__class__.__name__, packets)
        self._frame_index += 1

    def _timestamp(self, packet: av.Packet) -> float:
        """Obtain the standardized time stamp for the given packet

        Parameters
        ----------
        packet
            The packet to obtain the timestamp for

        Returns
        -------
        The standardized timestamp
        """
        assert packet.pts is not None
        return float(packet.pts * packet.time_base)

    def _get_audio_packet(self, timestamp: float) -> av.Packet | None:
        """Obtain the next audio packet if it should be output prior to the current timestamp and
        queue the next audio packet for output

        Parameters
        ----------
        timestamp
            The timestamp of the next video packet to be output
        """
        assert self._next_audio_packet is not None
        next_ts = self._timestamp(self._next_audio_packet)
        if next_ts >= timestamp:
            logger.trace(  # type:ignore[attr-defined]
                "[%s] Next audio timestamp %s >= video timestamp %s. No audio to stream",
                self.__class__.__name__, next_ts, timestamp)
            return None

        assert self._audio_packets is not None
        retval = self._next_audio_packet
        self._next_audio_packet = next(self._audio_packets)
        logger.trace(  # type:ignore[attr-defined]
            "[%s] Returning audio packet %s for timestamp %s < video timestamp: %s. Next  queued "
            "packet: %s",
            self.__class__.__name__, retval, next_ts, timestamp, self._next_audio_packet)
        retval.stream = self._streams["audio"]
        return retval

    def _mux(self) -> None:
        """Mux any audio and video packets that are ready to be output"""
        out = T.cast("OutputContainer", self._containers["dst"])
        while self._video_packets:
            video = self._video_packets.popleft()
            if self._mux_audio:
                while True:
                    audio = self._get_audio_packet(self._timestamp(video))
                    if audio is None:
                        break
                    logger.trace("[%s] Muxing audio: %s",  # type:ignore[attr-defined]
                                 self.__class__.__name__, audio)
                    out.mux(audio)
            logger.trace("[%s] Muxing video: %s",  # type:ignore[attr-defined]
                         self.__class__.__name__, video)
            out.mux(video)

    def encode(self, image: npt.NDArray[np.uint8] | None) -> None:
        """Encode a frame to the video

        Parameters
        ----------
        image
            The 3 channel BGR UINT8 image to encode to the video or ``None`` to finalize the video
        """
        if image is None:
            logger.debug("[%s] EOF Received. Flushing", self.__class__.__name__)
            self._video_packets.extend(self._streams["video"].encode())
            self._mux()
            for container in self._containers.values():
                container.close()
            return

        if not self._initialized:
            self._initialize_video(image)

        self._encode_frame(image)
        self._mux()


get_module_objects(__name__)
