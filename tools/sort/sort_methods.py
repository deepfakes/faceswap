#!/usr/bin/env python3
"""Sorting methods for the sorting tool.

All sorting methods inherit from :class:`SortMethod` and control functions for sorting one item,
sorting a full list of scores and binning based on those sorted scores.
"""
from __future__ import annotations
import logging
import operator
import sys
import typing as T
from queue import Queue

import cv2
import numpy as np
from tqdm import tqdm

from lib.align import DetectedFace, LandmarkType
from lib.multithreading import FSThread
from lib.utils import get_module_objects, FaceswapError
from lib.infer.identity import Cluster, Identity

from .info_loader import InfoLoader

if T.TYPE_CHECKING:
    from argparse import Namespace
    import numpy.typing as npt
    from lib.align.alignments import PNGHeaderAlignmentsDict
    from lib.infer.runner import ExtractRunner
    from lib.infer.handler import ExtractHandlerFace

logger = logging.getLogger(__name__)


class SortMethod():
    """Parent class for sort methods. All sort methods should inherit from this class

    Parameters
    ----------
    arguments
        The command line arguments passed to the sort process
    loader_type
        The type of image loader to use. "face" just loads the image with the filename, "meta"
        just loads the image alignment data with the filename. "all" loads the image and the
        alignment data with the filename
    is_group
        Set to ``True`` if this class is going to be called exclusively for binning.
        Default: ``False``
    """
    _log_mask_once = False

    def __init__(self,
                 arguments: Namespace,
                 loader_type: T.Literal["face", "meta", "all"] = "meta",
                 is_group: bool = False) -> None:
        logger.debug("Initializing %s: loader_type: '%s' is_group: %s, arguments: %s",
                     self.__class__.__name__, loader_type, is_group, arguments)
        self._is_group = is_group
        self._log_once = True
        self._method = arguments.group_method if self._is_group else arguments.sort_method

        self._num_bins: int = arguments.num_bins
        self._bin_names: list[str] = []

        self._loader_type: T.Literal["face", "meta", "all"] = loader_type
        self._iterator = self._get_file_iterator(arguments.input_dir)

        self._result: list[tuple[str, float | np.ndarray]] = []
        self._binned: list[list[str]] = []
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def loader_type(self) -> T.Literal["face", "meta", "all"]:
        """ ["face", "meta", "all"]: The loader that this sorter uses"""
        return self._loader_type

    @property
    def binned(self) -> list[list[str]]:
        """List of bins (list) containing the filenames belonging to the bin. The binning
        process is called when this property is first accessed"""
        if not self._binned:
            self._binned = self._binning()
            logger.debug({f"bin_{idx}": len(bin_) for idx, bin_ in enumerate(self._binned)})
        return self._binned

    @property
    def sorted_filelist(self) -> list[str]:
        """List of sorted filenames for given sorter in a single list. The sort process is
        called when this property is first accessed"""
        if not self._result:
            self._sort_filelist()
            retval = [item[0] for item in self._result]
            logger.debug(retval)
        else:
            retval = [item[0] for item in self._result]
        return retval

    @property
    def bin_names(self) -> list[str]:
        """The name of each created bin, if they exist, otherwise an empty list"""
        return self._bin_names

    def _get_file_iterator(self, input_dir: str) -> InfoLoader:
        """Override for method specific iterators.

        Parameters
        ----------
        input_dir
            Full path to containing folder of faces to be supported

        Returns
        -------
        The correct InfoLoader iterator for the current sort method
        """
        return InfoLoader(input_dir, self.loader_type)

    def _sort_filelist(self) -> None:
        """Call the sort method's logic to populate the :attr:`_results` attribute.

        Put logic for scoring an individual frame in in :attr:`score_image` of the child

        Returns
        -------
        The sorted file. A list of tuples with the filename in the first position and score in
        the second position
        """
        for filename, image, alignments in self._iterator():
            self.score_image(filename, image, alignments)

        self.sort()
        logger.debug("sorted list: %s",
                     [r[0] if isinstance(r, (tuple, list)) else r for r in self._result])

    @classmethod
    def _get_unique_labels(cls, numbers: np.ndarray) -> list[str]:
        """For a list of threshold values for displaying in the bin name, get the lowest number of
        decimal figures (down to int) required to have a unique set of folder names and return the
        formatted numbers.

        Parameters
        ----------
        numbers
            The list of floating point threshold numbers being used as boundary points

        Returns
        -------
        The string formatted numbers at the lowest precision possible to represent them
        uniquely
        """
        i = 0
        while True:
            rounded = [round(n, i) for n in numbers]
            if len(set(rounded)) == len(numbers):
                break
            i += 1

        if i == 0:
            retval = [str(int(n)) for n in rounded]
        else:
            pre, post = zip(*[str(r).split(".") for r in rounded])
            rpad = max(len(x) for x in post)
            retval = [f"{str(int(left))}.{str(int(right)).ljust(rpad, '0')}"
                      for left, right in zip(pre, post)]
        logger.debug("rounded values: %s, formatted labels: %s", rounded, retval)
        return retval

    def _binning_linear_threshold(self, units: str = "", multiplier: int = 1) -> list[list[str]]:
        """Standard linear binning method for binning by threshold.

        The minimum and maximum result from :attr:`_result` are taken, A range is created between
        these min and max values and is divided to get the number of bins to hold the data

        Parameters
        ----------
        units
            The units to use for the bin name for displaying the threshold values. This this should
            correspond the value in position 1 of :attr:`_result`.
            Default: "" (no units)
        multiplier
            The amount to multiply the contents in position 1 of :attr:`_results` for displaying in
            the bin folder name

        Returns
        -------
        List of bins of filenames
        """
        sizes = np.array([i[1] for i in self._result])
        thresholds = np.linspace(sizes.min(), sizes.max(), self._num_bins + 1)
        labels = self._get_unique_labels(thresholds * multiplier)

        self._bin_names = [f"{self._method}_{idx:03d}_"
                           f"{labels[idx]}{units}_to_{labels[idx + 1]}{units}"
                           for idx in range(self._num_bins)]

        bins: list[list[str]] = [[] for _ in range(self._num_bins)]
        for filename, result in self._result:
            bin_idx = next(bin_id for bin_id, thresh in enumerate(thresholds)
                           if result <= thresh) - 1
            bins[bin_idx].append(filename)

        return bins

    def _binning(self) -> list[list[str]]:
        """Called when :attr:`binning` is first accessed. Checks if sorting has been done, if not
        triggers it, then does binning

        Returns
        -------
        List of bins of filenames
        """
        if not self._result:
            self._sort_filelist()
        retval = self.binning()

        if not self._bin_names:
            self._bin_names = [f"{self._method}_{i:03d}" for i in range(len(retval))]

        logger.debug({bin_name: len(bin_) for bin_name, bin_ in zip(self._bin_names, retval)})

        return retval

    def sort(self) -> None:
        """Override for method specific logic for sorting the loaded statistics.

        The scored list :attr:`_result` should be sorted in place
        """
        raise NotImplementedError()

    def score_image(self,
                    filename: str,
                    image: np.ndarray | None,
                    alignments: PNGHeaderAlignmentsDict | None) -> None:
        """Override for sort method's specific logic. This method should be executed to get a
        single score from a single image  and add the result to :attr:`_result`

        Parameters
        ----------
        filename
            The filename of the currently processing image
        image
            A face image loaded from disk or ``None``
        alignments
            The alignments dictionary for the aligned face or ``None``
        """
        raise NotImplementedError()

    def binning(self) -> list[list[str]]:
        """Group into bins by their sorted score. Override for method specific binning techniques.

        Binning takes the results from :attr:`_result` compiled during :func:`_sort_filelist` and
        organizes into bins for output.

        Returns
        -------
        list
            List of bins of filenames
        """
        raise NotImplementedError()

    @classmethod
    def _mask_face(cls, image: np.ndarray, alignments: PNGHeaderAlignmentsDict) -> np.ndarray:
        """Function for applying the mask to an aligned face if both the face image and alignment
        data are available.

        Parameters
        ----------
        image
            The aligned face image loaded from disk
        alignments
            The alignments data corresponding to the loaded image

        Returns
        -------
        The original image with the mask applied
        """
        det_face = DetectedFace().from_png_meta(alignments)
        det_face.load_aligned(image,
                              size=256,
                              centering="legacy",
                              is_aligned=True)
        aln_face = det_face.aligned
        if aln_face.landmark_type != LandmarkType.LM_2D_68:
            mask = None
        else:
            mask = det_face.get_landmark_mask("face", 0, 0)

        if mask is None and not cls._log_mask_once:
            logger.warning("Masks cannot be generated for the available landmark types. Results "
                           "are likely to be sub-standard")
            cls._log_mask_once = True

        assert aln_face.face is not None
        if mask is None:
            return aln_face.face

        return np.minimum(aln_face.face, mask)


class SortMultiMethod(SortMethod):
    """A Parent sort method that runs 2 different underlying methods (one for sorting one for
    binning) in instances where grouping has been requested, but the sort method is different from
    the group method

    Parameters
    ----------
    arguments
        The command line arguments passed to the sort process
    sort_method
        A sort method object for sorting the images
    group_method
        A sort method object used for sorting and binning the images
    """
    def __init__(self,
                 arguments: Namespace,
                 sort_method: SortMethod,
                 group_method: SortMethod) -> None:
        self._sorter = sort_method
        self._grouper = group_method
        self._is_built = False
        super().__init__(arguments)

    def _get_file_iterator(self, input_dir: str) -> InfoLoader:
        """Override to get a group specific iterator. If the sorter and grouper use the same kind
        of iterator, use that. Otherwise return the 'all' iterator, as which ever way it is cut all
        outputs will be required. Monkey patch the actual loader used into the children in case of
        any callbacks.

        Parameters
        ----------
        input_dir
            Full path to containing folder of faces to be supported

        Returns
        -------
        The correct InfoLoader iterator for the current sort method
        """
        if self._sorter.loader_type == self._grouper.loader_type:
            retval = InfoLoader(input_dir, self._sorter.loader_type)
        else:
            retval = InfoLoader(input_dir, "all")
        self._sorter._iterator = retval  # pylint:disable=protected-access
        self._grouper._iterator = retval  # pylint:disable=protected-access
        return retval

    def score_image(self,
                    filename: str,
                    image: np.ndarray | None,
                    alignments: PNGHeaderAlignmentsDict | None) -> None:
        """Score a single image for sort method and add the result to :attr:`_result`

        Parameters
        ----------
        filename
            The filename of the currently processing image
        image
            A face image loaded from disk or ``None``
        alignments
            The alignments dictionary for the aligned face or ``None``
        """
        self._sorter.score_image(filename, image, alignments)
        self._grouper.score_image(filename, image, alignments)

    def sort(self) -> None:
        """Sort the sorter and grouper methods"""
        logger.debug("Sorting")
        self._sorter.sort()
        self._result = self._sorter.sorted_filelist  # type:ignore
        self._grouper.sort()
        self._binned = self._grouper.binned
        self._bin_names = self._grouper.bin_names
        logger.debug("Sorted")

    def binning(self) -> list[list[str]]:
        """Override standard binning, to bin by the group-by method and sort by the sorting
        method.

        Go through the grouped binned results, and reorder each bin contents based on the
        sorted list

        Returns
        -------
        List of bins of filenames
        """
        sorted_ = self._result
        output: list[list[str]] = []
        for bin_ in tqdm(self._binned, desc="Binning and sorting", file=sys.stdout, leave=False):
            indices: dict[int, str] = {}
            for filename in bin_:
                indices[sorted_.index(filename)] = filename
            output.append([indices[idx] for idx in sorted(indices)])
        return output


class SortBlur(SortMethod):
    """Sort images by blur or blur-fft amount

    Parameters
    ----------
    arguments
        The command line arguments passed to the sort process
    is_group
        Set to ``True`` if this class is going to be called exclusively for binning.
        Default: ``False``
    """
    def __init__(self, arguments: Namespace, is_group: bool = False) -> None:
        super().__init__(arguments, loader_type="all", is_group=is_group)
        method = arguments.group_method if self._is_group else arguments.sort_method
        self._use_fft = method == "blur_fft"

    def estimate_blur(self, image: np.ndarray, alignments=None) -> float:
        """Estimate the amount of blur an image has with the variance of the Laplacian.
        Normalize by pixel number to offset the effect of image size on pixel gradients & variance.

        Parameters
        ----------
        image
            The face image to calculate blur for
        alignments
            The metadata for the face image or ``None`` if no metadata is available. If metadata is
            provided the face will be masked by the "components" mask prior to calculating blur.
            Default:``None``

        Returns
        -------
        The estimated blur score for the face
        """
        if alignments is not None:
            image = self._mask_face(image, alignments)
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_map = T.cast(np.ndarray, cv2.Laplacian(image, cv2.CV_32F))
        score = np.var(blur_map) / np.sqrt(image.shape[0] * image.shape[1])
        return score

    def estimate_blur_fft(self,
                          image: np.ndarray,
                          alignments: PNGHeaderAlignmentsDict | None = None) -> float:
        """Estimate the amount of blur a fft filtered image has.

        Parameters
        ----------
        image
            Use Fourier Transform to analyze the frequency characteristics of the masked
            face using 2D Discrete Fourier Transform (DFT) filter to find the frequency domain.
            A mean value is assigned to the magnitude spectrum and returns a blur score.
            Adapted from https://www.pyimagesearch.com/2020/06/15/
            opencv-fast-fourier-transform-fft-for-blur-detection-in-images-and-video-streams/
        alignments
            The metadata for the face image or ``None`` if no metadata is available. If metadata is
            provided the face will be masked by the "components" mask prior to calculating blur.
            Default:``None``

        Returns
        -------
        The estimated fft blur score for the face
        """
        if alignments is not None:
            image = self._mask_face(image, alignments)

        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        height, width = image.shape
        c_height, c_width = (int(height / 2.0), int(width / 2.0))
        fft = np.fft.fft2(image)
        fft_shift = np.fft.fftshift(fft)
        fft_shift[c_height - 75:c_height + 75, c_width - 75:c_width + 75] = 0
        ifft_shift = np.fft.ifftshift(fft_shift)
        shift_back = np.fft.ifft2(ifft_shift)
        magnitude = np.log(np.abs(shift_back))
        score = float(np.mean(magnitude))

        return score

    def score_image(self,
                    filename: str,
                    image: np.ndarray | None,
                    alignments: PNGHeaderAlignmentsDict | None) -> None:
        """Score a single image for blur or blur-fft and add the result to :attr:`_result`

        Parameters
        ----------
        filename
            The filename of the currently processing image
        image
            A face image loaded from disk
        alignments
            The alignments dictionary for the aligned face or ``None``
        """
        assert image is not None
        if self._log_once:
            msg = "Grouping" if self._is_group else "Sorting"
            inf = " fft_filtered" if self._use_fft else ""
            logger.info("%s by estimated%s image blur...", msg, inf)
            self._log_once = False

        estimator = self.estimate_blur_fft if self._use_fft else self.estimate_blur
        self._result.append((filename, estimator(image, alignments)))

    def sort(self) -> None:
        """Sort by metric score. Order in reverse for distance sort."""
        logger.info("Sorting...")
        self._result = sorted(self._result, key=operator.itemgetter(1), reverse=True)

    def binning(self) -> list[list[str]]:
        """Create bins to split linearly from the lowest to the highest sample value

        Returns
        -------
        List of bins of filenames
        """
        return self._binning_linear_threshold(multiplier=100)


class SortColor(SortMethod):
    """Score by channel average intensity or black pixels.

    Parameters
    ----------
    arguments
        The command line arguments passed to the sort process
    is_group
        Set to ``True`` if this class is going to be called exclusively for binning.
        Default: ``False``
    """
    def __init__(self, arguments: Namespace, is_group: bool = False) -> None:
        super().__init__(arguments, loader_type="face", is_group=is_group)
        self._desired_channel = {'gray': 0, 'luma': 0, 'orange': 1, 'green': 2}

        method = arguments.group_method if self._is_group else arguments.sort_method
        self._method = method.replace("color_", "")

    def _convert_color(self, image: np.ndarray) -> np.ndarray:
        """Helper function to convert color spaces

        Parameters
        ----------
        image
            The original image to convert color space for

        Returns
        -------
        The color converted image
        """
        if self._method == 'gray':
            conversion = np.array([[0.0722], [0.7152], [0.2126]])
        else:
            conversion = np.array([[0.25, 0.5, 0.25], [-0.5, 0.0, 0.5], [-0.25, 0.5, -0.25]])

        operation = 'ijk, kl -> ijl' if self._method == "gray" else 'ijl, kl -> ijk'
        path = np.einsum_path(operation, image[..., :3], conversion, optimize='optimal')[0]
        return np.einsum(operation, image[..., :3], conversion, optimize=path).astype('float32')

    def _near_split(self, bin_range: int) -> list[int]:
        """Obtain the split for the given number of bins for the given range

        Parameters
        ----------
        bin_range
            The range of data to separate into bins

        Returns
        -------
        The split dividers for the given number of bins for the given range
        """
        quotient, remainder = divmod(bin_range, self._num_bins)
        seps = [quotient + 1] * remainder + [quotient] * (self._num_bins - remainder)
        uplimit = 0
        bins = [0]
        for sep in seps:
            bins.append(uplimit + sep)
            uplimit += sep
        return bins

    def binning(self) -> list[list[str]]:
        """Group into bins by percentage of black pixels"""
        # TODO. Only grouped by black pixels. Check color

        logger.info("Grouping by percentage of %s...", self._method)

        # Starting the binning process
        bins: list[list[str]] = [[] for _ in range(self._num_bins)]
        # Get edges of bins from 0 to 100
        bins_edges = self._near_split(100)
        # Get the proper bin number for each img order
        img_bins = np.digitize([float(x[1]) for x in self._result], bins_edges, right=True)

        # Place imgs in bins
        for idx, _bin in enumerate(img_bins):
            bins[_bin].append(self._result[idx][0])

        retval = [b for b in bins if b]
        return retval

    def score_image(self,
                    filename: str,
                    image: np.ndarray | None,
                    alignments: PNGHeaderAlignmentsDict | None) -> None:
        """Score a single image for color

        Parameters
        ----------
        filename
            The filename of the currently processing image
        image
            A face image loaded from disk
        alignments
            The alignments dictionary for the aligned face or ``None``
        """
        if self._log_once:
            msg = "Grouping" if self._is_group else "Sorting"
            if self._method == "black":
                logger.info("%s by percentage of black pixels...", msg)
            else:
                logger.info("%s by channel average intensity...", msg)
            self._log_once = False

        assert image is not None
        if self._method == "black":
            score = np.ndarray.all(image == [0, 0, 0], axis=2).sum()/image.size*100*3
        else:
            channel_to_sort = self._desired_channel[self._method]
            score = np.average(self._convert_color(image), axis=(0, 1))[channel_to_sort]
        self._result.append((filename, score))

    def sort(self) -> None:
        """Sort by metric score. Order in reverse for distance sort."""
        if self._method == "black":
            self._sort_black_pixels()
            return
        self._result = sorted(self._result, key=operator.itemgetter(1), reverse=True)

    def _sort_black_pixels(self) -> None:
        """Sort by percentage of black pixels

        Calculates the sum of black pixels, gets the percentage X 3 channels
        """
        img_list_len = len(self._result)
        for i in tqdm(range(0, img_list_len - 1),
                      desc="Comparing black pixels", file=sys.stdout,
                      leave=False):
            for j in range(0, img_list_len-i-1):
                if self._result[j][1] > self._result[j+1][1]:
                    temp = self._result[j]
                    self._result[j] = self._result[j+1]
                    self._result[j+1] = temp


class SortFace(SortMethod):
    """Sort by identity similarity using an Identity plugin

    Parameters
    ----------
    arguments
        The command line arguments passed to the sort process
    is_group
        Set to ``True`` if this class is going to be called exclusively for binning.
        Default: ``False``
    """
    def __init__(self, arguments: Namespace, is_group: bool = False) -> None:
        super().__init__(arguments, loader_type="all", is_group=is_group)
        self._plugin = Identity(arguments.identity, config_file=arguments.config_file)
        self._runner: ExtractRunner[ExtractHandlerFace] | None = None
        self._storage_name = self._plugin.storage_name
        threshold = arguments.threshold
        self._threshold: float | None = 0.25 if threshold < 0 else threshold

        self._count_seen = 0
        self._plugin_thread = FSThread(self._score_from_plugin)
        self._from_plugin: list[tuple[str, npt.NDArray[np.float32]]] = []
        self._alignment_queue: Queue[tuple[str, PNGHeaderAlignmentsDict]] = Queue(
            maxsize=self._plugin.batch_size * 3)

    def _score_from_header(self,
                           filename: str,
                           alignments: PNGHeaderAlignmentsDict) -> bool:
        """Reads header information from the PNG file to look for the identity embedding

        Parameters
        ----------
        filename
            The filename of the currently processing image
        alignments
            The alignments dictionary for the aligned face or ``None``

        Returns
        -------
        ``True`` if embedding information was read from the PNG header, otherwise ``False``
        """
        if not alignments.get("identity", {}).get(self._storage_name):
            return False
        embedding = np.array(alignments["identity"][self._storage_name], dtype="float32")
        self._result.append((filename, embedding))
        return True

    def _score_from_plugin(self):
        """Obtain the embedding from the identity plugin"""
        logger.info("%s Embeddings are being written to the image header. "
                    "Sorting by this method should be quicker next time",
                    self._storage_name.title())

        assert self._runner is not None
        for media in self._runner:
            if self._plugin_thread.error_state.has_error:
                self._plugin_thread.error_state.re_raise()
            embedding = media.detected_faces[0].identity[self._storage_name]
            self._from_plugin.append((media.filename, embedding))
            filename, alignments = self._alignment_queue.get()
            assert filename == media.filename
            alignments.setdefault("identity", {})[self._storage_name] = embedding.tolist()
            self._iterator.update_png_header(filename, alignments)

    def _handle_plugin(self) -> None:
        """Check if we have seen all input and shutdown the plugin if so"""
        if self._count_seen != self._iterator.filelist_count:
            return

        if not self._plugin_thread.is_alive():
            return
        assert self._runner is not None
        logger.debug("Shutting down Identity plugin")
        self._runner.stop()
        self._plugin_thread.join()
        self._result += self._from_plugin

    def score_image(self,
                    filename: str,
                    image: np.ndarray | None,
                    alignments: PNGHeaderAlignmentsDict | None) -> None:
        """Score a single image for sort method and add the result to :attr:`_result`. Attempts
        to pull identity information from the PNG metadata. If not available, pulls the information
        from the Identity plugin and stores in the PNG header for future use

        Parameters
        ----------
        filename
            The filename of the currently processing image
        image
            A face image loaded from disk or ``None``
        alignments
            The alignments dictionary for the aligned face or ``None``
        """
        if not alignments:
            msg = ("The images to be sorted do not contain alignment data. Images must have "
                   "been generated by Faceswap's Extract process.\nIf you are sorting an "
                   "older face set, then you should re-extract the faces from your source "
                   "alignments file to generate this data.")
            raise FaceswapError(msg)

        if self._plugin_thread.error_state.has_error:
            self._plugin_thread.error_state.re_raise()

        self._count_seen += 1
        if self._score_from_header(filename, alignments):
            self._handle_plugin()
            return

        if not self._plugin_thread.is_alive():
            logger.debug("Starting Identity plugin")
            self._runner = self._plugin()
            self._plugin_thread.start()

        self._alignment_queue.put((filename, alignments))

        face = DetectedFace(left=alignments["x"],  # Only include required items
                            width=alignments["w"],
                            top=alignments["y"],
                            height=alignments["h"],
                            landmarks_xy=np.array(alignments["landmarks_xy"], dtype="float32"))
        assert self._runner is not None
        try:
            self._runner.put(filename,
                             T.cast("npt.NDArray[np.uint8]", image),
                             [face],
                             is_aligned=True,
                             frame_metadata=self._iterator.cached_source_data[filename])
        except Exception:
            self._plugin_thread.error_state.set(sys.exc_info())
            raise
        self._handle_plugin()

    def sort(self) -> None:
        """Sort by dendrogram.

        Parameters
        ----------
        matched_list
            The list of tuples with filename in first position and face encoding in the 2nd

        Returns
        -------
        The original list, sorted for this metric
        """
        logger.info("Sorting by ward linkage. This may take some time...")
        results = np.array([item[1] for item in self._result])
        indices = Cluster(np.array(results), "ward", threshold=self._threshold)()
        self._result = [(self._result[idx][0], float(score)) for idx, score in indices]

    def binning(self) -> list[list[str]]:
        """Group into bins by their sorted score

        The bin ID has been output in the 2nd column of :attr:`_result` so use that for binning

        Returns
        -------
        List of bins of filenames
        """
        num_bins = len(set(int(i[1]) for i in self._result))
        logger.info("Grouping by %s...", self.__class__.__name__.replace("Sort", ""))
        bins: list[list[str]] = [[] for _ in range(num_bins)]

        for filename, bin_id in self._result:
            bins[int(bin_id)].append(filename)

        return bins


class SortHistogram(SortMethod):
    """Sort by image histogram similarity or dissimilarity

    Parameters
    ----------
    arguments
        The command line arguments passed to the sort process
    is_group
        Set to ``True`` if this class is going to be called exclusively for binning.
        Default: ``False``
    """
    def __init__(self, arguments: Namespace, is_group: bool = False) -> None:
        super().__init__(arguments, loader_type="all", is_group=is_group)
        method = arguments.group_method if self._is_group else arguments.sort_method
        self._is_dissim = method == "hist-dissim"
        self._threshold: float = 0.3 if arguments.threshold < 0.0 else arguments.threshold

    def _calc_histogram(self,
                        image: np.ndarray,
                        alignments: PNGHeaderAlignmentsDict | None) -> np.ndarray:
        if alignments:
            image = self._mask_face(image, alignments)
        return cv2.calcHist([image], [0], None, [256], [0, 256])

    def _sort_dissim(self) -> None:
        """Sort histograms by dissimilarity"""
        result = T.cast(list[tuple[str, np.ndarray]], self._result)
        img_list_len = len(result)
        for i in tqdm(range(0, img_list_len),
                      desc="Comparing histograms",
                      file=sys.stdout,
                      leave=False):
            score_total = 0.0
            for j in range(0, img_list_len):
                if i == j:
                    continue
                score_total += cv2.compareHist(result[i][1],
                                               result[j][1],
                                               cv2.HISTCMP_BHATTACHARYYA)
            result[i][2] = score_total

        self._result = sorted(result, key=operator.itemgetter(2), reverse=True)

    def _sort_sim(self) -> None:
        """Sort histograms by similarity"""
        result = T.cast(list[tuple[str, np.ndarray]], self._result)
        img_list_len = len(result)
        for i in tqdm(range(0, img_list_len - 1),
                      desc="Comparing histograms",
                      file=sys.stdout,
                      leave=False):
            min_score = float("inf")
            j_min_score = i + 1
            for j in range(i + 1, img_list_len):
                score = cv2.compareHist(result[i][1],
                                        result[j][1],
                                        cv2.HISTCMP_BHATTACHARYYA)
                if score < min_score:
                    min_score = score
                    j_min_score = j
            (self._result[i + 1], self._result[j_min_score]) = (result[j_min_score], result[i + 1])

    @classmethod
    def _get_avg_score(cls, image: np.ndarray, references: list[np.ndarray]) -> float:
        """Return the average histogram score between a face and reference images

        Parameters
        ----------
        image
            The image to test
        references: list
            List of reference images to test the original image against

        Returns
        -------
        The average score between the histograms
        """
        scores = []
        for img2 in references:
            score = cv2.compareHist(image, img2, cv2.HISTCMP_BHATTACHARYYA)
            scores.append(score)
        return sum(scores) / len(scores)

    def binning(self) -> list[list[str]]:
        """Group into bins by histogram"""
        # pylint:disable=duplicate-code
        msg = "dissimilarity" if self._is_dissim else "similarity"
        logger.info("Grouping by %s...", msg)

        # Groups are of the form: group_num -> reference histogram
        reference_groups: dict[int, list[np.ndarray]] = {}

        # Bins array, where index is the group number and value is
        # an array containing the file paths to the images in that group
        bins: list[list[str]] = []

        threshold = self._threshold

        img_list_len = len(self._result)
        reference_groups[0] = [T.cast(np.ndarray, self._result[0][1])]
        bins.append([self._result[0][0]])

        for i in tqdm(range(1, img_list_len),
                      desc="Grouping",
                      file=sys.stdout,
                      leave=False):
            current_key = -1
            current_score = float("inf")
            img = T.cast(np.ndarray, self._result[i][1])
            for key, value in reference_groups.items():
                score = self._get_avg_score(img, value)
                if score < current_score:
                    current_key, current_score = key, score

            if current_score < threshold:
                reference_groups[T.cast(int, current_key)].append(img)
                bins[current_key].append(self._result[i][0])
            else:
                reference_groups[len(reference_groups)] = [img]
                bins.append([self._result[i][0]])

        return bins

    def score_image(self,
                    filename: str,
                    image: np.ndarray | None,
                    alignments: PNGHeaderAlignmentsDict | None) -> None:
        """Collect the histogram for the given face

        Parameters
        ----------
        filename: str
            The filename of the currently processing image
        image
            A face image loaded from disk
        alignments
            The alignments dictionary for the aligned face or ``None``
        """
        if self._log_once:
            msg = "Grouping" if self._is_group else "Sorting"
            logger.info("%s by histogram similarity...", msg)
            self._log_once = False

        assert image is not None
        self._result.append((filename, self._calc_histogram(image, alignments)))

    def sort(self) -> None:
        """Sort by histogram."""
        logger.info("Comparing histograms and sorting...")
        if self._is_dissim:
            self._sort_dissim()
            return
        self._sort_sim()


__all__ = get_module_objects(__name__)
