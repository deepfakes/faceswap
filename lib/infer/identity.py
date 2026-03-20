#! /usr/env/bin/python3
"""Handles face identity plugins and runners"""
from __future__ import annotations

import logging
import os
import sys
import typing as T

import cv2
import numpy as np
import psutil
from fastcluster import linkage, linkage_vector

from lib.align.detected_face import DetectedFace
from lib.image import png_read_meta
from lib.logger import parse_class_init
from lib.utils import FaceswapError, get_module_objects, IMAGE_EXTENSIONS

from .objects import ExtractBatch
from .handler import ExtractHandlerFace

if T.TYPE_CHECKING:
    import numpy.typing as npt
    from collections.abc import Generator
    from lib.align.alignments import PNGHeaderDict
    from .runner import ExtractRunner

logger = logging.getLogger(__name__)


class Identity(ExtractHandlerFace):
    """Responsible for handling Identity/Recognition plugins within the extract pipeline

    Parameters
    ----------
    plugin
        The plugin that this runner is to use
    filter_threshold
        The threshold to use when filtering faces by identity. Default: 0.4
    compile_model
        ``True`` to compile any PyTorch models
    config_file
        Full path to a custom config file to load. ``None`` for default config
    """
    def __init__(self,
                 plugin: str,
                 threshold: float = 0.4,
                 compile_model: bool = False,
                 config_file: str | None = None) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__(plugin, compile_model=compile_model, config_file=config_file)
        self._filter = IdentityFilter(threshold, self.storage_name)

    def __repr__(self) -> str:
        """Pretty print for logging"""
        retval = super().__repr__()[:-1]
        retval += (f", threshold={self._filter.threshold})")
        return retval

    def pre_process(self, batch: ExtractBatch) -> None:
        """Obtain the aligned face images at the requested size, centering and image format.
        Perform any plugin specific pre-processing

        Parameters
        ----------
        batch
            The incoming ExtractBatch to use for pre-processing
        """
        self._maybe_log_warning(batch.landmark_type)
        if batch.is_aligned:
            data = self._get_faces_aligned(batch.images,
                                           batch.frame_ids,
                                           batch.aligned.offsets_head,
                                           getattr(batch.aligned, self._aligned_offsets_name))
        else:
            matrices = self._get_matrices(getattr(batch.aligned, self._aligned_mat_name))
            data = self._get_faces(batch.images, batch.frame_ids, matrices, with_alpha=False)
        data = self._format_images(data)
        batch.data = self.plugin.pre_process(data)

    def post_process(self, batch: ExtractBatch) -> None:
        """Perform recognition post processing.

        Obtains the final output from the identity plugin and performs any plugin specific post-
        processing

        Parameters
        ----------
        batch
            The incoming ExtractBatch to use for post-processing
        """
        identity = batch.data
        if self._overridden["post_process"]:
            identity = self.plugin.post_process(identity)
        batch.identities[self.storage_name] = identity
        self._filter(batch)

    def add_filter_identities(self, identities: npt.NDArray[np.float32], is_filter: bool) -> None:
        """Add the given identities to the identity filter

        Parameters
        ----------
        identities
            The identity embeddings to add to the filter
        is_filter
            ``True`` for filter, ``False`` for nFilter
        """
        self._filter.add_identities(identities, is_filter)

    def output_info(self) -> None:
        """Output the counts from the identity filter"""
        self._filter.output_counts()


class IdentityFilter:
    """Handles filtering of faces based on provided image files

    Parameters
    ----------
    threshold
        The threshold value for filtering out items
    name
        The name of the identity plugin running
    """
    def __init__(self, threshold: float, name: str) -> None:
        logger.debug(parse_class_init(locals()))
        self.threshold = threshold
        """The threshold for accepting a filter result"""
        self._plugin_name = name
        self._name = f"{name}.Filter"

        self._filters = {"filter": np.empty([0], dtype="float32"),
                         "nfilter": np.empty([0], dtype="float32")}
        self._active: set[T.Literal["filter", "nfilter"]] = set()
        self._counts = {"filter": 0, "nfilter": 0, "combined": 0}
        self._active_count = 0
        self.enabled = False
        """``True`` if the identity filter is enabled"""

    def add_identities(self, identities: npt.NDArray[np.float32], is_filter: bool) -> None:
        """Add the given identities to the filter

        Parameters
        ----------
        identities
            The identity embeddings to add to the filter
        is_filter
            ``True`` for filter, ``False`` for nFilter
        """
        logger.debug("[%s] Adding identities: %s, is_filter: %s",
                     self._name, identities.shape, is_filter)
        key: T.Literal["filter", "nfilter"] = "filter" if is_filter else "nfilter"
        self._filters[key] = identities
        if np.any(identities):
            self._active.add(key)
        self.enabled = bool(self._active)
        self._active_count = len(self._active)

    def output_counts(self) -> None:
        """If filter is enabled info log the number of faces filtered"""
        # pylint:disable=duplicate-code
        if not self.enabled:
            return
        counts = []
        for key, count in self._counts.items():
            if not count:
                continue
            txt = key.title() if key != "nfilter" else "nFilter"
            counts.append(txt + f": {count}")
        if counts:
            logger.info("[Identity filter] %s", ", ".join(counts))

    @classmethod
    def _find_cosine_similarity(cls,
                                source: npt.NDArray[np.float32],
                                batch: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
        """Find the cosine similarity between a source face identity and a test face identity

        Parameters
        ---------
        source
            The identity encoding for the source face identities
        batch
            A batch of face identities to test against the sources

        Returns
        -------
        The cosine similarity between the face identities and the source identities
        """
        s_norms = source / np.linalg.norm(source, axis=1, keepdims=True)
        t_norms = batch / np.linalg.norm(batch, axis=1, keepdims=True)
        retval = t_norms @ s_norms.T
        return retval

    def __call__(self, batch: ExtractBatch) -> None:
        """Apply the identity filter to the given batch

        Parameters
        ----------
        batch
            The batch object to perform filtering on with the identities populated
        """
        if not self.enabled:
            return
        identities = batch.identities[self._plugin_name]
        mask = np.empty((self._active_count, batch.bboxes.shape[0]), dtype="bool")
        for idx, f_type in enumerate(sorted(self._active)):
            similarities = self._find_cosine_similarity(self._filters[f_type], identities)
            matches = np.any(similarities >= self.threshold, axis=1)
            mask[idx] = ~matches if f_type == "nfilter" else matches
            self._counts[f_type] += int(np.sum(~mask[idx]))

        if np.all(mask):
            return

        if self._active_count > 1:
            mask = T.cast("npt.NDArray[np.bool_]", mask.all(axis=0))
            self._counts["combined"] += int(np.sum(~mask))
        else:
            mask = mask[0]
        batch.apply_mask(mask)


class FilterLoader:
    """Obtains face embeddings from images and loads the IdentityFilter as part of the extraction
    pipeline

    Parameters
    ----------
    threshold
        The threshold value for filtering out items. Default: 0.4
    filter_files
        The list of full paths to the files to use for filtering. Default: ``None`` (don't use
        filter)
    nfilter_files
        The list of full paths to the files to use to nfilter. Default: ``None`` (don't use
        nfilter)
    """
    def __init__(self,
                 threshold: float,
                 filter_files: list[str] | None,
                 nfilter_files: list[str] | None) -> None:
        logger.debug(parse_class_init(locals()))
        self.threshold = threshold
        """The threshold value for filtering out items"""
        self.enabled = False
        """``True`` if identity face filtering is enabled"""
        if not filter_files and not nfilter_files:
            return
        self.enabled = True

        self._filter_files = self._validate_paths(filter_files, True)
        self._nfilter_files = self._validate_paths(nfilter_files, False)

        if self._filter_files.intersection(self._nfilter_files):
            logger.error("Filter and nFilter files should be unique. The following path(s) exist "
                         "in both: %s", self._filter_files.intersection(self._nfilter_files))
            sys.exit(1)

        self._runner: ExtractRunner[ExtractHandlerFace]

    def _validate_paths(self, full_paths: list[str] | None, is_filter: bool) -> set[str]:
        """Validates that the given image file paths are valid. Exits if paths are provided but no
        images could be found

        Parameters
        ----------
        full_paths
            The list of full paths to images to validate
        is_filter
            ``True`` for filter files. ``False`` for nfilter files

        Returns
        -------
        The list of validated full paths
        """
        if not full_paths:
            return set()
        name = "Filter" if is_filter else ("nFilter")
        retval: list[str] = []
        for file_path in full_paths:

            if os.path.isdir(file_path):
                files = [os.path.join(file_path, fname)
                         for fname in os.listdir(file_path)
                         if os.path.splitext(fname)[-1].lower() in IMAGE_EXTENSIONS]
                if not files:
                    logger.warning("%s folder '%s' contains no image files", name, file_path)
                else:
                    retval.extend(files)
                continue

            if not os.path.splitext(file_path)[-1] in IMAGE_EXTENSIONS:
                logger.warning("%s file '%s' is not an image file. Skipping", name, file_path)
                continue
            if not os.path.isfile(file_path):
                logger.warning("%s file '%s' does not exist. Skipping", name, file_path)
                continue
            retval.append(file_path)

        if not retval:
            logger.error("None of the provided %s files are valid.", name)
            sys.exit(1)

        unique = set(retval)
        logger.debug("[IdentityFilter] %s files: %s", name, unique)
        return unique

    def add_identity_plugin(self, runner: ExtractRunner) -> None:
        """Add the identity plugin for updating with embedding information

        Parameters
        ----------
        runner
            The identity runner for the pipeline
        """
        logger.debug("[IdentityFilter] Adding identity runner: %s", runner)
        self._runner = runner

    @classmethod
    def _get_meta(cls, filename: str, image: bytes) -> PNGHeaderDict | None:
        """Obtain the embedded meta data from a faceswap aligned image

        Parameters
        ----------
        filename
            Full path to the image file to load
        image
            The raw loaded image to obtain the meta data from

        Returns
        -------
        The faceswap meta data from a PNG image header
        """
        if os.path.splitext(filename)[-1].lower() != ".png":
            logger.debug("[IdentityFilter] '%s' not a png", filename)
            return None

        try:
            meta = png_read_meta(image)
        except AssertionError:
            logger.debug("[IdentityFilter] '%s' is not a faceswap extracted image", filename)
            return None

        if "alignments" not in meta:
            logger.debug("[IdentityFilter] '%s' is not a faceswap extracted image", filename)
            return None

        return T.cast("PNGHeaderDict", meta)

    def _from_pipeline(self, pipeline: ExtractRunner, images: dict[str, npt.NDArray[np.uint8]]
                       ) -> dict[str, npt.NDArray[np.float32]]:
        """Obtain embeddings from the full extraction pipeline when non-faceswap images have been
        provided

        Parameters
        ----------
        pipeline
            The extraction pipelines for obtaining embeddings from non-faceswap images
        images
            Dictionary of full file paths to images to run extraction on

        Returns
        -------
        The identity embeddings received for each image from the extraction pipeline
        """
        retval: dict[str, npt.NDArray[np.float32]] = {}
        for file_name, image in images.items():
            logger.debug("[IdentityFilter] Putting to extractor: '%s'", file_name)
            retval[file_name] = np.array(
                [f.identity[self._runner.handler.storage_name]
                 for f in pipeline.put(file_name, image, passthrough=True).detected_faces]
                ).squeeze(0)

        logger.debug("[IdentityFilter] Identity from extraction: %s",
                     {k: v.shape for k, v in retval.items()})
        return retval

    def _from_plugin(self, images: dict[str, tuple[PNGHeaderDict, npt.NDArray[np.uint8]]]
                     ) -> dict[str, npt.NDArray[np.float32]]:
        """Obtain embeddings from the identity when faceswap aligned images without identity
        information have been provided

        Parameters
        ----------
        images
            Dictionary of full file paths to the faceswap meta information and aligned images to
            obtain identity information for

        Returns
        -------
        The identity embeddings received for each image from the extraction pipeline
        """
        retval: dict[str, npt.NDArray[np.float32]] = {}
        for fname, (meta, image) in images.items():
            logger.debug("[IdentityFilter] Putting to plugin: '%s'", fname)
            out = self._runner.put_direct(fname,
                                          image,
                                          [DetectedFace().from_png_meta(meta["alignments"])],
                                          is_aligned=True,
                                          frame_size=meta["source"]["source_frame_dims"])
            retval[fname] = out.identities[self._runner.handler.plugin.storage_name].squeeze(0)

        logger.debug("[IdentityFilter] Identity from plugin: %s",
                     {k: v.shape for k, v in retval.items()})
        return retval

    def _add_embeds_to_plugin(self, embeds: dict[str, npt.NDArray[np.float32]]) -> None:
        """Validate that we have exactly one embedding per image and add to the identity filter

        Parameters
        ----------
        embeds
            The file name with embeddings to add to the plugin filter
        """
        for is_filter, file_list in zip((True, False), (self._filter_files, self._nfilter_files)):
            if not file_list:
                continue
            collated: list[npt.NDArray[np.float32]] = []
            name = "Filter" if is_filter else "nFilter"
            for fname in file_list:
                embed = embeds.pop(fname)
                if not np.any(embed):
                    logger.warning("%s file '%s' contains no detected faces. Skipping",
                                   name, os.path.basename(fname))
                    continue
                if embed.ndim != 1 and is_filter:
                    logger.warning("%s file '%s' contains %s detected faces. Skipping",
                                   name, os.path.basename(fname), embed.shape[0])
                    continue
                if embed.ndim != 1 and not is_filter:
                    logger.warning("%s file '%s' contains %s detected faces. All of "
                                   "these identities will be used",
                                   name, os.path.basename(fname), embed.shape[0])
                    collated.extend(list(embed))
                    continue
                collated.append(embed)
            if not collated:
                logger.error("None of the provided %s files are valid.", name)
                sys.exit(1)
            logger.info("Adding %s face%s to Identity %s",
                        len(collated), "s" if len(collated) > 1 else "", name)
            T.cast(Identity, self._runner.handler).add_filter_identities(
                np.stack(collated, dtype="float32"), is_filter)

    def get_embeddings(self, pipeline: ExtractRunner) -> None:
        """Obtain the embeddings that are to be used for face filtering and add to the identity
        plugin

        Parameters
        ----------
        pipeline
            The extraction pipelines for obtaining embeddings from non-faceswap images
        """
        embeds: dict[str, npt.NDArray[np.float32]] = {}
        non_aligned: dict[str, npt.NDArray[np.uint8]] = {}
        aligned: dict[str, tuple[PNGHeaderDict, npt.NDArray[np.uint8]]] = {}

        for filepath in self._filter_files.union(self._nfilter_files):
            with open(filepath, "rb") as in_file:
                raw_image = in_file.read()

            meta = self._get_meta(filepath, raw_image)
            if meta is not None:
                idn = T.cast(dict[str, list], meta.get("identity", {}))
                embed = np.array(idn.get(self._runner.handler.storage_name, []),
                                 dtype="float32")
                if np.any(embed):
                    logger.debug("[IdentityFilter] Identity from header '%s'. Shape: %s",
                                 filepath, embed.shape)
                    embeds[filepath] = embed
                    continue

            image = T.cast("npt.NDArray[np.uint8]",
                           cv2.imdecode(np.frombuffer(raw_image, dtype="uint8"), cv2.IMREAD_COLOR))

            if meta is None:
                non_aligned[filepath] = image
                continue

            logger.debug("[IdentityFilter] No identity in header: '%s'", filepath)
            aligned[filepath] = (meta, image)

        if aligned or non_aligned:
            logger.info("Extracting faces for Identity Filter...")
        if non_aligned:
            embeds |= self._from_pipeline(pipeline, non_aligned)
        if aligned:
            embeds |= self._from_plugin(aligned)
        self._add_embeds_to_plugin(embeds)


class Cluster():
    """Cluster the outputs from a VGG-Face 2 Model

    Parameters
    ----------
    predictions
        A stacked matrix of identity predictions of the shape (`N`, `D`) where `N` is the
        number of observations and `D` are the number of dimensions.  NB: The given
        :attr:`predictions` will be overwritten to save memory. If you still require the
        original values you should take a copy prior to running this method
    method
        The clustering method to use.
    threshold
        The threshold to start creating bins for. Set to ``None`` to disable binning
    """

    def __init__(self,
                 predictions: np.ndarray,
                 method: T.Literal["single", "centroid", "median", "ward"],
                 threshold: float | None = None) -> None:
        logger.debug("Initializing: %s (predictions: %s, method: %s, threshold: %s)",
                     self.__class__.__name__, predictions.shape, method, threshold)
        self._num_predictions = predictions.shape[0]

        self._should_output_bins = threshold is not None
        self._threshold = 0.0 if threshold is None else threshold
        self._bins: dict[int, int] = {}
        self._iterator = self._integer_iterator()

        self._result_linkage = self._do_linkage(predictions, method)
        logger.debug("Initialized %s", self.__class__.__name__)

    @classmethod
    def _integer_iterator(cls) -> Generator[int, None, None]:
        """Iterator that just yields consecutive integers"""
        i = -1
        while True:
            i += 1
            yield i

    def _use_vector_linkage(self, dims: int) -> bool:
        """Calculate the RAM that will be required to sort these images and select the appropriate
        clustering method.

        From fastcluster documentation:
            "While the linkage method requires Θ(N:sup:`2`) memory for clustering of N points, this
            [vector] method needs Θ(N D)for N points in RD, which is usually much smaller."
            also:
            "half the memory can be saved by specifying :attr:`preserve_input`=``False``"

        To avoid under calculating we divide the memory calculation by 1.8 instead of 2

        Parameters
        ----------
        dims
            The number of dimensions in the vgg_face output

        Returns
        -------
        ``True`` if vector_linkage should be used. ``False`` if linkage should be used
        """
        np_float = 24  # bytes size of a numpy float
        divider = 1024 * 1024  # bytes to MB

        free_ram = psutil.virtual_memory().available / divider
        linkage_required = (((self._num_predictions ** 2) * np_float) / 1.8) / divider
        vector_required = ((self._num_predictions * dims) * np_float) / divider
        logger.debug("free_ram: %sMB, linkage_required: %sMB, vector_required: %sMB",
                     int(free_ram), int(linkage_required), int(vector_required))

        if linkage_required < free_ram:
            logger.verbose("Using linkage method")  # type:ignore[attr-defined]
            retval = False
        elif vector_required < free_ram:
            logger.warning("Not enough RAM to perform linkage clustering. Using vector "
                           "clustering. This will be significantly slower. Free RAM: %sMB. "
                           "Required for linkage method: %sMB",
                           int(free_ram), int(linkage_required))
            retval = True
        else:
            raise FaceswapError("Not enough RAM available to sort faces. Try reducing "
                                f"the size of  your dataset. Free RAM: {int(free_ram)}MB. "
                                f"Required RAM: {int(vector_required)}MB")
        logger.debug(retval)
        return retval

    def _do_linkage(self,
                    predictions: np.ndarray,
                    method: T.Literal["single", "centroid", "median", "ward"]) -> np.ndarray:
        """Use FastCluster to perform vector or standard linkage

        Parameters
        ----------
        predictions
            A stacked matrix of identity predictions of the shape (`N`, `D`) where `N` is the
            number of observations and `D` are the number of dimensions.
        method
            The clustering method to use.

        Returns
        -------
        The [`num_predictions`, 4] linkage vector
        """
        dims = predictions.shape[-1]
        if self._use_vector_linkage(dims):
            retval = linkage_vector(predictions, method=method)
        else:
            retval = linkage(predictions, method=method, preserve_input=False)
        logger.debug("Linkage shape: %s", retval.shape)
        return retval

    def _process_leaf_node(self,
                           current_index: int,
                           current_bin: int) -> list[tuple[int, int]]:
        """Process the output when we have hit a leaf node"""
        if not self._should_output_bins:
            return [(current_index, 0)]

        if current_bin not in self._bins:
            next_val = 0 if not self._bins else max(self._bins.values()) + 1
            self._bins[current_bin] = next_val
        return [(current_index, self._bins[current_bin])]

    def _get_bin(self,
                 tree: np.ndarray,
                 points: int,
                 current_index: int,
                 current_bin: int) -> int:
        """Obtain the bin that we are currently in.

        If we are not currently below the threshold for binning, get a new bin ID from the integer
        iterator.

        Parameters
        ----------
        tree
           A hierarchical tree (dendrogram)
        points
            The number of points given to the clustering process
        current_index
            The position in the tree for the recursive traversal
        current_bin
            The ID for the bin we are currently in. Only used when binning is enabled

        Returns
        -------
        The current bin ID for the node
        """
        if tree[current_index - points, 2] >= self._threshold:
            current_bin = next(self._iterator)
            logger.debug("Creating new bin ID: %s", current_bin)
        return current_bin

    def _seriation(self,
                   tree: np.ndarray,
                   points: int,
                   current_index: int,
                   current_bin: int = 0) -> list[tuple[int, int]]:
        """Seriation method for sorted similarity.

        Seriation computes the order implied by a hierarchical tree (dendrogram).

        Parameters
        ----------
        tree
           A hierarchical tree (dendrogram)
        points
            The number of points given to the clustering process
        current_index
            The position in the tree for the recursive traversal
        current_bin
            The ID for the bin we are currently in. Only used when binning is enabled

        Returns
        -------
        The indices in the order implied by the hierarchical tree
        """
        if current_index < points:  # Output the leaf node
            return self._process_leaf_node(current_index, current_bin)

        if self._should_output_bins:
            current_bin = self._get_bin(tree, points, current_index, current_bin)

        left = int(tree[current_index-points, 0])
        right = int(tree[current_index-points, 1])

        serate_left = self._seriation(tree, points, left, current_bin=current_bin)
        serate_right = self._seriation(tree, points, right, current_bin=current_bin)

        return serate_left + serate_right  # type: ignore

    def __call__(self) -> list[tuple[int, int]]:
        """Process the linkages.

        Transforms a distance matrix into a sorted distance matrix according to the order implied
        by the hierarchical tree (dendrogram).

        Returns
        -------
        List of indices with the order implied by the hierarchical tree or list of tuples of
        (`index`, `bin`) if a binning threshold was provided
        """
        logger.info("Sorting face distances. Depending on your dataset this may take some time...")
        if self._threshold:
            self._threshold = self._result_linkage[:, 2].max() * self._threshold
        result_order = self._seriation(self._result_linkage,
                                       self._num_predictions,
                                       self._num_predictions + self._num_predictions - 2)
        return result_order


__all__ = get_module_objects(__name__)
