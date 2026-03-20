#! /usr/env/bin/python3
"""Objects used for extraction plugins, runners and pipeline """
from __future__ import annotations
import logging
import typing as T
from dataclasses import dataclass, field
from enum import IntEnum
from zlib import compress

import cv2
import numpy as np
import numpy.typing as npt

from lib.align.aligned_face import batch_umeyama
from lib.align.aligned_utils import batch_resize, batch_transform, points_to_68
from lib.align.aligned_mask import Mask
from lib.align.alignments import PNGAlignments, MaskAlignmentsFile
from lib.align.constants import LandmarkType, MEAN_FACE
from lib.align.detected_face import DetectedFace
from lib.align.pose import Batch3D
from lib.logger import parse_class_init, format_array
from lib.utils import get_module_objects

if T.TYPE_CHECKING:
    from lib.align.alignments import PNGHeaderSourceDict
    from lib.align.constants import CenteringType

logger = logging.getLogger(__name__)


class ExtractSignal(IntEnum):
    """Signals to send to the extraction pipeline"""
    FLUSH = 1
    """Flush all queued items"""
    SHUTDOWN = 2
    """Flush all queued items and shutdown"""


@dataclass
class ExtractBatchAligned:
    """Dataclass for working with batches of aligned images

    Parameters
    ----------
    landmarks
        The face landmarks found for this batch in frame space or ``None`` if not available.
        Default: ``None`` (to be populated later)
    landmark_type
        The type of landmarks that the batch holds or ``None`` if not available.
        Default: ``None`` (to be populated later)
    """
    landmarks: npt.NDArray[np.float32] | None = None
    """The face landmarks found for this batch in frame space or ``None`` if not populated"""
    landmark_type: LandmarkType | None = None
    """The type of landmarks that the batch holds"""

    # The following "_cache_" attributes are cached on demand and accessed through their
    # corresponding "non _cache_" properties
    _cache_landmarks_68: npt.NDArray[np.float32] | None = field(init=False, default=None)
    _cache_landmarks_normalized: npt.NDArray[np.float32] | None = field(init=False, default=None)
    _cache_matrices: npt.NDArray[np.float32] | None = field(init=False, default=None)
    _cache_offsets_legacy: npt.NDArray[np.float32] | None = field(init=False, default=None)
    _cache_offsets_face: npt.NDArray[np.float32] | None = field(init=False, default=None)
    _cache_offsets_head: npt.NDArray[np.float32] | None = field(init=False, default=None)
    _cache_rotation: npt.NDArray[np.float32] | None = field(init=False, default=None)
    _cache_translation: npt.NDArray[np.float32] | None = field(init=False, default=None)

    def __repr__(self) -> str:
        """Pretty print arrays"""
        params = {}
        for k, v in self.__dict__.items():
            key = k.replace("_cache_", "")
            if isinstance(v, np.ndarray):
                params[key] = format_array(v)
                continue
            params[key] = v
        s_params = ", ".join(f"{k}={v}" for k, v in params.items())
        return f"{self.__class__.__name__}({s_params})"

    @property
    def landmarks_68(self) -> npt.NDArray[np.float32]:
        """ The stored landmarks as 68 point landmarks if supported, or original landmarks if not (
        4 point ROI landmarks)"""
        if self._cache_landmarks_68 is not None:
            return self._cache_landmarks_68

        if self.landmarks is None or not self.landmarks.size:
            return np.empty((0, 68, 2), dtype=np.float32)

        lms = T.cast("npt.NDArray[np.float32]", self.landmarks)
        if self.landmark_type not in (LandmarkType.LM_2D_68, LandmarkType.LM_2D_4):
            lms = points_to_68(lms, landmark_type=self.landmark_type)
        self._cache_landmarks_68 = lms
        return self._cache_landmarks_68

    @property
    def landmarks_normalized(self) -> npt.NDArray[np.float32]:
        """The normalized, aligned 68 point landmarks"""
        if self._cache_landmarks_normalized is not None:
            return self._cache_landmarks_normalized

        if self.landmarks is None or not self.landmarks.size:
            return np.empty((0, 68, 2), dtype=np.float32)

        self._cache_landmarks_normalized = batch_transform(self.matrices, self.landmarks_68)
        return self._cache_landmarks_normalized

    @property
    def matrices(self) -> npt.NDArray[np.float32]:
        """The face alignment matrices to transform from frame space to normalized (0, 1) space"""
        if self._cache_matrices is not None:
            return self._cache_matrices

        if self.landmarks is None or not self.landmarks.size:
            return np.empty((0, 3, 3), dtype=np.float32)

        if self.landmark_type == LandmarkType.LM_2D_4:
            points = self.landmarks
            lookup = LandmarkType.LM_2D_4
        else:
            points = self.landmarks_68[:, 17:]
            lookup = LandmarkType.LM_2D_51
        self._cache_matrices = batch_umeyama(points, MEAN_FACE[lookup], True).astype(np.float32)
        return self._cache_matrices

    @property
    def matrices_face(self) -> npt.NDArray[np.float32]:
        """The alignment matrices to transform from normalized legacy space (0, 1) to normalized
        face space"""
        mats = self.matrices.copy()
        mats[:, :2, 2] -= self.offsets_face
        return mats

    @property
    def matrices_head(self) -> npt.NDArray[np.float32]:
        """The alignment matrices to transform from normalized legacy space (0, 1) to normalized
        head space"""
        mats = self.matrices.copy()
        mats[:, :2, 2] -= self.offsets_head
        return mats

    @property
    def offsets_legacy(self) -> npt.NDArray[np.float32]:
        """The (N, x, y) offsets for normalized (legacy) centering. This is always (0, 0) for all
        items in the batch"""
        if self._cache_offsets_legacy is not None:
            return self._cache_offsets_legacy

        if self.landmarks is None or not self.landmarks.size:
            return np.empty((0, 2), dtype=np.float32)

        if self.landmark_type == LandmarkType.LM_2D_4:
            num_points = self.landmarks.shape[0]
        else:
            num_points = self.landmarks_68.shape[0]

        self._cache_offsets_legacy = np.zeros((num_points, 2), dtype=np.float32)
        return self._cache_offsets_legacy

    @property
    def offsets_face(self) -> npt.NDArray[np.float32]:
        """The (N, x, y) offsets required to shift from normalized (legacy) centering to face
        centering"""
        if self._cache_offsets_face is not None:
            return self._cache_offsets_face

        if self.landmarks is None or not self.landmarks.size:
            return np.empty((0, 2), dtype=np.float32)

        if self.landmark_type == LandmarkType.LM_2D_4:
            offsets = np.zeros((self.landmarks.shape[0], 2), dtype=np.float32)
        else:
            offsets = Batch3D.get_offsets("face", self.rotation, self.translation)

        self._cache_offsets_face = offsets
        return self._cache_offsets_face

    @property
    def offsets_head(self) -> npt.NDArray[np.float32]:
        """The (N, x, y) offsets required to shift from normalized (legacy) centering to head
        centering"""
        if self._cache_offsets_head is not None:
            return self._cache_offsets_head

        if self.landmarks is None or not self.landmarks.size:
            return np.empty((0, 2), dtype=np.float32)

        if self.landmark_type == LandmarkType.LM_2D_4:
            offsets = np.zeros((self.landmarks.shape[0], 2), dtype=np.float32)
        else:
            offsets = Batch3D.get_offsets("head", self.rotation, self.translation)

        self._cache_offsets_head = offsets
        return self._cache_offsets_head

    @property
    def rotation(self) -> npt.NDArray[np.float32]:
        """The estimated (N, 3, 1) rotation vectors"""
        if self._cache_rotation is not None:
            return self._cache_rotation

        if self.landmarks is None or not self.landmarks.size:
            return np.empty((0, 3, 1), dtype=np.float32)

        if self.landmark_type == LandmarkType.LM_2D_4:
            rot_trans = np.zeros((2, self.landmarks.shape[0], 3, 1), dtype=np.float32)
        else:
            rot_trans = Batch3D.solve_pnp(self.landmarks_normalized)
        self._cache_rotation = T.cast("npt.NDArray[np.float32]", rot_trans[0])
        self._cache_translation = rot_trans[1]
        return self._cache_rotation

    @property
    def translation(self) -> npt.NDArray[np.float32]:
        """The estimated (N, 3, 1) translation vectors"""
        if self._cache_translation is not None:
            return self._cache_translation

        if self.landmarks is None or not self.landmarks.size:
            return np.empty((0, 3, 1), dtype=np.float32)

        if self.landmark_type == LandmarkType.LM_2D_4:
            rot_trans = np.zeros((2, self.landmarks.shape[0], 3, 1), dtype=np.float32)
        else:
            rot_trans = Batch3D.solve_pnp(self.landmarks_normalized)

        rot_trans = Batch3D.solve_pnp(self.landmarks_normalized)
        self._cache_rotation = rot_trans[0]
        self._cache_translation = T.cast("npt.NDArray[np.float32]", rot_trans[1])
        return self._cache_translation

    def __getitem__(self, indices: slice) -> ExtractBatchAligned:
        """Obtain a subset of this batch object with the data given by the start and end indices

        Parameters
        ----------
        indices
            The (start, stop, end) slice for extracting from the batch

        Returns
        -------
        A batch object containing the data from this object for the given indices
        """
        retval = ExtractBatchAligned(landmark_type=self.landmark_type)
        if self.landmarks is not None:
            retval.landmarks = self.landmarks[indices]

        for k, v in self.__dict__.items():
            if k.startswith("_cache_") and v is not None:
                setattr(retval, k, v[indices])

        return retval

    def append(self, batch: ExtractBatchAligned) -> None:
        """Append the data from the given batch object to this batch object

        Parameters
        ----------
        batch
            The object containing data to be appended to this object
        """
        if batch.landmarks is not None:
            self.landmarks = (np.concatenate([self.landmarks, batch.landmarks])
                              if self.landmarks is not None else batch.landmarks)
            if self.landmark_type is None:
                self.landmark_type = batch.landmark_type

        for k, v in batch.__dict__.items():
            if k.startswith("_cache_") and v is not None:
                exist = getattr(self, k)
                val = None if exist is None else np.concatenate([exist, v])
                setattr(self, k, val)

    def apply_mask(self, mask: npt.NDArray[np.bool_]) -> None:
        """Apply a boolean mask to the batch object. ``True`` values are kept, ``False`` values
        are discarded

        Parameters
        ----------
        mask
            The boolean mask to apply to the object. Must be of size (landmarks, )
        """
        if np.all(mask):
            return

        if self.landmarks is not None:
            self.landmarks = self.landmarks[mask]

        for k, v in self.__dict__.items():
            if k.startswith("_cache_") and v is not None:
                setattr(self, k, v[mask])


@dataclass
class ExtractBatchMask:
    """Dataclass for holding information about masks produced by the extraction pipeline

    Parameters
    ----------
    centering
        The centering type of the masks
    matrices
        The normalized matrices required to take the masks from (0, 1) to full frame
    storage_size
        The pixel size to store the mask at in the alignments file. Default: 0 (must be populated
        later)
    masks
        The masks for this batch. Default: empty array (must be populated later)
    """
    centering: CenteringType
    """The centering type of the masks"""
    matrices: npt.NDArray[np.float32]
    """The normalized matrices required to take the masks from (0, 1) to full frame"""
    storage_size: int = field(default=0)
    """The pixel size to store the mask at in the alignments file"""
    masks: npt.NDArray[np.uint8] = field(default_factory=lambda: np.empty((0, 0, 0),
                                                                          dtype=np.uint8))
    """The masks for this batch"""

    def __repr__(self) -> str:
        """Pretty print arrays"""
        params = {k: format_array(v) if isinstance(v, np.ndarray) else repr(v)
                  for k, v in self.__dict__.items()}
        s_params = ", ".join(f"{k}={v}" for k, v in params.items())
        return f"{self.__class__.__name__}({s_params})"

    def __getitem__(self, indices: slice) -> ExtractBatchMask:
        """Basic object slicing for splitting batches

        Parameters
        ----------
        indices
            The (start, stop, end) slice for extracting from the batch

        Returns
        -------
        The sliced data from this batch
        """
        return ExtractBatchMask(self.centering,
                                self.matrices[indices],
                                storage_size=self.storage_size,
                                masks=self.masks[indices])

    def append(self, mask_batch: ExtractBatchMask) -> None:
        """Append the given mask batch object to this batch mask object

        Parameters
        ----------
        mask_batch
            The object containing data to be appended to this object
        """
        self.matrices = np.concatenate([self.matrices, mask_batch.matrices], axis=0)
        self.masks = np.concatenate([self.masks, mask_batch.masks], axis=0)

    def apply_mask(self, mask: npt.NDArray[np.bool_]) -> None:
        """Apply a boolean mask to the batch object. ``True`` values are kept, ``False`` values
        are discarded

        Parameters
        ----------
        mask
            The boolean mask to apply to the object. Must be of size (num_masks, )
        """
        if np.all(mask):
            return
        self.masks = self.masks[mask]
        self.matrices = self.matrices[mask]


@dataclass
class ExtractBatch:  # pylint:disable=too-many-instance-attributes
    """Dataclass for holding a batch flowing through Extraction plugins.

    The batch size for post Detector plugins is not the same as the overall batch size.
    An image may contain 0 or more detected faces, and these need to be split and recombined
    to be able to utilize a plugin's internal batch size.

    Parameters
    ----------
    filenames
        The original frame filenames for the batch
    images
        The original frames
    sources
        The full path to the source folder or video file. Default: ``[]`` (Not provided)
    is_aligned
        ``True`` if :attr:`images` contains aligned faces. ``False`` if it contains full frames.
        Default: ``False``
    frame_sizes
        The original frame (height, width) dimensions that contained the aligned images when
        :attr:`images` are aligned faces. Default: ``None``
    frame_metadata
        The original frame meta data when aligned faces is ``True`` otherwise ``None``
    passthrough
        `True`` if the contents of this item are meant to pass straight through the extraction
        pipeline for immediate return
    """
    # Input required information
    filenames: list[str] = field(default_factory=list)
    """The original frame filenames"""
    images: list[np.ndarray] = field(default_factory=list)
    """The original frames"""
    sources: list[str | None] = field(default_factory=list)
    """The full paths to the source folder or video file. ``None`` if not provided"""
    is_aligned: bool = False
    """``True`` if :attr:`images` contains aligned faces. ``False`` for full frames"""
    frame_sizes: list[tuple[int, int]] | None = None
    """The original frame (heights, widths) when the images are aligned faces"""
    frame_metadata: list[PNGHeaderSourceDict] | None = None
    """The original frame metadata when aligned faces is ``True`` otherwise ``None``"""
    passthrough: bool = False
    """Whether this item should pass straight through the pipeline for immediate return"""

    # Final data for output
    bboxes: npt.NDArray[np.int32] = field(init=False,
                                          default_factory=lambda: np.empty((0, 4), dtype=np.int32))
    """The bounding boxes found for this batch"""
    aligned: ExtractBatchAligned = field(init=False, default_factory=ExtractBatchAligned)
    """Holds the face landmarks found for this batch any any aligned data"""
    masks: dict[str, ExtractBatchMask] = field(init=False, default_factory=dict)
    """The masks for this batch"""
    identities: dict[str, npt.NDArray[np.float32]] = field(init=False, default_factory=dict)
    """The identity matrices for face recognition found for this batch"""

    # Internal batch structure
    frame_ids: npt.NDArray[np.int32] = field(init=False,
                                             default_factory=lambda: np.empty((0, ),
                                                                              dtype=np.int32))
    """A mapping of each box to which frame they came from"""

    # Internal holder for passing data between processes. Deleted at output from each plugin
    data: np.ndarray = field(init=False)
    """The data for this batch that has been populated by a processing step for ingestion by the
    next processing step. Internally populated. Cleared at the end of each plugin"""
    matrices: npt.NDArray[np.float32] = field(init=False)
    """Transformation matrices for taking points from model input space to frame space. Cleared at
    the end of each plugin"""

    def __repr__(self) -> str:
        """Pretty print arrays"""
        params: dict[str, T.Any] = {}
        for k, v in self.__dict__.items():
            if isinstance(v, (list, tuple)) and v and isinstance(v[0], np.ndarray):
                params[k] = [format_array(x) for x in v]
                continue
            if k == "identities" and isinstance(v, dict):
                params[k] = {key: format_array(val) for key, val in v.items()}
                continue
            if isinstance(v, np.ndarray):
                params[k] = format_array(v)
                continue
            params[k] = v

        s_params = ", ".join(f"{k}={v}" for k, v in params.items())
        return f"{self.__class__.__name__}({s_params})"

    def __post__init__(self) -> None:
        """Populate sources if not provided"""
        if not self.sources:
            self.sources = [None for _ in range(len(self.filenames))]

    def __len__(self) -> int:
        """The number of faces contained within this object"""
        return len(self.bboxes)

    @property
    def landmarks(self) -> npt.NDArray[np.float32] | None:
        """The face landmarks found for this batch in frame space or ``None`` if not populated"""
        return self.aligned.landmarks

    @landmarks.setter
    def landmarks(self, value: npt.NDArray) -> None:
        """Set the landmarks attribute in the underlining ExtractBatchAlign object

        Parameters
        ----------
        value
            The landmarks to set
        """
        self.aligned.landmarks = value

    @property
    def landmark_type(self) -> LandmarkType | None:
        """The landmark type found for this batch or ``None`` if not populated"""
        return self.aligned.landmark_type

    @landmark_type.setter
    def landmark_type(self, value: LandmarkType) -> None:
        """Set the landmark_type attribute in the underlining ExtractBatchAlign object

        Parameters
        ----------
        value
            The landmark_type to set
        """
        self.aligned.landmark_type = value

    @property
    def lengths(self) -> npt.NDArray[np.int32]:
        """The number of bboxes that belong to each frame"""
        if self.frame_ids.size == 0:
            return np.zeros((len(self.images)), dtype=np.int32)
        return np.bincount(self.frame_ids, minlength=len(self.images)).astype(np.int32)

    def __getitem__(self, indices: slice) -> ExtractBatch:
        """Obtain a subset of this batch object with the data given by the start and end indices

        Parameters
        ----------
        indices
            The (start, stop, end) slice for extracting from the batch

        Returns
        -------
        A batch object containing the data from this object for the given indices
        """
        frame_ids = self.frame_ids[indices].copy()
        # If requesting the first bbox, we select all frames from the start
        frame_start = 0 if indices.start == 0 else frame_ids[0]

        frame_end = frame_ids[-1] + 1
        if indices.stop < self.bboxes.shape[0] and self.frame_ids[indices.stop] > frame_end:
            # catch any zero box frames between now and next split request
            frame_end = self.frame_ids[indices.stop]

        frame_sizes = None if self.frame_sizes is None else self.frame_sizes[frame_start:frame_end]
        frame_metadata = (None if self.frame_metadata is None
                          else self.frame_metadata[frame_start:frame_end])
        retval = ExtractBatch(self.filenames[frame_start:frame_end],
                              self.images[frame_start:frame_end],
                              sources=self.sources[frame_start:frame_end],
                              is_aligned=self.is_aligned,
                              frame_sizes=frame_sizes,
                              frame_metadata=frame_metadata,
                              passthrough=self.passthrough)
        retval.bboxes = self.bboxes[indices]
        retval.aligned = self.aligned[indices]
        retval.masks = {k: v[indices] for k, v in self.masks.items()}
        retval.identities = {k: v[indices] for k, v in self.identities.items()}

        if indices.start > 0:
            frame_ids -= frame_ids[0]  # Reset to zero
        retval.frame_ids = frame_ids

        if self.landmarks is not None:
            retval.landmarks = self.landmarks[indices]

        if hasattr(self, "data"):
            retval.data = self.data[indices]

        if hasattr(self, "matrices"):
            retval.matrices = self.matrices[indices]

        return retval

    def _populate_batch(self, batch: ExtractBatch) -> None:
        """Populate this batch with the data from the incoming batch when this batch is empty

        Parameters
        ----------
        batch
            The object containing data to populate to this object
        """
        for k, v in batch.__dict__.items():
            setattr(self, k, v)

    def append(self, batch: ExtractBatch) -> None:  # noqa[C901]
        """Append the data from the given batch object to this batch object

        Parameters
        ----------
        batch
            The object containing data to be appended to this object
        """
        if not self.filenames:
            self._populate_batch(batch)
            return
        frame_offset = len(self.filenames)
        if self.filenames[-1] == batch.filenames[0]:
            frame_offset -= 1  # We are still on the same frame
            if not np.any(self.images[-1]) and np.any(batch.images[0]):
                # Image was stripped for the faces in this batch, but exist for incoming batch
                self.images[-1] = batch.images[0]
        batch.frame_ids += frame_offset

        existing_filenames = self.filenames[:]
        self.filenames.extend(f for f in batch.filenames if f not in existing_filenames)
        self.images.extend(batch.images[i] for i, f in enumerate(batch.filenames)
                           if f not in existing_filenames)
        self.sources.extend(batch.sources[i] for i, f in enumerate(batch.filenames)
                            if f not in existing_filenames)

        if self.frame_sizes is not None and batch.frame_sizes is not None:
            self.frame_sizes.extend(batch.frame_sizes[i] for i, f in enumerate(batch.filenames)
                                    if f not in existing_filenames)
        if self.frame_metadata is not None and batch.frame_metadata is not None:
            self.frame_metadata.extend(batch.frame_metadata[i]
                                       for i, f in enumerate(batch.filenames)
                                       if f not in existing_filenames)

        self.bboxes = np.concatenate([self.bboxes, batch.bboxes])
        self.frame_ids = np.concatenate([self.frame_ids, batch.frame_ids])
        self.aligned.append(batch.aligned)

        for name, masks in batch.masks.items():
            if name in self.masks:
                self.masks[name].append(masks)
            else:
                self.masks[name] = masks

        for name, identities in batch.identities.items():
            self.identities[name] = (np.concatenate([self.identities[name], identities])
                                     if name in self.identities
                                     else identities)

        if hasattr(self, "data"):
            self.data = np.concatenate([self.data, batch.data])

        if hasattr(self, "matrices"):
            self.matrices = np.concatenate([self.matrices, batch.matrices])

    @classmethod
    def from_frame_faces(cls, media: FrameFaces) -> ExtractBatch:
        """Populate a new ExtractBatch with the contents of an FrameFaces object.

        Parameters
        ----------
        media
            The FrameFaces to populate this batch from

        Returns
        -------
        A new ExtractBatch object populated from the given FrameFaces object
        """
        retval = cls([media.filename],
                     [media.image],
                     sources=[media.source],
                     is_aligned=media.is_aligned,
                     frame_sizes=[media.image_size] if media.is_aligned else None,
                     frame_metadata=[media.frame_metadata] if media.frame_metadata else None,
                     passthrough=media.passthrough)
        retval.frame_ids = np.fromiter((0 for _ in range(len(media.bboxes))), dtype=np.int32)
        retval.bboxes = media.bboxes
        retval.identities = media.identities
        retval.masks = media.masks
        retval.aligned = media.aligned
        return retval

    def from_detected_faces(self, faces: list[DetectedFace]) -> None:
        """Populate an ExtractBatch with the contents of a DetectedFace object.

        Parameters
        ----------
        faces
            The DetectedFace objects to populate this batch

        Raises
        ------
        ValueError
            If attempting to add detected faces without pre-populating filename and image or if
            bounding boxes pre-exist or if more than one frame is held in this batch
        """
        if not self.filenames:
            raise ValueError("Filenames must be populated prior to adding detected faces")
        if not self.images:
            raise ValueError("Images must be populated prior to adding detected faces")
        if len(self.filenames) != len(self.images) != 1:
            raise ValueError("Only 1 filename and image should be the batch")
        if np.any(self.bboxes):
            raise ValueError("An empty ExtractBatch object is required to add detected faces")
        self.frame_ids = np.fromiter((0 for _ in range(len(faces))), dtype=np.int32)
        self.aligned.landmark_type = LandmarkType.from_shape(T.cast(tuple[int, int],
                                                             faces[0].landmarks_xy.shape))
        num_faces = len(faces)
        self.bboxes = np.empty((num_faces, 4), dtype=np.int32)
        self.aligned.landmarks = np.empty((num_faces, *faces[0].landmarks_xy.shape),
                                          dtype=np.float32)
        self.identities = {k: np.empty((num_faces, *v.shape), dtype=np.float32)
                           for k, v in faces[0].identity.items()}
        self.masks = {
            k: ExtractBatchMask(v.stored_centering,
                                np.empty((num_faces, 2, 3), dtype=np.float32),
                                storage_size=v.stored_size,
                                masks=np.empty((num_faces, v.stored_size, v.stored_size),
                                               dtype=np.uint8))
            for k, v in faces[0].mask.items()
            }
        for i, f in enumerate(faces):
            self.bboxes[i] = np.array([f.left, f.top, f.right, f.bottom], dtype=np.int32)
            self.aligned.landmarks[i] = f.landmarks_xy
            for k, idn in f.identity.items():
                self.identities[k][i] = idn
            for k, m in f.mask.items():
                mask = self.masks[k]
                mask.matrices[i] = m.affine_matrix
                mask.masks[i] = m.mask[:, :, 0]

    def apply_mask(self, mask: npt.NDArray[np.bool_]) -> None:
        """Apply a boolean mask to the batch object. ``True`` values are kept, ``False`` values
        are discarded

        Parameters
        ----------
        mask
            The boolean mask to apply to the object. Must be of size (num_boxes, )
        """
        if np.all(mask):
            return

        self.bboxes = self.bboxes[mask]
        self.frame_ids = self.frame_ids[mask]
        self.aligned.apply_mask(mask)

        if self.masks:
            for v in self.masks.values():
                v.apply_mask(mask)

        if self.identities:
            self.identities = {k: v[mask] for k, v in self.identities.items()}


class FrameFaces:  # pylint:disable=too-many-instance-attributes
    """An object for holding information about faces in a single frame

    Parameters
    ----------
    filename
        The original file name of the frame
    image
        The original frame or a faceswap aligned face image
    bboxes
        The (N, Left, Top, Right, Bottom) bounding boxes of the faces in the frame.
        Default: ``None`` (Not provided)
    landmarks
        The (N, M, 2) landmarks for each face in the frame, in frame space.
        Default: ``None`` (Not provided)
    identities
        The identity matrices for each face in the frame. Default: ``None`` (Not provided)
    masks
        The mask objects for each face in the frame. Default: ``None`` (Not provided)
    source
        The full path to the source folder or video file. Default: ``None`` (Not provided)
    is_aligned
        ``True`` if the :attr:`image` is an aligned faceswap image otherwise ``False``. Used for
        face filtering with vggface2. Aligned faceswap images will automatically skip detection,
        alignment and masking. Default: ``False``
    frame_metadata
        The frame metadata for aligned images. ``None`` if the image is not an aligned image
    passthrough
        ``True`` if this item is meant to be passed straight through the extraction pipeline with
        no batching or caching. for immediate return. Default: ``False``
    """
    def __init__(self,  # pylint:disable=too-many-arguments,too-many-positional-arguments
                 filename: str,
                 image: npt.NDArray[np.uint8],
                 bboxes: npt.NDArray[np.int32] | None = None,
                 landmarks: npt.NDArray[np.float32] | None = None,
                 identities: dict[str, npt.NDArray[np.float32]] | None = None,
                 masks: dict[str, ExtractBatchMask] | None = None,
                 source: str | None = None,
                 is_aligned: bool = False,
                 frame_metadata: PNGHeaderSourceDict | None = None,
                 passthrough: bool = False) -> None:
        logger.trace(parse_class_init(locals()))  # type:ignore[attr-defined]
        if is_aligned:
            assert frame_metadata is not None, "frame_metadata is required for aligned images"

        self.filename = filename
        """The original file name of the original frame"""
        self.image = image
        """The original frame or a faceswap aligned face image"""
        self.bboxes = np.empty((0, 4), dtype=np.int32) if bboxes is None else bboxes
        """The (N, Left, Top, Right, Bottom) bounding boxes of the faces in the frame"""
        self.identities = {} if identities is None else identities
        """The identity matrices for each face in the frame"""
        self.masks = {} if masks is None else masks
        """The mask objects for each face in the frame"""
        self.source = source
        """The full path to the source folder or video file or ``None`` if not provided"""
        self.frame_metadata: PNGHeaderSourceDict | None = frame_metadata
        """The frame metadata that has been added from an aligned image. ``None`` if metadata has
        not been added"""
        self.is_aligned = is_aligned
        """``True`` if :attr:`image` is an aligned faceswap image otherwise ``False``"""
        self.passthrough = passthrough
        """``True`` if the contents of this item are meant to pass straight through the extraction
        pipeline for immediate return"""
        self.image_shape = self._get_image_shape()
        """The shape of the original frame"""

        self.aligned = ExtractBatchAligned(
            landmarks=landmarks if landmarks is None else landmarks,
            landmark_type=(None if landmarks is None
                           else LandmarkType.from_shape(T.cast(tuple[int, int],
                                                               landmarks.shape[1:]))))
        """Holds the face landmarks found for this batch any any aligned data"""
        self._name = self.__class__.__name__
        """The name of this object for logging"""

    def __repr__(self) -> str:
        """Pretty print for logging"""
        params: dict[str, T.Any] = {}
        for k, v in self.__dict__.items():
            if k in ("image_shape", "_name"):
                continue
            if k == "identities":
                params[k] = {i: format_array(m) for i, m in v.items()}
                continue
            if k == "aligned":
                lms = v.landmarks
                params["landmarks"] = None if lms is None else format_array(lms)
                continue
            params[k] = format_array(v) if isinstance(v, np.ndarray) else repr(v)
        s_params = ", ".join(f"{k}={v}" for k, v in params.items())
        return f"{self._name}({s_params})"

    def __len__(self) -> int:
        """The number of faces contained within this object"""
        return len(self.bboxes)

    @property
    def landmarks(self) -> npt.NDArray[np.float32] | None:
        """The (N, M, 2) landmarks for each face in the frame, in frame space"""
        return self.aligned.landmarks

    @landmarks.setter
    def landmarks(self, value: npt.NDArray[np.float32]) -> None:
        """Set the landmarks attribute in the underlining ExtractBatchAlign object

        Parameters
        ----------
        value
            The landmarks to set
        """
        self.aligned.landmarks = value
        self.aligned.landmark_type = LandmarkType.from_shape(T.cast(tuple[int, int],
                                                                    value.shape[1:]))

    @property
    def detected_faces(self) -> list[DetectedFace]:
        """A list of DetectedFace objects within the :attr:`image`"""
        return [DetectedFace(left=int(box[0]),
                             top=int(box[1]),
                             width=int(box[2] - box[0]),
                             height=int(box[3] - box[1]),
                             landmarks_xy=(None if self.landmarks is None
                                           or not self.landmarks.size
                                           else self.landmarks[idx]),
                             mask={k: Mask(storage_size=m.storage_size,
                                           storage_centering=m.centering).add(
                                               m.masks[idx],
                                               m.matrices[idx])
                                   for k, m in self.masks.items()},
                             identity={k: i[idx] for k, i in self.identities.items()
                                       if i.size})
                for idx, box in enumerate(self.bboxes)]

    @detected_faces.setter
    def detected_faces(self, faces: list[DetectedFace]) -> None:
        """Set the underlying properties from a list of DetectedFace objects

        Parameters
        ----------
        faces
            The DetectedFace objects to populate to this object

        Raises
        ------
        ValueError
            If the FrameFaces object does not contain a filename and image or if any of the data
            fields are populated
        """
        if not self.filename or not np.any(self.image):
            raise ValueError("Filename and image must be populated before adding DetectedFace "
                             "objects")
        if np.any(self.bboxes) or self.landmarks is not None or self.masks or self.identities:
            raise ValueError("The FrameFaces object must not be pre-populated when adding"
                             "DetectedFace objects")
        for face in faces:
            if None not in (face.left, face.top, face.width, face.height):
                bbox = np.array([[face.left, face.top, face.right, face.bottom]], dtype=np.int32)
                self.bboxes = np.concatenate([self.bboxes, bbox])
            if face.has_landmarks:
                landmarks = np.array(face.landmarks_xy, dtype=np.float32)[None]
                self.landmarks = (landmarks if self.landmarks is None
                                  else np.concatenate([self.landmarks, landmarks]))
            for k, m in face.mask.items():
                msk = ExtractBatchMask(m.stored_centering,
                                       m.affine_matrix[None],
                                       m.stored_size,
                                       m.mask[None])
                if k not in self.masks:
                    self.masks[k] = msk
                else:
                    self.masks[k].append(msk)
            for k, i in face.identity.items():
                if k not in self.identities:
                    self.identities[k] = i[None]
                else:
                    self.identities[k] = np.concatenate([self.identities[k], i[None]])

    @property
    def image_size(self) -> tuple[int, int]:
        """The (`height`, `width`) of the stored :attr:`image`"""
        return self.image_shape[:2]

    def _get_image_shape(self) -> tuple[int, int, int]:
        """Obtain the shape of the original image. Either the given image's shape or the value
        stored in the metadata if this is an aligned face object

        Returns
        -------
        The shape of the original image
        """
        if self.is_aligned:
            assert self.frame_metadata is not None
            dims = T.cast(tuple[int, int], self.frame_metadata["source_frame_dims"])
            return (*dims, 3)
        return T.cast(tuple[int, int, int], self.image.shape)

    def append(self, batch: FrameFaces) -> None:
        """Append the data from the given batch object to this batch object

        Parameters
        ----------
        batch
            The object containing data to be appended to this object
        """
        assert batch.filename == self.filename
        assert batch.source == self.source
        assert batch.passthrough == self.passthrough
        assert batch.frame_metadata == self.frame_metadata

        if not np.any(self.image):  # Image potentially deleted from previous split batch
            self.image = batch.image
        self.bboxes = np.concatenate([self.bboxes, batch.bboxes])
        self.aligned.append(batch.aligned)
        for name, masks in batch.masks.items():
            if name in self.masks:
                self.masks[name].append(masks)
            else:
                self.masks[name] = masks

        for name, identities in batch.identities.items():
            self.identities[name] = (np.concatenate([self.identities[name], identities])
                                     if name in self.identities
                                     else identities)

    def remove_image(self) -> None:
        """Delete the image and reset :attr:`image` to ``None``."""
        logger.trace("[%s] Removing image for filename: '%s'",  # type:ignore[attr-defined]
                     self._name, self.filename)
        del self.image
        self.image = np.empty((0, 0, 3), dtype=np.uint8)


def frame_faces_to_alignment(media: FrameFaces) -> list[PNGAlignments]:
    """Convert the faces in a FrameFaces object into a list of dictionaries (one for each face)
    for serializing into image headers and alignments files"""
    if not media:
        return []
    assert media.landmarks is not None
    assert media.landmarks.shape[0] == len(media)
    assert all(m.masks.shape[0] == m.matrices.shape[0] == len(media) for m in media.masks.values())
    assert all(i.shape[0] == len(media) for i in media.identities.values())

    masks = {}
    for k, v in media.masks.items():
        scales = np.hypot(v.matrices[..., 0, 0], v.matrices[..., 1, 0])  # Always same x/y scaling
        interpolators = np.where(scales > 1.0, cv2.INTER_LINEAR, cv2.INTER_AREA)
        store_masks = v.masks
        mats = v.matrices
        if v.storage_size != v.masks.shape[1]:
            store_masks = batch_resize(v.masks[..., None], v.storage_size)[..., 0]
            mats = mats.copy()
            mats[:, :2] *= v.storage_size / v.masks.shape[1]
        masks[k] = {"mask": [compress(m.tobytes()) for m in store_masks],
                    "mats": mats.tolist(),
                    "interpolators": interpolators.tolist(),
                    "size": v.storage_size,
                    "centering": v.centering}

    return [PNGAlignments(x=int(bbox[0]),
                          y=int(bbox[1]),
                          w=int(bbox[2] - bbox[0]),
                          h=int(bbox[3] - bbox[1]),
                          landmarks_xy=lms,
                          mask={k: MaskAlignmentsFile(mask=m["mask"][idx],
                                                      affine_matrix=m["mats"][idx],
                                                      interpolator=int(m["interpolators"][idx]),
                                                      stored_size=m["size"],
                                                      stored_centering=m["centering"])
                                for k, m in masks.items()},
                          identity={k: i[idx].tolist() for k, i in media.identities.items()})
            for idx, (bbox, lms) in enumerate(zip(media.bboxes, media.landmarks.tolist()))]


__all__ = get_module_objects(__name__)
